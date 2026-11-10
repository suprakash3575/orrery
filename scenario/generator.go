// Package scenario builds reproducible Orrery maps from a seed.
//
// It lives outside package game on purpose: the engine deliberately takes an
// explicit planet list and has no map generator of its own, so randomized
// layouts belong to whoever is running matches, not to the rules. The engine
// stays the authority on what is legal — this package only proposes maps, and
// game.New() still validates and sizes everything it is handed.
package scenario

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math/rand"

	"orrery/game"
)

// GeneratorV1ID is the stable identity of this generator. Recorded alongside a
// seed so that a seed always maps back to the same map: the pair (generator
// version, seed) is the replay key, not the seed alone.
const GeneratorV1ID = "generator-v1"

const (
	// The stream names are part of generator-v1's replay contract. Hashing the
	// scenario seed with a different name gives each subsystem a deterministic
	// private stream, so adding an engine draw cannot shift map-generation
	// choices (or vice versa).
	mapRNGDomain    = "orrery-generator-v1-map"
	engineRNGDomain = "orrery-generator-v1-engine"
)

const (
	// minSpacing is the empty gap required between any two body surfaces, the
	// sun's included. Two bodies must therefore sit at least
	// radiusA + radiusB + minSpacing apart centre to centre. It is wide enough
	// that a fleet launched from one body spawns in open space instead of
	// immediately inside its neighbour.
	minSpacing = 2.0

	// Home planets are identical for both players and fixed across seeds, so
	// the only thing a seed varies is the layout. Production sits at the top
	// of the engine's validated 1..5 range: a home is the strongest planet on
	// the board, and both players get exactly one.
	homeShips      = 10
	homeProduction = 5

	// A neutral planet's garrison scales with its production, so a richer
	// planet is also better defended and has to be earned rather than simply
	// grabbed first. Production is a whole number, so this is exact.
	neutralShipsPerProduction = 10

	// Neutral production spans the engine's full validated range, giving a mix
	// of cheap and valuable targets on every map.
	minNeutralProduction = 1
	maxNeutralProduction = 5

	// Neutral planets are added in mirrored pairs, so 3..5 pairs plus the two
	// homes produce 8, 10, or 12 planets.
	minNeutralPairs = 3
	maxNeutralPairs = 5

	// maxPlacementAttempts bounds the rejection sampling for a single pair.
	// Exhausting it returns an error rather than looping forever. The bound is
	// far above what the standard board needs — the largest map fills well
	// under a third of the usable area — so reaching it signals a board or
	// spacing configuration that cannot hold the planets being asked of it,
	// not ordinary bad luck.
	maxPlacementAttempts = 1000
)

// GeneratorV1 builds a random but legal two-player map from a seed: 8 to 12
// planets with 2-fold rotational symmetry about the sun, the two home planets
// diagonally opposite each other, static planets, no comets, and no sun
// hazard.
//
// It implements arena.ScenarioFactory, so the existing match runner and CLI
// use it without changes.
type GeneratorV1 struct{}

// ID returns the versioned generator identity.
func (GeneratorV1) ID() string { return GeneratorV1ID }

// New builds an independent game for one scenario seed.
//
// The map and the engine draw from separate streams derived from the same
// seed rather than sharing one. That keeps a seed's map fixed no matter how
// engine randomness evolves: when a future experiment re-enables comets or
// orbits, adding or removing map-generation draws cannot silently shift the
// engine's later random sequence. The separate names also prevent the map and
// engine from starting with identical, correlated random values.
func (GeneratorV1) New(seed int64) (*game.Game, error) {
	config := game.DefaultConfig()
	// The reinforcement-learning V1 variant: two players on static planets,
	// with no comets and no sun hazard. Every mechanic removed here is one
	// fewer confounder while the learning pipeline is still being debugged;
	// each is re-enabled later as its own separate experiment.
	config.SunHazardEnabled = false
	config.CometSpawningEnabled = false
	config.OrbitsEnabled = false
	config.Rand = rand.New(rand.NewSource(deriveRNGSeed(seed, engineRNGDomain)))

	mapRNG := rand.New(rand.NewSource(deriveRNGSeed(seed, mapRNGDomain)))
	planets, err := generatePlanets(mapRNG, config)
	if err != nil {
		return nil, fmt.Errorf("scenario seed %d: %w", seed, err)
	}
	return game.New(config, planets, nil)
}

// generatePlanets builds one reproducible map. Its caller owns the RNG and
// every random map choice comes from that one stream. GeneratorV1.New creates
// a fresh RNG for each call, so concurrent arena workers cannot disturb one
// another's draws.
//
// The map is built in mirrored pairs about the sun: whatever one side of the
// board holds, the other holds an exact 180-degree copy. Symmetry is what
// makes a result attributable to play rather than to a lucky spawn, which
// matters more than usual here because a learning agent will happily exploit a
// biased map instead of learning to play well on a fair one.
func generatePlanets(rng *rand.Rand, config game.Config) ([]game.Planet, error) {
	// rng.Intn(3) returns one of 0, 1, or 2. Adding 3 produces 3, 4, or 5 neutral pairs
	// The +1 makes the maximum value, 5, possible. Without it, the result could only be 3 or 4.
	pairs := minNeutralPairs + rng.Intn(maxNeutralPairs-minNeutralPairs+1)
	planets := make([]game.Planet, 0, 2+2*pairs)

	// Homes are placed first so neither player can be crowded out by a neutral
	// that happened to take the spot a home needed.
	first, second, err := placePair(rng, config, planets, game.PlanetRadius(homeProduction))
	if err != nil {
		return nil, fmt.Errorf("home planets: %w", err)
	}
	planets = append(planets, planetAt(len(planets)+1, first, 0, homeShips, homeProduction))
	planets = append(planets, planetAt(len(planets)+1, second, 1, homeShips, homeProduction))

	for pair := range pairs {
		production := minNeutralProduction + rng.Intn(maxNeutralProduction-minNeutralProduction+1)
		first, second, err := placePair(rng, config, planets, game.PlanetRadius(production))
		if err != nil {
			return nil, fmt.Errorf("neutral pair %d: %w", pair, err)
		}
		// Both twins share one production roll, so the pair is a true mirror in
		// value as well as position — a player cannot be handed the richer half.
		ships := neutralShipsPerProduction * production
		planets = append(planets, planetAt(len(planets)+1, first, game.NeutralPlayer, ships, production))
		planets = append(planets, planetAt(len(planets)+1, second, game.NeutralPlayer, ships, production))
	}
	return planets, nil
}

// deriveRNGSeed deterministically assigns one scenario seed to a named random
// stream. The seed is encoded explicitly as a big-endian 64-bit word so the
// same input bytes are hashed on every machine; converting through uint64
// preserves the two's-complement bits of negative seeds. FNV-1a is fixed by
// the Go standard library, making this formula suitable for recorded replays.
func deriveRNGSeed(scenarioSeed int64, domain string) int64 {
	hash := fnv.New64a()
	_, _ = hash.Write([]byte(domain))
	var encoded [8]byte
	binary.BigEndian.PutUint64(encoded[:], uint64(scenarioSeed))
	_, _ = hash.Write(encoded[:])
	return int64(hash.Sum64())
}

// planetAt builds one planet. Radius is filled in from the engine's own
// formula because placement has already reasoned about that exact size;
// game.New() overwrites it with the same value when it validates the map.
func planetAt(id int, position game.Vec2, owner, ships, production int) game.Planet {
	return game.Planet{
		ID:         id,
		Position:   position,
		Radius:     game.PlanetRadius(production),
		OwnerID:    owner,
		Ships:      ships,
		Production: production,
	}
}

// placePair samples candidate positions until it finds one whose mirrored twin
// is also legal, or gives up after maxPlacementAttempts.
//
// Both halves are tested before either is accepted. Accepting a position on the
// strength of its own legality alone would let a twin land overlapping
// something, and the map would lose the symmetry every later guarantee rests
// on. The pair is also checked against itself, since near the sun a planet and
// its mirror are each other's closest neighbour.
func placePair(rng *rand.Rand, config game.Config, placed []game.Planet, radius float64) (game.Vec2, game.Vec2, error) {
	for range maxPlacementAttempts {
		position := game.Vec2{
			X: rng.Float64() * config.Width,
			Y: rng.Float64() * config.Height,
		}
		mirror := mirrorAbout(config.Sun, position)
		if legalPlacement(config, placed, position, radius) &&
			legalPlacement(config, placed, mirror, radius) &&
			clear(position, radius, mirror, radius) {
			return position, mirror, nil
		}
	}
	return game.Vec2{}, game.Vec2{}, fmt.Errorf("no legal placement in %d attempts", maxPlacementAttempts)
}

// mirrorAbout reflects a point through centre — a 180-degree rotation, which
// on the standard 100x100 board maps (x, y) to (100-x, 100-y). It is written
// against the configured centre rather than those constants so a differently
// sized board mirrors correctly too.
func mirrorAbout(centre, p game.Vec2) game.Vec2 {
	return game.Vec2{X: 2*centre.X - p.X, Y: 2*centre.Y - p.Y}
}

// legalPlacement reports whether a planet of the given radius may sit at
// position: entirely on the board, clear of the sun, and clear of every planet
// already placed.
func legalPlacement(config game.Config, placed []game.Planet, position game.Vec2, radius float64) bool {
	// The whole disc must be inside the board, not just its centre: a planet
	// hanging over the edge would sit partly in the region that destroys
	// fleets.
	if position.X-radius < 0 || position.X+radius > config.Width ||
		position.Y-radius < 0 || position.Y+radius > config.Height {
		return false
	}
	// The sun is spaced against even though its hazard is disabled for this
	// variant. A planet overlapping it would still distort every path through
	// the centre of the board, and a later experiment re-enables the hazard on
	// these same maps, where an overlap would become lethal rather than merely
	// odd.
	if !clear(position, radius, config.Sun, config.SunRadius) {
		return false
	}
	for _, p := range placed {
		if !clear(position, radius, p.Position, p.Radius) {
			return false
		}
	}
	return true
}

// clear reports whether two bodies leave at least minSpacing of empty space
// between their surfaces. Measuring surface to surface rather than centre to
// centre is what keeps the gap honest when the two bodies differ in size.
func clear(a game.Vec2, radiusA float64, b game.Vec2, radiusB float64) bool {
	return a.Distance(b) >= radiusA+radiusB+minSpacing
}
