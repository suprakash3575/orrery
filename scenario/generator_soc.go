package scenario

import (
	"fmt"
	"math/rand"

	"orrery/game"
)

// GeneratorV2ID identifies the seeded full-rules generator. Its independent
// map algorithm and RNG domains leave generator-v1 replays unchanged.
const GeneratorV2ID = "generator-v2"

// GeneratorV2FourPlayerID identifies the four-player variant of generator-v2.
// It has its own ID because the same seed starts with four owned homes instead
// of two, which is a different initial state for replay purposes.
const GeneratorV2FourPlayerID = "generator-v2-4p"

const (
	// V2 uses different map and engine streams from V1. The domain names are
	// part of its replay identity: a map draw cannot shift future engine draws,
	// and adding V2 draws cannot change any generator-v1 replay.
	v2MapRNGDomain    = "orrery-generator-v2-map"
	v2EngineRNGDomain = "orrery-generator-v2-engine"
)

// GeneratorV2 builds two- or four-player maps with fourfold planet symmetry,
// static and orbiting planets, the sun hazard, and scheduled comets.
type GeneratorV2 struct {
	// FourPlayer gives all four planets of the home group to players 0 to 3.
	// The zero value is the two-player game. Both variants share the map and
	// engine streams, so one seed places the same planets in each and only the
	// home owners differ.
	FourPlayer bool
}

// ID returns the versioned scenario identity.
func (f GeneratorV2) ID() string {
	if f.FourPlayer {
		return GeneratorV2FourPlayerID
	}
	return GeneratorV2ID
}

// Players returns how many players every map from this variant seats: 4 with
// FourPlayer set, 2 otherwise.
func (f GeneratorV2) Players() int {
	if f.FourPlayer {
		return 4
	}
	return 2
}

// New gives the map and engine independent random streams derived from one
// seed. Each call creates fresh generators, so concurrent games cannot share
// random state. The engine receives its stream before construction because
// game.New draws the shared orbit rate during setup.
func (f GeneratorV2) New(seed int64) (*game.Game, error) {
	config := game.DefaultConfig()
	// Pin the V2 rule switches explicitly so a future default-config change
	// cannot silently turn a recorded V2 seed into a different game variant.
	config.SunHazardEnabled = true
	config.OrbitsEnabled = true
	config.CometSpawningEnabled = true
	config.Rand = rand.New(rand.NewSource(deriveRNGSeed(seed, v2EngineRNGDomain)))
	mapRNG := rand.New(rand.NewSource(deriveRNGSeed(seed, v2MapRNGDomain)))

	planets, err := generateV2Planets(mapRNG, config, f.FourPlayer)
	if err != nil {
		return nil, fmt.Errorf("scenario seed %d: %w", seed, err)
	}
	g, err := game.New(config, planets, nil)
	if err != nil {
		return nil, fmt.Errorf("scenario seed %d: %w", seed, err)
	}
	return g, nil
}

// generateV2Planets finishes all neutral placement before choosing homes.
// The home draw uses the same map stream after placement, while game.New
// receives a separate engine stream for orbits and later comet waves.
func generateV2Planets(rng *rand.Rand, config game.Config, fourPlayer bool) ([]game.Planet, error) {
	planets, err := generateV2Groups(rng, config)
	if err != nil {
		return nil, err
	}
	if err := assignV2Homes(rng, planets, fourPlayer); err != nil {
		return nil, err
	}
	return planets, nil
}
