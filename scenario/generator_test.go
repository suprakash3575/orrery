package scenario

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"orrery/game"
)

// sweepSeeds is the breadth used for the legality sweep. Sampling a handful of
// seeds cannot show that every seed in a training range is legal, so the sweep
// is wide enough to stand in for that guarantee rather than merely spot-check
// it.
const sweepSeeds = 10000

// goldenPlanet is the stable subset of game.Planet controlled by the map
// generator. Keeping this fixture type local to the test prevents unrelated
// engine-state fields from changing generator-v1's golden hashes.
type goldenPlanet struct {
	ID         int
	X          float64
	Y          float64
	OwnerID    int
	Ships      int
	Production int
}

// TestSameSeedProducesIdenticalMap is the property every replay depends on: a
// recorded seed is only a useful record if it rebuilds the same board later.
// Comparing serialized planets rather than field by field also catches a new
// field being added and left unset by one path.
func TestSameSeedProducesIdenticalMap(t *testing.T) {
	for _, seed := range []int64{0, 1, 42, 999, 1_000_000} {
		first, err := GeneratorV1{}.New(seed)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		second, err := GeneratorV1{}.New(seed)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		firstJSON, err := json.Marshal(first.Planets)
		if err != nil {
			t.Fatal(err)
		}
		secondJSON, err := json.Marshal(second.Planets)
		if err != nil {
			t.Fatal(err)
		}
		if string(firstJSON) != string(secondJSON) {
			t.Fatalf("seed %d produced different maps:\n%s\n%s", seed, firstJSON, secondJSON)
		}
	}
}

// TestGeneratorV1GoldenMaps freezes the public (generator ID, seed) to map
// contract. A same-process determinism test would still pass if a refactor
// changed every recorded seed's map in the same way; these hashes catch that
// replay-breaking change and require an intentional generator version bump.
func TestGeneratorV1GoldenMaps(t *testing.T) {
	tests := []struct {
		seed int64
		want string
	}{
		{seed: 0, want: "c43fe0bbfb6a9f3609ea024c59624ef0b7c478b8f5849c418894ead935279013"},
		{seed: 1, want: "a0328e53141c5174d59fd7d361e36e453d15fb2c6c64b426e0480844d4097a77"},
		{seed: 42, want: "8faea0f5ea5405556cd9cd660c0fea1040d3c731255ddef6d3ad40f08314653c"},
	}

	for _, tt := range tests {
		g, err := GeneratorV1{}.New(tt.seed)
		if err != nil {
			t.Fatalf("seed %d: %v", tt.seed, err)
		}
		encoded, err := json.Marshal(goldenMapProjection(g.Planets))
		if err != nil {
			t.Fatal(err)
		}
		got := fmt.Sprintf("%x", sha256.Sum256(encoded))
		if got != tt.want {
			t.Errorf("seed %d map hash = %s, want %s", tt.seed, got, tt.want)
		}
	}
}

// goldenMapProjection copies only the planet details that generator-v1
// promises to reproduce for a given seed. This keeps the saved map hashes from
// changing when game.Planet gains or rearranges unrelated fields. A change to
// any copied detail still changes the hash and requires a new generator version.
func goldenMapProjection(planets []game.Planet) []goldenPlanet {
	projected := make([]goldenPlanet, len(planets))
	for i, planet := range planets {
		projected[i] = goldenPlanet{
			ID:         planet.ID,
			X:          planet.Position.X,
			Y:          planet.Position.Y,
			OwnerID:    planet.OwnerID,
			Ships:      planet.Ships,
			Production: planet.Production,
		}
	}
	return projected
}

// TestRNGDomainsAreStableAndSeparate freezes the seed-derivation formula and
// verifies that map generation and engine runtime cannot begin from the same
// private random sequence.
func TestRNGDomainsAreStableAndSeparate(t *testing.T) {
	const scenarioSeed int64 = -17
	const wantMapSeed int64 = -5908920341373977611
	const wantEngineSeed int64 = 1546811596372424791

	mapSeed := deriveRNGSeed(scenarioSeed, mapRNGDomain)
	engineSeed := deriveRNGSeed(scenarioSeed, engineRNGDomain)
	if mapSeed != wantMapSeed {
		t.Errorf("map seed = %d, want stable golden %d", mapSeed, wantMapSeed)
	}
	if engineSeed != wantEngineSeed {
		t.Errorf("engine seed = %d, want stable golden %d", engineSeed, wantEngineSeed)
	}
	if mapSeed == engineSeed {
		t.Fatal("map and engine RNG domains produced the same seed")
	}

	g, err := GeneratorV1{}.New(scenarioSeed)
	if err != nil {
		t.Fatal(err)
	}
	wantFirstEngineValue := rand.New(rand.NewSource(wantEngineSeed)).Int63()
	if got := g.Config.Rand.Int63(); got != wantFirstEngineValue {
		t.Errorf("first engine RNG value = %d, want %d", got, wantFirstEngineValue)
	}
}

// TestGeneratedMapsAreLegal sweeps a wide seed range and checks every rule the
// generator promises: the engine accepts the map, the planet count is one of
// the three legal sizes, both homes are present and identical, every neutral
// has an exact mirrored twin of equal value, and nothing overlaps anything.
func TestGeneratedMapsAreLegal(t *testing.T) {
	for seed := int64(0); seed < sweepSeeds; seed++ {
		g, err := GeneratorV1{}.New(seed)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		switch len(g.Planets) {
		case 8, 10, 12:
		default:
			t.Fatalf("seed %d: %d planets, want 8, 10 or 12", seed, len(g.Planets))
		}
		checkHomes(t, seed, g)
		checkNeutralPairs(t, seed, g)
		checkSpacing(t, seed, g)
	}
}

// checkHomes verifies the two players start equal in everything the generator
// controls. Any asymmetry here would let a policy win by seat rather than by
// play, which is exactly the confound symmetric maps exist to remove.
func checkHomes(t *testing.T, seed int64, g *game.Game) {
	t.Helper()
	first, second := g.Planets[0], g.Planets[1]
	if first.OwnerID != 0 || second.OwnerID != 1 {
		t.Fatalf("seed %d: home owners = %d, %d, want 0, 1", seed, first.OwnerID, second.OwnerID)
	}
	for _, home := range []game.Planet{first, second} {
		if home.Ships != homeShips || home.Production != homeProduction {
			t.Fatalf("seed %d: home %d has %d ships and production %d, want %d and %d",
				seed, home.ID, home.Ships, home.Production, homeShips, homeProduction)
		}
	}
	checkMirrored(t, seed, g.Config.Sun, first, second)
}

// checkNeutralPairs verifies the map is built from mirrored pairs: planets 3
// and 4 are twins, 5 and 6 are twins, and so on. Equal production matters as
// much as mirrored position — a twin worth more than its counterpart would be
// an asymmetry the geometry alone would not reveal.
func checkNeutralPairs(t *testing.T, seed int64, g *game.Game) {
	t.Helper()
	for i := 2; i < len(g.Planets); i += 2 {
		first, second := g.Planets[i], g.Planets[i+1]
		if first.OwnerID != game.NeutralPlayer || second.OwnerID != game.NeutralPlayer {
			t.Fatalf("seed %d: planets %d and %d owned by %d and %d, want neutral",
				seed, first.ID, second.ID, first.OwnerID, second.OwnerID)
		}
		if first.Production != second.Production {
			t.Fatalf("seed %d: twins %d and %d have production %d and %d, want equal",
				seed, first.ID, second.ID, first.Production, second.Production)
		}
		if first.Production < minNeutralProduction || first.Production > maxNeutralProduction {
			t.Fatalf("seed %d: planet %d production %d out of range", seed, first.ID, first.Production)
		}
		wantShips := neutralShipsPerProduction * first.Production
		if first.Ships != wantShips || second.Ships != wantShips {
			t.Fatalf("seed %d: twins %d and %d have %d and %d ships, want %d",
				seed, first.ID, second.ID, first.Ships, second.Ships, wantShips)
		}
		checkMirrored(t, seed, g.Config.Sun, first, second)
	}
}

// checkMirrored verifies two planets sit exactly 180 degrees apart about the
// sun. The tolerance absorbs only floating-point noise from the reflection
// arithmetic, not any real asymmetry.
//
// The reflection is written out here rather than calling the generator's own
// mirrorAbout: a test that checks a function using that same function proves
// only self-consistency, and would pass a mirror that was wrong in both places.
func checkMirrored(t *testing.T, seed int64, sun game.Vec2, first, second game.Planet) {
	t.Helper()
	const tolerance = 1e-9
	want := game.Vec2{X: 2*sun.X - first.Position.X, Y: 2*sun.Y - first.Position.Y}
	if math.Abs(second.Position.X-want.X) > tolerance || math.Abs(second.Position.Y-want.Y) > tolerance {
		t.Fatalf("seed %d: planet %d at %v is not the mirror of planet %d at %v (want %v)",
			seed, second.ID, second.Position, first.ID, first.Position, want)
	}
}

// requiredGap is the empty space the generator promises between any two body
// surfaces. It is written as a literal rather than read from the generator's
// minSpacing so that this test states the contract independently: checking the
// code against its own constant would accept a build that changed both.
const requiredGap = 2.0

// checkSpacing verifies the geometry the engine itself will not check: every
// planet fully on the board, clear of the sun, and clear of every other planet.
// The engine validates ownership and production but never positions, so an
// overlapping map would be accepted and only misbehave later.
//
// The distance requirement is recomputed here instead of calling the
// generator's clear(): a spacing bug inside that helper would otherwise hide
// itself, since the same wrong rule would be used to place planets and then to
// approve the placement.
func checkSpacing(t *testing.T, seed int64, g *game.Game) {
	t.Helper()
	separated := func(a game.Vec2, radiusA float64, b game.Vec2, radiusB float64) bool {
		return a.Distance(b) >= radiusA+radiusB+requiredGap
	}
	for i, p := range g.Planets {
		if p.Position.X-p.Radius < 0 || p.Position.X+p.Radius > g.Config.Width ||
			p.Position.Y-p.Radius < 0 || p.Position.Y+p.Radius > g.Config.Height {
			t.Fatalf("seed %d: planet %d at %v radius %v is not fully on the board",
				seed, p.ID, p.Position, p.Radius)
		}
		if !separated(p.Position, p.Radius, g.Config.Sun, g.Config.SunRadius) {
			t.Fatalf("seed %d: planet %d at %v is too close to the sun: %v apart, need %v",
				seed, p.ID, p.Position, p.Position.Distance(g.Config.Sun),
				p.Radius+g.Config.SunRadius+requiredGap)
		}
		for _, other := range g.Planets[i+1:] {
			if !separated(p.Position, p.Radius, other.Position, other.Radius) {
				t.Fatalf("seed %d: planets %d and %d are too close: %v apart, need %v",
					seed, p.ID, other.ID, p.Position.Distance(other.Position),
					p.Radius+other.Radius+requiredGap)
			}
		}
	}
}

// TestDifferentSeedsProduceDifferentMaps guards against a generator that is
// reproducible but degenerate. Determinism alone would be satisfied by
// returning one fixed map for every seed, which would train a policy to
// memorize a single board instead of learning to play.
func TestDifferentSeedsProduceDifferentMaps(t *testing.T) {
	const seeds = 1000
	seen := make(map[string]int64, seeds)
	for seed := int64(0); seed < seeds; seed++ {
		g, err := GeneratorV1{}.New(seed)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		encoded, err := json.Marshal(g.Planets)
		if err != nil {
			t.Fatal(err)
		}
		if previous, duplicate := seen[string(encoded)]; duplicate {
			t.Fatalf("seeds %d and %d produced identical maps", previous, seed)
		}
		seen[string(encoded)] = seed
	}
}

// TestGeneratedGamesUseTheStaticVariant pins the configuration the learning
// environment depends on. A map that quietly enabled comets or orbits would
// still play, so nothing else in the suite would fail — but the environment
// would no longer be the static two-player problem the pipeline is built for.
func TestGeneratedGamesUseTheStaticVariant(t *testing.T) {
	g, err := GeneratorV1{}.New(7)
	if err != nil {
		t.Fatal(err)
	}
	if g.Config.SunHazardEnabled {
		t.Error("sun hazard should be disabled")
	}
	if g.Config.CometSpawningEnabled {
		t.Error("comet spawning should be disabled")
	}
	if g.Config.OrbitsEnabled || g.OrbitRate != 0 {
		t.Errorf("orbits enabled = %v at rate %v, want disabled at 0", g.Config.OrbitsEnabled, g.OrbitRate)
	}
	if len(g.Comets) != 0 {
		t.Errorf("comets = %d, want 0", len(g.Comets))
	}
	for _, p := range g.Planets {
		if p.Orbit != nil {
			t.Errorf("planet %d orbits, want static", p.ID)
		}
	}
}

// TestGeneratedGameStaysCometFree runs a generated map past the first
// scheduled spawn turn. The configuration assertion above only proves the flag
// is set; this proves the flag actually holds for a whole match, which is what
// the environment relies on.
func TestGeneratedGameStaysCometFree(t *testing.T) {
	g, err := GeneratorV1{}.New(11)
	if err != nil {
		t.Fatal(err)
	}
	for g.Turn <= 50 {
		if err := g.AdvanceTurn(nil); err != nil {
			t.Fatal(err)
		}
	}
	if len(g.Comets) != 0 {
		t.Fatalf("comets after turn 50 = %d, want 0", len(g.Comets))
	}
}

// TestIDsAreSequentialFromOne documents the ordering the observation builder
// will later rely on to map a body slot back to an engine ID: homes first,
// then each neutral pair, numbered from 1 with no gaps.
func TestIDsAreSequentialFromOne(t *testing.T) {
	g, err := GeneratorV1{}.New(3)
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range g.Planets {
		if p.ID != i+1 {
			t.Fatalf("planet at index %d has ID %d, want %d", i, p.ID, i+1)
		}
	}
}

// TestPlacementGivesUpRatherThanLooping proves the attempt cap is real. A
// board too small to hold a legal pair must return an error, since the
// alternative — retrying forever — would hang a training run with no
// diagnosis.
func TestPlacementGivesUpRatherThanLooping(t *testing.T) {
	config := game.DefaultConfig()
	// A board barely larger than the sun leaves nowhere legal to stand.
	config.Width, config.Height = 25, 25
	config.Sun = game.Vec2{X: 12.5, Y: 12.5}

	rng := rand.New(rand.NewSource(1))
	if _, err := generatePlanets(rng, config); err == nil {
		t.Fatal("expected an error when no legal placement exists")
	}
}
