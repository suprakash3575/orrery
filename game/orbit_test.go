package game

import (
	"math"
	"math/rand"
	"testing"
)

// seededConfig returns the standard game with a fixed RNG, so every assertion
// below is about the draw's rules rather than about one lucky draw.
func seededConfig(seed int64) Config {
	config := DefaultConfig()
	config.Rand = rand.New(rand.NewSource(seed))
	return config
}

// orbitingAt places a production-1 planet (radius 1) at the given distance and
// angle from the default sun at (50, 50). Any distance below 49 orbits.
func orbitingAt(id, owner int, distance, angle float64) Planet {
	position := Vec2{50 + distance*math.Cos(angle), 50 + distance*math.Sin(angle)}
	return Planet{ID: id, OwnerID: owner, Production: 1, Position: position}
}

// orbitingPlanets is the setup most of these tests share: four planets that all
// orbit, listed in an order that is deliberately not ID order.
func orbitingPlanets() []Planet {
	return []Planet{
		orbitingAt(3, 0, 30, 0),
		orbitingAt(1, 1, 15, 1),
		orbitingAt(4, 2, 25, 2),
		orbitingAt(2, 3, 20, 3),
	}
}

func ratesByID(g *Game) map[int]float64 {
	rates := map[int]float64{}
	for _, p := range g.Planets {
		if p.Orbit != nil {
			rates[p.ID] = p.Orbit.RadiansTurn
		}
	}
	return rates
}

// TestNewAssignsReproducibleOrbitRates is the property the whole seeded-replay
// guarantee rests on: same seed, same rates.
func TestNewAssignsReproducibleOrbitRates(t *testing.T) {
	a, err := New(seededConfig(42), orbitingPlanets(), nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := New(seededConfig(42), orbitingPlanets(), nil)
	if err != nil {
		t.Fatal(err)
	}
	for id, want := range ratesByID(a) {
		if got := ratesByID(b)[id]; got != want {
			t.Fatalf("planet %d drew %v on the first run and %v on the second", id, want, got)
		}
	}
}

// TestNewOverwritesSuppliedOrbitRate pins that orbits are engine-computed: a
// caller-supplied Orbit carries no weight, the same way a supplied Radius does
// not.
func TestNewOverwritesSuppliedOrbitRate(t *testing.T) {
	const supplied = 0.5 // far outside [MinOrbitRate, MaxOrbitRate)
	planet := orbitingAt(1, 0, 10, 0)
	planet.Orbit = &Orbit{Radius: 10, RadiansTurn: supplied}
	g, err := New(seededConfig(7), []Planet{planet}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := g.Planets[0].Orbit.RadiansTurn; got == supplied {
		t.Fatalf("caller-supplied rate %v survived construction; New() must assign its own", supplied)
	}
}

// TestNewOrbitRatesStayInDocumentedRange checks the draw against the reference
// engine's range: [MinOrbitRate, MaxOrbitRate).
func TestNewOrbitRatesStayInDocumentedRange(t *testing.T) {
	// Many seeds, since one draw proves nothing about a range.
	for seed := int64(0); seed < 50; seed++ {
		g, err := New(seededConfig(seed), orbitingPlanets(), nil)
		if err != nil {
			t.Fatal(err)
		}
		for id, rate := range ratesByID(g) {
			if rate < MinOrbitRate || rate >= MaxOrbitRate {
				t.Fatalf("seed %d planet %d drew rate %v, want [%v, %v)", seed, id, rate, MinOrbitRate, MaxOrbitRate)
			}
		}
	}
}

// TestNewOrbitRatesAreShared guards the reference engine's lockstep rule: every orbiting
// planet uses the one Game.OrbitRate, so they all turn the same way at the
// same speed.
func TestNewOrbitRatesAreShared(t *testing.T) {
	for seed := int64(0); seed < 50; seed++ {
		g, err := New(seededConfig(seed), orbitingPlanets(), nil)
		if err != nil {
			t.Fatal(err)
		}
		for id, rate := range ratesByID(g) {
			if rate != g.OrbitRate {
				t.Fatalf("seed %d planet %d has rate %v, want the shared rate %v", seed, id, rate, g.OrbitRate)
			}
		}
	}
}

// TestNewLeavesStaticPlanetsAlone covers both halves of "a planet outside the
// limit does not orbit": it keeps standing still, and — because orbit
// assignment draws no randomness per planet — adding it cannot shift the rate
// every other planet receives.
func TestNewLeavesStaticPlanetsAlone(t *testing.T) {
	// (10, 10) is about 56.6 from the sun, well outside OrbitRadiusLimit.
	withStatic := append(orbitingPlanets(),
		Planet{ID: 9, Position: Vec2{10, 10}, OwnerID: 0, Production: 1})

	orbitingOnly, err := New(seededConfig(11), orbitingPlanets(), nil)
	if err != nil {
		t.Fatal(err)
	}
	mixed, err := New(seededConfig(11), withStatic, nil)
	if err != nil {
		t.Fatal(err)
	}

	if static := mixed.planet(9); static.Orbit != nil {
		t.Fatal("a planet outside OrbitRadiusLimit must not be given an Orbit")
	}
	for id, want := range ratesByID(orbitingOnly) {
		if got := ratesByID(mixed)[id]; got != want {
			t.Fatalf("adding a static planet changed planet %d's rate from %v to %v", id, want, got)
		}
	}
}

// TestNewOrbitRatesIgnoreSetupOrder checks that slice order, a caller
// accident, cannot change which rate a planet receives.
func TestNewOrbitRatesIgnoreSetupOrder(t *testing.T) {
	forward := orbitingPlanets()
	reversed := orbitingPlanets()
	for i, j := 0, len(reversed)-1; i < j; i, j = i+1, j-1 {
		reversed[i], reversed[j] = reversed[j], reversed[i]
	}

	a, err := New(seededConfig(3), forward, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := New(seededConfig(3), reversed, nil)
	if err != nil {
		t.Fatal(err)
	}
	for id, want := range ratesByID(a) {
		if got := ratesByID(b)[id]; got != want {
			t.Fatalf("planet %d got rate %v listed first and %v listed last; ID-to-rate mapping must not depend on setup order", id, want, got)
		}
	}
}

// TestNewOrbitsDisabledDrawsNothing covers the static configuration RL V1
// uses. No planet orbits, and skipping the draw entirely is what keeps later
// draws from the same seed (comet spawns) lined up with a game that has
// orbits enabled.
func TestNewOrbitsDisabledDrawsNothing(t *testing.T) {
	config := seededConfig(5)
	config.OrbitsEnabled = false
	g, err := New(config, orbitingPlanets(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if rates := ratesByID(g); len(rates) != 0 || g.OrbitRate != 0 {
		t.Fatalf("orbits disabled: %d planets orbit at rate %v, want none at 0", len(rates), g.OrbitRate)
	}
	// A fresh generator on the same seed must still be at its first value: if
	// construction had drawn, the two would have diverged.
	if got, want := g.Config.Rand.Float64(), rand.New(rand.NewSource(5)).Float64(); got != want {
		t.Fatalf("construction consumed randomness with orbits disabled: next draw %v, want %v", got, want)
	}
}
