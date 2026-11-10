package scenario

import (
	"math/rand"
	"slices"
	"testing"

	"orrery/game"
)

func TestV2PlanetGroupOrderAndGeometry(t *testing.T) {
	group := v2PlanetGroup(game.Vec2{X: 50, Y: 50}, 70, 80, 3, 12, 9)
	want := [...]game.Vec2{
		{X: 80, Y: 70},
		{X: 30, Y: 80},
		{X: 70, Y: 20},
		{X: 20, Y: 30},
	}
	for i, planet := range group {
		if planet.ID != 9+i || planet.Position != want[i] ||
			planet.OwnerID != game.NeutralPlayer || planet.Ships != 12 ||
			planet.Production != 3 || planet.Radius != game.PlanetRadius(3) {
			t.Errorf("planet %d = %+v, want ID %d at %+v with shared neutral values",
				i, planet, 9+i, want[i])
		}
	}
}

func TestGenerateV2GroupsAreSymmetricAndSeeded(t *testing.T) {
	config := game.DefaultConfig()
	var seenCounts [6]bool
	for seed := range 128 {
		mapSeed := deriveRNGSeed(int64(seed), v2MapRNGDomain)
		first, err := generateV2Groups(rand.New(rand.NewSource(mapSeed)), config)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		second, err := generateV2Groups(rand.New(rand.NewSource(mapSeed)), config)
		if err != nil {
			t.Fatalf("repeated seed %d: %v", seed, err)
		}
		if !slices.Equal(first, second) {
			t.Fatalf("seed %d produced different group lists", seed)
		}

		groups := len(first) / 4
		if len(first)%4 != 0 || groups < 5 || groups > 10 {
			t.Fatalf("seed %d produced %d planets, want 5..10 complete groups", seed, len(first))
		}
		seenCounts[groups-5] = true
		for i := 0; i < len(first); i += 4 {
			base := first[i]
			wantPositions := [...]game.Vec2{
				base.Position,
				{X: 2*config.Sun.X - base.Position.Y, Y: base.Position.X},
				{X: base.Position.Y, Y: 2*config.Sun.Y - base.Position.X},
				{X: 2*config.Sun.X - base.Position.X, Y: 2*config.Sun.Y - base.Position.Y},
			}
			for j, planet := range first[i : i+4] {
				if planet.ID != i+j+1 || planet.Position != wantPositions[j] ||
					planet.OwnerID != game.NeutralPlayer || planet.Ships != base.Ships ||
					planet.Production != base.Production || planet.Radius != game.PlanetRadius(base.Production) {
					t.Fatalf("seed %d group %d planet %d breaks symmetry or shared values: %+v",
						seed, i/4, j, planet)
				}
			}
			maxShips := 30
			if i/4 < 3 {
				maxShips = 99 // The outer static phase takes the lower of two 5..99 rolls.
			}
			if base.Production < 1 || base.Production > 5 || base.Ships < 5 || base.Ships > maxShips {
				t.Fatalf("seed %d group %d has production %d and ships %d out of range",
					seed, i/4, base.Production, base.Ships)
			}
		}
	}
	for offset, seen := range seenCounts {
		if !seen {
			t.Errorf("no sampled seed produced %d groups", offset+5)
		}
	}
}

// Check the quota against game.New's authoritative orbit assignment, not the
// generator's own orbit predicate. A last-slot retry must never add group 11.
func TestGenerateV2GroupsMeetStaticAndOrbitingQuotas(t *testing.T) {
	config := game.DefaultConfig()
	sawLastSlotOrbit := false
	for seed := range 1024 {
		scenarioSeed := int64(seed)
		mapSeed := deriveRNGSeed(scenarioSeed, v2MapRNGDomain)
		planets, err := generateV2Groups(rand.New(rand.NewSource(mapSeed)), config)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		config.Rand = rand.New(rand.NewSource(deriveRNGSeed(scenarioSeed, v2EngineRNGDomain)))
		g, err := game.New(config, planets, nil)
		if err != nil {
			t.Fatalf("seed %d was rejected by game.New: %v", seed, err)
		}
		staticGroups, orbitingGroups := 0, 0
		for i := 0; i < len(g.Planets); i += 4 {
			orbiting := g.Planets[i].Orbit != nil
			for _, planet := range g.Planets[i : i+4] {
				if (planet.Orbit != nil) != orbiting {
					t.Fatalf("seed %d group %d has mixed orbit status", seed, i/4)
				}
			}
			if orbiting {
				orbitingGroups++
			} else {
				staticGroups++
			}
			if i < 12 && orbiting {
				t.Fatalf("seed %d guaranteed static group %d orbits", seed, i/4)
			}
		}
		if staticGroups < 3 || orbitingGroups < 1 || len(g.Planets) > 40 {
			t.Fatalf("seed %d has %d static and %d orbiting groups across %d planets",
				seed, staticGroups, orbitingGroups, len(g.Planets))
		}
		if orbitingGroups == 1 && g.Planets[len(g.Planets)-4].Orbit != nil {
			sawLastSlotOrbit = true
		}
	}
	if !sawLastSlotOrbit {
		t.Fatal("sample never exercised a map whose only orbiting group uses the final slot")
	}
}
