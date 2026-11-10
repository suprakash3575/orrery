package scenario

import (
	"math"
	"math/rand"
	"strings"
	"testing"

	"orrery/game"
)

func TestV2PairClearIncludesFullOrbit(t *testing.T) {
	config := game.DefaultConfig()
	planet := func(x, y float64) game.Planet {
		return game.Planet{Position: game.Vec2{X: x, Y: y}, Radius: 1, Production: 1}
	}
	tests := []struct {
		name string
		a, b game.Planet
		want bool
	}{
		{name: "initial overlap", a: planet(90, 80), b: planet(92, 82), want: false},
		{name: "exact initial clearance", a: planet(70, 70), b: planet(79, 70), want: true},
		{name: "two orbiting planets keep their separation", a: planet(70, 70), b: planet(30, 70), want: true},
		{name: "rotating planet later sweeps past static planet", a: planet(94, 50), b: planet(90, 80), want: false},
		{name: "rotating and static orbital rings are far apart", a: planet(70, 70), b: planet(90, 80), want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := v2PairClear(config, tt.a, tt.b); got != tt.want {
				t.Errorf("v2PairClear(%+v, %+v) = %v, want %v", tt.a.Position, tt.b.Position, got, tt.want)
			}
		})
	}
}

func TestV2LegalGroupRejectsSunEdgeAndPlacedPlanet(t *testing.T) {
	config := game.DefaultConfig()
	tests := []struct {
		name   string
		group  [4]game.Planet
		placed []game.Planet
		want   bool
	}{
		{name: "clear", group: v2PlanetGroup(config.Sun, 70, 80, 1, 10, 1), want: true},
		{name: "sun gap", group: v2PlanetGroup(config.Sun, 65, 65, 5, 10, 1), want: false},
		{name: "board edge", group: v2PlanetGroup(config.Sun, 99.5, 80, 1, 10, 1), want: false},
		{name: "existing planet", group: v2PlanetGroup(config.Sun, 70, 80, 1, 10, 1),
			placed: []game.Planet{{Position: game.Vec2{X: 82, Y: 70}, Radius: 1}}, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := v2LegalGroup(config, tt.placed, tt.group); got != tt.want {
				t.Errorf("v2LegalGroup = %v, want %v", got, tt.want)
			}
		})
	}
}

// Verify finished group layouts independently of the placement helper. The
// engine decides which bodies orbit, and the mixed pair check uses that result.
func TestGenerateV2GroupsKeepRequiredClearances(t *testing.T) {
	config := game.DefaultConfig()
	for seed := range 10000 {
		scenarioSeed := int64(seed)
		mapSeed := deriveRNGSeed(scenarioSeed, v2MapRNGDomain)
		planets, err := generateV2Groups(rand.New(rand.NewSource(mapSeed)), config)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		config.Rand = rand.New(rand.NewSource(deriveRNGSeed(scenarioSeed, v2EngineRNGDomain)))
		g, err := game.New(config, planets, nil)
		if err != nil {
			t.Fatalf("seed %d: game.New: %v", seed, err)
		}
		for i, a := range g.Planets {
			if a.Position.X-a.Radius < 0 || a.Position.X+a.Radius > config.Width ||
				a.Position.Y-a.Radius < 0 || a.Position.Y+a.Radius > config.Height {
				t.Fatalf("seed %d planet %d leaves the board", seed, a.ID)
			}
			if a.Position.Distance(config.Sun) < config.SunRadius+a.Radius+10 {
				t.Fatalf("seed %d planet %d is too close to the sun", seed, a.ID)
			}
			for _, b := range g.Planets[:i] {
				gap := a.Radius + b.Radius + 7
				if a.Position.Distance(b.Position) < gap {
					t.Fatalf("seed %d planets %d and %d violate initial clearance", seed, a.ID, b.ID)
				}
				if (a.Orbit == nil) != (b.Orbit == nil) &&
					math.Abs(a.Position.Distance(config.Sun)-b.Position.Distance(config.Sun)) < gap {
					t.Fatalf("seed %d planets %d and %d violate full-orbit clearance", seed, a.ID, b.ID)
				}
			}
		}
	}
}

func TestGenerateV2GroupsFailsWhenBoardCannotHoldStaticGroup(t *testing.T) {
	config := game.DefaultConfig()
	config.Width, config.Height = 10, 10
	config.Sun = game.Vec2{X: 5, Y: 5}
	_, err := generateV2Groups(rand.New(rand.NewSource(1)), config)
	if err == nil || !strings.Contains(err.Error(), "static group 0") {
		t.Fatalf("impossible board error = %v, want bounded static-placement failure", err)
	}
}
