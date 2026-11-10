package game

import (
	"math"
	"testing"
)

func testGame(t *testing.T, planets ...Planet) *Game {
	t.Helper()
	g, err := New(DefaultConfig(), planets, nil)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestProduction(t *testing.T) {
	g := testGame(t, Planet{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 4, Production: 3}, bystander(99, 2))
	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	if got := g.Planets[0].Ships; got != 7 {
		t.Fatalf("ships = %d, want 7", got)
	}
}

func TestFleetSpeed(t *testing.T) {
	const tolerance = 1e-12
	tests := []struct {
		name     string
		ships    int
		maxSpeed float64
		want     float64
	}{
		{name: "negative ships do not move", ships: -1, maxSpeed: 6, want: 0},
		{name: "zero ships do not move", ships: 0, maxSpeed: 6, want: 0},
		{name: "one ship starts at base speed", ships: 1, maxSpeed: 8, want: 1},
		{name: "ten ships follow curve", ships: 10, maxSpeed: 6, want: 1.9622504486493764},
		{name: "one hundred ships follow curve", ships: 100, maxSpeed: 6, want: 3.721655269759087},
		{name: "one thousand ships reach maximum", ships: 1000, maxSpeed: 8, want: 8},
		{name: "ships above curve stay capped", ships: 1500, maxSpeed: 8, want: 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FleetSpeed(tt.ships, tt.maxSpeed); math.Abs(got-tt.want) > tolerance {
				t.Fatalf("FleetSpeed(%d, %v) = %v, want %v", tt.ships, tt.maxSpeed, got, tt.want)
			}
		})
	}
}

func TestGameFleetSpeedUsesConfiguredMaximum(t *testing.T) {
	for _, maxSpeed := range []float64{1, 2.5, 6, 12} {
		g := &Game{Config: Config{MaxFleetSpeed: maxSpeed}}
		for _, ships := range []int{-1, 0, 1, 2, 10, 100, 999, 1000, 1500} {
			want := FleetSpeed(ships, maxSpeed)
			if got := g.FleetSpeed(ships); got != want {
				t.Fatalf("max speed %v, ships %d: Game.FleetSpeed = %v, FleetSpeed = %v", maxSpeed, ships, got, want)
			}
		}
	}
}

func TestLaunchAndCapture(t *testing.T) {
	// Setup: two planets 3 units apart on the X axis, both Production: 1, so
	// New() derives Radius = 1 + ln(1) = 1 for each — planet 2 sits at
	// distance 3, well within one turn's fleet travel.
	//
	//   planet 1 (owner 1, 7 ships)        planet 2 (owner 2, 2 ships)
	//        (10,10) r=1  ---------3 units--------  (13,10) r=1
	//
	g := testGame(t, Planet{ID: 1, Position: Vec2{10, 10}, OwnerID: 1, Ships: 7, Production: 1}, Planet{ID: 2, Position: Vec2{13, 10}, OwnerID: 2, Ships: 2, Production: 1})

	// Angle 0 points along +X (see Action.Angle doc), i.e. straight from
	// planet 1 toward planet 2. Launching 5 of planet 1's 7 ships:
	//   - the fleet spawns just outside planet 1's radius plus spawnEpsilon,
	//     at x≈11.01
	//   - FleetSpeed(5) ≈ 1.56, so after this turn's single move it's at
	//     roughly x=12.57 — within planet 2's radius (1), so it "arrives"
	//     this same turn (see resolveArrivals' endpoint check).
	if err := g.AdvanceTurn([]Action{{PlayerID: 1, PlanetID: 1, Ships: 5, Angle: 0}}); err != nil {
		t.Fatal(err)
	}

	// Turn order matters here: produce() runs before the fleet's arrival is
	// resolved, so planet 2's garrison grows from 2 to 3 (2 + Production: 1)
	// before combat. The attack is therefore 5 vs. 3, not 5 vs. 2 — the
	// winning attacker (owner 1) is left with 5-3=2 ships and flips
	// ownership of planet 2.
	if got := g.Planets[1].OwnerID; got != 1 {
		t.Fatalf("owner = %d, want 1", got)
	}
	if got := g.Planets[1].Ships; got != 2 {
		t.Fatalf("ships = %d, want 2", got)
	}
}
func TestSunHazardSwitch(t *testing.T) {
	if !DefaultConfig().SunHazardEnabled {
		t.Fatal("standard game should enable the sun hazard")
	}
	for _, test := range []struct {
		name       string
		enabled    bool
		wantFleets int
	}{
		{name: "enabled destroys crossing fleet", enabled: true, wantFleets: 0},
		{name: "disabled lets same crossing survive", enabled: false, wantFleets: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			config := DefaultConfig()
			config.SunHazardEnabled = test.enabled
			g, err := New(config, []Planet{
				{ID: 1, Position: Vec2{43, 50}, OwnerID: 1, Ships: 2, Production: 1},
				bystander(99, 2),
			}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if err := g.AdvanceTurn([]Action{{PlayerID: 1, PlanetID: 1, Ships: 2, Angle: 0}}); err != nil {
				t.Fatal(err)
			}
			if got := len(g.Fleets); got != test.wantFleets {
				t.Fatalf("fleets after sun crossing = %d, want %d", got, test.wantFleets)
			}
		})
	}
}
