package arena

import (
	"reflect"
	"testing"

	"orrery/agent"
	"orrery/game"
)

func TestFixedScenarioInitialState(t *testing.T) {
	factory := FixedScenarioFactory{}
	if got := factory.ID(); got != FixedScenarioV1 {
		t.Fatalf("ID = %q, want %q", got, FixedScenarioV1)
	}

	g, err := factory.New(42)
	if err != nil {
		t.Fatal(err)
	}
	if g.Config.Rand == nil {
		t.Fatal("scenario did not install a seeded engine RNG")
	}
	if len(g.Comets) != 0 {
		t.Fatalf("initial comets = %#v, want none", g.Comets)
	}
	if got, want := len(g.Planets), 6; got != want {
		t.Fatalf("planet count = %d, want %d", got, want)
	}

	standard := game.DefaultConfig()
	gotConfig := agent.NewView(g).Config
	wantConfig := agent.ConfigView{
		Width:                standard.Width,
		Height:               standard.Height,
		Sun:                  standard.Sun,
		SunRadius:            standard.SunRadius,
		SunHazardEnabled:     standard.SunHazardEnabled,
		CometSpawningEnabled: standard.CometSpawningEnabled,
		MaxTurns:             standard.MaxTurns,
		MaxFleetSpeed:        standard.MaxFleetSpeed,
		AngularVelocity:      g.OrbitRate,
		CometSpeed:           standard.CometSpeed,
	}
	if !reflect.DeepEqual(gotConfig, wantConfig) {
		t.Fatalf("config = %#v, want standard %#v", gotConfig, wantConfig)
	}

	wantOwners := []int{0, 1, game.NeutralPlayer, game.NeutralPlayer, game.NeutralPlayer, game.NeutralPlayer}
	for i, planet := range g.Planets {
		if planet.ID != i+1 || planet.OwnerID != wantOwners[i] {
			t.Fatalf("planet %d = %#v, want ID %d owner %d", i, planet, i+1, wantOwners[i])
		}
		if planet.Orbit != nil {
			t.Fatalf("planet %d unexpectedly orbits", planet.ID)
		}
	}
}

func TestFixedScenarioSeededGamesAreIndependentAndDeterministic(t *testing.T) {
	factory := FixedScenarioFactory{}
	first, err := factory.New(73)
	if err != nil {
		t.Fatal(err)
	}
	second, err := factory.New(73)
	if err != nil {
		t.Fatal(err)
	}
	if first.Config.Rand == second.Config.Rand {
		t.Fatal("two games share an engine RNG")
	}

	for turn := range 51 {
		if err := first.AdvanceTurn(nil); err != nil {
			t.Fatalf("first turn %d: %v", turn, err)
		}
		if err := second.AdvanceTurn(nil); err != nil {
			t.Fatalf("second turn %d: %v", turn, err)
		}
		if got, want := agent.NewView(first), agent.NewView(second); !reflect.DeepEqual(got, want) {
			t.Fatalf("same-seed views differ after turn %d", turn)
		}
	}
	if len(first.Comets) != 4 {
		t.Fatalf("comets after scheduled spawn = %d, want 4", len(first.Comets))
	}

	first.Planets[0].Ships++
	if first.Planets[0].Ships == second.Planets[0].Ships {
		t.Fatal("games share mutable planet state")
	}
}

func TestFixedScenarioSeedControlsScheduledComets(t *testing.T) {
	factory := FixedScenarioFactory{}
	first, err := factory.New(1)
	if err != nil {
		t.Fatal(err)
	}
	second, err := factory.New(2)
	if err != nil {
		t.Fatal(err)
	}
	for range 51 {
		if err := first.AdvanceTurn(nil); err != nil {
			t.Fatal(err)
		}
		if err := second.AdvanceTurn(nil); err != nil {
			t.Fatal(err)
		}
	}
	if reflect.DeepEqual(first.Comets, second.Comets) {
		t.Fatal("different scenario seeds produced identical scheduled comets")
	}
}
