package agent

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"orrery/game"
	"orrery/scenario"
)

// TestPredictorFromViewMatchesLiveGame checks that a bot predicting from its
// observation alone gets exactly the engine's answers. The boards are real
// four-player games played with random launches, so they have orbiting
// planets, fleets in flight and, after turn 50, comets.
func TestPredictorFromViewMatchesLiveGame(t *testing.T) {
	const horizon = 100
	checkTurns := map[int]bool{0: true, 40: true, 80: true, 120: true, 160: true}
	for _, seed := range []int64{1000000, 1000001} {
		g, err := scenario.GeneratorV2{FourPlayer: true}.New(seed)
		if err != nil {
			t.Fatal(err)
		}
		// rng drives the test's own launches, separate from the game's.
		rng := rand.New(rand.NewSource(seed))
		for g.Turn <= 160 && !g.Finished() {
			if checkTurns[g.Turn] {
				checkBoard(t, g, horizon)
			}
			if err := g.AdvanceTurn(randomLaunches(g, rng)); err != nil {
				t.Fatalf("seed %d turn %d: %v", seed, g.Turn, err)
			}
		}
	}
}

// checkBoard compares the predictor built from g's View with g itself.
func checkBoard(t *testing.T, g *game.Game, horizon int) {
	t.Helper()
	view := NewView(g)
	before := NewView(g)
	predictor, err := game.NewPredictor(view.Snapshot())
	if err != nil {
		t.Fatalf("turn %d: NewPredictor: %v", g.Turn, err)
	}
	// The predictor must hold its own copy: scrambling the View afterwards
	// must not change any answer below.
	for i := range view.Planets {
		view.Planets[i].Position = game.Vec2{X: -1, Y: -1}
		if view.Planets[i].Orbit != nil {
			view.Planets[i].Orbit.StartAngle += 1
		}
	}

	if got, want := predictor.PredictFleets(horizon), g.PredictFleets(horizon); !reflect.DeepEqual(got, want) {
		t.Errorf("turn %d: PredictFleets differs from the live game\n got %+v\nwant %+v", g.Turn, got, want)
	}
	// Every owned source against every other body, for a small and a large
	// fleet, since fleet size sets the speed and so the aim.
	bodies := before.Snapshot()
	for _, source := range bodyIDs(bodies, true) {
		for _, target := range bodyIDs(bodies, false) {
			for _, ships := range []int{5, 150} {
				gotAim, gotOK := predictor.LeadAim(source, target, ships)
				wantAim, wantOK := g.LeadAim(source, target, ships)
				if gotAim != wantAim || gotOK != wantOK {
					t.Errorf("turn %d: LeadAim(%d, %d, %d) = %+v, %v; live game %+v, %v",
						g.Turn, source, target, ships, gotAim, gotOK, wantAim, wantOK)
				}
			}
		}
	}

	// Asking must not change the live game: a fresh View equals the one taken
	// before any question.
	if after := NewView(g); !reflect.DeepEqual(after, before) {
		t.Errorf("turn %d: predicting changed the live game", g.Turn)
	}
}

// bodyIDs lists planet and comet IDs, only player-owned ones if owned is set.
func bodyIDs(s game.Snapshot, owned bool) []int {
	var ids []int
	for _, p := range s.Planets {
		if !owned || p.OwnerID != game.NeutralPlayer {
			ids = append(ids, p.ID)
		}
	}
	for _, c := range s.Comets {
		if !owned || c.OwnerID != game.NeutralPlayer {
			ids = append(ids, c.ID)
		}
	}
	return ids
}

// randomLaunches sends half of some garrisons in random directions, so the
// boards have fleets in flight from every player.
func randomLaunches(g *game.Game, rng *rand.Rand) []game.Action {
	var actions []game.Action
	for _, p := range g.Planets {
		if p.OwnerID != game.NeutralPlayer && p.Ships >= 10 && rng.Float64() < 0.3 {
			actions = append(actions, game.Action{PlayerID: p.OwnerID, PlanetID: p.ID, Ships: p.Ships / 2, Angle: rng.Float64() * 2 * math.Pi})
		}
	}
	return actions
}
