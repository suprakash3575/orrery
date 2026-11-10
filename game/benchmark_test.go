package game_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"orrery/agent"
	"orrery/game"
	"orrery/scenario"
)

// These benchmarks measure fleet prediction and real turns on the same
// mid-game states, so a change to the collision path can be compared before
// and after on identical fixtures. They live in the external test package
// because building a realistic state needs the scenario generators and the
// scripted agents, which themselves import game.

// boardFactory is the part of a scenario generator the fixtures use.
type boardFactory interface {
	New(seed int64) (*game.Game, error)
}

// benchmarkBoards are the fixture boards. The static board has no orbits and
// no comets, so every body is tested as a fixed circle; the moving boards
// exercise the relative-motion test against orbiting planets and comets.
var benchmarkBoards = []struct {
	name    string
	factory boardFactory
	players int
}{
	{name: "static-2p", factory: scenario.GeneratorV1{}, players: 2},
	{name: "moving-2p", factory: scenario.GeneratorV2{}, players: 2},
	{name: "moving-4p", factory: scenario.GeneratorV2{FourPlayer: true}, players: 4},
}

const (
	// benchmarkSeed picks one generated board per fixture; any seed works, but
	// a fixed one keeps before-and-after runs on identical states.
	benchmarkSeed int64 = 1_000_001
	// benchmarkWarmTurns plays the fixture past the opening, so many fleets
	// are in flight. It stays clear of the comet waves at turns 50 and 150 for
	// the timed AdvanceTurn window that follows it.
	benchmarkWarmTurns = 100
	// benchmarkTimedTurns is how many real turns one AdvanceTurn operation
	// plays from the warmed state.
	benchmarkTimedTurns = 10
)

// benchmarkFleetCounts are the fleet populations PredictFleets is timed at.
// Zero and one cover the cheap cases a change must not slow down; -1 means
// every fleet in the fixture.
var benchmarkFleetCounts = []int{0, 1, 8, 64, -1}

// benchmarkHorizon is mlenv's prediction bound: the board's diagonal, the
// longest straight crossing a fleet can make.
func benchmarkHorizon(g *game.Game) int {
	return int(math.Ceil(math.Hypot(g.Config.Width, g.Config.Height)))
}

// playBenchmarkFixture plays a freshly generated board for turns turns with a
// heuristic agent in every seat. It returns the game and the actions every
// turn used, so replayBenchmarkFixture can rebuild exactly the same state.
func playBenchmarkFixture(b *testing.B, factory boardFactory, players, turns int) (*game.Game, [][]game.Action) {
	b.Helper()
	g, err := factory.New(benchmarkSeed)
	if err != nil {
		b.Fatal(err)
	}
	heuristic, err := agent.NewHeuristicFactory(agent.HeuristicConfig{
		Aggression: 1.2, Economy: 1, Efficiency: 1, Defense: 0.1,
	})
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	agents := make([]agent.Agent, players)
	for seat := range players {
		if agents[seat], err = heuristic.New(ctx, agent.AgentInit{MatchID: "benchmark", PlayerID: seat}); err != nil {
			b.Fatal(err)
		}
	}
	moves := make([][]game.Action, 0, turns)
	for turn := range turns {
		if g.Finished() {
			b.Fatalf("fixture finished early, at turn %d", turn)
		}
		var actions []game.Action
		for seat, a := range agents {
			decision, err := a.Decide(ctx, agent.TurnRequest{MatchID: "benchmark", PlayerID: seat, View: agent.NewView(g)})
			if err != nil {
				b.Fatal(err)
			}
			actions = append(actions, decision.Actions...)
		}
		if err := g.AdvanceTurn(actions); err != nil {
			b.Fatalf("turn %d: %v", turn, err)
		}
		moves = append(moves, actions)
	}
	return g, moves
}

// replayBenchmarkFixture rebuilds a fixture by generating the same board and
// replaying recorded actions. The engine is deterministic, so the result is
// the state those actions were recorded in.
func replayBenchmarkFixture(b *testing.B, factory boardFactory, moves [][]game.Action) *game.Game {
	b.Helper()
	g, err := factory.New(benchmarkSeed)
	if err != nil {
		b.Fatal(err)
	}
	for turn, actions := range moves {
		if err := g.AdvanceTurn(actions); err != nil {
			b.Fatalf("replay turn %d: %v", turn, err)
		}
	}
	return g
}

// BenchmarkPredictFleets times one full fleet prediction at several fleet
// counts, taken from the front of one warmed fixture's fleet list.
// PredictFleets never modifies the game, so one fixture serves every count.
func BenchmarkPredictFleets(b *testing.B) {
	for _, board := range benchmarkBoards {
		g, _ := playBenchmarkFixture(b, board.factory, board.players, benchmarkWarmTurns)
		all := g.Fleets
		horizon := benchmarkHorizon(g)
		for _, count := range benchmarkFleetCounts {
			label := fmt.Sprint(count)
			if count < 0 {
				count, label = len(all), "all"
			}
			b.Run(fmt.Sprintf("board=%s/fleets=%s", board.name, label), func(b *testing.B) {
				if count > len(all) {
					b.Skipf("fixture has only %d fleets", len(all))
				}
				g.Fleets = all[:count]
				defer func() { g.Fleets = all }()
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					g.PredictFleets(horizon)
				}
				// Reported after the loop: ResetTimer discards earlier metrics.
				b.ReportMetric(float64(count), "fleets")
			})
		}
	}
}

// BenchmarkLeadAim times one aim between a fixed source and target on each
// warmed fixture, in two cases: one LeadAim verifies, and one it cannot,
// because the sun, the board edge or another body claims the checking fleet.
// A failed aim still runs the full checking prediction, so both cases matter.
//
// Sources are the planets player 0 owns, each sending all its ships, as the
// learner's all-in action does; targets are every other planet and comet.
// The first reachable pair and the first blocked pair, in storage order, are
// the two cases. Choosing them is untimed.
func BenchmarkLeadAim(b *testing.B) {
	type aimCase struct{ source, target, ships int }
	for _, board := range benchmarkBoards {
		g, _ := playBenchmarkFixture(b, board.factory, board.players, benchmarkWarmTurns)
		var targets []int
		for _, p := range g.Planets {
			targets = append(targets, p.ID)
		}
		for _, c := range g.Comets {
			targets = append(targets, c.ID)
		}
		cases := map[string]aimCase{}
		for _, p := range g.Planets {
			if p.OwnerID != 0 || p.Ships < 1 {
				continue
			}
			for _, target := range targets {
				if target == p.ID {
					continue
				}
				_, ok := g.LeadAim(p.ID, target, p.Ships)
				name := "blocked"
				if ok {
					name = "reachable"
				}
				if _, seen := cases[name]; !seen {
					cases[name] = aimCase{source: p.ID, target: target, ships: p.Ships}
				}
			}
		}
		for _, name := range []string{"reachable", "blocked"} {
			b.Run(fmt.Sprintf("board=%s/aim=%s", board.name, name), func(b *testing.B) {
				c, found := cases[name]
				if !found {
					b.Skipf("fixture has no %s source-target pair", name)
				}
				benchmarkOneAim(b, g, c.source, c.target, c.ships, name == "reachable")
			})
		}
	}

	// The generated static board happens to have no blocked pair, so a small
	// hand-built static board supplies one: planet 2 sits directly between the
	// source, planet 1, and the target, planet 3, and claims the checking fleet.
	b.Run("board=static-line/aim=blocked", func(b *testing.B) {
		config := game.DefaultConfig()
		config.OrbitsEnabled, config.CometSpawningEnabled, config.SunHazardEnabled = false, false, false
		g, err := game.New(config, []game.Planet{
			{ID: 1, OwnerID: 0, Ships: 50, Production: 1, Position: game.Vec2{X: 10, Y: 20}},
			{ID: 2, OwnerID: game.NeutralPlayer, Ships: 5, Production: 5, Position: game.Vec2{X: 30, Y: 20}},
			{ID: 3, OwnerID: 1, Ships: 5, Production: 1, Position: game.Vec2{X: 60, Y: 20}},
		}, nil)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkOneAim(b, g, 1, 3, 50, false)
	})
}

// benchmarkOneAim times LeadAim for one source-target pair and fails if the
// answer is not the expected one, so a case cannot silently measure the
// other outcome.
func benchmarkOneAim(b *testing.B, g *game.Game, source, target, ships int, want bool) {
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, ok := g.LeadAim(source, target, ships); ok != want {
			b.Fatalf("LeadAim(%d, %d, %d) ok = %v, want %v", source, target, ships, ok, want)
		}
	}
}

// BenchmarkAdvanceTurn times benchmarkTimedTurns real turns from the warmed
// fixture, with the recorded actions. AdvanceTurn changes the game, so every
// operation starts from a freshly replayed copy; that replay is untimed.
func BenchmarkAdvanceTurn(b *testing.B) {
	for _, board := range benchmarkBoards {
		b.Run("board="+board.name, func(b *testing.B) {
			_, moves := playBenchmarkFixture(b, board.factory, board.players, benchmarkWarmTurns+benchmarkTimedTurns)
			warm, timed := moves[:benchmarkWarmTurns], moves[benchmarkWarmTurns:]
			fleets := len(replayBenchmarkFixture(b, board.factory, warm).Fleets)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				b.StopTimer()
				g := replayBenchmarkFixture(b, board.factory, warm)
				b.StartTimer()
				for turn, actions := range timed {
					if err := g.AdvanceTurn(actions); err != nil {
						b.Fatalf("timed turn %d: %v", turn, err)
					}
				}
			}
			// Reported after the loop: ResetTimer discards earlier metrics.
			b.ReportMetric(float64(fleets), "fleets-at-start")
		})
	}
}
