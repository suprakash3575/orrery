package mlenv

import (
	"context"
	"fmt"
	"testing"

	"orrery/agent"
	"orrery/game"
	"orrery/scenario"
)

// buildBenchmarkSeed and buildBenchmarkTurns fix the fixture state, so
// before-and-after runs of BenchmarkBuild time identical observations.
// Turn 100 is past the opening, with many fleets in flight.
const (
	buildBenchmarkSeed  int64 = 1_000_001
	buildBenchmarkTurns       = 100
)

// BenchmarkBuild times one complete obs-v2 observation, incoming-fleet
// prediction included, on mid-game generated boards. Every seat is played by
// heuristic-v1; playing the fixture is untimed.
func BenchmarkBuild(b *testing.B) {
	for _, players := range benchmarkPlayerCounts {
		b.Run(fmt.Sprintf("players=%d", players), func(b *testing.B) {
			g, frame := buildBenchmarkFixture(b, players)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := Build(g, frame); err != nil {
					b.Fatal(err)
				}
			}
			// Reported after the loop: ResetTimer discards earlier metrics.
			b.ReportMetric(float64(len(g.Fleets)), "fleets")
		})
	}
}

// buildBenchmarkFixture plays a generated board to buildBenchmarkTurns.
// Seat 0's frame is built first, at turn 0, the only turn NewSeatFrame
// accepts, and kept as an episode keeps it.
func buildBenchmarkFixture(b *testing.B, players int) (*game.Game, SeatFrame) {
	b.Helper()
	g, err := scenario.GeneratorV2{FourPlayer: players == 4}.New(buildBenchmarkSeed)
	if err != nil {
		b.Fatal(err)
	}
	frame, err := NewSeatFrame(g, 0)
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	factory := heuristicFactory(b)
	agents := make([]agent.Agent, players)
	for seat := range players {
		if agents[seat], err = factory.New(ctx, agent.AgentInit{MatchID: "benchmark", PlayerID: seat}); err != nil {
			b.Fatal(err)
		}
	}
	for turn := range buildBenchmarkTurns {
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
	}
	return g, frame
}
