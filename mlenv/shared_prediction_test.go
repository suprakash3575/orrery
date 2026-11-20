package mlenv

import (
	"context"
	"reflect"
	"sync"
	"testing"

	"orrery/agent"
	"orrery/game"
	"orrery/scenario"
)

// recordingFactory wraps an opponent factory so a test can see every request
// its agents receive and every decision they return. With strip set, it removes
// the shared prediction before the real agent decides, which makes that agent
// predict for itself: the behaviour before predictions were shared.
type recordingFactory struct {
	inner agent.Factory
	strip bool

	// mu guards the fields below. Batches here run with one worker, but the
	// lock keeps the recorder safe if a test ever raises that.
	mu sync.Mutex
	// shared counts requests that arrived carrying a prediction.
	shared int
	// actions holds every decision's actions, in decision order.
	actions [][]game.Action
}

func (f *recordingFactory) Spec() agent.Spec { return f.inner.Spec() }

func (f *recordingFactory) New(ctx context.Context, init agent.AgentInit) (agent.Agent, error) {
	inner, err := f.inner.New(ctx, init)
	if err != nil {
		return nil, err
	}
	return &recordingAgent{Agent: inner, factory: f}, nil
}

// recordingAgent forwards to the real agent and reports to its factory.
type recordingAgent struct {
	agent.Agent
	factory *recordingFactory
}

func (a *recordingAgent) Decide(ctx context.Context, request agent.TurnRequest) (agent.Decision, error) {
	f := a.factory
	f.mu.Lock()
	if request.Prediction != nil {
		f.shared++
	}
	f.mu.Unlock()
	if f.strip {
		request.Prediction = nil
	}
	decision, err := a.Agent.Decide(ctx, request)
	f.mu.Lock()
	f.actions = append(f.actions, append([]game.Action(nil), decision.Actions...))
	f.mu.Unlock()
	return decision, err
}

// TestSharedPredictionLeavesHeuristicV2GamesUnchanged plays the same games
// twice: once with heuristic-v2 opponents reusing the prediction the learner's
// observation already ran, once with each opponent predicting for itself.
// Sharing is only a speed-up, so every opponent move, every board, and every
// learner result must be identical.
func TestSharedPredictionLeavesHeuristicV2GamesUnchanged(t *testing.T) {
	ctx := context.Background()
	// Eight four-player games over 150 turns reach the midgame, where many
	// fleets are in flight and the bots aim at moving bodies.
	const size, turns = 8, 150
	opponent, err := agent.ParseFactory(agent.HeuristicV2Default)
	if err != nil {
		t.Fatalf("ParseFactory(%q) error = %v", agent.HeuristicV2Default, err)
	}
	seeds := scenario.SeedRange{Start: 0, End: 1000}
	sharing := &recordingFactory{inner: opponent}
	private := &recordingFactory{inner: opponent, strip: true}
	batches := [2]*Batch{
		newTestBatch(t, benchmarkScenario(4), sharing, size, seeds, ScheduleTrainV2, 0, 1),
		newTestBatch(t, benchmarkScenario(4), private, size, seeds, ScheduleTrainV2, 0, 1),
	}
	// Each batch has its own learner state, built from the same seeds, so the
	// learner moves stay identical as long as the games do.
	states := [2]*benchmarkState{newBenchmarkState(size), newBenchmarkState(size)}

	reset := func(ids []int) {
		t.Helper()
		for i, batch := range batches {
			result, err := batch.Reset(ctx, ids, nil)
			if err != nil {
				t.Fatalf("batch %d Reset() error = %v", i, err)
			}
			for _, env := range result.Envs {
				states[i].installReset(env.EpisodeInfo, env.Observation)
			}
		}
	}
	ids := benchmarkIDs(size)
	reset(ids)

	for turn := range turns {
		var results [2][]StepResult
		for i, batch := range batches {
			results[i], err = batch.Step(ctx, states[i].actions(ids))
			if err != nil {
				t.Fatalf("turn %d: batch %d Step() error = %v", turn, i, err)
			}
		}
		var ended []int
		for j := range results[0] {
			if !reflect.DeepEqual(results[0][j], results[1][j]) {
				t.Fatalf("turn %d: env %d step results differ between shared and private prediction", turn, results[0][j].EnvID)
			}
			for i := range batches {
				states[i].installStep(results[i][j])
			}
			if results[0][j].Terminated || results[0][j].Truncated {
				ended = append(ended, results[0][j].EnvID)
			}
		}
		for _, id := range ids {
			a, b := batches[0].envs[id].game, batches[1].envs[id].game
			if !reflect.DeepEqual(a.Planets, b.Planets) || !reflect.DeepEqual(a.Comets, b.Comets) || !reflect.DeepEqual(a.Fleets, b.Fleets) {
				t.Fatalf("turn %d: env %d boards differ between shared and private prediction", turn, id)
			}
		}
		if len(ended) > 0 {
			reset(ended)
		}
	}

	// Without this, a broken hand-off that never passed a prediction would
	// make both runs predict privately and the comparison would prove nothing.
	if sharing.shared == 0 {
		t.Fatal("no opponent received the shared prediction")
	}
	if !reflect.DeepEqual(sharing.actions, private.actions) {
		t.Fatalf("opponent moves differ: %d decisions with sharing, %d without", len(sharing.actions), len(private.actions))
	}
	launches := 0
	for _, actions := range sharing.actions {
		launches += len(actions)
	}
	if launches == 0 {
		t.Fatal("opponents never launched, so the comparison exercised no aiming")
	}
}
