package arena

import (
	"context"
	"errors"
	"reflect"
	"sync/atomic"
	"testing"

	"orrery/agent"
)

func TestRunExperimentWorkerCountDoesNotChangeResults(t *testing.T) {
	randomFactory, err := agent.NewRandomFactory(agent.RandomConfig{
		LaunchProbability: 0.4,
		MinFraction:       0.2,
		MaxFraction:       0.8,
	})
	if err != nil {
		t.Fatal(err)
	}
	heuristicFactory, err := agent.NewHeuristicFactory(agent.HeuristicConfig{
		Aggression: 1.2,
		Economy:    1,
		Efficiency: 1,
		Defense:    0.1,
	})
	if err != nil {
		t.Fatal(err)
	}
	pairing := PairingSpec{
		Scenario:      shortScenarioFactory(12),
		ScenarioCount: 8,
		SeedStart:     500,
		Contestants: []ContestantSpec{
			{ID: "random", Factory: randomFactory},
			{ID: "heuristic", Factory: heuristicFactory},
		},
		RecordActions: true,
	}

	serial, err := RunExperiment(context.Background(), ExperimentSpec{Pairing: pairing, Workers: 1})
	if err != nil {
		t.Fatal(err)
	}
	parallel, err := RunExperiment(context.Background(), ExperimentSpec{Pairing: pairing, Workers: 7})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(serial, parallel) {
		t.Fatal("workers=1 and workers=7 produced different results")
	}
	if got, want := len(serial), 2*pairing.ScenarioCount; got != want {
		t.Fatalf("result count = %d, want %d", got, want)
	}
	for index, result := range serial {
		if result.ScenarioIndex != index/2 || result.Leg != index%2 || result.MatchID != matchID(index/2, index%2) {
			t.Fatalf("result %d out of schedule order: %#v", index, result)
		}
		if result.Failure != nil || result.Actions == nil {
			t.Fatalf("result %d incomplete: %#v", index, result)
		}
	}
}

func TestRunExperimentValidatesBeforeStartingWorkers(t *testing.T) {
	valid := ExperimentSpec{
		Pairing: PairingSpec{
			Scenario:      shortScenarioFactory(1),
			ScenarioCount: 1,
			Contestants: []ContestantSpec{
				{ID: "left", Factory: agent.IdleFactory{}},
				{ID: "right", Factory: agent.IdleFactory{}},
			},
		},
		Workers: 1,
	}
	tests := []struct {
		name   string
		mutate func(*ExperimentSpec)
	}{
		{"zero workers", func(spec *ExperimentSpec) { spec.Workers = 0 }},
		{"negative workers", func(spec *ExperimentSpec) { spec.Workers = -1 }},
		{"invalid schedule", func(spec *ExperimentSpec) { spec.Pairing.ScenarioCount = 0 }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := valid
			test.mutate(&spec)
			if results, err := RunExperiment(context.Background(), spec); err == nil || results != nil {
				t.Fatalf("RunExperiment = %#v, %v; want nil results and error", results, err)
			}
		})
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if results, err := RunExperiment(ctx, valid); !errors.Is(err, context.Canceled) || results != nil {
		t.Fatalf("pre-cancelled RunExperiment = %#v, %v", results, err)
	}
}

func TestRunExperimentCancellationClosesEveryConstructedAgent(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	started := make(chan struct{}, 1)
	var constructed atomic.Int64
	var closed atomic.Int64
	factory := &countingFactory{
		id:          "blocking-v1",
		constructed: &constructed,
		closed:      &closed,
		decide: func(ctx context.Context, _ agent.TurnRequest) (agent.Decision, error) {
			select {
			case started <- struct{}{}:
			default:
			}
			<-ctx.Done()
			return agent.Decision{}, ctx.Err()
		},
	}
	spec := ExperimentSpec{
		Pairing: PairingSpec{
			Scenario:      shortScenarioFactory(10),
			ScenarioCount: 20,
			Contestants: []ContestantSpec{
				{ID: "left", Factory: factory},
				{ID: "right", Factory: factory},
			},
		},
		Workers: 3,
	}

	type runOutcome struct {
		results []MatchResult
		err     error
	}
	completed := make(chan runOutcome, 1)
	go func() {
		results, err := RunExperiment(ctx, spec)
		completed <- runOutcome{results: results, err: err}
	}()
	<-started
	cancel()
	outcome := <-completed

	if !errors.Is(outcome.err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", outcome.err)
	}
	if len(outcome.results) != 2*spec.Pairing.ScenarioCount {
		t.Fatalf("partial result slots = %d, want %d", len(outcome.results), 2*spec.Pairing.ScenarioCount)
	}
	if got := constructed.Load(); got == 0 || closed.Load() != got {
		t.Fatalf("constructed = %d, closed = %d", got, closed.Load())
	}
}

type countingFactory struct {
	id          string
	constructed *atomic.Int64
	closed      *atomic.Int64
	decide      func(context.Context, agent.TurnRequest) (agent.Decision, error)
}

func (factory *countingFactory) Spec() agent.Spec { return agent.NewSpec(factory.id, nil) }

func (factory *countingFactory) New(context.Context, agent.AgentInit) (agent.Agent, error) {
	factory.constructed.Add(1)
	return &countingAgent{id: factory.id, closed: factory.closed, decide: factory.decide}, nil
}

type countingAgent struct {
	id     string
	closed *atomic.Int64
	decide func(context.Context, agent.TurnRequest) (agent.Decision, error)
}

func (instance *countingAgent) ID() string { return instance.id }

func (instance *countingAgent) Decide(ctx context.Context, request agent.TurnRequest) (agent.Decision, error) {
	return instance.decide(ctx, request)
}

func (instance *countingAgent) Close() error {
	instance.closed.Add(1)
	return nil
}
