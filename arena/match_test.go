package arena

import (
	"context"
	"errors"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"orrery/agent"
	"orrery/game"
)

func TestRunMatchScriptedAgentsIsDeterministic(t *testing.T) {
	randomFactory, err := agent.NewRandomFactory(agent.RandomConfig{
		LaunchProbability: 0.35,
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
	spec := MatchSpec{
		ID:            "deterministic-scripted-match",
		Scenario:      FixedScenarioFactory{},
		ScenarioIndex: 7,
		Leg:           1,
		Seed:          8181,
		Seats: []SeatSpec{
			{ContestantID: "random", Factory: randomFactory, AgentSeed: 101},
			{ContestantID: "heuristic", Factory: heuristicFactory, AgentSeed: 202},
		},
		RecordActions: true,
	}

	first := RunMatch(context.Background(), spec)
	second := RunMatch(context.Background(), spec)
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("same match spec produced different results:\nfirst  %#v\nsecond %#v", first, second)
	}
	if first.Failure != nil {
		t.Fatalf("match failed: %#v", first.Failure)
	}
	if first.Termination != TerminationElimination && first.Termination != TerminationTurnLimit {
		t.Fatalf("termination = %q, want completed match", first.Termination)
	}
	if first.FinishTurn <= 0 || first.FinalState.Turn != first.FinishTurn {
		t.Fatalf("finish turn = %d, final state turn = %d", first.FinishTurn, first.FinalState.Turn)
	}
	if first.Actions == nil || len(first.Actions.Turns) != first.FinishTurn {
		t.Fatalf("recorded turns = %#v, want %d accepted turns", first.Actions, first.FinishTurn)
	}
	for turn, record := range first.Actions.Turns {
		if record.Turn != turn {
			t.Fatalf("record %d has turn %d", turn, record.Turn)
		}
	}
	if got, want := first.Seats, []SeatAssignment{
		{PlayerID: 0, ContestantID: "random", AgentSpec: randomFactory.Spec().String(), AgentSeed: 101},
		{PlayerID: 1, ContestantID: "heuristic", AgentSpec: heuristicFactory.Spec().String(), AgentSeed: 202},
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("seats = %#v, want %#v", got, want)
	}
	assertSortedResult(t, first)
}

func TestRunMatchUsesIndependentSameTurnViewsAndAgentInit(t *testing.T) {
	scenario := shortScenarioFactory(1)
	callOrder := make([]int, 0, 2)
	inits := make([]agent.AgentInit, 2)
	firstFactory := &testFactory{id: "view-mutator-v1"}
	firstFactory.newAgent = func(_ context.Context, init agent.AgentInit) (agent.Agent, error) {
		inits[0] = init
		return &testAgent{id: firstFactory.id, decide: func(_ context.Context, request agent.TurnRequest) (agent.Decision, error) {
			callOrder = append(callOrder, request.PlayerID)
			if request.View.Turn != 0 || request.MatchID != "isolated-views" {
				return agent.Decision{}, errors.New("seat 0 received incorrect request metadata")
			}
			request.View.Planets[0].Ships = 999
			return agent.Decision{}, nil
		}}, nil
	}
	secondFactory := &testFactory{id: "view-observer-v1"}
	secondFactory.newAgent = func(_ context.Context, init agent.AgentInit) (agent.Agent, error) {
		inits[1] = init
		return &testAgent{id: secondFactory.id, decide: func(_ context.Context, request agent.TurnRequest) (agent.Decision, error) {
			callOrder = append(callOrder, request.PlayerID)
			if request.View.Turn != 0 || request.View.Planets[0].Ships != 10 {
				return agent.Decision{}, errors.New("seat 1 observed another seat's view mutation")
			}
			return agent.Decision{}, nil
		}}, nil
	}

	result := RunMatch(context.Background(), MatchSpec{
		ID:       "isolated-views",
		Scenario: scenario,
		Seed:     77,
		Seats: []SeatSpec{
			{ContestantID: "first", Factory: firstFactory, AgentSeed: 11},
			{ContestantID: "second", Factory: secondFactory, AgentSeed: 22},
		},
	})
	if result.Failure != nil {
		t.Fatalf("match failed: %#v", result.Failure)
	}
	if got, want := callOrder, []int{0, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("decision order = %v, want %v", got, want)
	}
	for playerID, init := range inits {
		if init.MatchID != "isolated-views" || init.ScenarioSeed != 77 || init.PlayerID != playerID {
			t.Fatalf("seat %d init = %#v", playerID, init)
		}
	}
	if inits[0].ContestantID != "first" || inits[0].AgentSeed != 11 ||
		inits[1].ContestantID != "second" || inits[1].AgentSeed != 22 {
		t.Fatalf("agent init identities = %#v", inits)
	}
}

func TestRunMatchRecordsAcceptedTurnsOnlyWhenEnabled(t *testing.T) {
	spec := MatchSpec{
		ID:       "record-actions",
		Scenario: shortScenarioFactory(3),
		Seed:     9,
		Seats: []SeatSpec{
			{ContestantID: "left", Factory: agent.IdleFactory{}},
			{ContestantID: "right", Factory: agent.IdleFactory{}},
		},
		RecordActions: true,
	}
	recorded := RunMatch(context.Background(), spec)
	if recorded.Failure != nil || recorded.Termination != TerminationTurnLimit {
		t.Fatalf("recorded match = %#v", recorded)
	}
	if recorded.Actions == nil || len(recorded.Actions.Turns) != 3 {
		t.Fatalf("action record = %#v, want three turns", recorded.Actions)
	}
	if recorded.Actions.SchemaVersion != ActionRecordSchemaV1 || recorded.Actions.ScenarioID != spec.Scenario.ID() {
		t.Fatalf("action record metadata = %#v", recorded.Actions)
	}

	spec.RecordActions = false
	unrecorded := RunMatch(context.Background(), spec)
	if unrecorded.Actions != nil {
		t.Fatalf("actions = %#v, want nil when recording is disabled", unrecorded.Actions)
	}
	if !reflect.DeepEqual(recorded.FinalState, unrecorded.FinalState) {
		t.Fatal("recording changed the match result")
	}
}

func TestRunMatchPrefersEliminationOnFinalTurn(t *testing.T) {
	scenario := scenarioFunc{id: "final-turn-elimination-v1", new: func(seed int64) (*game.Game, error) {
		config := game.DefaultConfig()
		config.MaxTurns = 1
		config.Rand = rand.New(rand.NewSource(seed))
		return game.New(config, []game.Planet{
			{ID: 1, Position: game.Vec2{X: 10, Y: 10}, OwnerID: 0, Ships: 100, Production: 1},
			{ID: 2, Position: game.Vec2{X: 13, Y: 10}, OwnerID: 1, Ships: 0, Production: 1},
		}, nil)
	}}
	attacker := &testFactory{id: "final-attacker-v1", decide: func(_ context.Context, request agent.TurnRequest) (agent.Decision, error) {
		return agent.Decision{Actions: []game.Action{{
			PlayerID: request.PlayerID,
			PlanetID: 1,
			Ships:    100,
			Angle:    0,
		}}}, nil
	}}

	result := RunMatch(context.Background(), MatchSpec{
		ID:       "final-turn-elimination",
		Scenario: scenario,
		Seed:     5,
		Seats: []SeatSpec{
			{ContestantID: "attacker", Factory: attacker},
			{ContestantID: "defender", Factory: agent.IdleFactory{}},
		},
	})
	if result.Failure != nil {
		t.Fatalf("match failed: %#v", result.Failure)
	}
	if result.FinishTurn != 1 || result.FinishTurn < result.FinalState.Config.MaxTurns {
		t.Fatalf("finish turn = %d, max turns = %d", result.FinishTurn, result.FinalState.Config.MaxTurns)
	}
	if result.Termination != TerminationElimination {
		t.Fatalf("termination = %q, want elimination when both terminal conditions are true", result.Termination)
	}
	if got, want := result.Leaders, []int{0}; !reflect.DeepEqual(got, want) {
		t.Fatalf("leaders = %v, want %v", got, want)
	}
}

func TestRunMatchValidationFailures(t *testing.T) {
	valid := MatchSpec{
		ID:       "valid",
		Scenario: shortScenarioFactory(1),
		Seats: []SeatSpec{
			{ContestantID: "left", Factory: agent.IdleFactory{}},
			{ContestantID: "right", Factory: agent.IdleFactory{}},
		},
	}
	var nilFactory *testFactory
	tests := []struct {
		name   string
		mutate func(*MatchSpec)
	}{
		{"empty match ID", func(spec *MatchSpec) { spec.ID = "" }},
		{"negative scenario index", func(spec *MatchSpec) { spec.ScenarioIndex = -1 }},
		{"negative leg", func(spec *MatchSpec) { spec.Leg = -1 }},
		{"nil scenario", func(spec *MatchSpec) { spec.Scenario = nil }},
		{"empty scenario ID", func(spec *MatchSpec) { spec.Scenario = scenarioFunc{} }},
		{"no seats", func(spec *MatchSpec) { spec.Seats = nil }},
		{"too many seats", func(spec *MatchSpec) {
			spec.Seats = append(spec.Seats, spec.Seats[0], spec.Seats[1], SeatSpec{ContestantID: "fifth", Factory: agent.IdleFactory{}})
		}},
		{"empty contestant", func(spec *MatchSpec) { spec.Seats[0].ContestantID = "" }},
		{"duplicate contestant", func(spec *MatchSpec) { spec.Seats[1].ContestantID = spec.Seats[0].ContestantID }},
		{"typed nil factory", func(spec *MatchSpec) { spec.Seats[0].Factory = nilFactory }},
		{"empty agent ID", func(spec *MatchSpec) { spec.Seats[0].Factory = &testFactory{} }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := valid
			spec.Seats = append([]SeatSpec(nil), valid.Seats...)
			test.mutate(&spec)
			result := RunMatch(context.Background(), spec)
			if result.Failure == nil || result.Failure.Kind != FailureValidation {
				t.Fatalf("failure = %#v, want validation", result.Failure)
			}
			if result.Termination != TerminationFailed || result.FinishTurn != 0 {
				t.Fatalf("result = %#v, want failed before game creation", result)
			}
		})
	}
}

func TestRunMatchRejectsScenarioOwnerWithoutSeat(t *testing.T) {
	result := RunMatch(context.Background(), MatchSpec{
		ID:       "missing-seat",
		Scenario: shortScenarioFactory(1),
		Seats: []SeatSpec{
			{ContestantID: "only", Factory: agent.IdleFactory{}},
		},
	})
	if result.Failure == nil || result.Failure.Kind != FailureScenario || !strings.Contains(result.Failure.Message, "has no match seat") {
		t.Fatalf("failure = %#v, want missing scenario seat", result.Failure)
	}
}

func TestRunMatchClosesPartialConstructionInReverseOrder(t *testing.T) {
	constructionErr := errors.New("cannot construct")
	closeErr := errors.New("cannot close partial agent")
	closeOrder := make([]int, 0, 2)
	first := closingFactory("first-v1", 0, &closeOrder, nil)
	second := closingFactory("second-v1", 1, &closeOrder, closeErr)
	second.newAgent = func(_ context.Context, _ agent.AgentInit) (agent.Agent, error) {
		return &testAgent{id: second.id, close: func() error {
			closeOrder = append(closeOrder, 1)
			return closeErr
		}}, constructionErr
	}

	result := RunMatch(context.Background(), MatchSpec{
		ID:       "construction-failure",
		Scenario: shortScenarioFactory(1),
		Seats: []SeatSpec{
			{ContestantID: "first", Factory: first},
			{ContestantID: "second", Factory: second},
		},
	})
	assertSeatFailure(t, result, FailureConstruction, 1, constructionErr)
	if got, want := closeOrder, []int{1, 0}; !reflect.DeepEqual(got, want) {
		t.Fatalf("close order = %v, want %v", got, want)
	}
	if len(result.CleanupErrors) != 1 || result.CleanupErrors[0].PlayerID != 1 || result.CleanupErrors[0].Message != closeErr.Error() {
		t.Fatalf("cleanup errors = %#v", result.CleanupErrors)
	}
}

func TestRunMatchAttributesDecisionFailureAndCleansUp(t *testing.T) {
	decisionErr := errors.New("decision failed")
	closeOrder := make([]int, 0, 2)
	first := closingFactory("first-v1", 0, &closeOrder, nil)
	first.decide = func(context.Context, agent.TurnRequest) (agent.Decision, error) {
		return agent.Decision{}, decisionErr
	}
	second := closingFactory("second-v1", 1, &closeOrder, nil)

	result := RunMatch(context.Background(), MatchSpec{
		ID:       "decision-failure",
		Scenario: shortScenarioFactory(2),
		Seats: []SeatSpec{
			{ContestantID: "first", Factory: first},
			{ContestantID: "second", Factory: second},
		},
	})
	assertSeatFailure(t, result, FailureDecision, 0, decisionErr)
	if result.FinishTurn != 0 {
		t.Fatalf("finish turn = %d, want unadvanced game", result.FinishTurn)
	}
	if got, want := closeOrder, []int{1, 0}; !reflect.DeepEqual(got, want) {
		t.Fatalf("close order = %v, want %v", got, want)
	}
}

func TestRunMatchRecordsEngineRejectedBatchWithoutActionRecording(t *testing.T) {
	closeOrder := make([]int, 0, 2)
	first := closingFactory("invalid-v1", 0, &closeOrder, nil)
	invalid := game.Action{PlayerID: 0, PlanetID: 2, Ships: 1, Angle: 0}
	first.decide = func(context.Context, agent.TurnRequest) (agent.Decision, error) {
		return agent.Decision{Actions: []game.Action{invalid}}, nil
	}
	second := closingFactory("idle-test-v1", 1, &closeOrder, nil)

	result := RunMatch(context.Background(), MatchSpec{
		ID:       "engine-failure",
		Scenario: shortScenarioFactory(2),
		Seats: []SeatSpec{
			{ContestantID: "invalid", Factory: first},
			{ContestantID: "idle", Factory: second},
		},
	})
	if result.Failure == nil || result.Failure.Kind != FailureEngine {
		t.Fatalf("failure = %#v, want engine", result.Failure)
	}
	if result.Failure.PlayerID == nil || *result.Failure.PlayerID != 0 {
		t.Fatalf("failure player = %#v, want the seat that sent the rejected action (0)", result.Failure.PlayerID)
	}
	if !reflect.DeepEqual(result.Failure.Actions, []game.Action{invalid}) {
		t.Fatalf("failed actions = %#v", result.Failure.Actions)
	}
	if result.Actions != nil || result.FinishTurn != 0 {
		t.Fatalf("result recorded or advanced rejected batch: %#v", result)
	}
	if got, want := closeOrder, []int{1, 0}; !reflect.DeepEqual(got, want) {
		t.Fatalf("close order = %v, want %v", got, want)
	}
}

// TestRunMatchBlamesEngineRejectionOnOffendingSeat puts a valid action from
// seat 0 ahead of an invalid one from seat 1 in the merged batch, so blaming
// "the first seat" or "whoever acted first" would name the wrong player. Only
// following the engine's ActionError to the action at fault names seat 1.
func TestRunMatchBlamesEngineRejectionOnOffendingSeat(t *testing.T) {
	closeOrder := make([]int, 0, 2)
	first := closingFactory("valid-v1", 0, &closeOrder, nil)
	first.decide = func(context.Context, agent.TurnRequest) (agent.Decision, error) {
		return agent.Decision{Actions: []game.Action{{PlayerID: 0, PlanetID: 1, Ships: 1, Angle: 0}}}, nil
	}
	second := closingFactory("overspend-v1", 1, &closeOrder, nil)
	second.decide = func(context.Context, agent.TurnRequest) (agent.Decision, error) {
		// Planet 2 holds 10 ships when the turn is validated.
		return agent.Decision{Actions: []game.Action{{PlayerID: 1, PlanetID: 2, Ships: 11, Angle: 0}}}, nil
	}

	result := RunMatch(context.Background(), MatchSpec{
		ID:       "engine-failure-second-seat",
		Scenario: shortScenarioFactory(2),
		Seats: []SeatSpec{
			{ContestantID: "valid", Factory: first},
			{ContestantID: "overspend", Factory: second},
		},
	})
	if result.Failure == nil || result.Failure.Kind != FailureEngine {
		t.Fatalf("failure = %#v, want engine", result.Failure)
	}
	if result.Failure.PlayerID == nil || *result.Failure.PlayerID != 1 {
		t.Fatalf("failure player = %#v, want 1", result.Failure.PlayerID)
	}
	if result.Failure.ContestantID != "overspend" {
		t.Fatalf("failure contestant = %q, want overspend", result.Failure.ContestantID)
	}
}

// TestRunMatchRejectsSeatActingForAnotherPlayer covers impersonation: seat 1
// submits an action that is perfectly legal for player 0. The engine trusts
// each action's own PlayerID, so without the arena's check this would launch
// player 0's ships on seat 1's say-so. It must fail as seat 1's decision, and
// the game must not advance.
func TestRunMatchRejectsSeatActingForAnotherPlayer(t *testing.T) {
	closeOrder := make([]int, 0, 2)
	first := closingFactory("idle-v1", 0, &closeOrder, nil)
	second := closingFactory("impostor-v1", 1, &closeOrder, nil)
	impersonation := []game.Action{{PlayerID: 0, PlanetID: 1, Ships: 5, Angle: 0}}
	second.decide = func(context.Context, agent.TurnRequest) (agent.Decision, error) {
		return agent.Decision{Actions: impersonation}, nil
	}

	result := RunMatch(context.Background(), MatchSpec{
		ID:       "impersonation",
		Scenario: shortScenarioFactory(2),
		Seats: []SeatSpec{
			{ContestantID: "idle", Factory: first},
			{ContestantID: "impostor", Factory: second},
		},
	})
	if result.Failure == nil || result.Failure.Kind != FailureDecision {
		t.Fatalf("failure = %#v, want agent decision", result.Failure)
	}
	if result.Failure.PlayerID == nil || *result.Failure.PlayerID != 1 {
		t.Fatalf("failure player = %#v, want 1", result.Failure.PlayerID)
	}
	if !reflect.DeepEqual(result.Failure.Actions, impersonation) {
		t.Fatalf("failed actions = %#v, want the impersonating decision", result.Failure.Actions)
	}
	if result.FinishTurn != 0 {
		t.Fatalf("finish turn = %d, want unadvanced game", result.FinishTurn)
	}
}

func TestRunMatchCancellationAfterConstructionCleansUp(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	closeOrder := make([]int, 0, 2)
	first := closingFactory("canceller-v1", 0, &closeOrder, nil)
	first.decide = func(context.Context, agent.TurnRequest) (agent.Decision, error) {
		cancel()
		return agent.Decision{}, nil
	}
	second := closingFactory("other-v1", 1, &closeOrder, nil)

	result := RunMatch(ctx, MatchSpec{
		ID:       "cancelled",
		Scenario: shortScenarioFactory(2),
		Seats: []SeatSpec{
			{ContestantID: "first", Factory: first},
			{ContestantID: "second", Factory: second},
		},
	})
	if result.Failure == nil || result.Failure.Kind != FailureCancellation || result.Termination != TerminationCancelled {
		t.Fatalf("result = %#v, want cancellation", result)
	}
	if result.FinishTurn != 0 {
		t.Fatalf("finish turn = %d, want 0", result.FinishTurn)
	}
	if got, want := closeOrder, []int{1, 0}; !reflect.DeepEqual(got, want) {
		t.Fatalf("close order = %v, want %v", got, want)
	}
}

func TestRunMatchCleanupErrorDoesNotMaskCompletedResult(t *testing.T) {
	closeErr := errors.New("close failed")
	closeOrder := make([]int, 0, 2)
	first := closingFactory("first-v1", 0, &closeOrder, nil)
	second := closingFactory("second-v1", 1, &closeOrder, closeErr)

	result := RunMatch(context.Background(), MatchSpec{
		ID:       "cleanup-error",
		Scenario: shortScenarioFactory(1),
		Seats: []SeatSpec{
			{ContestantID: "first", Factory: first},
			{ContestantID: "second", Factory: second},
		},
	})
	if result.Failure != nil || result.Termination != TerminationTurnLimit {
		t.Fatalf("cleanup masked completed outcome: %#v", result)
	}
	if len(result.CleanupErrors) != 1 || result.CleanupErrors[0].Message != closeErr.Error() {
		t.Fatalf("cleanup errors = %#v", result.CleanupErrors)
	}
	if got, want := closeOrder, []int{1, 0}; !reflect.DeepEqual(got, want) {
		t.Fatalf("close order = %v, want %v", got, want)
	}
}

type scenarioFunc struct {
	id  string
	new func(int64) (*game.Game, error)
}

func (factory scenarioFunc) ID() string { return factory.id }

func (factory scenarioFunc) New(seed int64) (*game.Game, error) {
	if factory.new == nil {
		return nil, errors.New("scenario not configured")
	}
	return factory.new(seed)
}

func shortScenarioFactory(maxTurns int) ScenarioFactory {
	return scenarioFunc{id: "short-v1", new: func(seed int64) (*game.Game, error) {
		config := game.DefaultConfig()
		config.MaxTurns = maxTurns
		config.Rand = rand.New(rand.NewSource(seed))
		return game.New(config, []game.Planet{
			{ID: 1, Position: game.Vec2{X: 15, Y: 50}, OwnerID: 0, Ships: 10, Production: 1},
			{ID: 2, Position: game.Vec2{X: 85, Y: 50}, OwnerID: 1, Ships: 10, Production: 1},
		}, nil)
	}}
}

type testFactory struct {
	id       string
	newAgent func(context.Context, agent.AgentInit) (agent.Agent, error)
	decide   func(context.Context, agent.TurnRequest) (agent.Decision, error)
	close    func() error
}

func (factory *testFactory) Spec() agent.Spec { return agent.NewSpec(factory.id, nil) }

func (factory *testFactory) New(ctx context.Context, init agent.AgentInit) (agent.Agent, error) {
	if factory.newAgent != nil {
		return factory.newAgent(ctx, init)
	}
	return &testAgent{id: factory.id, decide: factory.decide, close: factory.close}, nil
}

type testAgent struct {
	id     string
	decide func(context.Context, agent.TurnRequest) (agent.Decision, error)
	close  func() error
}

func (a *testAgent) ID() string { return a.id }

func (a *testAgent) Decide(ctx context.Context, request agent.TurnRequest) (agent.Decision, error) {
	if a.decide == nil {
		return agent.Decision{}, nil
	}
	return a.decide(ctx, request)
}

func (a *testAgent) Close() error {
	if a.close == nil {
		return nil
	}
	return a.close()
}

func closingFactory(id string, playerID int, closeOrder *[]int, closeErr error) *testFactory {
	return &testFactory{id: id, close: func() error {
		*closeOrder = append(*closeOrder, playerID)
		return closeErr
	}}
}

func assertSeatFailure(t *testing.T, result MatchResult, kind FailureKind, playerID int, wantErr error) {
	t.Helper()
	if result.Failure == nil || result.Failure.Kind != kind {
		t.Fatalf("failure = %#v, want kind %q", result.Failure, kind)
	}
	if result.Failure.PlayerID == nil || *result.Failure.PlayerID != playerID {
		t.Fatalf("failure player = %#v, want %d", result.Failure.PlayerID, playerID)
	}
	if result.Failure.Message != wantErr.Error() {
		t.Fatalf("failure message = %q, want %q", result.Failure.Message, wantErr)
	}
}

func assertSortedResult(t *testing.T, result MatchResult) {
	t.Helper()
	for i := 1; i < len(result.Standings); i++ {
		if result.Standings[i-1].PlayerID >= result.Standings[i].PlayerID {
			t.Fatalf("standings are not sorted: %#v", result.Standings)
		}
	}
	for i := 1; i < len(result.Leaders); i++ {
		if result.Leaders[i-1] >= result.Leaders[i] {
			t.Fatalf("leaders are not sorted: %#v", result.Leaders)
		}
	}
}
