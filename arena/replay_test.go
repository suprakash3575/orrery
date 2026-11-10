package arena

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"orrery/agent"
	"orrery/game"
	"orrery/scenario"
)

func TestReplayReconstructsRecordedScriptedMatchExactly(t *testing.T) {
	randomFactory, err := agent.NewRandomFactory(agent.RandomConfig{
		LaunchProbability: 0.3,
		MinFraction:       0.2,
		MaxFraction:       0.7,
	})
	if err != nil {
		t.Fatal(err)
	}
	heuristicFactory, err := agent.NewHeuristicFactory(agent.HeuristicConfig{
		Aggression: 1,
		Economy:    1.3,
		Efficiency: 0.8,
		Defense:    0.15,
	})
	if err != nil {
		t.Fatal(err)
	}
	scenario := FixedScenarioFactory{}
	result := RunMatch(context.Background(), MatchSpec{
		ID:            "replay-scripted-match",
		Scenario:      scenario,
		ScenarioIndex: 12,
		Leg:           1,
		Seed:          4444,
		Seats: []SeatSpec{
			{ContestantID: "random", Factory: randomFactory, AgentSeed: 17},
			{ContestantID: "heuristic", Factory: heuristicFactory, AgentSeed: 29},
		},
		RecordActions: true,
	})
	if result.Failure != nil || result.Actions == nil {
		t.Fatalf("recorded match failed: %#v", result)
	}
	hasAction := false
	for _, turn := range result.Actions.Turns {
		if len(turn.Actions) > 0 {
			hasAction = true
			break
		}
	}
	if !hasAction {
		t.Fatal("scripted match unexpectedly recorded no actions")
	}

	replayed, err := Replay(context.Background(), scenario, *result.Actions)
	if err != nil {
		t.Fatal(err)
	}
	if got := agent.NewView(replayed); !reflect.DeepEqual(got, result.FinalState) {
		t.Fatalf("replayed final state differs:\ngot  %#v\nwant %#v", got, result.FinalState)
	}
	if replayed.Turn != result.FinishTurn || !replayed.Finished() {
		t.Fatalf("replayed turn/finished = %d/%v, want %d/true", replayed.Turn, replayed.Finished(), result.FinishTurn)
	}
	leaders, ok := replayed.Winner()
	if !ok || !reflect.DeepEqual(leaders, result.Leaders) {
		t.Fatalf("replayed leaders = %v, %v; want %v, true", leaders, ok, result.Leaders)
	}
}

func TestReplayReconstructsGeneratorV2Match(t *testing.T) {
	heuristic, err := agent.ParseFactory("heuristic-v1:aggression=1.2,economy=1,efficiency=1,defense=0.1")
	if err != nil {
		t.Fatal(err)
	}
	random, err := agent.ParseFactory("random-v1:launch_probability=0.3,min_fraction=0.2,max_fraction=0.8")
	if err != nil {
		t.Fatal(err)
	}
	factory := scenario.GeneratorV2{}
	result := RunMatch(context.Background(), MatchSpec{
		ID:       "generator-v2-replay",
		Scenario: factory,
		Seed:     42,
		Seats: []SeatSpec{
			{ContestantID: "heuristic", Factory: heuristic, AgentSeed: 17},
			{ContestantID: "random", Factory: random, AgentSeed: 29},
		},
		RecordActions: true,
	})
	if result.Failure != nil || result.Actions == nil {
		t.Fatalf("recorded V2 match failed: %#v", result)
	}
	if result.ScenarioID != scenario.GeneratorV2ID || result.Actions.ScenarioID != scenario.GeneratorV2ID {
		t.Fatalf("V2 result and action record have scenario IDs %q and %q",
			result.ScenarioID, result.Actions.ScenarioID)
	}
	hasAction := false
	for _, turn := range result.Actions.Turns {
		if len(turn.Actions) != 0 {
			hasAction = true
			break
		}
	}
	if !hasAction {
		t.Fatal("V2 match recorded no launches")
	}

	replayed, err := Replay(context.Background(), factory, *result.Actions)
	if err != nil {
		t.Fatal(err)
	}
	if got := agent.NewView(replayed); !reflect.DeepEqual(got, result.FinalState) {
		t.Fatalf("V2 replay final state differs:\ngot  %#v\nwant %#v", got, result.FinalState)
	}
	if replayed.Turn != result.FinishTurn {
		t.Fatalf("V2 replay finished turn %d, want %d", replayed.Turn, result.FinishTurn)
	}
	if _, err := Replay(context.Background(), scenario.GeneratorV1{}, *result.Actions); err == nil {
		t.Fatal("V2 action record accepted a V1 scenario factory")
	}
}

func TestReplayRejectsInvalidMetadata(t *testing.T) {
	scenario := shortScenarioFactory(1)
	valid := ActionRecord{
		SchemaVersion: ActionRecordSchemaV1,
		MatchID:       "valid-replay",
		ScenarioID:    scenario.ID(),
		Turns:         []TurnRecord{{Turn: 0}},
	}
	tests := []struct {
		name     string
		scenario ScenarioFactory
		mutate   func(*ActionRecord)
	}{
		{"unsupported schema", scenario, func(record *ActionRecord) { record.SchemaVersion++ }},
		{"empty match ID", scenario, func(record *ActionRecord) { record.MatchID = "" }},
		{"negative scenario index", scenario, func(record *ActionRecord) { record.ScenarioIndex = -1 }},
		{"negative leg", scenario, func(record *ActionRecord) { record.Leg = -1 }},
		{"nil scenario", nil, func(*ActionRecord) {}},
		{"empty scenario ID", scenario, func(record *ActionRecord) { record.ScenarioID = "" }},
		{"scenario mismatch", FixedScenarioFactory{}, func(*ActionRecord) {}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			record := valid
			test.mutate(&record)
			if _, err := Replay(context.Background(), test.scenario, record); err == nil {
				t.Fatal("Replay accepted invalid metadata")
			}
		})
	}
}

func TestReplayRejectsInvalidTurnSequenceAndActions(t *testing.T) {
	scenario := shortScenarioFactory(1)
	base := ActionRecord{
		SchemaVersion: ActionRecordSchemaV1,
		MatchID:       "invalid-turns",
		ScenarioID:    scenario.ID(),
	}
	tests := []struct {
		name    string
		turns   []TurnRecord
		message string
	}{
		{
			name:    "turn gap",
			turns:   []TurnRecord{{Turn: 1}},
			message: "names turn 1, want 0",
		},
		{
			name: "engine rejected action",
			turns: []TurnRecord{{Turn: 0, Actions: []game.Action{
				{PlayerID: 0, PlanetID: 2, Ships: 1},
			}}},
			message: "replay turn 0",
		},
		{
			name:    "record after terminal turn",
			turns:   []TurnRecord{{Turn: 0}, {Turn: 1}},
			message: "continues after game finished",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			record := base
			record.Turns = test.turns
			_, err := Replay(context.Background(), scenario, record)
			if err == nil || !strings.Contains(err.Error(), test.message) {
				t.Fatalf("error = %v, want containing %q", err, test.message)
			}
		})
	}
}

func TestReplayHonorsCancellation(t *testing.T) {
	scenario := shortScenarioFactory(2)
	record := ActionRecord{
		SchemaVersion: ActionRecordSchemaV1,
		MatchID:       "cancel-replay",
		ScenarioID:    scenario.ID(),
		Turns:         []TurnRecord{{Turn: 0}, {Turn: 1}},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Replay(ctx, scenario, record); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
}
