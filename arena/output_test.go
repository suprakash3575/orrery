package arena

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"testing"

	"orrery/agent"
)

func TestSummarizeCompletedOutcomesSeatsAndFailures(t *testing.T) {
	pairing := outputTestPairing(2)
	results := []MatchResult{
		outputTestResult(pairing, 0, 0, []int{0}, TerminationElimination, 10),
		outputTestResult(pairing, 0, 1, []int{0}, TerminationTurnLimit, 20),
		outputTestResult(pairing, 1, 0, []int{0, 1}, TerminationTurnLimit, 30),
		outputTestResult(pairing, 1, 1, nil, TerminationFailed, 5),
	}
	results[3].Failure = &MatchFailure{Kind: FailureDecision, Message: "failed"}
	results[3].CleanupErrors = []CleanupError{{Message: "first"}, {Message: "second"}}

	summary, err := Summarize(pairing, results)
	if err != nil {
		t.Fatal(err)
	}
	if summary.Scenarios != 2 || summary.Games != 4 || summary.CompletedGames != 3 || summary.AverageFinishTurn != 20 {
		t.Fatalf("summary totals = %#v", summary)
	}
	if got, want := summary.Terminations, (TerminationSummary{Eliminations: 1, TurnLimits: 2}); got != want {
		t.Fatalf("terminations = %#v, want %#v", got, want)
	}
	if got, want := summary.Failures, (FailureSummary{Agent: 1, Cleanup: 2}); got != want {
		t.Fatalf("failures = %#v, want %#v", got, want)
	}
	if got, want := len(summary.Contestants), 2; got != want {
		t.Fatalf("contestants = %d, want %d", got, want)
	}
	for _, contestant := range summary.Contestants {
		if contestant.Wins != 1 || contestant.Losses != 1 || contestant.Draws != 1 ||
			contestant.WinRate != 1.0/3 || contestant.LossRate != 1.0/3 || contestant.DrawRate != 1.0/3 {
			t.Fatalf("contestant %q outcomes = %#v", contestant.ContestantID, contestant.OutcomeCounts)
		}
	}

	left := summary.Contestants[0]
	if got, want := left.Seats[0].OutcomeCounts, (OutcomeCounts{Wins: 1, Draws: 1, WinRate: 0.5, DrawRate: 0.5}); got != want {
		t.Fatalf("left seat 0 = %#v, want %#v", got, want)
	}
	if got, want := left.Seats[1].OutcomeCounts, (OutcomeCounts{Losses: 1, LossRate: 1}); got != want {
		t.Fatalf("left seat 1 = %#v, want %#v", got, want)
	}
	right := summary.Contestants[1]
	if got, want := right.Seats[0].OutcomeCounts, (OutcomeCounts{Wins: 1, WinRate: 1}); got != want {
		t.Fatalf("right seat 0 = %#v, want %#v", got, want)
	}
	if got, want := right.Seats[1].OutcomeCounts, (OutcomeCounts{Losses: 1, Draws: 1, LossRate: 0.5, DrawRate: 0.5}); got != want {
		t.Fatalf("right seat 1 = %#v, want %#v", got, want)
	}
}

func TestCountFailureCategories(t *testing.T) {
	var summary FailureSummary
	for _, kind := range []FailureKind{FailureConstruction, FailureDecision} {
		countFailure(&summary, kind)
	}
	countFailure(&summary, FailureEngine)
	for _, kind := range []FailureKind{FailureValidation, FailureScenario, FailureCancellation} {
		countFailure(&summary, kind)
	}
	if got, want := summary, (FailureSummary{Agent: 2, Engine: 1, Other: 3}); got != want {
		t.Fatalf("failure summary = %#v, want %#v", got, want)
	}
}

func TestSummarizeRejectsMalformedResultSets(t *testing.T) {
	pairing := outputTestPairing(1)
	valid := []MatchResult{
		outputTestResult(pairing, 0, 0, []int{0}, TerminationElimination, 10),
		outputTestResult(pairing, 0, 1, []int{1}, TerminationElimination, 11),
	}
	tests := []struct {
		name   string
		mutate func([]MatchResult) []MatchResult
	}{
		{"wrong result count", func(results []MatchResult) []MatchResult { return results[:1] }},
		{"wrong result schema", func(results []MatchResult) []MatchResult {
			results[0].SchemaVersion++
			return results
		}},
		{"wrong match ID", func(results []MatchResult) []MatchResult {
			results[0].MatchID = "wrong-match"
			return results
		}},
		{"wrong scenario ID", func(results []MatchResult) []MatchResult {
			results[0].ScenarioID = "wrong-scenario"
			return results
		}},
		{"wrong result index", func(results []MatchResult) []MatchResult { results[0].Leg = 1; return results }},
		{"wrong scenario seed", func(results []MatchResult) []MatchResult {
			results[0].ScenarioSeed++
			return results
		}},
		{"wrong action record metadata", func(results []MatchResult) []MatchResult {
			results[0].Actions = &ActionRecord{
				SchemaVersion: ActionRecordSchemaV1,
				MatchID:       results[0].MatchID,
				ScenarioID:    results[0].ScenarioID,
				ScenarioIndex: results[0].ScenarioIndex,
				Leg:           results[0].Leg,
				ScenarioSeed:  results[0].ScenarioSeed + 1,
			}
			return results
		}},
		{"wrong seat count", func(results []MatchResult) []MatchResult { results[0].Seats = results[0].Seats[:1]; return results }},
		{"wrong seat assignment", func(results []MatchResult) []MatchResult {
			results[0].Seats[0], results[0].Seats[1] = results[0].Seats[1], results[0].Seats[0]
			results[0].Seats[0].PlayerID = 0
			results[0].Seats[1].PlayerID = 1
			return results
		}},
		{"unknown contestant", func(results []MatchResult) []MatchResult {
			results[0].Seats[0].ContestantID = "unknown"
			return results
		}},
		{"wrong agent spec", func(results []MatchResult) []MatchResult { results[0].Seats[0].AgentSpec = "wrong-v1"; return results }},
		{"wrong agent seed", func(results []MatchResult) []MatchResult {
			results[0].Seats[0].AgentSeed++
			return results
		}},
		{"invalid player", func(results []MatchResult) []MatchResult { results[0].Seats[0].PlayerID = 2; return results }},
		{"duplicate player", func(results []MatchResult) []MatchResult { results[0].Seats[1].PlayerID = 0; return results }},
		{"duplicate contestant", func(results []MatchResult) []MatchResult {
			results[0].Seats[1].ContestantID = results[0].Seats[0].ContestantID
			results[0].Seats[1].AgentSpec = results[0].Seats[0].AgentSpec
			return results
		}},
		{"leader without seat", func(results []MatchResult) []MatchResult { results[0].Leaders = []int{3}; return results }},
		{"duplicate leader", func(results []MatchResult) []MatchResult { results[0].Leaders = []int{0, 0}; return results }},
		{"incomplete without failure", func(results []MatchResult) []MatchResult {
			results[0].Termination = TerminationCancelled
			return results
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			results := cloneOutputResults(valid)
			results = test.mutate(results)
			if summary, err := Summarize(pairing, results); err == nil {
				t.Fatalf("Summarize = %#v, want error", summary)
			}
		})
	}
}

func TestWriteNDJSONIsWorkerCountIndependentAndOrdered(t *testing.T) {
	pairing := outputTestPairing(4)
	pairing.RecordActions = true
	serial, err := RunExperiment(context.Background(), ExperimentSpec{Pairing: pairing, Workers: 1})
	if err != nil {
		t.Fatal(err)
	}
	parallel, err := RunExperiment(context.Background(), ExperimentSpec{Pairing: pairing, Workers: 5})
	if err != nil {
		t.Fatal(err)
	}
	var serialOutput, parallelOutput bytes.Buffer
	if err := WriteNDJSON(&serialOutput, serial); err != nil {
		t.Fatal(err)
	}
	if err := WriteNDJSON(&parallelOutput, parallel); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(serialOutput.Bytes(), parallelOutput.Bytes()) {
		t.Fatal("serial and parallel NDJSON differ")
	}

	decoder := json.NewDecoder(bytes.NewReader(serialOutput.Bytes()))
	for index := range serial {
		var decoded MatchResult
		if err := decoder.Decode(&decoded); err != nil {
			t.Fatalf("decode result %d: %v", index, err)
		}
		if !reflect.DeepEqual(decoded, serial[index]) {
			t.Fatalf("decoded result %d differs", index)
		}
		if decoded.ScenarioIndex != index/2 || decoded.Leg != index%2 {
			t.Fatalf("decoded result %d out of order", index)
		}
	}
	var extra MatchResult
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		t.Fatalf("decode after final result = %v, want io.EOF", err)
	}
}

func TestWriteNDJSONReportsWriterFailure(t *testing.T) {
	writer := &errorWriter{err: errors.New("disk full")}
	err := WriteNDJSON(writer, []MatchResult{{MatchID: "one"}})
	if err == nil || !errors.Is(err, writer.err) {
		t.Fatalf("error = %v, want wrapped writer error", err)
	}
}

type errorWriter struct{ err error }

func (writer *errorWriter) Write([]byte) (int, error) { return 0, writer.err }

func outputTestPairing(scenarioCount int) PairingSpec {
	return PairingSpec{
		Scenario:      shortScenarioFactory(3),
		ScenarioCount: scenarioCount,
		SeedStart:     20,
		Contestants: []ContestantSpec{
			{ID: "left", Factory: agent.IdleFactory{}},
			{ID: "right", Factory: agent.IdleFactory{}},
		},
	}
}

func outputTestResult(pairing PairingSpec, scenarioIndex, leg int, leaders []int, termination TerminationReason, finishTurn int) MatchResult {
	contestants := []ContestantSpec{pairing.Contestants[0], pairing.Contestants[1]}
	if leg == 1 {
		contestants[0], contestants[1] = contestants[1], contestants[0]
	}
	return MatchResult{
		SchemaVersion: MatchResultSchemaV1,
		MatchID:       matchID(scenarioIndex, leg),
		ScenarioID:    pairing.Scenario.ID(),
		ScenarioIndex: scenarioIndex,
		Leg:           leg,
		ScenarioSeed:  pairing.SeedStart + int64(scenarioIndex),
		Seats: []SeatAssignment{
			{
				PlayerID:     0,
				ContestantID: contestants[0].ID,
				AgentSpec:    contestants[0].Factory.Spec().String(),
				AgentSeed:    DeriveAgentSeed(pairing.SeedStart+int64(scenarioIndex), contestants[0].ID, contestants[0].Factory.Spec()),
			},
			{
				PlayerID:     1,
				ContestantID: contestants[1].ID,
				AgentSpec:    contestants[1].Factory.Spec().String(),
				AgentSeed:    DeriveAgentSeed(pairing.SeedStart+int64(scenarioIndex), contestants[1].ID, contestants[1].Factory.Spec()),
			},
		},
		FinishTurn:  finishTurn,
		Termination: termination,
		Leaders:     append([]int(nil), leaders...),
	}
}

func cloneOutputResults(results []MatchResult) []MatchResult {
	cloned := append([]MatchResult(nil), results...)
	for index := range cloned {
		cloned[index].Seats = append([]SeatAssignment(nil), cloned[index].Seats...)
		cloned[index].Leaders = append([]int(nil), cloned[index].Leaders...)
	}
	return cloned
}
