package arena

import (
	"encoding/json"
	"fmt"
	"io"
)

// OutcomeCounts is a contestant's completed-match record.
type OutcomeCounts struct {
	Wins     int     `json:"wins"`
	Losses   int     `json:"losses"`
	Draws    int     `json:"draws"`
	WinRate  float64 `json:"win_rate"`
	LossRate float64 `json:"loss_rate"`
	DrawRate float64 `json:"draw_rate"`
}

// SeatSummary breaks one contestant's outcomes down by engine player ID.
type SeatSummary struct {
	PlayerID int `json:"player_id"`
	OutcomeCounts
}

// ContestantSummary preserves pairing order while reporting canonical agent
// identity, overall outcomes, and seat-specific outcomes.
type ContestantSummary struct {
	ContestantID string `json:"contestant_id"`
	AgentSpec    string `json:"agent_spec"`
	OutcomeCounts
	Seats []SeatSummary `json:"seats"`
}

// TerminationSummary counts completed matches by authoritative reason.
type TerminationSummary struct {
	Eliminations int `json:"eliminations"`
	TurnLimits   int `json:"turn_limits"`
}

// FailureSummary separates failures useful to the first arena CLI. Other
// includes validation, scenario, and cancellation failures; cleanup errors
// are counted independently because they may accompany another outcome.
type FailureSummary struct {
	Agent   int `json:"agent"`
	Engine  int `json:"engine"`
	Other   int `json:"other"`
	Cleanup int `json:"cleanup"`
}

// ExperimentSummary is the deterministic aggregate shown by cmd/arena.
type ExperimentSummary struct {
	Scenarios         int                 `json:"scenarios"`
	Games             int                 `json:"games"`
	CompletedGames    int                 `json:"completed_games"`
	AverageFinishTurn float64             `json:"average_finish_turn"`
	Terminations      TerminationSummary  `json:"terminations"`
	Failures          FailureSummary      `json:"failures"`
	Contestants       []ContestantSummary `json:"contestants"`
}

// Summarize validates the indexed paired result set and aggregates it without
// maps in the returned value, preserving deterministic contestant and seat
// order for human and machine output.
func Summarize(pairing PairingSpec, results []MatchResult) (ExperimentSummary, error) {
	matches, err := SchedulePairedMatches(pairing)
	if err != nil {
		return ExperimentSummary{}, fmt.Errorf("summarize pairing: %w", err)
	}
	if len(results) != len(matches) {
		return ExperimentSummary{}, fmt.Errorf("result count %d, want %d", len(results), len(matches))
	}

	summary := ExperimentSummary{
		Scenarios:   pairing.ScenarioCount,
		Games:       len(results),
		Contestants: make([]ContestantSummary, len(pairing.Contestants)),
	}
	contestantIndexes := make(map[string]int, len(pairing.Contestants))
	for index, contestant := range pairing.Contestants {
		contestantIndexes[contestant.ID] = index
		summary.Contestants[index] = ContestantSummary{
			ContestantID: contestant.ID,
			AgentSpec:    contestant.Factory.Spec().String(),
			Seats: []SeatSummary{
				{PlayerID: 0},
				{PlayerID: 1},
			},
		}
	}

	finishTurnTotal := 0
	for resultIndex, result := range results {
		expected := matches[resultIndex]
		if err := validateResultIdentity(resultIndex, result, expected); err != nil {
			return ExperimentSummary{}, err
		}
		seatContestants, err := validateResultSeats(result, expected, contestantIndexes)
		if err != nil {
			return ExperimentSummary{}, fmt.Errorf("result %d: %w", resultIndex, err)
		}
		if err := validateResultLeaders(result.Leaders, seatContestants); err != nil {
			return ExperimentSummary{}, fmt.Errorf("result %d: %w", resultIndex, err)
		}
		summary.Failures.Cleanup += len(result.CleanupErrors)
		if result.Failure != nil {
			countFailure(&summary.Failures, result.Failure.Kind)
			continue
		}
		if result.Termination != TerminationElimination && result.Termination != TerminationTurnLimit {
			return ExperimentSummary{}, fmt.Errorf("result %d has non-completed termination %q without failure", resultIndex, result.Termination)
		}

		summary.CompletedGames++
		finishTurnTotal += result.FinishTurn
		if result.Termination == TerminationElimination {
			summary.Terminations.Eliminations++
		} else {
			summary.Terminations.TurnLimits++
		}

		if len(result.Leaders) == 1 {
			winnerPlayerID := result.Leaders[0]
			winnerContestant, exists := seatContestants[winnerPlayerID]
			if !exists {
				return ExperimentSummary{}, fmt.Errorf("result %d leader player %d has no seat", resultIndex, winnerPlayerID)
			}
			for playerID, contestantIndex := range seatContestants {
				won := contestantIndex == winnerContestant
				addOutcome(&summary.Contestants[contestantIndex].OutcomeCounts, won, false)
				addOutcome(&summary.Contestants[contestantIndex].Seats[playerID].OutcomeCounts, won, false)
			}
		} else {
			for playerID, contestantIndex := range seatContestants {
				addOutcome(&summary.Contestants[contestantIndex].OutcomeCounts, false, true)
				addOutcome(&summary.Contestants[contestantIndex].Seats[playerID].OutcomeCounts, false, true)
			}
		}
	}

	if summary.CompletedGames > 0 {
		summary.AverageFinishTurn = float64(finishTurnTotal) / float64(summary.CompletedGames)
	}
	for index := range summary.Contestants {
		calculateRates(&summary.Contestants[index].OutcomeCounts)
		for seatIndex := range summary.Contestants[index].Seats {
			calculateRates(&summary.Contestants[index].Seats[seatIndex].OutcomeCounts)
		}
	}
	return summary, nil
}

func validateResultIdentity(index int, result MatchResult, expected MatchSpec) error {
	if result.SchemaVersion != MatchResultSchemaV1 {
		return fmt.Errorf("result %d has schema %d, want %d", index, result.SchemaVersion, MatchResultSchemaV1)
	}
	if result.MatchID != expected.ID {
		return fmt.Errorf("result %d has match ID %q, want %q", index, result.MatchID, expected.ID)
	}
	if result.ScenarioID != expected.Scenario.ID() {
		return fmt.Errorf("result %d has scenario ID %q, want %q", index, result.ScenarioID, expected.Scenario.ID())
	}
	if result.ScenarioIndex != expected.ScenarioIndex || result.Leg != expected.Leg {
		return fmt.Errorf("result %d identifies scenario %d leg %d, want scenario %d leg %d", index, result.ScenarioIndex, result.Leg, expected.ScenarioIndex, expected.Leg)
	}
	if result.ScenarioSeed != expected.Seed {
		return fmt.Errorf("result %d has scenario seed %d, want %d", index, result.ScenarioSeed, expected.Seed)
	}
	if result.Actions != nil {
		record := result.Actions
		if record.SchemaVersion != ActionRecordSchemaV1 ||
			record.MatchID != expected.ID ||
			record.ScenarioID != expected.Scenario.ID() ||
			record.ScenarioIndex != expected.ScenarioIndex ||
			record.Leg != expected.Leg ||
			record.ScenarioSeed != expected.Seed {
			return fmt.Errorf("result %d action record metadata does not match its scheduled match", index)
		}
	}
	return nil
}

func validateResultSeats(result MatchResult, expected MatchSpec, contestantIndexes map[string]int) (map[int]int, error) {
	if len(result.Seats) != len(expected.Seats) {
		return nil, fmt.Errorf("seat count %d, want %d", len(result.Seats), len(expected.Seats))
	}
	seats := make(map[int]int, len(result.Seats))
	seenContestants := make(map[int]bool, len(result.Seats))
	for _, seat := range result.Seats {
		contestantIndex, exists := contestantIndexes[seat.ContestantID]
		if !exists {
			return nil, fmt.Errorf("unknown contestant %q", seat.ContestantID)
		}
		if seat.PlayerID < 0 || seat.PlayerID >= len(expected.Seats) {
			return nil, fmt.Errorf("invalid player ID %d", seat.PlayerID)
		}
		if _, duplicate := seats[seat.PlayerID]; duplicate {
			return nil, fmt.Errorf("duplicate player ID %d", seat.PlayerID)
		}
		if seenContestants[contestantIndex] {
			return nil, fmt.Errorf("duplicate contestant %q", seat.ContestantID)
		}
		expectedSeat := expected.Seats[seat.PlayerID]
		if seat.ContestantID != expectedSeat.ContestantID {
			return nil, fmt.Errorf("player %d contestant %q, want %q", seat.PlayerID, seat.ContestantID, expectedSeat.ContestantID)
		}
		if seat.AgentSpec != expectedSeat.Factory.Spec().String() {
			return nil, fmt.Errorf("contestant %q agent spec %q, want %q", seat.ContestantID, seat.AgentSpec, expectedSeat.Factory.Spec().String())
		}
		if seat.AgentSeed != expectedSeat.AgentSeed {
			return nil, fmt.Errorf("contestant %q agent seed %d, want %d", seat.ContestantID, seat.AgentSeed, expectedSeat.AgentSeed)
		}
		seats[seat.PlayerID] = contestantIndex
		seenContestants[contestantIndex] = true
	}
	return seats, nil
}

func validateResultLeaders(leaders []int, seats map[int]int) error {
	seen := make(map[int]bool, len(leaders))
	for _, playerID := range leaders {
		if _, exists := seats[playerID]; !exists {
			return fmt.Errorf("leader player %d has no seat", playerID)
		}
		if seen[playerID] {
			return fmt.Errorf("duplicate leader player %d", playerID)
		}
		seen[playerID] = true
	}
	return nil
}

func addOutcome(counts *OutcomeCounts, won, draw bool) {
	switch {
	case draw:
		counts.Draws++
	case won:
		counts.Wins++
	default:
		counts.Losses++
	}
}

func calculateRates(counts *OutcomeCounts) {
	games := counts.Wins + counts.Losses + counts.Draws
	if games == 0 {
		return
	}
	counts.WinRate = float64(counts.Wins) / float64(games)
	counts.LossRate = float64(counts.Losses) / float64(games)
	counts.DrawRate = float64(counts.Draws) / float64(games)
}

func countFailure(summary *FailureSummary, kind FailureKind) {
	switch kind {
	case FailureConstruction, FailureDecision:
		summary.Agent++
	case FailureEngine:
		summary.Engine++
	default:
		summary.Other++
	}
}

// WriteNDJSON writes complete match results sequentially in their existing
// index order. Workers never call this function, so output bytes cannot depend
// on match completion order.
func WriteNDJSON(writer io.Writer, results []MatchResult) error {
	encoder := json.NewEncoder(writer)
	encoder.SetEscapeHTML(false)
	for index, result := range results {
		if err := encoder.Encode(result); err != nil {
			return fmt.Errorf("write result %d: %w", index, err)
		}
	}
	return nil
}
