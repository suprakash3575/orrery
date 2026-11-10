package arena

import (
	"context"
	"errors"
	"fmt"

	"orrery/game"
)

// Replay reconstructs a game from its versioned scenario identity, seed, and
// accepted simultaneous action batches. No agent is involved: recorded engine
// inputs remain replayable even when the original agent was nondeterministic or
// is no longer available.
func Replay(ctx context.Context, scenario ScenarioFactory, record ActionRecord) (*game.Game, error) {
	if err := validateReplayInput(scenario, record); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	g, err := scenario.New(record.ScenarioSeed)
	if err != nil {
		return nil, fmt.Errorf("construct replay scenario: %w", err)
	}
	if g == nil {
		return nil, errors.New("replay scenario returned a nil game")
	}
	for recordIndex, turn := range record.Turns {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if turn.Turn != g.Turn {
			return nil, fmt.Errorf("action record %d names turn %d, want %d", recordIndex, turn.Turn, g.Turn)
		}
		if g.Finished() {
			return nil, fmt.Errorf("action record continues after game finished on turn %d", g.Turn)
		}
		if err := g.AdvanceTurn(cloneActions(turn.Actions)); err != nil {
			return nil, fmt.Errorf("replay turn %d: %w", turn.Turn, err)
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return g, nil
}

func validateReplayInput(scenario ScenarioFactory, record ActionRecord) error {
	if record.SchemaVersion != ActionRecordSchemaV1 {
		return fmt.Errorf("unsupported action record schema %d", record.SchemaVersion)
	}
	if record.MatchID == "" {
		return errors.New("action record match ID must not be empty")
	}
	if record.ScenarioIndex < 0 || record.Leg < 0 {
		return errors.New("action record scenario index and leg must be non-negative")
	}
	if nilInterface(scenario) {
		return errors.New("replay scenario factory must not be nil")
	}
	if record.ScenarioID == "" {
		return errors.New("action record scenario ID must not be empty")
	}
	if scenario.ID() != record.ScenarioID {
		return fmt.Errorf("scenario ID %q does not match action record %q", scenario.ID(), record.ScenarioID)
	}
	return nil
}
