package arena

import (
	"orrery/agent"
	"orrery/game"
)

const (
	// MatchResultSchemaV1 versions the persisted match result shape.
	MatchResultSchemaV1 = 1
	// ActionRecordSchemaV1 versions the authoritative replay input shape.
	ActionRecordSchemaV1 = 1
)

// SeatSpec assigns one contestant and one independently constructed agent to
// a player ID. The slice index in MatchSpec.Seats is the engine player ID.
// AgentSeed is supplied directly for single matches; the paired scheduler in
// PR3 will derive it from scenario and contestant identity.
type SeatSpec struct {
	ContestantID string
	Factory      agent.Factory
	AgentSeed    int64
}

// MatchSpec contains everything required to execute one independent match.
type MatchSpec struct {
	ID            string
	Scenario      ScenarioFactory
	ScenarioIndex int
	Leg           int
	Seed          int64
	Seats         []SeatSpec
	RecordActions bool
}

// SeatAssignment is the stable, serializable identity of one match seat.
type SeatAssignment struct {
	PlayerID     int    `json:"player_id"`
	ContestantID string `json:"contestant_id"`
	AgentSpec    string `json:"agent_spec"`
	AgentSeed    int64  `json:"agent_seed"`
}

// Standing replaces the engine's unordered standings map in persisted output.
type Standing struct {
	PlayerID int `json:"player_id"`
	Ships    int `json:"ships"`
}

// TerminationReason distinguishes completed games from interrupted matches.
type TerminationReason string

const (
	TerminationElimination TerminationReason = "elimination"
	TerminationTurnLimit   TerminationReason = "turn-limit"
	TerminationFailed      TerminationReason = "failed"
	TerminationCancelled   TerminationReason = "cancelled"
)

// FailureKind identifies the stage that prevented a match from completing.
type FailureKind string

const (
	FailureValidation   FailureKind = "validation"
	FailureScenario     FailureKind = "scenario"
	FailureConstruction FailureKind = "agent-construction"
	FailureDecision     FailureKind = "agent-decision"
	FailureEngine       FailureKind = "engine"
	FailureCancellation FailureKind = "cancellation"
)

// MatchFailure records the first error that stops match execution. PlayerID is
// nil for errors not attributable to a seat. An engine-rejected batch is
// attributed to the seat whose action the engine named, though its Kind stays
// FailureEngine. Actions is populated for an engine-rejected batch even when
// MatchSpec.RecordActions is false.
type MatchFailure struct {
	Kind         FailureKind   `json:"kind"`
	Turn         int           `json:"turn"`
	PlayerID     *int          `json:"player_id,omitempty"`
	ContestantID string        `json:"contestant_id,omitempty"`
	AgentSpec    string        `json:"agent_spec,omitempty"`
	Message      string        `json:"message"`
	Actions      []game.Action `json:"actions,omitempty"`
}

// CleanupError preserves a resource-close failure without masking the match's
// primary failure or completed game result.
type CleanupError struct {
	PlayerID     int    `json:"player_id"`
	ContestantID string `json:"contestant_id"`
	AgentSpec    string `json:"agent_spec"`
	Message      string `json:"message"`
}

// TurnRecord is one accepted simultaneous action batch. Rejected batches live
// on MatchFailure because they did not advance the replay state.
type TurnRecord struct {
	Turn    int           `json:"turn"`
	Actions []game.Action `json:"actions"`
}

// ActionRecord contains the ordered engine inputs required to replay a match.
type ActionRecord struct {
	SchemaVersion int          `json:"schema_version"`
	MatchID       string       `json:"match_id"`
	ScenarioID    string       `json:"scenario_id"`
	ScenarioIndex int          `json:"scenario_index"`
	Leg           int          `json:"leg"`
	ScenarioSeed  int64        `json:"scenario_seed"`
	Turns         []TurnRecord `json:"turns"`
}

// MatchResult is a complete serializable outcome, including partial state and
// attribution when execution fails before the engine reaches a terminal game.
type MatchResult struct {
	SchemaVersion int               `json:"schema_version"`
	MatchID       string            `json:"match_id"`
	ScenarioID    string            `json:"scenario_id"`
	ScenarioIndex int               `json:"scenario_index"`
	Leg           int               `json:"leg"`
	ScenarioSeed  int64             `json:"scenario_seed"`
	Seats         []SeatAssignment  `json:"seats"`
	FinishTurn    int               `json:"finish_turn"`
	Termination   TerminationReason `json:"termination"`
	Standings     []Standing        `json:"standings"`
	Leaders       []int             `json:"leaders"`
	Failure       *MatchFailure     `json:"failure,omitempty"`
	CleanupErrors []CleanupError    `json:"cleanup_errors,omitempty"`
	Actions       *ActionRecord     `json:"actions,omitempty"`
	FinalState    agent.View        `json:"final_state"`
}
