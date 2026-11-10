// Package agent defines the boundary between Orrery's mutable rules engine
// and code that chooses a player's actions. Agents receive isolated state
// views and return plain game.Action values; the game package remains the sole
// authority that validates and applies those actions.
package agent

import (
	"context"
	"sort"
	"strings"

	"orrery/game"
)

// AgentInit is the stable per-match identity and randomness assigned to one
// agent instance. ContestantID identifies the same entrant across seat-swapped
// legs, while PlayerID is the seat it occupies in this particular match.
type AgentInit struct {
	MatchID      string // unique ID for this scheduled match
	ScenarioSeed int64  // seed used to construct the shared initial game state
	ContestantID string // stable entrant ID, unchanged when paired legs swap seats
	PlayerID     int    // player seat this instance controls in this match
	AgentSeed    int64  // private deterministic seed for this agent instance
}

// TurnRequest contains everything an agent may use to decide one turn. View is
// an isolated copy, so an agent cannot mutate the live game or another seat's
// input through it.
type TurnRequest struct {
	MatchID  string
	PlayerID int
	View     View
}

// Decision is one agent's complete submission for a turn. Keeping it as a
// struct leaves room for bounded diagnostics from future external agents
// without changing the Agent interface or the meaning of Actions.
type Decision struct {
	Actions []game.Action
}

// Agent chooses one player's actions. A nil/empty action slice with a nil error
// is a legal no-op; an error means the agent failed to decide and the match
// runner should not advance that turn.
//
// An implementation that owns a subprocess, socket, or other resource may
// additionally implement io.Closer. Match runners must close every such agent
// on normal completion and on all error or cancellation paths. Keeping cleanup
// optional avoids forcing stateless in-process agents to implement a no-op.
type Agent interface {
	ID() string
	Decide(ctx context.Context, request TurnRequest) (Decision, error)
}

// Factory constructs a fresh Agent for one seat in one match. Implementations
// must not share mutable random state between the agents returned by New.
type Factory interface {
	Spec() Spec
	New(ctx context.Context, init AgentInit) (Agent, error)
}

// Parameter is one canonical name/value pair in an agent specification. Values
// are strings because each concrete factory owns their parsing and validation;
// Spec only provides stable identity and ordering after validation succeeds.
type Parameter struct {
	Name  string
	Value string
}

// Spec is the immutable, canonical identity of one configured agent. Its
// fields stay private so callers cannot mutate a factory's identity through a
// shared map or slice.
type Spec struct {
	id         string
	parameters []Parameter
}

// NewSpec constructs a stable specification from validated parameters. Map
// iteration is deliberately normalized here so seeds, results, and logs never
// depend on Go's randomized map order.
func NewSpec(id string, parameters map[string]string) Spec {
	keys := make([]string, 0, len(parameters))
	for key := range parameters {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	canonical := make([]Parameter, 0, len(keys))
	for _, key := range keys {
		canonical = append(canonical, Parameter{Name: key, Value: parameters[key]})
	}
	return Spec{id: id, parameters: canonical}
}

// ID returns the versioned agent type, such as "idle-v1".
func (s Spec) ID() string { return s.id }

// Parameters returns a copy of the canonical parameter list.
func (s Spec) Parameters() []Parameter {
	return append([]Parameter(nil), s.parameters...)
}

// String returns the canonical form used in seeds, records, and CLI output.
func (s Spec) String() string {
	if len(s.parameters) == 0 {
		return s.id
	}

	var b strings.Builder
	b.WriteString(s.id)
	b.WriteByte(':')
	for i, parameter := range s.parameters {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(parameter.Name)
		b.WriteByte('=')
		b.WriteString(parameter.Value)
	}
	return b.String()
}
