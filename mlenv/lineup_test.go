package mlenv

import (
	"context"
	"strings"
	"testing"

	"orrery/agent"
	"orrery/scenario"
)

// Three built-in agents with different IDs, so a test can tell from an
// opponent's ID alone which lineup entry built it.
const (
	lineupIdle      = agent.IdleV1
	lineupRandom    = "random-v1:launch_probability=0.3,max_fraction=0.8,min_fraction=0.2"
	lineupHeuristic = agent.HeuristicV2Default
)

func TestParseLineup(t *testing.T) {
	for _, test := range []struct {
		name string
		text string
		// want is the canonical form; empty when the text must be refused.
		want string
		// wantErr is a fragment of the refusal message.
		wantErr string
	}{
		{name: "one agent", text: lineupHeuristic, want: lineupHeuristic},
		{
			name: "one per opponent seat",
			text: lineupHeuristic + "|" + lineupRandom + "|" + lineupRandom,
			want: lineupHeuristic + "|" + lineupRandom + "|" + lineupRandom,
		},
		{
			// Parameters are canonicalized per entry, so a reordered
			// specification records the same identity.
			name: "entries are canonicalized",
			text: "random-v1:min_fraction=0.2,max_fraction=0.8,launch_probability=0.3|" + lineupIdle,
			want: lineupRandom + "|" + lineupIdle,
		},
		{name: "empty text", text: "", wantErr: "entry 1 is empty"},
		{name: "doubled separator", text: lineupIdle + "||" + lineupRandom, wantErr: "entry 2 is empty"},
		{name: "trailing separator", text: lineupIdle + "|", wantErr: "entry 2 is empty"},
		{name: "unknown agent", text: lineupIdle + "|nobody-v1", wantErr: "entry 2"},
		{
			// The same agent in every seat is the single specification, so
			// both spellings record one identity.
			name: "same agent in every seat",
			text: lineupRandom + "|" + lineupRandom + "|" + lineupRandom,
			want: lineupRandom,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			lineup, err := ParseLineup(test.text, agent.ParseFactory)
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("ParseLineup(%q) error = %v, want it to contain %q", test.text, err, test.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseLineup(%q) error = %v", test.text, err)
			}
			if got := lineup.String(); got != test.want {
				t.Fatalf("String() = %q, want %q", got, test.want)
			}
		})
	}
}

// TestLineupCountMustMatchPlayers checks that NewBatch refuses a lineup that
// names neither one agent nor one per opponent seat.
func TestLineupCountMustMatchPlayers(t *testing.T) {
	for _, test := range []struct {
		name    string
		players int
		text    string
	}{
		{name: "two agents on a 4-player map", players: 4, text: lineupIdle + "|" + lineupRandom},
		{name: "three agents on a 2-player map", players: 2, text: lineupIdle + "|" + lineupRandom + "|" + lineupHeuristic},
	} {
		t.Run(test.name, func(t *testing.T) {
			lineup, err := ParseLineup(test.text, agent.ParseFactory)
			if err != nil {
				t.Fatalf("ParseLineup(%q) error = %v", test.text, err)
			}
			_, err = NewBatch(context.Background(), BatchConfig{
				Size: 1, Seeds: scenario.SeedRange{Start: 0, End: 10}, Schedule: ScheduleRotate,
				Scenario:  testScenario{id: "lineup-count-v1", maxTurns: 2, players: test.players},
				Opponents: lineup,
			})
			assertEnvErrorKind(t, err, ErrorValidation)
			if !strings.Contains(err.Error(), "opponent lineup names") {
				t.Fatalf("NewBatch() error = %q, want the lineup count named", err)
			}
		})
	}
}

// emptyIDFactory is an agent factory without an ID. Its pointer form also
// gives the typed nil that a plain == nil check would let through.
type emptyIDFactory struct{}

func (emptyIDFactory) Spec() agent.Spec { return agent.NewSpec("", nil) }

func (emptyIDFactory) New(context.Context, agent.AgentInit) (agent.Agent, error) {
	return agent.IdleAgent{}, nil
}

// TestNewBatchChecksEveryLineupFactory checks that NewBatch refuses an
// unusable factory however the lineup was built. SingleOpponent and the zero
// Lineup skip NewLineup's checks, so without this a nil factory would pass
// NewBatch and panic during the first reset.
func TestNewBatchChecksEveryLineupFactory(t *testing.T) {
	var typedNil *emptyIDFactory
	for _, test := range []struct {
		name       string
		lineup     Lineup
		wantInText string
	}{
		{name: "zero lineup", lineup: Lineup{}, wantInText: "at least one agent"},
		{name: "nil factory", lineup: SingleOpponent(nil), wantInText: "must not be nil"},
		{name: "typed nil factory", lineup: SingleOpponent(typedNil), wantInText: "must not be nil"},
		{name: "empty agent ID", lineup: SingleOpponent(emptyIDFactory{}), wantInText: "ID must not be empty"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewBatch(context.Background(), BatchConfig{
				Size: 1, Seeds: scenario.SeedRange{Start: 0, End: 10}, Schedule: ScheduleRotate,
				Scenario:  testScenario{id: "lineup-factories-v1", maxTurns: 2, players: 4},
				Opponents: test.lineup,
			})
			assertEnvErrorKind(t, err, ErrorValidation)
			if !strings.Contains(err.Error(), test.wantInText) {
				t.Fatalf("NewBatch() error = %q, want it to contain %q", err, test.wantInText)
			}
		})
	}
}

// TestLineupSeatsCountedFromLearner places the learner in each seat of a
// four-player map and checks that the seats one, two, and three places after
// it hold the agents named by lineup entries 0, 1, and 2, wrapping past seat 3.
func TestLineupSeatsCountedFromLearner(t *testing.T) {
	lineup, err := ParseLineup(lineupHeuristic+"|"+lineupRandom+"|"+lineupIdle, agent.ParseFactory)
	if err != nil {
		t.Fatalf("ParseLineup() error = %v", err)
	}
	wantIDs := []string{agent.HeuristicV2, agent.RandomV1, agent.IdleV1}
	batch, err := NewBatch(context.Background(), BatchConfig{
		Size: 1, Seeds: scenario.SeedRange{Start: 0, End: 10}, Schedule: ScheduleRotate,
		Scenario:  testScenario{id: "lineup-seats-v1", maxTurns: 2, players: 4},
		Opponents: lineup,
	})
	if err != nil {
		t.Fatalf("NewBatch() error = %v", err)
	}
	t.Cleanup(func() {
		if err := batch.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})

	for learner := range 4 {
		override := []EpisodeOverride{{EnvID: 0, Seed: 0, Seat: learner}}
		if _, err := batch.Reset(context.Background(), []int{0}, override); err != nil {
			t.Fatalf("learner seat %d: Reset() error = %v", learner, err)
		}
		opponents := batch.envs[0].opponents
		if opponents[learner] != nil {
			t.Fatalf("learner seat %d holds an opponent %q", learner, opponents[learner].ID())
		}
		for after, want := range wantIDs {
			seat := (learner + after + 1) % 4
			if got := opponents[seat].ID(); got != want {
				t.Errorf("learner seat %d: seat %d (%d after the learner) = %q, want %q", learner, seat, after+1, got, want)
			}
		}
	}
}
