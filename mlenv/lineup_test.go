package mlenv

import (
	"context"
	"crypto/sha256"
	"fmt"
	"reflect"
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

// TestLineupSeatsFollowBoardRoles places the learner in each seat of a
// four-player map and checks that the seats clockwise, opposite, and
// counter-clockwise of it hold lineup entries 0, 1, and 2.
//
// The expected seats are written out from the board, not computed with a
// formula: going clockwise around the sun the seats run 0 → 1 → 3 → 2, so
// counting player numbers upwards would put an entry in a different board
// position depending on the learner's seat.
func TestLineupSeatsFollowBoardRoles(t *testing.T) {
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

	// boardSeats[learner] lists the seats clockwise, opposite, and
	// counter-clockwise of the learner, read off the board's clockwise
	// order 0 → 1 → 3 → 2.
	boardSeats := [4][3]int{
		{1, 3, 2},
		{3, 2, 0},
		{0, 1, 3},
		{2, 0, 1},
	}
	roleNames := [3]string{"clockwise", "opposite", "counter-clockwise"}
	for learner, seats := range boardSeats {
		override := []EpisodeOverride{{EnvID: 0, Seed: 0, Seat: learner}}
		if _, err := batch.Reset(context.Background(), []int{0}, override); err != nil {
			t.Fatalf("learner seat %d: Reset() error = %v", learner, err)
		}
		opponents := batch.envs[0].opponents
		if opponents[learner] != nil {
			t.Fatalf("learner seat %d holds an opponent %q", learner, opponents[learner].ID())
		}
		for entry, seat := range seats {
			if got := opponents[seat].ID(); got != wantIDs[entry] {
				t.Errorf("learner seat %d: %s seat %d = %q, want %q", learner, roleNames[entry], seat, got, wantIDs[entry])
			}
		}
	}
}

// legacySingleRandomFingerprint is the SHA-256 of fmt's %v text of every step
// result in one game: four players, 30 turns, seed 7, learner in seat 1, the
// learner never launching, and a single random-v1 specification (lineupRandom)
// in every opponent seat. It was recorded with the code from before lineups
// existed (origin/main at 9037531, BatchConfig.Opponent), where the random
// bots had 18 fleets in flight at the end. The observations include every
// opponent fleet and garrison, so any change to which agent plays a seat, its
// seed, or its moves changes this value.
const legacySingleRandomFingerprint = "d9ace97f4b49f06c03fdb806e921ae8214cb12815babeeb53559384d43f273f2"

// playLineupGame plays the fingerprinted game above against lineup and
// returns every step result.
func playLineupGame(t *testing.T, lineup Lineup) []StepResult {
	t.Helper()
	batch, err := NewBatch(context.Background(), BatchConfig{
		Size: 1, Seeds: scenario.SeedRange{Start: 0, End: 10}, Schedule: ScheduleRotate,
		Scenario:  testScenario{id: "legacy-fingerprint-v1", maxTurns: 30, players: 4},
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
	override := []EpisodeOverride{{EnvID: 0, Seed: 7, Seat: 1}}
	if _, err := batch.Reset(context.Background(), []int{0}, override); err != nil {
		t.Fatalf("Reset() error = %v", err)
	}
	_, results := playToCompletion(t, batch)
	return results
}

func stepFingerprint(results []StepResult) string {
	return fmt.Sprintf("%x", sha256.Sum256([]byte(fmt.Sprintf("%v", results))))
}

func mustParseLineup(t *testing.T, text string) Lineup {
	t.Helper()
	lineup, err := ParseLineup(text, agent.ParseFactory)
	if err != nil {
		t.Fatalf("ParseLineup(%q) error = %v", text, err)
	}
	return lineup
}

// TestSingleSpecGameplayUnchangedByLineups replays the recorded legacy game
// through both spellings of "random-v1 in every seat" and requires the exact
// results the pre-lineup code produced.
func TestSingleSpecGameplayUnchangedByLineups(t *testing.T) {
	random, err := agent.ParseFactory(lineupRandom)
	if err != nil {
		t.Fatalf("ParseFactory() error = %v", err)
	}
	for _, test := range []struct {
		name   string
		lineup Lineup
	}{
		{name: "single specification", lineup: SingleOpponent(random)},
		{name: "same specification in every seat", lineup: mustParseLineup(t, lineupRandom+"|"+lineupRandom+"|"+lineupRandom)},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := stepFingerprint(playLineupGame(t, test.lineup)); got != legacySingleRandomFingerprint {
				t.Fatalf("step results fingerprint = %s, want the pre-lineup %s", got, legacySingleRandomFingerprint)
			}
		})
	}
}

// TestMixedLineupReplaysIdentically plays the same game twice in fresh
// batches against a mixed lineup with seeded random-v1 bots and requires
// identical results. It also requires them to differ from the all-random
// game, which shows the heuristic-v2 entry really took its seat.
func TestMixedLineupReplaysIdentically(t *testing.T) {
	text := lineupHeuristic + "|" + lineupRandom + "|" + lineupRandom
	first := playLineupGame(t, mustParseLineup(t, text))
	second := playLineupGame(t, mustParseLineup(t, text))
	if !reflect.DeepEqual(first, second) {
		t.Fatal("the same mixed lineup, seed and seat gave different step results")
	}
	if stepFingerprint(first) == legacySingleRandomFingerprint {
		t.Fatal("the mixed lineup played the same game as random-v1 in every seat")
	}
}
