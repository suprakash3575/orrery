package main

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"orrery/game"
)

// TestRunIsDeterministic is PLAN.md step 9's core requirement: two runs of
// the same simulation must produce byte-identical output. Beyond a plain
// string comparison, it also asserts the output actually reflects a
// completed, non-trivial match — two runs that both silently produced
// empty or truncated output would otherwise pass the equality check for
// the wrong reason.
func TestRunIsDeterministic(t *testing.T) {
	var first, second bytes.Buffer

	if err := run(&first); err != nil {
		t.Fatal(err)
	}
	if err := run(&second); err != nil {
		t.Fatal(err)
	}

	if first.String() != second.String() {
		t.Fatal("identical simulations produced different output")
	}

	out := first.String()
	for _, want := range []string{"turn 000:", "turn 499:", "finished turn=", "planets:", "comets:", "fleets:", "standings:", "leaders:"} {
		if !strings.Contains(out, want) {
			t.Fatalf("output missing %q; got:\n%s", want, out)
		}
	}
}

// TestRunReportsScheduledCometSpawns is the correctness check the
// determinism test alone can't provide: two runs that deterministically
// produce the same *wrong* output — an event log that's silently empty
// throughout, say — would still pass TestRunIsDeterministic. Turn 50 is
// guaranteed reached (MaxTurns 500, no launches to risk early elimination),
// so a spawn event must appear.
func TestRunReportsScheduledCometSpawns(t *testing.T) {
	var out bytes.Buffer
	if err := run(&out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "spawned") {
		t.Fatalf("expected at least one comet spawn event in a full 500-turn run, got:\n%s", out.String())
	}
}

// TestNotableEvents covers each event category directly against hand-built
// snapshots, independent of a full run: the default simulation's setup
// never actually triggers a capture or a fleet event (nothing ever attacks
// anything), so TestRunIsDeterministic and TestRunReportsScheduledCometSpawns
// can't exercise this formatting at all.
func TestNotableEvents(t *testing.T) {
	cases := []struct {
		name         string
		before       snapshot
		after        snapshot
		actions      []game.Action
		wantContains []string
	}{
		{
			name:   "quiet turn reports nothing",
			before: snapshot{planetOwners: map[int]int{1: 0}},
			after:  snapshot{planetOwners: map[int]int{1: 0}},
		},
		{
			name:         "accepted launch",
			before:       snapshot{planetOwners: map[int]int{1: 0}},
			after:        snapshot{planetOwners: map[int]int{1: 0}},
			actions:      []game.Action{{PlayerID: 0, PlanetID: 1, Ships: 5, Angle: 0.5}},
			wantContains: []string{"launch player=0 source=1 ships=5 angle=0.500"},
		},
		{
			name:         "planet captured",
			before:       snapshot{planetOwners: map[int]int{1: 0, 2: 1}},
			after:        snapshot{planetOwners: map[int]int{1: 0, 2: 0}},
			wantContains: []string{"planet 2 owner 1 -> 0"},
		},
		{
			name:         "planet reset to neutral",
			before:       snapshot{planetOwners: map[int]int{1: 0}},
			after:        snapshot{planetOwners: map[int]int{1: game.NeutralPlayer}},
			wantContains: []string{"planet 1 owner 0 -> neutral"},
		},
		{
			name: "comet captured",
			before: snapshot{
				cometOwners: map[int]int{5: game.NeutralPlayer}, cometShips: map[int]int{5: 3},
			},
			after: snapshot{
				cometOwners: map[int]int{5: 1}, cometShips: map[int]int{5: 3},
			},
			wantContains: []string{"comet 5 owner neutral -> 1"},
		},
		{
			name:   "comet spawned",
			before: snapshot{cometOwners: map[int]int{}, cometShips: map[int]int{}},
			after: snapshot{
				cometOwners: map[int]int{9: game.NeutralPlayer}, cometShips: map[int]int{9: 17},
			},
			// No ships on the spawn line: a snapshot diff can't tell
			// "spawned untouched" apart from "spawned, then attacked and
			// repelled while staying neutral" (see
			// TestNotableEventsSpawnAttackedButNotCaptured), so the ship
			// count carried alongside a bare spawn is never trustworthy.
			wantContains: []string{"comet 9 spawned owner=neutral"},
		},
		{
			name: "comet expired",
			before: snapshot{
				cometOwners: map[int]int{4: game.NeutralPlayer}, cometShips: map[int]int{4: 2},
			},
			after:        snapshot{cometOwners: map[int]int{}, cometShips: map[int]int{}},
			wantContains: []string{"comet 4 expired"},
		},
		{
			name:         "fleet entered flight",
			before:       snapshot{fleetShips: map[int]int{}},
			after:        snapshot{fleetShips: map[int]int{99: 5}},
			wantContains: []string{"fleet 99 entered flight ships=5"},
		},
		{
			name:         "fleet left flight",
			before:       snapshot{fleetShips: map[int]int{99: 5}},
			after:        snapshot{fleetShips: map[int]int{}},
			wantContains: []string{"fleet 99 left flight ships=5"},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := notableEvents(c.before, c.after, c.actions)
			for _, want := range c.wantContains {
				found := false
				for _, line := range got {
					if line == want {
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("events = %v, want a line %q", got, want)
				}
			}
			if len(c.wantContains) == 0 && len(got) != 0 {
				t.Fatalf("events = %v, want none", got)
			}
		})
	}
}

// TestNotableEventsExpiryExcludesCapture guards the ordering between the
// capture check and the expiry check directly: a comet that both changed
// owner at some point in its history and is now gone must be reported only
// as expired, not also as a capture from its last-known owner to nothing.
func TestNotableEventsExpiryExcludesCapture(t *testing.T) {
	before := snapshot{cometOwners: map[int]int{4: 1}, cometShips: map[int]int{4: 2}}
	after := snapshot{cometOwners: map[int]int{}, cometShips: map[int]int{}}

	got := notableEvents(before, after, nil)

	if len(got) != 1 || got[0] != "comet 4 expired" {
		t.Fatalf("events = %v, want exactly [\"comet 4 expired\"]", got)
	}
}

// TestNotableEventsSameTurnSpawnAndCapture covers the case a comet's
// snapshot diff alone can't tell apart from an ordinary spawn: spawning runs
// before fleet movement in AdvanceTurn's order, so an existing fleet can
// capture a newly spawned comet within the same turn it appears. Reporting
// only its post-turn owner would misreport it as having been born already
// owned by a player — every spawned comet actually starts neutral
// (spawnComets hardcodes this, the only code path that creates one), so
// both the spawn and the capture must be reported as separate transitions.
//
// after.cometShips is deliberately a small survivor count (3), not the
// batch's real spawn count: it stands in for whatever Battle() left behind
// after combat, which is not the same number the comet actually spawned
// with. The spawn line must not claim to know that pre-combat count — the
// snapshot mechanism has no mid-turn checkpoint to recover it from — so it
// carries no ships at all; the capture line is what correctly reports 3,
// since that line only ever describes state after the transition.
func TestNotableEventsSameTurnSpawnAndCapture(t *testing.T) {
	before := snapshot{cometOwners: map[int]int{}, cometShips: map[int]int{}}
	after := snapshot{cometOwners: map[int]int{9: 1}, cometShips: map[int]int{9: 3}}

	got := notableEvents(before, after, nil)

	want := []string{"comet 9 spawned owner=neutral", "comet 9 owner neutral -> 1 ships=3"}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("events = %v, want exactly %v", got, want)
	}
}

// TestNotableEventsSpawnAttackedButNotCaptured is the gap ownership alone
// can't cover: Battle()'s Stage 2 leaves a repelled defender's owner
// unchanged but its ships reduced (resolution.go) — so a comet that spawns
// neutral, gets attacked by a weaker fleet the same turn, and survives
// still reads as owner=neutral afterward, identical to a comet nothing
// touched at all. Ships=3 here stands in for that reduced count, not
// whatever the comet actually spawned with; since ownership never changes
// in this scenario, there's no capture line to attach the real post-attack
// count to either — the ship count is simply not reportable, and must not
// be guessed at.
func TestNotableEventsSpawnAttackedButNotCaptured(t *testing.T) {
	before := snapshot{cometOwners: map[int]int{}, cometShips: map[int]int{}}
	after := snapshot{cometOwners: map[int]int{9: game.NeutralPlayer}, cometShips: map[int]int{9: 3}}

	got := notableEvents(before, after, nil)

	if len(got) != 1 || got[0] != "comet 9 spawned owner=neutral" {
		t.Fatalf("events = %v, want exactly [\"comet 9 spawned owner=neutral\"]", got)
	}
}

// TestRunFinalSummaryIncludesGameState is a content check beyond
// TestRunIsDeterministic's bare substring search for section headers:
// PLAN.md step 9 asks for "a concise final state and standings", so the
// planets/comets/fleets sections must actually carry per-body data, not
// just an empty heading. Planet 1 never changes hands (the default
// simulation submits no actions, so nothing ever attacks it), but it does
// orbit under the default rules, so its final position is read from an
// independent replay of the same seeded match rather than hard-coded.
func TestRunFinalSummaryIncludesGameState(t *testing.T) {
	var out bytes.Buffer
	if err := run(&out); err != nil {
		t.Fatal(err)
	}

	replay, err := newSimulation()
	if err != nil {
		t.Fatal(err)
	}
	for !replay.Finished() {
		if err := replay.AdvanceTurn(actionsForTurn(replay.Turn)); err != nil {
			t.Fatal(err)
		}
	}
	planet := replay.Planets[0]
	if planet.ID != 1 || planet.Orbit == nil || planet.Position == (game.Vec2{X: 15, Y: 50}) {
		t.Fatalf("test setup: planet %d should orbit away from (15, 50), ended at %v", planet.ID, planet.Position)
	}
	want := fmt.Sprintf("  planet 1 owner=0 ships=1512 position=(%.3f, %.3f)", planet.Position.X, planet.Position.Y)
	if !strings.Contains(out.String(), want) {
		t.Fatalf("final summary missing %q", want)
	}
}
