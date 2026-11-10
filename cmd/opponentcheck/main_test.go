package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"orrery/agent"
	"orrery/game"
	"orrery/scenario"
)

// testConfig is the run's matchConfig with the pinned default agents.
func testConfig(t *testing.T) matchConfig {
	t.Helper()
	opponent, err := agent.ParseFactory(defaultOpponent)
	if err != nil {
		t.Fatal(err)
	}
	random, err := agent.ParseFactory(defaultRandom)
	if err != nil {
		t.Fatal(err)
	}
	return matchConfig{opponent: opponent, random: random, launchProbability: 0.42}
}

// testOptions is a one-tester, one-seed run into a fresh directory.
func testOptions(t *testing.T, name string, workers int) options {
	return options{
		testers: []string{"diag-lead"}, seedStart: 1_000_000, seeds: 1,
		launchProbability: 0.42, workers: workers,
		opponent: defaultOpponent, random: defaultRandom,
		out: filepath.Join(t.TempDir(), name),
	}
}

// TestRunIsIndependentOfWorkerCount plays the same four games (one tester, one
// seed, every seat) with 1 and with 4 workers. Games finish in a different
// order with more workers, but the sorted games.csv must be byte-identical,
// so the worker count can never change a result.
//
// The failure case checks the failure contract on the same small run: an
// agent that errors on turn 3 must leave every game failed, unscored, written
// to games.jsonl, and the run incomplete with no verdict.
func TestRunIsIndependentOfWorkerCount(t *testing.T) {
	ctx := context.Background()
	outputs := map[int][]byte{}
	for _, workers := range []int{1, 4} {
		opts := testOptions(t, "run", workers)
		complete, err := runWith(ctx, opts, testConfig(t), io.Discard)
		if err != nil || !complete {
			t.Fatalf("workers %d: complete = %v, err = %v; want a complete run", workers, complete, err)
		}
		data, err := os.ReadFile(filepath.Join(opts.out, "games.csv"))
		if err != nil {
			t.Fatal(err)
		}
		rows, err := csv.NewReader(bytes.NewReader(data)).ReadAll()
		if err != nil {
			t.Fatal(err)
		}
		if len(rows) != 1+players {
			t.Fatalf("workers %d: games.csv has %d rows, want a header and %d games", workers, len(rows), players)
		}
		for _, row := range rows[1:] {
			if row[3] != statusCompleted {
				t.Fatalf("workers %d: game %v has status %q, want completed", workers, row[:3], row[3])
			}
		}
		outputs[workers] = data
	}
	if !bytes.Equal(outputs[1], outputs[4]) {
		t.Fatalf("games.csv differs between 1 and 4 workers:\n1 worker:\n%s\n4 workers:\n%s", outputs[1], outputs[4])
	}

	t.Run("a failing agent leaves an incomplete, unscored run", func(t *testing.T) {
		opts := testOptions(t, "failing", 2)
		config := testConfig(t)
		config.newTester = func(context.Context, job, *game.Game) (agent.Agent, error) {
			return failOnTurn{turn: 3}, nil
		}
		complete, err := runWith(ctx, opts, config, io.Discard)
		if err != nil || complete {
			t.Fatalf("complete = %v, err = %v; want an incomplete run and no error", complete, err)
		}

		var summary struct {
			Run struct {
				Status    string `json:"status"`
				Completed int    `json:"completed"`
			} `json:"run"`
			Testers []testerSummary `json:"testers"`
			Verdict string          `json:"verdict"`
		}
		data, err := os.ReadFile(filepath.Join(opts.out, "summary.json"))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, &summary); err != nil {
			t.Fatal(err)
		}
		if summary.Run.Completed != 0 || !strings.HasPrefix(summary.Run.Status, "incomplete") {
			t.Fatalf("run = %+v, want incomplete with 0 completed", summary.Run)
		}
		if !strings.HasPrefix(summary.Verdict, "no verdict") {
			t.Fatalf("verdict = %q, want no verdict", summary.Verdict)
		}
		tester := summary.Testers[0]
		if tester.Failed != players || tester.Completed != 0 || tester.Wins+tester.Losses+tester.Draws != 0 {
			t.Fatalf("tester summary = %+v, want %d failed games and nothing scored", tester, players)
		}

		// Every finished game, failed ones included, is already in games.jsonl.
		stream, err := os.ReadFile(filepath.Join(opts.out, "games.jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(strings.TrimSpace(string(stream)), "\n")
		if len(lines) != players {
			t.Fatalf("games.jsonl has %d lines, want %d", len(lines), players)
		}
		for _, line := range lines {
			var r gameRecord
			if err := json.Unmarshal([]byte(line), &r); err != nil {
				t.Fatal(err)
			}
			if r.Status != statusFailed || r.FailTurn != 3 || r.FailSeat == nil || *r.FailSeat != r.Seat {
				t.Fatalf("record %+v, want failed on turn 3 in the tested seat", r)
			}
		}
	})
}

// TestVerdictAppliesOnlyToTheAgreedConfiguration checks that the decision
// criteria are applied only to the configuration they were fixed for, and
// that a verdict names the opponent actually played. The unit cases use
// results that would otherwise read as "keep"; the last case is the
// reported reproduction, run end to end: idle opponents and p = 0 must not
// produce a verdict about heuristic-v1.
func TestVerdictAppliesOnlyToTheAgreedConfiguration(t *testing.T) {
	agreed := runMeta{Opponent: defaultOpponent, Random: defaultRandom, LaunchProbability: defaultLaunchProbability}
	keepResults := []testerSummary{
		{Tester: "heuristic", WinRate: 0.10},
		{Tester: "idle", WinRate: 0.02},
		{Tester: "diag-lead", WinRate: 0.12},
	}
	override := func(change func(*runMeta)) runMeta {
		meta := agreed
		change(&meta)
		return meta
	}
	tests := []struct {
		name       string
		meta       runMeta
		wantPrefix string
		wantText   string
	}{
		{name: "agreed configuration", meta: agreed, wantPrefix: "keep ", wantText: defaultOpponent},
		// The pinned heuristic-v2 spec is judged against the same thresholds,
		// and the verdict names it rather than heuristic-v1.
		{name: "pinned heuristic-v2 opponent", meta: override(func(m *runMeta) { m.Opponent = agent.HeuristicV2Default }), wantPrefix: "keep ", wantText: agent.HeuristicV2Default},
		// Only the pinned spec: heuristic-v2 with other weights is not agreed.
		{name: "other heuristic-v2 weights", meta: override(func(m *runMeta) { m.Opponent = "heuristic-v2:candidates=4,enemy_bonus=1,margin=0.2" }), wantPrefix: "no verdict", wantText: "margin=0.2"},
		{name: "opponent overridden", meta: override(func(m *runMeta) { m.Opponent = "idle-v1" }), wantPrefix: "no verdict", wantText: `opponent "idle-v1"`},
		{name: "random tester overridden", meta: override(func(m *runMeta) { m.Random = "random-v1:launch_probability=1,max_fraction=1,min_fraction=1" }), wantPrefix: "no verdict", wantText: "launch_probability=1"},
		{name: "launch probability overridden", meta: override(func(m *runMeta) { m.LaunchProbability = 0.5 }), wantPrefix: "no verdict", wantText: "p = 0.5"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, _ := verdict(test.meta, true, keepResults, calibration{Seeds: 1, Passed: true})
			if !strings.HasPrefix(got, test.wantPrefix) || !strings.Contains(got, test.wantText) {
				t.Fatalf("verdict = %q, want it to start %q and mention %q", got, test.wantPrefix, test.wantText)
			}
		})
	}

	t.Run("reported reproduction, end to end", func(t *testing.T) {
		out := filepath.Join(t.TempDir(), "override")
		opts, err := parseOptions([]string{
			"-opponent", "idle-v1", "-launch-probability", "0", "-seeds", "1",
			"-testers", "heuristic,idle,diag-lead", "-workers", "4", "-out", out,
		}, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		complete, err := run(context.Background(), opts, io.Discard)
		if err != nil || !complete {
			t.Fatalf("complete = %v, err = %v; want a complete run", complete, err)
		}
		var summary struct {
			Verdict string `json:"verdict"`
		}
		data, err := os.ReadFile(filepath.Join(out, "summary.json"))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, &summary); err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(summary.Verdict, "no verdict") || !strings.Contains(summary.Verdict, `opponent "idle-v1"`) {
			t.Fatalf("verdict = %q, want no verdict naming opponent idle-v1", summary.Verdict)
		}
	})
}

// failOnTurn is a tester that plays idle until its chosen turn, then fails.
type failOnTurn struct{ turn int }

func (failOnTurn) ID() string { return "fail-on-turn" }

func (a failOnTurn) Decide(_ context.Context, request agent.TurnRequest) (agent.Decision, error) {
	if request.View.Turn == a.turn {
		return agent.Decision{}, errors.New("planned failure")
	}
	return agent.Decision{}, nil
}

// TestDiagAgentsChooseAlikeAndAimAsSpecified checks the diagnostic agents'
// contract on a real starting board, with p = 1 so every source launches:
// on the same state both make the same choices, every launch sends all of
// its source's ships, diag-direct aims at the target's current position, and
// diag-lead takes LeadAim's heading when it verifies one and the current
// position otherwise.
func TestDiagAgentsChooseAlikeAndAimAsSpecified(t *testing.T) {
	const seed = 1_000_000
	g, err := scenario.GeneratorV2{FourPlayer: true}.New(seed)
	if err != nil {
		t.Fatal(err)
	}
	view := agent.NewView(g)
	for seat := range players {
		direct := newDiagAgent(false, g, 1, seed, seat).choose(view, seat)
		lead := newDiagAgent(true, g, 1, seed, seat).choose(view, seat)
		if len(direct) == 0 {
			t.Fatalf("seat %d: no launch chosen with p = 1", seat)
		}
		if !slices.Equal(direct, lead) {
			t.Fatalf("seat %d: diag-direct chose %+v, diag-lead chose %+v", seat, direct, lead)
		}
		for _, c := range direct {
			if c.source.owner != seat || c.target.id == c.source.id {
				t.Fatalf("seat %d: choice %+v is not from an owned source to another body", seat, c)
			}
		}

		// Fresh agents draw the same stream again, so their actions follow the
		// choices checked above, in the same order.
		request := agent.TurnRequest{PlayerID: seat, View: view}
		directActions, err := newDiagAgent(false, g, 1, seed, seat).Decide(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		leadActions, err := newDiagAgent(true, g, 1, seed, seat).Decide(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		for i, c := range direct {
			d, l := directActions.Actions[i], leadActions.Actions[i]
			if d.PlanetID != c.source.id || d.Ships != c.source.ships || l.PlanetID != c.source.id || l.Ships != c.source.ships {
				t.Fatalf("seat %d launch %d: actions %+v and %+v, want all %d ships from body %d", seat, i, d, l, c.source.ships, c.source.id)
			}
			delta := c.target.position.Sub(c.source.position)
			current := math.Atan2(delta.Y, delta.X)
			if d.Angle != current {
				t.Fatalf("seat %d launch %d: diag-direct angle %v, want the current-position heading %v", seat, i, d.Angle, current)
			}
			want := current
			if aim, ok := g.LeadAim(c.source.id, c.target.id, c.source.ships); ok {
				want = aim.Angle
			}
			if l.Angle != want {
				t.Fatalf("seat %d launch %d: diag-lead angle %v, want %v", seat, i, l.Angle, want)
			}
		}
	}
}
