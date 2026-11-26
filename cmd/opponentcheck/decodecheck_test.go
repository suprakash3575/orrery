package main

import (
	"context"
	"math"
	"strings"
	"testing"

	"orrery/agent"
	"orrery/game"
)

// restrictedSpec is heuristic-v2 limited to the learner's moves: experiment
// 1's "All three, full cost" tester.
const restrictedSpec = agent.HeuristicV2Default + ",learner_aim=1,no_aim_check=1,whole_garrison=1"

// TestCheckDecodeAllowed checks that -check-decode accepts only a tester
// whose every move is a learner move, and refuses the rest before any game.
func TestCheckDecodeAllowed(t *testing.T) {
	tests := []struct {
		name    string
		tester  string // "" means no -tester-spec
		wantErr string // "" means accepted
	}{
		{name: "fully restricted", tester: restrictedSpec},
		{name: "no tester spec", wantErr: "needs -tester-spec"},
		{name: "a setting missing", tester: agent.HeuristicV2Default + ",learner_aim=1,whole_garrison=1", wantErr: "no_aim_check=1"},
		{name: "epsilon above 0", tester: restrictedSpec + ",epsilon=0.5", wantErr: "epsilon=0"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var tester agent.Factory
			if test.tester != "" {
				var err error
				if tester, err = agent.ParseFactory(test.tester); err != nil {
					t.Fatal(err)
				}
			}
			err := checkDecodeAllowed(tester)
			switch {
			case test.wantErr == "" && err != nil:
				t.Fatalf("checkDecodeAllowed(%q) = %v, want accepted", test.tester, err)
			case test.wantErr != "" && (err == nil || !strings.Contains(err.Error(), test.wantErr)):
				t.Fatalf("checkDecodeAllowed(%q) = %v, want an error containing %q", test.tester, err, test.wantErr)
			}
		})
	}
}

// TestCheckDecodeCatchesAlteredLaunch plays the fully restricted bot with the
// decode check on. Its games must all pass, and between them include a
// launch aimed straight because aiming ahead failed, so the decoder's
// fallback is checked too. Then the same games are played with one launch's
// angle moved by the smallest step a float64 can take: each must fail on the
// decode check, which shows the angles are compared exactly.
func TestCheckDecodeCatchesAlteredLaunch(t *testing.T) {
	ctx := context.Background()
	restricted, err := agent.ParseFactory(restrictedSpec)
	if err != nil {
		t.Fatal(err)
	}
	jobs := []job{{tester: specTester, seed: 1_000_000, seat: 0}, {tester: specTester, seed: 1_000_001, seat: 1}}

	// play runs the jobs with the restricted bot as the tester, wrapped in
	// alteredAngle when alter is set, and sums the bots' straight aims.
	play := func(alter bool) ([]gameRecord, agent.AimStats) {
		var total agent.AimStats
		var bots []*agent.HeuristicV2Agent
		config := testConfig(t)
		config.checkDecode = true
		config.newTester = func(ctx context.Context, j job, _ *game.Game) (agent.Agent, error) {
			a, err := restricted.New(ctx, agent.AgentInit{PlayerID: j.seat})
			if err != nil {
				return nil, err
			}
			bot := a.(*agent.HeuristicV2Agent)
			bots = append(bots, bot)
			if alter {
				return alteredAngle{bot}, nil
			}
			return bot, nil
		}
		var records []gameRecord
		for _, j := range jobs {
			records = append(records, playGame(ctx, config, j))
		}
		for _, bot := range bots {
			total.AimedStraight += bot.AimStats().AimedStraight
		}
		return records, total
	}

	records, stats := play(false)
	for _, r := range records {
		if r.Status != statusCompleted || r.Launches == 0 {
			t.Fatalf("record %+v, want a completed game with launches", r)
		}
	}
	if stats.AimedStraight == 0 {
		t.Fatal("no launch was aimed straight; pick maps where aiming ahead fails, so the fallback is checked")
	}

	altered, _ := play(true)
	for _, r := range altered {
		if r.Status != statusFailed || !strings.Contains(r.Message, "decode check") || r.FailSeat == nil || *r.FailSeat != r.Seat {
			t.Fatalf("altered record %+v, want a game failed by the decode check in the tested seat", r)
		}
	}
}

// TestSameLaunchComparesAngleBits checks that launches are compared bit for
// bit: angles of +0 and −0 are equal to Go's ==, but their bits differ, so
// they are different launches.
func TestSameLaunchComparesAngleBits(t *testing.T) {
	launch := game.Action{PlayerID: 1, PlanetID: 7, Ships: 12, Angle: 0}
	if !sameLaunch(launch, launch) {
		t.Fatalf("sameLaunch(%+v, itself) = false, want true", launch)
	}
	negativeZero := launch
	negativeZero.Angle = math.Copysign(0, -1)
	if sameLaunch(launch, negativeZero) {
		t.Fatal("sameLaunch treats angles +0 and −0 as the same, want different")
	}
}

// alteredAngle is the restricted bot with its first launch of each turn
// turned by the smallest possible step. LastChoices still comes from the
// bot, so the choices are learner moves but the launch no longer matches.
type alteredAngle struct{ *agent.HeuristicV2Agent }

func (a alteredAngle) Decide(ctx context.Context, request agent.TurnRequest) (agent.Decision, error) {
	decision, err := a.HeuristicV2Agent.Decide(ctx, request)
	if err == nil && len(decision.Actions) > 0 {
		decision.Actions[0].Angle = math.Nextafter(decision.Actions[0].Angle, math.Inf(1))
	}
	return decision, err
}
