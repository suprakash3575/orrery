package game

import "testing"

// TestBattle table-drives Battle() directly, covering the cases PLAN.md's
// Step 2 calls out by name: reinforcement, capture, a failed attack, rival
// attackers settling their own contest, and ties at each stage.
func TestBattle(t *testing.T) {
	cases := []struct {
		name          string
		defenderOwner int
		defenderShips int
		arrivals      map[int]int
		wantOwner     int
		wantShips     int
	}{
		{
			name:          "reinforcement",
			defenderOwner: 1, defenderShips: 5,
			arrivals:  map[int]int{1: 3},
			wantOwner: 1, wantShips: 8,
		},
		{
			name:          "capture",
			defenderOwner: 1, defenderShips: 3,
			arrivals:  map[int]int{2: 5},
			wantOwner: 2, wantShips: 2,
		},
		{
			name:          "failed attack, garrison holds",
			defenderOwner: 1, defenderShips: 5,
			arrivals:  map[int]int{2: 3},
			wantOwner: 1, wantShips: 2,
		},
		{
			// Unlike a Stage 1 tie among attackers, a Stage 2 tie against the
			// garrison does not reset the body to neutral — it's treated as a
			// losing attack: the garrison's owner stands, at 0 ships.
			name:          "exact tie against garrison leaves the defender at 0 ships",
			defenderOwner: 1, defenderShips: 4,
			arrivals:  map[int]int{2: 4},
			wantOwner: 1, wantShips: 0,
		},
		{
			name:          "no arrivals, garrison untouched",
			defenderOwner: 1, defenderShips: 5,
			arrivals:  map[int]int{},
			wantOwner: 1, wantShips: 5,
		},
		{
			name:          "unclaimed neutral body with no arrivals stays empty",
			defenderOwner: NeutralPlayer, defenderShips: 0,
			arrivals:  map[int]int{},
			wantOwner: NeutralPlayer, wantShips: 0,
		},
		{
			name:          "two rival attackers: larger survives Stage 1 then takes the garrison",
			defenderOwner: 1, defenderShips: 5,
			arrivals: map[int]int{2: 10, 3: 4},
			// Stage 1: owner 2 (10) beats owner 3 (4), survives with 6.
			// Stage 2: survivor (6) beats garrison (5), captures with 1.
			wantOwner: 2, wantShips: 1,
		},
		{
			name:          "two rival attackers of equal size destroy each other; garrison stands",
			defenderOwner: 1, defenderShips: 5,
			arrivals:  map[int]int{2: 4, 3: 4},
			wantOwner: 1, wantShips: 5,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			owner, ships := Battle(tc.defenderOwner, tc.defenderShips, tc.arrivals)
			if owner != tc.wantOwner || ships != tc.wantShips {
				t.Fatalf("Battle() = (%d, %d), want (%d, %d)", owner, ships, tc.wantOwner, tc.wantShips)
			}
		})
	}
}

// TestBattleTwoStageRegressionVsOldPooledLogic pins down the exact scenario
// PLAN.md's correction targets: a garrison plus two rival attackers, where
// pooling the garrison into one sorted contest (the old behavior) picks a
// different winner than settling the attackers' own contest first.
//
// Old (single-pool) logic sorted all three forces together — garrison (0,5),
// attacker A (1,4), attacker B (2,3) — and walked the ladder including the
// garrison: (0,5) beats (1,4) -> (0,1); (2,3) then upsets (0,1) -> (2,2).
// Result: owner 2 captures the body with 2 ships.
//
// Two-stage logic (the fix) makes the attackers fight each other first,
// garrison held out: A (4) beats B (3) -> survivor (1,1). Only then does the
// survivor meet the garrison: (0,5) beats (1,1) -> garrison holds with 4.
// Result: owner 0 keeps the body with 4 ships — the opposite outcome.
func TestBattleTwoStageRegressionVsOldPooledLogic(t *testing.T) {
	owner, ships := Battle(0, 5, map[int]int{1: 4, 2: 3})
	if owner != 0 || ships != 4 {
		t.Fatalf("Battle() = (%d, %d), want (0, 4) — the two-stage result; "+
			"(2, 2) would mean the old pooled-ladder behavior regressed back in", owner, ships)
	}
}
