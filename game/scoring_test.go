package game

import (
	"reflect"
	"testing"
)

func TestWinnerBeforeFinished(t *testing.T) {
	g := testGame(t, Planet{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 5, Production: 1}, bystander(99, 2))
	if _, ok := g.Winner(); ok {
		t.Fatal("Winner should not decide a match that has not reached its final turn")
	}
}

func TestWinnerHighestShips(t *testing.T) {
	config := DefaultConfig()
	config.MaxTurns = 1
	g, err := New(config, []Planet{
		{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 5, Production: 1},
		{ID: 2, Position: Vec2{80, 80}, OwnerID: 2, Ships: 3, Production: 1},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	if !g.Finished() {
		t.Fatal("game should be finished after its one and only turn")
	}
	leaders, ok := g.Winner()
	if !ok || !reflect.DeepEqual(leaders, []int{1}) {
		t.Fatalf("Winner() = %v, %v, want [1], true", leaders, ok)
	}
}

// TestWinnerTieIsExplicit is PLAN.md's correction in effect: an exact tie at
// the top must be represented as the set of tied leaders, not silently
// declined. This replaces the pre-PR8 behavior, where a tie and an unfinished
// match were both just `ok == false` and indistinguishable from each other.
func TestWinnerTieIsExplicit(t *testing.T) {
	config := DefaultConfig()
	config.MaxTurns = 1
	g, err := New(config, []Planet{
		{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 5, Production: 1},
		{ID: 2, Position: Vec2{80, 80}, OwnerID: 2, Ships: 5, Production: 1},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	leaders, ok := g.Winner()
	if !ok {
		t.Fatal("a finished match with a tied top score should still report ok = true")
	}
	if !reflect.DeepEqual(leaders, []int{1, 2}) {
		t.Fatalf("Winner() leaders = %v, want [1 2] (sorted, both tied)", leaders)
	}
}

// TestWinnerThreeWayTie covers more than two tied leaders, and that the
// result stays sorted regardless of Standings' unordered map.
func TestWinnerThreeWayTie(t *testing.T) {
	config := DefaultConfig()
	config.MaxTurns = 1
	g, err := New(config, []Planet{
		{ID: 1, Position: Vec2{10, 10}, OwnerID: 3, Ships: 5, Production: 1},
		{ID: 2, Position: Vec2{30, 30}, OwnerID: 1, Ships: 5, Production: 1},
		{ID: 3, Position: Vec2{50, 50}, OwnerID: 2, Ships: 5, Production: 1},
		{ID: 4, Position: Vec2{70, 70}, OwnerID: 0, Ships: 2, Production: 1},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	leaders, ok := g.Winner()
	if !ok || !reflect.DeepEqual(leaders, []int{1, 2, 3}) {
		t.Fatalf("Winner() = %v, %v, want [1 2 3], true", leaders, ok)
	}
}

// TestStandingsCountsCometShips is PLAN.md's other correction: an owned
// comet's ships count toward its owner's total exactly like a planet's,
// reversing the pre-PR8 behavior where comets were deliberately excluded.
func TestStandingsCountsCometShips(t *testing.T) {
	g, err := New(DefaultConfig(),
		[]Planet{{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 5, Production: 1}},
		[]Comet{{ID: 2, OwnerID: 1, Ships: 7, Path: fixedCometPath(Vec2{60, 60})}},
	)
	if err != nil {
		t.Fatal(err)
	}
	standings := g.Standings()
	if got := standings[1]; got != 12 {
		t.Fatalf("player 1 standing = %d, want 12 (5 planet + 7 comet)", got)
	}
}

// TestStandingsExcludesNeutralComet guards the obvious counterpart: an
// unclaimed comet's ships belong to nobody's total.
func TestStandingsExcludesNeutralComet(t *testing.T) {
	g, err := New(DefaultConfig(), nil,
		[]Comet{{ID: 1, OwnerID: NeutralPlayer, Ships: 9, Path: fixedCometPath(Vec2{60, 60})}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if standings := g.Standings(); len(standings) != 0 {
		t.Fatalf("standings = %v, want empty (comet is neutral)", standings)
	}
}

// TestEliminationEndsMatchEarly covers Finished()'s new second condition: a
// match that began as a genuine two-player contest ends the instant it
// narrows to one owner, well before the turn limit. The geometry mirrors
// TestLaunchAndCapture (game_test.go): planet 1 (owner 1) captures planet 2
// (owner 2) in a single turn, and since these are the *only* two planets on
// the board, that capture leaves exactly one distinct owner.
func TestEliminationEndsMatchEarly(t *testing.T) {
	config := DefaultConfig()
	config.MaxTurns = 500
	g, err := New(config, []Planet{
		{ID: 1, Position: Vec2{10, 10}, OwnerID: 1, Ships: 7, Production: 1},
		{ID: 2, Position: Vec2{13, 10}, OwnerID: 2, Ships: 2, Production: 1},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if g.Finished() {
		t.Fatal("test setup: match should not start finished — two distinct owners are present")
	}
	if err := g.AdvanceTurn([]Action{{PlayerID: 1, PlanetID: 1, Ships: 5, Angle: 0}}); err != nil {
		t.Fatal(err)
	}
	if g.Planets[1].OwnerID != 1 {
		t.Fatalf("test setup: expected planet 2 captured by player 1, owner = %d", g.Planets[1].OwnerID)
	}
	if !g.Finished() {
		t.Fatal("match should have ended: player 1 just captured the only other owner's last planet, well before MaxTurns (500)")
	}
	leaders, ok := g.Winner()
	if !ok || !reflect.DeepEqual(leaders, []int{1}) {
		t.Fatalf("Winner() = %v, %v, want [1], true", leaders, ok)
	}
}

// TestEliminationNotReachedWithTwoOwners is the counterpart guarding against
// a too-eager elimination check: two distinct owners must not end the match.
func TestEliminationNotReachedWithTwoOwners(t *testing.T) {
	config := DefaultConfig()
	config.MaxTurns = 500
	g, err := New(config, []Planet{
		{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 5, Production: 1},
		{ID: 2, Position: Vec2{80, 80}, OwnerID: 2, Ships: 3, Production: 1},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if g.Finished() {
		t.Fatal("two distinct owners on the board should not be finished")
	}
}

// TestEliminationIgnoresNeutralOwners confirms a neutral planet never counts
// toward the distinct-owner tally: it doesn't prevent elimination once only
// one real owner is left standing. Same capture geometry as
// TestEliminationEndsMatchEarly, plus an untouched neutral planet elsewhere
// on the board that never enters combat.
func TestEliminationIgnoresNeutralOwners(t *testing.T) {
	config := DefaultConfig()
	config.MaxTurns = 500
	g, err := New(config, []Planet{
		{ID: 1, Position: Vec2{10, 10}, OwnerID: 1, Ships: 7, Production: 1},
		{ID: 2, Position: Vec2{13, 10}, OwnerID: 2, Ships: 2, Production: 1},
		{ID: 3, Position: Vec2{80, 80}, OwnerID: NeutralPlayer, Ships: 4, Production: 1},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.AdvanceTurn([]Action{{PlayerID: 1, PlanetID: 1, Ships: 5, Angle: 0}}); err != nil {
		t.Fatal(err)
	}
	if g.Planets[1].OwnerID != 1 {
		t.Fatalf("test setup: expected planet 2 captured by player 1, owner = %d", g.Planets[1].OwnerID)
	}
	if !g.Finished() {
		t.Fatal("player 1 is the only real owner left; the untouched neutral planet must not prevent elimination")
	}
}

// TestEliminationSurvivesZeroShipOwnership guards against elimination
// triggering just because a garrison was fought down to 0 ships. Ownership
// persists through that (a Stage 2 combat tie leaves the garrison's owner in
// place at 0 ships rather than resetting to neutral), so a second player
// still owning a 0-ship planet must still count as a distinct owner.
func TestEliminationSurvivesZeroShipOwnership(t *testing.T) {
	config := DefaultConfig()
	config.MaxTurns = 500
	g, err := New(config, []Planet{
		{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 5, Production: 1},
		{ID: 2, Position: Vec2{80, 80}, OwnerID: 2, Ships: 0, Production: 1},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if g.Finished() {
		t.Fatal("a 0-ship planet is still owned by its player, not eliminated")
	}
}

// TestEliminationIsUnconditional directly confirms PLAN.md's rule has no
// exception for a match that only ever had one owner: a single-owner setup
// is finished the instant it's constructed, before a single turn runs. This
// pins the corrected behavior after an earlier, incorrect implementation
// gated elimination on having started with 2+ owners — a deviation from
// spec that silently exempted every single-player setup instead.
func TestEliminationIsUnconditional(t *testing.T) {
	g, err := New(DefaultConfig(),
		[]Planet{{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 5, Production: 1}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !g.Finished() {
		t.Fatal("a game with only one owner from construction should already be finished")
	}
	if err := g.AdvanceTurn(nil); err == nil {
		t.Fatal("AdvanceTurn should refuse to run a turn on an already-finished game")
	}
}

// TestEliminationSurvivesActiveFleetAfterLastBodyLost is PLAN.md's fleet
// clause in effect: a player who owns no planet or comet at all is not
// eliminated as long as they still have an active fleet in flight. Without
// counting fleets, this board (one real planet owner, plus a fleet owned by
// a different player who owns nothing else) would incorrectly read as
// "at most one owner" and end the match.
func TestEliminationSurvivesActiveFleetAfterLastBodyLost(t *testing.T) {
	config := DefaultConfig()
	g, err := New(config,
		[]Planet{{ID: 1, Position: Vec2{80, 80}, OwnerID: 2, Ships: 3, Production: 1}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Player 1 owns no planet or comet anywhere, but still has a fleet in
	// flight — that alone must count as a distinct, surviving owner.
	g.Fleets = []Fleet{{ID: 99, OwnerID: 1, Ships: 4, Position: Vec2{20, 20}}}
	if g.Finished() {
		t.Fatal("player 1 still owns an active fleet after losing every planet and comet; the match should not be finished")
	}
}

// TestWinnerOnFinishedEmptyBoard is the other half of ok's contract: false
// must mean only "not finished yet", never "finished but nobody's here to
// have a standing". A board where every planet is neutral has zero distinct
// owners, which already satisfies "at most one" — so it's finished by
// elimination from construction, with nobody in Standings at all. Winner()
// must still report ok == true, just with no leaders.
func TestWinnerOnFinishedEmptyBoard(t *testing.T) {
	g, err := New(DefaultConfig(),
		[]Planet{{ID: 1, Position: Vec2{20, 20}, OwnerID: NeutralPlayer, Ships: 5, Production: 1}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !g.Finished() {
		t.Fatal("test setup: an all-neutral board should already be finished (elimination: zero owners)")
	}
	leaders, ok := g.Winner()
	if !ok {
		t.Fatal(`ok = false on a finished game — false must mean only "not finished yet"`)
	}
	if len(leaders) != 0 {
		t.Fatalf("leaders = %v, want none (nobody owns anything)", leaders)
	}
}
