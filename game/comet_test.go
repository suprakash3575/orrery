package game

import (
	"testing"
)

// TestCometExpiresAfterItsPath covers expireComets directly, marking a comet
// as departed by hand (PathIndex past its last point) rather than stepping it
// there turn by turn.
func TestCometExpiresAfterItsPath(t *testing.T) {
	g, err := New(DefaultConfig(), nil, []Comet{
		{ID: 1, Radius: 1, OwnerID: NeutralPlayer, Path: fixedCometPath(Vec2{10, 10})},
	})
	if err != nil {
		t.Fatal(err)
	}
	g.Comets[0].PathIndex = len(g.Comets[0].Path)

	g.expireComets()

	if len(g.Comets) != 0 {
		t.Fatalf("a comet past the end of its path should have been dropped, got %+v", g.Comets)
	}
}

// TestExpiryRunsBeforeLaunchValidation is the reason expiry moved to the start
// of the turn at all (IMPLEMENTATION-PLAN.md PR6): a comet that left on a
// previous turn must already be gone by the time this turn's launches are
// checked, so nobody can launch from a comet that no longer exists.
func TestExpiryRunsBeforeLaunchValidation(t *testing.T) {
	g, err := New(DefaultConfig(), []Planet{bystander(99, 2)}, []Comet{
		{ID: 1, OwnerID: 1, Ships: 5, Path: fixedCometPath(Vec2{10, 10})},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Simulates the comet having finished its path on a prior turn, before
	// this AdvanceTurn call even begins.
	g.Comets[0].PathIndex = len(g.Comets[0].Path)

	err = g.AdvanceTurn([]Action{{PlayerID: 1, PlanetID: 1, Ships: 1, Angle: 0}})

	if err == nil {
		t.Fatal("launching from a comet that expired at the start of this turn should be rejected")
	}
	if len(g.Comets) != 0 {
		t.Fatalf("the expired comet should already be gone, got %+v", g.Comets)
	}
}

func TestLaunchFromCometCapturesPlanet(t *testing.T) {
	// Mirrors TestLaunchAndCapture in game_test.go, but with the attacker's
	// source flipped to a comet: comet (owner 1, 7 ships) vs. planet (owner
	// 2, 2 ships, Production 1), 3 units apart, same as the planet-to-planet
	// case. Proves launch() treats a comet source identically to a planet.
	g, err := New(DefaultConfig(),
		[]Planet{{ID: 2, Position: Vec2{13, 10}, OwnerID: 2, Ships: 2, Production: 1}},
		[]Comet{{ID: 1, OwnerID: 1, Ships: 7, Path: fixedCometPath(Vec2{10, 10})}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.AdvanceTurn([]Action{{PlayerID: 1, PlanetID: 1, Ships: 5, Angle: 0}}); err != nil {
		t.Fatal(err)
	}
	// Same arithmetic as TestLaunchAndCapture: planet 2's garrison grows to 3
	// (2 + Production 1) before combat, so the 5-ship attack wins and leaves
	// the attacker with 5-3=2 ships.
	if got := g.Planets[0].OwnerID; got != 1 {
		t.Fatalf("owner = %d, want 1", got)
	}
	if got := g.Planets[0].Ships; got != 2 {
		t.Fatalf("ships = %d, want 2", got)
	}
	// The comet is still owned by player 1 after launching, so it produces
	// too: 7 - 5 launched = 2, then +1 from produce() = 3.
	if got := g.Comets[0].Ships; got != 3 {
		t.Fatalf("comet garrison after launching 5 of 7 ships and producing = %d, want 3", got)
	}
}

func TestLaunchRejectsInsufficientCometShips(t *testing.T) {
	g, err := New(DefaultConfig(), []Planet{bystander(99, 2)}, []Comet{
		{ID: 1, OwnerID: 1, Ships: 3, Path: fixedCometPath(Vec2{10, 10})},
	})
	if err != nil {
		t.Fatal(err)
	}
	// Without the bystander, this game would already be finished (a single
	// owner satisfies elimination immediately), and AdvanceTurn would return
	// "game is already finished" — a different error that this check can't
	// distinguish from the one actually under test.
	err = g.AdvanceTurn([]Action{{PlayerID: 1, PlanetID: 1, Ships: 4, Angle: 0}})
	if err == nil {
		t.Fatal("launching more ships than the comet has should be rejected")
	}
	if g.Finished() {
		t.Fatalf("test setup: game should not be finished, got error %q instead of the expected insufficient-ships rejection", err)
	}
}

func TestCometCapture(t *testing.T) {
	g, err := New(DefaultConfig(),
		[]Planet{{ID: 1, Position: Vec2{10, 20}, OwnerID: 1, Ships: 7, Production: 1}, bystander(99, 2)},
		[]Comet{{ID: 2, Radius: 1, OwnerID: NeutralPlayer, Ships: 2, Path: fixedCometPath(Vec2{13, 20})}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.AdvanceTurn([]Action{{PlayerID: 1, PlanetID: 1, Ships: 4, Angle: 0}}); err != nil {
		t.Fatal(err)
	}
	if len(g.Comets) != 1 {
		t.Fatalf("comet should still be on the board, got %+v", g.Comets)
	}
	if got := g.Comets[0].OwnerID; got != 1 {
		t.Fatalf("owner = %d, want 1", got)
	}
	if got := g.Comets[0].Ships; got != 2 {
		t.Fatalf("ships = %d, want 2", got)
	}
}

// TestCometExpiresWithinItsDepartingTurn ties expireComets and moveComets
// together through AdvanceTurn: a comet that steps past its last point on
// turn N leaves before turn N's combat and scoring, as in the reference engine, so it is
// already gone when turn N ends and its ships no longer count.
func TestCometExpiresWithinItsDepartingTurn(t *testing.T) {
	// A one-point path: the comet's first move takes it past its last point.
	g, err := New(DefaultConfig(), []Planet{bystander(99, 2)}, []Comet{{
		ID: 1, OwnerID: 1, Ships: 2, Path: []Vec2{{99.5, 50}},
	}})
	if err != nil {
		t.Fatal(err)
	}

	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	if len(g.Comets) != 0 {
		t.Fatalf("comet should have expired within the turn it departs on, got %+v", g.Comets)
	}
	if _, counted := g.Standings()[1]; counted {
		t.Fatal("an expired comet's ships must not count in the standings")
	}
}
