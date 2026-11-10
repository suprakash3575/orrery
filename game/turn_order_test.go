package game

import (
	"math"
	"testing"
)

// TestFleetAndBodyMustMeetAtTheSameInstant pins the simultaneous collision
// rule. The fleet lands on the *midpoint* of the planet's path this turn, but
// only at the end of the turn; the planet passes that midpoint halfway
// through. The two are never within the planet's radius at the same instant,
// so the fleet must not arrive.
//
// The former two-pass check (fleets moved first, then bodies swept over where
// fleets ended up) reported exactly this pair as an arrival, which is the
// false collision the relative-motion check removes. The midpoint sits
// farther than the planet's radius from both the planet's start and end
// position, so a plain fixed-circle check cannot catch it either way.
func TestFleetAndBodyMustMeetAtTheSameInstant(t *testing.T) {
	g, err := New(DefaultConfig(),
		[]Planet{{ID: 1, OwnerID: 2, Ships: 2, Production: 1, Position: Vec2{80, 50}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	const rate = 0.1
	g.Planets[0].Orbit.RadiansTurn = rate

	sun, radius := g.Config.Sun, g.Planets[0].Orbit.Radius
	midpoint := sun.Add(Vec2{radius * math.Cos(rate/2), radius * math.Sin(rate/2)})
	if d := midpoint.Distance(g.Planets[0].Position); d <= g.Planets[0].Radius {
		t.Fatalf("test setup: midpoint %v is within the planet's own radius of its start position, distance %v", midpoint, d)
	}

	// Starts well clear of the planet entirely; its velocity carries it
	// exactly to the swept arc's midpoint this turn.
	start := Vec2{60, midpoint.Y}
	g.Fleets = []Fleet{{ID: 99, OwnerID: 1, Ships: 5, Position: start, Velocity: midpoint.Sub(start)}}

	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}

	if len(g.Fleets) != 1 {
		t.Fatal("fleet was claimed by a planet it never met at the same instant; it should still be in flight")
	}
	// Garrison 2 + Production 1 = 3, untouched by any attack.
	if g.Planets[0].OwnerID != 2 || g.Planets[0].Ships != 3 {
		t.Fatalf("planet held by %d with %d ships, want owner 2 with 3", g.Planets[0].OwnerID, g.Planets[0].Ships)
	}
}

// TestProduceRunsBeforeAnyMovement pins produce's new position in the order:
// right after launch, before fleets or bodies move. This is really only
// observable as "produce still happens exactly once, before combat" — its
// position relative to *movement* specifically has no test-visible effect
// today, since production depends only on ship counts, never position. The
// regression this guards is a future change to production that *does* depend
// on position (e.g. a planet's post-move location) silently reading stale
// data if produce ever drifted back to running after movement.
func TestProduceRunsBeforeAnyMovement(t *testing.T) {
	g := testGame(t, Planet{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 4, Production: 3}, bystander(99, 2))
	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	if got := g.Planets[0].Ships; got != 7 {
		t.Fatalf("ships = %d, want 7 (4 + Production 3)", got)
	}
}
