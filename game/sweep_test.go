package game

import (
	"math"
	"testing"
)

// The shared fixture for these tests: a planet orbiting at radius 30, rotating
// 0.1 radians, runs from (80,50) to (79.850, 52.995). sweptFleetPosition sits
// 0.025 units off that path — comfortably inside the planet's radius of 1, and
// far enough out from the sun at (50,50) that nothing here is decided by the
// sun hazard rather than by the sweep.
var sweptFleetPosition = Vec2{79.9, 51.5}

// sweptGame builds a game and parks one fleet at position, in flight and going
// nowhere, so that anything which happens to it came from a body's movement
// rather than its own. A planet given a Position within OrbitRadiusLimit of
// the sun orbits; rates are set after New(), since New() assigns the shared
// rate itself and these tests need exact geometry.
func sweptGame(t *testing.T, position Vec2, planets []Planet, comets []Comet, rates map[int]float64) *Game {
	t.Helper()
	g, err := New(DefaultConfig(), planets, comets)
	if err != nil {
		t.Fatal(err)
	}
	for id, rate := range rates {
		if p := g.planet(id); p != nil && p.Orbit != nil {
			p.Orbit.RadiansTurn = rate
		}
	}
	g.Fleets = []Fleet{{ID: 99, OwnerID: 0, Ships: 5, Position: position}}
	return g
}

// TestMovingPlanetSweepsUpStationaryFleet is the gap this PR closes: a planet
// driving straight over a fleet used to do nothing at all.
func TestMovingPlanetSweepsUpStationaryFleet(t *testing.T) {
	g := sweptGame(t, sweptFleetPosition,
		[]Planet{{ID: 1, OwnerID: 1, Ships: 2, Production: 1, Position: Vec2{80, 50}}},
		nil, map[int]float64{1: 0.1})

	sweeps := g.moveOrbits(nil)
	arrivals := g.moveFleets(sweeps, nil)

	if len(arrivals) != 1 || arrivals[0].bodyID != 1 {
		t.Fatalf("got arrivals %+v, want one at planet 1", arrivals)
	}
	if len(g.Fleets) != 0 {
		t.Fatal("a swept-up fleet must be consumed, not left in flight")
	}
	if arrivals[0].ownerID != 0 || arrivals[0].ships != 5 {
		t.Fatalf("arrival carried owner %d with %d ships, want owner 0 with 5", arrivals[0].ownerID, arrivals[0].ships)
	}
}

// TestSweptFleetFightsForTheBody carries that through a full turn: being run
// over is an attack on the body, resolved with everything else that turn.
func TestSweptFleetFightsForTheBody(t *testing.T) {
	g := sweptGame(t, sweptFleetPosition,
		[]Planet{{ID: 1, OwnerID: 1, Ships: 2, Production: 1, Position: Vec2{80, 50}}},
		nil, map[int]float64{1: 0.1})

	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	// Garrison 2 + 1 production = 3, met by the swept fleet's 5: the attacker
	// takes the planet with the surplus.
	if g.Planets[0].OwnerID != 0 || g.Planets[0].Ships != 2 {
		t.Fatalf("planet held by %d with %d ships, want owner 0 with 2", g.Planets[0].OwnerID, g.Planets[0].Ships)
	}
}

// TestSweepIgnoresFleetsLaunchedThisTurn is PR4's launch-source exclusion
// applied to the reverse check. Fleets move before bodies do (since PR7), so
// by the time this planet sweeps, the fleet it just launched has already
// moved on to its own post-movement position — without the exclusion, a
// planet would routinely sweep up the fleets it had only just launched, once
// its own path reaches that position, the same defect as flying into your own
// source, arriving down a different code path.
//
// Radius 30 (not 10) is what makes this genuinely reach that position: at
// radius 10, even this rate moves the planet nowhere near a 1-ship fleet's
// post-movement point, so a smaller radius would pass this test whether or
// not the exclusion existed at all.
func TestSweepIgnoresFleetsLaunchedThisTurn(t *testing.T) {
	g, err := New(DefaultConfig(),
		[]Planet{{ID: 1, OwnerID: 1, Ships: 10, Production: 1, Position: Vec2{80, 50}}, bystander(99, 2)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// A large rate, so the planet plainly overruns the fleet it just launched.
	g.Planets[0].Orbit.RadiansTurn = 0.1
	if err := g.AdvanceTurn([]Action{{PlayerID: 1, PlanetID: 1, Ships: 1, Angle: math.Pi / 2}}); err != nil {
		t.Fatal(err)
	}
	if len(g.Fleets) != 1 {
		t.Fatalf("launched fleet was swept up by the source it just left; %d fleets remain", len(g.Fleets))
	}
}

// TestSweepClaimsFleetFromItsSourceAfterLaunchTurn is the matching edge: the
// exclusion lasts one turn, so a planet that later catches a fleet up does
// claim it. Without this the sweep would have a permanent blind spot.
func TestSweepClaimsFleetFromItsSourceAfterLaunchTurn(t *testing.T) {
	g := sweptGame(t, sweptFleetPosition,
		[]Planet{{ID: 1, OwnerID: 1, Ships: 2, Production: 1, Position: Vec2{80, 50}}},
		nil, map[int]float64{1: 0.1})
	g.Turn = 5
	g.Fleets[0].FromPlanetID = 1
	g.Fleets[0].LaunchTurn = 2

	arrivals := g.moveFleets(g.moveOrbits(nil), nil)

	if len(arrivals) != 1 || arrivals[0].bodyID != 1 {
		t.Fatalf("got arrivals %+v, want one at planet 1 — the exclusion must not outlive the launch turn", arrivals)
	}
}

// TestStationaryPlanetSweepsNothing keeps the reverse check honest: a planet
// that does not move cannot collect a fleet that does not move either.
func TestStationaryPlanetSweepsNothing(t *testing.T) {
	// (5, 5) is about 63.6 from the sun, outside OrbitRadiusLimit, so the
	// planet is stationary.
	g := sweptGame(t, Vec2{5, 5},
		[]Planet{{ID: 1, Position: Vec2{5, 5}, OwnerID: 1, Ships: 2, Production: 1}}, nil, nil)

	if sweeps := g.moveOrbits(nil); len(sweeps) != 0 {
		t.Fatalf("a stationary planet produced %d sweeps, want none", len(sweeps))
	}
}

// TestZeroRateOrbitSweepsNothing is the case a nil-Orbit check cannot catch: an
// *orbiting* planet (non-nil Orbit) whose rate is exactly 0 does not move
// either, and must produce no path, so collision detection treats it exactly
// like a genuinely stationary planet.
func TestZeroRateOrbitSweepsNothing(t *testing.T) {
	g := sweptGame(t, Vec2{60, 50},
		[]Planet{{ID: 1, OwnerID: 1, Ships: 2, Production: 1, Position: Vec2{60, 50}}},
		nil, map[int]float64{1: 0})

	if sweeps := g.moveOrbits(nil); len(sweeps) != 0 {
		t.Fatalf("a zero-rate orbit produced %d sweeps, want none", len(sweeps))
	}
}

// TestDegenerateCometSweepsNothing is the comet counterpart of
// TestZeroRateOrbitSweepsNothing: a comet whose next path point is the same
// as its current one does not move, so it must produce no path and be tested
// exactly like a stationary body.
func TestDegenerateCometSweepsNothing(t *testing.T) {
	g, err := New(DefaultConfig(), nil, []Comet{{
		ID: 1, OwnerID: 1, Ships: 2, Path: fixedCometPath(Vec2{50, 50}),
	}})
	if err != nil {
		t.Fatal(err)
	}
	sweeps := g.moveComets(nil)
	if len(sweeps) != 0 {
		t.Fatalf("a motionless comet produced %d sweeps, want none", len(sweeps))
	}
	if len(g.Comets) != 1 {
		t.Fatal("test setup: the comet should still be on the board")
	}
}

// TestCometStillSweepsOnItsDepartingTurn pins movement on a comet's
// departing turn: stepping past its last point leaves it where it is, so a
// fleet moving that turn still runs into it and is consumed. AdvanceTurn then
// expires the comet before combat and discards that arrival. moveComets never
// drops anything itself — that is expireComets' job alone.
//
// A one-point path makes this comet's very first move its departing one.
func TestCometStillSweepsOnItsDepartingTurn(t *testing.T) {
	g, err := New(DefaultConfig(), nil, []Comet{{
		ID: 1, OwnerID: 1, Ships: 2, Path: []Vec2{{99.5, 50}},
	}})
	if err != nil {
		t.Fatal(err)
	}

	sweeps := g.moveComets(nil)

	if g.Comets[0].PathIndex < len(g.Comets[0].Path) {
		t.Fatalf("test setup: comet has not finished its path, PathIndex %d", g.Comets[0].PathIndex)
	}
	if len(g.Comets) != 1 {
		t.Fatal("moveComets must not drop the comet itself; only expireComets does that")
	}

	// Placed exactly where the departing comet stays, to prove it still
	// claims a fleet sitting there.
	g.Fleets = []Fleet{{ID: 99, OwnerID: 0, Ships: 5, Position: g.Comets[0].Position}}
	arrivals := g.moveFleets(sweeps, nil)
	if len(arrivals) != 1 || !arrivals[0].isComet || arrivals[0].bodyID != 1 {
		t.Fatalf("got arrivals %+v, want the fleet claimed by comet 1", arrivals)
	}
}

// TestSweptAndDirectArrivalsFightTogether proves the two halves of collision
// detection feed one contest rather than two. A fleet swept up by a planet and
// a rival fleet flying into it in the same turn must meet each other, not
// resolve one after the other.
func TestSweptAndDirectArrivalsFightTogether(t *testing.T) {
	g := sweptGame(t, sweptFleetPosition,
		[]Planet{{ID: 1, OwnerID: NeutralPlayer, Ships: 0, Production: 1, Position: Vec2{80, 50}}},
		nil, map[int]float64{1: 0.1})
	// Player 0's fleet is the swept one from the helper. Player 1 flies a
	// bigger fleet into the same planet on the same turn: from (90,53) it
	// enters the planet's circle at x=80.850, and stays clear of the sun the
	// whole way in.
	g.Fleets = append(g.Fleets, Fleet{ID: 100, OwnerID: 1, Ships: 9, Position: Vec2{90, 53}, Velocity: Vec2{-10, 0}})

	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	// Stage 1: 9 attackers beat the swept 5, leaving 4. Stage 2: the neutral
	// planet is empty, so the survivor takes it with all 4.
	if g.Planets[0].OwnerID != 1 || g.Planets[0].Ships != 4 {
		t.Fatalf("planet held by %d with %d ships, want owner 1 with 4", g.Planets[0].OwnerID, g.Planets[0].Ships)
	}
	if len(g.Fleets) != 0 {
		t.Fatalf("%d fleets survived, want none — both were consumed by the same body", len(g.Fleets))
	}
}
