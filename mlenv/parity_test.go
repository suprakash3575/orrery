package mlenv

import (
	"math/rand"
	"testing"

	"orrery/game"
)

// TestPredictedArrivalsMatchEngineMovement is the test that closes the last
// open item of the rules-engine gate (AGENTS.md, "Rules-engine gate before
// ML", item 7): collision-prediction parity between the engine and the
// incoming-fleet observation. Nothing about this package's static features
// depended on that gate — it was always about whether a prediction here is
// something the engine will actually do.
//
// Every other piece of incoming.go's correctness is already covered
// elsewhere: FirstHit's own geometry was validated exhaustively in
// game/predict_test.go, and TestArrivalMatchesClosedFormGeometry checks one
// fleet's number against an independent calculation. What neither of those
// touches is the thing unique to this package: repeatedly re-querying
// FirstHit turn by turn, across many fleets in the same state, and turning
// that into the "arrives at body B on turn N" claim a caller acts on.
//
// Proving the *timing* is straightforward: a fleet's ID is either in
// g.Fleets or it isn't, and that is unambiguous ground truth from the engine
// itself. Proving the *body* is not, because owned planets change every turn
// through production regardless of what lands on them — comparing a body's
// state "before everything" to "after everything" would pass even for a
// fleet predicted at the wrong body, since production alone guarantees a
// difference. The fix is a control: for every fleet predicted to land on a
// body, run a second, otherwise-identical game with only that one fleet
// removed, advance both the same number of turns with no new actions, and
// require the target body's state to actually differ between the two runs.
// Production is identical in both, so a difference isolates this fleet's own
// contribution, not just the passage of time.
func TestPredictedArrivalsMatchEngineMovement(t *testing.T) {
	exercised, bodyChecks := 0, 0

	for seed := range int64(300) {
		g := generated(t, seed)
		playRandomTurns(t, g, seed)
		if g.Finished() || len(g.Fleets) == 0 {
			continue
		}

		type expectation struct {
			bodyID   int // 0 for a hazard death: no body to check
			arriveOn int // AdvanceTurn calls from now, 1-indexed
		}
		expected := make(map[int]expectation, len(g.Fleets)) // keyed by fleet ID
		maxCalls := 0
		landings, err := predictFleets(g)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		for _, landing := range landings {
			arriveOn := landing.Segment + 1 // engine resolves a hit at the end of the call that performs that segment's move, whatever the fractional T
			bodyID := 0
			if landing.Arrives() {
				bodyID = landing.Hit.BodyID
			}
			expected[landing.FleetID] = expectation{bodyID: bodyID, arriveOn: arriveOn}
			if arriveOn > maxCalls {
				maxCalls = arriveOn
			}
		}
		exercised++

		// Each fleet's control clone is built from this exact starting point
		// and advanced only as far as THAT fleet's own predicted arrival —
		// never to a shared maxCalls. Advancing every control to the same
		// call count seems natural but is wrong: a fleet predicted to arrive
		// on call 5 would have its control also run calls 6, 7, ... up to
		// whatever the seed's slowest fleet needs, and any production or
		// combat in those extra calls (identical in both branches, but
		// *after* the moment being tested) has nothing to do with this
		// fleet — comparing there conflates "everything that happened since"
		// with "what this fleet did". Stopping exactly at arriveOn is what
		// keeps the comparison scoped to only this fleet's own contribution.
		controlBodyAtArrival := make(map[int]bodySnapshot, len(expected))
		for id, exp := range expected {
			if exp.bodyID == 0 {
				continue // hazard deaths need no body comparison
			}
			control := cloneWithoutFleet(t, g, id)
			reachedArrival := true
			for call := 1; call <= exp.arriveOn; call++ {
				if control.Finished() {
					// Removing this fleet ended the control match before its
					// predicted arrival turn — e.g. it was the sole
					// remaining representative of its owner. Not a
					// misprediction, just not checkable this way.
					reachedArrival = false
					break
				}
				if err := control.AdvanceTurn(nil); err != nil {
					t.Fatalf("seed %d control fleet %d call %d: %v", seed, id, call, err)
				}
			}
			if reachedArrival {
				controlBodyAtArrival[id] = findBody(control, exp.bodyID)
			}
		}

		// The one real trajectory, shared across every fleet: disappearance
		// timing needs the whole run regardless of outcome, and a body
		// snapshot recorded right after the call where some fleet is
		// predicted to land gives every fleet's own arrival-turn state
		// without re-running the real game once per fleet.
		seenGoneOn := make(map[int]int, len(expected))
		realBodyAtArrival := make(map[int]bodySnapshot, len(expected)) // keyed by fleet ID
		for call := 1; call <= maxCalls; call++ {
			if g.Finished() {
				break
			}
			if err := g.AdvanceTurn(nil); err != nil {
				t.Fatalf("seed %d call %d: %v", seed, call, err)
			}
			stillFlying := make(map[int]bool, len(g.Fleets))
			for _, f := range g.Fleets {
				stillFlying[f.ID] = true
			}
			for id := range expected {
				if _, seen := seenGoneOn[id]; seen || stillFlying[id] {
					continue
				}
				seenGoneOn[id] = call
			}
			for id, exp := range expected {
				if exp.arriveOn == call && exp.bodyID != 0 {
					realBodyAtArrival[id] = findBody(g, exp.bodyID)
				}
			}
		}

		for id, exp := range expected {
			gone, seen := seenGoneOn[id]
			if !seen {
				if g.Finished() {
					continue // elimination ended the match first; not a misprediction
				}
				t.Fatalf("seed %d fleet %d: predicted to resolve on call %d, but was still in flight after %d calls",
					seed, id, exp.arriveOn, maxCalls)
			}
			if gone != exp.arriveOn {
				t.Fatalf("seed %d fleet %d: predicted to resolve on call %d, actually vanished on call %d",
					seed, id, exp.arriveOn, gone)
			}
			if exp.bodyID == 0 {
				continue // hazard death: timing alone is the whole claim
			}
			controlBody, ok := controlBodyAtArrival[id]
			if !ok {
				continue // control's own match ended before the predicted turn
			}
			realBody, ok := realBodyAtArrival[id]
			if !ok {
				continue // g itself ended before reaching the predicted turn
			}
			bodyChecks++
			if realBody == controlBody {
				t.Fatalf("seed %d fleet %d: body %d is identical with and without this fleet right after call %d — "+
					"the predicted arrival had no observable effect", seed, id, exp.bodyID, exp.arriveOn)
			}
		}
	}

	if exercised == 0 {
		t.Fatal("no seed in the sample produced a resolvable in-flight fleet; the parity property was never exercised")
	}
	if bodyChecks == 0 {
		t.Fatal("no fleet in the sample resolved at a body; the control-comparison property was never exercised")
	}
}

// TestParityDestructionAtTheBoundary is the explicit boundary-death case
// TestPredictedArrivalsMatchEngineMovement's random sweep cannot be relied on
// to produce: a played-forward generated map rarely happens to send a fleet
// off the edge within a short window. Built by hand instead, so the case is
// guaranteed rather than hoped for.
func TestParityDestructionAtTheBoundary(t *testing.T) {
	config := game.DefaultConfig()
	config.SunHazardEnabled = false
	config.CometSpawningEnabled = false
	g, err := game.New(config, []game.Planet{
		{ID: 1, Position: game.Vec2{X: 95, Y: 50}, OwnerID: 0, Ships: 20, Production: 1},
		{ID: 2, Position: game.Vec2{X: 5, Y: 5}, OwnerID: 1, Ships: 1, Production: 1},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Aimed at +x, straight off the right edge, with nothing else in the way.
	if err := g.AdvanceTurn([]game.Action{{PlayerID: 0, PlanetID: 1, Ships: 10, Angle: 0}}); err != nil {
		t.Fatal(err)
	}
	verifyDeath(t, g, game.HitBoundary)
}

// TestParityDestructionAtTheSun is the explicit sun-death case: every
// generated V1 map disables the sun hazard, so the random sweep can never
// produce one. This scenario turns the hazard back on directly.
func TestParityDestructionAtTheSun(t *testing.T) {
	config := game.DefaultConfig() // SunHazardEnabled: true, sun at board centre
	config.CometSpawningEnabled = false
	g, err := game.New(config, []game.Planet{
		{ID: 1, Position: game.Vec2{X: 20, Y: 50}, OwnerID: 0, Ships: 20, Production: 1},
		{ID: 2, Position: game.Vec2{X: 5, Y: 5}, OwnerID: 1, Ships: 1, Production: 1},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Aimed directly at the sun's centre (50,50), on a clear line.
	if err := g.AdvanceTurn([]game.Action{{PlayerID: 0, PlanetID: 1, Ships: 10, Angle: 0}}); err != nil {
		t.Fatal(err)
	}
	verifyDeath(t, g, game.HitSun)
}

// verifyDeath predicts the sole in-flight fleet's fate, asserts it is the
// expected hazard, then actually runs the engine forward and confirms the
// fleet vanishes on exactly the predicted call — the same timing property the
// main sweep checks, guaranteed to exercise a hazard death rather than hoping
// a random seed produces one.
func verifyDeath(t *testing.T, g *game.Game, wantKind game.HitKind) {
	t.Helper()
	if len(g.Fleets) != 1 {
		t.Fatalf("test setup produced %d fleets, want exactly 1", len(g.Fleets))
	}
	fleetID := g.Fleets[0].ID

	landings, err := predictFleets(g)
	if err != nil {
		t.Fatal(err)
	}
	hit := landings[0].Hit
	if hit.Kind != wantKind {
		t.Fatalf("predicted hit kind = %v, want %v", hit.Kind, wantKind)
	}
	if hit.IsBody() {
		t.Fatalf("predicted a body arrival, want a %v death", wantKind)
	}
	arriveOn := landings[0].Segment + 1

	for call := 1; call <= arriveOn; call++ {
		if err := g.AdvanceTurn(nil); err != nil {
			t.Fatalf("call %d: %v", call, err)
		}
		stillFlying := false
		for _, f := range g.Fleets {
			if f.ID == fleetID {
				stillFlying = true
			}
		}
		if call < arriveOn && !stillFlying {
			t.Fatalf("fleet vanished on call %d, predicted call %d", call, arriveOn)
		}
		if call == arriveOn && stillFlying {
			t.Fatalf("fleet still in flight after predicted call %d", arriveOn)
		}
	}
}

// cloneWithoutFleet builds an independent game matching g's current planets,
// comets, and turn number exactly, carrying every in-flight fleet except the
// one named. It exists so a fleet's predicted effect on a body can be
// isolated: advance this clone and g the same number of turns with no new
// actions, and any difference at that body is attributable to the excluded
// fleet, since everything else about the two games is identical.
//
// A fresh Rand is given rather than sharing g's, even though nothing in the
// static, comet-free games this file generates ever reads it during
// AdvanceTurn — cheap insurance against two games silently sharing mutable
// state if that assumption ever changes.
func cloneWithoutFleet(t *testing.T, g *game.Game, excludeFleetID int) *game.Game {
	t.Helper()
	config := g.Config
	config.Rand = rand.New(rand.NewSource(1))

	planets := append([]game.Planet(nil), g.Planets...)
	comets := append([]game.Comet(nil), g.Comets...)
	clone, err := game.New(config, planets, comets)
	if err != nil {
		t.Fatalf("cloning game state: %v", err)
	}
	clone.Turn = g.Turn
	for _, f := range g.Fleets {
		if f.ID != excludeFleetID {
			clone.Fleets = append(clone.Fleets, f)
		}
	}
	return clone
}

func findBody(g *game.Game, id int) bodySnapshot {
	for _, p := range g.Planets {
		if p.ID == id {
			return bodySnapshot{owner: p.OwnerID, ships: p.Ships, exists: true}
		}
	}
	for _, c := range g.Comets {
		if c.ID == id {
			return bodySnapshot{owner: c.OwnerID, ships: c.Ships, exists: true}
		}
	}
	return bodySnapshot{}
}

type bodySnapshot struct {
	owner  int
	ships  int
	exists bool
}
