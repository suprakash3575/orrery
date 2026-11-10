package mlenv

import (
	"math"
	"reflect"
	"strings"
	"testing"

	"orrery/game"
)

// decodeSeed is the generated four-player map every Decode test starts
// from. Each test regenerates it rather than sharing one game, because a
// launch or a fixture change would otherwise leak between cases.
const decodeSeed = 3

// TestDecodeAims checks which heading Decode picks and that it sends the
// whole garrison. Whether LeadAim's heading really intercepts is covered by
// the engine's own tests; here the question is only whether Decode uses it,
// unrotated, and falls back to the target's current centre when it must.
func TestDecodeAims(t *testing.T) {
	tests := []struct {
		name     string
		observer int
		// pick chooses the source and target body IDs. It may change the
		// board first: the frame is fixed at turn 0, so later changes do not
		// disturb it.
		pick func(t *testing.T, g *game.Game, frame SeatFrame) (source, target int)
		// wantLead is whether LeadAim finds a verified heading for the pair.
		// The test asserts it, so a fixture that drifts fails loudly instead
		// of quietly testing the other branch.
		wantLead bool
	}{
		{name: "moving target", observer: 0, pick: pickMovingTarget, wantLead: true},
		// Observer 2's frame turns the board a quarter-turn. LeadAim's angle
		// is in engine coordinates, so matching it exactly shows the frame
		// was not applied to the action.
		{name: "rotated observer", observer: 2, pick: pickMovingTarget, wantLead: true},
		{name: "comet source", observer: 0, pick: pickCometSource, wantLead: true},
		{name: "unverified aim falls back to the centre", observer: 0, pick: pickTargetBehindSun, wantLead: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			g := generatedV2(t, decodeSeed, true)
			frame, err := NewSeatFrame(g, test.observer)
			if err != nil {
				t.Fatal(err)
			}
			source, target := test.pick(t, g, frame)
			obs, err := Build(g, frame)
			if err != nil {
				t.Fatal(err)
			}
			garrison := bodyShips(t, g, source)

			var decisions [BodySlots]SlotAction
			decisions[slotFor(t, obs, source)] = SlotAction{Launch: true, Target: int32(slotFor(t, obs, target))}
			actions, err := Decode(g, frame, obs, decisions)
			if err != nil {
				t.Fatal(err)
			}
			if len(actions) != 1 {
				t.Fatalf("decoded %d actions, want 1", len(actions))
			}
			action := actions[0]
			if action.PlayerID != test.observer || action.PlanetID != source || action.Ships != garrison {
				t.Fatalf("action = %+v, want player %d launching all %d ships from body %d", action, test.observer, garrison, source)
			}

			from, to := bodyPosition(t, g, source), bodyPosition(t, g, target)
			centre := math.Atan2(to.Y-from.Y, to.X-from.X)
			aim, ok := g.LeadAim(source, target, garrison)
			if ok != test.wantLead {
				t.Fatalf("fixture: LeadAim found a heading = %v, want %v", ok, test.wantLead)
			}
			want := centre
			if ok {
				// Equal headings would let a decoder that ignores LeadAim pass.
				if aim.Angle == centre {
					t.Fatal("fixture: lead aim and centre aim agree, so the case cannot tell them apart")
				}
				want = aim.Angle
			}
			if action.Angle != want {
				t.Fatalf("angle = %v, want %v (lead found: %v, centre %v)", action.Angle, want, ok, centre)
			}
			// The decoder's output must also be something the engine accepts,
			// including a launch from a comet.
			if err := g.AdvanceTurn(actions); err != nil {
				t.Fatalf("engine rejected the decoded action: %v", err)
			}
		})
	}
}

// TestDecodeReturnsNothingWhenAnyLaunchIsInvalid checks that one bad
// decision voids the whole turn. Returning the launches decoded before it
// would let a caller submit half a turn without noticing.
//
// Each case puts a valid launch from the home first and the bad decision in a
// later slot, so the valid one has already been decoded when the bad one is
// rejected. Comet slots come after every planet slot, and a generated map has
// no comets at turn 0, so the first comet slot is padding.
func TestDecodeReturnsNothingWhenAnyLaunchIsInvalid(t *testing.T) {
	const padding = firstCometSlot
	tests := []struct {
		name string
		// setup may change the board after the frame is fixed. It runs on the
		// comparison copy too, so the two only differ if decoding changed one.
		setup func(t *testing.T, g *game.Game, homeID int)
		// bad returns the invalid decision's source and target slots.
		bad  func(t *testing.T, g *game.Game, obs Observation, homeID int) (source, target int)
		want string
	}{
		{
			name:  "padding source",
			setup: func(*testing.T, *game.Game, int) {},
			bad: func(t *testing.T, _ *game.Game, obs Observation, homeID int) (int, int) {
				return padding, slotFor(t, obs, homeID)
			},
			want: "source slot 40 is padding",
		},
		{
			// A padding slot's SlotBodyID is 0, which can also be a real body
			// ID. Without the padding check this launch would quietly aim at
			// that body instead of failing.
			name: "padding target",
			setup: func(t *testing.T, g *game.Game, homeID int) {
				p := nextPlanetAfter(t, g, homeID)
				p.OwnerID, p.Ships = 0, 5
			},
			bad: func(t *testing.T, g *game.Game, obs Observation, homeID int) (int, int) {
				return slotFor(t, obs, nextPlanetAfter(t, g, homeID).ID), padding
			},
			want: "target slot 40 is padding",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			g := generatedV2(t, decodeSeed, true)
			frame, err := NewSeatFrame(g, 0)
			if err != nil {
				t.Fatal(err)
			}
			// Find the home before setup, which may give player 0 a second
			// planet, after which homeOf could return either.
			homeID := homeOf(t, g, 0).ID
			test.setup(t, g, homeID)
			obs, err := Build(g, frame)
			if err != nil {
				t.Fatal(err)
			}
			if obs.BodyMask[padding] {
				t.Fatal("fixture: the first comet slot holds a body")
			}

			home := slotFor(t, obs, homeID)
			var decisions [BodySlots]SlotAction
			for target := range BodySlots {
				if obs.TargetMask[home][target] {
					decisions[home] = SlotAction{Launch: true, Target: int32(target)}
					break
				}
			}
			if !decisions[home].Launch {
				t.Fatal("fixture: the home has no allowed target")
			}
			source, target := test.bad(t, g, obs, homeID)
			if source <= home {
				t.Fatalf("fixture: bad decision in slot %d does not follow the home's slot %d", source, home)
			}
			decisions[source] = SlotAction{Launch: true, Target: int32(target)}

			actions, err := Decode(g, frame, obs, decisions)
			if err == nil {
				t.Fatal("Decode returned nil error")
			}
			// The message pins which check fired, so the case cannot pass on
			// some other rejection.
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error %q does not contain %q", err, test.want)
			}
			if actions != nil {
				t.Fatalf("Decode returned %d actions with its error, want none", len(actions))
			}
			// The same seed regenerates the same map, so any difference was
			// made by decoding — including LeadAim's simulation of the valid
			// launch.
			fresh := generatedV2(t, decodeSeed, true)
			test.setup(t, fresh, homeID)
			if !reflect.DeepEqual(g, fresh) {
				t.Fatal("Decode changed the game")
			}
		})
	}
}

// pickMovingTarget launches from the observer's home at the first orbiting
// planet, in ID order, that aiming at its current centre would miss and lead
// aiming would reach.
func pickMovingTarget(t *testing.T, g *game.Game, frame SeatFrame) (source, target int) {
	t.Helper()
	home := homeOf(t, g, frame.Observer())
	for _, p := range collectBodies(g) {
		if p.isComet || p.owner == frame.Observer() || orbitOf(t, g, p.id) == nil {
			continue
		}
		if _, ok := g.LeadAim(home.ID, p.id, home.Ships); !ok {
			continue
		}
		if !centreAimReaches(t, frame.Observer(), home.ID, p.id) {
			return home.ID, p.id
		}
	}
	t.Fatal("fixture: no orbiting planet that only lead aiming reaches")
	return 0, 0
}

// pickCometSource gives the observer a comet clear of every planet and the
// sun, and launches from it at the observer's home.
func pickCometSource(t *testing.T, g *game.Game, frame SeatFrame) (source, target int) {
	t.Helper()
	home := homeOf(t, g, frame.Observer())
	clear := func(p game.Vec2) bool {
		if p.Distance(g.Config.Sun) < g.Config.SunRadius+6 || p.Distance(home.Position) < 20 {
			return false
		}
		for _, planet := range g.Planets {
			if p.Distance(planet.Position) < planet.Radius+6 {
				return false
			}
		}
		return true
	}
	step := game.Vec2{X: g.Config.CometSpeed}
	for x := 10.0; x <= 86; x++ {
		for y := 10.0; y <= 90; y++ {
			start := game.Vec2{X: x, Y: y}
			if !clear(start) || !clear(start.Add(step)) {
				continue
			}
			g.Comets = append(g.Comets, game.Comet{
				ID: 10_000, Position: start, Radius: 1, OwnerID: frame.Observer(), Ships: 9, Production: 1,
				Path: []game.Vec2{start, start.Add(step)},
			})
			// The case is about the source being a comet, so take the first
			// orbiting planet in ID order that lead aiming reaches. It must
			// orbit: a still target's lead and centre headings are the same.
			for _, p := range g.Planets {
				if p.Orbit == nil {
					continue
				}
				if _, ok := g.LeadAim(10_000, p.ID, 9); ok {
					return 10_000, p.ID
				}
			}
			t.Fatal("fixture: lead aiming reaches no planet from the comet")
		}
	}
	t.Fatal("fixture: no clear spot for the comet")
	return 0, 0
}

// pickTargetBehindSun freezes the board and launches from the observer's home
// at the opposite home. Generated opposite homes are mirror images through
// the sun, so with nothing moving the only heading LeadAim tries runs
// straight through the sun and is never verified.
func pickTargetBehindSun(t *testing.T, g *game.Game, frame SeatFrame) (source, target int) {
	t.Helper()
	g.Config.OrbitsEnabled = false
	for i := range g.Planets {
		g.Planets[i].Orbit = nil
	}
	opposite, ok := frame.PlayerFor(RoleOpposite)
	if !ok {
		t.Fatal("fixture: no opposite player")
	}
	return homeOf(t, g, frame.Observer()).ID, homeOf(t, g, opposite).ID
}

// centreAimReaches reports whether a full-garrison launch from source aimed
// at target's current centre would land on target. It plays the launch out on
// a fresh copy of the map so the game under test is untouched.
func centreAimReaches(t *testing.T, player, source, target int) bool {
	t.Helper()
	g := generatedV2(t, decodeSeed, true)
	from, to := bodyPosition(t, g, source), bodyPosition(t, g, target)
	launch := game.Action{
		PlayerID: player, PlanetID: source, Ships: bodyShips(t, g, source),
		Angle: math.Atan2(to.Y-from.Y, to.X-from.X),
	}
	if err := g.AdvanceTurn([]game.Action{launch}); err != nil {
		t.Fatal(err)
	}
	// The map starts with no fleets, so after one turn the launch is the
	// only one still flying — unless it already landed or died.
	if len(g.Fleets) != 1 {
		t.Fatalf("fixture: %d fleets after one turn, want the single launch", len(g.Fleets))
	}
	landing := g.PredictFleets(maxPredictionSegments(g))[0]
	return landing.Arrives() && landing.Hit.BodyID == target
}

// nextPlanetAfter returns the planet with the smallest ID above id. Planets
// fill slots in ID order, so its slot comes straight after id's.
func nextPlanetAfter(t *testing.T, g *game.Game, id int) *game.Planet {
	t.Helper()
	var next *game.Planet
	for i := range g.Planets {
		p := &g.Planets[i]
		if p.ID > id && (next == nil || p.ID < next.ID) {
			next = p
		}
	}
	if next == nil {
		t.Fatalf("fixture: no planet has an ID above %d", id)
	}
	return next
}

// slotFor returns the obs-v2 slot holding body id.
func slotFor(t *testing.T, obs Observation, id int) int {
	t.Helper()
	for slot := range BodySlots {
		if obs.BodyMask[slot] && int(obs.SlotBodyID[slot]) == id {
			return slot
		}
	}
	t.Fatalf("observation has no slot for body %d", id)
	return 0
}

// bodyShips returns body id's current garrison.
func bodyShips(t *testing.T, g *game.Game, id int) int {
	t.Helper()
	for _, body := range collectBodies(g) {
		if body.id == id {
			return body.ships
		}
	}
	t.Fatalf("game has no body %d", id)
	return 0
}

// orbitOf returns planet id's orbit, or nil when it stands still.
func orbitOf(t *testing.T, g *game.Game, id int) *game.Orbit {
	t.Helper()
	for i := range g.Planets {
		if g.Planets[i].ID == id {
			return g.Planets[i].Orbit
		}
	}
	t.Fatalf("game has no planet %d", id)
	return nil
}
