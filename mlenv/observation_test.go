package mlenv

import (
	"math"
	"testing"

	"orrery/game"
	"orrery/scenario"
)

// generatedV2 returns a generated obs-v2 map at turn 0.
func generatedV2(t *testing.T, seed int64, fourPlayer bool) *game.Game {
	t.Helper()
	g, err := scenario.GeneratorV2{FourPlayer: fourPlayer}.New(seed)
	if err != nil {
		t.Fatalf("generate seed %d: %v", seed, err)
	}
	return g
}

// homeOf returns the planet player owns. At turn 0 of a generated map that is
// the player's only planet, its starting home.
func homeOf(t *testing.T, g *game.Game, player int) *game.Planet {
	t.Helper()
	for i := range g.Planets {
		if g.Planets[i].OwnerID == player {
			return &g.Planets[i]
		}
	}
	t.Fatalf("player %d owns no planet", player)
	return nil
}

// TestSeatFrameRolesAndRotation pins every seat's rotation and role table on
// generated maps, and checks that a board breaking the seating is refused.
//
// A wrong role does not fail anywhere: it just labels every opponent column
// with the wrong opponent, and the policy trains on that. So the table is
// asserted directly rather than inferred from other behaviour.
func TestSeatFrameRolesAndRotation(t *testing.T) {
	const none = game.NeutralPlayer
	tests := []struct {
		name       string
		fourPlayer bool
		observer   int
		turns      int
		// players lists who holds each role: me, clockwise, opposite,
		// counter-clockwise.
		players [roleCount]int
	}{
		{"four players, observer 0", true, 0, 0, [roleCount]int{0, 1, 3, 2}},
		{"four players, observer 1", true, 1, 3, [roleCount]int{1, 3, 2, 0}},
		{"four players, observer 2", true, 2, 1, [roleCount]int{2, 0, 1, 3}},
		{"four players, observer 3", true, 3, 2, [roleCount]int{3, 2, 0, 1}},
		{"two players, observer 0", false, 0, 0, [roleCount]int{0, none, 1, none}},
		{"two players, observer 1", false, 1, 2, [roleCount]int{1, none, 0, none}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Several layouts, so the table cannot pass only on one map.
			for seed := range int64(20) {
				g := generatedV2(t, seed, tt.fourPlayer)
				frame, err := NewSeatFrame(g, tt.observer)
				if err != nil {
					t.Fatalf("seed %d: %v", seed, err)
				}
				if frame.quarterTurns != tt.turns {
					t.Fatalf("seed %d: quarter-turns = %d, want %d", seed, frame.quarterTurns, tt.turns)
				}
				for role := range roleCount {
					player, held := frame.PlayerFor(role)
					if !held {
						player = none
					}
					if player != tt.players[role] {
						t.Fatalf("seed %d: role %d held by %d, want %d", seed, role, player, tt.players[role])
					}
					if held {
						if back, ok := frame.RoleOf(player); !ok || back != role {
							t.Fatalf("seed %d: RoleOf(%d) = %d, %v, want %d", seed, player, back, ok, role)
						}
					}
				}
				// The rotation is what puts every observer's home in player 0's
				// corner; the table above only names the turn count.
				got := frame.Position(homeOf(t, g, tt.observer).Position)
				if want := homeOf(t, g, 0).Position; got.Distance(want) > seatHomeTolerance {
					t.Fatalf("seed %d: home turned to (%g, %g), want player 0's home (%g, %g)", seed, got.X, got.Y, want.X, want.Y)
				}
			}
		})
	}

	// Swapping two opponents' homes leaves observer 0's own home in place, so
	// a check of the observer's home alone would accept this board and hand
	// out the wrong roles. Every seated home must be checked.
	t.Run("swapped opponents are refused", func(t *testing.T) {
		g := generatedV2(t, 1, true)
		two, three := homeOf(t, g, 2), homeOf(t, g, 3)
		two.OwnerID, three.OwnerID = 3, 2
		if _, err := NewSeatFrame(g, 0); err == nil {
			t.Fatal("NewSeatFrame accepted a board with players 2 and 3 swapped")
		}
	})
}

// TestBuildSameViewFromEverySeat builds a fourfold-symmetric board and
// checks that all four seats see exactly the same observation.
//
// This is the property obs-v2 exists for: a policy should face the same
// numbers whichever seat it plays. The board has orbiting planets, a comet
// wave, and one fleet per player in flight, launched symmetrically, so the
// rotated positions, owner roles, comet steps, incoming-fleet roles, and
// player rows are all exercised at once. A body's row is matched by its
// rotated position, because the same situation puts different engine bodies
// in a given slot for different seats.
func TestBuildSameViewFromEverySeat(t *testing.T) {
	g := generatedV2(t, 3, true)
	var frames [4]SeatFrame
	for observer := range frames {
		frame, err := NewSeatFrame(g, observer)
		if err != nil {
			t.Fatal(err)
		}
		frames[observer] = frame
	}

	// Play to the first comet wave, then have every player launch the same
	// way: player 0 towards its nearest neutral planet, and each other player
	// at that heading turned by its own seat's quarter-turns. Turning a
	// heading clockwise by a quarter adds pi/2 on this downward-Y board.
	for g.Turn < 50 {
		if err := g.AdvanceTurn(nil); err != nil {
			t.Fatal(err)
		}
	}
	home0 := homeOf(t, g, 0)
	nearest := math.Inf(1)
	var heading float64
	for _, p := range g.Planets {
		if d := p.Position.Distance(home0.Position); p.OwnerID == game.NeutralPlayer && d < nearest {
			nearest = d
			offset := p.Position.Sub(home0.Position)
			heading = math.Atan2(offset.Y, offset.X)
		}
	}
	var launches []game.Action
	for player, position := range fourPlayerCycle {
		launches = append(launches, game.Action{
			PlayerID: player,
			PlanetID: homeOf(t, g, player).ID,
			Ships:    5,
			Angle:    heading + float64(position)*math.Pi/2,
		})
	}
	if err := g.AdvanceTurn(launches); err != nil {
		t.Fatal(err)
	}

	// Guard the fixture: without these the comparison below could pass on a
	// board that never tested the moving parts.
	orbiting := false
	for _, p := range g.Planets {
		orbiting = orbiting || p.Orbit != nil
	}
	if !orbiting || len(g.Comets) != 4 || len(g.Fleets) != 4 {
		t.Fatalf("fixture lacks motion: orbiting=%v comets=%d fleets=%d", orbiting, len(g.Comets), len(g.Fleets))
	}

	// float32 columns computed from turned float64 headings agree to well
	// within this; a wrong role or rotation is off by whole columns.
	const tolerance = 1e-5
	reference, err := Build(g, frames[0])
	if err != nil {
		t.Fatal(err)
	}
	incoming := false
	for slot := range BodySlots {
		row := reference.Bodies[slot]
		for role := range roleCount {
			incoming = incoming || row[FeatureIncomingMe+int(role)] > 0
		}
	}
	if !incoming {
		t.Fatal("fixture has no predicted arrival, so the incoming columns go untested")
	}

	for observer := 1; observer < len(frames); observer++ {
		obs, err := Build(g, frames[observer])
		if err != nil {
			t.Fatal(err)
		}
		matched := 0
		for slot := range BodySlots {
			if !obs.BodyMask[slot] {
				continue
			}
			row := obs.Bodies[slot]
			found := false
			for refSlot := range BodySlots {
				ref := reference.Bodies[refSlot]
				if !reference.BodyMask[refSlot] ||
					math.Abs(float64(row[FeatureX]-ref[FeatureX])) > tolerance ||
					math.Abs(float64(row[FeatureY]-ref[FeatureY])) > tolerance {
					continue
				}
				for col := range BodyFeatures {
					if math.Abs(float64(row[col]-ref[col])) > tolerance {
						t.Fatalf("observer %d body %d column %d = %g, observer 0 sees %g",
							observer, obs.SlotBodyID[slot], col, row[col], ref[col])
					}
				}
				found = true
				break
			}
			if !found {
				t.Fatalf("observer %d sees body %d at (%g, %g), where observer 0 sees nothing",
					observer, obs.SlotBodyID[slot], row[FeatureX], row[FeatureY])
			}
			matched++
		}
		if want := len(g.Planets) + len(g.Comets); matched != want {
			t.Fatalf("observer %d matched %d bodies, want %d", observer, matched, want)
		}
		for role := range roleCount {
			for col := range PlayerFeatures {
				if math.Abs(float64(obs.Players[role][col]-reference.Players[role][col])) > tolerance {
					t.Fatalf("observer %d player row %d column %d = %g, observer 0 sees %g",
						observer, role, col, obs.Players[role][col], reference.Players[role][col])
				}
			}
		}
	}
}

// TestBuildEncodesEachOwnerSeparately checks body and player values against
// numbers worked out by hand, on a board where every role does something
// different.
//
// The cross-seat test cannot catch a mistake every seat shares: routing all
// fleets into IncomingMe, say, would still look the same from every seat. So
// here three fleets of different sizes and owners head for the observer's
// home at known distances, and the observer owns a comet, and each column is
// compared with what it must be.
func TestBuildEncodesEachOwnerSeparately(t *testing.T) {
	g := generatedV2(t, 3, true)
	frame, err := NewSeatFrame(g, 0)
	if err != nil {
		t.Fatal(err)
	}
	home := homeOf(t, g, 0)

	// Freeze the board, so a fleet's contact time is simple geometry: a
	// fleet gap units outside a still planet's edge, flying at its centre,
	// touches it after gap/speed turns.
	g.Config.OrbitsEnabled = false
	for i := range g.Planets {
		g.Planets[i].Orbit = nil
	}

	// Each fleet starts on a different bearing from the home, between the
	// home and the sun so it is on the board and away from the edge. The gap
	// is a fraction of the fleet's own speed, so all three land in the first
	// turn at a known fraction of it.
	toSun := math.Atan2(g.Config.Sun.Y-home.Position.Y, g.Config.Sun.X-home.Position.X)
	fleets := []struct {
		owner, ships int
		bearing      float64 // radians from the home-to-sun direction
		eta          float64 // turns until contact, by construction
	}{
		{owner: 0, ships: 7, bearing: 0, eta: 0.75},     // me
		{owner: 1, ships: 12, bearing: -0.7, eta: 0.25}, // clockwise
		{owner: 3, ships: 20, bearing: 0.7, eta: 0.5},   // opposite
	}
	for i, f := range fleets {
		speed := g.FleetSpeed(f.ships)
		out := game.Vec2{X: math.Cos(toSun + f.bearing), Y: math.Sin(toSun + f.bearing)}
		g.Fleets = append(g.Fleets, game.Fleet{
			ID:         10_001 + i,
			OwnerID:    f.owner,
			Ships:      f.ships,
			Position:   home.Position.Add(out.Scale(home.Radius + f.eta*speed)),
			Velocity:   out.Scale(-speed),
			LaunchTurn: -1, // not launched this turn, so no source is ignored
		})
	}

	// Give the observer a comet well away from everything, so it sits in
	// slot 40 with fewer than 40 planets and padding in between.
	cometAt := func(p game.Vec2) bool {
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
	var start game.Vec2
	found := false
	for x := 10.0; x <= 86 && !found; x++ {
		for y := 10.0; y <= 90 && !found; y++ {
			start = game.Vec2{X: x, Y: y}
			found = cometAt(start) && cometAt(start.Add(step))
		}
	}
	if !found {
		t.Fatal("no clear spot for the comet")
	}
	g.Comets = append(g.Comets, game.Comet{
		ID: 10_000, Position: start, Radius: 1, OwnerID: 0, Ships: 9, Production: 1,
		Path: []game.Vec2{start, start.Add(step)},
	})

	// Guard the fixture: each fleet must actually land on the home, or the
	// incoming checks below would be measuring a different board.
	for _, l := range g.PredictFleets(maxPredictionSegments(g)) {
		if !l.Arrives() || l.Hit.BodyID != home.ID {
			t.Fatalf("fleet %d does not land on the home: %+v", l.FleetID, l)
		}
	}

	obs, err := Build(g, frame)
	if err != nil {
		t.Fatal(err)
	}
	near := func(what string, got float32, want float64) {
		t.Helper()
		if math.Abs(float64(got)-want) > 1e-6 {
			t.Fatalf("%s = %g, want %g", what, got, want)
		}
	}

	// Incoming at the home: each owner's ships in its own role column, and
	// counter-clockwise (player 2) sent nothing.
	var homeSlot int
	for slot := range planetSlots {
		if obs.SlotBodyID[slot] == int32(home.ID) {
			homeSlot = slot
		}
	}
	row := obs.Bodies[homeSlot]
	horizon := float64(maxPredictionSegments(g)) + 1
	ship := func(n int) float64 { return math.Log1p(float64(n)) / math.Log1p(shipReference) }
	near("incoming me", row[FeatureIncomingMe], ship(7))
	near("incoming clockwise", row[FeatureIncomingCW], ship(12))
	near("incoming opposite", row[FeatureIncomingOpposite], ship(20))
	near("incoming counter-clockwise", row[FeatureIncomingCCW], 0)
	near("arrival me", row[FeatureArrivalMe], 0.75/horizon)
	near("arrival clockwise", row[FeatureArrivalCW], 0.25/horizon)
	near("arrival opposite", row[FeatureArrivalOpposite], 0.5/horizon)
	near("arrival counter-clockwise", row[FeatureArrivalCCW], noArrival)

	// The owned comet: slot 40, mine, two points left, stepping CometSpeed
	// to the right. Observer 0's frame does not turn the board.
	const cometSlot = firstCometSlot
	if obs.SlotBodyID[cometSlot] != 10_000 || !obs.OwnedMask[cometSlot] || obs.BodyMask[len(g.Planets)] {
		t.Fatalf("comet slot %d holds body %d (owned %v), and slot %d real = %v",
			cometSlot, obs.SlotBodyID[cometSlot], obs.OwnedMask[cometSlot], len(g.Planets), obs.BodyMask[len(g.Planets)])
	}
	comet := obs.Bodies[cometSlot]
	near("comet owner me", comet[FeatureOwnerMe], 1)
	near("comet flag", comet[FeatureIsComet], 1)
	near("comet turns left", comet[FeatureCometTurnsLeft], 2.0/game.CometMaxPathPoints)
	near("comet dx", comet[FeatureCometDX], 1)
	near("comet dy", comet[FeatureCometDY], 0)

	// Player totals per role: garrisons, the comet, and ships in flight.
	homeShips := func(player int) int { return homeOf(t, g, player).Ships }
	production := func(player int) int { return homeOf(t, g, player).Production }
	players := []struct {
		role              Role
		ships, production int
	}{
		{RoleMe, homeShips(0) + 9 + 7, production(0) + 1},
		{RoleCW, homeShips(1) + 12, production(1)},
		{RoleOpposite, homeShips(3) + 20, production(3)},
		{RoleCCW, homeShips(2), production(2)},
	}
	for _, p := range players {
		got := obs.Players[p.role]
		near("player ships", got[PlayerFeatureShips], ship(p.ships))
		near("player production", got[PlayerFeatureProduction], float64(p.production)/playerProductionScale)
	}
}

// TestBuildRejectsUnusableInput checks that Build refuses inputs it
// cannot describe truthfully, and returns a zero observation when it does,
// never a partly filled one a caller might use by mistake.
func TestBuildRejectsUnusableInput(t *testing.T) {
	tests := []struct {
		name string
		// setup returns the game and frame to build from.
		setup func(t *testing.T) (*game.Game, SeatFrame)
	}{
		{"nil game", func(t *testing.T) (*game.Game, SeatFrame) {
			g := generatedV2(t, 1, false)
			frame, _ := NewSeatFrame(g, 0)
			return nil, frame
		}},
		{"unbuilt frame", func(t *testing.T) (*game.Game, SeatFrame) {
			return generatedV2(t, 1, false), SeatFrame{}
		}},
		{"frame from another game", func(t *testing.T) (*game.Game, SeatFrame) {
			frame, _ := NewSeatFrame(generatedV2(t, 2, false), 0)
			return generatedV2(t, 1, false), frame
		}},
		{"frame from the original of a copied game", func(t *testing.T) (*game.Game, SeatFrame) {
			g := generatedV2(t, 1, false)
			frame, _ := NewSeatFrame(g, 0)
			copied := *g
			return &copied, frame
		}},
		{"neutral fleet", func(t *testing.T) (*game.Game, SeatFrame) {
			g := generatedV2(t, 1, false)
			frame, _ := NewSeatFrame(g, 0)
			g.Fleets = append(g.Fleets, game.Fleet{
				ID: 10_000, OwnerID: game.NeutralPlayer, Ships: 5,
				Position: game.Vec2{X: 20, Y: 20}, Velocity: game.Vec2{X: 1},
			})
			return g, frame
		}},
		{"square resize leaving the sun off-centre", func(t *testing.T) (*game.Game, SeatFrame) {
			g := generatedV2(t, 1, false)
			frame, _ := NewSeatFrame(g, 0)
			g.Config.Width, g.Config.Height = 200, 200
			return g, frame
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, frame := tt.setup(t)
			obs, err := Build(g, frame)
			if err == nil {
				t.Fatal("Build accepted the input")
			}
			if obs != (Observation{}) {
				t.Fatal("Build returned a non-zero observation with its error")
			}
		})
	}
}

// TestBuildMasks checks the masks that decide which actions the policy can
// pick, on a two-player board with one extra friendly planet that has no ships.
func TestBuildMasks(t *testing.T) {
	g := generatedV2(t, 1, false)
	frame, err := NewSeatFrame(g, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Find the homes before player 0 gains a second planet, after which
	// homeOf could return either.
	home := homeOf(t, g, 0)
	enemy := homeOf(t, g, 1)
	// The frame is fixed at turn 0, so the board can be changed afterwards:
	// give player 0 a second, empty planet to act as a friendly target.
	var empty *game.Planet
	for i := range g.Planets {
		if g.Planets[i].OwnerID == game.NeutralPlayer {
			empty = &g.Planets[i]
			break
		}
	}
	empty.OwnerID, empty.Ships = 0, 0

	obs, err := Build(g, frame)
	if err != nil {
		t.Fatal(err)
	}
	slotOf := map[int]int{}
	for slot := range BodySlots {
		if obs.BodyMask[slot] {
			slotOf[int(obs.SlotBodyID[slot])] = slot
		}
	}
	homeSlot, emptySlot, enemySlot := slotOf[home.ID], slotOf[empty.ID], slotOf[enemy.ID]

	// A stocked home may target every other real body, friendly ones
	// included, and never itself or padding.
	for target := range BodySlots {
		want := obs.BodyMask[target] && target != homeSlot
		if obs.TargetMask[homeSlot][target] != want {
			t.Fatalf("home -> slot %d = %v, want %v", target, obs.TargetMask[homeSlot][target], want)
		}
	}
	if !obs.TargetMask[homeSlot][emptySlot] {
		t.Fatal("home cannot reinforce its own planet")
	}
	// An owned planet with no ships is still mine, but can launch nothing. An
	// enemy planet is never a source.
	if !obs.OwnedMask[emptySlot] {
		t.Fatal("empty friendly planet is not marked owned")
	}
	for target := range BodySlots {
		if obs.TargetMask[emptySlot][target] || obs.TargetMask[enemySlot][target] {
			t.Fatalf("empty or enemy planet may launch to slot %d", target)
		}
	}
	want := [PlayerSlots]bool{true, false, true, false}
	if obs.PlayerMask != want {
		t.Fatalf("player mask = %v, want %v", obs.PlayerMask, want)
	}

	// Eliminate the opponent. It was still in the game, so its row and mask
	// stay; only its totals fall to zero.
	enemy.OwnerID = game.NeutralPlayer
	obs, err = Build(g, frame)
	if err != nil {
		t.Fatal(err)
	}
	if obs.PlayerMask != want {
		t.Fatalf("player mask after elimination = %v, want %v", obs.PlayerMask, want)
	}
	opposite := obs.Players[RoleOpposite]
	if opposite[PlayerFeatureRoleOpposite] != 1 || opposite[PlayerFeatureShips] != 0 {
		t.Fatalf("eliminated opponent's row = %v, want the opposite one-hot and no ships", opposite)
	}
}
