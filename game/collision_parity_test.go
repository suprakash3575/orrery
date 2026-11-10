package game

import (
	"math"
	"math/rand"
	"testing"
)

// The tests in this file check that bodies prepared once per turn
// (prepareTurnMotion) give exactly the collisions and landings of the search
// they replaced. The replaced search is kept here as an oracle. Its motion is
// built independently: from body positions read before and after each move,
// never from paths or the prepared list. A preparation bug therefore cannot
// shape both sides of a comparison and still pass.

// referenceFirstCollision is the collision search as it was before bodies were
// prepared: the sun and the board edge first, then every body, with the
// moving bodies' starting positions and the skipped bodies looked up by ID in
// maps for every fleet. It shares only the exact solver (segmentCircleEntry,
// boundaryExit) and the ranking (precedes) with the code under test.
func referenceFirstCollision(g *Game, start, end Vec2, excludeBody int, moved map[int]Vec2, ignored map[int]bool) (collision, bool) {
	var best collision
	found := false
	consider := func(t float64, ok bool, kind hazardKind, id int) {
		candidate := collision{t: t, kind: kind, id: id}
		if !ok || (found && !candidate.precedes(best)) {
			return
		}
		best, found = candidate, true
	}
	entry := func(id int, position Vec2, radius float64) (float64, bool) {
		from, ok := moved[id]
		if !ok {
			return segmentCircleEntry(start, end, position, radius)
		}
		return segmentCircleEntry(start.Sub(from), end.Sub(position), Vec2{}, radius)
	}

	t, ok := g.boundaryExit(start, end)
	consider(t, ok, hazardBoundary, 0)
	if g.Config.SunHazardEnabled {
		t, ok = segmentCircleEntry(start, end, g.Config.Sun, g.Config.SunRadius)
		consider(t, ok, hazardSun, 0)
	}
	for _, p := range g.Planets {
		if p.ID == excludeBody || ignored[p.ID] {
			continue
		}
		t, ok = entry(p.ID, p.Position, p.Radius)
		consider(t, ok, hazardPlanet, p.ID)
	}
	for _, c := range g.Comets {
		if c.ID == excludeBody || ignored[c.ID] {
			continue
		}
		t, ok = entry(c.ID, c.Position, c.Radius)
		consider(t, ok, hazardComet, c.ID)
	}
	return best, found
}

// bodyPositions records where every planet and comet is now, by ID. Planets
// and comets share one ID space, so one map holds both.
func bodyPositions(g *Game) map[int]Vec2 {
	positions := make(map[int]Vec2, len(g.Planets)+len(g.Comets))
	for _, p := range g.Planets {
		positions[p.ID] = p.Position
	}
	for _, c := range g.Comets {
		positions[c.ID] = c.Position
	}
	return positions
}

// movedFrom compares positions read before and after a move. Every body whose
// position changed maps to where it started; a body that did not move is
// absent, as it was from the replaced search's map.
func movedFrom(before, after map[int]Vec2) map[int]Vec2 {
	moved := make(map[int]Vec2)
	for id, now := range after {
		if was, ok := before[id]; ok && was != now {
			moved[id] = was
		}
	}
	return moved
}

// referencePredictLandings is predictLandings as it was before bodies were
// prepared, run on referenceFirstCollision with independently captured motion.
// Each predicted turn follows the same steps in the same order: safeguard
// expiry, move the bodies, test every fleet, expire the departing comets.
func referencePredictLandings(g *Game, fleets []Fleet, horizon int) []Landing {
	landings := make([]Landing, len(fleets))
	type flying struct {
		index int
		fleet Fleet
	}
	pending := make([]flying, len(fleets))
	for i, f := range fleets {
		landings[i].FleetID = f.ID
		pending[i] = flying{index: i, fleet: f}
	}
	sim := g.predictionCopy()
	for segment := range horizon {
		if len(pending) == 0 {
			break
		}
		sim.expireComets()
		before := bodyPositions(sim)
		sim.moveOrbits(nil)
		sim.moveComets(nil)
		moved := movedFrom(before, bodyPositions(sim))

		still := pending[:0]
		for _, p := range pending {
			start := p.fleet.Position
			end := start.Add(p.fleet.Velocity)
			exclude := 0
			if p.fleet.LaunchTurn == sim.Turn {
				exclude = p.fleet.FromPlanetID
			}
			found, ok := referenceFirstCollision(sim, start, end, exclude, moved, nil)
			if !ok {
				p.fleet.Position = end
				still = append(still, p)
				continue
			}
			landings[p.index] = Landing{FleetID: p.fleet.ID, Resolved: true, Hit: publicHit(found), Segment: segment}
		}
		pending = still

		if sim.expireComets() {
			for i := range landings {
				l := &landings[i]
				if l.Resolved && l.Segment == segment && l.Hit.Kind == HitComet && sim.comet(l.Hit.BodyID) == nil {
					l.Departed = true
				}
			}
		}
		sim.Turn++
	}
	return landings
}

// sameCollision compares two collision results exactly, contact time included
// to the bit, so a change in arithmetic shows up even when the winner agrees.
func sameCollision(a collision, aOK bool, b collision, bOK bool) bool {
	if aOK != bOK {
		return false
	}
	return !aOK || (a.kind == b.kind && a.id == b.id && math.Float64bits(a.t) == math.Float64bits(b.t))
}

// parityGame builds a seeded board with orbits, the sun hazard and comet waves
// all on: eight to twelve non-overlapping planets, the ones near the sun
// orbiting, and players 0 to players-1 owning one large garrison each.
//
// Two or three comets start on straight paths that cross the board, so comets
// move and depart from the first turns. Waves may add more at turns 50 and
// 150, but their path search can fail on a crowded random board, so the
// test does not rely on them.
func parityGame(t *testing.T, seed int64, players int) *Game {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	config := DefaultConfig()
	config.Rand = rand.New(rand.NewSource(seed))
	var planets []Planet
	for len(planets) < 8+rng.Intn(5) {
		production := 1 + rng.Intn(5)
		radius := PlanetRadius(production)
		position := Vec2{5 + 90*rng.Float64(), 5 + 90*rng.Float64()}
		if position.Distance(config.Sun) < config.SunRadius+radius+2 {
			continue
		}
		clear := true
		for _, other := range planets {
			if position.Distance(other.Position) < radius+other.Radius+2 {
				clear = false
			}
		}
		if !clear {
			continue
		}
		planets = append(planets, Planet{
			ID: len(planets) + 1, Position: position, Radius: radius, Production: production,
			OwnerID: NeutralPlayer, Ships: rng.Intn(21),
		})
	}
	for player := range players {
		planets[player].OwnerID, planets[player].Ships = player, 60
	}
	var comets []Comet
	for i := range 2 + rng.Intn(2) {
		through := Vec2{20 + 60*rng.Float64(), 20 + 60*rng.Float64()}
		angle := 2 * math.Pi * rng.Float64()
		step := Vec2{math.Cos(angle), math.Sin(angle)}.Scale(config.CometSpeed)
		var path []Vec2
		for k := -40; k <= 40; k++ {
			point := through.Add(step.Scale(float64(k)))
			if point.X >= 0 && point.X <= config.Width && point.Y >= 0 && point.Y <= config.Height {
				path = append(path, point)
			}
		}
		comets = append(comets, Comet{
			ID: 100 + i, OwnerID: NeutralPlayer, Ships: rng.Intn(21), Path: path,
		})
	}
	g, err := New(config, planets, comets)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// parityLaunches launches from about a third of the planets each player owns,
// part of the garrison in a random direction, so fleets of many sizes and
// headings are in flight.
func parityLaunches(rng *rand.Rand, g *Game) []Action {
	var actions []Action
	for _, p := range g.Planets {
		if p.OwnerID == NeutralPlayer || p.Ships < 2 || rng.Float64() >= 0.35 {
			continue
		}
		actions = append(actions, Action{
			PlayerID: p.OwnerID, PlanetID: p.ID,
			Ships: 1 + rng.Intn(p.Ships-1), Angle: 2 * math.Pi * rng.Float64(),
		})
	}
	return actions
}

// TestPreparedCollisionsMatchReference plays seeded two- and four-player games
// for 200 turns, through the comet waves at turns 50 and 150, with launches
// every turn. Every turn it moves a copy of the bodies once and checks, for
// every fleet's segment and for extra random segments, that the prepared
// search equals the reference. A random third of the comets is marked as
// spawned this turn, so skipped bodies are compared too. Every tenth turn it
// also checks complete PredictFleets landings against the reference
// predictor. Real turns then run on the code under test, so the states
// themselves come from the prepared path.
func TestPreparedCollisionsMatchReference(t *testing.T) {
	cases := []struct {
		seed    int64
		players int
	}{{1, 2}, {2, 2}, {3, 4}, {4, 4}}
	horizon := int(math.Ceil(math.Hypot(DefaultConfig().Width, DefaultConfig().Height)))
	for _, c := range cases {
		g := parityGame(t, c.seed, c.players)
		// Separate streams: launches decide the game, so they must not shift
		// with how many segments or spawned comets the checks happen to draw.
		launches := rand.New(rand.NewSource(c.seed + 1_000))
		rng := rand.New(rand.NewSource(c.seed + 2_000))
		compared, movingBodies, cometsSeen := 0, 0, 0
		for range 200 {
			if g.Finished() {
				break
			}

			// One turn of body movement on a copy, with the reference's motion
			// captured from positions around that same move.
			sim := g.predictionCopy()
			before := bodyPositions(sim)
			paths := sim.moveOrbits(nil)
			paths = sim.moveComets(paths)
			moved := movedFrom(before, bodyPositions(sim))
			var spawned []int
			ignored := map[int]bool{}
			for _, cm := range sim.Comets {
				if rng.Intn(3) == 0 {
					spawned = append(spawned, cm.ID)
					ignored[cm.ID] = true
				}
			}
			motion := sim.prepareTurnMotion(paths, spawned, nil)
			movingBodies += len(moved)
			cometsSeen += len(sim.Comets)

			check := func(start, end Vec2, exclude int) {
				got, gotOK := sim.firstCollision(start, end, exclude, motion)
				want, wantOK := referenceFirstCollision(sim, start, end, exclude, moved, ignored)
				if !sameCollision(got, gotOK, want, wantOK) {
					t.Fatalf("seed %d turn %d: segment %v→%v exclude %d: prepared = %+v (ok %v), reference = %+v (ok %v)",
						c.seed, g.Turn, start, end, exclude, got, gotOK, want, wantOK)
				}
				compared++
			}
			for _, f := range g.Fleets {
				exclude := 0
				if f.LaunchTurn == g.Turn {
					exclude = f.FromPlanetID
				}
				check(f.Position, f.Position.Add(f.Velocity), exclude)
			}
			for range 20 {
				start := Vec2{rng.Float64() * 100, rng.Float64() * 100}
				step := Vec2{rng.Float64()*12 - 6, rng.Float64()*12 - 6}
				exclude := 0
				if rng.Intn(4) == 0 {
					exclude = 1 + rng.Intn(len(g.Planets))
				}
				check(start, start.Add(step), exclude)
			}

			if g.Turn%10 == 0 {
				got := g.PredictFleets(horizon)
				want := referencePredictLandings(g, g.Fleets, horizon)
				for i := range want {
					if got[i] != want[i] || math.Float64bits(got[i].Hit.T) != math.Float64bits(want[i].Hit.T) {
						t.Fatalf("seed %d turn %d: landing %d = %+v, reference = %+v", c.seed, g.Turn, i, got[i], want[i])
					}
				}
			}

			if err := g.AdvanceTurn(parityLaunches(launches, g)); err != nil {
				t.Fatalf("seed %d turn %d: %v", c.seed, g.Turn, err)
			}
		}
		// Guard the guard: the sweep must really have run through the comet
		// waves and compared moving bodies, comets and many segments, not an
		// empty or static board.
		if g.Turn < 160 || compared < 5_000 || movingBodies == 0 || cometsSeen == 0 {
			t.Fatalf("seed %d: reached turn %d, compared %d segments with %d moving bodies and %d comets seen; the sweep is too thin",
				c.seed, g.Turn, compared, movingBodies, cometsSeen)
		}
	}
}

// TestPreparedButEmptyIsNotTheStaticLookup pins the difference between the
// zero-value motion (every body stands still and takes part) and a prepared
// motion in which no body takes part. A comet that spawned this turn lies
// across the fleet's path on a board with no planets. The static lookup finds
// it; the prepared motion leaves it out, and its now-empty list must not fall
// back to the static lookup, so the fleet only leaves the board.
func TestPreparedButEmptyIsNotTheStaticLookup(t *testing.T) {
	g, err := New(DefaultConfig(), nil, []Comet{{ID: 5, OwnerID: NeutralPlayer, Path: fixedCometPath(Vec2{95, 10})}})
	if err != nil {
		t.Fatal(err)
	}
	start, end := Vec2{90, 10}, Vec2{105, 10}

	static, ok := g.firstCollision(start, end, 0, turnMotion{})
	if !ok || static.kind != hazardComet || static.id != 5 {
		t.Fatalf("static lookup = %+v (ok %v), want comet 5", static, ok)
	}
	motion := g.prepareTurnMotion(nil, []int{5}, nil)
	if !motion.prepared || len(motion.bodies) != 0 {
		t.Fatalf("prepared motion = %+v, want prepared with no bodies", motion)
	}
	got, ok := g.firstCollision(start, end, 0, motion)
	if !ok || got.kind != hazardBoundary {
		t.Fatalf("prepared but empty = %+v (ok %v), want the board edge", got, ok)
	}
}

// TestPredictionKeepsBodyStorageAcrossStillTurns guards the prediction's
// scratch storage. A turn on which nothing moves uses the static lookup, which
// holds no storage, and the prepared-body buffer must survive it: otherwise a
// comet that pauses on its path (a point repeated, A → B → B → C) would cost a
// fresh allocation every time it moved again.
//
// Two boards differ only in their one comet's path: one moves every turn, the
// other pauses every other turn. Nothing else moves, so on the second board
// every other predicted turn is still. Pauses must not add allocations, so the
// pausing board may allocate no more than the moving one.
func TestPredictionKeepsBodyStorageAcrossStillTurns(t *testing.T) {
	const horizon = 40
	config := DefaultConfig()
	config.OrbitsEnabled, config.CometSpawningEnabled, config.SunHazardEnabled = false, false, false
	board := func(path []Vec2) *Game {
		g, err := New(config, []Planet{
			{ID: 1, OwnerID: 0, Ships: 5, Production: 1, Position: Vec2{10, 50}},
			{ID: 2, OwnerID: 1, Ships: 5, Production: 1, Position: Vec2{90, 50}},
		}, []Comet{{ID: 3, OwnerID: NeutralPlayer, Path: path}})
		if err != nil {
			t.Fatal(err)
		}
		// One slow fleet that touches nothing within the horizon, so it is
		// followed through every predicted turn.
		g.Fleets = []Fleet{{ID: 10, OwnerID: 0, Ships: 1, Position: Vec2{10, 10}, Velocity: Vec2{0.5, 0}, LaunchTurn: -1}}
		return g
	}
	// Both paths outlast the horizon, so the comet never departs.
	var moving, pausing []Vec2
	for i := range 2 * horizon {
		moving = append(moving, Vec2{5 + 0.5*float64(i), 90})
	}
	for i := range horizon {
		point := Vec2{5 + float64(i), 90}
		pausing = append(pausing, point, point)
	}

	allocs := func(g *Game) float64 {
		if landing := g.PredictFleets(horizon)[0]; landing.Resolved {
			t.Fatalf("test setup: fleet resolved at %+v, want it followed through the horizon", landing)
		}
		return testing.AllocsPerRun(50, func() { g.PredictFleets(horizon) })
	}
	movingAllocs, pausingAllocs := allocs(board(moving)), allocs(board(pausing))
	if pausingAllocs > movingAllocs {
		t.Fatalf("a pausing comet costs %v allocations per prediction against %v for one moving every turn; the body storage is not kept across still turns",
			pausingAllocs, movingAllocs)
	}
}
