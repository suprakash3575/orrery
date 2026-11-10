package game

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// fixedCometPath is a test comet that sits at one point for longer than any
// match lasts. A comet always steps along its path, so a path of repeated
// points is how a test pins a stationary comet without it expiring.
func fixedCometPath(point Vec2) []Vec2 {
	path := make([]Vec2, 600)
	for i := range path {
		path[i] = point
	}
	return path
}

// TestSpawnedCometsAreReproducible pins the replay contract for spawning: the
// same seed must produce the same batch, same as every other seeded roll in
// the engine.
func TestSpawnedCometsAreReproducible(t *testing.T) {
	newAtTurn50 := func(seed int64) *Game {
		config := DefaultConfig()
		config.Rand = rand.New(rand.NewSource(seed))
		g, err := New(config, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		g.Turn = 50
		g.spawnComets()
		return g
	}
	a, b := newAtTurn50(99), newAtTurn50(99)
	if len(a.Comets) != 4 || len(b.Comets) != 4 {
		t.Fatalf("got %d and %d comets, want 4 and 4", len(a.Comets), len(b.Comets))
	}
	if !reflect.DeepEqual(a.Comets, b.Comets) {
		t.Fatalf("comets diverged between runs:\na=%+v\nb=%+v", a.Comets, b.Comets)
	}
}

// TestSpawnedCometsAreFourFoldSymmetric checks actual positions: each comet's
// path, measured from the sun, must be a genuine 90-degree rotation of comet
// 0's, point for point. The reference engine's four copies are ordered so that comets 1, 3
// and 2 are comet 0 rotated by one, two and three quarter-turns.
func TestSpawnedCometsAreFourFoldSymmetric(t *testing.T) {
	config := DefaultConfig()
	config.Rand = rand.New(rand.NewSource(7))
	g, err := New(config, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	g.Turn = 50
	g.spawnComets()

	if len(g.Comets) != 4 {
		t.Fatalf("got %d comets, want 4", len(g.Comets))
	}
	want := g.Comets[0]
	sun := g.Config.Sun
	quarterTurns := [4]int{0, 1, 3, 2}
	for k, c := range g.Comets {
		if c.Ships != want.Ships {
			t.Fatalf("comet %d ships = %d, want %d (shared across the batch)", k, c.Ships, want.Ships)
		}
		if len(c.Path) != len(want.Path) {
			t.Fatalf("comet %d path has %d points, comet 0 has %d", k, len(c.Path), len(want.Path))
		}
		for i, point := range c.Path {
			rotated := want.Path[i].Sub(sun)
			for range quarterTurns[k] {
				rotated = Vec2{-rotated.Y, rotated.X}
			}
			got := point.Sub(sun)
			if math.Abs(got.X-rotated.X) > 1e-9 || math.Abs(got.Y-rotated.Y) > 1e-9 {
				t.Fatalf("comet %d point %d at offset %+v from the sun, want %+v", k, i, got, rotated)
			}
		}
	}
}

// TestSpawnedCometIDsDoNotCollide covers the same ID-safety guarantee fleets
// already have, extended to spawned comets: they draw from the same
// monotonic g.nextID counter as everything else.
func TestSpawnedCometIDsDoNotCollide(t *testing.T) {
	g, err := New(DefaultConfig(),
		[]Planet{{ID: 1, Position: Vec2{20, 20}, OwnerID: 0, Production: 1}},
		[]Comet{{ID: 2, OwnerID: 0, Path: fixedCometPath(Vec2{60, 60})}},
	)
	if err != nil {
		t.Fatal(err)
	}
	existing := len(g.Comets)
	g.Turn = 50
	g.spawnComets()

	seen := map[int]bool{1: true, 2: true}
	for _, c := range g.Comets[existing:] {
		if seen[c.ID] {
			t.Fatalf("spawned comet ID %d collides with an existing planet or comet", c.ID)
		}
		seen[c.ID] = true
	}
}

// TestCometsSpawnOnlyOnScheduledTurns guards both edges of the schedule: no
// spawn on an ordinary turn, and a spawn on every listed one.
func TestCometsSpawnOnlyOnScheduledTurns(t *testing.T) {
	for _, turn := range []int{0, 1, 49, 51, 100, 449, 451, 500} {
		g, err := New(DefaultConfig(), nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		g.Turn = turn
		g.spawnComets()
		if len(g.Comets) != 0 {
			t.Fatalf("turn %d: spawned %d comets, want 0 (not a scheduled turn)", turn, len(g.Comets))
		}
	}
	for _, turn := range cometSpawnTurns {
		g, err := New(DefaultConfig(), nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		g.Turn = turn
		g.spawnComets()
		if len(g.Comets) != 4 {
			t.Fatalf("turn %d: spawned %d comets, want 4", turn, len(g.Comets))
		}
	}
}

// TestFailedSpawnTurnDoesNotDuplicateOnRetry covers the case where a
// scheduled spawn turn's launch actions fail validation. Since Turn is not
// incremented on a failed AdvanceTurn, a caller correcting its actions and
// retrying re-enters the same scheduled turn — spawnComets must not have
// already run (and must not run again) until that turn actually completes,
// or a retry would double the wave and diverge Config.Rand from a replay
// that succeeded on the first attempt.
func TestFailedSpawnTurnDoesNotDuplicateOnRetry(t *testing.T) {
	config := DefaultConfig()
	config.Rand = rand.New(rand.NewSource(1))
	// A third owner (bystander) keeps this a genuine two-... three-player
	// board throughout: without it, the single owner (player 1) would
	// already satisfy elimination from construction, and both AdvanceTurn
	// calls below would fail with "game is already finished" instead of the
	// specific errors — or lack of one — this test actually checks for.
	g, err := New(config, []Planet{
		{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 5, Production: 1},
		bystander(99, 3),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	g.Turn = 50

	// player 2 does not own planet 1: fails validation before any spawn.
	err = g.AdvanceTurn([]Action{{PlayerID: 2, PlanetID: 1, Ships: 1, Angle: 0}})
	if err == nil {
		t.Fatal("expected the invalid action to be rejected")
	}
	if g.Finished() {
		t.Fatalf("test setup: game should not be finished, got error %q instead of the expected ownership rejection", err)
	}
	if len(g.Comets) != 0 {
		t.Fatalf("a failed turn must not spawn anything, got %d comets", len(g.Comets))
	}
	if g.Turn != 50 {
		t.Fatalf("a failed turn must not advance Turn, got %d", g.Turn)
	}

	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	if len(g.Comets) != 4 {
		t.Fatalf("the retried (now valid) turn should spawn exactly one wave, got %d comets", len(g.Comets))
	}
}

// TestSpawnedCometsAlwaysSatisfyCometSpeed is the end-to-end check of the
// path contract: for a range of seeds, every spawned path stays on the board,
// lasts the reference engine's range of turns, and steps about CometSpeed per turn. Like
// the reference engine, a point is kept at the first dense sample at or past each multiple
// of CometSpeed, so a straight-line step can overshoot CometSpeed slightly;
// the bound allows a 10% margin either way of that.
func TestSpawnedCometsAlwaysSatisfyCometSpeed(t *testing.T) {
	for seed := int64(0); seed < 30; seed++ {
		config := DefaultConfig()
		config.Rand = rand.New(rand.NewSource(seed))
		g, err := New(config, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		g.Turn = 50
		g.spawnComets()
		if len(g.Comets) != 4 {
			t.Fatalf("seed %d: expected a spawn wave", seed)
		}
		path := g.Comets[0].Path
		if len(path) < cometMinPathPoints || len(path) > CometMaxPathPoints {
			t.Fatalf("seed %d: path has %d points, want %d to %d", seed, len(path), cometMinPathPoints, CometMaxPathPoints)
		}
		for i, point := range path {
			if !g.inBounds(point) {
				t.Fatalf("seed %d: path point %d %v is off the board", seed, i, point)
			}
			if i == 0 {
				continue
			}
			if dist := point.Distance(path[i-1]); dist < 0.9*config.CometSpeed || dist > 1.1*config.CometSpeed {
				t.Fatalf("seed %d: step %d covers %v, want close to CometSpeed %v", seed, i, dist, config.CometSpeed)
			}
		}
	}
}

// TestCometSpawningSwitch drives full turns through AdvanceTurn rather than
// calling spawnComets directly: the switch guards the call site, so only the
// real turn path proves a disabled game truly skips the wave (and its Rand
// draws) instead of merely hiding it.
func TestCometSpawningSwitch(t *testing.T) {
	if !DefaultConfig().CometSpawningEnabled {
		t.Fatal("standard game should enable comet spawning")
	}
	for _, test := range []struct {
		name       string
		enabled    bool
		wantComets int
	}{
		{name: "enabled spawns the turn-50 wave", enabled: true, wantComets: 4},
		{name: "disabled stays comet-free past turn 50", enabled: false, wantComets: 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			config := DefaultConfig()
			config.CometSpawningEnabled = test.enabled
			config.Rand = rand.New(rand.NewSource(1))
			// Two owners so elimination cannot end the game before turn 50.
			g, err := New(config, []Planet{
				{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 5, Production: 1},
				bystander(99, 2),
			}, nil)
			if err != nil {
				t.Fatal(err)
			}
			for g.Turn <= 50 {
				if err := g.AdvanceTurn(nil); err != nil {
					t.Fatal(err)
				}
			}
			if got := len(g.Comets); got != test.wantComets {
				t.Fatalf("comets after turn 50 = %d, want %d", got, test.wantComets)
			}
		})
	}
}

// TestSpawnedCometIgnoresCollisionOnItsSpawnTurn pins the reference engine's first-placement
// rule: a comet takes no part in collision on the turn it spawns, and does from
// the next turn on. A fleet parked exactly where the comet appears survives
// the spawn turn, then is claimed once the comet moves off that point.
func TestSpawnedCometIgnoresCollisionOnItsSpawnTurn(t *testing.T) {
	newGame := func() *Game {
		config := DefaultConfig()
		config.Rand = rand.New(rand.NewSource(3))
		g, err := New(config, []Planet{
			{ID: 1, Position: Vec2{5, 5}, OwnerID: 0, Ships: 1, Production: 1},
			{ID: 2, Position: Vec2{95, 95}, OwnerID: 1, Ships: 1, Production: 1},
		}, nil)
		if err != nil {
			t.Fatal(err)
		}
		g.Turn = 50
		return g
	}
	// An identical game spawns the identical wave, which shows where the first
	// comet will appear before the real game reaches its spawn turn.
	preview := newGame()
	if preview.spawnComets() == nil {
		t.Fatal("test setup: this seed spawned no wave")
	}
	landing := preview.Comets[0].Path[0]

	g := newGame()
	g.Fleets = []Fleet{{ID: 90, OwnerID: 0, Ships: 3, Position: landing}}
	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	if len(g.Comets) == 0 || g.Comets[0].Position != landing {
		t.Fatalf("test setup: comet 0 should end its spawn turn at %v, got %+v", landing, g.Comets)
	}
	if len(g.Fleets) != 1 {
		t.Fatal("a comet claimed a fleet on its spawn turn; it must take no part in collision until the next turn")
	}

	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	if len(g.Fleets) != 0 {
		t.Fatal("the turn after spawning, the comet moving off the fleet's point should claim it")
	}
}

// TestDepartingCometTakesItsArrivalsWithIt pins the departing turn end to
// end: a fleet that reaches a comet on the turn the comet steps past its last
// point is consumed with it, the comet leaves before combat, and neither the
// comet's garrison nor the fleet counts in the standings afterwards.
func TestDepartingCometTakesItsArrivalsWithIt(t *testing.T) {
	g, err := New(DefaultConfig(), []Planet{
		{ID: 1, Position: Vec2{5, 5}, OwnerID: 1, Ships: 1, Production: 1},
		{ID: 2, Position: Vec2{95, 95}, OwnerID: 2, Ships: 1, Production: 1},
	}, []Comet{{ID: 3, OwnerID: 1, Ships: 4, Path: []Vec2{{80, 20}}}})
	if err != nil {
		t.Fatal(err)
	}
	// A one-point path makes the comet's first move its departing one, and the
	// fleet is parked on that point, so it runs into the comet this turn.
	g.Fleets = []Fleet{{ID: 90, OwnerID: 2, Ships: 7, Position: Vec2{80, 20}}}

	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	if len(g.Comets) != 0 {
		t.Fatalf("the departing comet should be gone by the end of the turn, got %+v", g.Comets)
	}
	if len(g.Fleets) != 0 {
		t.Fatalf("the fleet that reached the departing comet should be consumed, got %+v", g.Fleets)
	}
	// Each planet holds 1 ship plus 1 produced this turn; nothing else counts.
	standings := g.Standings()
	if standings[1] != 2 || standings[2] != 2 {
		t.Fatalf("standings = %v, want only planet garrisons (2 each)", standings)
	}
}

// TestCometPathTouchesPlanet checks both clearance rules directly. A static
// planet needs cometStaticClearance beyond its radius from every point. An
// orbiting planet is checked at the position it will hold when the comet
// reaches each point, not where it is now: the same point is a collision at
// its future position and clear at its current one.
func TestCometPathTouchesPlanet(t *testing.T) {
	g, err := New(DefaultConfig(), []Planet{
		// About 56.6 from the sun, outside the orbit limit, so static.
		{ID: 1, Position: Vec2{90, 10}, OwnerID: NeutralPlayer, Production: 1},
		// 30 from the sun, so it orbits.
		{ID: 2, Position: Vec2{80, 50}, OwnerID: NeutralPlayer, Production: 1},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	g.Turn = 50
	// The same points in all four copies keep the cases about one planet.
	wave := func(points ...Vec2) [][]Vec2 {
		return [][]Vec2{points, points, points, points}
	}
	far := Vec2{10, 90}
	orbit := g.Planets[1].Orbit
	futureAngle := orbit.StartAngle + orbit.RadiansTurn*float64(g.Turn+1+3)
	future := g.Config.Sun.Add(Vec2{orbit.Radius * math.Cos(futureAngle), orbit.Radius * math.Sin(futureAngle)})

	cases := []struct {
		name string
		wave [][]Vec2
		want bool
	}{
		// Planet 1 has radius 1, so its keep-out circle is 1 + 1.5 = 2.5.
		{"inside a static planet's clearance", wave(far, Vec2{92.4, 10}), true},
		{"just outside a static planet's clearance", wave(far, Vec2{92.6, 10}), false},
		{"on an orbiting planet's position when the comet gets there", wave(far, far, far, future), true},
		{"on an orbiting planet's current position, which it will have left", wave(far, far, far, g.Planets[1].Position), false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := g.cometPathTouchesPlanet(c.wave); got != c.want {
				t.Fatalf("cometPathTouchesPlanet = %v, want %v", got, c.want)
			}
		})
	}
}

// TestImpossibleCometSpeedSkipsTheWave checks the shortcut for a CometSpeed so
// large that even the biggest ellipse cannot hold a usable path: the wave is
// skipped before any try, so no random number is drawn for it.
func TestImpossibleCometSpeedSkipsTheWave(t *testing.T) {
	if cometPathCannotFit(cometMinSemiMajor*(cometArcEnd-cometArcStart), DefaultConfig().CometSpeed) {
		t.Fatal("the default CometSpeed must fit even the smallest ellipse")
	}
	config := DefaultConfig()
	config.CometSpeed = 300
	// Orbits off, so New() draws nothing and the generator is untouched.
	config.OrbitsEnabled = false
	config.Rand = rand.New(rand.NewSource(5))
	g, err := New(config, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	g.Turn = 50
	if ids := g.spawnComets(); ids != nil || len(g.Comets) != 0 {
		t.Fatalf("an impossible CometSpeed spawned comets %v", ids)
	}
	if got, want := g.Config.Rand.Float64(), rand.New(rand.NewSource(5)).Float64(); got != want {
		t.Fatalf("a skipped wave consumed randomness: next draw %v, want %v", got, want)
	}
}

// TestSpawnedCometsFitANonSquareBoard covers the rotated-copy check. On a
// board wider than it is tall, a copy rotated 90 degrees can fall off the
// board where the original is on it; such tries are rejected, so every
// spawned comet both starts and ends its path on the board.
func TestSpawnedCometsFitANonSquareBoard(t *testing.T) {
	found := 0
	for seed := int64(0); seed < 30; seed++ {
		config := DefaultConfig()
		config.Width, config.Height, config.Sun = 110, 90, Vec2{55, 45}
		config.Rand = rand.New(rand.NewSource(seed))
		g, err := New(config, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		g.Turn = 50
		if g.spawnComets() == nil {
			continue
		}
		found++
		for _, c := range g.Comets {
			for _, end := range [2]Vec2{c.Path[0], c.Path[len(c.Path)-1]} {
				if !g.inBounds(end) {
					t.Fatalf("seed %d: comet %d path end %v is off the %vx%v board", seed, c.ID, end, config.Width, config.Height)
				}
			}
		}
	}
	if found == 0 {
		t.Fatal("test setup: no seed spawned a wave on this board, so nothing was checked")
	}
}
