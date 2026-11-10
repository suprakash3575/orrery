package game

import (
	"math"
	"slices"
	"testing"
)

// TestSegmentCircleEntry pins the primitive the rest of continuous collision is
// built on. The first case is the reason this helper exists at all rather than
// reusing a closest-approach measurement: for a path that cuts through a
// circle, the closest approach is halfway *through* it, while the moment the
// path arrives is the entry point. Ordering hits by the wrong one of those lets
// a large distant body outrank a small one the fleet plainly reaches first.
func TestSegmentCircleEntry(t *testing.T) {
	cases := []struct {
		name         string
		a, b, center Vec2
		radius       float64
		wantHit      bool
		wantT        float64
	}{
		{"entry point, not closest approach", Vec2{0, 0}, Vec2{10, 0}, Vec2{5, 0}, 2, true, 0.3},
		{"starting inside touches immediately", Vec2{5, 0}, Vec2{10, 0}, Vec2{5, 0}, 2, true, 0},
		{"starting exactly on the edge touches immediately", Vec2{3, 0}, Vec2{10, 0}, Vec2{5, 0}, 2, true, 0},
		{"passing alongside misses", Vec2{0, 5}, Vec2{10, 5}, Vec2{5, 0}, 2, false, 0},
		{"circle behind the start", Vec2{10, 0}, Vec2{20, 0}, Vec2{0, 0}, 2, false, 0},
		{"circle past the end of the segment", Vec2{0, 0}, Vec2{5, 0}, Vec2{20, 0}, 2, false, 0},
		{"grazing the edge exactly counts", Vec2{0, 2}, Vec2{10, 2}, Vec2{5, 0}, 2, true, 0.5},
		{"stationary segment outside the circle", Vec2{0, 0}, Vec2{0, 0}, Vec2{5, 0}, 2, false, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, hit := segmentCircleEntry(c.a, c.b, c.center, c.radius)
			if hit != c.wantHit {
				t.Fatalf("hit = %v, want %v", hit, c.wantHit)
			}
			if hit && math.Abs(got-c.wantT) > 1e-9 {
				t.Fatalf("t = %v, want %v", got, c.wantT)
			}
		})
	}
}

// TestSegmentCircleEntryTangentsAtAnyAngle guards the roundoff edge of that
// primitive. The axis-aligned tangent in the table above passes trivially
// because nothing cancels; at a slant, B*B and 4*A*C are large and nearly
// equal, and an exact tangent's discriminant evaluates to roughly -2e-12
// instead of 0. Without the tolerance these all report a clean miss, so a
// fleet grazing a planet's edge would fly straight on.
//
// Each case is constructed to touch exactly: take a point one radius off the
// circle's centre along the path's perpendicular, and run the path through it.
func TestSegmentCircleEntryTangentsAtAnyAngle(t *testing.T) {
	const radius = 1.0
	center := Vec2{50, 50}
	for i := 0; i < 360; i++ {
		theta := float64(i) * math.Pi / 180
		direction := Vec2{math.Cos(theta), math.Sin(theta)}
		perpendicular := Vec2{-direction.Y, direction.X}
		touch := center.Add(perpendicular.Scale(radius))
		// Straddle the touch point so it falls in the middle of the segment.
		a, b := touch.Sub(direction.Scale(5)), touch.Add(direction.Scale(5))

		got, hit := segmentCircleEntry(a, b, center, radius)
		if !hit {
			t.Fatalf("tangent at %.0f degrees reported as a miss", float64(i))
		}
		// Touching at the midpoint of a 10-unit segment centred on the touch
		// point means t = 0.5. A loose tolerance would drift this.
		if math.Abs(got-0.5) > 1e-6 {
			t.Fatalf("tangent at %.0f degrees touched at t=%v, want 0.5", float64(i), got)
		}
	}
}

// TestSegmentCircleEntryStillRejectsClearMisses is the counterweight: the
// tolerance must not be so generous that it invents collisions. A path clearing
// a circle by a margin the board can actually represent stays a miss.
func TestSegmentCircleEntryStillRejectsClearMisses(t *testing.T) {
	const radius = 1.0
	center := Vec2{50, 50}
	for _, gap := range []float64{1e-6, 1e-3, 0.1, 5} {
		for i := 0; i < 360; i += 15 {
			theta := float64(i) * math.Pi / 180
			direction := Vec2{math.Cos(theta), math.Sin(theta)}
			perpendicular := Vec2{-direction.Y, direction.X}
			offset := center.Add(perpendicular.Scale(radius + gap))
			a, b := offset.Sub(direction.Scale(5)), offset.Add(direction.Scale(5))

			if _, hit := segmentCircleEntry(a, b, center, radius); hit {
				t.Fatalf("path clearing the circle by %v at %d degrees reported as a hit", gap, i)
			}
		}
	}
}

// TestSegmentCircleEntryTreatsNaNAsMiss feeds a NaN into each input in turn.
// Every one of them makes the solved entry time NaN, and NaN fails both
// "t < 0" and "t > 1", so a range check written that way reported a hit at
// time NaN — a time collision.precedes cannot order against any other.
func TestSegmentCircleEntryTreatsNaNAsMiss(t *testing.T) {
	nan := math.NaN()
	cases := []struct {
		name   string
		a, b   Vec2
		center Vec2
		radius float64
	}{
		{"NaN start", Vec2{nan, 50}, Vec2{60, 50}, Vec2{55, 50}, 1},
		{"NaN end", Vec2{40, 50}, Vec2{nan, 50}, Vec2{55, 50}, 1},
		{"NaN centre", Vec2{40, 50}, Vec2{60, 50}, Vec2{nan, 50}, 1},
		{"NaN radius", Vec2{40, 50}, Vec2{60, 50}, Vec2{55, 50}, nan},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got, hit := segmentCircleEntry(c.a, c.b, c.center, c.radius); hit {
				t.Fatalf("reported a hit at t=%v, want a miss", got)
			}
		})
	}
}

// TestFirstCollisionHitPriority covers the matrix PLAN.md step 4 implies, under
// the reference engine's ranking: every hazard is judged against the whole segment
// travelled; any touched body claims the fleet ahead of the sun or the board
// edge, and among several bodies the lowest ID wins. Fleets are placed directly
// rather than launched so each path is exact.
func TestFirstCollisionHitPriority(t *testing.T) {
	// Production 1 gives radius 1 + ln(1) = 1, so every planet below is a
	// 1-unit circle at its stated position unless noted otherwise.
	planet := func(id int, x, y float64) Planet {
		return Planet{ID: id, Position: Vec2{x, y}, OwnerID: NeutralPlayer, Production: 1}
	}
	// A comet on a fixed path sits still, which keeps these paths exact;
	// New() forces its radius to 1.
	comet := func(id int, x, y float64) Comet {
		return Comet{ID: id, OwnerID: NeutralPlayer, Path: fixedCometPath(Vec2{x, y})}
	}

	cases := []struct {
		name     string
		planets  []Planet
		comets   []Comet
		start    Vec2
		velocity Vec2
		// wantBody is the ID the fleet should arrive at, or 0 for no arrival.
		// wantInFlight distinguishes "destroyed" from "still travelling" when
		// there is no arrival.
		wantBody     int
		wantComet    bool
		wantInFlight bool
	}{{
		// The tunnelling gap this PR closes: the fleet ends up at (40,10),
		// nowhere near the planet, but its path went straight through it.
		name:     "tunnels through a planet",
		planets:  []Planet{planet(1, 30, 10)},
		start:    Vec2{20, 10},
		velocity: Vec2{20, 0},
		wantBody: 1,
	}, {
		name:      "tunnels through a comet",
		comets:    []Comet{comet(2, 30, 10)},
		start:     Vec2{20, 10},
		velocity:  Vec2{20, 0},
		wantBody:  2,
		wantComet: true,
	}, {
		// Planet entry at t=0.225 beats the sun's edge at t=0.5. Under the old
		// code the sun was checked first, in a different function entirely, so
		// this fleet died without ever being offered to the planet.
		name:     "planet before sun: planet claims the fleet",
		planets:  []Planet{planet(1, 30, 50)},
		start:    Vec2{20, 50},
		velocity: Vec2{40, 0},
		wantBody: 1,
	}, {
		// Reverse the geometry: the sun's edge at t=0.4 comes before the
		// planet's t=0.88. The reference engine checks bodies before the sun, so the planet
		// still claims the fleet.
		name:     "sun before planet: planet still claims the fleet",
		planets:  []Planet{planet(1, 65, 50)},
		start:    Vec2{20, 50},
		velocity: Vec2{50, 0},
		wantBody: 1,
	}, {
		// Planet entry at t=0.3 beats leaving the board at t=2/3.
		name:     "planet before board exit: planet claims the fleet",
		planets:  []Planet{planet(1, 90, 10)},
		start:    Vec2{80, 10},
		velocity: Vec2{30, 0},
		wantBody: 1,
	}, {
		name:     "leaving the board with nothing in the way destroys the fleet",
		start:    Vec2{95, 10},
		velocity: Vec2{10, 0},
		wantBody: 0,
	}, {
		// Both planets sit on the path. The nearer one is deliberately given
		// the *higher* ID, so passing this cannot be an accident of entry time
		// or slice ordering — only the lower-ID rule picks planet 1.
		name:     "two planets on one path: the lower ID claims the fleet",
		planets:  []Planet{planet(1, 40, 10), planet(2, 30, 10)},
		start:    Vec2{20, 10},
		velocity: Vec2{30, 0},
		wantBody: 1,
	}, {
		name:         "clear path leaves the fleet in flight",
		planets:      []Planet{planet(1, 80, 10)},
		start:        Vec2{20, 10},
		velocity:     Vec2{5, 0},
		wantBody:     0,
		wantInFlight: true,
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g, err := New(DefaultConfig(), c.planets, c.comets)
			if err != nil {
				t.Fatal(err)
			}
			g.Fleets = []Fleet{{ID: 99, OwnerID: 0, Ships: 5, Position: c.start, Velocity: c.velocity}}

			// moveFleets below uses a prepared motion. Nothing on these boards
			// moves, so the static lookup must name the same hazard and body.
			static, staticOK := g.firstCollision(c.start, c.start.Add(c.velocity), 0, turnMotion{})
			switch {
			case c.wantBody != 0 && (!staticOK || static.id != c.wantBody || (static.kind == hazardComet) != c.wantComet):
				t.Fatalf("static lookup = %+v (ok = %v), want body %d (comet=%v)", static, staticOK, c.wantBody, c.wantComet)
			case c.wantBody == 0 && c.wantInFlight && staticOK:
				t.Fatalf("static lookup = %+v, want no collision", static)
			case c.wantBody == 0 && !c.wantInFlight && (!staticOK || static.kind == hazardPlanet || static.kind == hazardComet):
				t.Fatalf("static lookup = %+v (ok = %v), want the sun or the board edge", static, staticOK)
			}

			arrivals := g.moveFleets(nil, nil)

			inFlight := len(g.Fleets) == 1
			if inFlight != c.wantInFlight {
				t.Fatalf("fleet in flight = %v, want %v", inFlight, c.wantInFlight)
			}
			if c.wantBody == 0 {
				if len(arrivals) != 0 {
					t.Fatalf("got %d arrivals, want none", len(arrivals))
				}
				return
			}
			if len(arrivals) != 1 {
				t.Fatalf("got %d arrivals, want 1 at body %d", len(arrivals), c.wantBody)
			}
			got := arrivals[0]
			if got.bodyID != c.wantBody || got.isComet != c.wantComet {
				t.Fatalf("arrived at body %d (comet=%v), want body %d (comet=%v)", got.bodyID, got.isComet, c.wantBody, c.wantComet)
			}
			if got.ownerID != 0 || got.ships != 5 {
				t.Fatalf("arrival carried owner %d with %d ships, want owner 0 with 5", got.ownerID, got.ships)
			}
		})
	}
}

// TestFirstCollisionTieBreakIgnoresStorageOrder pins the exact-tie rule without
// relying on how the board happens to store its bodies. Each case places two
// bodies whose entry times are bit-for-bit equal — mirror images across the
// path, or the very same circle — and runs the search over both storage
// orders, so a search that let whichever candidate it met first win a tie
// would fail one of them.
func TestFirstCollisionTieBreakIgnoresStorageOrder(t *testing.T) {
	// Production 1 gives radius 1 + ln(1) = 1, matching the radius New()
	// forces on every comet, so a planet and a comet at one position are the
	// same circle.
	planet := func(id int, x, y float64) Planet {
		return Planet{ID: id, Position: Vec2{x, y}, OwnerID: NeutralPlayer, Production: 1}
	}
	// A fixed path holds a comet still at one point.
	comet := func(id int, x, y float64) Comet {
		return Comet{ID: id, OwnerID: NeutralPlayer, Path: fixedCometPath(Vec2{x, y})}
	}
	// A horizontal path along y=10. A body centred at y=10±0.5 is the mirror
	// image of its partner across it, so the entry quadratics are identical.
	start, end := Vec2{20, 10}, Vec2{40, 10}

	cases := []struct {
		name     string
		planets  []Planet
		comets   []Comet
		wantKind hazardKind
		wantID   int
	}{{
		name:     "two planets tie: the lower ID claims the fleet",
		planets:  []Planet{planet(8, 30, 10.5), planet(3, 30, 9.5)},
		wantKind: hazardPlanet,
		wantID:   3,
	}, {
		name:     "two comets tie: the lower ID claims the fleet",
		comets:   []Comet{comet(9, 30, 10.5), comet(4, 30, 9.5)},
		wantKind: hazardComet,
		wantID:   4,
	}, {
		// Planets and comets share one ID space and are ranked by ID alone,
		// so the comet's lower ID beats the planet it is sitting on.
		name:     "planet and comet tie: the lower-ID comet claims the fleet",
		planets:  []Planet{planet(6, 30, 10)},
		comets:   []Comet{comet(2, 30, 10)},
		wantKind: hazardComet,
		wantID:   2,
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g, err := New(DefaultConfig(), c.planets, c.comets)
			if err != nil {
				t.Fatal(err)
			}

			// Prove the tie is exact first. Without this, a geometry mistake
			// would let the case pass on entry time alone and pin nothing.
			var entries []float64
			for _, p := range g.Planets {
				entry, ok := segmentCircleEntry(start, end, p.Position, p.Radius)
				if ok {
					entries = append(entries, entry)
				}
			}
			for _, cm := range g.Comets {
				entry, ok := segmentCircleEntry(start, end, cm.Position, cm.Radius)
				if ok {
					entries = append(entries, entry)
				}
			}
			if len(entries) != 2 || entries[0] != entries[1] {
				t.Fatalf("entry times = %v, want two bit-identical hits", entries)
			}

			for _, reversed := range []bool{false, true} {
				if reversed {
					slices.Reverse(g.Planets)
					slices.Reverse(g.Comets)
				}
				// Both search paths: the static lookup reads the game's slices,
				// and a prepared motion, built after any reversal, holds them in
				// that same storage order.
				motions := map[string]turnMotion{
					"static":   {},
					"prepared": g.prepareTurnMotion(nil, nil, nil),
				}
				for _, path := range []string{"static", "prepared"} {
					got, ok := g.firstCollision(start, end, 0, motions[path])
					if !ok || got.kind != c.wantKind || got.id != c.wantID {
						t.Fatalf("%s path, reversed storage = %v: firstCollision = %+v (ok = %v), want kind %d body %d",
							path, reversed, got, ok, c.wantKind, c.wantID)
					}
				}
			}
		})
	}
}

// TestFirstCollisionDoesNotAllocate guards the property that lets batched
// environment steps run in parallel. Incoming-fleet observations call
// firstCollision once per fleet per predicted turn; when it copied the body
// slices on every call, garbage collection became the dominant cost of a
// 64-environment step and serialized its workers. The board holds planets, a
// comet and the sun so every branch of the search runs, and the segment hits.
func TestFirstCollisionDoesNotAllocate(t *testing.T) {
	config := DefaultConfig()
	config.SunHazardEnabled = true
	g, err := New(config, []Planet{
		{ID: 5, Position: Vec2{30, 10}, OwnerID: NeutralPlayer, Production: 1},
		{ID: 1, Position: Vec2{70, 10}, OwnerID: NeutralPlayer, Production: 3},
	}, []Comet{{ID: 7, OwnerID: NeutralPlayer, Path: fixedCometPath(Vec2{50, 10})}})
	if err != nil {
		t.Fatal(err)
	}
	start, end := Vec2{20, 10}, Vec2{80, 10}
	if hit, ok := g.firstCollision(start, end, 0, turnMotion{}); !ok || hit.kind != hazardPlanet || hit.id != 1 {
		t.Fatalf("firstCollision = %+v (ok = %v), want planet 1", hit, ok)
	}

	allocs := testing.AllocsPerRun(100, func() {
		g.firstCollision(start, end, 0, turnMotion{})
	})
	if allocs != 0 {
		t.Fatalf("firstCollision allocated %v times per call, want 0", allocs)
	}
}

// TestTunnellingFleetCapturesPlanet is the end-to-end counterpart: a fleet fast
// enough to overshoot a planet still captures it through a full AdvanceTurn,
// not merely in moveFleets' return value.
func TestTunnellingFleetCapturesPlanet(t *testing.T) {
	g, err := New(DefaultConfig(), []Planet{
		{ID: 1, Position: Vec2{30, 10}, OwnerID: NeutralPlayer, Ships: 2, Production: 1},
		// A distant, uninvolved second owner. Without it, the fleet below
		// (owner 0) would be the only non-neutral owner on the board and
		// this game would already satisfy elimination (PLAN.md step 8)
		// before AdvanceTurn ever runs.
		{ID: 99, Position: Vec2{95, 95}, OwnerID: 1, Ships: 1, Production: 1},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// End point (40,10) is 10 units past the planet, so an endpoint-only test
	// would leave this fleet flying and the planet neutral.
	g.Fleets = []Fleet{{ID: 99, OwnerID: 0, Ships: 5, Position: Vec2{20, 10}, Velocity: Vec2{20, 0}}}
	if err := g.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	if len(g.Fleets) != 0 {
		t.Fatal("fleet should have been consumed by the planet it passed through")
	}
	// Neutral planets do not produce, so the garrison is still 2: 5 attackers
	// beat it and hold the planet with the surplus.
	if g.Planets[0].OwnerID != 0 || g.Planets[0].Ships != 3 {
		t.Fatalf("planet held by %d with %d ships, want owner 0 with 3", g.Planets[0].OwnerID, g.Planets[0].Ships)
	}
}

// TestLaunchedFleetSurvivesItsMovingSource is the case spawnEpsilon alone
// cannot cover. Fleets move before bodies do (since PR7), so a fleet placed
// just outside its source at launch has already moved on under its own power
// by the time that source sweeps — but the source's own path can still reach
// the fleet's new, post-movement position, and without the launch-turn
// exclusion it would be swallowed there as a friendly arrival.
//
// The planet's orbit radius (30) is deliberately larger than earlier PRs used
// (10): at radius 10, neither a 0.035 nor even a 0.1 rate moves the planet far
// enough to reach a 1-ship fleet's post-movement position (confirmed by
// disabling the exclusion and watching that geometry's test keep passing
// regardless — it wasn't exercising anything). At radius 30 the same rates
// sweep far enough to genuinely reach it, which is what makes this a real
// regression test rather than one that would still pass with the exclusion
// deleted.
//
// The planet's rate is set after New(), not in the setup: New() assigns orbit
// rates itself, and a rolled rate could be as low as half the configured mean,
// which would weaken the very margin this test depends on.
// bystander is a distant, uninvolved second owner. Without one, a
// single-owner setup already satisfies elimination (PLAN.md step 8) before
// any turn runs, and AdvanceTurn refuses to proceed on an already-finished
// game.
func bystander(id, owner int) Planet {
	return Planet{ID: id, Position: Vec2{95, 95}, OwnerID: owner, Ships: 1, Production: 1}
}

func TestLaunchedFleetSurvivesItsMovingSource(t *testing.T) {
	cases := []struct {
		name    string
		planets []Planet
		comets  []Comet
	}{{
		name: "orbiting planet",
		planets: []Planet{{
			ID: 1, OwnerID: 1, Ships: 10, Production: 1,
			Position: Vec2{80, 50}, // radius 30 from the sun, so it orbits
		}, bystander(99, 2)},
	}, {
		name:    "moving comet",
		planets: []Planet{bystander(99, 2)},
		comets: []Comet{{
			ID: 1, OwnerID: 1, Ships: 10,
			// Points on a radius-10 circle about (50, 50), 0.4 radians (4 units
			// of arc, the default CometSpeed) apart, starting at angle 0.
			Path: func() []Vec2 {
				path := make([]Vec2, 10)
				for k := range path {
					angle := 0.4 * float64(k)
					path[k] = Vec2{50 + 10*math.Cos(angle), 50 + 10*math.Sin(angle)}
				}
				return path
			}(),
		}},
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g, err := New(DefaultConfig(), c.planets, c.comets)
			if err != nil {
				t.Fatal(err)
			}
			for i := range g.Planets {
				if g.Planets[i].Orbit != nil {
					g.Planets[i].Orbit.RadiansTurn = 0.1
				}
			}
			// Angle pi/2 is straight down the board, perpendicular to the
			// source's travel at angle 0, so the fleet cannot simply outrun it.
			if err := g.AdvanceTurn([]Action{{PlayerID: 1, PlanetID: 1, Ships: 1, Angle: math.Pi / 2}}); err != nil {
				t.Fatal(err)
			}
			if len(g.Fleets) != 1 {
				t.Fatalf("launched fleet was reabsorbed by the source it just left; %d fleets remain", len(g.Fleets))
			}
		})
	}
}

// TestFleetCollidesWithSourceAfterLaunchTurn pins the other edge of that
// exclusion: it lasts exactly one turn. A source that catches its own fleet up
// on any later turn is a genuine collision and must be treated as one, or a
// fleet could shelter inside its origin planet indefinitely.
func TestFleetCollidesWithSourceAfterLaunchTurn(t *testing.T) {
	g, err := New(DefaultConfig(),
		[]Planet{{ID: 1, Position: Vec2{30, 10}, OwnerID: 1, Ships: 5, Production: 1}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// A fleet claiming planet 1 as its source, launched on an earlier turn, and
	// aimed straight back at it. The exclusion must not apply.
	g.Turn = 5
	g.Fleets = []Fleet{{ID: 99, OwnerID: 1, Ships: 3, Position: Vec2{20, 10}, Velocity: Vec2{20, 0}, FromPlanetID: 1, LaunchTurn: 2}}

	arrivals := g.moveFleets(nil, nil)

	if len(arrivals) != 1 || arrivals[0].bodyID != 1 {
		t.Fatalf("got arrivals %+v, want one at body 1", arrivals)
	}
}

// TestLaunchedFleetSurvivesItsOwnSource is the other half of PR3's spawnEpsilon:
// the spawn point sits strictly outside the source, so the very first
// continuous-collision check of the fleet's life must not read as an arrival
// back onto the planet it just left. The source is deliberately the largest
// planet size (production 5, radius 1 + ln(5)), where the margin is tightest
// relative to a small fleet's step.
func TestLaunchedFleetSurvivesItsOwnSource(t *testing.T) {
	g, err := New(DefaultConfig(),
		[]Planet{{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 10, Production: 5}, bystander(99, 2)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.AdvanceTurn([]Action{{PlayerID: 1, PlanetID: 1, Ships: 3, Angle: 0}}); err != nil {
		t.Fatal(err)
	}
	if len(g.Fleets) != 1 {
		t.Fatalf("expected the launched fleet to still be in flight, got %d fleets", len(g.Fleets))
	}
	// A fleet wrongly re-absorbed by its source would show up as the garrison
	// getting its ships straight back.
	if got := g.Planets[0].Ships; got != 12 {
		t.Fatalf("source holds %d ships, want 12 (10 - 3 launched + 5 produced)", got)
	}
}
