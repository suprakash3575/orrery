package game

import (
	"math"
	"math/rand"
	"testing"
)

// TestFirstHitMatchesEngineMovement is the point of exporting the predictor at
// all: what a caller is told will happen has to be what the turn loop then
// does. Each case predicts a fleet's next segment, advances the real engine,
// and checks the outcome the prediction implied — an arrival hands ships to a
// body, a hazard destroys the fleet, a clear path leaves it flying.
//
// Fleets are placed directly rather than launched so each geometry is exact.
// FromPlanetID is left zero so no launch-source exclusion applies here; that
// rule has its own test below.
func TestFirstHitMatchesEngineMovement(t *testing.T) {
	const target = 5
	tests := []struct {
		name        string
		sunHazard   bool
		position    Vec2
		velocity    Vec2
		wantKind    HitKind
		wantBodyID  int
		wantT       float64
		wantNoHit   bool
		verifyAfter func(t *testing.T, g *Game)
	}{
		{
			// The path enters planet 5 (radius 1, spanning x 23..25) at x=23,
			// three fifths of the way along a five-unit segment.
			name:       "reaches a planet",
			sunHazard:  true,
			position:   Vec2{X: 20, Y: 50},
			velocity:   Vec2{X: 5, Y: 0},
			wantKind:   HitPlanet,
			wantBodyID: target,
			wantT:      0.6,
			verifyAfter: func(t *testing.T, g *Game) {
				if len(g.Fleets) != 0 {
					t.Fatalf("fleets = %d, want 0: the fleet should have arrived", len(g.Fleets))
				}
				planet := planetWithID(t, g, target)
				if planet.OwnerID != 1 {
					t.Fatalf("planet %d owner = %d, want 1: arrival did not resolve", target, planet.OwnerID)
				}
			},
		},
		{
			name:      "leaves the board",
			sunHazard: true,
			position:  Vec2{X: 95, Y: 50},
			velocity:  Vec2{X: 10, Y: 0},
			wantKind:  HitBoundary,
			wantT:     0.5,
			verifyAfter: func(t *testing.T, g *Game) {
				if len(g.Fleets) != 0 {
					t.Fatalf("fleets = %d, want 0: the fleet should have been destroyed", len(g.Fleets))
				}
				if planetWithID(t, g, target).OwnerID != NeutralPlayer {
					t.Fatal("a destroyed fleet must not deliver ships anywhere")
				}
			},
		},
		{
			// The sun spans x 40..60, so this ten-unit segment enters halfway.
			name:      "crosses the sun while the hazard is enabled",
			sunHazard: true,
			position:  Vec2{X: 35, Y: 50},
			velocity:  Vec2{X: 10, Y: 0},
			wantKind:  HitSun,
			wantT:     0.5,
			verifyAfter: func(t *testing.T, g *Game) {
				if len(g.Fleets) != 0 {
					t.Fatalf("fleets = %d, want 0: the sun should have destroyed the fleet", len(g.Fleets))
				}
			},
		},
		{
			// Identical geometry with the hazard off. The prediction must stop
			// reporting the sun, because the engine stops enforcing it.
			name:      "crosses the same point while the hazard is disabled",
			sunHazard: false,
			position:  Vec2{X: 35, Y: 50},
			velocity:  Vec2{X: 10, Y: 0},
			wantNoHit: true,
			verifyAfter: func(t *testing.T, g *Game) {
				if len(g.Fleets) != 1 {
					t.Fatalf("fleets = %d, want 1: the fleet should have survived", len(g.Fleets))
				}
				if got := g.Fleets[0].Position; got != (Vec2{X: 45, Y: 50}) {
					t.Fatalf("fleet position = %v, want {45 50}", got)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			g := predictionGame(t, test.sunHazard)
			g.Fleets = []Fleet{{
				ID:       50,
				OwnerID:  1,
				Ships:    5,
				Position: test.position,
				Velocity: test.velocity,
			}}

			start := test.position
			end := start.Add(test.velocity)
			hit, ok := g.FirstHit(start, end, 0)

			if test.wantNoHit {
				if ok {
					t.Fatalf("FirstHit(%v, %v) = %+v, want no hit", start, end, hit)
				}
			} else {
				if !ok {
					t.Fatalf("FirstHit(%v, %v) reported no hit, want %v", start, end, test.wantKind)
				}
				if hit.Kind != test.wantKind {
					t.Fatalf("kind = %v, want %v", hit.Kind, test.wantKind)
				}
				if hit.BodyID != test.wantBodyID {
					t.Fatalf("body = %d, want %d", hit.BodyID, test.wantBodyID)
				}
				if math.Abs(hit.T-test.wantT) > 1e-9 {
					t.Fatalf("t = %v, want %v", hit.T, test.wantT)
				}
				if got, want := hit.IsBody(), test.wantBodyID != 0; got != want {
					t.Fatalf("IsBody() = %v, want %v", got, want)
				}
			}

			// The prediction is only worth anything if the engine then agrees.
			if err := g.AdvanceTurn(nil); err != nil {
				t.Fatal(err)
			}
			test.verifyAfter(t, g)
		})
	}
}

// TestFirstHitHonoursLaunchSourceExclusion covers the rule a caller is most
// likely to get wrong: on its launch turn a fleet ignores the body it came
// from. A predictor that omitted the exclusion would announce an arrival back
// at the source that the engine never produces.
//
// The fleet is built exactly where launch() would place it — spawnEpsilon
// outside the source radius — so the geometry is the real one rather than an
// approximation of it.
func TestFirstHitHonoursLaunchSourceExclusion(t *testing.T) {
	const source = 1
	g := predictionGame(t, true)
	origin := planetWithID(t, g, source)

	// Aim straight up, away from every other body on the board.
	direction := Vec2{X: 0, Y: -1}
	start := origin.Position.Add(direction.Scale(origin.Radius + spawnEpsilon))
	velocity := direction.Scale(g.FleetSpeed(5))
	end := start.Add(velocity)

	if hit, ok := g.FirstHit(start, end, source); ok && hit.BodyID == source {
		t.Fatalf("excluding the source still predicted a hit on it: %+v", hit)
	}

	// Without the exclusion the same segment does strike the source, which is
	// what makes the exclusion above meaningful rather than vacuous: the fleet
	// starts a hair outside the radius, so the check has something to skip.
	included, ok := g.FirstHit(origin.Position, end, 0)
	if !ok || included.BodyID != source {
		t.Fatalf("without exclusion, a path from the source's centre should strike it; got %+v ok=%v", included, ok)
	}

	// The engine must reach the same conclusion through a real launch.
	if err := g.AdvanceTurn([]Action{{PlayerID: 1, PlanetID: source, Ships: 5, Angle: -math.Pi / 2}}); err != nil {
		t.Fatal(err)
	}
	if len(g.Fleets) != 1 {
		t.Fatalf("fleets = %d, want 1: the launch was swallowed by its own source", len(g.Fleets))
	}
}

// TestFirstHitMatchesFirstCollision is what keeps the exported predictor a
// wrapper rather than a second implementation. Randomised segments across a
// populated board are compared against the engine's own call, so any future
// divergence — a reordered tie-break, a dropped hazard, a copied-and-edited
// body of code — fails here rather than silently telling callers something the
// turn loop will not do.
func TestFirstHitMatchesFirstCollision(t *testing.T) {
	g := predictionGame(t, true)
	rng := rand.New(rand.NewSource(9))

	for range 2000 {
		start := Vec2{X: rng.Float64() * 100, Y: rng.Float64() * 100}
		end := Vec2{X: rng.Float64() * 100, Y: rng.Float64() * 100}
		exclude := 0
		if rng.Intn(4) == 0 {
			exclude = 1 + rng.Intn(5)
		}

		want, wantOK := g.firstCollision(start, end, exclude, turnMotion{})
		got, gotOK := g.FirstHit(start, end, exclude)

		if gotOK != wantOK {
			t.Fatalf("segment %v→%v exclude %d: FirstHit ok = %v, firstCollision ok = %v",
				start, end, exclude, gotOK, wantOK)
		}
		if !wantOK {
			continue
		}
		if got.Kind != hitKind(want.kind) || got.T != want.t {
			t.Fatalf("segment %v→%v exclude %d: FirstHit = %+v, firstCollision = %+v",
				start, end, exclude, got, want)
		}
		wantBody := 0
		if want.kind == hazardPlanet || want.kind == hazardComet {
			wantBody = want.id
		}
		if got.BodyID != wantBody {
			t.Fatalf("segment %v→%v: body = %d, want %d", start, end, got.BodyID, wantBody)
		}
	}
}

// TestHitKindCoversEveryHazard pins the mapping between the engine's internal
// classification and the public one. The two enumerations are deliberately not
// assumed to share an iota order, so this checks every hazard maps to a
// distinct kind — a silent relabelling would otherwise be invisible.
func TestHitKindCoversEveryHazard(t *testing.T) {
	tests := []struct {
		hazard hazardKind
		want   HitKind
		name   string
		isBody bool
	}{
		{hazardBoundary, HitBoundary, "boundary", false},
		{hazardSun, HitSun, "sun", false},
		{hazardPlanet, HitPlanet, "planet", true},
		{hazardComet, HitComet, "comet", true},
	}
	seen := map[HitKind]bool{}
	for _, test := range tests {
		got := hitKind(test.hazard)
		if got != test.want {
			t.Errorf("hitKind(%d) = %v, want %v", int(test.hazard), got, test.want)
		}
		if seen[got] {
			t.Errorf("hazard %d maps to %v, already used by another hazard", int(test.hazard), got)
		}
		seen[got] = true
		if got.String() != test.name {
			t.Errorf("%v.String() = %q, want %q", got, got.String(), test.name)
		}
		if isBody := (Hit{Kind: got}).IsBody(); isBody != test.isBody {
			t.Errorf("%v IsBody() = %v, want %v", got, isBody, test.isBody)
		}
	}
}

// predictionGame builds a board with a home planet for each of two players, a
// neutral target, and the standard sun. Two owners keep elimination from ending
// the match before a turn can run.
func predictionGame(t *testing.T, sunHazard bool) *Game {
	t.Helper()
	config := DefaultConfig()
	config.SunHazardEnabled = sunHazard
	// FirstHit treats every body as stationary, so the board must not move.
	// Planet 5 sits 26 from the sun, inside the orbit limit; with orbits on it
	// would turn at a rate drawn from an unseeded source and sometimes drift
	// off the fleet's path before the engine's turn resolves.
	config.OrbitsEnabled = false
	g, err := New(config, []Planet{
		{ID: 1, Position: Vec2{X: 10, Y: 80}, OwnerID: 1, Ships: 20, Production: 1},
		{ID: 5, Position: Vec2{X: 24, Y: 50}, OwnerID: NeutralPlayer, Ships: 2, Production: 1},
		bystander(99, 2),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func planetWithID(t *testing.T, g *Game, id int) Planet {
	t.Helper()
	for _, planet := range g.Planets {
		if planet.ID == id {
			return planet
		}
	}
	t.Fatalf("planet %d not found", id)
	return Planet{}
}

// TestPredictFleetsMovingBodies pins the three things PredictFleets does that
// FirstHit cannot: a body that moves onto a fleet claims it, a comet that
// departs the same turn takes the arrival with it, and the launch-source
// exclusion holds for exactly the launch turn even when the source moves.
func TestPredictFleetsMovingBodies(t *testing.T) {
	tests := []struct {
		name  string
		setup func(t *testing.T) *Game
		want  Landing
	}{
		{
			// The parked fleet is 1.5 from the planet's starting point, outside
			// its radius of 1, so FirstHit on a zero-length segment reports
			// nothing; the planet's turn sweeps over it (see sweep_test.go).
			name: "orbiting planet sweeps up a parked fleet",
			setup: func(t *testing.T) *Game {
				return sweptGame(t, sweptFleetPosition,
					[]Planet{{ID: 1, OwnerID: 1, Ships: 2, Production: 1, Position: Vec2{80, 50}}},
					nil, map[int]float64{1: 0.1})
			},
			want: Landing{FleetID: 99, Resolved: true, Hit: Hit{Kind: HitPlanet, BodyID: 1}, Segment: 0},
		},
		{
			// A one-point path makes the comet's first move its departing one,
			// as in TestDepartingCometTakesItsArrivalsWithIt.
			name: "comet departs the turn the fleet reaches it",
			setup: func(t *testing.T) *Game {
				return sweptGame(t, Vec2{80, 20},
					[]Planet{bystander(98, 1)},
					[]Comet{{ID: 3, OwnerID: 1, Ships: 4, Path: []Vec2{{80, 20}}}}, nil)
			},
			want: Landing{FleetID: 99, Resolved: true, Hit: Hit{Kind: HitComet, BodyID: 3}, Segment: 0, Departed: true},
		},
		{
			// A one-ship fleet launched straight down from a planet orbiting at
			// 0.1 rad/turn. The planet moves down after it on the launch turn,
			// so only the exclusion saves it there; the next turn the planet
			// starts that segment already overlapping the fleet and claims it.
			name: "orbiting source claims its fleet only after the launch turn",
			setup: func(t *testing.T) *Game {
				g := sweptGame(t, Vec2{},
					[]Planet{{ID: 1, OwnerID: 1, Ships: 10, Production: 1, Position: Vec2{80, 50}}, bystander(98, 2)},
					nil, map[int]float64{1: 0.1})
				g.Fleets = nil
				if err := g.launch([]Action{{PlayerID: 1, PlanetID: 1, Ships: 1, Angle: math.Pi / 2}}); err != nil {
					t.Fatal(err)
				}
				return g
			},
			want: Landing{FleetID: 99, Resolved: true, Hit: Hit{Kind: HitPlanet, BodyID: 1}, Segment: 1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			g := test.setup(t)
			got := g.PredictFleets(10)
			if len(got) != 1 {
				t.Fatalf("PredictFleets returned %d landings, want 1", len(got))
			}
			// Hit.T is geometry already covered by FirstHit's tests; these
			// cases are about which body, on which turn.
			got[0].Hit.T = 0
			if got[0] != test.want {
				t.Fatalf("PredictFleets = %+v, want %+v", got[0], test.want)
			}
		})
	}
}

// TestPredictFleetsMatchesEngine is the parity check: on random boards with
// orbiting planets and hand-placed comets, a fleet must vanish on exactly the
// turn PredictFleets names, and the body it was predicted to arrive at must be
// the one it captures.
//
// Each board has one fleet, owned by player 0, who owns nothing else. The
// fleet carries more ships than any garrison can reach in the turns it flies,
// so an arrival always captures, and a body flipping to player 0 identifies
// exactly where it landed. A hazard death or a departing comet leaves no body
// owned by player 0. Player 1 owns one planet so the match is not over before
// it starts. Comet spawning is off: PredictFleets does not simulate waves.
func TestPredictFleetsMatchesEngine(t *testing.T) {
	movingHits := 0
	for seed := range int64(300) {
		g := randomPredictionBoard(t, seed)
		landing := g.PredictFleets(200)[0]
		if !landing.Resolved {
			t.Fatalf("seed %d: fleet unresolved after 200 turns, longer than any crossing of this board", seed)
		}
		if landing.Hit.Kind == HitComet || (landing.Hit.Kind == HitPlanet && planetWithID(t, g, landing.Hit.BodyID).Orbit != nil) {
			movingHits++
		}

		// The engine resolves a hit at the end of the call that moves that
		// segment, so a Segment-0 landing is gone after the first call.
		for call := range landing.Segment + 1 {
			if err := g.AdvanceTurn(nil); err != nil {
				t.Fatalf("seed %d call %d: %v", seed, call, err)
			}
			flying := len(g.Fleets) == 1
			if call < landing.Segment && !flying {
				t.Fatalf("seed %d: fleet gone after call %d, predicted %+v", seed, call, landing)
			}
			if call == landing.Segment && flying {
				t.Fatalf("seed %d: fleet still flying after predicted call %d, predicted %+v", seed, call, landing)
			}
		}

		var captured []int
		for _, p := range g.Planets {
			if p.OwnerID == 0 {
				captured = append(captured, p.ID)
			}
		}
		for _, c := range g.Comets {
			if c.OwnerID == 0 {
				captured = append(captured, c.ID)
			}
		}
		switch {
		case landing.Arrives() && (len(captured) != 1 || captured[0] != landing.Hit.BodyID):
			t.Fatalf("seed %d: player 0 owns %v, want only body %d from %+v", seed, captured, landing.Hit.BodyID, landing)
		case !landing.Arrives() && len(captured) != 0:
			t.Fatalf("seed %d: player 0 owns %v, want nothing after %+v", seed, captured, landing)
		}
	}
	// The sweep must actually test moving bodies, not only static ones. A
	// departing comet is too rare to rely on at random; the table test above
	// covers it.
	if movingHits == 0 {
		t.Fatal("sweep never hit an orbiting planet or a comet")
	}
}

// randomPredictionBoard builds one seeded board for TestPredictFleetsMatchesEngine:
// six to ten non-overlapping planets (those near the sun orbit), up to three
// comets on straight paths that cross the board, and one fleet in flight.
func randomPredictionBoard(t *testing.T, seed int64) *Game {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	config := DefaultConfig()
	config.CometSpawningEnabled = false
	config.Rand = rand.New(rand.NewSource(seed))
	sun := config.Sun

	// Rejection sampling keeps every planet clear of the sun and of each other,
	// so each one is a distinct target.
	var planets []Planet
	for len(planets) < 6+rng.Intn(5) {
		production := 1 + rng.Intn(5)
		radius := PlanetRadius(production)
		position := Vec2{5 + 90*rng.Float64(), 5 + 90*rng.Float64()}
		if position.Distance(sun) < config.SunRadius+radius+2 {
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
	// Production 1 keeps player 1's garrison below the fleet's ships for every
	// turn the fleet can fly (at most ~150 here).
	planets[0].OwnerID, planets[0].Ships, planets[0].Production = 1, 0, 1

	// Comets step CometSpeed along a straight line through a random board
	// point, keeping only the on-board points, and start partway along.
	var comets []Comet
	for i := range rng.Intn(4) {
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
			ID: 100 + i, OwnerID: NeutralPlayer, Ships: rng.Intn(21),
			Path: path, PathIndex: rng.Intn(len(path)),
		})
	}

	g, err := New(config, planets, comets)
	if err != nil {
		t.Fatal(err)
	}
	ships := 200 + rng.Intn(400)
	angle := 2 * math.Pi * rng.Float64()
	g.Fleets = []Fleet{{
		ID: 1000, OwnerID: 0, Ships: ships,
		Position: Vec2{5 + 90*rng.Float64(), 5 + 90*rng.Float64()},
		Velocity: Vec2{math.Cos(angle), math.Sin(angle)}.Scale(g.FleetSpeed(ships)),
		// No launch-source exclusion: that rule has its own case above.
		LaunchTurn: -1,
	}}
	return g
}

// aimGame builds a lead-aiming board: comet spawning off, so nothing appears
// mid-flight, and exact orbit rates set after New(), which otherwise assigns
// the shared rate itself. Player 1's bystander keeps the match from ending.
func aimGame(t *testing.T, config Config, planets []Planet, comets []Comet, rates map[int]float64) *Game {
	t.Helper()
	config.CometSpawningEnabled = false
	g, err := New(config, append(planets, bystander(99, 1)), comets)
	if err != nil {
		t.Fatal(err)
	}
	for id, rate := range rates {
		g.planet(id).Orbit.RadiansTurn = rate
	}
	return g
}

// TestLeadAimReachesMovingTargets launches for real at every angle LeadAim
// returns and checks the fleet captures the target within the turn LeadAim
// named. For the moving targets it also checks that aiming at the target's
// current position misses, which is the reason LeadAim exists.
func TestLeadAimReachesMovingTargets(t *testing.T) {
	// Source planet 1 belongs to player 0. Every source here is more than 50
	// from the sun, so it stays still.
	source := func(position Vec2) Planet {
		return Planet{ID: 1, Position: position, OwnerID: 0, Ships: 50, Production: 1}
	}
	tests := []struct {
		name        string
		planets     []Planet
		comets      []Comet
		rates       map[int]float64
		target      int
		ships       int
		naiveMisses bool
	}{
		{
			name:    "static planet",
			planets: []Planet{source(Vec2{10, 90}), {ID: 2, Position: Vec2{90, 90}, OwnerID: NeutralPlayer, Production: 1}},
			target:  2, ships: 20,
		},
		{
			// 30 from the sun at 0.05 rad/turn, it covers about 24 units in
			// the ~16 turns a 20-ship fleet needs to get there.
			name:    "orbiting planet",
			planets: []Planet{source(Vec2{90, 10}), {ID: 2, Position: Vec2{80, 50}, OwnerID: NeutralPlayer, Production: 1}},
			rates:   map[int]float64{2: 0.05},
			target:  2, ships: 20, naiveMisses: true,
		},
		{
			// The comet runs left along y=80 at 4 units/turn, faster than the
			// 20-ship fleet's 2.43, and is caught about 12 turns out.
			name:    "comet faster than the fleet",
			planets: []Planet{source(Vec2{10, 90})},
			comets: []Comet{{ID: 3, OwnerID: NeutralPlayer, Path: func() []Vec2 {
				path := make([]Vec2, 21)
				for k := range path {
					path[k] = Vec2{90 - 4*float64(k), 80}
				}
				return path
			}()}},
			target: 3, ships: 20, naiveMisses: true,
		},
		{
			// A one-ship fleet against a comet sweeping past just below the
			// source. The comet is within the fleet's reach only partway
			// through the first turn, so a search that samples whole turns
			// finds no contact at all.
			name:    "comet in reach only mid-turn",
			planets: []Planet{source(Vec2{10, 90})},
			comets: []Comet{{ID: 3, OwnerID: NeutralPlayer, Path: func() []Vec2 {
				path := make([]Vec2, 21)
				for k := range path {
					path[k] = Vec2{8 + 4*float64(k), 88.8}
				}
				return path
			}()}},
			target: 3, ships: 1,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			g := aimGame(t, DefaultConfig(), test.planets, test.comets, test.rates)
			aim, ok := g.LeadAim(1, test.target, test.ships)
			if !ok {
				t.Fatal("LeadAim reported the target unreachable")
			}

			if test.naiveMisses {
				src := g.launchSource(1)
				delta := g.launchSource(test.target).position().Sub(src.position())
				naive := g.newFleet(src, test.ships, math.Atan2(delta.Y, delta.X))
				if l := g.predictLandings([]Fleet{naive}, 200)[0]; l.Arrives() && l.Hit.BodyID == test.target {
					t.Fatal("aiming at the target's current position also reaches it; the case does not need lead aiming")
				}
			}

			if err := g.AdvanceTurn([]Action{{PlayerID: 0, PlanetID: 1, Ships: test.ships, Angle: aim.Angle}}); err != nil {
				t.Fatal(err)
			}
			calls := 1
			for len(g.Fleets) > 0 {
				if err := g.AdvanceTurn(nil); err != nil {
					t.Fatal(err)
				}
				calls++
			}
			// The fleet resolves during call number calls, which covers
			// turns calls-1 to calls after launch.
			if aim.Turns < float64(calls-1) || aim.Turns > float64(calls) {
				t.Fatalf("fleet resolved on call %d, LeadAim predicted %.3f turns", calls, aim.Turns)
			}
			if owner := g.launchSource(test.target).owner(); owner != 0 {
				t.Fatalf("target %d owner = %d after the fleet resolved, want 0", test.target, owner)
			}
		})
	}
}

// TestEarliestReach checks the per-turn contact solver on hand-built cases,
// including the ones where a careless quadratic goes wrong. For every case
// with a contact it checks the defining property rather than a hand-computed
// number: the target is in reach at the returned moment, and was not a
// millionth of a turn earlier, so the moment is the first contact.
func TestEarliestReach(t *testing.T) {
	tests := []struct {
		name   string
		offset Vec2 // target minus source at the start of the turn
		move   Vec2 // the target's movement over the turn
		reach  float64
		speed  float64
		wantOK bool
	}{
		// Out of reach at both ends of the turn, in reach at its middle.
		{name: "in reach only mid-turn", offset: Vec2{-5, 1.2}, move: Vec2{10, 0}, reach: 1, speed: 1, wantOK: true},
		// The target's speed equals the fleet's, so the quadratic term is
		// exactly zero and only the linear root exists.
		{name: "equal speeds, approaching", offset: Vec2{2.5, 0}, move: Vec2{-1, 0}, reach: 1, speed: 1, wantOK: true},
		{name: "equal speeds, receding", offset: Vec2{2.5, 0}, move: Vec2{1, 0}, reach: 1, speed: 1, wantOK: false},
		// A quadratic term of about 2e-9: the textbook formula's large root
		// is meaningless here and the useful one loses its digits.
		{name: "nearly linear", offset: Vec2{2.5, 0}, move: Vec2{-(1 + 1e-9), 0}, reach: 1, speed: 1, wantOK: true},
		// The target's path just touches the fleet's reach at mid-turn; 0.09
		// is not exact in binary, so the discriminant rounds near zero.
		{name: "grazing contact", offset: Vec2{-0.3, 1}, move: Vec2{0.6, 0}, reach: 1, speed: 0, wantOK: true},
		{name: "contact as the turn starts", offset: Vec2{2, 0}, move: Vec2{0, 1}, reach: 2, speed: 1, wantOK: true},
		{name: "contact as the turn ends", offset: Vec2{3, 0}, reach: 1, speed: 2, wantOK: true},
		{name: "never in reach", offset: Vec2{10, 0}, reach: 1, speed: 1, wantOK: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// gap is how far the target is beyond the fleet's reach at tau;
			// zero or less means in reach.
			gap := func(tau float64) float64 {
				return test.offset.Add(test.move.Scale(tau)).Length() - (test.reach + test.speed*tau)
			}
			tau, ok := earliestReach(test.offset, test.move, test.reach, test.speed)
			if ok != test.wantOK {
				t.Fatalf("earliestReach = (%v, %v), want ok %v", tau, ok, test.wantOK)
			}
			if !ok {
				return
			}
			if tau < 0 || tau > 1 {
				t.Fatalf("contact at tau %v, outside the turn", tau)
			}
			if g := gap(tau); g > 1e-9 {
				t.Fatalf("target is %v beyond reach at the returned tau %v", g, tau)
			}
			if tau > 0 && gap(tau-1e-6) <= 0 {
				t.Fatalf("target was already in reach just before tau %v", tau)
			}
		})
	}
}

// TestLeadAimReportsUnreachable covers the two blockers the plan names: the
// sun standing between source and target, and a comet that leaves the board
// before any fleet of that size can reach it.
func TestLeadAimReportsUnreachable(t *testing.T) {
	// Orbits are off so the sun case's planets stay put inside the orbit limit.
	config := DefaultConfig()
	config.OrbitsEnabled = false
	tests := []struct {
		name    string
		planets []Planet
		comets  []Comet
		target  int
	}{
		{
			name: "target behind the sun",
			planets: []Planet{
				{ID: 1, Position: Vec2{20, 50}, OwnerID: 0, Ships: 50, Production: 1},
				{ID: 2, Position: Vec2{80, 50}, OwnerID: NeutralPlayer, Production: 1},
			},
			target: 2,
		},
		{
			// Three points, so the comet is gone after three turns; a 20-ship
			// fleet needs about 30 to cover the distance.
			name:    "comet leaves first",
			planets: []Planet{{ID: 1, Position: Vec2{10, 90}, OwnerID: 0, Ships: 50, Production: 1}},
			comets:  []Comet{{ID: 3, OwnerID: NeutralPlayer, Path: []Vec2{{90, 80}, {86, 80}, {82, 80}}}},
			target:  3,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			g := aimGame(t, config, test.planets, test.comets, nil)
			if aim, ok := g.LeadAim(1, test.target, 20); ok {
				t.Fatalf("LeadAim = %+v, want unreachable", aim)
			}
		})
	}
}
