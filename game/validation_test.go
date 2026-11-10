package game

import (
	"errors"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func TestNewRejectsInvalidOwner(t *testing.T) {
	_, err := New(DefaultConfig(), []Planet{
		{ID: 1, Position: Vec2{20, 20}, OwnerID: 4, Ships: 1, Production: 1},
	}, nil)
	if err == nil {
		t.Fatal("owner ID outside 0-3 (and not neutral) should be rejected")
	}
}

func TestNewRejectsProductionOutOfRange(t *testing.T) {
	for _, production := range []int{0, 6} {
		_, err := New(DefaultConfig(), []Planet{
			{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 1, Production: production},
		}, nil)
		if err == nil {
			t.Fatalf("production %d outside [1,5] should be rejected", production)
		}
	}
}

func TestNewDerivesPlanetRadiusFromProduction(t *testing.T) {
	g, err := New(DefaultConfig(), []Planet{
		{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 1, Production: 5, Radius: 99},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := 1 + math.Log(5)
	if got := g.Planets[0].Radius; got != want {
		t.Fatalf("radius = %v, want %v (caller-supplied radius must be overwritten)", got, want)
	}
}

func TestNewForcesCometRadiusAndProduction(t *testing.T) {
	g, err := New(DefaultConfig(), nil, []Comet{
		{ID: 1, OwnerID: NeutralPlayer, Radius: 50, Production: 99, Path: fixedCometPath(Vec2{50, 70})},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := g.Comets[0].Radius; got != 1 {
		t.Fatalf("comet radius = %v, want 1", got)
	}
	if got := g.Comets[0].Production; got != 1 {
		t.Fatalf("comet production = %v, want 1", got)
	}
}

func TestNewRejectsNonPositiveID(t *testing.T) {
	_, err := New(DefaultConfig(), []Planet{
		{ID: 0, Position: Vec2{20, 20}, OwnerID: 1, Ships: 1, Production: 1},
	}, nil)
	if err == nil {
		t.Fatal("planet ID 0 should be rejected")
	}
}

func TestNewRejectsDuplicatePlanetID(t *testing.T) {
	_, err := New(DefaultConfig(), []Planet{
		{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 1, Production: 1},
		{ID: 1, Position: Vec2{60, 60}, OwnerID: 2, Ships: 1, Production: 1},
	}, nil)
	if err == nil {
		t.Fatal("duplicate planet ID should be rejected")
	}
}

func TestNewRejectsCometIDCollidingWithPlanetID(t *testing.T) {
	// Planets and comets share one ID space, so a comet reusing a planet's
	// ID must be rejected exactly like a duplicate within one kind.
	_, err := New(DefaultConfig(), []Planet{
		{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 1, Production: 1},
	}, []Comet{
		{ID: 1, OwnerID: NeutralPlayer},
	})
	if err == nil {
		t.Fatal("comet ID colliding with an existing planet ID should be rejected")
	}
}

func TestNewFleetIDsDoNotCollideWithExistingIDs(t *testing.T) {
	g, err := New(DefaultConfig(), []Planet{
		{ID: 1, Position: Vec2{10, 10}, OwnerID: 1, Ships: 5, Production: 1},
		bystander(99, 2),
	}, []Comet{
		{ID: 7, OwnerID: NeutralPlayer, Path: fixedCometPath(Vec2{80, 80})},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := g.AdvanceTurn([]Action{{PlayerID: 1, PlanetID: 1, Ships: 1, Angle: 0}}); err != nil {
		t.Fatal(err)
	}
	if len(g.Fleets) != 1 {
		t.Fatalf("expected one fleet, got %d", len(g.Fleets))
	}
	if got := g.Fleets[0].ID; got == 1 || got == 7 {
		t.Fatalf("new fleet ID %d collides with an existing planet or comet ID", got)
	}
}

func TestLaunchSpawnsStrictlyOutsideSourceRadius(t *testing.T) {
	g, err := New(DefaultConfig(),
		[]Planet{{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 5, Production: 1}},
		[]Comet{{ID: 2, Position: Vec2{60, 60}, OwnerID: 1, Ships: 5, Path: fixedCometPath(Vec2{60, 60})}},
	)
	if err != nil {
		t.Fatal(err)
	}
	actions := []Action{
		{PlayerID: 1, PlanetID: 1, Ships: 1, Angle: 0},
		{PlayerID: 1, PlanetID: 2, Ships: 1, Angle: 0},
	}
	// Call launch() directly rather than AdvanceTurn: AdvanceTurn also runs
	// moveFleets, which would move every fleet further from its source on
	// this same turn regardless of the spawn point, masking a regression
	// where spawnEpsilon stopped being applied at launch time.
	if err := g.launch(actions); err != nil {
		t.Fatal(err)
	}
	if len(g.Fleets) != 2 {
		t.Fatalf("expected 2 fleets, got %d", len(g.Fleets))
	}
	sources := []struct {
		name     string
		position Vec2
		radius   float64
	}{
		{"planet", Vec2{20, 20}, 1 + math.Log(1)},
		{"comet", Vec2{60, 60}, 1},
	}
	for i, src := range sources {
		dist := g.Fleets[i].Position.Distance(src.position)
		want := src.radius + spawnEpsilon
		if math.Abs(dist-want) > 1e-9 {
			t.Fatalf("%s-launched fleet spawned at distance %v from its source, want %v (radius + spawnEpsilon)", src.name, dist, want)
		}
	}
}

// TestValidateLaunchTotalsShipsWithoutOverflow covers the per-source total
// that validateLaunch checks against the garrison. The overflow cases are the
// regression: adding math.MaxInt to a running total of 10 wraps it negative,
// and the old "total > garrison" comparison then let the turn through,
// leaving the planet with a hugely negative garrison and launching a fleet of
// ships that never existed. A rejected turn must leave the garrison and the
// fleet list exactly as they were.
func TestValidateLaunchTotalsShipsWithoutOverflow(t *testing.T) {
	cases := []struct {
		name    string
		ships   []int // one action per entry, all from planet 1's garrison of 10
		wantErr bool
	}{
		{"exactly the garrison across two actions", []int{4, 6}, false},
		{"one ship over the garrison", []int{4, 7}, true},
		{"MaxInt after the full garrison", []int{10, math.MaxInt}, true},
		{"MaxInt after part of the garrison", []int{1, math.MaxInt}, true},
		{"two MaxInt actions", []int{math.MaxInt, math.MaxInt}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g, err := New(DefaultConfig(), []Planet{
				{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 10, Production: 1},
				bystander(99, 2),
			}, nil)
			if err != nil {
				t.Fatal(err)
			}
			actions := make([]Action, len(tc.ships))
			for i, ships := range tc.ships {
				actions[i] = Action{PlayerID: 1, PlanetID: 1, Ships: ships, Angle: 0}
			}

			err = g.AdvanceTurn(actions)
			if !tc.wantErr {
				if err != nil {
					t.Fatalf("a launch within the garrison was rejected: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("launch was accepted: garrison now %d, %d fleet(s) in flight", g.planet(1).Ships, len(g.Fleets))
			}
			if got := g.planet(1).Ships; got != 10 {
				t.Fatalf("rejected launch changed the garrison to %d, want 10", got)
			}
			if len(g.Fleets) != 0 {
				t.Fatalf("rejected launch still created %d fleet(s)", len(g.Fleets))
			}
		})
	}
}

func TestNewDeepCopiesOrbit(t *testing.T) {
	// RadiansTurn is set here deliberately even though New() overwrites it: the
	// caller's own struct must come back untouched, which matters more now that
	// New() writes to the orbit rather than only reading it.
	setup := []Planet{
		{ID: 1, Position: Vec2{60, 50}, OwnerID: 1, Ships: 1, Production: 1,
			Orbit: &Orbit{Radius: 10, Angle: 0, RadiansTurn: 0.03}},
		bystander(99, 2),
	}
	a, err := New(DefaultConfig(), setup, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := New(DefaultConfig(), setup, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.AdvanceTurn(nil); err != nil {
		t.Fatal(err)
	}
	if got := b.Planets[0].Orbit.Angle; got != 0 {
		t.Fatalf("game B's orbit angle = %v, want 0 (games must not share Orbit state)", got)
	}
	if got := setup[0].Orbit.Angle; got != 0 {
		t.Fatalf("caller's setup slice orbit angle = %v, want 0 (New() must not mutate the caller's Orbit)", got)
	}
	if got := setup[0].Orbit.RadiansTurn; got != 0.03 {
		t.Fatalf("caller's setup slice orbit rate = %v, want 0.03 (New() assigns rates to its own copy only)", got)
	}
}

func TestNewRejectsInvalidConfig(t *testing.T) {
	valid := DefaultConfig()
	cases := []struct {
		name   string
		modify func(*Config)
	}{
		{"zero width", func(c *Config) { c.Width = 0 }},
		{"zero height", func(c *Config) { c.Height = 0 }},
		{"negative sun radius", func(c *Config) { c.SunRadius = -1 }},
		{"zero max turns", func(c *Config) { c.MaxTurns = 0 }},
		{"fleet speed below 1", func(c *Config) { c.MaxFleetSpeed = 0.5 }},
		{"zero comet speed", func(c *Config) { c.CometSpeed = 0 }},
		{"negative comet speed", func(c *Config) { c.CometSpeed = -1 }},
		// NaN compares false against everything, so each of these used to slip
		// past the range checks above: "NaN <= 0" is false just as "5 <= 0" is.
		// +Inf passes "> 0" and "< 1" checks the same way.
		{"NaN width", func(c *Config) { c.Width = math.NaN() }},
		{"infinite width", func(c *Config) { c.Width = math.Inf(1) }},
		{"NaN height", func(c *Config) { c.Height = math.NaN() }},
		{"NaN sun x", func(c *Config) { c.Sun.X = math.NaN() }},
		{"infinite sun y", func(c *Config) { c.Sun.Y = math.Inf(-1) }},
		{"NaN sun radius", func(c *Config) { c.SunRadius = math.NaN() }},
		{"NaN max fleet speed", func(c *Config) { c.MaxFleetSpeed = math.NaN() }},
		{"infinite max fleet speed", func(c *Config) { c.MaxFleetSpeed = math.Inf(1) }},
		{"NaN comet speed", func(c *Config) { c.CometSpeed = math.NaN() }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			config := valid
			tc.modify(&config)
			if _, err := New(config, nil, nil); err == nil {
				t.Fatalf("expected error for invalid config (%s)", tc.name)
			}
		})
	}
}

// TestNewRejectsNonFiniteBodyGeometry covers the planet and comet numbers that
// place a body on the board. A non-finite one leaves the body nowhere: every
// collision test against it fails silently, so it can never be reached. A
// comet is checked at every point of its path, not only its current one,
// since it will visit each of them.
func TestNewRejectsNonFiniteBodyGeometry(t *testing.T) {
	nan := math.NaN()
	inf := math.Inf(1)
	cometPath := func(later Vec2) []Comet {
		return []Comet{{ID: 2, OwnerID: NeutralPlayer, Path: []Vec2{{50, 50}, later}}}
	}
	cases := []struct {
		name    string
		planets []Planet
		comets  []Comet
		wantErr bool
	}{
		{"static planet NaN x", []Planet{{ID: 1, Position: Vec2{nan, 10}, Production: 1, OwnerID: NeutralPlayer}}, nil, true},
		{"static planet infinite y", []Planet{{ID: 1, Position: Vec2{10, inf}, Production: 1, OwnerID: NeutralPlayer}}, nil, true},
		{"comet NaN later path point", nil, cometPath(Vec2{nan, 50}), true},
		{"comet infinite later path point", nil, cometPath(Vec2{50, inf}), true},
		{"comet finite path", nil, cometPath(Vec2{54, 50}), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New(DefaultConfig(), tc.planets, tc.comets)
			if tc.wantErr && err == nil {
				t.Fatal("expected New to reject non-finite geometry")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("expected New to accept this setup, got %v", err)
			}
		})
	}
}

// TestAdvanceTurnReportsRejectedActionAsActionError checks that every way a
// launch can be rejected comes back as an *ActionError pointing at the
// failing action. Each case puts a valid action from player 1 first and the bad
// one second, so a correct Index of 1 proves the error names the culprit
// rather than the first action in the batch.
func TestAdvanceTurnReportsRejectedActionAsActionError(t *testing.T) {
	valid := Action{PlayerID: 1, PlanetID: 1, Ships: 2, Angle: 0}
	cases := []struct {
		name string
		bad  Action
	}{
		{"neutral player", Action{PlayerID: NeutralPlayer, PlanetID: 1, Ships: 1}},
		{"zero ships", Action{PlayerID: 2, PlanetID: 2, Ships: 0}},
		{"NaN angle", Action{PlayerID: 2, PlanetID: 2, Ships: 1, Angle: math.NaN()}},
		{"infinite angle", Action{PlayerID: 2, PlanetID: 2, Ships: 1, Angle: math.Inf(1)}},
		{"source owned by someone else", Action{PlayerID: 2, PlanetID: 1, Ships: 1}},
		{"source that does not exist", Action{PlayerID: 2, PlanetID: 42, Ships: 1}},
		// Planet 1 holds 5; the valid action already claims 2 of them.
		{"total exceeds the garrison", Action{PlayerID: 1, PlanetID: 1, Ships: 4}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g, err := New(DefaultConfig(), []Planet{
				{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 5, Production: 1},
				{ID: 2, Position: Vec2{80, 80}, OwnerID: 2, Ships: 5, Production: 1},
			}, nil)
			if err != nil {
				t.Fatal(err)
			}
			err = g.AdvanceTurn([]Action{valid, tc.bad})
			var actionErr *ActionError
			if !errors.As(err, &actionErr) {
				t.Fatalf("error = %v (%T), want *ActionError", err, err)
			}
			if actionErr.Index != 1 || actionErr.PlayerID != tc.bad.PlayerID || actionErr.SourceID != tc.bad.PlanetID {
				t.Fatalf("error = %+v, want index 1, player %d, source %d", *actionErr, tc.bad.PlayerID, tc.bad.PlanetID)
			}
			if actionErr.Reason == "" {
				t.Fatal("error has no reason")
			}
		})
	}
}

func TestSeededConfigIsDeterministic(t *testing.T) {
	// Nothing reads Config.Rand yet (comet spawning arrives in a later PR),
	// so this locks in whole-match determinism from setup+seed+actions today
	// and will keep covering RNG-driven rules once they land.
	newGame := func() *Game {
		config := DefaultConfig()
		config.Rand = rand.New(rand.NewSource(42))
		g, err := New(config, []Planet{
			// About 42 from the sun, inside OrbitRadiusLimit, so it orbits.
			{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 20, Production: 3},
			{ID: 2, Position: Vec2{80, 80}, OwnerID: 2, Ships: 20, Production: 3},
		}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	a, b := newGame(), newGame()
	actions := []Action{{PlayerID: 1, PlanetID: 1, Ships: 5, Angle: 0.7}}
	for turn := 0; turn < 5; turn++ {
		if err := a.AdvanceTurn(actions); err != nil {
			t.Fatal(err)
		}
		if err := b.AdvanceTurn(actions); err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(a.Planets, b.Planets) {
		t.Fatalf("planets diverged with the same seed:\na=%+v\nb=%+v", a.Planets, b.Planets)
	}
	if !reflect.DeepEqual(a.Fleets, b.Fleets) {
		t.Fatalf("fleets diverged with the same seed:\na=%+v\nb=%+v", a.Fleets, b.Fleets)
	}
	if a.Turn != b.Turn {
		t.Fatalf("turn counters diverged: %d vs %d", a.Turn, b.Turn)
	}
}

func TestNewDefaultsRandWhenNil(t *testing.T) {
	config := DefaultConfig()
	if config.Rand != nil {
		t.Fatal("DefaultConfig should leave Rand nil so New() can seed it")
	}
	g, err := New(config, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if g.Config.Rand == nil {
		t.Fatal("New() should resolve a nil Rand to a time-seeded source")
	}
}

// TestPlanetRadiusMatchesConstruction pins the exported formula to what New()
// actually assigns. The generator in package scenario spaces planets using
// PlanetRadius before any Game exists, so the two drifting apart would let it
// build maps whose planets overlap once the engine sizes them for real.
func TestPlanetRadiusMatchesConstruction(t *testing.T) {
	for production := 1; production <= 5; production++ {
		g := testGame(t,
			Planet{ID: 1, Position: Vec2{20, 20}, OwnerID: 1, Ships: 1, Production: production},
			bystander(99, 2),
		)
		if got, want := g.Planets[0].Radius, PlanetRadius(production); got != want {
			t.Fatalf("production %d: New() radius = %v, PlanetRadius = %v", production, got, want)
		}
	}
}
