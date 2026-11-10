package agent

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"orrery/game"
)

func TestNewViewCopiesConfigAndSortsEntities(t *testing.T) {
	g := viewTestGame(t)
	view := NewView(g)

	wantConfig := ConfigView{
		Width:                g.Config.Width,
		Height:               g.Config.Height,
		Sun:                  g.Config.Sun,
		SunRadius:            g.Config.SunRadius,
		SunHazardEnabled:     g.Config.SunHazardEnabled,
		CometSpawningEnabled: g.Config.CometSpawningEnabled,
		MaxTurns:             g.Config.MaxTurns,
		MaxFleetSpeed:        g.Config.MaxFleetSpeed,
		AngularVelocity:      g.OrbitRate,
		CometSpeed:           g.Config.CometSpeed,
	}
	if view.Turn != g.Turn {
		t.Fatalf("Turn = %d, want %d", view.Turn, g.Turn)
	}
	if !reflect.DeepEqual(view.Config, wantConfig) {
		t.Fatalf("Config = %#v, want %#v", view.Config, wantConfig)
	}
	if got, want := planetIDs(view.Planets), []int{1, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("planet IDs = %v, want %v", got, want)
	}
	if got, want := cometIDs(view.Comets), []int{4, 6}; !reflect.DeepEqual(got, want) {
		t.Fatalf("comet IDs = %v, want %v", got, want)
	}
	if got, want := fleetIDs(view.Fleets), []int{8, 9}; !reflect.DeepEqual(got, want) {
		t.Fatalf("fleet IDs = %v, want %v", got, want)
	}

	for _, ships := range []int{0, 1, 10, 1000} {
		if got, want := view.FleetSpeed(ships), game.FleetSpeed(ships, g.Config.MaxFleetSpeed); got != want {
			t.Fatalf("FleetSpeed(%d) = %v, want %v", ships, got, want)
		}
	}
}

func TestNewViewIsolatesGameAndSeats(t *testing.T) {
	g := viewTestGame(t)
	first := NewView(g)
	second := NewView(g)

	livePlanet := planetByID(t, g.Planets, 1)
	livePlanetShips := livePlanet.Ships
	liveOrbitAngle := livePlanet.Orbit.Angle
	secondPlanetShips := planetByID(t, second.Planets, 1).Ships
	secondOrbitAngle := planetByID(t, second.Planets, 1).Orbit.Angle
	liveCometShips := cometByID(t, g.Comets, 4).Ships
	liveFleetShips := fleetByID(t, g.Fleets, 8).Ships

	firstPlanet := planetByID(t, first.Planets, 1)
	if firstPlanet.Orbit == livePlanet.Orbit {
		t.Fatal("view planet shares Orbit pointer with live game")
	}
	if firstPlanet.Orbit == planetByID(t, second.Planets, 1).Orbit {
		t.Fatal("two views share a planet Orbit pointer")
	}

	first.Planets[0].Ships++
	first.Planets[0].Orbit.Angle++
	first.Comets[0].Ships++
	first.Fleets[0].Ships++
	first.Config.Width++

	if livePlanet.Ships != livePlanetShips || livePlanet.Orbit.Angle != liveOrbitAngle {
		t.Fatalf("live planet changed through view: ships=%d angle=%v", livePlanet.Ships, livePlanet.Orbit.Angle)
	}
	secondPlanet := planetByID(t, second.Planets, 1)
	if secondPlanet.Ships != secondPlanetShips || secondPlanet.Orbit.Angle != secondOrbitAngle {
		t.Fatalf("second view changed through first: ships=%d angle=%v", secondPlanet.Ships, secondPlanet.Orbit.Angle)
	}
	if got := cometByID(t, g.Comets, 4).Ships; got != liveCometShips {
		t.Fatalf("live comet ships = %d, want %d", got, liveCometShips)
	}
	if got := fleetByID(t, g.Fleets, 8).Ships; got != liveFleetShips {
		t.Fatalf("live fleet ships = %d, want %d", got, liveFleetShips)
	}
	if g.Config.Width == first.Config.Width {
		t.Fatal("view config mutation unexpectedly changed live config")
	}
}

func viewTestGame(t *testing.T) *game.Game {
	t.Helper()
	config := game.DefaultConfig()
	config.Width = 120
	config.Height = 90
	config.Sun = game.Vec2{X: 60, Y: 45}
	config.SunRadius = 8
	config.MaxTurns = 321
	config.MaxFleetSpeed = 7
	config.CometSpeed = 3
	config.Rand = rand.New(rand.NewSource(17))

	g, err := game.New(config, []game.Planet{
		{ID: 3, Position: game.Vec2{X: 90, Y: 60}, OwnerID: 0, Ships: 11, Production: 2},
		// 20 from the sun at angle 0.4, inside OrbitRadiusLimit, so it orbits.
		{ID: 1, OwnerID: 1, Ships: 13, Production: 3, Position: game.Vec2{X: 60 + 20*math.Cos(0.4), Y: 45 + 20*math.Sin(0.4)}},
	}, []game.Comet{
		{ID: 6, OwnerID: game.NeutralPlayer, Ships: 7, Path: []game.Vec2{{X: 80, Y: 20}, {X: 80, Y: 24}}},
		{ID: 4, OwnerID: 0, Ships: 5, Path: []game.Vec2{{X: 20, Y: 70}, {X: 20, Y: 74}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	g.Turn = 19
	g.Fleets = []game.Fleet{
		{ID: 9, OwnerID: 1, Ships: 4},
		{ID: 8, OwnerID: 0, Ships: 6},
	}
	return g
}

func planetIDs(planets []game.Planet) []int {
	ids := make([]int, len(planets))
	for i, planet := range planets {
		ids[i] = planet.ID
	}
	return ids
}

func cometIDs(comets []game.Comet) []int {
	ids := make([]int, len(comets))
	for i, comet := range comets {
		ids[i] = comet.ID
	}
	return ids
}

func fleetIDs(fleets []game.Fleet) []int {
	ids := make([]int, len(fleets))
	for i, fleet := range fleets {
		ids[i] = fleet.ID
	}
	return ids
}

func planetByID(t *testing.T, planets []game.Planet, id int) *game.Planet {
	t.Helper()
	for i := range planets {
		if planets[i].ID == id {
			return &planets[i]
		}
	}
	t.Fatalf("planet %d not found", id)
	return nil
}

func cometByID(t *testing.T, comets []game.Comet, id int) *game.Comet {
	t.Helper()
	for i := range comets {
		if comets[i].ID == id {
			return &comets[i]
		}
	}
	t.Fatalf("comet %d not found", id)
	return nil
}

func fleetByID(t *testing.T, fleets []game.Fleet, id int) *game.Fleet {
	t.Helper()
	for i := range fleets {
		if fleets[i].ID == id {
			return &fleets[i]
		}
	}
	t.Fatalf("fleet %d not found", id)
	return nil
}
