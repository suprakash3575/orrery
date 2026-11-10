package agent

import (
	"sort"

	"orrery/game"
)

// ConfigView is the decision-relevant, read-only portion of game.Config. It
// intentionally has no Rand field: an agent cannot consume the live engine
// RNG merely by receiving a View.
type ConfigView struct {
	Width, Height        float64
	Sun                  game.Vec2
	SunRadius            float64
	SunHazardEnabled     bool
	CometSpawningEnabled bool
	MaxTurns             int
	MaxFleetSpeed        float64
	// AngularVelocity is the game's shared orbit rate (Game.OrbitRate) in
	// radians per turn, 0 when orbits are off. It is named after the reference
	// engine's angular_velocity observation field, which carries the same value.
	AngularVelocity float64
	CometSpeed      float64
}

// View is an isolated snapshot of the public game state at the start of one
// turn. Its slices and the Orbit values reachable through its planets are
// independent copies. Agents may modify their own View without changing the
// live game or the View given to another seat, with one exception: each
// comet's Path is shared with the live game and must be treated as read-only.
type View struct {
	Turn    int
	Config  ConfigView
	Planets []game.Planet
	Comets  []game.Comet
	Fleets  []game.Fleet
}

// NewView copies the state an agent may inspect and normalizes entity order by
// ID. Stable order prevents a caller's slice layout from changing an otherwise
// deterministic agent decision, seed, record, or test.
func NewView(g *game.Game) View {
	planets := append([]game.Planet(nil), g.Planets...)
	for i := range planets {
		if planets[i].Orbit != nil {
			orbit := *planets[i].Orbit
			planets[i].Orbit = &orbit
		}
	}
	sort.Slice(planets, func(i, j int) bool { return planets[i].ID < planets[j].ID })

	// Copying a Comet copies only its Path slice header, so every view shares
	// the live game's points. That is safe because the engine never changes a
	// path once built, and it saves copying every comet path for every seat on
	// every turn. Agents must therefore treat Path as read-only.
	comets := append([]game.Comet(nil), g.Comets...)
	sort.Slice(comets, func(i, j int) bool { return comets[i].ID < comets[j].ID })

	fleets := append([]game.Fleet(nil), g.Fleets...)
	sort.Slice(fleets, func(i, j int) bool { return fleets[i].ID < fleets[j].ID })

	return View{
		Turn: g.Turn,
		Config: ConfigView{
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
		},
		Planets: planets,
		Comets:  comets,
		Fleets:  fleets,
	}
}

// Snapshot returns the board in the form the engine's prediction helpers take
// (see game.NewPredictor). It copies fields one to one and does no physics, so
// a bot can predict from its observation alone. The slices are shared with the
// View; game.NewPredictor makes its own copy.
func (v View) Snapshot() game.Snapshot {
	return game.Snapshot{
		Turn:             v.Turn,
		OrbitRate:        v.Config.AngularVelocity,
		Width:            v.Config.Width,
		Height:           v.Config.Height,
		Sun:              v.Config.Sun,
		SunRadius:        v.Config.SunRadius,
		SunHazardEnabled: v.Config.SunHazardEnabled,
		MaxFleetSpeed:    v.Config.MaxFleetSpeed,
		Planets:          v.Planets,
		Comets:           v.Comets,
		Fleets:           v.Fleets,
	}
}

// FleetSpeed evaluates the authoritative engine curve using the maximum speed
// copied into this View.
func (v View) FleetSpeed(ships int) float64 {
	return game.FleetSpeed(ships, v.Config.MaxFleetSpeed)
}
