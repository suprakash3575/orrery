// Package arena coordinates complete Orrery matches without duplicating
// any rules from the game engine.
package arena

import (
	"math/rand"

	"orrery/game"
)

// ScenarioFactory constructs independent games from reproducible scenario
// seeds. Future generated layouts can implement this same boundary without
// changing the match runner or any agent.
type ScenarioFactory interface {
	ID() string
	New(seed int64) (*game.Game, error)
}

// FixedScenarioV1 identifies the first explicit two-player arena layout.
const FixedScenarioV1 = "fixed-v1"

// FixedScenarioFactory builds the same symmetric starting layout for every
// seed. The seed still controls rule-driven randomness such as scheduled comet
// spawns, so replaying a match requires both this factory version and its seed.
type FixedScenarioFactory struct{}

// ID returns the versioned fixed-scenario identity.
func (FixedScenarioFactory) ID() string { return FixedScenarioV1 }

// New returns an independent standard game with a fresh seeded engine RNG.
// The layout intentionally starts without comets; AdvanceTurn retains the
// standard engine's scheduled comet spawning behavior.
func (FixedScenarioFactory) New(seed int64) (*game.Game, error) {
	config := game.DefaultConfig()
	// fixed-v1 has always been a static layout. Every planet here sits inside
	// game.OrbitRadiusLimit, so keeping orbits off is what preserves that
	// layout now that the engine decides orbiting from position.
	config.OrbitsEnabled = false
	config.Rand = rand.New(rand.NewSource(seed))
	return game.New(config, []game.Planet{
		{ID: 1, Position: game.Vec2{X: 15, Y: 50}, OwnerID: 0, Ships: 20, Production: 3},
		{ID: 2, Position: game.Vec2{X: 85, Y: 50}, OwnerID: 1, Ships: 20, Production: 3},
		{ID: 3, Position: game.Vec2{X: 30, Y: 25}, OwnerID: game.NeutralPlayer, Ships: 8, Production: 2},
		{ID: 4, Position: game.Vec2{X: 70, Y: 75}, OwnerID: game.NeutralPlayer, Ships: 8, Production: 2},
		{ID: 5, Position: game.Vec2{X: 30, Y: 75}, OwnerID: game.NeutralPlayer, Ships: 12, Production: 4},
		{ID: 6, Position: game.Vec2{X: 70, Y: 25}, OwnerID: game.NeutralPlayer, Ships: 12, Production: 4},
	}, nil)
}
