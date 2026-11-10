package mlenv

import (
	"math"

	"orrery/game"
)

// spawnWaitScale divides the turns until the next comet wave. It is one more
// than the longest gap the schedule has (100 turns between waves), so every
// real wait stays below the 1 that means "no wave ahead". Like the ship
// reference, it is fixed by the schema rather than configured per run.
const spawnWaitScale = 101.0

// encodeGlobals fills body columns 24–26 with facts about the whole board:
// how far the game has run, when the next comet wave is due, and how large
// the sun hazard is.
//
// They are copied into every real body row rather than given a row of their
// own, so the model sees them whichever body it is attending to. Padding rows
// stay zero.
func encodeGlobals(obs *Observation, g *game.Game, slots slotTable) {
	cfg := g.Config

	// game.New rejects MaxTurns <= 0, so the division is safe.
	turnProgress := float32(float64(g.Turn) / float64(cfg.MaxTurns))

	// The schedule comes from the engine, so the observation never keeps its
	// own copy of the spawn turns, and it never looks at random draws.
	nextSpawn := float32(noArrival)
	if wait, ok := g.NextCometWave(); ok {
		nextSpawn = float32(math.Min(float64(wait)/spawnWaitScale, 1))
	}

	// The sun only matters when it destroys fleets. With the hazard off,
	// fleets fly straight through it, so its size carries nothing to act on
	// and is reported as 0. The seat frame requires a square board, so either
	// side gives the same half-width.
	sunRadius := float32(0)
	if cfg.SunHazardEnabled {
		sunRadius = float32(cfg.SunRadius / (math.Min(cfg.Width, cfg.Height) / 2))
	}

	for slot, body := range slots.bodies {
		if !body.real() {
			continue
		}
		row := &obs.Bodies[slot]
		row[FeatureTurnProgress] = turnProgress
		row[FeatureNextCometSpawn] = nextSpawn
		row[FeatureSunRadius] = sunRadius
	}
}
