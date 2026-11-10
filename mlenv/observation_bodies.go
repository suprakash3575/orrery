package mlenv

import (
	"math"

	"orrery/game"
)

// obs-v2 normalization references. These are fixed by the schema, not
// configured per run, so they need no recording beyond the schema ID: obs-v2
// always means these values.
const (
	// shipReference is the ship count that maps to 1.0 under log1p. It is a
	// reference, not a cap: larger counts produce values above 1 rather than
	// being clipped, so the policy can still tell 2,000 ships from 20,000.
	shipReference = 1000.0
	// productionMax is the engine's highest planet production.
	productionMax = 5
)

// shipFeature compresses a ship count with log1p, so the difference between
// 5 and 50 ships stays visible next to the difference between 500 and 5,000.
func shipFeature(count int) float32 {
	if count <= 0 {
		return 0
	}
	return float32(math.Log1p(float64(count)) / math.Log1p(shipReference))
}

// encodeBodies fills body columns 0–15 for every real slot: position,
// size, garrison, owner role, orbit, and comet motion.
//
// It reads the game and frame and writes only obs. Columns 16–26 (incoming
// fleets and global facts) are filled by later stages, and padding rows are
// left all zero.
func encodeBodies(obs *Observation, g *game.Game, frame SeatFrame, slots slotTable) {
	cfg := g.Config
	radiusMax := game.PlanetRadius(productionMax)
	for slot, body := range slots.bodies {
		if !body.real() {
			continue
		}
		row := &obs.Bodies[slot]

		// Stage 1: the fields planets and comets share. Positions are turned
		// into the observer's frame before being scaled to -1..1, so every
		// seat sees its own home in the same corner.
		var position game.Vec2
		var radius float64
		var owner, ships, production int
		if body.planet != nil {
			p := body.planet
			position, radius, owner, ships, production = p.Position, p.Radius, p.OwnerID, p.Ships, p.Production
		} else {
			c := body.comet
			position, radius, owner, ships, production = c.Position, c.Radius, c.OwnerID, c.Ships, c.Production
		}
		turned := frame.Position(position)
		row[FeatureX] = float32(centred(turned.X, cfg.Width))
		row[FeatureY] = float32(centred(turned.Y, cfg.Height))
		row[FeatureRadius] = float32(radius / radiusMax)
		row[FeatureProduction] = float32(production) / productionMax
		row[FeatureShips] = shipFeature(ships)

		// Stage 2: owner one-hot. The four role columns are laid out in role
		// order, so a seated owner's column is the first one plus its role.
		// buildSlotTable has already rejected owners outside the roster, so a
		// non-neutral owner always has a role here.
		if role, ok := frame.RoleOf(owner); ok {
			row[FeatureOwnerMe+int(role)] = 1
		} else {
			row[FeatureOwnerNeutral] = 1
		}

		// Stage 3: kind-specific motion. A planet's motion is fully described
		// by its orbit rate, because its position and the sun give the rest.
		// A comet follows a precomputed path, so it needs its own direction:
		// without history, a single position cannot tell an approaching comet
		// from one moving away.
		if p := body.planet; p != nil {
			if p.Orbit != nil {
				row[FeatureIsOrbiting] = 1
				// A quarter-turn of the whole board keeps every orbit turning
				// the same way, so the sign needs no adjustment.
				row[FeatureOrbitSpeed] = float32(p.Orbit.RadiansTurn / game.MaxOrbitRate)
			}
			continue
		}
		c := body.comet
		row[FeatureIsComet] = 1
		// Points left counts the current one, so a comet on its last point
		// has 1/40 left, not 0: it is still on the board this turn.
		row[FeatureCometTurnsLeft] = float32(len(c.Path)-c.PathIndex) / game.CometMaxPathPoints
		if next := c.PathIndex + 1; next < len(c.Path) {
			// A step is a displacement, so it turns with the board but is not
			// moved about the sun. Spawned comets space points CometSpeed
			// apart, so this is close to a unit vector.
			step := frame.Vector(c.Path[next].Sub(c.Path[c.PathIndex]))
			row[FeatureCometDX] = float32(step.X / cfg.CometSpeed)
			row[FeatureCometDY] = float32(step.Y / cfg.CometSpeed)
		}
	}
}
