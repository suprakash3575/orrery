package scenario

import (
	"fmt"
	"math"
	"math/rand"

	"orrery/game"
)

const (
	v2MinGroups            = 5
	v2MaxGroups            = 10
	v2MinStaticGroups      = 3
	v2MaxAttempts          = 5000
	v2MaxMapAttempts       = 20
	v2NarrowRegionAttempts = 500
)

// generateV2Groups builds the neutral groups before home assignment. Every
// choice comes from the caller's map RNG, never the engine's runtime stream.
//
// The first three groups use an outer polar sampler and are static. The
// remaining groups use a rectangular sampler, with the final slot reserved for
// an orbiting group if none appeared earlier. A group is accepted atomically
// only after all four copies clear the sun, board, and earlier groups.
func generateV2Groups(rng *rand.Rand, config game.Config) ([]game.Planet, error) {
	groups := v2MinGroups + rng.Intn(v2MaxGroups-v2MinGroups+1)
	// Early legal choices can leave no room for a later group. Start the
	// whole layout over with the same private RNG stream rather than returning
	// fewer planets than the chosen count or retrying a blocked slot forever.
	var lastErr error
	for range v2MaxMapAttempts {
		planets, err := tryV2Groups(rng, config, groups)
		if err == nil {
			return planets, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("cannot place %d groups in %d map attempts: %w", groups, v2MaxMapAttempts, lastErr)
}

// tryV2Groups keeps one layout attempt local. A failed group discards the
// entire slice, so the next attempt starts with no partial bodies or IDs.
func tryV2Groups(rng *rand.Rand, config game.Config, groups int) ([]game.Planet, error) {
	planets := make([]game.Planet, 0, 4*groups)
	for groupIndex := range v2MinStaticGroups {
		group, err := sampleV2StaticGroup(rng, config, planets, len(planets)+1)
		if err != nil {
			return nil, fmt.Errorf("static group %d: %w", groupIndex, err)
		}
		planets = append(planets, group[:]...)
	}

	hasOrbiting := false
	for groupIndex := v2MinStaticGroups; groupIndex < groups; groupIndex++ {
		// Reserving the last slot enforces the orbit quota without allowing an
		// extra group beyond the chosen count.
		requireOrbit := groupIndex == groups-1 && !hasOrbiting
		group, orbiting, err := sampleV2OrdinaryGroup(rng, config, planets, len(planets)+1, requireOrbit)
		if err != nil {
			return nil, fmt.Errorf("group %d: %w", groupIndex, err)
		}
		planets = append(planets, group[:]...)
		hasOrbiting = hasOrbiting || orbiting
	}
	return planets, nil
}

// v2OrbitLimit duplicates the engine's effective boundary on the standard
// board: a planet at distance d orbits exactly when d + its radius < limit.
// The generated game's orbit assignment is checked against this in tests.
func v2OrbitLimit(config game.Config) float64 {
	sun := config.Sun
	return min(game.OrbitRadiusLimit, sun.X, config.Width-sun.X, sun.Y, config.Height-sun.Y)
}

// sampleV2StaticGroup uses polar coordinates. Sampling at or beyond
// limit-radius makes the entire group static; the axis gap keeps its four
// rotated copies apart. The attempt bound turns an impossible board into an
// error instead of an unbounded retry loop.
func sampleV2StaticGroup(rng *rand.Rand, config game.Config, placed []game.Planet, firstID int) ([4]game.Planet, error) {
	sun := config.Sun
	limit := v2OrbitLimit(config)
	for range v2MaxAttempts {
		production := 1 + rng.Intn(5)
		radius := game.PlanetRadius(production)
		angle := rng.Float64() * math.Pi / 2
		cos, sin := math.Cos(angle), math.Sin(angle)
		minOrbital := limit - radius
		// Divide the available horizontal and vertical reach by the angle's
		// components. The tighter bound keeps the whole planet on the board;
		// sin can be zero when the sampled angle lies exactly on the X axis.
		maxOrbital := (config.Width - sun.X - radius) / cos
		if sin > 0 {
			maxOrbital = min(maxOrbital, (config.Height-sun.Y-radius)/sin)
		}
		if minOrbital > maxOrbital {
			continue
		}
		orbital := minOrbital + rng.Float64()*(maxOrbital-minOrbital)
		x, y := sun.X+orbital*cos, sun.Y+orbital*sin
		if x-sun.X < radius+5 || y-sun.Y < radius+5 {
			continue
		}
		// Taking the lower of two rolls gives the guaranteed static groups
		// a wider but low-skewed 5..99 neutral garrison distribution.
		ships := min(5+rng.Intn(95), 5+rng.Intn(95))
		group := v2PlanetGroup(sun, x, y, production, ships, firstID)
		if v2LegalGroup(config, placed, group) {
			return group, nil
		}
	}
	return [4]game.Planet{}, fmt.Errorf("no static candidate in %d attempts", v2MaxAttempts)
}

// sampleV2OrdinaryGroup starts with a rectangular first-quadrant
// sample and its 5..30 neutral-ship range. If a crowded map blocks that
// narrow region, it searches closer to the axes while keeping the same
// spacing rules. The optional orbit requirement applies only to the last
// available group slot when earlier groups were all static.
func sampleV2OrdinaryGroup(rng *rand.Rand, config game.Config, placed []game.Planet, firstID int, requireOrbit bool) ([4]game.Planet, bool, error) {
	sun := config.Sun
	limit := v2OrbitLimit(config)
	for attempt := range v2MaxAttempts {
		production := 1 + rng.Intn(5)
		radius := game.PlanetRadius(production)
		minX, minY := sun.X+15, sun.Y+15
		if attempt >= v2NarrowRegionAttempts {
			// A quarter-turn copy can meet this candidate across either axis.
			// Half the required centre gap on each side keeps them clear.
			axisOffset := radius + v2PlanetClearance/2
			minX, minY = sun.X+axisOffset, sun.Y+axisOffset
		}
		maxX, maxY := config.Width-radius-5, config.Height-radius-5
		if maxX <= minX || maxY <= minY {
			break
		}
		x := minX + rng.Float64()*(maxX-minX)
		y := minY + rng.Float64()*(maxY-minY)
		orbital := math.Hypot(x-sun.X, y-sun.Y)
		if orbital < config.SunRadius+radius+v2SunClearance {
			continue
		}
		orbiting := orbital+radius < limit
		if requireOrbit && !orbiting {
			continue
		}
		ships := 5 + rng.Intn(26)
		group := v2PlanetGroup(sun, x, y, production, ships, firstID)
		if v2LegalGroup(config, placed, group) {
			return group, orbiting, nil
		}
	}
	return [4]game.Planet{}, false, fmt.Errorf("no ordinary candidate in %d attempts", v2MaxAttempts)
}

// v2PlanetGroup uses this group order: quadrant one, a quarter-turn
// clockwise, a quarter-turn counterclockwise, then the opposite planet. That
// last member is the other player's home when a group is selected later.
// IDs start at one because game.New requires positive planet IDs.
func v2PlanetGroup(sun game.Vec2, x, y float64, production, ships, firstID int) [4]game.Planet {
	positions := [...]game.Vec2{
		{X: y, Y: x},
		{X: 2*sun.X - x, Y: y},
		{X: x, Y: 2*sun.Y - y},
		{X: 2*sun.X - y, Y: 2*sun.Y - x},
	}
	var group [4]game.Planet
	for i, position := range positions {
		group[i] = game.Planet{
			ID:         firstID + i,
			Position:   position,
			Radius:     game.PlanetRadius(production),
			OwnerID:    game.NeutralPlayer,
			Ships:      ships,
			Production: production,
		}
	}
	return group
}
