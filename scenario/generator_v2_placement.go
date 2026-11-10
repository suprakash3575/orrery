package scenario

import (
	"math"

	"orrery/game"
)

const (
	// Clearance is measured between planet surfaces, not centres. The sun
	// uses a separate, wider gap from every planet's surface.
	v2PlanetClearance = 7.0
	v2SunClearance    = 10.0
)

// v2LegalGroup checks all four copies before any is added to the map. That
// atomic decision preserves rotational symmetry when one copy is blocked.
// Checking the candidate members against each other also protects the group
// if a later sampler proposes positions closer to the symmetry axes.
func v2LegalGroup(config game.Config, placed []game.Planet, group [4]game.Planet) bool {
	for i, candidate := range group {
		r := candidate.Radius
		p := candidate.Position
		if p.X-r < 0 || p.X+r > config.Width || p.Y-r < 0 || p.Y+r > config.Height {
			return false
		}
		if p.Distance(config.Sun) < config.SunRadius+r+v2SunClearance {
			return false
		}
		for _, existing := range placed {
			if !v2PairClear(config, candidate, existing) {
				return false
			}
		}
		for j := range i {
			if !v2PairClear(config, candidate, group[j]) {
				return false
			}
		}
	}
	return true
}

// v2PairClear first checks the visible start positions. When exactly one
// planet orbits, its full circle will eventually pass closest to the static
// planet at the difference between their distances from the sun; that radial
// gap must also clear both radii and seven empty units. Two orbiting
// planets share one angular rate, so their relative positions stay fixed.
func v2PairClear(config game.Config, a, b game.Planet) bool {
	gap := a.Radius + b.Radius + v2PlanetClearance
	if a.Position.Distance(b.Position) < gap {
		return false
	}
	aOrbitRadius := a.Position.Distance(config.Sun)
	bOrbitRadius := b.Position.Distance(config.Sun)
	limit := v2OrbitLimit(config)
	aOrbits := config.OrbitsEnabled && aOrbitRadius+a.Radius < limit
	bOrbits := config.OrbitsEnabled && bOrbitRadius+b.Radius < limit
	return aOrbits == bOrbits || math.Abs(aOrbitRadius-bOrbitRadius) >= gap
}
