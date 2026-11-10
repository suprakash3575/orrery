package game

import "math"

// expireComets drops every comet that has used up its path. A comet's ships
// are a loan against it staying on the board, so they go with it.
//
// AdvanceTurn calls this twice. The main call comes after fleet movement and
// before combat: a comet that steps past its last point this turn has still
// launched, produced, and blocked fleets, but leaves before combat and
// scoring, as in the reference engine, so its ships cannot decide a result for an extra
// turn. A second call at the start of the turn, before launches are
// validated, removes any comet placed in that state by hand, so nobody can
// launch from it.
//
// It reports whether it removed any comet, so AdvanceTurn only filters
// queued arrivals on the rare turns when one actually left.
func (g *Game) expireComets() bool {
	alive := g.Comets[:0]
	for _, c := range g.Comets {
		if c.PathIndex >= len(c.Path) {
			continue
		}
		alive = append(alive, c)
	}
	removed := len(alive) < len(g.Comets)
	g.Comets = alive
	return removed
}

// CometMaxPathPoints is the most points a spawned comet's path can have, so
// the most turns it stays on the board. It is exported because observations
// scale a comet's remaining lifetime by it; a copy of the number elsewhere
// would silently go stale if spawning changed. Comets placed by hand are not
// bound by it.
const CometMaxPathPoints = 40

// cometSpawnTurns lists the turns at which a wave of four comets appears.
var cometSpawnTurns = [...]int{50, 150, 250, 350, 450}

// Comet path constants from the reference engine's generate_comet_paths. A comet's ellipse has the sun at one focus, so the
// comet swings in past the sun and back out rather than circling the board.
const (
	// cometMinEccentricity and cometMaxEccentricity bound how elongated the
	// ellipse is: close to 1 is a long, thin ellipse.
	cometMinEccentricity = 0.75
	cometMaxEccentricity = 0.93
	// cometMinSemiMajor and cometMaxSemiMajor bound the ellipse's semi-major
	// axis, in board units. Most of an ellipse this large lies off the board.
	cometMinSemiMajor = 60.0
	cometMaxSemiMajor = 150.0
	// cometMinPathPoints and CometMaxPathPoints bound how many turns the
	// visible stretch lasts, so every comet is a brief visit.
	cometMinPathPoints = 5
	// cometPathAttempts is how many ellipses spawning tries before giving up
	// on a wave, and cometDenseSamples is how finely each ellipse is sampled
	// before points are picked out at Config.CometSpeed spacing.
	cometPathAttempts = 300
	cometDenseSamples = 5000
	// cometStaticClearance and cometOrbitClearance are how far, beyond a
	// planet's radius, every comet path point must stay from it: 1.5 for a
	// stationary planet and 1 (the comet's own radius) for an orbiting one,
	// as in the reference engine. Without them a comet could overlap a planet, and the
	// lower-ID planet would take hits meant for the comet.
	cometStaticClearance = 1.5
	cometOrbitClearance  = 1.0
)

// NextCometWave reports how many AdvanceTurn calls will run before the next
// scheduled comet wave is due: 0 means the very next call spawns it. It
// reports false when spawning is disabled or no wave is left before the turn
// limit ends the game.
//
// It reads the schedule only. A wave that is due can still be skipped when no
// usable path is found, and the paths themselves come from random draws that
// have not happened yet, so nothing here can say what a wave will contain.
func (g *Game) NextCometWave() (int, bool) {
	if !g.Config.CometSpawningEnabled {
		return 0, false
	}
	// spawnComets fires when the turn being processed equals a scheduled
	// turn, and the game is over once Turn reaches MaxTurns, so a wave still
	// lies ahead exactly when Turn <= turn < MaxTurns. cometSpawnTurns is in
	// ascending order, so the first match is the next wave.
	for _, turn := range cometSpawnTurns {
		if turn >= g.Turn && turn < g.Config.MaxTurns {
			return turn - g.Turn, true
		}
	}
	return 0, false
}

// spawnComets checks whether this turn is scheduled to receive a wave of
// comets and, if so, adds one: four comets, one per quadrant, sharing one
// path shape and ship count. If no usable path is found the wave is skipped,
// as the reference engine does.
//
// Each comet starts at PathIndex -1 with its Position on the first path
// point. moveComets runs later this same turn and advances it to point 0, so
// a newly spawned comet ends its first turn on its first point, matching
// the reference engine. It returns the new comets' IDs so AdvanceTurn can leave them out of
// that turn's collisions, as the reference engine does for a comet's first placement.
//
// Spawned comets start neutral: a comet is an environmental event, not a
// grant to any player, and stays unclaimed until someone's fleet reaches it.
func (g *Game) spawnComets() []int {
	due := false
	for _, turn := range cometSpawnTurns {
		if g.Turn == turn {
			due = true
			break
		}
	}
	if !due {
		return nil
	}

	paths := g.cometPaths()
	if paths == nil {
		return nil
	}

	// The minimum of four rolls in [1,99] is what skews the shared count low:
	// a batch needs all four rolls to land high to itself be strong, which is
	// far less likely than at least one landing low, without reaching for a
	// non-uniform distribution to get the same skew.
	ships := 99
	for range 4 {
		if roll := 1 + g.Config.Rand.Intn(99); roll < ships {
			ships = roll
		}
	}

	ids := make([]int, 0, len(paths))
	for _, path := range paths {
		g.nextID++
		ids = append(ids, g.nextID)
		g.Comets = append(g.Comets, Comet{
			ID:         g.nextID,
			OwnerID:    NeutralPlayer,
			Ships:      ships,
			Radius:     1,
			Production: 1,
			Position:   path[0],
			Path:       path,
			PathIndex:  -1,
		})
	}
	return ids
}

// cometPaths builds the four symmetric paths for one comet wave, following
// the reference engine's generate_comet_paths, or returns nil if none of
// cometPathAttempts tries gives a usable path.
//
// Each try:
//  1. Rolls an ellipse with the sun at one focus. One whose closest approach
//     (perihelion) would touch the sun is rejected, so a comet can never
//     enter the sun. So is one too short to hold a usable path at all (see
//     cometPathCannotFit), before any sampling work is done.
//  2. Samples it densely, then keeps one point every Config.CometSpeed units
//     of distance, so the comet moves at that speed.
//  3. Keeps only the stretch from the first to the last point on the board,
//     and accepts it only if that lasts cometMinPathPoints to
//     CometMaxPathPoints turns.
//  4. Builds the four rotated copies and rejects the try if any copy leaves
//     the board where the original is on it, or passes too close to a
//     planet (see cometPathTouchesPlanet).
//
// The four copies are exact 90-degree rotations of each other about the sun,
// built from the reference engine's formulas, so every quadrant gets the same comet. They
// are only guaranteed to fit the board when it is square with the sun at its
// centre, as the standard board is, which is why step 4 checks them.
func (g *Game) cometPaths() [][]Vec2 {
	sun := g.Config.Sun
	rng := g.Config.Rand
	// Even the largest ellipse, sampled over its whole range, cannot hold a
	// usable path at this speed, so no try can succeed: skip the wave without
	// drawing or sampling anything.
	if cometPathCannotFit(cometMaxSemiMajor*(cometArcEnd-cometArcStart), g.Config.CometSpeed) {
		return nil
	}
	// reach is the farthest any board point lies from the sun. A point farther
	// than reach can never be on the board, so sampling covers only the
	// stretch of each ellipse within reach of the sun.
	reach := 0.0
	for _, corner := range [4]Vec2{{0, 0}, {g.Config.Width, 0}, {0, g.Config.Height}, {g.Config.Width, g.Config.Height}} {
		reach = max(reach, corner.Distance(sun))
	}
	// path is reused across tries; each try overwrites it.
	var path []Vec2
	for range cometPathAttempts {
		e := cometMinEccentricity + (cometMaxEccentricity-cometMinEccentricity)*rng.Float64()
		a := cometMinSemiMajor + (cometMaxSemiMajor-cometMinSemiMajor)*rng.Float64()
		tStart, tEnd := cometReachableArc(a, e, reach)
		if a*(1-e) < g.Config.SunRadius+1 || cometPathCannotFit(a*(tEnd-tStart), g.Config.CometSpeed) {
			continue
		}
		b := a * math.Sqrt(1-e*e)
		focusOffset := a * e
		// phi turns the ellipse so its closest approach points into one
		// quadrant, between 30 and 60 degrees.
		phi := math.Pi/6 + (math.Pi/6)*rng.Float64()
		cos, sin := math.Cos(phi), math.Sin(phi)

		// Walk the stretch of the ellipse that can reach the board in fine
		// samples, keeping one point each time the distance travelled passes
		// the next multiple of CometSpeed. Placing the ellipse's centre
		// focusOffset along its axis puts one focus exactly on the sun. first
		// and last track the first and last kept points on the board, since
		// only that stretch is used.
		samples := 1 + int(math.Ceil((tEnd-tStart)/cometSampleStep))
		path = path[:0]
		first, last := -1, -1
		tooLong := false
		var previous Vec2
		travelled, target := 0.0, g.Config.CometSpeed
		for i := range samples {
			t := min(tStart+cometSampleStep*float64(i), tEnd)
			ex := focusOffset + a*math.Cos(t)
			ey := b * math.Sin(t)
			sample := Vec2{sun.X + ex*cos - ey*sin, sun.Y + ex*sin + ey*cos}
			keep := i == 0
			if i > 0 {
				// A plain square root rather than Vec2.Distance: Distance uses
				// math.Hypot, which guards against overflow that cannot occur
				// for board-sized coordinates and costs noticeably more in this
				// hot loop.
				dx, dy := sample.X-previous.X, sample.Y-previous.Y
				travelled += math.Sqrt(dx*dx + dy*dy)
				if travelled >= target {
					keep = true
					target += g.Config.CometSpeed
				}
			}
			previous = sample
			if keep {
				path = append(path, sample)
				if g.inBounds(sample) {
					if first < 0 {
						first = len(path) - 1
					}
					last = len(path) - 1
					// The on-board stretch already holds too many points,
					// so this try is rejected whatever comes later.
					if last-first+1 > CometMaxPathPoints {
						tooLong = true
						break
					}
				}
			}
		}
		if tooLong || first < 0 {
			continue
		}
		visible := path[first : last+1]
		if len(visible) < cometMinPathPoints {
			continue
		}

		// The reference engine's four copies, written relative to the sun. With
		// (dx, dy) = point - sun, they are (dy, dx), (-dx, dy), (dx, -dy) and
		// (-dy, -dx): 90-degree rotations of one another about the sun, so
		// every quadrant gets one copy.
		paths := make([][]Vec2, 4)
		for k := range paths {
			paths[k] = make([]Vec2, 0, len(visible))
		}
		fits := true
		for _, point := range visible {
			dx, dy := point.X-sun.X, point.Y-sun.Y
			copies := [4]Vec2{
				{sun.X + dy, sun.Y + dx},
				{sun.X - dx, sun.Y + dy},
				{sun.X + dx, sun.Y - dy},
				{sun.X - dy, sun.Y - dx},
			}
			for k, copied := range copies {
				// On the standard board every copy is on the board exactly
				// where the original is. Checking only those points keeps
				// the reference engine's behaviour there while catching a non-square or
				// off-centre board, where a rotated copy can fall off.
				if g.inBounds(point) && !g.inBounds(copied) {
					fits = false
				}
				paths[k] = append(paths[k], copied)
			}
		}
		if !fits || g.cometPathTouchesPlanet(paths) {
			continue
		}
		return paths
	}
	return nil
}

// cometArcStart and cometArcEnd bound the ellipse parameter t sampled for a
// comet: the half of the ellipse nearest the sun, centred on its closest
// approach at t = pi. cometSampleStep is the spacing between samples, the
// same density as cometDenseSamples spread over that whole range.
const (
	cometArcStart   = 0.3 * math.Pi
	cometArcEnd     = 1.7 * math.Pi
	cometSampleStep = (cometArcEnd - cometArcStart) / (cometDenseSamples - 1)
)

// cometReachableArc returns the range of t, within [cometArcStart,
// cometArcEnd], where an ellipse with semi-major axis a and eccentricity e
// comes no farther than reach from the sun. Points outside it are farther
// from the sun than any board point, so they can never be on the board and
// need not be sampled.
//
// A point's distance from the sun (the focus) is a*(1 + e*cos t): smallest at
// t = pi and growing steadily away from it on both sides. So the points within
// reach are one stretch around pi, where cos t <= (reach/a - 1)/e.
func cometReachableArc(a, e, reach float64) (float64, float64) {
	limit := (reach/a - 1) / e
	if limit >= 1 {
		// Every point of the ellipse is within reach.
		return cometArcStart, cometArcEnd
	}
	// limit > -1 always holds here: the closest approach a*(1-e) is never
	// farther than reach on a board the sun sits on.
	half := math.Pi - math.Acos(max(limit, -1))
	return max(cometArcStart, math.Pi-half), min(cometArcEnd, math.Pi+half)
}

// cometPathCannotFit reports whether a stretch of ellipse at most maxLength
// long is too short to hold a usable path at speed, so a try can be skipped
// before any sampling. A usable path needs cometMinPathPoints points, which
// span at least (cometMinPathPoints-1) steps of about speed each along the
// curve. The 1% allowance covers a step that falls slightly short of speed
// because points are picked from discrete samples, so the check never
// rejects an ellipse that could have fitted.
//
// Callers bound the length cheaply: a point on the ellipse moves
// sqrt(a^2*sin^2 t + b^2*cos^2 t) per unit of t, never more than a, so a
// stretch spanning a range of t is at most a times that range. Every
// on-board point lies within the reachable stretch (cometReachableArc), so
// that stretch's bound covers the whole usable path.
func cometPathCannotFit(maxLength, speed float64) bool {
	return float64(cometMinPathPoints-1)*speed > 1.01*maxLength
}

// cometPathTouchesPlanet reports whether any of a wave's four paths passes too
// close to a planet, following the reference engine's checks. A stationary planet must stay
// cometStaticClearance beyond its radius from every point. An orbiting planet
// moves, so each point is checked against where that planet will be on the
// turn the comet reaches it: point k is reached at the end of turn
// g.Turn + k, when the planet has completed g.Turn+1+k turns of its orbit.
//
// All four paths have the same length, so the planet's position for point k
// is computed once and checked against all four copies of that point.
func (g *Game) cometPathTouchesPlanet(paths [][]Vec2) bool {
	for _, p := range g.Planets {
		for k := range paths[0] {
			position, clearance := p.Position, cometStaticClearance
			if p.Orbit != nil {
				position = g.orbitPoint(p.Orbit, orbitAngleAfter(p.Orbit, g.Turn+1+k))
				clearance = cometOrbitClearance
			}
			for _, path := range paths {
				if path[k].Distance(position) < p.Radius+clearance {
					return true
				}
			}
		}
	}
	return false
}
