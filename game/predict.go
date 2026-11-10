package game

import (
	"fmt"
	"math"
)

// HitKind names what a straight path ran into.
//
// The distinction that matters to a caller is arrival versus death: a planet or
// comet captures the fleet and hands its ships to that body, while the boundary
// and the sun destroy it outright. IsBody answers that question directly, so
// callers rarely need to switch on the kind itself.
type HitKind int

const (
	// HitBoundary means the path left the board, which destroys the fleet.
	HitBoundary HitKind = iota
	// HitSun means the path crossed the sun, which destroys the fleet. It can
	// only occur when Config.SunHazardEnabled is set; a game with the hazard
	// disabled never reports it.
	HitSun
	// HitPlanet means the path reached a planet: an arrival, not a death.
	HitPlanet
	// HitComet means the path reached a comet: likewise an arrival.
	HitComet
)

// String renders the kind for test failures and diagnostics.
func (k HitKind) String() string {
	switch k {
	case HitBoundary:
		return "boundary"
	case HitSun:
		return "sun"
	case HitPlanet:
		return "planet"
	case HitComet:
		return "comet"
	default:
		return fmt.Sprintf("HitKind(%d)", int(k))
	}
}

// Hit describes the first thing a straight path strikes.
type Hit struct {
	Kind HitKind
	// BodyID is the planet or comet struck, and 0 when Kind is HitBoundary or
	// HitSun. Body IDs are always positive, so 0 unambiguously means "no body"
	// rather than being a real identifier a caller might act on by mistake.
	BodyID int
	// T is how far along the segment the strike happened: 0 is the start and 1
	// is the end. A caller following a fleet turn by turn adds T to the number
	// of whole turns already travelled to get an arrival time.
	T float64
}

// IsBody reports whether the path reached a planet or comet — an arrival that
// hands the fleet's ships to that body — rather than a hazard that destroys it.
func (h Hit) IsBody() bool { return h.Kind == HitPlanet || h.Kind == HitComet }

// FirstHit reports what a fleet travelling start→end strikes first, honouring
// the same hazard set, ordering, and exclusion rules the engine itself applies
// when it moves fleets. It returns false when the path is clear.
//
// This exists so that code outside the engine — an observation builder deciding
// which body an in-flight fleet is heading for, or an agent judging whether a
// launch is reachable — can ask the authoritative question rather than
// reimplementing collision geometry and drifting from it. It delegates to the
// same unexported firstCollision the turn loop uses, so there is exactly one
// implementation of these rules and no second copy that could disagree.
//
// Limitation: FirstHit tests every body where it currently sits, as if no body
// moved during the segment. It therefore matches AdvanceTurn exactly only on a
// board where no body moves: no orbiting planets (every Planet.Orbit is nil, as
// with Config.OrbitsEnabled false) and no comets in play or due to spawn. The
// engine tests a fleet against each moving body's path over the same turn, so
// with orbits or comets the two can disagree — the engine can claim a fleet
// FirstHit reports as clear, and the reverse. To follow fleets on such a
// board, use PredictFleets, which moves the bodies turn by turn.
//
// excludeBody names one planet or comet to skip, or 0 for none. The engine uses
// it to stop a fleet arriving back at the source it launched from on the turn it
// left; a caller predicting that same turn must pass the same exclusion, or it
// will predict an arrival the engine will not produce.
func (g *Game) FirstHit(start, end Vec2, excludeBody int) (Hit, bool) {
	// nil: a prediction treats every body as standing still for the segment.
	found, ok := g.firstCollision(start, end, excludeBody, turnMotion{})
	if !ok {
		return Hit{}, false
	}
	return publicHit(found), true
}

// publicHit converts the engine's internal collision into the public Hit.
func publicHit(found collision) Hit {
	hit := Hit{Kind: hitKind(found.kind), T: found.t}
	// The engine leaves id meaningless for the sun and the boundary. Copying it
	// through unchanged would publish a stale planet ID alongside a hazard that
	// has nothing to do with it, so only a real body populates the field.
	if hit.IsBody() {
		hit.BodyID = found.id
	}
	return hit
}

// Landing is where one in-flight fleet's straight path ends, as predicted by
// PredictFleets.
type Landing struct {
	FleetID int
	// Resolved is false when the fleet struck nothing within the horizon; Hit,
	// Segment, and Departed are then zero and mean nothing.
	Resolved bool
	// Hit is what the fleet strikes first. Hit.T is the fraction of the way
	// through segment Segment at which it does.
	Hit Hit
	// Segment counts turns from now, starting at 0: a fleet with Segment 0 is
	// resolved by the very next AdvanceTurn, and its arrival time in turns is
	// Segment + Hit.T.
	Segment int
	// Departed is true when Hit is a comet that leaves the board on the same
	// turn the fleet reaches it. The engine discards that arrival with the
	// comet, so the fleet's ships count for nobody, exactly as if it had died.
	Departed bool
}

// Arrives reports whether the fleet hands its ships to a body: it was
// resolved, struck a planet or comet, and that comet did not leave the board
// the same turn.
func (l Landing) Arrives() bool { return l.Resolved && l.Hit.IsBody() && !l.Departed }

// PredictFleets reports, for every fleet in g.Fleets and in the same order,
// what it will strike first if the game runs on with no new launches,
// looking at most horizon turns ahead.
//
// Unlike FirstHit, this moves the bodies: orbiting planets turn and comets
// step along their paths and expire, turn by turn, using the engine's own
// moveOrbits, moveComets, firstCollision, and expireComets on a private copy
// of the board. Each fleet is therefore tested against each moving body in
// relative motion, just as AdvanceTurn tests it, and the prediction cannot
// drift from what a real turn does.
//
// Three things AdvanceTurn does are left out, each for a stated reason:
//
//   - Launches: a prediction asks where the fleets already flying will land.
//     Fleets never collide with each other, so new launches cannot change it.
//   - Production and combat: they change garrisons and owners, never where a
//     body is or which body a fleet strikes first.
//   - Comet spawning: a wave's paths are drawn from the game's random stream,
//     which a prediction must not consume, and agents cannot know them in
//     advance either. A fleet still flying after the next scheduled wave (turns 50, 150,
//     250, 350, 450) could be caught by a comet this prediction never saw.
//
// It also ignores the game ending. A fleet predicted to land after the turn
// limit, or after an elimination ends the match, never actually does; the
// caller decides whether that matters.
func (g *Game) PredictFleets(horizon int) []Landing {
	return g.predictLandings(g.Fleets, horizon)
}

// predictLandings is PredictFleets for any list of fleets, real or not:
// LeadAim passes a fleet it is only considering launching. The fleets are
// copied, never modified, and none of them has to be in g.Fleets.
func (g *Game) predictLandings(fleets []Fleet, horizon int) []Landing {
	landings := make([]Landing, len(fleets))
	// flying holds the fleets not yet resolved, each a private copy whose
	// Position advances one segment per predicted turn. index points back to
	// the fleet's slot in landings, so the result keeps the input order.
	type flying struct {
		index int
		fleet Fleet
	}
	// With nothing to follow, return before copying the board: the copy is
	// the only cost of an empty prediction. The result is the same empty,
	// non-nil slice the full path would return.
	if len(fleets) == 0 {
		return landings
	}
	pending := make([]flying, len(fleets))
	for i, f := range fleets {
		landings[i].FleetID = f.ID
		pending[i] = flying{index: i, fleet: f}
	}

	sim := g.predictionCopy()
	// paths and bodies are scratch storage reused by every predicted turn:
	// each is emptied at the start of a turn, before that turn's movement and
	// bodies are collected, so one allocation of each serves the whole
	// prediction. bodies is kept apart from the turn's motion because a turn on
	// which nothing moves uses the static lookup, whose motion holds no
	// storage. Keeping the buffer here means a comet that pauses for a turn
	// does not cost a fresh allocation when it moves again.
	var paths []bodySweep
	var bodies []collisionBody
	for segment := range horizon {
		if len(pending) == 0 {
			break
		}
		// Each predicted turn mirrors AdvanceTurn's steps 1 and 6–8 (AGENTS.md
		// "Target turn order"), in the same order: safeguard expiry, move the
		// bodies and record their paths, test every fleet against those paths,
		// then expire the comets that stepped past their last point.
		sim.expireComets()
		paths = sim.moveOrbits(paths[:0])
		paths = sim.moveComets(paths)
		motion := sim.turnMotionFor(paths, nil, bodies[:0])
		if motion.prepared {
			bodies = motion.bodies
		}

		still := pending[:0]
		for _, p := range pending {
			start := p.fleet.Position
			end := start.Add(p.fleet.Velocity)
			// The launch-turn exclusion applies on exactly one turn, as in
			// moveFleets. sim.Turn is the turn this segment runs as.
			exclude := 0
			if p.fleet.LaunchTurn == sim.Turn {
				exclude = p.fleet.FromPlanetID
			}
			found, ok := sim.firstCollision(start, end, exclude, motion)
			if !ok {
				p.fleet.Position = end
				still = append(still, p)
				continue
			}
			landings[p.index] = Landing{FleetID: p.fleet.ID, Resolved: true, Hit: publicHit(found), Segment: segment}
		}
		pending = still

		// A comet that departs this turn takes its arrivals with it, as
		// withoutExpiredComets does in AdvanceTurn. Departures are rare, so
		// this scans every landing on those turns rather than keeping a list
		// of this turn's hits, which would allocate on every turn.
		if sim.expireComets() {
			for i := range landings {
				l := &landings[i]
				if l.Resolved && l.Segment == segment && l.Hit.Kind == HitComet && sim.comet(l.Hit.BodyID) == nil {
					l.Departed = true
				}
			}
		}
		sim.Turn++
	}
	return landings
}

// predictionCopy returns a copy of g's bodies that PredictFleets can move
// without touching g.
//
// Planets are copied with their own Orbit, because moveOrbits writes the new
// angle through that pointer. Comets are copied by value; their Path slices
// stay shared, which is safe because the engine never writes to a Path. The
// copy has no fleets (PredictFleets moves its own) and no random source:
// nothing a prediction runs may draw randomness, and a nil Rand turns any
// such mistake into a panic instead of a silent change to g's random stream.
func (g *Game) predictionCopy() *Game {
	sim := &Game{Config: g.Config, Turn: g.Turn, OrbitRate: g.OrbitRate}
	sim.Config.Rand = nil
	sim.Planets = make([]Planet, len(g.Planets))
	for i, p := range g.Planets {
		if p.Orbit != nil {
			orbit := *p.Orbit
			p.Orbit = &orbit
		}
		sim.Planets[i] = p
	}
	sim.Comets = append([]Comet(nil), g.Comets...)
	return sim
}

// Aim is a launch heading that LeadAim has checked will reach its target.
type Aim struct {
	// Angle is the launch heading in radians, ready to use as Action.Angle.
	Angle float64
	// Turns is when the fleet reaches the target, counted from the start of
	// the launch turn: whole turns flown plus the fraction of the arrival
	// turn, Landing.Segment + Hit.T. It comes from the checking simulation,
	// not from the estimate used to aim.
	Turns float64
}

// reachRootSlack is how far outside a turn's [0,1] window a contact time may
// fall and still count as inside it, clamped to the nearer end. Contact times
// are fractions of a turn, so this is an absolute tolerance on a value of
// order 1. A crossing exactly at a turn boundary can be computed a few ulps
// (about 1e-16) to the wrong side by rounding; 1e-9 absorbs that with wide
// margin while shifting an aim by under a billionth of a turn's travel.
const reachRootSlack = 1e-9

// LeadAim returns the launch angle at which a fleet of ships launched from
// sourceID this turn meets targetID where the target will be when the fleet
// gets there, plus the travel time. It returns false when it finds no
// verified aim: the one heading it tries crosses the sun, another body claims
// the fleet first, a comet target leaves the board before (or as) the fleet
// arrives, or the fleet can never catch the target. It also returns false for
// an unknown source or target, a target equal to the source, or fewer than one
// ship.
//
// False means "no verified aim found", not "no launch can reach the target".
// LeadAim checks a single heading, the one that reaches the target at the
// earliest possible moment. When the sun or another body blocks that heading,
// a different one that meets the target later might still get through.
// LeadAim does not search for it, because every check is a full simulation
// and callers ask about many source-target pairs each turn.
//
// It does not check that the source's owner may launch that many ships; that
// is validateLaunch's job when the launch is submitted.
//
// Aiming at a body's current position misses anything that moves, because the
// body has moved on by the time the fleet arrives. LeadAim works in two
// stages:
//
//  1. Aim. Find the earliest moment a fleet flying straight out could touch
//     the target, and aim at the target's centre at that moment. Within one
//     turn the target moves in a straight line at constant speed, so each
//     turn is solved exactly (see earliestReach), in time order.
//  2. Check. Fly the would-be fleet through predictLandings, which moves every
//     body with the engine's own code, and accept the aim only if the target
//     is what the fleet reaches. Stage 1 treats the target as a point and
//     ignores everything else on the board; this stage is what makes the
//     answer authoritative.
//
// A simpler approach re-estimates the travel time a fixed number of times
// and re-aims after each. That settles only while the target is slower than the
// fleet; a comet moves 4 units a turn, faster than any fleet under about 135
// ships, and the estimate then swings back and forth instead of settling.
// Solving for the contact time directly works at any speed, including a target
// that comes into reach and leaves again within a single turn.
//
// Like PredictFleets, the check cannot see comet waves that have not spawned
// yet.
func (g *Game) LeadAim(sourceID, targetID, ships int) (Aim, bool) {
	source, target := g.launchSource(sourceID), g.launchSource(targetID)
	if source == nil || target == nil || sourceID == targetID || ships < 1 {
		return Aim{}, false
	}
	origin := source.position()
	// A fleet starts this far out from the source's centre (see newFleet), so
	// that much of the distance is already covered at launch.
	head := source.radius() + spawnEpsilon
	speed := g.FleetSpeed(ships)
	// Every fleet leaves the board within the board's diagonal divided by its
	// speed, so no catch-up or landing can come later than this.
	horizon := int(math.Ceil(math.Hypot(g.Config.Width, g.Config.Height)/speed)) + 1

	// Stage 1: aim. The fleet never slows down, so at any moment it sits
	// exactly head + speed*t from the source, on whichever heading it took.
	// It can touch the target then only if the target's centre is within the
	// target's radius of that distance (see earliestReach).
	radius := target.radius()
	// The search runs in time order and stops at the first contact, which is
	// what earliestReach's precondition relies on: every turn after the first
	// begins with the target still out of reach. The first turn is checked
	// here instead. A target whose centre starts closer to the source than the
	// fleet's start point, by more than its radius, has the fleet already past
	// it; LeadAim does not look for a later contact in that rare overlap.
	if target.position().Distance(origin) < head-radius {
		return Aim{}, false
	}
	var aimPoint Vec2
	found := false
	for turn := 0; !found && turn < horizon; turn++ {
		from := g.bodyPositionAfterTurns(targetID, turn)
		move := g.bodyPositionAfterTurns(targetID, turn+1).Sub(from)
		reach := head + speed*float64(turn) + radius
		if tau, ok := earliestReach(from.Sub(origin), move, reach, speed); ok {
			aimPoint, found = from.Add(move.Scale(tau)), true
		}
	}
	if !found {
		return Aim{}, false
	}
	delta := aimPoint.Sub(origin)
	angle := math.Atan2(delta.Y, delta.X)

	// Stage 2: check.
	fleet := g.newFleet(source, ships, angle)
	landing := g.predictLandings([]Fleet{fleet}, horizon)[0]
	if !landing.Arrives() || landing.Hit.BodyID != targetID {
		return Aim{}, false
	}
	return Aim{Angle: angle, Turns: float64(landing.Segment) + landing.Hit.T}, true
}

// earliestReach returns the first moment tau in [0,1] of one turn at which a
// fleet can touch a target moving in a straight line, and whether there is
// one. Within the turn the target sits at offset + move*tau from the source,
// and the fleet's reach is reach + speed*tau, where reach already includes the
// target's radius. "Can touch" means the target's centre is within reach:
//
//	|offset + move*tau| <= reach + speed*tau
//
// Precondition: the target is not already behind the fleet at tau = 0, that
// is, its centre is at least reach - 2*radius from the source. The fleet sits
// on a circle, not a disc, so a target well inside that circle is not
// touchable even though the inequality holds. LeadAim guarantees this by
// checking the first turn itself and searching later turns in time order: a
// turn is only searched after the previous one ended with the target out of
// reach, so it starts outside. With the precondition, the first moment the
// inequality holds is a real first contact.
//
// Both sides are non-negative, so squaring keeps the inequality and turns it
// into f(tau) = a*tau² + b*tau + k <= 0 with the coefficients below. k > 0
// means the target starts out of reach; the answer is then the smallest root
// of f in the window, where f first drops to zero.
func earliestReach(offset, move Vec2, reach, speed float64) (float64, bool) {
	dot := func(u, v Vec2) float64 { return u.X*v.X + u.Y*v.Y }
	a := dot(move, move) - speed*speed
	b := 2 * (dot(offset, move) - reach*speed)
	k := dot(offset, offset) - reach*reach
	if k <= 0 {
		return 0, true // in reach as the turn starts
	}

	// The candidate roots of f. Two cases need their own handling before the
	// quadratic formula applies.
	var roots [2]float64
	n := 0
	switch {
	case a == 0 && b == 0:
		// f is the constant k > 0: the target stays out of reach all turn.
		return 0, false
	case a == 0:
		// The target's speed equals the fleet's, so f is a straight line.
		roots[0], n = -k/b, 1
	default:
		discriminant := b*b - 4*a*k
		// A path that only grazes the target gives a discriminant of exactly
		// zero in exact arithmetic, but rounding can push it slightly
		// negative. As in segmentCircleEntry, a value inside the shared,
		// scale-aware tangentTolerance counts as that single grazing root.
		if discriminant < 0 {
			if discriminant < -tangentTolerance*(b*b+math.Abs(4*a*k)) {
				return 0, false
			}
			discriminant = 0
		}
		// The textbook (-b ± sqrt)/2a subtracts two nearly equal numbers
		// for one of its roots when 4ak is small next to b², and loses most
		// of that root's digits. This form never subtracts like-signed
		// values: q takes the sign of b, so b and the square root add. The
		// second root then comes from the product of roots, k/a, which is
		// equal to q/a * k/q. q is non-zero here because k > 0 and a != 0
		// make b and the discriminant not both zero.
		q := -0.5 * (b + math.Copysign(math.Sqrt(discriminant), b))
		roots[0], roots[1], n = q/a, k/q, 2
	}

	// Keep the roots in the window, allowing reachRootSlack either side, and
	// return the smallest. Which formula gives the smaller root depends on the
	// signs of a and q, so both are compared rather than assumed.
	best, found := 0.0, false
	for _, root := range roots[:n] {
		if !(root >= -reachRootSlack && root <= 1+reachRootSlack) {
			continue // also rejects NaN
		}
		root = min(max(root, 0), 1)
		if !found || root < best {
			best, found = root, true
		}
	}
	return best, found
}

// bodyPositionAfterTurns is where planet or comet id will be once turns more
// whole turns have run, using the formulas moveOrbits and moveComets apply.
// A comet past the end of its path stays on its last point, as moveComets
// leaves it; whether it is still there to be reached is stage 2's question.
func (g *Game) bodyPositionAfterTurns(id, turns int) Vec2 {
	if p := g.planet(id); p != nil {
		if p.Orbit == nil {
			return p.Position
		}
		return g.orbitPoint(p.Orbit, orbitAngleAfter(p.Orbit, g.Turn+turns))
	}
	c := g.comet(id)
	return c.Path[min(c.PathIndex+turns, len(c.Path)-1)]
}

// hitKind maps the engine's internal hazard classification onto the public one.
//
// The mapping is written out rather than relying on the two enumerations
// sharing an iota order. Reordering hazardKind is a reasonable thing to do —
// its order is the engine's tie-break rule, not an external contract — and if
// these were assumed to line up, such a change would silently relabel every
// prediction instead of failing to compile or failing a test.
func hitKind(kind hazardKind) HitKind {
	switch kind {
	case hazardBoundary:
		return HitBoundary
	case hazardSun:
		return HitSun
	case hazardPlanet:
		return HitPlanet
	case hazardComet:
		return HitComet
	default:
		panic(fmt.Sprintf("unknown hazard kind %d", int(kind)))
	}
}
