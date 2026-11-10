package game

import (
	"math"
	"slices"
	"sort"
)

// arrival records one fleet reaching a body. moveFleets produces these as it
// detects collisions, and resolveArrivals consumes them. It exists because
// collision detection and combat cannot happen in the same pass: a fleet's
// old position only exists during movement, while combat must wait until
// *every* fleet has moved so that a turn resolves simultaneously.
type arrival struct {
	bodyID  int
	isComet bool // which of Game.Planets / Game.Comets bodyID names
	ownerID int
	ships   int
}

// moveFleets advances every fleet one turn along its heading and decides what,
// if anything, that turn's travel ran into. Fleets never steer or slow down, so
// a launch angle is a commitment for the rest of that fleet's life.
//
// Every hazard is judged against the whole segment travelled this turn, and
// only one hit counts: any planet or comet ahead of the sun or the board edge —
// see firstCollision. Movement and collision
// have to happen together in this one pass because the check needs both the
// pre-move and post-move positions, and the pre-move one is gone as soon as
// this loop moves on to the next fleet.
//
// paths holds every body that moved this turn (from moveOrbits and
// moveComets, which have already run). A moving body is tested in relative
// motion against the fleet, so the two are compared at the same instant; a
// body absent from paths stood still and is tested at its position.
//
// spawned lists comets placed on the board this turn. They take no part in
// collision until next turn, as in the reference engine, whose first placement of a comet
// is not tested against fleets.
//
// The two outcomes differ in kind. The sun and the board edge destroy a fleet
// outright: the ships cease to exist and count for nobody at scoring. A planet
// or comet instead *consumes* the fleet into an arrival record, which is
// returned for resolveArrivals to fight out once the whole pass is done.
func (g *Game) moveFleets(paths []bodySweep, spawned []int) []arrival {
	// No fleet means nothing to test: skip preparing the bodies altogether.
	if len(g.Fleets) == 0 {
		return nil
	}
	// Bodies are resolved once for the turn, and every fleet below is tested
	// against them.
	motion := g.turnMotionFor(paths, spawned, nil)
	var arrivals []arrival
	// Filter in place. Writing survivors back over the same backing array as we
	// read it is safe because the write index never outruns the read index, and
	// it avoids allocating a fresh slice every single turn of a 500-turn match.
	alive := g.Fleets[:0]
	for _, f := range g.Fleets {
		// Keep the pre-move position: the hazard checks below need the whole
		// path travelled this turn, not just where the fleet ended up.
		start := f.Position
		f.Position = start.Add(f.Velocity)
		// On its launch turn a fleet ignores the body it launched from. The
		// fleet spawns spawnEpsilon outside its source, but an orbiting planet
		// or comet moves during the same turn and can catch its own fleet from
		// behind, so the margin alone is not enough. After this turn the
		// exclusion lapses, so a source that later catches a fleet up is a real
		// collision again.
		exclude := 0
		if f.LaunchTurn == g.Turn {
			exclude = f.FromPlanetID
		}
		hit, ok := g.firstCollision(start, f.Position, exclude, motion)
		if !ok {
			alive = append(alive, f)
			continue
		}
		// Either way the fleet stops existing here: skipping the append is what
		// removes it. Only a body hit leaves anything behind.
		if hit.kind == hazardPlanet || hit.kind == hazardComet {
			arrivals = append(arrivals, arrival{bodyID: hit.id, isComet: hit.kind == hazardComet, ownerID: f.OwnerID, ships: f.Ships})
		}
	}
	g.Fleets = alive
	return arrivals
}

// inBounds includes the board edge. A fleet is destroyed only once it moves past it.
func (g *Game) inBounds(p Vec2) bool {
	return p.X >= 0 && p.X <= g.Config.Width && p.Y >= 0 && p.Y <= g.Config.Height
}

// moveComets steps every comet to the next point on its path, as the reference
// engine does, and returns the path each one travelled this turn, for
// moveFleets.
//
// A comet already on its last point has nowhere to go: its PathIndex still
// advances, past the end, which marks it as departed, and it stays where it is
// so fleets moving this turn still run into it. AdvanceTurn then expires it
// before combat; this never drops anything itself.
//
// As in moveOrbits, the paths are appended to storage and the extended slice
// is returned, so the callers collect both kinds of movement in one list. With
// nil storage, the list is allocated only once a comet actually moves.
func (g *Game) moveComets(storage []bodySweep) []bodySweep {
	sweeps := storage
	for i := range g.Comets {
		c := &g.Comets[i]
		from := c.Position
		c.PathIndex++
		if c.PathIndex < len(c.Path) {
			c.Position = c.Path[c.PathIndex]
		}
		if from == c.Position {
			// Same as moveOrbits' zero-rate case: a comet that did not move
			// is tested as a fixed circle.
			continue
		}
		sweeps = append(sweeps, bodySweep{id: c.ID, isComet: true, from: from, to: c.Position, radius: c.Radius})
	}
	return sweeps
}

// bodySweep is the path one moving body covered this turn: where it started,
// where it ended, and how wide it is. moveFleets tests every fleet against
// these paths in relative motion, so a body that drives over a fleet drags it
// into that body's combat just as a fleet flying into the body would.
type bodySweep struct {
	id      int
	isComet bool // which of Game.Planets / Game.Comets id names
	from    Vec2
	to      Vec2
	radius  float64
}

// withoutExpiredComets drops every arrival at a comet that is no longer in
// the game. AdvanceTurn expires comets between fleet movement and combat, so
// a fleet that reached a departing comet this turn was consumed by it and has
// nothing left to fight over. Filtering in place reuses the slice, since the
// unfiltered list is not needed afterwards.
func withoutExpiredComets(g *Game, arrivals []arrival) []arrival {
	kept := arrivals[:0]
	for _, a := range arrivals {
		if a.isComet && g.comet(a.bodyID) == nil {
			continue
		}
		kept = append(kept, a)
	}
	return kept
}

// resolveArrivals fights out everything that reached a body this turn. It is
// deliberately separate from the movement pass that detected those arrivals:
// resolving each fleet the moment it was found would make combat depend on the
// order fleets happen to sit in the slice, so a player could win a contest
// purely by having launched earlier. Collecting every arrival first makes a
// turn genuinely simultaneous — everything that arrives together is weighed
// together.
func (g *Game) resolveArrivals(arrivals []arrival) {
	// Both maps are keyed body ID -> owner ID -> ships. The inner map is what
	// merges one player's separate fleets into a single force, so two fleets of
	// 5 arriving together fight as 10 rather than being defeated piecemeal.
	planetArrivals := map[int]map[int]int{}
	cometArrivals := map[int]map[int]int{}
	for _, a := range arrivals {
		byBody := planetArrivals
		if a.isComet {
			byBody = cometArrivals
		}
		if byBody[a.bodyID] == nil {
			byBody[a.bodyID] = map[int]int{}
		}
		byBody[a.bodyID][a.ownerID] += a.ships
	}
	// Map iteration order is random in Go, but that is harmless here: each body
	// is resolved independently against its own arrivals, so no result can
	// depend on which body is settled first.
	for id, forces := range planetArrivals {
		p := g.planet(id)
		p.OwnerID, p.Ships = Battle(p.OwnerID, p.Ships, forces)
	}
	for id, forces := range cometArrivals {
		c := g.comet(id)
		c.OwnerID, c.Ships = Battle(c.OwnerID, c.Ships, forces)
	}
}

// Battle decides who holds a planet or comet after everything that arrived this
// turn has fought, and returns the surviving owner and ship count. It is shared
// by planets and comets so the one combat rule cannot drift between the two.
// arrivals maps each arriving owner to its total ships. It is exported, like
// FleetSpeed, so a bot forecasting a fight from its observation uses the real
// rule rather than reimplementing it; it reads only its arguments.
//
// Combat runs in two stages, matching the reference engine:
//
//  1. Arriving fleets fight *each other* first, with the standing garrison
//     held out entirely. Only the two largest forces count: the largest
//     survives with its size minus the second largest's. Every smaller force
//     is destroyed without reducing the survivor, so 10, 8 and 5 ships leave
//     the largest owner with 2. If the two largest are exactly equal, no
//     attacker survives.
//  2. The lone attacking survivor (if any) then meets the garrison: it
//     reinforces if it shares the garrison's owner, otherwise it attacks —
//     same attrition rule, so a smaller assault is never pure waste, it
//     still costs the defender exactly what it cost the attacker. Ownership
//     only flips when the survivor strictly outnumbers the garrison; a tie
//     (like a losing attack) leaves the garrison's owner in place at 0
//     ships rather than resetting the body to neutral — that annihilation
//     only happens to ties among attackers in Stage 1.
//
// The two stages matter once three or more forces meet on one body: pooling
// the garrison into the same contest as the attackers (the old behavior)
// could let the garrison's size change which *attacker* wins Stage 1, which
// the rules do not allow — attackers settle their own contest first.
func Battle(defenderOwner, defenderShips int, arrivals map[int]int) (owner, ships int) {
	type force struct{ owner, ships int }
	attackers := make([]force, 0, len(arrivals))
	for o, s := range arrivals {
		// Drop empty forces so a zero-ship entry can't stand in as a "free"
		// second-largest attacker below.
		if s > 0 {
			attackers = append(attackers, force{o, s})
		}
	}
	// Sort largest first. Owner ID breaks equal-size ties so map iteration
	// order can never change the result; the outcome only depends on it when
	// the two largest are equal, and that case leaves no survivor either way.
	sort.Slice(attackers, func(i, j int) bool {
		if attackers[i].ships == attackers[j].ships {
			return attackers[i].owner < attackers[j].owner
		}
		return attackers[i].ships > attackers[j].ships
	})

	// Stage 1: attackers fight each other, garrison held out. A lone attacker
	// survives whole; otherwise only the top two forces matter.
	var survivor *force
	switch {
	case len(attackers) == 1:
		survivor = &attackers[0]
	case len(attackers) > 1 && attackers[0].ships > attackers[1].ships:
		survivor = &force{attackers[0].owner, attackers[0].ships - attackers[1].ships}
	}

	// Stage 2: the lone survivor (if the attackers didn't wipe each other
	// out) reinforces or assaults the garrison. No survivor at all leaves
	// the garrison untouched.
	if survivor == nil {
		return defenderOwner, defenderShips
	}
	if survivor.owner == defenderOwner {
		return defenderOwner, defenderShips + survivor.ships
	}
	if survivor.ships > defenderShips {
		return survivor.owner, survivor.ships - defenderShips
	}
	// survivor.ships <= defenderShips: the garrison stands, reduced by
	// whatever the survivor cost it. On an exact tie that leaves the
	// garrison at 0 ships but still under defenderOwner — unlike Stage 1,
	// a Stage 2 tie does not hand the body to NeutralPlayer.
	return defenderOwner, defenderShips - survivor.ships
}

// hazardKind names what a travel segment ran into. For the sun and the board
// edge, the declaration order is also the tie-break order used by
// firstCollision when both are struck at the exact same moment.
type hazardKind int

const (
	hazardBoundary hazardKind = iota // the fleet left the board
	hazardSun                        // the fleet crossed the sun
	hazardPlanet                     // the fleet reached a planet: an arrival, not a death
	hazardComet                      // the fleet reached a comet: likewise an arrival
)

// collision is one hazard struck by a travel segment, and when along that
// segment it was struck.
type collision struct {
	t    float64    // fraction of the way along the segment, 0..1; smaller is earlier
	kind hazardKind //
	id   int        // planet or comet ID; meaningless for the sun and the boundary
}

// firstCollision evaluates a single straight segment against every hazard on
// the board — the boundary, the sun, every planet and every comet — and returns
// the one hit that claims the fleet, or false if the path is clear.
//
// The ranking matches the reference engine, which
// checks bodies before it checks the board edge or the sun. Any planet or
// comet the segment touches therefore claims the fleet, even when the sun or
// the edge lies earlier along the path, and among several touched bodies the
// lowest body ID wins rather than the nearest. Only a segment that touches no
// body can be destroyed, and then by whichever of the edge and sun comes first.
//
// excludeBody names one planet or comet to skip, or 0 for none — moveFleets
// uses it to keep a fleet from arriving back at the source it launched from on
// the turn it left. IDs are always positive, so 0 can never name a real body.
//
// motion lists the bodies for this turn's tests (see turnMotion). Its zero
// value means every body stood still and none is skipped, which is how
// FirstHit's static predictions call it; moveFleets and PredictFleets pass a
// prepared motion instead.
//
// The two outcomes are ranked in two separate stages, and the order is safe
// because of precedes' first tier: any planet or comet claims a fleet ahead of
// the sun and the board edge. So the bodies are tested first, and the sun and
// the edge only when no body was touched at all.
//
// The result must not depend on the order planets or comets are stored in:
// that order is the caller's setup order and, once comets expire mid-match,
// changes as the match runs. Rather than visiting bodies in sorted order and
// letting the first offered candidate win a tie, every candidate is ranked by
// collision.precedes, which states the whole tie-break explicitly. The search
// therefore reads the slices in place and allocates nothing — which matters
// because incoming-fleet observations call it once per fleet per predicted
// turn, and a per-call sorted copy made garbage collection the dominant cost
// of a batched environment step.
func (g *Game) firstCollision(start, end Vec2, excludeBody int, motion turnMotion) (collision, bool) {
	var best collision
	found := false
	// consider replaces the best so far only with a candidate that outranks it,
	// so the winner is the same whatever order candidates are offered in.
	consider := func(t float64, ok bool, kind hazardKind, id int) {
		candidate := collision{t: t, kind: kind, id: id}
		if !ok || (found && !candidate.precedes(best)) {
			return
		}
		best, found = candidate, true
	}

	// Stage 1: the bodies. One segment can touch several; precedes settles them
	// by body ID alone, so storage order is free to be anything here.
	if motion.prepared {
		// A prepared turn holds every body that takes part, already resolved
		// into its fixed or moving form, so this loop reads one record per body
		// and looks nothing up. An empty prepared list means no body takes part;
		// it must not fall back to the static loop below, which would bring
		// back the comets the preparation deliberately left out.
		for i := range motion.bodies {
			body := &motion.bodies[i]
			if body.id == excludeBody {
				continue
			}
			t, ok := body.entry(start, end)
			consider(t, ok, body.kind, body.id)
		}
	} else {
		// The static lookup: every body stood still and takes part. It reads the
		// game's own slices, so this path allocates nothing at all.
		for _, p := range g.Planets {
			if p.ID == excludeBody {
				continue
			}
			t, ok := segmentCircleEntry(start, end, p.Position, p.Radius)
			consider(t, ok, hazardPlanet, p.ID)
		}
		for _, c := range g.Comets {
			if c.ID == excludeBody {
				continue
			}
			t, ok := segmentCircleEntry(start, end, c.Position, c.Radius)
			consider(t, ok, hazardComet, c.ID)
		}
	}
	if found {
		return best, true
	}

	// Stage 2: no body was touched, so the sun and the board edge decide.
	// Between those two, precedes takes the earlier entry, and an exact tie
	// falls to the board edge.
	t, ok := g.boundaryExit(start, end)
	consider(t, ok, hazardBoundary, 0)
	if g.Config.SunHazardEnabled {
		t, ok = segmentCircleEntry(start, end, g.Config.Sun, g.Config.SunRadius)
		consider(t, ok, hazardSun, 0)
	}
	return best, found
}

// turnMotion describes the bodies for one turn's collision tests. Its zero
// value is the static lookup: every body stood still and none is skipped.
type turnMotion struct {
	// prepared marks a motion built by prepareTurnMotion. It is what
	// distinguishes "prepared, but no body takes part" (an empty bodies list)
	// from the zero value's static lookup.
	prepared bool
	// bodies holds every planet and comet that takes part this turn, in the
	// game's storage order. Comets spawned this turn are left out.
	bodies []collisionBody
}

// collisionBody is one planet or comet as a collision test needs it, resolved
// once per turn so the per-fleet loop reads a record rather than looking up
// where the body started.
type collisionBody struct {
	id   int
	kind hazardKind // hazardPlanet or hazardComet
	// position is where the body ends this turn: its current Position.
	position Vec2
	// from is where a moving body started this turn; it is unused when moved
	// is false.
	from   Vec2
	radius float64
	// moved reports that the body moved this turn, so it is tested in
	// relative motion rather than as a fixed circle.
	moved bool
}

// turnMotionFor chooses the cheapest motion that gives exact answers for one
// turn. When nothing moved and nothing spawned, every body is a fixed circle
// that takes part, which is exactly the static lookup: it reads the game's own
// slices and needs no preparation or storage. Otherwise the bodies are
// prepared. moveFleets and PredictFleets both call this.
func (g *Game) turnMotionFor(paths []bodySweep, spawned []int, storage []collisionBody) turnMotion {
	if len(paths) == 0 && len(spawned) == 0 {
		return turnMotion{}
	}
	return g.prepareTurnMotion(paths, spawned, storage)
}

// prepareTurnMotion resolves every planet and comet for one turn's collision
// tests. moveFleets and PredictFleets both reach it through turnMotionFor, so
// a predicted turn tests fleets against the bodies exactly as a real turn does.
//
// paths holds every body that moved this turn, and spawned lists the comets
// placed this turn, which take no part in collision until next turn. The
// records are appended to storage, so a caller preparing many turns can pass
// the previous turn's list, emptied, and reuse one allocation.
//
// Each body's path is found by a linear scan of paths, matched on both ID and
// kind. That runs once per body per turn, not once per fleet, and paths holds
// only the bodies that moved, so the scan is cheap and needs no map.
func (g *Game) prepareTurnMotion(paths []bodySweep, spawned []int, storage []collisionBody) turnMotion {
	// Size the list once for every body, so it never grows by repeated appends.
	bodies := storage
	if need := len(storage) + len(g.Planets) + len(g.Comets); cap(bodies) < need {
		bodies = make([]collisionBody, len(storage), need)
		copy(bodies, storage)
	}
	add := func(id int, kind hazardKind, position Vec2, radius float64) {
		if slices.Contains(spawned, id) {
			return
		}
		body := collisionBody{id: id, kind: kind, position: position, radius: radius}
		isComet := kind == hazardComet
		for _, path := range paths {
			if path.id == id && path.isComet == isComet {
				body.from, body.moved = path.from, true
				break
			}
		}
		bodies = append(bodies, body)
	}
	for _, p := range g.Planets {
		add(p.ID, hazardPlanet, p.Position, p.Radius)
	}
	for _, c := range g.Comets {
		add(c.ID, hazardComet, c.Position, c.Radius)
	}
	return turnMotion{prepared: true, bodies: bodies}
}

// entry reports when, as a fraction of the turn, a fleet travelling
// start->end first comes within the body's radius, and whether it does.
//
// A body that stood still is a fixed circle at its position. A body that moved
// this turn went from from to position over the same turn as the fleet, both
// in straight lines at constant speed. Subtracting the body's position at each
// instant from the fleet's turns that into one segment, from start-from to
// end-position, tested against a circle at the origin. The two are compared at
// the same instant, so a pair that meets midway is caught and a pair that
// never meets is not.
func (b *collisionBody) entry(start, end Vec2) (float64, bool) {
	if !b.moved {
		return segmentCircleEntry(start, end, b.position, b.radius)
	}
	return segmentCircleEntry(start.Sub(b.from), end.Sub(b.position), Vec2{}, b.radius)
}

// precedes reports whether c claims a fleet ahead of other, in three tiers:
//
//  1. A body (planet or comet) always beats the sun and the board edge.
//  2. Between two bodies, the lower body ID wins regardless of which is
//     reached first. Planets and comets share one ID space, so this is a
//     total order. The reference engine checks bodies in list order, which is ID order.
//  3. Between the sun and the board edge, the earlier entry wins, and an
//     exact tie falls to the board edge (hazardKind declaration order).
func (c collision) precedes(other collision) bool {
	cBody, otherBody := c.isBody(), other.isBody()
	switch {
	case cBody != otherBody:
		return cBody
	case cBody:
		return c.id < other.id
	case c.t != other.t:
		return c.t < other.t
	default:
		return c.kind < other.kind
	}
}

// isBody reports whether the collision is an arrival at a planet or comet
// rather than a fleet-destroying hazard.
func (c collision) isBody() bool {
	return c.kind == hazardPlanet || c.kind == hazardComet
}

// segmentCircleEntry returns the fraction of the way along segment a->b at
// which it *first* touches the circle at center with the given radius, and
// whether it touches at all.
//
// This is deliberately not the same question as "how close does this segment
// pass to the circle's centre". The closest approach happens partway *through*
// a circle the path cuts across, so ordering hits by closest approach can rank
// a large distant body ahead of a small near one that the fleet plainly reaches
// first. Only the entry point orders hits by time, which is what firstCollision
// needs to pick a winner among several hazards.
//
// A segment that starts inside the circle returns 0: already touching.
//
// tangentTolerance is how far below zero the quadratic's discriminant may fall,
// relative to the size of the terms that produced it, before the path counts as
// a genuine miss rather than a tangent lost to floating-point cancellation.
// Doubles carry roughly 1e-16 of relative precision, so 1e-9 leaves seven
// orders of magnitude of headroom over the observed error while still being far
// tighter than any distance the board can meaningfully represent: on a 100-unit
// board it admits only paths within about a billionth of a unit of touching.
const tangentTolerance = 1e-9

func segmentCircleEntry(a, b, center Vec2, radius float64) (float64, bool) {
	d := b.Sub(a)
	// Offset of the segment's start from the circle's centre. Working relative
	// to the centre turns the problem into "when does |m + t*d| equal radius".
	m := a.Sub(center)
	// Squaring both sides of that gives a quadratic in t: At² + Bt + C = 0.
	A := d.X*d.X + d.Y*d.Y
	B := 2 * (m.X*d.X + m.Y*d.Y)
	C := m.X*m.X + m.Y*m.Y - radius*radius
	// C <= 0 means the start point is already on or inside the circle, so there
	// is nothing to solve: contact happens at t=0. Checking this first also
	// covers the degenerate zero-length segment of a fleet that did not move.
	if C <= 0 {
		return 0, true
	}
	if A == 0 {
		return 0, false
	}
	discriminant := B*B - 4*A*C
	// A negative discriminant means the infinite line misses the circle — but
	// only once it is negative by more than roundoff. B*B and 4*A*C are both
	// large and nearly equal for a path that grazes a circle's edge, so
	// subtracting them loses most of their significant digits: a geometrically
	// exact tangent at ordinary board coordinates lands around -2e-12 rather
	// than 0. Since touching the edge counts as contact, treating that as a
	// miss would silently drop real collisions.
	//
	// The error is proportional to the magnitude of the terms that cancelled,
	// so the tolerance has to scale with them rather than being an absolute
	// distance. Anything inside it is a tangent, and is clamped to exactly one
	// root at -B/2A.
	if scale := B*B + math.Abs(4*A*C); discriminant < 0 {
		if discriminant < -tangentTolerance*scale {
			return 0, false
		}
		discriminant = 0
	}
	// The two roots are where the line crosses the circle; the smaller one is
	// where it goes in. Since C > 0 here (start strictly outside), a negative
	// entry root means the whole circle lies behind the start, and an entry root
	// past 1 means the fleet stops short of it this turn — neither is a hit.
	//
	// The test is written as "not inside [0,1]" rather than "t < 0 || t > 1"
	// so that a NaN t, from a NaN input coordinate, counts as a miss. NaN
	// fails every comparison, so the shorter form would report it as a hit at
	// time NaN, and collision.precedes cannot rank a NaN time against any
	// other: the earliest-hit search would then depend on candidate order.
	t := (-B - math.Sqrt(discriminant)) / (2 * A)
	if !(t >= 0 && t <= 1) {
		return 0, false
	}
	return t, true
}

// boundaryExit returns the fraction of the way along segment a->b at which it
// leaves the board, and whether it leaves at all.
//
// The board being convex is what makes this cheap: a straight path between two
// in-bounds points cannot stray outside, so if the end point is in bounds there
// is no exit to find. When the end point *is* out, the exit is the first of the
// four edges the path crosses, so only the edges actually violated at the end
// need testing, and the earliest of those crossings wins.
func (g *Game) boundaryExit(a, b Vec2) (float64, bool) {
	// A fleet that somehow starts out of bounds is already gone.
	if !g.inBounds(a) {
		return 0, true
	}
	if g.inBounds(b) {
		return 0, false
	}
	d := b.Sub(a)
	exit := math.Inf(1)
	// cross solves for where the path meets one edge line: from + t*delta = limit.
	// delta is guaranteed non-zero for a violated edge, since the path had to
	// move along that axis to get from in-bounds to past that limit.
	cross := func(from, delta, limit float64) {
		if t := (limit - from) / delta; t < exit {
			exit = t
		}
	}
	if b.X < 0 {
		cross(a.X, d.X, 0)
	}
	if b.X > g.Config.Width {
		cross(a.X, d.X, g.Config.Width)
	}
	if b.Y < 0 {
		cross(a.Y, d.Y, 0)
	}
	if b.Y > g.Config.Height {
		cross(a.Y, d.Y, g.Config.Height)
	}
	return exit, true
}
