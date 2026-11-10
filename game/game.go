package game

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// New creates a game from supplied planets and comets. IDs must be unique
// and positive, and are shared between the two kinds so a comet can never be
// mistaken for a planet with the same ID. A copy of each slice is made so
// callers can safely reuse their setup slices.
func New(config Config, planets []Planet, comets []Comet) (*Game, error) {
	// Every float in Config must be finite before any range check below can
	// be trusted: NaN fails every comparison, so "Width <= 0" is false for a
	// NaN width and a range check alone waves it through. Each of these feeds
	// geometry the whole match runs on — a NaN Width makes every position
	// fail inBounds, so every fleet dies on launch, and a NaN CometSpeed would
	// leave every spawned comet path with NaN spacing.
	if !finite(config.Width, config.Height, config.Sun.X, config.Sun.Y, config.SunRadius,
		config.MaxFleetSpeed, config.CometSpeed) {
		return nil, fmt.Errorf("invalid configuration: every Config float must be finite")
	}
	if config.Width <= 0 || config.Height <= 0 || config.SunRadius < 0 || config.MaxTurns <= 0 || config.MaxFleetSpeed < 1 {
		return nil, fmt.Errorf("invalid configuration")
	}
	// CometSpeed is the spacing between a spawned comet's path points, so it
	// must be positive or every point would coincide and the comet would
	// never move. A speed too large to fit a usable path on the board simply
	// makes spawning find none, as the reference engine does.
	if config.CometSpeed <= 0 {
		return nil, fmt.Errorf("invalid configuration")
	}
	if config.Rand == nil {
		config.Rand = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	g := &Game{Config: config, Planets: append([]Planet(nil), planets...), Comets: append([]Comet(nil), comets...)}
	// Draw the shared orbit rate before anything else, matching the order
	// the reference engine uses (angular velocity first). Skipping the draw
	// entirely when orbits are off keeps a static game's later draws, such as
	// comet spawns, unaffected by this feature.
	if config.OrbitsEnabled {
		g.OrbitRate = MinOrbitRate + (MaxOrbitRate-MinOrbitRate)*config.Rand.Float64()
	}
	// seen tracks IDs across both planets and comets, since the two kinds
	// share one ID space and a duplicate between them is still a collision.
	seen := map[int]bool{}
	maxID := 0
	for i := range g.Planets {
		p := &g.Planets[i]
		if p.ID <= 0 || seen[p.ID] {
			return nil, fmt.Errorf("invalid planet %d", p.ID)
		}
		if !validOwner(p.OwnerID) {
			return nil, fmt.Errorf("planet %d has invalid owner %d", p.ID, p.OwnerID)
		}
		if p.Production < 1 || p.Production > 5 {
			return nil, fmt.Errorf("planet %d production %d out of range [1,5]", p.ID, p.Production)
		}
		if p.Ships < 0 {
			return nil, fmt.Errorf("invalid planet %d", p.ID)
		}
		seen[p.ID] = true
		if p.ID > maxID {
			maxID = p.ID
		}
		// Radius is derived from production, not caller-supplied, so every
		// planet's size stays consistent with how many ships it can build.
		p.Radius = PlanetRadius(p.Production)
		// The starting Position places every planet, orbiting or not, so it
		// must be finite. A non-finite value would put the planet nowhere:
		// every collision test against it would fail silently, so it could
		// never be reached.
		if !finite(p.Position.X, p.Position.Y) {
			return nil, fmt.Errorf("planet %d has a non-finite position", p.ID)
		}
		g.assignOrbit(p)
	}
	// Comets are validated in the same pass and against the same `seen` set as
	// planets, so the two kinds share one ID space. That is what lets the rest
	// of the engine treat a comet as just another capturable body: an ID names
	// exactly one thing, and a launch or arrival never has to ask which kind it
	// is holding before it can look it up.
	for i := range g.Comets {
		c := &g.Comets[i]
		if c.ID <= 0 || seen[c.ID] {
			return nil, fmt.Errorf("invalid comet %d", c.ID)
		}
		if !validOwner(c.OwnerID) {
			return nil, fmt.Errorf("comet %d has invalid owner %d", c.ID, c.OwnerID)
		}
		if c.Ships < 0 {
			return nil, fmt.Errorf("invalid comet %d", c.ID)
		}
		seen[c.ID] = true
		if c.ID > maxID {
			maxID = c.ID
		}
		// Comets are uniform: every one is the same size and out-produces
		// nothing, so unlike a planet these are not caller choices at all.
		// Overwriting rather than validating keeps a comet spawned mid-match
		// (step 5) identical to one supplied at setup.
		c.Radius = 1
		c.Production = 1
		// A comet must have somewhere to be: at least one path point, and a
		// PathIndex naming one of them. Every point must be finite, since a
		// NaN point would put the comet nowhere and every collision test
		// against it would fail silently.
		if len(c.Path) == 0 || c.PathIndex < 0 || c.PathIndex >= len(c.Path) {
			return nil, fmt.Errorf("comet %d needs a path point at PathIndex %d", c.ID, c.PathIndex)
		}
		for _, point := range c.Path {
			if !finite(point.X, point.Y) {
				return nil, fmt.Errorf("comet %d has a non-finite path point", c.ID)
			}
		}
		// Deep-copy: append() above only copied the slice header, so without
		// this two games built from the same setup would share one path.
		c.Path = append([]Vec2(nil), c.Path...)
		c.Position = c.Path[c.PathIndex]
	}
	// Fleets (and, later, spawned comets) draw new IDs from here, so they must
	// start above every ID already in use. maxID is tracked across both loops
	// above for exactly this reason: seeding from only one kind, or from a
	// count rather than the true maximum, would hand a fleet an ID that already
	// names a planet or comet, and lookups would then find the wrong body.
	g.nextID = maxID
	return g, nil
}

// assignOrbit decides from a planet's starting Position whether it orbits, as
// the reference engine does, and records the orbit if so. A planet orbits when
// orbits are enabled and its distance from the sun plus its own radius is
// below the board's orbit limit (see orbitRadiusLimit); otherwise its Orbit is
// nil and it never moves.
//
// Whatever Orbit a caller supplied is replaced, exactly as Radius is replaced
// from Production. The new Orbit is freshly allocated, so two games built from
// the same setup slice never share orbit state.
//
// Every orbiting planet gets the same Game.OrbitRate, so they all turn in
// lockstep in the same direction. No randomness is drawn here, so the order
// planets are listed in cannot change the result.
func (g *Game) assignOrbit(p *Planet) {
	p.Orbit = nil
	if !g.Config.OrbitsEnabled {
		return
	}
	offset := p.Position.Sub(g.Config.Sun)
	radius := offset.Length()
	if radius+p.Radius >= g.orbitRadiusLimit() {
		return
	}
	angle := math.Atan2(offset.Y, offset.X)
	p.Orbit = &Orbit{Radius: radius, StartAngle: angle, Angle: angle, RadiansTurn: g.OrbitRate}
}

// orbitRadiusLimit is the effective orbit limit for this board: the reference
// engine's OrbitRadiusLimit, or the sun's distance to the nearest board edge if that is
// smaller. A whole circle of that radius around the sun stays on the board, so
// an orbiting planet can never leave it. On the standard 100x100 board with the
// sun at its centre the two are equal, so the reference engine's rule is unchanged there.
func (g *Game) orbitRadiusLimit() float64 {
	sun := g.Config.Sun
	nearestEdge := min(sun.X, g.Config.Width-sun.X, sun.Y, g.Config.Height-sun.Y)
	return min(OrbitRadiusLimit, nearestEdge)
}

// finite reports whether every value is an ordinary number: neither NaN nor
// ±Inf. New() checks this before any range check, because NaN compares false
// against everything, so a range check alone cannot reject it.
func finite(values ...float64) bool {
	for _, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return true
}

// validOwner reports whether id is a valid planet/comet owner: one of the
// four player slots, or NeutralPlayer.
func validOwner(id int) bool {
	return id == NeutralPlayer || (id >= 0 && id <= 3)
}

// Finished reports whether the match is over: either the turn limit was
// reached, or elimination has — at most one distinct non-neutral player owns
// anything at all, anywhere on the board. A 4-player free-for-all collapsing
// to a single owner ends the match immediately rather than playing out the
// remaining turns with nothing left to contest.
func (g *Game) Finished() bool {
	return g.Turn >= g.Config.MaxTurns || g.EliminationReached()
}

// EliminationReached reports whether at most one distinct non-neutral owner
// remains across planets, comets, and fleets combined — including a board
// with zero owners at all, which is still "at most one". This is
// unconditional: PLAN.md's rule has no exception for a setup that only ever
// had one owner to begin with, so a single-player match is finished from the
// moment it's constructed. Test fixtures that need to run multiple turns
// without finishing must include a second, otherwise-uninvolved owner.
func (g *Game) EliminationReached() bool {
	owners := map[int]bool{}
	for _, p := range g.Planets {
		if p.OwnerID != NeutralPlayer {
			owners[p.OwnerID] = true
		}
	}
	for _, c := range g.Comets {
		if c.OwnerID != NeutralPlayer {
			owners[c.OwnerID] = true
		}
	}
	for _, f := range g.Fleets {
		// A fleet is never neutral (Action validation requires a real
		// player), unlike a planet or comet.
		owners[f.OwnerID] = true
	}
	return len(owners) <= 1
}

// AdvanceTurn resolves one simultaneous turn. Actions are launches requested at its start.
func (g *Game) AdvanceTurn(actions []Action) error {
	if g.Finished() {
		return fmt.Errorf("game is already finished")
	}
	// Keep this order in one place. It is the authoritative definition of a
	// turn, which prevents a UI or AI from applying rules in a different
	// order (AGENTS.md "Target turn order" lists it): safeguard expiry,
	// validate, spawn, launch, produce, move bodies, move fleets against the
	// bodies' paths, expire departed comets, resolve combat, advance.
	//
	// Expiry runs first as a safeguard: a comet that has already finished its
	// path must be gone before anyone can try to launch from it. The turn's
	// own expiry happens later, after fleet movement, so this only removes
	// comets placed in that state by hand. Expiry is safe to repeat (it only
	// drops what has finished its path, and draws no randomness) if this
	// call is retried after an earlier failure.
	//
	// spawnComets sits between validation and application, not strictly
	// before both as PLAN.md's shorthand "expire → spawn → validate & launch"
	// might suggest: it draws from Config.Rand and adds comets every time it
	// runs on a scheduled turn, so it must not run until validation
	// guarantees this turn will actually complete, or a retried failed turn
	// would spawn a second wave. A spawned comet starts neutral, so no
	// action this turn could legally reference one anyway — validating
	// first costs nothing, and the two orderings are otherwise equivalent.
	g.expireComets()
	if err := g.validateLaunch(actions); err != nil {
		return err
	}
	// The switch guards the schedule here, at the single call site, rather
	// than inside spawnComets: a disabled game must also skip the wave's
	// Rand draws, so the whole spawn step has to vanish, not just the append.
	var spawned []int
	if g.Config.CometSpawningEnabled {
		spawned = g.spawnComets()
	}
	g.applyLaunch(actions)
	// Production banks this turn's ships before anything moves, so an
	// attacker must beat the garrison plus its production, never the
	// garrison alone — unchanged from before PR7, just now written in its
	// final position rather than after body movement.
	g.produce()
	// Bodies and fleets move over the same turn, so collisions must compare
	// them at the same instant. As in the reference engine, every body's path for
	// this turn is computed first; each fleet is then tested against those
	// paths in relative motion (see firstCollision). Testing a fleet against
	// where a body was, and a body against where a fleet ended up, would miss
	// a pair that meets midway and can report one that never met.
	paths := g.moveOrbits(nil)
	paths = g.moveComets(paths)
	// Collisions are queued, not resolved — combat waits until every fleet
	// has moved, so a turn resolves simultaneously rather than in fleet order.
	// Comets spawned this turn are left out of collision entirely, as the
	// reference engine does for a comet's first placement; they can be hit from next turn.
	queued := g.moveFleets(paths, spawned)
	// A comet that stepped past its last point this turn still launched,
	// produced, and blocked fleets during movement, but leaves now, before
	// combat and scoring, as in the reference engine. Fleets that reached it are consumed
	// with it rather than fighting over a body that no longer exists, so its
	// ships cannot affect the result for an extra turn.
	if g.expireComets() {
		queued = withoutExpiredComets(g, queued)
	}
	g.resolveArrivals(queued)
	g.Turn++
	return nil
}

// spawnEpsilon is added to a source's radius when placing a freshly launched
// fleet, so the fleet's starting point sits strictly outside the source
// circle rather than exactly on its boundary. Continuous collision detection
// treats a point exactly on a circle's edge as touching it, so without this
// margin a fleet would register as colliding with the very source it just
// left, on the turn it launched.
//
// The margin alone is not enough, which is why Fleet.LaunchTurn exists too: an
// orbiting planet or a comet moves after its fleets are placed, and a source
// that travels further in a turn than this margin catches its own fleet from
// behind no matter how far outside the radius the fleet started.
const spawnEpsilon = 0.01

// launch validates every action and, if all pass, applies them. It is split
// into validateLaunch and applyLaunch (below) so AdvanceTurn can run
// something irreversible — spawnComets — strictly after validation succeeds,
// rather than between the two; launch itself is a convenience wrapper for
// callers, like tests, that just want "validate and launch" in one call with
// nothing in between.
func (g *Game) launch(actions []Action) error {
	if err := g.validateLaunch(actions); err != nil {
		return err
	}
	g.applyLaunch(actions)
	return nil
}

// validateLaunch checks every action without mutating anything. In
// particular, it adds all requests from the same source together so actions
// are simultaneous rather than order-based.
//
// A failure is always an *ActionError naming the first action that broke a
// rule, so a caller that merged several players' actions into one turn can
// tell whose action it was without parsing the message.
func (g *Game) validateLaunch(actions []Action) error {
	requested := map[int]int{}
	for i, a := range actions {
		reject := func(reason string) error {
			return &ActionError{Index: i, PlayerID: a.PlayerID, SourceID: a.PlanetID, Reason: reason}
		}
		if a.PlayerID == NeutralPlayer || a.Ships <= 0 {
			return reject("launch needs a player and positive ships")
		}
		// A NaN or infinite angle has no direction: cos and sin of it are NaN,
		// so the fleet would be placed at a NaN position and destroyed on its
		// first move, silently throwing the ships away. Rejecting it tells the
		// caller about the mistake instead.
		if !finite(a.Angle) {
			return reject("angle must be a finite number")
		}
		src := g.launchSource(a.PlanetID)
		if src == nil || src.owner() != a.PlayerID {
			return reject("player does not own this source")
		}
		// Compare each action against what this source still has unclaimed,
		// rather than adding it to the running total and comparing that. Ship
		// counts are caller-chosen ints, and the sum of two of them can pass
		// the int maximum and wrap round to a negative number, which would slip
		// under the garrison check, drive the garrison negative, and launch a
		// fleet of ships that never existed. The subtraction cannot wrap:
		// requested never exceeds src.ships(), which is never negative.
		if a.Ships > src.ships()-requested[a.PlanetID] {
			return reject("source does not have enough ships")
		}
		requested[a.PlanetID] += a.Ships
	}
	return nil
}

// applyLaunch removes ships and creates fleets for every action. Callers must
// only reach this after validateLaunch has already succeeded against the same
// actions and no relevant state has changed in between — it does not
// re-check anything itself.
func (g *Game) applyLaunch(actions []Action) {
	// This loop re-walks actions rather than a pre-aggregated total, since
	// each action still needs its own Angle to compute a fleet's direction.
	for _, a := range actions {
		src := g.launchSource(a.PlanetID)
		src.deduct(a.Ships)
		g.nextID++
		fleet := g.newFleet(src, a.Ships, a.Angle)
		fleet.ID, fleet.OwnerID = g.nextID, a.PlayerID
		g.Fleets = append(g.Fleets, fleet)
	}
}

// newFleet builds the fleet a launch of ships from src at angle would place
// this turn, leaving ID and OwnerID for the caller. applyLaunch uses it for
// real launches and LeadAim for a launch it is only checking, so the two
// start from the same position and velocity, bit for bit.
func (g *Game) newFleet(src launchSource, ships int, angle float64) Fleet {
	direction := Vec2{math.Cos(angle), math.Sin(angle)}
	// Fleets begin strictly outside the launching source (radius plus
	// spawnEpsilon), so they cannot immediately collide with the planet or
	// comet they just left.
	return Fleet{
		Ships:        ships,
		Position:     src.position().Add(direction.Scale(src.radius() + spawnEpsilon)),
		Velocity:     direction.Scale(g.FleetSpeed(ships)),
		Angle:        angle,
		FromPlanetID: src.sourceID(),
		LaunchTurn:   g.Turn,
	}
}

// launchSource is the common surface launch() needs from whatever a fleet is
// launched from: where it is, how big, who owns it, how many ships it can
// spare, and a way to draw that garrison down. Planets and comets both
// implement it, so launch() never has to branch on which kind it is holding.
type launchSource interface {
	sourceID() int
	position() Vec2
	radius() float64
	owner() int
	ships() int
	deduct(n int)
}

func (p *Planet) sourceID() int   { return p.ID }
func (p *Planet) position() Vec2  { return p.Position }
func (p *Planet) radius() float64 { return p.Radius }
func (p *Planet) owner() int      { return p.OwnerID }
func (p *Planet) ships() int      { return p.Ships }
func (p *Planet) deduct(n int)    { p.Ships -= n }

func (c *Comet) sourceID() int   { return c.ID }
func (c *Comet) position() Vec2  { return c.Position }
func (c *Comet) radius() float64 { return c.Radius }
func (c *Comet) owner() int      { return c.OwnerID }
func (c *Comet) ships() int      { return c.Ships }
func (c *Comet) deduct(n int)    { c.Ships -= n }

// launchSource looks up a planet or comet by ID, whichever kind holds it.
// Returns nil if neither does — planets and comets share one ID space
// (enforced in New()), so an ID never names both.
func (g *Game) launchSource(id int) launchSource {
	if p := g.planet(id); p != nil {
		return p
	}
	if c := g.comet(id); c != nil {
		return c
	}
	return nil
}

// FleetSpeed applies Orrery's logarithmic speed curve for an explicit
// maximum speed. One ship moves at 1.0; 1,000 ships reach maxSpeed.
//
// The curve is what makes fleet size a real decision rather than an obvious
// one. Splitting a garrison into small fleets sends more attacks but each
// crawls; committing it to one mass arrives far sooner. Because the growth is
// logarithmic, the early ships buy the most speed and later ones add very
// little. At the default maximum of 6.0: 10 ships travel at 1.96, 100 at 3.72,
// and 500 at 5.27 — a fiftyfold increase in size not quite tripling speed.
//
// Keeping the calculation independent of Game lets read-only callers, such as
// an agent evaluating a copied state view, use the authoritative formula
// without receiving a mutable *Game or reimplementing the rule themselves.
func FleetSpeed(ships int, maxSpeed float64) float64 {
	if ships < 1 {
		return 0
	}
	// Guard the curve's upper end. Past 1,000 ships the formula would keep
	// climbing above the configured ceiling, so clamp instead.
	if ships >= 1000 {
		return maxSpeed
	}
	// Express the fleet's size as a 0..1 position along the curve. Dividing two
	// logs rebases the logarithm to 1,000, so 1 ship gives 0 (log 1 is 0) and
	// 1,000 ships give exactly 1. Using logs is what compresses the huge 1..1000
	// range into that span, so each *tenfold* increase adds a similar amount of
	// speed rather than each extra ship doing so.
	ratio := math.Log(float64(ships)) / math.Log(1000)
	// Map that 0..1 position onto the 1..MaxFleetSpeed range: start at 1, then
	// add the fraction of the remaining speed the fleet has earned. Raising the
	// ratio to 1.5 bends the curve so mid-size fleets stay slower than a plain
	// straight-line mapping would make them, keeping the top speed a genuine
	// reward for committing a large fleet.
	return 1 + (maxSpeed-1)*math.Pow(ratio, 1.5)
}

// PlanetRadius derives a planet's size from its production. A planet that
// builds more ships is physically larger, so it is both a bigger target to hit
// and a shorter flight to reach — production buys presence on the board, not
// only ships.
//
// The logarithm keeps that growth gentle: across the validated 1..5 production
// range the radius only spans 1.00 to 2.61, so the most productive planet is
// not an unmissably large target compared with the least.
//
// New() is the authority that assigns this to every planet, overwriting
// whatever a caller supplied. The calculation is exported separately, like
// FleetSpeed, so read-only callers — a scenario generator spacing planets
// before the engine exists, or an agent reasoning about a state view — can use
// the real formula rather than reimplementing it and drifting from it.
func PlanetRadius(production int) float64 {
	return 1 + math.Log(float64(production))
}

// FleetSpeed applies the authoritative speed curve using this game's
// configured maximum.
func (g *Game) FleetSpeed(ships int) float64 {
	return FleetSpeed(ships, g.Config.MaxFleetSpeed)
}

// produce is the economy step of a turn: every owned planet adds its
// Production to its garrison. Growth is unbounded, so an uncontested planet
// compounds indefinitely and holding ground is what wins a match.
//
// Neutral planets deliberately do not produce, even when they hold ships. That
// makes an unclaimed planet a fixed-price target rather than one that gets more
// expensive the longer it is ignored.
//
// Production is the source of truth for a planet's size, not the other way
// round: New() derives Radius from it as 1 + ln(Production). A high-output
// planet is therefore physically larger, and so an easier target to hit.
//
// AdvanceTurn calls this before fleets move, which means a planet banks this
// turn's ships before any attacker reaches it. Attackers must beat the garrison
// plus its production, never the garrison alone.
//
// Comets produce too, at a flat +1/turn regardless of Production — which New()
// pins to 1 for every comet anyway, so this is really the same rule as a
// planet's, just written directly rather than through the field. A comet is
// free production for as long as it stays on the board, which is the upside
// half of the tension the comet rule creates: banking ships on one is a loan
// that only pays out until it reaches the end of its path (see moveComets).
func (g *Game) produce() {
	for i := range g.Planets {
		planet := &g.Planets[i]
		if planet.OwnerID != NeutralPlayer {
			planet.Ships += planet.Production
		}
	}
	for i := range g.Comets {
		comet := &g.Comets[i]
		if comet.OwnerID != NeutralPlayer {
			comet.Ships++
		}
	}
}

// moveOrbits rotates every orbiting planet one step along its circle and
// returns the straight-line path each one covered, for moveFleets.
//
// The returned path is a chord, not the arc actually flown: a planet's real
// path this turn is a slight curve, and collision detection tests the
// straight line between its endpoints instead. At the default rate a planet turns a fraction
// of a degree per turn, so the two differ by far less than a planet's own
// radius and no fleet can slip between them. The reference engine makes the same
// straight-line approximation.
//
// Stationary planets return no path at all, so moveFleets tests them as fixed
// circles.
//
// The paths are appended to storage, and the extended slice is returned.
// AdvanceTurn passes nil; PredictFleets passes the same list, emptied, on
// every predicted turn, so its many predicted turns reuse one allocation
// rather than making one each. With nil, the list is allocated only once a
// planet actually moves, so a static board allocates nothing here.
func (g *Game) moveOrbits(storage []bodySweep) []bodySweep {
	sweeps := storage
	// AdvanceTurn calls this before it increments Turn, so the turn now
	// completing is Turn+1.
	turnsCompleted := g.Turn + 1
	for i := range g.Planets {
		p := &g.Planets[i]
		if p.Orbit == nil {
			continue
		}
		from := p.Position
		p.Orbit.Angle = orbitAngleAfter(p.Orbit, turnsCompleted)
		p.Position = g.orbitPoint(p.Orbit, p.Orbit.Angle)
		// A planet that didn't actually move this turn (a test set RadiansTurn
		// to exactly 0) returns no path, so it is tested exactly like a
		// stationary planet rather than through the relative-motion check.
		if from == p.Position {
			continue
		}
		sweeps = append(sweeps, bodySweep{id: p.ID, from: from, to: p.Position, radius: p.Radius})
	}
	return sweeps
}

// orbitAngleAfter is an orbit's angle once turnsCompleted turns of it have
// run. It is computed from the starting angle and the turn count, as the
// reference engine does, rather than added to turn by turn, so no rounding error builds up over
// a 500-turn match. It is the one place this formula lives: moveOrbits uses it
// for the current turn, and comet path search uses it for future turns.
func orbitAngleAfter(o *Orbit, turnsCompleted int) float64 {
	return o.StartAngle + o.RadiansTurn*float64(turnsCompleted)
}

// orbitPoint converts an angle on an orbit into board coordinates. An orbiting
// planet stores only how far around the circle it is and how far out from the
// sun it sits (Orbit.Radius); this turns that pair into an actual x,y.
//
// Angle 0 places the planet directly right of the sun, and the angle grows
// clockwise on screen because the board's Y axis points down.
func (g *Game) orbitPoint(o *Orbit, angle float64) Vec2 {
	// cos and sin give a unit vector pointing at the angle; scaling both parts
	// by the orbit radius walks that far out from the sun, and adding the sun's
	// own position shifts the result from "relative to the sun" to "on the board".
	return g.Config.Sun.Add(Vec2{math.Cos(angle) * o.Radius, math.Sin(angle) * o.Radius})
}

// planet finds a planet by its stable ID. The initial prototype has few enough
// planets that a simple scan is clearer than maintaining an index map.
func (g *Game) planet(id int) *Planet {
	for i := range g.Planets {
		if g.Planets[i].ID == id {
			return &g.Planets[i]
		}
	}
	return nil
}

// comet finds a comet by its stable ID, the same way planet does.
func (g *Game) comet(id int) *Comet {
	for i := range g.Comets {
		if g.Comets[i].ID == id {
			return &g.Comets[i]
		}
	}
	return nil
}
