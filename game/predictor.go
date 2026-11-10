package game

import "fmt"

// Snapshot is the public board at the start of one turn: what every player
// may see, and everything fleet prediction reads. It is plain data, with no
// random source and no engine counters, so a bot can build one from its
// observation alone, and it could later be sent over a network as it is.
//
// The board settings are only the ones prediction uses: the board edge, the
// sun and the fleet speed curve. Turn limits, comet spawning and orbit
// switches are left out because a prediction never ends the game, spawns
// comets or decides which planets orbit.
type Snapshot struct {
	Turn int // turns completed so far, as Game.Turn
	// OrbitRate is the shared orbit rate in radians per turn, as
	// Game.OrbitRate. Each orbiting planet also carries it in
	// Orbit.RadiansTurn, which is what movement actually reads.
	OrbitRate        float64
	Width, Height    float64
	Sun              Vec2
	SunRadius        float64
	SunHazardEnabled bool
	MaxFleetSpeed    float64
	Planets          []Planet
	Comets           []Comet
	Fleets           []Fleet
}

// Predictor answers questions about where fleets go on one Snapshot, using
// the engine's own movement and collision code, so a bot and the engine can
// never disagree. It only answers questions: it cannot advance turns, take
// launches or draw randomness, and asking never changes its board.
type Predictor struct {
	// board is a private Game built from the snapshot, never the live match,
	// with no random source. It is a Game only so the engine's own prediction
	// methods can run on it. Every method only reads it; LeadAim and
	// PredictFleets move a further copy of their own (see predictionCopy).
	board *Game
}

// NewPredictor validates s and builds a Predictor on a private copy of it, so
// a later change to s does not change the predictor's answers. Comet paths
// are shared, not copied, because the engine never writes to a Path.
//
// Validation rejects what would make a prediction wrong or panic: a board
// New() would refuse, non-finite numbers, reused IDs, unknown owners, empty
// fleets and comets without a current path point. It does not check that the
// board could arise in a real game.
func NewPredictor(s Snapshot) (*Predictor, error) {
	if err := validateSnapshot(s); err != nil {
		return nil, err
	}
	board := &Game{
		Config: Config{
			Width: s.Width, Height: s.Height, Sun: s.Sun, SunRadius: s.SunRadius,
			SunHazardEnabled: s.SunHazardEnabled, MaxFleetSpeed: s.MaxFleetSpeed,
		},
		Turn:      s.Turn,
		OrbitRate: s.OrbitRate,
		Planets:   make([]Planet, len(s.Planets)),
		Comets:    append([]Comet(nil), s.Comets...),
		Fleets:    append([]Fleet(nil), s.Fleets...),
	}
	// Planets carry their Orbit by pointer, so copy each one: sharing it
	// would let the caller move a planet under the predictor.
	for i, p := range s.Planets {
		if p.Orbit != nil {
			orbit := *p.Orbit
			p.Orbit = &orbit
		}
		board.Planets[i] = p
	}
	return &Predictor{board: board}, nil
}

// LeadAim is Game.LeadAim on the snapshot's board: the launch angle at which
// a fleet of ships from sourceID meets targetID where the target will be, and
// the travel time, or false when it finds no verified aim.
func (p *Predictor) LeadAim(sourceID, targetID, ships int) (Aim, bool) {
	return p.board.LeadAim(sourceID, targetID, ships)
}

// PredictFleets is Game.PredictFleets on the snapshot's board: where each of
// its fleets lands, in the snapshot's fleet order, looking at most horizon
// turns ahead.
func (p *Predictor) PredictFleets(horizon int) []Landing {
	return p.board.PredictFleets(horizon)
}

// validateSnapshot reports the first problem that would make a prediction on
// s wrong or panic, or nil.
func validateSnapshot(s Snapshot) error {
	// The board checks are New()'s, so a predictor accepts exactly the boards
	// a game can be built on. finite runs first: NaN passes every range check.
	if !finite(s.OrbitRate, s.Width, s.Height, s.Sun.X, s.Sun.Y, s.SunRadius, s.MaxFleetSpeed) {
		return fmt.Errorf("invalid snapshot: every board number must be finite")
	}
	if s.Turn < 0 || s.Width <= 0 || s.Height <= 0 || s.SunRadius < 0 || s.MaxFleetSpeed < 1 {
		return fmt.Errorf("invalid snapshot: board settings out of range")
	}

	// Planets, comets and fleets share one ID space, as in a real game, so a
	// lookup by ID always finds the one thing it names.
	seen := map[int]bool{}
	claim := func(kind string, id int) error {
		if id <= 0 || seen[id] {
			return fmt.Errorf("invalid snapshot: %s %d has a non-positive or reused ID", kind, id)
		}
		seen[id] = true
		return nil
	}
	for _, p := range s.Planets {
		if err := claim("planet", p.ID); err != nil {
			return err
		}
		if !validOwner(p.OwnerID) || p.Ships < 0 || !finite(p.Position.X, p.Position.Y, p.Radius) || p.Radius <= 0 {
			return fmt.Errorf("invalid snapshot: planet %d has an invalid owner, garrison, position or radius", p.ID)
		}
		if o := p.Orbit; o != nil && !finite(o.Radius, o.StartAngle, o.Angle, o.RadiansTurn) {
			return fmt.Errorf("invalid snapshot: planet %d has a non-finite orbit", p.ID)
		}
	}
	for _, c := range s.Comets {
		if err := claim("comet", c.ID); err != nil {
			return err
		}
		if !validOwner(c.OwnerID) || c.Ships < 0 || !finite(c.Position.X, c.Position.Y, c.Radius) || c.Radius <= 0 {
			return fmt.Errorf("invalid snapshot: comet %d has an invalid owner, garrison, position or radius", c.ID)
		}
		// Prediction looks up a comet's future positions by PathIndex, so the
		// index must name a real point and every point must be finite.
		if c.PathIndex < 0 || c.PathIndex >= len(c.Path) {
			return fmt.Errorf("invalid snapshot: comet %d needs a path point at PathIndex %d", c.ID, c.PathIndex)
		}
		for _, point := range c.Path {
			if !finite(point.X, point.Y) {
				return fmt.Errorf("invalid snapshot: comet %d has a non-finite path point", c.ID)
			}
		}
	}
	for _, f := range s.Fleets {
		if err := claim("fleet", f.ID); err != nil {
			return err
		}
		// A fleet always belongs to a player and carries at least one ship.
		if f.OwnerID < 0 || f.OwnerID > 3 || f.Ships < 1 || !finite(f.Position.X, f.Position.Y, f.Velocity.X, f.Velocity.Y) {
			return fmt.Errorf("invalid snapshot: fleet %d has an invalid owner, size, position or velocity", f.ID)
		}
	}
	return nil
}
