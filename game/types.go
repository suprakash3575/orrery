// Package game contains the rules-only simulation for Orrery.
package game

import (
	"fmt"
	"math"
	"math/rand"
)

// NeutralPlayer marks an unclaimed planet. Players are numbered 0 through 3.
const NeutralPlayer = -1

// Vec2 is a point or direction in the game's 2D world. The board uses a
// logical 100 by 100 coordinate system rather than screen pixels. The origin
// is the top-left corner, so Y grows downward and an angle of pi/2 points down.
type Vec2 struct {
	X, Y float64 // board units, not pixels; the playing field spans 0..100 on both axes
}

func (a Vec2) Add(b Vec2) Vec2         { return Vec2{a.X + b.X, a.Y + b.Y} }
func (a Vec2) Sub(b Vec2) Vec2         { return Vec2{a.X - b.X, a.Y - b.Y} }
func (a Vec2) Scale(n float64) Vec2    { return Vec2{a.X * n, a.Y * n} }
func (a Vec2) Length() float64         { return math.Hypot(a.X, a.Y) }
func (a Vec2) Distance(b Vec2) float64 { return a.Sub(b).Length() }

// Orbit describes a planet that circles the sun. Angles and RadiansTurn are
// measured in radians; a nil Orbit means that the planet stays still.
//
// Orbit is engine-computed, not caller input, following the reference
// engine: New() decides from each planet's starting Position whether it
// orbits (see OrbitRadiusLimit) and fills this in, overwriting whatever a
// caller supplied, exactly as Radius is overwritten from Production.
type Orbit struct {
	Radius float64 // distance from the sun; the planet's own Radius is separate and comes from Production
	// StartAngle is the planet's angle around the sun at turn 0, radians;
	// 0 places it directly right of the sun. Every later position is computed
	// from it rather than accumulated turn by turn, as the reference engine does.
	StartAngle float64
	Angle      float64 // current position along the circle: StartAngle + RadiansTurn * Turn
	// RadiansTurn is how far Angle advances each turn. It is the game's single
	// shared Game.OrbitRate, so every orbiting planet turns at the same rate in
	// the same direction. A test that needs an exact rate may set it after
	// New() returns.
	RadiansTurn float64
}

// Orbit constants from the reference engine.
const (
	// OrbitRadiusLimit decides which planets orbit: one whose distance from
	// the sun plus its own radius is below this limit circles the sun, while
	// one reaching it or beyond stays still. On the standard 100x100 board
	// this keeps every orbiting planet fully on the board; a smaller or
	// off-centre board lowers the limit further (see Game.orbitRadiusLimit).
	OrbitRadiusLimit = 50.0
	// MinOrbitRate and MaxOrbitRate bound the shared orbit rate New() draws
	// for each game, in radians per turn: [MinOrbitRate, MaxOrbitRate).
	MinOrbitRate = 0.025
	MaxOrbitRate = 0.05
)

// Planet is a source of ships and may be captured by another player.
type Planet struct {
	ID       int
	Position Vec2 // if Orbit is non-nil, New() overwrites this from the orbit instead of using the caller's value
	Radius   float64
	OwnerID  int
	Ships    int
	// Production is validated to [1,5]; Radius is always derived from it
	// (1 + ln(Production)), so a caller-supplied Radius is also overwritten.
	Production int
	Orbit      *Orbit // nil means stationary; always assigned by New(), see Orbit
}

// Comet is a temporary, grey, mobile planet: while it is on the board it can
// be fought over and captured exactly like a Planet, and it disappears for
// good once it reaches the end of its path.
//
// Following the reference engine, a comet does not solve its motion
// each turn: it carries its whole visit as a precomputed list of points, one
// per turn, and steps to the next point every turn (see moveComets).
type Comet struct {
	ID       int
	Position Vec2 // always Path[PathIndex]; New() overwrites a caller-supplied value
	Radius   float64
	OwnerID  int
	Ships    int
	// Production is forced to 1 regardless of what the caller sets.
	Production int
	// Path is every position the comet visits, one per turn, in order. Spawned
	// comets space consecutive points Config.CometSpeed apart along their
	// ellipse and keep only the stretch that is on the board. New() copies it,
	// so a game never shares it with its caller or another game. After that
	// the engine never modifies it — only PathIndex advances — so read-only
	// views such as agent.View share it rather than copying it.
	Path []Vec2
	// PathIndex is the comet's current point in Path. moveComets advances it
	// by one each turn; once it passes the last point the comet has left, and
	// expireComets removes it at the start of the next turn.
	PathIndex int
}

// Fleet is a group of ships travelling in a constant straight-line direction.
// Velocity already includes the fleet's speed, so moving a fleet is just
// Position + Velocity. A fleet never changes heading, size, or speed once
// launched; it only ever arrives, or is destroyed by the sun or the board edge.
type Fleet struct {
	ID           int     // unique across planets, comets and fleets alike; drawn from Game.nextID
	OwnerID      int     // the launching player; a fleet is never neutral
	Ships        int     // fixed for the fleet's whole life, and what decides its speed
	Position     Vec2    // current location; starts just outside the source's radius, not at its centre
	Velocity     Vec2    // per-turn movement, already scaled by FleetSpeed(Ships)
	Angle        float64 // the launch heading in radians, kept for observers; Velocity is what actually moves the fleet
	FromPlanetID int     // the planet or comet this fleet launched from, for display and debugging
	// LaunchTurn is the turn this fleet was created on. Collision detection
	// uses it to ignore FromPlanetID for exactly that turn: the source may
	// move after the fleet is placed, and would otherwise swallow it whole.
	LaunchTurn int
}

// Action is one launch request made at the beginning of a turn. A player may
// issue several actions in the same turn, including several from the same
// source. They are validated together as one total rather than one at a time,
// so no action can succeed merely because it happened to be listed first.
type Action struct {
	PlayerID int     // who is launching; must own the source, and cannot be NeutralPlayer
	PlanetID int     // the source to launch from: a planet or a comet ID, either is valid
	Ships    int     // how many of the source's garrison to send; must be positive and available
	Angle    float64 // direction of travel in radians; 0 points right, pi/2 points down
}

// ActionError is how AdvanceTurn reports a rejected launch. A turn is still
// all-or-nothing — one bad action rejects every action in it, and nothing is
// mutated — but the error says which action was at fault. A caller that
// merges several players' actions into one turn needs that to blame the right
// player; a plain string would force it to parse the message.
//
// Only the first failing action is reported, in submitted order. Further bad
// actions later in the slice may exist; fixing this one and retrying will
// surface the next.
type ActionError struct {
	Index    int    // position of the failing action in the slice passed to AdvanceTurn
	PlayerID int    // the PlayerID the action claimed, as submitted, even when that ID is itself the problem (e.g. NeutralPlayer)
	SourceID int    // the action's PlanetID, as submitted; may name no body at all
	Reason   string // which rule the action broke, in plain words
}

func (e *ActionError) Error() string {
	return fmt.Sprintf("action %d (player %d, source %d): %s", e.Index, e.PlayerID, e.SourceID, e.Reason)
}

// Config contains the game-wide constants. Keeping these in one type lets a
// test or future game mode use a different board without changing the rules.
type Config struct {
	Width, Height    float64 // board extent; a fleet is destroyed once it leaves 0..Width by 0..Height
	Sun              Vec2    // centre of the board, and the point orbiting planets circle
	SunRadius        float64 // radius used when SunHazardEnabled makes the sun destroy crossing fleets
	SunHazardEnabled bool    // whether the sun participates in first-hit collision checks
	// CometSpawningEnabled controls the scheduled comet waves. It exists as a
	// real switch because merely starting a game with no comets is not
	// enough: AdvanceTurn spawns fresh waves on its own schedule from turn 50
	// onward, and each wave also draws from Rand. A variant that must stay
	// comet-free for its whole run (and keep its RNG stream untouched by
	// spawn rolls) needs spawning itself disabled. Comets already on the
	// board are unaffected; only the schedule is silenced.
	CometSpawningEnabled bool
	MaxTurns             int     // the match ends after this many turns; the standard game is 500
	MaxFleetSpeed        float64 // speed of a 1,000-ship fleet, the ceiling of the speed curve
	// OrbitsEnabled turns planet orbits on. When true, New() draws one shared
	// orbit rate and makes every planet inside OrbitRadiusLimit orbit; when
	// false, every planet stays still and no randomness is drawn for orbits.
	// It is a real switch so a static variant such as RL V1 keeps its game
	// even when some of its planets sit inside the limit.
	OrbitsEnabled bool
	CometSpeed    float64    // distance between consecutive points of a spawned comet's path, so units it moves per turn
	Rand          *rand.Rand // nil defaults to a time-seeded source in New(), for reproducible comet spawns
}

// DefaultConfig returns the standard 100x100, 500-turn game described by the
// rules. The sun is exactly at the centre of the board.
func DefaultConfig() Config {
	return Config{
		Width: 100, Height: 100, Sun: Vec2{50, 50}, SunRadius: 10, SunHazardEnabled: true,
		CometSpawningEnabled: true,
		MaxTurns:             500, MaxFleetSpeed: 6, OrbitsEnabled: true, CometSpeed: 4,
	}
}

// Game holds all mutable state for one match. Code using the engine calls
// AdvanceTurn and then reads Planets and Fleets to display or save the result.
type Game struct {
	Config Config // fixed for the whole match; New() validates it and fills in a Rand if none was given
	Turn   int    // turns completed so far, so the first AdvanceTurn runs as turn 0
	// OrbitRate is the one orbit rate every orbiting planet shares, in radians
	// per turn, drawn by New() from [MinOrbitRate, MaxOrbitRate). It is 0 when
	// Config.OrbitsEnabled is false. It is part of the replay contract.
	OrbitRate float64
	Planets   []Planet // a private copy of the caller's setup, safe to read between turns
	Comets    []Comet  // only those currently on the board; expired comets are dropped, not kept
	Fleets    []Fleet  // every fleet in flight, in launch order
	nextID    int      // last ID handed out; incremented before each new fleet or spawned comet
}
