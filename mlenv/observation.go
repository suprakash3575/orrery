// Package mlenv translates Orrery game states into the fixed-shape numeric
// form a neural network can consume, and translates decisions back into engine
// actions.
//
// It sits outside package game deliberately. The engine stays the authority on
// what the rules are; this package only re-expresses a state it is given, and
// asks the engine itself — through game.FirstHit — whenever it needs to know
// what the physics will do. Nothing here reimplements a rule.
package mlenv

import (
	"fmt"
	"math"
	"sort"

	"orrery/game"
)

// obs-v2 dimensions: the fixed shape of one observation, covering moving
// planets, comets, the sun, and up to four seated players.
//
// python/orrery/protocol.py repeats these values, and the two must change
// together: the env-v3 frame size is computed from them on both sides.
const (
	SchemaID = "obs-v2"
	// BodySlots is 40 planet slots plus 4 comet slots. Generated maps hold
	// at most 10 groups of 4 planets, and at most one comet wave of 4 is on
	// the board at a time: waves arrive 100 turns apart and a comet's path
	// lasts at most 40 turns. Reserving a slot for every comet of every wave
	// would leave most of them permanently empty while still paying the
	// O(n^2) attention cost for them. The split itself lives with the slot
	// table in observation_slots.go.
	BodySlots    = planetSlots + cometSlots
	BodyFeatures = 27
	PlayerSlots  = 4
	// PlayerFeatures carries two totals plus a four-way role one-hot. The
	// model treats rows as an unordered set, so a player's role must be a
	// value in the row rather than implied by which row it occupies.
	PlayerFeatures = 6
)

// noArrival is the value an arrival-time feature carries when nothing is
// inbound.
//
// It is 1.0 rather than 0 because these are *times*, normalized so that 0 means
// "arriving right now". Zero would therefore read as the most urgent possible
// threat rather than the absence of one, which is exactly backwards, and a
// policy trained against it would learn to panic about empty space.
const noArrival = 1.0

// obs-v2 body feature indices. The numbering lives here once and both Go and
// Python refer to it by name.
//
// Every role-indexed group — owner, incoming ships, earliest arrival — lists
// its four columns in the same role order: me, clockwise, opposite,
// counter-clockwise. Code can therefore address a group as base + role
// instead of switching on the role.
const (
	FeatureX          = 0 // rotated position, board-centred, -1 to 1
	FeatureY          = 1 // rotated position, board-centred, -1 to 1
	FeatureRadius     = 2 // radius over PlanetRadius(5)
	FeatureProduction = 3 // production over 5
	FeatureShips      = 4 // log1p(ships) over log1p(1000); not clipped

	// Owner one-hot. Exactly one of these five is 1 on every real body.
	FeatureOwnerMe       = 5
	FeatureOwnerCW       = 6
	FeatureOwnerOpposite = 7
	FeatureOwnerCCW      = 8
	FeatureOwnerNeutral  = 9

	FeatureIsOrbiting = 10 // 1 for a planet with an orbit
	FeatureOrbitSpeed = 11 // signed radians per turn over game.MaxOrbitRate

	FeatureIsComet        = 12 // 1 for a comet
	FeatureCometTurnsLeft = 13 // path points left, including the current one, over 40
	FeatureCometDX        = 14 // rotated step to the next path point over CometSpeed
	FeatureCometDY        = 15 //

	// Incoming ships per role: log1p of the summed ships whose predicted first
	// hit is this body, over log1p(1000).
	FeatureIncomingMe       = 16
	FeatureIncomingCW       = 17
	FeatureIncomingOpposite = 18
	FeatureIncomingCCW      = 19

	// Earliest predicted contact per role, in turns over the forecast horizon
	// plus one. 1 means nothing from that role is inbound.
	FeatureArrivalMe       = 20
	FeatureArrivalCW       = 21
	FeatureArrivalOpposite = 22
	FeatureArrivalCCW      = 23

	// Global facts, copied into every real body row so the model sees them
	// wherever it attends. There is no sun position: the seat frame requires
	// the sun at the board centre, so its normalized position is always 0.
	FeatureTurnProgress   = 24 // turn over MaxTurns
	FeatureNextCometSpawn = 25 // turns until the next scheduled wave over 101; 1 if none
	FeatureSunRadius      = 26 // sun radius over half the board; 0 when the hazard is off
)

// obs-v2 player feature indices. Rows are placed in role order too, but the
// role one-hot repeats it as a value because the model cannot see row order.
const (
	PlayerFeatureShips        = 0 // log1p(total ships) over log1p(1000); not clipped
	PlayerFeatureProduction   = 1 // total production over 204
	PlayerFeatureRoleMe       = 2
	PlayerFeatureRoleCW       = 3
	PlayerFeatureRoleOpposite = 4
	PlayerFeatureRoleCCW      = 5
)

// Observation is the fixed-shape obs-v2 view of a game state from one seat.
//
// Everything is sized at compile time: real bodies fill slots, padding stays
// zeroed, and the masks say which is which. Planets use slots 0–39 and comets
// 40–43. Coordinates are rotated to the observer's seat, and owners, incoming
// fleets, and player rows are described by role (me, clockwise, opposite,
// counter-clockwise) rather than as a single "enemy".
type Observation struct {
	Bodies  [BodySlots][BodyFeatures]float32
	Players [PlayerSlots][PlayerFeatures]float32
	// BodyMask marks slots holding a real planet or active comet.
	BodyMask [BodySlots]bool
	// OwnedMask marks bodies the observing player owns, including ones with
	// no ships left to launch.
	OwnedMask [BodySlots]bool
	// PlayerMask marks the roles seated at the start of the episode. It never
	// changes afterwards: an eliminated player was still a participant.
	PlayerMask [PlayerSlots]bool
	// TargetMask[source][target] reports whether that launch is available.
	TargetMask [BodySlots][BodySlots]bool
	// SlotBodyID maps a slot back to the engine body it describes, and is 0
	// for padding.
	SlotBodyID [BodySlots]int32
}

// Build translates one game state into the obs-v2 view from frame's
// observer.
//
// frame must come from NewSeatFrame on this same game object at its starting
// state; it fixes the rotation and roles for the whole episode, so Build
// never recomputes them, and refuses a frame bound to any other game. There is no normalization argument: every obs-v2 scale is fixed by the
// schema, so the schema ID alone says how a value was produced.
//
// The stages run in a fixed order:
//
//	validate inputs
//	  → build the slot table (and reject states obs-v2 cannot describe)
//	  → encode body columns 0–15
//	  → predict fleets and encode incoming columns 16–23
//	  → encode global columns 24–26
//	  → encode player rows
//	  → build masks
//
// Rejections of the state happen in the first two stages, before anything is
// written. The one later failure is a fleet that prediction cannot resolve,
// which means a broken state. Either way an error returns a zero observation,
// never a half-built one. Build only
// reads g: it moves nothing, spawns or expires no comet, leaves the frame
// alone, and draws no randomness — fleet prediction runs on a copy with no
// random source.
func Build(g *game.Game, frame SeatFrame) (Observation, error) {
	// Stage 1: inputs. A frame is a promise about one particular board, so
	// one that was never checked, or that describes another board, would
	// rotate and label this one wrongly without any sign of it.
	if g == nil {
		return Observation{}, fmt.Errorf("obs-v2: nil game")
	}
	if !frame.built {
		return Observation{}, fmt.Errorf("obs-v2: seat frame was not built by NewSeatFrame")
	}
	if frame.game != g {
		return Observation{}, fmt.Errorf("obs-v2: seat frame belongs to a different game")
	}
	// Identity says this is the frame's game, not that its board still suits
	// the frame: Config is a plain field a caller could change after turn 0.
	// The rotation is only valid about a centred sun on a square board.
	if g.Config.Sun != frame.sun {
		return Observation{}, fmt.Errorf("obs-v2: seat frame rotates about (%g, %g) but this board's sun is at (%g, %g)",
			frame.sun.X, frame.sun.Y, g.Config.Sun.X, g.Config.Sun.Y)
	}
	if err := checkSeatBoard(g.Config); err != nil {
		return Observation{}, fmt.Errorf("obs-v2: %w", err)
	}

	// Stage 2: slot table. Holds every remaining rejection.
	slots, err := buildSlotTable(g, frame)
	if err != nil {
		return Observation{}, err
	}

	// Stages 3–7: encoding. Each stage writes its own columns or masks and
	// reads only the game, the frame, and the slot table.
	var obs Observation
	encodeBodies(&obs, g, frame, slots)
	if err := encodeIncoming(&obs, g, frame, slots); err != nil {
		// Only an unresolvable fleet or an engine/slot disagreement reaches
		// here — both mean a broken state, so nothing partial is returned.
		return Observation{}, err
	}
	encodeGlobals(&obs, g, slots)
	encodePlayers(&obs, g, frame)
	encodeMasks(&obs, frame, slots)
	return obs, nil
}

// bodyRef is one planet or comet, flattened so slot assignment does not care
// which kind it holds. The engine gives planets and comets one shared ID space
// precisely so the rest of the system can treat them uniformly.
type bodyRef struct {
	id         int
	position   game.Vec2
	radius     float64
	owner      int
	ships      int
	production int
	isComet    bool
}

// collectBodies returns every planet and comet in ascending ID order.
//
// Sorting is what makes a slot assignment a function of the state rather than
// of how the engine's slices happen to be arranged. Two observations of the
// same state must agree on which slot holds which body, or an action decoded
// against one would name a different body in the other.
func collectBodies(g *game.Game) []bodyRef {
	bodies := make([]bodyRef, 0, len(g.Planets)+len(g.Comets))
	for _, p := range g.Planets {
		bodies = append(bodies, bodyRef{
			id: p.ID, position: p.Position, radius: p.Radius,
			owner: p.OwnerID, ships: p.Ships, production: p.Production,
		})
	}
	for _, c := range g.Comets {
		bodies = append(bodies, bodyRef{
			id: c.ID, position: c.Position, radius: c.Radius,
			owner: c.OwnerID, ships: c.Ships, production: c.Production,
			isComet: true,
		})
	}
	sort.Slice(bodies, func(i, j int) bool { return bodies[i].id < bodies[j].id })
	return bodies
}

// centred maps a coordinate onto -1..1 about the middle of the board.
func centred(value, extent float64) float64 {
	if extent == 0 {
		return 0
	}
	half := extent / 2
	return clampSigned((value - half) / half)
}

func clampSigned(v float64) float64 {
	if math.IsNaN(v) {
		return 0
	}
	return math.Min(math.Max(v, -1), 1)
}
