package mlenv

import (
	"fmt"
	"slices"

	"orrery/game"
)

// obs-v2 splits its body slots into two fixed ranges. A planet always sits in
// 0..planetSlots-1 and a comet always in planetSlots..BodySlots-1, so
// the kinds never compete for room: a map with 40 planets still has all four
// comet slots free when a wave arrives.
const (
	planetSlots = 40
	cometSlots  = 4
	// firstCometSlot is where the comet range begins.
	firstCometSlot = planetSlots
)

// slotBody is one occupied slot. Exactly one of planet and comet is set; both
// nil means padding.
//
// The pointers refer into the game being observed and are read-only: the
// builder must never change the state it describes. They are kept rather than
// copied so later stages can read a planet's orbit or a comet's path without
// this struct growing a copy of every field.
type slotBody struct {
	planet *game.Planet
	comet  *game.Comet
}

// id returns the engine body ID, or 0 for padding.
func (b slotBody) id() int {
	switch {
	case b.planet != nil:
		return b.planet.ID
	case b.comet != nil:
		return b.comet.ID
	}
	return 0
}

// owner returns the body's owner, or game.NeutralPlayer for padding.
func (b slotBody) owner() int {
	switch {
	case b.planet != nil:
		return b.planet.OwnerID
	case b.comet != nil:
		return b.comet.OwnerID
	}
	return game.NeutralPlayer
}

// ships returns the body's raw garrison, or 0 for padding.
func (b slotBody) ships() int {
	switch {
	case b.planet != nil:
		return b.planet.Ships
	case b.comet != nil:
		return b.comet.Ships
	}
	return 0
}

// real reports whether the slot holds a body rather than padding.
func (b slotBody) real() bool { return b.planet != nil || b.comet != nil }

// slotTable is the slot table every later stage of Build works from.
//
// Building it once and reading it everywhere is what keeps the stages in
// agreement: body columns, incoming fleets, and masks all need to know which
// slot holds which body, and two separate derivations could disagree.
type slotTable struct {
	// bodies[slot] is the body in that slot, or padding.
	bodies [BodySlots]slotBody
	// slotOf maps an engine body ID to its slot. It is only ever looked up by
	// key, never iterated, so map order cannot leak into the observation.
	slotOf map[int]int
}

// buildSlotTable assigns every planet and active comet a slot and rejects a
// state obs-v2 cannot describe faithfully.
//
// Planets fill slots 0, 1, 2, ... in ascending ID order, and comets fill 40,
// 41, ... the same way. "Ascending ID order" means sorting and then numbering
// consecutively — it does not use the raw ID as an index. With 20 planets and
// two comets, slots 0–19 and 40–41 are real and 20–39 and 42–43 are padding.
//
// Each rejection exists because the alternative is a plausible but wrong
// observation rather than a crash:
//
//   - Too many planets or comets would mean silently dropping bodies.
//   - A malformed comet path would make its lifetime and direction columns
//     meaningless.
//   - An owner outside the starting roster has no role, so its bodies and
//     fleets would be encoded as nobody's.
func buildSlotTable(g *game.Game, frame SeatFrame) (slotTable, error) {
	// Stage 1: counts. Checked before anything is assigned, so a rejected
	// state never produces a half-filled table.
	if len(g.Planets) > planetSlots {
		return slotTable{}, fmt.Errorf("obs-v2: game has %d planets, exceeding the %d planet slots", len(g.Planets), planetSlots)
	}
	if len(g.Comets) > cometSlots {
		return slotTable{}, fmt.Errorf("obs-v2: game has %d active comets, exceeding the %d comet slots", len(g.Comets), cometSlots)
	}

	// Stage 2: per-body validity. Comets must sit on a real point of their
	// path; the engine keeps Position equal to Path[PathIndex], and the step
	// to the next point is read from the same path. That step is scaled by
	// CometSpeed, so a comet on a board without a positive speed has no
	// meaningful direction.
	if len(g.Comets) > 0 && g.Config.CometSpeed <= 0 {
		return slotTable{}, fmt.Errorf("obs-v2: comets need a positive CometSpeed to scale their direction, got %g", g.Config.CometSpeed)
	}
	for _, c := range g.Comets {
		if len(c.Path) == 0 {
			return slotTable{}, fmt.Errorf("obs-v2: comet %d has an empty path", c.ID)
		}
		if c.PathIndex < 0 || c.PathIndex >= len(c.Path) {
			return slotTable{}, fmt.Errorf("obs-v2: comet %d has path index %d outside its %d points", c.ID, c.PathIndex, len(c.Path))
		}
	}
	seated := func(kind string, id, owner int) error {
		if owner == game.NeutralPlayer {
			return nil
		}
		if _, ok := frame.RoleOf(owner); !ok {
			return fmt.Errorf("obs-v2: %s %d belongs to player %d, who is not in the starting roster", kind, id, owner)
		}
		return nil
	}
	for _, p := range g.Planets {
		if err := seated("planet", p.ID, p.OwnerID); err != nil {
			return slotTable{}, err
		}
	}
	for _, c := range g.Comets {
		if err := seated("comet", c.ID, c.OwnerID); err != nil {
			return slotTable{}, err
		}
	}
	// Fleets have no slot, but their owner decides which incoming-role column
	// they count towards, so they need a role just as bodies do. Unlike a
	// body, a fleet can never be neutral: someone launched it. The engine
	// never makes one, but a hand-built game could, and with no role its
	// ships would have no honest column to go in.
	for _, f := range g.Fleets {
		if f.OwnerID == game.NeutralPlayer {
			return slotTable{}, fmt.Errorf("obs-v2: fleet %d has no owner; every fleet must belong to a seated player", f.ID)
		}
		if err := seated("fleet", f.ID, f.OwnerID); err != nil {
			return slotTable{}, err
		}
	}

	// Stage 3: assignment. Sorting makes the table a function of the state
	// rather than of how the engine's slices happen to be ordered.
	planets := make([]*game.Planet, len(g.Planets))
	for i := range g.Planets {
		planets[i] = &g.Planets[i]
	}
	slices.SortFunc(planets, func(a, b *game.Planet) int { return a.ID - b.ID })
	comets := make([]*game.Comet, len(g.Comets))
	for i := range g.Comets {
		comets[i] = &g.Comets[i]
	}
	slices.SortFunc(comets, func(a, b *game.Comet) int { return a.ID - b.ID })

	slots := slotTable{slotOf: make(map[int]int, len(planets)+len(comets))}
	for i, p := range planets {
		slots.bodies[i] = slotBody{planet: p}
		slots.slotOf[p.ID] = i
	}
	for i, c := range comets {
		slot := firstCometSlot + i
		slots.bodies[slot] = slotBody{comet: c}
		slots.slotOf[c.ID] = slot
	}
	return slots, nil
}
