package mlenv

import (
	"fmt"
	"strings"

	"orrery/agent"
)

// LineupSeparator separates the agent specifications in a lineup's text form.
// Specifications use ':', ',' and '=' themselves, so '|' never appears inside
// one and splitting on it cannot cut a specification in half.
const LineupSeparator = "|"

// Lineup chooses the scripted agent for every seat the learner does not hold.
//
// A lineup with one entry plays that agent in every opponent seat. That is the
// only form that existed before lineups, and it is recorded and seeded exactly
// as it was then, so earlier runs replay unchanged.
//
// A longer lineup names one agent per opponent seat by its board role from
// the learner, the same roles the learner's observation uses (see SeatFrame):
// entry 0 plays the seat clockwise of the learner, entry 1 the seat opposite,
// and entry 2 the seat counter-clockwise. The learner's seat changes from game
// to game, and assigning by role gives every game the same mix of opponents
// in the same places around the board. Fixed seat numbers would not: whichever
// fixed seat the learner took would lose its agent.
//
// Roles follow the board, not player numbers: going clockwise around the sun
// the seats run 0 → 1 → 3 → 2. For example, with the learner in seat 2 of a
// four-player game, the lineup "a|b|c" seats a in seat 0 (clockwise), b in
// seat 1 (opposite), and c in seat 3 (counter-clockwise).
type Lineup struct {
	// factories holds one agent factory per opponent seat, in role order
	// (clockwise, opposite, counter-clockwise of the learner), or a single
	// factory for every opponent seat.
	// NewLineup never builds an empty one; the zero Lineup is empty, and
	// NewBatch refuses it.
	factories []agent.Factory
}

// SingleOpponent returns the lineup that plays one agent in every opponent
// seat, as every batch did before lineups existed. It doesn't check the
// factory, so callers keep the one-value form they had; NewBatch checks every
// lineup's factories, however it was built, before any episode starts.
func SingleOpponent(factory agent.Factory) Lineup {
	return Lineup{factories: []agent.Factory{factory}}
}

// NewLineup builds a lineup from factories in role order: clockwise,
// opposite, then counter-clockwise of the learner. It refuses an empty list, a nil factory (including a typed nil),
// and an agent without an ID.
//
// Two or more entries with the same specification are accepted and
// normalized to that single specification, which already means "this agent
// in every seat". Both forms then record the same identity and derive the
// same seeds, so writing a setup either way gives the same games.
func NewLineup(factories ...agent.Factory) (Lineup, error) {
	if err := checkFactories(factories); err != nil {
		return Lineup{}, err
	}
	if len(factories) > 1 && allSameSpec(factories) {
		return Lineup{factories: []agent.Factory{factories[0]}}, nil
	}
	return Lineup{factories: append([]agent.Factory(nil), factories...)}, nil
}

// checkFactories refuses an empty lineup, a nil factory, and an agent without
// an ID. NewLineup and NewBatch both call it, so a lineup built by
// SingleOpponent or as a zero value is checked before it is used.
func checkFactories(factories []agent.Factory) error {
	if len(factories) == 0 {
		return fmt.Errorf("opponent lineup must name at least one agent")
	}
	for i, factory := range factories {
		// nilInterface also catches an interface holding a typed nil
		// pointer, which a plain == nil comparison would miss.
		if nilInterface(factory) {
			return fmt.Errorf("opponent lineup entry %d: factory must not be nil", i+1)
		}
		if factory.Spec().ID() == "" {
			return fmt.Errorf("opponent lineup entry %d: agent ID must not be empty", i+1)
		}
	}
	return nil
}

// ParseLineup reads a lineup's text form: one agent specification, or several
// joined by LineupSeparator. parse turns one specification into a factory;
// the protocol passes agent.ParseFactory, and tests may pass their own.
func ParseLineup(text string, parse func(string) (agent.Factory, error)) (Lineup, error) {
	parts := strings.Split(text, LineupSeparator)
	factories := make([]agent.Factory, 0, len(parts))
	for i, part := range parts {
		// An empty part comes from empty text, a doubled separator, or one at
		// either end. Refuse it here, where its position is still known.
		if part == "" {
			return Lineup{}, fmt.Errorf("opponent lineup entry %d is empty", i+1)
		}
		factory, err := parse(part)
		if err != nil {
			if len(parts) == 1 {
				return Lineup{}, err
			}
			return Lineup{}, fmt.Errorf("opponent lineup entry %d: %w", i+1, err)
		}
		factories = append(factories, factory)
	}
	return NewLineup(factories...)
}

// parseLineupText reads a create request's opponent text with the built-in
// agent types. It is the protocol server's production parser.
func parseLineupText(text string) (Lineup, error) {
	return ParseLineup(text, agent.ParseFactory)
}

// String returns the canonical text form recorded as a run's opponent: each
// entry's canonical specification, joined by LineupSeparator. A one-entry
// lineup is that specification alone, as before lineups existed.
func (l Lineup) String() string {
	specs := make([]string, len(l.factories))
	for i, factory := range l.factories {
		specs[i] = factory.Spec().String()
	}
	return strings.Join(specs, LineupSeparator)
}

// validateFor checks that every factory is usable and that the lineup fits a
// map with this many players: one entry for every seat, or exactly one entry
// per opponent seat.
func (l Lineup) validateFor(players int) error {
	if err := checkFactories(l.factories); err != nil {
		return err
	}
	switch count := len(l.factories); {
	case count == 1 || count == players-1:
		return nil
	default:
		return fmt.Errorf(
			"opponent lineup names %d agents; a %d-player map takes 1 (every seat) or %d (one per opponent seat)",
			count, players, players-1,
		)
	}
}

// forRole returns the factory for the opponent seat holding role, as seen
// from the learner. role is never RoleMe.
//
// A one-entry lineup serves every role. Otherwise validateFor has already
// required one entry per opponent seat, which on a four-player map is three:
// RoleCW, RoleOpposite, and RoleCCW are 1, 2, and 3, so role-1 indexes
// entries 0, 1, and 2. A two-player map seats only RoleOpposite and takes a
// one-entry lineup, so it never reaches the index.
func (l Lineup) forRole(role Role) agent.Factory {
	if len(l.factories) == 1 {
		return l.factories[0]
	}
	return l.factories[role-1]
}

// allSameSpec reports whether every factory has the same canonical
// specification.
func allSameSpec(factories []agent.Factory) bool {
	first := factories[0].Spec().String()
	for _, factory := range factories[1:] {
		if factory.Spec().String() != first {
			return false
		}
	}
	return true
}
