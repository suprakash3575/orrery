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
// A longer lineup names one agent per opponent seat, counted from the learner:
// entry 0 plays the first seat after the learner's, entry 1 the second, and so
// on, wrapping from the last seat back to seat 0. The learner's seat changes
// from game to game, and counting from it gives every game the same mix of
// opponents. Fixed seat numbers would not: whichever fixed seat the learner
// took would lose its agent, and the mix would depend on the learner's seat.
//
// For example, with the learner in seat 2 of a four-player game, the lineup
// "a|b|c" seats a in seat 3, b in seat 0, and c in seat 1.
type Lineup struct {
	// factories holds one agent factory per opponent seat, in seat order
	// counted from the learner, or a single factory for every opponent seat.
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

// NewLineup builds a lineup from factories in seat order counted from the
// learner. It refuses an empty list, a nil factory (including a typed nil),
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

// forSeat returns the factory for an opponent seat. seat and learner are
// player IDs on a map with players seats, and seat is never the learner's.
//
// The offset counts how many seats after the learner this seat comes, from 1
// to players-1. Adding players before taking the remainder keeps it positive
// when the seat number is lower than the learner's: with the learner in seat
// 2 of 4, seat 0 has offset (0 - 2 + 4) % 4 = 2, the second seat after it.
func (l Lineup) forSeat(seat, learner, players int) agent.Factory {
	if len(l.factories) == 1 {
		return l.factories[0]
	}
	offset := (seat - learner + players) % players
	return l.factories[offset-1]
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
