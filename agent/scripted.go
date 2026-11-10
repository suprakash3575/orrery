package agent

import (
	"math"
	"sort"
	"strconv"

	"orrery/game"
)

// body is the common decision surface of planets and comets. Keeping this
// adapter in the agent package lets scripted agents treat both launch sources
// consistently without teaching the game engine anything about strategies.
type body struct {
	id         int
	position   game.Vec2
	radius     float64
	ownerID    int
	ships      int
	production int
}

// sortedBodies returns every capturable body in the engine's shared ID order.
// NewView sorts planets and comets independently, but their IDs share one
// namespace, so combining and sorting again makes action order independent of
// whether a source happens to be a planet or a comet.
func sortedBodies(view View) []body {
	bodies := make([]body, 0, len(view.Planets)+len(view.Comets))
	for _, planet := range view.Planets {
		bodies = append(bodies, body{
			id:         planet.ID,
			position:   planet.Position,
			radius:     planet.Radius,
			ownerID:    planet.OwnerID,
			ships:      planet.Ships,
			production: planet.Production,
		})
	}
	for _, comet := range view.Comets {
		// The engine removes a comet within the turn it steps past the end of
		// its path, so a View built from a live game never holds one. A
		// hand-built View can, and AdvanceTurn's first phase would expire it
		// before launches are checked, so it is neither a legal launch source
		// nor a real target for this decision.
		if comet.PathIndex >= len(comet.Path) {
			continue
		}
		bodies = append(bodies, body{
			id:         comet.ID,
			position:   comet.Position,
			radius:     comet.Radius,
			ownerID:    comet.OwnerID,
			ships:      comet.Ships,
			production: comet.Production,
		})
	}
	sort.Slice(bodies, func(i, j int) bool { return bodies[i].id < bodies[j].id })
	return bodies
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// canonicalFloat produces a round-trippable, locale-independent parameter
// value. Normalizing signed zero prevents equivalent configurations from
// acquiring different identities and, later, different derived seeds.
func canonicalFloat(value float64) string {
	if value == 0 {
		value = 0
	}
	return strconv.FormatFloat(value, 'g', -1, 64)
}
