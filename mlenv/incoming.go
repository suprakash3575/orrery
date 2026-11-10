package mlenv

import (
	"fmt"
	"math"

	"orrery/game"
)

// maxPredictionSegments bounds predictFleets' look-ahead at the longest crossing
// any legal fleet's straight-line path could physically need on this board:
// the board's own diagonal, at the slowest speed any fleet ever travels
// (FleetSpeed floors at 1 unit/turn for a one-ship fleet).
//
// This is deliberately derived from the game's own Config, not from any
// normalization limit used to scale a feature into 0..1. A normalization limit
// is a display choice a caller may tune, even far below the board's true
// diagonal; using it to bound this loop would make every ordinary
// long-distance fleet fail to resolve, not because anything about the fleet is
// wrong, but because a display setting quietly became a correctness bound. The
// two concerns are kept apart on purpose.
func maxPredictionSegments(g *game.Game) int {
	return int(math.Ceil(math.Hypot(g.Config.Width, g.Config.Height)))
}

// predictFleets predicts every fleet in g.Fleets, in the same order, and
// fails if any is still flying after maxPredictionSegments turns. It has no
// opinion on ownership or which body slot a landing maps to — those are the
// caller's concerns.
//
// An unresolved fleet is an error rather than a fleet that lands nowhere:
// every straight path leaves the board within that bound, so reaching it means
// the state is not one the engine can produce.
func predictFleets(g *game.Game) ([]game.Landing, error) {
	horizon := maxPredictionSegments(g)
	landings := g.PredictFleets(horizon)
	for _, l := range landings {
		if !l.Resolved {
			return nil, fmt.Errorf("fleet %d did not resolve within %d turns, the longest crossing this board's geometry allows",
				l.FleetID, horizon)
		}
	}
	return landings, nil
}
