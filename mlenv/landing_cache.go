package mlenv

import (
	"fmt"

	"orrery/game"
)

// landingCache carries each fleet's predicted landing from one turn to the
// next, so a turn predicts only the fleets launched since the last one.
//
// Why a landing can be carried over: the prediction tests every fleet against
// the moving bodies alone, never against other fleets, and AdvanceTurn moves
// fleets and bodies with the same code the prediction runs. A fleet predicted
// at turn T to land in segment s, and still flying at T+1, is therefore
// predicted at T+1 to land in segment s-1 with the same hit — the turn that
// was segment 0 has simply happened. Two things break that, and each makes
// the cache start over:
//
//   - A comet wave. The prediction cannot see comets that have not spawned
//     yet, so every landing made before a wave may be wrong after it.
//   - Anything other than exactly one completed turn of the same game, such as
//     a reset. The cache only ever describes one game at one turn.
//
// The result is exactly what predictFleets(g) returns, landing for landing;
// a test compares the two every turn.
type landingCache struct {
	// game and turn identify the board the cached landings describe. A nil
	// game means the cache is empty.
	game *game.Game
	turn int
	// comets holds the IDs of the comets on that board. A comet ID missing
	// from it on the next turn means a wave spawned in between.
	comets map[int]bool
	// byFleet holds each fleet's landing as predicted at turn, keyed by
	// fleet ID.
	byFleet map[int]game.Landing
	// reused counts landings carried over instead of predicted again. It only
	// lets tests and benchmarks confirm the cache is doing something.
	reused int
}

// predict returns one landing per fleet in g.Fleets, in that order, looked
// ahead maxPredictionSegments(g) turns, and fails like predictFleets if any
// fleet is still flying after that.
//
// Stages:
//  1. Decide whether last turn's landings still apply to this board.
//  2. Carry over every landing that does, one segment closer.
//  3. Predict the rest in one engine call.
//  4. Remember this turn's landings for the next call.
func (c *landingCache) predict(g *game.Game) ([]game.Landing, error) {
	// Stage 1. Only the turn right after the cached one, on the same game,
	// with no new comet, can reuse anything.
	carry := c.game == g && g.Turn == c.turn+1
	for _, comet := range g.Comets {
		if carry && !c.comets[comet.ID] {
			carry = false
		}
	}

	// Stage 2. A carried landing in segment 0 belongs to a fleet that should
	// have landed during the turn just played. Finding it still in flight
	// means the board did not move as predicted, so nothing is reused rather
	// than trusting any of it.
	landings := make([]game.Landing, len(g.Fleets))
	var fresh []game.Fleet
	var freshAt []int
	reused := 0
	for i, fleet := range g.Fleets {
		if carry {
			if landing, ok := c.byFleet[fleet.ID]; ok {
				if landing.Segment == 0 {
					carry = false
				} else {
					landing.Segment--
					landings[i] = landing
					reused++
					continue
				}
			}
		}
		fresh = append(fresh, fleet)
		freshAt = append(freshAt, i)
	}
	if !carry {
		fresh, freshAt, reused = g.Fleets, nil, 0
	}

	// Stage 3. PredictFleets follows the fleets of the game it is called on.
	// A shallow copy with only the fresh fleets predicts just those: the copy
	// shares the bodies, which the prediction reads and never writes, and the
	// fleets cannot affect one another's landings.
	horizon := maxPredictionSegments(g)
	board := *g
	board.Fleets = fresh
	predicted := board.PredictFleets(horizon)
	for j, landing := range predicted {
		if !landing.Resolved {
			return nil, fmt.Errorf("fleet %d did not resolve within %d turns, the longest crossing this board's geometry allows",
				landing.FleetID, horizon)
		}
		if freshAt == nil {
			landings[j] = landing
		} else {
			landings[freshAt[j]] = landing
		}
	}

	// Stage 4. Replace the cache wholesale, so landings of fleets that have
	// landed or died since last turn are dropped.
	c.game, c.turn = g, g.Turn
	c.comets = make(map[int]bool, len(g.Comets))
	for _, comet := range g.Comets {
		c.comets[comet.ID] = true
	}
	c.byFleet = make(map[int]game.Landing, len(landings))
	for _, landing := range landings {
		c.byFleet[landing.FleetID] = landing
	}
	c.reused += reused
	return landings, nil
}
