package agent

import (
	"context"
	"fmt"
	"math"

	"orrery/game"
)

// heuristic-v1's turn policy:
//
//  1. Collect active planets and comets in stable body-ID order.
//  2. For each owned body with ships, reserve ceil(Defense * ships).
//  3. Skip the source if no ships remain available to launch.
//  4. Consider every neutral or enemy body as a possible target.
//  5. Estimate travel time from edge-to-edge distance at the available
//     fleet's authoritative speed.
//  6. Score each target as:
//     Aggression*(available-targetShips) + Economy*targetProduction -
//     Efficiency*travelTurns.
//  7. Select the highest-scoring target, resolving an exact score tie in
//     favor of the lower target ID.
//  8. Return one action that launches every available ship directly toward
//     that target. The arena later batches it with every other player's
//     actions, and the game engine validates and applies the launch.
//  9. Repeat independently for every owned body.
//
// The policy intentionally does not predict hazards, lead moving targets,
// account for incoming fleets, or coordinate launches between sources.

// HeuristicV1 is the stable identity of the first deterministic strategy.
const HeuristicV1 = "heuristic-v1"

// HeuristicConfig weights target strength, production, travel time, and the
// fraction of each garrison held back for defense.
type HeuristicConfig struct {
	Aggression float64 // weight for the attacking force's advantage over the target garrison
	Economy    float64 // weight for a target's production value
	Efficiency float64 // penalty weight for the estimated travel time to a target
	Defense    float64 // fraction of each owned garrison reserved instead of launched
}

// HeuristicFactory is an immutable, validated heuristic configuration.
type HeuristicFactory struct {
	config HeuristicConfig
	spec   Spec
}

// NewHeuristicFactory validates every weight before any match begins.
func NewHeuristicFactory(config HeuristicConfig) (*HeuristicFactory, error) {
	weights := []struct {
		name  string
		value float64
	}{
		{"aggression", config.Aggression},
		{"economy", config.Economy},
		{"efficiency", config.Efficiency},
	}
	for _, weight := range weights {
		if !finite(weight.value) || weight.value < 0 {
			return nil, fmt.Errorf("%s must be finite and non-negative", weight.name)
		}
	}
	if !finite(config.Defense) || config.Defense < 0 || config.Defense > 1 {
		return nil, fmt.Errorf("defense must be finite and in [0,1]")
	}

	return &HeuristicFactory{
		config: config,
		spec: NewSpec(HeuristicV1, map[string]string{
			"aggression": canonicalFloat(config.Aggression),
			"economy":    canonicalFloat(config.Economy),
			"efficiency": canonicalFloat(config.Efficiency),
			"defense":    canonicalFloat(config.Defense),
		}),
	}, nil
}

// Spec returns the canonical validated configuration.
func (factory *HeuristicFactory) Spec() Spec { return factory.spec }

// New constructs a stateless per-match heuristic agent.
func (factory *HeuristicFactory) New(ctx context.Context, _ AgentInit) (Agent, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &HeuristicAgent{config: factory.config}, nil
}

// HeuristicAgent directly aims available ships at its highest-scoring body.
type HeuristicAgent struct {
	config HeuristicConfig
}

// ID returns the versioned heuristic-agent identity.
func (*HeuristicAgent) ID() string { return HeuristicV1 }

// Decide emits at most one all-available launch per owned body. For an
// available force A and target T, heuristic-v1 scores:
//
//	aggression*(A - T.ships) + economy*T.production - efficiency*travelTurns
//
// where travelTurns is the unobstructed edge-to-edge distance divided by the
// authoritative fleet speed. This is target ranking, not collision
// prediction: the direct heading deliberately ignores intervening hazards and
// does not lead moving bodies. Exact score ties go to the lower target ID.
func (a *HeuristicAgent) Decide(ctx context.Context, request TurnRequest) (Decision, error) {
	if err := ctx.Err(); err != nil {
		return Decision{}, err
	}

	// The request is a snapshot of the current turn. Put planets and comets in
	// body-ID order before choosing actions so slice order cannot change a tie.
	bodies := sortedBodies(request.View)
	actions := make([]game.Action, 0)
	// Each owned body makes one independent choice. Skipping an empty or foreign
	// body means it cannot supply ships for an action this turn.
	for _, source := range bodies {
		if source.ownerID != request.PlayerID || source.ships <= 0 {
			continue
		}

		// Defense is the fraction to keep at home. Round the reserve up so a
		// small garrison does not accidentally leave fewer defenders than asked.
		reserve := int(math.Ceil(a.config.Defense * float64(source.ships)))
		available := source.ships - reserve
		if available <= 0 {
			continue
		}

		// A source with no neutral or enemy body to attack stays idle.
		target, ok := a.bestTarget(request.View, source, available, bodies)
		if !ok {
			continue
		}
		// The engine receives a launch angle, not a target ID. Aim from the
		// source's current position toward the chosen body's current position;
		// the engine later validates and applies this action with both players'.
		delta := target.position.Sub(source.position)
		actions = append(actions, game.Action{
			PlayerID: request.PlayerID,
			PlanetID: source.id,
			Ships:    available,
			Angle:    math.Atan2(delta.Y, delta.X),
		})
	}
	return Decision{Actions: actions}, nil
}

// bestTarget ranks every body this source could attack and reports whether it
// found one. The score is only a quick preference estimate: it uses a straight,
// unobstructed trip and does not simulate collisions or future body movement.
func (a *HeuristicAgent) bestTarget(view View, source body, available int, bodies []body) (body, bool) {
	var best body
	bestScore := math.Inf(-1)
	found := false
	// Start below every possible score and replace the winner as candidates
	// arrive. found distinguishes "no legal target" from a real zero-value body.
	for _, target := range bodies {
		if target.ownerID == source.ownerID {
			continue
		}
		// Fleets leave from the source edge and arrive at the target edge, so
		// subtract both radii from the center-to-center distance. Clamp at zero
		// if the edges already touch rather than estimating negative travel time.
		distance := source.position.Distance(target.position) - source.radius - target.radius
		if distance < 0 {
			distance = 0
		}
		// Fleet speed depends on the number of ships this source can send.
		// Favor a larger advantage and more production; penalize a longer trip.
		travelTurns := distance / view.FleetSpeed(available)
		score := a.config.Aggression*float64(available-target.ships) +
			a.config.Economy*float64(target.production) -
			a.config.Efficiency*travelTurns
		// Bodies arrive in ascending ID order, so retaining the first target
		// on an equal score is the lower-ID tie-break.
		if !found || score > bestScore {
			best = target
			bestScore = score
			found = true
		}
	}
	return best, found
}
