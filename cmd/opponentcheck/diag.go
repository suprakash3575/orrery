package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"math/rand"
	"sort"

	"orrery/agent"
	"orrery/game"
)

// The two diagnostic agents play like a barely trained learner: each turn,
// every owned body with ships launches with probability p, sending all of its
// ships at a uniformly random target. They differ only in aiming. diagDirect
// aims at the target's current position, as heuristic-v1 does; diagLead aims
// with Game.LeadAim, as the learner's action decoding does. Comparing them
// shows what lead aiming is worth to such a player against heuristic-v1.
//
// They exist only in this harness. They are not registered with
// agent.ParseFactory, so no existing agent name or behaviour changes.

const (
	diagDirectID = "diag-direct-v1"
	diagLeadID   = "diag-lead-v1"

	// diagSeedDomain separates the diagnostic agents' seeds from every other
	// hash in the project. Its bytes are part of how a run is reproduced.
	diagSeedDomain = "orrery-opponent-check-diag-v1"
)

// diagSeed is the diagnostic agents' private random seed for one game: the
// first eight bytes of SHA-256 over the domain text, then the scenario seed
// and the tested seat as big-endian 64-bit integers.
//
// The agent's name is deliberately left out, unlike arena.DeriveAgentSeed,
// which hashes the agent's spec. Both diagnostic agents therefore get the same
// stream for the same game, so they make the same choices on the same state.
func diagSeed(scenarioSeed int64, seat int) int64 {
	var input [len(diagSeedDomain) + 16]byte
	offset := copy(input[:], diagSeedDomain)
	binary.BigEndian.PutUint64(input[offset:], uint64(scenarioSeed))
	binary.BigEndian.PutUint64(input[offset+8:], uint64(seat))
	digest := sha256.Sum256(input[:])
	return int64(binary.BigEndian.Uint64(digest[:8]))
}

// diagAgent is diagDirect or diagLead. It reads the turn's View for the board
// and, when lead is set, the live game for LeadAim. LeadAim works on a private
// copy of the board, so this access never changes the game.
type diagAgent struct {
	id string
	// lead selects LeadAim aiming; false aims at the target's current position.
	lead bool
	// live is the match's own game, used read-only by LeadAim. The match loop
	// asks every agent for its turn before it advances the game, so live and
	// the View describe the same moment.
	live *game.Game
	// launchProbability is p: the chance each eligible source launches.
	launchProbability float64
	rng               *rand.Rand
}

// newDiagAgent builds diagDirect (lead false) or diagLead (lead true) for one
// game, seeded with diagSeed.
func newDiagAgent(lead bool, live *game.Game, launchProbability float64, scenarioSeed int64, seat int) *diagAgent {
	id := diagDirectID
	if lead {
		id = diagLeadID
	}
	return &diagAgent{
		id: id, lead: lead, live: live, launchProbability: launchProbability,
		rng: rand.New(rand.NewSource(diagSeed(scenarioSeed, seat))),
	}
}

func (a *diagAgent) ID() string { return a.id }

// diagBody is the part of a planet or comet the diagnostic agents need.
type diagBody struct {
	id       int
	owner    int
	ships    int
	position game.Vec2
}

// diagBodies lists every planet and comet in the View in ascending body ID,
// which fixes the order of both sources and targets.
func diagBodies(view agent.View) []diagBody {
	bodies := make([]diagBody, 0, len(view.Planets)+len(view.Comets))
	for _, p := range view.Planets {
		bodies = append(bodies, diagBody{id: p.ID, owner: p.OwnerID, ships: p.Ships, position: p.Position})
	}
	for _, c := range view.Comets {
		bodies = append(bodies, diagBody{id: c.ID, owner: c.OwnerID, ships: c.Ships, position: c.Position})
	}
	sort.Slice(bodies, func(i, j int) bool { return bodies[i].id < bodies[j].id })
	return bodies
}

// diagChoice is one launch decision before it is aimed: which body launches
// and which body it is sent at.
type diagChoice struct {
	source, target diagBody
}

// Decide makes the turn's launches: choose, then aim each choice and send all
// of the source's ships.
func (a *diagAgent) Decide(ctx context.Context, request agent.TurnRequest) (agent.Decision, error) {
	if err := ctx.Err(); err != nil {
		return agent.Decision{}, err
	}
	var actions []game.Action
	for _, c := range a.choose(request.View, request.PlayerID) {
		actions = append(actions, game.Action{
			PlayerID: request.PlayerID,
			PlanetID: c.source.id,
			Ships:    c.source.ships,
			Angle:    a.heading(c.source, c.target),
		})
	}
	return agent.Decision{Actions: actions}, nil
}

// choose makes the turn's random choices. For every source (a body player
// owns with at least one ship, in ascending ID) it draws, in this order:
//
//  1. one launch roll, launching if it is below p;
//  2. if launching, one target draw, uniform over the legal targets: every
//     other planet and comet, friendly ones included, the source excluded.
//
// No other draw is made: the heading comes from the aim, not the stream. So
// both agents, on the same state, make the same choices.
func (a *diagAgent) choose(view agent.View, player int) []diagChoice {
	bodies := diagBodies(view)
	var choices []diagChoice
	for i, source := range bodies {
		if source.owner != player || source.ships < 1 {
			continue
		}
		if a.rng.Float64() >= a.launchProbability {
			continue
		}
		// The legal targets are every body except the source, still in ID
		// order: draw an index into that list by skipping the source's slot.
		pick := a.rng.Intn(len(bodies) - 1)
		if pick >= i {
			pick++
		}
		choices = append(choices, diagChoice{source: source, target: bodies[pick]})
	}
	return choices
}

// heading is the launch angle from source towards target. diagLead takes
// LeadAim's verified heading; otherwise, and whenever LeadAim finds none, the
// fleet is aimed at the target's current centre, as the learner's decoder
// falls back.
func (a *diagAgent) heading(source, target diagBody) float64 {
	if a.lead {
		if aim, ok := a.live.LeadAim(source.id, target.id, source.ships); ok {
			return aim.Angle
		}
	}
	delta := target.position.Sub(source.position)
	return math.Atan2(delta.Y, delta.X)
}
