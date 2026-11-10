package agent

import (
	"context"
	"fmt"
	"math"
	"sort"

	"orrery/game"
)

// heuristic-v2's turn policy. Like v1 it sees only its View, uses no
// randomness, and takes bodies in ID order, so one board always gives the
// same moves. Unlike v1 it asks the engine's prediction helpers (see
// game.NewPredictor) where fleets will go:
//
//  1. Look ahead: predict where every fleet in flight lands, and when.
//  2. Defend: keep at home enough ships to survive the enemy fleets predicted
//     to land on each owned body; only the rest may launch.
//  3. Price: for each body it does not own, the ships needed to take it when
//     a fleet would arrive (see price).
//  4. Aim: ask LeadAim for that fleet size. A target with no verified aim is
//     skipped; v2 never falls back to aiming at the current position.
//  5. Choose: score each reachable target by the production it gains over
//     the rest of the game against the ships it costs, and send exactly the
//     ships needed.
//  6. Coordinate: a target this player's fleets in flight are already
//     predicted to take is never attacked again, and once a source claims a
//     target this turn, no later source attacks it.
//
// Defense, pricing and claims all come from one forecast of each body's
// predicted landings, fought with the engine's own combat rule (see
// forecast).
//
// Each source launches at most once per turn, and only when the best score is
// positive. To keep a turn cheap, only a source's Candidates best targets by a
// straight-line estimate are aimed with LeadAim.
//
// The forecast sees only fleets already flying, not launches other players
// make later. The policy does not reinforce threatened bodies from elsewhere
// or split a garrison between several targets.

// HeuristicV2 is the stable identity of the prediction-based strategy.
const HeuristicV2 = "heuristic-v2"

// HeuristicV2Default is the one pinned heuristic-v2 specification, in
// canonical form: the opponent the strength check and, once accepted,
// training and the release gate play. Its weights are not tuned yet; tuning
// changes this constant on purpose, and every result records the spec it ran.
const HeuristicV2Default = "heuristic-v2:candidates=4,enemy_bonus=1,margin=0.1"

// heuristicV2LookAhead is the most turns ahead fleet landings are predicted,
// the same horizon the observation's incoming-fleet columns use. Near the end
// of the game the horizon shrinks to the turns left (see Decide). A fleet
// landing later is ignored for defense, pricing and claims.
const heuristicV2LookAhead = 100

// HeuristicV2Config holds heuristic-v2's tunable weights.
type HeuristicV2Config struct {
	// Margin is the fraction of the bare minimum sent on top of it, so a
	// slightly wrong estimate still captures. 0 sends exactly the minimum
	// that the forecast says captures the target.
	Margin float64
	// EnemyBonus scales up the gain from taking a player's body, because that
	// also removes the player's production: the gain is multiplied by
	// 1+EnemyBonus. 0 values enemy and neutral bodies alike.
	EnemyBonus float64
	// Candidates is how many targets per source, best first by a
	// straight-line estimate, are aimed with LeadAim. It bounds the cost of a
	// turn; at least 1.
	Candidates int
}

// HeuristicV2Factory is an immutable, validated heuristic-v2 configuration.
type HeuristicV2Factory struct {
	config HeuristicV2Config
	spec   Spec
}

// NewHeuristicV2Factory validates every weight before any match begins.
func NewHeuristicV2Factory(config HeuristicV2Config) (*HeuristicV2Factory, error) {
	if !finite(config.Margin) || config.Margin < 0 {
		return nil, fmt.Errorf("margin must be finite and non-negative")
	}
	if !finite(config.EnemyBonus) || config.EnemyBonus < 0 {
		return nil, fmt.Errorf("enemy_bonus must be finite and non-negative")
	}
	if config.Candidates < 1 {
		return nil, fmt.Errorf("candidates must be at least 1")
	}
	return &HeuristicV2Factory{
		config: config,
		spec: NewSpec(HeuristicV2, map[string]string{
			"margin":      canonicalFloat(config.Margin),
			"enemy_bonus": canonicalFloat(config.EnemyBonus),
			"candidates":  fmt.Sprint(config.Candidates),
		}),
	}, nil
}

// Spec returns the canonical validated configuration.
func (factory *HeuristicV2Factory) Spec() Spec { return factory.spec }

// New constructs a per-match heuristic-v2 agent. It keeps no state between
// turns.
func (factory *HeuristicV2Factory) New(ctx context.Context, _ AgentInit) (Agent, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &HeuristicV2Agent{config: factory.config}, nil
}

// HeuristicV2Agent prices, aims and coordinates launches with the engine's
// prediction helpers.
type HeuristicV2Agent struct {
	config HeuristicV2Config
}

// ID returns the versioned agent identity.
func (*HeuristicV2Agent) ID() string { return HeuristicV2 }

// arrival is one predicted fleet landing on a body.
type arrival struct {
	owner int
	ships int
	// segment is Landing.Segment: 0 means the fleet lands during the next
	// AdvanceTurn. Every body produces once per AdvanceTurn before fleets
	// move, so a body produces segment+1 times before the landing's combat.
	segment int
}

// launchOption is one launch a source could make: a target, the fleet to
// send, its aim and how good it is.
type launchOption struct {
	target body
	ships  int // fleet size to send
	aim    game.Aim
	score  float64
}

// beats reports whether o is a better launch than other: a higher score, or
// an equal score on a lower target ID. The candidates are checked in estimate
// order, not ID order, so the tie must be compared, not assumed.
func (o launchOption) beats(other launchOption) bool {
	if o.score != other.score {
		return o.score > other.score
	}
	return o.target.id < other.target.id
}

// Decide emits at most one launch per owned body.
func (a *HeuristicV2Agent) Decide(ctx context.Context, request TurnRequest) (Decision, error) {
	if err := ctx.Err(); err != nil {
		return Decision{}, err
	}
	view := request.View
	bodies := sortedBodies(view)

	// Stage 1: look ahead. The predictor runs the engine's own movement and
	// collision code on a private copy of this observation. A comet already
	// past its path is gone before launches are checked (see sortedBodies),
	// and the predictor rejects one, so it is left out here.
	snapshot := view.Snapshot()
	snapshot.Comets = nil
	for _, comet := range view.Comets {
		if comet.PathIndex < len(comet.Path) {
			snapshot.Comets = append(snapshot.Comets, comet)
		}
	}
	predictor, err := game.NewPredictor(snapshot)
	if err != nil {
		return Decision{}, fmt.Errorf("%s: %w", HeuristicV2, err)
	}
	// PredictFleets ignores the end of the game, so stop it at the turn
	// limit: only MaxTurns-Turn more turns run, and a fleet landing in
	// segment s needs s+1 of them. A landing after that never happens, and
	// counting it would claim or defend against nothing. An early end by
	// elimination can't be known in advance.
	horizon := min(heuristicV2LookAhead, view.Config.MaxTurns-view.Turn)
	arrivals := predictedArrivals(view, predictor.PredictFleets(horizon))

	// A body this player's fleets already in flight are predicted to own once
	// every predicted landing has fought is claimed before any new launch is
	// chosen, so no source attacks it again.
	claimed := map[int]bool{}
	for _, target := range bodies {
		if target.ownerID == request.PlayerID {
			continue
		}
		if owner, _ := forecast(target, target.ships, arrivals[target.id], arrival{}, -1); owner == request.PlayerID {
			claimed[target.id] = true
		}
	}
	actions := make([]game.Action, 0)
	for _, source := range bodies {
		if source.ownerID != request.PlayerID || source.ships <= 0 {
			continue
		}
		// Stage 2: defend. Launches leave before this turn's production, so
		// only ships beyond the reserve may go.
		available := source.ships - reserve(source, arrivals[source.id])
		if available <= 0 {
			continue
		}
		// Stages 3–5 for this source.
		best, ok := a.bestTarget(view, predictor, source, available, bodies, arrivals, claimed)
		if !ok {
			continue
		}
		// Stage 6: coordinate. The claim is made as soon as the launch is
		// chosen, so a later source in ID order sees it.
		claimed[best.target.id] = true
		actions = append(actions, game.Action{
			PlayerID: request.PlayerID,
			PlanetID: source.id,
			Ships:    best.ships,
			Angle:    best.aim.Angle,
		})
	}
	return Decision{Actions: actions}, nil
}

// predictedArrivals groups the landings that hand ships to a body by that
// body's ID, each body's list in landing order (segment, then fleet order).
func predictedArrivals(view View, landings []game.Landing) map[int][]arrival {
	owners := make(map[int]int, len(view.Fleets))
	ships := make(map[int]int, len(view.Fleets))
	for _, fleet := range view.Fleets {
		owners[fleet.ID], ships[fleet.ID] = fleet.OwnerID, fleet.Ships
	}
	arrivals := map[int][]arrival{}
	for _, landing := range landings {
		if !landing.Arrives() {
			continue // destroyed by the sun or the edge, lost with a departing comet, or still flying
		}
		id := landing.Hit.BodyID
		arrivals[id] = append(arrivals[id], arrival{owner: owners[landing.FleetID], ships: ships[landing.FleetID], segment: landing.Segment})
	}
	for _, list := range arrivals {
		sort.SliceStable(list, func(i, j int) bool { return list[i].segment < list[j].segment })
	}
	return arrivals
}

// forecast replays the landings predicted for one body, turn by turn, and
// returns who owns it afterwards. The body starts with its owner and the given
// garrison. extra is a fleet being considered, added to the landings of its
// segment; ships 0 means none. The replay stops after segment through's
// battle, or after the last landing when through is negative. held reports
// whether the starting owner kept the body after every battle.
//
// Each turn mirrors the engine: the owner's production first (only a player's
// body produces), then every fleet landing that turn fights together through
// game.Battle, the engine's own combat rule. Grouping a turn's landings before
// the battle is what lets same-turn fleets fight each other first, and makes
// the result independent of fleet order.
//
// The forecast knows only fleets already flying, plus extra. Launches players
// make on later turns are unknowable from this observation.
func forecast(b body, garrison int, incoming []arrival, extra arrival, through int) (owner int, held bool) {
	landings := incoming
	if extra.ships > 0 {
		// Insert extra after the landings of earlier or equal segments, so the
		// list stays in landing order. A copy keeps incoming untouched.
		at := sort.Search(len(incoming), func(i int) bool { return incoming[i].segment > extra.segment })
		landings = append(append(append(make([]arrival, 0, len(incoming)+1), incoming[:at]...), extra), incoming[at:]...)
	}

	owner, ships, held := b.ownerID, garrison, true
	produced := 0 // turns of production already added
	forces := map[int]int{}
	for i := 0; i < len(landings); {
		segment := landings[i].segment
		if through >= 0 && segment > through {
			break
		}
		// A body produces once per turn before that turn's combat, so by the
		// battle of segment s it has produced s+1 times since now.
		if owner != game.NeutralPlayer {
			ships += b.production * (segment + 1 - produced)
		}
		produced = segment + 1
		clear(forces)
		for ; i < len(landings) && landings[i].segment == segment; i++ {
			forces[landings[i].owner] += landings[i].ships
		}
		owner, ships = game.Battle(owner, ships, forces)
		if owner != b.ownerID {
			held = false
		}
	}
	return owner, held
}

// reserve is how many of an owned body's ships must stay home so that it
// survives every landing predicted for it: the fewest that forecast says hold
// the body. More ships at home can only help, so a binary search finds that
// number. A body that can't be held even with every ship keeps them all and
// fights.
func reserve(source body, incoming []arrival) int {
	return sort.Search(source.ships, func(kept int) bool {
		_, held := forecast(source, kept, incoming, arrival{}, -1)
		return held
	})
}

// bestTarget runs stages 3–5 for one source and returns its best positive-
// scoring launch, if any.
func (a *HeuristicV2Agent) bestTarget(view View, predictor *game.Predictor, source body, available int,
	bodies []body, arrivals map[int][]arrival, claimed map[int]bool) (launchOption, bool) {
	// First rank every target by a straight-line estimate at the speed of
	// the whole available force, which is cheap. Only the best few are then
	// aimed with LeadAim, which runs a full simulation each time.
	var candidates []launchOption
	for _, target := range bodies {
		if target.ownerID == source.ownerID || claimed[target.id] {
			continue
		}
		distance := max(source.position.Distance(target.position)-source.radius-target.radius, 0)
		turns := distance / view.FleetSpeed(available)
		ships, ok := a.price(source.ownerID, target, arrivals[target.id], int(turns), available)
		if !ok {
			continue // out of reach of this force
		}
		candidates = append(candidates, launchOption{target: target, ships: ships, score: a.score(view, target, ships, turns)})
	}
	// Best first, lower ID on equal scores; the stable sort keeps body order.
	sort.SliceStable(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })
	if len(candidates) > a.config.Candidates {
		candidates = candidates[:a.config.Candidates]
	}

	var best launchOption
	found := false
	for _, c := range candidates {
		aimed, ok := a.aim(view, predictor, source, available, c, arrivals[c.target.id])
		if !ok || aimed.score <= 0 {
			continue
		}
		if !found || aimed.beats(best) {
			best, found = aimed, true
		}
	}
	return best, found
}

// aim verifies one candidate with LeadAim and re-prices it at the arrival
// time LeadAim reports. A fleet's speed depends on its size, so if the real
// arrival needs more ships than the estimate, it aims once more with that many
// and accepts only if they then suffice.
func (a *HeuristicV2Agent) aim(view View, predictor *game.Predictor, source body, available int, c launchOption, incoming []arrival) (launchOption, bool) {
	ships := c.ships
	for range 2 {
		aim, ok := predictor.LeadAim(source.id, c.target.id, ships)
		if !ok {
			return launchOption{}, false
		}
		// Aim.Turns is Landing.Segment plus the fraction of the arrival turn,
		// so its whole part is the arrival's segment.
		needed, ok := a.price(source.ownerID, c.target, incoming, int(aim.Turns), available)
		if !ok {
			return launchOption{}, false
		}
		if needed <= ships {
			return launchOption{target: c.target, ships: ships, aim: aim, score: a.score(view, c.target, ships, aim.Turns)}, true
		}
		ships = needed
	}
	return launchOption{}, false
}

// price is how many ships to send so that a fleet landing on target in
// segment makes player its owner after that turn's battle, and whether at
// most available ships can do it.
//
// The minimum is the fewest ships for which forecast ends with player owning
// the target, fighting alongside every other fleet predicted to land by then.
// A bigger fleet can only do better, so a binary search finds it. Margin then
// adds a fraction of that minimum, as far as available allows.
func (a *HeuristicV2Agent) price(player int, target body, incoming []arrival, segment, available int) (int, bool) {
	minimum := 1 + sort.Search(available, func(i int) bool {
		owner, _ := forecast(target, target.ships, incoming, arrival{owner: player, ships: i + 1, segment: segment}, segment)
		return owner == player
	})
	if minimum > available {
		return 0, false
	}
	return min(minimum+int(math.Ceil(a.config.Margin*float64(minimum))), available), true
}

// score is the production a capture gains over the turns left after the
// fleet lands, minus the ships it costs; both are counts of ships. A comet
// produces only until it leaves the board, and taking a player's body counts
// 1+EnemyBonus times.
func (a *HeuristicV2Agent) score(view View, target body, ships int, turns float64) float64 {
	left := float64(view.Config.MaxTurns - view.Turn)
	for _, comet := range view.Comets {
		if comet.ID == target.id {
			left = min(left, float64(len(comet.Path)-1-comet.PathIndex))
		}
	}
	gain := float64(target.production) * max(left-turns, 0)
	if target.ownerID != game.NeutralPlayer {
		gain *= 1 + a.config.EnemyBonus
	}
	return gain - float64(ships)
}
