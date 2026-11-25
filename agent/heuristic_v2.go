package agent

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"orrery/game"
)

// heuristic-v2's turn policy. Like v1 it sees only its View, uses no
// randomness unless Epsilon is set, and takes bodies in ID order, so one board
// always gives the same moves. Unlike v1 it asks the engine's prediction helpers (see
// game.NewPredictor) where fleets will go:
//
//  1. Look ahead: predict where every fleet in flight lands, and when.
//  2. Defend: keep at home enough ships to survive the enemy fleets predicted
//     to land on each owned body; only the rest may launch.
//  3. Price: for each body it does not own, the ships needed to take it when
//     a fleet would arrive (see price).
//  4. Aim: aim ahead with that fleet size (see below). A target it can't aim
//     ahead at is skipped; v2 never aims straight.
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
// straight-line estimate are aimed.
//
// The forecast sees only fleets already flying, not launches other players
// make later. The policy does not reinforce threatened bodies from elsewhere
// or split a garrison between several targets.
//
// Aiming, in this file's words:
//
//   - aim ahead: fly towards where the target will be when the fleet arrives,
//     since bodies move. The engine's game.LeadAim works out that direction
//     by simulating the flight, so it also says when the fleet lands. It can
//     find no way to get there, for example when the path crosses the sun.
//   - aim straight: fly towards where the target is now. Simple, but a moving
//     target may have left by the time the fleet arrives, so the fleet can
//     miss, or hit something else.
//
// Normal v2 only aims ahead, and skips a target it can't aim ahead at. The
// learner aims ahead when it can and aims straight otherwise; LearnerAim makes
// v2 do the same.
//
// HeuristicV2Config's optional restrictions change some stages, so the bot can
// play under the learner's controls:
//
//   - WholeGarrison (stages 2, 3 and 5): a body sends every ship or none.
//     Reserve says whether a body that needs defenders holds, and
//     GarrisonCost whether the score subtracts every ship sent as the
//     launch's cost, or only those the capture needs.
//   - LearnerAim (stage 4): aim once with the fleet already chosen, and aim
//     straight when it can't aim ahead, instead of skipping the target.
//   - NoAimCheck (stages 4 and 5): choose the target from the straight-line
//     estimate alone, then aim it, so an aim result can't change the choice.
//   - Epsilon (before stage 2): a dial that makes v2 weaker on purpose.
//     Every turn, before planning a planet it owns, v2 rolls a die. With
//     chance Epsilon, that planet makes a random move (random-v1's rule)
//     instead of v2's planned one. For example, Epsilon 0.25 means about one
//     planet in four moves randomly each turn. 0 is normal v2, and 1 is fully
//     random. It makes opponents that sit between random-v1 and normal v2.

// HeuristicV2 is the stable identity of the prediction-based strategy.
const HeuristicV2 = "heuristic-v2"

// heuristicV2LookAhead is the most turns ahead fleet landings are predicted,
// the same horizon the observation's incoming-fleet columns use. Near the end
// of the game the horizon shrinks to the turns left (see Decide). A fleet
// landing later is ignored for defense, pricing and claims.
const heuristicV2LookAhead = 100

// HeuristicV2Agent prices, aims and coordinates launches with the engine's
// prediction helpers.
type HeuristicV2Agent struct {
	config HeuristicV2Config
	// randomMove is the random-v1 rule Epsilon's random moves use (see
	// HeuristicV2Factory.randomMove).
	randomMove RandomConfig
	// rng is this match's private random source, seeded from
	// AgentInit.AgentSeed so a game replays exactly. It exists only when
	// Epsilon is above 0; otherwise it is nil and v2 draws nothing.
	rng *rand.Rand
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
	// cost is the ships the score subtracts for this launch. It equals ships,
	// except with WholeGarrison and GarrisonCostNeeded, where it is only the
	// ships the capture needs (see fleet).
	cost  int
	aim   game.Aim
	score float64
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
	landings, err := sharedLandings(view, request.Prediction, horizon)
	if err != nil {
		return Decision{}, fmt.Errorf("%s: %w", HeuristicV2, err)
	}
	if landings == nil {
		landings = predictor.PredictFleets(horizon)
	}
	arrivals := predictedArrivals(view, landings)

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
		// Epsilon: roll the die for this planet. If it says "random", make
		// random-v1's move here and skip v2's planning. A random launch
		// doesn't mark its target as taken, so v2's other planets may still
		// attack that target.
		if a.randomTurn() {
			if action, ok := randomMove(a.rng, a.randomMove, request.PlayerID, source); ok {
				actions = append(actions, action)
			}
			continue
		}
		// Stage 2: defend. Launches leave before this turn's production, so
		// only ships beyond the reserve may go.
		kept := reserve(source, arrivals[source.id])
		available := source.ships - kept
		if a.config.WholeGarrison {
			// The learner can only hold or send every ship, so a body keeps
			// all of them or none. With ReserveHold a body that needs
			// defenders holds; with ReserveIgnore it may launch them all.
			if kept > 0 && a.config.Reserve == ReserveHold {
				continue
			}
			available = source.ships
		}
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

// randomTurn rolls the Epsilon die for one planet: true means this planet
// makes a random move this turn.
//
// A random move is random-v1's move (see randomMove): no targets, no
// planning. With the epsilon_* settings at their defaults it usually holds,
// and sometimes sends a random share of the planet's ships in a random
// direction, where they may hit nothing, a neutral planet or the sun. For a
// planet with 100 ships, that might be "hold" this turn and "45 ships towards
// 210°" the next.
//
// The die is a random number from 0 to 1, and the answer is "random" when it
// is below Epsilon. With Epsilon 0.25, a roll of 0.1 says random and a roll of
// 0.7 says plan as v2.
//
// It rolls once for every planet it is asked about, whatever the answer. So
// which planets move randomly depends only on the seed, never on what v2
// decided, and a game replays exactly.
//
// At Epsilon 0 and 1 the answer is always the same, so it doesn't roll. That
// keeps Epsilon 1 exactly random-v1: random-v1 makes no such roll, so the
// two draw the same numbers in the same order and make the same moves.
func (a *HeuristicV2Agent) randomTurn() bool {
	switch {
	case a.config.Epsilon <= 0:
		return false
	case a.config.Epsilon >= 1:
		return true
	default:
		return a.rng.Float64() < a.config.Epsilon
	}
}

// predictedArrivals groups the landings that hand ships to a body by that
// body's ID, each body's list in landing order (segment, then fleet order).
//
// A landing names only a fleet and where and when it lands, so the fleet's
// owner and size are looked up from the view. For example, if fleet 7 (player
// 1, 15 ships) lands on body 3 in segment 2, and fleet 9 (player 0, 4 ships)
// lands there in segment 0, the result has
//
//	3 → [{owner 0, 4 ships, segment 0}, {owner 1, 15 ships, segment 2}]
//
// Everything that later reasons about a body's future (forecast, reserve,
// price) reads only that body's list.
func predictedArrivals(view View, landings []game.Landing) map[int][]arrival {
	// Look up each fleet's owner and size once, by fleet ID.
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
	// Landings arrive in fleet order, not time order. Sort each body's list
	// by segment; the stable sort keeps fleet order within one segment, so
	// the lists never depend on map iteration.
	for _, list := range arrivals {
		sort.SliceStable(list, func(i, j int) bool { return list[i].segment < list[j].segment })
	}
	return arrivals
}

// sharedLandings turns a caller's prediction into exactly what
// predictor.PredictFleets(horizon) would return on this view: one landing per
// fleet, in View.Fleets order, with nothing resolved at or beyond horizon. It
// returns nil, and no error, when there is no prediction or its horizon is
// shorter than the one wanted; the caller then predicts for itself.
//
// Why dropping later landings is exact: the prediction simulates one turn at a
// time and never looks past the turn it is on, so a fleet that lands in
// segment s is reported the same way by every horizon above s. A fleet that
// lands at or after horizon is one PredictFleets(horizon) reports as
// unresolved, with every other field zero.
//
// A prediction that names a different set of fleets than the view is an error
// rather than a reason to predict again: it means the caller predicted another
// board, and silently ignoring that would hide the bug.
func sharedLandings(view View, prediction *FleetPrediction, horizon int) ([]game.Landing, error) {
	if prediction == nil || prediction.Horizon < horizon {
		return nil, nil
	}
	if len(prediction.Landings) != len(view.Fleets) {
		return nil, fmt.Errorf("shared prediction has %d landings for %d fleets", len(prediction.Landings), len(view.Fleets))
	}
	byFleet := make(map[int]game.Landing, len(prediction.Landings))
	for _, landing := range prediction.Landings {
		if _, repeated := byFleet[landing.FleetID]; repeated {
			return nil, fmt.Errorf("shared prediction repeats fleet %d", landing.FleetID)
		}
		byFleet[landing.FleetID] = landing
	}
	landings := make([]game.Landing, len(view.Fleets))
	for i, fleet := range view.Fleets {
		landing, ok := byFleet[fleet.ID]
		if !ok {
			return nil, fmt.Errorf("shared prediction has no landing for fleet %d", fleet.ID)
		}
		if !landing.Resolved || landing.Segment >= horizon {
			landing = game.Landing{FleetID: fleet.ID}
		}
		landings[i] = landing
	}
	return landings, nil
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
//
// Example: player 0 owns a body with a garrison of 10 and production 2, and
// 15 enemy ships land in segment 1, the second turn from now.
//
//	segment 1: production twice (turns 0 and 1)  10 + 2×2 = 14 ships
//	           battle: 15 attackers against 14    the enemy wins with 1
//
// forecast returns the enemy as owner, and held = false. With a garrison of
// 13 instead, 13 + 4 = 17 beats 15, so player 0 keeps it with 2.
//
// Three callers ask it different questions:
//
//	Decide   (no extra, whole list)      who owns a target once every fleet now
//	                                     flying has landed? (claims)
//	reserve  (no extra, whole list)      is this garrison enough to keep my body?
//	price    (extra = my new fleet,      does a fleet of this size capture the
//	          stop after its battle)     target when it lands?
func forecast(b body, garrison int, incoming []arrival, extra arrival, through int) (owner int, held bool) {
	// Stage 1: the landings to replay, with extra added in time order.
	landings := incoming
	if extra.ships > 0 {
		// incoming is sorted by segment, so a binary search finds at, the
		// first landing later than extra: segments [0 1 1 3] with extra in
		// segment 1 give at = 3, and the list [0 1 1 extra 3]. Order within a
		// segment doesn't change the battle, which adds them all up.
		at := sort.Search(len(incoming), func(i int) bool { return incoming[i].segment > extra.segment })
		landings = append(append(append(make([]arrival, 0, len(incoming)+1), incoming[:at]...), extra), incoming[at:]...)
	}

	// Stage 2: replay one battle per segment that has landings. Turns with
	// no landing need no battle; their production is added in one step at
	// the next battle, which is why produced tracks how much is already in.
	owner, ships, held := b.ownerID, garrison, true
	produced := 0 // turns of production already added
	forces := map[int]int{}
	for i := 0; i < len(landings); {
		segment := landings[i].segment
		if through >= 0 && segment > through {
			break // later landings can't change the battle price asks about
		}
		// A body produces once per turn before that turn's combat, so by the
		// battle of segment s it has produced s+1 times since now.
		if owner != game.NeutralPlayer {
			ships += b.production * (segment + 1 - produced)
		}
		produced = segment + 1
		// Add up every fleet landing this segment by owner; two fleets of
		// one player join forces. game.Battle then settles them and the
		// garrison exactly as the engine will.
		clear(forces)
		for ; i < len(landings) && landings[i].segment == segment; i++ {
			forces[landings[i].owner] += landings[i].ships
		}
		owner, ships = game.Battle(owner, ships, forces)
		// Losing the body even once counts as not held, even if a later
		// landing wins it back: the defender lost its production meanwhile.
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
//
// Example, with forecast's body (production 2, 15 enemy ships landing in
// segment 1, so the garrison grows by 4 before the battle):
//
//	kept 10 → 14 against 15: lost
//	kept 11 → 15 against 15: a tie, which leaves the owner in place: held
//
// so reserve is 11. With 30 ships, 19 may launch. A body with no landings
// predicted needs nothing kept: reserve is 0.
func reserve(source body, incoming []arrival) int {
	// sort.Search tries kept = 0 … ships-1 and returns the smallest kept for
	// which the body is held, or ships if none is enough.
	return sort.Search(source.ships, func(kept int) bool {
		_, held := forecast(source, kept, incoming, arrival{}, -1)
		return held
	})
}

// bestTarget runs stages 3–5 for one source and returns its best positive-
// scoring launch, if any.
//
// It works in two passes because aiming ahead is expensive: it simulates the
// fleet's whole flight, while a straight-line estimate is one division:
//
//	pass 1, every target:  estimate travel time, price, cost and score
//	pass 2, best few:      aim ahead, re-price at the real arrival time,
//	                       and keep the best positive score
//
// With Candidates = 3 and ten possible targets, ten are estimated but only
// three are aimed.
func (a *HeuristicV2Agent) bestTarget(view View, predictor *game.Predictor, source body, available int,
	bodies []body, arrivals map[int][]arrival, claimed map[int]bool) (launchOption, bool) {
	// Pass 1: rank every target by a straight-line estimate at the speed of
	// the whole available force, which is cheap. Only the best few are then
	// aimed ahead, which simulates a whole flight each time.
	var candidates []launchOption
	for _, target := range bodies {
		// Own bodies aren't targets, and neither is one already being taken
		// by this player's fleets (see Decide's claims).
		if target.ownerID == source.ownerID || claimed[target.id] {
			continue
		}
		// Fleets leave from the source's edge and land on the target's
		// edge, so the gap between the two edges is the distance flown.
		//
		// The estimate flies every available ship. Bigger fleets fly faster,
		// up to a top speed (see game.FleetSpeed), so this is the fastest
		// fleet the source can send and the earliest it could arrive. The
		// fleet v2 actually sends is usually smaller, so slower: it lands
		// later, when the target may have produced more ships, and aim
		// re-prices it at that real arrival. With WholeGarrison the fleet
		// is every available ship, so the estimate's speed is the real one.
		distance := max(source.position.Distance(target.position)-source.radius-target.radius, 0)
		turns := distance / view.FleetSpeed(available)
		needed, ok := a.price(source.ownerID, target, arrivals[target.id], int(turns), available)
		if !ok {
			continue // out of reach of this force
		}
		ships, cost := a.fleet(needed, available)
		candidates = append(candidates, launchOption{target: target, ships: ships, cost: cost, score: a.score(view, target, cost, turns)})
	}
	// Best first, lower ID on equal scores; the stable sort keeps body order.
	sort.SliceStable(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })

	// NoAimCheck: pick the target without aiming first, as the learner must.
	// The learner chooses a target, and only afterwards is its fleet aimed,
	// so it can't know whether the aim will work.
	//
	// So take pass 1's top-ranked target and launch at it if its score is
	// above 0, aiming ahead if it can and straight if not; a failed aim
	// never cancels the launch. Pass 2 is skipped, so Candidates has no
	// effect.
	if a.config.NoAimCheck {
		if len(candidates) == 0 || candidates[0].score <= 0 {
			return launchOption{}, false
		}
		chosen := candidates[0]
		chosen.aim, _ = aimAheadOrStraight(predictor, source, chosen.target, chosen.ships)
		return chosen, true
	}

	if len(candidates) > a.config.Candidates {
		candidates = candidates[:a.config.Candidates]
	}

	// Pass 2: aim the shortlist. The estimate's order is only a guess, since
	// a real flight can take longer and need more ships, so every shortlisted
	// target is aimed and the best aimed score wins. A target that can't be
	// aimed, or no longer pays for itself, drops out.
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

// aim aims one candidate ahead and re-prices it at the arrival time that
// reports. A fleet's speed depends on its size: bigger fleets fly faster (see
// game.FleetSpeed). The estimate assumed every available ship, the fastest
// fleet, so the smaller fleet actually sent lands later and may need more
// ships by then. If so, it aims once more with that many and accepts only if
// they then suffice.
//
// With WholeGarrison the fleet is already every available ship, and price
// never asks for more than available, so the first attempt always suffices:
// the fleet is never resized. Only its cost is re-priced at the real arrival.
//
// Example: the straight-line estimate said 11 ships, landing in 4 turns. But
// 11 ships fly slower than the whole force the estimate assumed, and the
// target moves, so the real flight is longer:
//
//	attempt 1: aim 11 ships → lands in 6 turns; by then 13 are needed
//	attempt 2: aim 13 ships → a different speed, lands in 6 turns; 13 suffice
//
// The launch is 13 ships in attempt 2's direction. If attempt 2 had needed
// more than 13, the target would be dropped: resizing could go on forever,
// because each new size changes the speed and so the arrival again.
//
// With LearnerAim, learnerAim aims instead.
func (a *HeuristicV2Agent) aim(view View, predictor *game.Predictor, source body, available int, c launchOption, incoming []arrival) (launchOption, bool) {
	if a.config.LearnerAim {
		return a.learnerAim(view, predictor, source, available, c, incoming)
	}
	ships := c.ships
	for range 2 {
		// Aim ahead, for a fleet of exactly this size. If it can't aim
		// ahead, v2 skips the target.
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
			// v2 sends the fleet it aimed, which may be more than needed, and
			// that whole fleet is the cost. A whole-garrison fleet's cost
			// comes from fleet.
			cost := ships
			if a.config.WholeGarrison {
				ships, cost = a.fleet(needed, available)
			}
			return launchOption{target: c.target, ships: ships, cost: cost, aim: aim, score: a.score(view, c.target, cost, aim.Turns)}, true
		}
		ships = needed
	}
	return launchOption{}, false
}

// learnerAim aims a launch the way the learner's controls allow, for the
// LearnerAim setting.
//
// The learner can't resize a fleet, and can't skip a target because aiming
// ahead fails: it picks a target, then its decoder (mlenv.Decode) aims the
// fleet it already chose. So this aims c's fleet once, as it is:
//
//   - It can aim ahead: check the fleet still captures the target when it
//     actually lands. If it does, keep the launch; if not, drop the target,
//     because the fleet size is fixed once chosen.
//   - It can't aim ahead: keep the launch anyway, and aim straight.
//
// Example, with WholeGarrison: c sends all 50 ships at a target the
// straight-line estimate says needs 11.
//
//	aimed ahead, lands in 6 turns, 13 needed then   → keep: 50 is enough
//	aimed ahead, lands in 9 turns, 60 needed then   → drop: 50 is too few
//	can't aim ahead                                 → keep, aimed straight
//
// v2's own aim would instead resize the fleet, or skip a target it can't aim
// ahead at (see aim).
func (a *HeuristicV2Agent) learnerAim(view View, predictor *game.Predictor, source body, available int, c launchOption, incoming []arrival) (launchOption, bool) {
	aim, aimedAhead := aimAheadOrStraight(predictor, source, c.target, c.ships)
	if !aimedAhead {
		// Aiming straight simulates no flight, so there is no real arrival
		// time to re-price at: keep pass 1's estimated price and score.
		c.aim = aim
		return c, true
	}
	// Re-price at the real arrival. Aim.Turns' whole part is the turn it
	// lands in (its segment), as in aim.
	needed, ok := a.price(source.ownerID, c.target, incoming, int(aim.Turns), available)
	if !ok || needed > c.ships {
		return launchOption{}, false // the fleet no longer captures
	}
	// Score at the real arrival. The cost is the fleet sent, except that a
	// whole-garrison fleet's cost comes from fleet.
	cost := c.ships
	if a.config.WholeGarrison {
		_, cost = a.fleet(needed, available)
	}
	return launchOption{target: c.target, ships: c.ships, cost: cost, aim: aim, score: a.score(view, c.target, cost, aim.Turns)}, true
}

// aimAheadOrStraight aims a fleet of ships from source at target the way the
// learner's decoder does: ahead if it can, straight if not. aimedAhead
// reports which.
//
//   - aimedAhead = true: the direction meets the target where it will be,
//     and aim.Turns is when the fleet lands.
//   - aimedAhead = false: the direction points at where the target is now,
//     and aim.Turns is 0, because no flight was simulated.
//
// Both are computed exactly as mlenv.Decode computes them, so the bot's
// launch and the decoder's are identical to the last bit.
func aimAheadOrStraight(predictor *game.Predictor, source, target body, ships int) (aim game.Aim, aimedAhead bool) {
	// game.LeadAim is the engine's aim-ahead helper.
	if aim, ok := predictor.LeadAim(source.id, target.id, ships); ok {
		return aim, true
	}
	// Aim straight: the angle of the line from the source's centre to the
	// target's.
	delta := target.position.Sub(source.position)
	return game.Aim{Angle: math.Atan2(delta.Y, delta.X)}, false
}

// fleet returns the ships to send for a capture priced at needed ships, from
// a source that can send available, and the cost: the ships the score
// subtracts for this launch.
//
// v2 sends exactly what the capture needs, and that is the cost.
// WholeGarrison sends every available ship. With GarrisonCostSent the cost is
// every ship sent; with GarrisonCostNeeded it is only those needed, because
// the rest land on the captured body as its new garrison and are not spent.
//
// Example: the capture needs 11 and the source has 50 available.
//
//	normal v2                        sends 11, cost 11
//	WholeGarrison, GarrisonCostSent    sends 50, cost 50
//	WholeGarrison, GarrisonCostNeeded  sends 50, cost 11
func (a *HeuristicV2Agent) fleet(needed, available int) (ships, cost int) {
	switch {
	case !a.config.WholeGarrison:
		return needed, needed
	case a.config.GarrisonCost == GarrisonCostNeeded:
		return available, needed
	default:
		return available, available
	}
}

// price is how many ships to send so that a fleet landing on target in
// segment makes player its owner after that turn's battle, and whether at
// most available ships can do it.
//
// The minimum is the fewest ships for which forecast ends with player owning
// the target, fighting alongside every other fleet predicted to land by then.
// A bigger fleet can only do better, so a binary search finds it. Margin then
// adds a fraction of that minimum, as far as available allows.
//
// Example: a neutral target with 10 ships, which never produces, and no other
// fleet landing on it. A fleet must strictly outnumber the garrison to take
// it, so 10 ships tie and leave it neutral, and 11 take it: the minimum is 11.
// If an enemy fleet of 6 lands in the same segment, the two attacking fleets
// fight first, so the minimum becomes 6 + 11 = 17.
//
// Margin covers small forecast errors: the result is the minimum plus
// ceil(Margin × minimum), but never more than available.
func (a *HeuristicV2Agent) price(player int, target body, incoming []arrival, segment, available int) (int, bool) {
	// sort.Search tries fleets of i+1 = 1 … available ships and returns the
	// first i whose fleet captures; it returns available when none does,
	// which makes minimum = available + 1, too many.
	minimum := 1 + sort.Search(available, func(i int) bool {
		owner, _ := forecast(target, target.ships, incoming, arrival{owner: player, ships: i + 1, segment: segment}, segment)
		return owner == player
	})
	if minimum > available {
		return 0, false // out of reach of this force
	}
	return min(minimum+int(math.Ceil(a.config.Margin*float64(minimum))), available), true
}

// score is the production a capture gains over the turns left after the
// fleet lands, minus cost, the ships the launch is counted as spending; both
// are counts of ships. A comet produces only until it leaves the board, and
// taking a player's body counts 1+EnemyBonus times.
//
// Example: 100 turns are left, the fleet lands in 4, the target produces 3
// ships a turn, and the launch costs 11.
//
//	neutral target:  3 × (100 − 4) − 11           = 288 − 11 = 277
//	enemy target:    3 × (100 − 4) × (1 + EnemyBonus) − 11
//
// An enemy body is worth more because taking it also stops its owner's
// production. Near the end of the game the gain shrinks towards 0, so a
// costly launch scores below 0 and the bot holds instead.
func (a *HeuristicV2Agent) score(view View, target body, cost int, turns float64) float64 {
	// Turns the captured body can still produce for: the rest of the game,
	// or for a comet, until it leaves the board, if that is sooner.
	left := float64(view.Config.MaxTurns - view.Turn)
	for _, comet := range view.Comets {
		if comet.ID == target.id {
			left = min(left, float64(len(comet.Path)-1-comet.PathIndex))
		}
	}
	// Production starts only once the fleet has landed, and a fleet landing
	// after the last productive turn gains nothing.
	gain := float64(target.production) * max(left-turns, 0)
	if target.ownerID != game.NeutralPlayer {
		gain *= 1 + a.config.EnemyBonus
	}
	return gain - float64(cost)
}
