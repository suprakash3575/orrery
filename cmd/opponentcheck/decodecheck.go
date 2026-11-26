package main

import (
	"cmp"
	"errors"
	"fmt"
	"math"
	"slices"

	"orrery/agent"
	"orrery/game"
	"orrery/mlenv"
)

// The decode check (-check-decode) proves that a restricted heuristic-v2 bot
// made only moves the learner could have made. Each turn it takes the bot's
// choices (which body launched at which), writes them as the learner's slot
// actions, and runs them through the learner's own decoder, mlenv.Decode.
// The launches Decode builds must be exactly the bot's: same sources, same
// ship counts, same angles to the last bit. Any difference fails the game.
//
// Exact angles are expected, not merely close ones: the bot aims with
// Predictor.LeadAim on a copy of the board, and Decode with Game.LeadAim on
// the live game, which run the same code on the same state.

// restrictedSettings are the heuristic-v2 settings that limit the bot to the
// learner's controls: send every ship, aim once with the learner's fallback,
// and commit to a target before aiming. They are written as the canonical
// spec writes a setting that is on.
var restrictedSettings = []string{"whole_garrison", "learner_aim", "no_aim_check"}

// choiceReporter is the part of agent.HeuristicV2Agent the decode check
// reads: the launches of its last turn, as (source, target) body pairs.
type choiceReporter interface {
	LastChoices() []agent.LaunchChoice
}

// checkDecodeAllowed refuses -check-decode unless the tester is heuristic-v2
// with every restricted setting on and epsilon at 0. Any other bot can make
// moves the learner can't (a part of a garrison, a random heading), so the
// check would fail for reasons that say nothing about the decoder.
//
// It reads the canonical spec, which leaves out every setting at its
// default: a setting that is on appears as "1", and epsilon appears only
// when it is above 0.
func checkDecodeAllowed(tester agent.Factory) error {
	if tester == nil {
		return errors.New("-check-decode needs -tester-spec")
	}
	spec := tester.Spec()
	if spec.ID() != agent.HeuristicV2 {
		return fmt.Errorf("-check-decode needs a %s tester, not %s", agent.HeuristicV2, spec.ID())
	}
	parameters := map[string]string{}
	for _, p := range spec.Parameters() {
		parameters[p.Name] = p.Value
	}
	for _, name := range restrictedSettings {
		if parameters[name] != "1" {
			return fmt.Errorf("-check-decode needs a tester with %s=1; the tester is %q", name, spec.String())
		}
	}
	if _, ok := parameters["epsilon"]; ok {
		return fmt.Errorf("-check-decode needs a tester with epsilon=0; %q makes random moves", spec.String())
	}
	return nil
}

// checkDecode compares one turn of the tested bot with what the learner's
// decoder makes of its choices. live must be the board the bot just decided
// on, not yet advanced, and frame the tested seat's frame for this game.
//
// The stages are: match each choice to its action; skip the work for a turn
// with no launches; turn body IDs into observation slots; decode; compare.
func checkDecode(live *game.Game, frame mlenv.SeatFrame, choices []agent.LaunchChoice, actions []game.Action) error {
	// LastChoices()[i] describes actions[i], so the counts must agree before
	// anything else can be compared.
	if len(choices) != len(actions) {
		return fmt.Errorf("%d choices for %d launches", len(choices), len(actions))
	}
	// A turn with no launch decodes to no launch, so there is nothing to
	// build or compare. Building the observation every turn would only
	// cost time.
	if len(actions) == 0 {
		return nil
	}

	obs, err := mlenv.Build(live, frame)
	if err != nil {
		return fmt.Errorf("build observation: %w", err)
	}
	// slotOf finds the observation slot showing each body. Padding slots
	// have SlotBodyID 0, which is also a real body ID, so only slots the
	// body mask marks as real are used. The map is only looked up, never
	// iterated, so its order can't matter.
	slotOf := map[int]int{}
	for slot, id := range obs.SlotBodyID {
		if obs.BodyMask[slot] {
			slotOf[int(id)] = slot
		}
	}

	// The learner gives each source slot at most one launch, so a bot that
	// launched twice from one body made a move the learner can't make.
	var acts [mlenv.BodySlots]mlenv.SlotAction
	for i, c := range choices {
		source, sourceOK := slotOf[c.Source]
		target, targetOK := slotOf[c.Target]
		// A random move has Target agent.NoTarget, which no slot shows.
		if !sourceOK || !targetOK {
			return fmt.Errorf("launch %d: body %d → %d is not a launch between two bodies the learner sees", i, c.Source, c.Target)
		}
		if acts[source].Launch {
			return fmt.Errorf("launch %d: body %d launches twice in one turn", i, c.Source)
		}
		acts[source] = mlenv.SlotAction{Launch: true, Target: int32(target)}
	}
	decoded, err := mlenv.Decode(live, frame, obs, acts)
	if err != nil {
		return fmt.Errorf("decode: %w", err)
	}

	// Decode emits launches in slot order, and the bot in its own order.
	// Each source launches at most once, so sorting both by source pairs
	// each bot launch with its decoded twin.
	got := slices.Clone(actions)
	bySource := func(a, b game.Action) int { return cmp.Compare(a.PlanetID, b.PlanetID) }
	slices.SortFunc(got, bySource)
	slices.SortFunc(decoded, bySource)
	if len(decoded) != len(got) {
		return fmt.Errorf("decoder made %d launches from %d choices", len(decoded), len(got))
	}
	for i := range got {
		if !sameLaunch(got[i], decoded[i]) {
			return fmt.Errorf("body %d: bot launched %+v, decoder made %+v", got[i].PlanetID, got[i], decoded[i])
		}
	}
	return nil
}

// sameLaunch reports whether two launches are identical, the angle bit for
// bit. Comparing the structs with == isn't enough: Go's float == treats +0
// and −0 as equal although their bits differ, and an angle of exactly 0 can
// come out with either sign (math.Atan2(−0, 1) is −0).
func sameLaunch(a, b game.Action) bool {
	return a.PlayerID == b.PlayerID && a.PlanetID == b.PlanetID && a.Ships == b.Ships &&
		math.Float64bits(a.Angle) == math.Float64bits(b.Angle)
}
