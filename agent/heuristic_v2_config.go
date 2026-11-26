package agent

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"slices"
)

// heuristic-v2's configuration: its weights, its optional restrictions, the
// factory that validates them, and the spec they are written as. The bot's
// turn policy is in heuristic_v2.go.

// HeuristicV2Default is the one pinned heuristic-v2 specification, in
// canonical form: the opponent the strength check and, once accepted,
// training and the release gate play. Its weights are not tuned yet; tuning
// changes this constant on purpose, and every result records the spec it ran.
const HeuristicV2Default = "heuristic-v2:candidates=4,enemy_bonus=1,margin=0.1"

// heuristicV2DefaultRandomMove is the rule for Epsilon's random moves when a
// spec sets none: the pinned random-v1 opponent's settings, so a random move
// is then exactly the move that opponent would make.
var heuristicV2DefaultRandomMove = RandomConfig{
	LaunchProbability: 0.3,
	MinFraction:       0.2,
	MaxFraction:       0.8,
}

// GarrisonCost says which ships of a whole-garrison launch heuristic-v2's
// score subtracts as the launch's cost. It matters only with WholeGarrison.
type GarrisonCost int

const (
	// GarrisonCostSent counts every ship sent as the cost, as heuristic-v2
	// always has.
	GarrisonCostSent GarrisonCost = iota
	// GarrisonCostNeeded counts only the ships the capture needs, margin
	// included, as the cost. The rest land on the captured body and become
	// its garrison, so they are not lost.
	GarrisonCostNeeded
)

// String returns the spec spelling: "sent" or "needed".
func (c GarrisonCost) String() string {
	switch c {
	case GarrisonCostSent:
		return "sent"
	case GarrisonCostNeeded:
		return "needed"
	default:
		return fmt.Sprintf("GarrisonCost(%d)", int(c))
	}
}

// ReserveRule says what a whole-garrison bot does at a body that, by the
// forecast, needs defenders at home. It matters only with WholeGarrison.
type ReserveRule int

const (
	// ReserveHold launches nothing from such a body this turn.
	ReserveHold ReserveRule = iota
	// ReserveIgnore launches the whole garrison anyway.
	ReserveIgnore
)

// String returns the spec spelling: "hold" or "ignore".
func (r ReserveRule) String() string {
	switch r {
	case ReserveHold:
		return "hold"
	case ReserveIgnore:
		return "ignore"
	default:
		return fmt.Sprintf("ReserveRule(%d)", int(r))
	}
}

// HeuristicV2Config holds heuristic-v2's tunable weights and its optional
// restrictions.
//
// The restrictions make the bot play under the learner's controls (see
// mlenv.Decode) or weaken it on purpose. Every one has a zero value that
// keeps v2's own behaviour, so a config that sets none plays exactly as v2
// always has.
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
	// straight-line estimate, are aimed ahead (see heuristic_v2.go). It bounds
	// the cost of a turn; at least 1.
	Candidates int

	// WholeGarrison makes every launch send all of the source's ships, as the
	// learner's decoder does. The bot can still hold. False sends v2's own
	// priced fleet.
	WholeGarrison bool
	// GarrisonCost is which ships of a whole-garrison launch its score
	// subtracts as the cost. It must be GarrisonCostSent unless WholeGarrison
	// is set.
	GarrisonCost GarrisonCost
	// Reserve is what a whole-garrison bot does when the forecast says a
	// body needs defenders. It must be ReserveHold unless WholeGarrison is
	// set.
	Reserve ReserveRule
	// LearnerAim aims each launch once with its chosen fleet, never resizing
	// it, and aims straight when it can't aim ahead, as the learner's decoder
	// does (see heuristic_v2.go for both terms). False keeps v2's aiming: up
	// to two attempts, and a target it can't aim ahead at is skipped.
	LearnerAim bool
	// NoAimCheck commits to a target before aiming, choosing by the
	// straight-line estimate alone, so an aim result can neither cancel nor
	// change the launch. It requires LearnerAim.
	NoAimCheck bool
	// Epsilon is the chance, drawn separately for each owned body each turn,
	// that the body's move comes from random-v1's per-body rule instead of
	// v2's. 0 keeps v2 deterministic; 1 makes every move random. In [0,1].
	Epsilon float64
	// RandomMove is the random-v1 rule Epsilon's moves use. Nil means the
	// pinned random-v1 opponent's settings. It must be nil when Epsilon is 0,
	// because it would change nothing.
	RandomMove *RandomConfig
}

// HeuristicV2Factory is an immutable, validated heuristic-v2 configuration.
type HeuristicV2Factory struct {
	// config is the validated configuration. Its RandomMove, when set, points
	// at the factory's own copy, so a caller can't change it afterwards.
	config HeuristicV2Config
	// randomMove is the rule Epsilon's random moves use: config.RandomMove,
	// or heuristicV2DefaultRandomMove when that is nil. Unused when Epsilon
	// is 0.
	randomMove RandomConfig
	spec       Spec
}

// NewHeuristicV2Factory validates every weight and restriction before any
// match begins.
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
	if config.GarrisonCost != GarrisonCostSent && config.GarrisonCost != GarrisonCostNeeded {
		return nil, fmt.Errorf("garrison_cost %v is not a known rule", config.GarrisonCost)
	}
	if config.Reserve != ReserveHold && config.Reserve != ReserveIgnore {
		return nil, fmt.Errorf("reserve %v is not a known rule", config.Reserve)
	}
	// A restriction that would change nothing is refused rather than
	// ignored, so a spec never claims a setting the bot doesn't apply.
	if !config.WholeGarrison && (config.GarrisonCost != GarrisonCostSent || config.Reserve != ReserveHold) {
		return nil, fmt.Errorf("garrison_cost and reserve apply only with whole_garrison=1")
	}
	if config.NoAimCheck && !config.LearnerAim {
		return nil, fmt.Errorf("no_aim_check requires learner_aim=1")
	}
	if !finite(config.Epsilon) || config.Epsilon < 0 || config.Epsilon > 1 {
		return nil, fmt.Errorf("epsilon must be finite and in [0,1]")
	}
	randomMove := heuristicV2DefaultRandomMove
	if config.RandomMove != nil {
		if config.Epsilon == 0 {
			return nil, fmt.Errorf("epsilon_* settings require epsilon above 0")
		}
		if err := validateRandomConfig(*config.RandomMove, "epsilon_"); err != nil {
			return nil, err
		}
		randomMove = *config.RandomMove
		config.RandomMove = &randomMove
	}
	return &HeuristicV2Factory{
		config:     config,
		randomMove: randomMove,
		spec:       NewSpec(HeuristicV2, heuristicV2SpecParameters(config, randomMove)),
	}, nil
}

// heuristicV2SpecParameters builds the canonical parameters of a validated
// config. The three weights are always written. Every other setting is
// written only when it differs from its default, so a spec from before the
// settings existed keeps its exact identity, and with it the seeds derived
// from that identity and the results recorded under it.
func heuristicV2SpecParameters(config HeuristicV2Config, randomMove RandomConfig) map[string]string {
	parameters := map[string]string{
		"margin":      canonicalFloat(config.Margin),
		"enemy_bonus": canonicalFloat(config.EnemyBonus),
		"candidates":  fmt.Sprint(config.Candidates),
	}
	if config.WholeGarrison {
		parameters["whole_garrison"] = "1"
	}
	if config.GarrisonCost != GarrisonCostSent {
		parameters["garrison_cost"] = config.GarrisonCost.String()
	}
	if config.Reserve != ReserveHold {
		parameters["reserve"] = config.Reserve.String()
	}
	if config.LearnerAim {
		parameters["learner_aim"] = "1"
	}
	if config.NoAimCheck {
		parameters["no_aim_check"] = "1"
	}
	if config.Epsilon != 0 {
		parameters["epsilon"] = canonicalFloat(config.Epsilon)
	}
	// randomMove differs from the default only when Epsilon is above 0,
	// because validation refuses a RandomMove without it.
	defaults := heuristicV2DefaultRandomMove
	if randomMove.LaunchProbability != defaults.LaunchProbability {
		parameters["epsilon_launch_probability"] = canonicalFloat(randomMove.LaunchProbability)
	}
	if randomMove.MinFraction != defaults.MinFraction {
		parameters["epsilon_min_fraction"] = canonicalFloat(randomMove.MinFraction)
	}
	if randomMove.MaxFraction != defaults.MaxFraction {
		parameters["epsilon_max_fraction"] = canonicalFloat(randomMove.MaxFraction)
	}
	return parameters
}

// Spec returns the canonical validated configuration.
func (factory *HeuristicV2Factory) Spec() Spec { return factory.spec }

// New constructs a per-match heuristic-v2 agent. Without Epsilon it keeps no
// state between turns. With Epsilon it keeps a private random source, seeded
// from init.AgentSeed, so concurrent matches can't disturb one another's draws
// and every game replays exactly.
func (factory *HeuristicV2Factory) New(ctx context.Context, init AgentInit) (Agent, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	agent := &HeuristicV2Agent{config: factory.config, randomMove: factory.randomMove}
	if factory.config.Epsilon > 0 {
		agent.rng = rand.New(rand.NewSource(init.AgentSeed))
	}
	return agent, nil
}

// heuristicV2Weights are the parameters every heuristic-v2 spec must name.
var heuristicV2Weights = []string{"margin", "enemy_bonus", "candidates"}

// heuristicV2Settings are its optional restrictions. Each one left out keeps
// v2's own behaviour (see HeuristicV2Config).
var heuristicV2Settings = []string{
	"whole_garrison", "garrison_cost", "reserve", "learner_aim", "no_aim_check",
	"epsilon", "epsilon_launch_probability", "epsilon_min_fraction", "epsilon_max_fraction",
}

// parseHeuristicV2Config reads heuristic-v2's parameter list into a config.
// It checks the types of values and which settings were named together;
// NewHeuristicV2Factory then checks their ranges.
//
// The stages run in this order:
//
//	read each key=value in input order, converting it to its type
//	  → require the three weights
//	  → check settings that apply only alongside another one
//	  → assemble the config
func parseHeuristicV2Config(input string) (HeuristicV2Config, error) {
	// Stage 1: read. Each value is converted as soon as it is read, so the
	// first bad entry in the input is the one reported.
	var config HeuristicV2Config
	named := make(map[string]bool)
	numbers := make(map[string]float64)
	err := scanParameters(input, slices.Concat(heuristicV2Weights, heuristicV2Settings), func(name, valueText string) error {
		named[name] = true
		var err error
		switch name {
		case "garrison_cost":
			config.GarrisonCost, err = parseGarrisonCost(valueText)
		case "reserve":
			config.Reserve, err = parseReserveRule(valueText)
		case "whole_garrison":
			config.WholeGarrison, err = parseSwitch(name, valueText)
		case "learner_aim":
			config.LearnerAim, err = parseSwitch(name, valueText)
		case "no_aim_check":
			config.NoAimCheck, err = parseSwitch(name, valueText)
		default:
			numbers[name], err = parseNumber(name, valueText)
		}
		return err
	})
	if err != nil {
		return HeuristicV2Config{}, err
	}

	// Stage 2: the weights have no default.
	for _, name := range heuristicV2Weights {
		if !named[name] {
			return HeuristicV2Config{}, fmt.Errorf("missing parameter %q", name)
		}
	}
	// candidates is a count, so it must be a whole number; 4.5 is rejected
	// rather than rounded, so a spec means exactly one thing.
	candidates := numbers["candidates"]
	if candidates != math.Trunc(candidates) || candidates < 1 || candidates > math.MaxInt32 {
		return HeuristicV2Config{}, fmt.Errorf("candidates must be a whole number of at least 1")
	}
	config.Margin = numbers["margin"]
	config.EnemyBonus = numbers["enemy_bonus"]
	config.Candidates = int(candidates)
	config.Epsilon = numbers["epsilon"]

	// Stage 3: settings that apply only alongside another. These look at
	// what the spec names, not only at values: garrison_cost=sent without
	// whole_garrison=1 is refused too, because it would claim a setting the
	// bot doesn't apply. NewHeuristicV2Factory repeats the value checks for
	// configs built in code.
	if (named["garrison_cost"] || named["reserve"]) && !config.WholeGarrison {
		return HeuristicV2Config{}, fmt.Errorf("garrison_cost and reserve apply only with whole_garrison=1")
	}
	randomNames := []string{"epsilon_launch_probability", "epsilon_min_fraction", "epsilon_max_fraction"}
	if !slices.ContainsFunc(randomNames, func(name string) bool { return named[name] }) {
		return config, nil
	}
	if config.Epsilon == 0 {
		return HeuristicV2Config{}, fmt.Errorf("epsilon_* settings require epsilon above 0")
	}

	// Stage 4: an epsilon_* setting left out keeps the pinned random-v1
	// value, so naming one changes only that one.
	randomMove := heuristicV2DefaultRandomMove
	if named["epsilon_launch_probability"] {
		randomMove.LaunchProbability = numbers["epsilon_launch_probability"]
	}
	if named["epsilon_min_fraction"] {
		randomMove.MinFraction = numbers["epsilon_min_fraction"]
	}
	if named["epsilon_max_fraction"] {
		randomMove.MaxFraction = numbers["epsilon_max_fraction"]
	}
	config.RandomMove = &randomMove
	return config, nil
}

// parseSwitch reads an on/off setting written as 0 or 1. Any other number is
// refused rather than rounded, so a spec means exactly one thing.
func parseSwitch(name, valueText string) (bool, error) {
	value, err := parseNumber(name, valueText)
	if err != nil {
		return false, err
	}
	switch value {
	case 0:
		return false, nil
	case 1:
		return true, nil
	default:
		return false, fmt.Errorf("%s must be 0 or 1", name)
	}
}

// parseGarrisonCost reads garrison_cost's spec spelling.
func parseGarrisonCost(valueText string) (GarrisonCost, error) {
	for _, cost := range []GarrisonCost{GarrisonCostSent, GarrisonCostNeeded} {
		if valueText == cost.String() {
			return cost, nil
		}
	}
	return 0, fmt.Errorf("garrison_cost must be %q or %q, got %q", GarrisonCostSent, GarrisonCostNeeded, valueText)
}

// parseReserveRule reads reserve's spec spelling.
func parseReserveRule(valueText string) (ReserveRule, error) {
	for _, rule := range []ReserveRule{ReserveHold, ReserveIgnore} {
		if valueText == rule.String() {
			return rule, nil
		}
	}
	return 0, fmt.Errorf("reserve must be %q or %q, got %q", ReserveHold, ReserveIgnore, valueText)
}
