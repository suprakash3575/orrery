package agent

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
)

// ParseFactory converts one CLI agent specification into an immutable,
// validated factory. Parameter order and numeric spelling may vary on input;
// Factory.Spec returns the canonical identity used in seeds and results.
func ParseFactory(input string) (Factory, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return nil, fmt.Errorf("agent specification must not be empty")
	}

	id, rawParameters, hasParameters := strings.Cut(input, ":")
	switch id {
	case IdleV1:
		if hasParameters {
			return nil, fmt.Errorf("%s does not accept parameters", IdleV1)
		}
		return IdleFactory{}, nil
	case RandomV1:
		if !hasParameters {
			return nil, fmt.Errorf("%s requires parameters", RandomV1)
		}
		parameters, err := parseFloatParameters(rawParameters, "launch_probability", "min_fraction", "max_fraction")
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", RandomV1, err)
		}
		return NewRandomFactory(RandomConfig{
			LaunchProbability: parameters["launch_probability"],
			MinFraction:       parameters["min_fraction"],
			MaxFraction:       parameters["max_fraction"],
		})
	case HeuristicV1:
		if !hasParameters {
			return nil, fmt.Errorf("%s requires parameters", HeuristicV1)
		}
		parameters, err := parseFloatParameters(rawParameters, "aggression", "economy", "efficiency", "defense")
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", HeuristicV1, err)
		}
		return NewHeuristicFactory(HeuristicConfig{
			Aggression: parameters["aggression"],
			Economy:    parameters["economy"],
			Efficiency: parameters["efficiency"],
			Defense:    parameters["defense"],
		})
	case HeuristicV2:
		if !hasParameters {
			return nil, fmt.Errorf("%s requires parameters", HeuristicV2)
		}
		config, err := parseHeuristicV2Config(rawParameters)
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", HeuristicV2, err)
		}
		return NewHeuristicV2Factory(config)
	default:
		return nil, fmt.Errorf("unknown agent type %q", id)
	}
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

// parseNumber reads one numeric parameter value.
func parseNumber(name, valueText string) (float64, error) {
	value, err := strconv.ParseFloat(valueText, 64)
	if err != nil {
		return 0, fmt.Errorf("parameter %q: %w", name, err)
	}
	return value, nil
}

func parseFloatParameters(input string, requiredNames ...string) (map[string]float64, error) {
	parameters := make(map[string]float64, len(requiredNames))
	err := scanParameters(input, requiredNames, func(name, valueText string) error {
		value, err := parseNumber(name, valueText)
		parameters[name] = value
		return err
	})
	if err != nil {
		return nil, err
	}

	for _, name := range requiredNames {
		if _, exists := parameters[name]; !exists {
			return nil, fmt.Errorf("missing parameter %q", name)
		}
	}
	return parameters, nil
}

// scanParameters walks a comma-separated key=value list in input order. It
// refuses malformed, unknown and duplicate entries itself, and hands each
// value's text to visit, which converts it to that parameter's type.
// Converting inside the walk keeps errors in input order, whatever type each
// parameter has. visit's error stops the walk and is returned as it is.
func scanParameters(input string, allowed []string, visit func(name, valueText string) error) error {
	if input == "" {
		return fmt.Errorf("parameter list must not be empty")
	}
	known := make(map[string]bool, len(allowed))
	for _, name := range allowed {
		known[name] = true
	}

	seen := make(map[string]bool, len(allowed))
	remaining := input
	for {
		// Cut consumes one assignment without allocating a slice for every
		// comma-separated parameter. Processing a final empty assignment also
		// preserves the existing rejection of a trailing comma.
		assignment, rest, hasNext := strings.Cut(remaining, ",")
		name, valueText, hasValue := strings.Cut(assignment, "=")
		if !hasValue || strings.Contains(valueText, "=") {
			return fmt.Errorf("parameter %q must use key=value", assignment)
		}
		if name == "" || valueText == "" {
			return fmt.Errorf("parameter %q must have a name and value", assignment)
		}
		if !known[name] {
			return fmt.Errorf("unknown parameter %q", name)
		}
		if seen[name] {
			return fmt.Errorf("duplicate parameter %q", name)
		}
		seen[name] = true
		if err := visit(name, valueText); err != nil {
			return err
		}

		if !hasNext {
			break
		}
		remaining = rest
	}
	return nil
}
