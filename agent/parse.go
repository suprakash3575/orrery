package agent

import (
	"fmt"
	"math"
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
		parameters, err := parseFloatParameters(rawParameters, "margin", "enemy_bonus", "candidates")
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", HeuristicV2, err)
		}
		// candidates is a count, so it must be a whole number; 4.5 is
		// rejected rather than rounded, so a spec means exactly one thing.
		candidates := parameters["candidates"]
		if candidates != math.Trunc(candidates) || candidates < 1 || candidates > math.MaxInt32 {
			return nil, fmt.Errorf("parse %s: candidates must be a whole number of at least 1", HeuristicV2)
		}
		return NewHeuristicV2Factory(HeuristicV2Config{
			Margin:     parameters["margin"],
			EnemyBonus: parameters["enemy_bonus"],
			Candidates: int(candidates),
		})
	default:
		return nil, fmt.Errorf("unknown agent type %q", id)
	}
}

func parseFloatParameters(input string, requiredNames ...string) (map[string]float64, error) {
	if input == "" {
		return nil, fmt.Errorf("parameter list must not be empty")
	}
	required := make(map[string]bool, len(requiredNames))
	for _, name := range requiredNames {
		required[name] = true
	}

	parameters := make(map[string]float64, len(requiredNames))
	remaining := input
	for {
		// Cut consumes one assignment without allocating a slice for every
		// comma-separated parameter. Processing a final empty assignment also
		// preserves the existing rejection of a trailing comma.
		assignment, rest, hasNext := strings.Cut(remaining, ",")
		name, valueText, hasValue := strings.Cut(assignment, "=")
		if !hasValue || strings.Contains(valueText, "=") {
			return nil, fmt.Errorf("parameter %q must use key=value", assignment)
		}
		if name == "" || valueText == "" {
			return nil, fmt.Errorf("parameter %q must have a name and value", assignment)
		}
		if !required[name] {
			return nil, fmt.Errorf("unknown parameter %q", name)
		}
		if _, exists := parameters[name]; exists {
			return nil, fmt.Errorf("duplicate parameter %q", name)
		}
		value, err := strconv.ParseFloat(valueText, 64)
		if err != nil {
			return nil, fmt.Errorf("parameter %q: %w", name, err)
		}
		parameters[name] = value

		if !hasNext {
			break
		}
		remaining = rest
	}

	for _, name := range requiredNames {
		if _, exists := parameters[name]; !exists {
			return nil, fmt.Errorf("missing parameter %q", name)
		}
	}
	return parameters, nil
}
