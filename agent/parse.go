package agent

import (
	"fmt"
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
