package agent

import (
	"context"
	"fmt"
	"math"
	"math/rand"

	"orrery/game"
)

// random-v1's turn policy, applied independently to every owned planet or
// comet with ships, in stable body-ID order:
//
//  1. Roll a launch chance. LaunchProbability 0 never launches, 1 always
//     launches, and an intermediate value launches with that probability.
//  2. If launching, sample a fraction from MinFraction up to MaxFraction.
//     Equal bounds make the fraction fixed.
//  3. Send round(fraction * garrison ships). A zero-ship result is skipped.
//  4. Choose a uniformly random heading angle and return the launch action.
//
// For example, a 100-ship body with MinFraction 0.2 and MaxFraction 0.8 sends
// roughly 20 through 80 ships whenever its launch-chance roll succeeds. Each
// body makes its own rolls and may therefore launch or wait independently.

// RandomV1 is the stable identity of the first random scripted strategy.
const RandomV1 = "random-v1"

// RandomConfig controls independent launch choices for each owned body.
type RandomConfig struct {
	LaunchProbability float64 // chance that an eligible owned body launches this turn
	MinFraction       float64 // lower endpoint of the sampled launched-garrison fraction
	MaxFraction       float64 // upper endpoint of the sampled launched-garrison fraction
}

// RandomFactory is an immutable, validated random-agent configuration.
type RandomFactory struct {
	config RandomConfig
	spec   Spec
}

// NewRandomFactory validates a random strategy before any match begins.
func NewRandomFactory(config RandomConfig) (*RandomFactory, error) {
	if err := validateRandomConfig(config, ""); err != nil {
		return nil, err
	}

	return &RandomFactory{
		config: config,
		spec: NewSpec(RandomV1, map[string]string{
			"launch_probability": canonicalFloat(config.LaunchProbability),
			"min_fraction":       canonicalFloat(config.MinFraction),
			"max_fraction":       canonicalFloat(config.MaxFraction),
		}),
	}, nil
}

// validateRandomConfig checks random-v1's rule wherever it is configured.
// prefix is put in front of each parameter name in errors, so a rule set
// through other spec names, such as heuristic-v2's epsilon_* settings, is
// reported under the names the spec actually used.
func validateRandomConfig(config RandomConfig, prefix string) error {
	values := []struct {
		name  string
		value float64
	}{
		{"launch_probability", config.LaunchProbability},
		{"min_fraction", config.MinFraction},
		{"max_fraction", config.MaxFraction},
	}
	for _, candidate := range values {
		if !finite(candidate.value) || candidate.value < 0 || candidate.value > 1 {
			return fmt.Errorf("%s%s must be finite and in [0,1]", prefix, candidate.name)
		}
	}
	if config.MinFraction > config.MaxFraction {
		return fmt.Errorf("%smin_fraction must not exceed %smax_fraction", prefix, prefix)
	}
	return nil
}

// Spec returns the canonical validated configuration.
func (factory *RandomFactory) Spec() Spec { return factory.spec }

// New gives each match its own RNG so concurrent matches cannot perturb one
// another's decisions.
func (factory *RandomFactory) New(ctx context.Context, init AgentInit) (Agent, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &RandomAgent{
		config: factory.config,
		rng:    rand.New(rand.NewSource(init.AgentSeed)),
	}, nil
}

// RandomAgent makes bounded random launch decisions from private RNG state.
type RandomAgent struct {
	config RandomConfig
	rng    *rand.Rand
}

// ID returns the versioned random-agent identity.
func (*RandomAgent) ID() string { return RandomV1 }

// Decide considers each owned body once in stable ID order. The exact draw
// order is part of random-v1's replay behavior: launch roll, optional fraction
// roll, then optional angle roll for each source.
func (a *RandomAgent) Decide(ctx context.Context, request TurnRequest) (Decision, error) {
	if err := ctx.Err(); err != nil {
		return Decision{}, err
	}

	actions := make([]game.Action, 0)
	for _, source := range sortedBodies(request.View) {
		if source.ownerID != request.PlayerID || source.ships <= 0 {
			continue
		}
		if action, ok := randomMove(a.rng, a.config, request.PlayerID, source); ok {
			actions = append(actions, action)
		}
	}
	return Decision{Actions: actions}, nil
}

// randomMove applies random-v1's rule to one owned source with ships, and
// returns its launch, or false when it holds. heuristic-v2's Epsilon uses it
// too, so a random move there is exactly random-v1's move.
//
// The draws come from rng in a fixed order, which is part of random-v1's
// replay behaviour: the launch roll; then, only when launching, the fraction
// roll (skipped when the fraction is fixed) and the angle roll (skipped when
// the fraction rounds to zero ships).
func randomMove(rng *rand.Rand, config RandomConfig, player int, source body) (game.Action, bool) {
	if rng.Float64() >= config.LaunchProbability {
		return game.Action{}, false
	}

	fraction := config.MinFraction
	if config.MinFraction < config.MaxFraction {
		fraction += rng.Float64() * (config.MaxFraction - config.MinFraction)
	}
	ships := int(math.Round(fraction * float64(source.ships)))
	if ships <= 0 {
		return game.Action{}, false
	}
	if ships > source.ships {
		ships = source.ships
	}

	return game.Action{
		PlayerID: player,
		PlanetID: source.id,
		Ships:    ships,
		Angle:    rng.Float64() * 2 * math.Pi,
	}, true
}
