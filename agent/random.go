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
			return nil, fmt.Errorf("%s must be finite and in [0,1]", candidate.name)
		}
	}
	if config.MinFraction > config.MaxFraction {
		return nil, fmt.Errorf("min_fraction must not exceed max_fraction")
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
		if a.rng.Float64() >= a.config.LaunchProbability {
			continue
		}

		fraction := a.config.MinFraction
		if a.config.MinFraction < a.config.MaxFraction {
			fraction += a.rng.Float64() * (a.config.MaxFraction - a.config.MinFraction)
		}
		ships := int(math.Round(fraction * float64(source.ships)))
		if ships <= 0 {
			continue
		}
		if ships > source.ships {
			ships = source.ships
		}

		actions = append(actions, game.Action{
			PlayerID: request.PlayerID,
			PlanetID: source.id,
			Ships:    ships,
			Angle:    a.rng.Float64() * 2 * math.Pi,
		})
	}
	return Decision{Actions: actions}, nil
}
