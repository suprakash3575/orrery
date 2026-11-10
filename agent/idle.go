package agent

import (
	"context"
)

// IdleV1 is the stable identity of the no-op reference agent.
const IdleV1 = "idle-v1"

// IdleFactory constructs stateless idle agents.
type IdleFactory struct{}

// Spec returns the idle agent's parameter-free identity.
func (IdleFactory) Spec() Spec { return NewSpec(IdleV1, nil) }

// New returns a fresh idle agent unless construction was already cancelled.
func (IdleFactory) New(ctx context.Context, _ AgentInit) (Agent, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return IdleAgent{}, nil
}

// IdleAgent always chooses a legal no-op turn. It is useful for exercising
// match coordination and state progression without introducing strategy.
type IdleAgent struct{}

// ID returns the versioned idle-agent identity.
func (IdleAgent) ID() string { return IdleV1 }

// Decide returns no actions unless the match has been cancelled.
func (IdleAgent) Decide(ctx context.Context, _ TurnRequest) (Decision, error) {
	if err := ctx.Err(); err != nil {
		return Decision{}, err
	}
	return Decision{}, nil
}
