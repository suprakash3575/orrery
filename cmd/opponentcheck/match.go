package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	"orrery/agent"
	"orrery/arena"
	"orrery/game"
	"orrery/scenario"
)

// players is the seat count on every map this harness plays: generator-v2-4p.
const players = 4

// Game statuses. Only completed games are scored; a failed or cancelled game
// is never counted as a win, loss or draw.
const (
	statusCompleted = "completed"
	statusFailed    = "failed"
	statusCancelled = "cancelled"
)

// job is one game: a tester in one seat of one generated map, against
// heuristic opponents in the other three seats.
type job struct {
	tester string
	seed   int64
	seat   int
}

// gameRecord is everything kept about one game. Fields after Status are
// meaningful only for a completed game, except FailTurn, FailSeat and
// Message, which describe a failed or cancelled one.
type gameRecord struct {
	Tester string `json:"tester"`
	Seed   int64  `json:"seed"`
	Seat   int    `json:"seat"`
	Status string `json:"status"`

	// Result is the tester's reward as mlenv scores it: +1 for finishing
	// strictly first by ships, 0 for a tie for first (or no owner left), -1
	// otherwise.
	Result int `json:"result"`
	// FinishTurn is the engine turn the game ended on.
	FinishTurn int `json:"finish_turn"`
	// Termination is "elimination" when at most one owner remained, and
	// "turn-limit" otherwise.
	Termination string `json:"termination"`
	// EliminatedTurn is the first turn after which the tester owned nothing,
	// or 0 if it was never eliminated. The game itself plays on.
	EliminatedTurn int `json:"eliminated_turn"`
	// Launches counts the tester's launch actions over the whole game.
	Launches int `json:"launches"`
	// Ships is every seat's final ship total, by seat; Leaders are the seats
	// tied for the most ships, in ascending order.
	Ships   [players]int `json:"ships"`
	Leaders []int        `json:"leaders"`
	// OpponentsEliminated counts the other seats that owned nothing at the end.
	OpponentsEliminated int `json:"opponents_eliminated"`

	FailTurn int    `json:"fail_turn,omitempty"`
	FailSeat *int   `json:"fail_seat,omitempty"`
	Message  string `json:"message,omitempty"`
	// Seconds is the wall-clock time this game took.
	Seconds float64 `json:"seconds"`
}

// matchConfig is what every game of a run shares.
type matchConfig struct {
	opponent          agent.Factory
	random            agent.Factory
	launchProbability float64
	// newTester overrides how a tester's agent is built; tests use it to
	// inject a failing agent. Nil uses buildTester.
	newTester func(ctx context.Context, j job, live *game.Game) (agent.Agent, error)
}

// playGame plays one job to the end, or until it fails or the run is
// cancelled. Errors never escape: they become the record's status, so one
// game's failure cannot be mistaken for a loss or stop the other games.
//
// The stages are: build the map; build one agent per seat; then, each turn,
// collect every seat's actions in seat order and let the engine apply them.
// The loop holds no rules of its own; AdvanceTurn decides everything.
func playGame(ctx context.Context, config matchConfig, j job) (record gameRecord) {
	started := time.Now()
	record = gameRecord{Tester: j.tester, Seed: j.seed, Seat: j.seat}
	defer func() { record.Seconds = time.Since(started).Seconds() }()
	fail := func(turn int, seat *int, err error) gameRecord {
		record.Status = statusFailed
		if ctx.Err() != nil {
			record.Status = statusCancelled
		}
		record.FailTurn, record.FailSeat, record.Message = turn, seat, err.Error()
		return record
	}

	g, err := scenario.GeneratorV2{FourPlayer: true}.New(j.seed)
	if err != nil {
		return fail(0, nil, fmt.Errorf("generate map: %w", err))
	}

	agents, err := buildAgents(ctx, config, j, g)
	defer closeAgents(agents)
	if err != nil {
		return fail(0, nil, err)
	}

	for !g.Finished() {
		if err := ctx.Err(); err != nil {
			return fail(g.Turn, nil, err)
		}
		var actions []game.Action
		for seat, a := range agents {
			// Each seat gets its own copy of the board, as mlenv gives each
			// opponent, so no agent can affect another's view.
			decision, err := a.Decide(ctx, agent.TurnRequest{
				MatchID: matchID(j), PlayerID: seat, View: agent.NewView(g),
			})
			if err != nil {
				return fail(g.Turn, &seat, fmt.Errorf("decide: %w", err))
			}
			for i, action := range decision.Actions {
				if action.PlayerID != seat {
					return fail(g.Turn, &seat, fmt.Errorf("action %d claims player %d", i, action.PlayerID))
				}
			}
			if seat == j.seat {
				record.Launches += len(decision.Actions)
			}
			actions = append(actions, decision.Actions...)
		}
		if err := g.AdvanceTurn(actions); err != nil {
			return fail(g.Turn, nil, fmt.Errorf("engine rejected the turn: %w", err))
		}
		if record.EliminatedTurn == 0 && !ownsAnything(g, j.seat) {
			record.EliminatedTurn = g.Turn
		}
	}

	record.Status = statusCompleted
	record.FinishTurn = g.Turn
	record.Termination = "turn-limit"
	if g.EliminationReached() {
		record.Termination = "elimination"
	}
	standings := g.Standings()
	for seat := range players {
		record.Ships[seat] = standings[seat]
		if seat != j.seat && !ownsAnything(g, seat) {
			record.OpponentsEliminated++
		}
	}
	record.Leaders, _ = g.Winner()
	record.Result = result(record.Leaders, j.seat)
	return record
}

// matchID names one game for agent requests.
func matchID(j job) string {
	return fmt.Sprintf("%s-seed-%d-seat-%d", j.tester, j.seed, j.seat)
}

// buildAgents creates one agent per seat. The tested seat gets the tester;
// every other seat gets the opponent, built exactly as mlenv builds training
// opponents: contestant ID "opponent-seat-<n>" and a seed from
// arena.DeriveAgentSeed, so they behave as they do in training.
func buildAgents(ctx context.Context, config matchConfig, j job, live *game.Game) ([]agent.Agent, error) {
	agents := make([]agent.Agent, 0, players)
	for seat := range players {
		var a agent.Agent
		var err error
		if seat == j.seat {
			build := config.newTester
			if build == nil {
				build = config.buildTester
			}
			a, err = build(ctx, j, live)
		} else {
			contestant := fmt.Sprintf("opponent-seat-%d", seat)
			a, err = config.opponent.New(ctx, agent.AgentInit{
				MatchID: matchID(j), ScenarioSeed: j.seed, ContestantID: contestant, PlayerID: seat,
				AgentSeed: arena.DeriveAgentSeed(j.seed, contestant, config.opponent.Spec()),
			})
		}
		if err != nil {
			return agents, fmt.Errorf("seat %d: build agent: %w", seat, err)
		}
		agents = append(agents, a)
	}
	return agents, nil
}

// buildTester creates the tested seat's agent for one of the five testers.
func (config matchConfig) buildTester(ctx context.Context, j job, live *game.Game) (agent.Agent, error) {
	init := agent.AgentInit{
		MatchID: matchID(j), ScenarioSeed: j.seed, ContestantID: "tested", PlayerID: j.seat,
	}
	switch j.tester {
	case "heuristic":
		// The calibration tester is the configured opponent playing itself:
		// heuristic-v1 by default, whatever -opponent names otherwise.
		init.AgentSeed = arena.DeriveAgentSeed(j.seed, init.ContestantID, config.opponent.Spec())
		return config.opponent.New(ctx, init)
	case "idle":
		return agent.IdleFactory{}.New(ctx, init)
	case "random":
		init.AgentSeed = arena.DeriveAgentSeed(j.seed, init.ContestantID, config.random.Spec())
		return config.random.New(ctx, init)
	case "diag-direct":
		return newDiagAgent(false, live, config.launchProbability, j.seed, j.seat), nil
	case "diag-lead":
		return newDiagAgent(true, live, config.launchProbability, j.seed, j.seat), nil
	default:
		return nil, fmt.Errorf("unknown tester %q", j.tester)
	}
}

// closeAgents releases any agent that holds resources. The built-in agents
// hold none, but the agent contract allows it.
func closeAgents(agents []agent.Agent) {
	for _, a := range agents {
		if closer, ok := a.(io.Closer); ok {
			_ = closer.Close()
		}
	}
}

// ownsAnything reports whether player still has a planet, comet or fleet, the
// same rule the engine uses for elimination.
func ownsAnything(g *game.Game, player int) bool {
	for _, p := range g.Planets {
		if p.OwnerID == player {
			return true
		}
	}
	for _, c := range g.Comets {
		if c.OwnerID == player {
			return true
		}
	}
	for _, f := range g.Fleets {
		if f.OwnerID == player {
			return true
		}
	}
	return false
}

// result converts the engine's leaders into the tester's reward, as mlenv's
// terminalReward does: a sole leader gets +1, a seat tied for the lead or a
// board with no leader at all gets 0, and anyone behind a leader gets -1.
func result(leaders []int, seat int) int {
	for _, leader := range leaders {
		if leader == seat {
			if len(leaders) == 1 {
				return 1
			}
			return 0
		}
	}
	if len(leaders) == 0 {
		return 0
	}
	return -1
}

// errUnknownTester reports a tester name the harness does not know.
var errUnknownTester = errors.New("unknown tester")
