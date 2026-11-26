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
	"orrery/mlenv"
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

	// TesterCounts holds the tester's aim and decode counts. It is set only
	// for a completed game whose tester reports AimStats (heuristic-v2), and
	// nil otherwise. Embedded as a pointer, its fields appear at the top
	// level of the JSON when set and are left out entirely when nil, so a
	// record of any other tester is written exactly as before. The type is
	// exported only because encoding/json can't fill an embedded pointer to
	// an unexported struct when reading a record back.
	*TesterCounts
}

// TesterCounts are one game's totals for the tested bot, read from its
// AimStats at the end of the game (see agent.AimStats for each count).
type TesterCounts struct {
	// AimAttempts is how many times the bot tried to aim ahead, counting
	// targets it considered but didn't launch at; AimFailures is how many of
	// those found no way to the target.
	AimAttempts int `json:"aim_attempts"`
	AimFailures int `json:"aim_failures"`
	// FallbackLaunches is how many launches were aimed straight at the
	// target because aiming ahead failed. Only learner_aim=1 makes them.
	FallbackLaunches int `json:"fallback_launches"`
	// RandomMoves is how many times a body's move was chosen at random
	// (epsilon), whether it launched or held.
	RandomMoves int `json:"random_moves"`
	// DecodeCheckedLaunches is how many launches passed the decode check:
	// every launch of a completed game when -check-decode is on, else 0.
	DecodeCheckedLaunches int `json:"decode_checked_launches"`
}

// aimReporter is the part of agent.HeuristicV2Agent the counters read.
type aimReporter interface {
	AimStats() agent.AimStats
}

// matchConfig is what every game of a run shares.
type matchConfig struct {
	// opponents is the -opponent lineup: one agent for every opponent seat,
	// or one per board role from the tested seat (see mlenv.Lineup).
	opponents mlenv.Lineup
	random    agent.Factory
	// tester is the -tester-spec agent, played as specTester; nil when the
	// fixed testers are played.
	tester            agent.Factory
	launchProbability float64
	// checkDecode runs the decode check (see decodecheck.go) on every turn of
	// the tested seat. run allows it only for a fully restricted tester.
	checkDecode bool
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

	// The tested seat's frame says which seat is clockwise, opposite and
	// counter-clockwise of it, and is what the decode check observes the
	// board through. NewSeatFrame needs the starting board, which g still
	// is: no turn has been played yet.
	frame, err := mlenv.NewSeatFrame(g, j.seat)
	if err != nil {
		return fail(0, nil, fmt.Errorf("seat frame: %w", err))
	}

	agents, err := buildAgents(ctx, config, j, g, frame)
	defer closeAgents(agents)
	if err != nil {
		return fail(0, nil, err)
	}
	// decodeChecked counts the tester's launches that passed the decode
	// check. It is copied into the record only if the game completes.
	decodeChecked := 0
	var reporter choiceReporter
	if config.checkDecode {
		var ok bool
		if reporter, ok = agents[j.seat].(choiceReporter); !ok {
			return fail(0, &j.seat, fmt.Errorf("decode check: tester %s does not report its choices", agents[j.seat].ID()))
		}
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
				// g has not advanced yet, so it is still the board the
				// tester decided on, as the decoder needs.
				if reporter != nil {
					if err := checkDecode(g, frame, reporter.LastChoices(), decision.Actions); err != nil {
						return fail(g.Turn, &seat, fmt.Errorf("decode check: %w", err))
					}
					decodeChecked += len(decision.Actions)
				}
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
	// AimStats are running totals for the match, so reading them once at the
	// end gives the whole game's counts.
	if aims, ok := agents[j.seat].(aimReporter); ok {
		stats := aims.AimStats()
		record.TesterCounts = &TesterCounts{
			AimAttempts: stats.AimAheadAttempts, AimFailures: stats.AimAheadFailed,
			FallbackLaunches: stats.AimedStraight, RandomMoves: stats.RandomMoves,
			DecodeCheckedLaunches: decodeChecked,
		}
	}
	return record
}

// matchID names one game for agent requests.
func matchID(j job) string {
	return fmt.Sprintf("%s-seed-%d-seat-%d", j.tester, j.seed, j.seat)
}

// buildAgents creates one agent per seat. The tested seat gets the tester;
// every other seat gets its opponent, chosen and built exactly as mlenv
// builds training opponents: the lineup entry for the seat's board role as
// seen from the tested seat, contestant ID "opponent-seat-<n>", and a seed
// from arena.DeriveAgentSeed, so they behave as they do in training. frame
// is the tested seat's frame, which gives each other seat's role.
func buildAgents(ctx context.Context, config matchConfig, j job, live *game.Game, frame mlenv.SeatFrame) ([]agent.Agent, error) {
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
			// Every seat of a generator-v2-4p map is seated, so RoleOf always
			// finds a role here.
			role, _ := frame.RoleOf(seat)
			factory := config.opponents.ForRole(role)
			contestant := seatContestant(seat)
			a, err = factory.New(ctx, agent.AgentInit{
				MatchID: matchID(j), ScenarioSeed: j.seed, ContestantID: contestant, PlayerID: seat,
				AgentSeed: arena.DeriveAgentSeed(j.seed, contestant, factory.Spec()),
			})
		}
		if err != nil {
			return agents, fmt.Errorf("seat %d: build agent: %w", seat, err)
		}
		agents = append(agents, a)
	}
	return agents, nil
}

// seatContestant is the contestant ID of the opponent in seat, as mlenv
// names training opponents. arena.DeriveAgentSeed hashes it, so it fixes that
// seat's random stream.
func seatContestant(seat int) string {
	return fmt.Sprintf("opponent-seat-%d", seat)
}

// buildTester creates the tested seat's agent for one of the five testers.
func (config matchConfig) buildTester(ctx context.Context, j job, live *game.Game) (agent.Agent, error) {
	init := agent.AgentInit{
		MatchID: matchID(j), ScenarioSeed: j.seed, ContestantID: "tested", PlayerID: j.seat,
	}
	switch j.tester {
	case "heuristic":
		// The calibration tester is the configured opponent playing itself:
		// heuristic-v1 by default, whatever -opponent names otherwise. run
		// refuses this tester with a mixed lineup, so every role holds the
		// same agent and any role's entry is it.
		//
		// It is seeded as the opponent in its seat would be, not as
		// "tested". Calibration needs the four seat runs of a map to be one
		// game, so each seat must draw the same random stream whichever
		// seat carries the "tested" label. A seed tied to the label would
		// give a random opponent (random-v1, or v2 with epsilon) a different
		// stream in the tested seat, and calibration would fail falsely.
		opponent := config.opponents.ForRole(mlenv.RoleCW)
		init.AgentSeed = arena.DeriveAgentSeed(j.seed, seatContestant(j.seat), opponent.Spec())
		return opponent.New(ctx, init)
	case "idle":
		return agent.IdleFactory{}.New(ctx, init)
	case "random":
		init.AgentSeed = arena.DeriveAgentSeed(j.seed, init.ContestantID, config.random.Spec())
		return config.random.New(ctx, init)
	case "diag-direct":
		return newDiagAgent(false, live, config.launchProbability, j.seed, j.seat), nil
	case "diag-lead":
		return newDiagAgent(true, live, config.launchProbability, j.seed, j.seat), nil
	case specTester:
		if config.tester == nil {
			return nil, errors.New("no -tester-spec agent configured")
		}
		// A tester that is the opponent spec is the calibration, so it is
		// seeded by its seat, exactly as the heuristic tester is. Any other
		// tester spec isn't calibrated and keeps the "tested" seed.
		contestant := init.ContestantID
		if config.calibrationTester() == specTester {
			contestant = seatContestant(j.seat)
		}
		init.AgentSeed = arena.DeriveAgentSeed(j.seed, contestant, config.tester.Spec())
		return config.tester.New(ctx, init)
	default:
		return nil, fmt.Errorf("unknown tester %q", j.tester)
	}
}

// calibrationTester names the tester whose games the calibration checks:
// "heuristic" for the fixed testers, and specTester when the -tester-spec
// agent is the opponent spec itself (same canonical form), so every seat
// plays the same agent. A lineup whose three entries are all that spec
// counts too: its canonical form is the one spec. Any other tester spec has
// no calibration: "".
//
// The check expects the four seat runs of a seed to be one identical game.
// That holds for random agents too, because the calibration tester is seeded
// by its seat, exactly as the opponent there would be (see buildTester).
func (config matchConfig) calibrationTester() string {
	switch {
	case config.tester == nil:
		return "heuristic"
	case config.tester.Spec().String() == config.opponents.String():
		return specTester
	default:
		return ""
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
