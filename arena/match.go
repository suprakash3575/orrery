package arena

import (
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"

	"orrery/agent"
	"orrery/game"
)

type initializedAgent struct {
	playerID     int
	contestantID string
	agentSpec    string
	instance     agent.Agent
}

// RunMatch executes one match and reports both completed outcomes and runtime
// failures as data. The result is therefore usable by a later experiment
// runner without one failed agent aborting unrelated matches.
func RunMatch(ctx context.Context, spec MatchSpec) (result MatchResult) {
	result = baseResult(spec)
	if err := validateMatchSpec(spec); err != nil {
		setFailure(&result, nil, FailureValidation, err, nil, nil)
		return result
	}

	result.ScenarioID = spec.Scenario.ID()
	result.Seats = make([]SeatAssignment, len(spec.Seats))
	for playerID, seat := range spec.Seats {
		result.Seats[playerID] = SeatAssignment{
			PlayerID:     playerID,
			ContestantID: seat.ContestantID,
			AgentSpec:    seat.Factory.Spec().String(),
			AgentSeed:    seat.AgentSeed,
		}
	}
	if result.Actions != nil {
		result.Actions.ScenarioID = result.ScenarioID
	}

	if err := ctx.Err(); err != nil {
		setFailure(&result, nil, FailureCancellation, err, nil, nil)
		return result
	}
	g, err := spec.Scenario.New(spec.Seed)
	if err != nil {
		setFailure(&result, nil, FailureScenario, err, nil, nil)
		return result
	}
	if g == nil {
		setFailure(&result, nil, FailureScenario, errors.New("scenario returned a nil game"), nil, nil)
		return result
	}
	if err := validateGameSeats(g, len(spec.Seats)); err != nil {
		setFailure(&result, g, FailureScenario, err, nil, nil)
		return result
	}
	if err := ctx.Err(); err != nil {
		setFailure(&result, g, FailureCancellation, err, nil, nil)
		return result
	}

	initialized := make([]initializedAgent, 0, len(spec.Seats))
	defer func() {
		closeAgents(&result, initialized)
	}()
	for playerID, seat := range spec.Seats {
		instance, createErr := seat.Factory.New(ctx, agent.AgentInit{
			MatchID:      spec.ID,
			ScenarioSeed: spec.Seed,
			ContestantID: seat.ContestantID,
			PlayerID:     playerID,
			AgentSeed:    seat.AgentSeed,
		})
		if !nilInterface(instance) {
			initialized = append(initialized, initializedAgent{
				playerID:     playerID,
				contestantID: seat.ContestantID,
				agentSpec:    seat.Factory.Spec().String(),
				instance:     instance,
			})
		}
		if createErr != nil {
			kind := FailureConstruction
			if cancellationError(createErr) {
				kind = FailureCancellation
			}
			setFailure(&result, g, kind, createErr, seatFailure(result.Seats[playerID]), nil)
			return result
		}
		if nilInterface(instance) {
			setFailure(&result, g, FailureConstruction, errors.New("agent factory returned a nil agent"), seatFailure(result.Seats[playerID]), nil)
			return result
		}
		if instance.ID() != seat.Factory.Spec().ID() {
			err := fmt.Errorf("agent ID %q does not match factory ID %q", instance.ID(), seat.Factory.Spec().ID())
			setFailure(&result, g, FailureConstruction, err, seatFailure(result.Seats[playerID]), nil)
			return result
		}
		if err := ctx.Err(); err != nil {
			setFailure(&result, g, FailureCancellation, err, nil, nil)
			return result
		}
	}

	for !g.Finished() {
		if err := ctx.Err(); err != nil {
			setFailure(&result, g, FailureCancellation, err, nil, nil)
			return result
		}

		// Build every view before asking any seat to decide. Each call makes a
		// separate deep copy, so neither decision order nor an agent mutating its
		// own request can change another seat's pre-turn facts.
		views := make([]agent.View, len(initialized))
		for playerID := range initialized {
			views[playerID] = agent.NewView(g)
		}

		batch := make([]game.Action, 0)
		for playerID, initializedSeat := range initialized {
			if err := ctx.Err(); err != nil {
				setFailure(&result, g, FailureCancellation, err, nil, nil)
				return result
			}
			decision, decideErr := initializedSeat.instance.Decide(ctx, agent.TurnRequest{
				MatchID:  spec.ID,
				PlayerID: playerID,
				View:     views[playerID],
			})
			if decideErr != nil {
				kind := FailureDecision
				if cancellationError(decideErr) {
					kind = FailureCancellation
				}
				setFailure(&result, g, kind, decideErr, seatFailure(result.Seats[playerID]), nil)
				return result
			}
			// The engine takes one merged batch and trusts each action's own
			// PlayerID, so it cannot tell which seat submitted what. Without
			// this check a seat could launch another player's ships simply by
			// writing their ID. Checking here, before the merge, is the last
			// point at which "who sent this" is still known.
			for i, action := range decision.Actions {
				if action.PlayerID != playerID {
					err := fmt.Errorf("action %d claims player %d, but this seat plays %d", i, action.PlayerID, playerID)
					setFailure(&result, g, FailureDecision, err, seatFailure(result.Seats[playerID]), decision.Actions)
					return result
				}
			}
			batch = append(batch, cloneActions(decision.Actions)...)
		}
		if err := ctx.Err(); err != nil {
			setFailure(&result, g, FailureCancellation, err, nil, nil)
			return result
		}

		turn := g.Turn
		if err := g.AdvanceTurn(batch); err != nil {
			// The engine rejected the batch, so the kind stays FailureEngine,
			// but a rejected action is still some seat's mistake. The check
			// above guarantees every action's PlayerID is its sender's seat,
			// so an ActionError's PlayerID names who to blame.
			var seat *SeatAssignment
			var actionErr *game.ActionError
			if errors.As(err, &actionErr) && actionErr.PlayerID >= 0 && actionErr.PlayerID < len(result.Seats) {
				seat = seatFailure(result.Seats[actionErr.PlayerID])
			}
			setFailure(&result, g, FailureEngine, err, seat, batch)
			return result
		}
		if result.Actions != nil {
			result.Actions.Turns = append(result.Actions.Turns, TurnRecord{
				Turn:    turn,
				Actions: cloneActions(batch),
			})
		}
	}

	finalizeResult(&result, g)
	// Elimination takes priority when the final turn also reaches MaxTurns.
	// That preserves the meaningful game event for arena summaries: a pure
	// turn-limit result is one where multiple owners still remain.
	if g.EliminationReached() {
		result.Termination = TerminationElimination
	} else {
		result.Termination = TerminationTurnLimit
	}
	return result
}

func baseResult(spec MatchSpec) MatchResult {
	result := MatchResult{
		SchemaVersion: MatchResultSchemaV1,
		MatchID:       spec.ID,
		ScenarioIndex: spec.ScenarioIndex,
		Leg:           spec.Leg,
		ScenarioSeed:  spec.Seed,
		Termination:   TerminationFailed,
	}
	if spec.RecordActions {
		result.Actions = &ActionRecord{
			SchemaVersion: ActionRecordSchemaV1,
			MatchID:       spec.ID,
			ScenarioIndex: spec.ScenarioIndex,
			Leg:           spec.Leg,
			ScenarioSeed:  spec.Seed,
			Turns:         make([]TurnRecord, 0),
		}
	}
	return result
}

func validateMatchSpec(spec MatchSpec) error {
	if spec.ID == "" {
		return errors.New("match ID must not be empty")
	}
	if spec.ScenarioIndex < 0 || spec.Leg < 0 {
		return errors.New("scenario index and leg must be non-negative")
	}
	if nilInterface(spec.Scenario) {
		return errors.New("scenario factory must not be nil")
	}
	if spec.Scenario.ID() == "" {
		return errors.New("scenario ID must not be empty")
	}
	if len(spec.Seats) == 0 || len(spec.Seats) > 4 {
		return fmt.Errorf("seat count must be in [1,4], got %d", len(spec.Seats))
	}
	contestants := make(map[string]bool, len(spec.Seats))
	for playerID, seat := range spec.Seats {
		if seat.ContestantID == "" {
			return fmt.Errorf("seat %d contestant ID must not be empty", playerID)
		}
		if contestants[seat.ContestantID] {
			return fmt.Errorf("duplicate contestant ID %q", seat.ContestantID)
		}
		contestants[seat.ContestantID] = true
		if nilInterface(seat.Factory) {
			return fmt.Errorf("seat %d agent factory must not be nil", playerID)
		}
		if seat.Factory.Spec().ID() == "" {
			return fmt.Errorf("seat %d agent factory ID must not be empty", playerID)
		}
	}
	return nil
}

// validateGameSeats prevents an initial owner from participating without an
// agent. It intentionally permits extra seats with no starting body, which is
// useful for the later three- and four-player scenario work without adding
// p0/p1 branches to the loop now.
func validateGameSeats(g *game.Game, seatCount int) error {
	validateOwner := func(kind string, id, ownerID int) error {
		if ownerID == game.NeutralPlayer {
			return nil
		}
		if ownerID < 0 || ownerID >= seatCount {
			return fmt.Errorf("%s %d owner %d has no match seat", kind, id, ownerID)
		}
		return nil
	}
	for _, planet := range g.Planets {
		if err := validateOwner("planet", planet.ID, planet.OwnerID); err != nil {
			return err
		}
	}
	for _, comet := range g.Comets {
		if err := validateOwner("comet", comet.ID, comet.OwnerID); err != nil {
			return err
		}
	}
	for _, fleet := range g.Fleets {
		if err := validateOwner("fleet", fleet.ID, fleet.OwnerID); err != nil {
			return err
		}
	}
	return nil
}

func setFailure(result *MatchResult, g *game.Game, kind FailureKind, err error, seat *SeatAssignment, actions []game.Action) {
	failure := &MatchFailure{
		Kind:    kind,
		Message: err.Error(),
		Actions: cloneActions(actions),
	}
	if g != nil {
		failure.Turn = g.Turn
	}
	if seat != nil {
		playerID := seat.PlayerID
		failure.PlayerID = &playerID
		failure.ContestantID = seat.ContestantID
		failure.AgentSpec = seat.AgentSpec
	}
	result.Failure = failure
	if kind == FailureCancellation {
		result.Termination = TerminationCancelled
	} else {
		result.Termination = TerminationFailed
	}
	finalizeResult(result, g)
}

func finalizeResult(result *MatchResult, g *game.Game) {
	if g == nil {
		return
	}
	result.FinishTurn = g.Turn
	result.FinalState = agent.NewView(g)

	totals := g.Standings()
	players := make(map[int]bool, len(result.Seats)+len(totals))
	for _, seat := range result.Seats {
		players[seat.PlayerID] = true
	}
	for playerID := range totals {
		players[playerID] = true
	}
	playerIDs := make([]int, 0, len(players))
	for playerID := range players {
		playerIDs = append(playerIDs, playerID)
	}
	sort.Ints(playerIDs)
	result.Standings = make([]Standing, 0, len(playerIDs))
	for _, playerID := range playerIDs {
		result.Standings = append(result.Standings, Standing{PlayerID: playerID, Ships: totals[playerID]})
	}
	if leaders, ok := g.Winner(); ok {
		result.Leaders = append([]int(nil), leaders...)
	}
}

func closeAgents(result *MatchResult, initialized []initializedAgent) {
	for i := len(initialized) - 1; i >= 0; i-- {
		closer, ok := initialized[i].instance.(io.Closer)
		if !ok {
			continue
		}
		if err := closer.Close(); err != nil {
			result.CleanupErrors = append(result.CleanupErrors, CleanupError{
				PlayerID:     initialized[i].playerID,
				ContestantID: initialized[i].contestantID,
				AgentSpec:    initialized[i].agentSpec,
				Message:      err.Error(),
			})
		}
	}
}

func cancellationError(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

func seatFailure(seat SeatAssignment) *SeatAssignment { return &seat }

func cloneActions(actions []game.Action) []game.Action {
	return append([]game.Action(nil), actions...)
}

// nilInterface handles typed nil pointers stored in an interface. Without it,
// invoking Spec, ID, or Decide on a typed nil implementation could panic in
// validation instead of producing an attributable match failure.
func nilInterface(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	default:
		return false
	}
}
