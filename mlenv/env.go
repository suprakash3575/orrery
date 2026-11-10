package mlenv

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"runtime"
	"sort"
	"sync"

	"orrery/agent"
	"orrery/arena"
	"orrery/game"
	"orrery/scenario"
)

// ScheduleMode determines how an episode index maps to a scenario seed and
// learner seat.
type ScheduleMode string

const (
	// ScheduleTrainV2 repeats complete seed-range epochs and derives the
	// learner's seat from the versioned hash mapping in trainV2Seat, over
	// however many players the generated map seats.
	ScheduleTrainV2 ScheduleMode = "train-v2"
	// ScheduleRotate plays every seed once from each seat on the map, seat 0
	// first: 2 games per seed on a two-player map, 4 on a four-player one. On
	// a two-player map it is the same sequence the earlier paired schedule
	// gave. The seed order and seat formula are part of the replay contract;
	// changing either needs a new schedule name.
	ScheduleRotate ScheduleMode = "rotate"
)

// trainV2SeatDomain separates schedule hashes from every other SHA-256 use.
// Its exact bytes are part of the replay contract: changing them requires a
// new schedule name because old cursors would choose different learner seats.
const trainV2SeatDomain = "orrery-train-v2-seat"

// EndReason says why a step result is terminal. Its numeric values are part of
// the env-v3 wire format.
type EndReason uint8

const (
	// EndOngoing means the episode continues.
	EndOngoing EndReason = 0
	// EndGameFinished means the engine ended the game: the turn limit was
	// reached, or at most one owner is left.
	EndGameFinished EndReason = 1
	// EndLearnerEliminated means the learner owns nothing while other players
	// play on. The episode ends there; the game itself is not finished.
	EndLearnerEliminated EndReason = 2
)

// ErrorKind is a stable, machine-readable category for an environment error.
// PR 9's protocol can transport this value without having to parse Error text.
type ErrorKind string

const (
	// ErrorValidation means the caller supplied an invalid configuration, ID,
	// override, or request shape.
	ErrorValidation ErrorKind = "validation"
	// ErrorExhausted means the configured episode schedule has no next seed.
	ErrorExhausted ErrorKind = "seed-range-exhausted"
	// ErrorNotReset means Step named a slot that has no active episode.
	ErrorNotReset ErrorKind = "not-reset"
	// ErrorTerminated means Step attempted to advance an already-finished game.
	ErrorTerminated ErrorKind = "terminated"
	// ErrorPoisoned means stateful work failed and Reset is required before use.
	ErrorPoisoned ErrorKind = "poisoned"
	// ErrorClosed means an operation attempted to reuse a closed batch.
	ErrorClosed ErrorKind = "closed"
	// ErrorScenario identifies failure while constructing a generated game.
	ErrorScenario ErrorKind = "scenario"
	// ErrorOpponent identifies opponent creation or decision failure.
	ErrorOpponent ErrorKind = "opponent"
	// ErrorDecode identifies an invalid learner action for its observation.
	ErrorDecode ErrorKind = "decode"
	// ErrorEngine identifies failure while advancing the authoritative game.
	ErrorEngine ErrorKind = "engine"
	// ErrorObservation identifies failure while describing the current game.
	ErrorObservation ErrorKind = "observation"
)

// EnvError describes an environment failure without coupling the Go API to a
// wire format. EnvID is -1 when the error applies to the whole batch.
type EnvError struct {
	Kind ErrorKind // Kind lets callers handle the failure without parsing its text.
	// EnvID identifies the affected batch slot, or is -1 for a batch-wide error.
	EnvID int
	// Err preserves the underlying cause for errors.Is and errors.As.
	Err error
}

func (e *EnvError) Error() string {
	if e.EnvID >= 0 {
		return fmt.Sprintf("env %d %s: %v", e.EnvID, e.Kind, e.Err)
	}
	return fmt.Sprintf("%s: %v", e.Kind, e.Err)
}

// Unwrap exposes the underlying cause to errors.Is and errors.As.
func (e *EnvError) Unwrap() error { return e.Err }

// SeatedScenarioFactory is a map factory that also declares how many players
// every map it builds seats. A schedule that fixes the learner's seat before
// the map exists needs that count up front, and taking it from the factory
// rather than a separate setting leaves nothing that could disagree with it.
// newEnvironment still checks every generated map against the declared count.
type SeatedScenarioFactory interface {
	arena.ScenarioFactory
	// Players is the number of seated players on every generated map: 2 or 4.
	Players() int
}

// BatchConfig fixes the replay-relevant inputs and execution width of one
// batch. Scenario and Opponent are factories: every reset constructs an
// independent game and opponent instance for that environment.
type BatchConfig struct {
	// Size is the fixed number of independent environment slots in the batch.
	Size int
	// Seeds is the half-open range of scenario seeds available to the schedule.
	Seeds scenario.SeedRange
	// Schedule maps cursor positions to scenario seeds and learner seats.
	Schedule ScheduleMode
	// Cursor is the next scheduled episode index, allowing a run to resume.
	Cursor int64
	// Scenario creates a fresh deterministic game for each reset seed, and is
	// the one source of the batch's player count.
	Scenario SeatedScenarioFactory
	// Opponent creates an independent scripted agent for every seat the
	// learner does not hold, in every episode.
	Opponent agent.Factory
	// Workers limits concurrent turns; zero uses the available Go processors.
	Workers int
}

// EpisodeOverride selects a particular seed and learner seat for replay or
// debugging. It bypasses the scheduled cursor and is marked explicit in every
// result so it cannot be mistaken for a training or evaluation episode.
type EpisodeOverride struct {
	// EnvID selects the batch slot to reset explicitly.
	EnvID int
	// Seed selects the exact scenario to reproduce.
	Seed int64
	// Seat selects which player the learner controls. It must be seated on
	// the generated map: 0..1 on a two-player map, 0..3 on a four-player one.
	Seat int
}

// EpisodeInfo is the replay identity of one environment episode.
type EpisodeInfo struct {
	// EnvID identifies the batch slot that owns this episode.
	EnvID int
	// Seed identifies the generated starting scenario.
	Seed int64
	// Seat is the player controlled by the learner, in 0..players-1.
	Seat int
	// Explicit distinguishes a replay override from a scheduled episode.
	Explicit bool
}

// ResetEnv is one freshly constructed episode and its initial observation.
type ResetEnv struct {
	// EpisodeInfo records the identity needed to replay this episode.
	EpisodeInfo
	// Observation is the learner's view of the newly generated starting state.
	Observation Observation
}

// ResetResult reports reset environments in ascending ID order. Cursor is the
// next scheduled episode index after the reset; explicit overrides do not
// advance it.
type ResetResult struct {
	// Cursor is the next unused position in the configured episode schedule.
	Cursor int64
	// Envs contains the reset slots in ascending environment-ID order.
	Envs []ResetEnv
}

// StepResult is the learner-facing result of advancing one environment by one
// complete game turn. Normal operation never truncates; Truncated is reserved
// for a later external-cutoff API.
type StepResult struct {
	// EpisodeInfo identifies the episode that produced this turn result.
	EpisodeInfo
	// Observation is the learner's view after the completed turn.
	Observation Observation
	// Reward is zero during play and the learner's result on termination.
	Reward float32
	// Terminated reports that the episode ended: the game finished, or the
	// learner was eliminated while others play on. EndReason says which.
	Terminated bool
	// Truncated is reserved for a future cutoff outside the game rules.
	Truncated bool
	// EndReason is EndOngoing exactly when Terminated is false.
	EndReason EndReason
	// Turn is the engine turn number after this step completed.
	Turn int
}

// environment holds the mutable state owned by one batch slot.
type environment struct {
	// info remains stable until the slot is reset with a new episode.
	info EpisodeInfo
	// matchID gives the episode a deterministic identity for agent requests.
	matchID string
	// game is the authoritative state advanced by each successful step.
	game *game.Game
	// frame is the learner's seat frame, built once from game at turn 0. It
	// fixes the rotation and roles for the whole episode, and Build and
	// Decode both refuse a frame bound to any other game.
	frame SeatFrame
	// opponents[p] is the scripted agent playing seat p. It is nil for the
	// learner's seat and for seats the map leaves empty, so a loop over the
	// array in index order visits every opponent in ascending seat order.
	opponents [PlayerSlots]agent.Agent
	// observation caches the learner's current view used to decode its actions.
	observation Observation
	// terminated prevents another step after the game has ended.
	terminated bool
	// poisoned records a stateful step failure and requires the slot to reset.
	poisoned error
}

// Batch owns a fixed set of independent environment slots. Public operations
// are serialized by mu, while Step may advance distinct games concurrently;
// no game, RNG, or opponent instance is shared between slots.
type Batch struct {
	// mu serializes lifecycle operations and protects all fields below.
	mu sync.Mutex

	// config holds the replay and execution settings fixed at construction.
	config BatchConfig
	// workers is the validated number of slots that Step may advance at once.
	workers int
	// players is config.Scenario.Players(), read once and validated as 2 or 4
	// by NewBatch. Every generated map must seat exactly this many players.
	players int
	// cursor identifies the next episode in the configured schedule.
	cursor int64
	// envs stores one independently owned episode per batch slot.
	envs []*environment
	// closed makes Close idempotent and rejects later resets or steps.
	closed bool
	// cleanup retains replacement-close failures until the batch is closed.
	cleanup []error
}

// NewBatch validates a batch definition. It does not consume an episode or
// construct a game; Reset is the only operation that advances the cursor.
func NewBatch(ctx context.Context, config BatchConfig) (*Batch, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Validate every replay-relevant setting before allocating the batch. A
	// successfully constructed Batch can therefore assume its factories,
	// schedule, and cursor are usable for later Reset and Step calls. There
	// are no normalization caps to check: obs-v2 scales are fixed by schema.
	if config.Size <= 0 {
		return nil, batchError(ErrorValidation, "batch size must be positive")
	}
	if config.Seeds.Start >= config.Seeds.End {
		return nil, batchError(ErrorValidation, "seed range %s contains no seeds", config.Seeds)
	}
	if config.Schedule != ScheduleTrainV2 && config.Schedule != ScheduleRotate {
		return nil, batchError(ErrorValidation, "unknown schedule %q", config.Schedule)
	}
	if config.Cursor < 0 {
		return nil, batchError(ErrorValidation, "cursor must be non-negative")
	}
	// The factory is checked before the schedule length: the length will depend
	// on the player count, and asking a nil factory for it would panic.
	if nilInterface(config.Scenario) {
		return nil, batchError(ErrorValidation, "scenario factory must not be nil")
	}
	if config.Scenario.ID() == "" {
		return nil, batchError(ErrorValidation, "scenario ID must not be empty")
	}
	players := config.Scenario.Players()
	if players != 2 && players != 4 {
		return nil, batchError(ErrorValidation, "scenario %q declares %d players; want 2 or 4", config.Scenario.ID(), players)
	}
	length, err := scheduleLength(config.Seeds, config.Schedule, players)
	if err != nil {
		return nil, err
	}
	if config.Cursor > length {
		return nil, batchError(ErrorExhausted, "cursor %d exceeds schedule length %d", config.Cursor, length)
	}
	if nilInterface(config.Opponent) {
		return nil, batchError(ErrorValidation, "opponent factory must not be nil")
	}
	if config.Opponent.Spec().ID() == "" {
		return nil, batchError(ErrorValidation, "opponent ID must not be empty")
	}

	// Zero means "choose a sensible runtime default," while the size cap avoids
	// creating workers that could never receive an environment slot.
	workers := config.Workers
	if workers < 0 {
		return nil, batchError(ErrorValidation, "workers must not be negative")
	}
	if workers == 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	workers = min(workers, config.Size)

	return &Batch{
		config:  config,
		workers: workers,
		players: players,
		cursor:  config.Cursor,
		envs:    make([]*environment, config.Size),
	}, nil
}

// Cursor returns the next scheduled episode index. It is sufficient to resume
// either schedule because seed and seat are pure functions of this value.
func (b *Batch) Cursor() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.cursor
}

// Reset constructs new episodes for the requested environment IDs. IDs are
// sorted before scheduled episodes are assigned, so request ordering cannot
// change which seed and seat an environment receives.
func (b *Batch) Reset(ctx context.Context, ids []int, explicit []EpisodeOverride) (ResetResult, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return ResetResult{}, batchError(ErrorClosed, "batch is closed")
	}
	if err := ctx.Err(); err != nil {
		return ResetResult{}, err
	}

	sortedIDs, err := normalizeIDs(ids, len(b.envs))
	if err != nil {
		return ResetResult{}, err
	}
	overrides, err := normalizeOverrides(sortedIDs, explicit)
	if err != nil {
		return ResetResult{}, err
	}

	// Resolve the complete episode schedule before constructing anything.
	// Explicit overrides do not consume the cursor; scheduled episodes do, in
	// sorted environment-ID order, so input ordering cannot alter which slot
	// receives which episode. A rotate seat is already fixed here from the
	// factory's player count; a train-v2 seat is drawn in newEnvironment, once
	// the map's roster has been checked against that count.
	plans := make([]episodeAssignment, 0, len(sortedIDs))
	nextCursor := b.cursor
	for _, id := range sortedIDs {
		if override, ok := overrides[id]; ok {
			plans = append(plans, episodeAssignment{info: EpisodeInfo{
				EnvID: id, Seed: override.Seed, Seat: override.Seat, Explicit: true,
			}})
			continue
		}
		plan, episodeErr := scheduledEpisode(b.config.Seeds, b.config.Schedule, nextCursor, b.players)
		if episodeErr != nil {
			return ResetResult{}, episodeErr
		}
		plan.info.EnvID = id
		plans = append(plans, plan)
		if nextCursor == math.MaxInt64 {
			return ResetResult{}, batchError(ErrorExhausted, "cursor cannot advance beyond %d", nextCursor)
		}
		nextCursor++
	}

	// Construct every replacement before installing any of them. If one setup
	// fails — including a seat the generated map cannot seat — close the
	// replacements already created and leave the current batch slots and
	// published cursor unchanged instead of applying a partial reset.
	replacements := make([]*environment, len(plans))
	for i, plan := range plans {
		if err := ctx.Err(); err != nil {
			closeEnvironments(replacements[:i])
			return ResetResult{}, err
		}
		created, createErr := b.newEnvironment(ctx, plan)
		if createErr != nil {
			closeEnvironments(replacements[:i])
			return ResetResult{}, createErr
		}
		replacements[i] = created
	}

	// All setup succeeded, so commit the planned replacements. Failure to close
	// an old opponent is retained for Batch.Close rather than aborting midway
	// after some slots have already been replaced.
	result := ResetResult{Cursor: nextCursor, Envs: make([]ResetEnv, len(plans))}
	for i, created := range replacements {
		id := created.info.EnvID
		if previous := b.envs[id]; previous != nil {
			if closeErr := closeOpponents(previous.opponents); closeErr != nil {
				b.cleanup = append(b.cleanup, fmt.Errorf("env %d previous opponents: %w", id, closeErr))
			}
		}
		b.envs[id] = created
		result.Envs[i] = ResetEnv{EpisodeInfo: created.info, Observation: created.observation}
	}
	// Publish the new cursor only after every requested slot has been replaced.
	b.cursor = nextCursor
	return result, nil
}

// Step advances each environment named in decisions by exactly one complete
// game turn. Results are returned in ascending environment-ID order regardless
// of map iteration or worker completion order.
//
// The stages run in a fixed order, and the first two are what make a failed
// request safe:
//
//	reject lifecycle errors (nothing has happened yet)
//	  → decode every environment's learner actions (read-only)
//	  → ask every opponent, in seat order within each game (stateful)
//	  → advance each game and rebuild its observation
//
// Any failure from the opponent stage on marks every environment in the
// request as needing a reset, because some opponent may already have consumed
// private state that a retry could not reproduce.
func (b *Batch) Step(ctx context.Context, decisions map[int][BodySlots]SlotAction) ([]StepResult, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil, batchError(ErrorClosed, "batch is closed")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Maps deliberately have no stable iteration order. Sort the requested IDs
	// once so validation, opponent calls, worker indexes, errors, and returned
	// results all refer to environments in the same deterministic order.
	ids := make([]int, 0, len(decisions))
	for id := range decisions {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	// Reject every lifecycle error before decoding or calling an opponent. This
	// keeps a request that was invalid at entry entirely free of state changes.
	for _, id := range ids {
		if id < 0 || id >= len(b.envs) {
			return nil, envError(ErrorValidation, id, "environment ID is outside 0..%d", len(b.envs)-1)
		}
		if b.envs[id] == nil {
			return nil, envError(ErrorNotReset, id, "environment must be reset before stepping")
		}
		if b.envs[id].poisoned != nil {
			return nil, &EnvError{
				Kind:  ErrorPoisoned,
				EnvID: id,
				Err:   fmt.Errorf("environment must be reset after a failed batch step: %w", b.envs[id].poisoned),
			}
		}
		if b.envs[id].terminated {
			return nil, envError(ErrorTerminated, id, "environment must be reset before stepping again")
		}
	}

	type preparedStep struct {
		// id is the stable batch slot used for errors and result ordering.
		id int
		// env owns the game and opponents advanced by this prepared turn.
		env *environment
		// learnerActions is the read-only Decode result.
		learnerActions []game.Action
		// actions holds every seat's actions in ascending seat order.
		actions []game.Action
	}
	prepared := make([]preparedStep, len(ids))
	// Decode the whole request before contacting any opponent. Decide may
	// advance private RNG or other state, whereas Decode is read-only; a bad
	// learner choice in any environment must therefore reject the request
	// before stateful work starts in every environment.
	for i, id := range ids {
		env := b.envs[id]
		learnerActions, err := Decode(env.game, env.frame, env.observation, decisions[id])
		if err != nil {
			return nil, &EnvError{Kind: ErrorDecode, EnvID: id, Err: err}
		}
		prepared[i] = preparedStep{id: id, env: env, learnerActions: learnerActions}
	}

	// Opponent decisions begin the stateful phase: an agent may advance private
	// RNG or external state even though the game itself has not moved yet. Each
	// environment owns its opponents, so distinct slots can decide concurrently
	// without sharing mutable state; within one game the opponents decide one
	// after another. Store errors by the already-sorted index so worker
	// completion order can never change which failure is reported.
	opponentErrs := make([]error, len(prepared))
	decide := func(i int) {
		step := &prepared[i]
		actions, err := step.env.seatActions(ctx, step.learnerActions)
		if err != nil {
			opponentErrs[i] = &EnvError{Kind: ErrorOpponent, EnvID: step.id, Err: err}
			return
		}
		step.actions = actions
	}
	runIndexedWorkers(b.workers, len(prepared), decide)
	for _, err := range opponentErrs {
		if err != nil {
			return nil, b.poisonStep(ids, err)
		}
	}
	// Cancellation after an opponent acted is also stateful failure; retrying
	// would ask that opponent to decide again from an advanced private state.
	if err := ctx.Err(); err != nil {
		return nil, b.poisonStep(ids, err)
	}

	// Each worker writes only to its own prepared index and environment. Keeping
	// indexed result and error slices preserves deterministic output order even
	// when different games complete in a different order.
	results := make([]StepResult, len(prepared))
	errs := make([]error, len(prepared))
	advance := func(index int) {
		step := prepared[index]
		// AdvanceTurn is the first operation here that mutates the authoritative
		// game. Rebuild the observation only from the resulting state, with the
		// frame fixed at turn 0, so the next learner decision sees exactly what
		// the engine produced. Terminal turns get a fresh observation too: it is
		// the final state the learner's episode ended in.
		if err := step.env.game.AdvanceTurn(step.actions); err != nil {
			errs[index] = &EnvError{Kind: ErrorEngine, EnvID: step.id, Err: err}
			return
		}
		obs, err := Build(step.env.game, step.env.frame)
		if err != nil {
			errs[index] = &EnvError{Kind: ErrorObservation, EnvID: step.id, Err: err}
			return
		}
		step.env.observation = obs
		reason, reward := episodeEnd(step.env.game, step.env.info.Seat)
		step.env.terminated = reason != EndOngoing
		results[index] = StepResult{
			EpisodeInfo: step.env.info,
			Observation: obs,
			Reward:      reward,
			Terminated:  step.env.terminated,
			EndReason:   reason,
			Turn:        step.env.game.Turn,
		}
	}

	runIndexedWorkers(b.workers, len(prepared), advance)

	for i, err := range errs {
		if err != nil {
			// Other workers may already have completed their turn. Invalidate
			// every episode named by this request and return no results, so the
			// caller cannot mistake a mixed partial result for an atomic failure
			// and retry it. Reset is the only legal recovery.
			return nil, b.poisonStep(ids, fmt.Errorf("step result %d: %w", i, err))
		}
	}
	return results, nil
}

// runIndexedWorkers applies work to every index with bounded concurrency. The
// caller owns result storage by index, so this helper controls execution width
// without allowing goroutine completion order to affect observable ordering.
func runIndexedWorkers(workerLimit, count int, work func(int)) {
	// Avoid channel and goroutine overhead for empty or explicitly serial work.
	if count == 0 || workerLimit == 1 {
		for i := range count {
			work(i)
		}
		return
	}

	jobs := make(chan int)
	var workers sync.WaitGroup
	for range min(workerLimit, count) {
		workers.Go(func() {
			for index := range jobs {
				work(index)
			}
		})
	}
	for i := range count {
		jobs <- i
	}
	close(jobs)
	workers.Wait()
}

// Close releases every opponent owned by the batch. It is safe to call more
// than once. Cleanup failures accumulated while replacing episodes are joined
// with failures from the final close.
func (b *Batch) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	// Publish the closed state before invoking user-supplied agents. Even if an
	// agent's Close fails, the batch must not accept more work with a mixture of
	// released and unreleased resources.
	b.closed = true
	// Start with failures retained from opponents replaced by earlier resets,
	// then attempt every currently owned opponent so one failure cannot prevent
	// the remaining resources from being released.
	errs := append([]error(nil), b.cleanup...)
	for id, env := range b.envs {
		if env == nil {
			continue
		}
		if err := closeOpponents(env.opponents); err != nil {
			errs = append(errs, fmt.Errorf("env %d opponents: %w", id, err))
		}
	}
	return errors.Join(errs...)
}

// newEnvironment constructs every resource owned by one episode. It returns a
// fully usable environment, or an error after closing every opponent it had
// already created, so a failed reset leaks nothing.
//
// The stages run in this order:
//
//	generate the map
//	  → read its roster, check it against the factory's player count, and
//	    settle the learner's seat
//	  → build the learner's frame and first observation
//	  → create one opponent per other seat, in ascending seat order
//
// Everything that can reject the episode without owning a resource comes
// before the first opponent, so only the last stage needs cleanup.
func (b *Batch) newEnvironment(ctx context.Context, plan episodeAssignment) (*environment, error) {
	info := plan.info
	g, err := b.config.Scenario.New(info.Seed)
	if err != nil {
		return nil, &EnvError{Kind: ErrorScenario, EnvID: info.EnvID, Err: err}
	}
	if g == nil {
		return nil, envError(ErrorScenario, info.EnvID, "scenario returned a nil game")
	}
	if g.Finished() {
		return nil, envError(ErrorScenario, info.EnvID, "scenario returned an already-finished game")
	}

	// The roster comes from the map, so the seat is settled only now. Seat 0
	// is seated on every map NewSeatFrame accepts, which makes it a safe
	// observer for reading the roster before the learner's seat is known.
	roster, err := NewSeatFrame(g, 0)
	if err != nil {
		return nil, &EnvError{Kind: ErrorScenario, EnvID: info.EnvID, Err: err}
	}
	// A schedule may fix the seat from the declared count before the map
	// exists, so a map that seats a different number would make that seat
	// meaningless. Refuse it for every schedule, before anything is installed.
	if roster.Players() != b.players {
		return nil, envError(ErrorValidation, info.EnvID, "scenario %q declares %d players but its map for seed %d seats %d", b.config.Scenario.ID(), b.players, info.Seed, roster.Players())
	}
	info.Seat, err = plan.seatFor(roster.Players())
	if err != nil {
		return nil, err
	}
	frame := roster
	if info.Seat != 0 {
		if frame, err = NewSeatFrame(g, info.Seat); err != nil {
			return nil, &EnvError{Kind: ErrorScenario, EnvID: info.EnvID, Err: err}
		}
	}
	obs, err := Build(g, frame)
	if err != nil {
		return nil, &EnvError{Kind: ErrorObservation, EnvID: info.EnvID, Err: err}
	}

	// Including the learner's seat in the match ID distinguishes the possible
	// viewpoints of the same generated scenario.
	matchID := fmt.Sprintf("env-%d-seed-%d-seat-%d", info.EnvID, info.Seed, info.Seat)
	var opponents [PlayerSlots]agent.Agent
	for seat := range PlayerSlots {
		if _, seated := frame.RoleOf(seat); !seated || seat == info.Seat {
			continue
		}
		opponent, err := b.newOpponent(ctx, info, matchID, seat)
		if err != nil {
			// newOpponent has already closed the failed agent itself; close the
			// ones before it so the episode leaves nothing running.
			if closeErr := closeOpponents(opponents); closeErr != nil {
				err = errors.Join(err, fmt.Errorf("close earlier opponents: %w", closeErr))
			}
			return nil, &EnvError{Kind: ErrorOpponent, EnvID: info.EnvID, Err: err}
		}
		opponents[seat] = opponent
	}
	return &environment{
		info: info, matchID: matchID, game: g, frame: frame,
		opponents: opponents, observation: obs,
	}, nil
}

// newOpponent creates the scripted agent for one seat. On failure it closes
// whatever the factory returned, so the caller only has to clean up the
// opponents created before this one.
func (b *Batch) newOpponent(ctx context.Context, info EpisodeInfo, matchID string, seat int) (agent.Agent, error) {
	// Each seat gets its own contestant ID, and so its own seed:
	// DeriveAgentSeed leaves the seat out of its hash, so opponents sharing one
	// ID would also share one random stream. heuristic-v1 ignores the seed
	// today, but a seeded opponent must not mirror its neighbours. The seed is
	// also kept apart from the scenario and game RNGs, so the episode seed and
	// canonical opponent specification reproduce it.
	contestant := fmt.Sprintf("opponent-seat-%d", seat)
	opponent, err := b.config.Opponent.New(ctx, agent.AgentInit{
		MatchID:      matchID,
		ScenarioSeed: info.Seed,
		ContestantID: contestant,
		PlayerID:     seat,
		AgentSeed:    arena.DeriveAgentSeed(info.Seed, contestant, b.config.Opponent.Spec()),
	})
	if err != nil {
		// A factory may return a partially constructed agent together with an
		// error. Close it here so failed setup cannot leak subprocesses, sockets,
		// or other resources owned by that agent.
		if closeErr := closeAgent(opponent); closeErr != nil {
			err = errors.Join(err, fmt.Errorf("close partially constructed opponent: %w", closeErr))
		}
		return nil, fmt.Errorf("seat %d: %w", seat, err)
	}
	// An Agent is an interface, so a simple opponent == nil check would miss an
	// interface containing a typed nil pointer.
	if nilInterface(opponent) {
		return nil, fmt.Errorf("seat %d: opponent factory returned a nil agent", seat)
	}
	// The factory specification is recorded as replay metadata. Reject an agent
	// with a different ID so the recorded opponent still identifies what ran.
	if opponent.ID() != b.config.Opponent.Spec().ID() {
		err := fmt.Errorf("seat %d: opponent ID %q does not match factory ID %q", seat, opponent.ID(), b.config.Opponent.Spec().ID())
		if closeErr := closeAgent(opponent); closeErr != nil {
			err = errors.Join(err, fmt.Errorf("close mismatched opponent: %w", closeErr))
		}
		return nil, err
	}
	return opponent, nil
}

// seatActions asks every opponent for its turn and joins all seats' actions in
// ascending seat order, the order arena.RunMatch uses. The learner's actions
// take its own seat's place in that order.
//
// Opponents are asked one after another, never concurrently: they belong to
// one game, and a fixed order keeps any shared external state deterministic.
// Eliminated opponents are still asked. They own nothing, so they return
// nothing, and skipping them would be an unmeasured optimization.
func (env *environment) seatActions(ctx context.Context, learnerActions []game.Action) ([]game.Action, error) {
	actions := make([]game.Action, 0, len(learnerActions))
	for seat, opponent := range env.opponents {
		if seat == env.info.Seat {
			actions = append(actions, learnerActions...)
			continue
		}
		if opponent == nil {
			continue
		}
		decision, err := opponent.Decide(ctx, agent.TurnRequest{
			MatchID:  env.matchID,
			PlayerID: seat,
			View:     agent.NewView(env.game),
		})
		if err != nil {
			return nil, fmt.Errorf("seat %d: %w", seat, err)
		}
		// The engine checks that a launch's player owns its source, not that
		// the agent which sent it plays that player. An opponent acting for
		// another seat would otherwise slip through whenever it named a body
		// that seat owns — including the learner's.
		for i, action := range decision.Actions {
			if action.PlayerID != seat {
				return nil, fmt.Errorf("seat %d: action %d claims player %d", seat, i, action.PlayerID)
			}
		}
		actions = append(actions, decision.Actions...)
	}
	return actions, nil
}

// episodeEnd decides, after a completed engine turn, whether the learner's
// episode is over and with what reward. The checks run in a fixed order:
//
//  1. The game finished: the engine's result decides, through terminalReward.
//     This comes first so a turn that wipes out everyone at once scores as
//     the engine sees it — no winner, a tie — rather than as a learner loss.
//  2. The learner owns nothing while others play on: the episode ends with
//     -1. The game itself continues to exist but is never stepped again.
//  3. Otherwise play continues with reward 0.
func episodeEnd(g *game.Game, learner int) (EndReason, float32) {
	switch {
	case g.Finished():
		return EndGameFinished, terminalReward(g, learner)
	case !ownsAnything(g, learner):
		return EndLearnerEliminated, -1
	default:
		return EndOngoing, 0
	}
}

// ownsAnything reports whether player still has a planet, comet, or fleet —
// the same ownership rule game.EliminationReached applies to every owner. An
// owned planet with no ships counts: it still produces, so the player is
// still in the game. A lone fleet or comet counts for the same reason.
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

// poisonStep invalidates every episode named by a step whose stateful phase
// failed. Opponent decisions may have consumed private state and concurrent
// engine workers may have completed different amounts of work, so retrying
// any member of the request could no longer reproduce an uninterrupted run.
// Requiring Reset makes that loss explicit and discards the whole failed
// request instead of letting a caller silently continue a divergent episode.
func (b *Batch) poisonStep(ids []int, cause error) error {
	for _, id := range ids {
		b.envs[id].poisoned = cause
	}
	return batchError(ErrorPoisoned, "step failed after stateful work began; environments %v must be reset: %w", ids, cause)
}

// episodeAssignment is one episode chosen for a slot before its map exists.
//
// A train-v2 seat is drawn over the map's player count once newEnvironment
// has read the roster. rotate and explicit episodes fix their seat up front;
// an explicit seat still has to be checked against the map. seatFor finishes
// the choice.
type episodeAssignment struct {
	// info carries the slot, seed, and provenance. Its Seat is final for
	// rotate and explicit episodes and is ignored for train-v2.
	info EpisodeInfo
	// mode is the schedule that produced the assignment. It is empty for an
	// explicit override, which info.Explicit also marks.
	mode ScheduleMode
	// epoch is the train-v2 pass over the seed range, an input to the seat
	// hash. Other modes leave it zero.
	epoch uint64
}

// seatFor returns the learner's seat on a map seating players players, or a
// validation error when the assignment cannot be played on that map.
func (a episodeAssignment) seatFor(players int) (int, error) {
	switch {
	case a.info.Explicit:
		if a.info.Seat < 0 || a.info.Seat >= players {
			return 0, envError(ErrorValidation, a.info.EnvID, "explicit learner seat %d is not seated on this %d-player map", a.info.Seat, players)
		}
		return a.info.Seat, nil
	case a.mode == ScheduleRotate:
		// scheduledEpisode chose the seat in 0..factory players-1, and
		// newEnvironment has already refused a map that seats a different count,
		// so the seat is seated on this map.
		return a.info.Seat, nil
	default:
		return trainV2Seat(a.epoch, a.info.Seed, players), nil
	}
}

// scheduledEpisode converts the next assignment number (cursor) into a
// scenario seed and, where the schedule fixes it, the learner's seat.
// train-v2 walks through every seed in the configured range. After assigning
// the last seed, it starts another epoch at the first seed instead of
// exhausting the schedule; a hash of the epoch and seed later chooses the
// learner's seat. rotate assigns each seed players times in succession, with
// the learner in seat 0, then 1, and so on, and exhausts the schedule after
// that single pass. players is the factory's declared count, which NewBatch
// has validated; train-v2 ignores it here.
func scheduledEpisode(seeds scenario.SeedRange, mode ScheduleMode, cursor int64, players int) (episodeAssignment, error) {
	if cursor < 0 {
		return episodeAssignment{}, batchError(ErrorValidation, "cursor must be non-negative")
	}
	length := seeds.Len()
	if length == 0 {
		return episodeAssignment{}, batchError(ErrorExhausted, "%s schedule has no seeds in range %s", mode, seeds)
	}

	switch mode {
	case ScheduleTrainV2:
		epoch := uint64(cursor / length)
		offset := cursor % length
		return episodeAssignment{info: EpisodeInfo{Seed: seeds.Start + offset}, mode: mode, epoch: epoch}, nil
	case ScheduleRotate:
		// Integer division keeps one seed for players consecutive cursors, and
		// the remainder walks the seats within that run.
		offset := cursor / int64(players)
		if offset >= length {
			return episodeAssignment{}, batchError(ErrorExhausted, "cursor %d exhausts %s schedule for seed range %s", cursor, mode, seeds)
		}
		return episodeAssignment{info: EpisodeInfo{Seed: seeds.Start + offset, Seat: int(cursor % int64(players))}, mode: mode}, nil
	default:
		return episodeAssignment{}, batchError(ErrorValidation, "unknown schedule %q", mode)
	}
}

// trainV2Seat implements the schedule's cross-language replay formula:
//
//  1. Input bytes: the ASCII domain text, then the epoch as eight big-endian
//     bytes, then the seed as eight big-endian bytes. Converting a negative
//     seed to uint64 keeps its two's-complement bit pattern.
//  2. Digest: SHA-256 of those bytes.
//  3. Read the digest's first eight bytes as a big-endian uint64.
//  4. The seat is that value modulo players.
//
// 2^64 divides evenly by both 2 and 4, so neither supported player count
// favours any seat.
func trainV2Seat(epoch uint64, seed int64, players int) int {
	var input [len(trainV2SeatDomain) + 16]byte
	offset := copy(input[:], trainV2SeatDomain)
	binary.BigEndian.PutUint64(input[offset:offset+8], epoch)
	binary.BigEndian.PutUint64(input[offset+8:], uint64(seed))
	digest := sha256.Sum256(input[:])
	return int(binary.BigEndian.Uint64(digest[:8]) % uint64(players))
}

// scheduleLength reports how many scheduled episodes a supported mode can
// supply. An empty range supplies zero episodes for either mode, matching
// scheduledEpisode's refusal to assign from it, while an unknown mode returns
// a validation error. train-v2 repeats until the cursor type itself is
// exhausted. rotate supplies seeds × players episodes; saturating that
// multiplication prevents it from overflowing a cursor whose own type cannot
// represent a larger finite schedule.
func scheduleLength(seeds scenario.SeedRange, mode ScheduleMode, players int) (int64, error) {
	if mode != ScheduleTrainV2 && mode != ScheduleRotate {
		return 0, batchError(ErrorValidation, "unknown schedule %q", mode)
	}
	length := seeds.Len()
	if length == 0 {
		return 0, nil
	}
	if mode == ScheduleTrainV2 {
		return math.MaxInt64, nil
	}
	if length > math.MaxInt64/int64(players) {
		return math.MaxInt64, nil
	}
	return length * int64(players), nil
}

// normalizeIDs returns a sorted, independent copy and rejects IDs that could
// make reset assignment or step result ordering ambiguous.
func normalizeIDs(ids []int, size int) ([]int, error) {
	normalized := append([]int(nil), ids...)
	sort.Ints(normalized)
	for i, id := range normalized {
		if id < 0 || id >= size {
			return nil, envError(ErrorValidation, id, "environment ID is outside 0..%d", size-1)
		}
		if i > 0 && normalized[i-1] == id {
			return nil, envError(ErrorValidation, id, "environment ID appears more than once")
		}
	}
	return normalized, nil
}

// normalizeOverrides validates that every explicit replay belongs to this
// reset request and indexes overrides by environment ID for deterministic plan
// construction after the original slices have been normalized.
func normalizeOverrides(ids []int, overrides []EpisodeOverride) (map[int]EpisodeOverride, error) {
	requested := make(map[int]bool, len(ids))
	for _, id := range ids {
		requested[id] = true
	}
	normalized := make(map[int]EpisodeOverride, len(overrides))
	for _, override := range overrides {
		if !requested[override.EnvID] {
			return nil, envError(ErrorValidation, override.EnvID, "explicit override does not name a requested environment")
		}
		// Only the generated map can say whether the seat is seated; that check
		// runs in seatFor. This one rejects a seat no map could ever have.
		if override.Seat < 0 || override.Seat >= PlayerSlots {
			return nil, envError(ErrorValidation, override.EnvID, "explicit learner seat %d is outside 0..%d", override.Seat, PlayerSlots-1)
		}
		if _, duplicate := normalized[override.EnvID]; duplicate {
			return nil, envError(ErrorValidation, override.EnvID, "explicit override appears more than once")
		}
		normalized[override.EnvID] = override
	}
	return normalized, nil
}

// terminalReward converts the engine's possibly tied winner set into the
// learner-relative reward. It is called only after Finished reports true.
func terminalReward(g *game.Game, learner int) float32 {
	// Rewards are terminal-only: a sole winner receives +1, a learner tied for
	// the lead receives 0, and a learner behind any other leader receives -1,
	// whatever the number of players.
	leaders, _ := g.Winner()
	for _, leader := range leaders {
		if leader == learner {
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

// closeEnvironments is best-effort rollback for environments constructed by a
// Reset that failed before commit. There is no result channel for cleanup at
// that point, so the original construction or cancellation error takes
// precedence.
func closeEnvironments(envs []*environment) {
	for _, env := range envs {
		if env != nil {
			_ = closeOpponents(env.opponents)
		}
	}
}

// closeOpponents closes every opponent of one episode, in seat order, and
// joins their failures so one failed close cannot keep the rest running.
func closeOpponents(opponents [PlayerSlots]agent.Agent) error {
	var errs []error
	for seat, opponent := range opponents {
		if err := closeAgent(opponent); err != nil {
			errs = append(errs, fmt.Errorf("seat %d: %w", seat, err))
		}
	}
	return errors.Join(errs...)
}

// closeAgent releases agents that opt into io.Closer. Non-closable agents have
// no lifecycle work, and typed nil closers must not receive a method call.
func closeAgent(value agent.Agent) error {
	closer, ok := value.(io.Closer)
	if !ok || nilInterface(closer) {
		return nil
	}
	return closer.Close()
}

// nilInterface detects both a nil interface and an interface holding a typed
// nil value. Interface comparison with nil alone detects only the first case.
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

// batchError creates an EnvError whose -1 ID marks a batch-wide failure.
func batchError(kind ErrorKind, format string, args ...any) error {
	return envError(kind, -1, format, args...)
}

// envError keeps error construction and message formatting consistent across
// validation, lifecycle, and per-environment failures.
func envError(kind ErrorKind, envID int, format string, args ...any) error {
	return &EnvError{Kind: kind, EnvID: envID, Err: fmt.Errorf(format, args...)}
}
