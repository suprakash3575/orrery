package mlenv

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"orrery/agent"
	"orrery/scenario"
)

// ProtocolVersion identifies the fixed-layout binary transport used for every
// request and response between Python and Go. It carries obs-v2 observations
// (SchemaID) and is the only protocol cmd/env speaks. The "-v3" stays in the
// value because existing checkpoints record it.
const ProtocolVersion = "env-v3"

const (
	protocolErrorMalformedBinary = "malformed-binary"
	protocolErrorUnknownCommand  = "unknown-command"
	protocolErrorState           = "invalid-state"
	protocolErrorValidation      = "validation"
)

// ProtocolConfig supplies process-wide facts that are not chosen by a create
// request. Scenario is fixed for the process so every create uses the same
// map generator; EngineVersion is returned to the caller as replay metadata.
// There are no normalization caps: obs-v2 scales are fixed by its schema.
type ProtocolConfig struct {
	EngineVersion string
	// Scenario also declares the player count of every batch the process
	// creates; NewBatch validates it.
	Scenario SeatedScenarioFactory
}

// DefaultProtocolConfig constructs the generated-map environment used by
// cmd/env for the requested player count: generator-v2 for 2 players and
// generator-v2-4p for 4. Both variants place the same planets for a seed, so
// the seed ranges mean the same in each; in a two-player game the other two
// planets of the home group stay neutral. Any other count is an error, so
// this is the one place the 2-or-4 rule is enforced. Tests may inject a
// shorter scenario through ProtocolConfig without adding a test-only field to
// the wire format.
func DefaultProtocolConfig(engineVersion string, players int) (ProtocolConfig, error) {
	if players != 2 && players != 4 {
		return ProtocolConfig{}, fmt.Errorf("unsupported player count %d; want 2 or 4", players)
	}
	return ProtocolConfig{
		EngineVersion: engineVersion,
		Scenario:      scenario.GeneratorV2{FourPlayer: players == 4},
	}, nil
}

// ProtocolError is the stable structured form of a rejected request. EnvID is
// present only when the failure belongs to one environment rather than the
// protocol or batch as a whole.
type ProtocolError struct {
	Kind    string
	Message string
	EnvID   *int
}

// protocolCreateResult is the result of creating a batch: every identity the
// create response reports so the caller can reconstruct the same environment.
// The opponent string is canonicalized by its factory.
type protocolCreateResult struct {
	engine    string
	schema    string
	generator string
	opponent  string
	seedRange [2]int64
	schedule  string
	cursor    int64
}

type protocolFailure struct {
	kind string
	err  error
}

func (failure *protocolFailure) Error() string { return failure.err.Error() }
func (failure *protocolFailure) Unwrap() error { return failure.err }

// closeInputOnCancel turns context cancellation into a read interruption for
// pipes and files. An io.Reader alone has no cancellation operation, but the
// process boundary supplies os.Stdin and Python supplies a pipe, both of which
// are closable. Closing them makes a blocked frame read return immediately
// instead of leaving an interrupted idle process waiting for another request.
func closeInputOnCancel(ctx context.Context, input io.Reader) func() {
	closer, ok := input.(io.Closer)
	if !ok {
		return func() {}
	}
	stopped := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			// Closing a file descriptor does not reliably interrupt a read that
			// another goroutine already issued on every platform. Pipes exposed
			// as *os.File support deadlines, which wake that read before Close
			// releases the descriptor. io.PipeReader has no deadline support,
			// but its Close method directly wakes a blocked read.
			if deadliner, ok := input.(interface{ SetReadDeadline(time.Time) error }); ok {
				_ = deadliner.SetReadDeadline(time.Now())
			}
			_ = closer.Close()
		case <-stopped:
		}
	}()
	return func() { close(stopped) }
}

// protocolServer holds the one live batch a protocol session may own. The
// binary codec in protocol_codec.go decodes requests and encodes responses; this
// type owns the lifecycle rules and batch construction those requests share.
type protocolServer struct {
	config        ProtocolConfig
	parseOpponent func(string) (agent.Factory, error)
	batch         *Batch
}

// joinCleanupError closes a batch left behind by EOF, cancellation, or a fatal
// transport error. Joining rather than replacing the existing error preserves
// both the communication failure and any opponent cleanup failure.
func (server *protocolServer) joinCleanupError(returnErr *error) {
	if server.batch == nil {
		return
	}
	if err := server.batch.Close(); err != nil {
		*returnErr = errors.Join(*returnErr, fmt.Errorf("close environment batch: %w", err))
	}
}

func (server *protocolServer) requireNotCreated() error {
	if server.batch != nil {
		return &protocolFailure{kind: protocolErrorState, err: errors.New("batch is already created")}
	}
	return nil
}

func (server *protocolServer) requireCreated(command string) error {
	if server.batch == nil {
		return &protocolFailure{
			kind: protocolErrorState,
			err:  fmt.Errorf("create must succeed before %s", command),
		}
	}
	return nil
}

// createBatch performs the semantic validation of a decoded create request
// and records the resulting live batch.
func (server *protocolServer) createBatch(
	ctx context.Context,
	batchSize int,
	seedRange [2]int64,
	opponentText string,
	schedule ScheduleMode,
	cursor int64,
) (protocolCreateResult, error) {
	opponent, err := server.parseOpponent(opponentText)
	if err != nil {
		return protocolCreateResult{}, validationFailure("opponent: %v", err)
	}
	seeds := scenario.SeedRange{Start: seedRange[0], End: seedRange[1]}
	batch, err := NewBatch(ctx, BatchConfig{
		Size: batchSize, Seeds: seeds, Schedule: schedule, Cursor: cursor,
		Scenario: server.config.Scenario, Opponent: opponent,
	})
	if err != nil {
		return protocolCreateResult{}, err
	}
	server.batch = batch
	configuredSeeds := batch.config.Seeds
	configuredSchedule := batch.config.Schedule
	return protocolCreateResult{
		engine: server.config.EngineVersion, schema: SchemaID,
		generator: server.config.Scenario.ID(), opponent: opponent.Spec().String(),
		seedRange: [2]int64{configuredSeeds.Start, configuredSeeds.End},
		schedule:  string(configuredSchedule), cursor: batch.Cursor(),
	}, nil
}

// closeBatch clears the pointer even when cleanup fails. A close failure means
// the protocol must stop, and retaining the pointer would make deferred cleanup
// call the same already-failed close operation a second time.
func (server *protocolServer) closeBatch() error {
	err := server.batch.Close()
	server.batch = nil
	return err
}

func validateProtocolConfig(config ProtocolConfig) error {
	if strings.TrimSpace(config.EngineVersion) == "" {
		return errors.New("protocol engine version must not be empty")
	}
	if nilInterface(config.Scenario) || config.Scenario.ID() == "" {
		return errors.New("protocol scenario factory must have a non-empty ID")
	}
	return nil
}

func validationFailure(format string, args ...any) error {
	return &protocolFailure{kind: protocolErrorValidation, err: fmt.Errorf(format, args...)}
}

// protocolErrorDetails extracts the stable kind, environment ID, and message
// that an error response reports for a rejected request.
func protocolErrorDetails(err error) ProtocolError {
	kind := protocolErrorValidation
	var envID *int
	var envErr *EnvError
	var protocolErr *protocolFailure
	switch {
	case errors.As(err, &envErr):
		kind = string(envErr.Kind)
		if envErr.EnvID >= 0 {
			id := envErr.EnvID
			envID = &id
		}
	case errors.As(err, &protocolErr):
		kind = protocolErr.kind
	}
	return ProtocolError{Kind: kind, Message: err.Error(), EnvID: envID}
}
