package mlenv

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"testing"

	"orrery/agent"
	"orrery/scenario"
)

const (
	// benchmarkWarmupTurn moves every game far enough from its empty opening
	// state for launches, travelling fleets, prediction, and combat to be active.
	// Warm-up and any replacement-game resets stay outside the timed region.
	benchmarkWarmupTurn = 100
	// benchmarkLaunchProbability approximates the exploratory learner workload
	// used when the allocation bottleneck was found. Only sources that currently
	// have at least one legal target are eligible for this random draw.
	benchmarkLaunchProbability = 0.30
	// benchmarkActionSeed makes learner decisions repeatable across benchmark
	// runs and worker counts. It does not affect scenario or opponent RNGs.
	benchmarkActionSeed int64 = 7_431_921
)

var (
	// These sizes show the current 64-game cost and whether larger batches expose
	// useful Go scaling before Python adopts a larger policy batch.
	benchmarkEnvironmentCounts = [...]int{64, 256, 1024}
	// A case above the runtime's effective GOMAXPROCS is skipped rather than
	// presenting deliberate oversubscription as an ordinary scaling result.
	benchmarkWorkerCounts = [...]int{1, 8, 16, 46}
	// Both map sizes run on the same seed range, so their difference is the
	// cost of two extra seats: two more opponents, bodies, and fleets.
	benchmarkPlayerCounts = [...]int{2, 4}
	// The opening has few fleets in flight; by turn 100 launches, travelling
	// fleets, prediction, and combat are all active.
	benchmarkPhases = [...]benchmarkPhase{
		{name: "opening", warmTurn: 0, resetAfter: 20},
		{name: "midgame", warmTurn: benchmarkWarmupTurn},
	}
)

// benchmarkPhase picks the part of a game the timed steps sample.
type benchmarkPhase struct {
	name string
	// warmTurn is the turn every slot reaches, untimed, before timing starts.
	warmTurn int
	// resetAfter, when positive, replaces a slot once it reaches this turn, so
	// an opening case keeps measuring openings however many iterations run.
	// Zero keeps a slot until its game ends.
	resetAfter int
}

// benchmarkScenario returns the generated map family for a player count.
func benchmarkScenario(players int) scenario.GeneratorV2 {
	return scenario.GeneratorV2{FourPlayer: players == 4}
}

// benchmarkState retains the observation and learner RNG belonging to each
// reusable environment slot. Actions always come from the observation produced
// by the immediately preceding reset or step.
type benchmarkState struct {
	observations []Observation
	rngs         []*rand.Rand
}

// newBenchmarkState allocates stable, environment-ID-indexed benchmark state.
// Reset installation below supplies the first observation and RNG for each row.
func newBenchmarkState(size int) *benchmarkState {
	return &benchmarkState{
		observations: make([]Observation, size),
		rngs:         make([]*rand.Rand, size),
	}
}

// installReset starts the deterministic learner stream for one new episode.
// Deriving it from seed and seat makes the same scheduled episode receive the
// same learner actions in the API and protocol benchmarks.
func (state *benchmarkState) installReset(info EpisodeInfo, observation Observation) {
	state.observations[info.EnvID] = observation
	seed := benchmarkActionSeed + info.Seed*2 + int64(info.Seat)
	state.rngs[info.EnvID] = rand.New(rand.NewSource(seed))
}

// installStep advances the stored view after one successful environment turn.
// A terminal row remains useful for reporting but is replaced before another
// action is requested for that slot.
func (state *benchmarkState) installStep(result StepResult) {
	state.observations[result.EnvID] = result.Observation
}

// actions builds one legal, seeded action row for each requested environment.
// IDs arrive sorted, so RNG consumption and request construction never depend
// on Go map iteration or worker completion order.
func (state *benchmarkState) actions(ids []int) map[int][BodySlots]SlotAction {
	actions := make(map[int][BodySlots]SlotAction, len(ids))
	for _, id := range ids {
		actions[id] = seededBenchmarkActions(
			state.observations[id],
			state.rngs[id],
		)
	}
	return actions
}

// seededBenchmarkActions launches from about 30% of currently legal sources.
// Targets are selected uniformly from Go's authoritative target mask, keeping
// the workload realistic without allowing invalid actions into the benchmark.
func seededBenchmarkActions(observation Observation, rng *rand.Rand) [BodySlots]SlotAction {
	var actions [BodySlots]SlotAction
	for source := range BodySlots {
		var targets [BodySlots]int
		targetCount := 0
		for target, legal := range observation.TargetMask[source] {
			if legal {
				targets[targetCount] = target
				targetCount++
			}
		}
		if targetCount == 0 || rng.Float64() >= benchmarkLaunchProbability {
			continue
		}
		actions[source] = SlotAction{
			Launch: true,
			Target: int32(targets[rng.Intn(targetCount)]),
		}
	}
	return actions
}

// BenchmarkBatchHeuristic measures the Go API without any wire encoding. Each
// operation advances every environment once, so ns/op is one complete batch
// turn and env-steps/s is directly comparable across environment counts.
func BenchmarkBatchHeuristic(b *testing.B) {
	effectiveProcessors := runtime.GOMAXPROCS(0)
	for _, players := range benchmarkPlayerCounts {
		for _, phase := range benchmarkPhases {
			for _, size := range benchmarkEnvironmentCounts {
				for _, workers := range benchmarkWorkerCounts {
					name := fmt.Sprintf("players=%d/phase=%s/envs=%d/workers=%d", players, phase.name, size, workers)
					b.Run(name, func(b *testing.B) {
						if workers > effectiveProcessors {
							b.Skipf(
								"workers=%d exceeds effective GOMAXPROCS=%d",
								workers,
								effectiveProcessors,
							)
						}
						benchmarkBatchHeuristic(b, players, phase, size, workers)
					})
				}
			}
		}
	}
}

// benchmarkBatchHeuristic runs one API benchmark case after preparing every
// slot at the same minimum game age. Action generation, resets, and warm-up are
// excluded so the result isolates Batch.Step.
func benchmarkBatchHeuristic(b *testing.B, players int, phase benchmarkPhase, size, workers int) {
	ctx := context.Background()
	batch := newTestBatch(
		b,
		benchmarkScenario(players),
		heuristicFactory(b),
		size,
		scenario.SeedRange{Start: 0, End: 1 << 60},
		ScheduleTrainV2,
		0,
		workers,
	)
	state := newBenchmarkState(size)
	ids := benchmarkIDs(size)
	resetAndWarmBatchBenchmark(b, batch, state, ids, phase.warmTurn)

	steps := 0
	launches := 0
	fleets := 0
	b.ReportAllocs()
	b.ResetTimer()
	b.StopTimer()
	for range b.N {
		actions := state.actions(ids)
		launches += benchmarkLaunchCount(actions)

		// Only the authoritative concurrent step belongs to the API benchmark.
		b.StartTimer()
		results, err := batch.Step(ctx, actions)
		b.StopTimer()
		if err != nil {
			b.Fatalf("Step() error = %v", err)
		}

		steps += len(results)
		ended := make([]int, 0)
		for _, result := range results {
			state.installStep(result)
			if result.Terminated || result.Truncated || (phase.resetAfter > 0 && result.Turn >= phase.resetAfter) {
				ended = append(ended, result.EnvID)
			}
		}
		fleets += benchmarkFleetCount(batch)
		if len(ended) > 0 {
			resetAndWarmBatchBenchmark(b, batch, state, ended, phase.warmTurn)
		}
	}

	reportBenchmarkMetrics(b, steps, launches, fleets, size, 0)
	b.ReportMetric(float64(runtime.GOMAXPROCS(0)), "gomaxprocs")
}

// BenchmarkDecode measures learner-action decoding alone, on the same
// warmed states BenchmarkBatchHeuristic steps. Step decodes environments one
// after another before any opponent acts, so this is the serial share of a
// batch turn that decoding in parallel could recover. Decoding only reads the
// game, so the same states and decisions are decoded on every iteration.
func BenchmarkDecode(b *testing.B) {
	const size = 256
	for _, players := range benchmarkPlayerCounts {
		for _, phase := range benchmarkPhases {
			b.Run(fmt.Sprintf("players=%d/phase=%s/envs=%d", players, phase.name, size), func(b *testing.B) {
				batch := newTestBatch(
					b, benchmarkScenario(players), heuristicFactory(b), size,
					scenario.SeedRange{Start: 0, End: 1 << 60}, ScheduleTrainV2, 0, 0,
				)
				state := newBenchmarkState(size)
				ids := benchmarkIDs(size)
				resetAndWarmBatchBenchmark(b, batch, state, ids, phase.warmTurn)
				actions := state.actions(ids)
				launches := benchmarkLaunchCount(actions)

				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					for _, id := range ids {
						env := batch.envs[id]
						if _, err := Decode(env.game, env.frame, env.observation, actions[id]); err != nil {
							b.Fatalf("env %d: Decode() error = %v", id, err)
						}
					}
				}
				b.ReportMetric(float64(b.N*size)/b.Elapsed().Seconds(), "env-decodes/s")
				b.ReportMetric(float64(launches)/float64(size), "launches/env")
			})
		}
	}
}

// resetAndWarmBatchBenchmark installs fresh scheduled episodes and advances
// only those slots until each reaches the target turn. A game that ends during
// warm-up is reset and tried again, while already-warmed slots remain frozen.
func resetAndWarmBatchBenchmark(
	b *testing.B,
	batch *Batch,
	state *benchmarkState,
	ids []int,
	warmTurn int,
) {
	b.Helper()
	reset, err := batch.Reset(context.Background(), ids, nil)
	if err != nil {
		b.Fatalf("benchmark Reset() error = %v", err)
	}
	for _, environment := range reset.Envs {
		state.installReset(environment.EpisodeInfo, environment.Observation)
	}
	warmBatchEnvironments(b, batch, state, ids, warmTurn)
}

// warmBatchEnvironments performs untimed legal turns for the named API slots.
// The attempt limit turns an unexpected population of very short games into a
// clear benchmark setup failure rather than an endless warm-up loop.
func warmBatchEnvironments(
	b *testing.B,
	batch *Batch,
	state *benchmarkState,
	ids []int,
	warmTurn int,
) {
	b.Helper()
	// A fresh reset is at turn 0, so a zero target needs no warm-up turns.
	if warmTurn == 0 {
		return
	}
	pending := append([]int(nil), ids...)
	for attempt := 0; len(pending) > 0; attempt++ {
		if attempt >= warmTurn*20 {
			b.Fatalf("could not warm %d environments to turn %d", len(pending), warmTurn)
		}
		results, err := batch.Step(context.Background(), state.actions(pending))
		if err != nil {
			b.Fatalf("warm-up Step() error = %v", err)
		}
		pending = pending[:0]
		ended := make([]int, 0)
		for _, result := range results {
			state.installStep(result)
			switch {
			case result.Terminated || result.Truncated:
				ended = append(ended, result.EnvID)
			case result.Turn < warmTurn:
				pending = append(pending, result.EnvID)
			}
		}
		if len(ended) > 0 {
			reset, resetErr := batch.Reset(context.Background(), ended, nil)
			if resetErr != nil {
				b.Fatalf("warm-up Reset() error = %v", resetErr)
			}
			for _, environment := range reset.Envs {
				state.installReset(environment.EpisodeInfo, environment.Observation)
				pending = append(pending, environment.EnvID)
			}
		}
		// Step returns sorted results, but reset IDs can be interleaved with
		// continuing IDs. Sorting preserves deterministic RNG consumption.
		sort.Ints(pending)
	}
}

// benchmarkProtocolTransport gives the shared workload two interchangeable
// boundaries. The core transport calls the protocol handler directly, while
// the pipe transport runs the public server over OS pipes. Both exchange raw
// bytes, so the interface is independent of which wire format produced them.
type benchmarkProtocolTransport interface {
	exchange([]byte) ([]byte, error)
	close() error
	activeFleetCount() (int, bool)
}

// The benchmark request builders and response decoders below speak the
// fixed-layout env-v3 frame format. They reuse the builders and decoders
// already written for the routing tests, so the benchmark cannot drift away
// from the format those tests pin.
//
// Encoding and decoding both run outside the timed region, matching where
// Python performs the same work relative to its Go-wait phase.

func encodeBenchmarkCreate(size int, opponent string) []byte {
	return binaryTestFrame(binaryCreateRequestType, binaryCreateRequestPayload(
		uint32(size),
		[2]int64{0, 1 << 60},
		0,
		opponent,
		string(ScheduleTrainV2),
	))
}

func encodeBenchmarkReset(ids []int) []byte {
	return binaryTestFrame(
		binaryResetRequestType,
		binaryResetRequestPayload(benchmarkBinaryIDs(ids), nil),
	)
}

func encodeBenchmarkStep(actions map[int][BodySlots]SlotAction) []byte {
	return binaryTestFrame(binaryStepRequestType, benchmarkBinaryStepPayload(actions))
}

func decodeBenchmarkCreate(b *testing.B, raw []byte) protocolCreateResult {
	b.Helper()
	return decodeBinaryTestCreateResponse(
		b,
		benchmarkBinaryResponsePayload(b, raw, binaryCreateResponseType),
	)
}

func decodeBenchmarkReset(b *testing.B, raw []byte) ResetResult {
	b.Helper()
	return decodeBinaryTestResetResponse(
		b,
		benchmarkBinaryResponsePayload(b, raw, binaryResetResponseType),
	)
}

func decodeBenchmarkStep(b *testing.B, raw []byte) []StepResult {
	b.Helper()
	return decodeBinaryTestStepResponse(
		b,
		benchmarkBinaryResponsePayload(b, raw, binaryStepResponseType),
	)
}

// benchmarkBinaryStepPayload packs per-slot decisions into the env-v3 column
// layout: an environment count, ascending environment IDs,
// then one signed target byte per source slot with -1 meaning no launch.
//
// IDs are sorted because that is what the Python client transmits. The server
// accepts any order, so sorting keeps the benchmark representative rather than
// satisfying a requirement.
func benchmarkBinaryStepPayload(actions map[int][BodySlots]SlotAction) []byte {
	ids := make([]int, 0, len(actions))
	for id := range actions {
		ids = append(ids, id)
	}
	sort.Ints(ids)

	payload := make([]byte, 0, 4+len(ids)*(4+BodySlots))
	payload = binary.LittleEndian.AppendUint32(payload, uint32(len(ids)))
	for _, id := range ids {
		payload = binary.LittleEndian.AppendUint32(payload, uint32(id))
	}
	for _, id := range ids {
		for _, decision := range actions[id] {
			if !decision.Launch {
				// 0xff is -1 read as a signed byte: this source stays idle.
				payload = append(payload, 0xff)
				continue
			}
			payload = append(payload, byte(decision.Target))
		}
	}
	return payload
}

// benchmarkBinaryIDs converts environment IDs to the unsigned width used on the
// env-v3 wire, in the ascending order the server returns rows in.
func benchmarkBinaryIDs(ids []int) []uint32 {
	sorted := append([]int(nil), ids...)
	sort.Ints(sorted)
	converted := make([]uint32, len(sorted))
	for index, id := range sorted {
		converted[index] = uint32(id)
	}
	return converted
}

// benchmarkBinaryResponsePayload unwraps exactly one response frame. Checking
// the error type first means a setup failure reports the protocol's own
// message instead of a confusing decode error.
func benchmarkBinaryResponsePayload(b *testing.B, raw []byte, want byte) []byte {
	b.Helper()
	frames := decodeBinaryTestFrames(b, raw)
	if len(frames) != 1 {
		b.Fatalf("env-v3 response frame count = %d, want 1", len(frames))
	}
	frame := frames[0]
	if frame.messageType == binaryErrorResponseType {
		_, details := decodeBinaryTestErrorResponse(b, frame.payload)
		b.Fatalf("env-v3 protocol error %s: %s", details.Kind, details.Message)
	}
	if frame.messageType != want {
		b.Fatalf("env-v3 response type = %#x, want %#x", frame.messageType, want)
	}
	return frame.payload
}

// benchmarkProtocolCoreV3 calls the env-v3 protocol handler directly, with no
// pipe. Its timed region covers parsing the request off a byte slice, stepping
// the batch, and encoding the response, so subtracting BenchmarkBatchHeuristic
// isolates env-v3 encode/decode cost.
type benchmarkProtocolCoreV3 struct {
	server protocolServer
}

// benchmarkProtocolConfig serves the two-player maps the protocol benchmarks
// have always measured, so their numbers stay comparable across runs.
func benchmarkProtocolConfig() ProtocolConfig {
	return ProtocolConfig{EngineVersion: "benchmark-engine", Scenario: benchmarkScenario(2)}
}

func newBenchmarkProtocolCoreV3() *benchmarkProtocolCoreV3 {
	return &benchmarkProtocolCoreV3{server: protocolServer{
		config:        benchmarkProtocolConfig(),
		parseOpponent: agent.ParseFactory,
	}}
}

func (transport *benchmarkProtocolCoreV3) exchange(raw []byte) ([]byte, error) {
	frame, err := readBinaryFrame(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("decode env-v3 request frame: %w", err)
	}
	response, err := transport.server.handleBinary(context.Background(), frame)
	if err != nil {
		return nil, err
	}
	if response.closeAfter {
		return nil, errors.New("request unexpectedly closed protocol server")
	}
	var encoded bytes.Buffer
	if err := writeBinaryFrame(&encoded, response.messageType, response.payload); err != nil {
		return nil, fmt.Errorf("encode env-v3 response frame: %w", err)
	}
	return encoded.Bytes(), nil
}

func (transport *benchmarkProtocolCoreV3) close() error {
	if transport.server.batch == nil {
		return nil
	}
	err := transport.server.batch.Close()
	transport.server.batch = nil
	return err
}

func (transport *benchmarkProtocolCoreV3) activeFleetCount() (int, bool) {
	return benchmarkFleetCount(transport.server.batch), true
}

// benchmarkProtocolPipeV3 runs ServeBinaryProtocol over two OS pipes, matching
// the boundary Python's EnvClient crosses under env-v3. Comparing it with
// benchmarkProtocolCoreV3 estimates the process-boundary cost: the scheduling
// and pipe-copy work added on top of the codec itself.
type benchmarkProtocolPipeV3 struct {
	requestWriter  *os.File
	responseFile   *os.File
	responseReader *bufio.Reader
	cancel         context.CancelFunc
	done           <-chan error
	closed         bool
}

func newBenchmarkProtocolPipeV3(b *testing.B) *benchmarkProtocolPipeV3 {
	b.Helper()
	serverInput, requestWriter, err := os.Pipe()
	if err != nil {
		b.Fatalf("create env-v3 request pipe: %v", err)
	}
	responseReader, serverOutput, err := os.Pipe()
	if err != nil {
		_ = serverInput.Close()
		_ = requestWriter.Close()
		b.Fatalf("create env-v3 response pipe: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		// Closing the server-owned descriptors after ServeBinaryProtocol returns
		// makes EOF visible to the client even when the server exits with an
		// error. These closes are discarded because the protocol loop may have
		// released the same descriptors already, which is not a protocol fault.
		serveErr := ServeBinaryProtocol(
			ctx,
			serverInput,
			serverOutput,
			benchmarkProtocolConfig(),
		)
		_ = serverInput.Close()
		_ = serverOutput.Close()
		done <- serveErr
		close(done)
	}()
	return &benchmarkProtocolPipeV3{
		requestWriter:  requestWriter,
		responseFile:   responseReader,
		responseReader: bufio.NewReader(responseReader),
		cancel:         cancel,
		done:           done,
	}
}

func (transport *benchmarkProtocolPipeV3) exchange(raw []byte) ([]byte, error) {
	if transport.closed {
		return nil, errors.New("env-v3 protocol pipe is closed")
	}
	if _, err := transport.requestWriter.Write(raw); err != nil {
		return nil, fmt.Errorf("write env-v3 request: %w", err)
	}
	return readBenchmarkBinaryFrameBytes(transport.responseReader)
}

func (transport *benchmarkProtocolPipeV3) close() error {
	if transport.closed {
		return nil
	}

	// A close frame lets ServeBinaryProtocol finish its normal cleanup path.
	// Cancellation and descriptor closure below still release every blocked
	// reader or writer if that exchange fails.
	_, exchangeErr := transport.exchange(binaryTestFrame(binaryCloseRequestType, nil))
	transport.closed = true
	transport.cancel()
	writerErr := transport.requestWriter.Close()
	readerErr := transport.responseFile.Close()
	serverErr := <-transport.done
	return errors.Join(exchangeErr, writerErr, readerErr, serverErr)
}

func (transport *benchmarkProtocolPipeV3) activeFleetCount() (int, bool) {
	// The public protocol intentionally does not expose engine internals, so
	// fleet counts come from the core case.
	return 0, false
}

// readBenchmarkBinaryFrameBytes returns one complete frame without parsing and
// re-encoding it. Reading the declared payload length straight from the header
// keeps the timed pipe region to the transfer itself, so the measurement is not
// inflated by a copy that a real client never makes.
func readBenchmarkBinaryFrameBytes(reader io.Reader) ([]byte, error) {
	var header [binaryFrameHeaderSize]byte
	if _, err := io.ReadFull(reader, header[:]); err != nil {
		return nil, fmt.Errorf("read env-v3 response header: %w", err)
	}
	payloadSize := binary.LittleEndian.Uint32(header[8:12])
	if err := validateBinaryPayloadSize(uint64(payloadSize)); err != nil {
		return nil, err
	}
	frame := make([]byte, binaryFrameHeaderSize+int(payloadSize))
	copy(frame, header[:])
	if _, err := io.ReadFull(reader, frame[binaryFrameHeaderSize:]); err != nil {
		return nil, fmt.Errorf("read env-v3 response payload: %w", err)
	}
	return frame, nil
}

// BenchmarkProtocolCoreHeuristicV3 measures the protocol's Go work without
// pipe transfer. Subtracting BenchmarkBatchHeuristic from it gives the binary
// codec's own cost. Request construction and response inspection remain
// outside timing.
func BenchmarkProtocolCoreHeuristicV3(b *testing.B) {
	benchmarkProtocolCases(b, func(*testing.B) benchmarkProtocolTransport {
		return newBenchmarkProtocolCoreV3()
	})
}

// BenchmarkProtocolPipeHeuristicV3 measures the same workload through
// ServeBinaryProtocol and real OS pipes. Its gap from the core case estimates
// the process-boundary cost, which is what decides whether further Go work or
// overlapping Go with the learner is the better next step. Use it to compare
// code revisions; Python's profiler remains authoritative for Go-wait time
// across a complete rollout.
func BenchmarkProtocolPipeHeuristicV3(b *testing.B) {
	benchmarkProtocolCases(b, func(b *testing.B) benchmarkProtocolTransport {
		return newBenchmarkProtocolPipeV3(b)
	})
}

func benchmarkProtocolCases(
	b *testing.B,
	newTransport func(*testing.B) benchmarkProtocolTransport,
) {
	for _, size := range []int{64, 256} {
		b.Run(fmt.Sprintf("envs=%d", size), func(b *testing.B) {
			benchmarkProtocolHeuristic(b, size, newTransport(b))
		})
	}
}

// benchmarkProtocolHeuristic runs one common seeded workload so differences
// between results come from the boundary under test, not from game state.
// The core and pipe transports share this driver so their comparison cannot be
// confounded by a differing workload.
func benchmarkProtocolHeuristic(
	b *testing.B,
	size int,
	transport benchmarkProtocolTransport,
) {
	b.Cleanup(func() {
		if err := transport.close(); err != nil {
			b.Errorf("close benchmark protocol: %v", err)
		}
	})
	opponent := heuristicFactory(b)
	decodeBenchmarkCreate(b, exchangeBenchmarkBytes(
		b,
		transport,
		encodeBenchmarkCreate(size, opponent.Spec().String()),
	))
	state := newBenchmarkState(size)
	ids := benchmarkIDs(size)
	resetAndWarmProtocolBenchmark(b, transport, state, ids)

	steps := 0
	launches := 0
	fleets := 0
	reportFleets := true
	responseBytes := 0
	b.ReportAllocs()
	b.ResetTimer()
	b.StopTimer()
	for range b.N {
		actions := state.actions(ids)
		launches += benchmarkLaunchCount(actions)
		raw := encodeBenchmarkStep(actions)

		// The request has already been encoded. The core timer now begins at
		// request decoding; the pipe timer begins when those same bytes enter
		// the OS pipe. Holding this boundary fixed is what makes the core and
		// pipe numbers comparable.
		b.StartTimer()
		encoded, exchangeErr := transport.exchange(raw)
		b.StopTimer()
		if exchangeErr != nil {
			b.Fatalf("step protocol exchange: %v", exchangeErr)
		}
		step := decodeBenchmarkStep(b, encoded)

		responseBytes += len(encoded)
		steps += len(step)
		ended := make([]int, 0)
		for _, result := range step {
			state.observations[result.EnvID] = result.Observation
			if result.Terminated || result.Truncated {
				ended = append(ended, result.EnvID)
			}
		}
		if fleetCount, available := transport.activeFleetCount(); available {
			fleets += fleetCount
		} else {
			reportFleets = false
		}
		if len(ended) > 0 {
			resetAndWarmProtocolBenchmark(b, transport, state, ended)
		}
	}

	if !reportFleets {
		fleets = -1
	}
	reportBenchmarkMetrics(b, steps, launches, fleets, size, responseBytes)
	b.ReportMetric(float64(runtime.GOMAXPROCS(0)), "gomaxprocs")
}

// resetAndWarmProtocolBenchmark resets and warms only the named protocol slots.
// Setup uses the selected transport but remains outside benchmark timing.
func resetAndWarmProtocolBenchmark(
	b *testing.B,
	transport benchmarkProtocolTransport,
	state *benchmarkState,
	ids []int,
) {
	b.Helper()
	reset := decodeBenchmarkReset(b, exchangeBenchmarkBytes(
		b,
		transport,
		encodeBenchmarkReset(ids),
	))
	for _, environment := range reset.Envs {
		state.installReset(environment.EpisodeInfo, environment.Observation)
	}
	warmProtocolEnvironments(b, transport, state, ids)
}

// warmProtocolEnvironments mirrors the API warm-up through the protocol so
// every timed protocol benchmark starts with identical mid-game fleet traffic
// regardless of transport.
func warmProtocolEnvironments(
	b *testing.B,
	transport benchmarkProtocolTransport,
	state *benchmarkState,
	ids []int,
) {
	b.Helper()
	pending := append([]int(nil), ids...)
	for attempt := 0; len(pending) > 0; attempt++ {
		if attempt >= benchmarkWarmupTurn*20 {
			b.Fatalf("could not warm %d protocol environments to turn %d", len(pending), benchmarkWarmupTurn)
		}
		step := decodeBenchmarkStep(b, exchangeBenchmarkBytes(
			b,
			transport,
			encodeBenchmarkStep(state.actions(pending)),
		))
		pending = pending[:0]
		ended := make([]int, 0)
		for _, result := range step {
			state.observations[result.EnvID] = result.Observation
			switch {
			case result.Terminated || result.Truncated:
				ended = append(ended, result.EnvID)
			case result.Turn < benchmarkWarmupTurn:
				pending = append(pending, result.EnvID)
			}
		}
		if len(ended) > 0 {
			reset := decodeBenchmarkReset(b, exchangeBenchmarkBytes(
				b,
				transport,
				encodeBenchmarkReset(ended),
			))
			for _, environment := range reset.Envs {
				state.installReset(environment.EpisodeInfo, environment.Observation)
				pending = append(pending, environment.EnvID)
			}
		}
		sort.Ints(pending)
	}
}

// exchangeBenchmarkBytes sends one untimed setup request through the selected
// boundary and returns the raw response for decoding, so setup traffic crosses
// exactly the same boundary as the timed step requests.
func exchangeBenchmarkBytes(
	b *testing.B,
	transport benchmarkProtocolTransport,
	request []byte,
) []byte {
	b.Helper()
	response, err := transport.exchange(request)
	if err != nil {
		b.Fatalf("exchange benchmark protocol request: %v", err)
	}
	return response
}

// reportBenchmarkMetrics adds comparable domain units to Go's standard ns/op,
// B/op, and allocs/op output. Fleet and launch rates confirm that the benchmark
// continues to exercise the incoming-fleet predictor instead of drifting back
// toward an idle workload.
func reportBenchmarkMetrics(
	b *testing.B,
	steps, launches, fleets, environmentCount, responseBytes int,
) {
	b.Helper()
	elapsed := b.Elapsed()
	b.ReportMetric(float64(steps)/elapsed.Seconds(), "env-steps/s")
	b.ReportMetric(elapsed.Seconds()*1000/float64(b.N), "ms/turn")
	b.ReportMetric(float64(launches)/float64(steps), "launches/env-step")
	if fleets >= 0 {
		b.ReportMetric(float64(fleets)/float64(b.N*environmentCount), "fleets/env")
	}
	if responseBytes > 0 {
		b.ReportMetric(float64(responseBytes)/float64(b.N*1024), "response-KiB/turn")
	}
}

// benchmarkLaunchCount counts true launch decisions without depending on map
// iteration order. It runs outside the timed region and only describes load.
func benchmarkLaunchCount(actions map[int][BodySlots]SlotAction) int {
	count := 0
	for _, row := range actions {
		for _, action := range row {
			if action.Launch {
				count++
			}
		}
	}
	return count
}

// benchmarkFleetCount reads the post-step number of active travelling fleets.
// Batch.Step has completed before this helper runs, so no worker is mutating the
// games concurrently and the diagnostic does not need to enter the timed path.
func benchmarkFleetCount(batch *Batch) int {
	count := 0
	for _, environment := range batch.envs {
		if environment != nil {
			count += len(environment.game.Fleets)
		}
	}
	return count
}

// benchmarkIDs returns every reusable slot in ascending order for deterministic
// schedule assignment, action generation, and result comparison.
func benchmarkIDs(size int) []int {
	ids := make([]int, size)
	for id := range size {
		ids[id] = id
	}
	return ids
}
