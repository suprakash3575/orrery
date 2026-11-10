package mlenv

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"orrery/agent"
	"orrery/scenario"
)

// TestBinaryProtocolMatchesBatchAndIsByteDeterministic drives one complete
// create → reset → step → close exchange through the wire protocol and through
// a Batch built directly from the same settings. Decoding the frames back into
// Batch's own result types means any field the codec drops, reorders, or
// rescales shows up as a difference. The four-player case puts the learner in
// seat 3 so a seat byte above the two-player range crosses the wire.
func TestBinaryProtocolMatchesBatchAndIsByteDeterministic(t *testing.T) {
	tests := []struct {
		name      string
		players   int
		overrides []EpisodeOverride
	}{
		{name: "two players", players: 2},
		{name: "four players", players: 4, overrides: []EpisodeOverride{{EnvID: 0, Seed: 12, Seat: 3}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := context.Background()
			factory := testScenario{id: "protocol-parity-v1", maxTurns: 3, players: test.players}
			config := ProtocolConfig{EngineVersion: "parity-engine-v1", Scenario: factory}

			// The reference applies the same create, reset, and no-launch step
			// requests as method calls, with no bytes in between.
			seeds := scenario.SeedRange{Start: 10, End: 20}
			reference := newTestBatch(t, factory, agent.IdleFactory{}, 1, seeds, ScheduleTrainV2, 0, 1)
			wantReset, err := reference.Reset(ctx, []int{0}, test.overrides)
			if err != nil {
				t.Fatalf("reference Reset() error = %v", err)
			}
			wantStep, err := reference.Step(ctx, map[int][BodySlots]SlotAction{0: {}})
			if err != nil {
				t.Fatalf("reference Step() error = %v", err)
			}
			wantCreate := protocolCreateResult{
				engine: "parity-engine-v1", schema: "obs-v2",
				generator: factory.ID(), opponent: agent.IdleV1,
				seedRange: [2]int64{seeds.Start, seeds.End},
				schedule:  string(ScheduleTrainV2), cursor: 0,
			}

			binaryInput := joinBinaryTestFrames(
				binaryTestFrame(binaryCreateRequestType, binaryCreateRequestPayload(1, [2]int64{10, 20}, 0, "idle-v1", "train-v2")),
				binaryTestFrame(binaryResetRequestType, binaryResetRequestPayload([]uint32{0}, test.overrides)),
				binaryTestFrame(binaryStepRequestType, binaryNoLaunchStepPayload([]uint32{0})),
				binaryTestFrame(binaryCloseRequestType, nil),
			)
			firstOutput := runBinaryProtocol(t, binaryInput, config)
			secondOutput := runBinaryProtocol(t, binaryInput, config)
			if !bytes.Equal(firstOutput, secondOutput) {
				t.Fatalf("identical env-v3 sessions produced different bytes: first=%d second=%d", len(firstOutput), len(secondOutput))
			}

			frames := decodeBinaryTestFrames(t, firstOutput)
			if len(frames) != 4 {
				t.Fatalf("env-v3 returned %d frames, want 4", len(frames))
			}
			assertBinaryResponseTypes(t, frames, []byte{
				binaryCreateResponseType,
				binaryResetResponseType,
				binaryStepResponseType,
				binaryCloseResponseType,
			})
			binaryCreate := decodeBinaryTestCreateResponse(t, frames[0].payload)
			binaryReset := decodeBinaryTestResetResponse(t, frames[1].payload)
			binaryStep := decodeBinaryTestStepResponse(t, frames[2].payload)
			if got := frames[3].payload; !bytes.Equal(got, []byte{1}) {
				t.Fatalf("binary close payload = %v, want [1]", got)
			}

			if !reflect.DeepEqual(binaryCreate, wantCreate) {
				t.Errorf("create response differs\nbatch:  %+v\nenv-v3: %+v", wantCreate, binaryCreate)
			}
			if !reflect.DeepEqual(binaryReset, wantReset) {
				t.Errorf("reset response differs\nbatch:  %+v\nenv-v3: %+v", wantReset, binaryReset)
			}
			if !reflect.DeepEqual(binaryStep, wantStep) {
				t.Errorf("step response differs\nbatch:  %+v\nenv-v3: %+v", wantStep, binaryStep)
			}
		})
	}
}

// TestBinaryProtocolJoinsCleanupFailureWithFatalOutputError checks that when a
// response write fails, the server still closes the live batch and reports both
// failures. Returning only the write error would hide a leaked opponent.
func TestBinaryProtocolJoinsCleanupFailureWithFatalOutputError(t *testing.T) {
	writeFailure := errors.New("output failed")
	closeFailure := errors.New("opponent close failed")
	factory := cleanupFailureFactory{closeErr: closeFailure}
	config := ProtocolConfig{
		EngineVersion: "cleanup-test-engine",
		Scenario:      testScenario{id: "cleanup-test-v1", maxTurns: 2},
	}
	// The create response is the one write allowed to succeed; the reset
	// response write then fails while the batch holds a live opponent.
	input := joinBinaryTestFrames(
		binaryTestFrame(binaryCreateRequestType, binaryCreateRequestPayload(1, [2]int64{0, 2}, 0, "cleanup-test-v1", "train-v2")),
		binaryTestFrame(binaryResetRequestType, binaryResetRequestPayload([]uint32{0}, nil)),
	)
	output := &failAfterOneWrite{err: writeFailure}
	err := serveBinaryProtocol(
		context.Background(), bytes.NewReader(input), output, config,
		func(string) (agent.Factory, error) { return factory, nil },
	)
	if !errors.Is(err, writeFailure) || !errors.Is(err, closeFailure) {
		t.Fatalf("serveBinaryProtocol() error = %v, want joined output and cleanup failures", err)
	}
}

func TestBinaryProtocolReturnsRecoverableErrorsAndContinues(t *testing.T) {
	config := ProtocolConfig{
		EngineVersion: "error-engine-v1",
		Scenario:      testScenario{id: "protocol-errors-v1", maxTurns: 3},
	}
	invalidTargets := binaryNoLaunchStepPayload([]uint32{0})
	invalidTargets[8] = BodySlots
	input := joinBinaryTestFrames(
		// Lifecycle checks run before payload decoding.
		binaryTestFrame(binaryResetRequestType, nil),
		binaryTestFrame(0x55, nil),
		binaryTestFrame(binaryCreateRequestType, binaryCreateRequestPayload(1, [2]int64{0, 10}, 0, "idle-v1", "train-v2")),
		binaryTestFrame(binaryResetRequestType, binaryResetRequestPayload([]uint32{0}, nil)),
		binaryTestFrame(binaryStepRequestType, invalidTargets),
		binaryTestFrame(binaryStepRequestType, binaryNoLaunchStepPayload([]uint32{0})),
		binaryTestFrame(binaryCloseRequestType, []byte{0}),
		binaryTestFrame(binaryCloseRequestType, nil),
	)
	frames := decodeBinaryTestFrames(t, runBinaryProtocol(t, input, config))
	if len(frames) != 8 {
		t.Fatalf("response count = %d, want 8", len(frames))
	}

	wantTypes := []byte{
		binaryErrorResponseType,
		binaryErrorResponseType,
		binaryCreateResponseType,
		binaryResetResponseType,
		binaryErrorResponseType,
		binaryStepResponseType,
		binaryErrorResponseType,
		binaryCloseResponseType,
	}
	assertBinaryResponseTypes(t, frames, wantTypes)

	errorsToCheck := []struct {
		frameIndex  int
		requestType byte
		kind        string
	}{
		{frameIndex: 0, requestType: binaryResetRequestType, kind: protocolErrorState},
		{frameIndex: 1, requestType: 0x55, kind: protocolErrorUnknownCommand},
		{frameIndex: 4, requestType: binaryStepRequestType, kind: protocolErrorValidation},
		{frameIndex: 6, requestType: binaryCloseRequestType, kind: protocolErrorMalformedBinary},
	}
	for _, want := range errorsToCheck {
		failedType, details := decodeBinaryTestErrorResponse(t, frames[want.frameIndex].payload)
		if failedType != want.requestType || details.Kind != want.kind || details.Message == "" {
			t.Errorf(
				"error frame %d = type %#x details %+v; want type %#x kind %q",
				want.frameIndex, failedType, details, want.requestType, want.kind,
			)
		}
	}
}

func TestBinaryProtocolRejectsFatalFramingWithoutResponse(t *testing.T) {
	validHeader := func(payloadSize uint32) []byte {
		header := make([]byte, binaryFrameHeaderSize)
		copy(header[:4], binaryFrameMagic)
		header[4] = binaryCreateRequestType
		binary.LittleEndian.PutUint32(header[8:12], payloadSize)
		return header
	}
	tests := []struct {
		name string
		wire []byte
		want string
	}{
		{name: "wrong version", wire: append([]byte("OWE1"), make([]byte, 8)...), want: "unsupported binary protocol version"},
		{name: "env-v2 client", wire: append([]byte("OWE2"), make([]byte, 8)...), want: "unsupported binary protocol version"},
		{name: "oversized", wire: validHeader(binaryMaxPayloadSize + 1), want: "maximum"},
		{name: "payload entirely missing", wire: validHeader(100), want: "frame payload: unexpected EOF"},
		{name: "payload partly missing", wire: append(validHeader(2), 1), want: "frame payload: unexpected EOF"},
	}
	config := ProtocolConfig{
		EngineVersion: "framing-engine-v1",
		Scenario:      testScenario{id: "protocol-framing-v1", maxTurns: 3},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var output bytes.Buffer
			err := ServeBinaryProtocol(context.Background(), bytes.NewReader(test.wire), &output, config)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want text %q", err, test.want)
			}
			if output.Len() != 0 {
				t.Fatalf("fatal framing error wrote %d response bytes", output.Len())
			}
		})
	}
}

func TestBinaryProtocolAcceptsCleanEOFAtFrameBoundary(t *testing.T) {
	config := ProtocolConfig{
		EngineVersion: "eof-engine-v1",
		Scenario:      testScenario{id: "protocol-eof-v1", maxTurns: 3},
	}
	var output bytes.Buffer
	if err := ServeBinaryProtocol(context.Background(), bytes.NewReader(nil), &output, config); err != nil {
		t.Fatalf("ServeBinaryProtocol() at clean EOF = %v", err)
	}
	if output.Len() != 0 {
		t.Fatalf("clean EOF wrote %d response bytes", output.Len())
	}
}

func TestBinaryProtocolCloseReturnsWhilePipeWriterRemainsOpen(t *testing.T) {
	reader, writer := io.Pipe()
	defer writer.Close()
	config := ProtocolConfig{
		EngineVersion: "pipe-engine-v1",
		Scenario:      testScenario{id: "protocol-pipe-v1", maxTurns: 3},
	}
	var output bytes.Buffer
	done := make(chan error, 1)
	go func() {
		done <- ServeBinaryProtocol(context.Background(), reader, &output, config)
	}()
	requests := joinBinaryTestFrames(
		binaryTestFrame(binaryCreateRequestType, binaryCreateRequestPayload(1, [2]int64{0, 2}, 0, "idle-v1", "train-v2")),
		binaryTestFrame(binaryCloseRequestType, nil),
	)
	if _, err := writer.Write(requests); err != nil {
		t.Fatalf("write requests: %v", err)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("ServeBinaryProtocol() error = %v", err)
		}
	case <-time.After(2 * time.Second):
		_ = reader.Close()
		t.Fatal("binary protocol did not return after close while writer remained open")
	}
	if _, err := writer.Write([]byte{0}); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("write after server return = %v, want io.ErrClosedPipe", err)
	}
}

func TestBinaryProtocolCancellationUnblocksIdleInput(t *testing.T) {
	reader, writer := io.Pipe()
	defer writer.Close()
	ctx, cancel := context.WithCancel(context.Background())
	config := ProtocolConfig{
		EngineVersion: "cancel-engine-v1",
		Scenario:      testScenario{id: "protocol-cancel-v1", maxTurns: 3},
	}
	done := make(chan error, 1)
	go func() {
		done <- ServeBinaryProtocol(ctx, reader, io.Discard, config)
	}()

	// No request bytes are written. Cancellation must interrupt the ReadFull
	// that is waiting for an env-v3 frame header rather than leaving the server
	// goroutine tied to an idle parent process.
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("ServeBinaryProtocol() after cancellation = %v", err)
		}
	case <-time.After(2 * time.Second):
		_ = reader.Close()
		t.Fatal("binary protocol did not return after cancellation")
	}
}

func runBinaryProtocol(t *testing.T, input []byte, config ProtocolConfig) []byte {
	t.Helper()
	var output bytes.Buffer
	if err := ServeBinaryProtocol(context.Background(), bytes.NewReader(input), &output, config); err != nil {
		t.Fatalf("ServeBinaryProtocol() error = %v", err)
	}
	return append([]byte(nil), output.Bytes()...)
}

func binaryTestFrame(messageType byte, payload []byte) []byte {
	frame := make([]byte, binaryFrameHeaderSize, binaryFrameHeaderSize+len(payload))
	copy(frame[:4], binaryFrameMagic)
	frame[4] = messageType
	binary.LittleEndian.PutUint32(frame[8:12], uint32(len(payload)))
	return append(frame, payload...)
}

func joinBinaryTestFrames(frames ...[]byte) []byte {
	return bytes.Join(frames, nil)
}

func binaryCreateRequestPayload(
	batch uint32,
	seedRange [2]int64,
	cursor int64,
	opponent string,
	schedule string,
) []byte {
	encoder := newBinaryTestEncoder(64 + len(opponent) + len(schedule))
	encoder.u32(batch)
	encoder.i64(seedRange[0])
	encoder.i64(seedRange[1])
	encoder.i64(cursor)
	encoder.string(opponent)
	encoder.string(schedule)
	return encoder.bytes()
}

func binaryResetRequestPayload(ids []uint32, overrides []EpisodeOverride) []byte {
	encoder := newBinaryTestEncoder(8 + len(ids)*4 + len(overrides)*13)
	encoder.u32(uint32(len(ids)))
	for _, id := range ids {
		encoder.u32(id)
	}
	encoder.u32(uint32(len(overrides)))
	for _, override := range overrides {
		encoder.u32(uint32(override.EnvID))
		encoder.i64(override.Seed)
		encoder.u8(byte(override.Seat))
	}
	return encoder.bytes()
}

func binaryNoLaunchStepPayload(ids []uint32) []byte {
	encoder := newBinaryTestEncoder(4 + len(ids)*(4+BodySlots))
	encoder.u32(uint32(len(ids)))
	for _, id := range ids {
		encoder.u32(id)
	}
	for range ids {
		for range BodySlots {
			encoder.u8(0xff)
		}
	}
	return encoder.bytes()
}

func decodeBinaryTestFrames(t testing.TB, wire []byte) []binaryFrame {
	t.Helper()
	reader := bytes.NewReader(wire)
	var frames []binaryFrame
	for {
		frame, err := readBinaryFrame(reader)
		if errors.Is(err, io.EOF) {
			return frames
		}
		if err != nil {
			t.Fatalf("decode response frame: %v", err)
		}
		frames = append(frames, frame)
	}
}

func assertBinaryResponseTypes(t *testing.T, frames []binaryFrame, want []byte) {
	t.Helper()
	if len(frames) != len(want) {
		t.Fatalf("response count = %d, want %d", len(frames), len(want))
	}
	for index, frame := range frames {
		if frame.messageType != want[index] {
			t.Errorf("response %d type = %#x, want %#x", index, frame.messageType, want[index])
		}
	}
}

// decodeBinaryTestCreateResponse reads a create payload back into the
// protocolCreateResult the server encoded it from, so tests can compare the two
// directly. The obs-v2 dimensions that follow have no field there; they are
// checked against the schema's shapes instead.
func decodeBinaryTestCreateResponse(t testing.TB, payload []byte) protocolCreateResult {
	t.Helper()
	decoder := newBinaryResponseTestDecoder(t, payload)
	response := protocolCreateResult{
		engine:    decoder.string(),
		schema:    decoder.string(),
		generator: decoder.string(),
		opponent:  decoder.string(),
	}
	response.seedRange = [2]int64{decoder.i64(), decoder.i64()}
	response.schedule = decoder.string()
	response.cursor = decoder.i64()
	dimensions := [4]uint32{decoder.u32(), decoder.u32(), decoder.u32(), decoder.u32()}
	if want := [4]uint32{BodySlots, BodyFeatures, PlayerSlots, PlayerFeatures}; dimensions != want {
		t.Fatalf("create dimensions = %v, want %v", dimensions, want)
	}
	decoder.finish()
	return response
}

// decodeBinaryTestResetResponse reads a reset payload back into Batch.Reset's
// own result type.
func decodeBinaryTestResetResponse(t testing.TB, payload []byte) ResetResult {
	t.Helper()
	decoder := newBinaryResponseTestDecoder(t, payload)
	cursor := decoder.i64()
	count := decoder.count()
	identities := decoder.episodeIdentities(count)
	observations := decoder.observations(count)
	response := ResetResult{Cursor: cursor, Envs: make([]ResetEnv, count)}
	for index := range count {
		response.Envs[index] = ResetEnv{EpisodeInfo: identities[index], Observation: observations[index]}
	}
	decoder.finish()
	return response
}

// decodeBinaryTestStepResponse reads a step payload back into Batch.Step's own
// result type.
func decodeBinaryTestStepResponse(t testing.TB, payload []byte) []StepResult {
	t.Helper()
	decoder := newBinaryResponseTestDecoder(t, payload)
	count := decoder.count()
	identities := decoder.episodeIdentities(count)
	rewards := make([]float32, count)
	terminated := make([]bool, count)
	truncated := make([]bool, count)
	endReasons := make([]EndReason, count)
	turns := make([]int, count)
	for index := range count {
		rewards[index] = decoder.f32()
	}
	for index := range count {
		terminated[index] = decoder.boolean()
	}
	for index := range count {
		truncated[index] = decoder.boolean()
	}
	for index := range count {
		endReasons[index] = EndReason(decoder.u8())
	}
	for index := range count {
		turns[index] = int(decoder.u32())
	}
	observations := decoder.observations(count)
	response := make([]StepResult, count)
	for index := range count {
		response[index] = StepResult{
			EpisodeInfo: identities[index], Observation: observations[index],
			Reward: rewards[index], Terminated: terminated[index],
			Truncated: truncated[index], EndReason: endReasons[index], Turn: turns[index],
		}
	}
	decoder.finish()
	return response
}

func decodeBinaryTestErrorResponse(t testing.TB, payload []byte) (byte, ProtocolError) {
	t.Helper()
	decoder := newBinaryResponseTestDecoder(t, payload)
	requestType := decoder.u8()
	envIDValue := decoder.i32()
	details := ProtocolError{Kind: decoder.string(), Message: decoder.string()}
	if envIDValue >= 0 {
		envID := int(envIDValue)
		details.EnvID = &envID
	}
	decoder.finish()
	return requestType, details
}

// binaryResponseTestDecoder accepts testing.TB rather than *testing.T so the
// same response decoders serve both the routing tests and the protocol
// benchmarks. Benchmarks must read real env-v3 responses so the observations
// they feed back into the next step are the ones the protocol produced.
type binaryResponseTestDecoder struct {
	t       testing.TB
	payload []byte
	offset  int
}

func newBinaryResponseTestDecoder(t testing.TB, payload []byte) *binaryResponseTestDecoder {
	t.Helper()
	return &binaryResponseTestDecoder{t: t, payload: payload}
}

func (decoder *binaryResponseTestDecoder) take(size int) []byte {
	decoder.t.Helper()
	if size < 0 || size > len(decoder.payload)-decoder.offset {
		decoder.t.Fatalf(
			"test response decoder at byte %d needs %d bytes, has %d",
			decoder.offset, size, len(decoder.payload)-decoder.offset,
		)
	}
	start := decoder.offset
	decoder.offset += size
	return decoder.payload[start:decoder.offset]
}

func (decoder *binaryResponseTestDecoder) u8() byte {
	return decoder.take(1)[0]
}

func (decoder *binaryResponseTestDecoder) boolean() bool {
	value := decoder.u8()
	if value > 1 {
		decoder.t.Fatalf("test response bool = %d, want 0 or 1", value)
	}
	return value == 1
}

func (decoder *binaryResponseTestDecoder) u32() uint32 {
	return binary.LittleEndian.Uint32(decoder.take(4))
}

func (decoder *binaryResponseTestDecoder) i32() int32 {
	return int32(decoder.u32())
}

func (decoder *binaryResponseTestDecoder) i64() int64 {
	return int64(binary.LittleEndian.Uint64(decoder.take(8)))
}

func (decoder *binaryResponseTestDecoder) f32() float32 {
	return math.Float32frombits(decoder.u32())
}

func (decoder *binaryResponseTestDecoder) f64() float64 {
	return math.Float64frombits(binary.LittleEndian.Uint64(decoder.take(8)))
}

func (decoder *binaryResponseTestDecoder) string() string {
	length := decoder.u32()
	if uint64(length) > uint64(len(decoder.payload)-decoder.offset) {
		decoder.t.Fatalf("test response string length %d exceeds remaining payload", length)
	}
	return string(decoder.take(int(length)))
}

func (decoder *binaryResponseTestDecoder) count() int {
	value := decoder.u32()
	if uint64(value) > uint64(math.MaxInt) {
		decoder.t.Fatalf("test response count %d exceeds int range", value)
	}
	return int(value)
}

func (decoder *binaryResponseTestDecoder) episodeIdentities(count int) []EpisodeInfo {
	identities := make([]EpisodeInfo, count)
	for index := range count {
		identities[index].EnvID = int(decoder.u32())
	}
	for index := range count {
		identities[index].Seed = decoder.i64()
	}
	for index := range count {
		identities[index].Seat = int(decoder.u8())
	}
	for index := range count {
		identities[index].Explicit = decoder.boolean()
	}
	return identities
}

func (decoder *binaryResponseTestDecoder) observations(count int) []Observation {
	observations := make([]Observation, count)
	for envIndex := range count {
		for row := range BodySlots {
			for feature := range BodyFeatures {
				observations[envIndex].Bodies[row][feature] = decoder.f32()
			}
		}
	}
	for envIndex := range count {
		for row := range PlayerSlots {
			for feature := range PlayerFeatures {
				observations[envIndex].Players[row][feature] = decoder.f32()
			}
		}
	}
	for envIndex := range count {
		for index := range BodySlots {
			observations[envIndex].BodyMask[index] = decoder.boolean()
		}
	}
	for envIndex := range count {
		for index := range BodySlots {
			observations[envIndex].OwnedMask[index] = decoder.boolean()
		}
	}
	for envIndex := range count {
		for index := range PlayerSlots {
			observations[envIndex].PlayerMask[index] = decoder.boolean()
		}
	}
	for envIndex := range count {
		for source := range BodySlots {
			for target := range BodySlots {
				observations[envIndex].TargetMask[source][target] = decoder.boolean()
			}
		}
	}
	for envIndex := range count {
		for index := range BodySlots {
			observations[envIndex].SlotBodyID[index] = decoder.i32()
		}
	}
	return observations
}

func (decoder *binaryResponseTestDecoder) finish() {
	decoder.t.Helper()
	if decoder.offset != len(decoder.payload) {
		decoder.t.Fatalf("test response decoder left %d trailing bytes", len(decoder.payload)-decoder.offset)
	}
}

// cleanupFailureFactory builds opponents whose Close always fails, so a test
// can see whether the server's deferred batch cleanup reported that failure.
type cleanupFailureFactory struct {
	closeErr error
}

func (cleanupFailureFactory) Spec() agent.Spec {
	return agent.NewSpec("cleanup-test-v1", nil)
}

func (factory cleanupFailureFactory) New(context.Context, agent.AgentInit) (agent.Agent, error) {
	return cleanupFailureAgent{closeErr: factory.closeErr}, nil
}

type cleanupFailureAgent struct {
	closeErr error
}

func (cleanupFailureAgent) ID() string { return "cleanup-test-v1" }

func (cleanupFailureAgent) Decide(context.Context, agent.TurnRequest) (agent.Decision, error) {
	return agent.Decision{}, nil
}

func (agent cleanupFailureAgent) Close() error { return agent.closeErr }

// failAfterOneWrite accepts its first write and fails every later one, which
// lets one response reach the caller before the output breaks.
type failAfterOneWrite struct {
	writes int
	err    error
}

func (writer *failAfterOneWrite) Write(data []byte) (int, error) {
	if writer.writes > 0 {
		return 0, writer.err
	}
	writer.writes++
	return len(data), nil
}
