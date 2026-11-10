package mlenv

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"unicode/utf8"

	"orrery/agent"
)

const (
	// An env-v3 frame has a small fixed header followed by one payload. Keeping
	// framing separate from command decoding lets the stream reader reject a
	// corrupt length before allocating or interpreting application data.
	binaryFrameHeaderSize = 12
	binaryMaxPayloadSize  = 64 * 1024 * 1024
	binaryMaxStringSize   = 1024 * 1024
	// Opponent and schedule names select small, versioned implementations. They
	// do not carry bulk data, so tighter limits prevent a validator that quotes
	// an invalid name from expanding a megabyte of control characters into a
	// multi-megabyte error message.
	binaryMaxOpponentSize = 1024
	binaryMaxScheduleSize = 64
	binaryErrorSuffix     = "... [truncated]"

	binaryCreateRequestType  byte = 0x01
	binaryResetRequestType   byte = 0x02
	binaryStepRequestType    byte = 0x03
	binaryCloseRequestType   byte = 0x04
	binaryCreateResponseType byte = 0x81
	binaryResetResponseType  byte = 0x82
	binaryStepResponseType   byte = 0x83
	binaryCloseResponseType  byte = 0x84
	binaryErrorResponseType  byte = 0xff

	// One obs-v2 observation on the wire: seven columns in a fixed order —
	// body features, player features, body mask, owned mask, player mask,
	// target mask, slot body IDs. Floats and body IDs occupy four bytes; each
	// mask value occupies one byte. The size is asserted again after response
	// encoding.
	binaryObservationBytesPerEnvironment = BodySlots*BodyFeatures*4 +
		PlayerSlots*PlayerFeatures*4 +
		BodySlots + BodySlots + PlayerSlots +
		BodySlots*BodySlots + BodySlots*4
	binaryResetIdentityBytesPerEnvironment = 4 + 8 + 1 + 1
	// A step result adds reward (f32), terminated, truncated, end reason (one
	// byte each), and turn (u32) to the reset identity.
	binaryStepResultBytesPerEnvironment = binaryResetIdentityBytesPerEnvironment + 4 + 1 + 1 + 1 + 4

	// binaryMaxBatchSize is the largest batch whose full step response fits one
	// frame: 9,482 environments. The step response is the largest message per
	// environment (7,077 bytes, against 7,066 for reset), and the 4 is its
	// leading environment count. Create accepts a batch by checking both
	// responses with binaryPayloadSize; this value only names the limit in that
	// rejection's message. It is a wire ceiling, not a sensible training size.
	binaryMaxBatchSize = (binaryMaxPayloadSize - 4) /
		(binaryStepResultBytesPerEnvironment + binaryObservationBytesPerEnvironment)
)

// binaryFrameMagic marks env-v3 frames. It differs from the old env-v2's
// "OWE2", so a reader still speaking that protocol fails on the header instead
// of misreading obs-v2 payloads.
const binaryFrameMagic = "OWE3"

// binaryFrame is one already-delimited message. Payload owns its bytes so the
// next read cannot mutate data while the current command is being processed.
type binaryFrame struct {
	// messageType tells the receiver how to decode the payload: 0x01 through
	// 0x04 are create/reset/step/close requests, the matching response sets the
	// high bit (for example, step 0x03 becomes 0x83), and 0xff is an error.
	messageType byte
	payload     []byte
}

// binaryEncodedPayload owns a response buffer whose first 12 bytes are
// reserved for its frame header. Response encoders build directly after that
// prefix, allowing the completed frame to cross the process pipe with one
// Write call and no second full-payload copy.
type binaryEncodedPayload struct {
	framed []byte
}

func (payload binaryEncodedPayload) bytes() []byte {
	return payload.framed[binaryFrameHeaderSize:]
}

// binaryCreateRequest contains the values needed to construct the existing
// Batch. It deliberately has no binary-specific game behavior.
type binaryCreateRequest struct {
	batch     int
	seedRange [2]int64
	cursor    int64
	opponent  string
	schedule  ScheduleMode
}

// binaryResetRequest maps directly onto Batch.Reset's two arguments.
type binaryResetRequest struct {
	ids       []int
	overrides []EpisodeOverride
}

// readBinaryFrame reads exactly one env-v3 frame from a stream.
//
// Header failures are fatal to the stream: after a bad magic, impossible
// length, or truncated body, the caller cannot know where another frame would
// begin and must discard the process instead of attempting to continue.
func readBinaryFrame(input io.Reader) (binaryFrame, error) {
	var header [binaryFrameHeaderSize]byte
	read, err := io.ReadFull(input, header[:])
	if err != nil {
		if read == 0 && errors.Is(err, io.EOF) {
			return binaryFrame{}, io.EOF
		}
		return binaryFrame{}, fmt.Errorf("read env-v3 frame header: %w", err)
	}

	if header[0] == 'O' && header[1] == 'W' && header[2] == 'E' && header[3] != binaryFrameMagic[3] {
		return binaryFrame{}, fmt.Errorf(
			"unsupported binary protocol version %q; want %q",
			header[3:4], binaryFrameMagic[3:4],
		)
	}
	if string(header[:4]) != binaryFrameMagic {
		return binaryFrame{}, fmt.Errorf("invalid env-v3 frame magic %q", header[:4])
	}
	if header[5] != 0 || binary.LittleEndian.Uint16(header[6:8]) != 0 {
		return binaryFrame{}, errors.New("env-v3 frame flags and reserved bytes must be zero")
	}

	payloadSize := binary.LittleEndian.Uint32(header[8:12])
	if err := validateBinaryPayloadSize(uint64(payloadSize)); err != nil {
		return binaryFrame{}, err
	}
	payload := make([]byte, int(payloadSize))
	if _, err := io.ReadFull(input, payload); err != nil {
		// io.ReadFull returns plain EOF when the peer sends a complete header
		// and then closes before writing even the first payload byte. Plain EOF
		// means a clean boundary to the stream loop, so translate this case to
		// UnexpectedEOF to preserve the fatal truncated-frame distinction.
		if errors.Is(err, io.EOF) {
			err = io.ErrUnexpectedEOF
		}
		return binaryFrame{}, fmt.Errorf("read env-v3 frame payload: %w", err)
	}
	return binaryFrame{messageType: header[4], payload: payload}, nil
}

// writeBinaryFrame fills the header space reserved by binaryEncodedPayload and
// publishes the complete contiguous frame. A normal os.File write therefore
// crosses the process pipe once without copying the payload into another
// buffer. writeAll still handles writers that accept only part of that slice.
func writeBinaryFrame(output io.Writer, messageType byte, payload binaryEncodedPayload) error {
	if len(payload.framed) < binaryFrameHeaderSize {
		return errors.New("env-v3 response buffer does not reserve a frame header")
	}
	payloadBytes := payload.bytes()
	if err := validateBinaryPayloadSize(uint64(len(payloadBytes))); err != nil {
		return err
	}
	header := payload.framed[:binaryFrameHeaderSize]
	clear(header)
	copy(header[:4], binaryFrameMagic)
	header[4] = messageType
	binary.LittleEndian.PutUint32(header[8:12], uint32(len(payloadBytes)))
	if err := writeAll(output, payload.framed); err != nil {
		return fmt.Errorf("write env-v3 frame: %w", err)
	}
	return nil
}

func writeAll(output io.Writer, data []byte) error {
	for len(data) > 0 {
		written, err := output.Write(data)
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
		data = data[written:]
	}
	return nil
}

// ServeBinaryProtocol runs the stateful create/reset/step/close environment
// lifecycle over env-v3 frames until a successful close request or EOF.
// Recoverable, fully framed request failures receive error frames; framing,
// I/O, and response-encoding failures terminate the stream.
func ServeBinaryProtocol(
	ctx context.Context,
	input io.Reader,
	output io.Writer,
	config ProtocolConfig,
) error {
	return serveBinaryProtocol(ctx, input, output, config, agent.ParseFactory)
}

func serveBinaryProtocol(
	ctx context.Context,
	input io.Reader,
	output io.Writer,
	config ProtocolConfig,
	parseOpponent func(string) (agent.Factory, error),
) (returnErr error) {
	if err := validateProtocolConfig(config); err != nil {
		return err
	}

	server := protocolServer{config: config, parseOpponent: parseOpponent}
	defer server.joinCleanupError(&returnErr)

	// Frame reading blocks while Python is idle. Read in a child goroutine so
	// context cancellation can stop the server, while closeInputOnCancel wakes
	// an os.File or pipe that is already blocked inside ReadFull.
	readCtx, cancelRead := context.WithCancel(ctx)
	stopCancelRead := closeInputOnCancel(readCtx, input)
	frames, readDone := scanBinaryProtocolInput(readCtx, input)
	defer func() {
		cancelRead()
		if ctx.Err() != nil {
			return
		}
		<-readDone
		stopCancelRead()
	}()

	for {
		select {
		case <-readCtx.Done():
			return nil
		case read := <-frames:
			if read.done {
				// Cancellation closes a pipe or file to wake ReadFull. Prefer
				// the caller's cancellation over that induced read error if the
				// two become selectable at the same time.
				if read.err != nil && ctx.Err() == nil {
					return fmt.Errorf("read env-v3 request: %w", read.err)
				}
				return nil
			}
			response, err := server.handleBinary(ctx, read.frame)
			if err != nil {
				return err
			}
			if err := writeBinaryFrame(output, response.messageType, response.payload); err != nil {
				return fmt.Errorf("write env-v3 response: %w", err)
			}
			if response.closeAfter {
				return nil
			}
		}
	}
}

type binaryProtocolRead struct {
	frame binaryFrame
	err   error
	done  bool
}

// binaryProtocolResponse keeps the three pieces of a dispatch result together,
// including structured responses for recoverable request errors. The server
// writes messageType and payload as one frame, then stops only when closeAfter
// says that the response completed the protocol lifecycle. Fatal failures
// remain the separate error returned by handleBinary because no response can
// safely be written for them.
type binaryProtocolResponse struct {
	messageType byte
	payload     binaryEncodedPayload
	closeAfter  bool
}

// scanBinaryProtocolInput owns all reads from the stream. It delivers complete
// frames in order and reports one terminal result for EOF or a fatal framing
// error, so the request loop never observes a partially read message.
func scanBinaryProtocolInput(
	ctx context.Context,
	input io.Reader,
) (<-chan binaryProtocolRead, <-chan struct{}) {
	results := make(chan binaryProtocolRead)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			frame, err := readBinaryFrame(input)
			if err != nil {
				if errors.Is(err, io.EOF) {
					err = nil
				}
				select {
				case results <- binaryProtocolRead{err: err, done: true}:
				case <-ctx.Done():
				}
				return
			}
			select {
			case results <- binaryProtocolRead{frame: frame}:
			case <-ctx.Done():
				return
			}
		}
	}()
	return results, done
}

// handleBinary applies lifecycle checks before decoding each payload, so a
// request sent in the wrong state reports invalid-state rather than a decode
// error. Once decoded, requests call the batch methods in protocol.go; this
// switch only selects wire codecs.
func (server *protocolServer) handleBinary(
	ctx context.Context,
	frame binaryFrame,
) (binaryProtocolResponse, error) {
	switch frame.messageType {
	case binaryCreateRequestType:
		if err := server.requireNotCreated(); err != nil {
			return rejectedBinaryRequest(frame.messageType, err, false), nil
		}
		request, err := decodeBinaryCreateRequest(frame.payload)
		if err != nil {
			return rejectedBinaryRequest(frame.messageType, err, false), nil
		}
		result, err := server.createBatch(
			ctx, request.batch, request.seedRange, request.opponent,
			request.schedule, request.cursor,
		)
		if err != nil {
			return rejectedBinaryRequest(frame.messageType, err, false), nil
		}
		response, err := encodeBinaryCreateResponse(result)
		if err != nil {
			return binaryProtocolResponse{}, fmt.Errorf("encode env-v3 create response: %w", err)
		}
		return binaryProtocolResponse{
			messageType: binaryCreateResponseType,
			payload:     response,
		}, nil

	case binaryResetRequestType:
		if err := server.requireCreated("reset"); err != nil {
			return rejectedBinaryRequest(frame.messageType, err, false), nil
		}
		request, err := decodeBinaryResetRequest(frame.payload)
		if err != nil {
			return rejectedBinaryRequest(frame.messageType, err, false), nil
		}
		result, err := server.batch.Reset(ctx, request.ids, request.overrides)
		if err != nil {
			return rejectedBinaryRequest(frame.messageType, err, false), nil
		}
		response, err := encodeBinaryResetResponse(result)
		if err != nil {
			return binaryProtocolResponse{}, fmt.Errorf("encode env-v3 reset response: %w", err)
		}
		return binaryProtocolResponse{
			messageType: binaryResetResponseType,
			payload:     response,
		}, nil

	case binaryStepRequestType:
		if err := server.requireCreated("step"); err != nil {
			return rejectedBinaryRequest(frame.messageType, err, false), nil
		}
		actions, err := decodeBinaryStepRequest(frame.payload)
		if err != nil {
			return rejectedBinaryRequest(frame.messageType, err, false), nil
		}
		results, err := server.batch.Step(ctx, actions)
		if err != nil {
			return rejectedBinaryRequest(frame.messageType, err, false), nil
		}
		response, err := encodeBinaryStepResponse(results)
		if err != nil {
			// Step has already advanced Go. Continuing after losing its response
			// would make Python and Go disagree about the current states.
			return binaryProtocolResponse{}, fmt.Errorf("encode env-v3 step response: %w", err)
		}
		return binaryProtocolResponse{
			messageType: binaryStepResponseType,
			payload:     response,
		}, nil

	case binaryCloseRequestType:
		if err := server.requireCreated("close"); err != nil {
			return rejectedBinaryRequest(frame.messageType, err, false), nil
		}
		if err := decodeBinaryCloseRequest(frame.payload); err != nil {
			return rejectedBinaryRequest(frame.messageType, err, false), nil
		}
		if err := server.closeBatch(); err != nil {
			return rejectedBinaryRequest(frame.messageType, err, true), nil
		}
		return binaryProtocolResponse{
			messageType: binaryCloseResponseType,
			payload:     encodeBinaryCloseResponse(),
			closeAfter:  true,
		}, nil

	default:
		return rejectedBinaryRequest(
			frame.messageType,
			&protocolFailure{
				kind: protocolErrorUnknownCommand,
				err:  fmt.Errorf("unknown env-v3 message type %#x", frame.messageType),
			},
			false,
		), nil
	}
}

func rejectedBinaryRequest(
	requestType byte,
	err error,
	closeAfter bool,
) binaryProtocolResponse {
	return binaryProtocolResponse{
		messageType: binaryErrorResponseType,
		payload:     encodeBinaryErrorResponse(requestType, err),
		closeAfter:  closeAfter,
	}
}

// decodeBinaryCreateRequest turns the fixed create payload into ordinary Go
// values. NewBatch remains responsible for semantic checks such as positive
// batch size, a valid seed range, and a supported schedule.
func decodeBinaryCreateRequest(payload []byte) (binaryCreateRequest, error) {
	decoder := newBinaryDecoder(payload)
	batchValue, err := decoder.u32("batch size")
	if err != nil {
		return binaryCreateRequest{}, err
	}
	batch, err := binaryUint32ToInt(batchValue, "batch size")
	if err != nil {
		return binaryCreateRequest{}, err
	}
	seedStart, err := decoder.i64("seed-range start")
	if err != nil {
		return binaryCreateRequest{}, err
	}
	seedEnd, err := decoder.i64("seed-range end")
	if err != nil {
		return binaryCreateRequest{}, err
	}
	cursor, err := decoder.i64("schedule cursor")
	if err != nil {
		return binaryCreateRequest{}, err
	}
	opponent, err := decoder.string("opponent")
	if err != nil {
		return binaryCreateRequest{}, err
	}
	schedule, err := decoder.string("schedule")
	if err != nil {
		return binaryCreateRequest{}, err
	}
	if err := decoder.finish(); err != nil {
		return binaryCreateRequest{}, err
	}
	if len(opponent) > binaryMaxOpponentSize {
		return binaryCreateRequest{}, validationFailure(
			"opponent identity is %d bytes, maximum is %d",
			len(opponent), binaryMaxOpponentSize,
		)
	}
	if len(schedule) > binaryMaxScheduleSize {
		return binaryCreateRequest{}, validationFailure(
			"schedule identity is %d bytes, maximum is %d",
			len(schedule), binaryMaxScheduleSize,
		)
	}
	// A batch whose full reset or step response cannot fit one frame would be
	// created and then fail while encoding a response, after Go had already
	// changed its games. Checking here, with the same size function the
	// encoders use, refuses it before NewBatch builds anything.
	_, stepErr := binaryPayloadSize(4, batch, binaryStepResultBytesPerEnvironment+binaryObservationBytesPerEnvironment)
	_, resetErr := binaryPayloadSize(12, batch, binaryResetIdentityBytesPerEnvironment+binaryObservationBytesPerEnvironment)
	if stepErr != nil || resetErr != nil {
		return binaryCreateRequest{}, validationFailure(
			"batch size %d does not fit one env-v3 step response, maximum is %d",
			batch, binaryMaxBatchSize,
		)
	}
	return binaryCreateRequest{
		batch: batch, seedRange: [2]int64{seedStart, seedEnd}, cursor: cursor,
		opponent: opponent, schedule: ScheduleMode(schedule),
	}, nil
}

// decodeBinaryResetRequest reads IDs and optional replay overrides without
// changing their order. Batch.Reset performs the authoritative normalization,
// duplicate detection, range checks, and deterministic sorting.
func decodeBinaryResetRequest(payload []byte) (binaryResetRequest, error) {
	decoder := newBinaryDecoder(payload)
	count, err := decoder.count("reset environment count", 4)
	if err != nil {
		return binaryResetRequest{}, err
	}
	ids := make([]int, count)
	for index := range count {
		value, readErr := decoder.u32(fmt.Sprintf("reset environment ID %d", index))
		if readErr != nil {
			return binaryResetRequest{}, readErr
		}
		ids[index], err = binaryUint32ToInt(value, "reset environment ID")
		if err != nil {
			return binaryResetRequest{}, err
		}
	}

	overrideCount, err := decoder.count("explicit override count", 13)
	if err != nil {
		return binaryResetRequest{}, err
	}
	overrides := make([]EpisodeOverride, overrideCount)
	for index := range overrideCount {
		idValue, readErr := decoder.u32(fmt.Sprintf("override %d environment ID", index))
		if readErr != nil {
			return binaryResetRequest{}, readErr
		}
		id, conversionErr := binaryUint32ToInt(idValue, "override environment ID")
		if conversionErr != nil {
			return binaryResetRequest{}, conversionErr
		}
		seed, readErr := decoder.i64(fmt.Sprintf("override %d seed", index))
		if readErr != nil {
			return binaryResetRequest{}, readErr
		}
		seat, readErr := decoder.u8(fmt.Sprintf("override %d seat", index))
		if readErr != nil {
			return binaryResetRequest{}, readErr
		}
		overrides[index] = EpisodeOverride{EnvID: id, Seed: seed, Seat: int(seat)}
	}
	if err := decoder.finish(); err != nil {
		return binaryResetRequest{}, err
	}
	return binaryResetRequest{ids: ids, overrides: overrides}, nil
}

// decodeBinaryStepRequest converts each packed target byte into the existing
// SlotAction type. This is the only representation change: Batch.Step still
// validates and executes the actions itself.
func decodeBinaryStepRequest(payload []byte) (map[int][BodySlots]SlotAction, error) {
	decoder := newBinaryDecoder(payload)
	countValue, err := decoder.u32("step environment count")
	if err != nil {
		return nil, err
	}
	count := uint64(countValue)
	expected := uint64(4) + count*uint64(4+BodySlots)
	if expected != uint64(len(payload)) {
		return nil, malformedBinary(
			"step payload has %d bytes, want %d for %d environments",
			len(payload), expected, countValue,
		)
	}
	ids := make([]int, int(count))
	// Install an empty row while reading IDs so this same map detects
	// duplicates before any target bytes are interpreted. The second pass then
	// replaces each placeholder with its decoded decisions.
	actions := make(map[int][BodySlots]SlotAction, int(count))
	for index := range ids {
		value, readErr := decoder.u32(fmt.Sprintf("step environment ID %d", index))
		if readErr != nil {
			return nil, readErr
		}
		id, conversionErr := binaryUint32ToInt(value, "step environment ID")
		if conversionErr != nil {
			return nil, conversionErr
		}
		if _, duplicate := actions[id]; duplicate {
			return nil, validationFailure("step repeats environment ID %d", id)
		}
		actions[id] = [BodySlots]SlotAction{}
		ids[index] = id
	}

	for _, id := range ids {
		var decisions [BodySlots]SlotAction
		for source := range BodySlots {
			target, readErr := decoder.i8(fmt.Sprintf("env %d source slot %d target", id, source))
			if readErr != nil {
				return nil, readErr
			}
			switch {
			case target == -1:
				decisions[source] = SlotAction{}
			case target >= 0 && int(target) < BodySlots:
				decisions[source] = SlotAction{Launch: true, Target: int32(target)}
			default:
				return nil, validationFailure(
					"env %d source slot %d has invalid packed target %d",
					id, source, target,
				)
			}
		}
		actions[id] = decisions
	}
	if err := decoder.finish(); err != nil {
		return nil, err
	}
	return actions, nil
}

// decodeBinaryCloseRequest enforces that close carries no hidden or trailing
// data before the server releases its batch.
func decodeBinaryCloseRequest(payload []byte) error {
	if len(payload) != 0 {
		return malformedBinary("close payload has %d bytes, want 0", len(payload))
	}
	return nil
}

// encodeBinaryCreateResponse records every identity required to interpret and
// replay the run created by the server, then the obs-v2 dimensions. The
// dimensions are fixed by the schema, so they are redundant with it; sending
// them lets the reader check its own shapes against the server's before it
// decodes a single observation.
func encodeBinaryCreateResponse(result protocolCreateResult) (binaryEncodedPayload, error) {
	encoder := newBinaryResponseEncoder(256)
	for _, field := range []struct {
		name  string
		value string
	}{
		{name: "engine", value: result.engine},
		{name: "schema", value: result.schema},
		{name: "generator", value: result.generator},
		{name: "opponent", value: result.opponent},
	} {
		if err := encoder.string(field.name, field.value); err != nil {
			return binaryEncodedPayload{}, err
		}
	}
	encoder.i64(result.seedRange[0])
	encoder.i64(result.seedRange[1])
	if err := encoder.string("schedule", result.schedule); err != nil {
		return binaryEncodedPayload{}, err
	}
	encoder.i64(result.cursor)
	encoder.u32(BodySlots)
	encoder.u32(BodyFeatures)
	encoder.u32(PlayerSlots)
	encoder.u32(PlayerFeatures)
	return encoder.encodedPayload(), nil
}

// encodeBinaryResetResponse writes identities first and observations second.
// Keeping every field column contiguous lets Python construct seven complete
// NumPy arrays rather than allocating one Python object per value.
func encodeBinaryResetResponse(result ResetResult) (binaryEncodedPayload, error) {
	count := len(result.Envs)
	expected, err := binaryPayloadSize(12, count, binaryResetIdentityBytesPerEnvironment+binaryObservationBytesPerEnvironment)
	if err != nil {
		return binaryEncodedPayload{}, err
	}
	encoder := newBinaryResponseEncoder(expected)
	encoder.i64(result.Cursor)
	encoder.u32(uint32(count))
	if err := encodeBinaryEpisodeIdentities(encoder, count, func(index int) EpisodeInfo {
		return result.Envs[index].EpisodeInfo
	}); err != nil {
		return binaryEncodedPayload{}, err
	}
	if err := encodeBinaryObservations(encoder, count, func(index int) *Observation {
		return &result.Envs[index].Observation
	}); err != nil {
		return binaryEncodedPayload{}, err
	}
	if len(encoder.bytes()) != expected {
		return binaryEncodedPayload{}, fmt.Errorf("encoded reset payload has %d bytes, want %d", len(encoder.bytes()), expected)
	}
	return encoder.encodedPayload(), nil
}

// encodeBinaryStepResponse uses the same identity and observation columns as
// reset, with reward and terminal data inserted between them.
func encodeBinaryStepResponse(results []StepResult) (binaryEncodedPayload, error) {
	count := len(results)
	expected, err := binaryPayloadSize(4, count, binaryStepResultBytesPerEnvironment+binaryObservationBytesPerEnvironment)
	if err != nil {
		return binaryEncodedPayload{}, err
	}
	encoder := newBinaryResponseEncoder(expected)
	encoder.u32(uint32(count))
	if err := encodeBinaryEpisodeIdentities(encoder, count, func(index int) EpisodeInfo {
		return results[index].EpisodeInfo
	}); err != nil {
		return binaryEncodedPayload{}, err
	}
	for _, result := range results {
		if !finiteFloat32(result.Reward) {
			return binaryEncodedPayload{}, fmt.Errorf("encode env %d reward: value must be finite", result.EnvID)
		}
		encoder.f32(result.Reward)
	}
	for _, result := range results {
		encoder.boolean(result.Terminated)
	}
	for _, result := range results {
		encoder.boolean(result.Truncated)
	}
	// The end reason must agree with Terminated: a reader that trusts either
	// column alone would otherwise be told two different things.
	for _, result := range results {
		if (result.EndReason != EndOngoing) != result.Terminated || result.EndReason > EndLearnerEliminated {
			return binaryEncodedPayload{}, fmt.Errorf("encode env %d end reason %d with terminated %t", result.EnvID, result.EndReason, result.Terminated)
		}
		encoder.u8(byte(result.EndReason))
	}
	for _, result := range results {
		if result.Turn < 0 || uint64(result.Turn) > math.MaxUint32 {
			return binaryEncodedPayload{}, fmt.Errorf("encode env %d turn %d outside u32 range", result.EnvID, result.Turn)
		}
		encoder.u32(uint32(result.Turn))
	}
	if err := encodeBinaryObservations(encoder, count, func(index int) *Observation {
		return &results[index].Observation
	}); err != nil {
		return binaryEncodedPayload{}, err
	}
	if len(encoder.bytes()) != expected {
		return binaryEncodedPayload{}, fmt.Errorf("encoded step payload has %d bytes, want %d", len(encoder.bytes()), expected)
	}
	return encoder.encodedPayload(), nil
}

// encodeBinaryCloseResponse returns the one-byte acknowledgement required by
// the contract. It is separate from framing so the server can use one writer
// for success and error payloads.
func encodeBinaryCloseResponse() binaryEncodedPayload {
	encoder := newBinaryResponseEncoder(1)
	encoder.boolean(true)
	return encoder.encodedPayload()
}

// encodeBinaryErrorResponse writes the stable error kind, environment ID, and
// message, plus the request type needed to correlate a binary response.
//
// This encoder is deliberately total: it must always produce a bounded payload
// for a recoverable request. Error strings can contain invalid UTF-8 supplied
// by an implementation or grow when a validator quotes control characters.
// Sanitizing and truncating here prevents reporting an error from becoming a
// second protocol error that leaves the two processes out of sync.
func encodeBinaryErrorResponse(requestType byte, err error) binaryEncodedPayload {
	details := protocolErrorDetails(err)
	kind := boundedBinaryErrorString(details.Kind)
	message := boundedBinaryErrorString(details.Message)
	envID := int32(-1)
	if details.EnvID != nil {
		if *details.EnvID >= 0 && int64(*details.EnvID) <= math.MaxInt32 {
			envID = int32(*details.EnvID)
		} else {
			// A batch large enough to produce this ID cannot exist in practice,
			// but error reporting must remain encodable even for a bad caller.
			message = boundedBinaryErrorString(fmt.Sprintf(
				"environment ID %d cannot be represented by env-v3: %s",
				*details.EnvID, message,
			))
		}
	}
	// Compute the capacity after the exceptional unrepresentable-ID branch,
	// because that branch adds context to the message before it is encoded.
	encoder := newBinaryResponseEncoder(64 + len(kind) + len(message))
	encoder.u8(requestType)
	encoder.i32(envID)
	encoder.prevalidatedString(kind)
	encoder.prevalidatedString(message)
	return encoder.encodedPayload()
}

// boundedBinaryErrorString returns valid UTF-8 that fits one wire string. It
// cuts only at a rune boundary so the truncation itself never creates malformed
// text. The suffix tells a reader that the diagnostic is incomplete.
func boundedBinaryErrorString(value string) string {
	value = strings.ToValidUTF8(value, "\uFFFD")
	if len(value) <= binaryMaxStringSize {
		return value
	}
	limit := binaryMaxStringSize - len(binaryErrorSuffix)
	for limit > 0 && !utf8.RuneStart(value[limit]) {
		limit--
	}
	return value[:limit] + binaryErrorSuffix
}

func encodeBinaryEpisodeIdentities(
	encoder *binaryResponseEncoder,
	count int,
	identity func(int) EpisodeInfo,
) error {
	previousID := -1
	for index := range count {
		info := identity(index)
		if info.EnvID < 0 || uint64(info.EnvID) > math.MaxUint32 {
			return fmt.Errorf("encode environment ID %d outside u32 range", info.EnvID)
		}
		if index > 0 && info.EnvID <= previousID {
			return fmt.Errorf("encode environment IDs must be strictly increasing; %d follows %d", info.EnvID, previousID)
		}
		encoder.u32(uint32(info.EnvID))
		previousID = info.EnvID
	}
	for index := range count {
		info := identity(index)
		encoder.i64(info.Seed)
	}
	for index := range count {
		info := identity(index)
		seat := info.Seat
		if seat < 0 || seat >= PlayerSlots {
			return fmt.Errorf(
				"encode env %d seat %d, want 0..%d",
				info.EnvID, seat, PlayerSlots-1,
			)
		}
		encoder.u8(byte(seat))
	}
	for index := range count {
		info := identity(index)
		encoder.boolean(info.Explicit)
	}
	return nil
}

func encodeBinaryObservations(
	encoder *binaryResponseEncoder,
	count int,
	observation func(int) *Observation,
) error {
	// Each loop writes one complete column across the batch. The matching
	// Python decoder can then make an ndarray directly over that byte range.
	for envIndex := range count {
		obs := observation(envIndex)
		for row := range BodySlots {
			for feature := range BodyFeatures {
				value := obs.Bodies[row][feature]
				if !finiteFloat32(value) {
					return fmt.Errorf("encode env row %d body[%d][%d]: value must be finite", envIndex, row, feature)
				}
				encoder.f32(value)
			}
		}
	}
	for envIndex := range count {
		obs := observation(envIndex)
		for row := range PlayerSlots {
			for feature := range PlayerFeatures {
				value := obs.Players[row][feature]
				if !finiteFloat32(value) {
					return fmt.Errorf("encode env row %d player[%d][%d]: value must be finite", envIndex, row, feature)
				}
				encoder.f32(value)
			}
		}
	}
	for envIndex := range count {
		for _, value := range observation(envIndex).BodyMask {
			encoder.boolean(value)
		}
	}
	for envIndex := range count {
		for _, value := range observation(envIndex).OwnedMask {
			encoder.boolean(value)
		}
	}
	for envIndex := range count {
		for _, value := range observation(envIndex).PlayerMask {
			encoder.boolean(value)
		}
	}
	for envIndex := range count {
		obs := observation(envIndex)
		for source := range BodySlots {
			for target := range BodySlots {
				encoder.boolean(obs.TargetMask[source][target])
			}
		}
	}
	for envIndex := range count {
		for _, id := range observation(envIndex).SlotBodyID {
			encoder.i32(id)
		}
	}
	return nil
}

func binaryPayloadSize(prefix int, count int, bytesPerEnvironment int) (int, error) {
	if count < 0 || uint64(count) > math.MaxUint32 {
		return 0, fmt.Errorf("environment count %d outside u32 range", count)
	}
	size := uint64(prefix) + uint64(count)*uint64(bytesPerEnvironment)
	if err := validateBinaryPayloadSize(size); err != nil {
		return 0, err
	}
	return int(size), nil
}

// validateBinaryPayloadSize centralizes the frame limit so readers, writers,
// and size calculations enforce the same boundary. Taking a number rather than
// a byte slice also lets boundary tests avoid allocating a 64 MiB buffer.
func validateBinaryPayloadSize(size uint64) error {
	if size > binaryMaxPayloadSize {
		return fmt.Errorf("env-v3 payload is %d bytes, maximum is %d", size, binaryMaxPayloadSize)
	}
	return nil
}

func finiteFloat32(value float32) bool {
	converted := float64(value)
	return !math.IsNaN(converted) && !math.IsInf(converted, 0)
}

func binaryUint32ToInt(value uint32, name string) (int, error) {
	if uint64(value) > uint64(math.MaxInt) {
		return 0, validationFailure("%s %d exceeds platform integer range", name, value)
	}
	return int(value), nil
}

func malformedBinary(format string, args ...any) error {
	return &protocolFailure{
		kind: protocolErrorMalformedBinary,
		err:  fmt.Errorf(format, args...),
	}
}

// binaryDecoder advances one checked offset through a bounded frame payload.
// Every field names itself so malformed-input errors identify the boundary the
// sender violated instead of reporting only an opaque end-of-buffer failure.
type binaryDecoder struct {
	payload []byte
	offset  int
}

func newBinaryDecoder(payload []byte) *binaryDecoder {
	return &binaryDecoder{payload: payload}
}

func (decoder *binaryDecoder) take(size int, name string) ([]byte, error) {
	if size < 0 || size > len(decoder.payload)-decoder.offset {
		return nil, malformedBinary(
			"decode %s at byte %d: need %d bytes, have %d",
			name, decoder.offset, size, len(decoder.payload)-decoder.offset,
		)
	}
	start := decoder.offset
	decoder.offset += size
	return decoder.payload[start:decoder.offset], nil
}

func (decoder *binaryDecoder) u8(name string) (byte, error) {
	data, err := decoder.take(1, name)
	if err != nil {
		return 0, err
	}
	return data[0], nil
}

func (decoder *binaryDecoder) i8(name string) (int8, error) {
	value, err := decoder.u8(name)
	return int8(value), err
}

func (decoder *binaryDecoder) u32(name string) (uint32, error) {
	data, err := decoder.take(4, name)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(data), nil
}

func (decoder *binaryDecoder) i64(name string) (int64, error) {
	data, err := decoder.take(8, name)
	if err != nil {
		return 0, err
	}
	return int64(binary.LittleEndian.Uint64(data)), nil
}

func (decoder *binaryDecoder) string(name string) (string, error) {
	length, err := decoder.u32(name + " length")
	if err != nil {
		return "", err
	}
	if length > binaryMaxStringSize {
		return "", malformedBinary("%s is %d bytes, maximum is %d", name, length, binaryMaxStringSize)
	}
	data, err := decoder.take(int(length), name)
	if err != nil {
		return "", err
	}
	if !utf8.Valid(data) {
		return "", malformedBinary("%s is not valid UTF-8", name)
	}
	return string(data), nil
}

func (decoder *binaryDecoder) count(name string, minimumItemBytes int) (int, error) {
	value, err := decoder.u32(name)
	if err != nil {
		return 0, err
	}
	if minimumItemBytes > 0 && uint64(value) > uint64((len(decoder.payload)-decoder.offset)/minimumItemBytes) {
		return 0, malformedBinary(
			"%s %d exceeds remaining %d-byte payload",
			name, value, len(decoder.payload)-decoder.offset,
		)
	}
	return binaryUint32ToInt(value, name)
}

func (decoder *binaryDecoder) finish() error {
	if decoder.offset != len(decoder.payload) {
		return malformedBinary("payload has %d trailing bytes", len(decoder.payload)-decoder.offset)
	}
	return nil
}

// binaryResponseEncoder always starts with space for a frame header, then
// appends response scalars explicitly in wire order. There is no headerless
// mode, so every value returned by encodedPayload is valid by construction.
type binaryResponseEncoder struct {
	framed []byte
}

func newBinaryResponseEncoder(payloadCapacity int) *binaryResponseEncoder {
	return &binaryResponseEncoder{
		framed: make([]byte, binaryFrameHeaderSize, binaryFrameHeaderSize+payloadCapacity),
	}
}

func (encoder *binaryResponseEncoder) u8(value byte) {
	encoder.framed = append(encoder.framed, value)
}

func (encoder *binaryResponseEncoder) boolean(value bool) {
	if value {
		encoder.u8(1)
		return
	}
	encoder.u8(0)
}

func (encoder *binaryResponseEncoder) u32(value uint32) {
	encoder.framed = binary.LittleEndian.AppendUint32(encoder.framed, value)
}

func (encoder *binaryResponseEncoder) i32(value int32) {
	encoder.u32(uint32(value))
}

func (encoder *binaryResponseEncoder) i64(value int64) {
	encoder.framed = binary.LittleEndian.AppendUint64(encoder.framed, uint64(value))
}

func (encoder *binaryResponseEncoder) f32(value float32) {
	encoder.u32(math.Float32bits(value))
}

func (encoder *binaryResponseEncoder) f64(value float64) {
	encoder.framed = binary.LittleEndian.AppendUint64(encoder.framed, math.Float64bits(value))
}

func (encoder *binaryResponseEncoder) string(name string, value string) error {
	if !utf8.ValidString(value) {
		return fmt.Errorf("encode %s: value is not valid UTF-8", name)
	}
	if len(value) > binaryMaxStringSize {
		return fmt.Errorf("encode %s: %d bytes, maximum is %d", name, len(value), binaryMaxStringSize)
	}
	encoder.u32(uint32(len(value)))
	encoder.framed = append(encoder.framed, value...)
	return nil
}

// prevalidatedString is used only by the error encoder after it has repaired
// UTF-8 and enforced the size limit. Keeping this operation infallible is what
// lets a recoverable request always receive a response.
func (encoder *binaryResponseEncoder) prevalidatedString(value string) {
	encoder.u32(uint32(len(value)))
	encoder.framed = append(encoder.framed, value...)
}

func (encoder *binaryResponseEncoder) bytes() []byte {
	return encoder.framed[binaryFrameHeaderSize:]
}

func (encoder *binaryResponseEncoder) encodedPayload() binaryEncodedPayload {
	return binaryEncodedPayload{framed: encoder.framed}
}
