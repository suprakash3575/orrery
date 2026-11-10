package mlenv

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"

	"orrery/agent"
)

func TestBinaryFrameRoundTrip(t *testing.T) {
	payload := []byte{0, 1, 2, 0xff}
	encoder := newBinaryResponseEncoder(len(payload))
	encoder.framed = append(encoder.framed, payload...)
	var wire bytes.Buffer
	if err := writeBinaryFrame(&wire, binaryStepRequestType, encoder.encodedPayload()); err != nil {
		t.Fatalf("writeBinaryFrame() error = %v", err)
	}

	frame, err := readBinaryFrame(&wire)
	if err != nil {
		t.Fatalf("readBinaryFrame() error = %v", err)
	}
	if frame.messageType != binaryStepRequestType {
		t.Errorf("message type = %#x, want %#x", frame.messageType, binaryStepRequestType)
	}
	if !bytes.Equal(frame.payload, payload) {
		t.Errorf("payload = %v, want %v", frame.payload, payload)
	}
	if wire.Len() != 0 {
		t.Fatalf("reader left %d bytes after one frame", wire.Len())
	}
}

func TestWriteBinaryFrameUsesOneWrite(t *testing.T) {
	encoder := newBinaryResponseEncoder(4)
	encoder.framed = append(encoder.framed, 1, 2, 3, 4)
	output := &countingBinaryWriter{}
	if err := writeBinaryFrame(output, binaryStepResponseType, encoder.encodedPayload()); err != nil {
		t.Fatalf("writeBinaryFrame() error = %v", err)
	}
	if output.calls != 1 {
		t.Errorf("Write calls = %d, want 1", output.calls)
	}
}

func TestReadBinaryFrameRejectsBrokenFraming(t *testing.T) {
	validHeader := func(messageType byte, payloadSize uint32) []byte {
		header := make([]byte, binaryFrameHeaderSize)
		copy(header[:4], binaryFrameMagic)
		header[4] = messageType
		binary.LittleEndian.PutUint32(header[8:12], payloadSize)
		return header
	}

	tests := []struct {
		name    string
		wire    []byte
		want    string
		wantEOF bool
	}{
		{name: "empty stream", wantEOF: true},
		{name: "truncated header", wire: []byte("OWE"), want: "frame header"},
		{name: "wrong version", wire: append([]byte("OWE1"), make([]byte, 8)...), want: "unsupported binary protocol version"},
		// An env-v2 client must fail on the header, not on an obs-v1-sized payload.
		{name: "env-v2 frame", wire: append([]byte("OWE2"), make([]byte, 8)...), want: `unsupported binary protocol version "2"; want "3"`},
		{name: "wrong magic", wire: append([]byte("NOPE"), make([]byte, 8)...), want: "frame magic"},
		{name: "nonzero flags", wire: func() []byte { value := validHeader(binaryStepRequestType, 0); value[5] = 1; return value }(), want: "must be zero"},
		{name: "nonzero reserved", wire: func() []byte { value := validHeader(binaryStepRequestType, 0); value[6] = 1; return value }(), want: "must be zero"},
		{name: "oversized payload", wire: validHeader(binaryStepRequestType, binaryMaxPayloadSize+1), want: "maximum"},
		{name: "truncated payload", wire: append(validHeader(binaryStepRequestType, 2), 1), want: "frame payload"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := readBinaryFrame(bytes.NewReader(test.wire))
			if test.wantEOF {
				if !errors.Is(err, io.EOF) {
					t.Fatalf("error = %v, want io.EOF", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want text %q", err, test.want)
			}
		})
	}
}

func TestDecodeBinaryCreateAndResetRequests(t *testing.T) {
	create := newBinaryTestEncoder(64)
	create.u32(64)
	create.i64(100)
	create.i64(200)
	create.i64(17)
	create.string("heuristic-v1")
	create.string("rotate")

	created, err := decodeBinaryCreateRequest(create.bytes())
	if err != nil {
		t.Fatalf("decodeBinaryCreateRequest() error = %v", err)
	}
	if created.batch != 64 || created.seedRange != [2]int64{100, 200} ||
		created.cursor != 17 || created.opponent != "heuristic-v1" ||
		created.schedule != ScheduleRotate {
		t.Fatalf("decoded create = %+v", created)
	}

	reset := newBinaryTestEncoder(64)
	reset.u32(2)
	reset.u32(10)
	reset.u32(2)
	reset.u32(1)
	reset.u32(2)
	reset.i64(4711)
	reset.u8(1)

	decodedReset, err := decodeBinaryResetRequest(reset.bytes())
	if err != nil {
		t.Fatalf("decodeBinaryResetRequest() error = %v", err)
	}
	if len(decodedReset.ids) != 2 || decodedReset.ids[0] != 10 || decodedReset.ids[1] != 2 {
		t.Fatalf("reset IDs = %v, want [10 2]", decodedReset.ids)
	}
	wantOverride := EpisodeOverride{EnvID: 2, Seed: 4711, Seat: 1}
	if len(decodedReset.overrides) != 1 || decodedReset.overrides[0] != wantOverride {
		t.Fatalf("reset overrides = %+v, want %+v", decodedReset.overrides, wantOverride)
	}
}

func TestDecodeBinaryCreateRequestRejectsOversizedIdentities(t *testing.T) {
	tests := []struct {
		name     string
		opponent string
		schedule string
		want     string
	}{
		{
			name:     "opponent",
			opponent: strings.Repeat("x", binaryMaxOpponentSize+1),
			schedule: "train-v2",
			want:     "opponent identity is 1025 bytes, maximum is 1024",
		},
		{
			name:     "schedule",
			opponent: "idle-v1",
			schedule: strings.Repeat("s", binaryMaxScheduleSize+1),
			want:     "schedule identity is 65 bytes, maximum is 64",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			encoder := newBinaryTestEncoder(128 + len(test.opponent) + len(test.schedule))
			encoder.u32(1)
			encoder.i64(0)
			encoder.i64(1)
			encoder.i64(0)
			encoder.string(test.opponent)
			encoder.string(test.schedule)

			_, err := decodeBinaryCreateRequest(encoder.bytes())
			if err == nil || err.Error() != test.want {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
			assertBinaryProtocolFailureKind(t, err, protocolErrorValidation)
		})
	}
}

func TestDecodeBinaryCreateRequestLimitsBatchSize(t *testing.T) {
	// The size arithmetic is what create relies on, so pin it first: the
	// largest accepted batch's full step and reset responses fit one frame,
	// and one more environment no longer fits a step response.
	stepBytes := binaryStepResultBytesPerEnvironment + binaryObservationBytesPerEnvironment
	resetBytes := binaryResetIdentityBytesPerEnvironment + binaryObservationBytesPerEnvironment
	if _, err := binaryPayloadSize(4, binaryMaxBatchSize, stepBytes); err != nil {
		t.Errorf("step response for %d environments: %v", binaryMaxBatchSize, err)
	}
	if _, err := binaryPayloadSize(12, binaryMaxBatchSize, resetBytes); err != nil {
		t.Errorf("reset response for %d environments: %v", binaryMaxBatchSize, err)
	}
	if _, err := binaryPayloadSize(4, binaryMaxBatchSize+1, stepBytes); err == nil {
		t.Errorf("step response for %d environments fits, want it rejected", binaryMaxBatchSize+1)
	}

	// The literal 9482 checks the derived constant against the documented
	// limit. Every case decodes only, so no large buffer is allocated.
	tests := []struct {
		name  string
		batch uint32
		want  string
	}{
		{name: "largest batch", batch: 9482},
		{name: "one more", batch: 9483, want: "batch size 9483 does not fit one env-v3 step response, maximum is 9482"},
		// MaxInt32 is the largest batch that converts to int on every platform.
		// Anything larger fails that conversion first on a 32-bit build, with a
		// different message, before the size check is reached.
		{name: "int32 maximum", batch: math.MaxInt32, want: "batch size 2147483647 does not fit one env-v3 step response, maximum is 9482"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			payload := binaryCreateRequestPayload(test.batch, [2]int64{0, 10}, 0, "idle-v1", "train-v2")
			request, err := decodeBinaryCreateRequest(payload)
			if test.want == "" {
				if err != nil {
					t.Fatalf("decodeBinaryCreateRequest() error = %v", err)
				}
				if request.batch != int(test.batch) {
					t.Fatalf("batch = %d, want %d", request.batch, test.batch)
				}
				return
			}
			if err == nil || err.Error() != test.want {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
			assertBinaryProtocolFailureKind(t, err, protocolErrorValidation)
		})
	}
}

func TestBinaryRequestDecodersRejectMalformedPayloads(t *testing.T) {
	invalidUTF8 := newBinaryTestEncoder(32)
	invalidUTF8.u32(1)
	invalidUTF8.i64(0)
	invalidUTF8.i64(1)
	invalidUTF8.i64(0)
	invalidUTF8.u32(1)
	invalidUTF8.u8(0xff)
	invalidUTF8.u32(8)
	invalidUTF8.data = append(invalidUTF8.data, "train-v2"...)

	resetImpossibleCount := newBinaryTestEncoder(8)
	resetImpossibleCount.u32(2)
	resetImpossibleCount.u32(0)

	tests := []struct {
		name    string
		decode  func() error
		message string
	}{
		{
			name: "create truncation",
			decode: func() error {
				_, err := decodeBinaryCreateRequest(make([]byte, 3))
				return err
			},
			message: "need 4 bytes",
		},
		{
			name: "create invalid UTF-8",
			decode: func() error {
				_, err := decodeBinaryCreateRequest(invalidUTF8.bytes())
				return err
			},
			message: "not valid UTF-8",
		},
		{
			name: "reset impossible count",
			decode: func() error {
				_, err := decodeBinaryResetRequest(resetImpossibleCount.bytes())
				return err
			},
			message: "exceeds remaining",
		},
		{
			name:    "close trailing byte",
			decode:  func() error { return decodeBinaryCloseRequest([]byte{0}) },
			message: "want 0",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.decode()
			if err == nil || !strings.Contains(err.Error(), test.message) {
				t.Fatalf("error = %v, want text %q", err, test.message)
			}
			assertBinaryProtocolFailureKind(t, err, protocolErrorMalformedBinary)
		})
	}
}

func TestDecodeBinaryStepRequestConvertsPackedTargets(t *testing.T) {
	payload := make([]byte, 4+2*4+2*BodySlots)
	binary.LittleEndian.PutUint32(payload[:4], 2)
	binary.LittleEndian.PutUint32(payload[4:8], 10)
	binary.LittleEndian.PutUint32(payload[8:12], 2)
	for index := 12; index < len(payload); index++ {
		payload[index] = 0xff // int8(-1) means no launch.
	}
	payload[12+3] = 7
	payload[12+BodySlots+5] = 0
	// The last source slot of the first row may target the last body slot,
	// which only exists in obs-v2's 44-slot layout.
	payload[12+BodySlots-1] = BodySlots - 1

	actions, err := decodeBinaryStepRequest(payload)
	if err != nil {
		t.Fatalf("decodeBinaryStepRequest() error = %v", err)
	}
	if len(actions) != 2 {
		t.Fatalf("decoded %d action rows, want 2", len(actions))
	}
	if got := actions[10][3]; !got.Launch || got.Target != 7 {
		t.Errorf("env 10 source 3 = %+v, want launch at 7", got)
	}
	if got := actions[2][5]; !got.Launch || got.Target != 0 {
		t.Errorf("env 2 source 5 = %+v, want launch at 0", got)
	}
	if got := actions[10][0]; got != (SlotAction{}) {
		t.Errorf("env 10 source 0 = %+v, want no launch", got)
	}
	if got := actions[10][BodySlots-1]; !got.Launch || got.Target != BodySlots-1 {
		t.Errorf("env 10 last source = %+v, want launch at the last slot", got)
	}
}

func TestDecodeBinaryStepRequestRejectsInvalidRows(t *testing.T) {
	valid := make([]byte, 4+4+BodySlots)
	binary.LittleEndian.PutUint32(valid[:4], 1)
	binary.LittleEndian.PutUint32(valid[4:8], 3)
	for index := 8; index < len(valid); index++ {
		valid[index] = 0xff
	}

	invalidTarget := append([]byte(nil), valid...)
	invalidTarget[8] = BodySlots
	invalidNegativeTarget := append([]byte(nil), valid...)
	invalidNegativeTarget[8] = 0xfe // int8(-2); only -1 is the no-launch sentinel.

	duplicateID := make([]byte, 4+2*4+2*BodySlots)
	binary.LittleEndian.PutUint32(duplicateID[:4], 2)
	binary.LittleEndian.PutUint32(duplicateID[4:8], 3)
	binary.LittleEndian.PutUint32(duplicateID[8:12], 3)
	for index := 12; index < len(duplicateID); index++ {
		duplicateID[index] = 0xff
	}

	tests := []struct {
		name string
		data []byte
		kind string
		want string
	}{
		{name: "truncated", data: valid[:len(valid)-1], kind: protocolErrorMalformedBinary, want: "payload has"},
		{name: "trailing", data: append(valid, 0), kind: protocolErrorMalformedBinary, want: "payload has"},
		{name: "invalid target", data: invalidTarget, kind: protocolErrorValidation, want: "invalid packed target"},
		{name: "invalid negative target", data: invalidNegativeTarget, kind: protocolErrorValidation, want: "invalid packed target -2"},
		{name: "duplicate ID", data: duplicateID, kind: protocolErrorValidation, want: "repeats environment ID"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := decodeBinaryStepRequest(test.data)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want text %q", err, test.want)
			}
			assertBinaryProtocolFailureKind(t, err, test.kind)
		})
	}
}

func TestEncodeBinaryResetResponseUsesDocumentedColumns(t *testing.T) {
	observation := binaryTestObservation(1)
	encoded, err := encodeBinaryResetResponse(ResetResult{
		Cursor: 19,
		Envs: []ResetEnv{{
			EpisodeInfo: EpisodeInfo{EnvID: 3, Seed: 4711, Seat: 3, Explicit: true},
			Observation: observation,
		}},
	})
	if err != nil {
		t.Fatalf("encodeBinaryResetResponse() error = %v", err)
	}
	payload := encoded.bytes()
	if got, want := len(payload), 12+binaryResetIdentityBytesPerEnvironment+binaryObservationBytesPerEnvironment; got != want {
		t.Fatalf("payload length = %d, want %d", got, want)
	}
	if got := int64(binary.LittleEndian.Uint64(payload[0:8])); got != 19 {
		t.Errorf("cursor = %d, want 19", got)
	}
	if got := binary.LittleEndian.Uint32(payload[8:12]); got != 1 {
		t.Errorf("count = %d, want 1", got)
	}
	if got := binary.LittleEndian.Uint32(payload[12:16]); got != 3 {
		t.Errorf("env ID = %d, want 3", got)
	}
	if got := int64(binary.LittleEndian.Uint64(payload[16:24])); got != 4711 {
		t.Errorf("seed = %d, want 4711", got)
	}
	if payload[24] != 3 || payload[25] != 1 {
		t.Errorf("seat/explicit = %v, want [3 1]", payload[24:26])
	}
	assertBinaryObservationOffsets(t, payload, 26, 1, observation)
}

func TestEncodeBinaryStepResponseUsesDocumentedColumns(t *testing.T) {
	first := binaryTestObservation(1)
	second := binaryTestObservation(2)
	results := []StepResult{
		{
			EpisodeInfo: EpisodeInfo{EnvID: 2, Seed: 100, Seat: 0},
			Observation: first, Reward: -1, Terminated: true, EndReason: EndLearnerEliminated, Turn: 7,
		},
		{
			EpisodeInfo: EpisodeInfo{EnvID: 10, Seed: 200, Seat: 1, Explicit: true},
			Observation: second, Reward: 1, Truncated: true, Turn: 9,
		},
	}
	encoded, err := encodeBinaryStepResponse(results)
	if err != nil {
		t.Fatalf("encodeBinaryStepResponse() error = %v", err)
	}
	payload := encoded.bytes()
	if got, want := len(payload), 4+2*(binaryStepResultBytesPerEnvironment+binaryObservationBytesPerEnvironment); got != want {
		t.Fatalf("payload length = %d, want %d", got, want)
	}

	// The scalar arrays are columns rather than per-environment records. These
	// offsets are calculated independently from the documented field widths.
	if got := binary.LittleEndian.Uint32(payload[0:4]); got != 2 {
		t.Errorf("count = %d, want 2", got)
	}
	if got := []uint32{binary.LittleEndian.Uint32(payload[4:8]), binary.LittleEndian.Uint32(payload[8:12])}; got[0] != 2 || got[1] != 10 {
		t.Errorf("environment IDs = %v, want [2 10]", got)
	}
	if got := []byte{payload[28], payload[29]}; got[0] != 0 || got[1] != 1 {
		t.Errorf("seats = %v, want [0 1]", got)
	}
	if got := []byte{payload[30], payload[31]}; got[0] != 0 || got[1] != 1 {
		t.Errorf("explicit flags = %v, want [0 1]", got)
	}
	if got := math.Float32frombits(binary.LittleEndian.Uint32(payload[32:36])); got != -1 {
		t.Errorf("first reward = %v, want -1", got)
	}
	if got := math.Float32frombits(binary.LittleEndian.Uint32(payload[36:40])); got != 1 {
		t.Errorf("second reward = %v, want 1", got)
	}
	if got := payload[40:44]; !bytes.Equal(got, []byte{1, 0, 0, 1}) {
		t.Errorf("terminal columns = %v, want [1 0 0 1]", got)
	}
	if got := payload[44:46]; !bytes.Equal(got, []byte{byte(EndLearnerEliminated), byte(EndOngoing)}) {
		t.Errorf("end reasons = %v, want [2 0]", got)
	}
	if got := binary.LittleEndian.Uint32(payload[46:50]); got != 7 {
		t.Errorf("first turn = %d, want 7", got)
	}
	if got := binary.LittleEndian.Uint32(payload[50:54]); got != 9 {
		t.Errorf("second turn = %d, want 9", got)
	}
	assertBinaryObservationOffsets(t, payload, 54, 2, first)
	assertBinaryObservationOffsets(t, payload, 54, 2, second)
}

func TestBinaryResponseEncodingRejectsInvalidResults(t *testing.T) {
	tests := []struct {
		name   string
		result []StepResult
		want   string
	}{
		{
			name: "unsorted IDs",
			result: []StepResult{
				{EpisodeInfo: EpisodeInfo{EnvID: 2, Seat: 0}},
				{EpisodeInfo: EpisodeInfo{EnvID: 1, Seat: 0}},
			},
			want: "strictly increasing",
		},
		{
			name: "non-finite reward",
			result: []StepResult{{
				EpisodeInfo: EpisodeInfo{EnvID: 0, Seat: 0}, Reward: float32(math.NaN()),
			}},
			want: "reward",
		},
		{
			name: "non-finite observation",
			result: []StepResult{{
				EpisodeInfo: EpisodeInfo{EnvID: 0, Seat: 0},
				Observation: func() Observation {
					value := Observation{}
					value.Bodies[0][0] = float32(math.Inf(1))
					return value
				}(),
			}},
			want: "must be finite",
		},
		{
			name: "seat outside the player slots",
			result: []StepResult{{
				EpisodeInfo: EpisodeInfo{EnvID: 0, Seat: PlayerSlots},
			}},
			want: "want 0..",
		},
		{
			name: "terminated without an end reason",
			result: []StepResult{{
				EpisodeInfo: EpisodeInfo{EnvID: 0, Seat: 0}, Terminated: true,
			}},
			want: "end reason",
		},
		{
			name: "end reason while still playing",
			result: []StepResult{{
				EpisodeInfo: EpisodeInfo{EnvID: 0, Seat: 0}, EndReason: EndGameFinished,
			}},
			want: "end reason",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := encodeBinaryStepResponse(test.result)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want text %q", err, test.want)
			}
		})
	}
}

func TestBinaryEncoderRangeGuards(t *testing.T) {
	if strconv.IntSize == 64 {
		// Go checks constant conversions at compile time, and this value does
		// not fit a 32-bit int, so the IntSize guard alone would not stop a
		// 32-bit build from failing. Going through a variable defers the
		// conversion to run time, where the guard keeps it from happening.
		maxUint32 := uint64(math.MaxUint32)
		aboveUint32 := int(maxUint32 + 1)
		tests := []struct {
			name   string
			result StepResult
			want   string
		}{
			{
				name: "environment ID",
				result: StepResult{EpisodeInfo: EpisodeInfo{
					EnvID: aboveUint32, Seat: 0,
				}},
				want: "environment ID",
			},
			{
				name: "turn",
				result: StepResult{
					EpisodeInfo: EpisodeInfo{EnvID: 0, Seat: 0}, Turn: aboveUint32,
				},
				want: "turn",
			},
		}
		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				_, err := encodeBinaryStepResponse([]StepResult{test.result})
				if err == nil || !strings.Contains(err.Error(), test.want) {
					t.Fatalf("error = %v, want text %q", err, test.want)
				}
			})
		}
	}

	if _, err := binaryPayloadSize(4, binaryMaxPayloadSize, 2); err == nil || !strings.Contains(err.Error(), "maximum") {
		t.Fatalf("binaryPayloadSize() error = %v, want maximum-size rejection", err)
	}
	encoder := newBinaryResponseEncoder(binaryMaxStringSize + 8)
	if err := encoder.string("oversized", strings.Repeat("x", binaryMaxStringSize+1)); err == nil || !strings.Contains(err.Error(), "maximum") {
		t.Fatalf("encoder.string() error = %v, want maximum-size rejection", err)
	}
	if err := validateBinaryPayloadSize(binaryMaxPayloadSize + 1); err == nil || !strings.Contains(err.Error(), "maximum") {
		t.Fatalf("validateBinaryPayloadSize() error = %v, want maximum-size rejection", err)
	}
}

func TestEncodeBinaryCreateAndErrorResponses(t *testing.T) {
	encoded, err := encodeBinaryCreateResponse(protocolCreateResult{
		engine: "engine", schema: SchemaID, generator: "generator-v2",
		opponent: "idle-v1", seedRange: [2]int64{11, 22}, schedule: "rotate",
		cursor: 9,
	})
	if err != nil {
		t.Fatalf("encodeBinaryCreateResponse() error = %v", err)
	}
	payload := encoded.bytes()
	offset := 0
	for _, want := range []string{"engine", "obs-v2", "generator-v2", "idle-v1"} {
		got := readBinaryTestString(t, payload, &offset)
		if got != want {
			t.Fatalf("decoded string = %q, want %q", got, want)
		}
	}
	for index, want := range []int64{11, 22} {
		got := int64(binary.LittleEndian.Uint64(payload[offset : offset+8]))
		offset += 8
		if got != want {
			t.Errorf("seed range value %d = %d, want %d", index, got, want)
		}
	}
	if got := readBinaryTestString(t, payload, &offset); got != "rotate" {
		t.Errorf("schedule = %q, want rotate", got)
	}
	if got := int64(binary.LittleEndian.Uint64(payload[offset : offset+8])); got != 9 {
		t.Errorf("cursor = %d, want 9", got)
	}
	offset += 8
	// Written out rather than read from the constants, so a change to the
	// obs-v2 shapes has to be made here on purpose.
	for index, want := range []uint32{44, 27, 4, 6} {
		got := binary.LittleEndian.Uint32(payload[offset : offset+4])
		offset += 4
		if got != want {
			t.Errorf("dimension %d = %d, want %d", index, got, want)
		}
	}
	if offset != len(payload) {
		t.Fatalf("create response left %d trailing bytes", len(payload)-offset)
	}

	envErr := &EnvError{Kind: ErrorValidation, EnvID: 7, Err: errors.New("bad action")}
	errorPayload := encodeBinaryErrorResponse(binaryStepRequestType, envErr).bytes()
	if errorPayload[0] != binaryStepRequestType {
		t.Errorf("failed request type = %#x, want %#x", errorPayload[0], binaryStepRequestType)
	}
	if got := int32(binary.LittleEndian.Uint32(errorPayload[1:5])); got != 7 {
		t.Errorf("error env ID = %d, want 7", got)
	}
}

func TestEncodeBinaryErrorResponseBoundsExpandedValidationMessage(t *testing.T) {
	// ParseFactory quotes an unknown opponent with %q. Each control byte becomes
	// the four printable bytes "\\x01", reproducing the expansion that could
	// previously make the error itself impossible to encode.
	_, parseErr := agent.ParseFactory(strings.Repeat("\x01", binaryMaxStringSize))
	if parseErr == nil {
		t.Fatal("ParseFactory() unexpectedly accepted the invalid opponent")
	}
	if len(parseErr.Error()) <= binaryMaxStringSize {
		t.Fatalf("expanded parse error length = %d, want over %d", len(parseErr.Error()), binaryMaxStringSize)
	}

	payload := encodeBinaryErrorResponse(binaryCreateRequestType, parseErr).bytes()
	if payload[0] != binaryCreateRequestType {
		t.Fatalf("failed request type = %#x, want %#x", payload[0], binaryCreateRequestType)
	}
	decoder := newBinaryDecoder(payload[5:]) // Skip request type and i32 env ID.
	kind, err := decoder.string("error kind")
	if err != nil {
		t.Fatalf("decode error kind: %v", err)
	}
	message, err := decoder.string("error message")
	if err != nil {
		t.Fatalf("decode error message: %v", err)
	}
	if err := decoder.finish(); err != nil {
		t.Fatalf("finish error response: %v", err)
	}
	if kind != protocolErrorValidation {
		t.Errorf("error kind = %q, want %q", kind, protocolErrorValidation)
	}
	if len(message) > binaryMaxStringSize {
		t.Errorf("error message length = %d, maximum is %d", len(message), binaryMaxStringSize)
	}
	if !utf8.ValidString(message) {
		t.Error("truncated error message is not valid UTF-8")
	}
	if !strings.HasSuffix(message, binaryErrorSuffix) {
		t.Errorf("error message suffix = %q, want %q", message[len(message)-len(binaryErrorSuffix):], binaryErrorSuffix)
	}
}

func TestEncodeBinaryErrorResponseRepairsInvalidUTF8(t *testing.T) {
	payload := encodeBinaryErrorResponse(binaryStepRequestType, errors.New(string([]byte{0xff}))).bytes()
	decoder := newBinaryDecoder(payload[5:])
	if _, err := decoder.string("error kind"); err != nil {
		t.Fatalf("decode error kind: %v", err)
	}
	message, err := decoder.string("error message")
	if err != nil {
		t.Fatalf("decode error message: %v", err)
	}
	if message != "\uFFFD" {
		t.Errorf("repaired message = %q, want replacement rune", message)
	}
}

func FuzzDecodeBinaryStepRequest(f *testing.F) {
	valid := make([]byte, 4+4+BodySlots)
	binary.LittleEndian.PutUint32(valid[:4], 1)
	for index := 8; index < len(valid); index++ {
		valid[index] = 0xff
	}
	f.Add(valid)
	f.Add([]byte{})
	f.Add([]byte{1, 0, 0, 0})

	f.Fuzz(func(_ *testing.T, payload []byte) {
		// A wire decoder must reject arbitrary process input with an error rather
		// than panic, allocate past the frame cap, or access beyond the payload.
		_, _ = decodeBinaryStepRequest(payload)
	})
}

func FuzzDecodeBinaryCreateRequest(f *testing.F) {
	valid := newBinaryTestEncoder(64)
	valid.u32(1)
	valid.i64(0)
	valid.i64(10)
	valid.i64(0)
	valid.string("idle-v1")
	valid.string("train-v2")
	f.Add(append([]byte(nil), valid.bytes()...))
	f.Add([]byte{})
	f.Add([]byte{1, 0, 0, 0})

	f.Fuzz(func(_ *testing.T, payload []byte) {
		_, _ = decodeBinaryCreateRequest(payload)
	})
}

func FuzzDecodeBinaryResetRequest(f *testing.F) {
	valid := newBinaryTestEncoder(32)
	valid.u32(1)
	valid.u32(0)
	valid.u32(1)
	valid.u32(0)
	valid.i64(4711)
	valid.u8(1)
	f.Add(append([]byte(nil), valid.bytes()...))
	f.Add([]byte{})
	f.Add([]byte{1, 0, 0, 0})

	f.Fuzz(func(_ *testing.T, payload []byte) {
		_, _ = decodeBinaryResetRequest(payload)
	})
}

func binaryTestObservation(marker int) Observation {
	var observation Observation
	observation.Bodies[0][0] = float32(marker) + 0.25
	observation.Players[0][0] = float32(marker) + 0.5
	observation.BodyMask[0] = true
	observation.OwnedMask[1] = true
	observation.PlayerMask[0] = true
	observation.TargetMask[1][2] = true
	observation.SlotBodyID[0] = int32(marker * 100)
	return observation
}

func assertBinaryObservationOffsets(
	t *testing.T,
	payload []byte,
	start int,
	count int,
	want Observation,
) {
	t.Helper()

	// Identify the row by its marker. This lets the same helper verify that the
	// second environment starts one whole row into each separate column.
	marker := int(want.SlotBodyID[0] / 100)
	row := marker - 1
	bodyStart := start + row*BodySlots*BodyFeatures*4
	if got := math.Float32frombits(binary.LittleEndian.Uint32(payload[bodyStart : bodyStart+4])); got != want.Bodies[0][0] {
		t.Errorf("body marker = %v, want %v", got, want.Bodies[0][0])
	}
	playersStart := start + count*BodySlots*BodyFeatures*4
	playerRowStart := playersStart + row*PlayerSlots*PlayerFeatures*4
	if got := math.Float32frombits(binary.LittleEndian.Uint32(payload[playerRowStart : playerRowStart+4])); got != want.Players[0][0] {
		t.Errorf("player marker = %v, want %v", got, want.Players[0][0])
	}
	bodyMaskStart := playersStart + count*PlayerSlots*PlayerFeatures*4
	if got := payload[bodyMaskStart+row*BodySlots]; got != 1 {
		t.Errorf("body mask marker = %d, want 1", got)
	}
	ownedMaskStart := bodyMaskStart + count*BodySlots
	if got := payload[ownedMaskStart+row*BodySlots+1]; got != 1 {
		t.Errorf("owned mask marker = %d, want 1", got)
	}
	playerMaskStart := ownedMaskStart + count*BodySlots
	if got := payload[playerMaskStart+row*PlayerSlots]; got != 1 {
		t.Errorf("player mask marker = %d, want 1", got)
	}
	targetMaskStart := playerMaskStart + count*PlayerSlots
	if got := payload[targetMaskStart+row*BodySlots*BodySlots+1*BodySlots+2]; got != 1 {
		t.Errorf("target mask marker = %d, want 1", got)
	}
	slotIDStart := targetMaskStart + count*BodySlots*BodySlots
	if got := int32(binary.LittleEndian.Uint32(payload[slotIDStart+row*BodySlots*4:])); got != want.SlotBodyID[0] {
		t.Errorf("slot body ID marker = %d, want %d", got, want.SlotBodyID[0])
	}
}

func assertBinaryProtocolFailureKind(t *testing.T, err error, want string) {
	t.Helper()
	var failure *protocolFailure
	if !errors.As(err, &failure) {
		t.Fatalf("error type = %T, want *protocolFailure", err)
	}
	if failure.kind != want {
		t.Fatalf("error kind = %q, want %q", failure.kind, want)
	}
}

type countingBinaryWriter struct {
	bytes.Buffer
	calls int
}

func (writer *countingBinaryWriter) Write(data []byte) (int, error) {
	writer.calls++
	return writer.Buffer.Write(data)
}

// binaryTestEncoder builds request payload fixtures without sharing the
// production response encoder. Its deliberately small API prevents tests from
// creating the headerless production-encoder mode that previously made
// encodedPayload panic, and it gives decoder tests an independent byte writer.
type binaryTestEncoder struct {
	data []byte
}

func newBinaryTestEncoder(capacity int) *binaryTestEncoder {
	return &binaryTestEncoder{data: make([]byte, 0, capacity)}
}

func (encoder *binaryTestEncoder) u8(value byte) {
	encoder.data = append(encoder.data, value)
}

func (encoder *binaryTestEncoder) u32(value uint32) {
	encoder.data = binary.LittleEndian.AppendUint32(encoder.data, value)
}

func (encoder *binaryTestEncoder) i64(value int64) {
	encoder.data = binary.LittleEndian.AppendUint64(encoder.data, uint64(value))
}

func (encoder *binaryTestEncoder) string(value string) {
	encoder.u32(uint32(len(value)))
	encoder.data = append(encoder.data, value...)
}

func (encoder *binaryTestEncoder) bytes() []byte {
	return encoder.data
}

func readBinaryTestString(t *testing.T, payload []byte, offset *int) string {
	t.Helper()
	if len(payload)-*offset < 4 {
		t.Fatalf("test decoder needs string length at offset %d", *offset)
	}
	length := int(binary.LittleEndian.Uint32(payload[*offset : *offset+4]))
	*offset += 4
	if len(payload)-*offset < length {
		t.Fatalf("test decoder needs %d string bytes at offset %d", length, *offset)
	}
	value := string(payload[*offset : *offset+length])
	*offset += length
	return value
}
