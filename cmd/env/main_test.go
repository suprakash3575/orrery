package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"runtime/debug"
	"slices"
	"strings"
	"testing"
	"time"

	"orrery/mlenv"
	"orrery/scenario"
)

// env-v3 message types used by these process tests. Requests use the low
// range; each response sets the high bit of its request type, and 0xff is an
// error response.
const (
	createRequestType  byte = 0x01
	resetRequestType   byte = 0x02
	stepRequestType    byte = 0x03
	closeRequestType   byte = 0x04
	createResponseType byte = 0x81
	resetResponseType  byte = 0x82
	stepResponseType   byte = 0x83
	closeResponseType  byte = 0x84
	errorResponseType  byte = 0xff
)

// TestEnvHelperProcess turns the Go test binary into the real stdin/stdout
// process used by the integration tests. A separate process is essential here:
// signal handling and blocking reads on a real stdin pipe only exist at the
// process boundary. The words after "--" on the helper's command line are
// cmd/env's own arguments, handed to run exactly as main would, so startup
// parsing is part of every process test.
func TestEnvHelperProcess(t *testing.T) {
	if os.Getenv("ORRERY_ENV_HELPER") != "1" {
		return
	}
	var arguments []string
	if separator := slices.Index(os.Args, "--"); separator >= 0 {
		arguments = os.Args[separator+1:]
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	status := run(ctx, arguments, os.Stdin, os.Stdout, os.Stderr)
	stop()
	os.Exit(status)
}

func TestParseArguments(t *testing.T) {
	tests := []struct {
		name      string
		arguments []string
		want      int
		wantError bool
	}{
		{name: "default", want: 2},
		{name: "explicit two", arguments: []string{"--players", "2"}, want: 2},
		{name: "four", arguments: []string{"--players", "4"}, want: 4},
		{name: "four with equals", arguments: []string{"--players=4"}, want: 4},
		{name: "malformed value", arguments: []string{"--players", "abc"}, wantError: true},
		{name: "missing value", arguments: []string{"--players"}, wantError: true},
		{name: "unknown flag", arguments: []string{"--format", "4"}, wantError: true},
		{name: "removed protocol flag", arguments: []string{"--protocol", "env-v3"}, wantError: true},
		{name: "positional argument", arguments: []string{"--players", "4", "extra"}, wantError: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := parseArguments(test.arguments)
			if test.wantError {
				if err == nil {
					t.Fatalf("parseArguments(%v) succeeded with %d", test.arguments, got)
				}
				return
			}
			if err != nil || got != test.want {
				t.Fatalf("parseArguments(%v) = %d, %v; want %d", test.arguments, got, err, test.want)
			}
		})
	}
}

// TestRunServesChosenMap plays one turn of a real generated map through run,
// in-process, for each supported player count. Each seed is pinned to the
// seat its first train-v2 draw gives, so the four-player case reaches seat 2,
// one of the seats only a four-player map has.
func TestRunServesChosenMap(t *testing.T) {
	tests := []struct {
		players   int
		seed      int64
		generator string
		seat      byte
		// playerMask is indexed by role: me, clockwise, opposite,
		// counter-clockwise. A two-player opponent always sits opposite.
		playerMask [mlenv.PlayerSlots]bool
	}{
		{players: 2, seed: 2, generator: scenario.GeneratorV2ID, seat: 1, playerMask: [mlenv.PlayerSlots]bool{true, false, true, false}},
		{players: 4, seed: 1, generator: scenario.GeneratorV2FourPlayerID, seat: 2, playerMask: [mlenv.PlayerSlots]bool{true, true, true, true}},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%d players", test.players), func(t *testing.T) {
			input := appendTestProtocolFrame(nil, createRequestType, idleCreatePayload(test.seed, test.seed+1))
			input = appendTestProtocolFrame(input, resetRequestType, resetPayload(0))
			input = appendTestProtocolFrame(input, stepRequestType, noLaunchStepPayload(0))
			input = appendTestProtocolFrame(input, closeRequestType, nil)

			var stdout, stderr bytes.Buffer
			arguments := []string{"--players", fmt.Sprint(test.players)}
			if status := run(context.Background(), arguments, bytes.NewReader(input), &stdout, &stderr); status != exitOK {
				t.Fatalf("run(%v) status = %d, want %d; stderr=%q", arguments, status, exitOK, stderr.String())
			}
			frames := splitTestProtocolFrames(t, stdout.Bytes())
			wantTypes := []byte{createResponseType, resetResponseType, stepResponseType, closeResponseType}
			if len(frames) != len(wantTypes) {
				t.Fatalf("got %d response frames, want %d", len(frames), len(wantTypes))
			}
			for index, frame := range frames {
				if frame.messageType != wantTypes[index] {
					t.Fatalf("response %d type = %#x, want %#x; payload=%q", index, frame.messageType, wantTypes[index], frame.payload)
				}
			}

			// The create response opens with four strings: engine, schema,
			// generator, opponent.
			offset := 0
			readTestProtocolString(t, frames[0].payload, &offset)
			readTestProtocolString(t, frames[0].payload, &offset)
			if got := readTestProtocolString(t, frames[0].payload, &offset); got != test.generator {
				t.Fatalf("generator = %q, want %q", got, test.generator)
			}

			reset := frames[1].payload
			if got := reset[resetSeatOffset]; got != test.seat {
				t.Fatalf("learner seat = %d, want %d from the pinned train-v2 seed", got, test.seat)
			}
			var mask [mlenv.PlayerSlots]bool
			for role := range mask {
				mask[role] = reset[resetPlayerMaskOffset+role] == 1
			}
			if mask != test.playerMask {
				t.Fatalf("PlayerMask = %v, want %v", mask, test.playerMask)
			}

			if terminated, _, turn := stepOutcome(t, frames[2].payload); terminated || turn != 1 {
				t.Fatalf("step = terminated %t, turn %d; want an ongoing game at turn 1", terminated, turn)
			}
			if closed := frames[3].payload; len(closed) != 1 || closed[0] != 1 {
				t.Fatalf("close payload = %v, want [1]", closed)
			}
		})
	}
}

// TestStartupRejectsUnsupportedPlayerCount runs the real helper process with
// counts that parse as numbers but have no map generator. Stdin stays open
// and empty, so a process that reached the server would block waiting for a
// frame; exiting at all shows the rejection came before any input was read.
func TestStartupRejectsUnsupportedPlayerCount(t *testing.T) {
	for _, players := range []string{"0", "3"} {
		t.Run(players+" players", func(t *testing.T) {
			command := helperCommand("--players", players)
			stdin, err := command.StdinPipe()
			if err != nil {
				t.Fatalf("StdinPipe() error = %v", err)
			}
			// Close stdin only after the process is gone, so it is never the
			// reason the process stopped.
			defer stdin.Close()
			var stdout, stderr bytes.Buffer
			command.Stdout = &stdout
			command.Stderr = &stderr
			if err := command.Start(); err != nil {
				t.Fatalf("start env helper process: %v", err)
			}
			done := make(chan error, 1)
			go func() { done <- command.Wait() }()

			var waitErr error
			select {
			case waitErr = <-done:
			case <-time.After(5 * time.Second):
				_ = command.Process.Kill()
				<-done
				t.Fatal("env process kept running with an unsupported player count")
			}
			var exitErr *exec.ExitError
			if !errors.As(waitErr, &exitErr) || exitErr.ExitCode() != exitUsage {
				t.Fatalf("exit = %v, want status %d; stderr=%q", waitErr, exitUsage, stderr.String())
			}
			if message := stderr.String(); !strings.Contains(message, "player count "+players) || !strings.Contains(message, "2 or 4") {
				t.Fatalf("stderr = %q, want the bad count and the allowed values 2 and 4", message)
			}
			if stdout.Len() != 0 {
				t.Fatalf("stdout = %q, want nothing before the protocol starts", stdout.Bytes())
			}
		})
	}
}

func TestProtocolProcessInterruptUnblocksIdleInput(t *testing.T) {
	session := startSession(t)
	session.send(createRequestType, idleCreatePayload(0, 1000), createResponseType)
	// Deliberately leave the parent's stdin pipe open. The child is now idle
	// waiting for the next frame header, which is the exact state that used to
	// ignore SIGINT.
	if err := session.command.Process.Signal(os.Interrupt); err != nil {
		t.Fatalf("send interrupt: %v", err)
	}
	done := make(chan error, 1)
	go func() { done <- session.command.Wait() }()
	select {
	case err := <-done:
		_ = session.stdin.Close()
		if err != nil {
			t.Fatalf("interrupted env helper failed: %v; stderr=%q", err, session.stderr.String())
		}
		if session.stderr.Len() != 0 {
			t.Fatalf("interrupted env helper wrote stderr: %q", session.stderr.String())
		}
	case <-time.After(5 * time.Second):
		_ = session.command.Process.Kill()
		_ = session.stdin.Close()
		<-done
		t.Fatal("env process did not exit promptly after interrupt while stdin remained open")
	}
}

// TestProtocolRoundTripMatchesReferenceAndIsByteDeterministic plays one idle
// game to the end through the real process, checks that it ends on the same
// turn as the same scenario played directly, then replays the identical
// request bytes in a fresh process and requires identical response bytes.
// Field-by-field agreement with mlenv.Batch is covered inside mlenv; this test
// covers what only a separate process can: stdin/stdout framing end to end.
func TestProtocolRoundTripMatchesReferenceAndIsByteDeterministic(t *testing.T) {
	first := startSession(t)
	first.send(createRequestType, idleCreatePayload(0, 1000), createResponseType)
	reset := first.send(resetRequestType, resetPayload(0), resetResponseType)
	// A reset payload starts with the next cursor (i64) and the environment
	// count (u32); the first identity follows as env ID (u32) then seed (i64).
	seed := int64(binary.LittleEndian.Uint64(reset[16:24]))

	turn := 0
	for {
		step := first.send(stepRequestType, noLaunchStepPayload(0), stepResponseType)
		terminated, endReason, stepTurn := stepOutcome(t, step)
		turn = stepTurn
		if terminated {
			// Nobody launches, so nobody can be eliminated: only the engine
			// can end this game.
			if endReason != byte(mlenv.EndGameFinished) {
				t.Fatalf("idle game ended with reason %d, want game finished", endReason)
			}
			break
		}
	}
	first.send(closeRequestType, nil, closeResponseType)
	firstOutput, requests := first.finish()

	// cmd/env serves two-player generator-v2 maps.
	reference, err := (scenario.GeneratorV2{}).New(seed)
	if err != nil {
		t.Fatalf("reference scenario error = %v", err)
	}
	for !reference.Finished() {
		if err := reference.AdvanceTurn(nil); err != nil {
			t.Fatalf("reference AdvanceTurn() error = %v", err)
		}
	}
	if turn != reference.Turn {
		t.Fatalf("protocol game ended on turn %d, direct engine on turn %d", turn, reference.Turn)
	}

	secondOutput := runScript(t, requests)
	if !bytes.Equal(firstOutput, secondOutput) {
		t.Fatalf("identical complete sessions produced different stdout\nfirst bytes: %d\nsecond bytes: %d", len(firstOutput), len(secondOutput))
	}
}

func TestEngineVersionIsNeverEmpty(t *testing.T) {
	if got := engineVersion(); strings.TrimSpace(got) == "" {
		t.Fatal("engineVersion() returned an empty identity")
	}
}

// TestChooseEngineVersionPrefersTheMostExactIdentity covers each source in
// priority order. Archived values win once git archive has filled them; an
// unfilled placeholder falls through to what the Go toolchain embedded; and
// "development" is reserved for having nothing at all.
func TestChooseEngineVersionPrefersTheMostExactIdentity(t *testing.T) {
	const commit = "0123456789abcdef0123456789abcdef01234567"
	// The placeholders exactly as an ordinary checkout compiles them.
	describePlaceholder := archivePlaceholder + "%(describe:tags)$"
	commitPlaceholder := archivePlaceholder + "%H$"
	buildInfo := func(version string, settings ...debug.BuildSetting) *debug.BuildInfo {
		return &debug.BuildInfo{Main: debug.Module{Version: version}, Settings: settings}
	}
	vcs := func(revision, modified string) []debug.BuildSetting {
		return []debug.BuildSetting{
			{Key: "vcs.revision", Value: revision},
			{Key: "vcs.modified", Value: modified},
		}
	}

	tests := []struct {
		name     string
		describe string
		commit   string
		info     *debug.BuildInfo
		want     string
	}{
		{
			name: "archived at a tag", describe: "v0.1.0", commit: commit,
			info: buildInfo("(devel)"), want: "v0.1.0+" + commit,
		},
		{
			name: "archived after a tag", describe: "v0.1.0-3-gabc1234", commit: commit,
			info: buildInfo("(devel)"), want: "v0.1.0-3-gabc1234+" + commit,
		},
		{
			name: "archived without any tag", describe: "", commit: commit,
			info: nil, want: "untagged+" + commit,
		},
		{
			name: "archived values beat embedded VCS", describe: "v0.1.0", commit: commit,
			info: buildInfo("(devel)", vcs("fedcba", "true")...), want: "v0.1.0+" + commit,
		},
		{
			name: "checkout with clean embedded commit", describe: describePlaceholder,
			commit: commitPlaceholder, info: buildInfo("(devel)", vcs("fedcba", "false")...),
			want: "fedcba",
		},
		{
			name: "checkout with uncommitted changes", describe: describePlaceholder,
			commit: commitPlaceholder, info: buildInfo("(devel)", vcs("fedcba", "true")...),
			want: "fedcba+dirty",
		},
		{
			name: "module version without VCS", describe: describePlaceholder,
			commit: commitPlaceholder, info: buildInfo("v1.2.3"), want: "v1.2.3",
		},
		{
			name: "go run from a checkout", describe: describePlaceholder,
			commit: commitPlaceholder, info: buildInfo("(devel)"), want: "development",
		},
		{
			name: "no build information", describe: describePlaceholder,
			commit: commitPlaceholder, info: nil, want: "development",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := chooseEngineVersion(test.describe, test.commit, test.info); got != test.want {
				t.Fatalf("chooseEngineVersion() = %q, want %q", got, test.want)
			}
		})
	}
}

// stepOutcome reads the terminated flag, end reason, and turn number of the
// only environment in a one-environment step payload. The payload is
// column-major: count (u32), identity (u32 ID, i64 seed, u8 seat, u8
// explicit), reward (f32), terminated (u8), truncated (u8), end reason (u8),
// turn (u32), then the observation.
func stepOutcome(t *testing.T, payload []byte) (terminated bool, endReason byte, turn int) {
	t.Helper()
	const terminatedOffset = 4 + 14 + 4
	const endReasonOffset = terminatedOffset + 2
	const turnOffset = endReasonOffset + 1
	if len(payload) < turnOffset+4 || binary.LittleEndian.Uint32(payload[:4]) != 1 {
		t.Fatalf("step payload has %d bytes, want one environment", len(payload))
	}
	return payload[terminatedOffset] == 1, payload[endReasonOffset], int(binary.LittleEndian.Uint32(payload[turnOffset:]))
}

func idleCreatePayload(seedStart, seedEnd int64) []byte {
	payload := make([]byte, 0, 64)
	payload = binary.LittleEndian.AppendUint32(payload, 1)
	payload = binary.LittleEndian.AppendUint64(payload, uint64(seedStart))
	payload = binary.LittleEndian.AppendUint64(payload, uint64(seedEnd))
	payload = binary.LittleEndian.AppendUint64(payload, 0)
	payload = appendTestProtocolString(payload, "idle-v1")
	return appendTestProtocolString(payload, "train-v2")
}

// resetPayload names one environment and no explicit seed overrides.
func resetPayload(id uint32) []byte {
	payload := binary.LittleEndian.AppendUint32(nil, 1)
	payload = binary.LittleEndian.AppendUint32(payload, id)
	return binary.LittleEndian.AppendUint32(payload, 0)
}

// noLaunchStepPayload sends one environment whose every body slot holds the
// target byte 0xff, which is -1 read as signed: that source stays idle.
func noLaunchStepPayload(id uint32) []byte {
	payload := binary.LittleEndian.AppendUint32(nil, 1)
	payload = binary.LittleEndian.AppendUint32(payload, id)
	return append(payload, bytes.Repeat([]byte{0xff}, mlenv.BodySlots)...)
}

// Offsets into a one-environment reset payload: next cursor (i64), count
// (u32), then the identity (u32 ID, i64 seed, u8 seat, u8 explicit) and the
// observation. The observation's columns run bodies (f32), players (f32),
// body mask, owned mask, then the player mask, one byte per entry.
const (
	resetSeatOffset        = 8 + 4 + 4 + 8
	resetObservationOffset = resetSeatOffset + 2
	resetPlayerMaskOffset  = resetObservationOffset +
		4*mlenv.BodySlots*mlenv.BodyFeatures +
		4*mlenv.PlayerSlots*mlenv.PlayerFeatures +
		2*mlenv.BodySlots
)

type testProtocolFrame struct {
	messageType byte
	payload     []byte
}

// splitTestProtocolFrames cuts a complete response stream into its frames,
// failing on a bad magic or a frame that runs past the end of the stream.
func splitTestProtocolFrames(t *testing.T, wire []byte) []testProtocolFrame {
	t.Helper()
	var frames []testProtocolFrame
	for len(wire) > 0 {
		if len(wire) < 12 || string(wire[:4]) != "OWE3" {
			t.Fatalf("response frame header = %v, want an env-v3 header", wire[:min(len(wire), 12)])
		}
		end := 12 + int(binary.LittleEndian.Uint32(wire[8:12]))
		if end > len(wire) {
			t.Fatalf("response frame needs %d bytes, stream has %d", end, len(wire))
		}
		frames = append(frames, testProtocolFrame{messageType: wire[4], payload: wire[12:end]})
		wire = wire[end:]
	}
	return frames
}

// readTestProtocolString reads one u32-length-prefixed string at *offset and
// moves *offset past it.
func readTestProtocolString(t *testing.T, payload []byte, offset *int) string {
	t.Helper()
	if *offset+4 > len(payload) {
		t.Fatalf("string length at offset %d is past the payload's %d bytes", *offset, len(payload))
	}
	length := int(binary.LittleEndian.Uint32(payload[*offset:]))
	start := *offset + 4
	if start+length > len(payload) {
		t.Fatalf("string of %d bytes at offset %d is past the payload's %d bytes", length, start, len(payload))
	}
	*offset = start + length
	return string(payload[start:*offset])
}

func appendTestProtocolString(destination []byte, value string) []byte {
	destination = binary.LittleEndian.AppendUint32(destination, uint32(len(value)))
	return append(destination, value...)
}

func appendTestProtocolFrame(destination []byte, messageType byte, payload []byte) []byte {
	headerStart := len(destination)
	destination = append(destination, make([]byte, 12)...)
	copy(destination[headerStart:headerStart+4], "OWE3")
	destination[headerStart+4] = messageType
	binary.LittleEndian.PutUint32(destination[headerStart+8:headerStart+12], uint32(len(payload)))
	return append(destination, payload...)
}

type processSession struct {
	t        *testing.T
	command  *exec.Cmd
	stdin    io.WriteCloser
	stdout   io.Reader
	stderr   bytes.Buffer
	output   bytes.Buffer
	requests []byte
}

// helperCommand re-runs this test binary as cmd/env. Everything after "--"
// reaches run as cmd/env's own arguments; the test framework stops reading
// flags at "--", so it never sees them.
func helperCommand(arguments ...string) *exec.Cmd {
	command := exec.Command(os.Args[0], append([]string{"-test.run=^TestEnvHelperProcess$", "--"}, arguments...)...)
	command.Env = append(os.Environ(), "ORRERY_ENV_HELPER=1")
	return command
}

func startSession(t *testing.T) *processSession {
	t.Helper()
	command := helperCommand()
	stdin, err := command.StdinPipe()
	if err != nil {
		t.Fatalf("StdinPipe() error = %v", err)
	}
	stdout, err := command.StdoutPipe()
	if err != nil {
		t.Fatalf("StdoutPipe() error = %v", err)
	}
	session := &processSession{t: t, command: command, stdin: stdin, stdout: stdout}
	command.Stderr = &session.stderr
	if err := command.Start(); err != nil {
		t.Fatalf("start env helper process: %v", err)
	}
	return session
}

// send writes one request frame, reads exactly one response frame, and fails
// the test unless that response has the wanted type. It returns the response
// payload and records both directions for the byte-determinism replay.
func (session *processSession) send(requestType byte, payload []byte, wantType byte) []byte {
	session.t.Helper()
	request := appendTestProtocolFrame(nil, requestType, payload)
	session.requests = append(session.requests, request...)
	if _, err := session.stdin.Write(request); err != nil {
		session.t.Fatalf("write request %#x: %v", requestType, err)
	}
	var header [12]byte
	if _, err := io.ReadFull(session.stdout, header[:]); err != nil {
		session.t.Fatalf("read response header after %#x: %v; stderr=%q", requestType, err, session.stderr.String())
	}
	if string(header[:4]) != "OWE3" {
		session.t.Fatalf("response magic = %q, want OWE3", header[:4])
	}
	response := make([]byte, binary.LittleEndian.Uint32(header[8:12]))
	if _, err := io.ReadFull(session.stdout, response); err != nil {
		session.t.Fatalf("read response payload after %#x: %v", requestType, err)
	}
	if header[4] == errorResponseType {
		session.t.Fatalf("request %#x returned an error response: %q", requestType, response)
	}
	if header[4] != wantType {
		session.t.Fatalf("response type = %#x, want %#x", header[4], wantType)
	}
	session.output.Write(header[:])
	session.output.Write(response)
	return response
}

func (session *processSession) finish() ([]byte, []byte) {
	session.t.Helper()
	_ = session.stdin.Close()
	if err := session.command.Wait(); err != nil {
		session.t.Fatalf("env helper process failed: %v; stderr=%q", err, session.stderr.String())
	}
	if session.stderr.Len() != 0 {
		session.t.Fatalf("protocol session wrote diagnostics to stderr: %q", session.stderr.String())
	}
	return append([]byte(nil), session.output.Bytes()...), append([]byte(nil), session.requests...)
}

// runScript replays recorded request bytes into a fresh process in one write
// and returns everything it printed.
func runScript(t *testing.T, requests []byte) []byte {
	t.Helper()
	command := helperCommand()
	command.Stdin = bytes.NewReader(requests)
	var stdout, stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	if err := command.Run(); err != nil {
		t.Fatalf("scripted env process failed: %v; stderr=%q", err, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("scripted env process wrote diagnostics to stderr: %q", stderr.String())
	}
	return stdout.Bytes()
}
