package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"orrery/agent"
	"orrery/arena"
	"orrery/scenario"
)

func TestParseConfigDefaultsAndOverrides(t *testing.T) {
	parsed, err := parseConfig(nil, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	wantDefaults := config{
		scenarios: 10,
		workers:   runtime.GOMAXPROCS(0),
		seed:      1,
		scenario:  arena.FixedScenarioV1,
		p0:        defaultP0,
		p1:        defaultP1,
	}
	if !reflect.DeepEqual(parsed, wantDefaults) {
		t.Fatalf("defaults = %#v, want %#v", parsed, wantDefaults)
	}

	parsed, err = parseConfig([]string{
		"-scenarios", "7",
		"-workers", "3",
		"-seed", "-9",
		"-scenario", scenario.GeneratorV1ID,
		"-p0", "idle-v1",
		"-p1", "idle-v1",
		"-record", "matches.ndjson",
	}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	wantOverrides := config{
		scenarios: 7,
		workers:   3,
		seed:      -9,
		scenario:  scenario.GeneratorV1ID,
		p0:        "idle-v1",
		p1:        "idle-v1",
		record:    "matches.ndjson",
	}
	if !reflect.DeepEqual(parsed, wantOverrides) {
		t.Fatalf("overrides = %#v, want %#v", parsed, wantOverrides)
	}
}

func TestParseConfigErrorsAndHelp(t *testing.T) {
	tests := []struct {
		name      string
		arguments []string
		want      error
	}{
		{"help", []string{"-h"}, flag.ErrHelp},
		{"unknown flag", []string{"-unknown"}, nil},
		{"invalid integer", []string{"-workers", "many"}, nil},
		{"positional argument", []string{"extra"}, nil},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var diagnostics bytes.Buffer
			_, err := parseConfig(test.arguments, &diagnostics)
			if err == nil {
				t.Fatal("parseConfig accepted invalid arguments")
			}
			if test.want != nil && !errors.Is(err, test.want) {
				t.Fatalf("error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestRunIsDeterministicAcrossWorkerCounts(t *testing.T) {
	base := config{
		scenarios: 2,
		seed:      321,
		p0:        defaultP0,
		p1:        defaultP1,
	}
	serialConfig := base
	serialConfig.workers = 1
	parallelConfig := base
	parallelConfig.workers = 4

	var serial, parallel bytes.Buffer
	if err := run(&serial, serialConfig); err != nil {
		t.Fatal(err)
	}
	if err := run(&parallel, parallelConfig); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(serial.Bytes(), parallel.Bytes()) {
		t.Fatalf("worker counts produced different summaries:\nserial:\n%s\nparallel:\n%s", serial.Bytes(), parallel.Bytes())
	}
	for _, want := range []string{
		"scenarios: 2",
		"games: 4 (completed: 4)",
		"terminations:",
		"failures: agent=0 engine=0 other=0 cleanup=0",
		"contestant p0 (heuristic-v1:",
		"contestant p1 (random-v1:",
		"  seat 0:",
		"  seat 1:",
	} {
		if !strings.Contains(serial.String(), want) {
			t.Fatalf("summary missing %q:\n%s", want, serial.String())
		}
	}
}

func TestRunWritesDeterministicReplayableRecords(t *testing.T) {
	temporaryDirectory := t.TempDir()
	serialPath := filepath.Join(temporaryDirectory, "serial.ndjson")
	parallelPath := filepath.Join(temporaryDirectory, "parallel.ndjson")
	base := config{
		scenarios: 1,
		seed:      777,
		p0:        defaultP0,
		p1:        defaultP1,
	}
	serialConfig := base
	serialConfig.workers = 1
	serialConfig.record = serialPath
	parallelConfig := base
	parallelConfig.workers = 3
	parallelConfig.record = parallelPath

	var serialSummary, parallelSummary bytes.Buffer
	if err := run(&serialSummary, serialConfig); err != nil {
		t.Fatal(err)
	}
	if err := run(&parallelSummary, parallelConfig); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(serialSummary.Bytes(), parallelSummary.Bytes()) {
		t.Fatal("recording runs produced different summaries")
	}
	serialBytes, err := os.ReadFile(serialPath)
	if err != nil {
		t.Fatal(err)
	}
	parallelBytes, err := os.ReadFile(parallelPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(serialBytes, parallelBytes) {
		t.Fatal("workers=1 and workers=3 produced different NDJSON")
	}

	decoder := json.NewDecoder(bytes.NewReader(serialBytes))
	for resultIndex := range 2 {
		var result arena.MatchResult
		if err := decoder.Decode(&result); err != nil {
			t.Fatalf("decode result %d: %v", resultIndex, err)
		}
		if result.ScenarioIndex != 0 || result.Leg != resultIndex || result.Actions == nil {
			t.Fatalf("result %d metadata = %#v", resultIndex, result)
		}
		replayed, err := arena.Replay(context.Background(), arena.FixedScenarioFactory{}, *result.Actions)
		if err != nil {
			t.Fatalf("replay result %d: %v", resultIndex, err)
		}
		if got := agent.NewView(replayed); !reflect.DeepEqual(got, result.FinalState) {
			t.Fatalf("replayed result %d differs from recorded final state", resultIndex)
		}
	}
	var extra arena.MatchResult
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		t.Fatalf("decode after two results = %v, want io.EOF", err)
	}
}

func TestRunRejectsInvalidConfigurationAndCancellation(t *testing.T) {
	valid := config{scenarios: 1, workers: 1, seed: 1, p0: "idle-v1", p1: "idle-v1"}
	tests := []struct {
		name   string
		mutate func(*config)
		want   string
	}{
		{"invalid p0", func(parsed *config) { parsed.p0 = "unknown-v1" }, "parse -p0"},
		{"invalid p1", func(parsed *config) { parsed.p1 = "unknown-v1" }, "parse -p1"},
		{"zero scenarios", func(parsed *config) { parsed.scenarios = 0 }, "schedule experiment"},
		{"zero workers", func(parsed *config) { parsed.workers = 0 }, "worker count"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			parsed := valid
			test.mutate(&parsed)
			var output bytes.Buffer
			err := run(&output, parsed)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want containing %q", err, test.want)
			}
			if output.Len() != 0 {
				t.Fatalf("output on failure = %q", output.String())
			}
		})
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := runContext(ctx, io.Discard, valid); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled error = %v, want context.Canceled", err)
	}
}

func TestWriteRecordAndSummaryReportWriterFailures(t *testing.T) {
	if err := writeRecord(t.TempDir(), nil); err == nil || !strings.Contains(err.Error(), "create record") {
		t.Fatalf("writeRecord directory error = %v", err)
	}

	writerErr := errors.New("writer failed")
	err := writeSummary(errorWriter{err: writerErr}, arena.ExperimentSummary{
		Scenarios: 1,
		Games:     2,
	})
	if !errors.Is(err, writerErr) {
		t.Fatalf("writeSummary error = %v, want wrapped writer error", err)
	}
}

func TestWriteRecordCreatesMissingParentDirectories(t *testing.T) {
	path := filepath.Join(t.TempDir(), "runs", "nested", "matches.ndjson")
	if err := writeRecord(path, nil); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.IsDir() {
		t.Fatalf("record path %q is a directory, want a file", path)
	}
}

type errorWriter struct{ err error }

func (writer errorWriter) Write([]byte) (int, error) { return 0, writer.err }

// TestScenarioFlagSelectsGenerator covers the flag end to end: an unknown name
// is rejected during parsing rather than after workers start, both known names
// resolve to their versioned factory, and a full generated-scenario run stays
// deterministic across worker counts the way the fixed layout does.
func TestScenarioFlagSelectsGenerator(t *testing.T) {
	if _, err := parseConfig([]string{"-scenario", "nonesuch"}, io.Discard); err == nil {
		t.Fatal("expected an error for an unknown scenario name")
	}

	resolutions := []struct {
		name string
		want string
	}{
		{name: "", want: arena.FixedScenarioV1},
		{name: arena.FixedScenarioV1, want: arena.FixedScenarioV1},
		{name: scenario.GeneratorV1ID, want: scenario.GeneratorV1ID},
		{name: scenario.GeneratorV2ID, want: scenario.GeneratorV2ID},
	}
	for _, resolution := range resolutions {
		factory, err := scenarioFactory(resolution.name)
		if err != nil {
			t.Fatalf("scenario %q: %v", resolution.name, err)
		}
		if got := factory.ID(); got != resolution.want {
			t.Fatalf("scenario %q resolved to factory %q, want %q", resolution.name, got, resolution.want)
		}
	}

	base := config{
		scenarios: 2,
		seed:      4242,
		scenario:  scenario.GeneratorV1ID,
		p0:        defaultP0,
		p1:        defaultP1,
	}
	serialConfig := base
	serialConfig.workers = 1
	parallelConfig := base
	parallelConfig.workers = 4

	var serial, parallel bytes.Buffer
	if err := run(&serial, serialConfig); err != nil {
		t.Fatal(err)
	}
	if err := run(&parallel, parallelConfig); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(serial.Bytes(), parallel.Bytes()) {
		t.Fatalf("worker counts produced different summaries:\nserial:\n%s\nparallel:\n%s", serial.Bytes(), parallel.Bytes())
	}
	if !strings.Contains(serial.String(), "games: 4 (completed: 4)") {
		t.Fatalf("generated-scenario run did not complete all games:\n%s", serial.String())
	}
}

func TestScenarioFlagRunsGeneratorV2Deterministically(t *testing.T) {
	parsed, err := parseConfig([]string{
		"-scenario", scenario.GeneratorV2ID,
		"-scenarios", "1",
		"-seed", "42",
		"-workers", "1",
	}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	var serial, parallel bytes.Buffer
	if err := run(&serial, parsed); err != nil {
		t.Fatal(err)
	}
	parsed.workers = 2
	if err := run(&parallel, parsed); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(serial.Bytes(), parallel.Bytes()) {
		t.Fatalf("worker counts produced different V2 summaries:\nserial:\n%s\nparallel:\n%s",
			serial.Bytes(), parallel.Bytes())
	}
	if !strings.Contains(serial.String(), "games: 2 (completed: 2)") {
		t.Fatalf("V2 scenario did not complete both seats:\n%s", serial.String())
	}
}
