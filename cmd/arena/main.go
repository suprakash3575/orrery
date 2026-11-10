// Command arena runs deterministic paired Orrery matches between two
// configured agents and prints an aggregate summary.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"

	"orrery/agent"
	"orrery/arena"
	"orrery/scenario"
)

const (
	defaultP0 = "heuristic-v1:aggression=1.2,economy=1,efficiency=1,defense=0.1"
	defaultP1 = "random-v1:launch_probability=0.3,min_fraction=0.2,max_fraction=0.8"
)

type config struct {
	scenarios int
	workers   int
	seed      int64
	scenario  string
	p0        string
	p1        string
	record    string
}

// scenarioFactory resolves the -scenario flag to a factory. Each name is a
// versioned identity rather than a free-form label. A saved result is only
// replayable if it records which generator built its maps — the same seed
// means a different board under a different scenario.
func scenarioFactory(name string) (arena.ScenarioFactory, error) {
	switch name {
	// The empty name is the zero value of config.scenario, which a caller
	// constructing a config directly rather than through flag parsing will
	// leave unset. It resolves to the same default the flag declares, so a
	// bare config means the fixed layout instead of failing.
	case "", arena.FixedScenarioV1:
		return arena.FixedScenarioFactory{}, nil
	case scenario.GeneratorV1ID:
		return scenario.GeneratorV1{}, nil
	case scenario.GeneratorV2ID:
		return scenario.GeneratorV2{}, nil
	default:
		return nil, fmt.Errorf("unknown scenario %q, want %q, %q, or %q", name,
			arena.FixedScenarioV1, scenario.GeneratorV1ID, scenario.GeneratorV2ID)
	}
}

func main() {
	parsed, err := parseConfig(os.Args[1:], os.Stderr)
	if errors.Is(err, flag.ErrHelp) {
		return
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := runContext(ctx, os.Stdout, parsed); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func parseConfig(arguments []string, diagnostics io.Writer) (config, error) {
	parsed := config{}
	flags := flag.NewFlagSet("arena", flag.ContinueOnError)
	flags.SetOutput(diagnostics)
	flags.IntVar(&parsed.scenarios, "scenarios", 10, "number of paired scenario seeds (total games = 2 * scenarios)")
	flags.IntVar(&parsed.workers, "workers", runtime.GOMAXPROCS(0), "maximum concurrent matches")
	flags.Int64Var(&parsed.seed, "seed", 1, "first scenario seed")
	flags.StringVar(&parsed.scenario, "scenario", arena.FixedScenarioV1,
		"scenario factory: "+arena.FixedScenarioV1+", "+scenario.GeneratorV1ID+", or "+scenario.GeneratorV2ID)
	flags.StringVar(&parsed.p0, "p0", defaultP0, "contestant p0 agent specification")
	flags.StringVar(&parsed.p1, "p1", defaultP1, "contestant p1 agent specification")
	flags.StringVar(&parsed.record, "record", "", "optional NDJSON match-result path")
	if err := flags.Parse(arguments); err != nil {
		return config{}, err
	}
	if flags.NArg() != 0 {
		return config{}, fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}
	// Reject an unknown scenario during parsing, alongside the other flag
	// validation, so a typo fails before any worker starts rather than after
	// a long run has already been scheduled.
	if _, err := scenarioFactory(parsed.scenario); err != nil {
		return config{}, err
	}
	return parsed, nil
}

// run is the testable, context-free CLI boundary requested by the first arena
// release. main uses runContext so an interrupt can cancel active matches.
func run(output io.Writer, parsed config) error {
	return runContext(context.Background(), output, parsed)
}

func runContext(ctx context.Context, output io.Writer, parsed config) error {
	left, err := agent.ParseFactory(parsed.p0)
	if err != nil {
		return fmt.Errorf("parse -p0: %w", err)
	}
	right, err := agent.ParseFactory(parsed.p1)
	if err != nil {
		return fmt.Errorf("parse -p1: %w", err)
	}
	factory, err := scenarioFactory(parsed.scenario)
	if err != nil {
		return err
	}
	pairing := arena.PairingSpec{
		Scenario:      factory,
		ScenarioCount: parsed.scenarios,
		SeedStart:     parsed.seed,
		Contestants: []arena.ContestantSpec{
			{ID: "p0", Factory: left},
			{ID: "p1", Factory: right},
		},
		RecordActions: parsed.record != "",
	}
	results, err := arena.RunExperiment(ctx, arena.ExperimentSpec{
		Pairing: pairing,
		Workers: parsed.workers,
	})
	if err != nil {
		return fmt.Errorf("run experiment: %w", err)
	}
	summary, err := arena.Summarize(pairing, results)
	if err != nil {
		return fmt.Errorf("summarize experiment: %w", err)
	}
	if parsed.record != "" {
		if err := writeRecord(parsed.record, results); err != nil {
			return err
		}
	}
	return writeSummary(output, summary)
}

func writeRecord(path string, results []arena.MatchResult) error {
	// A record path commonly includes a new run directory. Create only its
	// parent, leaving os.Create to retain the documented overwrite behavior for
	// the record file itself.
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create record directory for %q: %w", path, err)
	}
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create record %q: %w", path, err)
	}
	if err := arena.WriteNDJSON(file, results); err != nil {
		_ = file.Close()
		return fmt.Errorf("write record %q: %w", path, err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close record %q: %w", path, err)
	}
	return nil
}

func writeSummary(output io.Writer, summary arena.ExperimentSummary) error {
	if _, err := fmt.Fprintf(output,
		"scenarios: %d\ngames: %d (completed: %d)\nterminations: elimination=%d turn-limit=%d\nfailures: agent=%d engine=%d other=%d cleanup=%d\naverage finishing turn: %.2f\n",
		summary.Scenarios,
		summary.Games,
		summary.CompletedGames,
		summary.Terminations.Eliminations,
		summary.Terminations.TurnLimits,
		summary.Failures.Agent,
		summary.Failures.Engine,
		summary.Failures.Other,
		summary.Failures.Cleanup,
		summary.AverageFinishTurn,
	); err != nil {
		return fmt.Errorf("write summary: %w", err)
	}
	for _, contestant := range summary.Contestants {
		if _, err := fmt.Fprintf(output,
			"contestant %s (%s): wins=%d losses=%d draws=%d rates=%.3f/%.3f/%.3f\n",
			contestant.ContestantID,
			contestant.AgentSpec,
			contestant.Wins,
			contestant.Losses,
			contestant.Draws,
			contestant.WinRate,
			contestant.LossRate,
			contestant.DrawRate,
		); err != nil {
			return fmt.Errorf("write summary: %w", err)
		}
		for _, seat := range contestant.Seats {
			if _, err := fmt.Fprintf(output,
				"  seat %d: wins=%d losses=%d draws=%d rates=%.3f/%.3f/%.3f\n",
				seat.PlayerID,
				seat.Wins,
				seat.Losses,
				seat.Draws,
				seat.WinRate,
				seat.LossRate,
				seat.DrawRate,
			); err != nil {
				return fmt.Errorf("write summary: %w", err)
			}
		}
	}
	return nil
}
