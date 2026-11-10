// Command opponentcheck measures how strong an opponent is on four-player
// generator-v2-4p maps. It plays each tested player in one seat against three
// opponents (heuristic-v1 by default), on the reusable validation seeds, with
// the tested player in every seat in turn. It runs entirely in Go: no model,
// no Python, no GPU.
//
// Each game is appended to games.jsonl as soon as it finishes, so an
// interrupted run keeps its completed work. At the end it writes the sorted
// games.csv, summary.json and summary.md. A run with any failed or unplayed
// game is incomplete: it gives no verdict and exits with status 1.
package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"orrery/agent"
)

const (
	// defaultOpponent and defaultRandom are the rl-v2 run config's training
	// and pipeline opponents, pinned so a run measures exactly those agents.
	defaultOpponent = "heuristic-v1:aggression=1.2,defense=0.1,economy=1,efficiency=1"
	defaultRandom   = "random-v1:launch_probability=0.3,max_fraction=0.8,min_fraction=0.2"
	// defaultLaunchProbability is the diagnostic agents' agreed p. Together
	// with defaultRandom and one of agreedOpponents it is the configuration
	// the decision criteria were fixed for; a run that overrides any of them
	// gives no verdict.
	defaultLaunchProbability = 0.42

	// progressEvery and heartbeatEvery pace the console output: a progress
	// line per batch of finished games, and a heartbeat so a slow stretch is
	// never silent.
	progressEvery  = 50
	heartbeatEvery = 60 * time.Second
)

// agreedOpponents are the opponent specs the decision criteria apply to: the
// training heuristic-v1 (the default) and the one pinned heuristic-v2 spec,
// the candidate to replace it. Both are judged against the same thresholds,
// fixed before either was measured. Any other opponent still gives no verdict.
var agreedOpponents = []string{defaultOpponent, agent.HeuristicV2Default}

// options are the parsed command-line settings.
type options struct {
	testers           []string
	seedStart         int64
	seeds             int
	launchProbability float64
	workers           int
	opponent          string
	random            string
	out               string
}

// runMeta records how a run was made and how it ended.
type runMeta struct {
	Command           []string `json:"command"`
	Commit            string   `json:"commit"`
	GoVersion         string   `json:"go_version"`
	Status            string   `json:"status"`
	Testers           []string `json:"testers"`
	SeedStart         int64    `json:"seed_start"`
	Seeds             int      `json:"seeds"`
	Opponent          string   `json:"opponent"`
	Random            string   `json:"random"`
	LaunchProbability float64  `json:"launch_probability"`
	Workers           int      `json:"workers"`
	Planned           int      `json:"planned"`
	Completed         int      `json:"completed"`
	ElapsedSeconds    float64  `json:"elapsed_seconds"`
}

func main() {
	opts, err := parseOptions(os.Args[1:], os.Stderr)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			os.Exit(0)
		}
		fmt.Fprintln(os.Stderr, "opponentcheck:", err)
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	complete, err := run(ctx, opts, os.Stdout)
	if err != nil {
		fmt.Fprintln(os.Stderr, "opponentcheck:", err)
		os.Exit(1)
	}
	if !complete {
		os.Exit(1)
	}
}

func parseOptions(args []string, stderr io.Writer) (options, error) {
	flags := flag.NewFlagSet("opponentcheck", flag.ContinueOnError)
	flags.SetOutput(stderr)
	testers := flags.String("testers", strings.Join(testerOrder, ","), "comma-separated testers: "+strings.Join(testerOrder, ", ")+
		" (heuristic is the -opponent agent playing itself, the calibration)")
	seedStart := flags.Int64("seed-start", 1_000_000, "first scenario seed (the reusable validation range starts at 1000000)")
	seeds := flags.Int("seeds", 25, "number of consecutive seeds; each is played from every seat")
	probability := flags.Float64("launch-probability", defaultLaunchProbability, "diagnostic agents' chance to launch from each source each turn")
	workers := flags.Int("workers", runtime.GOMAXPROCS(0), "games played at once")
	opponent := flags.String("opponent", defaultOpponent, "agent spec for the three opponent seats")
	random := flags.String("random", defaultRandom, "agent spec for the random tester")
	out := flags.String("out", "", "new directory for the results (required; never overwritten)")
	if err := flags.Parse(args); err != nil {
		return options{}, err
	}
	if flags.NArg() != 0 {
		return options{}, fmt.Errorf("unexpected arguments: %v", flags.Args())
	}
	opts := options{
		seedStart: *seedStart, seeds: *seeds, launchProbability: *probability,
		workers: *workers, opponent: *opponent, random: *random, out: *out,
	}
	for _, name := range strings.Split(*testers, ",") {
		name = strings.TrimSpace(name)
		if !slices.Contains(testerOrder, name) {
			return options{}, fmt.Errorf("%w %q; want some of %s", errUnknownTester, name, strings.Join(testerOrder, ", "))
		}
		if !slices.Contains(opts.testers, name) {
			opts.testers = append(opts.testers, name)
		}
	}
	switch {
	case opts.out == "":
		return options{}, errors.New("-out is required")
	case opts.seeds < 1:
		return options{}, errors.New("-seeds must be at least 1")
	case opts.workers < 1:
		return options{}, errors.New("-workers must be at least 1")
	case opts.launchProbability < 0 || opts.launchProbability > 1:
		return options{}, errors.New("-launch-probability must be in [0, 1]")
	}
	return opts, nil
}

// run plays every job and writes the results. It returns whether every game
// completed; an error means the run could not be set up or written at all.
//
// The stages are: validate the agents and claim the output directory; play
// the games on a fixed pool of workers, appending each to games.jsonl as it
// finishes; then sort everything and write the summaries.
func run(ctx context.Context, opts options, console io.Writer) (bool, error) {
	opponent, err := agent.ParseFactory(opts.opponent)
	if err != nil {
		return false, fmt.Errorf("-opponent: %w", err)
	}
	random, err := agent.ParseFactory(opts.random)
	if err != nil {
		return false, fmt.Errorf("-random: %w", err)
	}
	return runWith(ctx, opts, matchConfig{opponent: opponent, random: random, launchProbability: opts.launchProbability}, console)
}

// runWith is run with a ready matchConfig, so tests can inject agents.
func runWith(ctx context.Context, opts options, config matchConfig, console io.Writer) (bool, error) {
	started := time.Now()
	if err := os.MkdirAll(filepath.Dir(opts.out), 0o755); err != nil {
		return false, err
	}
	// Mkdir, not MkdirAll: an existing directory is refused, so earlier
	// results are never overwritten.
	if err := os.Mkdir(opts.out, 0o755); err != nil {
		return false, fmt.Errorf("-out must name a new directory: %w", err)
	}
	stream, err := os.OpenFile(filepath.Join(opts.out, "games.jsonl"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return false, err
	}
	defer stream.Close()

	var jobs []job
	planned := map[string]int{}
	for _, tester := range opts.testers {
		for i := range opts.seeds {
			for seat := range players {
				jobs = append(jobs, job{tester: tester, seed: opts.seedStart + int64(i), seat: seat})
				planned[tester]++
			}
		}
	}
	fmt.Fprintf(console, "opponentcheck: %d games (%d testers x %d seeds x %d seats) on %d workers\n",
		len(jobs), len(opts.testers), opts.seeds, players, opts.workers)

	records := playAll(ctx, opts.workers, jobs, config, stream, console, started)

	// Every record is written, completed or not; sorting by (tester, seed,
	// seat) makes the files independent of the order games finished in.
	sort.Slice(records, func(i, j int) bool {
		a, b := records[i], records[j]
		if a.Tester != b.Tester {
			return testerIndex(a.Tester) < testerIndex(b.Tester)
		}
		if a.Seed != b.Seed {
			return a.Seed < b.Seed
		}
		return a.Seat < b.Seat
	})
	completed := 0
	for _, r := range records {
		if r.Status == statusCompleted {
			completed++
		}
	}
	complete := completed == len(jobs)
	meta := runMeta{
		Command: os.Args, Commit: buildCommit(), GoVersion: runtime.Version(),
		Status: "complete", Testers: opts.testers, SeedStart: opts.seedStart, Seeds: opts.seeds,
		Opponent: opts.opponent, Random: opts.random, LaunchProbability: opts.launchProbability,
		Workers: opts.workers, Planned: len(jobs), Completed: completed,
		ElapsedSeconds: time.Since(started).Seconds(),
	}
	if !complete {
		meta.Status = fmt.Sprintf("incomplete (%d of %d games completed)", completed, len(jobs))
	}
	summaries := summarize(records, planned)
	calib := checkCalibration(records)
	decision, notes := verdict(meta, complete, summaries, calib)

	if err := writeCSV(filepath.Join(opts.out, "games.csv"), records); err != nil {
		return false, err
	}
	summaryJSON, err := json.MarshalIndent(map[string]any{
		"run": meta, "testers": summaries, "calibration": calib, "verdict": decision, "notes": notes,
	}, "", "  ")
	if err != nil {
		return false, err
	}
	if err := os.WriteFile(filepath.Join(opts.out, "summary.json"), append(summaryJSON, '\n'), 0o644); err != nil {
		return false, err
	}
	if err := os.WriteFile(filepath.Join(opts.out, "summary.md"), []byte(markdownSummary(meta, summaries, calib, decision, notes)), 0o644); err != nil {
		return false, err
	}
	fmt.Fprintf(console, "opponentcheck: %s; verdict: %s\nresults: %s\n", meta.Status, decision, opts.out)
	return complete, nil
}

// playAll runs the jobs on a fixed pool of workers and returns one record per
// job that was started. Each finished game is appended to stream straight
// away. Once the context is cancelled, no new game starts; games already
// running stop at their next turn and are recorded as cancelled.
func playAll(ctx context.Context, workers int, jobs []job, config matchConfig, stream io.Writer, console io.Writer, started time.Time) []gameRecord {
	queue := make(chan job)
	results := make(chan gameRecord)
	var wg sync.WaitGroup
	for range min(workers, len(jobs)) {
		wg.Go(func() {
			for j := range queue {
				results <- playGame(ctx, config, j)
			}
		})
	}
	go func() {
		defer close(queue)
		for _, j := range jobs {
			select {
			case queue <- j:
			case <-ctx.Done():
				return
			}
		}
	}()
	go func() {
		wg.Wait()
		close(results)
	}()

	heartbeat := time.NewTicker(heartbeatEvery)
	defer heartbeat.Stop()
	var records []gameRecord
	completed := 0
	for {
		select {
		case r, ok := <-results:
			if !ok {
				return records
			}
			records = append(records, r)
			if r.Status == statusCompleted {
				completed++
			}
			if line, err := json.Marshal(r); err == nil {
				// One write per game, unbuffered, so a crash loses at most the
				// game being written.
				_, _ = stream.Write(append(line, '\n'))
			}
			if len(records)%progressEvery == 0 || len(records) == len(jobs) {
				elapsed := time.Since(started)
				left := time.Duration(float64(elapsed) / float64(len(records)) * float64(len(jobs)-len(records)))
				fmt.Fprintf(console, "progress: %d/%d games finished (%d completed), elapsed %s, about %s left\n",
					len(records), len(jobs), completed, elapsed.Round(time.Second), left.Round(time.Second))
			}
		case <-heartbeat.C:
			fmt.Fprintf(console, "heartbeat: %d/%d games finished, elapsed %s\n",
				len(records), len(jobs), time.Since(started).Round(time.Second))
		}
	}
}

// writeCSV writes one row per game, in the records' order.
func writeCSV(path string, records []gameRecord) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()
	w := csv.NewWriter(file)
	header := []string{"tester", "seed", "seat", "status", "result", "finish_turn", "termination",
		"eliminated_turn", "launches", "ships_seat0", "ships_seat1", "ships_seat2", "ships_seat3",
		"leaders", "opponents_eliminated", "fail_turn", "fail_seat", "message"}
	if err := w.Write(header); err != nil {
		return err
	}
	for _, r := range records {
		failSeat := ""
		if r.FailSeat != nil {
			failSeat = strconv.Itoa(*r.FailSeat)
		}
		leaders := make([]string, len(r.Leaders))
		for i, l := range r.Leaders {
			leaders[i] = strconv.Itoa(l)
		}
		row := []string{r.Tester, strconv.FormatInt(r.Seed, 10), strconv.Itoa(r.Seat), r.Status,
			strconv.Itoa(r.Result), strconv.Itoa(r.FinishTurn), r.Termination, strconv.Itoa(r.EliminatedTurn),
			strconv.Itoa(r.Launches), strconv.Itoa(r.Ships[0]), strconv.Itoa(r.Ships[1]),
			strconv.Itoa(r.Ships[2]), strconv.Itoa(r.Ships[3]), strings.Join(leaders, " "),
			strconv.Itoa(r.OpponentsEliminated), strconv.Itoa(r.FailTurn), failSeat, r.Message}
		if err := w.Write(row); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

// buildCommit is the source revision Go recorded at build time, or
// "unknown" when the build carries none (for example, a copied bundle).
func buildCommit() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}
	revision, modified := "", false
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			revision = setting.Value
		case "vcs.modified":
			modified = setting.Value == "true"
		}
	}
	if revision == "" {
		return "unknown"
	}
	if modified {
		return revision + "+dirty"
	}
	return revision
}

// testerIndex is a tester's position in testerOrder, for sorting records.
func testerIndex(name string) int {
	return slices.Index(testerOrder, name)
}
