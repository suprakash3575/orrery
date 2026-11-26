// Command opponentcheck measures how strong an opponent is on four-player
// generator-v2-4p maps. It plays each tested player in one seat against three
// opponents (heuristic-v1 by default), on the reusable validation seeds, with
// the tested player in every seat in turn. It runs entirely in Go: no model,
// no Python, no GPU.
//
// With -tester-spec, any agent spec plays the tested seat instead of the
// fixed testers. Such a run only measures; it gives no verdict. -opponent
// also takes a lineup of three specs joined by "|", seated by board role
// from the tested seat, as mlenv seats training opponents. -check-decode
// proves a restricted heuristic-v2 tester made only the learner's moves.
// -canonical prints specs in canonical form and plays nothing, so a run
// script can check its specs before starting.
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
	"orrery/mlenv"
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
	// testers are the fixed testers to play, or just specTester when
	// testerSpec is set.
	testers []string
	// testerSpec is the agent spec for the tested seat, as typed; empty means
	// the fixed testers are played.
	testerSpec string
	// checkDecode turns on the decode check (see decodecheck.go).
	checkDecode bool
	// canonical holds the specs to print in canonical form (-canonical).
	// When it is set, no other option is, and nothing is played.
	canonical         []string
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
	Command   []string `json:"command"`
	Commit    string   `json:"commit"`
	GoVersion string   `json:"go_version"`
	Status    string   `json:"status"`
	Testers   []string `json:"testers"`
	// TesterSpec is the -tester-spec agent, left out of the JSON when the
	// fixed testers were played, so those runs' summaries are unchanged.
	TesterSpec string `json:"tester_spec,omitempty"`
	// CheckDecode records that the decode check ran on every tested turn;
	// left out when it didn't, so other runs' summaries are unchanged.
	CheckDecode       bool    `json:"check_decode,omitempty"`
	SeedStart         int64   `json:"seed_start"`
	Seeds             int     `json:"seeds"`
	Opponent          string  `json:"opponent"`
	Random            string  `json:"random"`
	LaunchProbability float64 `json:"launch_probability"`
	Workers           int     `json:"workers"`
	Planned           int     `json:"planned"`
	Completed         int     `json:"completed"`
	ElapsedSeconds    float64 `json:"elapsed_seconds"`
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
	if opts.canonical != nil {
		forms, err := canonicalForms(opts.canonical)
		if err != nil {
			fmt.Fprintln(os.Stderr, "opponentcheck:", err)
			os.Exit(1)
		}
		for _, form := range forms {
			fmt.Println(form)
		}
		return
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
	testerSpec := flags.String("tester-spec", "", "agent spec for the tested seat instead of the fixed testers; cannot be combined with -testers")
	checkDecode := flags.Bool("check-decode", false, "fail any game in which the tester makes a launch the learner's decoder would not make the same way; "+
		"needs a -tester-spec of heuristic-v2 with whole_garrison=1, learner_aim=1, no_aim_check=1 and no epsilon")
	seedStart := flags.Int64("seed-start", 1_000_000, "first scenario seed (the reusable validation range starts at 1000000)")
	seeds := flags.Int("seeds", 25, "number of consecutive seeds; each is played from every seat")
	probability := flags.Float64("launch-probability", defaultLaunchProbability, "diagnostic agents' chance to launch from each source each turn")
	workers := flags.Int("workers", runtime.GOMAXPROCS(0), "games played at once")
	opponent := flags.String("opponent", defaultOpponent, "agent spec for the three opponent seats, or three specs joined by | for the seats clockwise, opposite and counter-clockwise of the tested seat")
	random := flags.String("random", defaultRandom, "agent spec for the random tester")
	out := flags.String("out", "", "new directory for the results (required; never overwritten)")
	canonical := flags.Bool("canonical", false, "print the canonical form of each spec or lineup given as an argument, one per line, and exit; plays nothing")
	if err := flags.Parse(args); err != nil {
		return options{}, err
	}
	// -canonical is a separate mode: it takes specs as arguments and plays
	// nothing, so no other flag applies and -out isn't needed.
	if *canonical {
		others := 0
		flags.Visit(func(f *flag.Flag) {
			if f.Name != "canonical" {
				others++
			}
		})
		switch {
		case others > 0:
			return options{}, errors.New("-canonical cannot be combined with other flags")
		case flags.NArg() == 0:
			return options{}, errors.New("-canonical needs at least one spec")
		}
		return options{canonical: flags.Args()}, nil
	}
	if flags.NArg() != 0 {
		return options{}, fmt.Errorf("unexpected arguments: %v", flags.Args())
	}
	opts := options{
		testerSpec: strings.TrimSpace(*testerSpec), checkDecode: *checkDecode,
		seedStart: *seedStart, seeds: *seeds, launchProbability: *probability,
		workers: *workers, opponent: *opponent, random: *random, out: *out,
	}
	// Whether a flag was typed is asked of flags.Visit (which sees just the
	// flags actually given), never of its value. -testers has a default, so
	// its value can't tell; and -tester-spec typed with an empty value, say
	// from an unset variable in a script, must not quietly fall back to the
	// fixed testers.
	testersGiven, testerSpecGiven := false, false
	flags.Visit(func(f *flag.Flag) {
		testersGiven = testersGiven || f.Name == "testers"
		testerSpecGiven = testerSpecGiven || f.Name == "tester-spec"
	})
	switch {
	case testerSpecGiven && opts.testerSpec == "":
		return options{}, errors.New("-tester-spec must not be empty")
	case testerSpecGiven && testersGiven:
		return options{}, errors.New("-tester-spec cannot be combined with -testers")
	case testerSpecGiven:
		opts.testers = []string{specTester}
	default:
		for _, name := range strings.Split(*testers, ",") {
			name = strings.TrimSpace(name)
			if !slices.Contains(testerOrder, name) {
				return options{}, fmt.Errorf("%w %q; want some of %s", errUnknownTester, name, strings.Join(testerOrder, ", "))
			}
			if !slices.Contains(opts.testers, name) {
				opts.testers = append(opts.testers, name)
			}
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

// canonicalForms returns the canonical form of each input, in input order.
// Each is read exactly as -opponent reads it: one agent spec, or a lineup of
// three joined by "|", parsed and checked by mlenv.Lineup. So a lineup of
// three identical entries comes back as that one spec, which is the only
// spelling training accepts for it.
//
// A caller checks a spec by comparing it with its canonical form: any
// difference (parameter order, a setting written at its default, 0.50 for
// 0.5) means the spelling isn't canonical. If any input doesn't parse, it
// returns no forms and an error naming every bad input, so a caller never
// pairs a form with the wrong input.
func canonicalForms(inputs []string) ([]string, error) {
	forms := make([]string, 0, len(inputs))
	var errs []error
	for _, input := range inputs {
		lineup, err := mlenv.ParseLineup(input, agent.ParseFactory)
		if err == nil {
			err = lineup.ValidateFor(players)
		}
		if err != nil {
			errs = append(errs, fmt.Errorf("%q: %w", input, err))
			continue
		}
		forms = append(forms, lineup.String())
	}
	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}
	return forms, nil
}

// run plays every job and writes the results. It returns whether every game
// completed; an error means the run could not be set up or written at all.
//
// The stages are: validate the agents and claim the output directory; play
// the games on a fixed pool of workers, appending each to games.jsonl as it
// finishes; then sort everything and write the summaries.
func run(ctx context.Context, opts options, console io.Writer) (bool, error) {
	// -opponent is read exactly as mlenv reads a training opponent: one spec
	// for every opponent seat, or three joined by "|", one per board role.
	opponents, err := mlenv.ParseLineup(opts.opponent, agent.ParseFactory)
	if err == nil {
		err = opponents.ValidateFor(players)
	}
	if err != nil {
		return false, fmt.Errorf("-opponent: %w", err)
	}
	// The heuristic tester is "the opponent playing itself", which only means
	// something when every opponent seat plays one agent. A lineup's
	// canonical form keeps the separator only when its entries differ.
	if strings.Contains(opponents.String(), mlenv.LineupSeparator) && slices.Contains(opts.testers, "heuristic") {
		return false, errors.New("-opponent: the heuristic tester needs one agent in every opponent seat; leave it out of -testers when the lineup is mixed")
	}
	random, err := agent.ParseFactory(opts.random)
	if err != nil {
		return false, fmt.Errorf("-random: %w", err)
	}
	config := matchConfig{opponents: opponents, random: random, launchProbability: opts.launchProbability, checkDecode: opts.checkDecode}
	if opts.testerSpec != "" {
		if config.tester, err = agent.ParseFactory(opts.testerSpec); err != nil {
			return false, fmt.Errorf("-tester-spec: %w", err)
		}
	}
	// Checked here, not in parseOptions, because only the parsed tester's
	// canonical spec says which settings are on.
	if opts.checkDecode {
		if err := checkDecodeAllowed(config.tester); err != nil {
			return false, err
		}
	}
	return runWith(ctx, opts, config, console)
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
		Status: "complete", Testers: opts.testers, TesterSpec: opts.testerSpec, CheckDecode: opts.checkDecode, SeedStart: opts.seedStart, Seeds: opts.seeds,
		Opponent: opts.opponent, Random: opts.random, LaunchProbability: opts.launchProbability,
		Workers: opts.workers, Planned: len(jobs), Completed: completed,
		ElapsedSeconds: time.Since(started).Seconds(),
	}
	if !complete {
		meta.Status = fmt.Sprintf("incomplete (%d of %d games completed)", completed, len(jobs))
	}
	summaries := summarize(records, planned)
	calib := checkCalibration(records, config.calibrationTester())
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

// testerIndex is a tester's position in reportOrder, for sorting records.
func testerIndex(name string) int {
	return slices.Index(reportOrder, name)
}
