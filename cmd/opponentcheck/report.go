package main

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"
)

// testerOrder is the fixed testers -testers accepts, in the order they are
// listed everywhere.
var testerOrder = []string{"heuristic", "idle", "random", "diag-direct", "diag-lead"}

// specTester is the tester name recorded for the -tester-spec agent. It is
// not one of testerOrder, so -testers cannot ask for it.
const specTester = "spec"

// reportOrder is every tester a run can record, in report order. A
// -tester-spec run plays only specTester, so it never mixes with the others.
var reportOrder = append(slices.Clone(testerOrder), specTester)

// testerSummary is one tester's results over the games it completed.
type testerSummary struct {
	Tester    string `json:"tester"`
	Planned   int    `json:"planned"`
	Completed int    `json:"completed"`
	Failed    int    `json:"failed"`
	Cancelled int    `json:"cancelled"`
	Wins      int    `json:"wins"`
	Losses    int    `json:"losses"`
	Draws     int    `json:"draws"`
	// WinRate is wins over completed games; 0 with no completed game.
	WinRate float64 `json:"win_rate"`
	// SeatWinRate is the win rate for each seat the tester sat in.
	SeatWinRate     [players]float64 `json:"seat_win_rate"`
	Eliminated      int              `json:"eliminated"`
	LaunchesPerGame float64          `json:"launches_per_game"`
	ZeroLaunchGames int              `json:"zero_launch_games"`
	// LengthP10, LengthP50 and LengthP90 are percentiles of the finish turn.
	LengthP10 int `json:"length_p10"`
	LengthP50 int `json:"length_p50"`
	LengthP90 int `json:"length_p90"`
	// GameSeconds is the summed time of this tester's games. Games run in
	// parallel, so it is work done, not wall-clock time.
	GameSeconds float64 `json:"game_seconds"`

	// CounterSummary totals the TesterCounts of the completed games. It is
	// set only when they carry counts (a heuristic-v2 tester); embedded as a
	// pointer, it is left out of the JSON otherwise, so other testers'
	// summaries are unchanged.
	*CounterSummary
}

// CounterSummary is one tester's counts over its completed games, and the
// rates made from them. A rate is a pointer so that it can be null in the
// JSON, meaning undefined: nothing was counted to divide by. A rate of 0
// would claim something was counted and none of it failed.
type CounterSummary struct {
	AimAttempts int `json:"aim_attempts"`
	AimFailures int `json:"aim_failures"`
	// AimFailureRate is AimFailures / AimAttempts: how often aiming ahead
	// found no way to the target.
	AimFailureRate   *float64 `json:"aim_failure_rate"`
	FallbackLaunches int      `json:"fallback_launches"`
	// FallbackRate is FallbackLaunches over every launch in the same games:
	// the share of launches aimed straight because aiming ahead failed.
	FallbackRate *float64 `json:"fallback_rate"`
	RandomMoves  int      `json:"random_moves"`
	// RandomMovesPerGame is RandomMoves over the games counted.
	RandomMovesPerGame    float64 `json:"random_moves_per_game"`
	DecodeCheckedLaunches int     `json:"decode_checked_launches"`
}

// rate is part / whole, or nil (undefined) when whole is 0.
func rate(part, whole int) *float64 {
	if whole == 0 {
		return nil
	}
	r := float64(part) / float64(whole)
	return &r
}

// summarizeCounts totals the counts of the completed records that carry
// them, or returns nil when none does.
func summarizeCounts(records []gameRecord) *CounterSummary {
	var c CounterSummary
	games, launches := 0, 0
	for _, r := range records {
		if r.TesterCounts == nil {
			continue
		}
		games++
		launches += r.Launches
		c.AimAttempts += r.AimAttempts
		c.AimFailures += r.AimFailures
		c.FallbackLaunches += r.FallbackLaunches
		c.RandomMoves += r.RandomMoves
		c.DecodeCheckedLaunches += r.DecodeCheckedLaunches
	}
	if games == 0 {
		return nil
	}
	c.AimFailureRate = rate(c.AimFailures, c.AimAttempts)
	c.FallbackRate = rate(c.FallbackLaunches, launches)
	c.RandomMovesPerGame = float64(c.RandomMoves) / float64(games)
	return &c
}

// formatRate writes a rate as a percentage, or "undefined" when nil.
func formatRate(r *float64) string {
	if r == nil {
		return "undefined"
	}
	return fmt.Sprintf("%.1f%%", 100**r)
}

// calibration is the exact check on the calibration tester: the opponent
// playing itself. With the same agent in every seat, and each seat seeded by
// its seat rather than by the "tested" label, the four seat runs of one seed
// are the same game; only which seat counts as "tested" differs.
type calibration struct {
	// Seeds is how many seeds had all four seat runs completed.
	Seeds int `json:"seeds"`
	// Violations lists every seed whose four runs disagree on standings or
	// finish turn, or whose tested-seat wins are not exactly one (for a unique
	// winner) or zero (for a tie).
	Violations []string `json:"violations"`
	Passed     bool     `json:"passed"`
	// LeaderSeat counts, for seeds with a unique winner, how often each seat
	// won. Seat bias is reported here and is not a pass/fail check.
	LeaderSeat [players]int `json:"leader_seat"`
	TiedSeeds  int          `json:"tied_seeds"`
}

// summarize builds one testerSummary per tester, in reportOrder, from every
// game record, including the games that did not complete.
func summarize(records []gameRecord, planned map[string]int) []testerSummary {
	var summaries []testerSummary
	for _, tester := range reportOrder {
		if planned[tester] == 0 {
			continue
		}
		s := testerSummary{Tester: tester, Planned: planned[tester]}
		var seatGames, seatWins [players]int
		var lengths []int
		var completed []gameRecord
		launches := 0
		for _, r := range records {
			if r.Tester != tester {
				continue
			}
			s.GameSeconds += r.Seconds
			switch r.Status {
			case statusFailed:
				s.Failed++
				continue
			case statusCancelled:
				s.Cancelled++
				continue
			}
			s.Completed++
			completed = append(completed, r)
			seatGames[r.Seat]++
			switch {
			case r.Result > 0:
				s.Wins++
				seatWins[r.Seat]++
			case r.Result < 0:
				s.Losses++
			default:
				s.Draws++
			}
			if r.EliminatedTurn > 0 {
				s.Eliminated++
			}
			if r.Launches == 0 {
				s.ZeroLaunchGames++
			}
			launches += r.Launches
			lengths = append(lengths, r.FinishTurn)
		}
		if s.Completed > 0 {
			s.WinRate = float64(s.Wins) / float64(s.Completed)
			s.LaunchesPerGame = float64(launches) / float64(s.Completed)
		}
		for seat := range players {
			if seatGames[seat] > 0 {
				s.SeatWinRate[seat] = float64(seatWins[seat]) / float64(seatGames[seat])
			}
		}
		s.LengthP10, s.LengthP50, s.LengthP90 = percentile(lengths, 0.1), percentile(lengths, 0.5), percentile(lengths, 0.9)
		s.CounterSummary = summarizeCounts(completed)
		summaries = append(summaries, s)
	}
	return summaries
}

// percentile is the nearest-rank percentile of values, or 0 for none.
func percentile(values []int, fraction float64) int {
	if len(values) == 0 {
		return 0
	}
	sorted := slices.Clone(values)
	sort.Ints(sorted)
	rank := int(math.Ceil(fraction*float64(len(sorted)))) - 1
	return sorted[max(rank, 0)]
}

// checkCalibration applies the exact calibration rule to the named tester's
// completed games: the heuristic tester, or the -tester-spec agent when it is
// the opponent spec (see matchConfig.calibrationTester). An empty name checks
// nothing. Seeds missing any of their four seat runs are skipped, and only a
// complete run gives a verdict anyway.
func checkCalibration(records []gameRecord, tester string) calibration {
	bySeed := map[int64][]gameRecord{}
	for _, r := range records {
		if tester != "" && r.Tester == tester && r.Status == statusCompleted {
			bySeed[r.Seed] = append(bySeed[r.Seed], r)
		}
	}
	seeds := make([]int64, 0, len(bySeed))
	for seed := range bySeed {
		seeds = append(seeds, seed)
	}
	slices.Sort(seeds)

	var c calibration
	for _, seed := range seeds {
		runs := bySeed[seed]
		if len(runs) != players {
			continue
		}
		c.Seeds++
		first := runs[0]
		wins := 0
		consistent := true
		for _, r := range runs {
			if r.Ships != first.Ships || r.FinishTurn != first.FinishTurn || !slices.Equal(r.Leaders, first.Leaders) {
				consistent = false
			}
			if r.Result > 0 {
				wins++
			}
		}
		if !consistent {
			c.Violations = append(c.Violations, fmt.Sprintf("seed %d: the four seat runs did not end identically", seed))
			continue
		}
		switch len(first.Leaders) {
		case 1:
			c.LeaderSeat[first.Leaders[0]]++
			if wins != 1 {
				c.Violations = append(c.Violations, fmt.Sprintf("seed %d: unique winner seat %d, but %d tested-seat wins", seed, first.Leaders[0], wins))
			}
		default:
			c.TiedSeeds++
			if wins != 0 {
				c.Violations = append(c.Violations, fmt.Sprintf("seed %d: tied, but %d tested-seat wins", seed, wins))
			}
		}
	}
	c.Passed = c.Seeds > 0 && len(c.Violations) == 0
	return c
}

// verdict applies the plan's decision criteria. It gives one only for a
// complete run whose calibration passed and which played both idle and
// diag-lead; otherwise it says why there is none.
//
// The thresholds were fixed for the agreed configurations: one of the pinned
// opponent specs in agreedOpponents, the pinned random-tester spec, and
// p = 0.42. A run with any of them overridden still reports every result,
// but gives no verdict, so a result about some other agent can never read as
// a decision about an agreed opponent. Every message names the opponent
// actually played.
func verdict(meta runMeta, complete bool, summaries []testerSummary, c calibration) (string, []string) {
	if !complete {
		return "no verdict: the run is incomplete", nil
	}
	// The criteria were fixed for the fixed testers only. A -tester-spec run
	// measures some other agent, so its results are reported but not judged.
	if meta.TesterSpec != "" {
		return "no verdict: a -tester-spec run measures its tester but gives no verdict", nil
	}
	if !slices.Contains(agreedOpponents, meta.Opponent) || meta.Random != defaultRandom || meta.LaunchProbability != defaultLaunchProbability {
		return fmt.Sprintf("no verdict: the decision criteria apply only to the agreed configuration (opponent one of %q, random tester %q, p = %g); this run used opponent %q, random tester %q, p = %g",
			agreedOpponents, defaultRandom, defaultLaunchProbability, meta.Opponent, meta.Random, meta.LaunchProbability), nil
	}
	opponent := meta.Opponent
	rates := map[string]float64{}
	for _, s := range summaries {
		rates[s.Tester] = s.WinRate
	}
	if _, ok := rates["heuristic"]; !ok {
		return "no verdict: calibration (the heuristic tester) was not run", nil
	}
	if !c.Passed {
		return "stop: calibration failed, so the harness or its seat handling is wrong", nil
	}
	idle, haveIdle := rates["idle"]
	lead, haveLead := rates["diag-lead"]
	if !haveIdle || !haveLead {
		return "no verdict: the criteria need both the idle and diag-lead testers", nil
	}

	var notes []string
	if direct, ok := rates["diag-direct"]; ok && lead-direct >= 0.15 {
		notes = append(notes, fmt.Sprintf("lead aiming helps a challenger against %s: diag-lead %.1f%% vs diag-direct %.1f%%. This does not show how much that opponent itself would gain from better aiming.", opponent, 100*lead, 100*direct))
	}
	switch {
	case lead >= 0.40 || idle >= 0.25:
		return fmt.Sprintf("too weak: %s should not be the gate opponent; a stronger opponent is needed", opponent), notes
	case lead < 0.25 && idle < 0.10:
		return fmt.Sprintf("keep %s", opponent), notes
	default:
		return fmt.Sprintf("discuss: the results for %s fall between the agreed thresholds", opponent), notes
	}
}

// markdownSummary renders the run for a person: verdict first, then each
// tester's results, then calibration.
func markdownSummary(meta runMeta, summaries []testerSummary, c calibration, decision string, notes []string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Opponent check\n\n")
	fmt.Fprintf(&b, "**Status:** %s. **Verdict:** %s\n\n", meta.Status, decision)
	for _, note := range notes {
		fmt.Fprintf(&b, "- %s\n", note)
	}
	if len(notes) > 0 {
		b.WriteString("\n")
	}
	if meta.TesterSpec != "" {
		fmt.Fprintf(&b, "Tester `%s` is `%s`. ", specTester, meta.TesterSpec)
	}
	if meta.CheckDecode {
		b.WriteString("Every tester launch was checked against the learner's decoder; a mismatch fails its game. ")
	}
	against := fmt.Sprintf("three `%s` opponents", meta.Opponent)
	if strings.Contains(meta.Opponent, "|") {
		against = fmt.Sprintf("the lineup `%s` (clockwise | opposite | counter-clockwise of the tester)", meta.Opponent)
	}
	fmt.Fprintf(&b, "Seeds %d to %d, every seat, against %s. ", meta.SeedStart, meta.SeedStart+int64(meta.Seeds)-1, against)
	fmt.Fprintf(&b, "%d of %d games completed in %.0f s on %d workers.\n\n", meta.Completed, meta.Planned, meta.ElapsedSeconds, meta.Workers)

	b.WriteString("| Tester | Completed | Wins | Losses | Draws | Win rate | Seat 0 | Seat 1 | Seat 2 | Seat 3 | Eliminated | Launches/game | Zero-launch games | Length p10/p50/p90 |\n")
	b.WriteString("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
	for _, s := range summaries {
		fmt.Fprintf(&b, "| %s | %d/%d | %d | %d | %d | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %d | %.1f | %d | %d/%d/%d |\n",
			s.Tester, s.Completed, s.Planned, s.Wins, s.Losses, s.Draws, 100*s.WinRate,
			100*s.SeatWinRate[0], 100*s.SeatWinRate[1], 100*s.SeatWinRate[2], 100*s.SeatWinRate[3],
			s.Eliminated, s.LaunchesPerGame, s.ZeroLaunchGames, s.LengthP10, s.LengthP50, s.LengthP90)
	}

	// The counts table appears only when some tester reports counts, so
	// other runs' summaries are unchanged.
	if slices.ContainsFunc(summaries, func(s testerSummary) bool { return s.CounterSummary != nil }) {
		b.WriteString("\n## Aim and decode counts\n\n")
		b.WriteString("Over completed games. A rate is undefined when nothing was counted to divide by.\n\n")
		b.WriteString("| Tester | Aim attempts | Aim failures | Aim failure rate | Fallback launches | Fallback share of launches | Random moves | Random moves/game | Decode-checked launches |\n")
		b.WriteString("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
		for _, s := range summaries {
			if c := s.CounterSummary; c != nil {
				fmt.Fprintf(&b, "| %s | %d | %d | %s | %d | %s | %d | %.1f | %d |\n",
					s.Tester, c.AimAttempts, c.AimFailures, formatRate(c.AimFailureRate),
					c.FallbackLaunches, formatRate(c.FallbackRate), c.RandomMoves, c.RandomMovesPerGame, c.DecodeCheckedLaunches)
			}
		}
	}

	fmt.Fprintf(&b, "\n## Calibration\n\n")
	fmt.Fprintf(&b, "Seeds checked: %d. Passed: %v. Tied seeds: %d. ", c.Seeds, c.Passed, c.TiedSeeds)
	fmt.Fprintf(&b, "Unique winners by seat (seat bias, not pass/fail): %v.\n", c.LeaderSeat)
	for _, v := range c.Violations {
		fmt.Fprintf(&b, "- %s\n", v)
	}
	// These settings only shape the fixed diagnostic and random testers, so a
	// -tester-spec run, which plays neither, leaves them out here.
	// summary.json still records both.
	if meta.TesterSpec == "" {
		fmt.Fprintf(&b, "\nDiagnostic launch probability p = %g. Random tester: `%s`.\n", meta.LaunchProbability, meta.Random)
	}
	return b.String()
}
