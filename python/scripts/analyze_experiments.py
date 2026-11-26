#!/usr/bin/env python3
"""Analyse a finished (or partly finished) experiment session.

It reads a session folder written by run_experiments.py and writes
``analysis.md`` (for people) and ``analysis.json`` (for programs) into it:
each experiment's tables and, where the experiment file asks for one, its
answer, by the rules written out next to each answer function below. It
never plays games, and it can be rerun at any time; it rewrites only those
two files.

Words used throughout this file:

- **Win rate**: wins ÷ games. A win means finishing strictly first, so in a
  four-player game an equal player wins about 25% of the time.
- **Points**: a difference between two win rates, in percentage points.
  Example: 30% minus 25% is +5 points (not +20%).
- **Map**: one generated board, named by its seed. Every job plays each map
  from all 4 seats, so 100 maps give 400 games.
- **95% range**: how far a difference might move if other maps had been
  played. It comes from a bootstrap (see compare_jobs).

What it reports, by section of the experiment file:

- ``[[botcheck]]``: each tester's results, aim counts and calibration, and the
  map seed and seat of every win of a decode-checked tester, so it can be
  replayed.
- ``[[compare]]``: candidate minus baseline win rate, compared map by map with
  a 95% range. ``rule = "restricted-bot"`` adds experiment 1's answer.
- ``[[dial]]``: whether the win rate falls at every step (experiment 2).
- ``[[diagnose]]``: each model's win rate per opponent, and with a ``band``,
  whether some opponent sits inside it (experiment 3).
- ``[[train]]``: training win rate and diagnosis win rate at each checkpoint,
  side by side, and experiment 4's answer.

A job that isn't done is listed, never guessed: an answer that needs it says
so instead.

Run from the ``python/`` directory::

    uv run python -m scripts.analyze_experiments ../runs/stage1
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from scripts.compare_diagnoses import (
    DEFAULT_RESAMPLES,
    bootstrap_range,
    load_report,
    require_comparable,
    require_valid,
)
from scripts.run_experiments import Job, Plan, read_job_state, parse_plan

# Experiment 1's thresholds, as win-rate differences (a difference of -0.03
# would be -3 points):
# the fully limited bot "keeps its strength" if its whole 95% range is above
# KEPT_ABOVE, and "loses strength" if it is at most LOST_AT_MOST behind with
# the whole range below 0. See answer_restricted_bot.
KEPT_ABOVE = -0.05
LOST_AT_MOST = -0.10
# The bootstrap draws random samples. Starting its random numbers from a
# fixed seed makes it draw the same samples every time, so rerunning the
# analysis gives exactly the same 95% ranges.
BOOTSTRAP_SEED = 0
# The seats a bot job plays each map from: every Stage 1 game has four
# players, and run_experiments.py counts a job's games the same way.
BOT_SEATS = 4
# Training's win rate is compared over this many updates at each end of a run
# to decide whether it rose. Averaging several updates smooths out one lucky
# or unlucky update.
TREND_UPDATES = 10


def format_percent(value: float | None) -> str:
    """A share as a percentage with one decimal: 0.125 → "12.5%". None means
    nothing was counted, written as "undefined" rather than a misleading 0%."""

    return "undefined" if value is None else f"{100 * value:.1f}%"


def format_points(value: float | None) -> str:
    """A win-rate difference in percentage points, always with its sign:
    -0.125 → "-12.5 pts", 0.05 → "+5.0 pts"."""

    return "undefined" if value is None else f"{100 * value:+.1f} pts"


def format_diagnosis_cell(rate: float | None) -> str:
    """A checkpoint's diagnosis win rate as a table cell; "missing" when the
    checkpoint has no diagnosis report."""

    return "missing" if rate is None else format_percent(rate)


def job_folder(out: Path, job: Job) -> Path:
    """A job's folder in the session folder: <out>/<experiment>/<label>."""

    return out / job.experiment / job.label


# ---------------------------------------------------------------------------
# Reading each kind of job's results.
#
# Each job kind leaves its results in the files its command writes:
#   bot job       <job>/opponentcheck/games.jsonl and summary.json
#   diagnose job  <job>/diagnosis/report.json
#   train job     <job>/run/metrics.csv and <job>/diagnoses/<checkpoint>__<label>/report.json
# ---------------------------------------------------------------------------


def read_bot_games(out: Path, job: Job) -> list[dict[str, Any]]:
    """The completed games of a bot job, one row per (map seed, seat).

    games.jsonl holds one JSON object per line, one per game. ``result`` is
    the tester's reward: 1 a win, 0 a tie for first, -1 a loss. Failed or
    cancelled games are left out: they have no result to count.
    """

    path = job_folder(out, job) / "opponentcheck" / "games.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row["status"] == "completed"]


def read_failure(out: Path, job: Job) -> str:
    """Why a failed job failed: the error its failed.json records. The job's
    log.txt has the full output of its commands.

    Example: "opponentcheck exited with 1" for a bot job whose decode check
    found a mismatch.
    """

    try:
        return json.loads((job_folder(out, job) / "failed.json").read_text())["error"]
    except (OSError, ValueError, KeyError):
        return "no reason recorded"


def check_bot_games_complete(job_id: str, games: Sequence[Mapping[str, Any]], maps: Sequence[int]) -> str | None:
    """Why a bot job's completed games aren't a full set, or None when they
    are: every map seed of the session from every one of BOT_SEATS seats,
    exactly once.

    A comparison pairs two jobs game by game. If either job lacks some
    games, pairing would quietly compare only the games both have, so the
    comparison is refused instead. Example: with 2 maps, 8 games are
    expected; a job that completed 7 fails.
    """

    expected = sorted((seed, seat) for seed in range(*maps) for seat in range(BOT_SEATS))
    played = sorted((row["seed"], row["seat"]) for row in games)
    if played != expected:
        return (f"{job_id} completed {len(played)} games, not the {len(expected)} expected "
                f"(every map from each of {BOT_SEATS} seats once)")
    return None


def summarize_bot_job(out: Path, job: Job) -> dict[str, Any]:
    """One bot job's row: results, counts and calibration.

    Most numbers come straight from opponentcheck's summary.json. The aim
    counts are there only for a heuristic-v2 tester; .get() gives None for
    any other, which the table shows as "undefined".
    """

    summary = json.loads((job_folder(out, job) / "opponentcheck" / "summary.json").read_text())
    tester = summary["testers"][0]
    # Calibration runs only when the tester is the opponent itself (the
    # "normal" tester): all four seats then play one game, so moving the
    # "tested" label between seats must not change it. Zero seeds checked
    # means it didn't apply to this tester.
    calibration = summary["calibration"]
    if not calibration["seeds"]:
        calibration_text = "not run"
    else:
        calibration_text = "passed" if calibration["passed"] else "FAILED"
    wins = [
        {"seed": row["seed"], "seat": row["seat"]}
        for row in sorted(read_bot_games(out, job), key=lambda r: (r["seed"], r["seat"]))
        if row["result"] > 0
    ]
    return {
        "tester": job.label,
        "spec": job.settings["tester"],
        # False if any game failed or was cancelled: such a run gives no
        # trustworthy answer.
        "complete": summary["run"]["status"] == "complete",
        "completed": tester["completed"],
        "planned": tester["planned"],
        "wins": tester["wins"],
        "win_rate": tester["win_rate"] if tester["completed"] else None,
        "eliminated": tester["eliminated"],
        # The middle game length: half the games ended sooner, half later.
        "median_finish_turn": tester["length_p50"],
        "aim_failure_rate": tester.get("aim_failure_rate"),
        "fallback_rate": tester.get("fallback_rate"),
        "random_moves_per_game": tester.get("random_moves_per_game"),
        "decode_checked_launches": tester.get("decode_checked_launches"),
        "check_decode": job.settings["check_decode"],
        "calibration": calibration_text,
        # Kept only for decode-checked testers: proof that a learner-move bot
        # can win on these maps, replayable from the map seed and seat.
        "win_games": wins if job.settings["check_decode"] else [],
    }


def read_diagnosis_wins(report_path: Path) -> dict[str, int] | None:
    """A diagnosis's wins and games, from its report.json; None when the
    diagnosis is missing. Example: {"wins": 3, "games": 400}.

    Every diagnosis is read through this one function, so a missing report
    is handled the same way everywhere: a training run that stopped early,
    say, is reported as missing, never a crash.
    """

    if not report_path.is_file():
        return None
    games = json.loads(report_path.read_text())["games"]["measurements"]
    return {"wins": games["wins"], "games": games["games"]}


def read_diagnosis_win_rate(report_path: Path) -> float | None:
    """A diagnosis's win rate, wins ÷ games; None when it is missing.
    Example: 3 wins of 400 games gives 0.0075."""

    result = read_diagnosis_wins(report_path)
    return None if result is None else result["wins"] / result["games"]


def check_diagnosis_valid(report_folder: Path, opponent: str, maps: Sequence[int]) -> str | None:
    """Why one diagnosis can't be used as evidence, or None when it can.

    A win rate only measures the model if its games are a clean, complete
    record of the games that were asked for. So the report must pass
    compare_diagnoses.py's require_valid (every map seed played from every
    seat exactly once, and no learner action rejected, since a rejected
    action is a turn the model didn't really play), and it must have played
    the session's maps against the job's own opponent.

    ``maps`` is the session's [first seed, one past the last]. Example: a
    report that played seeds 1,000,000 to 1,000,099 in a session of maps
    [1,000,000, 1,000,100) passes that part; one that played
    [1,010,000, 1,010,100) doesn't.
    """

    try:
        report = load_report(report_folder)
        require_valid(report, "the")
    except ValueError as error:
        return str(error)
    except KeyError as error:
        # A report from an older or broken diagnose, missing a field the
        # check reads.
        return f"the report is missing the field {error}"
    if list(report["games"]["seeds"]) != list(maps):
        return f"it played seeds {report['games']['seeds']}, not the session's maps {list(maps)}"
    if report["games"]["opponent"] != opponent:
        return f"it played {report['games']['opponent']!r}, not {opponent!r}"
    return None


def read_training_win_rates(metrics_path: Path) -> list[tuple[int, int, int]]:
    """(update, wins, games) for every update in a run's metrics.csv.

    metrics.csv has one row per training update; its wins, losses and draws
    count the training games that finished during that update. Example row:
    update 5 with 3 wins, 9 losses and 0 draws gives (5, 3, 12).
    """

    with metrics_path.open(newline="") as handle:
        return [
            (int(row["update"]), int(row["wins"]), int(row["wins"]) + int(row["losses"]) + int(row["draws"]))
            for row in csv.DictReader(handle)
        ]


# ---------------------------------------------------------------------------
# Comparing two bot jobs map by map.
# ---------------------------------------------------------------------------


def compare_jobs(baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Candidate minus baseline win rate, map by map, with a 95% range.

    The four seats of one map share a board, so an easy or hard map moves all
    four results together. Games are paired by (map seed, seat), each map's
    paired seats are averaged, and the range comes from resampling whole maps
    (bootstrap_range), as compare_diagnoses.py does.

    Example: on one map the baseline wins 1 of 4 seats (0.25) and the
    candidate 0 of 4 (0.0), so that map's difference is -0.25.

    How the 95% range works (a "bootstrap"): pretend the maps played are all
    the maps there are, and draw a new set of the same size from them at
    random, allowing repeats. With 3 maps whose differences are -0.25, 0 and
    0, one draw might be (-0.25, -0.25, 0), mean -0.17, another (0, 0, 0),
    mean 0. After many draws, the middle 95% of those means is the range. A
    narrow range means the answer barely depends on which maps were played.

    Returns the number of maps compared, both win rates, their difference,
    the range (None with fewer than 2 maps), and "strength kept": the
    candidate's win rate as a share of the baseline's (None if the baseline
    never won, since dividing by 0 means nothing).
    """

    # Each baseline game's outcome by (seed, seat): 1.0 for a win, 0.0 for
    # anything else, so averaging gives a win rate.
    won = {(row["seed"], row["seat"]): float(row["result"] > 0) for row in baseline}
    # Pair each candidate game with the baseline game at the same map and
    # seat, grouped by map. analyze() checks first that both jobs hold the
    # full set of games (check_bot_games_complete), so every game pairs;
    # the "if key in won" only guards a direct caller.
    by_map: dict[int, list[tuple[float, float]]] = {}
    for row in candidate:
        key = (row["seed"], row["seat"])
        if key in won:
            by_map.setdefault(row["seed"], []).append((won[key], float(row["result"] > 0)))
    seeds = sorted(by_map)
    # Each map's win rate for each side: the average over its paired seats.
    base = np.array([np.mean([pair[0] for pair in by_map[seed]]) for seed in seeds])
    cand = np.array([np.mean([pair[1] for pair in by_map[seed]]) for seed in seeds])
    result: dict[str, Any] = {"maps": len(seeds), "baseline_rate": None, "candidate_rate": None,
                              "difference": None, "range": None, "strength_kept": None}
    if not seeds:
        return result
    # One difference per map; their average is the overall difference.
    differences = cand - base
    result.update(
        baseline_rate=float(base.mean()),
        candidate_rate=float(cand.mean()),
        difference=float(differences.mean()),
    )
    # A range needs at least two maps to resample from.
    if len(seeds) >= 2:
        result["range"] = list(bootstrap_range(differences, DEFAULT_RESAMPLES,
                                               np.random.default_rng(BOOTSTRAP_SEED)))
    if base.mean() > 0:
        result["strength_kept"] = float(cand.mean() / base.mean())
    return result


def answer_restricted_bot(compare: Mapping[str, Any]) -> str:
    """Experiment 1's answer from the main comparison: the fully limited bot
    (candidate) minus normal v2 (baseline).

    The question: is the learner losing because of the moves it's allowed to
    make? The fully limited bot is v2 restricted to the learner's moves, so:

    - Its whole 95% range is above KEPT_ABOVE (it loses at most a little):
      the learner's moves aren't the obstacle.
    - It is LOST_AT_MOST behind or worse, and the whole range is below 0
      (it clearly loses a lot): the learner's moves are an obstacle.
    - Anything else: unclear; check on fresh maps.

    Example: if LOST_AT_MOST were -10 points, a difference of -12 points
    with a range of -16 to -8 would be an obstacle: far enough behind, with
    the whole range below 0.

    It is only asked once both jobs are valid evidence with a full set of
    games; otherwise analyze() answers "Invalid evidence" with the reason
    (a failed job, such as a decode mismatch, a failed or cancelled game,
    a failed calibration, or missing games).
    """

    low_high = compare["range"]
    if low_high is None:
        return "Unclear: too few maps for a range."
    low, high = low_high
    if low > KEPT_ABOVE:
        return ("The learner's moves aren't the obstacle, at least for this bot: the learner's problem is "
                "finding good moves.")
    if compare["difference"] <= LOST_AT_MOST and high < 0:
        return ("The learner's moves are an obstacle. The single-limit comparisons show which; if it's the "
                "ship count, letting the learner choose fleet size comes first.")
    return "Unclear. Confirm on fresh maps before deciding."


# ---------------------------------------------------------------------------
# Building the whole analysis.
# ---------------------------------------------------------------------------


def analyze(out: Path) -> dict[str, Any]:
    """Every experiment's tables and answers, as one JSON-ready dict.

    The stages are: rebuild the plan from session.json; read every finished
    bot job; then build each section in turn: bot tables, comparisons, dials,
    diagnose tables, training tables. Only jobs that are done are read; every
    other job is reported as not done.
    """

    # Stage 1: the plan, rebuilt from the experiment file's text saved in
    # session.json (the same text the session ran), and each job's state.
    session = json.loads((out / "session.json").read_text())
    plan: Plan = parse_plan(session["experiment_text"])
    states = {job.id: read_job_state(out, job) for job in plan.jobs}
    jobs = {job.id: job for job in plan.jobs}
    done = {job_id for job_id, state in states.items() if state == "done"}

    analysis: dict[str, Any] = {
        "session": plan.session.name,
        # The maps played, as [first seed, one past the last].
        "maps": [plan.session.map_start, plan.session.map_start + plan.session.map_count],
        "runs": session.get("runs", []),
        # The code and inputs every job ran with, saved by the session's
        # first run; a resume refuses to run if they changed. None for a
        # session made before fingerprints were recorded.
        "fingerprint": session.get("fingerprint"),
        "jobs": states,
        "botcheck": {},
        "compare": [],
        "dial": [],
        "diagnose": {},
        "train": {},
    }

    # Stage 2: one row per finished bot job, then each [[botcheck]] section's
    # table, which also lists its jobs that aren't done.
    bot_rows: dict[str, dict[str, Any]] = {}
    for job in plan.jobs:
        if job.kind == "botcheck" and job.id in done:
            bot_rows[job.id] = summarize_bot_job(out, job)
    # dict.fromkeys lists each experiment name once, in file order.
    for experiment in dict.fromkeys(job.experiment for job in plan.jobs if job.kind == "botcheck"):
        group = [job for job in plan.jobs if job.experiment == experiment]
        analysis["botcheck"][experiment] = {
            "testers": [bot_rows[job.id] for job in group if job.id in bot_rows],
            # A failed job, with why, kept apart from one not run yet.
            "failed": {job.id: read_failure(out, job) for job in group if states[job.id] == "failed"},
            "not_done": [job.id for job in group if job.id not in bot_rows and states[job.id] != "failed"],
        }

    def evidence_problems(job_ids: Sequence[str]) -> list[str]:
        """Why these bot jobs can't support a conclusion: a failed job (a
        decode mismatch makes opponentcheck fail, for example), a run with a
        failed or cancelled game, or a failed calibration. Empty when they
        all can; a job that just isn't done yet is not a problem here."""

        problems = [f"{job_id} failed: {read_failure(out, jobs[job_id])}"
                    for job_id in job_ids if states[job_id] == "failed"]
        rows = [bot_rows[job_id] for job_id in job_ids if job_id in bot_rows]
        problems += [f"{row['tester']} is incomplete" for row in rows if not row["complete"]]
        problems += [f"{row['tester']} failed calibration" for row in rows if row["calibration"] == "FAILED"]
        return problems

    # Stage 3: each [[compare]]. Both jobs must be valid evidence (see
    # evidence_problems), done, and hold the full set of games, every map
    # from every seat (check_bot_games_complete), so the two are compared on
    # exactly the same games. With rule "restricted-bot", experiment 1's
    # answer is added. Any problem makes the answer "Invalid evidence".
    for compare in plan.compares:
        entry: dict[str, Any] = {**compare, "result": None, "answer": None}
        pair = (compare["baseline"], compare["candidate"])
        problems = evidence_problems(pair)
        missing = [job_id for job_id in pair if job_id not in done and states[job_id] != "failed"]
        if not problems and not missing:
            games = {job_id: read_bot_games(out, jobs[job_id]) for job_id in pair}
            problems = [problem for job_id in pair
                        if (problem := check_bot_games_complete(job_id, games[job_id], analysis["maps"]))]
        if problems:
            entry["answer"] = f"Invalid evidence: {'; '.join(problems)}."
        elif missing:
            entry["answer"] = f"Not ready: {', '.join(missing)} not done." if compare["rule"] else None
        else:
            entry["result"] = compare_jobs(games[compare["baseline"]], games[compare["candidate"]])
            if compare["rule"] == "restricted-bot":
                entry["answer"] = answer_restricted_bot(entry["result"])
        analysis["compare"].append(entry)

    # Stage 4: each [[dial]] (experiment 2). The dial is usable if the win
    # rate falls at every step of its order. Example: 25%, 18%, 9%, 4% is
    # usable; 25%, 18%, 19%, 4% isn't, because the third step rises. A step
    # that isn't valid evidence (see evidence_problems) makes the answer
    # "Invalid evidence": rates alone can look fine after a runner fault.
    for dial in plan.dials:
        rates = [bot_rows[job_id]["win_rate"] if job_id in bot_rows else None for job_id in dial["order"]]
        problems = evidence_problems(dial["order"])
        if problems:
            answer = f"Invalid evidence: {'; '.join(problems)}."
        elif any(rate is None for rate in rates):
            answer = "Not ready: some testers aren't done."
        else:
            # Every step whose rate is not strictly below the one before it.
            rises = [dial["order"][i + 1] for i in range(len(rates) - 1) if not rates[i + 1] < rates[i]]
            answer = ("Usable: the win rate falls at every step." if not rises
                      else f"Not usable as it stands: the win rate doesn't fall at {', '.join(rises)}. "
                           "The epsilon_* settings are the next thing to vary.")
        analysis["dial"].append({**dial, "win_rates": rates, "answer": answer})

    # Stage 5: each [[diagnose]] section (experiment 3): one row per job, and
    # the band answer when the section sets a band. A finished job's report
    # is checked before its win rate is used: an invalid one gets no rate,
    # and its reason in "invalid".
    for experiment in dict.fromkeys(job.experiment for job in plan.jobs if job.kind == "diagnose"):
        group = [job for job in plan.jobs if job.experiment == experiment]
        rows = []
        for job in group:
            rate = invalid = None
            if job.id in done:
                folder = job_folder(out, job) / "diagnosis"
                invalid = check_diagnosis_valid(folder, job.settings["opponent"], analysis["maps"])
                rate = None if invalid else read_diagnosis_win_rate(folder / "report.json")
            rows.append({"job": job.id, "opponent": job.settings["opponent_label"],
                         "run": Path(job.settings["run_dir"]).name, "update": job.settings["update"],
                         "win_rate": rate, "invalid": invalid})
        section: dict[str, Any] = {"rows": rows, "answer": None}
        options = plan.answers.get(experiment)
        if options:
            section["answer"] = answer_band(rows, options["band"], options["band_min_runs"])
        analysis["diagnose"][experiment] = section

    # Stage 6: each [[train]] section (experiment 4): one entry per run, and
    # the progress answer.
    for experiment in dict.fromkeys(job.experiment for job in plan.jobs if job.kind == "train"):
        group = [job for job in plan.jobs if job.experiment == experiment]
        options = plan.answers[experiment]
        runs = [summarize_train_job(out, job, options["progress_against"]) if job.id in done
                else {"job": job.id, "done": False, "opponent": job.settings["opponent_label"]} for job in group]
        analysis["train"][experiment] = {
            "runs": runs,
            "answer": answer_training(runs, options["progress_min_runs"], options["progress_min_extra_wins"],
                                      options["progress_against"]),
        }
    return analysis


def answer_band(rows: Sequence[Mapping[str, Any]], band: Sequence[float], min_runs: int) -> str:
    """Experiment 3: an opponent is found if, at one update, at least
    min_runs runs' models win inside the band against it.

    Training needs an opponent it sometimes beats: never winning teaches
    nothing about winning, and almost always winning leaves little to learn.
    The band is that "sometimes".

    Each run votes once per update: rows are grouped by (opponent, update),
    and a run has exactly one row in each group. Counting every row of an
    opponent instead would let one run with two in-band checkpoints count
    as two models.

    Example, band 10%-40% and min_runs 2: at update 153, three runs win
    12%, 35% and 50% against one opponent. Two of them are inside the band,
    so it is found. When none is found, each (opponent, update) is said to
    be "below" or "above" the band (by its runs' average), which shows which
    way to move next time.

    A row whose report failed check_diagnosis_valid makes the answer
    "Invalid evidence", with its reason: no verdict is drawn from games the
    model didn't cleanly play.
    """

    invalid = [f"{row['job']} ({row['invalid']})" for row in rows if row["invalid"]]
    if invalid:
        return f"Invalid evidence: {'; '.join(invalid)}."
    if any(row["win_rate"] is None for row in rows):
        return "Not ready: some diagnoses aren't done."
    low, high = band
    found = []
    sides = []
    for opponent, update in dict.fromkeys((row["opponent"], row["update"]) for row in rows):
        rates = [row["win_rate"] for row in rows if (row["opponent"], row["update"]) == (opponent, update)]
        # True counts as 1 in a sum, so this counts the runs inside the band.
        inside = sum(low <= rate <= high for rate in rates)
        if inside >= min_runs:
            found.append(f"{opponent} at update {update} ({inside} of {len(rates)} runs)")
        else:
            side = "below" if np.mean(rates) < low else "above" if np.mean(rates) > high else "around"
            sides.append(f"{opponent} at update {update}: {side}")
    if found:
        return f"Found: {', '.join(found)} gives {format_percent(low)}-{format_percent(high)}, an opponent with enough wins to learn from."
    return f"Not found. Each opponent's side of the band: {'; '.join(sides)}."


def summarize_train_job(out: Path, job: Job, against: str | None) -> dict[str, Any]:
    """One train job's checkpoints: training win rate since the previous
    checkpoint next to each diagnosis win rate, and the training trend.

    The two win rates measure different things, which is why both are shown:
    the training one is against the (weakened) training opponent, with moves
    sampled at random from the model's choices; the diagnosis one is against
    the diagnose_against opponent (full v2), with the model's top move.

    Example: checkpoints at updates 30 and 60. The training win rate at 60
    counts the training games of updates 31 to 60 together: their total wins
    ÷ their total games.

    ``against`` is the section's progress_against label, the opponent the
    progress measure uses (see measure_training_progress).
    """

    folder = job_folder(out, job)
    history = read_training_win_rates(folder / "run" / "metrics.csv")
    checkpoints = []
    previous = 0
    # Checkpoint names are zero-padded (update-000030.pt), so sorting them as
    # text puts them in update order.
    for path in sorted((folder / "run" / "checkpoints").glob("update-*.pt")):
        # "update-000030" → 30.
        update = int(path.stem.split("-")[1])
        window = [(wins, games) for u, wins, games in history if previous < u <= update]
        played = sum(games for _wins, games in window)
        diagnoses = {
            label: read_diagnosis_win_rate(folder / "diagnoses" / f"{path.stem}__{label}" / "report.json")
            for label in job.settings["diagnose_against"]
        }
        checkpoints.append({
            "update": update,
            "training_win_rate": sum(wins for wins, _games in window) / played if played else None,
            "diagnosis_win_rates": diagnoses,
        })
        previous = update
    # Rising: the last TREND_UPDATES updates win more often than the first.
    # A short run uses half its updates at each end instead (so the two ends
    # never overlap).
    span = min(TREND_UPDATES, len(history) // 2)

    def window_rate(window: Sequence[tuple[int, int, int]]) -> float | None:
        # None when no training game finished in the window: there is no
        # win rate, and treating it as 0% could make a rise out of nothing.
        games = sum(g for _u, _w, g in window)
        return sum(w for _u, w, _g in window) / games if games else None

    first = window_rate(history[:span]) if span else None
    last = window_rate(history[-span:]) if span else None
    return {
        "job": job.id,
        "done": True,
        "opponent": job.settings["opponent_label"],
        # The starting run's name, which carries its training seed.
        "start": Path(job.settings["init_from"]).parent.parent.name,
        "checkpoints": checkpoints,
        "first_training_win_rate": first,
        "last_training_win_rate": last,
        # None (undefined) when either end has no rate.
        "training_rose": None if first is None or last is None else last > first,
        "progress": measure_training_progress(folder, job, against),
    }


def measure_training_progress(folder: Path, job: Job, against: str | None) -> dict[str, Any]:
    """The progress measure of one train job: its final checkpoint's wins
    minus its starting checkpoint's wins, against the diagnose_against
    opponent labelled ``against`` (the section's progress_against; full v2
    in Stage 1). None means no target was named, so nothing is measured.

    The difference only means "what training added" if both diagnoses are
    clean and played the same games, so both are checked first (see
    check_diagnoses_comparable). The final checkpoint is the one at max_updates, the
    update the experiment file asked for.

    extra_wins is None when there is nothing to subtract:
    - either diagnosis is missing (training stopped early and never saved
      that checkpoint, say): the run is pending;
    - the two fail the checks: ``invalid`` says why.

    Example: the start wins 1 of 400 games and the final checkpoint 6 of
    the same 400, so extra_wins is 5.
    """

    final_update = job.settings["max_updates"]
    if against is None:
        return {"against": None, "final_update": final_update, "start": None, "final": None,
                "invalid": None, "extra_wins": None}
    spec = job.settings["diagnose_against"][against]
    start_folder = folder / "diagnoses" / f"start__{against}"
    final_folder = folder / "diagnoses" / f"update-{final_update:06d}__{against}"
    start = read_diagnosis_wins(start_folder / "report.json")
    final = read_diagnosis_wins(final_folder / "report.json")
    invalid = None
    if start is not None and final is not None:
        invalid = check_diagnoses_comparable(start_folder, final_folder, spec)
    usable = start is not None and final is not None and invalid is None
    return {
        "against": against,
        "final_update": final_update,
        "start": start,
        "final": final,
        "invalid": invalid,
        "extra_wins": final["wins"] - start["wins"] if usable else None,
    }


def check_diagnoses_comparable(start_folder: Path, final_folder: Path, opponent: str) -> str | None:
    """Why two diagnoses can't be subtracted, or None when they can.

    It reuses compare_diagnoses.py's checks, which every comparison of two
    diagnoses must pass:
    - each report is clean on its own (require_valid): every map seed
      played from every seat exactly once, and no learner action rejected,
      since a rejected action is a turn the model didn't really play;
    - the two played the same games the same way (require_comparable): the
      same opponent, map seeds, map-and-seat assignments, engine and
      evaluation settings, so the model is the only difference.
    Both must also have played the diagnose_against opponent the file asked
    for, not some other one.

    Example: a start that played seeds 1,000,000 to 1,000,099 and a final
    that played 1,000,100 to 1,000,199 fail, because the two played
    different maps.
    """

    try:
        start, final = load_report(start_folder), load_report(final_folder)
        require_valid(start, "start")
        require_valid(final, "final")
        require_comparable(start, final)
    except ValueError as error:
        return str(error)
    except KeyError as error:
        # A report from an older or broken diagnose, missing a field the
        # checks read.
        return f"a report is missing the field {error}"
    for name, report in (("start", start), ("final", final)):
        if report["games"]["opponent"] != opponent:
            return f"the {name} report played {report['games']['opponent']!r}, not {opponent!r}"
    return None


def answer_training(runs: Sequence[Mapping[str, Any]], min_runs: int, min_extra_wins: int | None,
                    against: str | None) -> str:
    """Experiment 4's answer. Training made progress if, for some training
    opponent, the final checkpoint won at least min_extra_wins more games
    than its starting checkpoint against the ``against`` opponent (the
    section's progress_against), in at least min_runs runs.

    The answers, checked in this order:

    1. **Unavailable**: min_extra_wins or against is None, because the
       session was made before progress_min_extra_wins or progress_against
       existed. There is no threshold or target to judge by, so no verdict;
       its tables are still shown.
    2. **Invalid evidence**: some run's start and final diagnoses fail the
       checks in check_diagnoses_comparable. Waiting won't fix that, so it comes
       before pending, and no verdict is given.
    3. **Pending**: some run isn't done, or has no final-versus-start result
       (see measure_training_progress). No verdict is given on part of the evidence.
    4. **Observed progress**: for one training opponent, at least min_runs
       runs reached min_extra_wins extra wins. It is called "observed"
       because the threshold is a screening rule on one set of maps, not a
       statistical test. Training on a weakened opponent is then a route
       forward.
    5. **Partial**: no such progress, but the training win rate rose in at
       least min_runs runs: the model improves against the weakened
       opponent, yet that doesn't carry over to the full one.
    6. **Not working**: neither.

    Partial and not working end with each training opponent's counts, so
    the answer says how many runs did gain wins, not just that too few did.
    A run whose training rise is undefined (a window with no finished
    games) doesn't count as rising.

    Example, min_runs 2 and min_extra_wins 4: three runs against one
    training opponent have 5, 4 and 0 extra wins. Two reach 4, so the
    answer is observed progress.
    """

    if min_extra_wins is None:
        return ("Unavailable: this session was made before progress_min_extra_wins existed, so there is no "
                "progress threshold to judge its training by.")
    if against is None:
        return ("Unavailable: diagnose_against has several labels and no progress_against names the one to "
                "judge training against.")
    invalid = [f"{run['job']} ({run['progress']['invalid']})" for run in runs
               if run["done"] and run["progress"]["invalid"]]
    if invalid:
        return f"Invalid evidence: the start and final diagnoses can't be compared for {'; '.join(invalid)}."
    pending = [run["job"] for run in runs if not run["done"] or run["progress"]["extra_wins"] is None]
    if pending:
        return f"Pending: no final-versus-start result yet for {', '.join(pending)}."
    # Each training opponent's counts: runs that gained enough wins, and runs
    # whose training win rate rose. True counts as 1 in a sum; "is True"
    # leaves out a rise that is undefined (None).
    counts = {}
    for opponent in dict.fromkeys(run["opponent"] for run in runs):
        group = [run for run in runs if run["opponent"] == opponent]
        counts[opponent] = {
            "runs": len(group),
            "progressed": sum(run["progress"]["extra_wins"] >= min_extra_wins for run in group),
            "rose": sum(run["training_rose"] is True for run in group),
        }
    for opponent, count in counts.items():
        if count["progressed"] >= min_runs:
            return (f"Observed progress: after training against {opponent}, the final checkpoint won at least "
                    f"{min_extra_wins} more games against {against} than its starting checkpoint in "
                    f"{count['progressed']} of {count['runs']} runs. The next step lowers epsilon towards 0.")
    # Example: "v2-eps0.75: 1 of 3 runs gained 4 wins against v2, training
    # rose in 2 of 3".
    summary = "; ".join(
        f"{opponent}: {count['progressed']} of {count['runs']} runs gained {min_extra_wins} wins against "
        f"{against}, training rose in {count['rose']} of {count['runs']}"
        for opponent, count in counts.items()
    )
    for opponent, count in counts.items():
        if count["rose"] >= min_runs:
            return (f"Partial: training wins rose against {opponent}, but fewer than {min_runs} runs gained "
                    f"{min_extra_wins} wins against {against} ({summary}). Use experiment 1's answer to choose "
                    "between widening the learner's moves and a smaller epsilon step.")
    return (f"Not working: training wins didn't rise either ({summary}). Experiment 1 says whether the "
            "learner's moves or the training method are to blame.")


# ---------------------------------------------------------------------------
# Writing analysis.md.
#
# The Markdown file shows the same numbers as analysis.json, as tables, in
# the order: bot tables, comparisons, dials, diagnose tables, training
# tables. A Markdown table is written as lines like "| a | b |", with a
# "|---|---|" line under the heading row.
# ---------------------------------------------------------------------------


def render_markdown(analysis: Mapping[str, Any]) -> str:
    """analysis.md's text, built from analyze()'s result."""

    # The heading: the maps played, how many jobs are in each state, and the
    # code that ran. The code comes from the session's fingerprint, saved
    # once by its first run, so a later resume can't relabel the results.
    lines = [f"# Analysis: {analysis['session']}", ""]
    start, end = analysis["maps"]
    states = list(analysis["jobs"].values())
    lines.append(f"Maps {start:,} to {end - 1:,}, every seat. Jobs: "
                 + ", ".join(f"{states.count(s)} {s}" for s in sorted(set(states))) + ".")
    fingerprint = analysis["fingerprint"]
    if fingerprint is None:
        lines.append("Code: provenance unknown (this session was made before fingerprints were recorded).")
    else:
        code = fingerprint["code"]
        # The first 12 characters of the hash are enough to tell sessions
        # apart by eye; session.json keeps the whole hash.
        lines.append(f"Code: source `{code['source_sha256'][:12]}`, {code.get('go')}, Python {code.get('python')}, "
                     f"Torch {code.get('torch')}, NumPy {code.get('numpy')}; "
                     f"{len(fingerprint['inputs'])} input files hashed.")
    lines.append("")

    # One table per [[botcheck]] section, one row per tester. "off" in the
    # decode column means the check wasn't asked for that tester.
    for experiment, section in analysis["botcheck"].items():
        lines += [f"## {experiment}", "",
                  "| Tester | Games | Wins | Win rate | Eliminated | Median finish | Aim failure rate | "
                  "Fallback share | Random moves/game | Decode-checked launches | Calibration |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in section["testers"]:
            checked = row["decode_checked_launches"] if row["check_decode"] else "off"
            random_moves = row["random_moves_per_game"]
            random_text = "undefined" if random_moves is None else f"{random_moves:.1f}"
            lines.append(
                f"| {row['tester']} | {row['completed']}/{row['planned']} | {row['wins']} | {format_percent(row['win_rate'])} "
                f"| {row['eliminated']} | {row['median_finish_turn']} | {format_percent(row['aim_failure_rate'])} "
                f"| {format_percent(row['fallback_rate'])} | {random_text} | {checked} | {row['calibration']} |")
        for job_id, error in section["failed"].items():
            lines.append(f"| {job_id.split('/', 1)[1]} | failed: {error} | | | | | | | | | |")
        for job_id in section["not_done"]:
            lines.append(f"| {job_id.split('/', 1)[1]} | not done | | | | | | | | | |")
        # The wins of decode-checked testers, by map seed and seat.
        for row in section["testers"]:
            if row["win_games"]:
                games = ", ".join(f"seed {g['seed']} seat {g['seat']}" for g in row["win_games"])
                lines.append(f"\n{row['tester']} won: {games}.")
        lines.append("")

    # All comparisons in one table, then each answer under it.
    if analysis["compare"]:
        lines += ["## Comparisons", "",
                  "Candidate minus baseline win rate, map by map; the range is the middle 95% over resampled maps.", "",
                  "| Question | Baseline | Candidate | Difference | 95% range | Strength kept |",
                  "|---|---|---|---|---|---|"]
        for entry in analysis["compare"]:
            result = entry["result"]
            if result is None:
                # No result: either invalid evidence (its answer, below,
                # says why) or a job not done yet.
                state = "invalid" if (entry["answer"] or "").startswith("Invalid") else "not done"
                lines.append(f"| {entry['question']} | {state} | | | | |")
                continue
            span = result["range"]
            span_text = f"{format_points(span[0])} to {format_points(span[1])}" if span else "undefined"
            lines.append(
                f"| {entry['question']} | {format_percent(result['baseline_rate'])} | {format_percent(result['candidate_rate'])} "
                f"| {format_points(result['difference'])} | {span_text} | {format_percent(result['strength_kept'])} |")
        for entry in analysis["compare"]:
            if entry.get("answer"):
                lines += ["", f"**Answer ({entry['question']}):** {entry['answer']}"]
        lines.append("")

    # Each dial: its steps in order with their win rates, then the answer.
    for dial in analysis["dial"]:
        lines += [f"## Dial: {dial['name']}", "", "| Step | Win rate |", "|---|---|"]
        lines += [f"| {job_id} | {format_percent(rate)} |" for job_id, rate in zip(dial["order"], dial["win_rates"])]
        lines += ["", f"**Answer:** {dial['answer']}", ""]

    # Each [[diagnose]] section as a grid: one row per opponent, one column
    # per (run, update), so each opponent's results across models sit on one
    # line.
    for experiment, section in analysis["diagnose"].items():
        runs = list(dict.fromkeys((row["run"], row["update"]) for row in section["rows"]))
        lines += [f"## {experiment}", "",
                  "| Opponent | " + " | ".join(f"{run} u{update}" for run, update in runs) + " |",
                  "|---|" + "---|" * len(runs)]
        for opponent in dict.fromkeys(row["opponent"] for row in section["rows"]):
            # For each column, the one row with this opponent and that
            # (run, update).
            # An invalid report shows "invalid" rather than a win rate.
            cells = [next("invalid" if r["invalid"] else format_percent(r["win_rate"]) for r in section["rows"]
                          if r["opponent"] == opponent and (r["run"], r["update"]) == run) for run in runs]
            lines.append(f"| {opponent} | " + " | ".join(cells) + " |")
        if section["answer"]:
            lines += ["", f"**Answer:** {section['answer']}"]
        lines.append("")

    # Each [[train]] section: a short line and a checkpoint table per run,
    # then the section's answer.
    for experiment, section in analysis["train"].items():
        lines += [f"## {experiment}", ""]
        for run in section["runs"]:
            if not run["done"]:
                lines += [f"- {run['job']}: not done", ""]
                continue
            labels = list(run["checkpoints"][0]["diagnosis_win_rates"]) if run["checkpoints"] else []
            progress = run["progress"]
            if progress["against"] is None:
                versus = "Progress: not measured (no progress_against target)."
            elif progress["invalid"]:
                versus = f"Against {progress['against']}: invalid evidence ({progress['invalid']})."
            elif progress["extra_wins"] is None:
                versus = (f"Against {progress['against']}: pending (no start or update-{progress['final_update']} "
                          "diagnosis to compare).")
            else:
                start, final = progress["start"], progress["final"]
                versus = (f"Against {progress['against']}: start {start['wins']}/{start['games']} wins, "
                          f"update {progress['final_update']} {final['wins']}/{final['games']} "
                          f"({progress['extra_wins']:+d}).")
            lines += [f"**{run['opponent']} from {run['start']}:** training win rate "
                      f"{format_percent(run['first_training_win_rate'])} at the start, "
                      f"{format_percent(run['last_training_win_rate'])} at the end.", "", versus, "",
                      "| Checkpoint | Training win rate since the previous one | "
                      + " | ".join(f"Against {label}" for label in labels) + " |",
                      "|---|---|" + "---|" * len(labels)]
            for checkpoint in run["checkpoints"]:
                lines.append(f"| {checkpoint['update']} | {format_percent(checkpoint['training_win_rate'])} | "
                             + " | ".join(format_diagnosis_cell(checkpoint["diagnosis_win_rates"][label]) for label in labels)
                             + " |")
            lines.append("")
        lines += [f"**Answer:** {section['answer']}", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: one argument, the session folder.

    It can run while a session is still going: finished jobs are analysed
    and the rest reported as not done. Rerunning it rewrites only
    analysis.md and analysis.json.
    """

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="session folder written by run_experiments.py")
    arguments = parser.parse_args(argv)
    out = arguments.out.resolve()
    if not (out / "session.json").exists():
        parser.error(f"{out} has no session.json; run a session there first")
    analysis = analyze(out)
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (out / "analysis.md").write_text(render_markdown(analysis))
    print(f"wrote {out / 'analysis.md'} and analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
