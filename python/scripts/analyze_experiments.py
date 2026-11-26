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

from scripts.compare_diagnoses import DEFAULT_RESAMPLES, bootstrap_range
from scripts.run_experiments import Job, Plan, job_state, parse_plan

# Experiment 1's thresholds, as win-rate differences (a difference of -0.03
# would be -3 points):
# the fully limited bot "keeps its strength" if its whole 95% range is above
# KEPT_ABOVE, and "loses strength" if it is at most LOST_AT_MOST behind with
# the whole range below 0. See restricted_bot_answer.
KEPT_ABOVE = -0.05
LOST_AT_MOST = -0.10
# The bootstrap draws random samples. Starting its random numbers from a
# fixed seed makes it draw the same samples every time, so rerunning the
# analysis gives exactly the same 95% ranges.
BOOTSTRAP_SEED = 0
# Training's win rate is compared over this many updates at each end of a run
# to decide whether it rose. Averaging several updates smooths out one lucky
# or unlucky update.
TREND_UPDATES = 10


def percent(value: float | None) -> str:
    """A share as a percentage with one decimal: 0.125 → "12.5%". None means
    nothing was counted, written as "undefined" rather than a misleading 0%."""

    return "undefined" if value is None else f"{100 * value:.1f}%"


def points(value: float | None) -> str:
    """A win-rate difference in percentage points, always with its sign:
    -0.125 → "-12.5 pts", 0.05 → "+5.0 pts"."""

    return "undefined" if value is None else f"{100 * value:+.1f} pts"


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


def bot_games(out: Path, job: Job) -> list[dict[str, Any]]:
    """The completed games of a bot job, one row per (map seed, seat).

    games.jsonl holds one JSON object per line, one per game. ``result`` is
    the tester's reward: 1 a win, 0 a tie for first, -1 a loss. Failed or
    cancelled games are left out: they have no result to count.
    """

    path = job_folder(out, job) / "opponentcheck" / "games.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row["status"] == "completed"]


def bot_summary(out: Path, job: Job) -> dict[str, Any]:
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
        for row in sorted(bot_games(out, job), key=lambda r: (r["seed"], r["seat"]))
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


def diagnosis_win_rate(report_path: Path) -> float:
    """A diagnosis's win rate, from its report.json: wins ÷ games."""

    games = json.loads(report_path.read_text())["games"]["measurements"]
    return games["wins"] / games["games"]


def diagnosis_wins(report_path: Path) -> dict[str, int] | None:
    """A diagnosis's wins and games, from its report.json; None when the
    diagnosis is missing. Example: {"wins": 3, "games": 400}."""

    if not report_path.is_file():
        return None
    games = json.loads(report_path.read_text())["games"]["measurements"]
    return {"wins": games["wins"], "games": games["games"]}


def training_win_rates(metrics_path: Path) -> list[tuple[int, int, int]]:
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
    # seat, grouped by map. A game only one side completed is left out, so
    # both sides always describe the same games.
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


def restricted_bot_answer(compare: Mapping[str, Any], valid: bool, why_invalid: str) -> str:
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

    The answer is "invalid" instead when the run can't be trusted: a game
    failed (which includes a decode mismatch, since that fails its game), or
    the normal tester failed calibration.
    """

    if not valid:
        return f"Invalid: {why_invalid}. Fix the bot or runner first."
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
    states = {job.id: job_state(out, job) for job in plan.jobs}
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
            bot_rows[job.id] = bot_summary(out, job)
    # dict.fromkeys lists each experiment name once, in file order.
    for experiment in dict.fromkeys(job.experiment for job in plan.jobs if job.kind == "botcheck"):
        group = [job for job in plan.jobs if job.experiment == experiment]
        analysis["botcheck"][experiment] = {
            "testers": [bot_rows[job.id] for job in group if job.id in bot_rows],
            "not_done": [job.id for job in group if job.id not in bot_rows],
        }

    # Stage 3: each [[compare]]. Both jobs must be done; with rule
    # "restricted-bot", experiment 1's answer is added, or "invalid" if
    # either job is incomplete or failed calibration.
    for compare in plan.compares:
        entry: dict[str, Any] = {**compare}
        missing = [job_id for job_id in (compare["baseline"], compare["candidate"]) if job_id not in done]
        if missing:
            entry["result"] = None
            entry["answer"] = f"Not ready: {', '.join(missing)} not done." if compare["rule"] else None
        else:
            entry["result"] = compare_jobs(bot_games(out, jobs[compare["baseline"]]),
                                           bot_games(out, jobs[compare["candidate"]]))
            if compare["rule"] == "restricted-bot":
                pair = [bot_rows[compare["baseline"]], bot_rows[compare["candidate"]]]
                problems = [f"{row['tester']} is incomplete" for row in pair if not row["complete"]]
                problems += [f"{row['tester']} failed calibration" for row in pair if row["calibration"] == "FAILED"]
                entry["answer"] = restricted_bot_answer(entry["result"], not problems, "; ".join(problems))
        analysis["compare"].append(entry)

    # Stage 4: each [[dial]] (experiment 2). The dial is usable if the win
    # rate falls at every step of its order. Example: 25%, 18%, 9%, 4% is
    # usable; 25%, 18%, 19%, 4% isn't, because the third step rises.
    for dial in plan.dials:
        rates = [bot_rows[job_id]["win_rate"] if job_id in bot_rows else None for job_id in dial["order"]]
        if any(rate is None for rate in rates):
            answer = "Not ready: some testers aren't done."
        else:
            # Every step whose rate is not strictly below the one before it.
            rises = [dial["order"][i + 1] for i in range(len(rates) - 1) if not rates[i + 1] < rates[i]]
            answer = ("Usable: the win rate falls at every step." if not rises
                      else f"Not usable as it stands: the win rate doesn't fall at {', '.join(rises)}. "
                           "The epsilon_* settings are the next thing to vary.")
        analysis["dial"].append({**dial, "win_rates": rates, "answer": answer})

    # Stage 5: each [[diagnose]] section (experiment 3): one row per job, and
    # the band answer when the section sets a band.
    for experiment in dict.fromkeys(job.experiment for job in plan.jobs if job.kind == "diagnose"):
        group = [job for job in plan.jobs if job.experiment == experiment]
        rows = []
        for job in group:
            rate = (diagnosis_win_rate(job_folder(out, job) / "diagnosis" / "report.json")
                    if job.id in done else None)
            rows.append({"opponent": job.settings["opponent_label"], "run": Path(job.settings["run_dir"]).name,
                         "update": job.settings["update"], "win_rate": rate})
        section: dict[str, Any] = {"rows": rows, "answer": None}
        options = plan.answers.get(experiment)
        if options:
            section["answer"] = band_answer(rows, options["band"], options["band_min_runs"])
        analysis["diagnose"][experiment] = section

    # Stage 6: each [[train]] section (experiment 4): one entry per run, and
    # the progress answer.
    for experiment in dict.fromkeys(job.experiment for job in plan.jobs if job.kind == "train"):
        group = [job for job in plan.jobs if job.experiment == experiment]
        runs = [train_run(out, job) if job.id in done else {"job": job.id, "done": False,
                "opponent": job.settings["opponent_label"]} for job in group]
        analysis["train"][experiment] = {
            "runs": runs,
            "answer": train_answer(runs, plan.answers[experiment]["progress_min_runs"],
                                   plan.answers[experiment]["progress_min_extra_wins"]),
        }
    return analysis


def band_answer(rows: Sequence[Mapping[str, Any]], band: Sequence[float], min_runs: int) -> str:
    """Experiment 3: an opponent is found if at least min_runs models win
    inside the band against it.

    Training needs an opponent it sometimes beats: never winning teaches
    nothing about winning, and almost always winning leaves little to learn.
    The band is that "sometimes".

    Example, band 10%-40% and min_runs 2: three models win 12%, 35% and 50%
    against one opponent. Two of them are inside the band, so it is found.
    When none is found, each opponent is said to be "below" or "above" the
    band (by its models' average), which shows which way to move next time.
    """

    if any(row["win_rate"] is None for row in rows):
        return "Not ready: some diagnoses aren't done."
    low, high = band
    found = []
    sides = []
    for opponent in dict.fromkeys(row["opponent"] for row in rows):
        rates = [row["win_rate"] for row in rows if row["opponent"] == opponent]
        # True counts as 1 in a sum, so this counts the models inside the band.
        inside = sum(low <= rate <= high for rate in rates)
        if inside >= min_runs:
            found.append(f"{opponent} ({inside} of {len(rates)} models)")
        else:
            side = "below" if np.mean(rates) < low else "above" if np.mean(rates) > high else "around"
            sides.append(f"{opponent}: {side}")
    if found:
        return f"Found: {', '.join(found)} gives {percent(low)}-{percent(high)}, an opponent with enough wins to learn from."
    return f"Not found. Each opponent's side of the band: {'; '.join(sides)}."


def train_run(out: Path, job: Job) -> dict[str, Any]:
    """One train job's checkpoints: training win rate since the previous
    checkpoint next to each diagnosis win rate, and the training trend.

    The two win rates measure different things, which is why both are shown:
    the training one is against the (weakened) training opponent, with moves
    sampled at random from the model's choices; the diagnosis one is against
    the diagnose_against opponent (full v2), with the model's top move.

    Example: checkpoints at updates 30 and 60. The training win rate at 60
    counts the training games of updates 31 to 60 together: their total wins
    ÷ their total games.
    """

    folder = job_folder(out, job)
    history = training_win_rates(folder / "run" / "metrics.csv")
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
            label: diagnosis_win_rate(folder / "diagnoses" / f"{path.stem}__{label}" / "report.json")
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
    # never overlap). max(1, ...) only guards against dividing by 0.
    span = min(TREND_UPDATES, len(history) // 2)
    first = sum(w for _u, w, _g in history[:span]) / max(1, sum(g for _u, _w, g in history[:span]))
    last = sum(w for _u, w, _g in history[-span:]) / max(1, sum(g for _u, _w, g in history[-span:]))
    return {
        "job": job.id,
        "done": True,
        "opponent": job.settings["opponent_label"],
        # The starting run's name, which carries its training seed.
        "start": Path(job.settings["init_from"]).parent.parent.name,
        "checkpoints": checkpoints,
        "first_training_win_rate": first if span else None,
        "last_training_win_rate": last if span else None,
        "training_rose": bool(span) and last > first,
        "progress": train_progress(folder, job),
    }


def train_progress(folder: Path, job: Job) -> dict[str, Any]:
    """The progress measure of one train job: its final checkpoint's wins
    minus its starting checkpoint's wins, against the first
    diagnose_against opponent (full v2 in Stage 1).

    Both diagnoses play the same maps, seats and opponent, so the difference
    is what training added. The final checkpoint is the one at max_updates,
    the update the experiment file asked for. extra_wins is None (the run is
    pending) when either diagnosis is missing, for example because training
    stopped early and never saved that checkpoint.

    Example: the start wins 1 of 400 games and the final checkpoint 6 of
    400, so extra_wins is 5.
    """

    against = next(iter(job.settings["diagnose_against"]))
    final_update = job.settings["max_updates"]
    start = diagnosis_wins(folder / "diagnoses" / f"start__{against}" / "report.json")
    final = diagnosis_wins(folder / "diagnoses" / f"update-{final_update:06d}__{against}" / "report.json")
    # Wins only compare if both played the same number of games, which they
    # do when both ran in this session.
    comparable = start is not None and final is not None and start["games"] == final["games"]
    return {
        "against": against,
        "final_update": final_update,
        "start": start,
        "final": final,
        "extra_wins": final["wins"] - start["wins"] if comparable else None,
    }


def train_answer(runs: Sequence[Mapping[str, Any]], min_runs: int, min_extra_wins: int) -> str:
    """Experiment 4: progress if, for some training opponent, at least
    min_runs runs' final checkpoints win at least min_extra_wins more games
    than their starting checkpoints against the full opponent.

    The four answers, checked in this order:

    1. **Pending**: some run isn't done, or has no final-versus-start result
       (see train_progress). No verdict is given on part of the evidence.
    2. **Observed progress**: for one training opponent, at least min_runs
       runs reached min_extra_wins extra wins. It is called "observed"
       because the threshold is a screening rule on one set of maps, not a
       statistical test. Training on a weakened opponent is then a route
       forward.
    3. **Partial**: no such progress, but the training win rate rose in at
       least min_runs runs: the model improves against the weakened
       opponent, yet that doesn't carry over to the full one.
    4. **Not working**: neither.

    Example, min_runs 2 and min_extra_wins 4: three runs against one
    training opponent have 5, 4 and 0 extra wins. Two reach 4, so the
    answer is observed progress.
    """

    pending = [run["job"] for run in runs if not run["done"] or run["progress"]["extra_wins"] is None]
    if pending:
        return f"Pending: no final-versus-start result yet for {', '.join(pending)}."
    against = runs[0]["progress"]["against"]
    for opponent in dict.fromkeys(run["opponent"] for run in runs):
        group = [run for run in runs if run["opponent"] == opponent]
        # True counts as 1 in a sum, so this counts the runs that progressed.
        progressed = sum(run["progress"]["extra_wins"] >= min_extra_wins for run in group)
        if progressed >= min_runs:
            return (f"Observed progress: after training against {opponent}, the final checkpoint won at least "
                    f"{min_extra_wins} more games against {against} than its starting checkpoint in {progressed} "
                    f"of {len(group)} runs. The next step lowers epsilon towards 0.")
    for opponent in dict.fromkeys(run["opponent"] for run in runs):
        group = [run for run in runs if run["opponent"] == opponent]
        if sum(run["training_rose"] for run in group) >= min_runs:
            return (f"Partial: training wins rose against {opponent}, but fewer than {min_runs} runs gained "
                    f"{min_extra_wins} wins against {against}. Use experiment 1's answer to choose between "
                    "widening the learner's moves and a smaller epsilon step.")
    return "Not working: training wins didn't rise either. Experiment 1 says whether the learner's moves or the training method are to blame."


# ---------------------------------------------------------------------------
# Writing analysis.md.
#
# The Markdown file shows the same numbers as analysis.json, as tables, in
# the order: bot tables, comparisons, dials, diagnose tables, training
# tables. A Markdown table is written as lines like "| a | b |", with a
# "|---|---|" line under the heading row.
# ---------------------------------------------------------------------------


def markdown(analysis: Mapping[str, Any]) -> str:
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
                f"| {row['tester']} | {row['completed']}/{row['planned']} | {row['wins']} | {percent(row['win_rate'])} "
                f"| {row['eliminated']} | {row['median_finish_turn']} | {percent(row['aim_failure_rate'])} "
                f"| {percent(row['fallback_rate'])} | {random_text} | {checked} | {row['calibration']} |")
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
                lines.append(f"| {entry['question']} | not done | | | | |")
                continue
            span = result["range"]
            span_text = f"{points(span[0])} to {points(span[1])}" if span else "undefined"
            lines.append(
                f"| {entry['question']} | {percent(result['baseline_rate'])} | {percent(result['candidate_rate'])} "
                f"| {points(result['difference'])} | {span_text} | {percent(result['strength_kept'])} |")
        for entry in analysis["compare"]:
            if entry.get("answer"):
                lines += ["", f"**Answer ({entry['question']}):** {entry['answer']}"]
        lines.append("")

    # Each dial: its steps in order with their win rates, then the answer.
    for dial in analysis["dial"]:
        lines += [f"## Dial: {dial['name']}", "", "| Step | Win rate |", "|---|---|"]
        lines += [f"| {job_id} | {percent(rate)} |" for job_id, rate in zip(dial["order"], dial["win_rates"])]
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
            cells = [next(percent(r["win_rate"]) for r in section["rows"]
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
            if progress["extra_wins"] is None:
                versus = (f"Against {progress['against']}: pending (no start or update-{progress['final_update']} "
                          "diagnosis to compare).")
            else:
                start, final = progress["start"], progress["final"]
                versus = (f"Against {progress['against']}: start {start['wins']}/{start['games']} wins, "
                          f"update {progress['final_update']} {final['wins']}/{final['games']} "
                          f"({progress['extra_wins']:+d}).")
            lines += [f"**{run['opponent']} from {run['start']}:** training win rate "
                      f"{percent(run['first_training_win_rate'])} at the start, "
                      f"{percent(run['last_training_win_rate'])} at the end.", "", versus, "",
                      "| Checkpoint | Training win rate since the previous one | "
                      + " | ".join(f"Against {label}" for label in labels) + " |",
                      "|---|---|" + "---|" * len(labels)]
            for checkpoint in run["checkpoints"]:
                lines.append(f"| {checkpoint['update']} | {percent(checkpoint['training_win_rate'])} | "
                             + " | ".join(percent(checkpoint["diagnosis_win_rates"][label]) for label in labels) + " |")
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
    (out / "analysis.md").write_text(markdown(analysis))
    print(f"wrote {out / 'analysis.md'} and analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
