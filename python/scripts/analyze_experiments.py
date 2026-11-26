#!/usr/bin/env python3
"""Analyse a finished (or partly finished) experiment session.

It reads a session folder written by run_experiments.py and writes
``analysis.md`` and ``analysis.json`` into it: each experiment's tables and,
where the experiment file asks for one, its answer, by the rules in
docs/stage1-implementation-spec.md section 6. It never plays games, and it can
be rerun at any time; it rewrites only those two files.

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

# Experiment 1's thresholds, in win-rate points (0.05 = 5 points). See
# docs/stage1-implementation-spec.md, experiment 1, "Comparing with the issue".
KEPT_ABOVE = -0.05
LOST_AT_MOST = -0.10
# A fixed seed makes the 95% ranges the same on every rerun.
BOOTSTRAP_SEED = 0
# Training's win rate is compared over this many updates at each end of a run
# to decide whether it rose.
TREND_UPDATES = 10


def percent(value: float | None) -> str:
    return "undefined" if value is None else f"{100 * value:.1f}%"


def points(value: float | None) -> str:
    return "undefined" if value is None else f"{100 * value:+.1f} pts"


def job_folder(out: Path, job: Job) -> Path:
    return out / job.experiment / job.label


# ---------------------------------------------------------------------------
# Reading each kind of job's results.
# ---------------------------------------------------------------------------


def bot_games(out: Path, job: Job) -> list[dict[str, Any]]:
    """The completed games of a bot job, one row per (map seed, seat)."""

    path = job_folder(out, job) / "opponentcheck" / "games.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row["status"] == "completed"]


def bot_summary(out: Path, job: Job) -> dict[str, Any]:
    """One bot job's row: results, counts and calibration."""

    summary = json.loads((job_folder(out, job) / "opponentcheck" / "summary.json").read_text())
    tester = summary["testers"][0]
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
        "complete": summary["run"]["status"] == "complete",
        "completed": tester["completed"],
        "planned": tester["planned"],
        "wins": tester["wins"],
        "win_rate": tester["win_rate"] if tester["completed"] else None,
        "eliminated": tester["eliminated"],
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
    games = json.loads(report_path.read_text())["games"]["measurements"]
    return games["wins"] / games["games"]


def training_win_rates(metrics_path: Path) -> list[tuple[int, int, int]]:
    """(update, wins, games) for every update in a run's metrics.csv."""

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
    """

    won = {(row["seed"], row["seat"]): float(row["result"] > 0) for row in baseline}
    by_map: dict[int, list[tuple[float, float]]] = {}
    for row in candidate:
        key = (row["seed"], row["seat"])
        if key in won:
            by_map.setdefault(row["seed"], []).append((won[key], float(row["result"] > 0)))
    seeds = sorted(by_map)
    base = np.array([np.mean([pair[0] for pair in by_map[seed]]) for seed in seeds])
    cand = np.array([np.mean([pair[1] for pair in by_map[seed]]) for seed in seeds])
    result: dict[str, Any] = {"maps": len(seeds), "baseline_rate": None, "candidate_rate": None,
                              "difference": None, "range": None, "strength_kept": None}
    if not seeds:
        return result
    differences = cand - base
    result.update(
        baseline_rate=float(base.mean()),
        candidate_rate=float(cand.mean()),
        difference=float(differences.mean()),
    )
    if len(seeds) >= 2:
        result["range"] = list(bootstrap_range(differences, DEFAULT_RESAMPLES,
                                               np.random.default_rng(BOOTSTRAP_SEED)))
    if base.mean() > 0:
        result["strength_kept"] = float(cand.mean() / base.mean())
    return result


def restricted_bot_answer(compare: Mapping[str, Any], valid: bool, why_invalid: str) -> str:
    """Experiment 1's answer from the main comparison."""

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
    """Every experiment's tables and answers, as one JSON-ready dict."""

    session = json.loads((out / "session.json").read_text())
    plan: Plan = parse_plan(session["experiment_text"])
    states = {job.id: job_state(out, job) for job in plan.jobs}
    jobs = {job.id: job for job in plan.jobs}
    done = {job_id for job_id, state in states.items() if state == "done"}

    analysis: dict[str, Any] = {
        "session": plan.session.name,
        "maps": [plan.session.map_start, plan.session.map_start + plan.session.map_count],
        "runs": session.get("runs", []),
        "jobs": states,
        "botcheck": {},
        "compare": [],
        "dial": [],
        "diagnose": {},
        "train": {},
    }

    bot_rows: dict[str, dict[str, Any]] = {}
    for job in plan.jobs:
        if job.kind == "botcheck" and job.id in done:
            bot_rows[job.id] = bot_summary(out, job)
    for experiment in dict.fromkeys(job.experiment for job in plan.jobs if job.kind == "botcheck"):
        group = [job for job in plan.jobs if job.experiment == experiment]
        analysis["botcheck"][experiment] = {
            "testers": [bot_rows[job.id] for job in group if job.id in bot_rows],
            "not_done": [job.id for job in group if job.id not in bot_rows],
        }

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

    for dial in plan.dials:
        rates = [bot_rows[job_id]["win_rate"] if job_id in bot_rows else None for job_id in dial["order"]]
        if any(rate is None for rate in rates):
            answer = "Not ready: some testers aren't done."
        else:
            rises = [dial["order"][i + 1] for i in range(len(rates) - 1) if not rates[i + 1] < rates[i]]
            answer = ("Usable: the win rate falls at every step." if not rises
                      else f"Not usable as it stands: the win rate doesn't fall at {', '.join(rises)}. "
                           "The epsilon_* settings are the next thing to vary.")
        analysis["dial"].append({**dial, "win_rates": rates, "answer": answer})

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

    for experiment in dict.fromkeys(job.experiment for job in plan.jobs if job.kind == "train"):
        group = [job for job in plan.jobs if job.experiment == experiment]
        runs = [train_run(out, job) if job.id in done else {"job": job.id, "done": False,
                "opponent": job.settings["opponent_label"]} for job in group]
        analysis["train"][experiment] = {
            "runs": runs,
            "answer": train_answer(runs, plan.answers[experiment]["progress_min_runs"]),
        }
    return analysis


def band_answer(rows: Sequence[Mapping[str, Any]], band: Sequence[float], min_runs: int) -> str:
    """Experiment 3: an opponent is found if at least min_runs models win
    inside the band against it."""

    if any(row["win_rate"] is None for row in rows):
        return "Not ready: some diagnoses aren't done."
    low, high = band
    found = []
    sides = []
    for opponent in dict.fromkeys(row["opponent"] for row in rows):
        rates = [row["win_rate"] for row in rows if row["opponent"] == opponent]
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
    checkpoint next to each diagnosis win rate, and the training trend."""

    folder = job_folder(out, job)
    history = training_win_rates(folder / "run" / "metrics.csv")
    checkpoints = []
    previous = 0
    for path in sorted((folder / "run" / "checkpoints").glob("update-*.pt")):
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
    span = min(TREND_UPDATES, len(history) // 2)
    first = sum(w for _u, w, _g in history[:span]) / max(1, sum(g for _u, _w, g in history[:span]))
    last = sum(w for _u, w, _g in history[-span:]) / max(1, sum(g for _u, _w, g in history[-span:]))
    return {
        "job": job.id,
        "done": True,
        "opponent": job.settings["opponent_label"],
        "start": Path(job.settings["init_from"]).parent.parent.name,
        "checkpoints": checkpoints,
        "first_training_win_rate": first if span else None,
        "last_training_win_rate": last if span else None,
        "training_rose": bool(span) and last > first,
    }


def train_answer(runs: Sequence[Mapping[str, Any]], min_runs: int) -> str:
    """Experiment 4: progress if, for some training opponent, at least
    min_runs runs beat the first diagnose_against opponent at any checkpoint."""

    if not all(run["done"] for run in runs):
        return "Not ready: some runs aren't done."
    for opponent in dict.fromkeys(run["opponent"] for run in runs):
        group = [run for run in runs if run["opponent"] == opponent]
        beat = sum(
            any(next(iter(c["diagnosis_win_rates"].values())) > 0 for c in run["checkpoints"]) for run in group
        )
        if beat >= min_runs:
            return (f"First real progress: training against {opponent} won games against the full opponent "
                    f"in {beat} of {len(group)} runs. The next step lowers epsilon towards 0.")
    for opponent in dict.fromkeys(run["opponent"] for run in runs):
        group = [run for run in runs if run["opponent"] == opponent]
        if sum(run["training_rose"] for run in group) >= min_runs:
            return (f"Partial: training wins rose against {opponent}, but still 0% against the full opponent. "
                    "Use experiment 1's answer to choose between widening the learner's moves and a smaller "
                    "epsilon step.")
    return "Not working: training wins didn't rise either. Experiment 1 says whether the learner's moves or the training method are to blame."


# ---------------------------------------------------------------------------
# Writing analysis.md.
# ---------------------------------------------------------------------------


def markdown(analysis: Mapping[str, Any]) -> str:
    lines = [f"# Analysis: {analysis['session']}", ""]
    start, end = analysis["maps"]
    states = list(analysis["jobs"].values())
    lines.append(f"Maps {start:,} to {end - 1:,}, every seat. Jobs: "
                 + ", ".join(f"{states.count(s)} {s}" for s in sorted(set(states))) + ".")
    if analysis["runs"]:
        last = analysis["runs"][-1]
        lines.append(f"Code: `{last.get('git_commit')}`{' (with uncommitted changes)' if last.get('git_dirty') else ''}.")
    lines.append("")

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
        for row in section["testers"]:
            if row["win_games"]:
                games = ", ".join(f"seed {g['seed']} seat {g['seat']}" for g in row["win_games"])
                lines.append(f"\n{row['tester']} won: {games}.")
        lines.append("")

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

    for dial in analysis["dial"]:
        lines += [f"## Dial: {dial['name']}", "", "| Step | Win rate |", "|---|---|"]
        lines += [f"| {job_id} | {percent(rate)} |" for job_id, rate in zip(dial["order"], dial["win_rates"])]
        lines += ["", f"**Answer:** {dial['answer']}", ""]

    for experiment, section in analysis["diagnose"].items():
        runs = list(dict.fromkeys((row["run"], row["update"]) for row in section["rows"]))
        lines += [f"## {experiment}", "",
                  "| Opponent | " + " | ".join(f"{run} u{update}" for run, update in runs) + " |",
                  "|---|" + "---|" * len(runs)]
        for opponent in dict.fromkeys(row["opponent"] for row in section["rows"]):
            cells = [next(percent(r["win_rate"]) for r in section["rows"]
                          if r["opponent"] == opponent and (r["run"], r["update"]) == run) for run in runs]
            lines.append(f"| {opponent} | " + " | ".join(cells) + " |")
        if section["answer"]:
            lines += ["", f"**Answer:** {section['answer']}"]
        lines.append("")

    for experiment, section in analysis["train"].items():
        lines += [f"## {experiment}", ""]
        for run in section["runs"]:
            if not run["done"]:
                lines += [f"- {run['job']}: not done", ""]
                continue
            labels = list(run["checkpoints"][0]["diagnosis_win_rates"]) if run["checkpoints"] else []
            lines += [f"**{run['opponent']} from {run['start']}:** training win rate "
                      f"{percent(run['first_training_win_rate'])} at the start, "
                      f"{percent(run['last_training_win_rate'])} at the end.", "",
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
