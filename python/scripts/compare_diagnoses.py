#!/usr/bin/env python3
"""Compare two diagnose reports map by map: did one model play better?

The usual pair is the reconstructed untrained baseline and a trained
checkpoint, both played by ``diagnose`` on the same validation maps and
seats against the same opponent. The script never plays games; it only reads
the two finished reports.

Why maps, not games: the four seats of one map share a board, so an easy or
hard map moves all four results together. Treating 400 games as 400
independent tests would overstate how sure we are. Games are first paired by
map and seat, keeping only seats both models measured; each map's paired seats
are averaged, the two models are compared map by map, and the uncertainty
comes from resampling whole maps with their seats kept together.

A difference is called better or worse only when its whole 95% range lies on
one side of zero. Anything else, including a difference too small to tell
apart from map-to-map noise, is reported as inconclusive.

Run from the ``python/`` directory::

    uv run python -m scripts.compare_diagnoses \\
        --baseline ../runs/diagnosis-untrained --candidate ../runs/diagnosis-153 \\
        --out ../runs/comparison-untrained-vs-153

It writes ``comparison.md``, ``comparison.json``, and ``maps.csv`` to a new or
empty output folder.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from orrery.jsonio import read_json_object

# Resampling repeats for the 95% range. More repeats only steady the range's
# last digit; 10,000 is plenty for 100 maps and still takes well under a second.
DEFAULT_RESAMPLES = 10_000
# Fewest repeats accepted. The range is the 2.5th to 97.5th percentile of the
# resampled means, so with very few repeats it is just the spread of a few
# random draws: one repeat collapses it to a single point, which can sit off
# zero by chance and produce a false verdict. Below this, comparing is refused.
MIN_RESAMPLES = 1_000
# Fixed so that rerunning a comparison reproduces its ranges exactly.
DEFAULT_BOOTSTRAP_SEED = 0
# Fewest maps for which a verdict is given. Resampling a handful of maps
# produces a falsely narrow range (two maps can only ever be recombined a few
# ways), so with fewer maps every measure is reported as inconclusive. The
# planned comparisons use 100 maps.
DEFAULT_MIN_MAPS = 20


@dataclass(frozen=True)
class Measure:
    """One per-game measurement and how to read it."""

    # Key in comparison.json and the column prefix in maps.csv.
    key: str
    # Plain-language name for comparison.md.
    label: str
    # Reads one game_results row. None means the game has no value (for
    # example, no action turns), and that game is left out of the map average.
    extract: Callable[[Mapping[str, Any]], float | None]
    # Whether a higher value is better play. Time without a productive body
    # is the one measure where lower is better.
    higher_is_better: bool
    # How to print a value: as a percentage or as a plain number.
    percent: bool


MEASURES = (
    Measure(
        "win_rate", "Win rate",
        lambda row: 1.0 if row["result"] == "win" else 0.0,
        higher_is_better=True, percent=True,
    ),
    Measure(
        "most_planets", "Most planets held at once",
        lambda row: float(row["most_planets"]),
        higher_is_better=True, percent=False,
    ),
    Measure(
        "reached_two_planets", "Games that held two planets at once",
        lambda row: 0.0 if row["first_two_planets_turn"] is None else 1.0,
        higher_is_better=True, percent=True,
    ),
    Measure(
        "share_without_productive_body", "Share of action turns with no planet or comet",
        lambda row: row["share_without_productive_body"],
        higher_is_better=False, percent=True,
    ),
)


# ---------------------------------------------------------------------------
# Stage 1: read both reports and refuse a comparison that would be unfair.
# ---------------------------------------------------------------------------


def load_report(directory: Path) -> dict[str, Any]:
    """Read one diagnose report that played games and measured every one."""

    report = read_json_object(directory / "report.json", "diagnosis report")
    if report.get("format") != "diagnosis-v2":
        raise ValueError(f"{directory} is not a diagnose report")
    games = report.get("games")
    if not games or "game_results" not in games:
        raise ValueError(f"{directory} has no per-game results; rerun diagnose with games")
    return report


def require_valid(report: Mapping[str, Any], name: str) -> None:
    """Refuse a report that is not a complete, clean record of play.

    Agreement between two reports proves nothing if both are broken in the
    same way, so each one is checked on its own first. Every map seed in its
    range must have been played from every seat exactly once; a missing seat
    would quietly change what a map average means. Go must have rejected no
    learner actions: a rejected action is a turn the model did not really
    play, so the games no longer measure its strength.
    """

    games = report["games"]
    start, end = games["seeds"]
    players = report["identity"]["players"]
    expected = [(seed, seat) for seed in range(start, end) for seat in range(players)]
    played = sorted((row["seed"], row["seat"]) for row in games["game_results"])
    if played != expected:
        raise ValueError(
            f"{name} report does not cover every seed in [{start}, {end}) from all "
            f"{players} seats exactly once ({len(played)} games, want {len(expected)})"
        )
    if games["rejected_actions"] != 0:
        raise ValueError(
            f"{name} report has {games['rejected_actions']} rejected actions; "
            "its games do not measure the model's play"
        )


def require_comparable(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    """Refuse two reports that did not play the same games the same way.

    The only intended difference is the model. Different maps, seats,
    opponent, game contract, map generator, engine, or evaluation settings
    (device, Torch version, batch size, numeric precision) could change
    results on their own, and the comparison would wrongly credit or blame
    the model. Each report must already have passed ``require_valid``.
    """

    problems = []
    for field, label in (
        ("players", "player count"),
        ("contract", "contract"),
        ("generator_version", "map generator"),
    ):
        if baseline["identity"][field] != candidate["identity"][field]:
            problems.append(label)
    for field in ("opponent", "seeds", "engine", "evaluation"):
        if baseline["games"][field] != candidate["games"][field]:
            problems.append(field)

    def assignments(report: Mapping[str, Any]) -> list[tuple[int, int]]:
        return sorted((row["seed"], row["seat"]) for row in report["games"]["game_results"])

    if assignments(baseline) != assignments(candidate):
        problems.append("map-and-seat assignments")
    if problems:
        raise ValueError("reports are not comparable; they differ in: " + ", ".join(problems))


# ---------------------------------------------------------------------------
# Stage 2: average each map's seats, then compare the models map by map.
# ---------------------------------------------------------------------------


def paired_map_averages(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    measure: Measure,
) -> dict[int, tuple[float, float, int]]:
    """Average one measure over each map's seats, using only seats both models measured.

    Games are paired by (map seed, seat) before anything is averaged. A seat
    whose value is missing for either model is left out for both, so a map's
    two averages always describe the same seats. Averaging each side on its
    own would let a seat that only one model measured shift that side alone,
    and the difference would partly compare different seats. A map with no
    complete pair is absent rather than counted as zero.

    Returns ``{map seed: (baseline average, candidate average, seat pairs)}``.
    """

    baseline = {(row["seed"], row["seat"]): measure.extract(row) for row in baseline_rows}
    pairs_by_map: dict[int, list[tuple[float, float]]] = {}
    for row in candidate_rows:
        before = baseline.get((row["seed"], row["seat"]))
        after = measure.extract(row)
        if before is not None and after is not None:
            pairs_by_map.setdefault(row["seed"], []).append((before, after))
    return {
        seed: (
            sum(before for before, _after in pairs) / len(pairs),
            sum(after for _before, after in pairs) / len(pairs),
            len(pairs),
        )
        for seed, pairs in sorted(pairs_by_map.items())
    }


def bootstrap_range(
    differences: np.ndarray, resamples: int, rng: np.random.Generator
) -> tuple[float, float]:
    """The middle 95% of mean differences over resampled sets of maps.

    Each resample draws as many maps as there are, with replacement, so some
    maps appear twice and others not at all. How much the mean moves between
    resamples shows how much the result depends on which maps were played.
    """

    picks = rng.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def compare_measure(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    measure: Measure,
    resamples: int,
    rng: np.random.Generator,
    min_maps: int = DEFAULT_MIN_MAPS,
) -> dict[str, Any]:
    """Compare one measure map by map and state what the difference shows.

    The range is reported whenever there are at least two maps, but a verdict
    other than inconclusive needs at least ``min_maps`` of them.
    """

    paired = paired_map_averages(baseline_rows, candidate_rows, measure)
    seeds = list(paired)
    differences = np.array([paired[seed][1] - paired[seed][0] for seed in seeds], dtype=float)

    result: dict[str, Any] = {
        "measure": measure.key,
        "label": measure.label,
        "higher_is_better": measure.higher_is_better,
        "maps": len(seeds),
        # Seats measured by both models, summed over maps: fewer than four per
        # map means some games had no value for this measure.
        "seat_pairs": sum(paired[seed][2] for seed in seeds),
        "baseline_mean": float(np.mean([paired[seed][0] for seed in seeds])) if seeds else None,
        "candidate_mean": float(np.mean([paired[seed][1] for seed in seeds])) if seeds else None,
        "mean_difference": float(differences.mean()) if seeds else None,
        "range_95": None,
        "conclusion": "inconclusive",
        "per_map": {seed: (paired[seed][0], paired[seed][1]) for seed in seeds},
    }
    # One map gives no spread to measure, so no range and no verdict.
    if len(seeds) < 2:
        return result
    low, high = bootstrap_range(differences, resamples, rng)
    result["range_95"] = [low, high]
    if len(seeds) < min_maps:
        result["conclusion"] = "inconclusive (too few maps)"
        return result
    # Better or worse only when the whole range is on one side of zero.
    if low > 0 or high < 0:
        candidate_higher = low > 0
        result["conclusion"] = (
            "candidate better" if candidate_higher == measure.higher_is_better else "candidate worse"
        )
    return result


# ---------------------------------------------------------------------------
# Stage 3: write the comparison files.
# ---------------------------------------------------------------------------


def _model_name(report: Mapping[str, Any]) -> str:
    games = report["games"]
    if games["checkpoint"] is None:
        return f"{games['model']} (learner seed {games['initialization_seed']})"
    return f"checkpoint after {games['completed_updates']} updates"


def _value(value: float | None, percent: bool, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign}.1%}" if percent else f"{value:{sign}.2f}"


def write_outputs(
    out: Path,
    baseline_dir: Path,
    candidate_dir: Path,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    resamples: int,
    bootstrap_seed: int,
    min_maps: int,
) -> None:
    """Write comparison.json, maps.csv, and comparison.md."""

    games = baseline["games"]
    models = {}
    for name, directory, report in (
        ("baseline", baseline_dir, baseline),
        ("candidate", candidate_dir, candidate),
    ):
        models[name] = {
            "report": str(directory),
            "model": _model_name(report),
            "checkpoint": report["games"]["checkpoint"],
            "weights_sha256": report["games"]["weights_sha256"],
        }
    summary = {
        "format": "diagnosis-comparison-v1",
        "models": models,
        "shared": {
            "opponent": games["opponent"],
            "seeds": games["seeds"],
            "players": baseline["identity"]["players"],
            "engine": games["engine"],
            "evaluation": games["evaluation"],
        },
        "resamples": resamples,
        "bootstrap_seed": bootstrap_seed,
        "min_maps": min_maps,
        "measures": [
            {key: value for key, value in result.items() if key != "per_map"} for result in results
        ],
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

    # One row per map: each measure's map average for both models and their
    # difference. An empty cell means that map had no value for that measure.
    seeds = sorted({seed for result in results for seed in result["per_map"]})
    with (out / "maps.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["seed"]
        for result in results:
            key = result["measure"]
            header += [f"baseline_{key}", f"candidate_{key}", f"difference_{key}"]
        writer.writerow(header)
        for seed in seeds:
            row: list[Any] = [seed]
            for result in results:
                pair = result["per_map"].get(seed)
                row += ["", "", ""] if pair is None else [pair[0], pair[1], pair[1] - pair[0]]
            writer.writerow(row)

    measures = {measure.key: measure for measure in MEASURES}
    lines = [
        "# Model comparison",
        "",
        f"- Baseline: {models['baseline']['model']} — `{baseline_dir}`",
        f"- Candidate: {models['candidate']['model']} — `{candidate_dir}`",
        f"- Opponent: `{games['opponent']}`",
        f"- Maps: seeds [{games['seeds'][0]:,}, {games['seeds'][1]:,}), every seat, "
        f"{len(baseline['games']['game_results'])} games per model",
        f"- Evaluator: `{games['evaluation']['policy']}` on {games['evaluation']['device']}, "
        f"Torch {games['evaluation'].get('torch_version', 'unknown')}",
        "",
        "Games are paired by map and seat, and only seats that both models measured",
        "are kept. Each map's paired seats are averaged, then the candidate is compared",
        "with the baseline map by map. The 95% range comes from resampling whole maps "
        f"({resamples:,} times), so related seats stay together. A result is better or",
        "worse only when its whole range is on one side of zero and at least",
        f"{min_maps} maps were compared; otherwise it is inconclusive. Per-map values",
        "are in [maps.csv](maps.csv).",
        "",
        "| Measure | Baseline | Candidate | Difference | 95% range | Maps | Seat pairs | Conclusion |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for result in results:
        percent = measures[result["measure"]].percent
        spread = result["range_95"]
        lines.append(
            f"| {result['label']} | {_value(result['baseline_mean'], percent)} "
            f"| {_value(result['candidate_mean'], percent)} "
            f"| {_value(result['mean_difference'], percent, signed=True)} "
            f"| {'n/a' if spread is None else _value(spread[0], percent, signed=True) + ' to ' + _value(spread[1], percent, signed=True)} "
            f"| {result['maps']} | {result['seat_pairs']} | {result['conclusion']} |"
        )
    lines += [
        "",
        "## Limits",
        "",
        "- The range describes these maps and this opponent only, not every possible game.",
        "- Both models played deterministically: each game is its top-scoring moves,",
        "  not the exploring moves used in training.",
        "- An inconclusive result means the maps cannot tell the models apart, not that",
        "  they are proven equal.",
        "",
    ]
    (out / "comparison.md").write_text("\n".join(lines))


def compare(
    baseline_dir: Path,
    candidate_dir: Path,
    out: Path,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    min_maps: int = DEFAULT_MIN_MAPS,
) -> list[dict[str, Any]]:
    """Compare two reports and write the comparison files. Returns every measure's result."""

    if resamples < MIN_RESAMPLES:
        raise ValueError(f"resamples must be at least {MIN_RESAMPLES:,} for a meaningful 95% range")
    baseline = load_report(baseline_dir)
    candidate = load_report(candidate_dir)
    require_valid(baseline, "baseline")
    require_valid(candidate, "candidate")
    require_comparable(baseline, candidate)
    # One generator for the whole comparison, used in MEASURES order, so the
    # same inputs and seed always give the same ranges.
    rng = np.random.default_rng(bootstrap_seed)
    results = [
        compare_measure(
            baseline["games"]["game_results"],
            candidate["games"]["game_results"],
            measure,
            resamples,
            rng,
            min_maps,
        )
        for measure in MEASURES
    ]
    write_outputs(
        out, baseline_dir, candidate_dir, baseline, candidate, results,
        resamples, bootstrap_seed, min_maps,
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Read options, compare the two reports, and print where the comparison went."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, required=True, help="diagnose output folder to compare against")
    parser.add_argument("--candidate", type=Path, required=True, help="diagnose output folder being judged")
    parser.add_argument(
        "--out", type=Path, required=True,
        help="new or empty output folder; earlier comparisons are never overwritten",
    )
    parser.add_argument(
        "--resamples", type=int, default=DEFAULT_RESAMPLES,
        help=f"map resamples for the 95%% range, at least {MIN_RESAMPLES:,} "
        f"(default: {DEFAULT_RESAMPLES:,})",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED,
        help=f"seed for the resampling, so ranges are reproducible (default: {DEFAULT_BOOTSTRAP_SEED})",
    )
    parser.add_argument(
        "--min-maps", type=int, default=DEFAULT_MIN_MAPS,
        help=f"fewest maps for a better/worse verdict (default: {DEFAULT_MIN_MAPS})",
    )
    arguments = parser.parse_args(argv)
    if arguments.resamples < MIN_RESAMPLES:
        parser.error(f"--resamples must be at least {MIN_RESAMPLES:,} for a meaningful 95% range")
    if arguments.min_maps < 2:
        parser.error("--min-maps must be at least 2: one map has no spread to measure")

    out = arguments.out
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} is not empty; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)

    results = compare(
        arguments.baseline.resolve(),
        arguments.candidate.resolve(),
        out,
        resamples=arguments.resamples,
        bootstrap_seed=arguments.bootstrap_seed,
        min_maps=arguments.min_maps,
    )
    for result in results:
        print(f"{result['label']}: {result['conclusion']}")
    print(f"comparison: {out / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
