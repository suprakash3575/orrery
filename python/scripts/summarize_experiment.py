"""Summarize the training-opponent experiment into learning-curve tables.

The experiment trains the same model against two opponents (setting A:
heuristic-v2, setting B: random-v1), with three learner seeds each, then
measures saved models with ``diagnose.py`` and compares them with
``compare_diagnoses.py``. This script reads those outputs and writes:

- one learning-curve table per evaluation opponent: for every run and saved
  model, the training turns, win rate, share of games that reached two planets
  by turn 50, and the win-rate change against that seed's untrained model;
- one table of direct comparisons between two trained models, such as A and B
  at update 153 for the same seed.

It only reads finished reports. It plays no games and changes no files other
than the two it writes.

Usage::

    uv run python -m scripts.summarize_experiment \\
        --results ../runs/exp-results --out ../runs/exp-results/summary
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orrery.jsonio import read_json_object

# The expansion target's deadline: a game counts when the learner first held
# two planets at once on or before this turn.
EXPANSION_DEADLINE_TURN = 50

DIAGNOSIS_FORMAT = "diagnosis-v2"
COMPARISON_FORMAT = "diagnosis-comparison-v1"
# diagnose.py labels a rebuilt never-trained model with this model name.
UNTRAINED_MODEL = "reconstructed untrained baseline"


def reached_two_planets_by(
    game_results: Iterable[Mapping[str, Any]],
    deadline: int = EXPANSION_DEADLINE_TURN,
) -> tuple[int, int]:
    """Count games that reached two planets at once by ``deadline``.

    Returns ``(games that reached it, games)``. ``first_two_planets_turn`` is
    the first recorded turn on which the learner held two or more planets at
    once, or ``None`` when that never happened. The rule therefore looks only
    at when expansion first happened, not at how the game ended: a game that
    reached two planets on turn 20 and ended on turn 40 counts, while a game
    that ended before ever reaching two does not. Turn ``deadline`` itself
    counts; the turn after it does not.
    """

    reached = 0
    games = 0
    for game in game_results:
        games += 1
        turn = game.get("first_two_planets_turn")
        if turn is not None and turn <= deadline:
            reached += 1
    return reached, games


@dataclass(frozen=True)
class ModelReport:
    """One ``diagnose.py`` report: one saved model against one opponent."""

    # path is the report's folder, for the summary to point back to.
    path: Path
    # training_opponent is what the run trained against; it tells A from B.
    training_opponent: str
    learner_seed: int
    # completed_updates is 0 for the untrained model.
    completed_updates: int
    # environment_steps is the training turns behind this model, the learning
    # curve's x-axis.
    environment_steps: int
    # opponent is what this report's games were played against.
    opponent: str
    # weights_sha256 identifies the model, so comparisons can be matched to
    # reports even after their folders move between machines.
    weights_sha256: str
    games: int
    wins: int
    reached_two_by_deadline: int

    @property
    def untrained(self) -> bool:
        return self.completed_updates == 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def expansion_share(self) -> float:
        return self.reached_two_by_deadline / self.games if self.games else 0.0


@dataclass(frozen=True)
class WinRateComparison:
    """The win-rate line of one ``compare_diagnoses.py`` comparison."""

    path: Path
    baseline_sha256: str
    candidate_sha256: str
    opponent: str
    difference: float
    low: float
    high: float
    conclusion: str


def load_report(folder: Path) -> ModelReport:
    """Read the fields the summary needs from one diagnose report."""

    report = read_json_object(folder / "report.json", "diagnosis report")
    if report.get("format") != DIAGNOSIS_FORMAT:
        raise ValueError(f"{folder}: not a {DIAGNOSIS_FORMAT} report")
    identity, games = report["identity"], report["games"]
    reached, count = reached_two_planets_by(games["game_results"])
    return ModelReport(
        path=folder,
        training_opponent=identity["training_opponent"],
        learner_seed=int(identity["learner_seed"]),
        completed_updates=int(games["completed_updates"]),
        environment_steps=int(games["environment_steps"]),
        opponent=games["opponent"],
        weights_sha256=games["weights_sha256"],
        games=count,
        wins=int(games["measurements"]["wins"]),
        reached_two_by_deadline=reached,
    )


def load_comparison(folder: Path) -> WinRateComparison:
    """Read the win-rate result from one comparison."""

    comparison = read_json_object(folder / "comparison.json", "diagnosis comparison")
    if comparison.get("format") != COMPARISON_FORMAT:
        raise ValueError(f"{folder}: not a {COMPARISON_FORMAT} comparison")
    win_rate = next(m for m in comparison["measures"] if m["measure"] == "win_rate")
    low, high = win_rate["range_95"]
    return WinRateComparison(
        path=folder,
        baseline_sha256=comparison["models"]["baseline"]["weights_sha256"],
        candidate_sha256=comparison["models"]["candidate"]["weights_sha256"],
        opponent=comparison["shared"]["opponent"],
        difference=float(win_rate["mean_difference"]),
        low=float(low),
        high=float(high),
        conclusion=win_rate["conclusion"],
    )


def discover(results: Path) -> tuple[list[ModelReport], list[WinRateComparison]]:
    """Find every report and comparison folder directly under ``results``.

    Folders are visited in sorted order so the output never depends on the
    order the file system lists them in.
    """

    reports: list[ModelReport] = []
    comparisons: list[WinRateComparison] = []
    for folder in sorted(p for p in results.iterdir() if p.is_dir()):
        if (folder / "report.json").is_file():
            reports.append(load_report(folder))
        elif (folder / "comparison.json").is_file():
            comparisons.append(load_comparison(folder))
    return reports, comparisons


def _short(opponent: str) -> str:
    """The opponent's name without its parameters, e.g. ``random-v1``."""

    return opponent.split(":", 1)[0]


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _points(value: float) -> str:
    return f"{value * 100:+.1f}"


def learning_curve_rows(
    reports: Sequence[ModelReport],
    comparisons: Sequence[WinRateComparison],
) -> list[dict[str, str]]:
    """One row per report, with its win-rate change against the untrained model.

    The change comes from the comparison whose baseline is the same seed's
    untrained model and whose candidate is this model, on the same opponent.
    It is blank when no such comparison was run, and for the untrained model
    itself.
    """

    by_sha = {(r.weights_sha256, r.opponent): r for r in reports}
    against_untrained: dict[tuple[str, str], WinRateComparison] = {}
    for comparison in comparisons:
        baseline = by_sha.get((comparison.baseline_sha256, comparison.opponent))
        if baseline is not None and baseline.untrained:
            against_untrained[(comparison.candidate_sha256, comparison.opponent)] = comparison

    rows: list[dict[str, str]] = []
    ordered = sorted(
        reports,
        key=lambda r: (r.opponent, r.training_opponent, r.learner_seed, r.completed_updates),
    )
    for report in ordered:
        change = against_untrained.get((report.weights_sha256, report.opponent))
        rows.append(
            {
                "played_against": _short(report.opponent),
                "trained_against": _short(report.training_opponent),
                "learner_seed": str(report.learner_seed),
                "updates": str(report.completed_updates),
                "training_turns": str(report.environment_steps),
                "games": str(report.games),
                "win_rate": _percent(report.win_rate),
                "two_planets_by_turn_50": _percent(report.expansion_share),
                "win_rate_change_vs_untrained": _points(change.difference) if change else "",
                "range_95": f"{_points(change.low)} to {_points(change.high)}" if change else "",
                "conclusion": change.conclusion if change else "",
                "report": report.path.name,
            }
        )
    return rows


def direct_comparison_rows(
    reports: Sequence[ModelReport],
    comparisons: Sequence[WinRateComparison],
) -> list[dict[str, str]]:
    """Comparisons between two trained models, such as A and B at update 153."""

    by_sha = {(r.weights_sha256, r.opponent): r for r in reports}
    rows: list[dict[str, str]] = []
    for comparison in comparisons:
        baseline = by_sha.get((comparison.baseline_sha256, comparison.opponent))
        candidate = by_sha.get((comparison.candidate_sha256, comparison.opponent))
        if baseline is None or candidate is None or baseline.untrained:
            continue

        def label(report: ModelReport) -> str:
            return (
                f"trained vs {_short(report.training_opponent)}, "
                f"seed {report.learner_seed}, update {report.completed_updates}"
            )

        rows.append(
            {
                "played_against": _short(comparison.opponent),
                "baseline": label(baseline),
                "candidate": label(candidate),
                "win_rate_difference": _points(comparison.difference),
                "range_95": f"{_points(comparison.low)} to {_points(comparison.high)}",
                "conclusion": comparison.conclusion,
                "comparison": comparison.path.name,
            }
        )
    return sorted(rows, key=lambda row: (row["played_against"], row["baseline"], row["candidate"]))


def _markdown_table(rows: Sequence[Mapping[str, str]]) -> list[str]:
    if not rows:
        return ["(none)"]
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(row[c] for c in columns) + " |" for row in rows]
    return lines


def write_summary(
    out: Path,
    curve: Sequence[Mapping[str, str]],
    direct: Sequence[Mapping[str, str]],
) -> None:
    """Write ``summary.md`` for reading and ``learning_curve.csv`` for charts."""

    out.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Training-opponent experiment summary",
        "",
        f"Two planets by turn {EXPANSION_DEADLINE_TURN}: the share of games in which",
        f"the learner first held two planets at once on or before turn {EXPANSION_DEADLINE_TURN},",
        "however the game ended. Win-rate changes are percentage points, with the",
        "95% range from compare_diagnoses' whole-map resampling.",
        "",
    ]
    for opponent in sorted({row["played_against"] for row in curve}):
        lines += [f"## Learning curves, played against {opponent}", ""]
        lines += _markdown_table([row for row in curve if row["played_against"] == opponent])
        lines.append("")
    lines += ["## Direct comparisons between trained models", ""]
    lines += _markdown_table(direct)
    lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    with (out / "learning_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        if curve:
            writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="folder holding diagnose report folders and comparison folders",
    )
    parser.add_argument("--out", type=Path, required=True, help="folder for summary.md and learning_curve.csv")
    arguments = parser.parse_args(argv)

    reports, comparisons = discover(arguments.results)
    curve = learning_curve_rows(reports, comparisons)
    direct = direct_comparison_rows(reports, comparisons)
    write_summary(arguments.out, curve, direct)
    print(
        f"summarized {len(reports)} reports and {len(comparisons)} comparisons "
        f"into {arguments.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
