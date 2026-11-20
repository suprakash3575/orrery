"""Tests for the training-opponent experiment summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.summarize_experiment import (
    discover,
    join_comparisons,
    learning_curve_rows,
    reached_two_planets_by,
)

_OPPONENT = "random-v1:launch_probability=0.3,max_fraction=0.8,min_fraction=0.2"
_SAMPLE: dict[str, Any] = {
    "opponent": _OPPONENT,
    "seeds": [1_000_000, 1_000_100],
    "players": 4,
    "engine": "v1.0.0+abc",
    "evaluation": {"policy": "eval-policy-v2", "device": "cuda"},
}


def _report(folder: Path, *, seed: int, updates: int, sha: str) -> None:
    """Write the fields summarize_experiment reads from a diagnose report."""

    folder.mkdir()
    report = {
        "format": "diagnosis-v2",
        "identity": {
            "training_opponent": _OPPONENT,
            "learner_seed": seed,
            "players": _SAMPLE["players"],
        },
        "games": {
            "completed_updates": updates,
            "environment_steps": updates * 32_768,
            "opponent": _SAMPLE["opponent"],
            "weights_sha256": sha,
            "seeds": _SAMPLE["seeds"],
            "engine": _SAMPLE["engine"],
            "evaluation": _SAMPLE["evaluation"],
            "measurements": {"wins": 1},
            "game_results": [{"first_two_planets_turn": None}] * 4,
        },
    }
    (folder / "report.json").write_text(json.dumps(report))


def _comparison(folder: Path, *, baseline: str, candidate: str, difference: float, **sample: Any) -> None:
    """Write a compare_diagnoses result; sample overrides change its games."""

    folder.mkdir()
    comparison = {
        "format": "diagnosis-comparison-v1",
        "models": {"baseline": {"weights_sha256": baseline}, "candidate": {"weights_sha256": candidate}},
        "shared": {**_SAMPLE, **sample},
        "measures": [
            {
                "measure": "win_rate",
                "mean_difference": difference,
                "range_95": [difference - 0.05, difference + 0.05],
                "conclusion": "candidate better",
            }
        ],
    }
    (folder / "comparison.json").write_text(json.dumps(comparison))


def _change_for(rows: list[dict[str, str]], report: str) -> str:
    return next(row for row in rows if row["report"] == report)["win_rate_change_vs_untrained"]


@pytest.mark.parametrize(
    ("game", "counts"),
    (
        # The deadline turn itself counts.
        ({"first_two_planets_turn": 50, "turns": 500}, True),
        # One turn late does not.
        ({"first_two_planets_turn": 51, "turns": 500}, False),
        # Expansion counts however the game ended: reaching two planets on
        # turn 20 still counts when the game was over by turn 40.
        ({"first_two_planets_turn": 20, "turns": 40}, True),
        # A game that never held two planets at once never counts.
        ({"first_two_planets_turn": None, "turns": 30}, False),
    ),
)
def test_reached_two_planets_by_turn_50(game: dict[str, int | None], counts: bool) -> None:
    assert reached_two_planets_by([game]) == (int(counts), 1)


def test_reached_two_planets_by_counts_every_game() -> None:
    games = [
        {"first_two_planets_turn": 12},
        {"first_two_planets_turn": None},
        {"first_two_planets_turn": 50},
        {"first_two_planets_turn": 77},
    ]

    assert reached_two_planets_by(games) == (2, 4)


def test_gain_is_measured_only_against_the_same_seeds_untrained_model(tmp_path: Path) -> None:
    """Seed 1502's gain must not be measured from seed 1501's starting point.

    Both comparisons exist in the results folder. The one against seed 1501's
    untrained model (+45) is left out with its reason; the one against seed
    1502's own (+20) is the gain shown.
    """

    _report(tmp_path / "untrained-1501", seed=1501, updates=0, sha="u1501")
    _report(tmp_path / "untrained-1502", seed=1502, updates=0, sha="u1502")
    _report(tmp_path / "trained-1502", seed=1502, updates=153, sha="t1502")
    _comparison(tmp_path / "cmp-other-seed", baseline="u1501", candidate="t1502", difference=0.45)
    _comparison(tmp_path / "cmp-own-seed", baseline="u1502", candidate="t1502", difference=0.20)

    reports, comparisons = discover(tmp_path)
    joined = join_comparisons(reports, comparisons)

    assert _change_for(learning_curve_rows(reports, joined), "trained-1502") == "+20.0"
    assert [(c.path.name, reason) for c, reason in joined.ignored] == [
        ("cmp-other-seed", "baseline is learner seed 1501 but the candidate is learner seed 1502"),
    ]


@pytest.mark.parametrize(
    "different",
    (
        {"engine": "v1.0.1+def"},
        {"seeds": [1_000_100, 1_000_200]},
        {"evaluation": {"policy": "eval-policy-v2", "device": "cpu"}},
    ),
)
def test_comparison_on_another_sample_is_not_joined(tmp_path: Path, different: dict[str, Any]) -> None:
    """A comparison is joined only to reports on exactly its games.

    The reports hold the same two models, but the comparison played other
    maps, another engine build, or another evaluator setting, so its result
    does not describe these reports.
    """

    _report(tmp_path / "untrained", seed=1501, updates=0, sha="u1501")
    _report(tmp_path / "trained", seed=1501, updates=153, sha="t1501")
    _comparison(tmp_path / "cmp", baseline="u1501", candidate="t1501", difference=0.30, **different)

    reports, comparisons = discover(tmp_path)
    joined = join_comparisons(reports, comparisons)

    assert _change_for(learning_curve_rows(reports, joined), "trained") == ""
    assert [(c.path.name, reason) for c, reason in joined.ignored] == [
        ("cmp", "baseline: no report of this model on the comparison's exact sample"),
    ]
