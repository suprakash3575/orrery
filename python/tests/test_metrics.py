"""Focused checks for PPO metric validation and CSV persistence."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from orrery.metrics import CSVMetricLogger, TrainingMetrics, explained_variance


def test_csv_logger_writes_header_and_resumes_with_the_next_update(tmp_path: Path) -> None:
    path = tmp_path / "run" / "metrics.csv"
    CSVMetricLogger(path).append(_metrics(update=1))
    CSVMetricLogger(path).append(_metrics(update=2, environment_steps=65_536))
    reopened = CSVMetricLogger(path)

    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))

    assert [int(row["update"]) for row in rows] == [1, 2]
    assert [int(row["environment_steps"]) for row in rows] == [32_768, 65_536]
    assert rows[0]["policy_loss"] == "-0.1"
    # Resume orchestration compares this cursor with its checkpoint before it
    # appends anything. Reading 2 proves the logger reconstructed the boundary
    # from the existing rows rather than starting again from zero.
    assert reopened.last_update == 2


def test_csv_logger_rejects_duplicate_updates_and_incompatible_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.csv"
    logger = CSVMetricLogger(path)
    logger.append(_metrics(update=1))

    with pytest.raises(ValueError, match="must be after 1"):
        logger.append(_metrics(update=1))

    incompatible = tmp_path / "old.csv"
    incompatible.write_text("update,old_metric\n1,2\n")
    with pytest.raises(ValueError, match="incompatible header"):
        CSVMetricLogger(incompatible)


def test_training_metrics_require_rates_to_match_episode_counts() -> None:
    with pytest.raises(ValueError, match="must match their counts"):
        replace(_metrics(update=1), win_rate=0.25)

    with pytest.raises(ValueError, match="illegal_action_count"):
        replace(_metrics(update=1), illegal_action_count=-1)


def test_explained_variance_covers_perfect_and_uninformative_critics() -> None:
    targets = torch.tensor([1.0, 2.0, 3.0])

    assert explained_variance(targets, targets) == pytest.approx(1.0)
    assert explained_variance(torch.zeros(3), targets) == pytest.approx(0.0)
    assert explained_variance(torch.ones(3), torch.ones(3)) == 0.0


def test_explained_variance_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned vectors"):
        explained_variance(torch.zeros(2), torch.zeros(2, 1))


def _metrics(*, update: int, environment_steps: int = 32_768) -> TrainingMetrics:
    return TrainingMetrics(
        update=update,
        environment_steps=environment_steps,
        wins=6,
        losses=3,
        draws=1,
        win_rate=0.6,
        loss_rate=0.3,
        draw_rate=0.1,
        mean_episode_length=250.5,
        mean_ship_advantage=12.0,
        policy_loss=-0.1,
        value_loss=0.2,
        explained_variance=0.4,
        launch_entropy=0.5,
        target_entropy=0.6,
        approximate_kl=0.01,
        clip_fraction=0.03,
        mean_gradient_norm=0.7,
        max_gradient_norm=1.2,
        no_launch_rate=0.25,
        launches_per_episode=8.5,
        illegal_action_count=0,
        environment_steps_per_second=500.0,
    )
