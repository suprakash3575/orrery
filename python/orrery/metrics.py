"""Validated CSV metrics for one PPO training run."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import torch
from torch import Tensor


@dataclass(frozen=True)
class TrainingMetrics:
    """One complete PPO update and the gameplay observed since the prior row."""

    update: int
    environment_steps: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    loss_rate: float
    draw_rate: float
    mean_episode_length: float
    mean_ship_advantage: float
    policy_loss: float
    value_loss: float
    explained_variance: float
    launch_entropy: float
    target_entropy: float
    approximate_kl: float
    clip_fraction: float
    mean_gradient_norm: float
    max_gradient_norm: float
    no_launch_rate: float
    launches_per_episode: float
    illegal_action_count: int
    environment_steps_per_second: float

    def __post_init__(self) -> None:
        integer_fields = (
            "update", "environment_steps", "wins", "losses", "draws",
            "illegal_action_count",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.update == 0:
            raise ValueError("update must be at least 1")

        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("win_rate", "loss_rate", "draw_rate", "clip_fraction", "no_launch_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if min(
            self.mean_episode_length,
            self.launch_entropy,
            self.target_entropy,
            self.mean_gradient_norm,
            self.max_gradient_norm,
            self.launches_per_episode,
            self.environment_steps_per_second,
        ) < 0.0:
            raise ValueError("length, entropy, gradient, launch, and throughput metrics must be non-negative")
        games = self.wins + self.losses + self.draws
        rates = (self.win_rate, self.loss_rate, self.draw_rate)
        if games:
            expected = (self.wins / games, self.losses / games, self.draws / games)
            if any(not math.isclose(actual, want, abs_tol=1e-6) for actual, want in zip(rates, expected, strict=True)):
                raise ValueError("win, loss, and draw rates must match their counts")
        elif any(rates):
            raise ValueError("win, loss, and draw rates must be zero when no games finished")


_FIELD_NAMES = tuple(field.name for field in fields(TrainingMetrics))
_FLOAT_FIELDS = tuple(
    name for name in _FIELD_NAMES
    if name not in {"update", "environment_steps", "wins", "losses", "draws", "illegal_action_count"}
)


class CSVMetricLogger:
    """Append validated metric rows while preserving one stable CSV schema."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_update = 0
        if self.path.exists():
            self._last_update = self._validate_existing_file()

    def append(self, metrics: TrainingMetrics) -> None:
        """Append one later update; duplicate or reordered rows fail."""

        if metrics.update <= self._last_update:
            raise ValueError(
                f"metrics update {metrics.update} must be after {self._last_update}"
            )
        write_header = not self.path.exists()
        with self.path.open("a", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=_FIELD_NAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(metrics))
            destination.flush()
        self._last_update = metrics.update

    @property
    def last_update(self) -> int:
        """Return the final completed update already present in the CSV."""

        return self._last_update

    def _validate_existing_file(self) -> int:
        with self.path.open(newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != _FIELD_NAMES:
                raise ValueError("existing metrics CSV has an incompatible header")
            last_update = 0
            for line, row in enumerate(reader, start=2):
                try:
                    update = int(row["update"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"invalid metrics update on CSV line {line}") from error
                if update <= last_update:
                    raise ValueError("existing metrics updates are not strictly increasing")
                last_update = update
            return last_update


def explained_variance(predicted_values: Tensor, return_targets: Tensor) -> float:
    """Measure how much variation in critic targets its predictions explain."""

    if predicted_values.shape != return_targets.shape or predicted_values.ndim != 1:
        raise ValueError("critic predictions and return targets must be aligned vectors")
    if predicted_values.numel() == 0:
        raise ValueError("explained variance requires at least one transition")
    if not torch.all(torch.isfinite(predicted_values)) or not torch.all(torch.isfinite(return_targets)):
        raise ValueError("explained variance inputs must be finite")
    targets = return_targets.detach().to(torch.float64)
    residuals = targets - predicted_values.detach().to(torch.float64)
    target_variance = torch.var(targets, unbiased=False)
    if target_variance == 0:
        return 0.0
    return float((1.0 - torch.var(residuals, unbiased=False) / target_variance).cpu())
