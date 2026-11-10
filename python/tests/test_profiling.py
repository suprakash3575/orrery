"""Focused checks for optional phase timing and its separate CSV."""

from __future__ import annotations

import csv

import pytest
import torch

from orrery.profiling import (
    CSVTimingLogger,
    PhaseProfiler,
    include_late_timing,
    profile_phase,
    update_timing,
)


def test_phase_profiler_accumulates_and_resets_named_cpu_phases() -> None:
    """Multiple intervals for one phase should become one update total."""

    # CPU profiling needs no device synchronization. Fixed timestamps make the
    # two policy intervals exactly 2 and 3 seconds so accumulation is explicit.
    profiler = PhaseProfiler("cpu", clock=iter((1.0, 3.0, 10.0, 13.0)).__next__)
    first = profiler.start()
    profiler.stop("policy", first)
    second = profiler.start()
    profiler.stop("policy", second)

    snapshot = profiler.snapshot_and_reset()

    assert snapshot["policy"] == 5.0
    assert profiler.snapshot_and_reset()["policy"] == 0.0


def test_cuda_phase_boundaries_synchronize_before_reading_the_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CUDA duration must include kernels queued before Python returns."""

    synchronized: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)
    profiler = PhaseProfiler("cuda", clock=iter((2.0, 5.0)).__next__)

    started = profiler.start()
    profiler.stop("sampling", started)

    assert synchronized == [torch.device("cuda"), torch.device("cuda")]
    assert profiler.snapshot_and_reset()["sampling"] == 3.0


def test_phase_context_measures_one_shared_operation() -> None:
    """The context manager should surround work without a second code path."""

    profiler = PhaseProfiler("cpu", clock=iter((4.0, 6.5)).__next__)

    with profile_phase(profiler, "ppo"):
        completed_operation = True

    assert completed_operation is True
    assert profiler.snapshot_and_reset()["ppo"] == 2.5


def test_timing_csv_keeps_coarse_and_profiled_measurements_separate(
    tmp_path,
) -> None:
    """A timing row should derive honest invocation-wide throughput."""

    timing = update_timing(
        update=7,
        environment_steps=240,
        invocation_start_steps=200,
        invocation_elapsed_seconds=8.0,
        update_elapsed_seconds=6.0,
        update_environment_steps=24,
        training_seconds=5.0,
        checkpoint_seconds=0.5,
        validation_seconds=0.25,
        profile_seconds={"policy": 1.5, "go_wait": 2.0},
    )
    path = tmp_path / "timings.csv"
    CSVTimingLogger(path).append(timing)

    # Reopening validates the stable header and proves resume may start timing
    # at a later checkpoint update even when an older run had no earlier rows.
    assert CSVTimingLogger(path).last_update == 7
    with path.open(newline="") as source:
        row = next(csv.DictReader(source))
    assert row["invocation_environment_steps"] == "40"
    assert float(row["end_to_end_steps_per_second"]) == 5.0
    # The per-update rate uses only this update's 24 transitions and 6 seconds,
    # so it differs from the cumulative rate that also carries earlier work.
    assert row["update_environment_steps"] == "24"
    assert float(row["update_steps_per_second"]) == 4.0
    assert row["profiling_enabled"] == "True"
    assert float(row["policy_seconds"]) == 1.5
    assert float(row["go_wait_seconds"]) == 2.0
    assert float(row["ppo_seconds"]) == 0.0


def test_late_validation_and_checkpoint_time_join_the_pending_update() -> None:
    """A ceiling gate belongs to the update whose weights it evaluates."""

    pending = update_timing(
        update=3,
        environment_steps=12,
        invocation_start_steps=0,
        invocation_elapsed_seconds=6.0,
        update_elapsed_seconds=2.0,
        update_environment_steps=6,
        training_seconds=1.5,
        checkpoint_seconds=0.25,
        validation_seconds=0.0,
        profile_seconds=None,
    )
    assert pending.update_steps_per_second == 3.0

    completed = include_late_timing(
        pending,
        invocation_elapsed_seconds=10.0,
        update_elapsed_seconds=6.0,
        checkpoint_seconds=0.75,
        validation_seconds=3.0,
    )

    assert completed.invocation_elapsed_seconds == 10.0
    assert completed.end_to_end_steps_per_second == 1.2
    assert completed.update_elapsed_seconds == 6.0
    # Late validation lengthened the update, so its own rate must be refreshed
    # from the same six transitions rather than keeping the earlier 3.0.
    assert completed.update_steps_per_second == 1.0
    assert completed.checkpoint_seconds == 1.0
    assert completed.validation_seconds == 3.0
