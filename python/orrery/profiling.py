"""Measure where training time is spent without changing training results.

The normal training log reports how quickly whole PPO updates finish. That is
enough to spot a slow run, but it does not show whether time was spent in the
Python policy, communication with Go, or the PPO optimizer. When profiling is
enabled, :class:`PhaseProfiler` measures those smaller pieces and this module
writes their totals to ``timings.csv``.

GPU work needs special handling because CUDA calls usually return before the
GPU has finished. The profiler waits for CUDA at phase boundaries so that work
is charged to the phase that launched it. Those waits make a profiled run
slower, which is why profiled timings explain *where* time went while an
unprofiled run remains the source for actual throughput comparisons.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
import csv
from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from time import monotonic
from types import TracebackType
from typing import Callable, Literal, Mapping

import torch


# These stable names become columns in timings.csv. Each value is the total
# number of seconds spent in that phase during one PPO update. Keeping all
# names here prevents the collector and trainer from silently recording the
# same work under different labels.
PROFILE_PHASES = (
    # Run the actor-critic network for the current batch of observations.
    "policy",
    # Sample learner actions from the probabilities produced by the policy.
    "sampling",
    # Calculate the selected actions' log-probabilities for the later PPO ratio.
    "log_probability",
    # Convert sampled action tensors into the per-environment request sent to Go.
    "action_encode",
    # Send a step request and wait until the Go process returns its response.
    "go_wait",
    # Parsed the removed env-v1 JSON step text. env-v2 never enters this phase,
    # so it records zero; it stays so existing timings.csv columns still load.
    "json_decode",
    # Validate a step response and convert its observation fields to NumPy.
    "observation_decode",
    # Move or replace observation and rollout data on the selected Torch device.
    "tensor_transfer",
    # Encode the request that asks Go to reset completed game slots.
    "reset_encode",
    # Wait for Go to create replacement games and return their observations.
    "reset_go_wait",
    # Decode a Go reset's binary frame into validated NumPy columns.
    "reset_decode",
    # Validate a reset response and convert its observation fields to NumPy.
    "reset_observation_decode",
    # Run the critic once more on the final observations needed to calculate GAE.
    "bootstrap",
    # Join the values collected turn by turn into one complete rollout batch.
    "rollout_build",
    # Calculate PPO losses, gradients, and optimizer updates.
    "ppo",
    # Summarize completed episodes for the human-readable training metrics.
    "episode_metrics",
)


class PhaseProfiler:
    """Accumulate optional phase durations on one Torch device.

    CUDA launches are asynchronous: the Python call can return while the GPU is
    still working. Synchronizing before and after a phase assigns that pending
    work to the phase that requested it. This deliberately slows a profiled
    run, so normal throughput must be measured with profiling disabled.
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """Create an empty set of phase totals for one training update.

        ``device`` tells the profiler whether CUDA synchronization is needed.
        ``clock`` is a function that takes no arguments and returns the current
        timer value in seconds. The default, ``monotonic``, measures elapsed
        time and cannot jump backward if the computer's date or wall clock is
        adjusted. The function is injectable so tests can return known timer
        values instead of waiting for real time to pass.
        """

        self.device = torch.device(device)
        self._clock = clock

        # A phase may occur hundreds of times while the 512 rollout turns are
        # collected. Store a running total rather than one value per call.
        self._seconds = {phase: 0.0 for phase in PROFILE_PHASES}

    def start(self) -> float:
        """Synchronize prior CUDA work and return a phase start timestamp."""

        # Waiting before reading the clock keeps unfinished work from the
        # preceding phase out of the new phase's measurement.
        self._synchronize()

        # Calling the clock does not start or create a timer. It reads the
        # current value from a continuously advancing timer. stop() later
        # subtracts this saved value from a second reading to get elapsed time.
        return self._clock()

    def phase(self, phase: str) -> _MeasuredPhase:
        """Return a ``with`` block that measures one named operation.

        Keeping the operation inside one block lets callers execute the exact
        same code with profiling enabled or disabled. The context manager only
        adds the clock reads and CUDA waits around that shared operation.
        """

        if phase not in self._seconds:
            raise ValueError(f"unknown profiling phase {phase!r}")
        return _MeasuredPhase(self, phase)

    def stop(self, phase: str, started: float) -> None:
        """Synchronize the phase's CUDA work and add its elapsed duration."""

        if phase not in self._seconds:
            raise ValueError(f"unknown profiling phase {phase!r}")

        # A Python CUDA call can finish while its GPU kernel is still running.
        # Waiting here makes the ending timestamp include that kernel's work.
        self._synchronize()

        # For example, if start() returned 10.0 and this call returns 10.25,
        # the measured phase took 0.25 seconds.
        elapsed = self._clock() - started
        if elapsed < 0.0 or not math.isfinite(elapsed):
            raise ValueError("profiling clock must produce finite increasing values")

        # Add instead of replace because phases such as policy and Go wait run
        # once per rollout turn and must be totaled across the whole update.
        self._seconds[phase] += elapsed

    def snapshot_and_reset(self) -> dict[str, float]:
        """Return this update's totals and clear them for the next update."""

        # Return a copy so clearing the internal counters cannot change the
        # completed update's values after the caller receives them.
        snapshot = dict(self._seconds)

        # Each timings.csv row describes exactly one PPO update, so the next
        # update must begin with every phase back at zero.
        self._seconds = {phase: 0.0 for phase in PROFILE_PHASES}
        return snapshot

    def _synchronize(self) -> None:
        """Wait for queued CUDA work only when the profiled device is CUDA."""

        # CPU operations finish before their Python calls return. CUDA work is
        # asynchronous and needs this explicit wait to measure it accurately.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class _MeasuredPhase:
    """Measure the code inside one ``with profiler.phase(...)`` block."""

    def __init__(self, profiler: PhaseProfiler, phase: str) -> None:
        self._profiler = profiler
        self._phase = phase
        self._started: float | None = None

    def __enter__(self) -> None:
        # __enter__ runs immediately before the caller's shared operation.
        self._started = self._profiler.start()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        # __exit__ also runs when the measured operation raises. Recording the
        # elapsed work must not suppress that original exception, so return False.
        assert self._started is not None
        self._profiler.stop(self._phase, self._started)
        return False


# Callers use this one shared context when profiling is disabled. It performs
# no clock reads and no CUDA synchronization while allowing one copy of the
# measured operation to remain inside a ``with`` statement.
_NO_PROFILE_PHASE = nullcontext()


def profile_phase(
    profiler: PhaseProfiler | None,
    phase: str,
) -> AbstractContextManager[None]:
    """Return a measured phase when enabled or the shared no-op otherwise."""

    if profiler is None:
        return _NO_PROFILE_PHASE
    return profiler.phase(phase)


# Whole-number UpdateTiming fields. They count updates or transitions, while
# every other numeric field is a duration or rate in seconds.
_TIMING_COUNTER_FIELDS = (
    "update",
    "environment_steps",
    "invocation_environment_steps",
    "update_environment_steps",
)


def _steps_per_second(steps: int, seconds: float) -> float:
    """Divide transitions by elapsed time, reporting zero for zero time.

    A zero duration is possible with an injected test clock. Returning zero
    keeps the CSV value finite instead of failing with a division by zero.
    """

    return steps / seconds if seconds > 0.0 else 0.0


@dataclass(frozen=True)
class UpdateTiming:
    """One ``timings.csv`` row for a completed PPO update.

    The first fields describe overall progress and wall-clock throughput. The
    ``*_seconds`` fields that follow split the update into smaller phases when
    profiling is enabled. With profiling disabled, those detailed fields are
    zero while the overall wall-clock fields are still recorded.
    """

    # One-based PPO update number written to the training artifacts.
    update: int
    # Total environment transitions collected over the run, including resumes.
    environment_steps: int
    # Transitions collected by this process invocation rather than older runs.
    invocation_environment_steps: int
    # Wall-clock seconds since this process invocation began.
    invocation_elapsed_seconds: float
    # Invocation transitions divided by invocation wall-clock seconds. This is
    # cumulative, so it also carries one-time setup such as starting Go.
    end_to_end_steps_per_second: float
    # Wall-clock time from the start to the end of this single update.
    update_elapsed_seconds: float
    # Transitions collected by this update alone: one complete rollout batch.
    update_environment_steps: int
    # This update's transitions divided by its own elapsed time. It excludes
    # setup and earlier updates, so a median over a window of updates, such as
    # updates 3 through 10 after warm-up, should be taken from this column.
    update_steps_per_second: float
    # Portion of the update spent collecting a rollout and optimizing the model.
    training_seconds: float
    # Time spent writing any checkpoint associated with this update.
    checkpoint_seconds: float
    # Time spent running any scheduled validation associated with this update.
    validation_seconds: float
    # True when the detailed phase measurements below were collected.
    profiling_enabled: bool

    # Detailed phase totals accumulated by PhaseProfiler during this update.
    policy_seconds: float
    sampling_seconds: float
    log_probability_seconds: float
    action_encode_seconds: float
    go_wait_seconds: float
    json_decode_seconds: float
    observation_decode_seconds: float
    tensor_transfer_seconds: float
    reset_encode_seconds: float
    reset_go_wait_seconds: float
    reset_decode_seconds: float
    reset_observation_decode_seconds: float
    bootstrap_seconds: float
    rollout_build_seconds: float
    ppo_seconds: float
    episode_metrics_seconds: float

    def __post_init__(self) -> None:
        """Reject rows that could make a timing comparison misleading."""

        # Counters describe completed work, so fractions, booleans, and
        # negative values would all indicate a caller or file-format mistake.
        for name in _TIMING_COUNTER_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        # Update zero means that no PPO update has completed and therefore
        # cannot be represented by a timing row.
        if self.update < 1:
            raise ValueError("update must be at least 1")
        if not isinstance(self.profiling_enabled, bool):
            raise TypeError("profiling_enabled must be boolean")

        # NaN, infinity, and negative durations would corrupt averages and can
        # otherwise be hard to notice after several runs are combined.
        for field in fields(self):
            if field.name in _TIMING_COUNTER_FIELDS or field.name == "profiling_enabled":
                continue
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field.name} must be finite")
            if value < 0.0:
                raise ValueError(f"{field.name} must be non-negative")


# Dataclass field order is also the stable CSV column order. Deriving the
# header here keeps serialization and resume-time validation in agreement.
_TIMING_FIELD_NAMES = tuple(field.name for field in fields(UpdateTiming))


class CSVTimingLogger:
    """Append timing rows without changing the existing ``metrics.csv``.

    Training can resume into an existing artifact directory. On construction,
    this logger therefore checks an existing ``timings.csv`` and remembers its
    final update number before it permits another row to be appended.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

        # A new run may provide a path whose artifact directory does not exist
        # yet. Creating only the parent leaves file creation to append().
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_update = 0
        if self.path.exists():
            # Fail before writing if a resumed file uses another schema or has
            # duplicate/out-of-order updates.
            self._last_update = self._validate_existing_file()

    def append(self, timing: UpdateTiming) -> None:
        """Write one later update and reject duplicates or reordered rows."""

        if timing.update <= self._last_update:
            raise ValueError(
                f"timing update {timing.update} must be after {self._last_update}"
            )

        # The header is written only for a new file. A resumed run appends to
        # the existing header that __init__ already validated.
        write_header = not self.path.exists()
        with self.path.open("a", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=_TIMING_FIELD_NAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(timing))

            # Push Python's buffered text to the operating system before the
            # row is treated as the latest successfully recorded update.
            destination.flush()
        self._last_update = timing.update

    @property
    def last_update(self) -> int:
        """Return the final completed update already present in timings.csv."""

        return self._last_update

    def _validate_existing_file(self) -> int:
        """Check schema and ordering before a resumed process appends rows."""

        with self.path.open(newline="") as source:
            reader = csv.DictReader(source)

            # Exact header equality prevents a newer process from placing
            # values under columns created by an incompatible older version.
            if tuple(reader.fieldnames or ()) != _TIMING_FIELD_NAMES:
                raise ValueError("existing timings CSV has an incompatible header")
            last_update = 0
            for line, row in enumerate(reader, start=2):
                try:
                    update = int(row["update"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid timing update on CSV line {line}"
                    ) from error

                # One row per increasing update lets readers align timings.csv
                # with metrics.csv and detect duplicate resume output.
                if update <= last_update:
                    raise ValueError(
                        "existing timing updates are not strictly increasing"
                    )
                last_update = update
            return last_update


def update_timing(
    *,
    update: int,
    environment_steps: int,
    invocation_start_steps: int,
    invocation_elapsed_seconds: float,
    update_elapsed_seconds: float,
    update_environment_steps: int,
    training_seconds: float,
    checkpoint_seconds: float,
    validation_seconds: float,
    profile_seconds: Mapping[str, float] | None,
) -> UpdateTiming:
    """Build one validated row and calculate end-to-end throughput.

    ``environment_steps`` is cumulative across the full training run, whereas
    ``invocation_start_steps`` is the saved counter at the moment this process
    started or resumed. Their difference is therefore the work done by this
    invocation. Dividing that work by its wall-clock lifetime includes rollout,
    PPO, metrics, checkpointing, and validation.

    That cumulative rate also includes one-time setup, so it keeps rising
    during early updates. ``update_environment_steps`` divided by
    ``update_elapsed_seconds`` gives each update its own end-to-end rate, which
    is the value to compare across a window of updates.
    """

    # Subtract the resume boundary so an older checkpoint's work is not counted
    # again in this process invocation's throughput.
    invocation_steps = environment_steps - invocation_start_steps
    if invocation_steps < 0:
        raise ValueError("environment steps cannot precede the invocation start")

    # Detailed fields always exist in the CSV. They remain zero in ordinary
    # unprofiled runs, which keeps one stable schema for both modes.
    phases = {phase: 0.0 for phase in PROFILE_PHASES}
    if profile_seconds is not None:
        # Reject spelling mistakes instead of silently creating or discarding
        # a measurement that has no corresponding CSV column.
        unknown = set(profile_seconds) - set(PROFILE_PHASES)
        if unknown:
            raise ValueError(f"unknown profile phases: {sorted(unknown)}")
        phases.update(profile_seconds)

    # UpdateTiming performs the final type, range, and finite-number checks.
    return UpdateTiming(
        update=update,
        environment_steps=environment_steps,
        invocation_environment_steps=invocation_steps,
        invocation_elapsed_seconds=invocation_elapsed_seconds,
        end_to_end_steps_per_second=_steps_per_second(
            invocation_steps,
            invocation_elapsed_seconds,
        ),
        update_elapsed_seconds=update_elapsed_seconds,
        update_environment_steps=update_environment_steps,
        update_steps_per_second=_steps_per_second(
            update_environment_steps,
            update_elapsed_seconds,
        ),
        training_seconds=training_seconds,
        checkpoint_seconds=checkpoint_seconds,
        validation_seconds=validation_seconds,
        profiling_enabled=profile_seconds is not None,
        **{f"{phase}_seconds": phases[phase] for phase in PROFILE_PHASES},
    )


def include_late_timing(
    timing: UpdateTiming,
    *,
    invocation_elapsed_seconds: float,
    update_elapsed_seconds: float,
    checkpoint_seconds: float,
    validation_seconds: float,
) -> UpdateTiming:
    """Return a finished copy of a pending timing row with late work added.

    **Why a row can be "pending".** The trainer builds update N's row as soon
    as that update's rollout, PPO step, metrics, and scheduled checkpoint or
    validation are done. It does not write the row yet, because more work can
    still belong to update N:

    1. Update N finishes and its row is built. The trainer then resets its
       ``checkpoint_seconds`` and ``validation_seconds`` counters to zero.
    2. The loop starts again. If another full rollout no longer fits below the
       phase ceiling, the trainer validates update N's weights *now*, and saves
       a checkpoint for that validation. Both add to the fresh counters.
    3. The row is finally written, either at the top of the next update or
       after the loop ends, where the final checkpoint also adds its time.

    Without this function, the work from step 2 was never recorded in any row.

    **What this function does.** It takes the row from step 1 plus the counter
    values accumulated since then, and returns a new row in which:

    - checkpoint and validation seconds are the row's earlier values plus the
      late ones, so both the scheduled and the late work are counted;
    - both elapsed times are replaced with values measured now, since the
      late work made the update, and the whole invocation, take longer;
    - both throughput rates are recalculated from those new elapsed times.
      Transition counts do not change because no new rollout was collected.

    Every other field, including the profiled phase totals, is copied as is.
    The original row is left unchanged; ``UpdateTiming`` is frozen.

    Example: a row built with ``validation_seconds=0`` and
    ``update_elapsed_seconds=2`` can receive a 3-second late validation and be
    written 4 seconds later. The returned row has ``validation_seconds=3`` and
    ``update_elapsed_seconds=6``, and the update rate is recalculated using 6.

    ``invocation_elapsed_seconds`` and ``update_elapsed_seconds`` are the
    current total times, not amounts to add. ``checkpoint_seconds`` and
    ``validation_seconds`` are only the late amounts added to the row.
    """

    return UpdateTiming(
        **{
            # Start from a plain dictionary of every field in the pending row.
            # The entries below override only the values that late work changes.
            **asdict(timing),
            # Elapsed times are totals measured at the moment of writing, so
            # they replace the earlier values instead of adding to them.
            "invocation_elapsed_seconds": invocation_elapsed_seconds,
            "update_elapsed_seconds": update_elapsed_seconds,
            # The transitions are the same, but the time taken is now longer,
            # so each rate must be divided again by its refreshed duration.
            "end_to_end_steps_per_second": _steps_per_second(
                timing.invocation_environment_steps,
                invocation_elapsed_seconds,
            ),
            "update_steps_per_second": _steps_per_second(
                timing.update_environment_steps,
                update_elapsed_seconds,
            ),
            # These durations are added: the row already holds any scheduled
            # checkpoint or validation from inside the update, and the late
            # amounts come from work that happened after the row was built.
            "checkpoint_seconds": timing.checkpoint_seconds + checkpoint_seconds,
            "validation_seconds": timing.validation_seconds + validation_seconds,
        }
        # Constructing a new UpdateTiming reruns its validation, so a negative
        # or non-finite value introduced here is rejected before it is written.
    )
