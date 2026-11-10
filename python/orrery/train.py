"""Run PPO training against the authoritative Go environment.

This module connects components implemented in earlier steps. Go owns the game
rules and returns observations. ``collect_rollout`` gathers a fixed batch of
turns and calculates GAE targets. ``ppo_update`` changes the policy weights.
This module owns the surrounding lifecycle: configuration, schedules, metrics,
validation timing, checkpoints, budgets, and the command-line entry point.

One training environment step means one game advancing by one turn. With the
standard configuration, 64 games advance for 512 rollout turns, so one complete
PPO update consumes ``64 * 512 = 32,768`` training environment steps.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import random
from time import monotonic
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .archive import freeze_passing_checkpoint
from .config import RunConfig, SeedRange, load_run_config
from .checkpoint import (
    ActiveEpisodeAudit,
    CheckpointState,
    capture_random_states,
    load_checkpoint,
    recorded_checkpoint_fingerprint,
    restore_random_states,
    save_checkpoint,
)
from .env_client import OBSERVATION_DIMENSIONS, EnvClient, Episode
from .evaluation_context import configure_cuda_workspace
from .protocol import PLAYER_FEATURE_SHIPS, SHIP_SCALE
from .evaluator import (
    EvaluationResult,
    ValidationProgress,
    checked_evaluation_result,
    evaluation_metadata,
    interval_evidence,
    pipeline_gate,
    release_readiness_gate,
    run_rotated_matches,
    timings_evidence,
    validation_summary,
)
from .jsonio import json_value, read_json_object, write_json_atomic, write_json_exclusive
from .metrics import CSVMetricLogger, TrainingMetrics, explained_variance
from .model import OrbitPolicy, policy_device
from .paths import REPOSITORY_ROOT
from .ppo import PPOUpdateMetrics, ppo_update
from .profiling import (
    CSVTimingLogger,
    PhaseProfiler,
    UpdateTiming,
    include_late_timing,
    profile_phase,
    update_timing,
)
from .rollout import CollectedRollout, RolloutBatch, collect_rollout


# The default starts Go from source. EnvClient runs it from the repository root,
# so callers do not need to build or locate a separate executable first.
DEFAULT_ENV_COMMAND = ("go", "run", "./cmd/env")

# Validation history uses these stable names to distinguish the small pipeline
# check from the later release-readiness check after checkpoint resume.
PIPELINE_GATE = "pipeline"
RELEASE_READINESS_GATE = "release-readiness"

# Engine versions that do not name exact code (see _engine_is_unidentified).
# Go reports "development" when it found no commit, and appends "+dirty" to a
# commit built with uncommitted changes.
DEVELOPMENT_ENGINE = "development"
DIRTY_ENGINE_SUFFIX = "+dirty"

# Tests can replace the real rotated-game runner with a small controlled result
# while production keeps the same call shape and uses run_rotated_matches.
ValidationRunner = Callable[..., EvaluationResult]
# Called as (policy, config, command, runner, on_progress=...). Callable cannot
# spell the keyword argument, so the parameters are left open here.
PhaseValidation = Callable[..., EvaluationResult]
ProgressCallback = Callable[[ValidationProgress], None]


@dataclass(frozen=True)
class ScheduleValues:
    """Coefficients held constant while one rollout is optimized.

    The trainer calculates these values from the number of environment steps
    already completed. Every minibatch in the next PPO update receives the
    same values; they change only between complete updates.
    """

    # AdamW step size. It gradually decreases toward zero over the run.
    learning_rate: float
    # Strength of the bonus that encourages uncertain launch/no-launch choices.
    launch_entropy: float
    # Strength of the bonus that encourages exploration among legal targets.
    target_entropy: float


@dataclass(frozen=True)
class GameplayMetrics:
    """Human-readable game behavior observed during one rollout.

    PPO returns optimization measurements such as loss and entropy separately.
    This structure describes what happened in the games: terminal outcomes,
    their lengths and final ship balance, and how often the learner launched.
    """

    # Terminal games whose learner-relative reward was positive, negative, or zero.
    wins: int
    losses: int
    draws: int
    # Average terminal turn among games that finished during this rollout.
    mean_episode_length: float
    # Average learner ships minus enemy ships when those games finished.
    mean_ship_advantage: float
    # Fraction of legal source decisions for which the learner did not launch.
    no_launch_rate: float
    # Average total launches over each game that finished during this rollout.
    launches_per_episode: float


@dataclass(frozen=True)
class TrainingResult:
    """Small summary returned when this invocation stops cleanly.

    The files in ``run_directory`` contain the detailed evidence. These fields
    let the command-line layer report where they are, how much training was
    completed, and whether the process stopped for a limit, review, or gate.
    """

    # Directory containing configuration, CSV metrics, checkpoints, and evaluations.
    run_directory: Path
    # Total game turns used for learning; validation turns are excluded.
    environment_steps: int
    # Number of full rollout batches that finished a complete PPO update.
    completed_updates: int
    # Human-readable lifecycle decision that caused this invocation to stop.
    reason: str


@dataclass(frozen=True)
class TrainingPhase:
    """The gate and budget contract active at one update boundary.

    Keeping these values together prevents a pipeline ceiling from being paired
    accidentally with release validation, or a release result from being saved
    under the pipeline filename. A phase changes only after its gate passes.
    """

    # Stable name stored in checkpoint validation history.
    gate_name: str
    # Maximum cumulative training steps available while this phase is active.
    ceiling: int
    # Prefix used for the human-readable validation evidence file.
    evidence_prefix: str
    # Purpose stored beside a validation checkpoint in the run manifest.
    checkpoint_purpose: str
    # Console label that distinguishes the larger release evaluation.
    progress_label: str
    # Total rotated games expected from this phase's validation runner: its
    # seed count times the run's player count.
    game_count: int
    # Function that copies the policy and plays this phase's fixed sample.
    validation: PhaseValidation


class EpisodeMetricTracker:
    """Carry per-game launch counts across rollout and update boundaries.

    A 512-turn rollout can end while a game is still in progress, and that game
    continues in the next rollout. Counting launches only within one rollout
    would therefore undercount long games. This tracker keeps a running total
    for each reusable environment slot until that slot reaches a terminal game
    state, then reports and resets the total for its newly assigned game.
    """

    def __init__(self, environment_ids: Sequence[int]) -> None:
        """Start every environment slot with zero recorded launches."""

        self._launches = {environment_id: 0 for environment_id in environment_ids}

    def consume(self, rollout: RolloutBatch) -> GameplayMetrics:
        """Read one rollout and return game metrics for episodes that finished.

        Rollout rows are stored in time-major order: all environment slots for
        turn 0, then all slots for turn 1, and so on. The modulo operation below
        maps each flat row back to its reusable environment slot. Launch totals
        remain stored for non-terminal games so the following rollout can keep
        counting the same episode.
        """

        # The tracker and rollout must use the same stable environment axis;
        # otherwise a launch could be credited to the wrong game.
        if tuple(self._launches) != rollout.environment_ids:
            raise ValueError("metric tracker environments do not match the rollout")

        # Only launchable sources represent a real choice. Padding, enemy-owned
        # bodies, and owned bodies without a legal target are excluded from the
        # no-launch rate instead of being counted as forced no-launch decisions.
        launchable = rollout.observations.launchable_mask
        launched = rollout.actions.launches & launchable

        # These values still live on the GPU after PPO collection. Calling
        # `.item()` inside the loop below would make Python stop and wait for
        # the GPU once for every value in every transition. A normal rollout
        # has 32,768 transitions, so those small waits previously accumulated
        # into a visible part of every update.
        #
        # Perform the simple counting on the GPU first. Each row of this tensor
        # contains three exact integers in this order:
        #
        #     [number of launches, terminal flag, completed turn]
        #
        # Keeping these values in an integer tensor avoids converting counts or
        # turn numbers through floating point merely to combine the transfer.
        integer_rows = torch.stack(
            (
                launched.sum(dim=1),
                rollout.terminated.to(torch.long),
                rollout.turns,
            ),
            dim=1,
        )
        # Flatten the per-transition rows and append two rollout-wide totals:
        # the number of real launch decisions and how many chose no launch.
        # One `.cpu()` waits for the complete integer payload, and `.tolist()`
        # then turns the CPU tensor into ordinary Python values for the loop.
        integer_payload = torch.cat(
            (
                integer_rows.reshape(-1),
                torch.stack(
                    (
                        launchable.sum(),
                        (launchable & ~launched).sum(),
                    )
                ),
            )
        ).detach().cpu().tolist()
        decision_count = int(integer_payload[-2])
        no_launch_count = int(integer_payload[-1])

        # Rewards and normalized ship features are floating-point values, so
        # transfer them as a second payload instead of mixing them with the
        # exact integer counts above. Each row contains:
        #
        #     [reward, learner ship feature, strongest opponent ship feature]
        #
        # Player row 0 is always the learner. Rows 1-3 are the clockwise,
        # opposite, and counter-clockwise opponents; a two-player opponent is
        # row 2. Padding rows and eliminated players are all zeros, which
        # decodes to 0 ships, so the row-wise maximum picks the strongest
        # opponent without consulting the player mask. Winning means finishing
        # strictly first, so the strongest opponent is the one to compare with.
        #
        # This second `.cpu()` is the only other GPU wait needed by the metric
        # tracker. The loop below reads plain Python lists.
        player_ships = rollout.next_observations.players[:, :, PLAYER_FEATURE_SHIPS]
        float_rows = torch.stack(
            (
                rollout.rewards,
                player_ships[:, 0],
                player_ships[:, 1:].amax(dim=1),
            ),
            dim=1,
        ).detach().cpu().tolist()

        # These lists collect only games that terminate in this rollout. Their
        # means are zero when no episode finishes; ongoing games remain tracked.
        rewards: list[float] = []
        lengths: list[int] = []
        ship_advantages: list[float] = []
        completed_launches: list[int] = []
        # obs-v2 fixes the ship scale in its schema, so it is a constant rather
        # than a value Go reports per run.
        ship_log_scale = math.log1p(SHIP_SCALE)

        for row in range(rollout.transition_count):
            # Every integer row occupies three adjacent payload values:
            # launches, terminal flag, and completed turn number.
            integer_offset = row * 3
            launch_count = int(integer_payload[integer_offset])
            terminated = bool(integer_payload[integer_offset + 1])
            turn = int(integer_payload[integer_offset + 2])
            reward, learner_ships, opponent_ships = float_rows[row]

            # Flat rollout rows cycle through the stable environment-ID tuple.
            environment_id = rollout.environment_ids[row % rollout.environment_count]
            self._launches[environment_id] += launch_count
            if not terminated:
                continue

            # A terminal row supplies the final reward, length, and next-state
            # ship totals. After recording them, the slot's next episode begins
            # with a fresh launch counter. The ship advantage is a diagnostic
            # decoded from approximate float features; the reward stays the
            # authority on wins, losses, and ties.
            rewards.append(float(reward))
            lengths.append(turn)
            ship_advantages.append(
                math.expm1(float(learner_ships) * ship_log_scale)
                - math.expm1(float(opponent_ships) * ship_log_scale)
            )
            completed_launches.append(self._launches[environment_id])
            self._launches[environment_id] = 0

        # Terminal rewards are +1 for a sole winner, -1 for any other loss
        # (including the learner's own elimination), and 0 for a tie.
        wins = sum(reward > 0.0 for reward in rewards)
        losses = sum(reward < 0.0 for reward in rewards)
        draws = len(rewards) - wins - losses
        return GameplayMetrics(
            wins=wins,
            losses=losses,
            draws=draws,
            mean_episode_length=_mean(lengths),
            mean_ship_advantage=_mean(ship_advantages),
            no_launch_rate=no_launch_count / decision_count if decision_count else 0.0,
            launches_per_episode=_mean(completed_launches),
        )

    def state_dict(self) -> dict[str, dict[int, int]]:
        """Copy unfinished launch totals into checkpoint-friendly data.

        The returned dictionary contains ordinary Python values and does not
        expose the tracker's mutable internal mapping. A resume records these
        partial totals with the games it abandons, then starts new per-episode
        accounting because those unfinished games are deliberately not replayed.
        """

        return {"launches_by_environment": dict(self._launches)}


def schedule_values(config: RunConfig, environment_steps: int) -> ScheduleValues:
    """Calculate learning and exploration settings for the next PPO update.

    Learning rate decays across the complete cumulative training budget. The
    two entropy bonuses reach their configured floors earlier, at the fraction
    named by ``entropy_anneal_fraction``. Passing the current completed-step
    count makes checkpointed progress, rather than wall-clock time, control all
    three schedules.
    """

    return ScheduleValues(
        learning_rate=_linear_decay(
            config.learning_rate,
            0.0,
            environment_steps,
            config.cumulative_step_ceiling,
        ),
        launch_entropy=_linear_decay(
            config.launch_entropy_initial,
            config.launch_entropy_final,
            environment_steps,
            round(config.cumulative_step_ceiling * config.entropy_anneal_fraction),
        ),
        target_entropy=_linear_decay(
            config.target_entropy_initial,
            config.target_entropy_final,
            environment_steps,
            round(config.cumulative_step_ceiling * config.entropy_anneal_fraction),
        ),
    )


def can_collect_full_rollout(
    environment_steps: int,
    batch_size: int,
    ceiling: int,
) -> bool:
    """Check whether another full rollout fits before the stage ceiling.

    PPO must optimize either the complete collected batch or none of it. For
    example, if 49,990 steps have been used, a 20-step batch cannot begin under
    a 50,000-step ceiling. Stopping below the exact ceiling is intentional and
    prevents a partial batch from changing the meaning of an update.
    """

    values = (environment_steps, batch_size, ceiling)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("step counts must be integers")
    if min(values) < 0 or batch_size == 0:
        raise ValueError("step counts must be non-negative and batch_size must be positive")
    # The next update is allowed when its final transition lands exactly on the
    # ceiling, but not when even one transition would cross it.
    return environment_steps + batch_size <= ceiling


def initial_stage_decision(
    config: RunConfig,
    environment_steps: int,
    *,
    review_reason: str | None,
) -> str | None:
    """Choose what happens after one complete initial-stage PPO update.

    Reaching the hard ceiling ends the experiment as failed. The earlier review
    boundary pauses unless the caller supplied a written reason to continue.
    A pipeline pass transitions phases before this helper is called, so this
    function handles only the still-active initial phase. ``None`` means another
    update may be considered.
    """

    # A run that did not prove learning by its hard ceiling must be diagnosed.
    if environment_steps >= config.initial_step_ceiling:
        return "initial pipeline gate failed"
    # Continuing beyond the review point is an explicit, recorded decision.
    if environment_steps >= config.review_steps and not review_reason:
        return "review required"
    return None


def _training_phase(config: RunConfig, gate_name: str) -> TrainingPhase:
    """Build the complete lifecycle settings for one training phase."""

    if gate_name == PIPELINE_GATE:
        return TrainingPhase(
            gate_name=PIPELINE_GATE,
            ceiling=config.initial_step_ceiling,
            evidence_prefix="pipeline",
            checkpoint_purpose="pipeline-validation",
            progress_label="validation",
            game_count=config.pipeline_games,
            validation=_run_pipeline_validation,
        )
    if gate_name == RELEASE_READINESS_GATE:
        return TrainingPhase(
            gate_name=RELEASE_READINESS_GATE,
            ceiling=config.cumulative_step_ceiling,
            evidence_prefix="release",
            checkpoint_purpose="release-readiness-validation",
            progress_label="release validation",
            game_count=config.release_games,
            validation=_run_release_readiness_validation,
        )
    raise ValueError(f"unknown training phase gate {gate_name!r}")


def _validation_record(
    phase: TrainingPhase,
    result: EvaluationResult,
    config: RunConfig,
    environment_steps: int,
    validation_boundary: int,
) -> dict[str, Any]:
    """Judge one checked result and build its complete durable evidence."""

    common = {
        "gate": phase.gate_name,
        "training_environment_steps": environment_steps,
        "validation_boundary": validation_boundary,
        "rejected_actions": result.rejected_actions,
        "metadata": dict(result.metadata),
        # Built field by field, so a new EvaluationOutcome field must be added
        # here too. This one record becomes the evidence file, the
        # checkpoint's validation history, and any archived release evidence.
        "outcomes": [
            {
                "seed": outcome.seed,
                "seat": outcome.seat,
                "reward": outcome.reward,
                "turns": outcome.turns,
                "end_reason": outcome.end_reason,
                "launches": outcome.launches,
            }
            for outcome in result.outcomes
        ],
        "summary": validation_summary(result.outcomes, config.players),
    }
    # Timings describe this run, not its replay identity. A result built
    # without the real runner has none, and none is recorded.
    if result.timings is not None:
        common["timings"] = timings_evidence(result.timings)
    if phase.gate_name == PIPELINE_GATE:
        gate = pipeline_gate(result, required_wins=config.pipeline_required_wins)
        common.update(
            wins=result.wins,
            games=len(result.outcomes),
            passed=gate.passed,
            failed_checks=list(gate.failed_checks),
        )
        return common

    if phase.gate_name != RELEASE_READINESS_GATE:
        raise ValueError(f"unknown validation gate {phase.gate_name!r}")
    gate = release_readiness_gate(
        result,
        expected_games=phase.game_count,
        required_wins=config.release_required_wins,
        minimum_lower_bound=config.release_minimum_lower_bound,
        confidence_level=config.release_confidence_level,
    )
    # The checked Go metadata already contains the authoritative seed interval.
    # Reusing it here avoids deriving a third copy that could drift from the
    # runner request and the metadata embedded below.
    seed_range = result.metadata["seed_range"]
    common.update(
        evaluation_policy=config.evaluation_policy,
        opponent=config.training_opponent,
        seed_range=list(seed_range),
        requirements={
            "validation_seeds": config.release_validation_seeds,
            "expected_games": phase.game_count,
            "required_wins": config.release_required_wins,
            "confidence_level": config.release_confidence_level,
            "minimum_lower_bound": config.release_minimum_lower_bound,
            "required_rejected_actions": 0,
            "required_results_per_seed_and_seat": 1,
        },
        wins=gate.wins,
        losses=gate.losses,
        draws=gate.draws,
        games=gate.games,
        win_rate=gate.win_rate,
        confidence_interval=interval_evidence(gate.interval),
        passed=gate.passed,
        failed_checks=list(gate.failed_checks),
    )
    return common


def _publish_passing_release(
    run_directory: Path,
    archive_root: Path,
    config: RunConfig,
    validation_record: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_fingerprint: str,
    completed_updates: int,
) -> dict[str, Any]:
    """Freeze and publish all evidence required to call training complete.

    A checkpoint stores the passing validation before archive creation starts.
    If the copy or evidence write fails, a later resume reconstructs this same
    request from that durable history and retries it. The archive operation is
    idempotent, so recovery safely accepts an already-complete matching archive.
    """

    record = dict(validation_record)
    if record.get("gate") != RELEASE_READINESS_GATE or record.get("passed") is not True:
        raise ValueError("release publication requires a passing release record")
    validation_boundary = record.get("validation_boundary")
    if (
        isinstance(validation_boundary, bool)
        or not isinstance(validation_boundary, int)
        or validation_boundary < 0
    ):
        raise ValueError("passing release record has an invalid validation boundary")

    # These fields are added after checkpoint saving because they identify the
    # newly written checkpoint itself. Resume can reconstruct them from the
    # checkpoint path and its already-verified sidecar.
    record["checkpoint"] = checkpoint_path.name
    record["checkpoint_sha256"] = checkpoint_fingerprint
    record["completed_updates"] = completed_updates
    archive = freeze_passing_checkpoint(
        checkpoint_path,
        archive_root,
        fingerprint=checkpoint_fingerprint,
        config=config,
        validation_record=record,
        completed_updates=completed_updates,
    )
    record["archive"] = archive.evidence_reference()
    write_json_atomic(
        run_directory / f"release-{validation_boundary:09d}.json",
        record,
    )
    return record


def run_training(
    config: RunConfig,
    run_directory: str | Path,
    *,
    command: Sequence[str] = DEFAULT_ENV_COMMAND,
    max_updates: int | None = None,
    review_reason: str | None = None,
    resume_checkpoint: str | Path | None = None,
    archive_root: str | Path | None = None,
    validation_runner: ValidationRunner = run_rotated_matches,
    clock: Callable[[], float] = monotonic,
    profile: bool = False,
    timing_clock: Callable[[], float] = monotonic,
    allow_development_engine: bool = False,
) -> TrainingResult:
    """Create or resume an experiment and train it with PPO until it must stop.

    The function runs these steps in this order; each is marked "Step N" in
    the code below:

    1. Check the caller's options and the requested device, before any file is
       written or Go is started.
    2. Build the untrained policy, its optimizer, and two random-number
       generators: one picks moves while playing, one shuffles minibatches.
    3. Create a new run directory, or check an existing one and restore the
       policy, optimizer, generators, counters, and validation history from the
       resume checkpoint. Open the metrics and timing logs.
    4. Work out which phase training is in from its validation history. A
       release that already passed only needs its archive published, so the
       function returns there. Resuming past the 25M review needs a reason.
    5. Start the Go process with the training maps and opponent, record a
       resume, and start a fresh game in every slot.
    6. Repeat one update per pass: play a rollout with the policy unchanged,
       stop if Go rejected any action, run PPO on the whole batch, write the
       metrics row, and save a periodic checkpoint.
    7. Validate at every 5M-turn boundary and when the phase's ceiling is
       reached. In the initial phase, play random-v1 until the pipeline gate
       passes; training pauses for review at 25M unless a reason was given,
       and the 50M ceiling ends the experiment as failed. After a pipeline
       pass the same learner continues toward 500M, validated against the
       training opponent (heuristic-v2). Training stops when release readiness
       passes, at either ceiling, at a required review, or at the optional
       update limit.
    8. Save a final checkpoint at the last completed update, write its timing
       row, and return why training stopped.

    ``validation_runner`` is injectable so orchestration tests can supply a
    small known result. ``archive_root`` redirects frozen candidates in tests;
    normal runs use the repository's Git-ignored ``runs/archive`` directory.
    ``clock`` lets deterministic tests remove wall-clock throughput differences
    from otherwise identical metric rows. This function starts a new run unless
    ``resume_checkpoint`` names a completed update from that same run. Resume
    deliberately abandons the checkpoint's live games and asks Go for new
    assignments beginning at its saved schedule cursor. ``profile`` adds CUDA
    synchronization around detailed phases and therefore must be disabled when
    measuring normal throughput; ``timing_clock`` supplies the coarse wall time
    written to the separate timings CSV. ``allow_development_engine`` lets a
    resume proceed when the engine version does not name exact code (a local
    ``development`` or ``+dirty`` build); it is refused otherwise.
    """

    # Step 1: check the caller's options and the device.
    #
    # Start this clock before model, checkpoint, and Go setup so end-to-end
    # throughput includes all work performed by this process invocation.
    invocation_started = timing_clock()
    # Validate caller-only controls before creating files or starting Go. The
    # production run omits max_updates; tests and short local checks can use it
    # to stop after a known number of complete updates.
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive when supplied")
    device = _training_device(config.device)
    profiler = PhaseProfiler(device) if profile else None
    frozen_archive_root = (
        Path(archive_root)
        if archive_root is not None
        else REPOSITORY_ROOT / "runs" / "archive"
    )

    # Step 2: build the untrained policy, its optimizer, and the generators.
    #
    # initial_policy seeds before constructing the model because PyTorch
    # randomly initializes its parameters.
    policy = initial_policy(config.learner_seed).to(device).eval()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # A "generator" is a source of random numbers. Giving it a fixed starting
    # number (a "seed") makes it produce the same sequence every run, so a run
    # can be repeated exactly. Training uses two separate generators, so the
    # random numbers used for one job never shift the numbers used for the
    # other.
    #
    # Generator 1: rolls the dice that pick the learner's moves while it plays
    # (during training it tries moves at random, guided by their probabilities).
    # It must live on the same device (CPU or GPU) as those probabilities,
    # because PyTorch only draws from a generator on the matching device.
    # Seed = learner seed + 1, so it differs from the seed that built the model.
    action_generator = torch.Generator(device=device).manual_seed(config.learner_seed + 1)
    # Generator 2: shuffles the order of the collected turns before PPO splits
    # them into small groups ("minibatches") to learn from. It stays on the CPU:
    # PPO makes the shuffled order there and then copies it to the GPU.
    # Seed = learner seed + 2, a third separate sequence.
    #
    # Both generators are saved in every checkpoint, so resuming a run
    # continues their random sequences exactly where they stopped.
    shuffle_generator = torch.Generator().manual_seed(config.learner_seed + 2)

    # Step 3: create the run directory, or check it and restore the learner
    # from the resume checkpoint; then restore the counters and open the logs.
    destination, resume = _open_run_directory(
        config,
        run_directory,
        resume_checkpoint,
        policy,
        optimizer,
        action_generator,
        shuffle_generator,
    )
    checkpoint = resume.checkpoint if resume is not None else None

    lifecycle = "resumed" if checkpoint is not None else "started"
    print(
        f"training {lifecycle}: device={device.type} "
        f"environments={config.environment_count} "
        f"rollout_turns={config.rollout_steps} "
        f"transitions_per_update={config.batch_size}; "
        f"profiling={'enabled' if profile else 'disabled'}; "
        f"artifacts={destination}",
        flush=True,
    )

    # Both CSV logs must end at the checkpoint's update.
    environment_steps = checkpoint.environment_steps if checkpoint is not None else 0
    completed_updates = checkpoint.completed_updates if checkpoint is not None else 0
    invocation_start_steps = environment_steps
    logger, timing_logger = _open_metric_logs(
        destination,
        completed_updates,
        resumed=checkpoint is not None,
        max_updates=max_updates,
    )
    next_validation_step = (
        checkpoint.next_validation_step
        if checkpoint is not None
        else config.pipeline_gate_every_steps
    )
    validation_history = (
        _validation_history(checkpoint.validation_state)
        if checkpoint is not None
        else []
    )
    saved_checkpoints: dict[int, tuple[Path, str]] = {}
    if resume is not None:
        saved_checkpoints[completed_updates] = (resume.path, resume.fingerprint)
    review_recorded = (destination / "review.json").exists()

    # Step 4: work out the active phase from validation history.
    phase, pipeline_previously_passed, previous_passing_release, reason = _restored_phase(
        config, validation_history
    )
    if previous_passing_release is not None:
        # A passing record proves model quality, but training is complete only
        # after its self-contained archive and readable release evidence exist.
        # A prior freeze may have failed after the checkpoint was safely saved;
        # retry that publication now without starting or resetting Go games.
        if resume is None:
            raise ValueError("passing release history requires a resume checkpoint")
        _publish_passing_release(
            destination,
            frozen_archive_root,
            config,
            previous_passing_release,
            resume.path,
            resume.fingerprint,
            completed_updates,
        )
        return TrainingResult(
            destination,
            environment_steps,
            completed_updates,
            "release readiness gate passed",
        )
    _require_review_reason(
        config,
        environment_steps,
        resumed=checkpoint is not None,
        pipeline_previously_passed=pipeline_previously_passed,
        review_reason=review_reason,
    )

    # Step 5: start Go, record a resume, and start a game in every slot. The
    # configured player count chooses cmd/env's maps.
    with EnvClient(
        command,
        players=config.players,
        profiler=profiler,
    ) as environment:
        environment_metadata = _create_training_environment(
            environment,
            config,
            checkpoint.seed_cursor if checkpoint is not None else 0,
        )
        if resume is not None:
            review_recorded = _record_resumed_run(
                destination,
                config,
                resume,
                environment_metadata,
                environment_steps=environment_steps,
                completed_updates=completed_updates,
                review_reason=review_reason,
                review_recorded=review_recorded,
                allow_development_engine=allow_development_engine,
            )

        # A fresh run fills empty slots. A resume replaces all recorded live games
        # with assignments beginning at the restored cursor. In both cases Go
        # returns the initial observations used for the next learner actions.
        episodes = tuple(environment.reset(range(config.environment_count)))
        if profiler is not None:
            # Initial population belongs to process setup rather than update 1.
            # Reset phases in an update row therefore describe only games that
            # actually ended and were replaced during that rollout.
            profiler.snapshot_and_reset()
        tracker = EpisodeMetricTracker(range(config.environment_count))

        # Gather what the loop needs into three objects: inputs that stay fixed
        # (run), values that change after every update (state), and the
        # bookkeeping behind timings.csv (timer).
        run = _RunSetup(
            config=config,
            destination=destination,
            command=command,
            validation_runner=validation_runner,
            clock=clock,
            timing_clock=timing_clock,
            frozen_archive_root=frozen_archive_root,
            review_reason=review_reason,
            profiler=profiler,
            environment=environment,
            environment_metadata=environment_metadata,
            logger=logger,
            resume=resume,
        )
        state = _TrainingState(
            policy=policy,
            optimizer=optimizer,
            action_generator=action_generator,
            shuffle_generator=shuffle_generator,
            environment_steps=environment_steps,
            completed_updates=completed_updates,
            next_validation_step=next_validation_step,
            validation_history=validation_history,
            saved_checkpoints=saved_checkpoints,
            phase=phase,
            reason=reason,
            review_recorded=review_recorded,
            episodes=episodes,
            tracker=tracker,
        )
        timer = _UpdateTimer(timing_clock, timing_logger, invocation_started, invocation_start_steps)

        # Steps 6 and 7 run inside this loop, one update per pass. Each pass
        # first checks whether the phase's ceiling is reached (step 7), then
        # trains one update (step 6), then validates if a boundary is due
        # (step 7). The initial phase uses 50M; after its gate passes, the
        # same counters and optimizer continue under the 500M cumulative
        # ceiling.
        while True:
            next_move = _check_phase_ceiling(run, state, timer)
            if next_move == _STOP_TRAINING:
                break
            if next_move == _NEXT_PASS:
                continue

            # A bounded local invocation stops at a completed update boundary.
            if max_updates is not None and state.completed_updates >= max_updates:
                state.reason = "requested update limit"
                break

            # Nothing more can belong to the previous update's timing row, so
            # write it before this update starts.
            timer.write_pending_row()

            # Step 6: train one update.
            collected = _train_one_update(run, state, timer)
            # Step 7: validate when a boundary is due, and decide whether to stop.
            if _validate_and_decide(run, state, timer, collected) == _STOP_TRAINING:
                break

        # Step 8: save the final checkpoint and timing row.
        #
        # Every clean stopping path leaves a checkpoint at the latest completed
        # update. If the same update was already saved for a cadence or
        # validation, the helper reuses that immutable file.
        final_checkpoint_started = timing_clock()
        _save_checkpoint(run, state, purpose="stopped")
        timer.finish(final_checkpoint_seconds=timing_clock() - final_checkpoint_started)

    return TrainingResult(
        destination, state.environment_steps, state.completed_updates, state.reason
    )


# ---------------------------------------------------------------------------
# run_training's steps. Each function below is one block of the training
# lifecycle, moved out so run_training reads as the sequence of steps.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResumePoint:
    """The checkpoint a resumed run continues from; a new run has none."""

    # Learner state, counters, and validation history, already loaded and
    # checked against the run configuration.
    checkpoint: CheckpointState
    # Where the checkpoint lives, for the resume record and evidence links.
    path: Path
    # SHA-256 from the checkpoint's sidecar; load_checkpoint already verified
    # it against the file's bytes, so it is not hashed again.
    fingerprint: str
    # Per-game launch counts of the live games the resume abandons, kept for
    # the resume record.
    abandoned_metric_state: dict[str, dict[int, int]]


@dataclass(frozen=True)
class _RunSetup:
    """The inputs of the training loop that stay fixed for the whole run."""

    config: RunConfig
    # The run directory: checkpoints, CSV logs, and evidence files go here.
    destination: Path
    # The command that starts Go; validation starts its own copy with it.
    command: Sequence[str]
    validation_runner: ValidationRunner
    # ``clock`` times collection plus PPO for metrics.csv; ``timing_clock``
    # times whole updates for timings.csv. Tests replace both.
    clock: Callable[[], float]
    timing_clock: Callable[[], float]
    # Where a passing release candidate is frozen.
    frozen_archive_root: Path
    # Written reason to continue past the 25M review, or None.
    review_reason: str | None
    # Detailed per-phase timer, only when profiling is on.
    profiler: PhaseProfiler | None
    # The running Go process and the identity it reported when created.
    environment: EnvClient
    environment_metadata: Mapping[str, Any]
    logger: CSVMetricLogger
    # The checkpoint this run resumed from; None for a new run.
    resume: _ResumePoint | None


@dataclass
class _TrainingState:
    """Everything the training loop changes as it goes.

    Every checkpoint saves these values (with Go's seed cursor), so a resume
    continues from exactly here. The loop's helper functions read and change
    this object in place.
    """

    # The learner, its optimizer, and its two random-number generators.
    policy: nn.Module
    optimizer: torch.optim.Optimizer
    action_generator: torch.Generator
    shuffle_generator: torch.Generator
    # Learner turns trained so far, and completed PPO updates.
    environment_steps: int
    completed_updates: int
    # The next scheduled gate boundary, in training turns.
    next_validation_step: int
    # Every validation record so far; each checkpoint saves the full list.
    validation_history: list[dict[str, Any]]
    # Checkpoints written so far, keyed by completed update, so one update is
    # saved only once.
    saved_checkpoints: dict[int, tuple[Path, str]]
    # The gate and step ceiling currently in force.
    phase: TrainingPhase
    # Why training stops if it stops now; returned in TrainingResult.
    reason: str
    # Whether review.json already records a reason to continue past 25M.
    review_recorded: bool
    # The live games and their running launch counts.
    episodes: Sequence[Episode]
    tracker: EpisodeMetricTracker


# What the training loop does next, as decided by its helper functions.
_TRAIN_UPDATE = "train update"
_NEXT_PASS = "next pass"
_STOP_TRAINING = "stop training"


class _UpdateTimer:
    """The wall-time bookkeeping behind timings.csv, one row per update.

    A row is prepared once its update's training, checkpoint, validation, and
    review work is done, but it stays pending: a ceiling validation at the top
    of the next loop pass, or the final checkpoint save, still belongs to it.
    The row is written when the next update starts or training stops.
    """

    def __init__(
        self,
        clock: Callable[[], float],
        logger: CSVTimingLogger,
        invocation_started: float,
        invocation_start_steps: int,
    ) -> None:
        self.clock = clock
        self.logger = logger
        # When this process started, and the step count it started from.
        self.invocation_started = invocation_started
        self.invocation_start_steps = invocation_start_steps
        # Seconds spent saving checkpoints and playing validation games since
        # the last reset. They are added to the update that caused them.
        self.checkpoint_seconds = 0.0
        self.validation_seconds = 0.0
        # When the current update began, when its training began, and how many
        # seconds its collection plus PPO took.
        self.update_started = 0.0
        self.training_started = 0.0
        self.training_seconds = 0.0
        # The latest update's prepared row, not yet written.
        self.pending: UpdateTiming | None = None

    @contextmanager
    def checkpointing(self) -> Iterator[None]:
        """Add the time spent inside this block to the checkpoint seconds."""

        started = self.clock()
        yield
        self.checkpoint_seconds += self.clock() - started

    @contextmanager
    def validating(self) -> Iterator[None]:
        """Add the time spent inside this block to the validation seconds."""

        started = self.clock()
        yield
        self.validation_seconds += self.clock() - started

    def write_pending_row(self) -> None:
        """Write the previous update's row, including any work done since.

        A phase-ceiling validation can run at the top of the loop after the
        previous update prepared its row; its cost is attached first.
        """

        if self.pending is None:
            return
        self.pending = include_late_timing(
            self.pending,
            invocation_elapsed_seconds=self.clock() - self.invocation_started,
            update_elapsed_seconds=self.clock() - self.update_started,
            checkpoint_seconds=self.checkpoint_seconds,
            validation_seconds=self.validation_seconds,
        )
        self.logger.append(self.pending)
        self.pending = None
        self.checkpoint_seconds = 0.0
        self.validation_seconds = 0.0

    def start_update(self) -> None:
        """Mark the start of an update and of its training work."""

        self.update_started = self.clock()
        self.training_started = self.clock()
        self.checkpoint_seconds = 0.0
        self.validation_seconds = 0.0

    def finish_training(self) -> None:
        """Record how long the update's collection plus PPO took."""

        self.training_seconds = self.clock() - self.training_started

    def prepare_row(
        self,
        state: _TrainingState,
        update_environment_steps: int,
        profiler: PhaseProfiler | None,
    ) -> None:
        """Prepare this update's row once its scheduled work is done.

        The row stays pending (see the class docstring). Detailed profiling
        phases stay zero in ordinary runs. The counters restart at zero, so
        from here on they hold only later work, such as a ceiling validation
        found on the next pass.
        """

        invocation_elapsed = self.clock() - self.invocation_started
        profile_seconds = profiler.snapshot_and_reset() if profiler is not None else None
        self.pending = update_timing(
            update=state.completed_updates,
            environment_steps=state.environment_steps,
            invocation_start_steps=self.invocation_start_steps,
            invocation_elapsed_seconds=invocation_elapsed,
            update_elapsed_seconds=self.clock() - self.update_started,
            update_environment_steps=update_environment_steps,
            training_seconds=self.training_seconds,
            checkpoint_seconds=self.checkpoint_seconds,
            validation_seconds=self.validation_seconds,
            profile_seconds=profile_seconds,
        )
        self.checkpoint_seconds = 0.0
        self.validation_seconds = 0.0

    def finish(self, *, final_checkpoint_seconds: float) -> None:
        """Write the last row when training stops, with the final save's time.

        The final save may create a new checkpoint or verify and reuse one
        already written by validation; either way its time belongs to the
        last update.
        """

        if self.pending is None:
            return
        self.pending = include_late_timing(
            self.pending,
            invocation_elapsed_seconds=self.clock() - self.invocation_started,
            update_elapsed_seconds=self.clock() - self.update_started,
            checkpoint_seconds=self.checkpoint_seconds + final_checkpoint_seconds,
            validation_seconds=self.validation_seconds,
        )
        self.logger.append(self.pending)


def _save_checkpoint(
    run: _RunSetup,
    state: _TrainingState,
    *,
    purpose: str,
    force_new: bool = False,
) -> tuple[Path, str]:
    """Save the current training state as a checkpoint; return its path and SHA-256."""

    return _save_training_checkpoint(
        run.destination,
        run.config,
        state.policy,
        state.optimizer,
        schedule_values(run.config, state.environment_steps),
        state.action_generator,
        state.shuffle_generator,
        run.environment_metadata,
        state.environment_steps,
        state.completed_updates,
        state.next_validation_step,
        state.validation_history,
        state.tracker,
        state.episodes,
        run.environment.cursor,
        state.saved_checkpoints,
        purpose=purpose,
        force_new=force_new,
    )


def _check_phase_ceiling(
    run: _RunSetup,
    state: _TrainingState,
    timer: _UpdateTimer,
) -> str:
    """Decide what happens when no full rollout fits below the phase ceiling.

    Returns ``_TRAIN_UPDATE`` when another update still fits. Otherwise the
    final weights are validated once, unless the last record already judged
    them (an earlier bounded invocation can stop after the final update but
    before this check). A pass ends training (release) or switches to the
    release phase and its larger ceiling (pipeline, ``_NEXT_PASS``); a fail
    ends training with the phase's failure reason. Rollouts are never
    shortened just to hit an exact limit.
    """

    config = run.config
    if can_collect_full_rollout(state.environment_steps, config.batch_size, state.phase.ceiling):
        return _TRAIN_UPDATE

    already_validated, last_validation_boundary = _last_validation_boundary(
        state.validation_history, state.phase, state.environment_steps
    )
    if state.completed_updates > 0 and not already_validated:
        if _run_and_record_validation(run, state, timer, state.phase, state.environment_steps):
            if state.phase.gate_name == RELEASE_READINESS_GATE:
                state.reason = "release readiness gate passed"
                return _STOP_TRAINING
            state.phase = _training_phase(config, RELEASE_READINESS_GATE)
            state.reason = "pipeline gate passed; release readiness active"
            return _NEXT_PASS

    if state.phase.gate_name == RELEASE_READINESS_GATE:
        state.reason = "release readiness ceiling reached"
        _record_release_ceiling_failure(
            run.destination,
            config,
            state.environment_steps,
            state.completed_updates,
            last_validation_boundary,
        )
    else:
        state.reason = "initial pipeline gate failed"
    return _STOP_TRAINING


def _train_one_update(
    run: _RunSetup,
    state: _TrainingState,
    timer: _UpdateTimer,
) -> CollectedRollout:
    """Step 6: play one rollout, run PPO on it, and record the update's metrics.

    Counters advance only after PPO has finished, so a failed update never
    counts. Returns the collected rollout, which the timing row needs.
    """

    config = run.config
    # Schedules are fixed now and reused by every optimizer minibatch in this
    # update. AdamW reads its learning rate from each parameter group.
    scheduled = schedule_values(config, state.environment_steps)
    for group in state.optimizer.param_groups:
        group["lr"] = scheduled.learning_rate

    # The legacy metric times collection plus PPO with its injectable clock.
    # The coarse timing clock separately covers the complete update
    # lifecycle, including metrics, checkpoints, and validation.
    started = run.clock()
    timer.start_update()
    collected, update_metrics, illegal_action_count = _collect_and_update(
        run.environment,
        state.policy,
        state.optimizer,
        state.episodes,
        config,
        scheduled,
        state.action_generator,
        state.shuffle_generator,
        run.profiler,
        state.completed_updates + 1,
    )

    # Only now has one update completed successfully. Advance counters, keep
    # the next observation for every live game, and summarize gameplay
    # without counting validation or failed requests.
    duration = run.clock() - started
    timer.finish_training()
    state.environment_steps += collected.batch.transition_count
    state.completed_updates += 1
    state.episodes = collected.next_episodes
    # Episode summaries use the same code in both modes; profiling merely
    # surrounds it with synchronized timestamps when enabled.
    with profile_phase(run.profiler, "episode_metrics"):
        gameplay = state.tracker.consume(collected.batch)

    _log_update(
        run.logger,
        state.completed_updates,
        state.environment_steps,
        collected.batch,
        gameplay,
        update_metrics,
        duration,
        illegal_action_count,
    )
    return collected


def _validate_and_decide(
    run: _RunSetup,
    state: _TrainingState,
    timer: _UpdateTimer,
    collected: CollectedRollout,
) -> str:
    """Step 7: save or validate after an update, then decide whether to go on.

    Returns ``_STOP_TRAINING`` (with ``state.reason`` set) or ``_NEXT_PASS``.
    """

    config = run.config
    # After an update, two saves can be due: the periodic save (every
    # checkpoint_every_updates updates) and a validation (at a 5M-turn boundary,
    # or on the last update that fits below the phase's step limit). Validation
    # saves its own checkpoint after adding its result to the history. So when
    # validation is due, skip the periodic save: otherwise this update would be
    # saved first without the result, and that file, which is saved only once
    # per update, would then be reused without it.
    phase_exhausted = not can_collect_full_rollout(
        state.environment_steps,
        config.batch_size,
        state.phase.ceiling,
    )
    # Validation is due in either of two cases: training has reached the next
    # scheduled boundary (every 5M turns), or the phase is used up (another
    # full update would go past its step limit, so these are the last weights
    # this phase will produce and must be tested now).
    validation_due = state.environment_steps >= state.next_validation_step or phase_exhausted
    # Periodic save: every checkpoint_every_updates updates (e.g. every 50th),
    # the "%" (remainder) is 0. Skipped when validation is due, because
    # validation saves this update itself, with its result included.
    if state.completed_updates % config.checkpoint_every_updates == 0 and not validation_due:
        with timer.checkpointing():
            _save_checkpoint(run, state, purpose="periodic")

    # One rollout cannot cross more than one configured boundary. When the
    # phase ceiling falls between scheduled boundaries, validate the final
    # trained weights at their actual step count before failing.
    gate_passed = False
    validation_phase = state.phase
    if validation_due:
        # Pick the boundary this validation is filed under. A scheduled check
        # uses the scheduled boundary (e.g. 5,000,000), even though the update
        # may have stopped a little past it, since updates come in fixed-size
        # batches. Otherwise the phase is used up between boundaries, and the
        # check uses the actual step count reached.
        if state.environment_steps >= state.next_validation_step:
            validation_boundary = state.next_validation_step
            # Move the cursor before checkpointing so a future resume knows
            # this scheduled boundary was already evaluated.
            state.next_validation_step += config.pipeline_gate_every_steps
        else:
            validation_boundary = state.environment_steps

        gate_passed = _run_and_record_validation(
            run, state, timer, validation_phase, validation_boundary
        )
        if gate_passed and validation_phase.gate_name == PIPELINE_GATE:
            # The next update keeps the same learner state but uses the
            # cumulative ceiling and heuristic validation gate.
            state.phase = _training_phase(config, RELEASE_READINESS_GATE)

    # A supplied review reason authorizes continuation beyond 25M.
    if (
        state.phase.gate_name == PIPELINE_GATE
        and state.environment_steps >= config.review_steps
        and run.review_reason
        and not state.review_recorded
    ):
        _write_review(
            run.destination, state.environment_steps, state.completed_updates, run.review_reason
        )
        state.review_recorded = True

    # Prepare timing only after this update's scheduled checkpoint,
    # validation, and review work has finished.
    timer.prepare_row(state, collected.batch.transition_count, run.profiler)

    if gate_passed:
        if validation_phase.gate_name == RELEASE_READINESS_GATE:
            state.reason = "release readiness gate passed"
            return _STOP_TRAINING
        # Passing ends only the initial phase. Collection continues on the
        # next pass, and the next scheduled 5M boundary receives the first
        # release-readiness validation.
        state.reason = "pipeline gate passed; release readiness active"
        return _NEXT_PASS

    # All ceiling exits return through the single no-fit check at the top of
    # the loop. If this update already ran the final gate, that check reuses
    # its recorded boundary instead of evaluating twice.
    if phase_exhausted:
        return _NEXT_PASS

    # A failed release gate does not invoke initial-stage review rules.
    if validation_phase.gate_name == RELEASE_READINESS_GATE:
        return _NEXT_PASS

    # Initial-stage stop decisions remain unchanged when its gate has not
    # passed: respect the 25M review and 50M diagnostic ceiling.
    stage_reason = initial_stage_decision(
        config,
        state.environment_steps,
        review_reason=run.review_reason,
    )
    if stage_reason is None:
        return _NEXT_PASS
    state.reason = stage_reason
    if stage_reason == "review required":
        _write_review_required(
            run.destination, state.environment_steps, state.completed_updates, stage_reason
        )
    return _STOP_TRAINING


def _open_run_directory(
    config: RunConfig,
    run_directory: str | Path,
    resume_checkpoint: str | Path | None,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    action_generator: torch.Generator,
    shuffle_generator: torch.Generator,
) -> tuple[Path, _ResumePoint | None]:
    """Create a new run directory, or verify an existing one and restore its checkpoint.

    A resume restores the policy, optimizer, and both random streams in place.
    Nothing here starts Go.
    """

    if resume_checkpoint is None:
        # A new run directory is created once and never reused. Its configuration
        # is written before training so even an early failure retains its identity.
        return _prepare_run_directory(run_directory, config), None

    # Directory and readable metadata checks happen before deserialization.
    # The model signature passed here lets load_checkpoint reject weights for
    # another architecture before invoking Torch's pickle-based loader.
    checkpoint_path = Path(resume_checkpoint)
    destination = _prepare_resume_directory(
        run_directory,
        config,
        checkpoint_path,
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_config=config,
        expected_model_state=policy.state_dict(),
    )
    # The loop validates a due boundary and moves next_validation_step
    # forward before saving that update, so no stopped or periodic save
    # leaves a boundary due. A checkpoint that has one was saved by an
    # older lifecycle that skipped validation; resuming it would judge the
    # boundary only after another update, not the saved weights. Refuse it
    # here, before Go starts or any file is written.
    if checkpoint.environment_steps >= checkpoint.next_validation_step:
        raise ValueError(
            "checkpoint has an overdue validation boundary; start a new run"
        )
    # load_checkpoint has already hashed every byte. Read its verified
    # sidecar value for audit records instead of hashing the file again.
    checkpoint_fingerprint = recorded_checkpoint_fingerprint(checkpoint_path)
    abandoned_metric_state = _restore_training_checkpoint(
        checkpoint,
        config,
        policy,
        optimizer,
        action_generator,
        shuffle_generator,
    )
    return destination, _ResumePoint(
        checkpoint, checkpoint_path, checkpoint_fingerprint, abandoned_metric_state
    )


def _open_metric_logs(
    destination: Path,
    completed_updates: int,
    *,
    resumed: bool,
    max_updates: int | None,
) -> tuple[CSVMetricLogger, CSVTimingLogger]:
    """Open both CSV logs and check they end where the run resumes.

    Metrics append one row after every completed PPO update. A resume must
    find the CSV ending at exactly the checkpoint boundary: an older or newer
    CSV would mix two different experiment histories. timings.csv has its own
    schema so adding performance measurements cannot make an existing
    metrics.csv unreadable. Old runs may legitimately have no timing rows; a
    resume begins recording at its next completed update.
    """

    logger = CSVMetricLogger(destination / "metrics.csv")
    timing_logger = CSVTimingLogger(destination / "timings.csv")
    if resumed and max_updates is not None and max_updates <= completed_updates:
        raise ValueError(
            "max_updates must be greater than the checkpoint's completed updates"
        )
    if logger.last_update != completed_updates:
        raise ValueError(
            f"metrics end at update {logger.last_update}, "
            f"but checkpoint is update {completed_updates}"
        )
    if timing_logger.last_update > completed_updates:
        raise ValueError(
            f"timings end at update {timing_logger.last_update}, "
            f"after checkpoint update {completed_updates}"
        )
    return logger, timing_logger


def _restored_phase(
    config: RunConfig,
    validation_history: Sequence[Mapping[str, Any]],
) -> tuple[TrainingPhase, bool, Mapping[str, Any] | None, str]:
    """Work out the active phase from validation history.

    Returns the phase, whether the pipeline gate already passed, the latest
    passing release record (if any), and the stop reason to report if the
    phase's ceiling is reached. Once the pipeline gate has passed, every later
    boundary uses the release gate and cumulative ceiling.
    """

    pipeline_previously_passed = any(
        record.get("passed") is True
        and record.get("gate", PIPELINE_GATE) == PIPELINE_GATE
        for record in validation_history
    )
    previous_passing_release = next(
        (
            record
            for record in reversed(validation_history)
            if record.get("passed") is True
            and record.get("gate") == RELEASE_READINESS_GATE
        ),
        None,
    )
    phase = _training_phase(
        config,
        RELEASE_READINESS_GATE if pipeline_previously_passed else PIPELINE_GATE,
    )
    reason = (
        "release readiness ceiling reached"
        if phase.gate_name == RELEASE_READINESS_GATE
        else "initial pipeline gate failed"
    )
    return phase, pipeline_previously_passed, previous_passing_release, reason


def _require_review_reason(
    config: RunConfig,
    environment_steps: int,
    *,
    resumed: bool,
    pipeline_previously_passed: bool,
    review_reason: str | None,
) -> None:
    """Refuse a resume at or past the 25M review boundary without a stated reason.

    The rule applies only in the initial stage, and only while another full
    rollout still fits below its ceiling.
    """

    if (
        resumed
        and environment_steps >= config.review_steps
        and can_collect_full_rollout(
            environment_steps,
            config.batch_size,
            config.initial_step_ceiling,
        )
        and not pipeline_previously_passed
        and not review_reason
    ):
        raise ValueError(
            "resuming at or beyond the review boundary requires --review-reason"
        )


def _create_training_environment(
    environment: EnvClient,
    config: RunConfig,
    seed_cursor: int,
) -> dict[str, Any]:
    """Configure Go once for the whole process and return its metadata.

    Create declares the slot count, training-only seed interval, scripted
    opponent, and deterministic schedule, but does not start any games. The
    configured player count chooses cmd/env's maps, and create checks that Go
    built the generator for that count.
    """

    environment_metadata = dict(
        environment.create(
            config.environment_count,
            (config.training_seeds.start, config.training_seeds.end),
            config.training_opponent,
            schedule=config.training_schedule,
            cursor=seed_cursor,
        )
    )
    # Compare Go's response with the Python configuration before trusting
    # its observations. A mismatch would make later replay metadata false.
    _validate_training_metadata(config, environment_metadata)
    # Say which code and opponent this run uses, on the first lines of the
    # console. A run started from a stale copy shows here at once: the
    # engine reads "development" or an old commit, and the opponent is not
    # the one expected.
    print(
        f"environment: engine={environment_metadata['engine']} "
        f"opponent={environment_metadata['opponent']} "
        f"generator={environment_metadata['generator']}",
        flush=True,
    )
    return environment_metadata


def _record_resumed_run(
    destination: Path,
    config: RunConfig,
    resume: _ResumePoint,
    environment_metadata: Mapping[str, Any],
    *,
    environment_steps: int,
    completed_updates: int,
    review_reason: str | None,
    review_recorded: bool,
    allow_development_engine: bool,
) -> bool:
    """Check the resumed engine, record the resume, and save a review reason.

    Returns whether review.json now exists.
    """

    _validate_resumed_environment(
        resume.checkpoint.environment_metadata,
        environment_metadata,
        resume.checkpoint.seed_cursor,
        allow_development=allow_development_engine,
    )
    # Persist the discarded game identities before asking Go to replace
    # them. Their already-trained turns remain in environment_steps; the
    # fresh games begin with zero per-episode metric totals.
    _record_resume(
        destination,
        resume.path,
        resume.fingerprint,
        resume.checkpoint,
        review_reason,
        resume.abandoned_metric_state,
        engine=environment_metadata["engine"],
        allow_development_engine=allow_development_engine,
    )
    if environment_steps >= config.review_steps and review_reason and not review_recorded:
        _write_review(destination, environment_steps, completed_updates, review_reason)
        return True
    return review_recorded


def _write_review(
    destination: Path,
    environment_steps: int,
    completed_updates: int,
    review_reason: str,
) -> None:
    """Save the reason to continue past the 25M review boundary.

    It is written once, as experiment evidence, instead of leaving the decision
    in a terminal session or relying on memory.
    """

    write_json_atomic(
        destination / "review.json",
        {
            "environment_steps": environment_steps,
            "completed_updates": completed_updates,
            "reason_to_continue": review_reason,
        },
    )


def _write_review_required(
    destination: Path,
    environment_steps: int,
    completed_updates: int,
    reason: str,
) -> None:
    """Mark why training paused and which completed update should be reviewed."""

    write_json_atomic(
        destination / "review-required.json",
        {
            "environment_steps": environment_steps,
            "completed_updates": completed_updates,
            "reason": reason,
        },
    )


def _validation_progress_printer(
    phase: TrainingPhase,
    validation_boundary: int,
) -> ProgressCallback:
    """Return the callback that prints a validation's periodic progress line.

    A release validation runs for many minutes. A periodic line shows it is
    alive, and the decide/environment split shows whether Python or the Go
    environment is the slower side.
    """

    def print_progress(progress: ValidationProgress) -> None:
        print(
            f"{phase.progress_label} progress: "
            f"boundary={validation_boundary} "
            f"finished={progress.finished}/{progress.total} "
            f"active={progress.active} "
            f"elapsed={progress.elapsed_seconds:.0f}s "
            f"decide={progress.decide_seconds:.0f}s "
            f"environment={progress.environment_seconds:.0f}s",
            flush=True,
        )

    return print_progress


def _print_validation_result(
    phase: TrainingPhase,
    record: Mapping[str, Any],
    result: EvaluationResult,
    validation_boundary: int,
) -> None:
    """Print one finished validation: wins, behaviour, and pass or fail.

    How the games looked, not only who won: a policy that has stopped
    launching shows up in zero_launch_games before its win count falls. The
    release gate also prints its confidence lower bound.
    """

    summary = record["summary"]
    behaviour = (
        f"mean_launches={summary['launches_per_game']['mean']:.1f} "
        f"zero_launch_games={summary['zero_launch_games']} "
        f"mean_episode_turns={summary['episode_turns']['mean']:.1f} "
        f"eliminations={summary['eliminations']} "
    )
    lower_bound = (
        f"lower_bound={record['confidence_interval']['lower_bound']:.6f} "
        if phase.gate_name == RELEASE_READINESS_GATE
        else ""
    )
    print(
        f"{phase.progress_label} complete: "
        f"boundary={validation_boundary} "
        f"wins={record['wins']}/{record['games']} "
        f"{lower_bound}"
        f"{behaviour}"
        f"rejected_actions={result.rejected_actions} "
        f"passed={record['passed']}",
        flush=True,
    )


def _publish_validation_record(
    destination: Path,
    frozen_archive_root: Path,
    config: RunConfig,
    phase: TrainingPhase,
    record: dict[str, Any],
    checkpoint_path: Path,
    fingerprint: str,
    completed_updates: int,
    validation_boundary: int,
) -> Mapping[str, Any]:
    """Link a validation record to its checkpoint and write its evidence file.

    The record is updated in place: it is the same object as the entry in
    validation history, so later checkpoints keep the link too. A passing
    release record is published with its frozen archive; training reports
    success only after both are durable. Returns the record as written.
    """

    record["checkpoint"] = checkpoint_path.name
    record["checkpoint_sha256"] = fingerprint
    record["completed_updates"] = completed_updates
    if phase.gate_name == RELEASE_READINESS_GATE and record["passed"] is True:
        return _publish_passing_release(
            destination,
            frozen_archive_root,
            config,
            record,
            checkpoint_path,
            fingerprint,
            completed_updates,
        )
    write_json_atomic(
        destination / f"{phase.evidence_prefix}-{validation_boundary:09d}.json",
        record,
    )
    return record


def _run_and_record_validation(
    run: _RunSetup,
    state: _TrainingState,
    timer: _UpdateTimer,
    phase: TrainingPhase,
    validation_boundary: int,
) -> bool:
    """Play one phase's validation games, then save and publish the result.

    Scheduled checks and final ceiling checks both use this function. The
    phase supplies the matching evaluator, checkpoint purpose, and evidence
    filename prefix; this function does the shared work in a fixed order:

    1. Play the validation games. They use a copy of the learned weights, so
       the policy and its optimizer are not changed.
    2. Add the result to the validation history and save a checkpoint that
       contains it. On the update a run resumed from, a new checkpoint is
       written even though that update was already saved.
    3. Link the record to that checkpoint and write its evidence file.

    The time spent playing and saving is added to ``timer``. Returns whether
    the gate passed.
    """

    config = run.config
    # Rotated validation can take noticeable time. Announce the complete
    # sample before starting so a quiet console is not mistaken for a stall.
    print(
        f"{phase.progress_label} progress: "
        f"boundary={validation_boundary} "
        f"playing={phase.game_count} games",
        flush=True,
    )
    with timer.validating():
        result = phase.validation(
            state.policy,
            config,
            run.command,
            run.validation_runner,
            on_progress=_validation_progress_printer(phase, validation_boundary),
        )
    record = _validation_record(
        phase,
        result,
        config,
        state.environment_steps,
        validation_boundary,
    )
    _print_validation_result(phase, record, result, validation_boundary)

    state.validation_history.append(record)
    resumed_update = (
        run.resume is not None
        and state.completed_updates == run.resume.checkpoint.completed_updates
    )
    with timer.checkpointing():
        checkpoint_path, fingerprint = _save_checkpoint(
            run, state, purpose=phase.checkpoint_purpose, force_new=resumed_update
        )
    # Link readable evidence to the exact immutable checkpoint that
    # contains this validation in its durable history.
    published = _publish_validation_record(
        run.destination,
        run.frozen_archive_root,
        config,
        phase,
        record,
        checkpoint_path,
        fingerprint,
        state.completed_updates,
        validation_boundary,
    )
    return bool(published["passed"])


def _record_release_ceiling_failure(
    destination: Path,
    config: RunConfig,
    environment_steps: int,
    completed_updates: int,
    validation_boundary: int,
) -> None:
    """Record terminal failure without selecting a protected eval pool."""

    write_json_atomic(
        destination / "release-readiness-failed.json",
        {
            "format": "release-readiness-failure-v1",
            "reason": "release readiness ceiling reached",
            "environment_steps": environment_steps,
            "completed_updates": completed_updates,
            "cumulative_step_ceiling": config.cumulative_step_ceiling,
            "last_validation_boundary": validation_boundary,
            "last_validation": f"release-{validation_boundary:09d}.json",
        },
    )


def _last_validation_boundary(
    validation_history: Sequence[Mapping[str, Any]],
    phase: TrainingPhase,
    environment_steps: int,
) -> tuple[bool, int]:
    """Whether the current weights were already validated, and at which boundary.

    An earlier bounded invocation can stop after training the final weights
    but before judging the phase ceiling. A scheduled gate can be crossed by a
    complete rollout, so its filename may use an earlier cadence boundary than
    the resulting environment-step count; the boundary from the existing
    record is kept when no new validation is needed.
    """

    last_gate = next(
        (
            record
            for record in reversed(validation_history)
            if record.get("gate", PIPELINE_GATE) == phase.gate_name
        ),
        None,
    )
    if last_gate is None or last_gate.get("training_environment_steps") != environment_steps:
        return False, environment_steps
    recorded_boundary = last_gate.get("validation_boundary")
    if (
        isinstance(recorded_boundary, bool)
        or not isinstance(recorded_boundary, int)
        or recorded_boundary < 0
    ):
        raise ValueError("validation history has an invalid validation boundary")
    return True, recorded_boundary


def _collect_and_update(
    environment: EnvClient,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    episodes: Sequence[Episode],
    config: RunConfig,
    scheduled: ScheduleValues,
    action_generator: torch.Generator,
    shuffle_generator: torch.Generator,
    profiler: PhaseProfiler | None,
    update_number: int,
) -> tuple[CollectedRollout, PPOUpdateMetrics, int]:
    """Collect one fixed-policy rollout, then run PPO on the complete batch.

    Collection advances all Go games while policy weights remain fixed. It
    returns observations, sampled actions, old log probabilities, rewards,
    critic values, GAE advantages, and return targets. Returns the collected
    rollout, the PPO metrics, and the rejected-action count, which is always
    zero: any rejection stops training before its transitions update weights.
    """

    print(
        f"update={update_number} progress: collecting "
        f"{config.batch_size} transitions "
        f"({config.rollout_steps} turns across "
        f"{config.environment_count} environments)",
        flush=True,
    )
    errors_before = environment.error_count
    collected = collect_rollout(
        environment,
        policy,
        episodes,
        steps=config.rollout_steps,
        gamma=config.gamma,
        lam=config.gae_lambda,
        generator=action_generator,
        profiler=profiler,
    )

    # Any structured Go rejection means the Python action path violated
    # the authoritative rules. Stop before those transitions can update weights.
    illegal_action_count = environment.error_count - errors_before
    if illegal_action_count != 0:
        raise RuntimeError(
            f"rollout contained {illegal_action_count} rejected actions"
        )

    # PPO normalizes all rollout advantages once, shuffles transitions
    # into minibatches for every epoch, calculates actor and critic
    # losses, and changes the policy weights through optimizer steps.
    minibatches_per_epoch = config.batch_size // config.minibatch_size
    optimizer_steps = config.ppo_epochs * minibatches_per_epoch
    print(
        f"update={update_number} progress: rollout complete; running "
        f"{optimizer_steps} optimizer steps "
        f"({config.ppo_epochs} epochs x {minibatches_per_epoch} minibatches)",
        flush=True,
    )
    # The no-op context used in ordinary runs keeps this as one PPO
    # call. Profiling can therefore measure it without creating a
    # second implementation that might later receive different args.
    with profile_phase(profiler, "ppo"):
        update_metrics = ppo_update(
            policy,
            optimizer,
            collected.batch,
            epochs=config.ppo_epochs,
            minibatch_size=config.minibatch_size,
            clip_epsilon=config.clip_epsilon,
            value_coefficient=config.value_coefficient,
            launch_entropy_coefficient=scheduled.launch_entropy,
            target_entropy_coefficient=scheduled.target_entropy,
            max_gradient_norm=config.max_gradient_norm,
            shuffle_generator=shuffle_generator,
        )
    return collected, update_metrics, illegal_action_count


def _log_update(
    logger: CSVMetricLogger,
    completed_updates: int,
    environment_steps: int,
    batch: RolloutBatch,
    gameplay: GameplayMetrics,
    update_metrics: PPOUpdateMetrics,
    duration: float,
    illegal_action_count: int,
) -> None:
    """Write one update's metrics row and print a short summary of it.

    The row combines game behaviour with PPO diagnostics; the printed subset
    lets a person monitor a long run.
    """

    metrics = _training_metrics(
        completed_updates,
        environment_steps,
        batch,
        gameplay,
        update_metrics,
        duration,
        illegal_action_count,
    )
    logger.append(metrics)
    print(
        f"update={metrics.update} steps={metrics.environment_steps} "
        f"wins={metrics.wins} losses={metrics.losses} draws={metrics.draws} "
        f"policy_loss={metrics.policy_loss:.6f} "
        f"value_loss={metrics.value_loss:.6f} "
        f"steps_per_second={metrics.environment_steps_per_second:.1f}",
        flush=True,
    )


def _restore_training_checkpoint(
    checkpoint: CheckpointState,
    config: RunConfig,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    action_generator: torch.Generator,
    shuffle_generator: torch.Generator,
) -> dict[str, dict[int, int]]:
    """Restore every learner-owned value saved at one completed update boundary.

    Validation happens before mutation where possible. Live Go games are not part
    of this state: the caller records and replaces them after starting a compatible
    environment process at the saved schedule cursor.
    """

    if checkpoint.environment_steps != checkpoint.completed_updates * config.batch_size:
        raise ValueError(
            "checkpoint environment steps do not match its completed PPO updates"
        )
    expected_schedule = asdict(schedule_values(config, checkpoint.environment_steps))
    if dict(checkpoint.schedule_state) != expected_schedule:
        raise ValueError("checkpoint schedule state does not match its step counter")
    required_random_states = {
        "python",
        "numpy",
        "torch_cpu",
        "action_sampling",
        "minibatch_shuffle",
    }
    missing_random_states = required_random_states - checkpoint.random_states.keys()
    if missing_random_states:
        raise ValueError(
            "checkpoint random states missing "
            f"{sorted(missing_random_states)}"
        )
    environment_ids = tuple(
        episode.environment_id for episode in checkpoint.active_episodes
    )
    if environment_ids != tuple(range(config.environment_count)):
        raise ValueError(
            "checkpoint active episodes must contain every environment exactly once"
        )
    abandoned_metric_state = _abandoned_metric_state(
        checkpoint.metric_state,
        environment_ids,
    )

    policy.load_state_dict(checkpoint.model_state, strict=True)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    restore_random_states(checkpoint.random_states)
    action_generator.set_state(checkpoint.random_states["action_sampling"])
    shuffle_generator.set_state(checkpoint.random_states["minibatch_shuffle"])
    return abandoned_metric_state


def _validation_history(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a mutable copy of the checkpoint's previously completed validations."""

    if set(state) != {"history"} or not isinstance(state["history"], list):
        raise ValueError("checkpoint validation state has an invalid structure")
    history: list[dict[str, Any]] = []
    for record in state["history"]:
        if not isinstance(record, Mapping):
            raise ValueError("checkpoint validation history contains a non-object record")
        history.append(dict(record))
    return history


def _abandoned_metric_state(
    state: Mapping[str, Any],
    environment_ids: Sequence[int],
) -> dict[str, dict[int, int]]:
    """Validate and copy per-game totals that end when resume abandons live games."""

    if set(state) != {"launches_by_environment"}:
        raise ValueError("checkpoint metric state has an invalid structure")
    launches = state["launches_by_environment"]
    if not isinstance(launches, Mapping) or set(launches) != set(environment_ids):
        raise ValueError("checkpoint metric state does not match its environments")
    copied: dict[int, int] = {}
    for environment_id in environment_ids:
        value = launches[environment_id]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("checkpoint launch totals must be non-negative integers")
        copied[environment_id] = value
    return {"launches_by_environment": copied}


def _engine_is_unidentified(version: object) -> bool:
    """True when an engine version does not name exact code.

    ``development`` means Go found no commit at all: a ``go run`` from a local
    checkout reports it. A ``+dirty`` suffix names a commit plus uncommitted
    changes nobody recorded. Two equal strings of either kind do not prove the
    code was the same.
    """

    return version == DEVELOPMENT_ENGINE or (
        isinstance(version, str) and version.endswith(DIRTY_ENGINE_SUFFIX)
    )


def _validate_resumed_environment(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    cursor: int,
    *,
    allow_development: bool = False,
) -> None:
    """Require the new Go process to match the one recorded by the checkpoint.

    Matching metadata only proves matching code when the engine version names
    exact code. A Vast.ai bundle reports ``<tag>+<commit>`` stamped by git
    archive, so its resumes are checked fully. A run from a local checkout
    reports ``development``; resuming it is refused unless
    ``allow_development`` is set, in which case the equal-strings check below
    is all that is enforced and the choice is recorded in the resume audit.
    """

    if not allow_development:
        for side, version in (
            ("checkpoint", saved.get("engine")),
            ("current", current.get("engine")),
        ):
            if _engine_is_unidentified(version):
                raise ValueError(
                    f"cannot resume: the {side} engine version {version!r} does not "
                    "name exact code, so matching versions would not prove matching "
                    "code; resume from a git-archive bundle, or pass "
                    "--allow-development to resume a local run anyway"
                )

    if current.get("cursor") != cursor:
        raise ValueError(
            f"resumed environment cursor is {current.get('cursor')}, want {cursor}"
        )
    keys = {
        "v",
        "engine",
        "schema",
        "generator",
        "opponent",
        "seed_range",
        "schedule",
        "dimensions",
    }
    mismatches = {
        key: {"saved": saved.get(key), "current": current.get(key)}
        for key in sorted(keys)
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        raise ValueError(
            f"resumed Go environment does not match checkpoint metadata: {mismatches}"
        )


def _record_resume(
    run_directory: Path,
    checkpoint_path: Path,
    fingerprint: str,
    checkpoint: CheckpointState,
    review_reason: str | None,
    abandoned_metric_state: Mapping[str, Mapping[int, int]],
    *,
    engine: str,
    allow_development_engine: bool,
) -> Path:
    """Record discarded games and the exact boundary from which training restarts.

    ``engine`` is the resumed Go process's version, and
    ``allow_development_engine`` says whether an unidentified version was
    allowed through, so a later reader can see which resumes were not proven
    to run the same code.
    """

    manifest_path = run_directory / "manifest.json"
    manifest = read_json_object(manifest_path, "run manifest")
    resumes = manifest.setdefault("resumes", [])
    if not isinstance(resumes, list):
        raise ValueError("run manifest resumes must be a list")

    sequence = len(resumes) + 1
    audit_path = run_directory / f"resume-{sequence:03d}.json"
    while audit_path.exists():
        sequence += 1
        audit_path = run_directory / f"resume-{sequence:03d}.json"
    checkpoint_name = str(checkpoint_path.resolve().relative_to(run_directory.resolve()))
    record = {
        "format": "resume-audit-v1",
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": fingerprint,
        "environment_steps": checkpoint.environment_steps,
        "completed_updates": checkpoint.completed_updates,
        "next_seed_cursor": checkpoint.seed_cursor,
        "review_reason": review_reason,
        "engine": engine,
        "allow_development_engine": allow_development_engine,
        "abandoned_episodes": [
            asdict(episode) for episode in checkpoint.active_episodes
        ],
        "abandoned_metric_state": {
            name: dict(values) for name, values in abandoned_metric_state.items()
        },
    }
    write_json_atomic(audit_path, record)
    resumes.append(
        {
            "file": audit_path.name,
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": fingerprint,
            "environment_steps": checkpoint.environment_steps,
            "completed_updates": checkpoint.completed_updates,
            "next_seed_cursor": checkpoint.seed_cursor,
        }
    )
    write_json_atomic(manifest_path, manifest)
    return audit_path


def _validate_training_metadata(
    config: RunConfig,
    metadata: Mapping[str, Any],
) -> None:
    """Ensure Go created the exact environment described by the run config.

    Python sends requested versions and settings, but Go is authoritative about
    what process actually started. Its create response repeats the protocol,
    observation schema, generator, opponent, schedule, and seed interval. This
    comparison catches a stale binary or wiring mistake before any game turns
    enter a training rollout.
    """

    # These fields must equal their configured values exactly. In particular,
    # accepting a different seed range could leak validation boards into training.
    expected = {
        "v": config.protocol_version,
        "schema": config.observation_schema,
        "generator": config.generator_version,
        "opponent": config.training_opponent,
        "schedule": config.training_schedule,
        "seed_range": [config.training_seeds.start, config.training_seeds.end],
    }
    mismatches = {
        name: {"got": metadata.get(name), "want": value}
        for name, value in expected.items()
        if metadata.get(name) != value
    }
    if mismatches:
        raise ValueError(f"environment rejected the configured training contract: {mismatches}")

    # The engine version is supplied by Go rather than predicted by Python. It
    # still must be present because checkpoints need it to identify how game
    # results were produced. The observation dimensions must equal the sizes
    # the model was built for.
    if not isinstance(metadata.get("engine"), str) or not metadata["engine"].strip():
        raise ValueError("environment metadata is missing its engine version")
    if metadata.get("dimensions") != OBSERVATION_DIMENSIONS:
        raise ValueError(
            f"environment metadata dimensions are {metadata.get('dimensions')!r}, "
            f"want {OBSERVATION_DIMENSIONS!r}"
        )


def _run_pipeline_validation(
    policy: nn.Module,
    config: RunConfig,
    command: Sequence[str],
    validation_runner: ValidationRunner,
    *,
    on_progress: ProgressCallback | None = None,
) -> EvaluationResult:
    """Evaluate the current weights under the fixed pipeline-gate contract.

    Validation runs on the device the policy trains on, normally the GPU,
    under ``eval-policy-v2``. The runner evaluates its own private copy, so
    the model that continues training is never moved or modified.

    The pipeline gate uses the first configured number of validation seeds,
    plays every learner seat through the rotate schedule, and faces the fixed
    random opponent in every other seat. Expected metadata prevents results
    from a different Go contract from being accepted silently.
    """

    # SeedRange uses a half-open interval. Adding the sample count therefore
    # produces exactly pipeline_validation_seeds distinct boards.
    validation_end = config.validation_seeds.start + config.pipeline_validation_seeds
    validation_seeds = type(config.validation_seeds)(
        config.validation_seeds.start,
        validation_end,
    )
    expected_metadata = evaluation_metadata(
        config,
        validation_seeds,
        config.pipeline_opponent,
    )
    result = validation_runner(
        policy,
        command=command,
        seeds=validation_seeds,
        opponent=config.pipeline_opponent,
        evaluation_policy=config.evaluation_policy,
        device=policy_device(policy),
        batch_size=config.environment_count,
        expected_metadata=expected_metadata,
        # Validation plays maps with the same player count as the training run
        # whose weights it is judging.
        players=config.players,
        on_progress=on_progress,
    )
    return checked_evaluation_result(
        result,
        validation_seeds,
        expected_metadata,
        players=config.players,
        evaluation_policy=config.evaluation_policy,
        context="pipeline validation",
    )


def _run_release_readiness_validation(
    policy: nn.Module,
    config: RunConfig,
    command: Sequence[str],
    validation_runner: ValidationRunner,
    *,
    on_progress: ProgressCallback | None = None,
) -> EvaluationResult:
    """Play the fixed checkpoint-selection sample: 1,000 games on two players.

    Release readiness uses the same deterministic decoder, device and
    rotated-match runner as the small pipeline gate. Its larger sample uses
    the first 500 reusable validation boards from every seat (2,000 games on
    four-player maps) and replaces random-v1 with the canonical heuristic
    opponent, in every opponent seat, used by the final learning objective.
    """

    validation_seeds = type(config.validation_seeds)(
        config.validation_seeds.start,
        config.validation_seeds.start + config.release_validation_seeds,
    )
    expected_metadata = evaluation_metadata(
        config,
        validation_seeds,
        config.training_opponent,
    )
    result = validation_runner(
        policy,
        command=command,
        seeds=validation_seeds,
        opponent=config.training_opponent,
        evaluation_policy=config.evaluation_policy,
        device=policy_device(policy),
        batch_size=config.environment_count,
        expected_metadata=expected_metadata,
        # Release readiness belongs to this training run, so it inherits the
        # player count stored in the run configuration and its checkpoints.
        players=config.players,
        on_progress=on_progress,
    )

    return checked_evaluation_result(
        result,
        validation_seeds,
        expected_metadata,
        players=config.players,
        evaluation_policy=config.evaluation_policy,
        context="release validation",
    )


def _save_training_checkpoint(
    run_directory: Path,
    config: RunConfig,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduled: ScheduleValues,
    action_generator: torch.Generator,
    shuffle_generator: torch.Generator,
    environment_metadata: Mapping[str, Any],
    environment_steps: int,
    completed_updates: int,
    next_validation_step: int,
    validation_history: Sequence[Mapping[str, Any]],
    tracker: EpisodeMetricTracker,
    episodes: Sequence[Episode],
    seed_cursor: int,
    saved: dict[int, tuple[Path, str]],
    *,
    purpose: str,
    force_new: bool = False,
) -> tuple[Path, str]:
    """Persist everything needed to identify one completed update boundary.

    Checkpoints contain more than model weights. They also retain AdamW state,
    schedule values, global and dedicated RNG streams, environment versions,
    progress counters, validation history, unfinished metric totals, active-game
    audit identities, and the next ``train-v1`` assignment cursor.

    Multiple events can request a save after the same update—for example, a
    periodic interval and a clean stop. ``saved`` lets those events reuse one
    immutable file rather than overwrite it or store duplicate weight states.
    A validation performed immediately after resume is the exception: its new
    history must be persisted even though no optimizer update occurred, so
    ``force_new`` writes a revision beside the immutable resume checkpoint.
    The returned fingerprint is the SHA-256 of every checkpoint byte.
    """

    # A checkpoint already written for this update represents the same policy,
    # optimizer, and counters, so return its path and fingerprint unchanged.
    if completed_updates in saved and not force_new:
        return saved[completed_updates]

    # Global libraries own one RNG state each. Action sampling and minibatch
    # shuffling deliberately use separate Torch generators, so their states
    # must be added explicitly to the checkpoint as well.
    random_states = capture_random_states()
    random_states["action_sampling"] = action_generator.get_state()
    random_states["minibatch_shuffle"] = shuffle_generator.get_state()

    # The usual filename identifies the completed PPO update. If an immutable
    # file already represents that update, choose the first unused state
    # revision so validation history can advance without replacing old bytes.
    path = run_directory / "checkpoints" / f"update-{completed_updates:06d}.pt"
    if force_new or path.exists():
        revision = 1
        while True:
            candidate = (
                run_directory
                / "checkpoints"
                / f"update-{completed_updates:06d}-state-{revision:03d}.pt"
            )
            if not candidate.exists():
                path = candidate
                break
            revision += 1
    state = CheckpointState(
        config=config,
        model_state=policy.state_dict(),
        optimizer_state=optimizer.state_dict(),
        schedule_state={
            "learning_rate": scheduled.learning_rate,
            "launch_entropy": scheduled.launch_entropy,
            "target_entropy": scheduled.target_entropy,
        },
        random_states=random_states,
        environment_metadata=dict(environment_metadata),
        environment_steps=environment_steps,
        completed_updates=completed_updates,
        next_validation_step=next_validation_step,
        validation_state={"history": list(validation_history)},
        metric_state=tracker.state_dict(),
        # Live game state is deliberately not serialized. These identities and
        # turns record which games a later resume will abandon before resetting
        # each slot from the saved scheduler cursor.
        active_episodes=tuple(
            ActiveEpisodeAudit(
                environment_id=episode.env_id,
                seed=episode.seed,
                learner_seat=episode.seat,
                turn=int(getattr(episode, "turn", 0)),
            )
            for episode in episodes
        ),
        seed_cursor=seed_cursor,
    )
    fingerprint = save_checkpoint(path, state)
    saved[completed_updates] = (path, fingerprint)

    # The small JSON manifest makes checkpoint discovery possible without
    # loading every large Torch file and records why this checkpoint was saved.
    _append_checkpoint_manifest(
        run_directory,
        filename=str(path.relative_to(run_directory)),
        fingerprint=fingerprint,
        environment_steps=environment_steps,
        completed_updates=completed_updates,
        purpose=purpose,
    )
    return path, fingerprint


def _append_checkpoint_manifest(
    run_directory: Path,
    *,
    filename: str,
    fingerprint: str,
    environment_steps: int,
    completed_updates: int,
    purpose: str,
) -> None:
    """Add one checkpoint and its fingerprint to the run's JSON index.

    The binary checkpoint remains the source of learner state. The manifest is
    a small, readable list that lets people and later commands find checkpoint
    files, see their step/update boundaries, and verify their SHA-256 values.
    Rewriting it atomically prevents an interrupted write from leaving partial JSON.
    """

    path = run_directory / "manifest.json"
    # Load the existing index, append exactly one new immutable checkpoint
    # record, and replace the manifest through the atomic JSON helper.
    manifest = read_json_object(path, "run manifest")
    manifest["checkpoints"].append(
        {
            "file": filename,
            "sha256": fingerprint,
            "environment_steps": environment_steps,
            "completed_updates": completed_updates,
            "purpose": purpose,
        }
    )
    write_json_atomic(path, manifest)


def _training_metrics(
    update: int,
    environment_steps: int,
    rollout: RolloutBatch,
    gameplay: GameplayMetrics,
    update_metrics: PPOUpdateMetrics,
    duration: float,
    illegal_action_count: int,
) -> TrainingMetrics:
    """Combine game outcomes and PPO diagnostics into one validated CSV row.

    ``gameplay`` describes terminal results and action behavior observed during
    collection. ``update_metrics`` describes the optimization that followed.
    The rollout supplies critic predictions and return targets for explained
    variance, while wall-clock duration supplies collection-plus-update
    throughput. Validation games never pass through this function.
    """

    # Rates use only games that terminated during this rollout. A rollout can
    # finish before any game does, in which case all three rates are defined as zero.
    games = gameplay.wins + gameplay.losses + gameplay.draws
    return TrainingMetrics(
        update=update,
        environment_steps=environment_steps,
        wins=gameplay.wins,
        losses=gameplay.losses,
        draws=gameplay.draws,
        win_rate=gameplay.wins / games if games else 0.0,
        loss_rate=gameplay.losses / games if games else 0.0,
        draw_rate=gameplay.draws / games if games else 0.0,
        mean_episode_length=gameplay.mean_episode_length,
        mean_ship_advantage=gameplay.mean_ship_advantage,
        policy_loss=update_metrics.policy_loss,
        value_loss=update_metrics.value_loss,
        explained_variance=explained_variance(rollout.values, rollout.return_targets),
        launch_entropy=update_metrics.launch_entropy,
        target_entropy=update_metrics.target_entropy,
        approximate_kl=update_metrics.approximate_kl,
        clip_fraction=update_metrics.clip_fraction,
        mean_gradient_norm=update_metrics.mean_gradient_norm,
        max_gradient_norm=update_metrics.max_gradient_norm,
        no_launch_rate=gameplay.no_launch_rate,
        launches_per_episode=gameplay.launches_per_episode,
        illegal_action_count=illegal_action_count,
        environment_steps_per_second=rollout.transition_count / max(duration, 1e-12),
    )


def _linear_decay(start: float, end: float, step: int, duration: int) -> float:
    """Interpolate from ``start`` to ``end`` using step-based progress.

    Before or at step zero the schedule returns its starting value. At or after
    the duration it stays exactly at the ending value. Between them, ordinary
    linear interpolation moves the same fraction through the value range as
    training has moved through the scheduled step range.
    """

    if duration <= 0 or step >= duration:
        return end
    if step <= 0:
        return start
    progress = min(max(step / duration, 0.0), 1.0)
    return start + progress * (end - start)


def _mean(values: Sequence[int | float]) -> float:
    """Return an arithmetic mean, using zero when no game finished."""

    return float(sum(values) / len(values)) if values else 0.0


def _seed_learner(seed: int) -> None:
    """Give every global learner RNG a reproducible initial state.

    Python, NumPy, and Torch maintain separate random-number generators. Model
    initialization or helper code may use any of them, so all receive the same
    recorded learner seed. Dedicated Torch generators for actions and shuffling
    are created separately by ``run_training``.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initial_policy(learner_seed: int) -> OrbitPolicy:
    """Build the untrained policy that a run with this learner seed starts from.

    Training and the untrained diagnostic baseline both call this, so the
    baseline cannot drift from how a real run begins. The global learner RNGs
    are seeded first because PyTorch draws the initial weights at random, which
    also resets the caller's global random state. The model is built on the
    CPU; the caller moves it to its device afterwards, so the starting weights
    do not depend on which device later trains or plays it.
    """

    _seed_learner(learner_seed)
    return OrbitPolicy()


def _training_device(name: str) -> torch.device:
    """Resolve the configured Torch device before creating run artifacts.

    CPU is always available. A CUDA request must fail clearly when this machine
    has no CUDA runtime, rather than starting a directory and failing later when
    the model is moved.
    """

    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("run configuration requests CUDA, but CUDA is unavailable")
    return torch.device(name)


def _prepare_run_directory(path: str | Path, config: RunConfig) -> Path:
    """Create a new artifact directory and write its immutable identity files.

    ``exist_ok=False`` protects earlier experiments from accidental reuse. The
    JSON configuration records every setting that produced the run, and the
    initially empty manifest becomes the readable index of checkpoints written
    after completed updates.
    """

    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(destination / "run-config.json", config.to_dict())
    write_json_atomic(
        destination / "manifest.json",
        {
            "format": "run-manifest-v1",
            "configuration": "run-config.json",
            "checkpoints": [],
            "resumes": [],
        },
    )
    return destination


def _prepare_resume_directory(
    path: str | Path,
    config: RunConfig,
    checkpoint: Path,
) -> Path:
    """Verify that a checkpoint belongs to the existing run being continued."""

    destination = Path(path)
    if not destination.is_dir():
        raise ValueError(f"resume run directory does not exist: {destination}")
    expected_checkpoint_directory = (destination / "checkpoints").resolve()
    try:
        checkpoint.resolve().relative_to(expected_checkpoint_directory)
    except ValueError as error:
        raise ValueError(
            "resume checkpoint must be inside the run's checkpoints directory"
        ) from error

    recorded_config = read_json_object(
        destination / "run-config.json",
        "run configuration",
    )
    expected_config = json_value(config.to_dict())
    if recorded_config != expected_config:
        raise ValueError("existing run configuration does not match the requested run")

    manifest = read_json_object(destination / "manifest.json", "run manifest")
    if manifest.get("format") != "run-manifest-v1":
        raise ValueError("run manifest has an incompatible format")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("run manifest checkpoints must be a list")
    checkpoint_name = str(checkpoint.resolve().relative_to(destination.resolve()))
    if not any(
        isinstance(record, Mapping) and record.get("file") == checkpoint_name
        for record in checkpoints
    ):
        raise ValueError("resume checkpoint is not recorded in the run manifest")
    return destination


def _default_run_directory() -> Path:
    """Choose a unique UTC timestamp below the Git-ignored ``runs`` directory.

    Microseconds reduce collisions between invocations, while UTC avoids local
    daylight-saving or timezone ambiguity in artifact names.
    """

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return REPOSITORY_ROOT / "runs" / timestamp


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line options, start training, and print its stop summary.

    ``--config`` identifies the versioned TOML contract. ``--run-dir`` can fix
    an artifact location; otherwise a timestamped local directory is selected.
    ``--resume`` continues the run that owns a completed checkpoint;
    ``--max-updates`` provides a bounded cumulative update count, and
    ``--review-reason`` records why training may continue after the 25M review.
    ``--profile`` adds synchronized phase timings for performance diagnosis.
    """

    # cuBLAS reads its workspace setting when it first starts, so it must be
    # fixed before training or validation touches the GPU.
    configure_cuda_workspace()

    # Argument parsing is kept at the process boundary. The reusable
    # run_training function receives already validated Python values.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        help="continue the run containing this completed-update checkpoint",
    )
    parser.add_argument("--max-updates", type=int)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="record synchronized per-phase timings in timings.csv",
    )
    parser.add_argument(
        "--review-reason",
        help="record why an initial run may continue beyond its 25M-step review",
    )
    parser.add_argument(
        "--allow-development",
        action="store_true",
        help=(
            "with --resume: allow an engine version that does not name exact code "
            "(a local 'development' or '+dirty' build); recorded in the resume audit"
        ),
    )
    arguments = parser.parse_args(argv)
    if arguments.allow_development and arguments.resume is None:
        parser.error("--allow-development only applies together with --resume")

    # The versioned TOML fields are the complete compatibility identity. This
    # deliberately requires neither Git nor a .git directory, so a copied cloud
    # deployment can train and resume from the same explicit contract values.
    config = load_run_config(arguments.config)
    run_directory = arguments.run_dir
    if run_directory is None:
        run_directory = (
            arguments.resume.parent.parent
            if arguments.resume is not None
            else _default_run_directory()
        )
    result = run_training(
        config,
        run_directory,
        max_updates=arguments.max_updates,
        review_reason=arguments.review_reason,
        resume_checkpoint=arguments.resume,
        profile=arguments.profile,
        allow_development_engine=arguments.allow_development,
    )
    # A clean stop always reports both counters, the lifecycle reason, and the
    # directory containing detailed metrics and checkpoints.
    print(
        f"training stopped after {result.environment_steps} environment steps "
        f"and {result.completed_updates} updates: {result.reason}; "
        f"artifacts: {result.run_directory}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
