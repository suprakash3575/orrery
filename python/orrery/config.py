"""Load and validate the complete configuration for one RL training run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any
import tomllib

from .env_client import (
    GENERATOR_FOR_PLAYERS,
    OBSERVATION_SCHEMA,
    PROTOCOL_VERSION,
    SCHEDULE_TRAIN_V2,
)


# rl-v2 is the only training contract. It fixes the rollout layout that gives
# one PPO update its meaning: 64 consecutive turns from each of 512 games, or
# 32,768 transitions. 64 x 512 holds as many transitions but is a different
# training layout, so it is not accepted under this name.
CONTRACT = "rl-v2"
CONTRACT_LAYOUT = (512, 64)

# These identities are the same in every rl-v2 run. The generator is fixed by
# the player count instead, through GENERATOR_FOR_PLAYERS.
_FIXED_IDENTITIES = {
    "training_schedule": SCHEDULE_TRAIN_V2,
    # The pinned heuristic-v2 spec (agent.HeuristicV2Default in Go). The
    # opponent check against it gave "keep": no skill-less tester won a game,
    # where diag-lead beat heuristic-v1 59.4% of the time. It is the opponent
    # in every training seat and in the release gate.
    "training_opponent": "heuristic-v2:candidates=4,enemy_bonus=1,margin=0.1",
    "pipeline_opponent": (
        "random-v1:launch_probability=0.3,max_fraction=0.8,min_fraction=0.2"
    ),
    "evaluation_policy": "eval-policy-v2",
    "protocol_version": PROTOCOL_VERSION,
    "observation_schema": OBSERVATION_SCHEMA,
    "checkpoint_format": "checkpoint-v2",
}


@dataclass(frozen=True)
class SeedRange:
    """A half-open scenario-seed interval: start is included and end is not."""

    # start is the first scenario seed available to this partition.
    start: int
    # end is the first scenario seed outside the partition, so it is not used.
    end: int

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or isinstance(self.end, bool):
            raise TypeError("seed range bounds must be integers")
        if self.start >= self.end:
            raise ValueError(f"seed range [{self.start},{self.end}) must not be empty")


@dataclass(frozen=True)
class RunConfig:
    """Every fixed input needed to identify and reproduce one training run."""

    # contract names the complete versioned RL experiment contract.
    contract: str
    # players is 2 or 4: how many seats every training map has. It chooses the
    # map generator and is passed to cmd/env as --players.
    players: int
    # device selects where Torch performs policy inference and PPO optimization.
    device: str
    # learner_seed initializes Python, NumPy, Torch, and derived learner RNG streams.
    learner_seed: int
    # environment_count is the number of reusable Go game slots stepped in parallel.
    environment_count: int
    # rollout_steps is the number of game turns collected from every slot per rollout.
    rollout_steps: int
    # minibatch_size is the number of stored transitions used by one optimizer step.
    minibatch_size: int
    # ppo_epochs is the number of complete passes over one collected rollout batch.
    ppo_epochs: int
    # learning_rate is AdamW's starting step size before its linear decay to zero.
    learning_rate: float
    # weight_decay is AdamW's explicit parameter-shrinkage coefficient.
    weight_decay: float
    # gamma discounts rewards by this factor for each game turn into the future.
    gamma: float
    # gae_lambda controls how far later TD errors propagate through GAE.
    gae_lambda: float
    # clip_epsilon bounds how far PPO's new/old action-probability ratio may help loss.
    clip_epsilon: float
    # value_coefficient scales the critic loss when actor and critic losses are combined.
    value_coefficient: float
    # launch_entropy_initial is the launch/no-launch exploration bonus at step zero.
    launch_entropy_initial: float
    # launch_entropy_final is the floor reached by the launch entropy schedule.
    launch_entropy_final: float
    # target_entropy_initial is the target-choice exploration bonus at step zero.
    target_entropy_initial: float
    # target_entropy_final is the floor reached by the target entropy schedule.
    target_entropy_final: float
    # entropy_anneal_fraction is the cumulative-budget fraction used for entropy decay.
    entropy_anneal_fraction: float
    # max_gradient_norm clips the combined gradient norm before each optimizer step.
    max_gradient_norm: float
    # approximate_kl_warning records the diagnostic KL level considered concerning.
    approximate_kl_warning: float
    # checkpoint_every_updates is the periodic save cadence in completed PPO updates.
    checkpoint_every_updates: int
    # smoke_evaluation_every_steps reserves a training-step cadence for smoke evaluation.
    smoke_evaluation_every_steps: int
    # pipeline_gate_every_steps is the initial validation cadence in training steps.
    pipeline_gate_every_steps: int
    # review_steps is the initial-stage step count that requires a continuation reason.
    review_steps: int
    # initial_step_ceiling limits the first pipeline-learning stage's training steps.
    initial_step_ceiling: int
    # cumulative_step_ceiling limits all training for one experiment.
    cumulative_step_ceiling: int
    # pipeline_validation_seeds is the number of validation boards used by each gate.
    pipeline_validation_seeds: int
    # pipeline_required_wins is the minimum wins across every seat for gate success,
    # out of pipeline_games.
    pipeline_required_wins: int
    # release_validation_seeds is the number of boards played from every learner seat.
    release_validation_seeds: int
    # release_required_wins is the minimum wins in the complete rotated release
    # sample, out of release_games.
    release_required_wins: int
    # release_confidence_level selects the exact binomial interval's coverage.
    release_confidence_level: float
    # release_minimum_lower_bound is the required conservative win-rate estimate.
    release_minimum_lower_bound: float
    # training_seeds is the Go scenario-seed partition from which training games draw.
    training_seeds: SeedRange
    # validation_seeds is the reusable scenario partition used for model selection.
    validation_seeds: SeedRange
    # evaluation_seeds contains protected, one-use final-evaluation partitions.
    evaluation_seeds: tuple[SeedRange, ...]
    # evaluation_pool_names identifies each protected range in manifests and results.
    evaluation_pool_names: tuple[str, ...]
    # training_schedule names Go's deterministic cursor-to-seed-and-seat mapping.
    training_schedule: str
    # training_opponent is the complete versioned opponent specification used to learn.
    training_opponent: str
    # pipeline_opponent is the complete opponent specification used by the initial gate.
    pipeline_opponent: str
    # evaluation_policy names deterministic action decoding used outside training.
    evaluation_policy: str
    # protocol_version identifies the Python/Go command and response contract.
    protocol_version: str
    # observation_schema identifies the shape and meaning of every model input feature.
    observation_schema: str
    # generator_version identifies how a scenario seed becomes an initial game board.
    generator_version: str
    # checkpoint_format identifies the serialized learner-state representation.
    checkpoint_format: str

    @property
    def batch_size(self) -> int:
        """Return transitions collected for one complete PPO update."""

        return self.environment_count * self.rollout_steps

    @property
    def pipeline_games(self) -> int:
        """Return games in one pipeline validation: every seed from every seat."""

        return self.pipeline_validation_seeds * self.players

    @property
    def release_games(self) -> int:
        """Return games in one release validation: every seed from every seat."""

        return self.release_validation_seeds * self.players

    @property
    def held_out_games(self) -> int:
        """Return rotated games in each fixed protected evaluation pool."""

        seeds = self.evaluation_seeds[0]
        return (seeds.end - seeds.start) * self.players

    @property
    def held_out_required_wins(self) -> int:
        """Apply the configured release win fraction to the held-out sample.

        Training uses the same 90% observed-win requirement for release validation
        and final evaluation. Expressing that relationship through RunConfig
        keeps evaluation thresholds with the archived experiment contract while
        avoiding a second hard-coded 1,800-win constant.
        """

        scaled_wins = self.release_required_wins * self.held_out_games
        if scaled_wins % self.release_games != 0:
            raise ValueError(
                "release win fraction does not produce a whole held-out win count"
            )
        return scaled_wins // self.release_games

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")

        positive_integers = (
            "environment_count", "rollout_steps", "minibatch_size", "ppo_epochs",
            "checkpoint_every_updates", "smoke_evaluation_every_steps",
            "pipeline_gate_every_steps", "review_steps", "initial_step_ceiling",
            "cumulative_step_ceiling", "pipeline_validation_seeds",
            "pipeline_required_wins", "release_validation_seeds",
            "release_required_wins",
        )
        for name in positive_integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.learner_seed, bool) or not isinstance(self.learner_seed, int):
            raise TypeError("learner_seed must be an integer")
        if not 0 <= self.learner_seed < 2**32:
            raise ValueError("learner_seed must fit NumPy's unsigned 32-bit seed range")

        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("rollout batch size must be divisible by minibatch_size")
        if self.batch_size > self.pipeline_gate_every_steps:
            raise ValueError(
                "one rollout batch must not cross more than one pipeline-gate boundary"
            )
        if self.pipeline_validation_seeds > (
            self.validation_seeds.end - self.validation_seeds.start
        ):
            raise ValueError("pipeline validation sample exceeds the validation seed range")
        if self.release_validation_seeds > (
            self.validation_seeds.end - self.validation_seeds.start
        ):
            raise ValueError("release validation sample exceeds the validation seed range")
        if not self.review_steps < self.initial_step_ceiling <= self.cumulative_step_ceiling:
            raise ValueError("step budgets must satisfy review < initial <= cumulative")

        unit_interval = ("gamma", "gae_lambda", "clip_epsilon", "entropy_anneal_fraction")
        for name in unit_interval:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        positive_floats = (
            "learning_rate", "value_coefficient", "launch_entropy_initial",
            "launch_entropy_final", "target_entropy_initial", "target_entropy_final",
            "max_gradient_norm", "approximate_kl_warning",
        )
        for name in positive_floats:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if (
            isinstance(self.release_confidence_level, bool)
            or not isinstance(self.release_confidence_level, (int, float))
            or not math.isfinite(self.release_confidence_level)
            or not 0.0 < self.release_confidence_level < 1.0
        ):
            raise ValueError("release_confidence_level must be in (0, 1)")
        if (
            isinstance(self.release_minimum_lower_bound, bool)
            or not isinstance(self.release_minimum_lower_bound, (int, float))
            or not math.isfinite(self.release_minimum_lower_bound)
            or not 0.0 <= self.release_minimum_lower_bound <= 1.0
        ):
            raise ValueError("release_minimum_lower_bound must be in [0, 1]")
        if self.launch_entropy_final > self.launch_entropy_initial:
            raise ValueError("launch entropy schedule must not increase")
        if self.target_entropy_final > self.target_entropy_initial:
            raise ValueError("target entropy schedule must not increase")

        identities = (
            "contract", "training_schedule", "training_opponent", "pipeline_opponent",
            "evaluation_policy", "protocol_version", "observation_schema",
            "generator_version", "checkpoint_format",
        )
        for name in identities:
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")

        if self.contract != CONTRACT:
            raise ValueError(f"contract must be {CONTRACT!r}, got {self.contract!r}")

        expected_environments, expected_rollout_steps = CONTRACT_LAYOUT
        if (
            self.environment_count != expected_environments
            or self.rollout_steps != expected_rollout_steps
        ):
            raise ValueError(
                f"contract {self.contract!r} requires "
                f"environment_count={expected_environments} and "
                f"rollout_steps={expected_rollout_steps}"
            )

        # bool is a subclass of int, so True must be refused before the lookup
        # below would treat it as a one-player count.
        if isinstance(self.players, bool) or self.players not in GENERATOR_FOR_PLAYERS:
            supported = " or ".join(str(count) for count in GENERATOR_FOR_PLAYERS)
            raise ValueError(f"players must be {supported}, got {self.players!r}")

        # The game counts multiply seeds by players, so these caps run only
        # once players is known to be 2 or 4.
        if self.pipeline_required_wins > self.pipeline_games:
            raise ValueError("pipeline_required_wins exceeds the pipeline game count")
        if self.release_required_wins > self.release_games:
            raise ValueError("release_required_wins exceeds the release game count")

        # Each player count has exactly one generator, so a mismatch names
        # maps the configured seat count does not describe.
        fixed_identities = {
            **_FIXED_IDENTITIES,
            "generator_version": GENERATOR_FOR_PLAYERS[self.players],
        }
        for name, expected in fixed_identities.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"{name} must be {expected!r} for {CONTRACT} "
                    f"with {self.players} players"
                )

        if len(self.evaluation_seeds) != len(self.evaluation_pool_names):
            raise ValueError("evaluation pool names and seed ranges must align")
        if len(set(self.evaluation_pool_names)) != len(self.evaluation_pool_names):
            raise ValueError("evaluation pool names must be unique")
        ranges = (self.training_seeds, self.validation_seeds, *self.evaluation_seeds)
        for index, left in enumerate(ranges):
            for right in ranges[index + 1 :]:
                if max(left.start, right.start) < min(left.end, right.end):
                    raise ValueError("training, validation, and evaluation seeds must be disjoint")
        expected_ranges = (
            SeedRange(0, 1_000_000),
            SeedRange(1_000_000, 1_010_000),
            SeedRange(1_010_000, 1_011_000),
            SeedRange(1_011_000, 1_012_000),
            SeedRange(1_012_000, 1_013_000),
            SeedRange(1_013_000, 1_014_000),
            SeedRange(1_014_000, 1_015_000),
        )
        if ranges != expected_ranges:
            raise ValueError("seed ranges must match the frozen seed partitions")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible run metadata for manifests and checkpoints."""

        return asdict(self)


def load_run_config(path: str | Path) -> RunConfig:
    """Parse one complete semantic training configuration from TOML."""

    config_path = Path(path)
    with config_path.open("rb") as source:
        raw = tomllib.load(source)
    return run_config_from_mapping(raw)


def run_config_from_mapping(raw: Mapping[str, Any]) -> RunConfig:
    """Build and fully validate a RunConfig from plain parsed values.

    Two sources reach this function. A TOML file writes each seed range as
    ``[start, end]``. A recorded configuration (``RunConfig.to_dict`` saved as
    JSON, for example in a checkpoint envelope) writes it as
    ``{"start": ..., "end": ...}``. Both pass the same key and contract checks,
    so a checkpoint recorded under another contract is refused here, before any
    caller unpickles it.
    """

    if not isinstance(raw, Mapping):
        raise ValueError("run configuration must be a mapping of field names")
    expected = set(RunConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise ValueError("invalid run configuration: " + "; ".join(details))

    values: dict[str, Any] = dict(raw)
    for name in ("training_seeds", "validation_seeds"):
        values[name] = _seed_range(values[name], name)
    if not isinstance(values["evaluation_seeds"], list):
        raise ValueError("evaluation_seeds must be a list of seed ranges")
    values["evaluation_seeds"] = tuple(
        _seed_range(item, f"evaluation_seeds[{index}]")
        for index, item in enumerate(values["evaluation_seeds"])
    )
    if not isinstance(values["evaluation_pool_names"], list):
        raise ValueError("evaluation_pool_names must be a list of names")
    values["evaluation_pool_names"] = tuple(values["evaluation_pool_names"])
    return RunConfig(**values)


def _seed_range(value: object, name: str) -> SeedRange:
    """Accept a seed range as TOML's [start, end] or a record's {start, end}."""

    if isinstance(value, Mapping):
        if set(value) != {"start", "end"}:
            raise ValueError(f"{name} must contain exactly start and end")
        value = [value["start"], value["end"]]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly [start, end]")
    if any(isinstance(bound, bool) or not isinstance(bound, int) for bound in value):
        raise TypeError(f"{name} bounds must be integers")
    return SeedRange(value[0], value[1])
