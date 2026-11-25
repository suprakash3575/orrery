"""Collect fixed-policy environment transitions for a later PPO update."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn

from .env_client import Episode, StepResult
from .gae import gae
from .model import (
    ObservationBatch,
    PolicyOutput,
    SampledActionBatch,
    policy_validation_flags,
    policy_device,
    require_evaluation_mode,
    run_policy,
    run_policy_for_collection,
    sample_actions_for_collection,
    validate_collection_sample,
    validate_policy_observation,
)
from .ppo import collection_joint_log_prob, validate_collection_log_prob
from .profiling import PhaseProfiler, profile_phase


REQUIRED_REPLAY_METADATA = frozenset(("engine", "schema", "generator", "opponent"))


class RolloutEnvironment(Protocol):
    """Environment operations needed by the reusable collector."""

    metadata: dict[str, object]

    def reset(self, ids: Sequence[int]) -> list[Episode]:
        """Start new episodes in the requested reusable slots."""

    def step_packed(
        self,
        action_targets: Mapping[int, Sequence[int]] | NDArray[np.integer[Any]],
        *,
        env_ids: Sequence[int] | None = None,
    ) -> list[StepResult]:
        """Advance games from target slots, using -1 to mean no launch.

        Collection always passes an ``(N, 44)`` integer array whose row ``i``
        belongs to ``env_ids[i]``. The mapping form remains part of the
        interface because evaluation passes ``{env_id: row}``.
        """


@dataclass(frozen=True)
class RolloutBatch:
    """One fixed-policy rollout flattened in time-major order.

    The leading row order is ``turn * environment_count + environment_position``.
    Consequently all transitions for turn zero come first in ascending
    environment-ID order, followed by turn one, and so on. ``time_major`` can
    restore the leading ``[turn, environment]`` dimensions when trajectory
    calculations need them; PPO minibatches use the flattened form directly.
    """

    # The observation before each action trains the current policy. The
    # observation returned after it is retained for replay and for the critic
    # bootstrap required at truncations and the final rollout boundary.
    observations: ObservationBatch
    next_observations: ObservationBatch
    actions: SampledActionBatch

    # These are fixed facts recorded from the rollout policy and environment.
    # They must never retain a gradient graph into collection.
    old_log_probabilities: Tensor
    values: Tensor
    next_values: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    advantages: Tensor
    return_targets: Tensor

    # Per-transition replay identity. The submitted action is already retained
    # in ``actions``; these fields identify the exact episode and resulting
    # turn to which it belongs.
    seeds: Tensor
    seats: Tensor
    explicit: Tensor
    turns: Tensor

    # One stable B-axis shared by every collected turn, plus process-wide
    # engine/schema/generator/opponent metadata copied from EnvClient.create.
    environment_ids: tuple[int, ...]
    time_steps: int
    replay_metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        """Reject misaligned storage before an optimizer can train from it."""

        if (
            isinstance(self.time_steps, bool)
            or not isinstance(self.time_steps, int)
            or self.time_steps < 1
        ):
            raise ValueError("time_steps must be a positive integer")
        if not self.environment_ids:
            raise ValueError("environment_ids must contain at least one environment")
        if tuple(sorted(set(self.environment_ids))) != self.environment_ids:
            raise ValueError("environment_ids must be unique and in ascending order")

        replay_metadata = deepcopy(dict(self.replay_metadata))
        missing_metadata = sorted(REQUIRED_REPLAY_METADATA - replay_metadata.keys())
        if missing_metadata:
            raise ValueError(
                f"replay_metadata is missing required fields {missing_metadata}"
            )
        for name in REQUIRED_REPLAY_METADATA:
            value = replay_metadata[name]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"replay_metadata field {name!r} must be a non-empty string"
                )
        # Copy before making the top-level mapping read-only. This separates
        # the stored replay identity from EnvClient.metadata, which can be
        # replaced by a later create call in another lifecycle.
        object.__setattr__(
            self,
            "replay_metadata",
            MappingProxyType(replay_metadata),
        )

        transition_count = self.time_steps * len(self.environment_ids)
        if self.observations.batch_size != transition_count:
            raise ValueError(
                "observations contain "
                f"{self.observations.batch_size} transitions, want {transition_count}"
            )
        if self.next_observations.batch_size != transition_count:
            raise ValueError(
                "next_observations contain "
                f"{self.next_observations.batch_size} transitions, "
                f"want {transition_count}"
            )
        if self.actions.launches.shape[0] != transition_count:
            raise ValueError(
                f"actions contain {self.actions.launches.shape[0]} transitions, "
                f"want {transition_count}"
            )

        vector_fields = (
            "old_log_probabilities",
            "values",
            "next_values",
            "rewards",
            "terminated",
            "truncated",
            "advantages",
            "return_targets",
            "seeds",
            "seats",
            "explicit",
            "turns",
        )
        for name in vector_fields:
            tensor = getattr(self, name)
            if tuple(tensor.shape) != (transition_count,):
                raise ValueError(
                    f"{name} has shape {tuple(tensor.shape)}, "
                    f"want {(transition_count,)}"
                )

        float_fields = (
            "old_log_probabilities",
            "values",
            "next_values",
            "rewards",
            "advantages",
            "return_targets",
        )
        expected_dtype = self.values.dtype
        for name in float_fields:
            tensor = getattr(self, name)
            if not tensor.is_floating_point():
                raise TypeError(
                    f"{name} must use a floating-point dtype, got {tensor.dtype}"
                )
            if tensor.dtype != expected_dtype:
                raise TypeError(
                    f"{name} must use {expected_dtype} to match values, got {tensor.dtype}"
                )
            if not torch.all(torch.isfinite(tensor)):
                raise ValueError(f"{name} must contain only finite values")

        for name in ("terminated", "truncated", "explicit"):
            tensor = getattr(self, name)
            if tensor.dtype != torch.bool:
                raise TypeError(f"{name} must use torch.bool, got {tensor.dtype}")
        for name in ("seeds", "seats", "turns"):
            tensor = getattr(self, name)
            if tensor.dtype != torch.long:
                raise TypeError(f"{name} must use torch.long, got {tensor.dtype}")
        if torch.any(self.terminated & self.truncated):
            raise ValueError("a transition cannot be both terminated and truncated")

        stored_tensors = (
            self.observations.bodies,
            self.observations.players,
            self.observations.body_mask,
            self.observations.owned_mask,
            self.observations.player_mask,
            self.observations.target_mask,
            self.next_observations.bodies,
            self.next_observations.players,
            self.next_observations.body_mask,
            self.next_observations.owned_mask,
            self.next_observations.player_mask,
            self.next_observations.target_mask,
            self.actions.launches,
            self.actions.targets,
            *(getattr(self, name) for name in vector_fields),
        )
        devices = {tensor.device for tensor in stored_tensors}
        if len(devices) != 1:
            raise ValueError(f"rollout tensors must share one device, got {devices}")
        if any(tensor.requires_grad for tensor in stored_tensors):
            raise ValueError("rollout tensors must be detached from collection graphs")

    @property
    def environment_count(self) -> int:
        """Return the number of parallel environment slots in each turn."""

        return len(self.environment_ids)

    @property
    def transition_count(self) -> int:
        """Return the number of environment turns stored in this rollout."""

        return self.time_steps * self.environment_count

    def time_major(self, tensor: Tensor) -> Tensor:
        """Restore this rollout's leading ``[turn, environment]`` axes.

        Shapes below use ``T`` for the number of turns stored in the rollout
        (``time_steps``) and ``B`` for the number of game slots played in
        parallel (``environment_count``). ``...`` stands for the tensor's
        remaining dimensions, which stay unchanged and differ per tensor: a
        reward has none, while body features add ``44, 27``.

        Collection stores transitions in time-major order, meaning all ``B``
        games for one turn come before any game's next turn. It flattens the
        first two axes to ``[T * B, ...]``, so turn ``t`` of game ``b`` is row
        ``t * B + b``. For example, with three turns and two environments,
        rows 0 and 1 belong to turn 0, rows 2 and 3 belong to turn 1, and rows
        4 and 5 belong to turn 2. This method reshapes those rows to
        ``[T, B, ...]`` without reordering or recalculating them.

        The flat form is convenient when PPO shuffles individual transitions
        into minibatches. The restored form is easier when code or tests need
        to follow one environment across turns, such as when checking GAE
        trajectory boundaries.
        """

        if tensor.ndim == 0 or tensor.shape[0] != self.transition_count:
            raise ValueError(
                "tensor must have one leading row per rollout transition, "
                f"got shape {tuple(tensor.shape)}"
            )
        return tensor.reshape(
            self.time_steps,
            self.environment_count,
            *tensor.shape[1:],
        )

    def collection_groups(self) -> Iterator[Tensor]:
        """Yield the flattened row indices from each original policy call.

        Collection evaluates all B environment slots together once per turn.
        Consumers that need to reproduce those calls must use these groups
        rather than regrouping rows according to an unrelated PPO minibatch.
        """

        for turn in range(self.time_steps):
            start = turn * self.environment_count
            yield torch.arange(
                start,
                start + self.environment_count,
                device=self.values.device,
            )

    def select_observations(self, indices: Tensor) -> ObservationBatch:
        """Return current observations for one flattened PPO minibatch."""

        return _select_observations(self.observations, indices)

    def select_actions(self, indices: Tensor) -> SampledActionBatch:
        """Return stored actions aligned with one flattened PPO minibatch."""

        return SampledActionBatch(
            launches=self.actions.launches[indices],
            targets=self.actions.targets[indices],
        )


@dataclass(frozen=True)
class CollectedRollout:
    """A completed training batch and states from which collection continues."""

    batch: RolloutBatch
    next_episodes: tuple[Episode, ...]

    def __post_init__(self) -> None:
        """Keep continuation episodes aligned with the rollout's B axis."""

        continuation_ids = tuple(episode.env_id for episode in self.next_episodes)
        if continuation_ids != self.batch.environment_ids:
            raise ValueError(
                f"continuation environment IDs are {continuation_ids}, "
                f"want {self.batch.environment_ids}"
            )


@dataclass(frozen=True)
class CollectedTransition:
    """Artifacts produced by the one environment step shared by tracing and rollout."""

    # Episode identity and model tensors all use this deterministic B-axis order.
    episodes: tuple[Episode, ...]
    observation: ObservationBatch
    policy_output: PolicyOutput
    # The tensor action and its probability are retained for PPO. The packed
    # form is the CPU representation used to build the Go request: a read-only
    # int64 array of shape (B, 44), one row per environment in B-axis order,
    # with -1 meaning no launch and 0 through 43 naming the selected target
    # slot.
    actions: SampledActionBatch
    old_log_probabilities: Tensor
    packed_action_targets: NDArray[np.int64]
    # Results and their converted observation describe the state after Go
    # completed the turn, before any finished slot is reset by collect_rollout.
    results: tuple[StepResult, ...]
    next_observation: ObservationBatch

    @property
    def environment_ids(self) -> tuple[int, ...]:
        """Return the B-axis environment order used throughout this step."""

        return tuple(episode.env_id for episode in self.episodes)


@torch.no_grad()
def collect_transition(
    environment: RolloutEnvironment,
    policy: nn.Module,
    episodes: Sequence[Episode],
    *,
    generator: torch.Generator | None = None,
    profiler: PhaseProfiler | None = None,
) -> CollectedTransition:
    """Run the shared observe, policy, sample, log-probability, and step path."""

    ordered_episodes = _ordered_episodes(episodes, "transition episodes")
    require_evaluation_mode(policy, "rollout collection")
    device = policy_device(policy)
    _require_generator_device(generator, device)

    # Validate and convert the caller's initial NumPy states once before the
    # shared transition stages begin. Standalone collection starts from NumPy
    # observations, so this first host-to-device conversion is tensor transfer.
    with profile_phase(profiler, "tensor_transfer"):
        observation = ObservationBatch.from_observations(
            [episode.observation for episode in ordered_episodes],
            device=device,
        )
    # Standalone callers begin with NumPy observations, so this wrapper creates
    # their first tensor batch. The rollout collector already owns that tensor
    # after its first turn and calls the helper below to avoid creating it again.
    return _collect_transition(
        environment,
        policy,
        ordered_episodes,
        observation,
        generator=generator,
        profiler=profiler,
    )


def _collect_transition(
    environment: RolloutEnvironment,
    policy: nn.Module,
    ordered_episodes: Sequence[Episode],
    observation: ObservationBatch,
    *,
    generator: torch.Generator | None,
    profiler: PhaseProfiler | None,
) -> CollectedTransition:
    """Advance one already-validated tensor batch through Python and Go.

    ``collect_rollout`` keeps Go's returned tensor batch for the following
    turn and enters here directly. The public ``collect_transition`` wrapper
    still performs all ordering and device checks for standalone callers.
    """

    # Episode order defines the batch axis sent to Go and stored for PPO. Read
    # the device from the supplied tensor batch because this hot-path helper
    # deliberately avoids looking up and validating the policy device again.
    environment_ids = tuple(episode.env_id for episode in ordered_episodes)
    device = observation.bodies.device

    # Python first asks the current fixed-weight policy to score every game in
    # this batch. Sampling converts those scores into concrete launch and target
    # choices, while the log probability records how likely those exact choices
    # were under the policy that collected them. PPO needs that value later.
    # A no-op context is used when profiling is disabled, so both modes run
    # exactly the same policy call, sampling, and probability calculations.
    with profile_phase(profiler, "policy"):
        output = run_policy_for_collection(policy, observation)
    with profile_phase(profiler, "sampling"):
        sample = sample_actions_for_collection(
            output,
            observation,
            generator=generator,
        )
        actions = sample.actions
    with profile_phase(profiler, "log_probability"):
        probability = collection_joint_log_prob(
            output,
            sample,
            observation,
        )
        old_log_probabilities = probability.values

    # Combine launch flags and target slots before crossing from CUDA to Python.
    # Append every validation flag to that same integer payload so one copy to
    # the CPU is the turn's only explicit GPU-to-CPU transfer. The action rows
    # use -1 as the internal no-launch sentinel expanded by EnvClient.
    with profile_phase(profiler, "action_encode"):
        packed_action_targets, validation_passed = _transfer_collection_payload(
            actions,
            torch.cat(
                (
                    policy_validation_flags(observation),
                    probability.validation_flags,
                )
            ),
        )
    # A failed aggregate flag says at least one original input was malformed.
    # Rerun the pure validators in pipeline order to raise the same detailed
    # error as the eager path. These helpers inspect the already sampled action
    # and retained selections; they never sample again or advance the generator.
    if not validation_passed:
        validate_policy_observation(observation)
        validate_collection_sample(output, observation, sample)
        validate_collection_log_prob(output, sample, observation)
        # Every transferred flag has a corresponding validator above. Reaching
        # here would mean those two definitions drifted and must fail closed.
        raise RuntimeError("deferred collection validation flags disagreed")
    # Row i of the packed array belongs to environment_ids[i]. The encoder
    # sorts IDs and rows together, so no per-game dictionary is built here.
    results = _ordered_results(
        environment.step_packed(packed_action_targets, env_ids=environment_ids),
        environment_ids,
    )
    _validate_episode_identity(ordered_episodes, results)

    # Go returns ordinary NumPy observations for the state after this turn.
    # Convert the complete response to one tensor batch now. collect_rollout()
    # can reuse it directly as the next policy input for continuing games.
    with profile_phase(profiler, "tensor_transfer"):
        next_observation = ObservationBatch.from_observations(
            [result.observation for result in results],
            device=device,
        )
    # Return both representations of the turn: tensors remain ready for PPO,
    # while packed targets record what the client expanded into Go actions.
    return CollectedTransition(
        episodes=ordered_episodes,
        observation=observation,
        policy_output=output,
        actions=actions,
        old_log_probabilities=old_log_probabilities,
        packed_action_targets=packed_action_targets,
        results=results,
        next_observation=next_observation,
    )


def _transfer_collection_payload(
    actions: SampledActionBatch,
    validation_flags: Tensor,
) -> tuple[NDArray[np.int64], bool]:
    """Transfer packed actions and validation results in one CPU operation.

    Action targets occupy the first ``games * 44`` values. The remaining
    values are one for a passing check and zero for a failed check. Splitting
    the NumPy array after the transfer does not touch CUDA again.
    """

    if validation_flags.ndim != 1 or validation_flags.dtype != torch.bool:
        raise TypeError(
            "collection validation flags must be a one-dimensional bool tensor"
        )
    if validation_flags.device != actions.targets.device:
        raise ValueError("collection actions and validation flags must share one device")

    packed_tensor = torch.where(
        actions.launches,
        actions.targets,
        torch.full_like(actions.targets, -1),
    )
    action_value_count = packed_tensor.numel()
    # ``.numpy()`` shares memory with the CPU tensor instead of creating one
    # Python int per value. That sharing is safe: ``torch.cat`` builds a new
    # tensor every turn and nothing else holds it, even on CPU where ``.cpu()``
    # does not copy. The targets are ``torch.long``, which SampledActionBatch
    # enforces, so the array is int64 and cannot carry a Boolean that the
    # encoder's integer check would miss.
    payload = torch.cat(
        (
            packed_tensor.reshape(-1),
            validation_flags.to(dtype=packed_tensor.dtype),
        )
    ).detach().cpu().numpy()

    # Slicing and reshaping make views, not copies. The views are marked
    # read-only because CollectedTransition is frozen and the array records
    # exactly what was sent to Go.
    packed_action_targets = payload[:action_value_count].reshape(
        actions.targets.shape
    )
    packed_action_targets.flags.writeable = False
    validation_passed = bool(np.all(payload[action_value_count:] == 1))
    return packed_action_targets, validation_passed


@torch.no_grad()
def collect_rollout(
    environment: RolloutEnvironment,
    policy: nn.Module,
    episodes: Sequence[Episode],
    *,
    steps: int,
    gamma: float = 0.999,
    lam: float = 0.95,
    generator: torch.Generator | None = None,
    profiler: PhaseProfiler | None = None,
) -> CollectedRollout:
    """Collect ``steps`` turns with fixed policy weights and no gradients.

    ``episodes`` supplies the state currently installed in every reusable
    environment slot. A terminal or externally truncated result is retained as
    the next observation for that transition, then only that finished slot is
    reset before the following collection turn. The returned
    ``next_episodes`` can therefore be passed directly into the next call.

    The collector obtains data only. It never changes policy parameters, calls
    an optimizer, or chooses PPO minibatches.
    """

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    # Put the games in a fixed order, sorted by environment ID, so row i of
    # every tensor in this rollout always means the same game, whatever order
    # ``episodes`` arrived in. This also stops with an error if the list is
    # empty or names an environment twice. "initial episodes" only labels
    # that error message.
    ordered_episodes = _ordered_episodes(episodes, "initial episodes")
    # Build both lookups in one pass over the ordered episodes.
    # ``active`` maps each environment ID to the episode currently playing in
    # it. Tensor rows use positions 0 through B-1, whereas protocol messages
    # name reusable environments by ID, so ``environment_positions`` tells
    # reset handling which tensor row belongs to each finished environment
    # without searching every turn.
    active: dict[int, Episode] = {}
    environment_positions: dict[int, int] = {}
    for position, episode in enumerate(ordered_episodes):
        active[episode.env_id] = episode
        environment_positions[episode.env_id] = position
    # A dict keeps insertion order, so these keys are the ascending IDs from
    # ``ordered_episodes``. No ID is lost to a repeated key because
    # ``_ordered_episodes`` has already rejected duplicates.
    environment_ids = tuple(active)

    # These invariants remain fixed for the complete rollout. Check them once,
    # then retain the first tensor batch as the learner's current state.
    require_evaluation_mode(policy, "rollout collection")
    device = policy_device(policy)
    _require_generator_device(generator, device)
    with profile_phase(profiler, "tensor_transfer"):
        current_observation = ObservationBatch.from_observations(
            [episode.observation for episode in ordered_episodes],
            device=device,
        )
    # Initial reset observations still live in NumPy. Their conversion occurs
    # once at the start of every collect_rollout() call and is grouped with the
    # later observation transfers performed during that same PPO update.

    rows = _RolloutRows()

    # Each turn has three stages: play one step, store it, then start new
    # games in any finished slots. The policy device is fixed for the
    # rollout, and _collect_transition converts every returned Go observation
    # back onto it, so the loop does not rediscover the device each turn.
    for _ in range(steps):
        # Rebuild only the lightweight identity tuple in the fixed B-axis order.
        # current_observation already contains the matching model tensors, either
        # returned by Go on the prior turn or installed by reset handling below.
        # Reading ``active`` by ID matters: after a reset, its insertion order no
        # longer matches the B axis.
        current_episodes = tuple(active[env_id] for env_id in environment_ids)
        transition = _collect_transition(
            environment,
            policy,
            current_episodes,
            current_observation,
            generator=generator,
            profiler=profiler,
        )
        # Store the turn before any reset. A finished game's final observation
        # and boundary flags belong to this transition, not to the next game.
        rows.record(transition)
        # Replace both values together so the identities and observations used
        # on the next turn always describe the same games.
        active, current_observation = _advance_finished_slots(
            environment,
            transition,
            environment_positions,
            device=device,
            profiler=profiler,
        )

    # Everything below turns the per-turn row groups (512 in the standard
    # rollout) into the fixed tensors consumed by bootstrap, GAE, and PPO. Profiling uses two build intervals so
    # bootstrap inference can remain visible as its own phase in between them.
    with profile_phase(profiler, "rollout_build"):
        # Critic values were already produced as one device tensor per policy
        # call, so torch.stack joins them without another host-to-device transfer.
        values = torch.stack(rows.values)

        # Protocol rewards and boundary flags are still Python scalars. Construct
        # each complete [T, B] tensor once with torch.tensor, which accepts the
        # nested Python lists directly. Previously every turn created seven small
        # tensors and torch.stack joined them later: 512 turns therefore caused
        # 3,584 tensor constructions. Creating one complete tensor per field
        # reduces that work to seven constructions for the standard rollout.
        rewards = torch.tensor(
            rows.rewards,
            dtype=values.dtype,
            device=device,
        )
        terminated = torch.tensor(rows.terminated, dtype=torch.bool, device=device)
        truncated = torch.tensor(rows.truncated, dtype=torch.bool, device=device)

        # GAE needs a critic estimate for the state following each stored action.
        # Most values can be reused from the following turn; the bootstrap helper
        # evaluates only truncations and the final rollout boundary that lack one.
        next_observations = _concatenate_observations(rows.next_observations)

    with profile_phase(profiler, "bootstrap"):
        next_values = calculate_bootstrap_values(
            policy,
            next_observations,
            values,
            terminated,
            truncated,
        )

    with profile_phase(profiler, "rollout_build"):
        # GAE combines rewards and critic estimates across time. advantages tell
        # PPO whether sampled actions performed better or worse than expected;
        # return_targets are the critic values the value head should learn toward.
        advantages, return_targets = gae(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            gamma=gamma,
            lam=lam,
        )

        # PPO consumes one flat time-major batch: all B environments from turn 0,
        # followed by all B environments from turn 1, and so on. Concatenation and
        # reshape preserve that order consistently across every stored field. The
        # seed, seat, explicit-assignment flag, and turn lists are also converted
        # here once rather than once per collection turn.
        batch = RolloutBatch(
            observations=_concatenate_observations(rows.observations),
            next_observations=next_observations,
            actions=SampledActionBatch(
                launches=torch.cat([row.launches for row in rows.actions]),
                targets=torch.cat([row.targets for row in rows.actions]),
            ),
            old_log_probabilities=torch.cat(rows.log_probabilities),
            values=values.reshape(-1),
            next_values=next_values.reshape(-1),
            rewards=rewards.reshape(-1),
            terminated=terminated.reshape(-1),
            truncated=truncated.reshape(-1),
            advantages=advantages.reshape(-1),
            return_targets=return_targets.reshape(-1),
            seeds=torch.tensor(rows.seeds, dtype=torch.long, device=device).reshape(-1),
            seats=torch.tensor(rows.seats, dtype=torch.long, device=device).reshape(-1),
            explicit=torch.tensor(
                rows.explicit,
                dtype=torch.bool,
                device=device,
            ).reshape(-1),
            turns=torch.tensor(rows.turns, dtype=torch.long, device=device).reshape(-1),
            environment_ids=environment_ids,
            time_steps=steps,
            replay_metadata=environment.metadata,
        )
    # Alongside the fixed training batch, return the episode now installed in
    # every reusable slot. The next rollout continues from these identities and
    # from their matching observations without resetting live games.
    return CollectedRollout(
        batch=batch,
        next_episodes=tuple(active[env_id] for env_id in environment_ids),
    )


@dataclass
class _RolloutRows:
    """Per-turn values gathered during collection and joined after the last turn.

    ``record`` appends exactly one entry to every list per collection turn, so
    after ``T`` turns each list has length ``T``. Entry ``t`` holds turn
    ``t``'s values for all ``B`` game slots in B-axis order. ``collect_rollout``
    owns one instance per call and turns these lists into the fixed
    ``RolloutBatch`` tensors once collection ends.
    """

    # Policy inputs and outputs. These are already tensors on the policy
    # device, one B-row batch per turn, so joining them later needs no
    # host-to-device copy.
    observations: list[ObservationBatch] = field(default_factory=list)
    # Go's observation after each turn, before any finished slot is reset.
    # Bootstrap values for truncated games are computed from these rows.
    next_observations: list[ObservationBatch] = field(default_factory=list)
    actions: list[SampledActionBatch] = field(default_factory=list)
    log_probabilities: list[Tensor] = field(default_factory=list)
    # Critic estimates for each turn's starting state.
    values: list[Tensor] = field(default_factory=list)

    # These values originate as Python scalars in protocol results and episode
    # identities. Retain plain rows during collection, then create one tensor
    # per field after all requested turns. In the standard 512-turn rollout,
    # constructing seven tiny CUDA tensors every turn caused thousands of
    # host-to-device operations.
    rewards: list[list[float]] = field(default_factory=list)
    terminated: list[list[bool]] = field(default_factory=list)
    truncated: list[list[bool]] = field(default_factory=list)
    # Replay identity of the episode that chose each action. These come from
    # the episodes the turn started with, not from the step results.
    seeds: list[list[int]] = field(default_factory=list)
    seats: list[list[int]] = field(default_factory=list)
    explicit: list[list[bool]] = field(default_factory=list)
    # The game turn reported by Go after each step.
    turns: list[list[int]] = field(default_factory=list)

    def record(self, transition: CollectedTransition) -> None:
        """Append one turn's values for every slot to each list.

        Each line builds one column with its own comprehension. Merging them
        into a single loop was measured and was slightly slower, because
        CPython runs a comprehension faster than repeated ``append`` calls.
        """

        results = transition.results
        episodes = transition.episodes
        self.observations.append(transition.observation)
        self.next_observations.append(transition.next_observation)
        self.actions.append(transition.actions)
        self.log_probabilities.append(transition.old_log_probabilities)
        self.values.append(transition.policy_output.values)
        self.rewards.append([result.reward for result in results])
        self.terminated.append([result.terminated for result in results])
        self.truncated.append([result.truncated for result in results])
        self.seeds.append([episode.seed for episode in episodes])
        self.seats.append([episode.seat for episode in episodes])
        self.explicit.append([episode.explicit for episode in episodes])
        self.turns.append([result.turn for result in results])


def _advance_finished_slots(
    environment: RolloutEnvironment,
    transition: CollectedTransition,
    environment_positions: Mapping[int, int],
    *,
    device: torch.device,
    profiler: PhaseProfiler | None,
) -> tuple[dict[int, Episode], ObservationBatch]:
    """Prepare every slot's episode and observation for the next turn.

    Slots whose game is still running keep the step result Go just returned.
    Slots whose game terminated or was truncated are reset, so Go starts a
    new game there. Returns the next ``active`` map, keyed by environment ID,
    and the matching policy-input observation in B-axis order.

    Call this only after the transition has been recorded. A finished game's
    final observation belongs to the stored transition, so a truncation
    bootstraps from its real last state rather than from the next game's
    starting state.
    """

    results = transition.results
    next_observation = transition.next_observation
    finished_ids = tuple(
        result.env_id
        for result in results
        if result.terminated or result.truncated
    )
    next_active: dict[int, Episode] = {
        result.env_id: result
        for result in results
        if not (result.terminated or result.truncated)
    }
    if not finished_ids:
        # For a continuing game, Go's observation after turn t is exactly
        # the policy input for turn t+1. Reusing this tensor avoids stacking
        # the same NumPy observations and copying them to CUDA again.
        return next_active, next_observation

    # Ask Go to assign a new game only to finished slots. Continuing slots
    # must retain both their episode identity and the observation Go returned
    # for them on this turn. Reset episodes are added at the end of the dict,
    # so its insertion order no longer matches the B axis. Callers must read
    # it by environment ID.
    reset_episodes = _ordered_reset_episodes(
        environment.reset(finished_ids),
        finished_ids,
    )
    next_active.update({episode.env_id: episode for episode in reset_episodes})
    # Only reset results are still NumPy values here. Convert that small
    # subset rather than rebuilding tensors for all environments.
    with profile_phase(profiler, "tensor_transfer"):
        reset_observation = ObservationBatch.from_observations(
            [episode.observation for episode in reset_episodes],
            device=device,
        )
        # index_copy_ works with tensor row positions, so translate each
        # stable environment ID into its position on the fixed batch axis.
        reset_positions = torch.tensor(
            [environment_positions[episode.env_id] for episode in reset_episodes],
            dtype=torch.long,
            device=device,
        )
        # Start from the complete post-step tensor and replace only the rows
        # whose games ended. The resulting batch therefore combines reset
        # starting states with untouched continuing states in B-axis order.
        current_observation = _replace_observation_rows(
            next_observation,
            reset_observation,
            reset_positions,
        )
    return next_active, current_observation


def _replace_observation_rows(
    base: ObservationBatch,
    replacements: ObservationBatch,
    indices: Tensor,
) -> ObservationBatch:
    """Return ``base`` with reset states installed at selected batch rows.

    The stored next-observation row must continue describing the completed
    transition, including terminal states. Clone each field before replacing
    rows so preparing the following policy input cannot rewrite that history.
    """

    # One replacement row is required for every reset observation. A mismatch
    # would either leave a reset state unused or write it into the wrong slot.
    if indices.ndim != 1 or indices.numel() != replacements.batch_size:
        raise ValueError("replacement indices must name every replacement row")

    def replaced(source: Tensor, values: Tensor) -> Tensor:
        """Copy one observation field and install its reset rows by position."""

        # base is also stored as the transition's next observation. Mutating it
        # in place would replace terminal history with the next game's starting
        # state, corrupting bootstrap calculations and replay evidence.
        result = source.clone()
        result.index_copy_(0, indices, values)
        return result

    # Apply the same row operation to every observation field so features and
    # masks cannot describe different games after a partial reset.
    return ObservationBatch(
        bodies=replaced(base.bodies, replacements.bodies),
        players=replaced(base.players, replacements.players),
        body_mask=replaced(base.body_mask, replacements.body_mask),
        owned_mask=replaced(base.owned_mask, replacements.owned_mask),
        player_mask=replaced(base.player_mask, replacements.player_mask),
        target_mask=replaced(base.target_mask, replacements.target_mask),
    )


@torch.no_grad()
def calculate_bootstrap_values(
    policy: nn.Module,
    next_observations: ObservationBatch,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
) -> Tensor:
    """Return the critic's estimate of what comes after each stored turn.

    GAE (gae.py) needs, for every turn t of every game, the critic's value of
    the position *after* that turn: its ``next_values``. The result has shape
    [turn, environment]. There are three cases:

    | After turn t, the game... | Next value |
    |---|---|
    | ended (won, lost or tied) | 0: nothing more can happen |
    | continues on the next stored turn | that turn's stored value, already computed when it was played, so it is reused |
    | was cut off without ending, or the rollout stopped here | the critic is asked now, about the position the game was left in |

    "Cut off without ending" is a truncation. Normal games never truncate
    (see ``Truncated`` in mlenv/env.go). The last stored turn always needs
    the critic, unless the game ended there, because no later turn was stored.

    ``@torch.no_grad()`` keeps these values fixed: they are targets to learn
    toward, so training must not change them through gradients.
    """

    next_values = torch.zeros_like(values)
    if values.shape[0] > 1:
        # Within one uninterrupted episode, the observation after turn t is
        # exactly the observation whose value was recorded before turn t+1.
        # Reusing it avoids an otherwise redundant policy forward pass.
        same_episode = ~(terminated[:-1] | truncated[:-1])
        next_values[:-1] = torch.where(same_episode, values[1:], next_values[:-1])

    # The third case in the docstring: positions the critic must be asked
    # about now.
    # - A truncated game still had a future, but its next stored row is the
    #   first turn of a fresh game started after the reset, so that row's
    #   value belongs to a different game and can't be reused.
    # - The last stored turn has no next row at all, so every game that
    #   didn't end on it needs a value.
    # Games that ended keep the 0 they were given at the start.
    needs_evaluation = truncated.clone()
    needs_evaluation[-1] |= ~terminated[-1]
    flat_indices = needs_evaluation.reshape(-1).nonzero().flatten()
    if flat_indices.numel() > 0:
        selected = _select_observations(next_observations, flat_indices)
        evaluated = run_policy(policy, selected).values
        flat_next_values = next_values.view(-1)
        flat_next_values.index_copy_(0, flat_indices, evaluated)
    return next_values


def _require_generator_device(
    generator: torch.Generator | None,
    policy_device: torch.device,
) -> None:
    """Require the generator and policy to use the same Torch backend.

    The V1 configuration chooses ``cpu`` or ``cuda`` rather than a numbered
    device. After a policy moves to the GPU, Torch reports its concrete device
    as ``cuda:0`` while a generator created from the configured shorthand still
    reports ``cuda``. Those names refer to the same supported CUDA backend, so
    comparing their optional indices would reject a valid single-GPU run.
    """

    if generator is None:
        return
    generator_device = torch.device(generator.device)
    if generator_device.type != policy_device.type:
        raise ValueError(
            f"generator uses {generator_device}, but policy uses {policy_device}"
        )


def _ordered_episodes(
    episodes: Sequence[Episode],
    name: str,
) -> tuple[Episode, ...]:
    """Return unique episodes in deterministic environment-ID order."""

    if not episodes:
        raise ValueError(f"{name} must contain at least one episode")
    by_id = {episode.env_id: episode for episode in episodes}
    if len(by_id) != len(episodes):
        raise ValueError(f"{name} contain duplicate environment IDs")
    return tuple(by_id[env_id] for env_id in sorted(by_id))


def _ordered_results(
    results: Sequence[StepResult],
    environment_ids: tuple[int, ...],
) -> tuple[StepResult, ...]:
    """Require exactly one step result for every action submitted."""

    by_id = {result.env_id: result for result in results}
    if len(by_id) != len(results):
        raise RuntimeError("environment returned duplicate step-result IDs")
    if set(by_id) != set(environment_ids):
        raise RuntimeError(
            f"environment returned step-result IDs {tuple(sorted(by_id))}, "
            f"want {environment_ids}"
        )
    return tuple(by_id[env_id] for env_id in environment_ids)


def _ordered_reset_episodes(
    episodes: Sequence[Episode],
    expected_ids: tuple[int, ...],
) -> tuple[Episode, ...]:
    """Require reset to replace every and only the finished slots."""

    ordered = _ordered_episodes(episodes, "reset episodes")
    actual_ids = tuple(episode.env_id for episode in ordered)
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"environment reset IDs {actual_ids}, want finished IDs {expected_ids}"
        )
    return ordered


def _validate_episode_identity(
    episodes: Sequence[Episode],
    results: Sequence[StepResult],
) -> None:
    """Ensure a step result still belongs to the episode that chose its action."""

    for episode, result in zip(episodes, results, strict=True):
        before = (episode.seed, episode.seat, episode.explicit)
        after = (result.seed, result.seat, result.explicit)
        if after != before:
            raise RuntimeError(
                f"environment {episode.env_id} changed episode identity from "
                f"{before} to {after} during one step"
            )
        if result.terminated and result.truncated:
            raise RuntimeError(
                f"environment {episode.env_id} returned both boundary flags"
            )


def _concatenate_observations(
    rows: Sequence[ObservationBatch],
) -> ObservationBatch:
    """Flatten time-major observation rows without changing row order."""

    return ObservationBatch(
        bodies=torch.cat([row.bodies for row in rows]),
        players=torch.cat([row.players for row in rows]),
        body_mask=torch.cat([row.body_mask for row in rows]),
        owned_mask=torch.cat([row.owned_mask for row in rows]),
        player_mask=torch.cat([row.player_mask for row in rows]),
        target_mask=torch.cat([row.target_mask for row in rows]),
    )


def _select_observations(
    observations: ObservationBatch,
    indices: Tensor,
) -> ObservationBatch:
    """Select aligned observation rows for bootstrap evaluation or training."""

    if indices.ndim != 1 or indices.dtype != torch.long:
        raise TypeError("observation indices must be a one-dimensional torch.long tensor")
    return ObservationBatch(
        bodies=observations.bodies[indices],
        players=observations.players[indices],
        body_mask=observations.body_mask[indices],
        owned_mask=observations.owned_mask[indices],
        player_mask=observations.player_mask[indices],
        target_mask=observations.target_mask[indices],
    )
