"""Probability, loss, and optimizer calculations used by Orrery PPO."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical
from torch.optim import Optimizer

from .env_client import BODY_SLOTS
from .model import (
    CollectionActionSample,
    ObservationBatch,
    PolicyOutput,
    SampledActionBatch,
    policy_device,
    require_evaluation_mode,
    run_policy,
    sampled_target_validation_flag,
)

if TYPE_CHECKING:
    from .rollout import RolloutBatch


# Collection-shaped replay is bit-identical on the tested CPU path. V1 uses
# float32, for which this floor is assumed to leave room for ordinary
# CUDA/TF32 replay rounding without accepting policy-scale drift. PR 15's
# first GPU run must measure and confirm that assumption.
DEFAULT_FLOAT32_LOG_PROBABILITY_ATOL = 1e-5
DEFAULT_LOG_PROBABILITY_RTOL = 1e-5


@dataclass(frozen=True)
class EntropyBonus:
    """Batch-mean entropy values used to encourage policy exploration."""

    # ``launch`` and ``target`` stay separate so training can coefficient and
    # log the two decision heads independently. ``total`` is their sum.
    total: Tensor
    launch: Tensor
    target: Tensor


@dataclass(frozen=True)
class CollectionLogProbability:
    """Old-policy probabilities and validation produced by one collection pass."""

    # ``values`` stores one joint action log probability per environment.
    # ``validation_flags`` remains on the same device until the action batch is
    # transferred to Python for the Go request.
    values: Tensor
    validation_flags: Tensor


@dataclass(frozen=True)
class PPOUpdateMetrics:
    """Batch-weighted summaries from one complete PPO update."""

    # Losses and entropies are averaged over every transition use across all
    # epochs. They describe the data seen before each optimizer step rather
    # than only the final minibatch.
    policy_loss: float
    value_loss: float
    launch_entropy: float
    target_entropy: float
    total_loss: float
    approximate_kl: float
    clip_fraction: float

    # Gradient norms are measured before clipping. The maximum exposes a spike
    # that an average could hide; the mean describes the typical update.
    mean_gradient_norm: float
    max_gradient_norm: float

    # These confirm that one fixed normalization was calculated from the full
    # rollout rather than independently inside each minibatch.
    advantage_mean: float
    advantage_std: float
    initial_log_probability_max_error: float
    optimization_steps: int


def joint_log_prob(
    output: PolicyOutput,
    actions: SampledActionBatch,
    launchable_mask: Tensor,
    target_mask: Tensor,
) -> Tensor:
    """Return one log-probability for each game's structured action.

    Launch/no-launch probabilities contribute for launchable learner-owned
    sources. Target probabilities contribute only for sources that actually
    launched. Forced no-launch sources and padding never enter a distribution,
    which keeps their fully masked logits from producing ``NaN`` values.
    """

    _validate_joint_action_masks(output, actions, launchable_mask, target_mask)

    # Select before constructing the Bernoulli distribution. In particular,
    # this excludes padding and forced no-launch entries whose policy logits
    # are negative infinity.
    launch_logits = output.launch_logits[launchable_mask]
    _validate_launch_logits(launch_logits)

    # The empty selection still retains a zero-gradient connection to current
    # policy output. Consequently an all-forced batch returns exact zeros whose
    # backward pass succeeds instead of returning a detached constant.
    batch_size = output.launch_logits.shape[0]
    total = torch.zeros(
        batch_size,
        dtype=output.launch_logits.dtype,
        device=output.launch_logits.device,
    ) + launch_logits.sum() * 0.0

    if torch.any(launchable_mask):
        launch_choices = actions.launches[launchable_mask].to(launch_logits.dtype)
        launch_log_prob = Bernoulli(logits=launch_logits).log_prob(launch_choices)
        launch_games = launchable_mask.nonzero(as_tuple=True)[0]
        total = total.index_add(0, launch_games, launch_log_prob)

    # Target placeholders for sources that stayed idle carry no probability.
    # Gathering only launched rows also avoids constructing a Categorical over
    # any fully masked target row belonging to a forced no-launch source.
    if torch.any(actions.launches):
        legal_targets = target_mask[actions.launches]
        target_logits = output.target_logits[actions.launches].masked_fill(
            ~legal_targets,
            -torch.inf,
        )
        stored_targets = actions.targets[actions.launches]

        _validate_stored_targets(stored_targets, legal_targets, target_logits)

        target_log_prob = Categorical(logits=target_logits).log_prob(stored_targets)
        target_games = actions.launches.nonzero(as_tuple=True)[0]
        total = total.index_add(0, target_games, target_log_prob)

    return total


def collection_joint_log_prob(
    output: PolicyOutput,
    sample: CollectionActionSample,
    observation: ObservationBatch,
) -> CollectionLogProbability:
    """Calculate collection log probabilities and flags from saved selections.

    Structural checks such as shape, dtype, and device alignment still run
    eagerly because they inspect Python metadata without making Python wait for
    CUDA. Sampling already calculated the launchable and launched indices and
    selected their score rows. This function reuses those exact tensors for
    both probability math and validation instead of repeating boolean-mask
    indexing in two separate passes.

    The returned flags mean, in order:

    1. no action launches from a forbidden source;
    2. every launchable source has at least one legal target;
    3. every launchable source has a finite launch score;
    4. every sampled launch has at least one finite legal target score;
    5. every stored launch target is inside the 44-slot range;
    6. every stored launch target is legal for its source;
    7. every legal target score used by a launch is finite.

    The target used for the legality lookup is clamped only in a temporary
    tensor. This lets the function record an out-of-range target as ``False``
    without allowing an unsafe GPU indexing operation or changing the stored
    action. The eager validator still reports the range error first.
    """

    actions = sample.actions
    launchable_mask = observation.launchable_mask
    target_mask = observation.target_mask
    _validate_action_inputs(output, actions, launchable_mask, target_mask)

    valid_launch_sources, launchable_sources_have_targets = (
        _joint_action_mask_flags(actions, launchable_mask, target_mask)
    )
    launch_logits_are_finite = _launch_logits_are_valid(sample.launch_logits)
    sampled_rows_have_finite_targets = sampled_target_validation_flag(
        sample.target_logits
    )
    target_range_is_valid, selected_targets_are_legal, target_logits_are_finite = (
        _stored_target_flags(
            sample.sampled_targets,
            sample.legal_targets,
            sample.target_logits,
        )
    )

    validation_flags = torch.stack(
        (
            valid_launch_sources,
            launchable_sources_have_targets,
            launch_logits_are_finite,
            sampled_rows_have_finite_targets,
            target_range_is_valid,
            selected_targets_are_legal,
            target_logits_are_finite,
        )
    )

    # Keep a differentiable zero connection to the policy output for a batch
    # with no optional decisions, matching the eager joint_log_prob contract.
    batch_size = output.launch_logits.shape[0]
    total = torch.zeros(
        batch_size,
        dtype=output.launch_logits.dtype,
        device=output.launch_logits.device,
    ) + sample.launch_logits.sum() * 0.0

    launch_games = sample.launchable_indices[:, 0]
    launch_sources = sample.launchable_indices[:, 1]
    launch_choices = actions.launches[
        launch_games,
        launch_sources,
    ].to(sample.launch_logits.dtype)
    # Validation is intentionally disabled only for this deferred path. Bad
    # logits may produce NaN temporary values, but the accompanying flag fails
    # and a later substep will stop the action before either value is consumed.
    launch_log_probabilities = Bernoulli(
        logits=sample.launch_logits,
        validate_args=False,
    ).log_prob(launch_choices)
    total = total.index_add(0, launch_games, launch_log_probabilities)

    launched_games = sample.launched_indices[:, 0]
    target_log_probabilities = Categorical(
        logits=sample.target_logits,
        validate_args=False,
    ).log_prob(sample.sampled_targets)
    total = total.index_add(0, launched_games, target_log_probabilities)

    return CollectionLogProbability(
        values=total,
        validation_flags=validation_flags,
    )


def validate_collection_log_prob(
    output: PolicyOutput,
    sample: CollectionActionSample,
    observation: ObservationBatch,
) -> None:
    """Rerun eager probability checks without repeating selection work.

    This function is called only after the combined GPU flags report failure.
    It uses the original policy scores retained by ``CollectionActionSample``
    and performs no sampling, so it cannot change the rollout generator state.
    """

    _validate_joint_action_masks(
        output,
        sample.actions,
        observation.launchable_mask,
        observation.target_mask,
    )
    _validate_launch_logits(sample.launch_logits)
    _validate_stored_targets(
        sample.sampled_targets,
        sample.legal_targets,
        sample.target_logits,
    )


def entropy_bonus(
    output: PolicyOutput,
    owned_mask: Tensor,
    launchable_mask: Tensor,
    target_mask: Tensor,
) -> EntropyBonus:
    """Return structured action entropy averaged over owned bodies and games.

    A launchable source contributes ``H(launch) + P(launch) * H(target)``.
    Non-launchable owned sources contribute zero but remain in the owned-body
    denominator. A game with no owned bodies contributes exact zero rather
    than taking an empty mean.
    """

    _validate_policy_masks(output, launchable_mask, target_mask)
    expected_owned_shape = tuple(output.launch_logits.shape)
    if tuple(owned_mask.shape) != expected_owned_shape:
        raise ValueError(
            f"owned_mask has shape {tuple(owned_mask.shape)}, "
            f"want {expected_owned_shape}"
        )
    if owned_mask.dtype != torch.bool:
        raise TypeError(f"owned_mask must use torch.bool, got {owned_mask.dtype}")
    if owned_mask.device != output.launch_logits.device:
        raise ValueError("policy output and owned_mask must share one device")

    expected_launchable = owned_mask & target_mask.any(dim=-1)
    if not torch.equal(launchable_mask, expected_launchable):
        raise ValueError(
            "launchable_mask must identify exactly the owned sources with legal targets"
        )

    launch_by_source = torch.zeros_like(output.launch_logits)
    target_by_source = torch.zeros_like(output.launch_logits)

    # Selection happens before distribution construction. Fully masked target
    # rows and the -inf launch logits used for forced decisions are therefore
    # never evaluated and cannot create NaN values that masking would hide too
    # late.
    if torch.any(launchable_mask):
        launch_logits = output.launch_logits[launchable_mask]
        if not torch.all(torch.isfinite(launch_logits)):
            raise ValueError("launchable sources must have finite launch logits")
        launch_distribution = Bernoulli(logits=launch_logits)
        launch_entropy = launch_distribution.entropy()

        legal_targets = target_mask[launchable_mask]
        target_logits = output.target_logits[launchable_mask].masked_fill(
            ~legal_targets,
            -torch.inf,
        )
        if not torch.all(torch.isfinite(target_logits[legal_targets])):
            raise ValueError("legal targets must have finite target logits")
        target_entropy = Categorical(logits=target_logits).entropy()
        weighted_target_entropy = launch_distribution.probs * target_entropy

        launch_by_source = launch_by_source.masked_scatter(
            launchable_mask,
            launch_entropy,
        )
        target_by_source = target_by_source.masked_scatter(
            launchable_mask,
            weighted_target_entropy,
        )

    # Clamp only protects zero-owned rows from division by zero. Their sums are
    # already exactly zero, so dividing by one preserves the intended zero
    # contribution while ordinary games still divide by their true body count.
    owned_counts = owned_mask.sum(dim=-1)
    safe_owned_counts = owned_counts.clamp_min(1).to(output.launch_logits.dtype)
    launch_per_game = launch_by_source.sum(dim=-1) / safe_owned_counts
    target_per_game = target_by_source.sum(dim=-1) / safe_owned_counts

    mean_launch = launch_per_game.mean()
    mean_target = target_per_game.mean()
    return EntropyBonus(
        total=mean_launch + mean_target,
        launch=mean_launch,
        target=mean_target,
    )


def clipped_policy_loss(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float = 0.2,
) -> Tensor:
    """Return PPO's negative mean clipped surrogate objective.

    PPO describes an objective to maximize, while PyTorch optimizers minimize
    losses. The leading negative sign converts between those conventions.
    Positive and negative advantages both use the elementwise minimum before
    that sign is applied, preserving PPO's asymmetric clipping behavior.
    """

    _validate_aligned_floats(
        ("new_log_prob", new_log_prob),
        ("old_log_prob", old_log_prob),
        ("advantages", advantages),
    )
    _require_fixed_tensor("old_log_prob", old_log_prob)
    _require_fixed_tensor("advantages", advantages)
    if not 0.0 <= clip_epsilon < 1.0:
        raise ValueError(f"clip_epsilon must be in [0, 1), got {clip_epsilon}")
    for name, tensor in (
        ("new_log_prob", new_log_prob),
        ("old_log_prob", old_log_prob),
        ("advantages", advantages),
    ):
        if not torch.all(torch.isfinite(tensor)):
            raise ValueError(f"{name} must contain only finite values")

    probability_ratio = torch.exp(new_log_prob - old_log_prob)
    if not torch.all(torch.isfinite(probability_ratio)):
        raise ValueError("new and old log-probabilities produced a non-finite ratio")
    limited_ratio = probability_ratio.clamp(
        min=1.0 - clip_epsilon,
        max=1.0 + clip_epsilon,
    )
    ordinary_score = probability_ratio * advantages
    limited_score = limited_ratio * advantages
    return -torch.minimum(ordinary_score, limited_score).mean()


def value_loss(predicted_values: Tensor, return_targets: Tensor) -> Tensor:
    """Return the mean squared error between critic values and GAE targets."""

    _validate_aligned_floats(
        ("predicted_values", predicted_values),
        ("return_targets", return_targets),
    )
    _require_fixed_tensor("return_targets", return_targets)
    for name, tensor in (
        ("predicted_values", predicted_values),
        ("return_targets", return_targets),
    ):
        if not torch.all(torch.isfinite(tensor)):
            raise ValueError(f"{name} must contain only finite values")

    return torch.square(predicted_values - return_targets).mean()


def ppo_update(
    policy: nn.Module,
    optimizer: Optimizer,
    rollout: RolloutBatch,
    *,
    epochs: int,
    minibatch_size: int,
    clip_epsilon: float = 0.2,
    value_coefficient: float = 0.5,
    launch_entropy_coefficient: float = 0.05,
    target_entropy_coefficient: float = 0.01,
    max_gradient_norm: float = 0.5,
    advantage_epsilon: float = 1e-8,
    log_probability_atol: float | None = None,
    log_probability_rtol: float = DEFAULT_LOG_PROBABILITY_RTOL,
    shuffle_generator: torch.Generator | None = None,
) -> PPOUpdateMetrics:
    """Apply repeated PPO minibatch updates to one fixed rollout.

    The policy must remain in evaluation mode. Evaluation mode disables
    stochastic layers such as dropout but does not disable gradient tracking,
    so ``backward()`` still trains every parameter. This matches collection's
    old-policy mode and keeps its stored action probabilities meaningful.

    Raw advantages are normalized exactly once over the complete rollout.
    Return targets, rewards, and old log-probabilities remain unchanged. Every
    epoch shuffles the flattened transitions, while the same normalized
    advantage tensor is reused by every minibatch and epoch.

    Before training, each turn's stored log-probabilities are recomputed using
    the same environment batch shape used during collection. This avoids
    mistaking batch-shape-dependent transformer rounding for changed actions,
    masks, or policy parameters. ``log_probability_atol`` and
    ``log_probability_rtol`` supply only a small absolute-plus-relative
    allowance for device-level nondeterminism when repeating that calculation.
    The defaults are based on the V1 float32 contract and assume sufficient
    headroom for CUDA reductions and TF32 kernels; PR 15's first GPU run must
    measure and confirm that assumption.

    ``shuffle_generator`` controls only transition ordering. Unlike the action
    sampling generator passed to ``collect_rollout``, it may stay on the CPU
    while a policy runs elsewhere because the resulting indices are moved to
    the rollout device before they select tensors.

    Configuration and stored-probability checks finish before any parameter
    changes. Runtime failures after optimization begins are fail-fast but not
    transactional: an error in a later minibatch leaves earlier optimizer
    steps applied, so a run should stop and reload its last checkpoint rather
    than continuing from the partially completed update.
    """

    _validate_update_configuration(
        policy,
        optimizer,
        rollout,
        epochs=epochs,
        minibatch_size=minibatch_size,
        clip_epsilon=clip_epsilon,
        value_coefficient=value_coefficient,
        launch_entropy_coefficient=launch_entropy_coefficient,
        target_entropy_coefficient=target_entropy_coefficient,
        max_gradient_norm=max_gradient_norm,
        advantage_epsilon=advantage_epsilon,
        log_probability_atol=log_probability_atol,
        log_probability_rtol=log_probability_rtol,
    )

    advantage_mean = rollout.advantages.mean()
    advantage_std = rollout.advantages.std(unbiased=False)
    normalized_advantages = (
        rollout.advantages - advantage_mean
    ) / (advantage_std + advantage_epsilon)

    # Check the entire fixed rollout before the first optimizer step. A later
    # mismatch would be ambiguous because changing policy parameters is the
    # purpose of PPO; here it unambiguously means collection and update do not
    # agree on the stored action probability.
    roundtrip_atol = _log_probability_atol(
        rollout.old_log_probabilities,
        log_probability_atol,
    )
    initial_log_probability_max_error = _verify_old_log_probabilities(
        policy,
        rollout,
        atol=roundtrip_atol,
        rtol=log_probability_rtol,
    )

    # All minibatches have the same size, so each optimizer step has equal
    # weight in these means. Keep the values as one device tensor throughout
    # the loop; converting every scalar to a Python float here would force a
    # GPU to stop and synchronize eight times after every optimizer step.
    metric_totals = torch.zeros(
        8,
        dtype=rollout.values.dtype,
        device=rollout.values.device,
    )
    max_observed_gradient_norm = torch.zeros(
        (),
        dtype=rollout.values.dtype,
        device=rollout.values.device,
    )
    optimization_steps = 0
    trainable_parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]

    for _ in range(epochs):
        permutation = _random_permutation(
            rollout.transition_count,
            rollout.values.device,
            shuffle_generator,
        )
        for start in range(0, rollout.transition_count, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            observation = rollout.select_observations(indices)
            actions = rollout.select_actions(indices)
            output = run_policy(policy, observation)
            new_log_probability = joint_log_prob(
                output,
                actions,
                observation.launchable_mask,
                observation.target_mask,
            )
            old_log_probability = rollout.old_log_probabilities[indices]
            minibatch_advantages = normalized_advantages[indices]

            actor_loss = clipped_policy_loss(
                new_log_probability,
                old_log_probability,
                minibatch_advantages,
                clip_epsilon=clip_epsilon,
            )
            critic_loss = value_loss(
                output.values,
                rollout.return_targets[indices],
            )
            entropy = entropy_bonus(
                output,
                observation.owned_mask,
                observation.launchable_mask,
                observation.target_mask,
            )
            total_loss = (
                actor_loss
                + value_coefficient * critic_loss
                - launch_entropy_coefficient * entropy.launch
                - target_entropy_coefficient * entropy.target
            )
            if not torch.isfinite(total_loss):
                raise ValueError("PPO losses produced a non-finite total")

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_gradient_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()

            # Approximate KL and clip fraction are diagnostics only. Detaching
            # prevents metric bookkeeping from retaining each minibatch graph.
            with torch.no_grad():
                log_ratio = new_log_probability - old_log_probability
                probability_ratio = torch.exp(log_ratio)
                approximate_kl = ((probability_ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    torch.abs(probability_ratio - 1.0) > clip_epsilon
                ).to(output.values.dtype).mean()

            metric_totals.add_(
                torch.stack(
                    (
                        actor_loss.detach(),
                        critic_loss.detach(),
                        entropy.launch.detach(),
                        entropy.target.detach(),
                        total_loss.detach(),
                        approximate_kl.detach(),
                        clip_fraction.detach(),
                        gradient_norm.detach().to(rollout.values.dtype),
                    )
                )
            )
            max_observed_gradient_norm = torch.maximum(
                max_observed_gradient_norm,
                gradient_norm.detach().to(rollout.values.dtype),
            )
            optimization_steps += 1

    # This is the only metrics transfer from the policy device to the CPU.
    # Keeping the values together avoids turning routine logging into repeated
    # synchronization points inside the optimizer loop.
    summary = torch.cat(
        (
            metric_totals / optimization_steps,
            max_observed_gradient_norm.unsqueeze(0),
            advantage_mean.unsqueeze(0),
            advantage_std.unsqueeze(0),
            initial_log_probability_max_error.unsqueeze(0),
        )
    ).detach().cpu().tolist()
    (
        policy_loss_value,
        value_loss_value,
        launch_entropy_value,
        target_entropy_value,
        total_loss_value,
        approximate_kl_value,
        clip_fraction_value,
        mean_gradient_norm_value,
        max_gradient_norm_value,
        advantage_mean_value,
        advantage_std_value,
        initial_log_probability_max_error_value,
    ) = summary

    return PPOUpdateMetrics(
        policy_loss=policy_loss_value,
        value_loss=value_loss_value,
        launch_entropy=launch_entropy_value,
        target_entropy=target_entropy_value,
        total_loss=total_loss_value,
        approximate_kl=approximate_kl_value,
        clip_fraction=clip_fraction_value,
        mean_gradient_norm=mean_gradient_norm_value,
        max_gradient_norm=max_gradient_norm_value,
        advantage_mean=advantage_mean_value,
        advantage_std=advantage_std_value,
        initial_log_probability_max_error=(
            initial_log_probability_max_error_value
        ),
        optimization_steps=optimization_steps,
    )


def _validate_update_configuration(
    policy: nn.Module,
    optimizer: Optimizer,
    rollout: RolloutBatch,
    *,
    epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    value_coefficient: float,
    launch_entropy_coefficient: float,
    target_entropy_coefficient: float,
    max_gradient_norm: float,
    advantage_epsilon: float,
    log_probability_atol: float | None,
    log_probability_rtol: float,
) -> None:
    """Reject an invalid update before changing any model or optimizer state."""

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if (
        isinstance(minibatch_size, bool)
        or not isinstance(minibatch_size, int)
        or minibatch_size < 1
        or minibatch_size > rollout.transition_count
    ):
        raise ValueError(
            "minibatch_size must be a positive integer no larger than the rollout"
        )
    if rollout.transition_count % minibatch_size != 0:
        raise ValueError(
            "minibatch_size must divide the rollout exactly so every optimizer "
            "step has equal weight"
        )
    if not 0.0 <= clip_epsilon < 1.0:
        raise ValueError(f"clip_epsilon must be in [0, 1), got {clip_epsilon}")
    for name, value in (
        ("value_coefficient", value_coefficient),
        ("launch_entropy_coefficient", launch_entropy_coefficient),
        ("target_entropy_coefficient", target_entropy_coefficient),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")
    if not math.isfinite(max_gradient_norm) or max_gradient_norm <= 0.0:
        raise ValueError(
            "max_gradient_norm must be finite and positive, "
            f"got {max_gradient_norm}"
        )
    if not math.isfinite(advantage_epsilon) or advantage_epsilon <= 0.0:
        raise ValueError(
            f"advantage_epsilon must be finite and positive, got {advantage_epsilon}"
        )
    if log_probability_atol is not None and (
        not math.isfinite(log_probability_atol)
        or log_probability_atol < 0.0
    ):
        raise ValueError(
            "log_probability_atol must be finite and non-negative, "
            f"got {log_probability_atol}"
        )
    if not math.isfinite(log_probability_rtol) or log_probability_rtol < 0.0:
        raise ValueError(
            "log_probability_rtol must be finite and non-negative, "
            f"got {log_probability_rtol}"
        )
    require_evaluation_mode(policy, "PPO updates")
    current_policy_device = policy_device(policy)
    if current_policy_device != rollout.values.device:
        raise ValueError(
            f"policy uses {current_policy_device}, "
            f"but rollout uses {rollout.values.device}"
        )

    policy_parameters = {id(parameter) for parameter in policy.parameters()}
    trainable_parameters = {
        id(parameter) for parameter in policy.parameters() if parameter.requires_grad
    }
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if not trainable_parameters:
        raise ValueError("policy must contain at least one trainable parameter")
    if missing := trainable_parameters - optimizer_parameters:
        raise ValueError(
            f"optimizer is missing {len(missing)} trainable policy parameters"
        )
    if foreign := optimizer_parameters - policy_parameters:
        raise ValueError(
            f"optimizer contains {len(foreign)} parameters outside the policy"
        )


@torch.no_grad()
def _verify_old_log_probabilities(
    policy: nn.Module,
    rollout: RolloutBatch,
    *,
    atol: float,
    rtol: float,
) -> Tensor:
    """Replay collection-shaped turns before changing policy weights."""

    max_error = torch.zeros(
        (),
        dtype=rollout.old_log_probabilities.dtype,
        device=rollout.old_log_probabilities.device,
    )
    largest_excess = torch.full_like(max_error, -torch.inf)
    # Collection called the policy once per turn with B environment rows. Use
    # those same contiguous time-major groups here. Replaying with an arbitrary
    # PPO minibatch shape changes transformer reduction order and can create a
    # harmless rounding difference large enough to resemble policy drift.
    for indices in rollout.collection_groups():
        observation = rollout.select_observations(indices)
        output = run_policy(policy, observation)
        recomputed = joint_log_prob(
            output,
            rollout.select_actions(indices),
            observation.launchable_mask,
            observation.target_mask,
        )
        stored = rollout.old_log_probabilities[indices]
        difference = torch.abs(recomputed - stored)
        allowed = atol + rtol * torch.abs(stored)
        max_error = torch.maximum(max_error, difference.max())
        largest_excess = torch.maximum(
            largest_excess,
            (difference - allowed).max(),
        )

    # A single host check is required before the optimizer may mutate state.
    if bool((largest_excess > 0).cpu()):
        max_error_value, largest_excess_value = (
            torch.stack((max_error, largest_excess)).cpu().tolist()
        )
        raise ValueError(
            "stored and recomputed log-probabilities differ before the first "
            f"optimizer step: max error {max_error_value}, largest tolerance "
            f"excess {largest_excess_value}, atol {atol}, rtol {rtol}"
        )
    return max_error


def _log_probability_atol(
    old_log_probabilities: Tensor,
    requested: float | None,
) -> float:
    """Use a dtype-aware zero-scale floor unless explicitly overridden."""

    if requested is not None:
        return requested
    dtype_floor = 8.0 * torch.finfo(old_log_probabilities.dtype).eps
    if old_log_probabilities.dtype == torch.float32:
        return max(DEFAULT_FLOAT32_LOG_PROBABILITY_ATOL, dtype_floor)
    return dtype_floor


def _random_permutation(
    transition_count: int,
    rollout_device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    """Shuffle on the generator's device, then move indices beside the rollout."""

    permutation_device = (
        torch.device(generator.device)
        if generator is not None
        else torch.device("cpu")
    )
    permutation = torch.randperm(
        transition_count,
        device=permutation_device,
        generator=generator,
    )
    return permutation.to(rollout_device)


def _validate_action_inputs(
    output: PolicyOutput,
    actions: SampledActionBatch,
    launchable_mask: Tensor,
    target_mask: Tensor,
) -> None:
    """Reject masks and action tensors that do not describe the same games."""

    _validate_policy_masks(output, launchable_mask, target_mask)
    batch_size = output.launch_logits.shape[0]
    if actions.launches.shape[0] != batch_size:
        raise ValueError("policy output and actions must contain the same batch size")

    if actions.launches.device != output.launch_logits.device:
        raise ValueError("policy output and actions must share one device")


def _validate_joint_action_masks(
    output: PolicyOutput,
    actions: SampledActionBatch,
    launchable_mask: Tensor,
    target_mask: Tensor,
) -> None:
    """Reject structurally invalid masks or a launch from a forbidden source.

    The helper only reads already-produced tensors. It neither samples an
    action nor changes model state, so the collection path can rerun it after a
    deferred validation flag reaches Python and still preserve replay state.
    """

    _validate_action_inputs(output, actions, launchable_mask, target_mask)
    valid_launch_sources, launchable_sources_have_targets = (
        _joint_action_mask_flags(actions, launchable_mask, target_mask)
    )
    if not valid_launch_sources:
        raise ValueError("a stored launch uses a source that is not launchable")
    if not launchable_sources_have_targets:
        raise ValueError("every launchable source must have at least one legal target")


def _joint_action_mask_flags(
    actions: SampledActionBatch,
    launchable_mask: Tensor,
    target_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build source-legality flags without converting them to Python values."""

    return (
        ~(actions.launches & ~launchable_mask).any(),
        ~(launchable_mask & ~target_mask.any(dim=-1)).any(),
    )


def _launch_logits_are_valid(launch_logits: Tensor) -> Tensor:
    """Build the finite-launch-score flag on the logits' current device."""

    return torch.isfinite(launch_logits).all()


def _validate_launch_logits(launch_logits: Tensor) -> None:
    """Require every optional launch decision to have a finite score."""

    if not _launch_logits_are_valid(launch_logits):
        raise ValueError("launchable sources must have finite launch logits")


def _validate_stored_targets(
    stored_targets: Tensor,
    legal_targets: Tensor,
    target_logits: Tensor,
) -> None:
    """Check target range, legality, and scores without changing the action.

    Checks remain in their original order so a malformed action reports the
    same first error as before this validation code was extracted.
    """

    target_range_is_valid, selected_targets_are_legal, target_logits_are_finite = (
        _stored_target_flags(stored_targets, legal_targets, target_logits)
    )
    if not target_range_is_valid:
        raise ValueError("every stored launch target must be a valid body slot")
    if not selected_targets_are_legal:
        raise ValueError("every stored launch target must be legal")
    if not target_logits_are_finite:
        raise ValueError("legal targets must have finite target logits")


def _stored_target_flags(
    stored_targets: Tensor,
    legal_targets: Tensor,
    target_logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build target validation flags while keeping unsafe indices out of gather.

    Clamping creates an indexing-safe copy only for checking legality. The
    separate range flag still rejects the original invalid target, and no
    replacement action can reach the environment.
    """

    target_range_is_valid = (
        (stored_targets >= 0) & (stored_targets < BODY_SLOTS)
    ).all()
    safe_targets = stored_targets.clamp(0, BODY_SLOTS - 1)
    selected_targets_are_legal = legal_targets.gather(
        1,
        safe_targets.unsqueeze(1),
    ).all()
    # Treat masked entries as automatically valid instead of selecting legal
    # entries into another dynamically sized tensor. This keeps the collection
    # path free from an extra boolean-index operation while testing the same
    # condition as ``isfinite(target_logits[legal_targets]).all()``.
    target_logits_are_finite = (
        torch.isfinite(target_logits) | ~legal_targets
    ).all()
    return (
        target_range_is_valid,
        selected_targets_are_legal,
        target_logits_are_finite,
    )


def _validate_policy_masks(
    output: PolicyOutput,
    launchable_mask: Tensor,
    target_mask: Tensor,
) -> None:
    """Reject masks that do not align with the policy's game and body slots."""

    if output.target_logits.dtype != output.launch_logits.dtype:
        raise TypeError(
            "launch_logits and target_logits must use the same floating-point dtype"
        )

    batch_size = output.launch_logits.shape[0]
    expected_launchable_shape = (batch_size, BODY_SLOTS)
    if tuple(launchable_mask.shape) != expected_launchable_shape:
        raise ValueError(
            f"launchable_mask has shape {tuple(launchable_mask.shape)}, "
            f"want {expected_launchable_shape}"
        )
    expected_target_shape = (batch_size, BODY_SLOTS, BODY_SLOTS)
    if tuple(target_mask.shape) != expected_target_shape:
        raise ValueError(
            f"target_mask has shape {tuple(target_mask.shape)}, "
            f"want {expected_target_shape}"
        )
    for name, mask in (
        ("launchable_mask", launchable_mask),
        ("target_mask", target_mask),
    ):
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must use torch.bool, got {mask.dtype}")

    devices = {
        output.launch_logits.device,
        launchable_mask.device,
        target_mask.device,
    }
    if len(devices) != 1:
        raise ValueError(f"policy output and masks must share one device, got {devices}")


def _validate_aligned_floats(*named_tensors: tuple[str, Tensor]) -> None:
    """Require non-empty floating tensors with matching shape, dtype, and device."""

    first_name, first = named_tensors[0]
    if first.numel() == 0:
        raise ValueError(f"{first_name} must contain at least one value")
    expected_shape = tuple(first.shape)
    expected_dtype = first.dtype
    expected_device = first.device

    for name, tensor in named_tensors:
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype, got {tensor.dtype}")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}, want {expected_shape}"
            )
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"{name} must use {expected_dtype} to match {first_name}, "
                f"got {tensor.dtype}"
            )
        if tensor.device != expected_device:
            raise ValueError(
                f"{name} must use {expected_device} to match {first_name}, "
                f"got {tensor.device}"
            )


def _require_fixed_tensor(name: str, tensor: Tensor) -> None:
    """Reject rollout data that could send gradients into its fixed target."""

    if tensor.requires_grad:
        raise ValueError(f"{name} must be detached from the rollout graph")
