"""Probability, loss, and optimizer calculations used by Orrery PPO.

Where this file fits in one training update:

1. Collection (rollout.py) plays ``environment_count`` games side by side
   for ``rollout_steps`` turns, both set in the training config. Each
   turn the model scores every possible move, model.py samples one, and
   ``collection_joint_log_prob`` here records how likely that move was.
2. gae.py turns the finished games into an **advantage** for every turn: how
   much better (positive) or worse (negative) the turn went than the critic,
   the model's own prediction of the result, expected.
3. ``ppo_update`` here adjusts the model's weights so moves with a positive
   advantage become more likely and moves with a negative one less likely.

Words used throughout this file:

- **logit**: a raw score from the model, any real number. A higher logit means
  a more likely choice. Sigmoid (for yes/no) or softmax (for picking one of
  many) turns logits into probabilities.
- **log-probability**: the natural log of a probability. log(1) = 0, and a
  smaller probability gives a more negative number: log(0.5) is about -0.69.
  Logs turn a product of probabilities into a sum, which stays accurate where
  multiplying many small probabilities would round to zero.
- **joint**: covering the learner's whole move for one turn. The learner
  decides for every planet it owns, so the move's probability is the product
  of every decision's probability, and its log-probability is their sum.
- **transition**: one turn of one game together with the learner's move.
  One update holds ``environment_count`` x ``rollout_steps`` transitions.
- **old policy**: the model's weights while the games were played. The update
  changes the weights and keeps comparing the new probabilities with the old.
"""

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


# Tolerances for the safety check in ``_verify_old_log_probabilities``. Before
# training, the update recomputes every stored log-probability and requires it
# to match what collection recorded. Computers round floating-point numbers,
# so "match" means "within an allowed difference":
#
#     allowed = ATOL + RTOL x |stored value|
#
# ATOL (absolute tolerance) is a fixed allowance. RTOL (relative tolerance)
# grows with the size of the value. For example, if both were 0.001, a stored
# value of -2.0 could differ by up to 0.001 + 0.001 x 2.0 = 0.003.
#
# Collection-shaped replay is bit-identical on the tested CPU path. V1 uses
# float32, for which this floor is assumed to leave room for ordinary
# CUDA/TF32 replay rounding without accepting policy-scale drift. PR 15's
# first GPU run must measure and confirm that assumption.
DEFAULT_FLOAT32_LOG_PROBABILITY_ATOL = 1e-5
DEFAULT_LOG_PROBABILITY_RTOL = 1e-5


@dataclass(frozen=True)
class EntropyBonus:
    """Batch-mean entropy values used to encourage policy exploration.

    Entropy measures how spread out a choice's probabilities are. A certain
    choice (100% launch) has entropy 0. A 50/50 launch-or-hold choice has the
    largest possible entropy for two options, log(2), about 0.69. Training
    adds a small reward for entropy so the model keeps trying less-favoured
    moves (exploration) instead of locking in its first guesses too early.
    """

    # ``launch`` and ``target`` stay separate so training can weight and log
    # the two decisions independently: the training config sets their weights
    # separately (``launch_entropy_*`` and ``target_entropy_*``). Each value
    # is a zero-dimensional tensor (a single number) that still carries
    # gradients, so the loss can push on it.
    # total: launch + target.
    total: Tensor
    # launch: average entropy of the launch-or-hold decision per owned planet.
    launch: Tensor
    # target: average entropy of the target choice per owned planet, weighted
    # by how likely that planet is to launch at all.
    target: Tensor


@dataclass(frozen=True)
class CollectionLogProbability:
    """Old-policy probabilities and validation produced by one collection pass.

    Collection creates one of these every turn. ``ppo_update`` later needs the
    stored log-probabilities as its "old policy" reference.
    """

    # values: shape [games]. One joint log-probability per game: the sum of
    # the log-probabilities of every decision the learner made that turn.
    values: Tensor
    # validation_flags: shape [7], one true/false per safety check (listed in
    # ``collection_joint_log_prob``). They stay on the GPU instead of raising
    # errors here. Reading a GPU value from Python makes the CPU wait for the
    # GPU to finish, so rollout.py copies these flags together with the chosen
    # actions in one transfer per turn, and raises only if a flag is false.
    validation_flags: Tensor


@dataclass(frozen=True)
class PPOUpdateMetrics:
    """Batch-weighted summaries from one complete PPO update.

    Training logs these after every update. They are plain Python floats,
    copied off the GPU once at the end of ``ppo_update``.
    """

    # Losses and entropies are averaged over every transition use across all
    # epochs. They describe the data seen before each optimizer step rather
    # than only the final minibatch.
    # policy_loss: the actor's loss from ``clipped_policy_loss``. More negative
    # means the update made positive-advantage moves more likely.
    policy_loss: float
    # value_loss: the critic's mean squared prediction error, from
    # ``value_loss``. Lower means it predicts game results better.
    value_loss: float
    # launch_entropy, target_entropy: how spread out the two decisions still
    # are (see ``EntropyBonus``). Falling values mean the model is growing
    # more certain.
    launch_entropy: float
    target_entropy: float
    # total_loss: the single number the optimizer reduced, combining the
    # three parts above (see ``ppo_update``).
    total_loss: float
    # approximate_kl: an estimate of how far the move probabilities moved away
    # from the old policy. 0 means unchanged. The config's
    # ``approximate_kl_warning`` records the level considered concerning: a
    # sign that one update changed the model too much.
    approximate_kl: float
    # clip_fraction: the share of transitions, 0 to 1, whose probability ratio
    # left the clip range (1 - clip_epsilon to 1 + clip_epsilon), so the clip
    # stopped them pushing further.
    clip_fraction: float

    # A gradient says, for every weight, which way and how strongly to change
    # it to reduce the loss. Its norm is its overall size, one number. Training
    # shrinks any gradient whose norm exceeds the config's
    # ``max_gradient_norm``, which is called gradient clipping.
    # Gradient norms are measured before clipping. The maximum exposes a spike
    # that an average could hide; the mean describes the typical update.
    mean_gradient_norm: float
    max_gradient_norm: float

    # These confirm that one fixed normalization was calculated from the full
    # rollout rather than independently inside each minibatch.
    # advantage_mean, advantage_std: the average and spread of the raw
    # advantages from gae.py, before normalization.
    advantage_mean: float
    advantage_std: float
    # initial_log_probability_max_error: the largest gap between a stored and
    # a recomputed log-probability in the pre-training check. Near 0 is normal.
    initial_log_probability_max_error: float
    # optimization_steps: how many times the weights changed. It equals
    # epochs x (transitions / minibatch_size), where the config sets
    # ``ppo_epochs`` and ``minibatch_size``.
    optimization_steps: int


def joint_log_prob(
    output: PolicyOutput,
    actions: SampledActionBatch,
    launchable_mask: Tensor,
    target_mask: Tensor,
) -> Tensor:
    """Return one log-probability for each game's whole move this turn.

    "How likely was this move under the model's current weights?" ``ppo_update``
    asks this for every stored move: once before training, to check the stored
    answers (``_verify_old_log_probabilities``), and again in every minibatch,
    where the answer changes as the weights change.

    Inputs, for a batch of games (``B``) with ``BODY_SLOTS`` body slots each:

    - ``output``: the model's scores. ``launch_logits`` is [B, slots], one
      launch score per body. ``target_logits`` is [B, slots, slots], one score
      per (source body, target body) pair.
    - ``actions``: the move that was actually played. ``launches`` is
      [B, slots] true/false; ``targets`` is [B, slots], the chosen target
      slot, meaningful only where ``launches`` is true.
    - ``launchable_mask``: [B, slots], true where the learner had a real
      launch-or-hold choice: a body it owns with at least one ship.
    - ``target_mask``: [B, slots, slots], true where a launch from the source
      to the target is allowed.

    Worked example: in one game the learner owns two bodies.

    | Body | Decision | Probability of what was chosen | Log |
    |---|---|---|---|
    | A | launch | 0.6 | -0.51 |
    | A | target X | 0.5 | -0.69 |
    | B | hold | 0.8 | -0.22 |

    The game's joint log-probability is -0.51 - 0.69 - 0.22 = -1.42. Body B
    adds no target term because it held. Bodies with no real choice add
    nothing: an empty slot, an enemy body, or an owned body with no ships.
    Their only possible decision is "hold", with probability 1 and log 0, so
    leaving them out doesn't change the sum.
    """

    # Stop with a clear error if the inputs don't describe the same games, or
    # if a stored move launches from a body that couldn't launch.
    _validate_joint_action_masks(output, actions, launchable_mask, target_mask)

    # Keep only the launch scores of bodies with a real choice. The rest are
    # -infinity, the model's way of saying "impossible", and leaving them out
    # keeps them from producing NaN ("not a number") values in the maths below.
    # Indexing with a true/false mask flattens the result to a 1-D list in
    # reading order: game 0's choices first, then game 1's, and so on.
    launch_logits = output.launch_logits[launchable_mask]
    _validate_launch_logits(launch_logits)

    # Start every game's total at 0. Adding ``launch_logits.sum() * 0.0``
    # changes no number, but it links the result to the model's outputs.
    # Without it, a batch where no body had a choice would return a plain
    # constant, and PyTorch's backward() (which traces the loss back to the
    # weights) would fail with "does not require grad".
    batch_size = output.launch_logits.shape[0]
    total = torch.zeros(
        batch_size,
        dtype=output.launch_logits.dtype,
        device=output.launch_logits.device,
    ) + launch_logits.sum() * 0.0

    # Part 1: the launch-or-hold decision of every body with a choice.
    if torch.any(launchable_mask):
        # The decision that was played, as 1.0 (launched) or 0.0 (held), in
        # the same flattened order as ``launch_logits``.
        launch_choices = actions.launches[launchable_mask].to(launch_logits.dtype)
        # Bernoulli is a yes/no distribution. With probability
        # p = sigmoid(logit), log_prob gives log(p) for a launch and
        # log(1 - p) for a hold.
        launch_log_prob = Bernoulli(logits=launch_logits).log_prob(launch_choices)
        # Which game each flattened entry came from, e.g. [0, 0, 1].
        launch_games = launchable_mask.nonzero(as_tuple=True)[0]
        # index_add adds each entry into its own game's total. With games
        # [0, 0, 1] and logs [-0.51, -0.22, -0.36], game 0 receives -0.73 and
        # game 1 receives -0.36.
        total = total.index_add(0, launch_games, launch_log_prob)

    # Part 2: the target choice, only for bodies that launched. A body that
    # held still has a placeholder target in ``actions.targets``, which means
    # nothing and must not count. Bodies with no real choice have every
    # target forbidden, and a choice among zero allowed targets would give
    # NaN, so selecting launched rows only also keeps them out.
    if torch.any(actions.launches):
        # One row per launching body: which targets were allowed, and the
        # model's score for each. Forbidden targets get -infinity, so softmax
        # below gives them probability 0. This repeats the masking done during
        # sampling, so a score for a forbidden target can never count.
        legal_targets = target_mask[actions.launches]
        target_logits = output.target_logits[actions.launches].masked_fill(
            ~legal_targets,
            -torch.inf,
        )
        # The target slot each launching body actually chose.
        stored_targets = actions.targets[actions.launches]

        # Stop with a clear error if a stored target is out of range, was not
        # allowed, or has a broken (infinite or NaN) score.
        _validate_stored_targets(stored_targets, legal_targets, target_logits)

        # Categorical picks one of many options. It applies softmax to each
        # row's scores, so they become probabilities summing to 1, and
        # log_prob gives the log of the chosen target's probability.
        target_log_prob = Categorical(logits=target_logits).log_prob(stored_targets)
        # As in part 1: add each launching body's log into its game's total.
        target_games = actions.launches.nonzero(as_tuple=True)[0]
        total = total.index_add(0, target_games, target_log_prob)

    # Shape [B]: one joint log-probability per game.
    return total


def collection_joint_log_prob(
    output: PolicyOutput,
    sample: CollectionActionSample,
    observation: ObservationBatch,
) -> CollectionLogProbability:
    """Calculate collection log probabilities and flags from saved selections.

    The same answer as ``joint_log_prob``, computed during collection, right
    after a move is sampled, and stored as the "old policy" probability that
    ``ppo_update`` later compares against. Two things make it differ:

    - **It reuses sampling's work.** While sampling, model.py already found
      which bodies had a choice and which launched, and picked out their
      score rows (``sample``). This function uses those same rows instead of
      searching the masks again.
    - **It doesn't stop to check.** Instead of raising an error, each safety
      check becomes a true/false flag that stays on the GPU. Raising needs
      Python to read a GPU value, which makes the CPU wait for the GPU to
      finish. rollout.py reads all the flags in the turn's single GPU-to-CPU
      copy, and only if one is false calls ``validate_collection_log_prob``
      to raise the matching error.

    Checks that only look at shapes, types and devices still raise right away:
    Python knows those without asking the GPU.

    The returned flags mean, in order:

    1. no action launches from a forbidden source;
    2. every launchable source has at least one legal target;
    3. every launchable source has a finite launch score;
    4. every sampled launch has at least one finite legal target score;
    5. every stored launch target is inside the ``BODY_SLOTS`` range;
    6. every stored launch target is legal for its source;
    7. every legal target score used by a launch is finite.

    For flag 6, a target outside the slot range would crash the lookup, so
    the lookup uses a temporary copy with each target forced into range.
    Flag 5 still reports the real target as invalid, and the stored action is
    never changed.
    """

    actions = sample.actions
    launchable_mask = observation.launchable_mask
    target_mask = observation.target_mask
    # Shape, type and device checks: these raise immediately, because Python
    # can read them without waiting for the GPU.
    _validate_action_inputs(output, actions, launchable_mask, target_mask)

    # Build the seven safety flags. Each is a single true/false value that
    # stays on the GPU; nothing here raises.
    # Flags 1 and 2: no launch from a body that couldn't launch, and every
    # body with a choice has at least one allowed target.
    valid_launch_sources, launchable_sources_have_targets = (
        _joint_action_mask_flags(actions, launchable_mask, target_mask)
    )
    # Flag 3: every launch score is a real number, not infinity or NaN.
    launch_logits_are_finite = _launch_logits_are_valid(sample.launch_logits)
    # Flag 4: every launching body had at least one allowed target with a
    # real-number score, so sampling had something to choose from.
    sampled_rows_have_finite_targets = sampled_target_validation_flag(
        sample.target_logits
    )
    # Flags 5, 6 and 7: each chosen target is a real slot number, was
    # allowed, and every allowed target's score is a real number.
    target_range_is_valid, selected_targets_are_legal, target_logits_are_finite = (
        _stored_target_flags(
            sample.sampled_targets,
            sample.legal_targets,
            sample.target_logits,
        )
    )

    # Put the seven flags into one tensor of shape [7], in the order listed
    # in the docstring, so rollout.py can copy them in one go.
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

    # Now the log-probability itself, in the same two parts as
    # ``joint_log_prob``. Start every game at 0, linked to the model's outputs
    # by ``* 0.0`` for the same reason given there.
    batch_size = output.launch_logits.shape[0]
    total = torch.zeros(
        batch_size,
        dtype=output.launch_logits.dtype,
        device=output.launch_logits.device,
    ) + sample.launch_logits.sum() * 0.0

    # Part 1: launch or hold. ``launchable_indices`` has one row per body with
    # a choice, as [game, body slot], e.g. [[0, 3], [0, 7], [1, 2]].
    launch_games = sample.launchable_indices[:, 0]
    launch_sources = sample.launchable_indices[:, 1]
    # The decision that was played for each of those bodies: 1.0 launched,
    # 0.0 held.
    launch_choices = actions.launches[
        launch_games,
        launch_sources,
    ].to(sample.launch_logits.dtype)
    # ``validate_args=False`` turns off PyTorch's own input checks, which
    # would make the CPU wait for the GPU. Broken scores may give NaN here,
    # but flag 3 is then false and rollout.py raises before the value is used.
    launch_log_probabilities = Bernoulli(
        logits=sample.launch_logits,
        validate_args=False,
    ).log_prob(launch_choices)
    total = total.index_add(0, launch_games, launch_log_probabilities)

    # Part 2: the target, for launching bodies only. Sampling already
    # selected their score rows (forbidden targets set to -infinity) and the
    # chosen targets, in the same order as ``launched_indices``.
    launched_games = sample.launched_indices[:, 0]

    # For each launching body: the log of the probability the model gave to
    # the target it actually chose. ``Categorical`` is a "pick one of many"
    # choice. It applies softmax to each row of target scores, so they become
    # probabilities summing to 1, and ``log_prob`` reads the chosen slot's.
    target_log_probabilities = Categorical(
        logits=sample.target_logits,
        validate_args=False,
    ).log_prob(sample.sampled_targets)
    # Add each launching body's value into its own game's total.
    total = total.index_add(0, launched_games, target_log_probabilities)

    # The probabilities and the flags travel together. rollout.py checks the
    # flags before the move reaches the Go environment.
    return CollectionLogProbability(
        values=total,
        validation_flags=validation_flags,
    )


def validate_collection_log_prob(
    output: PolicyOutput,
    sample: CollectionActionSample,
    observation: ObservationBatch,
) -> None:
    """Raise the detailed error behind a failed collection flag.

    ``collection_joint_log_prob`` only reports *that* a check failed, as a
    false flag. rollout.py calls this function only then, to find out *which*
    check failed and raise an error that says so. It reruns the raising
    versions of the same checks on the same saved scores and move.

    It samples nothing, so the random-number generator used for sampling is
    left exactly where it was. A replay of the run therefore still draws the
    same random numbers.
    """

    # Flags 1 and 2: launches only from bodies with a choice, and every such
    # body has an allowed target.
    _validate_joint_action_masks(
        output,
        sample.actions,
        observation.launchable_mask,
        observation.target_mask,
    )
    # Flag 3: every launch score is a real number.
    _validate_launch_logits(sample.launch_logits)
    # Flags 5, 6 and 7: each chosen target is in range, allowed, and every
    # allowed target's score is a real number. (Flag 4 is rechecked by
    # model.py's ``validate_collection_sample``, which rollout.py calls first.)
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
        # Walk through the shuffled transitions one minibatch at a time. Each
        # pass of this loop trains on one minibatch and changes the weights
        # once. ``range(0, count, size)`` counts up in steps of the minibatch
        # size, giving where each minibatch starts; the next line cuts out
        # that slice of the shuffled order.
        #
        # Example with 12 transitions and a minibatch size of 4:
        #
        #     permutation (shuffled) = [7, 2, 11, 0, 5, 9, 1, 4, 10, 3, 8, 6]
        #     start = 0  -> indices [7, 2, 11, 0]
        #     start = 4  -> indices [5, 9, 1, 4]
        #     start = 8  -> indices [10, 3, 8, 6]
        #
        # Every transition is used exactly once per epoch.
        # ``_validate_update_configuration`` requires minibatch_size to divide
        # the transition count exactly, so there is never a short last
        # minibatch and every weight change counts equally.
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
            # Gradient clipping: measure the overall size (L2 norm) of all
            # gradients together and, if it exceeds max_gradient_norm, scale
            # every gradient down so the size equals it. Directions are kept;
            # one unusual minibatch just can't move the weights too far.
            # Example: gradients [3, 4] have size 5; with a limit of 0.5 they
            # become [0.3, 0.4]. The trailing ``_`` means the gradients are
            # changed in place. Returns the size *before* clipping, for the
            # metrics. Raises if the size is infinite or NaN.
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
    # Recompute in exactly the groups collection used: one group per turn,
    # holding that turn's row from every game. The model's answer can shift
    # very slightly with the batch it runs in, because a different batch
    # changes the order in which it adds numbers, and floating-point sums
    # round differently in a different order. Regrouping into PPO's shuffled
    # minibatches could give a rounding gap big enough to look like a real
    # mismatch.
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

    # Bring one true/false answer to the CPU: did any value, in any turn, fall
    # outside its allowance? The loop above only kept a running worst case,
    # so this is asked once rather than once per turn. It must be answered
    # here, before the optimizer changes any weight.
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
    # Computers store numbers with limited precision, so tiny rounding
    # errors are normal. ``eps`` is the size of one such error near 1.0
    # (about 0.0000001 for float32). Allowing 8 of them gives the smallest
    # sensible ATOL. It matters when the stored value is 0 (a game where the
    # learner had no choice), because then the RTOL part allows nothing.
    dtype_floor = 8.0 * torch.finfo(old_log_probabilities.dtype).eps
    # float32 uses the larger of this floor and the default constant, which
    # leaves room for GPU rounding (see DEFAULT_FLOAT32_LOG_PROBABILITY_ATOL).
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
