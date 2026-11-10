"""Hand-calculated checks for Orrery's PPO probability helpers."""

from __future__ import annotations

import math

import pytest
import torch

from orrery.env_client import BODY_SLOTS
from orrery.model import PolicyOutput
from orrery.ppo import (
    EntropyBonus,
    clipped_policy_loss,
    entropy_bonus,
    value_loss,
)


def test_entropy_bonus_matches_two_owned_body_calculation() -> None:
    """A forced source stays in the mean but contributes exact zero."""

    launch_logits = torch.full((1, BODY_SLOTS), -torch.inf)
    launch_logits[0, 0] = 0.0  # P(launch) = 0.5 and H(launch) = log(2).
    launch_logits.requires_grad_()

    target_logits = torch.full((1, BODY_SLOTS, BODY_SLOTS), -torch.inf)
    target_logits[0, 0, 1:3] = 0.0  # Two equally likely targets: H = log(2).
    target_logits.requires_grad_()
    output = PolicyOutput(launch_logits, target_logits, torch.zeros(1))

    owned_mask = torch.zeros((1, BODY_SLOTS), dtype=torch.bool)
    owned_mask[0, :2] = True
    target_mask = torch.zeros((1, BODY_SLOTS, BODY_SLOTS), dtype=torch.bool)
    target_mask[0, 0, 1:3] = True
    launchable_mask = owned_mask & target_mask.any(dim=-1)

    bonus = entropy_bonus(output, owned_mask, launchable_mask, target_mask)

    # Source 0 contributes log(2) launch entropy and 0.5*log(2) target
    # entropy. Source 1 is forced no-launch, so both values are divided by the
    # two owned bodies without adding another contribution.
    expected_launch = torch.tensor(math.log(2.0) / 2)
    expected_target = torch.tensor(math.log(2.0) / 4)
    torch.testing.assert_close(bonus.launch, expected_launch)
    torch.testing.assert_close(bonus.target, expected_target)
    torch.testing.assert_close(bonus.total, expected_launch + expected_target)
    assert torch.isfinite(bonus.total)
    assert bonus.total.requires_grad

    # The entropy calculation must remain attached to current policy logits.
    # Masked entries are constants and therefore receive exact-zero gradients.
    bonus.total.backward()
    assert torch.isfinite(launch_logits.grad[launchable_mask]).all()
    assert torch.count_nonzero(launch_logits.grad[~launchable_mask]) == 0
    assert torch.isfinite(target_logits.grad[target_mask]).all()
    assert torch.count_nonzero(target_logits.grad[~target_mask]) == 0


def test_entropy_bonus_is_invariant_to_equally_uncertain_empire_size() -> None:
    """Repeating one decision must not increase a mean entropy regularizer."""

    small = _uniform_entropy_case(owned_bodies=2)
    large = _uniform_entropy_case(owned_bodies=4)

    torch.testing.assert_close(large.launch, small.launch)
    torch.testing.assert_close(large.target, small.target)
    torch.testing.assert_close(large.total, small.total)


def test_entropy_bonus_gives_zero_owned_game_zero_batch_weight() -> None:
    """An eliminated game must contribute zero instead of an empty-mean NaN."""

    launch_logits = torch.full((2, BODY_SLOTS), -torch.inf)
    launch_logits[1, 0] = 0.0
    target_logits = torch.full((2, BODY_SLOTS, BODY_SLOTS), -torch.inf)
    target_logits[1, 0, 1:3] = 0.0
    output = PolicyOutput(launch_logits, target_logits, torch.zeros(2))

    owned_mask = torch.zeros((2, BODY_SLOTS), dtype=torch.bool)
    owned_mask[1, 0] = True
    target_mask = torch.zeros((2, BODY_SLOTS, BODY_SLOTS), dtype=torch.bool)
    target_mask[1, 0, 1:3] = True
    launchable_mask = owned_mask & target_mask.any(dim=-1)

    bonus = entropy_bonus(output, owned_mask, launchable_mask, target_mask)

    # Game 0 owns nothing and contributes 0. Game 1 contributes log(2) launch
    # entropy plus 0.5*log(2) target entropy; the batch mean divides each by 2.
    torch.testing.assert_close(bonus.launch, torch.tensor(math.log(2.0) / 2))
    torch.testing.assert_close(bonus.target, torch.tensor(math.log(2.0) / 4))
    assert torch.isfinite(bonus.total)


def test_clipped_policy_loss_matches_positive_and_negative_example() -> None:
    """Both advantage signs must use PPO's conservative clipped score."""

    old_log_prob = torch.zeros(2)
    new_log_prob = torch.log(torch.tensor([1.5, 0.5]))
    advantages = torch.tensor([2.0, -2.0])

    loss = clipped_policy_loss(
        new_log_prob,
        old_log_prob,
        advantages,
        clip_epsilon=0.2,
    )

    # Ratios clamp from [1.5, 0.5] to [1.2, 0.8]. The conservative scores are
    # min(3.0, 2.4) = 2.4 and min(-1.0, -1.6) = -1.6. Their mean objective is
    # 0.4, and the loss carries the leading negative sign used by optimizers.
    torch.testing.assert_close(loss, torch.tensor(-0.4))


def test_clipped_policy_loss_keeps_unfavorable_ratio_changes() -> None:
    """Clipping limits rewarded improvement, not every out-of-range ratio."""

    old_log_prob = torch.zeros(2)
    new_log_prob = torch.log(torch.tensor([0.5, 1.5]))
    advantages = torch.tensor([2.0, -2.0])

    loss = clipped_policy_loss(
        new_log_prob,
        old_log_prob,
        advantages,
        clip_epsilon=0.2,
    )

    # These changes move in the wrong direction for their advantage signs, so
    # the ordinary scores 1.0 and -3.0 are more conservative than the clipped
    # scores 1.6 and -2.4. The objective is therefore -1.0 and the loss is 1.0.
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_value_loss_matches_mean_squared_error_and_retains_gradient() -> None:
    """The critic should regress directly toward fixed GAE return targets."""

    predicted_values = torch.tensor([0.2, -0.5], requires_grad=True)
    return_targets = torch.tensor([1.0, -1.0])

    loss = value_loss(predicted_values, return_targets)

    # Errors [-0.8, 0.5] square to [0.64, 0.25], whose mean is 0.445.
    torch.testing.assert_close(loss, torch.tensor(0.445))
    loss.backward()
    # For a two-entry mean squared error, 2*error/2 equals the raw error.
    torch.testing.assert_close(predicted_values.grad, torch.tensor([-0.8, 0.5]))


def test_losses_reject_gradient_carrying_fixed_rollout_data() -> None:
    """Only current policy predictions may receive gradients during an update."""

    fixed = torch.zeros(2)
    carrying_gradient = torch.zeros(2, requires_grad=True)

    with pytest.raises(ValueError, match="old_log_prob must be detached"):
        clipped_policy_loss(fixed, carrying_gradient, fixed)
    with pytest.raises(ValueError, match="advantages must be detached"):
        clipped_policy_loss(fixed, fixed, carrying_gradient)
    with pytest.raises(ValueError, match="return_targets must be detached"):
        value_loss(fixed, carrying_gradient)


def _uniform_entropy_case(owned_bodies: int) -> EntropyBonus:
    """Return entropy when every owned source has the same two choices."""

    launch_logits = torch.full((1, BODY_SLOTS), -torch.inf)
    launch_logits[0, :owned_bodies] = 0.0
    target_logits = torch.full((1, BODY_SLOTS, BODY_SLOTS), -torch.inf)
    owned_mask = torch.zeros((1, BODY_SLOTS), dtype=torch.bool)
    owned_mask[0, :owned_bodies] = True
    target_mask = torch.zeros((1, BODY_SLOTS, BODY_SLOTS), dtype=torch.bool)

    for source in range(owned_bodies):
        first_target = (source + 1) % BODY_SLOTS
        second_target = (source + 2) % BODY_SLOTS
        target_mask[0, source, first_target] = True
        target_mask[0, source, second_target] = True
        target_logits[0, source, first_target] = 0.0
        target_logits[0, source, second_target] = 0.0

    output = PolicyOutput(launch_logits, target_logits, torch.zeros(1))
    launchable_mask = owned_mask & target_mask.any(dim=-1)
    return entropy_bonus(output, owned_mask, launchable_mask, target_mask)
