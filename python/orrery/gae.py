"""Generalized advantage estimation for stored Orrery rollouts."""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float = 0.999,
    lam: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Return advantages and critic targets for a time-major rollout.

    Every tensor has the same ``[time, ...batch dimensions]`` shape.
    ``next_values[t]`` is the critic value of the observation following turn
    ``t``. Supplying it for every turn preserves the distinct bootstrap value
    at each truncation in an environment column, as well as at the final
    rollout boundary.

    A true terminal has no future, so its supplied next value is ignored. An
    external truncation still bootstraps from its next value, but both kinds of
    episode boundary stop advantages from flowing backward across a reset.
    The end of the stored rollout similarly stops the recurrence without
    discarding its final next-state value.

    Advantages and return targets describe the fixed rollout rather than the
    policy currently being optimized. This function therefore always returns
    tensors detached from any critic graph carried by its inputs.
    """

    _validate_inputs(rewards, values, next_values, terminated, truncated)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    advantages = torch.zeros_like(values)
    # Zero represents the absent following advantage beyond the final stored
    # turn. It stops recurrence at the rollout boundary without a special case
    # inside the loop.
    following_advantage = torch.zeros_like(values[0])

    # Start at the last stored turn and move toward the first. We go backward
    # because the advantage for an earlier turn uses information learned from
    # the turns that came after it.
    for turn in range(rewards.shape[0] - 1, -1, -1):
        # If the game truly ended on this turn, there is no future value, so
        # can_bootstrap is False and the next value becomes zero in the formula
        # below. If recording merely stopped because of a truncation, the game
        # still has a future, so we keep the critic's next-state estimate.
        can_bootstrap = ~terminated[turn]

        # Compare what actually happened with what the critic expected:
        #
        #     reward + estimated value after the action - old estimate
        #
        # gamma slightly discounts the next-state estimate. A positive delta
        # means the turn looks better than expected; a negative delta means it
        # looks worse.
        delta = (
            rewards[turn]
            + gamma * next_values[turn] * can_bootstrap
            - values[turn]
        )

        # A terminal or truncation may be followed in the buffer by the first
        # turn of a newly reset game. That new game's advantage must not leak
        # backward into the old game, so same_fragment becomes False at either
        # boundary. At the end of the rollout, following_advantage is already
        # zero because that is how we initialized it above.
        same_fragment = ~(terminated[turn] | truncated[turn])

        # Begin with this turn's delta, then bring back some information from
        # later turns in the same game. lambda controls how much of that later
        # information survives as we move farther backward.
        following_advantage = (
            delta + gamma * lam * same_fragment * following_advantage
        )
        # Store this turn's answer. On the next loop iteration, the same value
        # becomes the "following advantage" for the preceding turn.
        advantages[turn] = following_advantage

    # The advantage says how far the old value estimate was from its target, so
    # adding them gives the value the critic should learn toward:
    #
    #     return target = old value estimate + advantage
    #
    # gae() runs with no_grad, which keeps these answers fixed during training.
    return advantages, advantages + values


def _validate_inputs(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
) -> None:
    """Reject rollout tensors that cannot describe aligned transitions."""

    tensors = {
        "rewards": rewards,
        "values": values,
        "next_values": next_values,
        "terminated": terminated,
        "truncated": truncated,
    }
    expected_shape = tuple(rewards.shape)
    if rewards.ndim == 0 or rewards.shape[0] == 0:
        raise ValueError("rollout tensors must contain at least one time step")

    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}, want {expected_shape}"
            )

    for name, tensor in (
        ("rewards", rewards),
        ("values", values),
        ("next_values", next_values),
    ):
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype, got {tensor.dtype}")
        if tensor.dtype != rewards.dtype:
            raise TypeError(
                f"{name} must use {rewards.dtype} to match rewards, got {tensor.dtype}"
            )

    for name, tensor in (("terminated", terminated), ("truncated", truncated)):
        if tensor.dtype != torch.bool:
            raise TypeError(f"{name} must use torch.bool, got {tensor.dtype}")

    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        raise ValueError(f"rollout tensors must share one device, got {devices}")
