"""Hand-calculated checks for generalized advantage estimation."""

from __future__ import annotations

import pytest
import torch

from orrery.gae import gae


@pytest.mark.parametrize(
    ("rewards", "next_values", "terminated", "truncated", "expected"),
    [
        (
            [0.0, 0.0, 0.0, 0.0, 1.0],
            # A deliberately large final value proves a true terminal ignores it.
            [0.2, 0.3, 0.4, 0.5, 10.0],
            [False, False, False, False, True],
            [False, False, False, False, False],
            [0.31453568, 0.325744, 0.3552, 0.41, 0.5],
        ),
        (
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.3, 0.6, 0.5, 0.6],
            [False, False, False, False, False],
            [False, False, True, False, False],
            [0.254816, 0.2428, 0.24, 0.0788, 0.04],
        ),
        (
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [0.1909159424, 0.15404992, 0.116736, 0.0788, 0.04],
        ),
    ],
    ids=["true terminal", "external truncation", "rollout boundary"],
)
def test_gae_matches_five_turn_boundary_calculations(
    rewards: list[float],
    next_values: list[float],
    terminated: list[bool],
    truncated: list[bool],
    expected: list[float],
) -> None:
    """Each way a stored fragment ends should use the correct bootstrap."""

    reward_tensor = torch.tensor(rewards, dtype=torch.float64)
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
    next_value_tensor = torch.tensor(next_values, dtype=torch.float64)

    advantages, returns = gae(
        reward_tensor,
        values,
        next_value_tensor,
        torch.tensor(terminated),
        torch.tensor(truncated),
        gamma=0.9,
        lam=0.8,
    )

    expected_advantages = torch.tensor(expected, dtype=torch.float64)
    torch.testing.assert_close(advantages, expected_advantages)
    torch.testing.assert_close(returns, expected_advantages + values)


def test_gae_bootstraps_every_truncation_in_one_environment_column() -> None:
    """Reset episodes in one rollout column need independent bootstrap values."""

    # Turns 0 and 2 end different externally truncated episodes. Turn 4 is the
    # final rollout boundary of a third live episode. The five distinct next
    # values model critic evaluations of the actual observation after each turn.
    rewards = torch.zeros((5, 1), dtype=torch.float64)
    values = torch.full((5, 1), 0.1, dtype=torch.float64)
    next_values = torch.tensor(
        [[0.4], [0.2], [0.5], [0.2], [0.3]], dtype=torch.float64
    )
    terminated = torch.zeros((5, 1), dtype=torch.bool)
    truncated = torch.tensor([[True], [False], [True], [False], [False]])

    advantages, _ = gae(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        gamma=0.9,
        lam=0.8,
    )

    # At the truncations, the deltas bootstrap from 0.4 and 0.5 but do not
    # inherit advantages from the reset episodes that follow them.
    expected = torch.tensor(
        [[0.26], [0.332], [0.35], [0.2024], [0.17]], dtype=torch.float64
    )
    torch.testing.assert_close(advantages, expected)


def test_gae_detaches_advantages_and_return_targets() -> None:
    """Stored critic estimates must not become movable training targets."""

    rewards = torch.zeros((2, 1))
    values = torch.tensor([[0.2], [0.4]], requires_grad=True)
    next_values = torch.tensor([[0.4], [0.0]], requires_grad=True)
    terminated = torch.tensor([[False], [True]])
    truncated = torch.zeros_like(terminated)

    advantages, returns = gae(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
    )

    assert not advantages.requires_grad
    assert not returns.requires_grad
