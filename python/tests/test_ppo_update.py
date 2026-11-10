"""Checks for the reusable optimizer-facing PPO update."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch
from torch import nn

import orrery.ppo as ppo_module
from conftest import OPPONENT
from orrery.env_client import (
    BODY_FEATURES,
    BODY_SLOTS,
    EnvClient,
    PLAYER_FEATURES,
    PLAYER_SLOTS,
)
from orrery.model import (
    ObservationBatch,
    OrbitPolicy,
    PolicyOutput,
    SampledActionBatch,
)
from orrery.ppo import joint_log_prob, ppo_update
from orrery.rollout import RolloutBatch, collect_rollout


class _TwoChoicePolicy(nn.Module):
    """Learn one launch probability and one critic value for a fixed state."""

    def __init__(self) -> None:
        super().__init__()
        self.launch_logit = nn.Parameter(torch.tensor(0.0))
        self.critic_value = nn.Parameter(torch.tensor(0.0))

    def forward(self, observation: ObservationBatch) -> PolicyOutput:
        """Give source zero one optional launch with one forced target."""

        launch_logits = self.launch_logit.expand(
            observation.batch_size,
            BODY_SLOTS,
        ).masked_fill(~observation.launchable_mask, -torch.inf)
        target_logits = torch.zeros(
            (
                observation.batch_size,
                BODY_SLOTS,
                BODY_SLOTS,
            ),
            dtype=observation.bodies.dtype,
            device=observation.bodies.device,
        ).masked_fill(~observation.target_mask, -torch.inf)
        return PolicyOutput(
            launch_logits=launch_logits,
            target_logits=target_logits,
            values=(
                observation.bodies[:, 0, 0] * 0.0
                + self.critic_value
            ),
        )


def test_ppo_update_changes_parameters_and_clips_gradients() -> None:
    """One update should exercise losses, backward, clipping, and optimizer step."""

    policy = _TwoChoicePolicy().eval()
    advantages = torch.tensor([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0])
    launches = torch.tensor([False, False, False, False, True, True, True, True])
    rollout = _rollout(policy, advantages=advantages, launches=launches)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.05)
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }
    fixed_advantages = rollout.advantages.clone()
    fixed_returns = rollout.return_targets.clone()
    fixed_log_probabilities = rollout.old_log_probabilities.clone()

    metrics = ppo_update(
        policy,
        optimizer,
        rollout,
        epochs=2,
        minibatch_size=4,
        value_coefficient=0.5,
        launch_entropy_coefficient=0.02,
        target_entropy_coefficient=0.01,
        max_gradient_norm=0.05,
        shuffle_generator=torch.Generator().manual_seed(1403),
    )

    assert metrics.optimization_steps == 4
    assert metrics.initial_log_probability_max_error == 0.0
    assert metrics.advantage_mean == 0.0
    assert metrics.advantage_std == pytest.approx(
        float(advantages.std(unbiased=False))
    )
    assert 0.0 <= metrics.clip_fraction <= 1.0
    assert metrics.approximate_kl >= 0.0
    assert math.isfinite(metrics.total_loss)
    assert metrics.max_gradient_norm >= metrics.mean_gradient_norm >= 0.0
    assert metrics.total_loss == pytest.approx(
        metrics.policy_loss
        + 0.5 * metrics.value_loss
        - 0.02 * metrics.launch_entropy
        - 0.01 * metrics.target_entropy,
        abs=1e-7,
    )

    # Positive advantages belong to launches and negative advantages to idle
    # choices, so the actor should increase its launch score. Every return
    # target is one, so the critic should also move upward from zero.
    assert float(policy.launch_logit) > float(initial_parameters["launch_logit"])
    assert float(policy.critic_value) > float(initial_parameters["critic_value"])
    assert policy.training is False

    # clip_grad_norm_ leaves the final clipped gradients on the parameters.
    # Metrics retain the pre-clipping norm, while this calculation confirms
    # what the optimizer actually received was bounded.
    final_gradient_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                parameter.grad.detach().norm()
                for parameter in policy.parameters()
                if parameter.grad is not None
            ]
        )
    )
    assert float(final_gradient_norm) <= 0.050001

    # PPO may change current policy parameters, but rollout evidence and fixed
    # targets must remain byte-for-byte unchanged through every epoch.
    assert torch.equal(rollout.advantages, fixed_advantages)
    assert torch.equal(rollout.return_targets, fixed_returns)
    assert torch.equal(rollout.old_log_probabilities, fixed_log_probabilities)


def test_ppo_update_normalizes_advantages_once_across_full_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Minibatches must receive slices of one global normalized tensor."""

    policy = _TwoChoicePolicy().eval()
    advantages = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    rollout = _rollout(
        policy,
        advantages=advantages,
        launches=torch.tensor([False, False, True, True]),
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)
    received_advantages: list[torch.Tensor] = []
    real_clipped_policy_loss = ppo_module.clipped_policy_loss

    def record_advantages(
        new_log_probability: torch.Tensor,
        old_log_probability: torch.Tensor,
        minibatch_advantages: torch.Tensor,
        *,
        clip_epsilon: float,
    ) -> torch.Tensor:
        received_advantages.append(minibatch_advantages.detach().clone())
        return real_clipped_policy_loss(
            new_log_probability,
            old_log_probability,
            minibatch_advantages,
            clip_epsilon=clip_epsilon,
        )

    monkeypatch.setattr(ppo_module, "clipped_policy_loss", record_advantages)

    ppo_update(
        policy,
        optimizer,
        rollout,
        epochs=1,
        minibatch_size=2,
        value_coefficient=0.0,
        launch_entropy_coefficient=0.0,
        target_entropy_coefficient=0.0,
        shuffle_generator=torch.Generator().manual_seed(1404),
    )

    expected = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    received = torch.cat(received_advantages)
    torch.testing.assert_close(
        torch.sort(received).values,
        torch.sort(expected).values,
    )
    # The raw rollout tensor is the critic/diagnostic evidence and is never
    # replaced by its actor-only normalized copy.
    assert torch.equal(rollout.advantages, advantages)


def test_ppo_update_rejects_probability_mismatch_before_mutation() -> None:
    """A changed old policy must fail before parameters or optimizer state move."""

    policy = _TwoChoicePolicy().eval()
    rollout = _rollout(
        policy,
        advantages=torch.tensor([-1.0, 1.0]),
        launches=torch.tensor([False, True]),
    )
    corrupted = replace(
        rollout,
        old_log_probabilities=rollout.old_log_probabilities + 0.01,
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }

    with pytest.raises(
        ValueError,
        match="stored and recomputed log-probabilities differ before the first",
    ):
        ppo_update(
            policy,
            optimizer,
            corrupted,
            epochs=1,
            minibatch_size=2,
        )

    for name, parameter in policy.named_parameters():
        assert torch.equal(parameter, before[name])
    assert optimizer.state == {}


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"epochs": 0}, "epochs must be a positive integer"),
        (
            {"minibatch_size": 3},
            "minibatch_size must divide the rollout exactly",
        ),
        ({"clip_epsilon": 1.0}, r"clip_epsilon must be in \[0, 1\)"),
        (
            {"value_coefficient": -0.1},
            "value_coefficient must be finite and non-negative",
        ),
        (
            {"launch_entropy_coefficient": math.inf},
            "launch_entropy_coefficient must be finite and non-negative",
        ),
        (
            {"target_entropy_coefficient": -0.1},
            "target_entropy_coefficient must be finite and non-negative",
        ),
        (
            {"max_gradient_norm": 0.0},
            "max_gradient_norm must be finite and positive",
        ),
        (
            {"advantage_epsilon": 0.0},
            "advantage_epsilon must be finite and positive",
        ),
        (
            {"log_probability_atol": -1.0},
            "log_probability_atol must be finite and non-negative",
        ),
        (
            {"log_probability_rtol": -1.0},
            "log_probability_rtol must be finite and non-negative",
        ),
    ),
)
def test_ppo_update_rejects_invalid_configuration_before_mutation(
    changes: dict[str, float | int],
    error: str,
) -> None:
    """Configuration errors must not partially update model parameters."""

    policy = _TwoChoicePolicy().eval()
    rollout = _rollout(
        policy,
        advantages=torch.tensor([-3.0, -1.0, 1.0, 3.0]),
        launches=torch.tensor([False, False, True, True]),
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    arguments: dict[str, float | int | None] = {
        "epochs": 1,
        "minibatch_size": 2,
    }
    arguments.update(changes)
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    with pytest.raises(ValueError, match=error):
        ppo_update(policy, optimizer, rollout, **arguments)  # type: ignore[arg-type]

    for parameter, expected in zip(policy.parameters(), before, strict=True):
        assert torch.equal(parameter, expected)
    assert optimizer.state == {}


def test_ppo_update_requires_eval_mode_and_matching_optimizer() -> None:
    """Policy-mode and optimizer wiring errors must be reported directly."""

    policy = _TwoChoicePolicy()
    rollout = _rollout(
        policy.eval(),
        advantages=torch.tensor([-1.0, 1.0]),
        launches=torch.tensor([False, True]),
    )
    policy.train()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="policy must be in evaluation mode"):
        ppo_update(policy, optimizer, rollout, epochs=1, minibatch_size=2)

    policy.eval()
    missing_critic = torch.optim.SGD([policy.launch_logit], lr=0.1)
    with pytest.raises(ValueError, match="optimizer is missing 1 trainable"):
        ppo_update(policy, missing_critic, rollout, epochs=1, minibatch_size=2)

    foreign_parameter = nn.Parameter(torch.tensor(0.0))
    foreign_optimizer = torch.optim.SGD(
        [*policy.parameters(), foreign_parameter],
        lr=0.1,
    )
    with pytest.raises(ValueError, match="optimizer contains 1 parameters outside"):
        ppo_update(policy, foreign_optimizer, rollout, epochs=1, minibatch_size=2)


def test_ppo_update_accepts_explicit_log_probability_tolerances() -> None:
    """A caller may deliberately allow a known, bounded round-trip difference."""

    policy = _TwoChoicePolicy().eval()
    rollout = _rollout(
        policy,
        advantages=torch.tensor([-1.0, 1.0]),
        launches=torch.tensor([False, True]),
    )
    permitted_difference = 5e-5
    adjusted = replace(
        rollout,
        old_log_probabilities=(
            rollout.old_log_probabilities + permitted_difference
        ),
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)

    metrics = ppo_update(
        policy,
        optimizer,
        adjusted,
        epochs=1,
        minibatch_size=2,
        log_probability_atol=1e-4,
    )

    assert metrics.initial_log_probability_max_error == pytest.approx(
        permitted_difference,
        rel=0.01,
    )


def test_default_log_probability_tolerance_scales_with_large_magnitude() -> None:
    """Relative tolerance should cover device rounding on a large log-probability."""

    policy = _TwoChoicePolicy().eval()
    with torch.no_grad():
        policy.launch_logit.fill_(20.0)
    rollout = _rollout(
        policy,
        advantages=torch.tensor([-1.0, 1.0]),
        launches=torch.tensor([False, False]),
    )
    rounding_sized_difference = torch.full_like(
        rollout.old_log_probabilities,
        1e-4,
    )
    adjusted = replace(
        rollout,
        old_log_probabilities=(
            rollout.old_log_probabilities + rounding_sized_difference
        ),
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)

    metrics = ppo_update(
        policy,
        optimizer,
        adjusted,
        epochs=1,
        minibatch_size=2,
    )

    assert metrics.initial_log_probability_max_error == pytest.approx(
        1e-4,
        rel=0.02,
    )


def test_default_log_probability_absolute_floor_respects_dtype() -> None:
    """Only V1 float32 replay should receive its configured floor."""

    float32_atol = ppo_module._log_probability_atol(
        torch.zeros(1, dtype=torch.float32),
        None,
    )
    float64_atol = ppo_module._log_probability_atol(
        torch.zeros(1, dtype=torch.float64),
        None,
    )

    assert float32_atol == ppo_module.DEFAULT_FLOAT32_LOG_PROBABILITY_ATOL
    assert float64_atol == 8.0 * torch.finfo(torch.float64).eps


def test_ppo_update_rejects_runtime_contract_errors_before_mutation() -> None:
    """Policy output and trainable-parameter mistakes should fail at the boundary."""

    source_policy = _TwoChoicePolicy().eval()
    rollout = _rollout(
        source_policy,
        advantages=torch.tensor([-1.0, 1.0]),
        launches=torch.tensor([False, True]),
    )

    class WrongOutputPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, observation: ObservationBatch) -> torch.Tensor:
            return observation.bodies[:, 0, 0] * self.weight

    wrong_output = WrongOutputPolicy().eval()
    wrong_optimizer = torch.optim.SGD(wrong_output.parameters(), lr=0.1)
    with pytest.raises(TypeError, match="policy returned Tensor, want PolicyOutput"):
        ppo_update(
            wrong_output,
            wrong_optimizer,
            rollout,
            epochs=1,
            minibatch_size=2,
        )
    assert wrong_optimizer.state == {}

    frozen = _TwoChoicePolicy().eval()
    frozen_optimizer = torch.optim.SGD(frozen.parameters(), lr=0.1)
    frozen.requires_grad_(False)
    with pytest.raises(ValueError, match="at least one trainable parameter"):
        ppo_update(
            frozen,
            frozen_optimizer,
            rollout,
            epochs=1,
            minibatch_size=2,
        )
    assert frozen_optimizer.state == {}


def test_ppo_update_rejects_policy_rollout_device_mismatch() -> None:
    """Device disagreement should be named before attempting a policy forward pass."""

    source_policy = _TwoChoicePolicy().eval()
    rollout = _rollout(
        source_policy,
        advantages=torch.tensor([-1.0, 1.0]),
        launches=torch.tensor([False, True]),
    )
    policy = _TwoChoicePolicy().to("meta").eval()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="policy uses meta, but rollout uses cpu"):
        ppo_update(policy, optimizer, rollout, epochs=1, minibatch_size=2)
    assert optimizer.state == {}


def test_ppo_update_rejects_nonfinite_first_minibatch_without_mutation() -> None:
    """A first-minibatch overflow must fail before any optimizer mutation."""

    policy = _TwoChoicePolicy().eval()
    rollout = _rollout(
        policy,
        advantages=torch.tensor([-1.0, 1.0]),
        launches=torch.tensor([False, True]),
    )
    large_targets = replace(
        rollout,
        return_targets=torch.full_like(rollout.return_targets, 2.0),
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    # This deliberately fails on the first and only minibatch. Later runtime
    # failures are fail-fast but are not promised to roll back optimizer steps
    # that completed earlier in the update.
    with pytest.raises(ValueError, match="PPO losses produced a non-finite total"):
        ppo_update(
            policy,
            optimizer,
            large_targets,
            epochs=1,
            minibatch_size=2,
            value_coefficient=float(torch.finfo(torch.float32).max),
        )

    for parameter, expected in zip(policy.parameters(), before, strict=True):
        assert torch.equal(parameter, expected)
    assert optimizer.state == {}


@pytest.mark.parametrize("players", [2, 4])
def test_real_policy_update_accepts_deep_live_go_rollout(
    env_command: tuple[str, ...],
    players: int,
) -> None:
    """Verification must replay collection batches on a deeper real rollout.

    This is the full training path on real obs-v2 maps of both sizes: Go
    observations, sampling, the backward pass, and an optimizer step. Every
    sampled action must be legal, the losses finite, and the weights changed.
    """

    torch.manual_seed(1405)
    policy = OrbitPolicy().eval()
    # Larger weights sharpen the policy's action probabilities, reproducing
    # the log-probability magnitudes reached after training has begun rather
    # than testing only a freshly initialized network.
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.mul_(3.0)

    with EnvClient(env_command, players=players) as environment:
        environment.create(4, (0, 1_000_000), OPPONENT)
        initial = environment.reset([0, 1, 2, 3])
        rollout = collect_rollout(
            environment,
            policy,
            initial,
            steps=40,
            generator=torch.Generator().manual_seed(1406),
        ).batch

        # A single B=160 forward can differ from the stored B=4-per-turn
        # calculations, which is what the update's verification must tolerate.
        # Whether it does differ depends on the platform's math libraries: on
        # macOS it does, while on Linux CI the two can be bit-identical. So
        # only the ratio bound below is asserted, not that a difference exists.
        with torch.no_grad():
            flat_output = policy(rollout.observations)
            flat_log_probabilities = joint_log_prob(
                flat_output,
                rollout.actions,
                rollout.observations.launchable_mask,
                rollout.observations.target_mask,
            )
        flat_probability_ratios = torch.exp(
            flat_log_probabilities - rollout.old_log_probabilities
        )
        # The training-shaped recomputation is not bit-identical, but its
        # initial ratios remain effectively one and several orders of
        # magnitude inside PPO's default 0.2 clipping range.
        torch.testing.assert_close(
            flat_probability_ratios,
            torch.ones_like(flat_probability_ratios),
            rtol=0.0,
            atol=1e-3,
        )

        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
        before = [parameter.detach().clone() for parameter in policy.parameters()]

        metrics = ppo_update(
            policy,
            optimizer,
            rollout,
            epochs=1,
            # This update shape previously made the gate recompute all 160
            # rows at once even though collection evaluated B=4 each turn.
            minibatch_size=rollout.transition_count,
            shuffle_generator=torch.Generator().manual_seed(1407),
        )

    assert rollout.transition_count == 160
    assert metrics.optimization_steps == 1
    # Replaying collection's own B=4 batches must reproduce the stored
    # log-probabilities to within the strict same-calculation floor.
    same_calculation_atol = 8.0 * torch.finfo(torch.float32).eps
    assert (
        metrics.initial_log_probability_max_error
        <= same_calculation_atol
    )
    assert math.isfinite(metrics.total_loss)

    # Every sampled launch came from a launchable source, and every launched
    # source chose a target its row of the target mask allows. The sharpened
    # weights make launching rare, so real sources must still be launchable
    # even when none happened to launch.
    launches = rollout.actions.launches
    assert bool(rollout.observations.launchable_mask.any())
    assert not bool((launches & ~rollout.observations.launchable_mask).any())
    rows, sources = launches.nonzero(as_tuple=True)
    chosen = rollout.actions.targets[rows, sources]
    assert bool(rollout.observations.target_mask[rows, sources, chosen].all())
    # The seats Go assigned fit the map, so four-player maps reach seats 2-3
    # whenever the schedule draws them.
    assert int(rollout.seats.max()) < players
    assert any(
        not torch.equal(parameter, previous)
        for parameter, previous in zip(policy.parameters(), before, strict=True)
    )


def _rollout(
    policy: _TwoChoicePolicy,
    *,
    advantages: torch.Tensor,
    launches: torch.Tensor,
) -> RolloutBatch:
    """Build a valid fixed rollout whose learning signal is hand controlled."""

    transition_count = advantages.numel()
    if tuple(launches.shape) != (transition_count,):
        raise ValueError("launch fixture must have one choice per advantage")
    observations = _observations(transition_count)
    action_launches = torch.zeros(
        (transition_count, BODY_SLOTS),
        dtype=torch.bool,
    )
    action_launches[:, 0] = launches
    action_targets = torch.zeros(
        (transition_count, BODY_SLOTS),
        dtype=torch.long,
    )
    action_targets[:, 0] = 1
    actions = SampledActionBatch(action_launches, action_targets)
    with torch.no_grad():
        output = policy(observations)
        old_log_probabilities = joint_log_prob(
            output,
            actions,
            observations.launchable_mask,
            observations.target_mask,
        )

    return RolloutBatch(
        observations=observations,
        next_observations=observations,
        actions=actions,
        old_log_probabilities=old_log_probabilities,
        values=output.values,
        next_values=torch.zeros(transition_count),
        rewards=torch.where(launches, 1.0, -1.0),
        terminated=torch.ones(transition_count, dtype=torch.bool),
        truncated=torch.zeros(transition_count, dtype=torch.bool),
        advantages=advantages,
        return_targets=torch.ones(transition_count),
        seeds=torch.arange(transition_count),
        seats=torch.zeros(transition_count, dtype=torch.long),
        explicit=torch.ones(transition_count, dtype=torch.bool),
        turns=torch.ones(transition_count, dtype=torch.long),
        environment_ids=tuple(range(transition_count)),
        time_steps=1,
        replay_metadata={
            "engine": "ppo-update-test",
            "schema": "obs-v1",
            "generator": "fixed-test-v1",
            "opponent": "fixed-test-v1",
        },
    )


def _observations(batch_size: int) -> ObservationBatch:
    """Return identical states with one optional launch and forced target."""

    body_mask = torch.zeros((batch_size, BODY_SLOTS), dtype=torch.bool)
    body_mask[:, :2] = True
    owned_mask = torch.zeros_like(body_mask)
    owned_mask[:, 0] = True
    player_mask = torch.zeros((batch_size, PLAYER_SLOTS), dtype=torch.bool)
    player_mask[:, :2] = True
    target_mask = torch.zeros(
        (batch_size, BODY_SLOTS, BODY_SLOTS),
        dtype=torch.bool,
    )
    target_mask[:, 0, 1] = True
    return ObservationBatch(
        bodies=torch.zeros((batch_size, BODY_SLOTS, BODY_FEATURES)),
        players=torch.zeros((batch_size, PLAYER_SLOTS, PLAYER_FEATURES)),
        body_mask=body_mask,
        owned_mask=owned_mask,
        player_mask=player_mask,
        target_mask=target_mask,
    )
