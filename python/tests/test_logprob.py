"""Checks that rollout sampling and PPO agree on action probabilities."""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import Bernoulli, Categorical

from orrery.env_client import (
    BODY_FEATURES,
    BODY_SLOTS,
    PLAYER_FEATURES,
    PLAYER_SLOTS,
)
from orrery.model import (
    ObservationBatch,
    PolicyOutput,
    SampledActionBatch,
    sample_actions,
    sample_actions_for_collection,
)
from orrery.ppo import collection_joint_log_prob, joint_log_prob


def test_sampled_action_log_probability_round_trips_exactly() -> None:
    """The stored and recomputed values must use one probability contract."""

    observation = _observation_batch()
    output = _policy_output(observation)
    actions = sample_actions(
        output,
        observation,
        generator=torch.Generator().manual_seed(71),
    )

    stored = _sampling_log_prob_oracle(output, actions, observation)
    recomputed = joint_log_prob(
        output,
        actions,
        observation.launchable_mask,
        observation.target_mask,
    )

    # The loop-based sampling oracle and vectorized PPO helper are independent
    # paths. Exact equality proves they select and reduce the same decisions.
    assert torch.equal(recomputed, stored)
    assert torch.isfinite(recomputed).all()


def test_joint_log_probability_matches_hand_calculation() -> None:
    """Only launchable sources and a launched source's target should count."""

    launch_logits = torch.full((1, BODY_SLOTS), -torch.inf, dtype=torch.float64)
    # Source 0 launches with probability 0.8. Source 1 stays idle when its
    # launch probability is 0.3. All other sources are forced no-launch.
    launch_logits[0, 0] = math.log(0.8 / 0.2)
    launch_logits[0, 1] = math.log(0.3 / 0.7)

    target_logits = torch.full(
        (1, BODY_SLOTS, BODY_SLOTS),
        -torch.inf,
        dtype=torch.float64,
    )
    target_logits[0, 0, 1] = math.log(0.25)
    target_logits[0, 0, 2] = math.log(0.75)
    # Source 1 needs a legal target to make its launch decision optional, even
    # though its stored no-launch action means this target row is never scored.
    target_logits[0, 1, 0] = 0.0
    output = PolicyOutput(
        launch_logits=launch_logits,
        target_logits=target_logits,
        values=torch.zeros(1, dtype=torch.float64),
    )

    launches = torch.zeros((1, BODY_SLOTS), dtype=torch.bool)
    launches[0, 0] = True
    targets = torch.zeros((1, BODY_SLOTS), dtype=torch.long)
    targets[0, 0] = 2
    actions = SampledActionBatch(launches=launches, targets=targets)

    launchable_mask = torch.zeros((1, BODY_SLOTS), dtype=torch.bool)
    launchable_mask[0, :2] = True
    target_mask = torch.zeros((1, BODY_SLOTS, BODY_SLOTS), dtype=torch.bool)
    target_mask[0, 0, 1:3] = True
    target_mask[0, 1, 0] = True

    actual = joint_log_prob(output, actions, launchable_mask, target_mask)
    expected = torch.tensor(
        [math.log(0.8) + math.log(0.75) + math.log(0.7)],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected)


def test_joint_log_probability_retains_graph_when_every_action_is_forced() -> None:
    """An exact-zero policy loss must still permit the update's backward pass."""

    launch_logits = torch.full(
        (1, BODY_SLOTS),
        -torch.inf,
        requires_grad=True,
    )
    output = PolicyOutput(
        launch_logits=launch_logits,
        target_logits=torch.full((1, BODY_SLOTS, BODY_SLOTS), -torch.inf),
        values=torch.zeros(1),
    )
    actions = SampledActionBatch(
        launches=torch.zeros((1, BODY_SLOTS), dtype=torch.bool),
        targets=torch.zeros((1, BODY_SLOTS), dtype=torch.long),
    )
    launchable_mask = torch.zeros((1, BODY_SLOTS), dtype=torch.bool)
    target_mask = torch.zeros((1, BODY_SLOTS, BODY_SLOTS), dtype=torch.bool)

    log_probability = joint_log_prob(
        output,
        actions,
        launchable_mask,
        target_mask,
    )

    assert torch.equal(log_probability, torch.zeros(1))
    assert log_probability.requires_grad
    log_probability.sum().backward()
    assert launch_logits.grad is not None
    assert torch.count_nonzero(launch_logits.grad) == 0


def test_joint_log_probability_rejects_an_out_of_range_stored_target() -> None:
    """Validation must stop an unsafe target before Torch indexes with it."""

    observation = _observation_batch()
    output = _policy_output(observation)
    launches = torch.zeros_like(observation.launchable_mask)
    launches[0, 0] = True
    targets = torch.zeros_like(launches, dtype=torch.long)
    targets[0, 0] = BODY_SLOTS
    actions = SampledActionBatch(launches=launches, targets=targets)

    # This is also the validation helper that the deferred collection path
    # will rerun after a failed GPU flag. The precise ValueError is preferable
    # to a CUDA device assertion, which would poison the remaining process.
    with pytest.raises(ValueError, match="valid body slot"):
        joint_log_prob(
            output,
            actions,
            observation.launchable_mask,
            observation.target_mask,
        )


def test_collection_joint_log_probability_matches_the_eager_calculation() -> None:
    """One collection pass must return the same values plus passing flags."""

    observation = _observation_batch()
    output = _policy_output(observation)
    sample = sample_actions_for_collection(
        output,
        observation,
        generator=torch.Generator().manual_seed(73),
    )

    collection = collection_joint_log_prob(output, sample, observation)
    eager = joint_log_prob(
        output,
        sample.actions,
        observation.launchable_mask,
        observation.target_mask,
    )

    assert torch.equal(collection.values, eager)
    assert collection.validation_flags.tolist() == [True] * 7


def test_safe_collection_sampling_does_not_hide_malformed_scores() -> None:
    """Fallback probabilities keep CUDA safe while flags retain the real failure."""

    observation = _observation_batch()

    invalid_launch_output = _policy_output(observation)
    invalid_launch_output.launch_logits[observation.launchable_mask] = torch.nan
    launch_sample = sample_actions_for_collection(
        invalid_launch_output,
        observation,
        generator=torch.Generator().manual_seed(73),
    )
    launch_result = collection_joint_log_prob(
        invalid_launch_output,
        launch_sample,
        observation,
    )
    assert not launch_result.validation_flags[2]

    invalid_target_output = _policy_output(observation)
    invalid_target_output.launch_logits[observation.launchable_mask] = 100.0
    invalid_target_output.target_logits[observation.target_mask] = torch.nan
    target_sample = sample_actions_for_collection(
        invalid_target_output,
        observation,
        generator=torch.Generator().manual_seed(79),
    )
    target_result = collection_joint_log_prob(
        invalid_target_output,
        target_sample,
        observation,
    )
    # The safe uniform row permits sampling, but both the sampling-level
    # finite-row check and PPO's all-legal-logits check retain the failure.
    assert not target_result.validation_flags[3]
    assert not target_result.validation_flags[6]


def _observation_batch() -> ObservationBatch:
    """Build two games containing optional, forced, and padded sources."""

    batch_size = 2
    body_mask = torch.zeros((batch_size, BODY_SLOTS), dtype=torch.bool)
    body_mask[:, :4] = True
    owned_mask = torch.zeros_like(body_mask)
    owned_mask[0, :2] = True
    owned_mask[1, :3] = True
    player_mask = torch.zeros((batch_size, PLAYER_SLOTS), dtype=torch.bool)
    player_mask[:, :2] = True

    target_mask = torch.zeros(
        (batch_size, BODY_SLOTS, BODY_SLOTS), dtype=torch.bool
    )
    # Game 0 source 1 and game 1 source 2 are owned but have no targets. Their
    # fully masked rows must contribute exact zero without a distribution.
    target_mask[0, 0, 2:4] = True
    target_mask[1, 0, 3] = True
    target_mask[1, 1, 2:4] = True

    return ObservationBatch(
        bodies=torch.zeros((batch_size, BODY_SLOTS, BODY_FEATURES)),
        players=torch.zeros((batch_size, PLAYER_SLOTS, PLAYER_FEATURES)),
        body_mask=body_mask,
        owned_mask=owned_mask,
        player_mask=player_mask,
        target_mask=target_mask,
    )


def _policy_output(observation: ObservationBatch) -> PolicyOutput:
    """Give every optional choice a finite score and every forced row ``-inf``."""

    launch_logits = torch.full_like(
        observation.owned_mask,
        -torch.inf,
        dtype=torch.float32,
    )
    launch_logits[observation.launchable_mask] = torch.tensor([2.0, -1.0, 0.5])

    target_logits = torch.full(
        (observation.batch_size, BODY_SLOTS, BODY_SLOTS),
        -torch.inf,
    )
    target_logits[0, 0, 2:4] = torch.tensor([-0.5, 0.5])
    target_logits[1, 0, 3] = 0.25
    target_logits[1, 1, 2:4] = torch.tensor([1.0, -1.0])
    return PolicyOutput(
        launch_logits=launch_logits,
        target_logits=target_logits,
        values=torch.zeros(observation.batch_size),
    )


def _sampling_log_prob_oracle(
    output: PolicyOutput,
    actions: SampledActionBatch,
    observation: ObservationBatch,
) -> torch.Tensor:
    """Calculate sampled probabilities without PPO's selection or scatter code."""

    game_totals = []
    for game in range(observation.batch_size):
        # This selected sum is a differentiable zero even if a future fixture
        # contains no launchable sources in one game.
        game_total = output.launch_logits[
            game,
            observation.launchable_mask[game],
        ].sum() * 0.0

        # Match the mathematical contract in two explicit stages: every
        # launch decision first, followed by targets only for actual launches.
        for source in range(BODY_SLOTS):
            if not observation.launchable_mask[game, source]:
                continue
            launch = actions.launches[game, source].to(output.launch_logits.dtype)
            game_total = game_total + Bernoulli(
                logits=output.launch_logits[game, source]
            ).log_prob(launch)

        for source in range(BODY_SLOTS):
            if not actions.launches[game, source]:
                continue
            legal_targets = observation.target_mask[game, source]
            logits = output.target_logits[game, source].masked_fill(
                ~legal_targets,
                -torch.inf,
            )
            game_total = game_total + Categorical(logits=logits).log_prob(
                actions.targets[game, source]
            )

        game_totals.append(game_total)

    return torch.stack(game_totals)
