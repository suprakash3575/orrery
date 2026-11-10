"""End-to-end proof that an untrained policy can legally play complete games."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from conftest import run_games
from orrery.env_client import Episode, SlotAction
from orrery.model import ObservationBatch, OrbitPolicy, sample_actions


# Step 3.1 requires ten complete games before the policy path is accepted.
GAME_COUNT = 10

# Fixed model and action randomness makes a failure replayable.
TORCH_SEED = 20260823


@pytest.mark.acceptance
def test_untrained_policy_completes_ten_games_without_rejected_actions(
    env_command: tuple[str, ...],
) -> None:
    """Run ten games through the complete model-to-mask-to-Go action path.

    Winning is irrelevant here because the weights are random. The requirement
    is that every game finishes and Go rejects none of the model's actions.
    """

    # This seed recreates the same initial model weights on every test run.
    torch.manual_seed(TORCH_SEED)
    policy = OrbitPolicy().eval()
    # Action sampling has its own repeatable random stream. Keeping it separate
    # means constructing another Torch object cannot change the chosen actions.
    action_generator = torch.Generator().manual_seed(TORCH_SEED + 1)

    def decide(active: Mapping[int, Episode]) -> dict[int, list[SlotAction]]:
        """Turn the driver's sorted active episodes into policy actions."""

        environment_ids = list(active)
        # Preserve environment order when creating model rows. The same order
        # is reused below when rows become client actions.
        observation = ObservationBatch.from_observations(
            [active[env_id].observation for env_id in environment_ids]
        )
        # This milestone only plays with fixed random weights; it does not
        # train, so Torch does not need to retain gradient history.
        with torch.no_grad():
            output = policy(observation)
            sampled = sample_actions(
                output,
                observation,
                generator=action_generator,
            )
        # Convert Torch tensors into protocol objects only after all batched
        # model sampling is complete, preserving the ID-to-row pairing.
        actions = sampled.to_slot_actions()
        return dict(zip(environment_ids, actions, strict=True))

    # The shared driver owns create/reset/step ordering, truncation checks, and
    # completion bookkeeping. This test supplies only policy decisions.
    run = run_games(
        env_command,
        players=2,
        game_count=GAME_COUNT,
        batch_size=GAME_COUNT,
        decide=decide,
    )

    # These are the two acceptance conditions: all ten games reached a real
    # terminal state, and Go rejected none of the model's submitted actions.
    assert len(run.outcomes) == GAME_COUNT
    assert run.error_count == 0

    # The printed count is evidence for reviewers; it is not a performance or
    # playing-strength threshold.
    print(
        f"untrained-policy smoke: games={len(run.outcomes)} "
        f"environment_steps={run.environment_steps} rejected_actions=0"
    )
