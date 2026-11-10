"""End-to-end proof that OrbitPolicy can learn one tiny fixed scenario."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from orrery.env_client import (
    BODY_FEATURES,
    BODY_SLOTS,
    END_GAME_FINISHED,
    END_ONGOING,
    PLAYER_FEATURES,
    PLAYER_SLOTS,
    Episode,
    Observation,
    SlotAction,
    StepResult,
)
from orrery.model import ObservationBatch, OrbitPolicy, sample_actions
from orrery.ppo import ppo_update
from orrery.rollout import collect_rollout


ENVIRONMENT_COUNT = 32
ROLLOUT_STEPS = 4
TRANSITIONS_PER_UPDATE = ENVIRONMENT_COUNT * ROLLOUT_STEPS
MAX_ENVIRONMENT_STEPS = 4_096
EVALUATION_GAMES = 100
REQUIRED_WINS = 99
TORCH_SEED = 20260902


class _OneMoveEnvironment:
    """A two-turn game where source zero must launch at target one to win.

    Every reset returns the same observation, so this is deliberately an
    overfitting exercise rather than a generalization test. The class obeys
    the same reset/step boundary consumed by ``collect_rollout`` but keeps the
    reward rule visible here: the first-turn action determines the outcome,
    while its terminal reward arrives only on the following turn.
    """

    REPLAY_METADATA = {
        "engine": "one-move-test-v1",
        "schema": "obs-v1",
        "generator": "one-move-test-v1",
        "opponent": "one-move-test-v1",
    }

    def __init__(
        self,
        opening_observation: Observation,
        consequence_observation: Observation,
    ) -> None:
        self.metadata = dict(self.REPLAY_METADATA)
        self._opening_observation = opening_observation
        self._consequence_observation = consequence_observation
        self._active: dict[int, Episode] = {}
        self._opening_wins: dict[int, bool] = {}
        self._turns: dict[int, int] = {}

    def reset(self, ids: Sequence[int]) -> list[Episode]:
        """Install identical fresh two-turn games in the requested slots."""

        episodes = [
            Episode(
                env_id=env_id,
                seed=0,
                seat=0,
                explicit=True,
                observation=self._opening_observation,
            )
            for env_id in sorted(ids)
        ]
        for env_id in ids:
            self._opening_wins.pop(env_id, None)
            self._turns[env_id] = 0
        self._active.update({episode.env_id: episode for episode in episodes})
        return episodes

    def step_packed(
        self,
        action_targets: Mapping[int, Sequence[int]],
    ) -> list[StepResult]:
        """Delay the opening move's outcome until the second, terminal turn."""

        if tuple(sorted(action_targets)) != tuple(sorted(self._active)):
            raise ValueError("actions must name every active environment")

        results: list[StepResult] = []
        for env_id in sorted(action_targets):
            episode = self._active[env_id]
            turn = self._turns[env_id] + 1
            targets = action_targets[env_id]
            if len(targets) != BODY_SLOTS:
                raise ValueError("each action must contain every body slot")

            if turn == 2:
                result = StepResult(
                    env_id=env_id,
                    seed=episode.seed,
                    seat=episode.seat,
                    explicit=episode.explicit,
                    observation=self._consequence_observation,
                    reward=1.0 if self._opening_wins.pop(env_id) else -1.0,
                    terminated=True,
                    truncated=False,
                    end_reason=END_GAME_FINISHED,
                    turn=2,
                )
                del self._active[env_id]
                del self._turns[env_id]
            elif turn == 1:
                self._opening_wins[env_id] = targets[0] == 1
                result = StepResult(
                    env_id=env_id,
                    seed=episode.seed,
                    seat=episode.seat,
                    explicit=episode.explicit,
                    observation=self._consequence_observation,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    end_reason=END_ONGOING,
                    turn=1,
                )
                self._turns[env_id] = turn
            else:
                raise AssertionError(
                    f"environment {env_id} reached unexpected turn {turn}"
                )
            results.append(result)
        return results

    def step(
        self,
        actions: Mapping[int, Sequence[SlotAction]],
    ) -> list[StepResult]:
        """Retain the ordinary controller interface used by evaluation."""

        # The production EnvClient keeps this compatibility path as well. The
        # test environment converts it to the same compact representation so
        # training collection and post-training evaluation share game behavior.
        packed = {
            env_id: [action.target if action.launch else -1 for action in row]
            for env_id, row in actions.items()
        }
        return self.step_packed(packed)


def test_orbit_policy_overfits_one_move_scenario() -> None:
    """Learn at least a 99% sampled win rate within a fixed transition budget."""

    torch.manual_seed(TORCH_SEED)
    opening_observation, consequence_observation = _one_move_observations()
    environment = _OneMoveEnvironment(
        opening_observation,
        consequence_observation,
    )
    episodes = environment.reset(range(ENVIRONMENT_COUNT))
    policy = OrbitPolicy().eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    action_generator = torch.Generator().manual_seed(TORCH_SEED + 1)
    shuffle_generator = torch.Generator().manual_seed(TORCH_SEED + 2)
    environment_steps = 0
    (
        initial_win_probability,
        initial_launch_probability,
        initial_target_probability,
    ) = _winning_action_probabilities(
        policy,
        opening_observation,
    )

    # Each update sees another fixed-policy rollout from the same two-turn
    # game. The decisive first action initially receives zero reward; GAE must
    # carry the second turn's terminal outcome backward before PPO can raise
    # both its launch and target probabilities. Stop early once the stochastic
    # policy is at least 99.5% confident, leaving headroom for 100 sampled games.
    while environment_steps < MAX_ENVIRONMENT_STEPS:
        collected = collect_rollout(
            environment,
            policy,
            episodes,
            steps=ROLLOUT_STEPS,
            generator=action_generator,
        )
        ppo_update(
            policy,
            optimizer,
            collected.batch,
            epochs=4,
            minibatch_size=TRANSITIONS_PER_UPDATE,
            shuffle_generator=shuffle_generator,
        )
        episodes = list(collected.next_episodes)
        environment_steps += collected.batch.transition_count
        if _winning_action_probabilities(
            policy,
            opening_observation,
        )[0] >= 0.995:
            break

    (
        final_win_probability,
        final_launch_probability,
        final_target_probability,
    ) = _winning_action_probabilities(
        policy,
        opening_observation,
    )
    wins = _sample_wins(
        policy,
        opening_observation,
        consequence_observation,
        EVALUATION_GAMES,
    )

    assert initial_win_probability < 0.995
    assert final_launch_probability > initial_launch_probability
    assert final_target_probability > initial_target_probability
    assert final_win_probability >= 0.995
    assert wins >= REQUIRED_WINS, (
        f"policy won {wins}/{EVALUATION_GAMES} games after "
        f"{environment_steps} environment steps"
    )
    print(
        "tiny-scenario overfit: "
        f"initial_win_probability={initial_win_probability:.4f} "
        f"final_win_probability={final_win_probability:.4f} "
        f"launch_probability={final_launch_probability:.4f} "
        f"target_probability={final_target_probability:.4f} "
        f"wins={wins}/{EVALUATION_GAMES} environment_steps={environment_steps}"
    )


def _winning_action_probabilities(
    policy: OrbitPolicy,
    observation: Observation,
) -> tuple[float, float, float]:
    """Return joint, launch, and correct-target probabilities for the winning move."""

    batch = ObservationBatch.from_observations([observation])
    with torch.no_grad():
        output = policy(batch)
        launch_probability = torch.sigmoid(output.launch_logits[0, 0])
        target_probability = torch.softmax(
            output.target_logits[0, 0, [1, 2]],
            dim=0,
        )[0]
        probability = launch_probability * target_probability
    return (
        float(probability),
        float(launch_probability),
        float(target_probability),
    )


def _sample_wins(
    policy: OrbitPolicy,
    opening_observation: Observation,
    consequence_observation: Observation,
    game_count: int,
) -> int:
    """Play a fixed evaluation batch and count positive terminal rewards."""

    batch = ObservationBatch.from_observations(
        [opening_observation] * game_count
    )
    generator = torch.Generator().manual_seed(TORCH_SEED + 3)
    with torch.no_grad():
        actions = sample_actions(
            policy(batch),
            batch,
            generator=generator,
        )
    environment = _OneMoveEnvironment(
        opening_observation,
        consequence_observation,
    )
    environment_ids = tuple(range(game_count))
    environment.reset(environment_ids)
    first_results = environment.step(
        dict(
            zip(
                environment_ids,
                actions.to_slot_actions(),
                strict=True,
            )
        )
    )
    consequence_batch = ObservationBatch.from_observations(
        [result.observation for result in first_results]
    )
    with torch.no_grad():
        consequence_actions = sample_actions(
            policy(consequence_batch),
            consequence_batch,
            generator=generator,
        )
    terminal_results = environment.step(
        dict(
            zip(
                environment_ids,
                consequence_actions.to_slot_actions(),
                strict=True,
            )
        )
    )
    return sum(result.reward > 0.0 for result in terminal_results)


def _one_move_observations() -> tuple[Observation, Observation]:
    """Build a decision state and the state that precedes delayed reward."""

    bodies = np.zeros((BODY_SLOTS, BODY_FEATURES), dtype=np.float32)
    # Source zero belongs to the learner. Target one is visibly weak and
    # target two visibly strong, giving the target head distinct inputs it can
    # memorize instead of two interchangeable all-zero tokens.
    bodies[0, [0, 4, 5]] = (-0.8, 0.5, 1.0)
    bodies[1, [0, 4, 7]] = (0.2, 0.1, 1.0)
    bodies[2, [0, 4, 7]] = (0.8, 0.9, 1.0)
    bodies[:3, [10, 13, 15]] = 1.0
    players = np.zeros((PLAYER_SLOTS, PLAYER_FEATURES), dtype=np.float32)
    players[0, 2] = 1.0
    body_mask = np.zeros(BODY_SLOTS, dtype=np.bool_)
    body_mask[:3] = True
    owned_mask = np.zeros(BODY_SLOTS, dtype=np.bool_)
    owned_mask[0] = True
    player_mask = np.zeros(PLAYER_SLOTS, dtype=np.bool_)
    player_mask[:2] = True
    target_mask = np.zeros((BODY_SLOTS, BODY_SLOTS), dtype=np.bool_)
    target_mask[0, 1] = True
    target_mask[0, 2] = True
    slot_body_id = np.full(BODY_SLOTS, -1, dtype=np.int32)
    slot_body_id[:3] = (1, 2, 3)
    opening = Observation(
        bodies=bodies,
        players=players,
        body_mask=body_mask,
        owned_mask=owned_mask,
        player_mask=player_mask,
        target_mask=target_mask,
        slot_body_id=slot_body_id,
    )
    consequence_bodies = bodies.copy()
    consequence_bodies[:3, 14] = 0.5
    consequence = Observation(
        bodies=consequence_bodies,
        players=players.copy(),
        body_mask=body_mask.copy(),
        owned_mask=owned_mask.copy(),
        player_mask=player_mask.copy(),
        target_mask=np.zeros_like(target_mask),
        slot_body_id=slot_body_id.copy(),
    )
    return opening, consequence
