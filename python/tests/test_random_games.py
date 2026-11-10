"""Checks for the legal random controller and its reproducible game behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from conftest import OPPONENT, Outcome, run_games
from orrery.env_client import (
    BODY_FEATURES,
    BODY_SLOTS,
    PLAYER_FEATURES,
    PLAYER_SLOTS,
    EnvClient,
    Episode,
    Observation,
    SlotAction,
)
from orrery.random_agent import sample


BATCH_SIZE = 8
GAME_COUNT = 100
PYTHON_SEED = 20250821


class _RandomController:
    """Own one repeatable NumPy random stream per active episode slot."""

    def __init__(self, python_seed: int) -> None:
        """Remember the run seed used to create each episode's random stream."""

        self._python_seed = python_seed
        self._episode_rngs: dict[int, np.random.Generator] = {}

    def reset(self, episodes: Sequence[Episode]) -> None:
        """Start every reset episode from the beginning of its random stream.

        Reset is authoritative even if a slot receives the same scenario seed
        and seat again. Replacing the generator here prevents a new episode
        from continuing random draws consumed by the previous one.
        """

        for episode in episodes:
            self._episode_rngs[episode.env_id] = _episode_rng(
                self._python_seed,
                episode.seed,
                episode.seat,
            )

    def decide(
        self,
        active: Mapping[int, Episode],
    ) -> dict[int, list[SlotAction]]:
        """Choose repeatable random actions for every active episode."""

        return {
            env_id: sample(episode.observation, self._episode_rngs[env_id])
            for env_id, episode in active.items()
        }


def test_random_agent_samples_only_masked_actions() -> None:
    """The random controller should choose only sources and targets allowed by Go."""

    # Arrange a small artificial board. Slots 1, 4, and 7 belong to the
    # learner. Slots 1 and 4 have legal targets, while slot 7 has none and must
    # therefore choose no-launch.
    observation = _empty_observation()
    observation.owned_mask[[1, 4, 7]] = True
    observation.target_mask[1, [0, 2]] = True
    observation.target_mask[4, [3, 5, 6]] = True
    # Slot 7 is owned but has no legal target, as happens with an empty garrison.

    # Act twice with separate random generators initialized from the same seed.
    # Equal results prove that a recorded seed can replay the controller choice.
    # default_rng creates a modern NumPy random-number generator. Supplying 91
    # makes each new generator return the same repeatable sequence of choices.
    first = sample(observation, np.random.default_rng(91))
    second = sample(observation, np.random.default_rng(91))

    # The two seeded samples must match exactly, and the controller must return
    # one decision for every fixed body slot expected by the protocol.
    assert first == second
    assert len(first) == BODY_SLOTS

    # Every attempted launch must come from an owned slot and use a source-to-
    # target pair that Go marked legal. No-launch entries need no valid target.
    for source, action in enumerate(first):
        if action.launch:
            assert observation.owned_mask[source]
            assert observation.target_mask[source, action.target]

    # An owned body without a legal target cannot launch, regardless of the
    # random draw.
    assert not first[7].launch


def test_random_controller_restarts_stream_when_episode_identity_repeats() -> None:
    """Resetting a slot must restart randomness even when seed and seat repeat."""

    observation = _empty_observation()
    # Give every source many legal choices so two different points in the
    # random stream produce a visibly different complete action batch.
    observation.owned_mask[:] = True
    observation.target_mask[:] = True
    # fill_diagonal selects source/target pairs with equal slot numbers and
    # turns them off, matching the normal rule that a body cannot target itself.
    np.fill_diagonal(observation.target_mask, False)
    episode = Episode(
        env_id=0,
        seed=17,
        seat=1,
        explicit=False,
        observation=observation,
    )
    controller = _RandomController(PYTHON_SEED)

    controller.reset([episode])
    first_actions = controller.decide({episode.env_id: episode})
    # Consume more choices to model a completed episode that used its stream.
    for _ in range(3):
        controller.decide({episode.env_id: episode})
    continued_actions = controller.decide({episode.env_id: episode})
    # Confirm this fixed test case can distinguish a continued stream from a
    # restarted one; otherwise the final equality could pass by coincidence.
    assert continued_actions != first_actions

    # A new episode can legitimately reuse the exact same seed and seat. Its
    # first actions must replay from the beginning instead of continuing above.
    controller.reset([episode])
    restarted_actions = controller.decide({episode.env_id: episode})
    assert restarted_actions == first_actions


@pytest.mark.acceptance
def test_random_controller_records_100_game_baseline(
    env_command: tuple[str, ...],
) -> None:
    """Record a stable 100-game reference result for the random controller."""

    # Run the full controller/client/Go-engine path for the requested baseline.
    outcomes, error_count = _run_games(env_command, GAME_COUNT, PYTHON_SEED, players=2)

    # The acceptance requirement is complete collection and zero rejected
    # actions. Wins and losses are measurements of an intentionally weak random
    # controller, not a strength threshold that could fail this plumbing test.
    assert len(outcomes) == GAME_COUNT
    assert error_count == 0

    # Count and print the observed results so future runs can be compared with
    # the recorded baseline without pretending that the random agent must win.
    wins = sum(outcome.reward > 0 for outcome in outcomes)
    losses = sum(outcome.reward < 0 for outcome in outcomes)
    draws = sum(outcome.reward == 0 for outcome in outcomes)
    mean_turns = sum(outcome.turns for outcome in outcomes) / len(outcomes)
    print(
        "random-controller baseline: "
        f"games={len(outcomes)} wins={wins} losses={losses} draws={draws} "
        f"mean_episode_length={mean_turns:.2f}"
    )


@pytest.mark.parametrize("players", [2, 4])
def test_random_controller_is_deterministic(
    env_command: tuple[str, ...],
    players: int,
) -> None:
    """Repeating the same seeded run should produce the same game outcomes."""

    # Run the same eight scheduled episodes twice from fresh environment
    # processes. Both the Python random seed and Go scenario schedule match.
    first, first_errors = _run_games(env_command, BATCH_SIZE, PYTHON_SEED, players=players)
    second, second_errors = _run_games(env_command, BATCH_SIZE, PYTHON_SEED, players=players)

    # Equality compares each episode's seed, learner seat, reward, and final
    # turn. Zero errors additionally proves neither replay attempted an illegal
    # protocol action.
    assert first == second
    assert first_errors == 0
    assert second_errors == 0


def test_four_player_games_complete_with_every_seat_and_valid_results(
    env_command: tuple[str, ...],
) -> None:
    """Play complete four-player games through the real env-v3 process.

    The learner's seat comes from the map's four seats, every terminal reward
    is -1, 0, or +1, and run_games checks each step's end reason against its
    terminated flag. Seeds 0-7 of train-v2 seat the learner in seats 2 or 3 at
    least once, which two-player maps can never do.
    """

    outcomes, error_count = _run_games(env_command, BATCH_SIZE, PYTHON_SEED, players=4)

    assert len(outcomes) == BATCH_SIZE
    assert error_count == 0
    assert {outcome.seat for outcome in outcomes} <= {0, 1, 2, 3}
    assert any(outcome.seat >= 2 for outcome in outcomes)
    assert {outcome.reward for outcome in outcomes} <= {-1.0, 0.0, 1.0}


def test_episode_actions_match_in_isolation_and_batch(
    env_command: tuple[str, ...],
) -> None:
    """Other games in a batch must not change one episode's actions or result."""

    # Record environment slot 0 once by itself and once while seven other games
    # are progressing beside it.
    isolated, isolated_errors = _episode_trace(env_command, batch_size=1)
    batched, batched_errors = _episode_trace(env_command, batch_size=8)

    # The comparison includes every turn's launch/target choices and the final
    # outcome. Matching traces prove that batch ordering and other games'
    # termination times cannot consume slot 0's random choices.
    assert isolated == batched
    assert isolated_errors == 0
    assert batched_errors == 0


def _run_games(
    env_command: tuple[str, ...], game_count: int, python_seed: int, *, players: int
) -> tuple[list[Outcome], int]:
    """Play the requested number of games and return their results and errors."""

    controller = _RandomController(python_seed)

    run = run_games(
        env_command,
        players=players,
        game_count=game_count,
        batch_size=BATCH_SIZE,
        decide=controller.decide,
        on_reset=controller.reset,
    )
    # Preserve this helper's compact return shape for its existing callers.
    return list(run.outcomes), run.error_count


def _episode_trace(
    env_command: tuple[str, ...], batch_size: int
) -> tuple[tuple[tuple[tuple[bool, int], ...], Outcome], int]:
    """Record every action and the final outcome for environment slot zero."""

    # The target episode is always slot 0. Other slots exist only in the
    # batched run used to prove they cannot influence its random stream.
    target_env = 0
    active: dict[int, Observation] = {}
    # Each running environment receives its own NumPy random generator so other
    # batch members cannot consume or shift its future choices.
    episode_rngs: dict[int, np.random.Generator] = {}
    target_actions: list[tuple[tuple[bool, int], ...]] = []

    with EnvClient(env_command, players=2) as client:
        client.create(batch_size, (0, 1_000_000), OPPONENT)
        for episode in client.reset(list(range(batch_size))):
            active[episode.env_id] = episode.observation
            episode_rngs[episode.env_id] = _episode_rng(
                PYTHON_SEED, episode.seed, episode.seat
            )

        while True:
            # Sample all active slots normally, but record only slot 0's exact
            # launch flag and target index for every body on every turn.
            decisions = {
                env_id: sample(observation, episode_rngs[env_id])
                for env_id, observation in sorted(active.items())
            }
            target_actions.append(
                tuple(
                    (action.launch, action.target)
                    for action in decisions[target_env]
                )
            )
            for result in client.step(decisions):
                if result.env_id == target_env and result.terminated:
                    # Returning the complete action history plus outcome lets
                    # the caller compare an isolated run with a batched run.
                    outcome = Outcome(
                        result.seed, result.seat, result.reward, result.turn
                    )
                    return (tuple(target_actions), outcome), client.error_count
                if result.terminated:
                    # Non-target games can finish earlier. Removing them must
                    # not alter slot 0's episode-owned random choices.
                    del active[result.env_id]
                    del episode_rngs[result.env_id]
                else:
                    active[result.env_id] = result.observation


def _episode_rng(
    python_seed: int, scenario_seed: int, seat: int
) -> np.random.Generator:
    """Create repeatable random choices owned by one specific episode.

    Mixing the Python seed, board seed, and learner seat prevents other games
    in the batch from consuming this episode's random choices.
    """

    # NumPy SeedSequence accepts integer pieces. Split each 64-bit seed into
    # two 32-bit pieces, then include the learner seat as another distinguishing
    # value. The same episode identity always rebuilds the same random stream.
    word_mask = (1 << 32) - 1
    scenario_bits = scenario_seed & ((1 << 64) - 1)
    seed_material = [
        python_seed & word_mask,
        (python_seed >> 32) & word_mask,
        scenario_bits & word_mask,
        scenario_bits >> 32,
        seat,
    ]
    # SeedSequence mixes the integer pieces into stable seed data; default_rng
    # turns that data into the object used to draw the episode's random choices.
    return np.random.default_rng(np.random.SeedSequence(seed_material))


def _empty_observation() -> Observation:
    """Create a blank observation whose masks a test can fill deliberately."""

    # Zero-filled feature arrays are sufficient because the unit test above is
    # checking action masks, not model or game-state calculations.
    # np.zeros builds correctly shaped protocol arrays containing only zeros.
    # float32 stores decimal features, bool_ stores true/false masks, and int32
    # stores whole-number body IDs. Individual tests turn on only what they need.
    return Observation(
        bodies=np.zeros((BODY_SLOTS, BODY_FEATURES), dtype=np.float32),
        players=np.zeros((PLAYER_SLOTS, PLAYER_FEATURES), dtype=np.float32),
        body_mask=np.zeros(BODY_SLOTS, dtype=np.bool_),
        owned_mask=np.zeros(BODY_SLOTS, dtype=np.bool_),
        player_mask=np.zeros(PLAYER_SLOTS, dtype=np.bool_),
        target_mask=np.zeros((BODY_SLOTS, BODY_SLOTS), dtype=np.bool_),
        slot_body_id=np.zeros(BODY_SLOTS, dtype=np.int32),
    )
