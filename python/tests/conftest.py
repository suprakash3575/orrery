"""Shared test setup that would otherwise be repeated across integration files."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import subprocess

import pytest

from orrery.evaluation_context import configure_cuda_workspace
from orrery.env_client import (
    END_ONGOING,
    EnvClient,
    Episode,
    REPOSITORY_ROOT,
    SCHEDULE_TRAIN_V2,
    SlotAction,
)


# cuBLAS reads its workspace setting when it first starts. GPU tests share one
# process, so fix it once here, before any test can touch CUDA; the entry
# points' main() functions do the same for real commands.
configure_cuda_workspace()


# Both acceptance controllers must face this exact versioned opponent so their
# results remain comparable when either test is changed.
OPPONENT = "heuristic-v1:aggression=1.2,defense=0.1,economy=1,efficiency=1"


@dataclass(frozen=True)
class Outcome:
    """The replay identity and final result of one completed game."""

    seed: int
    seat: int
    reward: float
    turns: int


@dataclass(frozen=True)
class GameRun:
    """Results and protocol evidence returned by the shared game driver."""

    outcomes: tuple[Outcome, ...]
    environment_steps: int
    error_count: int


# A controller receives active episodes in sorted environment-ID order and
# returns one protocol action list for every active ID.
Controller = Callable[
    [Mapping[int, Episode]],
    Mapping[int, Sequence[SlotAction]],
]

# A stateful controller can use this hook to start fresh episode-owned state.
# It is called for every successful reset, even when a reused slot receives the
# same scenario seed and learner seat as its previous episode.
ResetHook = Callable[[Sequence[Episode]], None]


@pytest.fixture(scope="session")
def env_command(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, ...]:
    """Build one isolated environment executable for Python integration tests.

    A session fixture runs once for the complete pytest process. Every test that
    asks for ``env_command`` reuses the same compiled Go binary, avoiding a
    separate compilation for each game test.
    """

    # pytest owns this temporary directory and cleans it after the test session,
    # so the compiled binary never appears as a repository change.
    executable = tmp_path_factory.mktemp("go-env") / "orrery-env"
    # check=True turns a compiler failure into an immediate test setup failure
    # instead of returning a path to an executable that does not exist.
    subprocess.run(
        ["go", "build", "-o", str(executable), "./cmd/env"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    # EnvClient accepts a command sequence. A one-item tuple runs this exact
    # executable without another shell or `go run` wrapper.
    return (str(executable),)


def run_games(
    env_command: tuple[str, ...],
    *,
    players: int,
    game_count: int,
    batch_size: int,
    decide: Controller,
    on_reset: ResetHook | None = None,
) -> GameRun:
    """Play complete games while one shared loop owns the env protocol.

    The caller supplies the controller-specific decision function and may
    supply a reset hook for episode-owned state. This driver consistently
    handles sorted IDs, reset scheduling, termination, truncation rejection,
    and protocol error reporting for every acceptance controller.
    """

    outcomes: list[Outcome] = []
    active: dict[int, Episode] = {}
    environment_steps = 0

    def install_reset_episodes(episodes: Sequence[Episode]) -> None:
        """Notify the controller, then expose each fresh episode as active."""

        # Notify before the next decision. A reset is a lifecycle event in its
        # own right; comparing seed and seat cannot reliably infer that it
        # happened because a later schedule may deliberately repeat both.
        if on_reset is not None:
            on_reset(episodes)
        for episode in episodes:
            active[episode.env_id] = episode

    with EnvClient(env_command, players=players) as client:
        metadata = client.create(
            batch_size,
            (0, 1_000_000),
            OPPONENT,
            schedule=SCHEDULE_TRAIN_V2,
            cursor=0,
        )
        assert metadata["opponent"] == OPPONENT
        assert metadata["schedule"] == SCHEDULE_TRAIN_V2
        assert client.schedule == SCHEDULE_TRAIN_V2
        assert client.cursor == 0

        initial_ids = list(range(min(batch_size, game_count)))
        install_reset_episodes(client.reset(initial_ids))

        while active:
            # Building a new dictionary in sorted order gives every controller
            # the same stable mapping between environment IDs and batch rows.
            ordered_active = {
                env_id: active[env_id]
                for env_id in sorted(active)
            }
            decisions = decide(ordered_active)
            assert set(decisions) == set(active)

            results = client.step(decisions)
            environment_steps += len(results)
            finished_ids: list[int] = []
            for result in results:
                # A response must match a slot just stepped. RL V1 counts only
                # normally terminated games, never external truncations.
                assert result.env_id in active
                assert not result.truncated
                # The end reason and terminated flag describe the same fact.
                assert (result.end_reason != END_ONGOING) == result.terminated
                if result.terminated:
                    outcomes.append(
                        Outcome(result.seed, result.seat, result.reward, result.turn)
                    )
                    del active[result.env_id]
                    finished_ids.append(result.env_id)
                else:
                    # StepResult retains the episode identity and supplies the
                    # next authoritative observation returned by Go.
                    active[result.env_id] = result

            # Refill only as many finished slots as the requested total still
            # needs, even when several games end in the same batched step.
            remaining_starts = game_count - len(outcomes) - len(active)
            reset_ids = sorted(finished_ids)[:remaining_starts]
            if reset_ids:
                install_reset_episodes(client.reset(reset_ids))

        return GameRun(
            outcomes=tuple(outcomes),
            environment_steps=environment_steps,
            error_count=client.error_count,
        )
