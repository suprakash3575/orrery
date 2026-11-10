"""Focused orchestration tests for the first PPO training run."""

from __future__ import annotations

import csv
from copy import copy
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

import orrery.archive as archive_module
import orrery.checkpoint as checkpoint_module
import orrery.train as training
from orrery.archive import freeze_passing_checkpoint
from orrery.checkpoint import load_checkpoint, sha256_file
from orrery.config import RunConfig, load_run_config
from orrery.env_client import (
    END_GAME_FINISHED,
    END_LEARNER_ELIMINATED,
    OBSERVATION_DIMENSIONS,
)
from orrery.evaluator import (
    NOMINAL_INTERVAL_ASSUMPTION,
    EvaluationOutcome,
    EvaluationResult,
    RunnerTimings,
)
from orrery.model import OrbitPolicy
from orrery.train import (
    TrainingResult,
    can_collect_full_rollout,
    initial_stage_decision,
    run_training,
    schedule_values,
)


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "rl-v2-2p.toml"
# What the real runner records about how it chose actions. Fake runners
# report it too, because validation refuses results from another evaluator.
_EVALUATION = {"policy": "eval-policy-v2", "device": "cpu"}
# Tests that resume against the real Go environment use a binary built from the
# working tree. With uncommitted changes it reports "<commit>+dirty", which a
# resume refuses by default, so those tests allow it. The refusal itself is
# covered by its own tests below.
_TEST_BINARY_ENGINE_MAY_BE_DIRTY = True


def _test_config(**overrides: object) -> RunConfig:
    """Build a deliberately tiny configuration for orchestration tests.

    Production ``RunConfig`` construction requires the official rollout
    layout. Several integration tests need one or four transitions so they
    finish quickly. This helper copies an already validated official
    configuration and changes fields only inside this test module. Normal TOML
    loading and training commands cannot bypass the versioned layout checks.
    The production file trains on CUDA; these tests run on CPU unless an
    override says otherwise.
    """

    config = copy(load_run_config(CONFIG))
    object.__setattr__(config, "device", "cpu")
    for name, value in overrides.items():
        if name not in RunConfig.__dataclass_fields__:
            raise AssertionError(f"unknown RunConfig test override: {name}")
        object.__setattr__(config, name, value)
    return config


@pytest.fixture(autouse=True)
def _accept_test_layouts_in_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let checkpoints written from _test_config load again in this module.

    _test_config's tiny layouts bypass RunConfig's layout check, and loading a
    checkpoint rebuilds the recorded config through that same check, so it
    would refuse every checkpoint these orchestration tests write. The
    expected-config comparison still runs. test_checkpoint.py covers the
    recorded-identity check itself.
    """

    monkeypatch.setattr(checkpoint_module, "run_config_from_mapping", lambda _raw: None)


def test_schedules_decay_from_configured_values_and_stop_at_their_floors() -> None:
    """Explain how training coefficients change as environment steps accumulate.

    At step zero the optimizer and entropy bonuses use their configured starting
    values. Halfway through the 500M cumulative budget, entropy has reached its
    configured floor while the learning rate is halfway to zero. At the final
    ceiling, learning is zero and neither entropy coefficient falls below its
    floor. These checks keep scheduling tied to recorded environment steps.
    """

    config = load_run_config(CONFIG)

    start = schedule_values(config, 0)
    entropy_end = schedule_values(config, 250_000_000)
    training_end = schedule_values(config, config.cumulative_step_ceiling)

    assert start.learning_rate == config.learning_rate
    assert start.launch_entropy == config.launch_entropy_initial
    assert start.target_entropy == config.target_entropy_initial
    assert entropy_end.learning_rate == pytest.approx(config.learning_rate / 2)
    assert entropy_end.launch_entropy == config.launch_entropy_final
    assert entropy_end.target_entropy == config.target_entropy_final
    assert training_end.learning_rate == 0.0
    assert training_end.launch_entropy == config.launch_entropy_final
    assert training_end.target_entropy == config.target_entropy_final


def test_episode_metrics_bulk_transfer_matches_the_previous_row_loop() -> None:
    """Bulk CPU data must preserve metrics across rollout boundaries.

    Environment 1 remains active after the first rollout, so its launch count
    must carry into the second rollout. Environment 0 finishes once in each
    batch. Comparing against the item-by-item calculation covers reward,
    length, ship advantage, no-launch rate, and per-episode launch accounting.
    The games are two-player, so the opponent sits in player row 2.
    """

    scale = 1_000.0

    def encoded_ships(ships: float) -> float:
        """Encode a real ship total exactly like the Go observation contract."""

        return math.log1p(ships) / math.log1p(scale)

    def metric_rollout(
        *,
        launchable_sources: list[tuple[int, ...]],
        launched_sources: list[tuple[int, ...]],
        terminated: list[bool],
        rewards: list[float],
        turns: list[int],
        ships: list[tuple[float, float]],
    ) -> SimpleNamespace:
        """Build only the RolloutBatch fields consumed by the metric tracker."""

        row_count = len(terminated)
        launchable = torch.zeros((row_count, 44), dtype=torch.bool)
        launched = torch.zeros((row_count, 44), dtype=torch.bool)
        for row, sources in enumerate(launchable_sources):
            launchable[row, list(sources)] = True
        for row, sources in enumerate(launched_sources):
            launched[row, list(sources)] = True

        players = torch.zeros((row_count, 4, 6), dtype=torch.float32)
        for row, (learner, opponent) in enumerate(ships):
            players[row, 0, 0] = encoded_ships(learner)
            players[row, 2, 0] = encoded_ships(opponent)

        # SimpleNamespace keeps this test focused on metric calculation. These
        # fields have the same shapes, dtypes, and ordering as a RolloutBatch.
        return SimpleNamespace(
            environment_ids=(0, 1),
            environment_count=2,
            transition_count=row_count,
            observations=SimpleNamespace(launchable_mask=launchable),
            actions=SimpleNamespace(launches=launched),
            terminated=torch.tensor(terminated, dtype=torch.bool),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            turns=torch.tensor(turns, dtype=torch.long),
            next_observations=SimpleNamespace(players=players),
        )

    def row_by_row_reference(
        launch_state: dict[int, int],
        rollout: SimpleNamespace,
    ) -> training.GameplayMetrics:
        """Retain the pre-optimization algorithm as an independent oracle."""

        launchable = rollout.observations.launchable_mask
        launched = rollout.actions.launches & launchable
        decision_count = int(launchable.sum().item())
        no_launch_count = int((launchable & ~launched).sum().item())
        finished_rewards: list[float] = []
        lengths: list[int] = []
        advantages: list[float] = []
        completed_launches: list[int] = []

        for row in range(rollout.transition_count):
            env_id = rollout.environment_ids[row % rollout.environment_count]
            launch_state[env_id] += int(launched[row].sum().item())
            if not bool(rollout.terminated[row].item()):
                continue
            reward = float(rollout.rewards[row].item())
            players = rollout.next_observations.players[row]
            learner = math.expm1(
                float(players[0, 0].item()) * math.log1p(scale)
            )
            strongest_opponent = max(
                math.expm1(float(players[opponent, 0].item()) * math.log1p(scale))
                for opponent in (1, 2, 3)
            )
            finished_rewards.append(reward)
            lengths.append(int(rollout.turns[row].item()))
            advantages.append(learner - strongest_opponent)
            completed_launches.append(launch_state[env_id])
            launch_state[env_id] = 0

        wins = sum(reward > 0.0 for reward in finished_rewards)
        losses = sum(reward < 0.0 for reward in finished_rewards)
        draws = len(finished_rewards) - wins - losses

        def mean(values: list[int] | list[float]) -> float:
            """Match the production zero-for-empty arithmetic."""

            return sum(values) / len(values) if values else 0.0

        return training.GameplayMetrics(
            wins=wins,
            losses=losses,
            draws=draws,
            mean_episode_length=mean(lengths),
            mean_ship_advantage=mean(advantages),
            no_launch_rate=(
                no_launch_count / decision_count if decision_count else 0.0
            ),
            launches_per_episode=mean(completed_launches),
        )

    first = metric_rollout(
        launchable_sources=[(0, 1), (0,), (0, 1, 2), (0, 1)],
        launched_sources=[(0,), (), (0, 2), (1,)],
        terminated=[False, False, True, False],
        rewards=[0.0, 0.0, 1.0, 0.0],
        turns=[1, 1, 2, 2],
        ships=[(40, 45), (50, 50), (80, 50), (60, 55)],
    )
    second = metric_rollout(
        launchable_sources=[(0,), (0, 1), (0, 1), (0,)],
        launched_sources=[(), (0, 1), (1,), ()],
        terminated=[False, False, True, True],
        rewards=[0.0, 0.0, 0.0, -1.0],
        turns=[1, 3, 2, 4],
        ships=[(45, 45), (55, 60), (70, 70), (40, 90)],
    )

    tracker = training.EpisodeMetricTracker((0, 1))
    reference_state = {0: 0, 1: 0}
    assert tracker.consume(first) == row_by_row_reference(reference_state, first)
    assert tracker.state_dict()["launches_by_environment"] == reference_state
    assert tracker.consume(second) == row_by_row_reference(reference_state, second)
    assert tracker.state_dict()["launches_by_environment"] == reference_state


@pytest.mark.parametrize(
    ("ships_by_row", "expected_advantage"),
    [
        # Two players: the opponent is "opposite", row 2; rows 1 and 3 are
        # padding.
        ((80, 0, 50, 0), 30),
        # Four players: compare with the strongest of the three opponents.
        ((100, 30, 120, 60), -20),
        # An eliminated opponent's row is all zeros and never wins the maximum.
        ((40, 0, 25, 10), 15),
        # Every opponent eliminated: the advantage is the learner's own ships.
        ((10, 0, 0, 0), 10),
        # 1,000 is a reference scale, not a cap: larger fleets still decode.
        ((5_000, 1_200, 0, 300), 3_800),
    ],
)
def test_ship_advantage_compares_the_learner_with_the_strongest_opponent(
    ships_by_row: tuple[int, int, int, int],
    expected_advantage: int,
) -> None:
    """Decode each player's ships with obs-v2's fixed scale and take the gap.

    Player rows are me, clockwise, opposite, and counter-clockwise. Winning
    means finishing strictly first, so the learner is compared with whichever
    opponent holds the most ships at the end of the game.
    """

    players = torch.zeros((1, 4, 6), dtype=torch.float32)
    for row, ships in enumerate(ships_by_row):
        players[0, row, 0] = math.log1p(ships) / math.log1p(1_000)
    rollout = SimpleNamespace(
        environment_ids=(0,),
        environment_count=1,
        transition_count=1,
        observations=SimpleNamespace(
            launchable_mask=torch.zeros((1, 44), dtype=torch.bool)
        ),
        actions=SimpleNamespace(launches=torch.zeros((1, 44), dtype=torch.bool)),
        terminated=torch.tensor([True]),
        rewards=torch.tensor([0.0]),
        turns=torch.tensor([200]),
        next_observations=SimpleNamespace(players=players),
    )

    metrics = training.EpisodeMetricTracker((0,)).consume(rollout)

    # float32 features round the decoded counts slightly.
    assert metrics.mean_ship_advantage == pytest.approx(expected_advantage, abs=0.5)


def test_one_real_update_writes_config_and_complete_metrics(
    env_command: tuple[str, ...],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise one complete collect, learn, log, and checkpoint cycle.

    The production configuration is reduced to two Go games and two turns so
    the test is fast, but it still uses the real environment process, model,
    rollout collector, GAE calculation, and PPO optimizer. After four total
    transitions, the test checks the run configuration, CSV row, checkpoint,
    custom RNG streams, schedule cursor, and manifest fingerprint. This is the
    smallest end-to-end proof that all training stages are connected.
    """

    config = _test_config(
        environment_count=2,
        rollout_steps=2,
        minibatch_size=4,
        ppo_epochs=1,
    )
    run_directory = tmp_path / "run"

    result = run_training(
        config,
        run_directory,
        command=env_command,
        max_updates=1,
        profile=True,
    )

    assert result.environment_steps == 4
    assert result.completed_updates == 1
    assert result.reason == "requested update limit"
    output = capsys.readouterr().out
    assert "training started: device=cpu environments=2" in output
    # The engine version and opponent are on the first lines, so a run started
    # from stale code shows it at once.
    assert f"opponent={config.training_opponent} generator={config.generator_version}" in output
    assert "environment: engine=" in output
    assert "update=1 progress: collecting 4 transitions" in output
    assert (
        "update=1 progress: rollout complete; running 1 optimizer steps "
        "(1 epochs x 1 minibatches)"
    ) in output
    assert "update=1 steps=4" in output
    serialized_config = json.loads((run_directory / "run-config.json").read_text())
    assert "git_commit" not in serialized_config
    assert serialized_config["contract"] == config.contract
    assert serialized_config["learner_seed"] == config.learner_seed
    assert serialized_config["weight_decay"] == 0.01
    assert serialized_config["release_validation_seeds"] == 500
    assert serialized_config["release_required_wins"] == 900
    assert serialized_config["training_seeds"] == {"start": 0, "end": 1_000_000}
    with (run_directory / "metrics.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert rows[0]["update"] == "1"
    assert rows[0]["environment_steps"] == "4"
    assert float(rows[0]["environment_steps_per_second"]) > 0.0
    # Performance data has its own schema so metrics.csv remains compatible
    # with existing runs. This profiled update must record both honest
    # invocation-wide throughput and the major synchronized phase boundaries.
    with (run_directory / "timings.csv").open(newline="") as source:
        timing_rows = list(csv.DictReader(source))
    assert len(timing_rows) == 1
    timing = timing_rows[0]
    assert timing["update"] == "1"
    assert timing["environment_steps"] == "4"
    assert timing["invocation_environment_steps"] == "4"
    assert timing["profiling_enabled"] == "True"
    assert float(timing["end_to_end_steps_per_second"]) > 0.0
    # The per-update rate excludes process setup such as starting Go, so for
    # the first update it must exceed the cumulative invocation-wide rate.
    assert timing["update_environment_steps"] == "4"
    assert float(timing["update_steps_per_second"]) > float(
        timing["end_to_end_steps_per_second"]
    )
    assert float(timing["checkpoint_seconds"]) > 0.0
    for phase in (
        "policy_seconds",
        "sampling_seconds",
        "log_probability_seconds",
        "action_encode_seconds",
        "go_wait_seconds",
        "observation_decode_seconds",
        "tensor_transfer_seconds",
        "bootstrap_seconds",
        "rollout_build_seconds",
        "ppo_seconds",
        "episode_metrics_seconds",
    ):
        assert float(timing[phase]) > 0.0
    # Only the removed env-v1 JSON transport entered this phase. The column is
    # kept for existing timings.csv files and must now record no time.
    assert float(timing["json_decode_seconds"]) == 0.0
    checkpoint_path = run_directory / "checkpoints" / "update-000001.pt"
    checkpoint = load_checkpoint(checkpoint_path, expected_config=config)
    assert checkpoint.environment_steps == 4
    assert checkpoint.completed_updates == 1
    assert checkpoint.optimizer_state["param_groups"][0]["weight_decay"] == 0.01
    assert checkpoint.seed_cursor == 2
    assert checkpoint.environment_metadata["schedule"] == "train-v2"
    assert set(checkpoint.random_states) >= {
        "python",
        "numpy",
        "torch_cpu",
        "action_sampling",
        "minibatch_shuffle",
    }
    manifest = json.loads((run_directory / "manifest.json").read_text())
    assert manifest["checkpoints"][0]["sha256"] == (
        run_directory / "checkpoints" / "update-000001.pt.sha256"
    ).read_text().strip()


def test_two_resumes_from_one_checkpoint_continue_deterministically(
    env_command: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the completed-update resume contract with two real continuations.

    A small real run first completes one PPO update and saves its boundary. The
    entire run directory is then copied twice, which gives both continuations
    the exact same checkpoint, metrics history, and manifest. Each copy resumes
    for one more update through a real Go process and real PPO optimizer.

    The collector wrapper observes rather than replaces rollout collection. It
    records the new episode assignments and the resulting actions and rollout
    tensors so the test can compare what crossed the Python/Go training path.
    A controlled clock removes machine-speed differences from the CSV. Matching
    traces, weights, counters, cursors, and CSV bytes then demonstrates that two
    resumes from the same boundary agree on the fixed test window.
    """

    config = _test_config(
        environment_count=2,
        rollout_steps=2,
        minibatch_size=4,
        ppo_epochs=1,
        checkpoint_every_updates=1,
    )
    original_run = tmp_path / "original"
    run_training(
        config,
        original_run,
        command=env_command,
        max_updates=1,
        clock=iter((10.0, 11.0)).__next__,
    )
    original_checkpoint_path = (
        original_run / "checkpoints" / "update-000001.pt"
    )
    original_checkpoint = load_checkpoint(
        original_checkpoint_path,
        expected_config=config,
    )

    # Copy every artifact because resume verifies that its checkpoint belongs to
    # the run manifest and that metrics end at the same completed update.
    resumed_runs = (tmp_path / "resume-a", tmp_path / "resume-b")
    for destination in resumed_runs:
        shutil.copytree(original_run, destination)

    real_collect_rollout = training.collect_rollout
    traces: list[dict[str, object]] = []

    def traced_collect_rollout(
        environment: object,
        policy: object,
        episodes: object,
        **kwargs: object,
    ) -> object:
        # Capture the assignments supplied to the learner before collection,
        # then retain the exact sampled actions and outcome tensors afterward.
        assigned = tuple(
            (episode.env_id, episode.seed, episode.seat)
            for episode in episodes
        )
        collected = real_collect_rollout(
            environment,
            policy,
            episodes,
            **kwargs,
        )
        traces.append(
            {
                "assigned": assigned,
                "seeds": collected.batch.seeds.cpu().clone(),
                "seats": collected.batch.seats.cpu().clone(),
                "launches": collected.batch.actions.launches.cpu().clone(),
                "targets": collected.batch.actions.targets.cpu().clone(),
                "rewards": collected.batch.rewards.cpu().clone(),
                "advantages": collected.batch.advantages.cpu().clone(),
            }
        )
        return collected

    monkeypatch.setattr(training, "collect_rollout", traced_collect_rollout)

    results = []
    final_checkpoints = []
    for destination in resumed_runs:
        checkpoint_path = destination / "checkpoints" / "update-000001.pt"
        results.append(
            run_training(
                config,
                destination,
                command=env_command,
                max_updates=2,
                resume_checkpoint=checkpoint_path,
                clock=iter((20.0, 21.0)).__next__,
                allow_development_engine=_TEST_BINARY_ENGINE_MAY_BE_DIRTY,
            )
        )
        final_checkpoints.append(
            load_checkpoint(
                destination / "checkpoints" / "update-000002.pt",
                expected_config=config,
            )
        )

    # The first reset after resume starts at cursor 2 because the initial reset
    # consumed seeds 0 and 1. No abandoned assignment is replayed.
    assert traces[0]["assigned"] == traces[1]["assigned"]
    assert tuple(
        (environment_id, seed)
        for environment_id, seed, _seat in traces[0]["assigned"]
    ) == ((0, 2), (1, 3))
    for field in ("seeds", "seats", "launches", "targets", "rewards", "advantages"):
        assert torch.equal(traces[0][field], traces[1][field])

    # Both continuations complete exactly one additional update. Equal learned
    # tensors and exact CSV bytes cover the optimizer result and logged metrics.
    assert [result.environment_steps for result in results] == [8, 8]
    assert [result.completed_updates for result in results] == [2, 2]
    assert [result.reason for result in results] == [
        "requested update limit",
        "requested update limit",
    ]
    for name in final_checkpoints[0].model_state:
        assert torch.equal(
            final_checkpoints[0].model_state[name],
            final_checkpoints[1].model_state[name],
        )
    assert final_checkpoints[0].environment_steps == 8
    assert final_checkpoints[0].completed_updates == 2
    assert final_checkpoints[0].seed_cursor == final_checkpoints[1].seed_cursor
    assert (
        resumed_runs[0] / "metrics.csv"
    ).read_bytes() == (
        resumed_runs[1] / "metrics.csv"
    ).read_bytes()

    # Each audit names the exact source boundary and every live game discarded
    # there. The saved cursor remains the start of the replacement assignments.
    expected_abandoned = [
        {
            "environment_id": episode.environment_id,
            "seed": episode.seed,
            "learner_seat": episode.learner_seat,
            "turn": episode.turn,
        }
        for episode in original_checkpoint.active_episodes
    ]
    for destination in resumed_runs:
        audit = json.loads((destination / "resume-001.json").read_text())
        assert audit["completed_updates"] == 1
        assert audit["environment_steps"] == 4
        assert audit["next_seed_cursor"] == original_checkpoint.seed_cursor == 2
        assert audit["abandoned_episodes"] == expected_abandoned
        assert audit["checkpoint_sha256"] == (
            destination / "checkpoints" / "update-000001.pt.sha256"
        ).read_text().strip()
        # The audit names the resumed engine and whether an unidentified
        # version was allowed through.
        assert isinstance(audit["engine"], str) and audit["engine"].strip()
        assert audit["allow_development_engine"] is _TEST_BINARY_ENGINE_MAY_BE_DIRTY


def test_validation_after_resume_writes_a_revised_same_update_checkpoint(
    tmp_path: Path,
) -> None:
    """Persist new gate history without replacing the resume checkpoint.

    A restored run can evaluate immediately at its completed-update boundary,
    before another PPO update changes the weights. The original checkpoint is
    immutable, but reusing it would lose the newly appended validation record.
    This test writes both states and proves the gate-bearing revision has a new
    path, a new fingerprint, and the durable history needed by another resume.
    """

    config = _test_config(
        environment_count=1,
        rollout_steps=1,
        minibatch_size=1,
        ppo_epochs=1,
    )
    run_directory = tmp_path / "same-update"
    (run_directory / "checkpoints").mkdir(parents=True)
    (run_directory / "manifest.json").write_text(
        json.dumps({"format": "run-manifest-v1", "checkpoints": []})
    )
    policy = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(policy.parameters())
    action_generator = torch.Generator().manual_seed(1)
    shuffle_generator = torch.Generator().manual_seed(2)
    tracker = training.EpisodeMetricTracker(())
    saved: dict[int, tuple[Path, str]] = {}

    def save(history: list[dict[str, object]], *, force_new: bool) -> tuple[Path, str]:
        return training._save_training_checkpoint(
            run_directory,
            config,
            policy,
            optimizer,
            schedule_values(config, 1),
            action_generator,
            shuffle_generator,
            {},
            1,
            1,
            config.pipeline_gate_every_steps,
            history,
            tracker,
            (),
            0,
            saved,
            purpose="test-validation",
            force_new=force_new,
        )

    original_path, original_fingerprint = save([], force_new=False)
    gate_record = {
        "gate": "pipeline",
        "validation_boundary": 1,
        "training_environment_steps": 1,
        "passed": True,
    }
    revised_path, revised_fingerprint = save([gate_record], force_new=True)

    assert original_path.name == "update-000001.pt"
    assert revised_path.name == "update-000001-state-001.pt"
    assert revised_path != original_path
    assert revised_fingerprint != original_fingerprint
    assert load_checkpoint(original_path).validation_state == {"history": []}
    assert load_checkpoint(revised_path).validation_state == {
        "history": [gate_record]
    }
    assert saved[1] == (revised_path, revised_fingerprint)


def test_resume_reconstructs_release_phase_and_uses_the_release_gate(
    env_command: tuple[str, ...],
    tmp_path: Path,
) -> None:
    """Continue with heuristic validation after restoring a pipeline pass.

    The first invocation performs one update and passes the small pipeline gate,
    then stops at its requested update limit. The second invocation restores
    that checkpoint. Its next gate must use the release opponent and thresholds;
    replaying the pipeline gate would show that phase state was lost on resume.
    """

    config = _test_config(
        environment_count=2,
        rollout_steps=2,
        minibatch_size=4,
        ppo_epochs=1,
        pipeline_gate_every_steps=4,
        review_steps=8,
        initial_step_ceiling=12,
        cumulative_step_ceiling=20,
        pipeline_validation_seeds=1,
        pipeline_required_wins=2,
        release_validation_seeds=1,
        release_required_wins=2,
        release_minimum_lower_bound=0.1,
    )
    opponents: list[str] = []

    def passing_validation(_policy: object, **kwargs: object) -> EvaluationResult:
        opponents.append(kwargs["opponent"])
        seeds = kwargs["seeds"]
        metadata = dict(kwargs["expected_metadata"])
        metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
        return EvaluationResult(
            tuple(
                EvaluationOutcome(seed, seat, 1.0, 10, END_GAME_FINISHED, 1)
                for seed in range(seeds.start, seeds.end)
                for seat in (0, 1)
            ),
            metadata,
            rejected_actions=0,
        )

    run_directory = tmp_path / "release-resume"
    archive_root = tmp_path / "release-resume-archive"
    first = run_training(
        config,
        run_directory,
        command=env_command,
        max_updates=1,
        archive_root=archive_root,
        validation_runner=passing_validation,
    )
    pipeline_record = json.loads(
        (run_directory / "pipeline-000000004.json").read_text()
    )
    resumed = run_training(
        config,
        run_directory,
        command=env_command,
        max_updates=2,
        archive_root=archive_root,
        resume_checkpoint=(
            run_directory / "checkpoints" / pipeline_record["checkpoint"]
        ),
        validation_runner=passing_validation,
        allow_development_engine=_TEST_BINARY_ENGINE_MAY_BE_DIRTY,
    )

    assert first.reason == "requested update limit"
    assert resumed.reason == "release readiness gate passed"
    assert resumed.completed_updates == 2
    assert opponents == [config.pipeline_opponent, config.training_opponent]
    assert (run_directory / "release-000000008.json").is_file()


def test_resume_retries_passing_release_publication_after_freeze_failure(
    env_command: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover when validation was saved but its frozen archive was not.

    The first invocation fails at the archive boundary after its checkpoint
    already contains the passing release record. Resume must retry publication
    from that checkpoint without replaying validation or starting more training.
    """

    config = _test_config(
        environment_count=2,
        rollout_steps=2,
        minibatch_size=4,
        ppo_epochs=1,
        checkpoint_every_updates=50,
        pipeline_gate_every_steps=4,
        review_steps=8,
        initial_step_ceiling=12,
        cumulative_step_ceiling=20,
        pipeline_validation_seeds=1,
        pipeline_required_wins=2,
    )
    validation_calls = 0

    def passing_validation(_policy: object, **kwargs: object) -> EvaluationResult:
        nonlocal validation_calls
        validation_calls += 1
        seeds = kwargs["seeds"]
        metadata = dict(kwargs["expected_metadata"])
        metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
        return EvaluationResult(
            tuple(
                EvaluationOutcome(seed, seat, 1.0, 10, END_GAME_FINISHED, 1)
                for seed in range(seeds.start, seeds.end)
                for seat in (0, 1)
            ),
            metadata,
            rejected_actions=0,
        )

    real_json_write = archive_module.write_json_exclusive
    marker_writes = 0

    def fail_first_archive_marker(path: Path, value: object) -> None:
        nonlocal marker_writes
        if path.name == "archive.json":
            marker_writes += 1
        if path.name == "archive.json" and marker_writes == 1:
            raise OSError("simulated archive storage failure")
        real_json_write(path, value)

    monkeypatch.setattr(
        archive_module,
        "write_json_exclusive",
        fail_first_archive_marker,
    )
    run_directory = tmp_path / "freeze-recovery-run"
    archive_root = tmp_path / "freeze-recovery-archive"
    with pytest.raises(OSError, match="simulated archive storage failure"):
        run_training(
            config,
            run_directory,
            command=env_command,
            archive_root=archive_root,
            validation_runner=passing_validation,
        )

    checkpoint_path = run_directory / "checkpoints" / "update-000002.pt"
    assert checkpoint_path.is_file()
    assert not (run_directory / "release-000000008.json").exists()

    resumed = run_training(
        config,
        run_directory,
        command=env_command,
        archive_root=archive_root,
        resume_checkpoint=checkpoint_path,
        validation_runner=passing_validation,
    )

    assert resumed.reason == "release readiness gate passed"
    assert resumed.environment_steps == 8
    assert resumed.completed_updates == 2
    assert validation_calls == 2
    assert marker_writes == 2
    evidence = json.loads((run_directory / "release-000000008.json").read_text())
    assert evidence["archive"]["checkpoint_sha256"] == evidence["checkpoint_sha256"]
    assert (archive_root / evidence["archive"]["identifier"] / "archive.json").is_file()


def test_resume_rejects_metrics_ahead_of_the_checkpoint(
    env_command: tuple[str, ...],
    tmp_path: Path,
) -> None:
    """Prevent a checkpoint from continuing on top of another metrics history.

    After a real one-update run, this test appends a syntactically valid row for
    update 2 without creating the matching checkpoint. Resume must compare the
    CSV's reconstructed last update with checkpoint update 1 and stop before it
    starts Go or appends a third history.
    """

    config = _test_config(
        environment_count=1,
        rollout_steps=1,
        minibatch_size=1,
        ppo_epochs=1,
    )
    run_directory = tmp_path / "mixed-history"
    run_training(config, run_directory, command=env_command, max_updates=1)
    metrics_path = run_directory / "metrics.csv"
    with metrics_path.open(newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        first_row = next(reader)
    assert fieldnames is not None
    second_row = dict(first_row, update="2", environment_steps="2")
    with metrics_path.open("a", newline="") as destination:
        csv.DictWriter(destination, fieldnames=fieldnames).writerow(second_row)

    with pytest.raises(
        ValueError,
        match="metrics end at update 2, but checkpoint is update 1",
    ):
        run_training(
            config,
            run_directory,
            command=env_command,
            max_updates=2,
            resume_checkpoint=(
                run_directory / "checkpoints" / "update-000001.pt"
            ),
        )


def test_resume_at_review_boundary_requires_a_reason(
    env_command: tuple[str, ...],
    tmp_path: Path,
) -> None:
    """Require the observed-curve decision before resuming initial training.

    A one-transition update reaches the test's review boundary and pauses. Its
    checkpoint still has room below the initial ceiling, so resuming it without
    a written reason must fail before another rollout can change the policy.
    """

    config = _test_config(
        environment_count=1,
        rollout_steps=1,
        minibatch_size=1,
        ppo_epochs=1,
        pipeline_gate_every_steps=2,
        review_steps=1,
        initial_step_ceiling=3,
        cumulative_step_ceiling=4,
    )
    run_directory = tmp_path / "review-resume"
    first = run_training(
        config,
        run_directory,
        command=env_command,
        max_updates=1,
    )

    assert first.reason == "review required"
    with pytest.raises(ValueError, match="requires --review-reason"):
        run_training(
            config,
            run_directory,
            command=env_command,
            max_updates=2,
            resume_checkpoint=(
                run_directory / "checkpoints" / "update-000001.pt"
            ),
        )


def test_training_refuses_to_reuse_a_run_directory(tmp_path: Path) -> None:
    """Keep artifacts from separate experiments from being mixed together.

    A run directory identifies one immutable configuration and its resulting
    metrics and checkpoints. This test creates that directory in advance and
    verifies that a fresh training invocation refuses to reuse it. Without the
    check, a new run could append misleading metrics or replace prior evidence.
    """

    run_directory = tmp_path / "existing"
    run_directory.mkdir()
    config = _test_config()

    with pytest.raises(FileExistsError):
        run_training(config, run_directory, max_updates=1)


def test_budget_and_review_decisions_happen_only_at_complete_update_boundaries() -> None:
    """Make stage decisions only where no rollout is partially processed.

    A 10-transition batch fits when 40 of 50 steps have been used, but it does
    not fit when 41 have been used. The remaining assertions show that reaching
    25M steps pauses for review unless a continuation reason was recorded, and
    that a successful pipeline gate stops immediately. This prevents truncated
    PPO batches and preserves the meaning of the training budget.
    """

    config = load_run_config(CONFIG)

    assert can_collect_full_rollout(40, 10, 50)
    assert not can_collect_full_rollout(41, 10, 50)
    assert initial_stage_decision(
        config,
        config.review_steps,
        review_reason=None,
    ) == "review required"
    assert initial_stage_decision(
        config,
        config.review_steps,
        review_reason="win rate is still rising",
    ) is None


def test_pipeline_pass_continues_to_release_readiness_and_stops_on_success(
    env_command: tuple[str, ...],
    tmp_path: Path,
) -> None:
    """Move from the initial gate into the final training phase.

    The small configuration reaches a gate after one four-transition update.
    A controlled result passes the two-game pipeline sample against random-v1.
    Training then performs one more update and evaluates 1,000 complete games
    against heuristic-v1. Passing that harder gate stops the run at step 8.
    Neither validation sample increases the training-step counter.
    """

    config = _test_config(
        environment_count=2,
        rollout_steps=2,
        minibatch_size=4,
        ppo_epochs=1,
        checkpoint_every_updates=50,
        pipeline_gate_every_steps=4,
        review_steps=8,
        initial_step_ceiling=12,
        cumulative_step_ceiling=20,
        pipeline_validation_seeds=1,
        pipeline_required_wins=2,
    )
    calls: list[dict[str, object]] = []

    def passing_validation(_policy: object, **kwargs: object) -> EvaluationResult:
        calls.append(kwargs)
        seeds = kwargs["seeds"]
        outcomes = tuple(
            EvaluationOutcome(seed, seat, 1.0, 10, END_GAME_FINISHED, 1)
            for seed in range(seeds.start, seeds.end)
            for seat in (0, 1)
        )
        metadata = dict(kwargs["expected_metadata"])
        metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
        return EvaluationResult(outcomes, metadata, rejected_actions=0)

    archive_root = tmp_path / "frozen-archives"
    result = run_training(
        config,
        tmp_path / "passing-run",
        command=env_command,
        archive_root=archive_root,
        validation_runner=passing_validation,
    )

    assert result.environment_steps == 8
    assert result.completed_updates == 2
    assert result.reason == "release readiness gate passed"
    assert len(calls) == 2
    # Each gate saves a checkpoint containing its validation history. The
    # separate timing rows must attribute both costs to the update that caused
    # them without altering the training metrics schema.
    with (result.run_directory / "timings.csv").open(newline="") as source:
        timing_rows = list(csv.DictReader(source))
    assert [row["update"] for row in timing_rows] == ["1", "2"]
    assert all(float(row["validation_seconds"]) > 0.0 for row in timing_rows)
    assert all(float(row["checkpoint_seconds"]) > 0.0 for row in timing_rows)
    assert all(row["profiling_enabled"] == "False" for row in timing_rows)
    assert calls[0]["seeds"].start == config.validation_seeds.start
    assert calls[0]["seeds"].end == config.validation_seeds.start + 1
    assert calls[0]["opponent"] == config.pipeline_opponent
    assert calls[1]["seeds"].start == config.validation_seeds.start
    assert calls[1]["seeds"].end == config.validation_seeds.start + 500
    assert calls[1]["opponent"] == config.training_opponent
    pipeline_evidence = json.loads(
        (result.run_directory / "pipeline-000000004.json").read_text()
    )
    release_evidence = json.loads(
        (result.run_directory / "release-000000008.json").read_text()
    )
    assert pipeline_evidence["gate"] == "pipeline"
    assert pipeline_evidence["passed"] is True
    assert pipeline_evidence["wins"] == 2
    assert release_evidence["gate"] == "release-readiness"
    assert release_evidence["passed"] is True
    assert release_evidence["wins"] == 1_000
    assert release_evidence["confidence_interval"]["lower_bound"] > 0.85
    assert release_evidence["confidence_interval"]["nominal"] is True
    assert release_evidence["confidence_interval"]["assumption"] == (
        NOMINAL_INTERVAL_ASSUMPTION
    )
    assert release_evidence["requirements"] == {
        "validation_seeds": 500,
        "expected_games": 1_000,
        "required_wins": 900,
        "confidence_level": 0.95,
        "minimum_lower_bound": 0.85,
        "required_rejected_actions": 0,
        "required_results_per_seed_and_seat": 1,
    }
    archive_reference = release_evidence["archive"]
    archive_directory = archive_root / archive_reference["identifier"]
    archive_manifest = json.loads((archive_directory / "archive.json").read_text())
    archived_validation = json.loads(
        (archive_directory / "release-validation.json").read_text()
    )
    archived_checkpoint = archive_directory / archive_manifest["checkpoint"]
    source_checkpoint = (
        result.run_directory / "checkpoints" / release_evidence["checkpoint"]
    )

    # The archive contains independent checkpoint bytes and every adjacent file
    # needed to verify their origin. Nothing here points back to a required file
    # in the mutable training run.
    assert set(path.name for path in archive_directory.iterdir()) == {
        "archive.json",
        "release-validation.json",
        "run-config.json",
        archived_checkpoint.name,
        f"{archived_checkpoint.name}.sha256",
        f"{archived_checkpoint.name}.metadata.json",
    }
    assert not archived_checkpoint.is_symlink()
    assert archived_checkpoint.read_bytes() == source_checkpoint.read_bytes()
    assert sha256_file(archived_checkpoint) == archive_reference["checkpoint_sha256"]
    assert load_checkpoint(archived_checkpoint, expected_config=config).completed_updates == 2
    assert archive_manifest["identifier"] == archive_reference["identifier"]
    assert archive_manifest["checkpoint_sha256"] == release_evidence[
        "checkpoint_sha256"
    ]
    assert archived_validation["archive"] == archive_reference
    assert archived_validation["passed"] is True
    assert json.loads((archive_directory / "run-config.json").read_text()) == (
        json.loads(json.dumps(config.to_dict()))
    )

    # Repeating a completed freeze is an idempotent recovery operation. It
    # verifies and returns the existing archive without replacing any bytes.
    repeated = freeze_passing_checkpoint(
        source_checkpoint,
        archive_root,
        fingerprint=release_evidence["checkpoint_sha256"],
        config=config,
        validation_record=release_evidence,
        completed_updates=2,
    )
    assert repeated.directory == archive_directory
    assert repeated.checkpoint == archived_checkpoint
    assert sha256_file(archived_checkpoint) == archive_reference["checkpoint_sha256"]

    # Caller-supplied counters are checked against the checkpoint before a new
    # final identifier can be created. Stale orchestration data therefore fails
    # while rejection is still harmless.
    mismatched_steps = dict(
        release_evidence,
        training_environment_steps=release_evidence["training_environment_steps"] + 4,
    )
    mismatched_root = tmp_path / "mismatched-counter-archive"
    with pytest.raises(ValueError, match="training steps do not match"):
        freeze_passing_checkpoint(
            source_checkpoint,
            mismatched_root,
            fingerprint=release_evidence["checkpoint_sha256"],
            config=config,
            validation_record=mismatched_steps,
            completed_updates=2,
        )
    assert not mismatched_root.exists()

    with pytest.raises(ValueError, match="completed_updates does not match"):
        freeze_passing_checkpoint(
            source_checkpoint,
            mismatched_root,
            fingerprint=release_evidence["checkpoint_sha256"],
            config=config,
            validation_record=release_evidence,
            completed_updates=3,
        )
    assert not mismatched_root.exists()

    failed_record = dict(release_evidence, passed=False)
    with pytest.raises(ValueError, match="only a passing release-readiness"):
        freeze_passing_checkpoint(
            source_checkpoint,
            tmp_path / "must-not-freeze",
            fingerprint=release_evidence["checkpoint_sha256"],
            config=config,
            validation_record=failed_record,
            completed_updates=2,
        )
    assert not (tmp_path / "must-not-freeze").exists()


def test_release_readiness_ceiling_records_failure_without_spending_a_pool(
    env_command: tuple[str, ...],
    tmp_path: Path,
) -> None:
    """Validate final weights and stop before a rollout crosses the ceiling.

    Four-transition updates pass the pipeline gate at step 8. The next update
    ends at step 12, and another full batch would cross the artificial 15-step
    cumulative ceiling. The trainer therefore runs one final heuristic check at
    step 12, records its failure, and stops without creating evaluation-pool
    state. This is the small-config equivalent of the 500M production limit.
    """

    config = _test_config(
        environment_count=2,
        rollout_steps=2,
        minibatch_size=4,
        ppo_epochs=1,
        checkpoint_every_updates=50,
        pipeline_gate_every_steps=8,
        review_steps=4,
        initial_step_ceiling=12,
        cumulative_step_ceiling=15,
        pipeline_validation_seeds=1,
        pipeline_required_wins=2,
    )
    calls: list[dict[str, object]] = []

    def controlled_validation(_policy: object, **kwargs: object) -> EvaluationResult:
        calls.append(kwargs)
        seeds = kwargs["seeds"]
        reward = 1.0 if kwargs["opponent"] == config.pipeline_opponent else -1.0
        outcomes = tuple(
            EvaluationOutcome(seed, seat, reward, 10, END_GAME_FINISHED, 1)
            for seed in range(seeds.start, seeds.end)
            for seat in (0, 1)
        )
        metadata = dict(kwargs["expected_metadata"])
        metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
        return EvaluationResult(outcomes, metadata, rejected_actions=0)

    result = run_training(
        config,
        tmp_path / "release-ceiling-run",
        command=env_command,
        review_reason="the small test intentionally reaches its next gate",
        validation_runner=controlled_validation,
    )

    assert result.environment_steps == 12
    assert result.completed_updates == 3
    assert result.reason == "release readiness ceiling reached"
    assert [call["opponent"] for call in calls] == [
        config.pipeline_opponent,
        config.training_opponent,
    ]
    release_evidence = json.loads(
        (result.run_directory / "release-000000012.json").read_text()
    )
    failure = json.loads(
        (result.run_directory / "release-readiness-failed.json").read_text()
    )
    assert release_evidence["passed"] is False
    assert release_evidence["wins"] == 0
    assert failure["environment_steps"] == 12
    assert failure["cumulative_step_ceiling"] == 15
    assert not list(result.run_directory.glob("evaluation-pools*.json"))


def test_resumed_ceiling_failure_references_the_existing_release_record(
    env_command: tuple[str, ...],
    tmp_path: Path,
) -> None:
    """Keep a crossed cadence boundary distinct from its trained step count.

    Six-transition updates cross the gates scheduled for steps 8 and 16 at
    actual training steps 12 and 18. The first invocation fails release
    readiness at the 16-step boundary. Resuming its checkpoint cannot fit
    another update, so it must reuse that result and name release-000000016.json
    rather than inventing a file from the actual 18-step counter.
    """

    config = _test_config(
        environment_count=2,
        rollout_steps=3,
        minibatch_size=6,
        ppo_epochs=1,
        checkpoint_every_updates=50,
        pipeline_gate_every_steps=8,
        review_steps=7,
        initial_step_ceiling=20,
        cumulative_step_ceiling=23,
        pipeline_validation_seeds=1,
        pipeline_required_wins=2,
        release_validation_seeds=1,
        release_required_wins=2,
        release_minimum_lower_bound=0.1,
    )
    calls = 0

    def controlled_validation(_policy: object, **kwargs: object) -> EvaluationResult:
        nonlocal calls
        calls += 1
        seeds = kwargs["seeds"]
        reward = 1.0 if kwargs["opponent"] == config.pipeline_opponent else -1.0
        outcomes = tuple(
            EvaluationOutcome(seed, seat, reward, 10, END_GAME_FINISHED, 1)
            for seed in range(seeds.start, seeds.end)
            for seat in (0, 1)
        )
        metadata = dict(kwargs["expected_metadata"])
        metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
        return EvaluationResult(outcomes, metadata, rejected_actions=0)

    run_directory = tmp_path / "crossed-release-boundary"
    first = run_training(
        config,
        run_directory,
        command=env_command,
        validation_runner=controlled_validation,
    )
    release_record = json.loads(
        (run_directory / "release-000000016.json").read_text()
    )
    checkpoint_path = run_directory / "checkpoints" / release_record["checkpoint"]

    resumed = run_training(
        config,
        run_directory,
        command=env_command,
        resume_checkpoint=checkpoint_path,
        validation_runner=controlled_validation,
        allow_development_engine=_TEST_BINARY_ENGINE_MAY_BE_DIRTY,
    )
    failure = json.loads(
        (run_directory / "release-readiness-failed.json").read_text()
    )

    assert first.environment_steps == resumed.environment_steps == 18
    assert calls == 2
    assert failure["last_validation_boundary"] == 16
    assert failure["last_validation"] == "release-000000016.json"
    assert (run_directory / failure["last_validation"]).is_file()


def test_stage_exit_validates_weights_trained_after_the_last_periodic_gate(
    env_command: tuple[str, ...],
    tmp_path: Path,
) -> None:
    """Judge the final complete update when the ceiling is between gate boundaries.

    Four-transition updates fit at steps 4, 8, and 12 under a 15-step ceiling.
    The ordinary eight-step gate evaluates the second update. The third update
    then changes the weights, but a fourth batch cannot fit. A final gate at
    step 12 proves those newest weights are measured before the run is declared
    failed, and it uses the same evidence-file shape as the periodic gate.
    """

    config = _test_config(
        environment_count=2,
        rollout_steps=2,
        minibatch_size=4,
        ppo_epochs=1,
        # The third update is also a periodic checkpoint boundary. Validation
        # must run first so that its checkpoint includes the final gate evidence.
        checkpoint_every_updates=3,
        pipeline_gate_every_steps=8,
        review_steps=5,
        initial_step_ceiling=15,
        cumulative_step_ceiling=20,
        pipeline_validation_seeds=1,
        pipeline_required_wins=2,
    )
    calls = 0

    def failing_validation(_policy: object, **kwargs: object) -> EvaluationResult:
        nonlocal calls
        calls += 1
        seed = config.validation_seeds.start
        metadata = dict(kwargs["expected_metadata"])
        metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
        return EvaluationResult(
            outcomes=(
                EvaluationOutcome(seed, 0, -1.0, 10, END_GAME_FINISHED, 1),
                EvaluationOutcome(seed, 1, -1.0, 11, END_GAME_FINISHED, 1),
            ),
            metadata=metadata,
            rejected_actions=0,
        )

    result = run_training(
        config,
        tmp_path / "ceiling-run",
        command=env_command,
        review_reason="continue after reviewing the observed learning curve",
        validation_runner=failing_validation,
    )

    assert result.environment_steps == 12
    assert result.completed_updates == 3
    assert result.reason == "initial pipeline gate failed"
    assert calls == 2

    periodic = json.loads(
        (result.run_directory / "pipeline-000000008.json").read_text()
    )
    final = json.loads(
        (result.run_directory / "pipeline-000000012.json").read_text()
    )
    assert periodic["training_environment_steps"] == 8
    assert final["training_environment_steps"] == 12
    assert final["validation_boundary"] == 12
    assert final["passed"] is False
    assert final["failed_checks"] == ["at least 2 wins"]
    assert set(final) == set(periodic)
    checkpoint = load_checkpoint(
        result.run_directory / "checkpoints" / final["checkpoint"],
        expected_config=config,
    )
    assert len(checkpoint.validation_state["history"]) == 2
    assert checkpoint.validation_state["history"][-1][
        "training_environment_steps"
    ] == 12

    # The 15-step ceiling is not divisible by the four-transition batch. The
    # timing row for update 3 must still contain the final validation and its
    # checkpoint work instead of dropping those costs at the loop boundary.
    with (result.run_directory / "timings.csv").open(newline="") as source:
        timing_rows = list(csv.DictReader(source))
    assert [row["update"] for row in timing_rows] == ["1", "2", "3"]
    assert float(timing_rows[-1]["validation_seconds"]) > 0.0
    assert float(timing_rows[-1]["checkpoint_seconds"]) > 0.0


def test_four_player_run_validates_every_seat_and_resumes_into_release_readiness(
    env_command: tuple[str, ...],
    tmp_path: Path,
) -> None:
    """Validate a four-player run at each gate, across a resume.

    Training plays real four-player maps. The first invocation passes the
    pipeline gate after one four-transition update and stops at its update
    limit; the gate must move the next boundary forward before that save, so
    the resume is accepted. The resumed update then reaches release readiness.
    Each gate plays every seed from all four seats: 100 seeds make 400
    pipeline games against random-v1, and 500 make 2,000 release games
    against heuristic-v1. One elimination per sample checks that the saved
    record keeps each game's end reason.
    """

    config = _test_config(
        players=4,
        generator_version="generator-v2-4p",
        environment_count=2,
        rollout_steps=2,
        minibatch_size=4,
        ppo_epochs=1,
        pipeline_gate_every_steps=4,
        review_steps=8,
        initial_step_ceiling=12,
        cumulative_step_ceiling=20,
        pipeline_validation_seeds=100,
        pipeline_required_wins=320,
        release_required_wins=1_800,
    )
    calls: list[dict[str, object]] = []

    def passing_validation(_policy: object, **kwargs: object) -> EvaluationResult:
        calls.append(kwargs)
        seeds = kwargs["seeds"]
        outcomes = [
            EvaluationOutcome(seed, seat, 1.0, 10, END_GAME_FINISHED, 1)
            for seed in range(seeds.start, seeds.end)
            for seat in range(kwargs["players"])
        ]
        outcomes[-1] = EvaluationOutcome(seeds.end - 1, 3, -1.0, 4, END_LEARNER_ELIMINATED, 1)
        metadata = dict(kwargs["expected_metadata"])
        metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
        return EvaluationResult(tuple(outcomes), metadata, rejected_actions=0)

    run_directory = tmp_path / "four-player"
    archive_root = tmp_path / "four-player-archive"
    first = run_training(
        config,
        run_directory,
        command=env_command,
        max_updates=1,
        archive_root=archive_root,
        validation_runner=passing_validation,
    )

    assert first.reason == "requested update limit"
    pipeline_record = json.loads((run_directory / "pipeline-000000004.json").read_text())
    assert pipeline_record["passed"] is True
    assert (pipeline_record["games"], pipeline_record["wins"]) == (400, 399)
    assert pipeline_record["outcomes"][-1] == {
        "seed": config.validation_seeds.start + 99,
        "seat": 3,
        "reward": -1.0,
        "turns": 4,
        "end_reason": END_LEARNER_ELIMINATED,
        "launches": 1,
    }
    # The one eliminated seat appears in the summary saved beside the score.
    assert pipeline_record["summary"]["eliminations"] == 1
    assert pipeline_record["summary"]["zero_launch_games"] == 0
    checkpoint_path = run_directory / "checkpoints" / pipeline_record["checkpoint"]
    checkpoint = load_checkpoint(checkpoint_path, expected_config=config)
    assert (checkpoint.environment_steps, checkpoint.next_validation_step) == (4, 8)

    resumed = run_training(
        config,
        run_directory,
        command=env_command,
        max_updates=2,
        archive_root=archive_root,
        resume_checkpoint=checkpoint_path,
        validation_runner=passing_validation,
        allow_development_engine=_TEST_BINARY_ENGINE_MAY_BE_DIRTY,
    )

    assert resumed.reason == "release readiness gate passed"
    assert [call["opponent"] for call in calls] == [
        config.pipeline_opponent,
        config.training_opponent,
    ]
    assert [call["players"] for call in calls] == [4, 4]
    release_record = json.loads((run_directory / "release-000000008.json").read_text())
    assert (release_record["games"], release_record["wins"]) == (2_000, 1_999)
    assert release_record["requirements"]["expected_games"] == 2_000
    assert release_record["requirements"]["required_wins"] == 1_800
    assert release_record["outcomes"][-1]["end_reason"] == END_LEARNER_ELIMINATED


def test_resume_refuses_an_overdue_validation_boundary_before_starting(
    tmp_path: Path,
) -> None:
    """Refuse a checkpoint saved at or past its next validation boundary.

    The current loop always validates a due boundary before saving, so only an
    older lifecycle could leave one behind. The checkpoint here is written
    directly at step 8 with its boundary still at 8. The resume must fail
    before Go starts (the command is not runnable) and before any file in the
    run directory is created or changed.
    """

    config = _test_config(pipeline_gate_every_steps=8)
    run_directory = tmp_path / "overdue"
    (run_directory / "checkpoints").mkdir(parents=True)
    (run_directory / "run-config.json").write_text(json.dumps(config.to_dict()))
    (run_directory / "manifest.json").write_text(
        json.dumps({"format": "run-manifest-v1", "checkpoints": []})
    )
    policy = OrbitPolicy()
    checkpoint_path, _fingerprint = training._save_training_checkpoint(
        run_directory,
        config,
        policy,
        torch.optim.AdamW(policy.parameters()),
        schedule_values(config, 8),
        torch.Generator().manual_seed(1),
        torch.Generator().manual_seed(2),
        {},
        8,
        2,
        8,
        [],
        training.EpisodeMetricTracker(()),
        (),
        0,
        {},
        purpose="stopped",
    )
    before = {
        path: path.read_bytes() for path in sorted(run_directory.rglob("*")) if path.is_file()
    }

    def forbidden_validation(*_args: object, **_kwargs: object) -> EvaluationResult:
        pytest.fail("validation started")

    with pytest.raises(ValueError, match="overdue validation boundary; start a new run"):
        run_training(
            config,
            run_directory,
            command=("go-must-not-start",),
            resume_checkpoint=checkpoint_path,
            validation_runner=forbidden_validation,
        )

    after = {
        path: path.read_bytes() for path in sorted(run_directory.rglob("*")) if path.is_file()
    }
    assert after == before


def test_pipeline_validation_checks_the_full_go_metadata_contract() -> None:
    """Reject a pipeline result that omits its rotate-schedule identity.

    The initial gate unlocks the larger training budget, so it must verify the
    same Go evidence as release readiness. The injected runner returns complete
    games but removes `schedule` from its response. The trainer must reject the
    result even though the score itself would pass.
    """

    config = _test_config(
        pipeline_validation_seeds=1,
        pipeline_required_wins=2,
    )
    captured: dict[str, object] = {}

    def missing_schedule(_policy: object, **kwargs: object) -> EvaluationResult:
        captured.update(kwargs)
        seeds = kwargs["seeds"]
        metadata = dict(kwargs["expected_metadata"])
        metadata.pop("schedule")
        metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
        return EvaluationResult(
            tuple(
                EvaluationOutcome(seeds.start, seat, 1.0, 10, END_GAME_FINISHED, 1)
                for seat in (0, 1)
            ),
            metadata,
            rejected_actions=0,
        )

    with pytest.raises(ValueError, match="pipeline validation metadata 'schedule'"):
        training._run_pipeline_validation(
            torch.nn.Linear(1, 1),
            config,
            ("unused",),
            missing_schedule,
        )

    expected_metadata = captured["expected_metadata"]
    assert captured["players"] == 2
    assert expected_metadata["v"] == "env-v3"
    assert expected_metadata["schedule"] == "rotate"
    assert expected_metadata["seed_range"] == [1_000_000, 1_000_001]


def test_release_validation_uses_the_training_player_count() -> None:
    """Run release-readiness games on the training run's maps.

    The runner is replaced because this test needs only to observe its inputs.
    Returning the complete every-seat result also proves the argument passes
    through without weakening the existing coverage and metadata checks.
    """

    config = _test_config(
        release_validation_seeds=1,
        release_required_wins=2,
    )
    captured: dict[str, object] = {}
    timings = RunnerTimings(3.0, 1.0, 2.0)

    def complete_result(_policy: object, **kwargs: object) -> EvaluationResult:
        captured.update(kwargs)
        seeds = kwargs["seeds"]
        metadata = dict(kwargs["expected_metadata"])
        metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
        return EvaluationResult(
            tuple(
                EvaluationOutcome(seeds.start, seat, 1.0, 10, END_GAME_FINISHED, 1)
                for seat in (0, 1)
            ),
            metadata,
            rejected_actions=0,
            timings=timings,
        )

    def on_progress(_report: object) -> None:
        return None

    result = training._run_release_readiness_validation(
        torch.nn.Linear(1, 1),
        config,
        ("unused",),
        complete_result,
        on_progress=on_progress,
    )

    assert len(result.outcomes) == 2
    assert captured["players"] == 2
    # Training's progress printer reaches the runner, and the timings survive
    # the completeness check that rebuilds the result.
    assert captured["on_progress"] is on_progress
    assert result.timings is timings
    # Validation runs the configured evaluator on the policy's own device.
    assert captured["evaluation_policy"] == "eval-policy-v2"
    assert captured["device"] == torch.device("cpu")


def test_pipeline_validation_rechecks_seed_and_seat_completeness() -> None:
    """Prevent a duplicate pipeline game from replacing a required seat result."""

    config = _test_config(
        pipeline_validation_seeds=1,
        pipeline_required_wins=2,
    )

    def duplicate_seat(_policy: object, **kwargs: object) -> EvaluationResult:
        seeds = kwargs["seeds"]
        metadata = dict(kwargs["expected_metadata"])
        metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
        duplicate = EvaluationOutcome(seeds.start, 0, 1.0, 10, END_GAME_FINISHED, 1)
        return EvaluationResult(
            (duplicate, duplicate),
            metadata,
            rejected_actions=0,
        )

    with pytest.raises(ValueError, match="duplicate evaluation result"):
        training._run_pipeline_validation(
            torch.nn.Linear(1, 1),
            config,
            ("unused",),
            duplicate_seat,
        )


def test_rejected_rollout_stops_before_the_ppo_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stop before optimization when Go reports an illegal submitted action.

    The fake environment increments the same rejection counter used by the real
    protocol client during rollout collection. `ppo_update` is replaced with a
    function that fails the test if called. The expected runtime error therefore
    proves rejected environment data cannot change policy weights or produce a
    normal metrics row.
    """

    config = _test_config(
        environment_count=1,
        rollout_steps=1,
        minibatch_size=1,
        ppo_epochs=1,
    )

    constructed_players: int | None = None

    class RejectingEnvironment:
        """Provide only the environment lifecycle needed to simulate rejection."""

        error_count = 0
        cursor = 0

        def __init__(self, _command: object, **kwargs: object) -> None:
            # The production client also accepts the optional profiler. This
            # rejection fixture ignores it because collection stops before any
            # meaningful phase can complete.
            nonlocal constructed_players
            constructed_players = kwargs.get("players")

        def __enter__(self) -> RejectingEnvironment:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def create(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {
                "v": config.protocol_version,
                "engine": "abc123",
                "schema": "obs-v2",
                "generator": "generator-v2",
                "opponent": config.training_opponent,
                "schedule": "train-v2",
                "seed_range": [0, 1_000_000],
                "dimensions": OBSERVATION_DIMENSIONS,
            }

        def reset(self, _ids: object) -> list[object]:
            return [object()]

    environment: RejectingEnvironment | None = None

    def rejected_collection(client: RejectingEnvironment, *_args: object, **_kwargs: object) -> None:
        nonlocal environment
        environment = client
        client.error_count += 1

    def forbidden_update(*_args: object, **_kwargs: object) -> None:
        pytest.fail("PPO update ran after an environment rejection")

    monkeypatch.setattr(training, "EnvClient", RejectingEnvironment)
    monkeypatch.setattr(training, "collect_rollout", rejected_collection)
    monkeypatch.setattr(training, "ppo_update", forbidden_update)

    with pytest.raises(RuntimeError, match="1 rejected actions"):
        run_training(config, tmp_path / "rejected", max_updates=1)

    assert environment is not None
    assert constructed_players == 2
    assert not (tmp_path / "rejected" / "metrics.csv").exists()


def test_command_loads_the_config_and_starts_the_training_driver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Check the command-line boundary without starting a long training run.

    The test supplies the same flags a user passes to `python -m orrery.train`.
    A lightweight replacement records the arguments instead of collecting a
    full 32,768-transition update. The assertions prove that the TOML file was
    loaded without consulting Git, the requested directory was retained, and
    `--max-updates` reached the training driver as an integer.
    """

    run_directory = tmp_path / "cli-run"
    captured: dict[str, object] = {}

    def fake_run(config: object, destination: object, **kwargs: object) -> TrainingResult:
        captured.update(config=config, destination=destination, kwargs=kwargs)
        return TrainingResult(run_directory, 32_768, 1, "requested update limit")

    monkeypatch.setattr(training, "run_training", fake_run)

    exit_code = training.main(
        [
            "--config",
            str(CONFIG),
            "--run-dir",
            str(run_directory),
            "--max-updates",
            "1",
            "--profile",
        ]
    )

    assert exit_code == 0
    assert captured["config"].contract == "rl-v2"
    assert captured["destination"] == run_directory
    assert captured["kwargs"]["max_updates"] == 1
    assert captured["kwargs"]["profile"] is True


def test_command_derives_run_directory_from_resume_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use the checkpoint's owning run when --resume omits --run-dir."""

    checkpoint = tmp_path / "existing-run" / "checkpoints" / "update-000007.pt"
    captured: dict[str, object] = {}

    def fake_run(config: object, destination: object, **kwargs: object) -> TrainingResult:
        captured.update(config=config, destination=destination, kwargs=kwargs)
        return TrainingResult(checkpoint.parent.parent, 28, 7, "requested update limit")

    monkeypatch.setattr(training, "run_training", fake_run)

    assert training.main(
        ["--config", str(CONFIG), "--resume", str(checkpoint)]
    ) == 0
    assert captured["destination"] == checkpoint.parent.parent
    assert captured["kwargs"]["resume_checkpoint"] == checkpoint
    # Unidentified engine versions are refused unless the option is given.
    assert captured["kwargs"]["allow_development_engine"] is False

    assert training.main(
        ["--config", str(CONFIG), "--resume", str(checkpoint), "--allow-development"]
    ) == 0
    assert captured["kwargs"]["allow_development_engine"] is True


def test_command_refuses_allow_development_without_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--allow-development only changes a resume, so it is refused on its own."""

    monkeypatch.setattr(
        training, "run_training", lambda *_args, **_kwargs: pytest.fail("training started")
    )

    with pytest.raises(SystemExit):
        training.main(["--config", str(CONFIG), "--allow-development"])


_REAL_ENGINE = "v0.1.0+0123456789abcdef0123456789abcdef01234567"


@pytest.mark.parametrize(
    ("saved", "current", "allow", "message"),
    [
        ("development", "development", False, "checkpoint engine version 'development'"),
        (_REAL_ENGINE, "development", False, "current engine version 'development'"),
        ("abc123+dirty", "abc123+dirty", False, "'abc123\\+dirty' does not name exact code"),
        ("development", "development", True, None),
        ("abc123+dirty", "abc123+dirty", True, None),
        (_REAL_ENGINE, _REAL_ENGINE, False, None),
        # Allowing development builds never lets different code through.
        (_REAL_ENGINE, "untagged+fedcba", True, "does not match checkpoint metadata"),
    ],
    ids=[
        "both development", "current development", "dirty", "development allowed",
        "dirty allowed", "real versions match", "allowed but different",
    ],
)
def test_resume_refuses_engine_versions_that_do_not_name_exact_code(
    saved: str, current: str, allow: bool, message: str | None
) -> None:
    """Resume only trusts matching versions when they identify exact code.

    ``development`` and ``+dirty`` versions are refused on either side unless
    explicitly allowed. Allowing them only skips that refusal: two different
    versions are still a mismatch.
    """

    def metadata(engine: str) -> dict[str, object]:
        return {"engine": engine, "cursor": 7}

    def resume() -> None:
        training._validate_resumed_environment(
            metadata(saved), metadata(current), 7, allow_development=allow
        )

    if message is None:
        resume()
    else:
        with pytest.raises(ValueError, match=message):
            resume()
