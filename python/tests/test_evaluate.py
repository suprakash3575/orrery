"""Held-out command and protected-pool integration tests."""

from __future__ import annotations

import inspect
import json
import os
from types import SimpleNamespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import orrery.evaluate as evaluate_module
import orrery.evaluator as evaluator_module
from orrery.archive import FrozenCheckpointArchive, freeze_passing_checkpoint
from orrery.checkpoint import CheckpointState, save_checkpoint, sha256_file
from orrery.config import RunConfig, load_run_config
from orrery.env_client import REPOSITORY_ROOT
from orrery.evaluate import (
    ArchivedCandidate,
    load_archived_candidate,
    main,
    run_held_out_evaluation,
)
from orrery.evaluation_pool import (
    POOL_SPENT_COMPLETE,
    POOL_SPENT_INCOMPLETE,
    read_evaluation_pool_manifest,
)
from orrery.env_client import (
    END_GAME_FINISHED,
    END_LEARNER_ELIMINATED,
    OBSERVATION_DIMENSIONS,
)
from orrery.evaluator import (
    NOMINAL_INTERVAL_ASSUMPTION,
    EvaluationOutcome,
    EvaluationResult,
    run_rotated_matches,
)
from orrery.jsonio import write_json_exclusive
from orrery.model import OrbitPolicy


CONFIG = REPOSITORY_ROOT / "configs" / "rl-v2-2p.toml"
FOUR_PLAYER_CONFIG = REPOSITORY_ROOT / "configs" / "rl-v2-4p.toml"
CHECKPOINT_SHA256 = "a" * 64
# What the real runner records about how it chose actions. Fake runners
# report it too, because evaluation refuses results from another evaluator.
_EVALUATION = {"policy": "eval-policy-v2", "device": "cpu"}


@pytest.fixture
def config() -> RunConfig:
    """Load the exact evaluation partitions used by the command."""

    return load_run_config(CONFIG)


def test_command_requires_explicit_spend_eval_selection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject an evaluation command that could consume a pool accidentally."""

    with pytest.raises(SystemExit) as error:
        main(["--archive", "unused-for-parser-test"])

    assert error.value.code == 2
    assert "--spend-eval" in capsys.readouterr().err


@pytest.mark.parametrize(("passed", "status"), ((True, 0), (False, 1)))
def test_main_returns_a_status_matching_the_completed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    passed: bool,
    status: int,
) -> None:
    """Expose pass and failure through the documented shell exit codes."""

    evaluation = SimpleNamespace(
        decision=SimpleNamespace(passed=passed, wins=1_800, games=2_000),
        claim=SimpleNamespace(pool_name="eval-1"),
        result_path=tmp_path / "held-out.json",
    )
    monkeypatch.setattr(
        evaluate_module,
        "run_held_out_evaluation",
        lambda *_args, **_kwargs: evaluation,
    )

    assert main(["--archive", "archive", "--spend-eval", "eval-1"]) == status
    assert ("passed" if passed else "failed") in capsys.readouterr().out


@pytest.mark.parametrize(
    ("config_path", "players", "games", "required_wins"),
    [(CONFIG, 2, 2_000, 1_800), (FOUR_PLAYER_CONFIG, 4, 4_000, 3_600)],
    ids=["2p", "4p"],
)
def test_runner_observes_spent_manifest_before_returning_games(
    tmp_path: Path,
    config_path: Path,
    players: int,
    games: int,
    required_wins: int,
) -> None:
    """Connect the atomic claim directly to the shared rotated-match boundary.

    The fake runner represents ``run_rotated_matches`` without starting 2,000
    or 4,000 real games. Its first action reads the manifest and proves that
    ``eval-1`` is already unavailable. It then returns a complete passing
    sample, every seed from every seat, so the command can write immutable
    evidence and mark the attempt complete. Its losses are eliminations, so
    the stored outcomes must keep that end reason.
    """

    config = load_run_config(config_path)
    archive_directory = tmp_path / "release-example"
    archive_directory.mkdir()
    checkpoint = archive_directory / "checkpoint.pt"
    manifest_path = tmp_path / "evaluation-pools.json"
    candidate = ArchivedCandidate(
        directory=archive_directory,
        identifier=archive_directory.name,
        checkpoint=checkpoint,
        checkpoint_sha256=CHECKPOINT_SHA256,
        config=config,
        policy=OrbitPolicy().eval(),
        training_environment_steps=10_000_000,
        completed_updates=305,
    )
    runner_called = False

    def passing_runner(_policy: object, **kwargs: object) -> EvaluationResult:
        """Return complete outcomes after checking claim-before-play ordering."""

        nonlocal runner_called
        runner_called = True
        saved = read_evaluation_pool_manifest(manifest_path, config)
        assert saved["pools"][0]["status"] == POOL_SPENT_INCOMPLETE
        seeds = kwargs["seeds"]
        assert (seeds.start, seeds.end) == (1_010_000, 1_011_000)
        assert kwargs["opponent"] == config.training_opponent
        # The archived player count is forwarded, not the runner's default.
        assert kwargs["players"] == config.players
        rewards = (1.0,) * config.held_out_required_wins + (-1.0,) * (
            config.held_out_games - config.held_out_required_wins
        )
        outcomes = tuple(
            EvaluationOutcome(
                seed=seeds.start + index // players,
                seat=index % players,
                reward=reward,
                turns=100,
                end_reason=END_GAME_FINISHED if reward > 0 else END_LEARNER_ELIMINATED,
                launches=1,
            )
            for index, reward in enumerate(rewards)
        )
        metadata = dict(kwargs["expected_metadata"])
        metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
        return EvaluationResult(outcomes, metadata, rejected_actions=0)

    evaluation = run_held_out_evaluation(
        archive_directory,
        spend_eval="eval-1",
        manifest_path=manifest_path,
        evaluation_runner=passing_runner,
        candidate_loader=lambda _path, **_kwargs: candidate,
        attempt_id="attempt-one",
        started_at="2026-09-10T12:00:00Z",
        completed_at="2026-09-10T13:00:00Z",
    )

    assert runner_called
    assert evaluation.decision.passed
    evidence = json.loads(evaluation.result_path.read_text())
    assert evidence["claim"]["pool"] == "eval-1"
    assert evidence["claim"]["status_at_start"] == POOL_SPENT_INCOMPLETE
    assert evidence["pool_manifest_at_start"]["pools"][0]["status"] == (
        POOL_SPENT_INCOMPLETE
    )
    assert evidence["pool_manifest_at_start"]["pools"][1]["status"] == "available"
    assert evidence["checkpoint"]["archive_identifier"] == "release-example"
    assert evidence["checkpoint"]["checkpoint_format"] == config.checkpoint_format
    assert evidence["run_config"] == json.loads(json.dumps(config.to_dict()))
    assert evidence["requirements"]["expected_games"] == games
    assert evidence["requirements"]["required_wins"] == required_wins
    assert evidence["requirements"]["seats_per_seed"] == players
    assert evidence["contract"] == {
        "rl": config.contract,
        "checkpoint": config.checkpoint_format,
        "policy": config.evaluation_policy,
        "opponent": config.training_opponent,
        "schedule": "rotate",
        "protocol": config.protocol_version,
        "observation": config.observation_schema,
        "generator": config.generator_version,
        "engine": "test-engine",
        "players": players,
        "dimensions": OBSERVATION_DIMENSIONS,
        "evaluation": _EVALUATION,
    }
    assert evidence["aggregate"]["wins"] == required_wins
    assert evidence["aggregate"]["losses"] == games - required_wins
    assert evidence["aggregate"]["draws"] == 0
    # The fake runner ends every lost game by elimination, gives every game
    # 100 turns and one launch, so the saved summary must say exactly that.
    # Its wins come first while seats rotate, so every seat wins the same
    # share: 900 of its 1,000 games, on both map sizes.
    assert evidence["summary"] == {
        "games": games,
        "results": {
            "wins": required_wins,
            "losses": games - required_wins,
            "draws": 0,
            "win_rate": required_wins / games,
        },
        "seat_win_rate": [0.9] * players,
        "launches_per_game": {"mean": 1.0, "median": 1, "max": 1},
        "zero_launch_games": 0,
        "episode_turns": {
            "mean": 100.0, "min": 100, "p10": 100, "median": 100, "p90": 100, "max": 100,
        },
        "eliminations": games - required_wins,
    }
    assert evidence["aggregate"]["confidence_interval"]["nominal"] is True
    assert evidence["aggregate"]["confidence_interval"]["assumption"] == (
        NOMINAL_INTERVAL_ASSUMPTION
    )
    assert len(evidence["outcomes"]) == games
    assert evidence["outcomes"][0]["seed"] == 1_010_000
    assert evidence["outcomes"][0]["seat"] == 0
    assert evidence["outcomes"][0]["end_reason"] == END_GAME_FINISHED
    assert evidence["outcomes"][-1]["seed"] == 1_010_999
    assert evidence["outcomes"][-1]["seat"] == players - 1
    assert evidence["outcomes"][-1]["end_reason"] == END_LEARNER_ELIMINATED
    saved = read_evaluation_pool_manifest(manifest_path, config)
    assert saved["pools"][0]["status"] == POOL_SPENT_COMPLETE
    assert saved["pools"][0]["attempt"]["result"]["outcome"] == "passed"
    assert saved["pools"][0]["attempt"]["result"]["sha256"] == sha256_file(
        evaluation.result_path
    )
    # Result publication uses exclusive creation. A later command cannot
    # replace the evidence after its digest has entered the pool manifest.
    with pytest.raises(FileExistsError):
        write_json_exclusive(evaluation.result_path, {"changed": True})


def test_evaluate_reuses_the_shared_decoder_and_rotated_match_runner() -> None:
    """Keep held-out play on the evaluator implementation used by validation.

    ``run_held_out_evaluation`` defaults directly to the imported reusable
    runner. That runner decides each turn with ``_decide_packed``, which
    decodes through the public ``decode_evaluation_targets``. The command
    module must not gain its own EnvClient loop or action-decoding function,
    because two implementations could silently score the same checkpoint
    under different rules.
    """

    parameter = inspect.signature(run_held_out_evaluation).parameters[
        "evaluation_runner"
    ]
    command_source = inspect.getsource(evaluate_module)
    runner_source = inspect.getsource(run_rotated_matches)
    turn_source = inspect.getsource(evaluator_module._decide_packed)

    assert parameter.default is run_rotated_matches
    assert "def decode_evaluation_targets" not in command_source
    assert "EnvClient" not in command_source
    assert "_decide_packed(" in runner_source
    assert "decode_evaluation_targets(" in turn_source


def test_malformed_runner_output_spends_pool_without_completing_it(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Keep a pool unavailable when the runner omits required seed/seat games."""

    archive_directory = tmp_path / "release-example"
    archive_directory.mkdir()
    candidate = ArchivedCandidate(
        directory=archive_directory,
        identifier=archive_directory.name,
        checkpoint=archive_directory / "checkpoint.pt",
        checkpoint_sha256=CHECKPOINT_SHA256,
        config=config,
        policy=OrbitPolicy().eval(),
        training_environment_steps=10_000_000,
        completed_updates=305,
    )
    manifest_path = tmp_path / "evaluation-pools.json"

    def incomplete_runner(_policy: object, **kwargs: object) -> EvaluationResult:
        """Return only one seat, which is invalid for a rotated pool."""

        seeds = kwargs["seeds"]
        metadata = dict(kwargs["expected_metadata"])
        metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
        return EvaluationResult(
            (EvaluationOutcome(seeds.start, 0, 1.0, 50, END_GAME_FINISHED, 1),),
            metadata,
            rejected_actions=0,
        )

    with pytest.raises(ValueError, match="missing evaluation results"):
        run_held_out_evaluation(
            archive_directory,
            spend_eval="eval-1",
            manifest_path=manifest_path,
            evaluation_runner=incomplete_runner,
            candidate_loader=lambda _path, **_kwargs: candidate,
            attempt_id="malformed-attempt",
        )

    saved = read_evaluation_pool_manifest(manifest_path, config)
    assert saved["pools"][0]["status"] == POOL_SPENT_INCOMPLETE
    assert list(archive_directory.glob("held-out-*.json")) == []


def test_duplicate_seed_and_seat_spends_pool_without_writing_a_result(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Reject a repeated game identity instead of hiding a missing rotated game.

    Counting only list length would allow one duplicated outcome to replace a
    missing seat. The held-out command checks identities after the protected
    claim, so malformed coverage consumes the pool but cannot produce official
    result evidence.
    """

    candidate = _synthetic_candidate(tmp_path, config)
    manifest_path = tmp_path / "evaluation-pools.json"

    def duplicate_runner(_policy: object, **kwargs: object) -> EvaluationResult:
        """Return the same seed and seat twice to violate rotated coverage."""

        seeds = kwargs["seeds"]
        repeated = EvaluationOutcome(seeds.start, 0, 1.0, 50, END_GAME_FINISHED, 1)
        metadata = dict(kwargs["expected_metadata"])
        metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
        return EvaluationResult((repeated, repeated), metadata, rejected_actions=0)

    with pytest.raises(ValueError, match="duplicate evaluation result"):
        run_held_out_evaluation(
            candidate.directory,
            spend_eval="eval-1",
            manifest_path=manifest_path,
            evaluation_runner=duplicate_runner,
            candidate_loader=lambda _path, **_kwargs: candidate,
            attempt_id="duplicate-attempt",
        )

    saved = read_evaluation_pool_manifest(manifest_path, config)
    assert saved["pools"][0]["status"] == POOL_SPENT_INCOMPLETE
    assert list(candidate.directory.glob("held-out-*.json")) == []


@pytest.mark.parametrize(
    ("problem", "message"),
    (
        ("result-type", "invalid result type"),
        ("opponent", "metadata 'opponent'"),
        ("engine-missing", "missing the engine version"),
        ("engine-type", "missing the engine version"),
        ("dimensions-missing", "metadata dimensions are None"),
        ("dimensions-obs-v1", "metadata dimensions are"),
        # Games decided under the CPU evaluator's rules are not this
        # archive's evidence, even when every Go field matches.
        ("evaluator-v1", "evaluator is 'eval-policy-v1'"),
    ),
)
def test_invalid_runner_contract_spends_pool_without_official_evidence(
    tmp_path: Path,
    config: RunConfig,
    problem: str,
    message: str,
) -> None:
    """Fail closed when an injected runner violates the shared Go contract."""

    candidate = _synthetic_candidate(tmp_path, config)
    manifest_path = tmp_path / "evaluation-pools.json"

    def invalid_runner(_policy: object, **kwargs: object) -> object:
        if problem == "result-type":
            return object()
        result = _complete_result(
            kwargs,
            wins=config.held_out_required_wins,
            rejected_actions=0,
        )
        metadata = dict(result.metadata)
        if problem == "opponent":
            metadata["opponent"] = "wrong-opponent"
        elif problem == "engine-missing":
            metadata.pop("engine")
        elif problem == "engine-type":
            metadata["engine"] = 7
        elif problem == "dimensions-missing":
            metadata.pop("dimensions")
        elif problem == "evaluator-v1":
            metadata["evaluation"] = {**metadata["evaluation"], "policy": "eval-policy-v1"}
        else:
            # obs-v1's sizes: results from a server built for the old layout.
            metadata["dimensions"] = {
                **OBSERVATION_DIMENSIONS,
                "body_slots": 16,
                "body_features": 16,
            }
        return EvaluationResult(result.outcomes, metadata, result.rejected_actions)

    with pytest.raises(ValueError, match=message):
        run_held_out_evaluation(
            candidate.directory,
            spend_eval="eval-1",
            manifest_path=manifest_path,
            evaluation_runner=invalid_runner,
            candidate_loader=lambda _path, **_kwargs: candidate,
            attempt_id="invalid-contract",
        )

    saved = read_evaluation_pool_manifest(manifest_path, config)
    assert saved["pools"][0]["status"] == POOL_SPENT_INCOMPLETE
    assert list(candidate.directory.glob("held-out-*.json")) == []


@pytest.mark.parametrize("batch_size", (0, -1, True))
def test_invalid_batch_size_is_rejected_before_pool_claim(
    tmp_path: Path,
    config: RunConfig,
    batch_size: object,
) -> None:
    """Reject unusable Go slot counts without spending protected boards."""

    candidate = _synthetic_candidate(tmp_path, config)
    manifest_path = tmp_path / "evaluation-pools.json"

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        run_held_out_evaluation(
            candidate.directory,
            spend_eval="eval-1",
            manifest_path=manifest_path,
            batch_size=batch_size,
            candidate_loader=lambda _path, **_kwargs: candidate,
        )

    assert not manifest_path.exists()


@pytest.mark.parametrize(
    ("wins", "rejected_actions", "failed_checks"),
    (
        (1_799, 0, ("at least 1800 wins",)),
        (1_800, 1, ("zero rejected actions",)),
        (
            1_731,
            0,
            (
                "at least 1800 wins",
                "95% confidence lower bound at least 85%",
            ),
        ),
    ),
)
def test_held_out_gate_rejects_each_failed_quality_requirement(
    tmp_path: Path,
    config: RunConfig,
    wins: int,
    rejected_actions: int,
    failed_checks: tuple[str, ...],
) -> None:
    """Apply the held-out win, confidence, and action-validity thresholds.

    The existing passing integration covers exactly 1,800 wins. These cases
    move below the win threshold, introduce a Go rejection, and move below the
    Clopper-Pearson boundary. Every sample still contains all 2,000 games so a
    score failure cannot be confused with malformed coverage.
    """

    candidate = _synthetic_candidate(tmp_path, config)

    def failing_runner(_policy: object, **kwargs: object) -> EvaluationResult:
        """Return the requested complete score with trustworthy metadata."""

        return _complete_result(kwargs, wins=wins, rejected_actions=rejected_actions)

    evaluation = run_held_out_evaluation(
        candidate.directory,
        spend_eval="eval-1",
        manifest_path=tmp_path / "evaluation-pools.json",
        evaluation_runner=failing_runner,
        candidate_loader=lambda _path, **_kwargs: candidate,
        attempt_id="failed-quality",
    )

    assert not evaluation.decision.passed
    assert evaluation.decision.failed_checks == failed_checks
    saved = read_evaluation_pool_manifest(evaluation.claim.manifest_path, config)
    assert saved["pools"][0]["status"] == POOL_SPENT_COMPLETE
    assert saved["pools"][0]["attempt"]["result"]["outcome"] == "failed"


def test_archived_candidate_loader_accepts_a_self_contained_passing_archive(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Load only the exact checkpoint selected and copied by release readiness."""

    archive = _passing_archive(tmp_path, config)

    # The archived config trains on CUDA; this machine may have none, so the
    # test asks for the CPU explicitly.
    candidate = load_archived_candidate(archive.directory, device="cpu")

    assert next(candidate.policy.parameters()).device.type == "cpu"
    assert candidate.identifier == archive.identifier
    assert candidate.checkpoint == archive.checkpoint
    assert candidate.checkpoint_sha256 == archive.sha256
    assert candidate.config == config
    assert candidate.training_environment_steps == 32_768
    assert candidate.completed_updates == 1


def test_relative_archive_records_an_absolute_result_path(
    tmp_path: Path,
    config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the permanent pool ledger meaningful outside the launch directory.

    The documented command may name ``../runs/archive/...`` after changing
    into ``python``. Resolving that archive before deriving the result filename
    ensures the ledger stores a stable absolute path instead of shell context.
    """

    archive = _passing_archive(tmp_path, config)
    working_directory = tmp_path / "working"
    working_directory.mkdir()
    relative_archive = Path(os.path.relpath(archive.directory, working_directory))
    manifest_path = tmp_path / "evaluation-pools.json"
    monkeypatch.chdir(working_directory)

    def passing_runner(_policy: object, **kwargs: object) -> EvaluationResult:
        return _complete_result(
            kwargs,
            wins=config.held_out_required_wins,
            rejected_actions=0,
        )

    evaluation = run_held_out_evaluation(
        relative_archive,
        spend_eval="eval-1",
        manifest_path=manifest_path,
        device="cpu",
        evaluation_runner=passing_runner,
    )

    saved = read_evaluation_pool_manifest(manifest_path, config)
    result_file = Path(saved["pools"][0]["attempt"]["result"]["file"])
    assert evaluation.result_path.is_absolute()
    assert result_file.is_absolute()
    assert result_file == evaluation.result_path


def test_model_copy_failure_is_refused_before_spending_a_pool(
    tmp_path: Path,
    config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running out of memory while copying the model leaves the pool unspent.

    The evaluator's private model copy is a fresh allocation on the GPU. It is
    made before the pool is claimed, so a failure there, such as CUDA running
    out of memory, must leave no manifest behind and start no game.
    """

    candidate = _synthetic_candidate(tmp_path, config)
    manifest_path = tmp_path / "evaluation-pools.json"

    def out_of_memory(_value: object) -> object:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(evaluator_module, "copy", SimpleNamespace(deepcopy=out_of_memory))

    with pytest.raises(RuntimeError, match="out of memory"):
        run_held_out_evaluation(
            candidate.directory,
            spend_eval="eval-1",
            manifest_path=manifest_path,
            evaluation_runner=lambda *_args, **_kwargs: pytest.fail("games started"),
            candidate_loader=lambda _path, **_kwargs: candidate,
        )

    assert not manifest_path.exists()


def test_unavailable_cuda_is_refused_before_spending_a_pool(
    tmp_path: Path,
    config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing GPU stops held-out evaluation before the pool is claimed.

    The archived config asks for CUDA. With CUDA reported unavailable, the
    command must fail while loading the model, never fall back to the CPU,
    and leave the protected pool unspent for a correctly set-up machine.
    """

    archive = _passing_archive(tmp_path, config)
    manifest_path = tmp_path / "evaluation-pools.json"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA is not available"):
        run_held_out_evaluation(
            archive.directory,
            spend_eval="eval-1",
            manifest_path=manifest_path,
            evaluation_runner=lambda *_args, **_kwargs: pytest.fail("games started"),
        )

    assert not manifest_path.exists()


def test_failed_release_archive_is_rejected_before_spending_a_pool(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Keep protected seeds untouched when release readiness did not pass."""

    archive = _passing_archive(tmp_path, config)
    release_path = archive.directory / "release-validation.json"
    release = json.loads(release_path.read_text())
    release["passed"] = False
    release_path.write_text(json.dumps(release))
    manifest_path = tmp_path / "evaluation-pools.json"

    with pytest.raises(ValueError, match="lacks passing release-readiness evidence"):
        run_held_out_evaluation(
            archive.directory,
            spend_eval="eval-1",
            manifest_path=manifest_path,
        )

    assert not manifest_path.exists()


def test_changed_archived_checkpoint_is_rejected_before_spending_a_pool(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Re-hash the frozen bytes before exposing held-out boards to the model."""

    archive = _passing_archive(tmp_path, config)
    with archive.checkpoint.open("r+b") as output:
        output.seek(-1, 2)
        final_byte = output.read(1)
        output.seek(-1, 2)
        output.write(bytes([final_byte[0] ^ 1]))
    manifest_path = tmp_path / "evaluation-pools.json"

    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        run_held_out_evaluation(
            archive.directory,
            spend_eval="eval-1",
            manifest_path=manifest_path,
        )

    assert not manifest_path.exists()


def test_changed_archive_training_boundary_is_rejected_before_pool_spend(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Bind evaluation to the completed update selected by release readiness."""

    archive = _passing_archive(tmp_path, config)
    archive_path = archive.directory / "archive.json"
    manifest = json.loads(archive_path.read_text())
    manifest["training_environment_steps"] += config.batch_size
    archive_path.write_text(json.dumps(manifest))
    pool_path = tmp_path / "evaluation-pools.json"

    with pytest.raises(ValueError, match="training boundary disagrees"):
        run_held_out_evaluation(
            archive.directory,
            spend_eval="eval-1",
            manifest_path=pool_path,
        )

    assert not pool_path.exists()


def _synthetic_candidate(tmp_path: Path, config: RunConfig) -> ArchivedCandidate:
    """Create a lightweight candidate for orchestration tests with a fake runner."""

    directory = tmp_path / "release-example"
    directory.mkdir()
    return ArchivedCandidate(
        directory=directory,
        identifier=directory.name,
        checkpoint=directory / "checkpoint.pt",
        checkpoint_sha256=CHECKPOINT_SHA256,
        config=config,
        policy=OrbitPolicy().eval(),
        training_environment_steps=10_000_000,
        completed_updates=305,
    )


def _complete_result(
    runner_arguments: dict[str, object],
    *,
    wins: int,
    rejected_actions: int,
) -> EvaluationResult:
    """Create every seed/seat outcome expected from one held-out pool."""

    seeds = runner_arguments["seeds"]
    players = runner_arguments["players"]
    games = (seeds.end - seeds.start) * players
    rewards = (1.0,) * wins + (-1.0,) * (games - wins)
    outcomes = tuple(
        EvaluationOutcome(
            seed=seeds.start + index // players,
            seat=index % players,
            reward=reward,
            turns=100,
            end_reason=END_GAME_FINISHED,
            launches=1,
        )
        for index, reward in enumerate(rewards)
    )
    metadata = dict(runner_arguments["expected_metadata"])
    metadata.update(engine="test-engine", dimensions=OBSERVATION_DIMENSIONS, evaluation=_EVALUATION)
    return EvaluationResult(outcomes, metadata, rejected_actions)


def _passing_archive(
    tmp_path: Path,
    config: RunConfig,
) -> FrozenCheckpointArchive:
    """Create the exact self-contained archive consumed by the real loader."""

    policy = OrbitPolicy().eval()
    state = CheckpointState(
        config=config,
        model_state=policy.state_dict(),
        optimizer_state={},
        schedule_state={},
        random_states={},
        environment_metadata={},
        environment_steps=32_768,
        completed_updates=1,
        next_validation_step=5_000_000,
        validation_state={},
        metric_state={},
        active_episodes=(),
        seed_cursor=64,
    )
    source = tmp_path / "run" / "checkpoints" / "update-000001.pt"
    fingerprint = save_checkpoint(source, state)
    validation = {
        "gate": "release-readiness",
        "passed": True,
        "checkpoint": source.name,
        "checkpoint_sha256": fingerprint,
        "training_environment_steps": state.environment_steps,
    }
    return freeze_passing_checkpoint(
        source,
        tmp_path / "archive",
        fingerprint=fingerprint,
        config=config,
        validation_record=validation,
        completed_updates=state.completed_updates,
    )
