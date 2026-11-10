"""Evaluate one frozen checkpoint once on an explicitly spent held-out pool."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from .archive import FROZEN_ARCHIVE_FORMAT
from .checkpoint import (
    load_checkpoint,
    recorded_checkpoint_fingerprint,
    sha256_file,
)
from .config import RunConfig
from .env_client import SCHEDULE_ROTATE
from .evaluation_context import configure_cuda_workspace, resolve_evaluation_device
from .evaluation_pool import (
    DEFAULT_EVALUATION_POOL_MANIFEST,
    EvaluationPoolClaim,
    complete_evaluation_pool,
    read_evaluation_pool_manifest,
    run_with_claimed_evaluation_pool,
)
from .evaluator import (
    EvaluationModel,
    EvaluationResult,
    ReleaseReadinessResult,
    checked_evaluation_result,
    evaluation_metadata,
    interval_evidence,
    prepare_evaluation_model,
    release_readiness_gate,
    run_rotated_matches,
    timings_evidence,
    validation_summary,
)
from .jsonio import json_value, read_json_object, write_json_exclusive
from .model import OrbitPolicy, policy_device


HELD_OUT_RESULT_FORMAT = "held-out-evaluation-result-v1"
DEFAULT_ENV_COMMAND = ("go", "run", "./cmd/env")


@dataclass(frozen=True)
class ArchivedCandidate:
    """Verified frozen weights and metadata allowed to spend a held-out pool."""

    # Directory created by the release-readiness freeze operation.
    directory: Path
    # Stable archive identity stored in release-validation evidence.
    identifier: str
    # Independent checkpoint copy inside the archive.
    checkpoint: Path
    # Digest verified against the archive manifest, sidecar, and actual bytes.
    checkpoint_sha256: str
    # Immutable experiment contract deserialized from that checkpoint.
    config: RunConfig
    # Float32 model loaded with the frozen weights, in eval mode, already on
    # the evaluation device.
    policy: nn.Module
    # Training boundary that produced this selected checkpoint.
    training_environment_steps: int
    completed_updates: int


@dataclass(frozen=True)
class HeldOutEvaluation:
    """Location and decision produced by one completed protected attempt."""

    claim: EvaluationPoolClaim
    result_path: Path
    result_sha256: str
    decision: ReleaseReadinessResult


@dataclass(frozen=True)
class _ClaimedMatches:
    """Rotated-game output plus the manifest snapshot taken before game 1."""

    manifest_at_start: Mapping[str, Any]
    result: EvaluationResult


EvaluationRunner = Callable[..., EvaluationResult]
# Called as loader(archive_directory, device=...), like load_archived_candidate.
CandidateLoader = Callable[..., ArchivedCandidate]


def run_held_out_evaluation(
    archive_directory: str | Path,
    *,
    spend_eval: str | None,
    manifest_path: str | Path = DEFAULT_EVALUATION_POOL_MANIFEST,
    command: Sequence[str] = DEFAULT_ENV_COMMAND,
    batch_size: int | None = None,
    device: str | None = None,
    evaluation_runner: EvaluationRunner = run_rotated_matches,
    candidate_loader: CandidateLoader | None = None,
    attempt_id: str | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> HeldOutEvaluation:
    """Run the complete final-exam lifecycle for one frozen model.

    ``archive_directory`` identifies the checkpoint that training selected.
    ``spend_eval`` identifies the protected group of unseen boards that the
    user deliberately chose to consume. ``device`` overrides the archived
    config's evaluation device (``"cpu"`` or ``"cuda"``). The optional runner
    and loader inputs let tests replace expensive file and Go-process work;
    ordinary command-line use receives the real implementations.

    Archive verification deliberately happens before the protected pool is
    claimed. Once verification succeeds, ``run_with_claimed_evaluation_pool``
    writes ``spent-incomplete`` before invoking the rotated-match runner. Every
    error after that boundary consumes the pool, including Go startup errors,
    Ctrl+C, missing games, duplicate games, and malformed metadata.
    """

    # Stage 1: load and verify the model, and place it on its evaluation
    # device, before touching protected pool state. A damaged or ineligible
    # archive, or a missing GPU, therefore cannot waste an evaluation pool.
    loader = candidate_loader or load_archived_candidate
    candidate = loader(archive_directory, device=device)
    # Evaluation does not require the same number of simultaneous games as
    # training. When the user gives no override, reuse the archived run's
    # environment count so memory and process behavior remain familiar.
    if batch_size is None:
        batch_size = candidate.config.environment_count
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    # Make the evaluator's private copy of the model now, on its device. The
    # copy is a fresh allocation, so an out-of-memory error or a mismatched
    # evaluator surfaces here, while the pool is still unspent.
    prepared = prepare_evaluation_model(
        candidate.policy,
        candidate.config.evaluation_policy,
        policy_device(candidate.policy),
    )

    # Stage 2: make the irreversible manifest change, then enter the function
    # that starts Go matches. The wrapper guarantees those operations cannot be
    # accidentally reversed by a future refactor.
    claim, matches = run_with_claimed_evaluation_pool(
        manifest_path,
        candidate.config,
        spend_eval,
        archive_identifier=candidate.identifier,
        checkpoint_sha256=candidate.checkpoint_sha256,
        play=partial(
            _run_claimed_matches,
            candidate=candidate,
            prepared=prepared,
            command=command,
            batch_size=batch_size,
            evaluation_runner=evaluation_runner,
        ),
        attempt_id=attempt_id,
        started_at=started_at,
    )
    # Stage 3: reduce the complete game list to wins, losses, draws, confidence
    # bounds, and the final pass/fail decision. This changes no model weights.
    result = matches.result
    decision = release_readiness_gate(
        result,
        expected_games=candidate.config.held_out_games,
        required_wins=candidate.config.held_out_required_wins,
        minimum_lower_bound=candidate.config.release_minimum_lower_bound,
        confidence_level=candidate.config.release_confidence_level,
    )

    # Stage 4: publish the detailed evidence before declaring the manifest
    # entry complete. If publication fails, the pool correctly remains
    # spent-incomplete and no result path is advertised as trustworthy.
    # Results are added to the frozen archive under an attempt-specific name.
    # Exclusive creation preserves any earlier evidence and makes a repeated
    # attempt ID fail without changing its already-spent pool state.
    result_path = candidate.directory / f"held-out-{claim.pool_name}-{claim.attempt_id}.json"
    evidence = _evaluation_evidence(
        candidate,
        claim,
        matches.manifest_at_start,
        result,
        decision,
        command,
        batch_size,
    )
    write_json_exclusive(result_path, evidence)
    result_sha256 = sha256_file(result_path)
    # Stage 5: bind the saved result and its digest to the pool claim. Passing
    # and failing are both complete outcomes; neither makes the pool reusable.
    complete_evaluation_pool(
        claim,
        candidate.config,
        passed=decision.passed,
        result_file=str(result_path),
        result_sha256=result_sha256,
        completed_at=completed_at,
    )
    return HeldOutEvaluation(claim, result_path, result_sha256, decision)


def _run_claimed_matches(
    claim: EvaluationPoolClaim,
    *,
    candidate: ArchivedCandidate,
    prepared: EvaluationModel,
    command: Sequence[str],
    batch_size: int,
    evaluation_runner: EvaluationRunner,
) -> _ClaimedMatches:
    """Play every protected board after its pool is safely marked spent.

    ``claim`` supplies the exact 1,000-seed range. ``candidate`` supplies the
    run configuration and versioned opponent. ``prepared`` is the evaluator's
    private model copy, made before the claim so allocating it cannot spend
    the pool. The runner starts a Go environment,
    plays each seed from both learner seats, and returns 2,000 terminal outcomes.

    ``run_with_claimed_evaluation_pool`` invokes this function only after the
    irreversible manifest write succeeds. Keeping the match work in a named
    module-level function makes its inputs and two outputs explicit without a
    closure or mutable ``nonlocal`` state.
    """

    # Capture the complete ledger immediately after this attempt's claim. The
    # results can then prove which earlier pools were spent and which later
    # pools remained protected when game 1 became reachable.
    manifest_at_start = read_evaluation_pool_manifest(
        claim.manifest_path,
        candidate.config,
    )
    # Describe the Go contract expected for this attempt. The runner compares
    # these values with Go's create response before accepting any score.
    expected_metadata = evaluation_metadata(
        candidate.config,
        claim.seeds,
        candidate.config.training_opponent,
    )
    # This is the Python-to-Go game boundary. The reusable runner repeatedly
    # chooses deterministic actions, sends them to Go, and receives the next
    # observations until every scheduled game terminates.
    result = evaluation_runner(
        prepared,
        command=command,
        seeds=claim.seeds,
        opponent=candidate.config.training_opponent,
        batch_size=batch_size,
        evaluation_policy=prepared.evaluation_policy,
        device=prepared.device,
        expected_metadata=expected_metadata,
        # The archive owns the player count. Final evaluation must replay that
        # recorded contract instead of silently using a default.
        players=candidate.config.players,
    )
    # Check coverage and metadata once more at this command boundary. An
    # injected or future runner cannot bypass the held-out evidence rules.
    return _ClaimedMatches(
        manifest_at_start=manifest_at_start,
        result=checked_evaluation_result(
            result,
            claim.seeds,
            expected_metadata,
            players=candidate.config.players,
            evaluation_policy=candidate.config.evaluation_policy,
            context="held-out evaluation",
        ),
    )


def load_archived_candidate(
    archive_directory: str | Path,
    *,
    device: str | None = None,
) -> ArchivedCandidate:
    """Turn a frozen archive directory into a model ready for final games.

    ``device`` overrides where the model is placed; by default it goes to the
    device saved in the archived run configuration. Every archive check runs
    on the CPU first; only the verified model is then moved. Asking for CUDA
    on a machine without it raises here, which in held-out evaluation is
    before any protected pool is claimed.

    A beginner can view this as a chain of receipts. ``archive.json`` names the
    checkpoint; its sidecar records a fingerprint; hashing the actual bytes
    recomputes that fingerprint; the checkpoint contains its configuration;
    and ``release-validation.json`` proves that training selected it. Every
    receipt must describe the same file and training boundary.

    The archive is local trusted storage because the checkpoint uses Torch's
    pickle-based format. The safe JSON envelope, byte count, sidecar, and hash
    are still checked by ``load_checkpoint`` before deserialization.
    """

    # Read the small JSON manifest first. It is safe to inspect before the much
    # larger Torch checkpoint is deserialized.
    # The protected-pool ledger outlives the shell invocation that created it.
    # Resolve the archive now so the permanent result path remains meaningful
    # even when a later reader starts in a different working directory.
    directory = Path(archive_directory).resolve()
    archive = read_json_object(directory / "archive.json", "archive manifest")
    expected_archive_fields = {
        "format",
        "identifier",
        "immutable",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "checkpoint_metadata",
        "checkpoint_fingerprint",
        "run_config",
        "release_validation",
        "training_environment_steps",
        "completed_updates",
    }
    if set(archive) != expected_archive_fields:
        raise ValueError("archive manifest has an invalid structure")
    if archive["format"] != FROZEN_ARCHIVE_FORMAT or archive["immutable"] is not True:
        raise ValueError("checkpoint is not a supported immutable frozen archive")
    identifier = archive["identifier"]
    if not isinstance(identifier, str) or identifier != directory.name:
        raise ValueError("archive identifier does not match its directory")
    # Resolve only plain filenames inside the archive. This prevents a changed
    # manifest from redirecting the loader to an unrelated path elsewhere.
    checkpoint_name = _archive_filename(archive["checkpoint"], "checkpoint")
    checkpoint = directory / checkpoint_name
    checkpoint_sha256 = archive["checkpoint_sha256"]
    if recorded_checkpoint_fingerprint(checkpoint) != checkpoint_sha256:
        raise ValueError("archive checkpoint fingerprint disagrees with its manifest")
    if checkpoint.stat().st_size != archive["checkpoint_size_bytes"]:
        raise ValueError("archive checkpoint size disagrees with its manifest")
    if archive["checkpoint_metadata"] != checkpoint_name + ".metadata.json":
        raise ValueError("archive manifest names the wrong checkpoint metadata file")
    if archive["checkpoint_fingerprint"] != checkpoint_name + ".sha256":
        raise ValueError("archive manifest names the wrong checkpoint fingerprint file")

    # ``load_checkpoint`` repeats envelope, size, digest, and run-contract
    # validation before loading the pickled learner state onto CPU. The model
    # signature makes a checkpoint for another architecture fail there too,
    # before Torch deserializes it.
    state = load_checkpoint(
        checkpoint,
        expected_model_state=OrbitPolicy().state_dict(),
    )
    config_record = read_json_object(
        directory / _archive_filename(archive["run_config"], "run configuration"),
        "archived run configuration",
    )
    if config_record != json_value(state.config.to_dict()):
        raise ValueError("archived run configuration disagrees with its checkpoint")
    release_record = read_json_object(
        directory / _archive_filename(
            archive["release_validation"],
            "release validation",
        ),
        "archived release validation",
    )
    expected_reference = {
        "format": FROZEN_ARCHIVE_FORMAT,
        "identifier": identifier,
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if (
        release_record.get("gate") != "release-readiness"
        or release_record.get("passed") is not True
        or release_record.get("archive") != expected_reference
        or release_record.get("checkpoint") != checkpoint_name
        or release_record.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("archive lacks passing release-readiness evidence")
    if (
        state.environment_steps != archive["training_environment_steps"]
        or state.completed_updates != archive["completed_updates"]
    ):
        raise ValueError("archive training boundary disagrees with its checkpoint")

    # Recreate the known network architecture and copy the frozen tensors into
    # it on the CPU, where the checkpoint was loaded.
    policy = OrbitPolicy().to(device="cpu", dtype=torch.float32).eval()
    policy.load_state_dict(state.model_state, strict=True)
    # Only now choose the evaluation device and move the verified model there.
    # On CUDA this also fixes cuBLAS's workspace first (a no-op when main()
    # already did) and allocates the model on the GPU, so a missing GPU, a bad
    # workspace setting or an allocation failure all surface at this point.
    target = resolve_evaluation_device(device, state.config.device)
    if target.type == "cuda":
        configure_cuda_workspace()
    policy = policy.to(device=target)
    return ArchivedCandidate(
        directory=directory,
        identifier=identifier,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        config=state.config,
        policy=policy,
        training_environment_steps=state.environment_steps,
        completed_updates=state.completed_updates,
    )


def _evaluation_evidence(
    candidate: ArchivedCandidate,
    claim: EvaluationPoolClaim,
    manifest_at_start: Mapping[str, Any],
    result: EvaluationResult,
    decision: ReleaseReadinessResult,
    command: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    """Assemble all facts needed to understand and reproduce the final score.

    The returned dictionary is the exact object written to ``held-out-*.json``.
    It combines four kinds of information: what was tested, which protected
    pool was spent, how pass/fail was calculated, and what happened in every
    individual game. It does not run games or recalculate statistics.
    """

    evidence: dict[str, Any] = {
        "format": HELD_OUT_RESULT_FORMAT,
        "passed": decision.passed,
        "failed_checks": list(decision.failed_checks),
        "claim": claim.evidence(),
        "pool_manifest_at_start": manifest_at_start,
        "checkpoint": {
            "archive_format": FROZEN_ARCHIVE_FORMAT,
            "checkpoint_format": candidate.config.checkpoint_format,
            "archive_identifier": candidate.identifier,
            "file": candidate.checkpoint.name,
            "sha256": candidate.checkpoint_sha256,
            "training_environment_steps": candidate.training_environment_steps,
            "completed_updates": candidate.completed_updates,
        },
        # Store the entire immutable configuration copied into the archive.
        # Individual version fields below make common evidence easy to scan;
        # this complete record retains every training, seed, threshold, and
        # source-revision value needed to interpret the checkpoint later.
        "run_config": json_value(candidate.config.to_dict()),
        "requirements": {
            "expected_games": candidate.config.held_out_games,
            "required_wins": candidate.config.held_out_required_wins,
            "confidence_level": candidate.config.release_confidence_level,
            "minimum_lower_bound": candidate.config.release_minimum_lower_bound,
            "required_rejected_actions": 0,
            "required_results_per_seed_and_seat": 1,
            # Every seed is played once from each seat on the map.
            "seats_per_seed": candidate.config.players,
        },
        "aggregate": {
            "games": decision.games,
            "wins": decision.wins,
            "losses": decision.losses,
            "draws": decision.draws,
            "win_rate": decision.win_rate,
            "rejected_actions": result.rejected_actions,
            "confidence_interval": interval_evidence(decision.interval),
        },
        "contract": {
            "rl": candidate.config.contract,
            "checkpoint": candidate.config.checkpoint_format,
            "policy": candidate.config.evaluation_policy,
            "opponent": candidate.config.training_opponent,
            "schedule": SCHEDULE_ROTATE,
            "protocol": candidate.config.protocol_version,
            "observation": candidate.config.observation_schema,
            "generator": candidate.config.generator_version,
            "engine": result.metadata["engine"],
            "players": candidate.config.players,
            "dimensions": result.metadata["dimensions"],
            # Device, software versions and numeric settings actually used,
            # needed to tell an exact replay from separate evidence.
            "evaluation": result.metadata["evaluation"],
        },
        "command": {
            "environment": list(command),
            "batch_size": batch_size,
            "spend_eval": claim.pool_name,
        },
        # How the protected games looked, beyond the pass/fail decision.
        "summary": validation_summary(result.outcomes, candidate.config.players),
        "outcomes": [asdict(outcome) for outcome in result.outcomes],
    }
    # Timings describe this one attempt rather than the checkpoint's score.
    if result.timings is not None:
        evidence["timings"] = timings_evidence(result.timings)
    return evidence


def _archive_filename(value: object, description: str) -> str:
    """Require a filename such as ``checkpoint.pt`` rather than a path.

    Archive manifests are allowed to point only at sibling files within their
    own directory. Values containing ``..`` or directory separators could
    escape that directory, so they fail before any file is opened.
    """

    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"archive {description} must be a plain filename")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Read command-line options, run one attempt, and return a shell status.

    ``--archive`` chooses the frozen model and required ``--spend-eval`` makes
    consumption of unseen boards deliberate. A passing model returns status 0;
    a complete but failing model returns 1. Validation errors raise before or
    during the lifecycle and retain the pool state appropriate to their stage.
    """

    # Fix cuBLAS's workspace before any GPU work, including model loading.
    configure_cuda_workspace()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        required=True,
        help="directory of the frozen checkpoint archive selected by release readiness",
    )
    parser.add_argument(
        "--spend-eval",
        required=True,
        help="next protected pool from the archived run configuration",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_EVALUATION_POOL_MANIFEST),
        help="shared protected-pool ledger (default: runs/evaluation-pools-v1.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="parallel Go game slots (default: archived training environment count)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="where the model decides moves (default: the archived run's device)",
    )
    arguments = parser.parse_args(argv)

    evaluation = run_held_out_evaluation(
        arguments.archive,
        spend_eval=arguments.spend_eval,
        manifest_path=arguments.manifest,
        batch_size=arguments.batch_size,
        device=arguments.device,
    )
    print(
        f"held-out evaluation {'passed' if evaluation.decision.passed else 'failed'}: "
        f"pool={evaluation.claim.pool_name} "
        f"wins={evaluation.decision.wins}/{evaluation.decision.games} "
        f"result={evaluation.result_path}",
        flush=True,
    )
    return 0 if evaluation.decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
