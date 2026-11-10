"""Protected held-out pool accounting tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier

import pytest

from orrery.checkpoint import sha256_file
from orrery.config import RunConfig, load_run_config
from orrery.env_client import REPOSITORY_ROOT
from orrery.evaluation_pool import (
    POOL_AVAILABLE,
    POOL_SPENT_COMPLETE,
    POOL_SPENT_INCOMPLETE,
    EvaluationPoolClaim,
    claim_evaluation_pool,
    complete_evaluation_pool,
    read_evaluation_pool_manifest,
    run_with_claimed_evaluation_pool,
)


CHECKPOINT_SHA256 = "a" * 64
SECOND_CHECKPOINT_SHA256 = "b" * 64


@pytest.fixture
def config() -> RunConfig:
    """Load the real V1 pool names and ranges with a stable source identity."""

    return load_run_config(REPOSITORY_ROOT / "configs" / "rl-v2-2p.toml")


def test_first_claim_lazily_creates_the_versioned_pool_manifest(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Create the exact frozen pool order through the production claim path."""

    path = tmp_path / "evaluation-pools.json"
    claim_evaluation_pool(
        path,
        config,
        "eval-1",
        archive_identifier="release-first",
        checkpoint_sha256=CHECKPOINT_SHA256,
    )
    manifest = read_evaluation_pool_manifest(path, config)

    assert manifest["format"] == "evaluation-pool-manifest-v1"
    assert [pool["name"] for pool in manifest["pools"]] == list(
        config.evaluation_pool_names
    )
    assert [pool["seed_range"] for pool in manifest["pools"]] == [
        {"start": seeds.start, "end": seeds.end}
        for seeds in config.evaluation_seeds
    ]
    assert manifest["pools"][0]["status"] == POOL_SPENT_INCOMPLETE
    assert all(
        pool["status"] == POOL_AVAILABLE for pool in manifest["pools"][1:]
    )


def test_claim_requires_explicit_valid_selection_before_creating_manifest(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Do not spend protected data when ``--spend-eval`` is absent or wrong.

    Argument parsing will pass the command-line value to this boundary. The
    checks happen before manifest initialization so an accidental evaluation
    command has no persistent side effect at all.
    """

    path = tmp_path / "evaluation-pools.json"

    with pytest.raises(ValueError, match="requires --spend-eval"):
        claim_evaluation_pool(
            path,
            config,
            None,
            archive_identifier="release-example",
            checkpoint_sha256=CHECKPOINT_SHA256,
        )
    with pytest.raises(ValueError, match="unknown evaluation pool"):
        claim_evaluation_pool(
            path,
            config,
            "eval-6",
            archive_identifier="release-example",
            checkpoint_sha256=CHECKPOINT_SHA256,
        )

    assert not path.exists()


def test_claim_is_saved_as_spent_incomplete_before_control_returns(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Persist the irreversible spend before an evaluator could play game 1.

    ``claim_evaluation_pool`` is the boundary a later held-out driver calls
    immediately before its match runner. Once it returns, a fresh read already
    sees the pool as unavailable and bound to the selected checkpoint. A crash
    at the next instruction therefore cannot make that pool reusable.
    """

    path = tmp_path / "evaluation-pools.json"
    claim = claim_evaluation_pool(
        path,
        config,
        "eval-1",
        archive_identifier="release-example",
        checkpoint_sha256=CHECKPOINT_SHA256,
        attempt_id="attempt-one",
        started_at="2026-09-10T12:00:00Z",
    )
    saved = json.loads(path.read_text())
    first = saved["pools"][0]

    assert claim == EvaluationPoolClaim(
        pool_name="eval-1",
        seeds=config.evaluation_seeds[0],
        attempt_id="attempt-one",
        manifest_path=path,
        archive_identifier="release-example",
        checkpoint_sha256=CHECKPOINT_SHA256,
    )
    assert first["status"] == POOL_SPENT_INCOMPLETE
    assert first["attempt"] == {
        "id": "attempt-one",
        "archive_identifier": "release-example",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "started_at": "2026-09-10T12:00:00Z",
        "completed_at": None,
        "result": None,
    }


def test_interrupted_claim_cannot_be_reused_or_skipped(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Treat an unfinished attempt as spent and advance only in pool order.

    The test intentionally does not complete ``eval-1``. That models Ctrl+C,
    process failure, or malformed match output: its saved state remains
    ``spent-incomplete``. A new process must reject both retrying that pool and
    skipping over the next available pool.
    """

    path = tmp_path / "evaluation-pools.json"
    claim_evaluation_pool(
        path,
        config,
        "eval-1",
        archive_identifier="release-first",
        checkpoint_sha256=CHECKPOINT_SHA256,
    )

    with pytest.raises(ValueError, match="next protected evaluation pool is 'eval-2'"):
        claim_evaluation_pool(
            path,
            config,
            "eval-1",
            archive_identifier="release-retry",
            checkpoint_sha256=CHECKPOINT_SHA256,
        )
    with pytest.raises(ValueError, match="next protected evaluation pool is 'eval-2'"):
        claim_evaluation_pool(
            path,
            config,
            "eval-3",
            archive_identifier="release-skip",
            checkpoint_sha256=CHECKPOINT_SHA256,
        )

    second = claim_evaluation_pool(
        path,
        config,
        "eval-2",
        archive_identifier="release-second",
        checkpoint_sha256=SECOND_CHECKPOINT_SHA256,
    )
    assert second.pool_name == "eval-2"


@pytest.mark.parametrize(
    ("archive_identifier", "checkpoint_sha256", "message"),
    (
        ("release-first", SECOND_CHECKPOINT_SHA256, "archive .* already spent"),
        ("release-second", CHECKPOINT_SHA256, "SHA-256 already spent"),
    ),
)
def test_one_checkpoint_cannot_spend_a_second_protected_pool(
    tmp_path: Path,
    config: RunConfig,
    archive_identifier: str,
    checkpoint_sha256: str,
    message: str,
) -> None:
    """Require new release-ready weights before exposing the next seed pool.

    An interrupted first attempt already revealed eval-1. Changing only its
    archive name or only its recorded digest must not give those same weights
    another statistically independent chance on eval-2.
    """

    path = tmp_path / "evaluation-pools.json"
    claim_evaluation_pool(
        path,
        config,
        "eval-1",
        archive_identifier="release-first",
        checkpoint_sha256=CHECKPOINT_SHA256,
    )

    with pytest.raises(ValueError, match=message):
        claim_evaluation_pool(
            path,
            config,
            "eval-2",
            archive_identifier=archive_identifier,
            checkpoint_sha256=checkpoint_sha256,
        )

    saved = read_evaluation_pool_manifest(path, config)
    assert saved["pools"][1]["status"] == POOL_AVAILABLE


def test_claim_fails_after_all_five_pools_are_spent(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Stop official evaluation when the versioned pool supply is exhausted."""

    path = tmp_path / "evaluation-pools.json"
    for index, name in enumerate(config.evaluation_pool_names):
        claim_evaluation_pool(
            path,
            config,
            name,
            archive_identifier=f"release-{index}",
            checkpoint_sha256=f"{index + 1:064x}",
        )

    with pytest.raises(ValueError, match="all protected evaluation pools have been spent"):
        claim_evaluation_pool(
            path,
            config,
            "eval-5",
            archive_identifier="release-extra",
            checkpoint_sha256="f" * 64,
        )


def test_simultaneous_claims_cannot_both_spend_the_same_pool(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Serialize the check-and-write boundary across competing evaluators.

    Both workers ask for ``eval-1`` at nearly the same time. The manifest lock
    must let exactly one worker observe and spend it. The second worker reads
    the first worker's durable update and fails rather than launching another
    evaluation on the same protected boards.
    """

    path = tmp_path / "evaluation-pools.json"
    barrier = Barrier(2)

    def compete(attempt_id: str) -> EvaluationPoolClaim:
        """Wait for both test workers, then submit one conflicting claim."""

        barrier.wait()
        return claim_evaluation_pool(
            path,
            config,
            "eval-1",
            archive_identifier=f"release-{attempt_id}",
            checkpoint_sha256=CHECKPOINT_SHA256,
            attempt_id=attempt_id,
        )

    claims: list[EvaluationPoolClaim] = []
    failures: list[Exception] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(compete, name) for name in ("one", "two")]
        for future in futures:
            try:
                claims.append(future.result())
            except Exception as error:  # The losing claim is the behavior under test.
                failures.append(error)

    assert len(claims) == 1
    assert len(failures) == 1
    assert "next protected evaluation pool is 'eval-2'" in str(failures[0])
    saved = read_evaluation_pool_manifest(path, config)
    assert saved["pools"][0]["attempt"]["id"] == claims[0].attempt_id
    assert saved["pools"][1]["status"] == POOL_AVAILABLE


def test_runner_starts_only_after_claim_is_durable_and_interruption_keeps_it(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Observe the saved claim inside the first instruction allowed to play.

    The callback stands in for the paired Go match runner. It reads the ledger
    before raising KeyboardInterrupt, proving that game execution becomes
    reachable only after ``eval-1`` is already marked spent-incomplete. The
    interruption propagates, but it cannot roll the protected pool back.
    """

    path = tmp_path / "evaluation-pools.json"

    def interrupted_play(claim: EvaluationPoolClaim) -> None:
        """Model Ctrl+C immediately before the runner would reset game 1."""

        saved = read_evaluation_pool_manifest(path, config)
        first = saved["pools"][0]
        assert first["status"] == POOL_SPENT_INCOMPLETE
        assert first["attempt"]["id"] == claim.attempt_id
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_with_claimed_evaluation_pool(
            path,
            config,
            "eval-1",
            archive_identifier="release-example",
            checkpoint_sha256=CHECKPOINT_SHA256,
            play=interrupted_play,
            attempt_id="interrupted-attempt",
        )

    saved = read_evaluation_pool_manifest(path, config)
    assert saved["pools"][0]["status"] == POOL_SPENT_INCOMPLETE
    with pytest.raises(ValueError, match="next protected evaluation pool is 'eval-2'"):
        claim_evaluation_pool(
            path,
            config,
            "eval-1",
            archive_identifier="release-retry",
            checkpoint_sha256=CHECKPOINT_SHA256,
        )


@pytest.mark.parametrize("passed", (False, True))
def test_completed_pass_or_failure_remains_permanently_spent(
    tmp_path: Path,
    config: RunConfig,
    passed: bool,
) -> None:
    """Record either result while keeping the protected range unavailable.

    Passing and failing describe checkpoint quality. Neither result changes
    pool accounting: the boards have been revealed in both cases and cannot
    be used for another official attempt.
    """

    path = tmp_path / "evaluation-pools.json"
    claim = claim_evaluation_pool(
        path,
        config,
        "eval-1",
        archive_identifier="release-example",
        checkpoint_sha256=CHECKPOINT_SHA256,
        attempt_id="attempt-one",
    )
    result_path = tmp_path / "attempt-one.json"
    result_path.write_text('{"passed": true}\n')
    result_sha256 = sha256_file(result_path)
    completed = complete_evaluation_pool(
        claim,
        config,
        passed=passed,
        result_file=str(result_path),
        result_sha256=result_sha256,
        completed_at="2026-09-10T13:00:00Z",
    )

    first = completed["pools"][0]
    assert first["status"] == POOL_SPENT_COMPLETE
    assert first["attempt"]["result"] == {
        "outcome": "passed" if passed else "failed",
        "file": str(result_path),
        "sha256": result_sha256,
    }
    with pytest.raises(ValueError, match="next protected evaluation pool is 'eval-2'"):
        claim_evaluation_pool(
            path,
            config,
            "eval-1",
            archive_identifier="release-retry",
            checkpoint_sha256=CHECKPOINT_SHA256,
        )


def test_malformed_result_leaves_the_claim_spent_incomplete(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Reject unusable output without returning its protected pool to service.

    Match output must become valid result JSON before a spend can be labelled
    complete. The earlier irreversible claim remains authoritative when that
    validation fails, so a malformed result cannot be discarded and retried on
    the same final-exam boards.
    """

    manifest_path = tmp_path / "evaluation-pools.json"
    claim = claim_evaluation_pool(
        manifest_path,
        config,
        "eval-1",
        archive_identifier="release-example",
        checkpoint_sha256=CHECKPOINT_SHA256,
    )
    malformed_result = tmp_path / "malformed.json"
    malformed_result.write_text("{not valid json\n")

    with pytest.raises(ValueError, match="result file is invalid JSON"):
        complete_evaluation_pool(
            claim,
            config,
            passed=False,
            result_file=str(malformed_result),
            result_sha256=sha256_file(malformed_result),
        )

    saved = read_evaluation_pool_manifest(manifest_path, config)
    assert saved["pools"][0]["status"] == POOL_SPENT_INCOMPLETE
    with pytest.raises(ValueError, match="next protected evaluation pool is 'eval-2'"):
        claim_evaluation_pool(
            manifest_path,
            config,
            "eval-1",
            archive_identifier="release-retry",
            checkpoint_sha256=CHECKPOINT_SHA256,
        )


def test_malformed_manifest_fails_closed_instead_of_replenishing_pools(
    tmp_path: Path,
    config: RunConfig,
) -> None:
    """Refuse evaluation when earlier spending history cannot be trusted.

    Automatically replacing invalid JSON with a fresh manifest would silently
    make every protected pool available again. The safer behavior is to stop
    before any game until the existing evidence is investigated.
    """

    path = tmp_path / "evaluation-pools.json"
    path.write_text("{not valid json\n")

    with pytest.raises(ValueError, match="invalid JSON"):
        read_evaluation_pool_manifest(path, config)
    with pytest.raises(ValueError, match="invalid JSON"):
        claim_evaluation_pool(
            path,
            config,
            "eval-1",
            archive_identifier="release-example",
            checkpoint_sha256=CHECKPOINT_SHA256,
        )

    assert path.read_text() == "{not valid json\n"
