"""Permanently account for one-shot held-out evaluation pools."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, TypeVar
from uuid import uuid4

from .checkpoint import sha256_file
from .config import RunConfig, SeedRange
from .jsonio import json_object_copy, read_json_object, write_json_atomic
from .paths import REPOSITORY_ROOT


EVALUATION_POOL_MANIFEST_FORMAT = "evaluation-pool-manifest-v1"
# Every training run and frozen checkpoint shares this one repository-level
# ledger. A per-run manifest would incorrectly make eval-1 available again
# whenever a new experiment directory was created. This local default is an
# operational record rather than tamper-proof storage: ``runs/`` is Git-ignored,
# so deleting it, using a fresh clone, or evaluating on another machine without
# copying this manifest loses spending history and can incorrectly re-offer
# eval-1. Preserve and transfer this file with official evaluation artifacts.
DEFAULT_EVALUATION_POOL_MANIFEST = (
    REPOSITORY_ROOT / "runs" / "evaluation-pools-v1.json"
)
POOL_AVAILABLE = "available"
POOL_SPENT_INCOMPLETE = "spent-incomplete"
POOL_SPENT_COMPLETE = "spent-complete"
_POOL_STATUSES = {
    POOL_AVAILABLE,
    POOL_SPENT_INCOMPLETE,
    POOL_SPENT_COMPLETE,
}
_EvaluationOutput = TypeVar("_EvaluationOutput")


@dataclass(frozen=True)
class EvaluationPoolClaim:
    """The durable identity assigned before one held-out attempt starts.

    Creating this value means the corresponding manifest entry is already
    ``spent-incomplete`` on disk. The caller may therefore start games only
    after ``claim_evaluation_pool`` has returned successfully.
    """

    # Name selected explicitly by the user, such as ``eval-1``.
    pool_name: str
    # Exact protected scenario range owned by this one attempt.
    seeds: SeedRange
    # Unique identity that binds later completion to this particular claim.
    attempt_id: str
    # Shared manifest whose saved state proves the pool was spent before play.
    manifest_path: Path
    # Frozen checkpoint archive and digest tested by this attempt.
    archive_identifier: str
    checkpoint_sha256: str

    def evidence(self) -> dict[str, object]:
        """Convert this claim into plain values suitable for result JSON.

        The result records ``spent-incomplete`` because that was the pool state
        when games began. A later manifest update may say ``spent-complete``,
        but preserving the start state proves the pool was claimed before play.
        """

        return {
            "manifest_format": EVALUATION_POOL_MANIFEST_FORMAT,
            "pool": self.pool_name,
            "seed_range": {"start": self.seeds.start, "end": self.seeds.end},
            "status_at_start": POOL_SPENT_INCOMPLETE,
            "attempt_id": self.attempt_id,
            "archive_identifier": self.archive_identifier,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def claim_evaluation_pool(
    path: str | Path,
    config: RunConfig,
    selection: str | None,
    *,
    archive_identifier: str,
    checkpoint_sha256: str,
    attempt_id: str | None = None,
    started_at: str | None = None,
) -> EvaluationPoolClaim:
    """Reserve the next protected board set for one checkpoint forever.

    ``path`` is the shared ledger, ``config`` supplies the frozen pool order,
    and ``selection`` is the value of ``--spend-eval``. The archive identifier
    and checkpoint hash record exactly which model gained access to the boards.
    Tests may supply stable IDs and timestamps; real commands generate them.

    ``selection`` deliberately accepts ``None`` so a command that omitted
    ``--spend-eval`` fails here before creating or changing the manifest. A
    process lock makes the read-check-write sequence exclusive across local
    evaluator processes, and atomic replacement prevents a partial JSON file
    from becoming the official ledger.
    """

    # Validate all caller-controlled identifiers before creating a lock file or
    # manifest. A typo must not have any lasting effect on protected state.
    if selection is None:
        raise ValueError("held-out evaluation requires --spend-eval <eval-N>")
    if selection not in config.evaluation_pool_names:
        raise ValueError(f"unknown evaluation pool {selection!r}")
    if not archive_identifier or not archive_identifier.strip():
        raise ValueError("archive_identifier must not be empty")
    _validate_sha256(checkpoint_sha256)
    claim_id = attempt_id if attempt_id is not None else uuid4().hex
    if not isinstance(claim_id, str) or not claim_id.strip():
        raise ValueError("attempt_id must not be empty")
    start_time = started_at if started_at is not None else _utc_now()
    if not isinstance(start_time, str) or not start_time.strip():
        raise ValueError("started_at must not be empty")

    # The following read, order check, state change, and write form one locked
    # transaction from the perspective of other local evaluation processes.
    manifest_path = Path(path)
    with _locked_manifest(manifest_path):
        if manifest_path.exists():
            manifest = _read_manifest(manifest_path)
        else:
            manifest = _new_manifest(config)
        _validate_manifest(manifest, config)

        # The first available item is the only legal selection. Earlier items
        # may be complete or incomplete; either status means they were spent.
        next_pool = next(
            (
                pool
                for pool in manifest["pools"]
                if pool["status"] == POOL_AVAILABLE
            ),
            None,
        )
        if next_pool is None:
            raise ValueError("all protected evaluation pools have been spent")
        if next_pool["name"] != selection:
            raise ValueError(
                f"next protected evaluation pool is {next_pool['name']!r}, "
                f"not {selection!r}"
            )

        # A protected pool is one independent final exam for one frozen model.
        # Once any earlier pool has exposed its boards to a checkpoint, neither
        # that archive nor the same checkpoint bytes may receive another pool,
        # regardless of whether the earlier attempt passed, failed, crashed, or
        # produced malformed output.
        for pool in manifest["pools"]:
            attempt = pool["attempt"]
            if not isinstance(attempt, dict):
                continue
            if attempt["archive_identifier"] == archive_identifier:
                raise ValueError(
                    f"archive {archive_identifier!r} already spent "
                    f"protected evaluation pool {pool['name']!r}"
                )
            if attempt["checkpoint_sha256"] == checkpoint_sha256:
                raise ValueError(
                    "checkpoint SHA-256 already spent protected evaluation "
                    f"pool {pool['name']!r}"
                )

        # This is the irreversible transition. It happens while the manifest
        # lock is held and reaches disk before this function returns. If the
        # evaluator then crashes, Ctrl+C is pressed, or its output is invalid,
        # this entry remains spent-incomplete and can never be reclaimed.
        next_pool["status"] = POOL_SPENT_INCOMPLETE
        next_pool["attempt"] = {
            "id": claim_id,
            "archive_identifier": archive_identifier,
            "checkpoint_sha256": checkpoint_sha256,
            "started_at": start_time,
            "completed_at": None,
            "result": None,
        }
        _validate_manifest(manifest, config)
        write_json_atomic(manifest_path, manifest)

    # Return the same range recorded in the manifest so the match runner does
    # not need to look up or reconstruct seed boundaries on its own.
    seed_index = config.evaluation_pool_names.index(selection)
    return EvaluationPoolClaim(
        pool_name=selection,
        seeds=config.evaluation_seeds[seed_index],
        attempt_id=claim_id,
        manifest_path=manifest_path,
        archive_identifier=archive_identifier,
        checkpoint_sha256=checkpoint_sha256,
    )


def run_with_claimed_evaluation_pool(
    path: str | Path,
    config: RunConfig,
    selection: str | None,
    *,
    archive_identifier: str,
    checkpoint_sha256: str,
    play: Callable[[EvaluationPoolClaim], _EvaluationOutput],
    attempt_id: str | None = None,
    started_at: str | None = None,
) -> tuple[EvaluationPoolClaim, _EvaluationOutput]:
    """Enforce the order “save the spend, then call the game runner.”

    ``play`` is a callback supplied by ``evaluate.py``. It receives the saved
    claim, including the exact seed range, and returns whatever complete match
    data the caller needs. This small wrapper exists so no caller can reverse
    these two important operations by accident.

    Held-out orchestration supplies its complete rotated-match runner as
    ``play``. Keeping that call behind this boundary fixes the critical order:
    the manifest update finishes first, and only then can the runner reset a Go
    game slot. Exceptions are deliberately not used to undo the claim. A crash,
    Ctrl+C, or malformed result therefore leaves the pool spent-incomplete.

    A valid caller writes and verifies its immutable results after this returns,
    then calls ``complete_evaluation_pool``. Until that final evidence exists,
    incomplete is the honest manifest state even when all games happened to run.
    """

    # This call must finish its atomic disk write before ``play`` is evaluated.
    claim = claim_evaluation_pool(
        path,
        config,
        selection,
        archive_identifier=archive_identifier,
        checkpoint_sha256=checkpoint_sha256,
        attempt_id=attempt_id,
        started_at=started_at,
    )
    # Do not catch exceptions here. Propagating them tells the user what failed,
    # while the already-written spent-incomplete record remains on disk.
    return claim, play(claim)


def complete_evaluation_pool(
    claim: EvaluationPoolClaim,
    config: RunConfig,
    *,
    passed: bool,
    result_file: str,
    result_sha256: str,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Attach trustworthy result evidence to an existing pool claim.

    ``claim`` identifies the earlier irreversible spend. ``result_file`` and
    ``result_sha256`` identify the immutable JSON produced after all games were
    checked. ``passed`` describes model quality; both True and False are normal
    completed attempts and both keep the pool unavailable.

    This function is intentionally unavailable for interruptions or malformed
    output. Those attempts retain the ``spent-incomplete`` state written before
    play. A normal, complete result changes only the description of the spend;
    it never changes the fact that the protected seeds are unavailable.
    """

    # Validate and hash the result before taking the manifest lock. Invalid or
    # missing output leaves the existing spent-incomplete record untouched.
    if not isinstance(passed, bool):
        raise ValueError("passed must be a boolean")
    if not isinstance(result_file, str) or not result_file.strip():
        raise ValueError("result_file must not be empty")
    _validate_sha256(result_sha256)
    result_path = Path(result_file)
    result_value = read_json_object(result_path, "evaluation result file")
    actual_result_sha256 = sha256_file(result_path)
    if actual_result_sha256 != result_sha256:
        raise ValueError(
            "evaluation result SHA-256 mismatch: "
            f"recorded {result_sha256}, actual {actual_result_sha256}"
        )
    finish_time = completed_at if completed_at is not None else _utc_now()
    if not isinstance(finish_time, str) or not finish_time.strip():
        raise ValueError("completed_at must not be empty")

    # Reopen the latest ledger under its lock. Another process may have read it
    # since this attempt started, so completion must verify the saved attempt ID
    # and checkpoint identity instead of trusting an old in-memory copy.
    with _locked_manifest(claim.manifest_path):
        manifest = _read_manifest(claim.manifest_path)
        _validate_manifest(manifest, config)
        pool = next(
            (item for item in manifest["pools"] if item["name"] == claim.pool_name),
            None,
        )
        if pool is None:
            raise ValueError(f"manifest does not contain pool {claim.pool_name!r}")
        attempt = pool["attempt"]
        if pool["status"] != POOL_SPENT_INCOMPLETE or not isinstance(attempt, dict):
            raise ValueError(f"evaluation pool {claim.pool_name!r} is not incomplete")
        if (
            attempt.get("id") != claim.attempt_id
            or attempt.get("archive_identifier") != claim.archive_identifier
            or attempt.get("checkpoint_sha256") != claim.checkpoint_sha256
        ):
            raise ValueError("evaluation pool claim does not match the manifest")

        # Completion enriches the permanent spend with a result. There is no
        # code path from either spent status back to ``available``.
        pool["status"] = POOL_SPENT_COMPLETE
        attempt["completed_at"] = finish_time
        attempt["result"] = {
            "outcome": "passed" if passed else "failed",
            "file": result_file,
            "sha256": result_sha256,
        }
        _validate_manifest(manifest, config)
        write_json_atomic(claim.manifest_path, manifest)
        return json_object_copy(manifest)


def read_evaluation_pool_manifest(
    path: str | Path,
    config: RunConfig,
) -> dict[str, Any]:
    """Read the current ledger safely for evidence or human inspection.

    Even readers take the same lock as writers so they cannot observe the brief
    moment while a claim is being prepared. Validation then rejects unknown
    formats, changed ranges, or impossible ordering before returning a copy.
    """

    manifest_path = Path(path)
    with _locked_manifest(manifest_path):
        manifest = _read_manifest(manifest_path)
        _validate_manifest(manifest, config)
        return json_object_copy(manifest)


def _new_manifest(config: RunConfig) -> dict[str, Any]:
    """Build the first in-memory ledger from the frozen V1 configuration.

    ``zip(..., strict=True)`` pairs each pool name with exactly one seed range.
    Every entry begins available and has no attempt because no checkpoint has
    seen those boards yet. This function only builds data; its caller validates
    and atomically writes it.
    """

    return {
        "format": EVALUATION_POOL_MANIFEST_FORMAT,
        "contract": config.contract,
        "pools": [
            {
                "name": name,
                "seed_range": {"start": seeds.start, "end": seeds.end},
                "status": POOL_AVAILABLE,
                "attempt": None,
            }
            for name, seeds in zip(
                config.evaluation_pool_names,
                config.evaluation_seeds,
                strict=True,
            )
        ],
    }


def _validate_manifest(manifest: Mapping[str, Any], config: RunConfig) -> None:
    """Prove that a ledger still obeys the complete protected-pool contract.

    Validation checks the outer JSON shape, format version, RL contract, exact
    five names and ranges, every entry's state, and ascending spend order. It
    raises instead of repairing suspicious data because automatically resetting
    a damaged ledger could make previously revealed boards available again.
    """

    if set(manifest) != {"format", "contract", "pools"}:
        raise ValueError("evaluation pool manifest has an invalid structure")
    if manifest["format"] != EVALUATION_POOL_MANIFEST_FORMAT:
        raise ValueError(
            f"unsupported evaluation pool manifest format {manifest['format']!r}"
        )
    if manifest["contract"] != config.contract:
        raise ValueError("evaluation pool manifest has a different RL contract")
    pools = manifest["pools"]
    if not isinstance(pools, list) or len(pools) != len(config.evaluation_pool_names):
        raise ValueError("evaluation pool manifest has the wrong pool count")

    # Compare by position as well as value. The order itself is part of the
    # contract: eval-2 cannot become eligible before eval-1 is spent.
    for index, (pool, expected_name, expected_seeds) in enumerate(
        zip(
            pools,
            config.evaluation_pool_names,
            config.evaluation_seeds,
            strict=True,
        )
    ):
        if not isinstance(pool, dict) or set(pool) != {
            "name",
            "seed_range",
            "status",
            "attempt",
        }:
            raise ValueError(f"evaluation pool entry {index} has an invalid structure")
        expected_range = {"start": expected_seeds.start, "end": expected_seeds.end}
        if pool["name"] != expected_name or pool["seed_range"] != expected_range:
            raise ValueError("evaluation pool manifest names or ranges were changed")
        status = pool["status"]
        if status not in _POOL_STATUSES:
            raise ValueError(f"evaluation pool {expected_name!r} has an invalid status")
        if status == POOL_AVAILABLE:
            if pool["attempt"] is not None:
                raise ValueError("an available evaluation pool cannot contain an attempt")
            continue
        _validate_attempt(pool["attempt"], status, expected_name)

    # A later pool can be spent only after all earlier pools were spent. This
    # makes manual reordering or a corrupt hole fail closed instead of silently
    # exposing a protected range out of sequence.
    seen_available = False
    for pool in pools:
        if pool["status"] == POOL_AVAILABLE:
            seen_available = True
        elif seen_available:
            raise ValueError("evaluation pools were not spent in ascending order")


def _validate_attempt(attempt: object, status: str, pool_name: str) -> None:
    """Check the lifecycle fields stored for one already-spent pool.

    An incomplete attempt has a checkpoint and start time but no completion or
    result. A complete attempt adds a finish time plus a passed/failed JSON file
    and its hash. Mixing fields from the two states indicates corrupt evidence.
    """

    if not isinstance(attempt, dict) or set(attempt) != {
        "id",
        "archive_identifier",
        "checkpoint_sha256",
        "started_at",
        "completed_at",
        "result",
    }:
        raise ValueError(f"evaluation pool {pool_name!r} has an invalid attempt")
    for field in ("id", "archive_identifier", "started_at"):
        if not isinstance(attempt[field], str) or not attempt[field].strip():
            raise ValueError(
                f"evaluation pool {pool_name!r} has an invalid {field}"
            )
    _validate_sha256(attempt["checkpoint_sha256"])
    if status == POOL_SPENT_INCOMPLETE:
        if attempt["completed_at"] is not None or attempt["result"] is not None:
            raise ValueError("an incomplete evaluation pool cannot contain a result")
        return

    if not isinstance(attempt["completed_at"], str) or not attempt["completed_at"].strip():
        raise ValueError("a completed evaluation pool requires completed_at")
    result = attempt["result"]
    if not isinstance(result, dict) or set(result) != {"outcome", "file", "sha256"}:
        raise ValueError("a completed evaluation pool has an invalid result")
    if result["outcome"] not in {"passed", "failed"}:
        raise ValueError("a completed evaluation pool has an invalid outcome")
    if not isinstance(result["file"], str) or not result["file"].strip():
        raise ValueError("a completed evaluation pool has an invalid result file")
    _validate_sha256(result["sha256"])


def _validate_sha256(value: object) -> None:
    """Recognize the text form of one SHA-256 digest.

    SHA-256 produces 32 bytes. Writing each byte as two hexadecimal characters
    produces the 64 lowercase characters stored in sidecars and manifests.
    This method checks the representation; callers separately hash file bytes.
    """

    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("SHA-256 fingerprint must be 64 lowercase hexadecimal characters")


def _read_manifest(path: Path) -> dict[str, Any]:
    """Decode the ledger file without guessing how damaged data should look.

    A missing file is meaningful to initialization but is an error everywhere
    else. Invalid JSON or a non-object top level blocks evaluation, preserving
    protected pools instead of silently constructing replacement state.
    """

    return read_json_object(path, "evaluation pool manifest")


@contextmanager
def _locked_manifest(path: Path) -> Iterator[None]:
    """Hold an operating-system file lock around one manifest operation.

    Code before ``yield`` creates and locks the sibling ``.lock`` file. The
    caller's ``with`` block then reads or changes the manifest. Code in
    ``finally`` unlocks it even if validation, writing, or evaluation setup
    raises an exception. The surrounding file ``with`` closes the lock handle.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    # ``a+`` creates the stable lock file if this is the first operation. The
    # file's contents are irrelevant; the kernel-managed lock is what matters.
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _utc_now() -> str:
    """Return the current time in an unambiguous JSON-friendly UTC form.

    Replacing ``+00:00`` with ``Z`` uses the standard short marker for UTC, so
    timestamps from machines in different local time zones remain comparable.
    """

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
