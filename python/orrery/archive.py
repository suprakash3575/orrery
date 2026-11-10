"""Create self-contained immutable local archives for passing checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .checkpoint import load_checkpoint, recorded_checkpoint_fingerprint
from .config import RunConfig
from .model import OrbitPolicy
from .jsonio import (
    fsync_directory,
    json_value,
    read_json_object,
    write_json_exclusive,
    write_text_exclusive,
)


FROZEN_ARCHIVE_FORMAT = "frozen-checkpoint-archive-v1"


@dataclass(frozen=True)
class FrozenCheckpointArchive:
    """Paths and identity of one completed immutable checkpoint archive."""

    # Stable directory name derived from the training boundary and full digest.
    identifier: str
    # Directory containing every file required to inspect or load the candidate.
    directory: Path
    # Independent checkpoint copy used by later held-out evaluation.
    checkpoint: Path
    # SHA-256 of every byte in both the source and archived checkpoint.
    sha256: str

    def evidence_reference(self) -> dict[str, object]:
        """Return the location-independent identity stored in validation JSON."""

        return {
            "format": FROZEN_ARCHIVE_FORMAT,
            "identifier": self.identifier,
            "checkpoint": self.checkpoint.name,
            "checkpoint_sha256": self.sha256,
        }


def freeze_passing_checkpoint(
    source_checkpoint: str | Path,
    archive_root: str | Path,
    *,
    fingerprint: str,
    config: RunConfig,
    validation_record: Mapping[str, Any],
    completed_updates: int,
) -> FrozenCheckpointArchive:
    """Copy a passing release checkpoint into a new self-contained archive.

    Source files are completely verified before archive publication begins.
    Files are assembled in a unique staging directory, flushed, and atomically
    renamed to the final identifier only after ``archive.json`` marks the staged
    copy complete. A failed attempt can therefore be retried without a partial
    final directory blocking it. Repeating a successful freeze verifies and
    returns the existing immutable archive instead of replacing its bytes.
    """

    source = Path(source_checkpoint)
    _validate_release_record(validation_record, source, fingerprint)
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or completed_updates < 0
    ):
        raise ValueError("completed_updates must be a non-negative integer")

    # load_checkpoint validates the safe envelope, file size, sidecar, actual
    # bytes, configuration, model signature, and payload in one pass before
    # archive work starts.
    state = load_checkpoint(
        source,
        expected_config=config,
        expected_model_state=OrbitPolicy().state_dict(),
    )
    if recorded_checkpoint_fingerprint(source) != fingerprint:
        raise ValueError("passing checkpoint fingerprint does not match its sidecar")

    metadata_path = source.with_name(source.name + ".metadata.json")
    metadata = read_json_object(
        metadata_path,
        "passing checkpoint metadata envelope",
    )
    expected_config = json_value(config.to_dict())

    environment_steps = validation_record.get("training_environment_steps")
    if (
        isinstance(environment_steps, bool)
        or not isinstance(environment_steps, int)
        or environment_steps < 0
    ):
        raise ValueError("release validation has an invalid training step count")
    if state.environment_steps != environment_steps:
        raise ValueError(
            "release validation training steps do not match the checkpoint"
        )
    if state.completed_updates != completed_updates:
        raise ValueError("completed_updates does not match the checkpoint")
    identifier = f"release-{environment_steps:012d}-{fingerprint}"
    root = Path(archive_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / identifier

    if destination.exists():
        return _verify_existing_archive(
            destination,
            source_name=source.name,
            identifier=identifier,
            fingerprint=fingerprint,
            config=config,
            environment_steps=environment_steps,
            completed_updates=completed_updates,
        )

    # A unique staging name means an interrupted copy never occupies the final
    # identifier. We deliberately leave an interrupted staging directory for
    # diagnosis; the next attempt uses a different name and remains recoverable.
    staging = Path(
        tempfile.mkdtemp(prefix=f".{identifier}.", suffix=".tmp", dir=root)
    )
    archived_checkpoint = staging / source.name
    archived_fingerprint = _copy_exclusive(source, archived_checkpoint)
    write_text_exclusive(
        staging / f"{source.name}.sha256",
        fingerprint + "\n",
    )
    write_json_exclusive(
        staging / f"{source.name}.metadata.json",
        metadata,
    )
    write_json_exclusive(staging / "run-config.json", expected_config)

    archive = FrozenCheckpointArchive(
        identifier=identifier,
        directory=destination,
        checkpoint=destination / source.name,
        sha256=fingerprint,
    )
    archived_validation = dict(validation_record)
    archived_validation["archive"] = archive.evidence_reference()
    write_json_exclusive(
        staging / "release-validation.json",
        archived_validation,
    )
    # Re-hash the independent copy after every file has been written. This
    # proves the archived candidate, rather than only its training source, owns
    # the recorded fingerprint. The manifest is written only after this passes,
    # so its presence marks a complete archive.
    if archived_fingerprint != fingerprint:
        raise OSError(
            "archived checkpoint SHA-256 mismatch: "
            f"recorded {fingerprint}, actual {archived_fingerprint}"
        )
    # Persist every non-marker directory entry before publishing the marker.
    # This ordering keeps archive.json meaningful even across a power loss.
    fsync_directory(staging)
    write_json_exclusive(
        staging / "archive.json",
        {
            "format": FROZEN_ARCHIVE_FORMAT,
            "identifier": identifier,
            "immutable": True,
            "checkpoint": source.name,
            "checkpoint_sha256": fingerprint,
            "checkpoint_size_bytes": archived_checkpoint.stat().st_size,
            "checkpoint_metadata": f"{source.name}.metadata.json",
            "checkpoint_fingerprint": f"{source.name}.sha256",
            "run_config": "run-config.json",
            "release_validation": "release-validation.json",
            "training_environment_steps": environment_steps,
            "completed_updates": completed_updates,
        },
    )
    fsync_directory(staging)

    # Renaming a complete directory publishes all files under the deterministic
    # identifier in one operation. Flushing the root makes that rename durable.
    try:
        os.rename(staging, destination)
    except OSError as error:
        if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
            raise
        # Another identical freezer may have won the race. Never replace it;
        # verify its complete contents before treating this call as successful.
        return _verify_existing_archive(
            destination,
            source_name=source.name,
            identifier=identifier,
            fingerprint=fingerprint,
            config=config,
            environment_steps=environment_steps,
            completed_updates=completed_updates,
        )
    fsync_directory(root)
    return archive


def _verify_existing_archive(
    destination: Path,
    *,
    source_name: str,
    identifier: str,
    fingerprint: str,
    config: RunConfig,
    environment_steps: int,
    completed_updates: int,
) -> FrozenCheckpointArchive:
    """Return an existing archive only when every immutable receipt agrees.

    This is the idempotent recovery path used after training saved a passing
    checkpoint but failed before publishing its evidence. A partial, changed,
    or unrelated directory fails closed instead of being overwritten.
    """

    archive_path = destination / "archive.json"
    try:
        manifest = read_json_object(archive_path, "existing archive manifest")
    except ValueError as error:
        if archive_path.exists():
            raise
        raise ValueError(
            f"existing archive is incomplete: {archive_path} is missing"
        ) from error

    archived_checkpoint = destination / source_name
    try:
        source_checkpoint_size = archived_checkpoint.stat().st_size
    except FileNotFoundError as error:
        raise ValueError("existing archive checkpoint is missing") from error

    expected_manifest = {
        "format": FROZEN_ARCHIVE_FORMAT,
        "identifier": identifier,
        "immutable": True,
        "checkpoint": source_name,
        "checkpoint_sha256": fingerprint,
        "checkpoint_size_bytes": source_checkpoint_size,
        "checkpoint_metadata": f"{source_name}.metadata.json",
        "checkpoint_fingerprint": f"{source_name}.sha256",
        "run_config": "run-config.json",
        "release_validation": "release-validation.json",
        "training_environment_steps": environment_steps,
        "completed_updates": completed_updates,
    }
    if not isinstance(manifest, dict) or manifest != expected_manifest:
        raise ValueError("existing archive does not match the freeze request")

    state = load_checkpoint(
        archived_checkpoint,
        expected_config=config,
        expected_model_state=OrbitPolicy().state_dict(),
    )
    if (
        state.environment_steps != environment_steps
        or state.completed_updates != completed_updates
    ):
        raise ValueError("existing archive checkpoint has a different training boundary")
    if source_checkpoint_size != archived_checkpoint.stat().st_size:
        raise ValueError("existing archive checkpoint size does not match its manifest")
    if recorded_checkpoint_fingerprint(archived_checkpoint) != fingerprint:
        raise ValueError("existing archive checkpoint has a different SHA-256")

    try:
        archived_config = read_json_object(
            destination / "run-config.json",
            "existing archived run configuration",
        )
        validation = read_json_object(
            destination / "release-validation.json",
            "existing archive release validation",
        )
    except ValueError as error:
        raise ValueError("existing archive is missing required evidence") from error
    archive = FrozenCheckpointArchive(
        identifier=identifier,
        directory=destination,
        checkpoint=archived_checkpoint,
        sha256=fingerprint,
    )
    if archived_config != json_value(config.to_dict()):
        raise ValueError("existing archive has a different run configuration")
    if (
        not isinstance(validation, dict)
        or validation.get("gate") != "release-readiness"
        or validation.get("passed") is not True
        or validation.get("checkpoint") != source_name
        or validation.get("checkpoint_sha256") != fingerprint
        or validation.get("archive") != archive.evidence_reference()
    ):
        raise ValueError("existing archive lacks matching release validation")
    return archive


def _validate_release_record(
    record: Mapping[str, Any],
    checkpoint: Path,
    fingerprint: str,
) -> None:
    """Require evidence that selected this exact checkpoint for freezing."""

    if record.get("gate") != "release-readiness" or record.get("passed") is not True:
        raise ValueError("only a passing release-readiness checkpoint can be frozen")
    if record.get("checkpoint") != checkpoint.name:
        raise ValueError("release validation names a different checkpoint")
    if record.get("checkpoint_sha256") != fingerprint:
        raise ValueError("release validation names a different checkpoint SHA-256")


def _copy_exclusive(source: Path, destination: Path) -> str:
    """Copy, hash, and flush bytes to a new file without reading them twice."""

    digest = hashlib.sha256()
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            output_file.write(chunk)
            digest.update(chunk)
        output_file.flush()
        os.fsync(output_file.fileno())
    return digest.hexdigest()
