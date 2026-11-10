"""Atomic, fingerprinted checkpoints for deterministic training continuation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from .config import RunConfig, run_config_from_mapping
from .jsonio import json_value, read_json_object, write_json_atomic, write_text_atomic


CHECKPOINT_ENVELOPE_FORMAT = "checkpoint-envelope-v2"
SUPPORTED_CHECKPOINT_FORMAT = "checkpoint-v2"


@dataclass(frozen=True)
class ActiveEpisodeAudit:
    """Identity of one live game deliberately abandoned during a later resume."""

    environment_id: int
    seed: int
    learner_seat: int
    turn: int


@dataclass(frozen=True)
class CheckpointState:
    """All learner and scheduler state saved at a completed PPO-update boundary."""

    config: RunConfig
    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    schedule_state: Mapping[str, Any]
    random_states: Mapping[str, Any]
    environment_metadata: Mapping[str, Any]
    environment_steps: int
    completed_updates: int
    next_validation_step: int
    validation_state: Mapping[str, Any]
    metric_state: Mapping[str, Any]
    active_episodes: tuple[ActiveEpisodeAudit, ...]
    seed_cursor: int

    def __post_init__(self) -> None:
        for name in (
            "environment_steps",
            "completed_updates",
            "next_validation_step",
            "seed_cursor",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def capture_random_states() -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU, and already-initialized CUDA RNGs."""

    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_random_states(states: Mapping[str, Any]) -> None:
    """Restore the exact RNG families present when a checkpoint was written."""

    required = {"python", "numpy", "torch_cpu"}
    missing = sorted(required - states.keys())
    if missing:
        raise ValueError(f"checkpoint random states missing {missing}")
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def save_checkpoint(
    path: str | Path,
    state: CheckpointState,
    *,
    replace: bool = False,
) -> str:
    """Atomically save one checkpoint plus its safe verification metadata.

    The Torch payload contains pickled Python objects, so its format cannot be
    inspected safely by loading it. The adjacent JSON envelope records the
    compatibility information needed before deserialization. A save interrupted
    between files may leave an incomplete set, but load_checkpoint rejects that
    set rather than treating it as a usable checkpoint.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise FileExistsError(f"checkpoint already exists: {destination}")

    payload = {"format": state.config.checkpoint_format, "state": state}
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        fingerprint = sha256_file(temporary)
        envelope = {
            "envelope_format": CHECKPOINT_ENVELOPE_FORMAT,
            "checkpoint_format": state.config.checkpoint_format,
            "checkpoint_file": destination.name,
            "checkpoint_size_bytes": temporary.stat().st_size,
            "sha256": fingerprint,
            "run_config": json_value(state.config.to_dict()),
            "model_tensors": _model_tensor_metadata(state.model_state),
        }
        os.replace(temporary, destination)
        write_text_atomic(_fingerprint_path(destination), fingerprint + "\n")
        write_json_atomic(_envelope_path(destination), envelope)
        return fingerprint
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | Path,
    *,
    expected_config: RunConfig | None = None,
    expected_model_state: Mapping[str, Any] | None = None,
) -> CheckpointState:
    """Verify and load a checkpoint created in this run's trusted local storage.

    ``torch.load`` deserializes Python objects and may execute their reduction
    code. The SHA-256 sidecar detects accidental corruption, but someone able
    to replace the checkpoint can also replace that unsigned sidecar. Callers
    must therefore never pass a downloaded or otherwise untrusted checkpoint.

    The JSON envelope is deliberately not pickled. Its format, run configuration,
    optional expected model signature, file size, and SHA-256 are checked before
    ``torch.load``. The deserialized state is checked against the same metadata
    afterward so the envelope cannot disagree silently with its bound payload.

    The recorded run configuration is always rebuilt through the same checks as
    a TOML file, whether or not the caller passes ``expected_config``. A
    checkpoint recorded under another contract, such as a V1 run or a
    mismatched player count and generator, is therefore refused before
    unpickling for every caller. Callers that load the weights into a policy
    should also pass ``expected_model_state``.
    """

    source = Path(path)
    envelope = _read_envelope(source)
    _validate_envelope(envelope, source)
    try:
        run_config_from_mapping(envelope["run_config"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"checkpoint records an unsupported run configuration: {error}"
        ) from error
    if expected_config is not None and envelope["run_config"] != json_value(
        expected_config.to_dict()
    ):
        raise ValueError("checkpoint run configuration does not match the requested run")
    if expected_model_state is not None and envelope["model_tensors"] != (
        _model_tensor_metadata(expected_model_state)
    ):
        raise ValueError("checkpoint model tensor signature does not match the requested model")

    recorded = recorded_checkpoint_fingerprint(source)
    if recorded != envelope["sha256"]:
        raise ValueError("checkpoint envelope and SHA-256 sidecar disagree")
    try:
        size = source.stat().st_size
    except FileNotFoundError as error:
        raise ValueError(f"checkpoint file is missing: {source}") from error
    # Reject a truncated or unexpectedly large file before reading all of its
    # bytes. This check is especially useful for real checkpoints, which may
    # be hundreds of megabytes; the later hash still detects same-size changes.
    if size != envelope["checkpoint_size_bytes"]:
        raise ValueError(
            "checkpoint size mismatch: "
            f"recorded {envelope['checkpoint_size_bytes']}, actual {size}"
        )
    actual = sha256_file(source)
    if recorded != actual:
        raise ValueError(f"checkpoint SHA-256 mismatch: recorded {recorded}, actual {actual}")
    # A training checkpoint contains the CheckpointState dataclass plus optimizer,
    # scheduler, and random-generator state; it is intentionally more than a map
    # of model weights. PyTorch 2.6 and newer default to ``weights_only=True``,
    # which rejects those trusted local objects. State the required mode explicitly
    # so loading behaves the same across the pinned macOS and Linux Torch versions.
    # The trust requirement is enforced by this function's documented contract:
    # never pass it a downloaded or otherwise untrusted checkpoint.
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != {"format", "state"}:
        raise ValueError("checkpoint has an invalid top-level structure")
    state = payload["state"]
    if not isinstance(state, CheckpointState):
        raise ValueError("checkpoint state has an incompatible type")
    if payload["format"] != state.config.checkpoint_format:
        raise ValueError("checkpoint format does not match its run configuration")
    if payload["format"] != envelope["checkpoint_format"]:
        raise ValueError("checkpoint payload format does not match its envelope")
    if json_value(state.config.to_dict()) != envelope["run_config"]:
        raise ValueError("checkpoint run configuration does not match its envelope")
    if _model_tensor_metadata(state.model_state) != envelope["model_tensors"]:
        raise ValueError("checkpoint model tensor signature does not match its envelope")
    if expected_config is not None and state.config != expected_config:
        raise ValueError("checkpoint run configuration does not match the requested run")
    return state


def sha256_file(path: str | Path) -> str:
    """Hash every byte of a file without loading the whole checkpoint into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Fingerprint model weights by their values, independent of any file.

    Two models with equal fingerprints hold identical weights. Each tensor adds
    its name, dtype, and shape, then its raw bytes, in sorted-name order, so the
    result does not depend on dictionary order, device, or pickling. This is
    not a checkpoint's file SHA-256: that also covers optimizer and training
    state, and an untrained model has no file at all.
    """

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to("cpu").contiguous()
        # The header fixes how many bytes follow, so one tensor's bytes can
        # never be mistaken for the start of the next tensor's.
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}\n".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def recorded_checkpoint_fingerprint(path: str | Path) -> str:
    """Read the SHA-256 sidecar associated with a checkpoint.

    This returns the recorded digest; it does not hash the checkpoint bytes.
    ``load_checkpoint`` performs that expensive verification before a trainer
    uses this helper to retain the already-verified value for resume evidence.
    """

    source = Path(path)
    fingerprint_path = _fingerprint_path(source)
    try:
        recorded = fingerprint_path.read_text().strip()
    except FileNotFoundError as error:
        raise ValueError(
            f"checkpoint SHA-256 sidecar is missing: {fingerprint_path}"
        ) from error
    if len(recorded) != 64 or any(
        character not in "0123456789abcdef" for character in recorded
    ):
        raise ValueError("checkpoint SHA-256 sidecar is invalid")
    return recorded


def _fingerprint_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _envelope_path(path: Path) -> Path:
    return path.with_name(path.name + ".metadata.json")


def _read_envelope(checkpoint: Path) -> dict[str, Any]:
    """Read safe compatibility metadata without touching the Torch payload."""

    path = _envelope_path(checkpoint)
    return read_json_object(path, "checkpoint metadata envelope")


def _validate_envelope(envelope: Mapping[str, Any], checkpoint: Path) -> None:
    """Reject incompatible or malformed metadata before pickle deserialization."""

    expected_fields = {
        "envelope_format",
        "checkpoint_format",
        "checkpoint_file",
        "checkpoint_size_bytes",
        "sha256",
        "run_config",
        "model_tensors",
    }
    if set(envelope) != expected_fields:
        raise ValueError("checkpoint metadata envelope has an invalid structure")
    if envelope["envelope_format"] != CHECKPOINT_ENVELOPE_FORMAT:
        raise ValueError(
            f"unsupported checkpoint envelope format {envelope['envelope_format']!r}"
        )
    if envelope["checkpoint_format"] != SUPPORTED_CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format {envelope['checkpoint_format']!r}"
        )
    if envelope["checkpoint_file"] != checkpoint.name:
        raise ValueError("checkpoint metadata envelope names a different file")
    size = envelope["checkpoint_size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("checkpoint metadata envelope has an invalid file size")
    fingerprint = envelope["sha256"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError("checkpoint metadata envelope has an invalid SHA-256")
    if not isinstance(envelope["run_config"], dict):
        raise ValueError("checkpoint metadata envelope has an invalid run configuration")
    if not isinstance(envelope["model_tensors"], dict):
        raise ValueError("checkpoint metadata envelope has an invalid model tensor signature")


def _model_tensor_metadata(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Describe model tensor names, shapes, and dtypes in JSON-safe values."""

    metadata: dict[str, dict[str, Any]] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("model state must map string names to tensors")
        metadata[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return metadata


def _temporary_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)
