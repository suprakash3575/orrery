"""Checkpoint persistence tests for deterministic training continuation."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

import orrery.checkpoint as checkpoint_module
from orrery.checkpoint import (
    ActiveEpisodeAudit,
    CheckpointState,
    capture_random_states,
    load_checkpoint,
    recorded_checkpoint_fingerprint,
    restore_random_states,
    save_checkpoint,
    sha256_file,
)
from orrery.config import load_run_config
from orrery.env_client import OBSERVATION_DIMENSIONS


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "rl-v2-2p.toml"
FOUR_PLAYER_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "rl-v2-4p.toml"


def _state(
    *,
    environment_steps: int = 32_768,
    config_path: Path = CONFIG,
) -> CheckpointState:
    """Build a compact example containing every checkpoint information group.

    Real checkpoints contain a full neural network and AdamW optimizer. Small
    tensors and dictionaries are enough here because the persistence code must
    preserve their structure in exactly the same way regardless of size.
    """

    config = load_run_config(config_path)
    return CheckpointState(
        config=config,
        model_state={"weight": torch.tensor([1.0, 2.0])},
        optimizer_state={"step": 7},
        schedule_state={"learning_rate": 0.0002},
        random_states=capture_random_states(),
        environment_metadata={
            "engine": "abc123",
            "schema": "obs-v2",
            "generator": config.generator_version,
            "dimensions": dict(OBSERVATION_DIMENSIONS),
        },
        environment_steps=environment_steps,
        completed_updates=1,
        next_validation_step=5_000_000,
        validation_state={"best_wins": 117},
        metric_state={"launches_by_environment": {0: 3}},
        active_episodes=(
            ActiveEpisodeAudit(
                environment_id=0,
                seed=42,
                learner_seat=1,
                turn=83,
            ),
        ),
        seed_cursor=64,
    )


def test_checkpoint_round_trip_preserves_complete_state_and_fingerprint(
    tmp_path: Path,
) -> None:
    """Save a complete checkpoint, load it, and compare every information group.

    The test first writes the binary checkpoint, its text SHA-256 sidecar, and
    its safe JSON metadata envelope. It then loads through the public verification
    path and checks configuration, learner state, counters, validation data,
    active-game audit data, and the seed cursor. A passing result means an update
    boundary was not silently reduced to model weights alone.
    """

    path = tmp_path / "update-000001.pt"
    expected = _state()

    fingerprint = save_checkpoint(path, expected)
    restored = load_checkpoint(path, expected_config=expected.config)

    assert len(fingerprint) == 64
    assert fingerprint == sha256_file(path)
    assert fingerprint == recorded_checkpoint_fingerprint(path)
    assert (tmp_path / "update-000001.pt.sha256").read_text() == fingerprint + "\n"
    envelope = json.loads(
        (tmp_path / "update-000001.pt.metadata.json").read_text()
    )
    assert envelope["envelope_format"] == "checkpoint-envelope-v2"
    assert envelope["checkpoint_format"] == "checkpoint-v2"
    assert envelope["checkpoint_file"] == path.name
    assert envelope["checkpoint_size_bytes"] == path.stat().st_size
    assert envelope["sha256"] == fingerprint
    assert "git_commit" not in envelope["run_config"]
    assert envelope["run_config"]["release_required_wins"] == 900
    assert envelope["run_config"]["release_minimum_lower_bound"] == 0.85
    assert envelope["model_tensors"] == {
        "weight": {"dtype": "torch.float32", "shape": [2]}
    }
    assert restored.config == expected.config
    assert torch.equal(restored.model_state["weight"], expected.model_state["weight"])
    assert restored.optimizer_state == expected.optimizer_state
    assert restored.schedule_state == expected.schedule_state
    assert restored.environment_metadata == expected.environment_metadata
    assert restored.environment_steps == expected.environment_steps
    assert restored.completed_updates == expected.completed_updates
    assert restored.next_validation_step == expected.next_validation_step
    assert restored.validation_state == expected.validation_state
    assert restored.metric_state == expected.metric_state
    assert restored.active_episodes == expected.active_episodes
    assert restored.seed_cursor == expected.seed_cursor


@pytest.mark.parametrize(
    ("config_path", "players", "generator"),
    [(CONFIG, 2, "generator-v2"), (FOUR_PLAYER_CONFIG, 4, "generator-v2-4p")],
)
def test_both_configs_round_trip_with_their_identity(
    tmp_path: Path,
    config_path: Path,
    players: int,
    generator: str,
) -> None:
    """The envelope records the rl-v2 names; the payload keeps the dimensions."""

    path = tmp_path / "update-000001.pt"
    expected = _state(config_path=config_path)

    save_checkpoint(path, expected)
    restored = load_checkpoint(path, expected_config=expected.config)

    envelope = json.loads((tmp_path / "update-000001.pt.metadata.json").read_text())
    recorded = envelope["run_config"]
    assert recorded["contract"] == "rl-v2"
    assert recorded["players"] == players
    assert recorded["generator_version"] == generator
    assert recorded["protocol_version"] == "env-v3"
    assert recorded["observation_schema"] == "obs-v2"
    assert restored.config.players == players
    assert restored.environment_metadata["dimensions"] == OBSERVATION_DIMENSIONS


def test_checkpoint_explicitly_loads_complete_trusted_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep checkpoint loading stable across PyTorch's default-mode change.

    Orrery saves optimizer, scheduler, RNG, and dataclass state alongside
    model tensors. Newer PyTorch releases default to a weights-only loader that
    rejects those objects. Recording the call proves the loader deliberately
    requests complete deserialization instead of inheriting a version-dependent
    default. The production docstring limits this path to trusted local files.
    """

    path = tmp_path / "checkpoint.pt"
    expected = _state()
    save_checkpoint(path, expected)
    real_load = torch.load
    captured: dict[str, object] = {}

    def recording_load(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    restored = load_checkpoint(path, expected_config=expected.config)

    assert restored.environment_steps == expected.environment_steps
    assert captured["map_location"] == "cpu"
    assert captured["weights_only"] is False


def test_checkpoint_refuses_corrupted_file(tmp_path: Path) -> None:
    """Detect corruption before deserializing or trusting checkpoint contents.

    After a valid save, the test changes one byte without changing the file
    size and leaves its recorded fingerprint unchanged. The cheap size guard
    cannot detect this case, so hashing every byte must produce a mismatch.
    Rejecting the file prevents training from continuing with a damaged or
    manually altered learner state.
    """

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, _state())

    with path.open("r+b") as output:
        output.seek(-1, 2)
        final_byte = output.read(1)
        output.seek(-1, 2)
        output.write(bytes([final_byte[0] ^ 1]))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_checkpoint(path)


def test_checkpoint_rejects_wrong_size_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the envelope's byte count as a cheap guard before the full hash.

    The metadata is changed to claim that the checkpoint is one byte larger
    than the file on disk. The hash function is replaced with a failing stub;
    reaching the expected size error proves loading rejected the mismatch
    without reading the whole checkpoint first.
    """

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, _state())
    metadata_path = tmp_path / "checkpoint.pt.metadata.json"
    envelope = json.loads(metadata_path.read_text())
    envelope["checkpoint_size_bytes"] += 1
    metadata_path.write_text(json.dumps(envelope))
    monkeypatch.setattr(
        checkpoint_module,
        "sha256_file",
        lambda *_args, **_kwargs: pytest.fail("sha256_file must not be called"),
    )

    with pytest.raises(ValueError, match="checkpoint size mismatch"):
        load_checkpoint(path)


def test_checkpoint_refuses_a_different_run_configuration_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent one experiment from resuming under incompatible settings.

    The saved checkpoint belongs to a CUDA run, while the requested
    configuration says CPU. Even if its tensors can technically be loaded on
    the current machine, treating those configurations as the same experiment
    would break the recorded reproducibility contract, so loading must fail
    clearly.
    """

    path = tmp_path / "checkpoint.pt"
    state = _state()
    save_checkpoint(path, state)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("torch.load must not be called"),
    )

    with pytest.raises(ValueError, match="does not match the requested run"):
        load_checkpoint(path, expected_config=replace(state.config, device="cpu"))


def test_two_player_checkpoint_cannot_resume_as_a_four_player_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The player count is part of the run identity, like the rollout layout."""

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, _state())
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("torch.load must not be called"),
    )

    with pytest.raises(ValueError, match="does not match the requested run"):
        load_checkpoint(path, expected_config=load_run_config(FOUR_PLAYER_CONFIG))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        # A V1 record: its contract, protocol, schema and generator, with no
        # players field at all.
        (
            {
                "contract": "rl-v1",
                "protocol_version": "env-v2",
                "observation_schema": "obs-v1",
                "generator_version": "generator-v1",
                "players": None,
            },
            "missing keys: players",
        ),
        ({"contract": "rl-v1"}, "contract must be 'rl-v2'"),
        ({"protocol_version": "env-v2"}, "protocol_version must be 'env-v3'"),
        ({"players": 4}, "generator_version must be 'generator-v2-4p'"),
        ({"players": 3}, "players must be 2 or 4"),
    ],
)
def test_unsupported_recorded_identity_is_refused_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    message: str,
) -> None:
    """Refuse an incompatible checkpoint even when no expected config is given.

    Archived evaluation and checkpoint validation load without an expected
    config, so the recorded identity must be checked on its own. Rewriting the
    saved envelope stands in for a checkpoint recorded under another contract,
    since a current RunConfig cannot be built with these values at all. A None
    value removes the field.
    """

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, _state())
    metadata_path = tmp_path / "checkpoint.pt.metadata.json"
    envelope = json.loads(metadata_path.read_text())
    for name, value in changes.items():
        if value is None:
            del envelope["run_config"][name]
        else:
            envelope["run_config"][name] = value
    metadata_path.write_text(json.dumps(envelope))
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("torch.load must not be called"),
    )

    with pytest.raises(ValueError, match="unsupported run configuration") as caught:
        load_checkpoint(path)
    assert message in str(caught.value)


def test_checkpoint_refuses_incompatible_envelope_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read and reject an unsupported format without invoking pickle machinery."""

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, _state())
    metadata_path = tmp_path / "checkpoint.pt.metadata.json"
    envelope = json.loads(metadata_path.read_text())
    envelope["checkpoint_format"] = "checkpoint-v3"
    metadata_path.write_text(json.dumps(envelope))
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("torch.load must not be called"),
    )

    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(path)


def test_checkpoint_refuses_wrong_model_shape_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare the safe tensor signature with the model receiving the weights."""

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, _state())
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("torch.load must not be called"),
    )

    with pytest.raises(ValueError, match="tensor signature"):
        load_checkpoint(
            path,
            expected_model_state={"weight": torch.empty(3)},
        )


def test_checkpoint_reports_non_tensor_model_state_as_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep malformed model signatures inside load_checkpoint's error contract.

    A caller may accidentally provide a model-state mapping containing an
    ordinary Python value. Validation happens against the safe envelope before
    deserialization, and the public loader must describe malformed checkpoint
    inputs with ValueError consistently instead of leaking an implementation
    TypeError.
    """

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, _state())
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("torch.load must not be called"),
    )

    with pytest.raises(ValueError, match="must map string names to tensors"):
        load_checkpoint(path, expected_model_state={"weight": "not a tensor"})


def test_checkpoint_requires_metadata_envelope(tmp_path: Path) -> None:
    """Treat an interrupted save missing its final metadata file as unusable."""

    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"incomplete checkpoint")

    with pytest.raises(ValueError, match="metadata envelope is missing"):
        load_checkpoint(path)


def test_checkpoint_requires_explicit_replacement(tmp_path: Path) -> None:
    """Protect an existing checkpoint name from accidental overwriting.

    The second ordinary save uses the same path and must fail. The test then
    opts into replacement explicitly, verifies that the bytes have a different
    fingerprint, and loads the new step count. Training code never opts in;
    this escape hatch exists for deliberate low-level callers and tests.
    """

    path = tmp_path / "checkpoint.pt"
    first_fingerprint = save_checkpoint(path, _state())

    with pytest.raises(FileExistsError, match="already exists"):
        save_checkpoint(path, _state(environment_steps=65_536))

    second_fingerprint = save_checkpoint(
        path,
        _state(environment_steps=65_536),
        replace=True,
    )
    assert second_fingerprint != first_fingerprint
    assert load_checkpoint(path).environment_steps == 65_536


def test_random_state_round_trip_replays_each_rng_family() -> None:
    """Show that restoring RNG state reproduces the next random values.

    Python, NumPy, and Torch each own an independent random-number generator.
    The test captures all three states, records their next draws, advances them,
    then restores the snapshot. Receiving the original three draws again proves
    a checkpoint can continue the learner's stochastic choices deterministically.
    """

    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    states = capture_random_states()
    expected = (random.random(), np.random.random(), torch.rand(1))

    random.random()
    np.random.random()
    torch.rand(1)
    restore_random_states(states)

    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(1), expected[2])
