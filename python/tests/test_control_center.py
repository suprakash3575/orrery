"""The menu selects real artifacts and delegates lifecycle work safely."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from orrery.config import load_run_config
from scripts.control_center import main_menu


def _inputs(*answers: str):
    values = iter(answers)
    return lambda _prompt: next(values)


def _launcher(calls: list):
    def launch(command, cwd, environment):
        calls.append((list(command), cwd, environment))
        return 0

    return launch


def test_report_menu_reads_diagnosis_and_held_out_counts(tmp_path: Path) -> None:
    report = tmp_path / "runs" / "diagnosis" / "report.md"
    report.parent.mkdir(parents=True)
    report.write_text("# Diagnosis\n\n| Seat | Wins |\n| 0 | 7 |\n")
    result = tmp_path / "runs" / "archive" / "held-out-eval-1.json"
    result.parent.mkdir()
    result.write_text(json.dumps({"aggregate": {"wins": 8, "losses": 2, "draws": 0}, "passed": True}))

    shown: list[str] = []
    main_menu(tmp_path, read=_inputs("4", "2", "4", "1", "0"), show=shown.append)

    assert any("# Diagnosis" in line for line in shown)
    assert any("wins: 8; losses: 2; draws: 0; win rate: 80.0%" in line for line in shown)


SOURCE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "rl-v2-2p.toml"


@pytest.mark.parametrize(
    ("name", "device", "gomaxprocs"),
    [
        # The production file trains on CUDA without saying so in its name.
        ("rl-v2-2p.toml", "cuda", "32"),
        # A CPU copy keeps its CPU setting even when its name mentions CUDA.
        ("cuda-cpu-copy.toml", "cpu", None),
    ],
)
def test_start_training_uses_selected_contract_update_limit_and_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    device: str,
    gomaxprocs: str | None,
) -> None:
    monkeypatch.delenv("GOMAXPROCS", raising=False)
    configs = tmp_path / "configs"
    configs.mkdir()
    text = SOURCE_CONFIG.read_text().replace('device = "cuda"', f'device = "{device}"')
    (configs / name).write_text(text)
    calls: list = []

    main_menu(tmp_path, read=_inputs("1", "1", "3", "0"), show=lambda _: None, launch=_launcher(calls))

    assert len(calls) == 1
    command, cwd, environment = calls[0]
    assert command[1:] == ["-m", "orrery.train", "--config", str(configs / name), "--max-updates", "3"]
    assert cwd == tmp_path / "python"
    assert environment.get("GOMAXPROCS") == gomaxprocs


def test_start_training_refuses_an_unreadable_config_before_launching(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "broken.toml").write_text("placeholder")
    calls: list = []
    shown: list[str] = []

    main_menu(tmp_path, read=_inputs("1", "1", "3", "0"), show=shown.append, launch=_launcher(calls))

    assert calls == []
    assert any("Cannot read broken.toml" in line for line in shown)


def test_resume_requires_matching_config_and_existing_checkpoint(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "configs" / "rl-v2-2p.toml"
    configs = tmp_path / "configs"
    configs.mkdir()
    config = configs / "rl-v2-2p.toml"
    shutil.copyfile(source, config)
    run = tmp_path / "runs" / "example"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints" / "update-000050.pt"
    checkpoint.write_bytes(b"fixture")
    (run / "run-config.json").write_text(json.dumps(load_run_config(config).to_dict()))
    (run / "manifest.json").write_text(json.dumps({"checkpoints": [{"file": "checkpoints/update-000050.pt"}]}))
    calls: list = []

    main_menu(tmp_path, read=_inputs("2", "1", "1", "needed more validation", "0"), show=lambda _: None, launch=_launcher(calls))

    command = calls[0][0]
    assert command[1:] == [
        "-m", "orrery.train", "--config", str(config), "--resume", str(checkpoint),
        "--review-reason", "needed more validation",
    ]


def test_protected_evaluation_refuses_missing_spending_ledger(tmp_path: Path) -> None:
    archive = tmp_path / "runs" / "archive" / "release-one"
    archive.mkdir(parents=True)
    (archive / "archive.json").write_text(json.dumps({"checkpoint_sha256": "abc"}))
    (archive / "held-out-eval-1-old.json").write_text("{}")
    calls: list = []
    shown: list[str] = []

    main_menu(tmp_path, read=_inputs("5", "1", "0"), show=shown.append, launch=_launcher(calls))

    assert not calls
    assert any("restore the shared ledger" in line for line in shown)


def test_protected_evaluation_requires_exact_spend_phrase(tmp_path: Path) -> None:
    archive = tmp_path / "runs" / "archive" / "release-two"
    archive.mkdir(parents=True)
    (archive / "archive.json").write_text(json.dumps({"identifier": "release-two", "checkpoint_sha256": "new"}))
    ledger = tmp_path / "runs" / "evaluation-pools-v1.json"
    ledger.write_text(json.dumps({"pools": [
        {"name": "eval-1", "status": "spent-complete", "attempt": {"checkpoint_sha256": "old", "archive_identifier": "release-one"}},
        {"name": "eval-2", "status": "available", "attempt": None},
    ]}))
    calls: list = []

    main_menu(tmp_path, read=_inputs("5", "1", "yes", "5", "1", "SPEND eval-2", "0"), show=lambda _: None, launch=_launcher(calls))

    assert len(calls) == 1
    assert calls[0][0][1:] == ["-m", "orrery.evaluate", "--archive", str(archive), "--spend-eval", "eval-2"]


def test_menu_validates_selected_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "example"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints" / "update-000050.pt"
    checkpoint.write_bytes(b"fixture")
    (run / "run-config.json").write_text("{}")
    (run / "manifest.json").write_text(json.dumps({"checkpoints": [{"file": "checkpoints/update-000050.pt"}]}))
    calls: list = []

    main_menu(tmp_path, read=_inputs("7", "1", "1", "0"), show=lambda _: None, launch=_launcher(calls))

    command = calls[0][0]
    assert command[1:5] == ["-m", "scripts.validate_checkpoint", "--checkpoint", str(checkpoint)]
    assert Path(command[command.index("--out") + 1]).parent == tmp_path / "runs"
