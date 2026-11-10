"""Checkpoint validation reports must use the selected run and reusable seeds."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from orrery.config import load_run_config
from orrery.env_client import (
    END_GAME_FINISHED,
    END_LEARNER_ELIMINATED,
    OBSERVATION_DIMENSIONS,
)
from orrery.evaluator import (
    NOMINAL_INTERVAL_ASSUMPTION,
    EvaluationOutcome,
    EvaluationResult,
)
from scripts.validate_checkpoint import validate_checkpoint

# What the real runner records about how it chose actions. Fake runners
# report it too, because the report refuses results from another evaluator.
_EVALUATION = {"policy": "eval-policy-v2", "device": "cpu"}


class _Policy:
    def to(self, **_kwargs):
        return self

    def eval(self):
        return self

    def state_dict(self):
        return {}

    def load_state_dict(self, _state, *, strict):
        assert strict


def _fixture(tmp_path: Path, config_name: str = "rl-v2-2p.toml"):
    config = load_run_config(Path(__file__).resolve().parents[2] / "configs" / config_name)
    run = tmp_path / "runs" / "example"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints" / "update-000050.pt"
    checkpoint.write_bytes(b"fixture")
    fingerprint = "a" * 64
    (run / "checkpoints" / "update-000050.pt.sha256").write_text(fingerprint + "\n")
    (run / "run-config.json").write_text(json.dumps(config.to_dict()))
    (run / "manifest.json").write_text(json.dumps({"checkpoints": [{
        "file": "checkpoints/update-000050.pt", "sha256": fingerprint,
        "completed_updates": 50, "environment_steps": 1_638_400,
    }]}))
    state = SimpleNamespace(config=config, model_state={}, completed_updates=50, environment_steps=1_638_400)
    return checkpoint, state


def test_validate_checkpoint_writes_rotated_report_without_spending_pool(tmp_path: Path) -> None:
    checkpoint, state = _fixture(tmp_path)
    calls = []

    def runner(_policy, **kwargs):
        calls.append(kwargs)
        seeds = kwargs["seeds"]
        outcomes = tuple(
            EvaluationOutcome(seed, seat, 1.0 if seat == 0 else -1.0, 80, END_GAME_FINISHED, 1)
            for seed in range(seeds.start, seeds.end)
            for seat in (0, 1)
        )
        return EvaluationResult(
            outcomes,
            {
                **kwargs["expected_metadata"], "engine": "test",
                "dimensions": OBSERVATION_DIMENSIONS, "evaluation": _EVALUATION,
            },
            0,
        )

    loads = []

    def loader(path, **kwargs):
        loads.append(kwargs)
        return state

    out = tmp_path / "runs" / "validation-example"
    # The run config asks for CUDA; this test runs on the CPU explicitly.
    report = validate_checkpoint(
        checkpoint, out, runner=runner, loader=loader, policy_factory=_Policy, device="cpu",
    )

    # The policy's tensor signature reaches the loader, so another
    # architecture is refused before the checkpoint is unpickled.
    assert loads == [{"expected_model_state": {}}]
    assert (calls[0]["seeds"].start, calls[0]["seeds"].end) == (1_000_000, 1_000_500)
    assert calls[0]["evaluation_policy"] == "eval-policy-v2"
    assert calls[0]["device"].type == "cpu"
    assert "Evaluator: `eval-policy-v2` on cpu" in (out / "report.md").read_text()
    assert calls[0]["opponent"] == state.config.training_opponent
    assert calls[0]["players"] == 2
    assert "protocol_version" not in calls[0]
    assert report["wins"] == 500
    assert report["losses"] == 500
    assert report["results_by_seat"]["0"]["wins"] == 500
    assert "500 wins" in (out / "report.md").read_text()
    saved = json.loads((out / "report.json").read_text())
    assert len(saved["outcomes"]) == 1_000
    assert saved["confidence_interval"]["nominal"] is True
    assert saved["confidence_interval"]["assumption"] == NOMINAL_INTERVAL_ASSUMPTION
    assert not (tmp_path / "runs" / "evaluation-pools-v1.json").exists()


def test_validate_checkpoint_reports_every_seat_of_a_four_player_run(tmp_path: Path) -> None:
    """One row per seat, with eliminations counted inside the losses.

    The learner wins from seat 0, loses on ships from seats 1 and 2, and is
    eliminated from seat 3, on every one of the 500 boards.
    """

    checkpoint, state = _fixture(tmp_path, "rl-v2-4p.toml")
    calls = []

    def runner(_policy, **kwargs):
        calls.append(kwargs)
        seeds = kwargs["seeds"]
        outcomes = tuple(
            EvaluationOutcome(
                seed,
                seat,
                1.0 if seat == 0 else -1.0,
                80,
                END_LEARNER_ELIMINATED if seat == 3 else END_GAME_FINISHED,
                1,
            )
            for seed in range(seeds.start, seeds.end)
            for seat in range(4)
        )
        return EvaluationResult(
            outcomes,
            {
                **kwargs["expected_metadata"], "engine": "test",
                "dimensions": OBSERVATION_DIMENSIONS, "evaluation": _EVALUATION,
            },
            0,
        )

    out = tmp_path / "runs" / "validation-four-player"
    report = validate_checkpoint(
        checkpoint, out, runner=runner, loader=lambda _path, **_kwargs: state, policy_factory=_Policy,
        device="cpu",
    )

    assert calls[0]["players"] == 4
    assert report["games"] == 2_000
    assert report["results_by_seat"] == {
        "0": {"wins": 500, "losses": 0, "draws": 0, "eliminated": 0},
        "1": {"wins": 0, "losses": 500, "draws": 0, "eliminated": 0},
        "2": {"wins": 0, "losses": 500, "draws": 0, "eliminated": 0},
        "3": {"wins": 0, "losses": 500, "draws": 0, "eliminated": 500},
    }
    # Every seat-3 game is an elimination, and every game launched once.
    assert report["summary"]["eliminations"] == 500
    assert report["summary"]["zero_launch_games"] == 0
    markdown = (out / "report.md").read_text()
    assert "every seat" in markdown
    assert "| 3 | 0 | 500 | 0 | 500 |" in markdown
    assert "| Learner eliminations | 500 |" in markdown
    # The learner wins every game from seat 0 and none from the others.
    assert report["summary"]["seat_win_rate"] == [1.0, 0.0, 0.0, 0.0]
    assert "| Win rate by seat | 0: 100.0% / 1: 0.0% / 2: 0.0% / 3: 0.0% |" in markdown


def test_validate_checkpoint_plays_only_the_requested_seed_prefix(tmp_path: Path) -> None:
    """``--seeds 2`` plays the first two seeds from every seat and says so.

    Two seeds on a two-player map are four games. The report records that
    seed range, and the release thresholds are reported as not met because
    the sample is smaller than the release sample.
    """

    checkpoint, state = _fixture(tmp_path)
    calls = []

    def runner(_policy, **kwargs):
        calls.append(kwargs)
        seeds = kwargs["seeds"]
        outcomes = tuple(
            EvaluationOutcome(seed, seat, 1.0, 80, END_GAME_FINISHED, 1)
            for seed in range(seeds.start, seeds.end)
            for seat in (0, 1)
        )
        return EvaluationResult(
            outcomes,
            {
                **kwargs["expected_metadata"], "engine": "test",
                "dimensions": OBSERVATION_DIMENSIONS, "evaluation": _EVALUATION,
            },
            0,
        )

    report = validate_checkpoint(
        checkpoint, tmp_path / "report", runner=runner, loader=lambda _path, **_kwargs: state,
        policy_factory=_Policy, device="cpu", seeds=2,
    )

    assert (calls[0]["seeds"].start, calls[0]["seeds"].end) == (1_000_000, 1_000_002)
    assert report["seed_range"] == [1_000_000, 1_000_002]
    assert report["games"] == 4
    assert report["release_thresholds_met"] is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"device": "cpu", "seeds": 0}, "seeds must be between 1 and 500"),
        ({"device": "cpu", "seeds": 501}, "seeds must be between 1 and 500"),
        ({"device": "cuda"}, "CUDA is not available"),
    ],
    ids=["zero seeds", "more seeds than the release sample", "no CUDA"],
)
def test_validate_checkpoint_rejects_bad_options_before_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: dict, message: str
) -> None:
    """Refuse an impossible seed count or a missing GPU before any game starts.

    CUDA is reported unavailable, so the test behaves the same on any machine.
    """

    checkpoint, state = _fixture(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match=message):
        validate_checkpoint(
            checkpoint, tmp_path / "report", runner=lambda *_args, **_kwargs: pytest.fail("games started"),
            loader=lambda _path, **_kwargs: state, policy_factory=_Policy, **arguments,
        )
    assert not (tmp_path / "report").exists()


def test_validate_checkpoint_rejects_manifest_mismatch_before_games(tmp_path: Path) -> None:
    checkpoint, state = _fixture(tmp_path)
    manifest = checkpoint.parent.parent / "manifest.json"
    data = json.loads(manifest.read_text())
    data["checkpoints"][0]["sha256"] = "b" * 64
    manifest.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="fingerprint disagree"):
        validate_checkpoint(
            checkpoint, tmp_path / "report", runner=lambda *_args, **_kwargs: pytest.fail("games started"),
            loader=lambda _path, **_kwargs: state, policy_factory=_Policy,
        )
    assert not (tmp_path / "report").exists()
