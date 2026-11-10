"""The rl-v2 diagnosis must play the requested games and record what it played."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orrery.checkpoint import state_dict_sha256
from orrery.config import load_run_config
from orrery.env_client import (
    BODY_FEATURES,
    BODY_SLOTS,
    END_GAME_FINISHED,
    END_LEARNER_ELIMINATED,
    OBSERVATION_DIMENSIONS,
    PLAYER_FEATURES,
    PLAYER_SLOTS,
    Observation,
    SlotAction,
)
from orrery.evaluator import EvaluationOutcome, EvaluationResult, GameEndRecord, TurnRecord
from orrery.train import initial_policy
from scripts.diagnose import (
    FEATURE_OWNER_ME,
    ROLES,
    FinalState,
    GameDiagnosis,
    TurnSummary,
    diagnose_games,
    game_measurements,
)

# A complete opponent spec that differs from the four-player run's
# heuristic-v2 training opponent.
RANDOM_OPPONENT = "random-v1:launch_probability=0.3,max_fraction=0.8,min_fraction=0.2"


def _board(my_planets: int) -> Observation:
    """A four-player board with ``my_planets`` learner planets and one enemy planet.

    Only the fields the diagnosis reads are filled in: which slots are real
    planets, who owns them, and which player rows are present.
    """

    bodies = np.zeros((BODY_SLOTS, BODY_FEATURES), dtype=np.float32)
    body_mask = np.zeros(BODY_SLOTS, dtype=np.bool_)
    for slot in range(my_planets):
        bodies[slot, FEATURE_OWNER_ME] = 1.0
    # One planet owned by the clockwise opponent, the next ownership column.
    bodies[my_planets, FEATURE_OWNER_ME + 1] = 1.0
    body_mask[: my_planets + 1] = True
    return Observation(
        bodies,
        np.zeros((PLAYER_SLOTS, PLAYER_FEATURES), dtype=np.float32),
        body_mask,
        body_mask & (bodies[:, FEATURE_OWNER_ME] > 0.5),
        np.ones(PLAYER_SLOTS, dtype=np.bool_),
        np.zeros((BODY_SLOTS, BODY_SLOTS), dtype=np.bool_),
        np.arange(BODY_SLOTS, dtype=np.int32),
    )


def _fake_runner(calls: list, *, start_planets: int, final_planets: int, turns: int, end_reason: int):
    """A runner that plays every game as ``turns`` held turns, then a final board.

    It reports to the diagnosis exactly as the real evaluator does: one
    on_turn record per decision, then one on_game_end record, and an outcome
    whose metadata echoes what Go would confirm.
    """

    def runner(_policy, **kwargs):
        calls.append(kwargs)
        seeds = kwargs["seeds"]
        outcomes = []
        for seed in range(seeds.start, seeds.end):
            for seat in range(4):
                for turn in range(turns):
                    kwargs["on_turn"](TurnRecord(
                        seat, seed, seat, turn, _board(start_planets), -0.9,
                        tuple(SlotAction() for _ in range(BODY_SLOTS)),
                    ))
                kwargs["on_game_end"](GameEndRecord(
                    seat, seed, seat, turns, _board(final_planets), -1.0, end_reason,
                ))
                outcomes.append(EvaluationOutcome(seed, seat, -1.0, turns, end_reason, 0))
        return EvaluationResult(
            tuple(outcomes),
            {
                **kwargs["expected_metadata"], "engine": "test",
                "dimensions": OBSERVATION_DIMENSIONS,
                "evaluation": {"policy": "eval-policy-v2", "device": "cpu"},
            },
            0,
        )

    return runner


class _Policy:
    """Stands in for OrbitPolicy: the fake runner never calls the model."""

    def to(self, **_kwargs):
        return self

    def eval(self):
        return self

    def state_dict(self):
        return {}

    def load_state_dict(self, _state, *, strict):
        assert strict


def _run(tmp_path: Path):
    """A four-player run directory with one checkpoint and its manifest entry."""

    config = load_run_config(Path(__file__).resolve().parents[2] / "configs" / "rl-v2-4p.toml")
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
    return run, checkpoint, state


def test_explicit_opponent_reaches_the_runner_and_the_report(tmp_path: Path) -> None:
    """--opponent replaces the training opponent for these games only.

    The fake runner echoes the expected metadata, as Go does when it accepts
    the opponent.
    """

    run, checkpoint, state = _run(tmp_path)
    saved_config = (run / "run-config.json").read_bytes()
    calls = []
    runner = _fake_runner(calls, start_planets=1, final_planets=1, turns=1, end_reason=END_GAME_FINISHED)

    report, _charted = diagnose_games(
        checkpoint=checkpoint,
        seeds=1,
        device="cpu",
        out=tmp_path / "diagnosis",
        charts=False,
        game_charts=0,
        traced_wins=0,
        opponent=RANDOM_OPPONENT,
        runner=runner,
        loader_options={"loader": lambda _path, **_kwargs: state, "policy_factory": _Policy},
    )

    assert state.config.training_opponent != RANDOM_OPPONENT
    assert calls[0]["opponent"] == RANDOM_OPPONENT
    assert calls[0]["expected_metadata"]["opponent"] == RANDOM_OPPONENT
    assert report["opponent"] == RANDOM_OPPONENT
    assert (run / "run-config.json").read_bytes() == saved_config


def test_untrained_baseline_rebuilds_identically_from_its_learner_seed() -> None:
    """The baseline is only a fair comparison if every rebuild is the same model.

    No update-0 checkpoint exists, so its weights are recreated from the learner
    seed each time. The fingerprint proves two rebuilds match, and a different
    seed must give a different model, or the fingerprint would prove nothing.
    """

    first = state_dict_sha256(initial_policy(1501).state_dict())
    again = state_dict_sha256(initial_policy(1501).state_dict())
    other_seed = state_dict_sha256(initial_policy(1502).state_dict())

    assert first == again
    assert first != other_seed


def test_a_planet_lost_on_the_final_move_appears_in_the_final_board(tmp_path: Path) -> None:
    """The last move's result must be recorded, not just the boards before moves.

    Every game holds one planet on both of its decision boards (turns 0 and 1),
    then loses it on the final move and is eliminated. Only the final board,
    at turn 2, shows zero planets: without it the trace would end with the
    learner apparently still holding its home.
    """

    run, checkpoint, state = _run(tmp_path)
    out = tmp_path / "diagnosis"
    runner = _fake_runner(
        [], start_planets=1, final_planets=0, turns=2, end_reason=END_LEARNER_ELIMINATED
    )

    diagnose_games(
        checkpoint=checkpoint,
        seeds=1,
        device="cpu",
        out=out,
        charts=False,
        game_charts=0,
        traced_wins=0,
        runner=runner,
        loader_options={"loader": lambda _path, **_kwargs: state, "policy_factory": _Policy},
    )

    lines = [json.loads(line) for line in (out / "traces" / "seed-1000000-seat-0.ndjson").read_text().splitlines()]
    assert [line["turn"] for line in lines] == [0, 1, 2]
    assert [line["planets"]["me"] for line in lines] == [1, 1, 0]
    assert "final" not in lines[1]
    assert lines[-1]["final"] is True
    assert lines[-1]["result"] == "loss"


def _measured_game(
    planets: list[int], comets: list[int], final_planets: int, *, eliminated: bool
) -> GameDiagnosis:
    """A game whose learner holds ``planets[t]`` planets and ``comets[t]`` comets
    on the board before move t, then ``final_planets`` after the last move.

    Only the fields the measurements read are meaningful.
    """

    rows = len(ROLES)
    summaries = tuple(
        TurnSummary(
            turn=turn,
            value=-0.9,
            planets=(held,) + (1,) * (rows - 1),
            comets=(comets[turn],) + (0,) * (rows - 1),
            neutral_planets=0,
            ships_scaled=(0.0,) * rows,
            present=(True,) * rows,
            launches=(),
            launchable_sources=0,
        )
        for turn, held in enumerate(planets)
    )
    final = FinalState(
        turn=len(planets),
        planets=(final_planets,) + (1,) * (rows - 1),
        comets=(0,) * rows,
        neutral_planets=0,
        ships_scaled=(0.0,) * rows,
        present=(True,) * rows,
    )
    return GameDiagnosis(1_000_000, 0, -1.0, len(planets), eliminated, summaries, final)


@pytest.mark.parametrize(
    ("planets", "comets", "final_planets", "eliminated", "most", "first_two", "without"),
    [
        # Captures a second planet on the very last move: only the final board
        # shows two planets, so both measures depend on recording it.
        pytest.param([1, 1, 1], [0, 0, 0], 2, False, 2, 3, 0, id="capture-on-final-move"),
        # Loses the home, then retakes one planet. Two of four action turns had
        # no productive body, and one planet again is not two at once.
        pytest.param([1, 0, 0, 1], [0, 0, 0, 0], 1, False, 1, None, 2, id="loss-then-recovery"),
        # Eliminated before ever expanding: "not reached" is None, never turn 0.
        pytest.param([1, 1], [0, 0], 0, True, 1, None, 0, id="early-ending"),
        # A comet without any planet still produces ships, so it is productive.
        pytest.param([0, 0], [1, 0], 0, True, 0, None, 1, id="comet-only"),
    ],
)
def test_game_measurements_follow_the_definitions(
    planets: list[int],
    comets: list[int],
    final_planets: int,
    eliminated: bool,
    most: int,
    first_two: int | None,
    without: int,
) -> None:
    """Each case is small enough to check by hand against the definitions."""

    row = game_measurements(_measured_game(planets, comets, final_planets, eliminated=eliminated))

    assert row["most_planets"] == most
    assert row["first_two_planets_turn"] == first_two
    assert (row["turns_without_productive_body"], row["action_turns"]) == (without, len(planets))
    assert row["share_without_productive_body"] == pytest.approx(without / len(planets))
    assert row["end_reason"] == ("learner_eliminated" if eliminated else "game_finished")
