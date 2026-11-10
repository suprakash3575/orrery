"""Deterministic decoding and complete rotated-evaluation tests."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import orrery.evaluator as evaluator_module
from orrery.config import SeedRange
from orrery.env_client import (
    BODY_FEATURES, BODY_SLOTS, END_GAME_FINISHED, END_LEARNER_ELIMINATED,
    END_ONGOING, PLAYER_FEATURES, PLAYER_SLOTS, SCHEDULE_ROTATE, EnvClient, Episode,
    Observation, SlotAction, StepResult,
)
from orrery.evaluator import (
    EVALUATION_POLICY_V2, PROGRESS_INTERVAL_SECONDS, EvaluationOutcome,
    EvaluationResult, RunnerTimings,
    GameEndRecord, TurnRecord, ValidationProgress, clopper_pearson_interval,
    decode_evaluation_targets, pipeline_gate,
    prepare_evaluation_model, release_readiness_gate, require_complete_results, require_finite_scores,
    run_rotated_matches, timings_evidence, validation_summary,
)
from orrery.evaluation_context import evaluation_context
from orrery.model import ObservationBatch, OrbitPolicy, PolicyOutput

# Every runner call in these tests uses the GPU evaluator's rules on the CPU,
# where the small test policies live.
_V2 = {"evaluation_policy": EVALUATION_POLICY_V2, "device": "cpu"}


class _FixedPolicy(nn.Module):
    """Return caller-supplied scores so evaluator choices are predictable.

    A real policy calculates scores from a neural network. These tests need to
    control the scores exactly, so this small policy returns tensors prepared
    by the test. That lets the assertions focus on the evaluator's decoding
    rules instead of depending on randomly initialized model weights.
    """

    def __init__(self, launch: torch.Tensor, target: torch.Tensor) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros((), dtype=launch.dtype))
        self.launch = launch
        self.target = target

    def forward(self, observation: ObservationBatch) -> PolicyOutput:
        batch = observation.batch_size
        return PolicyOutput(
            self.launch[:batch] + self.marker,
            self.target[:batch] + self.marker,
            torch.zeros(batch, dtype=self.marker.dtype) + self.marker,
        )


class _IdlePolicy(nn.Module):
    """Return zero for every action score and value estimate.

    The evaluation contract launches only when a launch score is greater than
    zero. This policy therefore produces no launches and is useful when a test
    cares about match scheduling or completion rather than action quality.
    """

    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, observation: ObservationBatch) -> PolicyOutput:
        batch = observation.batch_size
        return PolicyOutput(
            torch.zeros((batch, BODY_SLOTS)) + self.marker,
            torch.zeros((batch, BODY_SLOTS, BODY_SLOTS)) + self.marker,
            torch.zeros(batch) + self.marker,
        )


_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    ),
]


@pytest.mark.parametrize("device", _DEVICES)
def test_eval_policy_v2_applies_each_decision_rule(device: str) -> None:
    """Decode one game per rule and require each row's expected action.

    Each row is one game built to exercise one rule, with the action the rule
    requires written next to it. Body slots 0 and 3 are owned; slot 0 may
    target slots 1 and 2 (body IDs 9 and 3), slot 3 has no legal target at
    all. The scores are decoded as tensors on ``device``. Row 5 is padding:
    its scores are NaN, which must decode without flagging, because padding
    output is ignored.
    """

    rows = [
        # 0: tie between legal slots 1 and 2 -> lower body ID (3) wins, so
        #    slot 2; illegal slot 3 has the highest score but is ignored.
        dict(launch=1.0, scores={1: 4.0, 2: 4.0, 3: 100.0}, expected=2),
        # 1: a clear best target, slot 1.
        dict(launch=0.5, scores={1: 7.0, 2: 4.0}, expected=1),
        # 2: a launch score of exactly zero means no launch.
        dict(launch=0.0, scores={1: 7.0}, expected=-1),
        # 3: a negative launch score means no launch.
        dict(launch=-2.0, scores={1: 7.0}, expected=-1),
        # 4: only the owned source with no legal target has a positive
        #    score, and it must still not launch.
        dict(launch=-1.0, scores={}, expected=-1, no_target_launch=5.0),
        # 5: padding row, scores all NaN.
        dict(launch=float("nan"), scores={}, expected=None),
    ]
    games = len(rows)
    launch = torch.full((games, BODY_SLOTS), -1.0)
    target = torch.zeros((games, BODY_SLOTS, BODY_SLOTS))
    for index, row in enumerate(rows):
        launch[index, 0] = row["launch"]
        launch[index, 3] = row.get("no_target_launch", -1.0)
        for slot, score in row["scores"].items():
            target[index, 0, slot] = score
    target[5] = float("nan")
    launch[5] = float("nan")
    policy = _FixedPolicy(launch, target).eval()
    observation = _decoder_observation()
    active = [row["expected"] is not None for row in rows]

    # Every row is decoded, padding included, on ``device``.
    batch = ObservationBatch.from_observations([observation] * games).to(device)
    with torch.inference_mode():
        output = policy(ObservationBatch.from_observations([observation] * games))
    output = PolicyOutput(
        output.launch_logits.to(device),
        output.target_logits.to(device),
        output.values.to(device),
    )
    body_ids = torch.from_numpy(np.stack([observation.slot_body_id] * games)).to(device)
    packed, flags = decode_evaluation_targets(
        output, batch, body_ids, torch.tensor(active, device=device)
    )
    packed = packed.cpu().tolist()

    for index, row in enumerate(rows):
        if row["expected"] is None:
            continue
        assert packed[index][0] == row["expected"]
        assert all(value == -1 for value in packed[index][1:])
    assert flags.cpu().tolist() == [True, True, True]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda launch, target, values: launch.__setitem__((0, 0), float("nan")), "launch scores"),
        (lambda launch, target, values: target.__setitem__((0, 0, 1), float("inf")), "target scores"),
        (lambda launch, target, values: values.__setitem__(0, float("nan")), "value estimates"),
    ],
    ids=["nan launch", "inf legal target", "nan value"],
)
def test_eval_policy_v2_refuses_non_finite_scores_that_matter(
    change: object, message: str
) -> None:
    """A non-finite score that could affect a decision stops validation.

    Without this check a NaN launch score would compare as "not greater than
    zero" and quietly become no launch, so a broken model would be scored as
    a passive one.
    """

    launch, target, values = _finite_scores()
    change(launch, target, values)

    _, flags = _decode(launch, target, values, active=[True])

    with pytest.raises(ValueError, match=message):
        require_finite_scores(flags.tolist())


def test_eval_policy_v2_ignores_non_finite_scores_that_cannot_matter() -> None:
    """Masked illegal targets, non-launchable sources and padding are not checked.

    The policy may score an illegal target as -inf, and a padding row's
    output is thrown away, so none of these may stop validation.
    """

    launch, target, values = _finite_scores(games=2)
    target[0, 0, 3] = float("-inf")  # Illegal target of a launchable source.
    launch[0, 1] = float("nan")  # Slot 1 is not owned, so it cannot launch.
    target[0, 1, 2] = float("nan")  # Targets of a non-launchable source.
    launch[1] = float("nan")  # Row 1 is padding.
    values[1] = float("nan")

    _, flags = _decode(launch, target, values, active=[True, False])

    require_finite_scores(flags.tolist())


def _decoder_observation() -> Observation:
    """``_observation`` plus a second owned body (slot 3) with no legal target."""

    observation = _observation()
    owned = observation.owned_mask.copy()
    owned[3] = True
    return Observation(
        observation.bodies, observation.players, observation.body_mask, owned,
        observation.player_mask, observation.target_mask, observation.slot_body_id,
    )


def _finite_scores(games: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ordinary finite launch scores, target scores and values for ``games`` rows."""

    return (
        torch.zeros((games, BODY_SLOTS)),
        torch.zeros((games, BODY_SLOTS, BODY_SLOTS)),
        torch.zeros(games),
    )


def _decode(
    launch: torch.Tensor, target: torch.Tensor, values: torch.Tensor, *, active: list[bool]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode supplied scores against ``_observation`` boards on the CPU."""

    observation = _observation()
    games = len(active)
    return decode_evaluation_targets(
        PolicyOutput(launch, target, values),
        ObservationBatch.from_observations([observation] * games),
        torch.from_numpy(np.stack([observation.slot_body_id] * games)),
        torch.tensor(active),
    )


def _complete(players: int) -> list[EvaluationOutcome]:
    """Return one finished game for seed 10 from every seat, seat 0 a win."""

    return [
        EvaluationOutcome(10, seat, 1.0 if seat == 0 else -1.0, 20, END_GAME_FINISHED, 1)
        for seat in range(players)
    ]


@pytest.mark.parametrize("players", [2, 4])
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda outcomes, _players: outcomes[:-1], "missing evaluation results"),
        (lambda outcomes, _players: outcomes + outcomes[:1], "duplicate evaluation result"),
        # A seat the map does not have belongs to a different sample.
        (
            lambda outcomes, players: outcomes
            + [EvaluationOutcome(10, players, 1.0, 20, END_GAME_FINISHED, 1)],
            "unexpected evaluation result",
        ),
        (
            lambda outcomes, _players: outcomes
            + [EvaluationOutcome(11, 0, 1.0, 20, END_GAME_FINISHED, 1)],
            "unexpected evaluation result",
        ),
        (
            lambda outcomes, _players: [EvaluationOutcome(10, 0, 0.5, 20, END_GAME_FINISHED, 1)]
            + outcomes[1:],
            "invalid terminal reward",
        ),
        # END_ONGOING means the game never ended; 3 is no end reason at all.
        (
            lambda outcomes, _players: [EvaluationOutcome(10, 0, 1.0, 20, END_ONGOING, 1)]
            + outcomes[1:],
            "invalid end reason 0",
        ),
        (
            lambda outcomes, _players: [EvaluationOutcome(10, 0, 1.0, 20, 3, 1)] + outcomes[1:],
            "invalid end reason 3",
        ),
        (
            lambda outcomes, _players: [EvaluationOutcome(10, 0, 0.0, 20, END_LEARNER_ELIMINATED, 1)]
            + outcomes[1:],
            "eliminated learner has reward 0.0",
        ),
    ],
    ids=[
        "missing seat", "duplicate", "foreign seat", "foreign seed", "invalid reward",
        "ongoing", "unknown end reason", "eliminated without -1",
    ],
)
def test_complete_results_reject_incomplete_or_malformed_games(
    players: int, change: object, message: str
) -> None:
    """Require one well-formed terminal result for every seed and seat.

    Missing or duplicated games would change the denominator and could produce
    a misleading validation win rate; a result that did not end, or ended in a
    way that disagrees with its reward, cannot be scored.
    """

    outcomes = change(_complete(players), players)

    with pytest.raises(ValueError, match=message):
        require_complete_results(outcomes, SeedRange(10, 11), players)


def test_pipeline_gate_reports_score_and_legality_failures() -> None:
    """Collect both runtime gate failures instead of reporting only the first.

    This fixture has too few wins and one rejected action. The gate should
    report both reasons in stable order so a learner can distinguish model
    quality from an environment-legality failure. Repository test results are
    established outside a training run and are not asserted here as runtime
    evidence.
    """

    result = EvaluationResult(
        (
            EvaluationOutcome(10, 0, 1.0, 20, END_GAME_FINISHED, 1),
            EvaluationOutcome(10, 1, -1.0, 21, END_GAME_FINISHED, 1),
        ),
        {},
        1,
    )

    gate = pipeline_gate(
        result,
        required_wins=2,
    )

    assert not gate.passed
    assert gate.wins == 1
    assert gate.failed_checks == (
        "zero rejected actions", "at least 2 wins"
    )


def test_clopper_pearson_zero_and_all_wins_match_closed_forms() -> None:
    """Derive both edge cases without using beta inversion or bisection.

    With zero wins, the lower endpoint is exactly zero and the upper endpoint
    has the closed form ``1 - 0.025**(1/n)``. With all wins, the endpoints are
    the reflected pair ``0.025**(1/n)`` and one. These calculations share none
    of the production beta-CDF or continued-fraction machinery.
    """

    games = 10
    tail_probability = 0.025
    zero_wins = clopper_pearson_interval(0, games)
    all_wins = clopper_pearson_interval(games, games)

    assert zero_wins.confidence_level == 0.95
    assert zero_wins.lower_bound == 0.0
    assert zero_wins.upper_bound == pytest.approx(
        1.0 - tail_probability ** (1.0 / games),
        abs=1e-12,
    )
    assert all_wins.lower_bound == pytest.approx(
        tail_probability ** (1.0 / games),
        abs=1e-12,
    )
    assert all_wins.upper_bound == 1.0


def test_clopper_pearson_matches_release_threshold_reference_value() -> None:
    """Retain the full interval expected for the contracted passing score."""

    interval = clopper_pearson_interval(900, 1_000)

    assert interval.lower_bound == pytest.approx(0.8797120634813074, abs=1e-12)
    assert interval.upper_bound == pytest.approx(0.9178946656444200, abs=1e-12)


@pytest.mark.parametrize(
    ("games", "failing_wins", "passing_wins", "failing_lower", "passing_lower"),
    (
        (1_000, 872, 873, 0.8496940714251292, 0.8507599696651418),
        (2_000, 1_731, 1_732, 0.8497606980764395, 0.8502834118100188),
    ),
)
def test_clopper_pearson_resolves_the_85_percent_boundary_with_binomial_oracle(
    games: int,
    failing_wins: int,
    passing_wins: int,
    failing_lower: float,
    passing_lower: float,
) -> None:
    """Check the beta implementation through an independent binomial path.

    The production code obtains the lower endpoint by evaluating a beta CDF and
    bisecting its inverse. This test instead sums the finite binomial upper tail
    directly at p=0.85. The Clopper-Pearson identity says the lower endpoint is
    at least p exactly when this tail is at most alpha/2, which is 0.025 for a
    95% interval. Checking both neighboring counts at both contracted sample
    sizes proves the implementation places the boundary on the correct side.
    """

    below = clopper_pearson_interval(failing_wins, games)
    above = clopper_pearson_interval(passing_wins, games)
    failing_tail = _binomial_tail_at_least(failing_wins, games, 0.85)
    passing_tail = _binomial_tail_at_least(passing_wins, games, 0.85)

    assert below.lower_bound == pytest.approx(failing_lower, abs=1e-12)
    assert below.lower_bound < 0.85
    assert failing_tail > 0.025
    assert above.lower_bound == pytest.approx(passing_lower, abs=1e-12)
    assert above.lower_bound >= 0.85
    assert passing_tail <= 0.025


def test_release_readiness_gate_accepts_the_complete_required_result() -> None:
    """Accept 900 wins only when all 1,000 games and runtime checks are present.

    Fifty losses and fifty draws demonstrate that both kinds of terminal
    non-win remain visible in the aggregate statistics. They both count against
    the 90% win rate and enter the confidence calculation as non-wins.
    """

    result = _evaluation_result(wins=900, losses=50, draws=50)

    gate = release_readiness_gate(result)

    assert gate.passed
    assert (gate.wins, gate.losses, gate.draws, gate.games) == (900, 50, 50, 1_000)
    assert gate.win_rate == 0.9
    assert gate.interval.lower_bound == pytest.approx(0.8797120634813074)
    assert gate.failed_checks == ()


def test_release_readiness_gate_reports_every_failed_requirement() -> None:
    """Explain incomplete, illegal, weak, and statistically weak evidence.

    This partial sample deliberately violates every gate rule. Reporting all
    four failures helps distinguish a weak policy from missing games or a bug
    in the Python-to-Go action path.
    """

    result = _evaluation_result(
        wins=650,
        losses=100,
        draws=50,
        rejected_actions=1,
    )

    gate = release_readiness_gate(result)

    assert not gate.passed
    assert gate.failed_checks == (
        "exactly 1000 completed games",
        "zero rejected actions",
        "at least 900 wins",
        "95% confidence lower bound at least 85%",
    )


@pytest.mark.parametrize("players", [2, 4])
def test_rotated_runner_returns_each_seed_and_seat_once(
    env_command: tuple[str, ...], players: int
) -> None:
    """Run complete rotated games through the real Go environment process.

    Two validation seeds should produce 2 × players games because the learner
    plays each board once from every seat. Three slots is fewer than either
    game count, so finished slots must be refilled. The idle policy makes the
    decisions simple while this test checks process communication, slot reuse,
    terminal collection, deterministic result ordering, and zero rejections.
    """

    result = run_rotated_matches(
        _IdlePolicy().eval(),
        command=env_command,
        seeds=SeedRange(1_000_000, 1_000_002),
        opponent="idle-v1",
        batch_size=3,
        players=players,
        **_V2,
        expected_metadata={"schedule": "rotate", "opponent": "idle-v1"},
    )

    assert [(item.seed, item.seat) for item in result.outcomes] == [
        (seed, seat) for seed in (1_000_000, 1_000_001) for seat in range(players)
    ]
    assert all(item.end_reason in (END_GAME_FINISHED, END_LEARNER_ELIMINATED) for item in result.outcomes)
    assert result.rejected_actions == 0


def test_rotated_runner_reports_each_games_final_board_once(
    env_command: tuple[str, ...],
) -> None:
    """Every game hands its post-final-turn board to on_game_end exactly once.

    on_turn only sees boards before decisions, so without this record a game's
    last state would be lost. Each record must name the same game, turn count,
    reward, and end reason as that game's outcome, and every game's decisions
    must cover turns 0 through turns - 1, so start-to-finish is complete.
    Three slots for eight four-player games forces slot reuse, the moment the
    final board would otherwise be overwritten.
    """

    ends: list[GameEndRecord] = []
    decisions: dict[tuple[int, int], list[int]] = {}

    def remember_end(record: GameEndRecord) -> None:
        # Check the board now; the assertions below read only the record's
        # plain fields, because the evaluator may reuse its arrays later.
        assert record.observation.body_mask.any()
        ends.append(record)

    result = run_rotated_matches(
        _IdlePolicy().eval(),
        command=env_command,
        seeds=SeedRange(1_000_000, 1_000_002),
        opponent="idle-v1",
        batch_size=3,
        players=4,
        **_V2,
        on_turn=lambda record: decisions.setdefault((record.seed, record.seat), []).append(record.turn),
        on_game_end=remember_end,
    )

    outcomes = {(item.seed, item.seat): item for item in result.outcomes}
    assert sorted((record.seed, record.seat) for record in ends) == sorted(outcomes)
    for record in ends:
        outcome = outcomes[(record.seed, record.seat)]
        assert (record.turn, record.reward, record.end_reason) == (
            outcome.turns, outcome.reward, outcome.end_reason,
        )
        assert decisions[(record.seed, record.seat)] == list(range(outcome.turns))


def test_rotated_runner_passes_the_selected_player_count_to_env_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start validation on the maps selected by its caller.

    The replacement client stops immediately after construction because this
    test is about routing, not game play.
    """

    captured: dict[str, object] = {}

    class ClientStarted(RuntimeError):
        """Stop the runner once it has exposed the constructor arguments."""

    def capture_client(command: object, **kwargs: object) -> object:
        captured.update(command=command, kwargs=kwargs)
        raise ClientStarted

    monkeypatch.setattr(evaluator_module, "EnvClient", capture_client)

    with pytest.raises(ClientStarted):
        run_rotated_matches(
            _IdlePolicy().eval(),
            command=("unused",),
            seeds=SeedRange(1_000_000, 1_000_001),
            opponent="idle-v1",
            batch_size=1,
            players=4,
            **_V2,
        )

    assert captured["command"] == ("unused",)
    assert captured["kwargs"] == {"players": 4}


def test_turn_hook_reports_each_decision_with_decoder_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Describe every decision exactly as the decoder made it.

    A replacement client plays two games, one per seat, that each last two
    turns. The hook must see turn 0 then turn 1 for each game, games in
    ascending environment-ID order within a turn, the policy's value estimate,
    and the actions the decision rules require for that observation.
    """

    observation = _observation()
    launch = torch.full((2, BODY_SLOTS), -1.0)
    launch[:, 0] = 3.0
    target = torch.zeros((2, BODY_SLOTS, BODY_SLOTS))
    target[:, 0, 2] = 5.0
    policy = _ValuePolicy(launch, target, value=0.25).eval()
    # Only slot 0 is owned with legal targets (slots 1 and 2). Its launch score
    # is positive and slot 2 scores highest, so it launches to slot 2; every
    # other slot holds.
    expected_actions = [SlotAction(launch=True, target=2)] + [
        SlotAction() for _ in range(BODY_SLOTS - 1)
    ]
    monkeypatch.setattr(evaluator_module, "EnvClient", _two_turn_client(observation))

    records: list[TurnRecord] = []
    run_rotated_matches(
        policy,
        command=("unused",),
        seeds=SeedRange(1_000_000, 1_000_001),
        opponent="idle-v1",
        batch_size=2,
        players=2,
        on_turn=records.append,
        **_V2,
    )

    assert [(record.env_id, record.turn) for record in records] == [
        (0, 0), (1, 0), (0, 1), (1, 1)
    ]
    assert [(record.seed, record.seat) for record in records[:2]] == [
        (1_000_000, 0), (1_000_000, 1)
    ]
    assert all(record.value == pytest.approx(0.25) for record in records)
    assert all(record.actions == tuple(expected_actions) for record in records)
    assert records[0].observation is observation


def test_turn_hook_covers_every_real_turn_without_changing_results(
    env_command: tuple[str, ...],
) -> None:
    """Observe complete Go games without changing what is played.

    Every game must produce one record per completed turn, numbered from 0 with
    no gaps, and the outcomes must match a run without the hook exactly.
    """

    def play(on_turn: object = None) -> EvaluationResult:
        return run_rotated_matches(
            _IdlePolicy().eval(),
            command=env_command,
            seeds=SeedRange(1_000_000, 1_000_002),
            opponent="idle-v1",
            batch_size=2,
            players=2,
            on_turn=on_turn,
            **_V2,
        )

    records: list[TurnRecord] = []
    observed = play(records.append)
    plain = play()

    assert observed.outcomes == plain.outcomes
    assert observed.rejected_actions == plain.rejected_actions == 0
    for outcome in observed.outcomes:
        turns = [
            record.turn
            for record in records
            if (record.seed, record.seat) == (outcome.seed, outcome.seat)
        ]
        assert turns == list(range(outcome.turns))


def test_validation_summary_matches_hand_counts() -> None:
    """Summarize four games whose statistics are easy to work out by hand.

    Rewards are +1, -1, -1 and 0: one win, two losses and one draw, a 25%
    win rate. Seat 0 won one of its two games (50%); seat 1 lost one and
    drew one (0%). Launches are 0, 3, 5 and 2: mean 2.5, median
    (2 + 3) / 2 = 2.5, max 5, and one game with no launches. Episode lengths
    are 10, 40, 20 and 30, sorted 10, 20, 30, 40: mean 25, min 10, median
    (20 + 30) / 2 = 25, max 40; by nearest rank p10 is the ceil(0.4) = 1st
    value (10) and p90 the ceil(3.6) = 4th (40). One learner was eliminated.
    An empty sample has no statistics and must be refused.
    """

    outcomes = [
        EvaluationOutcome(10, 0, 1.0, 10, END_GAME_FINISHED, 0),
        EvaluationOutcome(10, 1, -1.0, 40, END_GAME_FINISHED, 3),
        EvaluationOutcome(11, 0, -1.0, 20, END_LEARNER_ELIMINATED, 5),
        EvaluationOutcome(11, 1, 0.0, 30, END_GAME_FINISHED, 2),
    ]

    assert validation_summary(outcomes, 2) == {
        "games": 4,
        "results": {"wins": 1, "losses": 2, "draws": 1, "win_rate": 0.25},
        "seat_win_rate": [0.5, 0.0],
        "launches_per_game": {"mean": 2.5, "median": 2.5, "max": 5},
        "zero_launch_games": 1,
        "episode_turns": {
            "mean": 25.0, "min": 10, "p10": 10, "median": 25.0, "p90": 40, "max": 40,
        },
        "eliminations": 1,
    }
    # On a four-player map, seats with no game report None, not an invented 0.
    assert validation_summary(outcomes, 4)["seat_win_rate"] == [0.5, 0.0, None, None]
    with pytest.raises(ValueError, match="at least one outcome"):
        validation_summary([], 2)


def test_runner_counts_each_games_launches_and_resets_on_refill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count every launch a game sent, terminal turn included, per game.

    One environment slot plays two games in turn: seat 0 lasts 2 turns, then
    the refilled slot plays seat 1 for 3 turns. The policy launches from its
    one owned body every turn, so the totals must be 2 and 3. A counter that
    skipped the terminal turn would give 1 and 2; one that was not reset on
    refill would give 2 and 5.
    """

    monkeypatch.setattr(
        evaluator_module, "EnvClient", _scheduled_client(_observation(), [2, 3], players=2)
    )

    result = run_rotated_matches(
        _launching_policy(),
        command=("unused",),
        seeds=SeedRange(1_000_000, 1_000_001),
        opponent="idle-v1",
        batch_size=1,
        players=2,
        **_V2,
    )

    assert [(item.seat, item.turns, item.launches) for item in result.outcomes] == [
        (0, 2, 2), (1, 3, 3)
    ]
    # The policy runs on the CPU, so no GPU memory peak is reported.
    assert result.timings is not None
    assert result.timings.peak_allocated_bytes is None
    assert result.timings.peak_reserved_bytes is None


def test_progress_reports_are_spaced_and_end_with_final_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report progress at most once per interval, then once with final totals.

    A stepping clock advances 7 seconds per reading, so a few turns pass
    between periodic reports. A second run makes every progress callback
    cost 500 seconds: elapsed time must grow, but the decide and environment
    totals must stay exactly the same, because callback time belongs to
    neither phase.
    """

    monkeypatch.setattr(
        evaluator_module,
        "EnvClient",
        _scheduled_client(_observation(), [6, 6, 6, 6], players=2),
    )

    def play(callback_cost: float) -> tuple[EvaluationResult, list[ValidationProgress]]:
        clock = _SteppingClock(7.0)
        reports: list[ValidationProgress] = []

        def on_progress(report: ValidationProgress) -> None:
            reports.append(report)
            clock.now += callback_cost

        result = run_rotated_matches(
            _IdlePolicy().eval(),
            command=("unused",),
            seeds=SeedRange(1_000_000, 1_000_002),
            opponent="idle-v1",
            batch_size=2,
            players=2,
            on_progress=on_progress,
            **_V2,
            clock=clock,
        )
        return result, reports

    result, reports = play(0.0)
    periodic, final = reports[:-1], reports[-1]

    assert periodic, "expected at least one report before the end"
    assert periodic[0].elapsed_seconds >= PROGRESS_INTERVAL_SECONDS
    for earlier, later in zip(periodic, periodic[1:]):
        assert later.elapsed_seconds - earlier.elapsed_seconds >= PROGRESS_INTERVAL_SECONDS
    assert (final.finished, final.total, final.active) == (4, 4, 0)
    assert final.elapsed_seconds == result.timings.elapsed_seconds

    slow_result, _ = play(500.0)
    assert slow_result.timings.decide_seconds == result.timings.decide_seconds
    assert slow_result.timings.environment_seconds == result.timings.environment_seconds
    assert slow_result.timings.elapsed_seconds > result.timings.elapsed_seconds


def test_peak_memory_is_read_on_cuda_and_omitted_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measure GPU memory only on CUDA, and never record a CPU peak as zero.

    PyTorch's memory functions are replaced, so this needs no GPU. A CUDA
    device resets the peak once and reads both peaks; a CPU device touches
    neither, and its timings evidence leaves the peak fields out.
    """

    calls: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", calls.append)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 200)
    cuda = torch.device("cuda")
    cpu = torch.device("cpu")

    evaluator_module._reset_peak_memory(cuda)
    assert evaluator_module._peak_memory(cuda) == (100, 200)
    evaluator_module._reset_peak_memory(cpu)
    assert evaluator_module._peak_memory(cpu) == (None, None)
    assert calls == [cuda]

    assert timings_evidence(RunnerTimings(3.0, 1.0, 2.0)) == {
        "elapsed_seconds": 3.0, "decide_seconds": 1.0, "environment_seconds": 2.0,
    }
    with_peaks = timings_evidence(RunnerTimings(3.0, 1.0, 2.0, 100, 200))
    assert (with_peaks["peak_allocated_bytes"], with_peaks["peak_reserved_bytes"]) == (100, 200)


def test_runner_keeps_a_fixed_batch_with_padding_and_reads_rows_by_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Score every turn as one fixed-size batch, row i always slot i.

    Three slots play four games lasting 1, 3, 3 and 1 turns:

    - turn 1: slots hold games 0, 1, 2; game 0 ends and slot 0 is refilled
      with game 3;
    - turn 2: games 3, 1, 2; game 3 ends and no games are left, so slot 0
      goes idle;
    - turn 3: slot 0 is padding, a copy of the lowest active slot (game 1).

    Each observation carries its game number + 1, so the batches the policy
    sees must be [1, 2, 3], [4, 2, 3] and [2, 2, 3]. Go must be stepped only
    for active slots, and on turn 3 the diagnostic values must come from rows
    1 and 2: the policy's value is marker * 10 + row, so reading by position
    in the active list would give 20 instead of 21 for slot 1.
    """

    stepped: list[list[int]] = []
    monkeypatch.setattr(
        evaluator_module,
        "EnvClient",
        _scheduled_client(_observation(), [1, 3, 3, 1], players=2, stepped=stepped),
    )
    _RowRecordingPolicy.batches = []
    records: list[TurnRecord] = []

    result = run_rotated_matches(
        _RowRecordingPolicy().eval(),
        command=("unused",),
        seeds=SeedRange(1_000_000, 1_000_002),
        opponent="idle-v1",
        batch_size=3,
        players=2,
        on_turn=records.append,
        **_V2,
    )

    assert _RowRecordingPolicy.batches == [[1, 2, 3], [4, 2, 3], [2, 2, 3]]
    assert stepped == [[0, 1, 2], [0, 1, 2], [1, 2]]
    last_turn = [(record.env_id, record.value) for record in records if record.turn == 2]
    assert last_turn == [(1, 21.0), (2, 32.0)]
    assert [item.turns for item in result.outcomes] == [1, 3, 3, 1]
    evaluation = result.metadata["evaluation"]
    assert evaluation["policy"] == EVALUATION_POLICY_V2
    assert evaluation["effective_batch_size"] == 3
    assert evaluation["device"] == "cpu"
    assert evaluation["deterministic_algorithms"] is True
    assert evaluation["float32_matmul_precision"] == "highest"


@pytest.mark.parametrize(
    ("policy", "arguments", "message"),
    [
        (_IdlePolicy(), {"evaluation_policy": "eval-policy-v1"}, "supports 'eval-policy-v2'"),
        (_IdlePolicy(), {"device": "cuda"}, "requested on cuda"),
        (_IdlePolicy().double(), {}, "float32"),
    ],
    ids=["v1 evaluator", "wrong device", "float64 policy"],
)
def test_runner_refuses_a_mismatched_request_before_games_start(
    monkeypatch: pytest.MonkeyPatch, policy: nn.Module, arguments: dict, message: str
) -> None:
    """Refuse a different evaluator, device or number format before Go starts."""

    def no_games(*_args: object, **_kwargs: object) -> object:
        pytest.fail("games started")

    monkeypatch.setattr(evaluator_module, "EnvClient", no_games)

    with pytest.raises(ValueError, match=message):
        run_rotated_matches(
            policy.eval(),
            command=("unused",),
            seeds=SeedRange(1_000_000, 1_000_001),
            opponent="idle-v1",
            batch_size=1,
            players=2,
            **{**_V2, **arguments},
        )


@pytest.mark.parametrize("fail", [False, True], ids=["completes", "fails"])
def test_runner_never_changes_the_live_policy_or_leaves_settings_behind(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """Training's model and settings are untouched, whether or not validation fails.

    The live policy stays in train mode with its original parameters,
    because the runner evaluates a private copy. Deterministic mode, turned
    on only while games run, is off again afterwards, also after an error
    raised by the environment.
    """

    client = _scheduled_client(_observation(), [2, 2], players=2)
    if fail:
        class FailingClient(client):
            def step_packed(self, actions: dict) -> list[StepResult]:
                raise RuntimeError("environment failed")

        client = FailingClient
    monkeypatch.setattr(evaluator_module, "EnvClient", client)
    policy = _IdlePolicy()
    with torch.no_grad():
        policy.marker.fill_(-1.0)
    torch.use_deterministic_algorithms(False)

    def play() -> None:
        run_rotated_matches(
            policy,
            command=("unused",),
            seeds=SeedRange(1_000_000, 1_000_001),
            opponent="idle-v1",
            batch_size=2,
            players=2,
            **_V2,
        )

    if fail:
        with pytest.raises(RuntimeError, match="environment failed"):
            play()
    else:
        play()

    assert policy.training
    assert policy.marker.item() == -1.0
    assert policy.marker.device.type == "cpu"
    assert not torch.are_deterministic_algorithms_enabled()


def test_runner_refuses_non_finite_scores_before_stepping_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NaN launch score stops validation before any action reaches Go."""

    stepped: list[list[int]] = []
    monkeypatch.setattr(
        evaluator_module,
        "EnvClient",
        _scheduled_client(_observation(), [2], players=1, stepped=stepped),
    )
    launch = torch.zeros((1, BODY_SLOTS))
    launch[0, 0] = float("nan")
    policy = _FixedPolicy(launch, torch.zeros((1, BODY_SLOTS, BODY_SLOTS))).eval()

    with pytest.raises(ValueError, match="non-finite policy output: launch scores"):
        run_rotated_matches(
            policy,
            command=("unused",),
            seeds=SeedRange(1_000_000, 1_000_001),
            opponent="idle-v1",
            batch_size=1,
            players=1,
            **_V2,
        )
    assert stepped == []


def test_elapsed_time_includes_preparing_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model copy is part of the runner's lifetime, so it counts as elapsed.

    The clock stands still except while the model is copied, which costs a
    controlled ten seconds. Elapsed must be exactly those ten seconds, and
    neither phase total may include them.
    """

    monkeypatch.setattr(
        evaluator_module, "EnvClient", _scheduled_client(_observation(), [1, 1], players=2)
    )
    clock = _SteppingClock(0.0)
    real_deepcopy = evaluator_module.copy.deepcopy

    def slow_copy(value: object) -> object:
        clock.now += 10.0
        return real_deepcopy(value)

    monkeypatch.setattr(evaluator_module, "copy", SimpleNamespace(deepcopy=slow_copy))

    result = run_rotated_matches(
        _IdlePolicy().eval(),
        command=("unused",),
        seeds=SeedRange(1_000_000, 1_000_001),
        opponent="idle-v1",
        batch_size=2,
        players=2,
        clock=clock,
        **_V2,
    )

    assert result.timings.elapsed_seconds == 10.0
    assert result.timings.decide_seconds == 0.0
    assert result.timings.environment_seconds == 0.0


def test_runner_uses_a_prepared_model_without_copying_it_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model prepared earlier is played as is, and only for its own request.

    Held-out evaluation prepares the copy before claiming its pool. The runner
    must not copy it a second time, and must refuse it when the request names
    another evaluator or device.
    """

    monkeypatch.setattr(
        evaluator_module, "EnvClient", _scheduled_client(_observation(), [1, 1], players=2)
    )
    prepared = prepare_evaluation_model(_IdlePolicy(), EVALUATION_POLICY_V2, "cpu")

    def no_copy(_value: object) -> object:
        pytest.fail("prepared model was copied again")

    monkeypatch.setattr(evaluator_module, "copy", SimpleNamespace(deepcopy=no_copy))

    result = run_rotated_matches(
        prepared,
        command=("unused",),
        seeds=SeedRange(1_000_000, 1_000_001),
        opponent="idle-v1",
        batch_size=2,
        players=2,
        **_V2,
    )
    assert len(result.outcomes) == 2

    with pytest.raises(ValueError, match="prepared model is for"):
        run_rotated_matches(
            prepared,
            command=("unused",),
            seeds=SeedRange(1_000_000, 1_000_001),
            opponent="idle-v1",
            batch_size=2,
            players=2,
            evaluation_policy=EVALUATION_POLICY_V2,
            device="cuda",
        )


@pytest.mark.parametrize("device", _DEVICES)
def test_real_policy_scores_do_not_change_when_a_neighbour_becomes_padding(
    env_command: tuple[str, ...], device: str
) -> None:
    """The real transformer gives a game identical scores beside live or padding rows.

    Two real boards from Go fill a two-row batch. Then slot 1's game is
    removed, so slot 1 becomes padding (a copy of slot 0), at the same batch
    shape. Slot 0's actions and value must not change at all: the batch size
    is the same, and its own row's input is unchanged. This runs the real
    ``OrbitPolicy`` inference path under the evaluation settings, including
    any fused attention fast path, on the CPU and on CUDA when available.
    """

    with EnvClient(env_command, players=2) as environment:
        environment.create(2, (1_000_000, 1_000_001), "idle-v1", schedule=SCHEDULE_ROTATE)
        first, second = environment.reset([0, 1])
    torch.manual_seed(0)
    target = torch.device(device)
    prepared = prepare_evaluation_model(
        OrbitPolicy().to(target), EVALUATION_POLICY_V2, target
    )

    with evaluation_context(prepared.device):
        both = evaluator_module._decide_packed(
            prepared.model, {0: first, 1: second}, 2, prepared.device
        )
        alone = evaluator_module._decide_packed(prepared.model, {0: first}, 2, prepared.device)

    assert alone.targets[0] == both.targets[0]
    assert torch.equal(alone.values[0], both.values[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_smoke_repeats_real_policy_games_exactly(env_command: tuple[str, ...]) -> None:
    """Two CUDA runs of the real policy play exactly the same games.

    Two seeds on two-player maps are four games in three slots, so the last
    turns include a padding row. Every recorded action and value, every
    outcome and every recorded execution setting must match between runs.
    This test is skipped without CUDA, and a skip is not GPU verification.
    """

    torch.manual_seed(0)
    policy = OrbitPolicy().to("cuda")

    def play() -> tuple[EvaluationResult, list[tuple[object, ...]]]:
        records: list[TurnRecord] = []
        result = run_rotated_matches(
            policy,
            command=env_command,
            seeds=SeedRange(1_000_000, 1_000_002),
            opponent="idle-v1",
            batch_size=3,
            players=2,
            evaluation_policy=EVALUATION_POLICY_V2,
            device="cuda",
            on_turn=records.append,
        )
        decisions = [
            (record.env_id, record.turn, record.actions, record.value) for record in records
        ]
        return result, decisions

    first, first_decisions = play()
    second, second_decisions = play()

    assert first.outcomes == second.outcomes
    assert first_decisions == second_decisions
    assert first.metadata["evaluation"] == second.metadata["evaluation"]
    assert first.metadata["evaluation"]["device"] == "cuda"
    assert first.rejected_actions == second.rejected_actions == 0


class _RowRecordingPolicy(nn.Module):
    """Record each batch's game markers and report row-identifying values.

    The runner evaluates a deep copy of the policy, so the record lives on
    the class rather than the instance; each test resets it first.
    """

    batches: list[list[float]] = []

    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, observation: ObservationBatch) -> PolicyOutput:
        markers = observation.bodies[:, 0, 0]
        type(self).batches.append(markers.tolist())
        batch = observation.batch_size
        rows = torch.arange(batch, dtype=torch.float32)
        return PolicyOutput(
            torch.zeros((batch, BODY_SLOTS)) + self.marker,
            torch.zeros((batch, BODY_SLOTS, BODY_SLOTS)) + self.marker,
            markers * 10 + rows + self.marker,
        )


class _SteppingClock:
    """A fake monotonic clock that advances a fixed amount per reading."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def _launching_policy() -> nn.Module:
    """Launch from body slot 0 to legal target slot 1 on every turn."""

    launch = torch.full((1, BODY_SLOTS), -1.0)
    launch[0, 0] = 3.0
    target = torch.zeros((1, BODY_SLOTS, BODY_SLOTS))
    target[0, 0, 1] = 5.0
    return _FixedPolicy(launch, target).eval()


def _scheduled_client(
    observation: Observation,
    game_turns: list[int],
    *,
    players: int,
    stepped: list[list[int]] | None = None,
) -> type:
    """Build a replacement EnvClient that hands out rotate games in order.

    Game ``i`` is seed ``1_000_000 + i // players`` from seat ``i % players``
    and ends after ``game_turns[i]`` turns. Unlike ``_two_turn_client``, a
    refilled slot receives the next scheduled game, as the real rotate
    schedule does. Game ``i``'s observation carries ``i + 1`` in
    ``bodies[0, 0]``, so a test policy can tell which game sits in which
    batch row. When ``stepped`` is given, each step appends the environment
    IDs it was asked to advance.
    """

    def marked(game: int) -> Observation:
        bodies = observation.bodies.copy()
        bodies[0, 0] = game + 1
        return Observation(
            bodies, observation.players, observation.body_mask, observation.owned_mask,
            observation.player_mask, observation.target_mask, observation.slot_body_id,
        )

    class ScheduledClient:
        error_count = 0

        def __init__(self, command: object, **kwargs: object) -> None:
            self.next_game = 0
            self.games: dict[int, int] = {}
            self.steps: dict[int, int] = {}

        def __enter__(self) -> "ScheduledClient":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def create(self, *args: object, **kwargs: object) -> dict:
            return {"schedule": "rotate"}

        def reset(self, ids: object) -> list[Episode]:
            episodes = []
            for env_id in ids:
                game = self.next_game
                self.next_game += 1
                self.games[env_id] = game
                self.steps[env_id] = 0
                episodes.append(
                    Episode(env_id, 1_000_000 + game // players, game % players, False, marked(game))
                )
            return episodes

        def step_packed(self, actions: dict) -> list[StepResult]:
            if stepped is not None:
                stepped.append(sorted(actions))
            results = []
            for env_id in sorted(actions):
                game = self.games[env_id]
                self.steps[env_id] += 1
                turn = self.steps[env_id]
                done = turn == game_turns[game]
                results.append(
                    StepResult(
                        env_id, 1_000_000 + game // players, game % players, False, marked(game),
                        0.0, done, False, END_GAME_FINISHED if done else END_ONGOING, turn,
                    )
                )
            return results

    return ScheduledClient


class _ValuePolicy(_FixedPolicy):
    """A fixed-score policy that also reports a chosen value estimate."""

    def __init__(self, launch: torch.Tensor, target: torch.Tensor, *, value: float) -> None:
        super().__init__(launch, target)
        self.value = value

    def forward(self, observation: ObservationBatch) -> PolicyOutput:
        output = super().forward(observation)
        return PolicyOutput(
            output.launch_logits,
            output.target_logits,
            torch.full((observation.batch_size,), self.value) + self.marker,
        )


def _two_turn_client(observation: Observation) -> type:
    """Build a replacement EnvClient whose rotated games each last two turns.

    Reset assigns two-player rotate games in slot order: slot 0 gets seat 0 and
    slot 1 gets seat 1 of the same seed. Each step returns the same observation
    and ends a game after its second turn.
    """

    class TwoTurnClient:
        error_count = 0

        def __init__(self, command: object, **kwargs: object) -> None:
            self.steps: dict[int, int] = {}

        def __enter__(self) -> "TwoTurnClient":
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def create(self, batch_size: int, seed_range: tuple[int, int], *args: object, **kwargs: object) -> dict:
            self.seed = seed_range[0]
            return {"schedule": "rotate"}

        def reset(self, ids: object) -> list[Episode]:
            episodes = []
            for env_id in ids:
                self.steps[env_id] = 0
                episodes.append(Episode(env_id, self.seed, env_id % 2, False, observation))
            return episodes

        def step(self, actions: dict) -> list[StepResult]:
            results = []
            for env_id in sorted(actions):
                self.steps[env_id] += 1
                turn = self.steps[env_id]
                results.append(
                    StepResult(
                        env_id, self.seed, env_id % 2, False, observation,
                        0.0, turn == 2, False,
                        END_GAME_FINISHED if turn == 2 else END_ONGOING, turn,
                    )
                )
            return results

        # eval-policy-v2 sends packed target rows; only their keys matter here.
        step_packed = step

    return TwoTurnClient


def _observation() -> Observation:
    """Build a small legal observation whose target identities are easy to inspect.

    Only body slot 0 is learner-owned. Slots 1 and 2 are legal destinations,
    and their physical body IDs are intentionally ordered as 9 then 3 so a
    test can distinguish body-ID tie-breaking from ordinary slot ordering.
    """

    bodies = np.zeros((BODY_SLOTS, BODY_FEATURES), dtype=np.float32)
    players = np.zeros((PLAYER_SLOTS, PLAYER_FEATURES), dtype=np.float32)
    body_mask = np.zeros(BODY_SLOTS, dtype=np.bool_)
    body_mask[:4] = True
    owned_mask = np.zeros(BODY_SLOTS, dtype=np.bool_)
    owned_mask[0] = True
    player_mask = np.zeros(PLAYER_SLOTS, dtype=np.bool_)
    player_mask[:2] = True
    target_mask = np.zeros((BODY_SLOTS, BODY_SLOTS), dtype=np.bool_)
    target_mask[0, 1:3] = True
    body_ids = np.zeros(BODY_SLOTS, dtype=np.int32)
    body_ids[:4] = [5, 9, 3, 1]
    return Observation(
        bodies, players, body_mask, owned_mask, player_mask, target_mask, body_ids
    )


def _evaluation_result(
    *,
    wins: int,
    losses: int,
    draws: int,
    rejected_actions: int = 0,
) -> EvaluationResult:
    """Build deterministic synthetic outcomes for gate-only unit tests."""

    rewards = (1.0,) * wins + (-1.0,) * losses + (0.0,) * draws
    outcomes = tuple(
        EvaluationOutcome(index // 2, index % 2, reward, 100, END_GAME_FINISHED, 1)
        for index, reward in enumerate(rewards)
    )
    return EvaluationResult(outcomes, {}, rejected_actions)


def _binomial_tail_at_least(wins: int, games: int, probability: float) -> float:
    """Compute P(X >= wins) directly as an oracle for the beta implementation.

    This test helper deliberately does not call a beta function, continued
    fraction, quantile function, or bisection. Each possible win count from the
    requested threshold through ``games`` contributes one binomial probability.
    Logarithms keep the large combinations at n=2,000 representable as float64.
    """

    if wins <= 0:
        return 1.0
    log_probability = math.log(probability)
    log_non_win_probability = math.log1p(-probability)
    return math.fsum(
        math.exp(
            math.lgamma(games + 1)
            - math.lgamma(outcome_wins + 1)
            - math.lgamma(games - outcome_wins + 1)
            + outcome_wins * log_probability
            + (games - outcome_wins) * log_non_win_probability
        )
        for outcome_wins in range(wins, games + 1)
    )
