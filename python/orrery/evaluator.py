"""Run repeatable validation games and judge the resulting model checkpoint.

Training and evaluation use the same neural network but choose actions in
different ways. Training samples actions so the learner explores. Evaluation
removes that randomness and always converts the same model scores into the same
legal game actions. That makes scores from different checkpoints comparable.

The module has five separate responsibilities:

1. ``decode_evaluation_targets`` converts model output into deterministic
   actions as tensors on the policy's device (``eval-policy-v2``).
2. ``run_rotated_matches`` sends those actions to Go until every game finishes.
   An optional ``on_turn`` callback lets diagnostics observe each decision
   without changing how the games are played.
3. Shared metadata helpers verify the same Go contract for every scoring path.
4. ``pipeline_gate`` applies the small early-learning pass/fail check.
5. ``release_readiness_gate`` applies the stronger statistical check used
   before a checkpoint can become a final-evaluation candidate.

None of these functions trains the model. They read fixed weights, play games,
and summarize evidence about those weights.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import statistics
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .config import RunConfig, SeedRange
from .env_client import (
    BODY_SLOTS,
    END_GAME_FINISHED,
    END_LEARNER_ELIMINATED,
    OBSERVATION_DIMENSIONS,
    EnvClient,
    Episode,
    Observation,
    SCHEDULE_ROTATE,
    SlotAction,
)
from .evaluation_context import evaluation_context
from .model import (
    ObservationBatch,
    PolicyOutput,
    policy_device,
    policy_validation_flags,
    require_evaluation_mode,
    run_policy_for_collection,
    validate_policy_observation,
)


# This name identifies the exact rules used to turn model scores into actions:
# launch only when a launch score is above zero, to the highest-scoring legal
# target, computed as tensor operations on the policy's own device (normally
# the training GPU). Recording it with results prevents scores made under
# different decoding rules from being compared as though they used the same
# evaluator. The "-v2" stays because it is recorded in existing evidence.
EVALUATION_POLICY_V2 = "eval-policy-v2"

# Release readiness uses 500 validation boards, and the rotate schedule plays
# every board from each seat. On a two-player map that produces 500 * 2 =
# 1,000 games. These two counts are release_readiness_gate's two-player
# defaults; every caller passes its own config's values instead.
RELEASE_READINESS_GAMES = 1_000
# Draws and losses are non-wins, so 900 means a measured win rate of at least 90%.
RELEASE_READINESS_WINS = 900
# A two-sided 95% interval leaves 2.5% in each tail of the beta distribution.
RELEASE_READINESS_CONFIDENCE = 0.95
# The conservative lower estimate of the underlying win rate must reach 85%.
RELEASE_READINESS_MINIMUM_LOWER_BOUND = 0.85

# Saved beside every interval in JSON evidence, so a reader of that file alone
# knows the bound is nominal. See clopper_pearson_interval for the reason.
NOMINAL_INTERVAL_ASSUMPTION = (
    "each game is treated as an independent draw with the same win chance; "
    "games sharing a scenario seed are the same board from different seats "
    "and are correlated, so the stated confidence is nominal"
)

# A release validation can run for many minutes. One progress report a minute
# shows the run is alive without flooding the training log.
PROGRESS_INTERVAL_SECONDS = 60.0

@dataclass(frozen=True)
class EvaluationOutcome:
    """The identity and terminal result of one validation game."""

    # Scenario seed used by Go to construct this game's starting board.
    seed: int
    # Player position occupied by the learner: 0 to players-1.
    seat: int
    # Terminal reward from the learner's view: 1 for finishing strictly first
    # by ships, 0 for a tie for first or a draw, and -1 for a loss.
    reward: float
    # Number of game turns completed before the terminal result.
    turns: int
    # Why the game ended, copied from the terminal StepResult:
    # END_GAME_FINISHED (1) or END_LEARNER_ELIMINATED (2). Keeping it lets a
    # report tell an elimination apart from a loss on ships.
    end_reason: int
    # Launch actions the learner sent during this game, terminal turn
    # included. Zero over a whole game means the policy never launched.
    launches: int


@dataclass(frozen=True)
class RunnerTimings:
    """Where one validation run spent its wall-clock time.

    These numbers describe a run; they are not part of its replay identity.
    The same games played on a busier machine take longer but must produce
    the same outcomes.
    """

    # Whole runner lifetime, including Go startup and shutdown. The two phase
    # totals below need not add up to it.
    elapsed_seconds: float
    # Choosing actions in Python: batching observations, model inference,
    # decoding, and copying the chosen actions back to the CPU.
    decide_seconds: float
    # Calls that reset and step games: encoding, the pipe to Go, Go's own
    # work, and decoding the next observations. Not Go engine time alone.
    environment_seconds: float
    # Highest GPU memory held by tensors and reserved by PyTorch's allocator
    # during the run, in bytes. None on CPU runs, which have no such peak.
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None


@dataclass(frozen=True)
class ValidationProgress:
    """A snapshot of a running validation, handed to ``on_progress``."""

    # Games with a terminal result so far, out of the total requested.
    finished: int
    total: int
    # Games currently being played in environment slots.
    active: int
    # Time so far, split the same way as RunnerTimings.
    elapsed_seconds: float
    decide_seconds: float
    environment_seconds: float


@dataclass(frozen=True)
class TurnRecord:
    """One learner decision seen during a validation game.

    ``run_rotated_matches`` builds one record for every active game just before
    Go advances that game by a turn. A diagnostic callback can store these
    records to replay a game turn by turn. Building them changes nothing about
    how the game is played: the same deterministic actions are sent either way.
    """

    # Reusable Go environment slot that played this game. Slots are refilled
    # with new games, so pair it with seed and seat to identify one game.
    env_id: int
    # Scenario seed Go used to build this game's starting board.
    seed: int
    # Player position the learner occupies in this game: 0 to players-1.
    seat: int
    # Number of game turns already completed when this decision was made.
    # The first decision after a reset is turn 0.
    turn: int
    # The exact player-relative observation the policy read for this decision.
    # Its NumPy arrays are shared with the evaluator, so treat them as read-only.
    observation: Observation
    # The critic's estimate of the final result from the learner's view, as a
    # plain float. Training targets are discounted results between -1 and 1.
    value: float
    # One deterministic action for every body slot, in slot order, exactly as
    # sent to Go for this turn.
    actions: tuple[SlotAction, ...]


@dataclass(frozen=True)
class GameEndRecord:
    """The board one validation game ended on, after its final turn.

    A ``TurnRecord`` shows the board *before* each decision, so the board that
    the last decision produced never appears in one: a capture or loss on the
    final move would be invisible. Go rebuilds the observation after every
    turn, terminal ones included, and this record carries that last one.
    """

    # Reusable Go environment slot that played this game. It is refilled with
    # a new game right after this record, so pair it with seed and seat.
    env_id: int
    # Scenario seed and learner seat: together they identify one game.
    seed: int
    seat: int
    # Game turns completed when the game ended; equals EvaluationOutcome.turns.
    turn: int
    # The learner's view after the final turn. Its arrays belong to the
    # evaluator and may be reused once the slot is refilled, so a callback
    # must copy or summarize anything it keeps rather than store the arrays.
    observation: Observation
    # Terminal reward and end reason, as in EvaluationOutcome.
    reward: float
    end_reason: int


@dataclass(frozen=True)
class EvaluationResult:
    """A checked rotated-match result plus its replay metadata."""

    # One terminal result for every requested (scenario seed, learner seat).
    outcomes: tuple[EvaluationOutcome, ...]
    # Go contract identities such as engine, schema, generator, and opponent.
    metadata: Mapping[str, object]
    # Number of learner action batches the authoritative Go process rejected.
    rejected_actions: int
    # Where the run spent its time. The real runner always fills it; results
    # assembled by hand, such as gate-only test fixtures, may leave it None.
    timings: RunnerTimings | None = None

    @property
    def wins(self) -> int:
        """Count how many completed games the learner won.

        Go reports terminal reward ``1`` only when the learner finishes
        strictly first by ships, ``0`` for a tie for first or a draw, and
        ``-1`` for a loss. Only positive rewards increase this count. Reading
        the property does not modify or remove any individual outcome.
        """

        return sum(outcome.reward > 0 for outcome in self.outcomes)


def evaluation_metadata(
    config: RunConfig,
    seeds: SeedRange,
    opponent: str,
) -> dict[str, object]:
    """Build the one expected Go contract for validation or evaluation.

    Pipeline validation, release validation, and protected held-out evaluation
    differ in opponent and seed range. Their protocol, observation, generator,
    and rotate schedule are the same and are assembled here so a contract
    change cannot be applied to only one scoring path.
    """

    return {
        "v": config.protocol_version,
        "schema": config.observation_schema,
        "generator": config.generator_version,
        "opponent": opponent,
        "schedule": SCHEDULE_ROTATE,
        "seed_range": [seeds.start, seeds.end],
    }


def checked_evaluation_result(
    result: object,
    seeds: SeedRange,
    expected_metadata: Mapping[str, object],
    *,
    players: int,
    evaluation_policy: str,
    context: str,
) -> EvaluationResult:
    """Validate one runner result before it can influence an official score.

    The result must use the expected Python type, contain exactly one terminal
    outcome for each seed and each of the ``players`` learner seats, and
    report the complete expected Go contract plus an engine version and the
    obs-v2 dimensions. ``context`` appears in errors so a beginner can tell
    which gate rejected the evidence.

    ``expected_metadata`` covers only what Go reports. How Python chose the
    actions is recorded separately under ``metadata["evaluation"]``, and its
    evaluator must be ``evaluation_policy``: a result decided under other
    rules is not evidence for this run.
    """

    if not isinstance(result, EvaluationResult):
        raise ValueError(f"{context} runner returned an invalid result type")
    outcomes = require_complete_results(result.outcomes, seeds, players)
    for name, expected in expected_metadata.items():
        if result.metadata.get(name) != expected:
            raise ValueError(
                f"{context} metadata {name!r} is "
                f"{result.metadata.get(name)!r}, want {expected!r}"
            )
    engine = result.metadata.get("engine")
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError(f"{context} metadata is missing the engine version")
    if result.metadata.get("dimensions") != OBSERVATION_DIMENSIONS:
        raise ValueError(
            f"{context} metadata dimensions are "
            f"{result.metadata.get('dimensions')!r}, want {OBSERVATION_DIMENSIONS!r}"
        )
    evaluation = result.metadata.get("evaluation")
    recorded = evaluation.get("policy") if isinstance(evaluation, Mapping) else None
    if recorded != evaluation_policy:
        raise ValueError(
            f"{context} evaluator is {recorded!r}, want {evaluation_policy!r}"
        )
    return EvaluationResult(
        outcomes, dict(result.metadata), result.rejected_actions, result.timings
    )


def validation_summary(
    outcomes: Sequence[EvaluationOutcome], players: int
) -> dict[str, object]:
    """Describe what the validation games looked like, beyond who won.

    Win counts alone can hide a broken policy: one that has learned never to
    launch may still win some games against weak opponents. These numbers show
    how often the learner acted, how long its games lasted, and how often it
    was wiped out.

    The summary covers every measure the opponent-check harness
    (``cmd/opponentcheck``) reports for a tester, so a model's validation and
    an opponent check can be read side by side: wins, losses, draws and win
    rate; win rate from each seat; eliminations; launches; and length
    percentiles computed with the same nearest-rank rule. It adds what only
    validation needs, such as the launch median and maximum.

    ``players`` is the map's player count; ``seat_win_rate`` has one entry
    per seat, ``None`` for a seat with no game rather than an invented 0.

    ``episode_turns`` is the learner's episode length. An eliminated learner's
    episode stops before the full match does, so it is not the game length.
    (The opponent check reports the game's finish turn instead.) Only the
    learner's own elimination is counted: Go reports the learner's end
    reason, not the opponents'.

    An empty sequence raises ``ValueError``; zero-valued statistics would
    describe games that were never played.
    """

    if not outcomes:
        raise ValueError("validation_summary requires at least one outcome")
    launches = [outcome.launches for outcome in outcomes]
    turns = [outcome.turns for outcome in outcomes]
    wins = sum(outcome.reward > 0 for outcome in outcomes)
    seat_win_rate: list[float | None] = []
    for seat in range(players):
        seat_rewards = [outcome.reward for outcome in outcomes if outcome.seat == seat]
        seat_win_rate.append(
            sum(reward > 0 for reward in seat_rewards) / len(seat_rewards)
            if seat_rewards
            else None
        )
    return {
        "games": len(outcomes),
        # A positive reward is a win, a negative one a loss, and exactly zero
        # a draw (a tie for first), as in release_readiness_gate.
        "results": {
            "wins": wins,
            "losses": sum(outcome.reward < 0 for outcome in outcomes),
            "draws": sum(outcome.reward == 0 for outcome in outcomes),
            "win_rate": wins / len(outcomes),
        },
        "seat_win_rate": seat_win_rate,
        "launches_per_game": {
            "mean": statistics.fmean(launches),
            "median": statistics.median(launches),
            "max": max(launches),
        },
        "zero_launch_games": sum(count == 0 for count in launches),
        "episode_turns": {
            "mean": statistics.fmean(turns),
            "min": min(turns),
            "p10": _nearest_rank_percentile(turns, 0.10),
            "median": statistics.median(turns),
            "p90": _nearest_rank_percentile(turns, 0.90),
            "max": max(turns),
        },
        "eliminations": sum(
            outcome.end_reason == END_LEARNER_ELIMINATED for outcome in outcomes
        ),
    }


def _nearest_rank_percentile(values: Sequence[int], fraction: float) -> int:
    """Return the value at ``fraction`` of the sorted list, by nearest rank.

    Nearest rank picks an actual value rather than interpolating: the item at
    position ceil(fraction * n), counting from 1. For 10 games, p10 is the
    1st-shortest and p90 the 9th. This is the opponent check's ``percentile``
    rule, so the two reports give the same number for the same games.
    """

    ordered = sorted(values)
    rank = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(rank, 0)]


def timings_evidence(timings: RunnerTimings) -> dict[str, object]:
    """Convert runner timings to plain JSON values, leaving out absent peaks.

    A CPU run has no GPU memory peak. Omitting the fields says that directly,
    where a zero would claim a measurement that was never taken.
    """

    evidence: dict[str, object] = {
        "elapsed_seconds": timings.elapsed_seconds,
        "decide_seconds": timings.decide_seconds,
        "environment_seconds": timings.environment_seconds,
    }
    if timings.peak_allocated_bytes is not None:
        evidence["peak_allocated_bytes"] = timings.peak_allocated_bytes
    if timings.peak_reserved_bytes is not None:
        evidence["peak_reserved_bytes"] = timings.peak_reserved_bytes
    return evidence


def _reset_peak_memory(device: torch.device) -> None:
    """Start a fresh GPU memory peak for this run; nothing to do on the CPU."""

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> tuple[int | None, int | None]:
    """Read the run's allocated and reserved GPU memory peaks, or None on CPU."""

    if device.type != "cuda":
        return None, None
    return (
        torch.cuda.max_memory_allocated(device),
        torch.cuda.max_memory_reserved(device),
    )


@dataclass(frozen=True)
class PipelineGateResult:
    """Explain whether a checkpoint proved the complete learning pipeline.

    ``passed`` is true only when every required check succeeded. ``wins`` and
    ``games`` retain the score that was judged, while ``failed_checks`` names
    every unmet requirement. Keeping the reasons makes a failed gate useful
    for diagnosis instead of reducing it to an unexplained true/false value.
    """

    # Overall decision used by the trainer to stop or continue the initial stage.
    passed: bool
    # Number of validation games won; draws and losses are both non-wins.
    wins: int
    # Total number of completed validation games included in the decision.
    games: int
    # Stable list of requirements that prevented the checkpoint from passing.
    failed_checks: tuple[str, ...]


@dataclass(frozen=True)
class BinomialInterval:
    """Two-sided exact confidence interval for an underlying win probability."""

    # Fraction such as 0.95 that describes the interval's coverage rule.
    confidence_level: float
    # Conservative low estimate of the model's underlying probability of winning.
    lower_bound: float
    # Conservative high estimate of the model's underlying probability of winning.
    upper_bound: float


def interval_evidence(interval: BinomialInterval) -> dict[str, object]:
    """Describe an interval for saved evidence, qualified as nominal.

    Training records, held-out results, and checkpoint reports all store the
    interval through this one function, so none of them can drop the
    qualification that the Markdown report states in words.
    """

    return {
        "method": "clopper-pearson",
        "confidence_level": interval.confidence_level,
        "lower_bound": interval.lower_bound,
        "upper_bound": interval.upper_bound,
        "nominal": True,
        "assumption": NOMINAL_INTERVAL_ASSUMPTION,
    }


@dataclass(frozen=True)
class ReleaseReadinessResult:
    """Aggregate score and reasons for a release-readiness gate decision.

    Every game is reduced to either a win or a non-win for the confidence
    interval. Loss and draw counts remain separate here so the saved evidence
    can still explain what actually happened.
    """

    # True only when every release-readiness requirement below succeeds.
    passed: bool
    # Number of terminal outcomes actually supplied to the decision.
    games: int
    # Positive, negative, and zero learner rewards kept as separate counts.
    wins: int
    losses: int
    draws: int
    # Observed wins divided by games. This is the sample's measured rate.
    win_rate: float
    # Statistical range for the unknown underlying win probability.
    interval: BinomialInterval
    # Stable, human-readable reasons explaining every failed requirement.
    failed_checks: tuple[str, ...]


# Names for the three entries of decode_evaluation_targets' score flags, in
# order. require_finite_scores reports these names when a flag is False.
SCORE_FLAG_NAMES = (
    "launch scores of launchable sources",
    "target scores of legal targets",
    "value estimates",
)


def decode_evaluation_targets(
    output: PolicyOutput,
    observation: ObservationBatch,
    slot_body_id: torch.Tensor,
    active_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply eval-policy-v2's decision rules to a whole batch as tensors.

    The rules are:

    - launch from a launchable source only when its launch score is greater
      than zero;
    - send it to the legal target with the highest score, and on an exact tie
      to the target with the lowest physical body ID.

    Everything stays on the policy's device, so a GPU batch needs no
    per-slot reads back to Python. Inputs and their shapes:

    - ``output``: the policy's scores for ``games`` rows.
    - ``observation``: the same rows' masks. ``launchable_mask`` already
      requires at least one legal target, so a source without one never
      launches.
    - ``slot_body_id``: ``[games, 44]`` integers, the physical body ID in
      each observation slot. Tie-breaking uses these, not slot numbers.
    - ``active_rows``: ``[games]`` booleans, False for padding rows that
      hold no running game. Padding rows are still decoded, but never
      checked: their output is ignored by the caller.

    Returns two tensors on the same device:

    - packed targets, ``[games, 44]`` int64: ``-1`` for no launch, otherwise
      the target **slot** in 0..43, ready for ``EnvClient.step_packed``;
    - score flags, ``[3]`` booleans, True when every checked score in that
      group is finite (see ``SCORE_FLAG_NAMES``). Scores are checked only
      where they could influence a decision on an active row; an illegal
      target masked to ``-inf`` is fine. Without this check a NaN launch
      score would silently become "no launch".
    """

    launchable = observation.launchable_mask
    target_mask = observation.target_mask

    # Stage 1: launch decisions, [games, 44]. The rule is strictly greater
    # than zero, so a score of exactly 0.0 means no launch.
    launch = launchable & (output.launch_logits > 0)

    # Stage 2: best legal target for every source, [games, 44].
    # Illegal targets become -inf so they can never be the maximum, however
    # high the model scored them.
    scores = output.target_logits.masked_fill(~target_mask, float("-inf"))
    best = scores.amax(dim=-1, keepdim=True)
    # Every legal target whose score equals the maximum is a candidate. Exact
    # equality is intended: the rule breaks only exact ties by body ID.
    candidates = target_mask & (scores == best)
    # Among the candidates, choose the lowest physical body ID. Each source
    # row sees the same per-target body IDs, so they are broadcast across the
    # source dimension; non-candidates get the largest int64 so argmin never
    # picks them. A source with no candidate (no legal target) gets slot 0
    # here, but stage 3 discards it because such a source cannot launch.
    body_ids = slot_body_id.to(dtype=torch.long).unsqueeze(1).expand_as(candidates)
    largest = torch.iinfo(torch.long).max
    target = body_ids.masked_fill(~candidates, largest).argmin(dim=-1)

    # Stage 3: pack both decisions into the protocol's one-integer-per-slot
    # form: the target slot when launching, -1 otherwise.
    packed = torch.where(launch, target, torch.full_like(target, -1))

    # Stage 4: finiteness flags over the scores that matter on active rows.
    # A score that cannot affect a decision (padding row, non-launchable
    # source, illegal target) is not checked.
    checked_sources = launchable & active_rows.unsqueeze(1)
    checked_targets = target_mask & checked_sources.unsqueeze(2)
    flags = torch.stack(
        (
            torch.isfinite(output.launch_logits).logical_or(~checked_sources).all(),
            torch.isfinite(output.target_logits).logical_or(~checked_targets).all(),
            torch.isfinite(output.values).logical_or(~active_rows).all(),
        )
    )
    return packed, flags


def require_finite_scores(flags: Sequence[bool]) -> None:
    """Raise if any score group checked by the decoder was not finite.

    ``flags`` is the score-flag result of ``decode_evaluation_targets`` after
    its copy to the CPU. Raising here, before any action reaches Go, keeps a
    broken model from being scored as if it had simply chosen not to launch.
    """

    failed = [name for name, ok in zip(SCORE_FLAG_NAMES, flags) if not ok]
    if failed:
        raise ValueError(
            f"{EVALUATION_POLICY_V2} found non-finite policy output: {', '.join(failed)}"
        )


@dataclass(frozen=True)
class _PackedDecisions:
    """One turn's eval-policy-v2 actions, already copied to the CPU."""

    # Packed target row for each active environment ID, in ID order: -1 for
    # no launch, otherwise the target slot. Ready for EnvClient.step_packed.
    targets: dict[int, tuple[int, ...]]
    # Critic values for every batch row, still on the policy's device. Row i
    # is environment slot i. Read only when a diagnostic asks for them, so an
    # ordinary run pays for no extra GPU-to-CPU copy.
    values: torch.Tensor


def _decide_packed(
    policy: nn.Module,
    active: Mapping[int, Episode],
    slot_count: int,
    device: torch.device,
) -> _PackedDecisions:
    """Choose eval-policy-v2 actions for every active game in one batch.

    The batch always has ``slot_count`` rows, one per environment slot, so a
    game's scores never depend on how many other games are still running. A
    GPU may add numbers in a different order for a different batch size, and
    a fixed size removes that source of variation.

    Four stages: build the fixed batch, run the policy, decode on the device,
    then make the turn's single copy to the CPU and check it.
    """

    # Stage 1: row i is slot i. A slot with no running game repeats the
    # lowest active game's observation, so the row is valid model input; its
    # output is ignored and its scores are never checked.
    filler = active[min(active)]
    rows = [active.get(slot, filler) for slot in range(slot_count)]
    observations = [episode.observation for episode in rows]
    batch = ObservationBatch.from_observations(observations, device=device)
    slot_body_id = torch.from_numpy(
        np.stack([observation.slot_body_id for observation in observations])
    ).to(device)
    active_rows = torch.tensor(
        [slot in active for slot in range(slot_count)], dtype=torch.bool, device=device
    )

    # Stage 2 and 3: score and decode without building a backward graph and
    # without mixed precision. run_policy_for_collection leaves its input
    # checks as device flags instead of reading them back to Python.
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        output = run_policy_for_collection(policy, batch)
        packed, score_flags = decode_evaluation_targets(
            output, batch, slot_body_id, active_rows
        )
        flags = torch.cat((policy_validation_flags(batch), score_flags))
        # Stage 4: the turn's only required GPU-to-CPU copy. Packed targets
        # fill the first slot_count * 44 values; the flags follow as 1 or 0.
        payload = torch.cat((packed.reshape(-1), flags.to(packed.dtype))).cpu().tolist()

    action_count = slot_count * BODY_SLOTS
    policy_flag_count = flags.numel() - score_flags.numel()
    policy_ok = all(value == 1 for value in payload[action_count : action_count + policy_flag_count])
    if not policy_ok:
        # Rerun the eager input checks to raise their detailed error. Every
        # policy flag has a matching check, so falling through means the two
        # definitions drifted, and validation must stop rather than continue.
        validate_policy_observation(batch)
        raise RuntimeError("deferred policy validation flags disagreed")
    require_finite_scores(
        [value == 1 for value in payload[action_count + policy_flag_count :]]
    )

    targets = {
        slot: tuple(payload[slot * BODY_SLOTS : (slot + 1) * BODY_SLOTS])
        for slot in sorted(active)
    }
    return _PackedDecisions(targets, output.values)


@dataclass(frozen=True)
class EvaluationModel:
    """A checked private copy of a policy, ready for ``run_rotated_matches``.

    Made by ``prepare_evaluation_model``. Preparing it separately lets a
    caller allocate the copy before an irreversible step: held-out evaluation
    prepares it before claiming a protected pool, so an out-of-memory error
    while copying cannot spend the pool.
    """

    # The private copy, in eval mode, on ``device``. Nothing else holds it.
    model: nn.Module
    # Exact device of the copy, including the GPU index.
    device: torch.device
    # The decoder this copy was checked for.
    evaluation_policy: str


def _same_device(actual: torch.device, requested: torch.device) -> bool:
    """True when ``actual`` is ``requested``.

    "cuda" without an index means the current GPU, so indices are compared
    only when the request names one.
    """

    return actual.type == requested.type and (
        requested.index is None or actual.index == requested.index
    )


def prepare_evaluation_model(
    policy: nn.Module, evaluation_policy: str, device: torch.device | str
) -> EvaluationModel:
    """Check the request and make a private evaluation copy of the policy.

    Refused before any game starts: an evaluator other than eval-policy-v2, a
    policy that is not on the requested device, and non-float32 parameters.
    The copy lives on the policy's own device and is put in eval mode, so
    training's model, its device and its train/eval mode are never touched.
    Copying allocates the model's memory again on that device, so this is
    where an out-of-memory error appears.
    """

    if evaluation_policy != EVALUATION_POLICY_V2:
        raise ValueError(
            f"run_rotated_matches supports {EVALUATION_POLICY_V2!r}, got {evaluation_policy!r}"
        )
    requested = torch.device(device)
    actual = policy_device(policy)
    if not _same_device(actual, requested):
        raise ValueError(f"policy is on {actual}, but evaluation was requested on {requested}")
    if any(parameter.dtype != torch.float32 for parameter in policy.parameters()):
        raise ValueError(f"{EVALUATION_POLICY_V2} requires float32 policy parameters")
    model = copy.deepcopy(policy).eval()
    require_evaluation_mode(model, EVALUATION_POLICY_V2)
    return EvaluationModel(model, actual, evaluation_policy)


def run_rotated_matches(
    policy: nn.Module | EvaluationModel,
    *,
    command: Sequence[str],
    seeds: SeedRange,
    opponent: str,
    batch_size: int,
    players: int,
    evaluation_policy: str,
    device: torch.device | str,
    expected_metadata: Mapping[str, object] | None = None,
    on_turn: Callable[[TurnRecord], None] | None = None,
    on_game_end: Callable[[GameEndRecord], None] | None = None,
    on_progress: Callable[[ValidationProgress], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> EvaluationResult:
    """Play a complete deterministic validation sample through the Go process.

    The rotate schedule uses every requested map seed once from each seat:
    twice on a two-player map, four times on a four-player one, seat 0 first.
    This removes starting-seat advantage from the score. ``batch_size``
    controls how many games are active simultaneously; finished environment
    slots are reset with the next scheduled assignment until every
    seed-and-seat pair has completed.

    The policy acts through the fixed ``eval-policy-v2`` decoder and never
    samples training actions. ``evaluation_policy`` must name it, and
    ``device`` must be where ``policy`` already lives (normally the training
    GPU); both are checked before any game starts. The runner works on a
    private copy of the policy, so training's model is never changed. A
    caller that must allocate that copy before an irreversible step passes an
    ``EvaluationModel`` from ``prepare_evaluation_model`` instead, which is
    used as is after the same checks. The
    returned result contains every terminal outcome, the Go process metadata
    needed to identify the environment contract plus an ``"evaluation"``
    entry describing how the games were decided, and its rejected-action
    count. These validation turns do not update model weights or consume the
    training-step budget.

    ``batch_size`` is the number of reusable Go game slots, not the total game
    count. For 1,000 two-player seeds, the function can keep 64 games active
    and reset each finished slot onto the next assignment until all 2,000
    games are complete. Every turn scores exactly
    ``min(batch_size, total games)`` rows, one per slot, even when some slots
    have finished: see ``_decide_packed`` for why the shape stays fixed.

    ``players`` chooses two- or four-player maps for the Go process and so how
    many games each seed contributes. It has no default: a four-player caller
    that forgot it would otherwise quietly measure two-player maps.

    ``on_turn`` is an optional observer for diagnostics. When given, it is
    called once for every active game on every turn, after that game's action
    is chosen and before Go advances it, with a ``TurnRecord`` describing the
    decision. Calls within one turn follow ascending environment ID. The
    callback only observes: the actions sent to Go and every returned outcome
    are identical with or without it. An exception raised by the callback stops
    the evaluation and propagates to the caller.

    ``on_game_end`` is a second optional observer, called exactly once per game
    with a ``GameEndRecord`` holding the board after its final turn, before the
    slot is refilled. ``on_turn`` records begin with the starting board, so the
    two together cover every state of a game from start to finish. It observes
    only, like ``on_turn``.

    ``on_progress`` is an optional observer for long runs. It receives a
    ``ValidationProgress`` at most once every ``PROGRESS_INTERVAL_SECONDS``
    while games run, and always once more after the last game, with the final
    totals. ``clock`` is the monotonic wall clock used for every timing; tests
    replace it to control the reporting interval.

    The returned result carries ``RunnerTimings``. Time spent inside
    ``on_turn`` and ``on_progress`` counts toward elapsed time only, never
    toward the decide or environment phases, so a traced run cannot make
    either phase look slower than it is.
    """

    # Each seed creates one game per seat because the learner must occupy
    # every seat on the map.
    game_count = (seeds.end - seeds.start) * players
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    # Timing starts before the model copy is made, so elapsed covers the
    # runner's whole lifetime: preparing the model, Go startup and shutdown.
    # The two phase totals grow only around their own calls below.
    started = clock()
    if isinstance(policy, EvaluationModel):
        # Already copied and checked by the caller; confirm it was prepared
        # for this exact request rather than copying it a second time.
        if policy.evaluation_policy != evaluation_policy or not _same_device(
            policy.device, torch.device(device)
        ):
            raise ValueError(
                f"prepared model is for {policy.evaluation_policy!r} on {policy.device}, "
                f"but evaluation was requested with {evaluation_policy!r} on {device}"
            )
        prepared = policy
    else:
        prepared = prepare_evaluation_model(policy, evaluation_policy, device)
    model, device = prepared.model, prepared.device
    # Chosen once: the number of Go slots and of batch rows for every turn.
    slot_count = min(batch_size, game_count)
    outcomes: list[EvaluationOutcome] = []
    # Active games are indexed by reusable environment slot. A value changes
    # from Episode to StepResult as that game advances, but both carry the same
    # seed, seat, observation, and environment ID fields needed here.
    active: dict[int, Episode] = {}
    # Completed turns for the game in each active slot. Episode itself carries
    # no turn number, so a fresh reset starts at 0 and each StepResult supplies
    # the count for the following decision. Only TurnRecord uses this.
    turns: dict[int, int] = {}
    # Launch actions sent so far by the game in each active slot. A reset
    # starts the slot's count at 0; the count is copied into the outcome when
    # the game ends.
    launches: dict[int, int] = {}

    last_progress = started
    decide_seconds = 0.0
    environment_seconds = 0.0
    _reset_peak_memory(device)

    def progress() -> ValidationProgress:
        return ValidationProgress(
            finished=len(outcomes),
            total=game_count,
            active=len(active),
            elapsed_seconds=clock() - started,
            decide_seconds=decide_seconds,
            environment_seconds=environment_seconds,
        )

    # The evaluation context holds deterministic float32 settings for the
    # whole run and records them; training's settings return afterwards.
    # Opening EnvClient starts one long-lived Go process. Every exchange below
    # crosses its env-v3 binary boundary.
    with evaluation_context(device) as settings, EnvClient(
        command, players=players
    ) as environment:
        # Create no more slots than the total number of games. The rotate
        # schedule in Go decides the exact seed and learner seat for each reset.
        metadata = environment.create(
            slot_count,
            (seeds.start, seeds.end),
            opponent,
            schedule=SCHEDULE_ROTATE,
        )
        # Metadata describes the engine, observation schema, generator, and
        # opponent that actually ran. Rejecting a mismatch prevents results
        # from a different contract being attributed to this validation check.
        for key, expected in (expected_metadata or {}).items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"environment metadata {key!r} is {metadata.get(key)!r}, want {expected!r}"
                )

        # Fill every available slot with its first scheduled game. Reset gives
        # Python the initial observation; it does not execute a game turn.
        phase_started = clock()
        first_episodes = environment.reset(range(slot_count))
        environment_seconds += clock() - phase_started
        for episode in first_episodes:
            active[episode.env_id] = episode
            turns[episode.env_id] = 0
            launches[episode.env_id] = 0

        # Each pass advances every active game by one turn. Games finish at
        # different times, so completed slots are refilled below as needed.
        while active:
            # Choose one deterministic learner action for every active game.
            phase_started = clock()
            decisions = _decide_packed(model, active, slot_count, device)
            decide_seconds += clock() - phase_started
            # Count launches before stepping, so the action sent on a game's
            # terminal turn is included in that game's total.
            for env_id, row in decisions.targets.items():
                launches[env_id] += sum(target >= 0 for target in row)
            # Let a diagnostic observe each decision before the turn advances.
            # The records are built from the same decisions sent to Go below,
            # so observing never changes what is played. Values are copied
            # here, outside the decide timer, and read by slot row: row i is
            # slot i even when lower slots have gone idle.
            if on_turn is not None:
                values = decisions.values.cpu().tolist()
                for env_id, row in decisions.targets.items():
                    episode = active[env_id]
                    on_turn(
                        TurnRecord(
                            env_id=env_id,
                            seed=episode.seed,
                            seat=episode.seat,
                            turn=turns[env_id],
                            observation=episode.observation,
                            value=values[env_id],
                            actions=tuple(
                                SlotAction(launch=True, target=target)
                                if target >= 0
                                else SlotAction()
                                for target in row
                            ),
                        )
                    )
            # Ask Go to advance all of those games by one complete turn.
            phase_started = clock()
            results = environment.step_packed(decisions.targets)
            environment_seconds += clock() - phase_started
            finished: list[int] = []
            for result in results:
                # Every response must belong to a slot that was just stepped.
                # A foreign ID would mean Python and Go disagree about the
                # batch lifecycle, so the result cannot be trusted.
                if result.env_id not in active:
                    raise ValueError(f"unexpected environment result {result.env_id}")
                # Validation requires natural game completion. An externally
                # truncated game has no final win/draw/loss result to score.
                if result.truncated:
                    raise ValueError(f"validation game {result.env_id} was truncated")
                if result.terminated:
                    # Retain the replay identity, terminal reward, and end
                    # reason, then mark this reusable slot as available for
                    # another scheduled game. A diagnostic sees the final board
                    # first: it exists only in this result, and the slot's
                    # observation is replaced as soon as the slot is refilled.
                    if on_game_end is not None:
                        on_game_end(
                            GameEndRecord(
                                env_id=result.env_id,
                                seed=result.seed,
                                seat=result.seat,
                                turn=result.turn,
                                observation=result.observation,
                                reward=result.reward,
                                end_reason=result.end_reason,
                            )
                        )
                    outcomes.append(
                        EvaluationOutcome(
                            result.seed,
                            result.seat,
                            result.reward,
                            result.turn,
                            result.end_reason,
                            launches[result.env_id],
                        )
                    )
                    del active[result.env_id]
                    del turns[result.env_id]
                    del launches[result.env_id]
                    finished.append(result.env_id)
                else:
                    # An unfinished StepResult contains the next observation
                    # from which the evaluator chooses the following action.
                    active[result.env_id] = result
                    turns[result.env_id] = result.turn

            # Refill only the slots needed to reach the exact requested sample.
            # Near the end there may be more free slots than unstarted games.
            remaining = game_count - len(outcomes) - len(active)
            reset_ids = sorted(finished)[:remaining]
            if reset_ids:
                phase_started = clock()
                refilled = environment.reset(reset_ids)
                environment_seconds += clock() - phase_started
                for episode in refilled:
                    active[episode.env_id] = episode
                    turns[episode.env_id] = 0
                    launches[episode.env_id] = 0

            # Report at most once per interval while games are still running.
            # The final report is sent after shutdown, below.
            if on_progress is not None and active:
                now = clock()
                if now - last_progress >= PROGRESS_INTERVAL_SECONDS:
                    on_progress(progress())
                    last_progress = now

        # Before calculating a gate score, prove that every requested seed and
        # seat appears exactly once and that every terminal outcome is valid.
        checked = require_complete_results(outcomes, seeds, players)
        rejected_actions = environment.error_count

    # Go cannot know how Python chose its actions, so the evaluator, batch
    # size and numeric settings are recorded beside Go's own contract fields.
    # The settings were captured inside the evaluation context, before
    # training's settings were restored.
    metadata = dict(metadata)
    metadata["evaluation"] = {
        "policy": EVALUATION_POLICY_V2,
        "effective_batch_size": slot_count,
        **settings,
    }

    # Leaving the with block has shut Go down, so elapsed now covers the
    # runner's whole lifetime. The final progress report always follows,
    # even when the previous periodic report was moments ago.
    peak_allocated, peak_reserved = _peak_memory(device)
    timings = RunnerTimings(
        elapsed_seconds=clock() - started,
        decide_seconds=decide_seconds,
        environment_seconds=environment_seconds,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )
    if on_progress is not None:
        on_progress(
            ValidationProgress(
                finished=len(outcomes),
                total=game_count,
                active=0,
                elapsed_seconds=timings.elapsed_seconds,
                decide_seconds=decide_seconds,
                environment_seconds=environment_seconds,
            )
        )
    return EvaluationResult(checked, metadata, rejected_actions, timings)


def require_complete_results(
    outcomes: Sequence[EvaluationOutcome], seeds: SeedRange, players: int
) -> tuple[EvaluationOutcome, ...]:
    """Reject missing, duplicate, foreign, or malformed rotated outcomes.

    The rotate schedule promises exactly one result per board for each learner
    seat 0 to ``players``-1. Checking identities rather than only the list
    length catches a duplicated game that happens to replace a missing one.

    Each outcome must also have ended in a way that can be scored: the game
    finished, or the learner was eliminated, and an elimination must carry
    reward -1. These checks catch malformed terminal records. They cannot
    prove on their own that a +1 matches the final ship standings; Go stays the
    authority on scoring, and its reward tests pin "+1 only for strictly
    first by ships".

    The returned tuple contains those same valid outcomes sorted by seed and
    seat. Stable ordering makes saved evidence reproducible even when concurrent
    Go slots finish in a different order.
    """

    # This set is the complete expected Cartesian product of seeds and seats.
    # For two seeds on a four-player map, for example, it contains eight
    # distinct identities.
    expected = {
        (seed, seat) for seed in range(seeds.start, seeds.end) for seat in range(players)
    }
    actual: set[tuple[int, int]] = set()
    for outcome in outcomes:
        identity = (outcome.seed, outcome.seat)
        # Check duplication first so the error explains that repeated result
        # directly instead of later describing only the game it displaced.
        if identity in actual:
            raise ValueError(f"duplicate evaluation result for seed/seat {identity}")
        # A seed outside the selected range or a seat the map does not have
        # belongs to a different evaluation sample and must never affect this
        # score.
        if identity not in expected:
            raise ValueError(f"unexpected evaluation result for seed/seat {identity}")
        # Release statistics are binomial: win versus non-win. Restricting the
        # original reward first prevents an arbitrary positive number from
        # being silently counted as a legitimate win.
        if outcome.reward not in (-1.0, 0.0, 1.0):
            raise ValueError(f"invalid terminal reward {outcome.reward}")
        # END_ONGOING (0) would mean the game had not ended, so its reward is
        # not a result; any other value is not a known end reason at all.
        if outcome.end_reason not in (END_GAME_FINISHED, END_LEARNER_ELIMINATED):
            raise ValueError(
                f"invalid end reason {outcome.end_reason} for seed/seat {identity}"
            )
        # Go ends an eliminated learner's episode with -1. Any other reward
        # there disagrees with how the game ended.
        if outcome.end_reason == END_LEARNER_ELIMINATED and outcome.reward != -1.0:
            raise ValueError(
                f"eliminated learner has reward {outcome.reward}, want -1, "
                f"for seed/seat {identity}"
            )
        actual.add(identity)
    # Set subtraction finds every expected game that did not produce a result.
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"missing evaluation results: {missing}")
    # Sorting makes serialized outcomes independent of the order in which
    # concurrently running Go games happened to finish.
    return tuple(sorted(outcomes, key=lambda outcome: (outcome.seed, outcome.seat)))


def pipeline_gate(
    result: EvaluationResult,
    *,
    required_wins: int,
) -> PipelineGateResult:
    """Decide whether the small validation gate passes.

    This function does not play games, update the neural network, or load a
    checkpoint. ``run_rotated_matches`` has already produced ``result`` by
    playing the validation games. This function only judges that evidence.

    Passing requires both conditions below:

    1. The environment rejected zero submitted actions. A strong score cannot
       compensate for actions that violated the Go engine's rules.
    2. The checkpoint won at least ``required_wins`` games. Losses and draws do
       not increase ``EvaluationResult.wins`` and therefore count as non-wins.

    Repository correctness tests remain required evidence before a real run,
    but they are not runtime inputs to this function. Pretending that a
    hard-coded ``True`` proves a test ran would overstate the evidence recorded
    beside a validation checkpoint.

    The returned result includes every failed condition so the caller can see
    why the checkpoint failed. An empty failure list means the gate passed.

    This early gate answers whether the learning pipeline is working well
    enough to enter the longer release-readiness phase. It uses reusable
    validation boards and never consumes a protected final-evaluation pool.
    """

    # Keep checking after a failure so the resulting evidence explains every
    # observed problem rather than only the first one.
    failed: list[str] = []

    # A rejected action means Python and the authoritative Go rules disagreed
    # about legality. Training must diagnose that contract error rather than
    # treating the validation score as trustworthy.
    if result.rejected_actions != 0:
        failed.append("zero rejected actions")

    # Finally apply the learning threshold: on two-player maps, at least 160
    # wins from 200 rotated games against random-v1.
    if result.wins < required_wins:
        failed.append(f"at least {required_wins} wins")

    # No recorded failure means every independent requirement passed. The raw
    # counts travel with the decision so logs can show what was actually judged.
    return PipelineGateResult(not failed, result.wins, len(result.outcomes), tuple(failed))


def clopper_pearson_interval(
    wins: int,
    games: int,
    *,
    confidence_level: float = RELEASE_READINESS_CONFIDENCE,
) -> BinomialInterval:
    """Calculate the exact two-sided interval for a binomial win probability.

    Evaluation treats every game as either a win or a non-win. For ``k`` wins
    in ``n`` games, the Clopper-Pearson endpoints are beta-distribution
    quantiles with parameters ``(k, n-k+1)`` and ``(k+1, n-k)``. The zero-win
    and all-win cases have exact endpoints at zero and one respectively.

    No statistics package is added for this one calculation. The beta CDF is
    evaluated with a continued fraction, then a bounded binary search finds
    each requested quantile. This follows the exact beta-quantile definition;
    it does not substitute a normal approximation near the gate boundary.

    For example, 1,800 wins in 2,000 games gives an observed rate of 90%. The
    lower endpoint asks how low the underlying win probability could reasonably
    be after accounting for uncertainty from observing a finite sample.

    The interval is nominal: it treats every game as an independent draw with
    the same chance of a win. Games that share a scenario seed are the same
    board played from different seats, so they are not independent, and the
    stated confidence is not a coverage guarantee across unseen maps.
    """

    # Check integer types explicitly because Python considers True and False to
    # be integers. Accepting True as one win would hide a caller mistake.
    if isinstance(wins, bool) or not isinstance(wins, int):
        raise ValueError("wins must be an integer")
    if isinstance(games, bool) or not isinstance(games, int) or games < 1:
        raise ValueError("games must be a positive integer")
    if not 0 <= wins <= games:
        raise ValueError("wins must be between zero and games")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not math.isfinite(confidence_level)
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("confidence_level must be finite and in (0, 1)")

    # A 95% two-sided interval leaves 5% outside the interval. Dividing that
    # equally gives the 2.5% lower-tail and 2.5% upper-tail probabilities used
    # by the two beta quantiles below.
    tail_probability = (1.0 - float(confidence_level)) / 2.0

    # The lower endpoint asks: how small could the true win probability
    # reasonably be while still producing this many observed wins? With no
    # observed wins, zero is the exact endpoint and no beta inversion is needed.
    lower = (
        0.0
        if wins == 0
        else _inverse_regularized_beta(
            tail_probability,
            float(wins),
            float(games - wins + 1),
        )
    )
    # The upper endpoint asks the matching high-side question. When every game
    # is a win, one is the exact endpoint. Release readiness currently checks
    # only the lower endpoint, but saving both makes the reported interval
    # complete and reusable by held-out evaluation.
    upper = (
        1.0
        if wins == games
        else _inverse_regularized_beta(
            1.0 - tail_probability,
            float(wins + 1),
            float(games - wins),
        )
    )
    return BinomialInterval(float(confidence_level), lower, upper)


def release_readiness_gate(
    result: EvaluationResult,
    *,
    expected_games: int = RELEASE_READINESS_GAMES,
    required_wins: int = RELEASE_READINESS_WINS,
    minimum_lower_bound: float = RELEASE_READINESS_MINIMUM_LOWER_BOUND,
    confidence_level: float = RELEASE_READINESS_CONFIDENCE,
) -> ReleaseReadinessResult:
    """Judge completed validation games against the fixed release thresholds.

    This function neither plays games nor changes model weights. Its input is
    the result already returned by ``run_rotated_matches``. Passing requires
    the complete expected sample, enough wins, a nominal confidence lower bound
    at or above the configured floor, and zero rejected Go actions. Draws and
    losses are both non-wins in the interval calculation. The bound is nominal
    because games sharing a seed are not independent; see
    ``clopper_pearson_interval``.

    Training calls this method with its config's release-readiness values
    (1,000 games on two-player maps, 2,000 on four-player ones). Final
    evaluation calls the same method with the held-out counts. Sharing one
    method keeps both phases on the same tested counting and confidence
    calculation.
    """

    # These arguments are parameters because the later held-out evaluation uses
    # the same calculation with 2,000 games. The defaults above are the fixed
    # 1,000-game release-readiness contract.
    if (
        isinstance(expected_games, bool)
        or not isinstance(expected_games, int)
        or expected_games < 1
    ):
        raise ValueError("expected_games must be a positive integer")
    if (
        isinstance(required_wins, bool)
        or not isinstance(required_wins, int)
        or not 0 <= required_wins <= expected_games
    ):
        raise ValueError("required_wins must be between zero and expected_games")
    if (
        isinstance(minimum_lower_bound, bool)
        or not isinstance(minimum_lower_bound, (int, float))
        or not math.isfinite(minimum_lower_bound)
        or not 0.0 <= minimum_lower_bound <= 1.0
    ):
        raise ValueError("minimum_lower_bound must be finite and in [0, 1]")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not math.isfinite(confidence_level)
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("confidence_level must be finite and in (0, 1)")

    # Convert the individual game records into the summary a person will see in
    # evaluation evidence. A positive reward is a win, a negative reward is a
    # loss, and exactly zero is a draw.
    games = len(result.outcomes)
    wins = result.wins
    losses = sum(outcome.reward < 0.0 for outcome in result.outcomes)
    draws = sum(outcome.reward == 0.0 for outcome in result.outcomes)
    # A missing sample is a failed gate, but its partial score is still useful
    # diagnostic evidence. The vacuous [0, 1] interval handles an empty result
    # without pretending it measured the model's win probability.
    interval = (
        clopper_pearson_interval(
            wins,
            games,
            confidence_level=confidence_level,
        )
        if games
        else BinomialInterval(float(confidence_level), 0.0, 1.0)
    )

    # Evaluate every requirement instead of returning after the first failure.
    # A single validation run can therefore reveal an incomplete sample, an
    # action-protocol problem, and a weak score together.
    failed_checks: list[str] = []

    # A smaller sample must not quietly lower the denominator. A larger sample
    # is also rejected because it would no longer be the versioned fixed test.
    if games != expected_games:
        failed_checks.append(f"exactly {expected_games} completed games")

    # Any rejected action indicates that Python attempted something Go judged
    # illegal. The model cannot qualify while that correctness problem exists,
    # regardless of how many remaining games it won.
    if result.rejected_actions != 0:
        failed_checks.append("zero rejected actions")

    # This is the direct sample-score requirement: 900/1,000 is at least 90%.
    if wins < required_wins:
        failed_checks.append(f"at least {required_wins} wins")

    # The confidence check is deliberately more cautious than the observed
    # win rate. It asks whether the lower plausible underlying rate reaches the
    # contract floor, rather than judging only the point estimate above.
    #
    # At the configured release sizes this check does not bind after the win-count rule:
    # 900/1,000 has a lower bound near 87.97%, and 1,800/2,000 has one near
    # 88.60%, both already above 85%. Keeping the check explicit protects the
    # statistical meaning of the contract if a later version changes the sample
    # size while retaining the same observed-rate threshold. For example,
    # 90/100 wins has a lower bound near 82.38% and would fail this requirement.
    if interval.lower_bound < minimum_lower_bound:
        failed_checks.append(
            f"{confidence_level:.0%} confidence lower bound at least "
            f"{minimum_lower_bound:.0%}"
        )

    # Return both the overall decision and the evidence used to make it. Later
    # orchestration can serialize these fields without recalculating the gate.
    return ReleaseReadinessResult(
        passed=not failed_checks,
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / games if games else 0.0,
        interval=interval,
        failed_checks=tuple(failed_checks),
    )


def _inverse_regularized_beta(probability: float, alpha: float, beta: float) -> float:
    """Find a beta quantile inside the known probability domain ``[0, 1]``.

    A cumulative distribution function answers "how much probability lies at
    or below this value?" A quantile reverses that question: given the amount
    of probability, find the value. There is no simple closed formula for the
    beta quantile, so this function repeatedly narrows a bracket around it.

    ``probability`` is the requested cumulative amount. ``alpha`` and ``beta``
    describe the curve derived from wins and non-wins. The returned value is a
    possible underlying win probability between zero and one.
    """

    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0

    # A probability is always between zero and one, so these are guaranteed
    # starting bounds. Each loop keeps the half that still contains the answer.
    lower = 0.0
    upper = 1.0
    # Binary search cannot leave the probability domain and gives predictable
    # convergence even for the asymmetric beta shapes at zero/all-win edges.
    # Sixty iterations reduce the bracket below double-precision resolution.
    for _ in range(60):
        midpoint = (lower + upper) / 2.0
        # If too little beta probability lies below midpoint, the desired
        # quantile must be higher. Otherwise it is at or below midpoint.
        if _regularized_incomplete_beta(midpoint, alpha, beta) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _regularized_incomplete_beta(value: float, alpha: float, beta: float) -> float:
    """Evaluate a beta CDF using its stable continued-fraction forms.

    This is the forward calculation used by the binary search above. ``value``
    is a possible win probability, while ``alpha`` and ``beta`` describe the
    beta distribution derived from wins and non-wins. The returned number is
    the fraction of that distribution lying at or below ``value``.

    This is the forward question used by ``_inverse_regularized_beta``: the
    inverse method proposes a value, and this method reports whether its
    cumulative probability is too low or too high.
    """

    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0

    # The common factor is evaluated in log space so the parameters used by
    # 1,000-game gates do not overflow intermediate gamma or power values.
    factor = math.exp(
        math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + alpha * math.log(value)
        + beta * math.log1p(-value)
    )
    # The same mathematical value has two equivalent continued-fraction forms.
    # Using the direct form below the split and the complemented form above it
    # avoids subtracting nearly equal numbers in the less stable region.
    split = (alpha + 1.0) / (alpha + beta + 2.0)
    if value < split:
        return factor * _beta_continued_fraction(alpha, beta, value) / alpha
    return 1.0 - (
        factor
        * _beta_continued_fraction(beta, alpha, 1.0 - value)
        / beta
    )


def _beta_continued_fraction(alpha: float, beta: float, value: float) -> float:
    """Evaluate the incomplete-beta continued fraction to float64 precision.

    A continued fraction represents the beta integral as a sequence of smaller
    corrections. The loop uses the standard modified-Lentz recurrence: ``c``
    and ``d`` carry the numerator and denominator corrections, while ``result``
    accumulates their product. The details are numerical machinery rather than
    new evaluation policy; changing them must preserve the tested endpoints.

    Each loop calculates two smaller corrections, multiplies them into the
    running answer, and stops when a correction is so close to one that another
    iteration cannot meaningfully change a float64 result. Tiny-value guards
    prevent division by zero along the way.
    """

    # Stop after a generous fixed limit if the recurrence does not settle. The
    # supported 1,000- and 2,000-game calculations normally finish much sooner.
    maximum_iterations = 200
    # ``change`` below is a multiplier. When it differs from 1 by no more than
    # this amount, another level cannot meaningfully affect a float64 result.
    tolerance = 3.0e-14
    # Modified Lentz evaluation takes reciprocals. This tiny nonzero substitute
    # prevents division by zero without being large enough to influence a
    # normal intermediate value.
    smallest = 1.0e-300

    # These repeated combinations are named once to make the two coefficient
    # formulas below easier to relate to the original alpha and beta inputs.
    total = alpha + beta
    alpha_plus_one = alpha + 1.0
    alpha_minus_one = alpha - 1.0

    # ``c`` follows the numerator side of the modified-Lentz recurrence. Its
    # neutral starting value is one because the later updates divide by it.
    c = 1.0
    # ``d`` starts with the continued fraction's leading denominator term.
    # ``total * value / alpha_plus_one`` is the first correction to that term.
    d = 1.0 - total * value / alpha_plus_one
    # Replacing a near-zero denominator prevents its reciprocal from becoming
    # infinite while approximating the recurrence's limiting behavior.
    if abs(d) < smallest:
        d = smallest
    # Modified Lentz carries the reciprocal denominator between corrections,
    # so invert the leading term before using it.
    d = 1.0 / d
    # The first approximation contains only the leading denominator. Every
    # loop iteration below multiplies two further corrections into this value.
    result = d

    for iteration in range(1, maximum_iterations + 1):
        # Each iteration has two coefficients. Applying both completes one
        # level of the continued fraction before convergence is checked.
        # ``twice`` is the even index 2m used by both coefficient formulas.
        twice = 2 * iteration

        # This is the first, or even, coefficient for level m. It combines the
        # current level with the beta shape and the candidate probability. The
        # sign can change after iteration passes beta; that is expected in this
        # algebraic representation of the beta integral.
        coefficient = (
            iteration
            * (beta - iteration)
            * value
            / ((alpha_minus_one + twice) * (alpha + twice))
        )

        # Update the denominator recurrence with this coefficient and the
        # reciprocal denominator retained from the preceding half-level.
        d = 1.0 + coefficient * d
        # Guard the value before taking its reciprocal below.
        if abs(d) < smallest:
            d = smallest

        # Update the numerator recurrence. Dividing by the previous ``c`` is
        # the numerator-side counterpart of multiplying the prior ``d``.
        c = 1.0 + coefficient / c
        # Guard ``c`` because the next half-level will divide by it.
        if abs(c) < smallest:
            c = smallest

        # Turn the new denominator into reciprocal form, then multiply the
        # approximation by numerator correction ``c`` and reciprocal ``d``.
        d = 1.0 / d
        result *= d * c

        # This is the second, or odd, coefficient for the same level m. Its
        # leading minus sign is part of the incomplete-beta continued fraction,
        # rather than an indication that a probability has become negative.
        coefficient = -(
            (alpha + iteration)
            * (total + iteration)
            * value
            / ((alpha + twice) * (alpha_plus_one + twice))
        )

        # Repeat the denominator update for the second half-level.
        d = 1.0 + coefficient * d
        if abs(d) < smallest:
            d = smallest

        # Repeat the numerator update and its near-zero protection.
        c = 1.0 + coefficient / c
        if abs(c) < smallest:
            c = smallest

        # The product of the updated numerator and reciprocal denominator is
        # this level's final multiplicative correction.
        d = 1.0 / d
        change = d * c
        # Apply that correction before testing whether it is close enough to
        # one to declare convergence.
        result *= change
        # A multiplicative correction of one would make no change. Once the
        # correction is within float64 tolerance of one, further iterations
        # cannot materially move the reported confidence endpoints.
        if abs(change - 1.0) <= tolerance:
            return result

    # Reaching this point would mean the numerical method did not meet its
    # promised precision. Failing is safer than returning an uncertain value
    # that could approve the wrong checkpoint at the 85% boundary.
    raise ArithmeticError("incomplete beta continued fraction did not converge")
