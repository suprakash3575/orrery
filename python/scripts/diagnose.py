#!/usr/bin/env python3
"""Diagnose an rl-v2 training run: its training history and how a checkpoint plays.

The script never trains or changes a model. It has two parts:

- **Training history** reads the run's ``metrics.csv``: one row per PPO update,
  written while training ran. It needs no GPU or Go and works on any rl-v2
  run, including a ten-update smoke test.
- **Game diagnosis** plays one of the run's checkpoints on reusable
  validation boards, from every seat, against the run's training opponent
  (or another one chosen with ``--opponent``), with the deterministic
  evaluation decoder. Every decision is recorded, so
  the report can show results by seat, how well the critic predicted each
  result, and turn-by-turn traces and charts of individual games.

It produces one output folder containing:

- ``report.md`` and ``report.json``: the run identity, training milestones,
  results by seat, behaviour, value error by turn, and links to each game.
- ``charts/``: PNG charts of training history, value error, results by seat,
  and one chart per selected game. Charts need the optional ``analysis`` group.
- ``traces/``: one NDJSON file per lost or drawn game, plus a few wins, one
  line per turn, with real game values first.

Run from the ``python/`` directory::

    uv run --group analysis python -m scripts.diagnose \\
        --run-dir ../runs/<run> --out ../runs/diagnosis-<name>

``--no-games`` keeps only the training history (seconds, no GPU or Go).
``--seeds N`` sets how many validation boards are played; each is played from
every seat, so a four-player run plays ``4 × N`` games. ``--no-charts`` skips
the charts and the matplotlib dependency. ``--opponent SPEC`` plays a different
opponent than the one the run trained against, for example
``random-v1:launch_probability=0.3,max_fraction=0.8,min_fraction=0.2``.
``--untrained`` plays a never-trained model rebuilt from the run's learner
seed instead of a checkpoint, as the baseline that shows how much training
helped; its report records a weights fingerprint so rebuilds can be compared.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from orrery.checkpoint import state_dict_sha256
from orrery.config import SeedRange, run_config_from_mapping
from orrery.env_client import END_LEARNER_ELIMINATED, Observation
from orrery.evaluate import DEFAULT_ENV_COMMAND
from orrery.evaluation_context import configure_cuda_workspace, resolve_evaluation_device
from orrery.evaluator import (
    EvaluationOutcome,
    GameEndRecord,
    TurnRecord,
    checked_evaluation_result,
    evaluation_metadata,
    run_rotated_matches,
)
from orrery.jsonio import read_json_object
from orrery.protocol import PLAYER_FEATURE_SHIPS, SHIP_SCALE
from orrery.train import initial_policy
from scripts.validate_checkpoint import load_run_checkpoint, validation_seed_prefix


# The report's name for a never-trained model rebuilt from a run's learner
# seed. "Reconstructed" because no update-0 checkpoint was ever saved: the
# weights are recreated, and matched by fingerprint, not loaded from a file.
UNTRAINED_BASELINE = "reconstructed untrained baseline"

# Default number of reusable validation boards: the pipeline gate's sample
# size. Each board is played from every seat.
DEFAULT_SEEDS = 100

# Observation column numbers for obs-v2. The authoritative names live in
# mlenv/observation.go (FeatureOwnerMe … FeatureOwnerNeutral,
# FeatureIsComet); these mirror them.
FEATURE_OWNER_ME = 5
FEATURE_OWNER_NEUTRAL = 9
FEATURE_IS_COMET = 12

# Player rows are seat-relative roles, in this fixed order: row 0 is always
# the learner, rows 1-3 the clockwise, opposite, and counter-clockwise
# opponents. A two-player opponent sits opposite, in row 2. Body ownership
# columns 5-8 use the same role order.
ROLES = ("me", "clockwise", "opposite", "counter_clockwise")

# Turns are grouped into ranges of this size for the value-error summary.
VALUE_ERROR_BUCKET_TURNS = 25

# Training-history panels, as (title, columns). A panel with two columns plots
# two series that share a unit and a scale. win_rate and launches_per_turn are
# recalculated by training_history() rather than read as logged.
TRAINING_PANELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Win rate in finished training games", ("win_rate",)),
    ("Game length (turns)", ("mean_episode_length",)),
    ("Ship advantage at game end", ("mean_ship_advantage",)),
    ("Value loss (critic error)", ("value_loss",)),
    ("Explained variance (critic fit)", ("explained_variance",)),
    ("Entropy (exploration, nats)", ("launch_entropy", "target_entropy")),
    ("Policy loss", ("policy_loss",)),
    ("Approximate KL (update size)", ("approximate_kl",)),
    ("Clip fraction", ("clip_fraction",)),
    ("Gradient size before clipping", ("mean_gradient_norm", "max_gradient_norm")),
    ("No-launch rate", ("no_launch_rate",)),
    ("Launches per turn", ("launches_per_turn",)),
)
# Columns that describe finished games. An update in which no game finished
# has no measurement for them, even though metrics.csv stores 0.
GAME_COLUMNS = ("win_rate", "mean_episode_length", "mean_ship_advantage", "launches_per_turn")


# ---------------------------------------------------------------------------
# Stage 1: read the training history.
#
# metrics.csv holds text. Each row is converted to numbers once, and the
# per-game columns are recalculated so an update with no finished games reads
# as "not measured" (NaN, a gap in the chart) instead of a misleading zero.
# ---------------------------------------------------------------------------


def training_history(rows: Sequence[Mapping[str, str]]) -> list[dict[str, float]]:
    """Convert metrics.csv rows to numbers and add the derived columns.

    ``launches_per_turn`` divides launches per finished game by that game's
    length: launch counts grow with game length, so this shows play style
    without the length effect.
    """

    history = []
    for row in rows:
        values = {key: float(value) for key, value in row.items()}
        games = values["wins"] + values["losses"] + values["draws"]
        values["games"] = games
        if games:
            values["win_rate"] = values["wins"] / games
            length = values["mean_episode_length"]
            values["launches_per_turn"] = (
                values["launches_per_episode"] / length if length > 0 else math.nan
            )
        else:
            for column in GAME_COLUMNS:
                values[column] = math.nan
        history.append(values)
    return history


def training_milestones(history: Sequence[Mapping[str, float]], points: int = 10) -> list[dict[str, float]]:
    """Pick evenly spaced updates, from the first to the last.

    ``points`` intervals give ``points + 1`` rows at most: the start, every
    tenth of training by default, and the end.
    """

    if not history:
        return []
    if points <= 0:
        raise ValueError("points must be positive")
    indices = sorted({round(step * (len(history) - 1) / points) for step in range(points + 1)})
    columns = ["update", "environment_steps", "games"] + [
        column for _, group in TRAINING_PANELS for column in group
    ]
    return [{column: history[index][column] for column in columns} for index in indices]


def training_totals(history: Sequence[Mapping[str, float]], kl_warning: float) -> dict[str, Any]:
    """Whole-run game counts and the updates whose KL passed the warning line.

    The win rate is total wins over total finished games, not the average of
    each update's rate: updates finish different numbers of games.
    """

    wins = sum(row["wins"] for row in history)
    games = sum(row["games"] for row in history)
    return {
        "updates": len(history),
        "environment_steps": history[-1]["environment_steps"] if history else 0,
        "games": games,
        "wins": wins,
        "losses": sum(row["losses"] for row in history),
        "draws": sum(row["draws"] for row in history),
        "win_rate": wins / games if games else math.nan,
        "kl_warning_updates": [
            int(row["update"]) for row in history if row["approximate_kl"] > kl_warning
        ],
    }


def read_metrics(path: Path) -> list[dict[str, str]]:
    """Read the training run's metrics.csv as one dictionary per PPO update."""

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no metric rows")
    return rows


# ---------------------------------------------------------------------------
# Stage 2: describe one decision in game terms.
#
# The hook in run_rotated_matches hands over a TurnRecord holding the raw
# observation arrays, which the evaluator keeps reusing. Each record is
# reduced straight away to a few game facts: keeping every observation for
# hundreds of games would also use a lot of memory. Ship counts stay in the
# observation's log scale here and are decoded when a report is written.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnSummary:
    """The game facts behind one learner decision."""

    # Completed turns when the decision was made; 0 is the first decision.
    turn: int
    # Critic estimate of the final result from the learner's view.
    value: float
    # Planets (comets excluded) owned by each role, in ROLES order.
    planets: tuple[int, ...]
    # Comets owned by each role, in ROLES order. Comets produce ships like
    # planets, so a player with a comet but no planet still has income.
    comets: tuple[int, ...]
    # Planets nobody owns.
    neutral_planets: int
    # Each role's total ships, including fleets in flight, still in the
    # observation's log scale: log1p(ships) / log1p(SHIP_SCALE).
    ships_scaled: tuple[float, ...]
    # Whether each role's row is a real player in this game. A two-player
    # game has only "me" and "opposite"; an eliminated player stays present
    # with zero ships.
    present: tuple[bool, ...]
    # Launches sent this turn as (source body ID, target body ID) pairs.
    launches: tuple[tuple[int, int], ...]
    # Owned bodies that had at least one legal target, so could have launched.
    launchable_sources: int


@dataclass(frozen=True)
class FinalState:
    """The board after a game's final turn, from the learner's view.

    TurnSummary describes boards before decisions, so the board the final
    decision produced is recorded here instead. A capture or a loss on the
    last move shows up only in this state.
    """

    # Completed turns when the game ended; equals GameDiagnosis.turns.
    turn: int
    # The same board facts as TurnSummary, with the same meanings.
    planets: tuple[int, ...]
    comets: tuple[int, ...]
    neutral_planets: int
    ships_scaled: tuple[float, ...]
    present: tuple[bool, ...]


@dataclass(frozen=True)
class _BoardFacts:
    """The board facts shared by TurnSummary and FinalState, as plain values."""

    planets: tuple[int, ...]
    comets: tuple[int, ...]
    neutral_planets: int
    ships_scaled: tuple[float, ...]
    present: tuple[bool, ...]


def _board_facts(observation: Observation) -> _BoardFacts:
    """Read planet and comet counts, ships, and present players from one observation.

    Returns plain Python values, so nothing refers to the observation's arrays
    after the call: the evaluator may reuse those arrays for another game.
    """

    bodies = observation.bodies
    real = observation.body_mask.astype(bool)
    comet_bodies = real & (bodies[:, FEATURE_IS_COMET] > 0.5)
    planet_bodies = real & ~comet_bodies

    # Ownership columns are one-hot, so a value above 0.5 means "set".
    def owned(kind: np.ndarray, column: int) -> int:
        return int(np.count_nonzero(kind & (bodies[:, column] > 0.5)))

    return _BoardFacts(
        planets=tuple(owned(planet_bodies, FEATURE_OWNER_ME + row) for row in range(len(ROLES))),
        comets=tuple(owned(comet_bodies, FEATURE_OWNER_ME + row) for row in range(len(ROLES))),
        neutral_planets=owned(planet_bodies, FEATURE_OWNER_NEUTRAL),
        ships_scaled=tuple(
            float(observation.players[row, PLAYER_FEATURE_SHIPS]) for row in range(len(ROLES))
        ),
        present=tuple(bool(flag) for flag in observation.player_mask),
    )


def summarize_final(record: GameEndRecord) -> FinalState:
    """Reduce one GameEndRecord to the facts the report and traces use."""

    facts = _board_facts(record.observation)
    return FinalState(
        turn=record.turn,
        planets=facts.planets,
        comets=facts.comets,
        neutral_planets=facts.neutral_planets,
        ships_scaled=facts.ships_scaled,
        present=facts.present,
    )


def summarize_turn(record: TurnRecord) -> TurnSummary:
    """Reduce one TurnRecord to the facts the report and charts use."""

    observation = record.observation
    facts = _board_facts(observation)

    # A source could launch only if the learner owns it and Go marked at least
    # one legal target for it.
    launchable = observation.owned_mask.astype(bool) & observation.target_mask.any(axis=1)

    # Body IDs, not slot numbers, identify bodies in the trace. Slot order is
    # only the observation's layout.
    launches = tuple(
        (int(observation.slot_body_id[slot]), int(observation.slot_body_id[action.target]))
        for slot, action in enumerate(record.actions)
        if action.launch
    )
    return TurnSummary(
        turn=record.turn,
        value=float(record.value),
        planets=facts.planets,
        comets=facts.comets,
        neutral_planets=facts.neutral_planets,
        ships_scaled=facts.ships_scaled,
        present=facts.present,
        launches=launches,
        launchable_sources=int(np.count_nonzero(launchable)),
    )


class TurnRecorder:
    """Collect every recorded board, grouped by game.

    ``games`` holds a TurnSummary for every decision (the boards before each
    move, starting with the initial board) and ``finals`` the board after the
    last move. Together they cover a game from start to finish. The rotate
    schedule plays each (seed, seat) pair exactly once, so it identifies one
    game. Environment slots are reused, so they cannot.
    """

    def __init__(self) -> None:
        self.games: dict[tuple[int, int], list[TurnSummary]] = {}
        self.finals: dict[tuple[int, int], FinalState] = {}

    def __call__(self, record: TurnRecord) -> None:
        self.games.setdefault((record.seed, record.seat), []).append(summarize_turn(record))

    def game_ended(self, record: GameEndRecord) -> None:
        key = (record.seed, record.seat)
        if key in self.finals:
            raise ValueError(f"seed {record.seed} seat {record.seat}: game ended twice")
        self.finals[key] = summarize_final(record)


# ---------------------------------------------------------------------------
# Stage 3: turn the recorded games into measurements.
#
# These small functions hold every calculation that could mislead if it were
# wrong, so tests can check each one by hand.
# ---------------------------------------------------------------------------


def decode_ships(scaled: float) -> float:
    """Undo obs-v2's ``log1p(ships) / log1p(SHIP_SCALE)`` scaling.

    obs-v2 does not clip this feature, so totals above SHIP_SCALE decode
    exactly rather than stopping at a cap.
    """

    return math.expm1(scaled * math.log1p(SHIP_SCALE))


def discounted_result(reward: float, game_turns: int, turn: int, gamma: float) -> float:
    """The training target for a decision: the final result, discounted.

    The final reward arrives on the step that ends the game. A decision made at
    ``turn`` reaches that step after ``game_turns - 1 - turn`` more turns, and
    each turn scales the reward by ``gamma``. This is the return the critic was
    trained to predict.
    """

    if not 0 <= turn < game_turns:
        raise ValueError(f"turn {turn} is outside a {game_turns}-turn game")
    return reward * gamma ** (game_turns - 1 - turn)


@dataclass(frozen=True)
class GameDiagnosis:
    """One finished game with its per-turn summaries."""

    seed: int
    seat: int
    # Terminal reward from the learner's view: 1 sole winner, 0 tie, -1 loss.
    reward: float
    # The learner's turns before the game ended for it. An eliminated
    # learner's game ends at its elimination, not at the match's end.
    turns: int
    # True when the learner lost every planet and ship before the match ended.
    eliminated: bool
    # Boards before each decision, turns 0 to turns - 1; turn 0 is the start.
    summaries: tuple[TurnSummary, ...]
    # The board after the final decision, at turn ``turns``.
    final: FinalState


def outcome_name(reward: float) -> str:
    """Name a terminal reward."""

    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def results_by_seat(games: Sequence[GameDiagnosis], players: int) -> dict[int, dict[str, int]]:
    """Count wins, draws, losses, and eliminations for every learner seat.

    Every seat appears, even one without games, so a missing seat shows as
    zeros rather than disappearing from the table.
    """

    counts: dict[int, Counter[str]] = {seat: Counter() for seat in range(players)}
    for game in games:
        counts[game.seat][outcome_name(game.reward)] += 1
        counts[game.seat]["eliminated"] += game.eliminated
    return {
        seat: {name: counter.get(name, 0) for name in ("win", "draw", "loss", "eliminated")}
        for seat, counter in sorted(counts.items())
    }


# The measurements below separate stronger play from merely surviving longer:
# a learner whose fleets drift on after its planets are gone has a long game
# but no territory. Each reads the learner's own row (role 0, "me").


def _learner_planets_by_turn(game: GameDiagnosis) -> list[tuple[int, int]]:
    """(turn, learner planets) for every recorded board, start to finish.

    The boards are the one before each move, starting with the initial board,
    and then the final board after the last move.
    """

    return [(summary.turn, summary.planets[0]) for summary in game.summaries] + [
        (game.final.turn, game.final.planets[0])
    ]


def most_planets_owned(game: GameDiagnosis) -> int:
    """The most planets the learner held at the same time, over every board.

    The final board is included, so a capture on the last move counts. Comets
    are excluded here; they count towards productive time instead.
    """

    return max(planets for _turn, planets in _learner_planets_by_turn(game))


def first_two_planets_turn(game: GameDiagnosis) -> int | None:
    """The first turn on which the learner held two or more planets at once.

    ``None`` means "not reached", never turn 0: a game that ended first has no
    such turn. Planets must be held together, so recapturing a planet after
    losing the home, which is one planet again, does not count.
    """

    return next((turn for turn, planets in _learner_planets_by_turn(game) if planets >= 2), None)


def turns_without_productive_body(game: GameDiagnosis) -> tuple[int, int]:
    """Count the learner's action turns with no planet and no comet.

    Returns ``(turns without a productive body, action turns)``. Action turns
    are the boards on which the learner was asked to move, so the final board,
    which no move follows, is not one. A turn with a body but no legal launch
    still counts as productive: the body keeps producing ships.
    """

    without = sum(
        1 for summary in game.summaries if summary.planets[0] == 0 and summary.comets[0] == 0
    )
    return without, len(game.summaries)


def game_measurements(game: GameDiagnosis) -> dict[str, Any]:
    """Every measurement for one game, identified by its map seed and seat."""

    without, action_turns = turns_without_productive_body(game)
    return {
        "seed": game.seed,
        "seat": game.seat,
        "result": outcome_name(game.reward),
        "end_reason": "learner_eliminated" if game.eliminated else "game_finished",
        "turns": game.turns,
        "most_planets": most_planets_owned(game),
        "first_two_planets_turn": first_two_planets_turn(game),
        "action_turns": action_turns,
        "turns_without_productive_body": without,
        # The share is undefined, not zero, for a game with no action turns.
        "share_without_productive_body": without / action_turns if action_turns else None,
    }


def measurement_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Totals and per-game averages of game_measurements rows.

    Averages give every game equal weight, so one long game cannot dominate.
    The productive-time average covers only games with at least one action
    turn, and reports how many those were.
    """

    if not rows:
        raise ValueError("at least one game is required")
    shares = [row["share_without_productive_body"] for row in rows]
    shares = [share for share in shares if share is not None]
    results = Counter(row["result"] for row in rows)
    return {
        "games": len(rows),
        "wins": results["win"],
        "draws": results["draw"],
        "losses": results["loss"],
        "eliminations": sum(row["end_reason"] == "learner_eliminated" for row in rows),
        "mean_most_planets": sum(row["most_planets"] for row in rows) / len(rows),
        "reached_two_planets": sum(row["first_two_planets_turn"] is not None for row in rows),
        "games_with_action_turns": len(shares),
        "mean_share_without_productive_body": sum(shares) / len(shares) if shares else None,
    }


def behaviour_summary(games: Sequence[GameDiagnosis]) -> dict[str, float]:
    """Summarize how the deterministic policy played its validation games.

    The no-launch rate counts source-turns: every turn, each owned body that
    had a legal target is one chance to launch. It is the share of those
    chances the policy passed up. Training logs a similar rate for sampled,
    exploring actions; this one is for the evaluation decoder.
    """

    if not games:
        raise ValueError("at least one game is required")
    launches = sum(len(summary.launches) for game in games for summary in game.summaries)
    chances = sum(summary.launchable_sources for game in games for summary in game.summaries)
    turns = sum(game.turns for game in games)
    return {
        "games": len(games),
        "launches_per_game": launches / len(games),
        "launches_per_turn": launches / turns if turns else 0.0,
        "no_launch_rate": 1.0 - launches / chances if chances else 0.0,
        "mean_game_turns": turns / len(games),
        "eliminations": sum(game.eliminated for game in games),
    }


def value_error_by_turn(
    games: Sequence[GameDiagnosis],
    gamma: float,
    bucket_turns: int = VALUE_ERROR_BUCKET_TURNS,
) -> list[dict[str, float]]:
    """Measure how well the critic predicted each game's result, by turn range.

    For every decision, compare the value estimate with the discounted final
    result. ``mean_absolute_error`` is the average gap. ``sign_agreement`` is
    how often the estimate's sign matched the result's: did the critic know
    whether it was winning? Draws have no sign, so they count toward the error
    but not toward sign agreement. ``mean_value`` is the average estimate
    itself, which shows how hopeful the critic was at that stage.
    """

    if bucket_turns <= 0:
        raise ValueError("bucket_turns must be positive")
    errors: dict[int, list[float]] = {}
    values: dict[int, list[float]] = {}
    signs: dict[int, list[bool]] = {}
    for game in games:
        for summary in game.summaries:
            target = discounted_result(game.reward, game.turns, summary.turn, gamma)
            bucket = summary.turn // bucket_turns
            errors.setdefault(bucket, []).append(abs(summary.value - target))
            values.setdefault(bucket, []).append(summary.value)
            if target != 0:
                signs.setdefault(bucket, []).append((summary.value > 0) == (target > 0))
    rows = []
    for bucket in sorted(errors):
        bucket_signs = signs.get(bucket, [])
        rows.append(
            {
                "first_turn": bucket * bucket_turns,
                "last_turn": (bucket + 1) * bucket_turns - 1,
                "decisions": len(errors[bucket]),
                "mean_absolute_error": sum(errors[bucket]) / len(errors[bucket]),
                "mean_value": sum(values[bucket]) / len(values[bucket]),
                "sign_agreement": (
                    sum(bucket_signs) / len(bucket_signs) if bucket_signs else math.nan
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Stage 4: write traces a person can read.
#
# Each line is one turn, with real game values first: planets and real ship
# counts for each player present, launches by body ID, then the critic's
# estimate beside the result it was trying to predict.
# ---------------------------------------------------------------------------


def trace_lines(game: GameDiagnosis, gamma: float) -> list[str]:
    """Render one game as NDJSON lines: one per decision, then the final board.

    Decision lines describe the board before that turn's move. The last line,
    marked ``"final": true``, is the board after the final move, so a capture
    or loss on that move is visible; it has no launches or value because no
    decision follows it.
    """

    def board(state: TurnSummary | FinalState) -> dict[str, Any]:
        rows = [row for row, present in enumerate(state.present) if present]
        ships = {ROLES[row]: round(decode_ships(state.ships_scaled[row])) for row in rows}
        # The learner's lead over the strongest opponent: winning means
        # finishing strictly first, so that is the opponent to beat.
        opponents = [ships[ROLES[row]] for row in rows if row != 0]
        return {
            "turn": state.turn,
            "planets": {ROLES[row]: state.planets[row] for row in rows},
            "neutral_planets": state.neutral_planets,
            "ships": ships,
            "ship_lead": ships["me"] - max(opponents, default=0),
        }

    lines = []
    for summary in game.summaries:
        lines.append(
            json.dumps(
                {
                    **board(summary),
                    "launches": [{"from": source, "to": target} for source, target in summary.launches],
                    "value": round(summary.value, 4),
                    "discounted_result": round(
                        discounted_result(game.reward, game.turns, summary.turn, gamma), 4
                    ),
                }
            )
        )
    lines.append(
        json.dumps({**board(game.final), "final": True, "result": outcome_name(game.reward)})
    )
    return lines


# ---------------------------------------------------------------------------
# Stage 5: charts.
#
# Charts are static PNGs drawn with matplotlib's non-interactive Agg backend,
# so no window opens. Every chart keeps one y-axis: measures on different
# scales get separate panels rather than a second axis. Colours come from a
# colour-blind-checked categorical palette used in a fixed order, and a
# player role keeps the same colour in every chart. Text and reference lines
# stay in neutral ink rather than series colours.
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"
# Categorical slots 1-4 of the reference palette (blue, orange, aqua, yellow),
# validated for light mode. Slot n colours ROLES[n] in game charts.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
LINE_WIDTH = 1.5
# Short runs have few points; markers keep single updates visible.
MARKER_ROWS = 40


def _pyplot() -> Any:
    """Import matplotlib only when charts are requested."""

    try:
        import matplotlib
    except ImportError as error:
        raise SystemExit(
            "charts need the optional analysis group: "
            "uv run --group analysis python -m scripts.diagnose ... "
            "(or pass --no-charts)"
        ) from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "text.color": TEXT_PRIMARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _reference_line(axis: Any, value: float, label: str) -> None:
    """Draw a dashed threshold line in neutral ink, named in the legend."""

    axis.axhline(value, color=TEXT_SECONDARY, linewidth=1, linestyle="--", label=label)
    axis.legend(loc="best")


def plot_training(
    history: Sequence[Mapping[str, float]],
    path: Path,
    *,
    title: str,
    kl_warning: float,
    gradient_cap: float,
) -> None:
    """Draw the training-history panels on a shared x-axis."""

    plt = _pyplot()
    steps = [row["environment_steps"] / 1e6 for row in history]
    marker = "o" if len(history) <= MARKER_ROWS else None
    figure, axes = plt.subplots(4, 3, figsize=(12, 11), sharex=True)
    for axis, (panel_title, columns) in zip(axes.flat, TRAINING_PANELS):
        for colour, column in zip(SERIES, columns):
            axis.plot(
                steps,
                [row[column] for row in history],
                color=colour,
                linewidth=LINE_WIDTH,
                marker=marker,
                markersize=3,
                label=column.replace("_", " "),
            )
        axis.set_title(panel_title, loc="left")
        if len(columns) > 1:
            axis.legend(loc="best")
        if columns == ("win_rate",) or columns == ("no_launch_rate",):
            axis.set_ylim(-0.02, 1.02)
        if columns == ("mean_ship_advantage",) or columns == ("explained_variance",):
            axis.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
        if columns == ("approximate_kl",):
            _reference_line(axis, kl_warning, f"warning {kl_warning:g}")
        if columns == ("mean_gradient_norm", "max_gradient_norm"):
            _reference_line(axis, gradient_cap, f"clipping cap {gradient_cap:g}")
    for axis in axes[-1]:
        axis.set_xlabel("Training turns (millions)")
    figure.suptitle(title, x=0.01, ha="left", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_value_error(buckets: Sequence[Mapping[str, float]], path: Path) -> None:
    """Draw critic error, sign agreement, and mean estimate across turn ranges."""

    plt = _pyplot()
    starts = [row["first_turn"] for row in buckets]
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    panels = (
        ("mean_absolute_error", "Mean absolute value error", (0, None)),
        ("sign_agreement", "Did the critic know who was winning? (share)", (0, 1.02)),
        ("mean_value", "Mean critic estimate", (-1.05, 1.05)),
    )
    for axis, (column, panel_title, limits) in zip(axes, panels):
        axis.plot(
            starts,
            [row[column] for row in buckets],
            color=SERIES[0],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=4,
        )
        axis.set_title(panel_title, loc="left")
        axis.set_xlabel("Turn (start of range)")
        axis.set_ylim(*limits)
    axes[2].axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_seat_results(by_seat: Mapping[int, Mapping[str, int]], path: Path) -> None:
    """Draw wins, draws, and losses for each seat, labelled with their counts."""

    plt = _pyplot()
    seats = sorted(by_seat)
    names = ("win", "draw", "loss")
    figure, axis = plt.subplots(figsize=(7, 3.8))
    width = 0.26
    for offset, (colour, name) in enumerate(zip(SERIES, names)):
        positions = [seat + (offset - 1) * width for seat in seats]
        heights = [by_seat[seat][name] for seat in seats]
        bars = axis.bar(positions, heights, width=width * 0.92, color=colour, label=f"{name}s")
        axis.bar_label(bars, padding=2, fontsize=8, color=TEXT_PRIMARY)
    axis.set_xticks(seats, [f"Seat {seat}" for seat in seats])
    axis.set_ylabel("Games")
    axis.legend(loc="upper right")
    axis.set_title("Results by learner seat", loc="left")
    # Game counts are whole numbers starting at zero. Fixing the range also
    # keeps an all-zero result from centring the axis on zero.
    highest = max((by_seat[seat][name] for seat in seats for name in names), default=0)
    axis.set_ylim(0, max(1, highest) * 1.25)
    axis.yaxis.get_major_locator().set_params(integer=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_game(game: GameDiagnosis, gamma: float, path: Path) -> None:
    """Draw one game: critic estimate against the real result, then ships."""

    plt = _pyplot()
    turns = [summary.turn for summary in game.summaries]
    figure, (value_axis, ship_axis) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    value_axis.plot(
        turns,
        [summary.value for summary in game.summaries],
        color=SERIES[0],
        linewidth=LINE_WIDTH,
        label="value estimate",
    )
    value_axis.plot(
        turns,
        [discounted_result(game.reward, game.turns, turn, gamma) for turn in turns],
        color=SERIES[1],
        linewidth=LINE_WIDTH,
        label="discounted real result",
    )
    value_axis.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    value_axis.set_ylim(-1.05, 1.05)
    value_axis.set_title("Critic estimate vs real result", loc="left")
    value_axis.legend(loc="best")
    # A role keeps its palette slot whether or not the other roles are
    # present, so "opposite" is the same colour in two- and four-player games.
    first = game.summaries[0]
    for row, role in enumerate(ROLES):
        if not first.present[row]:
            continue
        ship_axis.plot(
            turns,
            [decode_ships(summary.ships_scaled[row]) for summary in game.summaries],
            color=SERIES[row],
            linewidth=LINE_WIDTH,
            label=role.replace("_", "-"),
        )
    ship_axis.set_title("Total ships, including fleets", loc="left")
    ship_axis.set_xlabel("Turn")
    ship_axis.set_ylim(bottom=0)
    ship_axis.legend(loc="best")
    ending = "eliminated" if game.eliminated else outcome_name(game.reward)
    figure.suptitle(
        f"Seed {game.seed}, seat {game.seat}: {ending} after {game.turns} turns",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Stage 6: assemble the report.
# ---------------------------------------------------------------------------


def game_label(game: GameDiagnosis) -> str:
    """A file-name-safe identifier for one game."""

    return f"seed-{game.seed}-seat-{game.seat}"


def _percent(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.1%}"


def _number(value: float, digits: int) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def _training_markdown(
    identity: Mapping[str, Any],
    totals: Mapping[str, Any],
    milestones: Sequence[Mapping[str, float]],
    charts: bool,
) -> list[str]:
    """The training-history section of report.md."""

    clip = float(identity["clip_epsilon"])
    lines = ["## Training history", ""]
    if charts:
        lines += ["![Training history](charts/training.png)", ""]
    lines += [
        "**Where it comes from:** one row of the run's `metrics.csv` per PPO update. Those",
        "games used sampled, exploring actions, so they look worse than evaluation games.",
        "Game columns count only games that finished during that update; an update with",
        "none shows `n/a`, a gap in the chart.",
        "",
        f"- Updates: {totals['updates']}; training turns: {totals['environment_steps']:,.0f}",
        f"- Finished training games: {totals['games']:,.0f}: {totals['wins']:,.0f} wins, "
        f"{totals['losses']:,.0f} losses, {totals['draws']:,.0f} draws "
        f"({_percent(totals['win_rate'])} win rate)",
        f"- Updates with KL above the {identity['approximate_kl_warning']:g} warning: "
        + (", ".join(str(update) for update in totals["kl_warning_updates"]) or "none"),
        "",
        "Below, `r = π_new(a|s) / π_old(a|s)` is how much one update changed an action's",
        "probability, `A` the normalised GAE advantage, `V` the critic's estimate, and `G`",
        "the GAE return target. `reading-training-charts.md` explains each panel.",
        "",
        "| Panel | Formula | How to read it |",
        "| --- | --- | --- |",
        "| Win rate | wins ÷ games finished in the update | Is it learning to win? |",
        "| Game length | mean turns of games finished in the update | Read before launch counts |",
        "| Ship advantage | learner ships − strongest opponent's ships, at game end | How far ahead or behind it finishes |",
        "| Value loss | `mean((V − G)²)` | Falling means better return predictions |",
        "| Explained variance | `1 − Var(G − V) / Var(G)` | 1 is perfect; 0 is no better than the average |",
        "| Entropy | launch `-p·ln p − (1−p)·ln(1−p)` (max ≈ 0.693); target `-Σ q·ln q`, weighted by `p` | Falling means more decisive |",
        f"| Policy loss | `-mean(min(r·A, clip(r, {1 - clip:g}, {1 + clip:g})·A))` | Near 0 is normal; spikes mean large updates |",
        f"| Approximate KL | `mean((r − 1) − ln r)` | Update size; the dashed line is the {identity['approximate_kl_warning']:g} warning |",
        f"| Clip fraction | share with `\\|r − 1\\| > {clip:g}` | How often clipping kicked in |",
        f"| Gradient size | gradient norm before clipping | Above the dashed {identity['max_gradient_norm']:g} cap means that step was clipped |",
        "| No-launch rate | no-launch decisions ÷ chances to launch | How passive the exploring policy was |",
        "| Launches per turn | launches per game ÷ game length | Play style without the game-length effect |",
        "",
        "| Update | Turns (M) | Games | Win rate | Length | Ship adv. | Value loss | Expl. var. "
        "| Launch ent. | Target ent. | KL | Clip | No-launch | Launches/turn |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in milestones:
        lines.append(
            f"| {row['update']:.0f} | {row['environment_steps'] / 1e6:.2f} | {row['games']:.0f} | "
            f"{_percent(row['win_rate'])} | {_number(row['mean_episode_length'], 1)} | "
            f"{_number(row['mean_ship_advantage'], 0)} | {row['value_loss']:.4f} | "
            f"{row['explained_variance']:.3f} | {row['launch_entropy']:.3f} | "
            f"{row['target_entropy']:.3f} | {row['approximate_kl']:.4f} | "
            f"{row['clip_fraction']:.3f} | {row['no_launch_rate']:.1%} | "
            f"{_number(row['launches_per_turn'], 2)} |"
        )
    return lines + [""]


def _games_markdown(
    identity: Mapping[str, Any],
    games_report: Mapping[str, Any],
    charted: set[str],
    charts: bool,
) -> list[str]:
    """The game-diagnosis section of report.md."""

    by_seat = games_report["results_by_seat"]
    behaviour = games_report["behaviour"]
    # The untrained baseline has no checkpoint file, so it is named for what
    # it is rather than shown with an empty checkpoint path and hash.
    if games_report["checkpoint"] is None:
        model_lines = [
            f"- Model: {UNTRAINED_BASELINE}, never trained, initialized from learner "
            f"seed {games_report['initialization_seed']}",
        ]
    else:
        model_lines = [
            f"- Checkpoint: `{games_report['checkpoint']}`",
            f"- Checkpoint SHA-256: `{games_report['checkpoint_sha256']}`",
            f"- Trained for: {games_report['environment_steps']:,} turns, "
            f"{games_report['completed_updates']} updates",
        ]
    lines = [
        "## Game diagnosis",
        "",
        *model_lines,
        f"- Weights SHA-256: `{games_report['weights_sha256']}` (Torch {games_report['torch_version']})",
        f"- Seeds: [{games_report['seeds'][0]:,}, {games_report['seeds'][1]:,}), every seat",
        f"- Opponent: `{games_report['opponent']}`"
        + (
            ""
            if games_report["opponent"] == identity["training_opponent"]
            else f" (not the training opponent, `{identity['training_opponent']}`)"
        ),
        f"- Evaluator: `{games_report['evaluation']['policy']}` on {games_report['evaluation']['device']}",
        f"- Engine: `{games_report['engine']}`",
        f"- Rejected actions: {games_report['rejected_actions']}",
        "",
        "### Results by seat",
        "",
    ]
    if charts:
        lines += ["![Results by seat](charts/seats.png)", ""]
    lines += [
        "Each game counts once, by the seat the learner played. A clear imbalance between",
        "seats suggests a starting-position effect; with few games, check whether it could",
        "be chance first. Eliminated games are a subset of losses.",
        "",
        "| Seat | Wins | Draws | Losses | Eliminated |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for seat, counts in by_seat.items():
        lines.append(
            f"| {seat} | {counts['win']} | {counts['draw']} | {counts['loss']} | {counts['eliminated']} |"
        )
    measured = games_report["measurements"]
    share = measured["mean_share_without_productive_body"]
    lines += [
        "",
        "### Expansion and productive time",
        "",
        "These separate stronger play from surviving longer: a learner whose fleets",
        "drift on after its planets are gone has a long game but no territory.",
        "Every game's values, by map seed and seat, are in [games.csv](games.csv);",
        "an empty cell there means \"not reached\".",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
        f"| Games (wins / draws / losses) | {measured['games']} "
        f"({measured['wins']} / {measured['draws']} / {measured['losses']}) |",
        f"| Mean most planets held at once | {measured['mean_most_planets']:.2f} |",
        f"| Games that held two planets at once | {measured['reached_two_planets']} "
        f"of {measured['games']} |",
        "| Mean share of action turns with no planet or comet | "
        + ("n/a" if share is None else f"{share:.1%}")
        + f" (over {measured['games_with_action_turns']} games) |",
        "",
        "### Behaviour (deterministic evaluation decoder)",
        "",
        f"- Games: {behaviour['games']:.0f}",
        f"- Mean learner game length: {behaviour['mean_game_turns']:.1f} turns",
        f"- Launches per game: {behaviour['launches_per_game']:.1f}; per turn: "
        f"{behaviour['launches_per_turn']:.2f}",
        f"- No-launch rate: {behaviour['no_launch_rate']:.1%} of chances to launch",
        f"- Learner eliminated: {behaviour['eliminations']:.0f} games",
        "",
        "### Value error by turn",
        "",
    ]
    if charts:
        lines += ["![Value error by turn](charts/value-error.png)", ""]
    lines += [
        "For a decision at turn `t` in a game of `T` turns that ended with result `z`",
        "(+1 sole winner, 0 tie, −1 loss):",
        "",
        "```text",
        f"target_t = z × {identity['gamma']}^(T − 1 − t)",
        "error_t  = |value_t − target_t|",
        "```",
        "",
        "Later turn ranges hold only the games still running, so their decision counts",
        "matter: a small count is a small, unusual set of games.",
        "",
        "| Turns | Decisions | Mean absolute error | Knew who was winning | Mean estimate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in games_report["value_error_by_turn"]:
        lines.append(
            f"| {row['first_turn']:.0f}–{row['last_turn']:.0f} | {row['decisions']:.0f} | "
            f"{row['mean_absolute_error']:.3f} | {_percent(row['sign_agreement'])} | "
            f"{row['mean_value']:+.3f} |"
        )
    lines += ["", "### Individual games", ""]
    if charts and charted:
        lines += [
            "**How to read a game chart:** the top panel shows the value estimate against",
            "the discounted real result, turn by turn. Turns where the estimate stays high",
            "while the result is near −1 are turns where the critic was confidently wrong.",
            "The bottom panel shows each player's total ships, garrisons plus fleets,",
            f"decoded from the observation as `exp(x × ln(1 + {SHIP_SCALE})) − 1`. Where the",
            "learner's line falls behind is usually where the game turned.",
            "",
        ]
    traced = games_report["traced_games"]
    if not traced:
        lines.append("No games were traced.")
    for entry in traced:
        label = f"seed-{entry['seed']}-seat-{entry['seat']}"
        line = (
            f"- **{entry['result']}**, seed {entry['seed']}, seat {entry['seat']}, "
            f"{entry['turns']} turns: [trace](traces/{label}.ndjson)"
        )
        if label in charted:
            line += f" · [chart](charts/games/{label}.png)"
        lines.append(line)
    return lines + [""]


def write_report_markdown(
    path: Path,
    *,
    identity: Mapping[str, Any],
    totals: Mapping[str, Any],
    milestones: Sequence[Mapping[str, float]],
    games_report: Mapping[str, Any] | None,
    charted: set[str],
    charts: bool,
) -> None:
    """Write the human-readable report, embedding charts beside their tables."""

    lines = [
        f"# {identity['players']}-player rl-v2 run diagnosis",
        "",
        f"- Run: `{identity['run']}`",
        f"- Contract: `{identity['contract']}`, {identity['players']} players, "
        f"generator `{identity['generator_version']}`",
        f"- Training opponent: `{identity['training_opponent']}`",
        f"- Learner seed: {identity['learner_seed']}",
        "",
    ]
    lines += _training_markdown(identity, totals, milestones, charts)
    if games_report is None:
        lines += ["## Game diagnosis", "", "Skipped (`--no-games`).", ""]
    else:
        lines += _games_markdown(identity, games_report, charted, charts)
    path.write_text("\n".join(lines))


def _json_safe(value: Any) -> Any:
    """Replace NaN floats with None, recursively, so the JSON stays valid."""

    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _game_from_outcome(outcome: EvaluationOutcome, recorder: TurnRecorder) -> GameDiagnosis:
    """Pair one terminal outcome with every board recorded for that game.

    A game is complete only with one decision per turn and its final board at
    the outcome's turn count; anything less would silently drop states, so it
    is refused instead.
    """

    summaries = recorder.games.get((outcome.seed, outcome.seat), [])
    if len(summaries) != outcome.turns:
        raise ValueError(
            f"seed {outcome.seed} seat {outcome.seat}: recorded {len(summaries)} decisions "
            f"for a {outcome.turns}-turn game"
        )
    final = recorder.finals.get((outcome.seed, outcome.seat))
    if final is None or final.turn != outcome.turns:
        raise ValueError(
            f"seed {outcome.seed} seat {outcome.seat}: no final board at turn {outcome.turns}"
        )
    return GameDiagnosis(
        outcome.seed,
        outcome.seat,
        outcome.reward,
        outcome.turns,
        outcome.end_reason == END_LEARNER_ELIMINATED,
        tuple(summaries),
        final,
    )


def latest_checkpoint(run: Path) -> Path:
    """The run manifest's checkpoint with the most completed updates."""

    manifest = read_json_object(run / "manifest.json", "run manifest")
    entries = manifest.get("checkpoints") or []
    if not entries:
        raise ValueError(f"{run / 'manifest.json'} lists no checkpoints")
    newest = max(entries, key=lambda entry: entry["completed_updates"])
    return run / newest["file"]


# ---------------------------------------------------------------------------
# Stage 7: run both parts and write every output file.
# ---------------------------------------------------------------------------


def diagnose_games(
    *,
    checkpoint: Path | None,
    seeds: int,
    device: str | None,
    out: Path,
    charts: bool,
    game_charts: int,
    traced_wins: int,
    opponent: str | None = None,
    run: Path | None = None,
    command: Sequence[str] = DEFAULT_ENV_COMMAND,
    runner: Callable[..., Any] = run_rotated_matches,
    loader_options: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], set[str]]:
    """Play a model's validation games and write their traces and charts.

    The model is ``checkpoint``, or, when that is ``None``, the reconstructed
    untrained baseline of ``run``. ``opponent`` is a complete opponent spec;
    ``None`` plays the run's training opponent. Returns the game section of
    the report and the labels of charted games.
    """

    # Choose the model before any game starts. A saved checkpoint is bound to
    # its run's manifest and configuration. The untrained baseline is rebuilt
    # from the run's configuration and learner seed exactly as training builds
    # its first policy; it has no file, optimizer, or training progress, so its
    # report says so rather than borrowing a checkpoint's identity.
    if checkpoint is None:
        if run is None:
            raise ValueError("the untrained baseline is rebuilt from a run directory")
        config = run_config_from_mapping(
            read_json_object(run / "run-config.json", "run configuration")
        )
        policy = initial_policy(config.learner_seed).to(dtype=torch.float32).eval()
        model: dict[str, Any] = {
            "model": UNTRAINED_BASELINE,
            "checkpoint": None,
            "checkpoint_sha256": None,
            "completed_updates": 0,
            "environment_steps": 0,
            "initialization_seed": config.learner_seed,
        }
    else:
        loaded = load_run_checkpoint(checkpoint, **(loader_options or {}))
        config, policy = loaded.state.config, loaded.policy
        model = {
            "model": "checkpoint",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": loaded.fingerprint,
            "completed_updates": loaded.state.completed_updates,
            "environment_steps": loaded.state.environment_steps,
        }
    # Fingerprint the weights themselves, on the CPU before they move to the
    # playing device. Equal fingerprints across machines prove two reports
    # played the same model; the Torch version is recorded because another
    # version may initialize an untrained model differently.
    model["weights_sha256"] = state_dict_sha256(policy.state_dict())
    model["torch_version"] = torch.__version__

    # Choose the boards: a prefix of the reusable validation seeds, never a
    # protected pool.
    seed_range = validation_seed_prefix(config, seeds)
    target = resolve_evaluation_device(device, config.device)
    policy = policy.to(device=target)
    # The override is local to these games: the run's saved configuration is
    # only read, never rewritten. The opponent goes into the expected Go
    # metadata, so Go must report back exactly this spec or the result is
    # refused; a mistyped spec cannot quietly fall back to another opponent.
    opponent = opponent or config.training_opponent

    # Play every seed from every seat, recording each decision through the hook.
    recorder = TurnRecorder()
    expected = evaluation_metadata(config, seed_range, opponent)
    total = config.players * (seed_range.end - seed_range.start)
    print(f"playing {total} games against {opponent} on {target} ...", flush=True)
    result = checked_evaluation_result(
        runner(
            policy,
            command=command,
            seeds=seed_range,
            opponent=opponent,
            batch_size=min(config.environment_count, total),
            evaluation_policy=config.evaluation_policy,
            device=target,
            expected_metadata=expected,
            players=config.players,
            on_turn=recorder,
            on_game_end=recorder.game_ended,
        ),
        seed_range,
        expected,
        players=config.players,
        evaluation_policy=config.evaluation_policy,
        context="V2 diagnosis",
    )

    # Join each outcome with its recorded turns, in seed and seat order.
    games = [
        _game_from_outcome(outcome, recorder)
        for outcome in sorted(result.outcomes, key=lambda o: (o.seed, o.seat))
    ]
    non_wins = [game for game in games if game.reward <= 0]
    wins = [game for game in games if game.reward > 0][:traced_wins]

    # Traces: every lost or drawn game, plus the first few wins to compare with.
    traces = out / "traces"
    traces.mkdir(parents=True)
    for game in non_wins + wins:
        (traces / f"{game_label(game)}.ndjson").write_text(
            "\n".join(trace_lines(game, config.gamma)) + "\n"
        )

    # Measurements for every game, not only the traced ones, keyed by map seed
    # and seat so two reports can later be compared game by game and map by
    # map. games.csv holds the same rows for spreadsheets; an empty cell means
    # "not reached" or "undefined", never zero.
    game_rows = [game_measurements(game) for game in games]
    with (out / "games.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(game_rows[0]))
        writer.writeheader()
        writer.writerows(game_rows)

    by_seat = results_by_seat(games, config.players)
    value_errors = value_error_by_turn(games, config.gamma)
    # Charts: the first game_charts non-wins and every traced win. A weak
    # policy can lose hundreds of games, and a chart for each would bury the
    # report; their traces are all kept.
    charted_games = non_wins[:game_charts] + wins
    charted = {game_label(game) for game in charted_games} if charts else set()
    if charts:
        chart_directory = out / "charts"
        (chart_directory / "games").mkdir(parents=True, exist_ok=True)
        plot_value_error(value_errors, chart_directory / "value-error.png")
        plot_seat_results(by_seat, chart_directory / "seats.png")
        for game in charted_games:
            plot_game(game, config.gamma, chart_directory / "games" / f"{game_label(game)}.png")

    report = {
        **model,
        "seeds": [seed_range.start, seed_range.end],
        "opponent": opponent,
        "engine": result.metadata.get("engine"),
        "evaluation": result.metadata["evaluation"],
        "rejected_actions": result.rejected_actions,
        "results_by_seat": {str(seat): counts for seat, counts in by_seat.items()},
        "measurements": measurement_summary(game_rows),
        "game_results": game_rows,
        "behaviour": behaviour_summary(games),
        "value_error_by_turn": value_errors,
        "traced_games": [
            {
                "seed": game.seed,
                "seat": game.seat,
                "result": "eliminated" if game.eliminated else outcome_name(game.reward),
                "turns": game.turns,
            }
            for game in non_wins + wins
        ],
    }
    return report, charted


def diagnose(
    *,
    run: Path,
    out: Path,
    checkpoint: Path | None,
    games: bool,
    seeds: int,
    device: str | None,
    charts: bool,
    game_charts: int,
    traced_wins: int,
    untrained: bool = False,
    **game_options: Any,
) -> dict[str, Any]:
    """Run the whole diagnosis and write every output file. Returns the report.

    ``untrained`` plays the run's reconstructed untrained baseline instead of a
    checkpoint; ``checkpoint`` must then be ``None``.
    """

    if untrained and checkpoint is not None:
        raise ValueError("choose a checkpoint or the untrained baseline, not both")

    # The run's own record of its contract, read as plain JSON: no Torch
    # deserialization is needed for the training history.
    config = read_json_object(run / "run-config.json", "run configuration")
    if config.get("contract") != "rl-v2":
        raise ValueError(f"{run} is a {config.get('contract')!r} run; diagnose reads rl-v2 runs")
    history = training_history(read_metrics(run / "metrics.csv"))
    totals = training_totals(history, float(config["approximate_kl_warning"]))
    milestones = training_milestones(history)
    identity = {
        "run": str(run),
        **{
            key: config[key]
            for key in (
                "contract", "players", "generator_version", "training_opponent", "learner_seed",
                "gamma", "clip_epsilon", "approximate_kl_warning", "max_gradient_norm",
            )
        },
    }
    if charts:
        (out / "charts").mkdir(parents=True, exist_ok=True)
        plot_training(
            history,
            out / "charts" / "training.png",
            title=f"{config['players']}-player rl-v2 training history",
            kl_warning=float(config["approximate_kl_warning"]),
            gradient_cap=float(config["max_gradient_norm"]),
        )

    games_report = None
    charted: set[str] = set()
    if games:
        games_report, charted = diagnose_games(
            checkpoint=None if untrained else checkpoint or latest_checkpoint(run),
            run=run,
            seeds=seeds,
            device=device,
            out=out,
            charts=charts,
            game_charts=game_charts,
            traced_wins=traced_wins,
            **game_options,
        )

    report = {
        # Saved reports record this name and compare_diagnoses checks it, so it
        # keeps its version even though the script itself is no longer "v2".
        "format": "diagnosis-v2",
        "identity": identity,
        "training": {"totals": totals, "milestones": milestones},
        "games": games_report,
    }
    # NaN is not valid JSON: an update without finished games, or a turn range
    # with only draws, has no measured value.
    (out / "report.json").write_text(json.dumps(_json_safe(report), indent=2) + "\n")
    write_report_markdown(
        out / "report.md",
        identity=identity,
        totals=totals,
        milestones=milestones,
        games_report=games_report,
        charted=charted,
        charts=charts,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Read options, run the diagnosis, and print where the report went."""

    # Fix cuBLAS's workspace before any GPU work, including model loading.
    configure_cuda_workspace()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="rl-v2 training run directory")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="new or empty output directory; earlier reports are never overwritten",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="checkpoint to play (default: the run manifest's latest checkpoint)",
    )
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="play a never-trained model rebuilt from the run's learner seed, as a baseline",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"validation boards to play, each from every seat (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="where the model decides moves (default: the run config's device)",
    )
    parser.add_argument(
        "--opponent",
        help="complete opponent spec to play against (default: the run's training opponent)",
    )
    parser.add_argument("--no-games", action="store_true", help="only chart the training history")
    parser.add_argument("--no-charts", action="store_true", help="skip charts and matplotlib")
    parser.add_argument(
        "--game-charts", type=int, default=12, help="lost or drawn games to chart (traces keep all)"
    )
    parser.add_argument("--traced-wins", type=int, default=5, help="wins to trace and chart for comparison")
    arguments = parser.parse_args(argv)
    if arguments.game_charts < 0 or arguments.traced_wins < 0:
        parser.error("--game-charts and --traced-wins must not be negative")
    if arguments.no_games and arguments.opponent is not None:
        parser.error("--opponent chooses who the games are played against; it has no effect with --no-games")
    if arguments.untrained and arguments.checkpoint is not None:
        parser.error("--untrained replaces --checkpoint; choose one model to play")
    if arguments.untrained and arguments.no_games:
        parser.error("--untrained chooses the model that plays; it has no effect with --no-games")

    # Each run gets its own folder, so an earlier report is never mixed with or
    # replaced by a new one.
    out = arguments.out
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} is not empty; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)

    report = diagnose(
        run=arguments.run_dir.resolve(),
        out=out,
        checkpoint=arguments.checkpoint,
        games=not arguments.no_games,
        seeds=arguments.seeds,
        device=arguments.device,
        opponent=arguments.opponent,
        untrained=arguments.untrained,
        charts=not arguments.no_charts,
        game_charts=arguments.game_charts,
        traced_wins=arguments.traced_wins,
    )
    totals = report["training"]["totals"]
    summary = f"training: {totals['wins']:.0f}W of {totals['games']:.0f} finished games"
    if report["games"] is not None:
        seats = report["games"]["results_by_seat"]
        summary += "; validation: " + ", ".join(
            f"seat {seat}: {counts['win']}W {counts['draw']}D {counts['loss']}L"
            for seat, counts in seats.items()
        )
    print(f"{summary}; report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
