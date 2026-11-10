"""Measure any saved training checkpoint on reusable validation boards.

This is an inspection report. It does not freeze a checkpoint or claim a
protected evaluation pool; the training release gate remains authoritative.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from orrery.checkpoint import load_checkpoint, recorded_checkpoint_fingerprint
from orrery.config import SeedRange
from orrery.evaluate import DEFAULT_ENV_COMMAND
from orrery.env_client import END_LEARNER_ELIMINATED
from orrery.evaluation_context import configure_cuda_workspace, resolve_evaluation_device
from orrery.evaluator import (
    checked_evaluation_result,
    evaluation_metadata,
    interval_evidence,
    release_readiness_gate,
    run_rotated_matches,
    timings_evidence,
    validation_summary,
)
from orrery.jsonio import json_value, read_json_object
from orrery.model import OrbitPolicy


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A checkpoint proven to belong to its run, with its weights on the CPU."""

    # Training run directory: the checkpoint's grandparent.
    run: Path
    # SHA-256 recorded beside the checkpoint and matched in the run manifest.
    fingerprint: str
    # Everything load_checkpoint restored, including the run configuration.
    state: Any
    # A float32 policy on the CPU holding the checkpoint's weights, in eval mode.
    policy: torch.nn.Module


def load_run_checkpoint(
    checkpoint: Path,
    *,
    loader: Callable = load_checkpoint,
    policy_factory: Callable = OrbitPolicy,
) -> LoadedCheckpoint:
    """Bind one checkpoint to its run's manifest and configuration, then load it.

    Every check here runs before a game starts, so a report can never describe
    a checkpoint other than the one its run recorded.
    """

    checkpoint = checkpoint.resolve()
    run = checkpoint.parent.parent
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("checkpoint must be inside a run's checkpoints/ directory")
    manifest = read_json_object(run / "manifest.json", "run manifest")
    relative = checkpoint.relative_to(run).as_posix()
    entry = next((item for item in manifest.get("checkpoints", []) if item.get("file") == relative), None)
    if entry is None:
        raise ValueError("checkpoint is not recorded in its run manifest")
    fingerprint = recorded_checkpoint_fingerprint(checkpoint)
    if entry.get("sha256") != fingerprint:
        raise ValueError("run manifest and checkpoint fingerprint disagree")

    # load_checkpoint checks the independent JSON envelope, run contract, byte
    # count, hash, and payload before Torch deserializes the trusted local
    # checkpoint. Building the policy first lets its tensor signature be
    # checked at that same point, so another architecture is refused before
    # unpickling.
    policy = policy_factory().to(device="cpu", dtype=torch.float32).eval()
    state = loader(checkpoint, expected_model_state=policy.state_dict())
    config = state.config
    if read_json_object(run / "run-config.json", "run configuration") != json_value(config.to_dict()):
        raise ValueError("run configuration disagrees with checkpoint")
    if entry.get("completed_updates") != state.completed_updates or entry.get("environment_steps") != state.environment_steps:
        raise ValueError("run manifest disagrees with checkpoint training boundary")
    policy.load_state_dict(state.model_state, strict=True)
    return LoadedCheckpoint(run, fingerprint, state, policy)


def validation_seed_prefix(config: Any, seeds: int | None) -> SeedRange:
    """The first ``seeds`` release validation seeds, or all of them for None."""

    seed_count = config.release_validation_seeds if seeds is None else seeds
    if (
        isinstance(seed_count, bool)
        or not isinstance(seed_count, int)
        or not 1 <= seed_count <= config.release_validation_seeds
    ):
        raise ValueError(
            f"seeds must be between 1 and {config.release_validation_seeds}, got {seed_count!r}"
        )
    seed_range = SeedRange(config.validation_seeds.start, config.validation_seeds.start + seed_count)
    if seed_range.end > config.validation_seeds.end:
        raise ValueError("release validation sample exceeds reusable validation seeds")
    return seed_range


def validate_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    runner: Callable = run_rotated_matches,
    loader: Callable = load_checkpoint,
    policy_factory: Callable = OrbitPolicy,
    command: Sequence[str] = DEFAULT_ENV_COMMAND,
    device: str | None = None,
    seeds: int | None = None,
) -> dict:
    """Load verified weights, play every seat, and publish a new report.

    The run manifest and configuration bind the selected checkpoint to its
    training run before a game starts. Validation uses that checkpoint's
    opponent, seed range, policy decoder, and protocol, so reports from
    different contracts cannot silently mix.

    ``device`` overrides the run config's device (``"cpu"`` or ``"cuda"``);
    CUDA on a machine without it is an error, never a silent CPU fallback.
    ``seeds`` plays only the first N validation seeds, from every seat, for a
    quick check such as comparing two runs. A partial sample still reports
    the release thresholds as not met, because its game count differs.
    """

    checkpoint = checkpoint.resolve()
    if output.exists():
        raise ValueError(f"report output already exists: {output}")
    loaded = load_run_checkpoint(checkpoint, loader=loader, policy_factory=policy_factory)
    fingerprint, state, config = loaded.fingerprint, loaded.state, loaded.state.config
    seed_range = validation_seed_prefix(config, seeds)
    # Check the device before any game starts, then load the verified weights
    # on the CPU and move them there.
    target = resolve_evaluation_device(device, config.device)
    policy = loaded.policy.to(device=target)
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("checkpoint must be inside a run's checkpoints/ directory")
    expected = evaluation_metadata(config, seed_range, config.training_opponent)
    raw = runner(
        policy,
        command=command,
        seeds=seed_range,
        opponent=config.training_opponent,
        batch_size=config.environment_count,
        evaluation_policy=config.evaluation_policy,
        device=target,
        expected_metadata=expected,
        players=config.players,
    )
    result = checked_evaluation_result(
        raw,
        seed_range,
        expected,
        players=config.players,
        evaluation_policy=config.evaluation_policy,
        context="checkpoint validation report",
    )
    evaluation = result.metadata["evaluation"]
    decision = release_readiness_gate(
        result,
        expected_games=config.release_games,
        required_wins=config.release_required_wins,
        minimum_lower_bound=config.release_minimum_lower_bound,
        confidence_level=config.release_confidence_level,
    )
    # "eliminated" is a subset of "losses": games where the learner was wiped
    # out before the game itself ended, as opposed to losing on ships.
    by_seat = {
        str(seat): {
            "wins": sum(item.reward > 0 for item in result.outcomes if item.seat == seat),
            "losses": sum(item.reward < 0 for item in result.outcomes if item.seat == seat),
            "draws": sum(item.reward == 0 for item in result.outcomes if item.seat == seat),
            "eliminated": sum(
                item.end_reason == END_LEARNER_ELIMINATED
                for item in result.outcomes
                if item.seat == seat
            ),
        }
        for seat in range(config.players)
    }
    summary = validation_summary(result.outcomes, config.players)
    report = {
        "format": "checkpoint-validation-report-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": fingerprint,
        "completed_updates": state.completed_updates,
        "environment_steps": state.environment_steps,
        "seed_range": [seed_range.start, seed_range.end],
        "opponent": config.training_opponent,
        "metadata": result.metadata,
        "wins": decision.wins,
        "losses": decision.losses,
        "draws": decision.draws,
        "games": len(result.outcomes),
        "win_rate": result.wins / len(result.outcomes),
        "rejected_actions": result.rejected_actions,
        "release_thresholds_met": decision.passed,
        "failed_checks": list(decision.failed_checks),
        "confidence_interval": interval_evidence(decision.interval),
        "results_by_seat": by_seat,
        "summary": summary,
        "outcomes": [asdict(item) for item in result.outcomes],
    }
    # Timings describe this run, not the checkpoint, so a repeated report may
    # differ here while every outcome above stays the same.
    if result.timings is not None:
        report["timings"] = timings_evidence(result.timings)
    # Finish every measurement before creating a new directory. An existing
    # report is never overwritten, including when validation is repeated.
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(json_value(report), indent=2) + "\n")
    lines = [
        "# Checkpoint validation report", "",
        f"- Checkpoint: `{checkpoint}`", f"- SHA-256: `{fingerprint}`",
        f"- Training steps: {state.environment_steps:,}; updates: {state.completed_updates}",
        f"- Reusable seeds: [{seed_range.start:,}, {seed_range.end:,}), every seat",
        # The device and evaluator that decided the moves. Reports from
        # different devices are separate evidence, not exact replays.
        f"- Evaluator: `{evaluation['policy']}` on {evaluation['device']}"
        + (f" ({evaluation['device_name']})" if "device_name" in evaluation else ""),
        f"- Opponent: `{config.training_opponent}`", "",
        f"**Results:** {decision.wins} wins, {decision.losses} losses, {decision.draws} draws in {len(result.outcomes)} games ({report['win_rate']:.1%} win rate).", "",
        "| Seat | Wins | Losses | Draws | Eliminated |", "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for seat, counts in by_seat.items():
        lines.append(
            f"| {seat} | {counts['wins']} | {counts['losses']} | {counts['draws']} | {counts['eliminated']} |"
        )
    launches = summary["launches_per_game"]
    turns = summary["episode_turns"]
    lines += [
        "", "**How the games looked:**", "",
        "| Measure | Value |", "| --- | ---: |",
        f"| Launches per game (mean / median / max) | {launches['mean']:.1f} / {launches['median']:g} / {launches['max']} |",
        f"| Games with no launches | {summary['zero_launch_games']} |",
        # An eliminated learner's episode ends early, so this is the learner's
        # episode length rather than the length of the whole match.
        f"| Learner episode turns (mean / min / p10 / median / p90 / max) | {turns['mean']:.1f} / {turns['min']} / {turns['p10']} / {turns['median']:g} / {turns['p90']} / {turns['max']} |",
        f"| Learner eliminations | {summary['eliminations']} |",
        # Win rate from each seat, as the opponent check reports it. A seat
        # with no games shows a dash rather than 0%.
        "| Win rate by seat | "
        + " / ".join(
            "–" if rate is None else f"{seat}: {rate:.1%}"
            for seat, rate in enumerate(summary["seat_win_rate"])
        )
        + " |",
    ]
    lines += [
        "", f"- Rejected actions: {result.rejected_actions}",
        # Nominal: games on the same board from different seats are not
        # independent, so this is not a coverage guarantee across maps.
        f"- Nominal 95% confidence interval: {decision.interval.lower_bound:.1%}–{decision.interval.upper_bound:.1%}",
        f"- Release thresholds met in this sample: {'yes' if decision.passed else 'no'}", "",
        "This is a reusable validation measurement. It does not freeze this checkpoint",
        "or authorize a protected held-out evaluation. The training release gate",
        "and archived checkpoint remain authoritative for that lifecycle.", "",
        "Individual seed and seat outcomes are in `report.json`.", "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    # Fix cuBLAS's workspace before any GPU work, including model loading.
    configure_cuda_workspace()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="where the model decides moves (default: the run config's device)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        help="play only the first N release validation seeds, from every seat",
    )
    arguments = parser.parse_args(argv)
    report = validate_checkpoint(
        arguments.checkpoint, arguments.out, device=arguments.device, seeds=arguments.seeds
    )
    print(f"validation: {report['wins']}W {report['losses']}L {report['draws']}D; report: {arguments.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
