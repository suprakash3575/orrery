"""One terminal entry point for training and inspecting RL run artifacts.

This menu reads existing evidence without loading checkpoint tensors. Long-running
training and evaluation still use their established commands and safety checks.
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping, Sequence

from orrery.config import load_run_config
from orrery.jsonio import json_value
from orrery.paths import REPOSITORY_ROOT


Input = Callable[[str], str]
Output = Callable[[str], None]
Launcher = Callable[[Sequence[str], Path, Mapping[str, str] | None], int]


def _launch(command: Sequence[str], cwd: Path, environment: Mapping[str, str] | None) -> int:
    """Run a selected workflow in this terminal and return to the menu afterward."""

    return subprocess.run(command, cwd=cwd, env=environment, check=False).returncode


def _json(path: Path) -> dict:
    with path.open() as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _paths(root: Path, filename: str) -> list[Path]:
    """Discover artifacts in ordinary runs and copied backups alike."""

    directory = root / "runs"
    return sorted(directory.rglob(filename)) if directory.is_dir() else []


def _choose(paths: Sequence[Path], root: Path, read: Input, show: Output) -> Path | None:
    if not paths:
        show("Nothing found under runs/.")
        return None
    for index, path in enumerate(paths, 1):
        show(f"{index}. {path.relative_to(root)}")
    while True:
        choice = read("Number (Enter to go back): ").strip()
        if not choice:
            return None
        if choice.isdecimal() and 1 <= int(choice) <= len(paths):
            return paths[int(choice) - 1]
        show("Choose one of the listed numbers.")


def _outcomes(evidence: Mapping) -> tuple[int, int, int, int] | None:
    """Use recorded aggregates where available; otherwise count raw outcomes."""

    aggregate = evidence.get("aggregate", evidence)
    if all(name in aggregate for name in ("wins", "losses", "draws")):
        wins, losses, draws = (int(aggregate[name]) for name in ("wins", "losses", "draws"))
        return wins, losses, draws, wins + losses + draws
    outcomes = evidence.get("outcomes")
    if not isinstance(outcomes, list):
        return None
    wins = sum(item["reward"] > 0 for item in outcomes)
    losses = sum(item["reward"] < 0 for item in outcomes)
    return wins, losses, len(outcomes) - wins - losses, len(outcomes)


def _show_evidence(path: Path, show: Output) -> None:
    try:
        evidence = _json(path)
        counts = _outcomes(evidence)
    except (OSError, ValueError, KeyError, TypeError) as error:
        show(f"Cannot read {path}: {error}")
        return
    show(str(path))
    show(f"Gate: {evidence.get('gate', evidence.get('format', 'unknown'))}")
    if counts:
        wins, losses, draws, games = counts
        show(f"Games: {games}; wins: {wins}; losses: {losses}; draws: {draws}; win rate: {wins / games:.1%}" if games else "No games recorded.")
    if "passed" in evidence:
        show(f"Passed: {evidence['passed']}")
    if "release_thresholds_met" in evidence:
        show(f"Release thresholds met in this diagnostic sample: {evidence['release_thresholds_met']}")
    rejected = evidence.get("rejected_actions", evidence.get("aggregate", {}).get("rejected_actions"))
    if rejected is not None:
        show(f"Rejected actions: {rejected}")


def _show_run(run: Path, show: Output) -> None:
    """Summarize completed training games and the latest update from metrics.csv."""

    metrics = run / "metrics.csv"
    try:
        with metrics.open(newline="") as source:
            rows = list(csv.DictReader(source))
        if not rows:
            raise ValueError("no updates recorded")
        totals = {name: sum(int(row[name]) for row in rows) for name in ("wins", "losses", "draws")}
        games = sum(totals.values())
        latest = rows[-1]
        show(f"Run: {run}")
        show(f"Updates: {latest['update']}; environment steps: {latest['environment_steps']}")
        show(f"Training games: {games}; {totals['wins']}W {totals['losses']}L {totals['draws']}D; win rate: {totals['wins'] / games:.1%}" if games else "No completed training games yet.")
        show(f"Latest update win rate: {float(latest['win_rate']):.1%}")
    except (OSError, ValueError, KeyError) as error:
        show(f"Cannot summarize {metrics}: {error}")
    evidence = sorted(run.glob("pipeline-*.json")) + sorted(run.glob("release-*.json"))
    if evidence:
        show("Validation records:")
        for path in evidence:
            show(f"  {path.name}")


def _matching_config(root: Path, run: Path) -> Path | None:
    """Resume only with a TOML whose full semantic contract matches the run."""

    recorded = _json(run / "run-config.json")
    for path in sorted((root / "configs").glob("*.toml")):
        try:
            if json_value(load_run_config(path).to_dict()) == recorded:
                return path
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def _run_command(
    module: str,
    arguments: Sequence[str],
    root: Path,
    show: Output,
    launch: Launcher,
    *,
    cuda: bool = False,
) -> None:
    environment = dict(os.environ)
    if cuda:
        environment.setdefault("GOMAXPROCS", "32")
    code = launch([sys.executable, "-m", module, *arguments], root / "python", environment)
    show(f"Command finished with exit code {code}.")


def _start_training(root: Path, read: Input, show: Output, launch: Launcher) -> None:
    config = _choose(sorted((root / "configs").glob("*.toml")), root, read, show)
    if config is None:
        return
    limit = read("Cumulative max updates (Enter for configured training limit): ").strip()
    if limit and (not limit.isdecimal() or int(limit) < 1):
        show("Max updates must be a positive integer.")
        return
    # Read the device from the config itself: the file name says nothing
    # reliable about it, and an unreadable config fails here rather than after
    # a training process has started.
    try:
        cuda = load_run_config(config).device == "cuda"
    except (OSError, ValueError, KeyError, TypeError) as error:
        show(f"Cannot read {config.name}: {error}")
        return
    args = ["--config", str(config)]
    if limit:
        args += ["--max-updates", limit]
    show(f"Starting training with {config.name}. Progress and artifact path appear here.")
    _run_command("orrery.train", args, root, show, launch, cuda=cuda)


def _resume_training(root: Path, read: Input, show: Output, launch: Launcher) -> None:
    run = _choose([path.parent for path in _paths(root, "manifest.json") if (path.parent / "run-config.json").is_file()], root, read, show)
    if run is None:
        return
    try:
        manifest = _json(run / "manifest.json")
        entries = manifest["checkpoints"]
        checkpoints = [run / entry["file"] for entry in entries if (run / entry["file"]).is_file()]
        config = _matching_config(root, run)
    except (OSError, ValueError, KeyError, TypeError) as error:
        show(f"Cannot inspect run: {error}")
        return
    if not checkpoints:
        show("No checkpoint files are present in this run.")
        return
    checkpoint = _choose(checkpoints, root, read, show)
    if checkpoint is None:
        return
    if config is None:
        show("No config TOML matches this run's complete contract; resume through orrery.train with its original config.")
        return
    reason = read("Review reason if this run crossed its review boundary (Enter if none): ").strip()
    args = ["--config", str(config), "--resume", str(checkpoint)]
    if reason:
        args += ["--review-reason", reason]
    show(f"Resuming {checkpoint.name} with {config.name}.")
    # _matching_config already loaded this file successfully.
    cuda = load_run_config(config).device == "cuda"
    _run_command("orrery.train", args, root, show, launch, cuda=cuda)


def _validate_checkpoint(root: Path, read: Input, show: Output, launch: Launcher) -> None:
    runs = [path.parent for path in _paths(root, "manifest.json") if (path.parent / "run-config.json").is_file()]
    run = _choose(runs, root, read, show)
    if run is None:
        return
    manifest = _json(run / "manifest.json")
    checkpoints = [
        run / entry["file"] for entry in manifest.get("checkpoints", [])
        if (run / entry["file"]).is_file()
    ]
    checkpoint = _choose(checkpoints, root, read, show)
    if checkpoint is None:
        return
    output = root / "runs" / f"validation-{checkpoint.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    show("Playing reusable validation games from every seat for this checkpoint.")
    _run_command("scripts.validate_checkpoint", ["--checkpoint", str(checkpoint), "--out", str(output)], root, show, launch)
    if (output / "report.md").is_file():
        show(f"Report: {output / 'report.md'}")


def _protected_pool(root: Path, archive: Path) -> str:
    """Read the shared ledger before offering a one-shot evaluation.

    A copied held-out result with no canonical ledger means the spending
    history may have been left on another machine. Refuse to guess a free pool.
    The evaluator performs the authoritative locked check again at execution.
    """

    ledger = root / "runs" / "evaluation-pools-v1.json"
    if not ledger.is_file():
        if _paths(root, "held-out-*.json"):
            raise ValueError("held-out results exist but runs/evaluation-pools-v1.json is missing; restore the shared ledger before another protected evaluation")
        return "eval-1"
    manifest = _json(ledger)
    archive_info = _json(archive / "archive.json")
    fingerprint = archive_info["checkpoint_sha256"]
    for pool in manifest["pools"]:
        attempt = pool.get("attempt")
        if attempt and (attempt.get("checkpoint_sha256") == fingerprint or attempt.get("archive_identifier") == archive_info["identifier"]):
            raise ValueError(f"this checkpoint already spent {pool['name']}")
    for pool in manifest["pools"]:
        if pool["status"] == "available":
            return str(pool["name"])
    raise ValueError("all protected evaluation pools have been spent")


def _held_out(root: Path, read: Input, show: Output, launch: Launcher) -> None:
    archive = _choose([path.parent for path in _paths(root, "archive.json")], root, read, show)
    if archive is None:
        return
    try:
        pool = _protected_pool(root, archive)
    except (OSError, ValueError, KeyError, TypeError) as error:
        show(str(error))
        return
    show(f"Protected evaluation will permanently spend {pool} for {archive.name} before playing any games.")
    if read(f"Type SPEND {pool} to continue: ").strip() != f"SPEND {pool}":
        show("Cancelled; no pool was spent.")
        return
    _run_command("orrery.evaluate", ["--archive", str(archive), "--spend-eval", pool], root, show, launch)


def _reports(root: Path, read: Input, show: Output) -> None:
    paths = _paths(root, "report.md")
    paths += _paths(root, "pipeline-*.json") + _paths(root, "release-*.json")
    paths += _paths(root, "release-validation.json") + _paths(root, "held-out-*.json")
    path = _choose(sorted(set(paths)), root, read, show)
    if path is None:
        return
    if path.suffix == ".md":
        show(path.read_text())
        show(f"Charts and traces are beside {path}. See reading-training-charts.md for interpretation.")
    else:
        _show_evidence(path, show)


def _compare_runs(root: Path, read: Input, show: Output, launch: Launcher) -> None:
    runs = [path.parent for path in _paths(root, "manifest.json") if (path.parent / "metrics.csv").is_file()]
    show("First run:")
    first = _choose(runs, root, read, show)
    if first is None:
        return
    show("Second run:")
    second = _choose([run for run in runs if run != first], root, read, show)
    if second is None:
        return
    _run_command(
        "scripts.compare_training_runs",
        ["--run", f"A={first}", "--run", f"B={second}"],
        root, show, launch,
    )


def main_menu(root: Path = REPOSITORY_ROOT, *, read: Input = input, show: Output = print, launch: Launcher = _launch) -> int:
    """Keep discovery, reports, and established workflows in one terminal session."""

    actions = {
        "1": lambda: _start_training(root, read, show, launch),
        "2": lambda: _resume_training(root, read, show, launch),
        "3": lambda: _browse_runs(root, read, show),
        "4": lambda: _reports(root, read, show),
        "5": lambda: _held_out(root, read, show, launch),
        "6": lambda: _compare_runs(root, read, show, launch),
        "7": lambda: _validate_checkpoint(root, read, show, launch),
    }
    while True:
        show("\nOrrery control center")
        show("1. Start training  2. Resume checkpoint  3. Training runs  4. Reports and validation  5. Protected evaluation  6. Compare runs  7. Validate checkpoint  0. Exit")
        choice = read("Choose: ").strip()
        if choice == "0" or not choice:
            return 0
        action = actions.get(choice)
        if action is None:
            show("Choose 0 through 7.")
            continue
        try:
            action()
        except (OSError, ValueError, KeyError, TypeError) as error:
            show(f"Could not complete selection: {error}")


def _browse_runs(root: Path, read: Input, show: Output) -> None:
    runs = [path.parent for path in _paths(root, "manifest.json") if (path.parent / "metrics.csv").is_file()]
    run = _choose(runs, root, read, show)
    if run:
        _show_run(run, show)


if __name__ == "__main__":
    raise SystemExit(main_menu())
