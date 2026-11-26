#!/usr/bin/env python3
"""Run a whole session of experiments from one experiment file, and resume it.

An experiment file (TOML, in ``experiments/``) lists everything a session
runs. Each list or table of choices expands into one **job** per
combination, and every job gets its own folder in the session folder:

- ``[[botcheck]]``: bot-only games with ``opponentcheck``, one job per tester.
- ``[[diagnose]]``: a trained checkpoint plays the session's maps, one job per
  (opponent, run, update).
- ``[[train]]``: train from a checkpoint against an opponent, then diagnose
  every saved checkpoint, one job per (training opponent, starting checkpoint).
- ``[[compare]]`` and ``[[dial]]``: no jobs; they tell analyze_experiments.py
  what to compare.

A run has four stages:

1. **Preflight.** Expand every job, check that every path exists and every
   agent spec is canonical, build the Go tools, and print what will run. Any
   mistake stops here, in seconds, before anything runs.
2. **Run.** Two lanes at once: the GPU lane runs diagnose and train jobs one
   at a time, and the CPU lane runs bot jobs one at a time.
3. **Record.** A job writes ``done.json`` last, so a folder with it is a
   finished job that can be trusted. ``session.json`` holds the experiment
   file, the code version and every job's state.
4. **Resume.** Running the same command again skips finished jobs. A folder
   without ``done.json`` (interrupted or failed) is renamed
   ``<name>.partial-<n>``, never deleted, and the job runs again.

Run from the ``python/`` directory::

    uv run python -m scripts.run_experiments preflight ../experiments/stage1.toml
    uv run python -m scripts.run_experiments run ../experiments/stage1.toml --out ../runs/stage1
    uv run python -m scripts.run_experiments status ../runs/stage1
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from typing import Any

from orrery.config import run_config_from_mapping
from orrery.paths import REPOSITORY_ROOT

# Where the Python commands the jobs start are run from: the python/ folder,
# so ``-m scripts.diagnose`` and ``-m orrery.train`` resolve.
PYTHON_ROOT = REPOSITORY_ROOT / "python"

# A job label becomes a folder name, so it is kept to characters that are
# safe in a path on every system.
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# The four kinds of job lane: bot games need only CPUs; diagnose and train
# use the model, on the GPU when there is one.
CPU_LANE, GPU_LANE = "cpu", "gpu"


class PlanError(ValueError):
    """The experiment file can't be run as written."""


# ---------------------------------------------------------------------------
# Stage 1a: read the experiment file and expand it into jobs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Session:
    """The ``[session]`` table: settings every job shares."""

    name: str
    # The maps every job plays: map_count consecutive validation seeds from
    # map_start, each played from every seat.
    map_start: int
    map_count: int
    # Where models decide moves: "cuda" on the session machine, "cpu" for a
    # local smoke test. Written into each training config too.
    device: str
    # Games a bot job plays at once, and the Go threads it may use.
    cpu_workers: int
    # Go threads for the environment of a train or diagnose job, as the
    # Makefile's train-4p uses, so the bot lane's CPUs stay free.
    gomaxprocs: int


@dataclass(frozen=True)
class Job:
    """One unit of work with its own folder: ``<experiment>/<label>``.

    ``settings`` holds everything the job needs, as plain JSON values, so it
    can be written into the job's folder and compared on a resume.
    """

    experiment: str
    kind: str
    label: str
    settings: Mapping[str, Any]

    @property
    def id(self) -> str:
        return f"{self.experiment}/{self.label}"

    @property
    def lane(self) -> str:
        return CPU_LANE if self.kind == "botcheck" else GPU_LANE


@dataclass(frozen=True)
class Plan:
    """A whole session: its settings, its jobs, and what to compare."""

    session: Session
    jobs: tuple[Job, ...]
    # [[compare]] tables: baseline and candidate job ids, a question, and an
    # optional answer rule. Read by analyze_experiments.py.
    compares: tuple[Mapping[str, Any], ...]
    # [[dial]] tables: a name and an order of job ids whose win rates should
    # fall at every step.
    dials: tuple[Mapping[str, Any], ...]
    # Per-experiment answer settings, such as a diagnose band, by name.
    answers: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def specs_of(jobs: Sequence[Job]) -> list[str]:
    """Every agent spec or lineup the jobs play, once each, in order."""

    found: list[str] = []
    for job in jobs:
        for key in ("tester", "opponent", "training_opponent"):
            if key in job.settings and job.settings[key] not in found:
                found.append(job.settings[key])
        for spec in job.settings.get("diagnose_against", {}).values():
            if spec not in found:
                found.append(spec)
    return found


def checkpoint_count(job: Job) -> int:
    """How many checkpoints a train job saves: one every N updates, plus the
    last update when it isn't a multiple of N."""

    every, updates = job.settings["checkpoint_every_updates"], job.settings["max_updates"]
    return updates // every + (updates % every != 0)


def load_plan(path: Path) -> Plan:
    """Read an experiment file and expand it into a Plan."""

    return parse_plan(path.read_text())


def parse_plan(text: str) -> Plan:
    """Expand an experiment file's text into a Plan, or raise PlanError.

    Only the file's own shape is checked here: required fields, types and
    labels. Whether paths exist and specs are canonical is preflight's job,
    because those checks need the files and Go.
    """

    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise PlanError(f"experiment file is not valid TOML: {error}") from error
    known = {"session", "botcheck", "diagnose", "train", "compare", "dial"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise PlanError(f"unknown sections {unknown}; known: {sorted(known)}")

    session = _session(_table(raw, "session", "the file"))
    jobs: list[Job] = []
    answers: dict[str, Mapping[str, Any]] = {}
    for table in raw.get("botcheck", []):
        jobs.extend(_botcheck_jobs(table))
    for table in raw.get("diagnose", []):
        jobs.extend(_diagnose_jobs(table))
        if "band" in table:
            answers[table["name"]] = _band(table)
    for table in raw.get("train", []):
        jobs.extend(_train_jobs(table))
        answers[table["name"]] = {"progress_min_runs": _int(table, "progress_min_runs", "[[train]]", default=2)}

    # Experiment names group jobs into folders, so two sections may not
    # share one, and job ids must be unique.
    names = [job.experiment for job in jobs]
    sections = [t["name"] for kind in ("botcheck", "diagnose", "train") for t in raw.get(kind, [])]
    duplicates = sorted({name for name in sections if sections.count(name) > 1})
    if duplicates:
        raise PlanError(f"experiment names must be unique; repeated: {duplicates}")
    ids = [job.id for job in jobs]
    repeated = sorted({job_id for job_id in ids if ids.count(job_id) > 1})
    if repeated:
        raise PlanError(f"two jobs would share a folder: {repeated}")
    if not names:
        raise PlanError("the experiment file has no [[botcheck]], [[diagnose]] or [[train]] section")

    bot_ids = {job.id for job in jobs if job.kind == "botcheck"}
    compares = tuple(_compare(table, bot_ids) for table in raw.get("compare", []))
    dials = tuple(_dial(table, bot_ids) for table in raw.get("dial", []))
    return Plan(session, tuple(jobs), compares, dials, answers)


def _session(table: Mapping[str, Any]) -> Session:
    where = "[session]"
    maps = _table(table, "maps", where)
    session = Session(
        name=_label(table, "name", where),
        map_start=_int(maps, "start", f"{where} maps"),
        map_count=_int(maps, "count", f"{where} maps"),
        device=_choice(table, "device", where, ("cuda", "cpu")),
        cpu_workers=_int(table, "cpu_workers", where),
        gomaxprocs=_int(table, "gomaxprocs", where, default=12),
    )
    for name in ("map_count", "cpu_workers", "gomaxprocs"):
        if getattr(session, name) < 1:
            raise PlanError(f"{where} {name} must be at least 1")
    return session


def _botcheck_jobs(table: Mapping[str, Any]) -> list[Job]:
    """One job per tester, each against the same opponent."""

    where = "[[botcheck]]"
    name = _label(table, "name", where)
    opponent = _string(table, "opponent", where)
    testers = _named_specs(table, "testers", where)
    check_decode = _strings(table, "check_decode", where, default=[])
    unknown = sorted(set(check_decode) - set(testers))
    if unknown:
        raise PlanError(f"{where} {name}: check_decode names testers it doesn't have: {unknown}")
    return [
        Job(name, "botcheck", label, {
            "tester": spec, "opponent": opponent, "check_decode": label in check_decode,
        })
        for label, spec in testers.items()
    ]


def _diagnose_jobs(table: Mapping[str, Any]) -> list[Job]:
    """One job per (opponent, run, update), named after all three."""

    where = "[[diagnose]]"
    name = _label(table, "name", where)
    run_dirs = _strings(table, "run_dirs", where)
    updates = _ints(table, "updates", where)
    opponents = _named_specs(table, "opponents", where)
    return [
        Job(name, "diagnose", f"{label}__{Path(run_dir).name}__u{update:06d}", {
            "run_dir": run_dir, "update": update, "opponent": spec, "opponent_label": label,
        })
        for label, spec in opponents.items()
        for run_dir in run_dirs
        for update in updates
    ]


def _train_jobs(table: Mapping[str, Any]) -> list[Job]:
    """One job per (training opponent, starting checkpoint).

    A starting checkpoint is named after its run: ``runs/x/checkpoints/u.pt``
    gives ``x``, which keeps the training seed in the folder name.
    """

    where = "[[train]]"
    name = _label(table, "name", where)
    starts = _strings(table, "init_from", where)
    opponents = _named_specs(table, "training_opponents", where)
    against = _named_specs(table, "diagnose_against", where)
    common = {
        "base_config": _string(table, "base_config", where),
        "max_updates": _int(table, "max_updates", where),
        "checkpoint_every_updates": _int(table, "checkpoint_every_updates", where),
        "diagnose_against": against,
    }
    if common["max_updates"] < 1 or common["checkpoint_every_updates"] < 1:
        raise PlanError(f"{where} {name}: max_updates and checkpoint_every_updates must be at least 1")
    return [
        Job(name, "train", f"{label}__{Path(start).parent.parent.name}", {
            **common, "init_from": start, "training_opponent": spec, "opponent_label": label,
        })
        for label, spec in opponents.items()
        for start in starts
    ]


def _compare(table: Mapping[str, Any], bot_ids: set[str]) -> Mapping[str, Any]:
    where = "[[compare]]"
    compare = {
        "baseline": _string(table, "baseline", where),
        "candidate": _string(table, "candidate", where),
        "question": _string(table, "question", where),
        "rule": table.get("rule"),
    }
    for key in ("baseline", "candidate"):
        if compare[key] not in bot_ids:
            raise PlanError(f"{where} {key} {compare[key]!r} is not a bot job; bot jobs: {sorted(bot_ids)}")
    if compare["rule"] not in (None, "restricted-bot"):
        raise PlanError(f"{where} rule must be 'restricted-bot' or left out, got {compare['rule']!r}")
    return compare


def _dial(table: Mapping[str, Any], bot_ids: set[str]) -> Mapping[str, Any]:
    where = "[[dial]]"
    dial = {"name": _label(table, "name", where), "order": _strings(table, "order", where)}
    unknown = [job_id for job_id in dial["order"] if job_id not in bot_ids]
    if unknown or len(dial["order"]) < 2:
        raise PlanError(f"{where} {dial['name']}: order needs at least two bot jobs; unknown: {unknown}")
    return dial


def _band(table: Mapping[str, Any]) -> Mapping[str, Any]:
    where = f"[[diagnose]] {table.get('name')}"
    band = table["band"]
    if (
        not isinstance(band, list) or len(band) != 2
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in band)
        or not 0 <= band[0] < band[1] <= 1
    ):
        raise PlanError(f"{where}: band must be [low, high] with 0 <= low < high <= 1")
    return {"band": [float(band[0]), float(band[1])], "band_min_runs": _int(table, "band_min_runs", where, default=2)}


# Small readers that name the field and section in every error.


def _table(raw: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise PlanError(f"{where} needs a [{key}] table")
    return value


def _string(raw: Mapping[str, Any], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{where} {key} must be a non-empty string")
    return value


def _label(raw: Mapping[str, Any], key: str, where: str) -> str:
    value = _string(raw, key, where)
    if not LABEL_PATTERN.match(value):
        raise PlanError(f"{where} {key} {value!r} must use only letters, digits, '.', '_' and '-'")
    return value


def _int(raw: Mapping[str, Any], key: str, where: str, default: int | None = None) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{where} {key} must be an integer")
    return value


def _choice(raw: Mapping[str, Any], key: str, where: str, choices: Sequence[str]) -> str:
    value = raw.get(key)
    if value not in choices:
        raise PlanError(f"{where} {key} must be one of {list(choices)}, got {value!r}")
    return value


def _strings(raw: Mapping[str, Any], key: str, where: str, default: list[str] | None = None) -> list[str]:
    value = raw.get(key, default)
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise PlanError(f"{where} {key} must be a list of non-empty strings")
    if default is None and not value:
        raise PlanError(f"{where} {key} must not be empty")
    return value


def _ints(raw: Mapping[str, Any], key: str, where: str) -> list[int]:
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        raise PlanError(f"{where} {key} must be a non-empty list of integers")
    return value


def _named_specs(raw: Mapping[str, Any], key: str, where: str) -> dict[str, str]:
    """A table of ``label = "spec"``. The label names the job's folder, so a
    long spec never has to be one."""

    value = raw.get(key)
    if not isinstance(value, dict) or not value:
        raise PlanError(f'{where} {key} must be a table of label = "agent spec"')
    for label, spec in value.items():
        if not LABEL_PATTERN.match(label):
            raise PlanError(f"{where} {key} label {label!r} must use only letters, digits, '.', '_' and '-'")
        if not isinstance(spec, str) or not spec.strip():
            raise PlanError(f"{where} {key}.{label} must be a non-empty agent spec")
    return dict(value)


# ---------------------------------------------------------------------------
# Stage 1b: preflight. Check files and specs, build Go, count the work.
# ---------------------------------------------------------------------------


def repository_path(path: str) -> Path:
    """A path from the experiment file, which is written from the repository root."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def checkpoint_path(run_dir: str, update: int) -> Path:
    return repository_path(run_dir) / "checkpoints" / f"update-{update:06d}.pt"


def training_config(job: Job, session: Session) -> dict[str, Any]:
    """The full training config a train job runs: its base config with this
    job's opponent, checkpoint cadence and the session's device."""

    with repository_path(job.settings["base_config"]).open("rb") as source:
        config = tomllib.load(source)
    config.update(
        training_opponent=job.settings["training_opponent"],
        checkpoint_every_updates=job.settings["checkpoint_every_updates"],
        device=session.device,
    )
    return config


def build_go_tools(bin_dir: Path, runner: Callable[..., Any] = subprocess.run) -> Path:
    """Build opponentcheck once, and compile cmd/env so a broken build shows
    now. Training and diagnosis start cmd/env with ``go run``, which then
    reuses this build from Go's cache. Returns opponentcheck's path."""

    bin_dir.mkdir(parents=True, exist_ok=True)
    tool = bin_dir / "opponentcheck"
    for package, output in (("./cmd/opponentcheck", tool), ("./cmd/env", bin_dir / "env")):
        result = runner(["go", "build", "-o", str(output), package], cwd=REPOSITORY_ROOT,
                        capture_output=True, text=True)
        if result.returncode != 0:
            raise PlanError(f"go build {package} failed:\n{result.stderr}")
    return tool


def canonical_problems(specs: Sequence[str], tool: Path,
                       runner: Callable[..., Any] = subprocess.run) -> list[str]:
    """Ask Go for each spec's canonical form; list every spec that differs.

    Training refuses a non-canonical opponent as it starts, and the analysis
    matches results by spec text, so the file must use Go's exact spelling.
    """

    if not specs:
        return []
    result = runner([str(tool), "-canonical", *specs], capture_output=True, text=True)
    if result.returncode != 0:
        return [f"opponentcheck refused a spec:\n{result.stderr.strip()}"]
    forms = result.stdout.splitlines()
    return [
        f"spec is not canonical: {spec!r}\n    write it as: {form!r}"
        for spec, form in zip(specs, forms, strict=True)
        if spec != form
    ]


def file_problems(plan: Plan, jobs: Sequence[Job]) -> list[str]:
    """Every missing checkpoint or run, and every train config that won't load."""

    problems = []
    for job in jobs:
        if job.kind == "diagnose":
            path = checkpoint_path(job.settings["run_dir"], job.settings["update"])
            if not path.is_file():
                problems.append(f"{job.id}: checkpoint {path} does not exist")
        elif job.kind == "train":
            start = repository_path(job.settings["init_from"])
            if not start.is_file():
                problems.append(f"{job.id}: starting checkpoint {start} does not exist")
            base = repository_path(job.settings["base_config"])
            if not base.is_file():
                problems.append(f"{job.id}: base config {base} does not exist")
                continue
            try:
                run_config_from_mapping(training_config(job, plan.session))
            except (TypeError, ValueError) as error:
                problems.append(f"{job.id}: training config is invalid: {error}")
    return problems


def describe(plan: Plan, jobs: Sequence[Job]) -> list[str]:
    """What the run will do, in a few lines: jobs, games and updates."""

    games_per_job = 4 * plan.session.map_count
    lines = [f"session {plan.session.name}: {len(jobs)} jobs, {plan.session.map_count} maps x 4 seats"]
    for experiment in dict.fromkeys(job.experiment for job in jobs):
        group = [job for job in jobs if job.experiment == experiment]
        kind = group[0].kind
        if kind == "train":
            updates = sum(job.settings["max_updates"] for job in group)
            diagnoses = sum(checkpoint_count(job) * len(job.settings["diagnose_against"]) for job in group)
            lines.append(f"  {experiment} (train, GPU lane): {len(group)} runs, {updates} updates, "
                         f"{diagnoses} checkpoint diagnoses, {diagnoses * games_per_job} games")
        else:
            lane = "CPU lane" if kind == "botcheck" else "GPU lane"
            lines.append(f"  {experiment} ({kind}, {lane}): {len(group)} jobs, {len(group) * games_per_job} games")
    return lines


def preflight(plan: Plan, jobs: Sequence[Job], bin_dir: Path) -> tuple[list[str], Path | None]:
    """Run every check. Returns the problems found and opponentcheck's path."""

    problems = file_problems(plan, jobs)
    try:
        tool = build_go_tools(bin_dir)
    except PlanError as error:
        return [*problems, str(error)], None
    problems += canonical_problems(specs_of(jobs), tool)
    return problems, tool


# ---------------------------------------------------------------------------
# Stages 2 and 3: run each job in its lane and record it.
# ---------------------------------------------------------------------------


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    """Write JSON to a temporary file, then rename it into place, so a reader
    never sees half a file."""

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def toml_text(values: Mapping[str, Any]) -> str:
    """A flat TOML table for a training config: strings, numbers, booleans and
    lists of them. Strings are written as JSON strings, which TOML reads the
    same way."""

    def value(item: Any) -> str:
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, (int, float)):
            return repr(item)
        if isinstance(item, str):
            return json.dumps(item)
        if isinstance(item, list):
            return "[" + ", ".join(value(element) for element in item) + "]"
        raise TypeError(f"can't write {item!r} as TOML")

    return "".join(f"{key} = {value(item)}\n" for key, item in values.items())


class Runner:
    """Runs jobs and keeps track of the processes they start, so an
    interrupted session can stop them all."""

    def __init__(self, plan: Plan, out: Path, tool: Path) -> None:
        self.plan = plan
        self.out = out
        self.tool = tool
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[bytes]] = set()
        self._stopping = False

    def stop(self) -> None:
        """Stop every running command; no new one starts afterwards."""

        with self._lock:
            self._stopping = True
            for process in self._processes:
                process.terminate()

    @property
    def stopping(self) -> bool:
        with self._lock:
            return self._stopping

    def command(self, arguments: Sequence[str], log: Path, *, cwd: Path, gomaxprocs: int) -> int:
        """Run one command, appending its output to the job's log."""

        env = {**os.environ, "GOMAXPROCS": str(gomaxprocs)}
        with log.open("ab") as handle:
            handle.write(f"$ {' '.join(arguments)}\n".encode())
            handle.flush()
            with self._lock:
                if self._stopping:
                    return -1
                process = subprocess.Popen(arguments, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
                self._processes.add(process)
            try:
                return process.wait()
            finally:
                with self._lock:
                    self._processes.discard(process)

    def run_job(self, job: Job) -> str:
        """Run one job in its folder and return "done" or "failed".

        The stages are: set aside any unfinished folder; create a fresh one
        and record what it runs; run the job's commands; write done.json last,
        or failed.json. A job that raises is recorded as failed and never
        stops the other jobs.
        """

        folder = self.out / job.experiment / job.label
        if (folder / "done.json").exists():
            return "done"
        if folder.exists():
            set_aside(folder)
        folder.mkdir(parents=True)
        started = time.monotonic()
        record = {"job": job.id, "kind": job.kind, "settings": dict(job.settings), "started": now()}
        write_json(folder / "job.json", record)
        log = folder / "log.txt"
        try:
            failed = self._run(job, folder, log)
        except Exception as error:  # recorded, so one broken job can't stop the session
            failed = f"{type(error).__name__}: {error}"
        record.update(finished=now(), seconds=round(time.monotonic() - started, 1))
        if failed:
            write_json(folder / "failed.json", {**record, "error": failed, "log": "log.txt"})
            return "failed"
        write_json(folder / "done.json", record)
        return "done"

    def _run(self, job: Job, folder: Path, log: Path) -> str | None:
        """Run a job's commands; return why it failed, or None."""

        session = self.plan.session
        maps = ["--seeds", str(session.map_count), "--seed-start", str(session.map_start)]
        if job.kind == "botcheck":
            arguments = [
                str(self.tool), "-tester-spec", job.settings["tester"], "-opponent", job.settings["opponent"],
                "-seed-start", str(session.map_start), "-seeds", str(session.map_count),
                "-workers", str(session.cpu_workers), "-out", str(folder / "opponentcheck"),
            ]
            if job.settings["check_decode"]:
                arguments.insert(-2, "-check-decode")
            code = self.command(arguments, log, cwd=REPOSITORY_ROOT, gomaxprocs=session.cpu_workers)
            return None if code == 0 else f"opponentcheck exited with {code}"

        if job.kind == "diagnose":
            run_dir = repository_path(job.settings["run_dir"])
            return self._diagnose(run_dir, checkpoint_path(job.settings["run_dir"], job.settings["update"]),
                                  job.settings["opponent"], folder / "diagnosis", maps, log)

        # A train job: write its own full config, train, then diagnose every
        # checkpoint the run saved against every diagnose_against opponent.
        config = folder / "config.toml"
        config.write_text(toml_text(training_config(job, session)))
        run_dir = folder / "run"
        code = self.command([
            sys.executable, "-m", "orrery.train", "--config", str(config), "--run-dir", str(run_dir),
            "--init-from", str(repository_path(job.settings["init_from"])),
            "--max-updates", str(job.settings["max_updates"]),
        ], log, cwd=PYTHON_ROOT, gomaxprocs=session.gomaxprocs)
        if code != 0:
            return f"training exited with {code}"
        checkpoints = sorted((run_dir / "checkpoints").glob("update-*.pt"))
        if not checkpoints:
            return "training saved no checkpoint"
        for checkpoint in checkpoints:
            for label, spec in job.settings["diagnose_against"].items():
                failed = self._diagnose(run_dir, checkpoint, spec,
                                        folder / "diagnoses" / f"{checkpoint.stem}__{label}", maps, log)
                if failed:
                    return failed
        return None

    def _diagnose(self, run_dir: Path, checkpoint: Path, opponent: str, out: Path,
                  maps: list[str], log: Path) -> str | None:
        code = self.command([
            sys.executable, "-m", "scripts.diagnose", "--run-dir", str(run_dir), "--checkpoint", str(checkpoint),
            "--opponent", opponent, *maps, "--device", self.plan.session.device, "--no-charts",
            "--out", str(out),
        ], log, cwd=PYTHON_ROOT, gomaxprocs=self.plan.session.gomaxprocs)
        return None if code == 0 else f"diagnose of {checkpoint.name} against {opponent} exited with {code}"


def set_aside(folder: Path) -> Path:
    """Rename an unfinished job folder to ``<name>.partial-<n>``, the first
    unused n, so its files are kept for inspection but never mixed with the
    rerun's."""

    n = 1
    while (target := folder.with_name(f"{folder.name}.partial-{n}")).exists():
        n += 1
    folder.rename(target)
    return target


# ---------------------------------------------------------------------------
# Stage 4 and the session record: session.json, status and the two lanes.
# ---------------------------------------------------------------------------


def job_state(out: Path, job: Job) -> str:
    """done, failed, unfinished (a folder without a result: running now, or
    interrupted), or waiting (not started)."""

    folder = out / job.experiment / job.label
    if (folder / "done.json").exists():
        return "done"
    if (folder / "failed.json").exists():
        return "failed"
    return "unfinished" if folder.exists() else "waiting"


def versions() -> dict[str, Any]:
    """The code and tools a session ran with, for replay."""

    def output(arguments: Sequence[str]) -> str | None:
        try:
            result = subprocess.run(arguments, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    dirty = output(["git", "status", "--porcelain", "--untracked-files=no"])
    return {
        "git_commit": output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "go": output(["go", "version"]),
        "python": platform.python_version(),
    }


def run_session(experiment: Path, out: Path, only: str | None = None) -> int:
    """Preflight, then run every job not yet done, two lanes at once.

    Returns 0 when every selected job is done, 1 when any failed, and 2 when
    preflight found problems or the session folder holds a different
    experiment file.
    """

    text = experiment.read_text()
    plan = parse_plan(text)
    jobs = [job for job in plan.jobs if only is None or job.experiment == only]
    if not jobs:
        print(f"no experiment named {only!r}; experiments: {sorted({j.experiment for j in plan.jobs})}")
        return 2

    # A session folder belongs to one experiment file. Resuming with an
    # edited file would mix results made under different settings.
    out.mkdir(parents=True, exist_ok=True)
    session_file = out / "session.json"
    record: dict[str, Any] = json.loads(session_file.read_text()) if session_file.exists() else {}
    if record and record["experiment_text"] != text:
        print(f"{out} holds a session of a different experiment file; use a new --out")
        return 2

    problems, tool = preflight(plan, jobs, out / "bin")
    for line in describe(plan, jobs):
        print(line)
    if problems:
        print("preflight found problems; nothing was run:")
        for problem in problems:
            print(f"- {problem}")
        return 2
    assert tool is not None

    record.setdefault("experiment_file", str(experiment))
    record.setdefault("experiment_text", text)
    record.setdefault("runs", []).append({"started": now(), "only": only, **versions()})
    lock = threading.Lock()

    def save(job: Job | None = None, state: str | None = None) -> None:
        with lock:
            if job is not None:
                record.setdefault("jobs", {})[job.id] = state
            write_json(session_file, record)

    for job in jobs:
        record.setdefault("jobs", {})[job.id] = job_state(out, job)
    save()

    runner = Runner(plan, out, tool)

    def lane(name: str) -> None:
        for job in (job for job in jobs if job.lane == name):
            # After Ctrl-C, start nothing new: the remaining jobs stay
            # waiting for the next run.
            if runner.stopping:
                return
            if job_state(out, job) == "done":
                continue
            print(f"[{name}] start {job.id}", flush=True)
            save(job, "running")
            state = runner.run_job(job)
            save(job, state)
            print(f"[{name}] {state} {job.id}", flush=True)

    threads = [threading.Thread(target=lane, args=(name,), daemon=True) for name in (GPU_LANE, CPU_LANE)]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            while thread.is_alive():
                thread.join(timeout=1)
    except KeyboardInterrupt:
        print("interrupted: stopping running jobs; run the same command again to resume", flush=True)
        runner.stop()
        for thread in threads:
            thread.join()
        for job in jobs:
            record["jobs"][job.id] = job_state(out, job)
        record["runs"][-1]["finished"] = now()
        save()
        return 1

    for job in jobs:
        record["jobs"][job.id] = job_state(out, job)
    record["runs"][-1]["finished"] = now()
    save()
    failed = [job.id for job in jobs if record["jobs"][job.id] != "done"]
    print(f"session {plan.session.name}: {len(jobs) - len(failed)} of {len(jobs)} jobs done; results in {out}")
    for job_id in failed:
        print(f"- {record['jobs'][job_id]}: {job_id}")
    return 1 if failed else 0


def print_status(out: Path) -> int:
    """Print each job's state from the session folder."""

    session_file = out / "session.json"
    if not session_file.exists():
        print(f"{out} has no session.json; no session has run there")
        return 2
    plan = parse_plan(json.loads(session_file.read_text())["experiment_text"])
    counts: dict[str, int] = {}
    for job in plan.jobs:
        state = job_state(out, job)
        counts[state] = counts.get(state, 0) + 1
        print(f"{state:10} {job.id}")
    print(", ".join(f"{count} {state}" for state, count in sorted(counts.items())))
    return 0


def run_preflight(experiment: Path, only: str | None = None) -> int:
    """Stage 1 only: check and count, run nothing, write nothing kept."""

    plan = load_plan(experiment)
    jobs = [job for job in plan.jobs if only is None or job.experiment == only]
    with tempfile.TemporaryDirectory() as scratch:
        problems, _tool = preflight(plan, jobs, Path(scratch))
    for line in describe(plan, jobs):
        print(line)
    for problem in problems:
        print(f"- {problem}")
    print("preflight passed" if not problems else f"preflight found {len(problems)} problems")
    return 2 if problems else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="check and count, run nothing")
    check.add_argument("experiment", type=Path)
    check.add_argument("--only", help="check one experiment by name")
    run = commands.add_parser("run", help="run everything not yet done; the same command resumes")
    run.add_argument("experiment", type=Path)
    run.add_argument("--out", type=Path, required=True, help="session folder, new or from an earlier run")
    run.add_argument("--only", help="run one experiment by name")
    status = commands.add_parser("status", help="each job's state in a session folder")
    status.add_argument("out", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "preflight":
            return run_preflight(arguments.experiment.resolve(), arguments.only)
        if arguments.command == "run":
            return run_session(arguments.experiment.resolve(), arguments.out.resolve(), arguments.only)
        return print_status(arguments.out.resolve())
    except PlanError as error:
        print(f"experiment file: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
