#!/usr/bin/env python3
"""Run a whole session of experiments from one experiment file, and resume it.

Words used throughout this file:

- **Experiment file**: a TOML file in ``experiments/`` that lists everything a
  session should run. TOML is a simple settings format: ``name = "value"``
  lines, grouped under headings in square brackets.
- **Session**: one run of an experiment file. All of its results go into one
  **session folder** (``--out``).
- **Experiment**: one ``[[botcheck]]``, ``[[diagnose]]`` or ``[[train]]``
  section of the file, named by its ``name`` field.
- **Job**: one unit of work inside an experiment, with its own folder. Each
  list or table of choices in a section expands into one job per
  combination. Example: a ``[[diagnose]]`` with 2 opponents, 3 runs and 1
  update gives 2 × 3 × 1 = 6 jobs.
- **Agent spec**: the text that names a bot and its settings, such as
  ``heuristic-v2:candidates=4,...``. Go defines one **canonical** spelling
  for each (settings in alphabetical order, defaults left out).

The kinds of section:

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

A session folder looks like this (one experiment shown)::

    runs/stage1/
        session.json                    what ran, with which code, and each job's state
        bin/opponentcheck               the Go tool, built once by preflight
        restricted-bot/                 one folder per experiment
            normal/                     one folder per job
                job.json                what this job runs (written first)
                log.txt                 everything its commands printed
                opponentcheck/          the command's own results
                done.json               written last: the job finished
            aim.partial-1/              an earlier, unfinished try, kept

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
# so ``-m scripts.diagnose`` and ``-m orrery.train`` resolve. (``python -m
# scripts.diagnose`` looks for a ``scripts`` folder in the current folder.)
PYTHON_ROOT = REPOSITORY_ROOT / "python"

# A job label becomes a folder name, so it is kept to characters that are
# safe in a path on every system: letters, digits, '.', '_' and '-', and it
# must start with a letter or digit (so it can't be ".." or a hidden ".x").
# Examples: "normal", "eps-0.5" and "v2-eps0.5" are fine; "my tester" (a
# space) and "a/b" (a slash) are refused.
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# The two lanes. A lane is a queue of jobs run one after another; the two
# lanes run at the same time. Bot games need only CPUs, so they go in the CPU
# lane; diagnose and train use the model, on the GPU when there is one, so
# they go in the GPU lane. One GPU job at a time keeps them from competing
# for the GPU's memory.
CPU_LANE, GPU_LANE = "cpu", "gpu"


class PlanError(ValueError):
    """The experiment file can't be run as written.

    Every check that reads the file raises this, with a message that names
    the section and field at fault, so the command can print it and stop.
    """


# ---------------------------------------------------------------------------
# Stage 1a: read the experiment file and expand it into jobs.
#
# Reading turns the file's text into a Plan: the session's settings plus a
# flat list of Jobs. Nothing here touches the disk or Go; only the file's own
# shape is checked (required fields, types, safe labels).
# ---------------------------------------------------------------------------


# frozen=True makes these objects read-only once built, so a job can't be
# changed by accident while two lanes are using the plan at the same time.
@dataclass(frozen=True)
class Session:
    """The ``[session]`` table: settings every job shares."""

    # The session's name, shown in messages and the analysis title.
    name: str
    # The maps every job plays: map_count consecutive validation seeds from
    # map_start, each played from every seat. Example: start 1,000,000 and
    # count 3 play seeds 1,000,000, 1,000,001 and 1,000,002, so a
    # four-player job plays 3 × 4 = 12 games.
    map_start: int
    map_count: int
    # Where models decide moves: "cuda" on the session machine, "cpu" for a
    # local smoke test. Written into each training config too.
    device: str
    # How many games a bot job plays at once, and how many Go threads it may
    # use to play them.
    cpu_workers: int
    # Go threads for the game environment of a train or diagnose job. Go
    # reads its thread limit from the GOMAXPROCS environment variable. Each
    # job gets its own value, so model jobs leave CPUs free for the bot lane,
    # and bot jobs aren't held to the model jobs' limit.
    gomaxprocs: int


@dataclass(frozen=True)
class Job:
    """One unit of work with its own folder: ``<experiment>/<label>``.

    ``settings`` holds everything the job needs, as plain JSON values (text,
    numbers, lists, tables), so it can be written into the job's folder as
    job.json and read back later. Which keys it has depends on ``kind``:

    - botcheck: tester, opponent, check_decode
    - diagnose: run_dir, update, opponent, opponent_label
    - train: base_config, max_updates, checkpoint_every_updates,
      diagnose_against, init_from, training_opponent, opponent_label
    """

    # The section's name, which is also the job's parent folder.
    experiment: str
    # "botcheck", "diagnose" or "train": which kind of section made it.
    kind: str
    # The job's own folder name, made from what this job varies.
    label: str
    settings: Mapping[str, Any]

    @property
    def id(self) -> str:
        """The job's unique name, also its path inside the session folder.
        Example: ``restricted-bot/normal``."""

        return f"{self.experiment}/{self.label}"

    @property
    def lane(self) -> str:
        """Which lane runs the job: bot jobs need no model, so they use the
        CPU lane; every other job uses the GPU lane."""

        return CPU_LANE if self.kind == "botcheck" else GPU_LANE


@dataclass(frozen=True)
class Plan:
    """A whole session: its settings, its jobs, and what to compare."""

    session: Session
    # Every job, in the order the file lists them. A tuple (a list that can't
    # be changed) keeps the frozen dataclass truly read-only.
    jobs: tuple[Job, ...]
    # [[compare]] tables: baseline and candidate job ids, a question, and an
    # optional answer rule. Read by analyze_experiments.py.
    compares: tuple[Mapping[str, Any], ...]
    # [[dial]] tables: a name and an order of job ids whose win rates should
    # fall at every step.
    dials: tuple[Mapping[str, Any], ...]
    # Per-experiment answer settings, such as a diagnose band, by experiment
    # name. Example: {"probe": {"band": [0.1, 0.4], "band_min_runs": 2}}.
    # field(default_factory=dict) gives each Plan its own empty table; a
    # plain ``= {}`` would be one table shared by every Plan.
    answers: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def specs_of(jobs: Sequence[Job]) -> list[str]:
    """Every agent spec or lineup the jobs play, once each, in order.

    Preflight sends this list to Go to check each spelling. A spec used by
    many jobs (such as the opponent of every bot job) is listed once.
    """

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
    last update when it isn't a multiple of N.

    Example: 7 updates with a checkpoint every 3 save at 3 and 6, and also at
    7, the last: 7 // 3 = 2, plus 1 because 7 % 3 isn't 0, so 3 checkpoints.
    """

    every, updates = job.settings["checkpoint_every_updates"], job.settings["max_updates"]
    # True counts as 1 when added to a number, so this adds the extra one.
    return updates // every + (updates % every != 0)


def load_plan(path: Path) -> Plan:
    """Read an experiment file and expand it into a Plan."""

    return parse_plan(path.read_text())


def parse_plan(text: str) -> Plan:
    """Expand an experiment file's text into a Plan, or raise PlanError.

    Only the file's own shape is checked here: required fields, types and
    labels. Whether paths exist and specs are canonical is preflight's job,
    because those checks need the files and Go.

    It takes the text rather than a path so that ``status`` and the analysis
    can rebuild the plan from the copy saved in session.json, even if the
    file on disk has since changed.
    """

    # Step 1: read the TOML. ``[[botcheck]]`` (double brackets) means "one
    # more item in a list", so raw["botcheck"] is a list of tables, one per
    # section in the file.
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise PlanError(f"experiment file is not valid TOML: {error}") from error
    # A misspelled section name ([[diagnoze]]) would otherwise be silently
    # ignored, and its jobs never run.
    known = {"session", "botcheck", "diagnose", "train", "compare", "dial"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise PlanError(f"unknown sections {unknown}; known: {sorted(known)}")

    # Step 2: expand each section into its jobs, in file order, and keep the
    # settings each answer needs.
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

    # Step 3: check the jobs fit together. Experiment names group jobs into
    # folders, so two sections may not share one, and job ids must be
    # unique, or two jobs would write into the same folder.
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

    # Step 4: [[compare]] and [[dial]] name bot jobs by id; check each one
    # exists now, rather than finding out in the analysis after the session.
    bot_ids = {job.id for job in jobs if job.kind == "botcheck"}
    compares = tuple(_compare(table, bot_ids) for table in raw.get("compare", []))
    dials = tuple(_dial(table, bot_ids) for table in raw.get("dial", []))
    return Plan(session, tuple(jobs), compares, dials, answers)


def _session(table: Mapping[str, Any]) -> Session:
    """Read and check the [session] table."""

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
    # Zero maps or zero threads would run nothing, or hang; refuse them.
    for name in ("map_count", "cpu_workers", "gomaxprocs"):
        if getattr(session, name) < 1:
            raise PlanError(f"{where} {name} must be at least 1")
    return session


def _botcheck_jobs(table: Mapping[str, Any]) -> list[Job]:
    """One job per tester, each against the same opponent.

    Example: testers ``normal`` and ``aim`` in an experiment named ``bots``
    give jobs ``bots/normal`` and ``bots/aim``.
    """

    where = "[[botcheck]]"
    name = _label(table, "name", where)
    opponent = _string(table, "opponent", where)
    testers = _named_specs(table, "testers", where)
    # check_decode lists the tester labels whose launches are checked against
    # the learner's decoder (opponentcheck -check-decode). A label that isn't
    # a tester is a typo, and the check would silently not run.
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
    """One job per (opponent, run, update), named after all three.

    Example: opponent label ``v2-eps0.5``, run folder
    ``runs/vast-experiment/exp-b-4p-s1501`` and update 153 give the job
    ``probe/v2-eps0.5__exp-b-4p-s1501__u000153``. The double underscore
    separates the parts, since a single one may appear inside a name. The
    update is padded to six digits, as checkpoint files are, so folders sort
    in update order.
    """

    where = "[[diagnose]]"
    name = _label(table, "name", where)
    run_dirs = _strings(table, "run_dirs", where)
    updates = _ints(table, "updates", where)
    opponents = _named_specs(table, "opponents", where)
    # Three nested loops: every opponent with every run with every update.
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
    gives ``x``, which keeps the training seed in the folder name. Example:
    training opponent ``v2-eps0.75`` from
    ``runs/vast-experiment/exp-b-4p-s1501/checkpoints/update-000153.pt``
    gives ``train-weakened/v2-eps0.75__exp-b-4p-s1501``.
    """

    where = "[[train]]"
    name = _label(table, "name", where)
    starts = _strings(table, "init_from", where)
    opponents = _named_specs(table, "training_opponents", where)
    against = _named_specs(table, "diagnose_against", where)
    # Settings every run of this section shares.
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
    """Read one [[compare]]: candidate minus baseline, both bot job ids.

    ``rule``, when present, names the answer the analysis gives for this
    comparison; "restricted-bot" is experiment 1's.
    """

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
    """Read one [[dial]]: bot jobs in the order their win rate should fall.

    A "dial" is a setting meant to make a bot weaker step by step, like
    turning a knob; experiment 2 checks epsilon behaves like one. It needs at
    least two steps to have anything to compare.
    """

    where = "[[dial]]"
    dial = {"name": _label(table, "name", where), "order": _strings(table, "order", where)}
    unknown = [job_id for job_id in dial["order"] if job_id not in bot_ids]
    if unknown or len(dial["order"]) < 2:
        raise PlanError(f"{where} {dial['name']}: order needs at least two bot jobs; unknown: {unknown}")
    return dial


def _band(table: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read a [[diagnose]]'s band: the win-rate range an answer looks for.

    Example: band [0.10, 0.40] with band_min_runs 2 means "found" when at
    least 2 models win between 10% and 40% against the same opponent.
    """

    where = f"[[diagnose]] {table.get('name')}"
    band = table["band"]
    # Two numbers (True and False count as numbers in Python, so they are
    # refused by name), each a share between 0 and 1, low below high.
    if (
        not isinstance(band, list) or len(band) != 2
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in band)
        or not 0 <= band[0] < band[1] <= 1
    ):
        raise PlanError(f"{where}: band must be [low, high] with 0 <= low < high <= 1")
    return {"band": [float(band[0]), float(band[1])], "band_min_runs": _int(table, "band_min_runs", where, default=2)}


# Small readers that name the field and section in every error. Each takes
# the table, the field's name, and ``where`` (the section, for the message),
# and returns the value only if it has the right type.
# Example error: "[[botcheck]] opponent must be a non-empty string".


def _table(raw: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    """A sub-table, such as [session] or maps = { ... }."""

    value = raw.get(key)
    if not isinstance(value, dict):
        raise PlanError(f"{where} needs a [{key}] table")
    return value


def _string(raw: Mapping[str, Any], key: str, where: str) -> str:
    """Text that isn't empty or only spaces."""

    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{where} {key} must be a non-empty string")
    return value


def _label(raw: Mapping[str, Any], key: str, where: str) -> str:
    """Text that is safe as a folder name (see LABEL_PATTERN)."""

    value = _string(raw, key, where)
    if not LABEL_PATTERN.match(value):
        raise PlanError(f"{where} {key} {value!r} must use only letters, digits, '.', '_' and '-'")
    return value


def _int(raw: Mapping[str, Any], key: str, where: str, default: int | None = None) -> int:
    """A whole number. With no default, the field is required."""

    value = raw.get(key, default)
    # In Python, True and False are also integers (1 and 0), so they are
    # refused by name: "count = true" is a mistake, not 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{where} {key} must be an integer")
    return value


def _choice(raw: Mapping[str, Any], key: str, where: str, choices: Sequence[str]) -> str:
    """One of a few fixed words, such as "cuda" or "cpu"."""

    value = raw.get(key)
    if value not in choices:
        raise PlanError(f"{where} {key} must be one of {list(choices)}, got {value!r}")
    return value


def _strings(raw: Mapping[str, Any], key: str, where: str, default: list[str] | None = None) -> list[str]:
    """A list of non-empty texts. With no default it is required and must
    not be empty; with a default (such as []) it may be left out."""

    value = raw.get(key, default)
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise PlanError(f"{where} {key} must be a list of non-empty strings")
    if default is None and not value:
        raise PlanError(f"{where} {key} must not be empty")
    return value


def _ints(raw: Mapping[str, Any], key: str, where: str) -> list[int]:
    """A required, non-empty list of whole numbers."""

    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        raise PlanError(f"{where} {key} must be a non-empty list of integers")
    return value


def _named_specs(raw: Mapping[str, Any], key: str, where: str) -> dict[str, str]:
    """A table of ``label = "spec"``. The label names the job's folder, so a
    long spec never has to be one.

    Example, in the file::

        [botcheck.testers]
        normal = "heuristic-v2:candidates=4,enemy_bonus=1,margin=0.1"

    gives {"normal": "heuristic-v2:..."}. TOML keeps the file's order, so
    jobs come out in the order they were written.
    """

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
#
# Everything that can be checked before a long session starts is checked
# here, and every problem is collected (not just the first), so one preflight
# shows everything to fix.
# ---------------------------------------------------------------------------


def repository_path(path: str) -> Path:
    """A path from the experiment file, which is written from the repository root.

    Example: "configs/rl-v2-4p.toml" becomes <repository>/configs/rl-v2-4p.toml,
    whatever folder the command runs from. An absolute path is kept as it is.
    """

    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def checkpoint_path(run_dir: str, update: int) -> Path:
    """Where training saves the checkpoint for one update. Example: update
    153 of runs/x is runs/x/checkpoints/update-000153.pt."""

    return repository_path(run_dir) / "checkpoints" / f"update-{update:06d}.pt"


def training_config(job: Job, session: Session) -> dict[str, Any]:
    """The full training config a train job runs: its base config with this
    job's opponent, checkpoint cadence and the session's device.

    Everything else (learning rate, seeds, game counts, ...) comes from the
    base config unchanged. The job writes the result into its folder as
    config.toml, so exactly what trained is kept with its results.
    """

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
    reuses this build from Go's cache. Returns opponentcheck's path.

    ``runner`` is the function that runs a command; tests pass a fake one so
    they need no Go.
    """

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

    ``opponentcheck -canonical`` prints one canonical form per line, in the
    order the specs were given, so the i-th line belongs to the i-th spec.
    Example: given "heuristic-v2:margin=0.1,candidates=4,enemy_bonus=1" it
    prints "heuristic-v2:candidates=4,enemy_bonus=1,margin=0.1"; they differ,
    so that spec is reported with the spelling to use instead.
    """

    if not specs:
        return []
    result = runner([str(tool), "-canonical", *specs], capture_output=True, text=True)
    if result.returncode != 0:
        # A spec Go can't read at all (a misspelled setting, say). Go names
        # every bad one in its error, so pass that on.
        return [f"opponentcheck refused a spec:\n{result.stderr.strip()}"]
    forms = result.stdout.splitlines()
    # strict=True makes zip fail loudly if the counts ever differ, rather
    # than quietly pairing a spec with the wrong form.
    return [
        f"spec is not canonical: {spec!r}\n    write it as: {form!r}"
        for spec, form in zip(specs, forms, strict=True)
        if spec != form
    ]


def file_problems(plan: Plan, jobs: Sequence[Job]) -> list[str]:
    """Every missing checkpoint or run, and every train config that won't load.

    Each train job's config is built and checked by the same code training
    uses (run_config_from_mapping), so a bad setting is found now, not when
    the job's turn comes hours later.
    """

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
    """What the run will do, in a few lines: jobs, games and updates.

    Every job plays each map from all 4 seats, so a job plays
    4 × map_count games. A train job also diagnoses each of its checkpoints
    against each diagnose_against opponent, and each diagnosis plays that
    many games too.
    """

    games_per_job = 4 * plan.session.map_count
    lines = [f"session {plan.session.name}: {len(jobs)} jobs, {plan.session.map_count} maps x 4 seats"]
    # dict.fromkeys keeps the first time each name appears and drops
    # repeats, in order: a simple way to list each experiment once.
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
    """Run every check. Returns the problems found and opponentcheck's path.

    The order matters only for speed: file checks need nothing built; the
    spec check needs opponentcheck, so it runs after the build. If the build
    fails there is no tool to check specs with, so that check is skipped.
    """

    problems = file_problems(plan, jobs)
    try:
        tool = build_go_tools(bin_dir)
    except PlanError as error:
        return [*problems, str(error)], None
    problems += canonical_problems(specs_of(jobs), tool)
    return problems, tool


# ---------------------------------------------------------------------------
# Stages 2 and 3: run each job in its lane and record it.
#
# Each job runs other programs (opponentcheck, training, diagnose) as
# separate processes, waits for them, and records the outcome in its folder.
# ---------------------------------------------------------------------------


def now() -> str:
    """The current time in UTC, to the second, as text. Example:
    "2026-11-26T02:33:02+00:00". UTC keeps times comparable between a
    laptop and a cloud machine in another time zone."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    """Write JSON to a temporary file, then rename it into place, so a reader
    never sees half a file.

    A rename within one folder happens all at once: anyone reading the file
    sees either the old version or the new one, never a half-written mix.
    That matters for done.json, which promises "this job finished", and for
    session.json, which both lanes update while the other may be reading.
    sort_keys gives the same key order every time, so two files are easy to
    compare.
    """

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def toml_text(values: Mapping[str, Any]) -> str:
    """A flat TOML table for a training config: strings, numbers, booleans and
    lists of them. Strings are written as JSON strings, which TOML reads the
    same way.

    Python can read TOML but has no built-in way to write it, and a training
    config needs only these few kinds of value, so this small writer avoids
    a new dependency. Example: {"players": 4, "device": "cuda",
    "training_seeds": [0, 100]} becomes::

        players = 4
        device = "cuda"
        training_seeds = [0, 100]
    """

    def value(item: Any) -> str:
        # bool is checked before int, because in Python True is also an int.
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, (int, float)):
            # repr writes a float with every digit it needs (0.1, not
            # 0.10000000000000001), so the config reads back as the same value.
            return repr(item)
        if isinstance(item, str):
            return json.dumps(item)
        if isinstance(item, list):
            return "[" + ", ".join(value(element) for element in item) + "]"
        raise TypeError(f"can't write {item!r} as TOML")

    return "".join(f"{key} = {value(item)}\n" for key, item in values.items())


class Runner:
    """Runs jobs and keeps track of the processes they start, so an
    interrupted session can stop them all.

    Both lanes share one Runner, each from its own thread (a thread is a
    second line of work inside the same program, running at the same time).
    The lock lets only one thread at a time change the shared set of
    processes and the stopping flag, so they can't trip over each other.
    """

    def __init__(self, plan: Plan, out: Path, tool: Path) -> None:
        self.plan = plan
        # The session folder.
        self.out = out
        # The opponentcheck binary built by preflight.
        self.tool = tool
        self._lock = threading.Lock()
        # Every process started and not yet finished, so stop() can end them.
        self._processes: set[subprocess.Popen[bytes]] = set()
        # Set by stop() after Ctrl-C; from then on no new command starts.
        self._stopping = False

    def stop(self) -> None:
        """Stop every running command; no new one starts afterwards.

        terminate() asks each process to end (the SIGTERM signal). The
        process ends, its command() call returns, and its job is left
        unfinished for the next run.
        """

        with self._lock:
            self._stopping = True
            for process in self._processes:
                process.terminate()

    @property
    def stopping(self) -> bool:
        """Whether stop() has been called."""

        with self._lock:
            return self._stopping

    def command(self, arguments: Sequence[str], log: Path, *, cwd: Path, gomaxprocs: int) -> int:
        """Run one command, appending its output to the job's log.

        Returns the command's exit code: 0 means it succeeded, anything else
        that it failed. -1 means it never started because the session is
        stopping. A negative code from a started process means it was ended
        by a signal (for example -15 after stop()).
        """

        # The child gets a copy of this program's environment variables, with
        # its own GOMAXPROCS (Go's thread limit) for this job.
        env = {**os.environ, "GOMAXPROCS": str(gomaxprocs)}
        # "ab" appends in binary, so the log keeps every earlier command's
        # output, and the child's bytes are written exactly as it printed them.
        with log.open("ab") as handle:
            handle.write(f"$ {' '.join(arguments)}\n".encode())
            handle.flush()
            # Start the process and register it in one locked step, so stop()
            # can't run in between and miss it.
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
        """Run one job in its folder and return "done", "failed" or
        "unfinished".

        The stages are: set aside any unfinished folder; create a fresh one
        and record what it runs; run the job's commands; write done.json last,
        or failed.json. A job that raises is recorded as failed and never
        stops the other jobs.
        """

        folder = self.out / job.experiment / job.label
        # Stage 1: a finished job is never run again. Any other folder is
        # left from an interrupted or failed try: rename it out of the way
        # (kept, never deleted), so the new try starts from an empty folder
        # and its files can't mix with the old ones.
        if (folder / "done.json").exists():
            return "done"
        if folder.exists():
            set_aside(folder)
        # Stage 2: a fresh folder, and job.json first, so even a job that
        # dies at once says what it was trying to run. monotonic() is a clock
        # that only moves forward, so the duration is right even if the
        # computer's clock is changed during the job.
        folder.mkdir(parents=True)
        started = time.monotonic()
        record = {"job": job.id, "kind": job.kind, "settings": dict(job.settings), "started": now()}
        write_json(folder / "job.json", record)
        log = folder / "log.txt"
        # Stage 3: run the commands. Any error is caught and recorded, so one
        # broken job can't crash the lane and stop the jobs after it.
        try:
            failed = self._run(job, folder, log)
        except Exception as error:  # recorded, so one broken job can't stop the session
            failed = f"{type(error).__name__}: {error}"
        record.update(finished=now(), seconds=round(time.monotonic() - started, 1))
        # Stage 4: record the outcome. done.json is written last of all, so
        # its presence promises everything before it finished.
        if failed and self.stopping:
            # Stopped by Ctrl-C, not failed: leave the folder without a result,
            # so status shows it unfinished and the next run sets it aside and
            # runs the job again.
            return "unfinished"
        if failed:
            write_json(folder / "failed.json", {**record, "error": failed, "log": "log.txt"})
            return "failed"
        write_json(folder / "done.json", record)
        return "done"

    def _run(self, job: Job, folder: Path, log: Path) -> str | None:
        """Run a job's commands; return why it failed, or None if it succeeded."""

        session = self.plan.session
        # The diagnose options that choose the session's maps.
        maps = ["--seeds", str(session.map_count), "--seed-start", str(session.map_start)]

        # A bot job: one opponentcheck run, the tester in the tested seat and
        # the opponent in the other three, on the session's maps.
        if job.kind == "botcheck":
            arguments = [
                str(self.tool), "-tester-spec", job.settings["tester"], "-opponent", job.settings["opponent"],
                "-seed-start", str(session.map_start), "-seeds", str(session.map_count),
                "-workers", str(session.cpu_workers), "-out", str(folder / "opponentcheck"),
            ]
            if job.settings["check_decode"]:
                # Inserted before the last two items, "-out" and its folder.
                arguments.insert(-2, "-check-decode")
            code = self.command(arguments, log, cwd=REPOSITORY_ROOT, gomaxprocs=session.cpu_workers)
            return None if code == 0 else f"opponentcheck exited with {code}"

        # A diagnose job: one checkpoint plays the session's maps against
        # one opponent.
        if job.kind == "diagnose":
            run_dir = repository_path(job.settings["run_dir"])
            return self._diagnose(run_dir, checkpoint_path(job.settings["run_dir"], job.settings["update"]),
                                  job.settings["opponent"], folder / "diagnosis", maps, log)

        # A train job: write its own full config, train, then diagnose every
        # checkpoint the run saved against every diagnose_against opponent.
        config = folder / "config.toml"
        config.write_text(toml_text(training_config(job, session)))
        run_dir = folder / "run"
        # sys.executable is the Python running this script, so training uses
        # the same Python and installed packages.
        code = self.command([
            sys.executable, "-m", "orrery.train", "--config", str(config), "--run-dir", str(run_dir),
            "--init-from", str(repository_path(job.settings["init_from"])),
            "--max-updates", str(job.settings["max_updates"]),
        ], log, cwd=PYTHON_ROOT, gomaxprocs=session.gomaxprocs)
        if code != 0:
            return f"training exited with {code}"
        # The checkpoint files actually saved, not the ones expected: names
        # are zero-padded (update-000030.pt), so sorting them as text puts
        # them in update order.
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
        """Run diagnose.py on one checkpoint; return why it failed, or None.

        --no-charts skips the pictures: the analysis reads only report.json,
        and the charts would need the optional plotting packages.
        """

        code = self.command([
            sys.executable, "-m", "scripts.diagnose", "--run-dir", str(run_dir), "--checkpoint", str(checkpoint),
            "--opponent", opponent, *maps, "--device", self.plan.session.device, "--no-charts",
            "--out", str(out),
        ], log, cwd=PYTHON_ROOT, gomaxprocs=self.plan.session.gomaxprocs)
        return None if code == 0 else f"diagnose of {checkpoint.name} against {opponent} exited with {code}"


def set_aside(folder: Path) -> Path:
    """Rename an unfinished job folder to ``<name>.partial-<n>``, the first
    unused n, so its files are kept for inspection but never mixed with the
    rerun's.

    Example: if "aim.partial-1" already exists, "aim" becomes "aim.partial-2".
    """

    n = 1
    # := assigns and tests in one step: try partial-1, partial-2, ... until
    # one doesn't exist yet.
    while (target := folder.with_name(f"{folder.name}.partial-{n}")).exists():
        n += 1
    folder.rename(target)
    return target


# ---------------------------------------------------------------------------
# Stage 4 and the session record: session.json, status and the two lanes.
# ---------------------------------------------------------------------------


def job_state(out: Path, job: Job) -> str:
    """done, failed, unfinished (a folder without a result: running now, or
    interrupted), or waiting (not started).

    The state is read from the job's folder, not from session.json, so it is
    right even after a crash that left session.json out of date.
    """

    folder = out / job.experiment / job.label
    if (folder / "done.json").exists():
        return "done"
    if (folder / "failed.json").exists():
        return "failed"
    return "unfinished" if folder.exists() else "waiting"


def versions() -> dict[str, Any]:
    """The code and tools a session ran with, for replay.

    git_dirty is True when tracked files have changes not yet committed:
    then the commit alone doesn't say exactly what code ran. A value that
    can't be found (no git, say) is recorded as None, never guessed.
    """

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

    Returns 0 when every selected job is done, 1 when any failed or the run
    was interrupted, and 2 when preflight found problems or the session
    folder holds a different experiment file.

    The stages are: read the plan and choose the jobs; check the session
    folder belongs to this file; preflight; record the run in session.json;
    start the two lanes and wait for both; record every job's final state.
    """

    # Stage 1: the plan, and the jobs this run covers (--only picks one
    # experiment by name).
    text = experiment.read_text()
    plan = parse_plan(text)
    jobs = [job for job in plan.jobs if only is None or job.experiment == only]
    if not jobs:
        print(f"no experiment named {only!r}; experiments: {sorted({j.experiment for j in plan.jobs})}")
        return 2

    # Stage 2: a session folder belongs to one experiment file. Resuming with
    # an edited file would mix results made under different settings, so the
    # file's text is compared with the copy saved on the first run.
    out.mkdir(parents=True, exist_ok=True)
    session_file = out / "session.json"
    record: dict[str, Any] = json.loads(session_file.read_text()) if session_file.exists() else {}
    if record and record["experiment_text"] != text:
        print(f"{out} holds a session of a different experiment file; use a new --out")
        return 2

    # Stage 3: preflight, with the Go tools built into the session folder.
    problems, tool = preflight(plan, jobs, out / "bin")
    for line in describe(plan, jobs):
        print(line)
    if problems:
        print("preflight found problems; nothing was run:")
        for problem in problems:
            print(f"- {problem}")
        return 2
    assert tool is not None

    # Stage 4: record this run. setdefault keeps what an earlier run already
    # saved; "runs" gets one entry per run, so a resumed session shows each
    # attempt with its own code version and times.
    record.setdefault("experiment_file", str(experiment))
    record.setdefault("experiment_text", text)
    record.setdefault("runs", []).append({"started": now(), "only": only, **versions()})
    # Both lanes update the one record; the lock makes each save whole.
    lock = threading.Lock()

    def save(job: Job | None = None, state: str | None = None) -> None:
        """Set one job's state (if given) and write session.json."""

        with lock:
            if job is not None:
                record.setdefault("jobs", {})[job.id] = state
            write_json(session_file, record)

    for job in jobs:
        record.setdefault("jobs", {})[job.id] = job_state(out, job)
    save()

    # Stage 5: the two lanes.
    runner = Runner(plan, out, tool)
    # Each lane sets its event when it has finished, including after Ctrl-C.
    # The main thread waits on these events rather than Thread.join: in
    # CPython a join interrupted by Ctrl-C can make a later join return
    # before the thread has really ended, and the session would then be
    # recorded while a lane is still writing its last job. (An Event is a
    # flag one thread sets and another waits for.)
    finished = {name: threading.Event() for name in (GPU_LANE, CPU_LANE)}

    def lane(name: str) -> None:
        """Run this lane's jobs one after another, skipping finished ones."""

        try:
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
        finally:
            # finally runs however the loop ends (finished, returned early, or
            # an error), so the main thread is never left waiting forever.
            finished[name].set()

    # daemon=True: if the main thread ends, these threads don't keep the
    # program alive on their own.
    for name in finished:
        threading.Thread(target=lane, args=(name,), daemon=True).start()
    try:
        # Wait in one-second steps rather than all at once, so Ctrl-C (which
        # Python delivers to the main thread as KeyboardInterrupt) is noticed
        # within a second.
        for event in finished.values():
            while not event.wait(timeout=1):
                pass
    except KeyboardInterrupt:
        # Ctrl-C: stop the running commands, wait for both lanes to finish
        # their current job (each now ends quickly), then record the states.
        print("interrupted: stopping running jobs; run the same command again to resume", flush=True)
        runner.stop()
        for event in finished.values():
            event.wait()
        for job in jobs:
            record["jobs"][job.id] = job_state(out, job)
        record["runs"][-1]["finished"] = now()
        save()
        return 1

    # Stage 6: every job's final state, read from its folder, and a summary.
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
    """Print each job's state from the session folder.

    The plan is rebuilt from the experiment file's text saved in
    session.json, so status works even if the file was moved or edited.
    Example output line: "done       restricted-bot/normal".
    """

    session_file = out / "session.json"
    if not session_file.exists():
        print(f"{out} has no session.json; no session has run there")
        return 2
    plan = parse_plan(json.loads(session_file.read_text())["experiment_text"])
    counts: dict[str, int] = {}
    for job in plan.jobs:
        state = job_state(out, job)
        counts[state] = counts.get(state, 0) + 1
        # {state:10} pads the state to 10 characters, so the ids line up.
        print(f"{state:10} {job.id}")
    print(", ".join(f"{count} {state}" for state, count in sorted(counts.items())))
    return 0


def run_preflight(experiment: Path, only: str | None = None) -> int:
    """Stage 1 only: check and count, run nothing, write nothing kept.

    The Go tools are built into a temporary folder that is deleted when the
    check ends, so a preflight leaves no files behind.
    """

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
    """The command line: three subcommands, preflight, run and status.

    Exit codes: 0 all good; 1 a job failed or the run was interrupted; 2 the
    file or folder can't be used as given (nothing was run).
    """

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
        # resolve() turns each path into a full one, so the jobs, which run
        # from other folders, still find it.
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
