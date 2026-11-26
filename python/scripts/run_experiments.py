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
- ``[[train]]``: diagnose the starting checkpoint, train from it against an
  opponent, then diagnose every saved checkpoint, one job per (training
  opponent, starting checkpoint). The analysis compares the final
  checkpoint's wins with the starting checkpoint's.
- ``[[compare]]`` and ``[[dial]]``: no jobs; they tell analyze_experiments.py
  what to compare.

A run has four stages:

1. **Preflight.** Expand every job, check that every path exists and every
   agent spec is canonical, build the Go tools, and print what will run. Any
   mistake stops here, in seconds, before anything runs.
2. **Run.** One job at a time: every bot job first (they are quick, so
   experiments 1 and 2 have answers early), then the diagnose and train jobs,
   each group in the order the file lists them.
3. **Record.** A job writes ``done.json`` last, so a folder with it is a
   finished job that can be trusted. ``session.json`` holds the experiment
   file, the code version and every job's state.
4. **Resume.** Running the same command again skips finished jobs. A folder
   without ``done.json`` (interrupted or failed) is renamed
   ``<name>.partial-<n>``, never deleted, and the job runs again. A train
   job whose training finished is the exception: its folder is kept, and
   the retry redoes only the diagnoses that didn't finish, each earlier
   attempt moved to ``diagnoses/.partial/<name>/attempt-<n>``. A resume
   refuses to run if the code or any input file changed since the first
   run, so after changing either, start a new session with a new ``--out``.

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
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
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

# The file a train job writes in its folder once training has finished
# successfully. A retry of that job keeps the trained run and redoes only
# what didn't finish (see keep_finished_training).
TRAINING_DONE = "training-done.json"

# The order kinds of job run in. Jobs run one at a time, never at once: each
# job already keeps the machine busy on its own (a bot job plays many games
# at once; training steps hundreds of games and uses the GPU), so running two
# together would mostly make them share the same CPUs and GPU. Bot jobs go
# first because they are the quickest: their answers arrive minutes into a
# session instead of after hours of training.
KIND_ORDER = ("botcheck", "diagnose", "train")


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


# frozen=True makes these objects read-only once built, so a job's settings
# can't be changed by accident after the file was checked.
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
    # reads its thread limit from the GOMAXPROCS environment variable. More
    # threads isn't always faster for the environment (the Makefile's train
    # target notes the measurements), so model jobs get this setting, while
    # bot jobs get cpu_workers instead.
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


def run_order(jobs: Sequence[Job]) -> list[Job]:
    """The jobs in the order they run: by kind (KIND_ORDER), and within a
    kind in file order.

    Example: a file listing a train section, then a bot section, runs the bot
    jobs first. sorted() keeps equal items in their original order, so jobs
    of one kind stay in the order the file lists them.
    """

    return sorted(jobs, key=lambda job: KIND_ORDER.index(job.kind))


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


def unique_specs(jobs: Sequence[Job]) -> list[str]:
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


def saved_checkpoint_count(job: Job) -> int:
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
    session = _read_session(_read_table(raw, "session", "the file"))
    jobs: list[Job] = []
    answers: dict[str, Mapping[str, Any]] = {}
    for table in raw.get("botcheck", []):
        jobs.extend(_expand_botcheck(table))
    for table in raw.get("diagnose", []):
        jobs.extend(_expand_diagnose(table))
        if "band" in table:
            answers[table["name"]] = _read_band(table)
    for table in raw.get("train", []):
        jobs.extend(_expand_train(table))
        answers[table["name"]] = {
            "progress_min_runs": _read_int(table, "progress_min_runs", "[[train]]", default=2),
            # How many more wins than its starting checkpoint a run's final
            # checkpoint needs to count as progress. A fair threshold
            # depends on how many games a diagnosis plays, so each file
            # states its own. Reading accepts it missing (as None), because
            # status and the analysis also read sessions saved before the
            # setting existed; preflight still refuses a new run without it
            # (check_required_settings).
            "progress_min_extra_wins": (
                _read_int(table, "progress_min_extra_wins", "[[train]]")
                if "progress_min_extra_wins" in table else None
            ),
            "progress_against": _read_progress_against(table),
        }
        # Both thresholds must be at least 1. With 0 extra wins, a run whose
        # final checkpoint did no better than its start would count as
        # progress; with 0 runs, progress would need no run at all.
        for key in ("progress_min_runs", "progress_min_extra_wins"):
            value = answers[table["name"]][key]
            if value is not None and value < 1:
                raise PlanError(f"[[train]] {table['name']}: {key} must be at least 1, got {value}")

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
    compares = tuple(_read_compare(table, bot_ids) for table in raw.get("compare", []))
    dials = tuple(_read_dial(table, bot_ids) for table in raw.get("dial", []))
    return Plan(session, tuple(jobs), compares, dials, answers)


def _read_session(table: Mapping[str, Any]) -> Session:
    """Read and check the [session] table."""

    where = "[session]"
    maps = _read_table(table, "maps", where)
    session = Session(
        name=_read_label(table, "name", where),
        map_start=_read_int(maps, "start", f"{where} maps"),
        map_count=_read_int(maps, "count", f"{where} maps"),
        device=_read_choice(table, "device", where, ("cuda", "cpu")),
        cpu_workers=_read_int(table, "cpu_workers", where),
        gomaxprocs=_read_int(table, "gomaxprocs", where, default=12),
    )
    # Zero maps or zero threads would run nothing, or hang; refuse them.
    for name in ("map_count", "cpu_workers", "gomaxprocs"):
        if getattr(session, name) < 1:
            raise PlanError(f"{where} {name} must be at least 1")
    return session


def _expand_botcheck(table: Mapping[str, Any]) -> list[Job]:
    """One job per tester, each against the same opponent.

    Example: testers ``normal`` and ``aim`` in an experiment named ``bots``
    give jobs ``bots/normal`` and ``bots/aim``.
    """

    where = "[[botcheck]]"
    name = _read_label(table, "name", where)
    opponent = _read_string(table, "opponent", where)
    testers = _read_named_specs(table, "testers", where)
    # check_decode lists the tester labels whose launches are checked against
    # the learner's decoder (opponentcheck -check-decode). A label that isn't
    # a tester is a typo, and the check would silently not run.
    check_decode = _read_strings(table, "check_decode", where, default=[])
    unknown = sorted(set(check_decode) - set(testers))
    if unknown:
        raise PlanError(f"{where} {name}: check_decode names testers it doesn't have: {unknown}")
    return [
        Job(name, "botcheck", label, {
            "tester": spec, "opponent": opponent, "check_decode": label in check_decode,
        })
        for label, spec in testers.items()
    ]


def _expand_diagnose(table: Mapping[str, Any]) -> list[Job]:
    """One job per (opponent, run, update), named after all three.

    Example: opponent label ``v2-eps0.5``, run folder
    ``runs/vast-experiment/exp-b-4p-s1501`` and update 153 give the job
    ``probe/v2-eps0.5__exp-b-4p-s1501__u000153``. The double underscore
    separates the parts, since a single one may appear inside a name. The
    update is padded to six digits, as checkpoint files are, so folders sort
    in update order.
    """

    where = "[[diagnose]]"
    name = _read_label(table, "name", where)
    run_dirs = _read_strings(table, "run_dirs", where)
    updates = _read_ints(table, "updates", where)
    opponents = _read_named_specs(table, "opponents", where)
    # Three nested loops: every opponent with every run with every update.
    return [
        Job(name, "diagnose", f"{label}__{Path(run_dir).name}__u{update:06d}", {
            "run_dir": run_dir, "update": update, "opponent": spec, "opponent_label": label,
        })
        for label, spec in opponents.items()
        for run_dir in run_dirs
        for update in updates
    ]


def _expand_train(table: Mapping[str, Any]) -> list[Job]:
    """One job per (training opponent, starting checkpoint).

    A starting checkpoint is named after its run: ``runs/x/checkpoints/u.pt``
    gives ``x``, which keeps the training seed in the folder name. Example:
    training opponent ``v2-eps0.75`` from
    ``runs/vast-experiment/exp-b-4p-s1501/checkpoints/update-000153.pt``
    gives ``train-weakened/v2-eps0.75__exp-b-4p-s1501``.
    """

    where = "[[train]]"
    name = _read_label(table, "name", where)
    starts = _read_strings(table, "init_from", where)
    opponents = _read_named_specs(table, "training_opponents", where)
    against = _read_named_specs(table, "diagnose_against", where)
    # Settings every run of this section shares.
    common = {
        "base_config": _read_string(table, "base_config", where),
        "max_updates": _read_int(table, "max_updates", where),
        "checkpoint_every_updates": _read_int(table, "checkpoint_every_updates", where),
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


def _read_progress_against(table: Mapping[str, Any]) -> str | None:
    """Which diagnose_against label the progress answer measures against.

    The answer compares the final checkpoint with the start against one
    opponent, so it must be the intended one, named by its label. With a
    single diagnose_against label there is no choice, and progress_against
    may be left out. With several, it must be given: guessing the first one
    could judge progress against a weaker opponent than full v2.

    None means "several labels and no choice made". Reading accepts that,
    so sessions saved before this setting stay readable; preflight refuses
    a new run with it (check_required_settings).

    Example: diagnose_against = { v2 = "...", weak = "..." } with
    progress_against = "v2" gives "v2"; with only { v2 = "..." }, leaving it
    out also gives "v2".
    """

    labels = list(table["diagnose_against"])
    against = table.get("progress_against")
    if against is None:
        return labels[0] if len(labels) == 1 else None
    if against not in labels:
        raise PlanError(f"[[train]] {table['name']}: progress_against {against!r} is not one of the "
                        f"diagnose_against labels {labels}")
    return against


def _read_compare(table: Mapping[str, Any], bot_ids: set[str]) -> Mapping[str, Any]:
    """Read one [[compare]]: candidate minus baseline, both bot job ids.

    ``rule``, when present, names the answer the analysis gives for this
    comparison; "restricted-bot" is experiment 1's.
    """

    where = "[[compare]]"
    compare = {
        "baseline": _read_string(table, "baseline", where),
        "candidate": _read_string(table, "candidate", where),
        "question": _read_string(table, "question", where),
        "rule": table.get("rule"),
    }
    for key in ("baseline", "candidate"):
        if compare[key] not in bot_ids:
            raise PlanError(f"{where} {key} {compare[key]!r} is not a bot job; bot jobs: {sorted(bot_ids)}")
    if compare["rule"] not in (None, "restricted-bot"):
        raise PlanError(f"{where} rule must be 'restricted-bot' or left out, got {compare['rule']!r}")
    return compare


def _read_dial(table: Mapping[str, Any], bot_ids: set[str]) -> Mapping[str, Any]:
    """Read one [[dial]]: bot jobs in the order their win rate should fall.

    A "dial" is a setting meant to make a bot weaker step by step, like
    turning a knob; experiment 2 checks epsilon behaves like one. It needs at
    least two steps to have anything to compare.
    """

    where = "[[dial]]"
    dial = {"name": _read_label(table, "name", where), "order": _read_strings(table, "order", where)}
    unknown = [job_id for job_id in dial["order"] if job_id not in bot_ids]
    if unknown or len(dial["order"]) < 2:
        raise PlanError(f"{where} {dial['name']}: order needs at least two bot jobs; unknown: {unknown}")
    return dial


def _read_band(table: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read a [[diagnose]]'s band: the win-rate range an answer looks for.

    Example: band [0.10, 0.40] with band_min_runs 2 means "found" when the
    models of at least 2 runs, at the same update, win between 10% and 40%
    against the same opponent. band_min_runs must be at least 1.
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
    min_runs = _read_int(table, "band_min_runs", where, default=2)
    # With 0, "found" would need no run inside the band at all.
    if min_runs < 1:
        raise PlanError(f"{where}: band_min_runs must be at least 1, got {min_runs}")
    return {"band": [float(band[0]), float(band[1])], "band_min_runs": min_runs}


# Small readers that name the field and section in every error. Each takes
# the table, the field's name, and ``where`` (the section, for the message),
# and returns the value only if it has the right type.
# Example error: "[[botcheck]] opponent must be a non-empty string".


def _read_table(raw: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    """A sub-table, such as [session] or maps = { ... }."""

    value = raw.get(key)
    if not isinstance(value, dict):
        raise PlanError(f"{where} needs a [{key}] table")
    return value


def _read_string(raw: Mapping[str, Any], key: str, where: str) -> str:
    """Text that isn't empty or only spaces."""

    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{where} {key} must be a non-empty string")
    return value


def _read_label(raw: Mapping[str, Any], key: str, where: str) -> str:
    """Text that is safe as a folder name (see LABEL_PATTERN)."""

    value = _read_string(raw, key, where)
    if not LABEL_PATTERN.match(value):
        raise PlanError(f"{where} {key} {value!r} must use only letters, digits, '.', '_' and '-'")
    return value


def _read_int(raw: Mapping[str, Any], key: str, where: str, default: int | None = None) -> int:
    """A whole number. With no default, the field is required."""

    value = raw.get(key, default)
    # In Python, True and False are also integers (1 and 0), so they are
    # refused by name: "count = true" is a mistake, not 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{where} {key} must be an integer")
    return value


def _read_choice(raw: Mapping[str, Any], key: str, where: str, choices: Sequence[str]) -> str:
    """One of a few fixed words, such as "cuda" or "cpu"."""

    value = raw.get(key)
    if value not in choices:
        raise PlanError(f"{where} {key} must be one of {list(choices)}, got {value!r}")
    return value


def _read_strings(raw: Mapping[str, Any], key: str, where: str, default: list[str] | None = None) -> list[str]:
    """A list of non-empty texts. With no default it is required and must
    not be empty; with a default (such as []) it may be left out."""

    value = raw.get(key, default)
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise PlanError(f"{where} {key} must be a list of non-empty strings")
    if default is None and not value:
        raise PlanError(f"{where} {key} must not be empty")
    return value


def _read_ints(raw: Mapping[str, Any], key: str, where: str) -> list[int]:
    """A required, non-empty list of whole numbers."""

    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        raise PlanError(f"{where} {key} must be a non-empty list of integers")
    return value


def _read_named_specs(raw: Mapping[str, Any], key: str, where: str) -> dict[str, str]:
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


def resolve_repository_path(path: str) -> Path:
    """A path from the experiment file, which is written from the repository root.

    Example: "configs/rl-v2-4p.toml" becomes <repository>/configs/rl-v2-4p.toml,
    whatever folder the command runs from. An absolute path is kept as it is.
    """

    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def relative_checkpoint_path(run_dir: str, update: int) -> str:
    """The checkpoint for one update, as a path from the repository root, the
    way the experiment file writes paths. Example: update 153 of runs/x is
    runs/x/checkpoints/update-000153.pt."""

    return (Path(run_dir) / "checkpoints" / f"update-{update:06d}.pt").as_posix()


def full_checkpoint_path(run_dir: str, update: int) -> Path:
    """Where training saved the checkpoint for one update, as a full path."""

    return resolve_repository_path(relative_checkpoint_path(run_dir, update))


def starting_run_dir(init_from: str) -> str:
    """The folder of the run that saved a starting checkpoint, as a path from
    the repository root. Example: runs/x/checkpoints/update-000153.pt gives
    runs/x. Diagnosing the checkpoint needs files from that folder."""

    return Path(init_from).parent.parent.as_posix()


def files_read_by_diagnosis(run_dir: str, checkpoint: str) -> list[str]:
    """Every file that diagnosing one checkpoint reads, as paths from the
    repository root.

    Preflight checks that each one exists, and the session fingerprint
    hashes each one. Both use this list, so they always cover the same
    files. The files are:
    - the checkpoint, and the two files training writes beside it: .sha256
      (its recorded hash, which must match the run's manifest) and
      .metadata.json (the description checked before the checkpoint loads);
    - from its run, run-config.json and manifest.json (the checkpoint is
      checked against both) and metrics.csv (the run's training history).

    Example: runs/x and runs/x/checkpoints/update-000153.pt give that .pt,
    its .pt.sha256 and .pt.metadata.json, and runs/x/run-config.json,
    runs/x/manifest.json and runs/x/metrics.csv.
    """

    return [
        checkpoint, f"{checkpoint}.sha256", f"{checkpoint}.metadata.json",
        *((Path(run_dir) / name).as_posix() for name in ("run-config.json", "manifest.json", "metrics.csv")),
    ]


def find_missing_files(names: Sequence[str]) -> list[str]:
    """The files in ``names`` that don't exist. Each name is a path from the
    repository root, such as runs/x/manifest.json."""

    return [name for name in names if not resolve_repository_path(name).is_file()]


def build_training_config(job: Job, session: Session) -> dict[str, Any]:
    """The full training config a train job runs: its base config with this
    job's opponent, checkpoint cadence and the session's device.

    Everything else (learning rate, seeds, game counts, ...) comes from the
    base config unchanged. The job writes the result into its folder as
    config.toml, so exactly what trained is kept with its results.
    """

    with resolve_repository_path(job.settings["base_config"]).open("rb") as source:
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


def check_specs_canonical(specs: Sequence[str], tool: Path,
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


def check_input_files(plan: Plan, jobs: Sequence[Job]) -> list[str]:
    """Every missing input file, and every train config that won't load.

    A diagnose job needs every file that diagnosing its checkpoint reads.
    A train job needs the same files for its starting checkpoint, because
    it diagnoses that first (see files_read_by_diagnosis). Each job's missing
    files are listed together on one line.
    Each train job's config is built and checked by the same code training
    uses (run_config_from_mapping), so a bad setting is found now, not when
    the job's turn comes hours later.
    """

    problems = []
    for job in jobs:
        if job.kind == "diagnose":
            run_dir = job.settings["run_dir"]
            missing = find_missing_files(files_read_by_diagnosis(run_dir, relative_checkpoint_path(run_dir, job.settings["update"])))
            if missing:
                problems.append(f"{job.id}: these files do not exist: {', '.join(missing)}")
        elif job.kind == "train":
            start = job.settings["init_from"]
            missing = find_missing_files(files_read_by_diagnosis(starting_run_dir(start), start))
            if missing:
                problems.append(f"{job.id}: these files of its starting checkpoint do not exist: {', '.join(missing)}")
            base = resolve_repository_path(job.settings["base_config"])
            if not base.is_file():
                problems.append(f"{job.id}: base config {base} does not exist")
                continue
            try:
                run_config_from_mapping(build_training_config(job, plan.session))
            except (TypeError, ValueError) as error:
                problems.append(f"{job.id}: training config is invalid: {error}")
    return problems


def summarize_plan(plan: Plan, jobs: Sequence[Job]) -> list[str]:
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
            # Each saved checkpoint, plus the starting one, is diagnosed
            # against each diagnose_against opponent.
            diagnoses = sum((saved_checkpoint_count(job) + 1) * len(job.settings["diagnose_against"]) for job in group)
            lines.append(f"  {experiment} (train): {len(group)} runs, {updates} updates, "
                         f"{diagnoses} diagnoses (start and checkpoints), {diagnoses * games_per_job} games")
        else:
            lines.append(f"  {experiment} ({kind}): {len(group)} jobs, {len(group) * games_per_job} games")
    return lines


def check_required_settings(plan: Plan, jobs: Sequence[Job]) -> list[str]:
    """Settings a new run must have, even though reading the file lets them
    be missing.

    Reading the file accepts a missing progress_min_extra_wins, and a
    missing progress_against when diagnose_against has several labels, so
    that old sessions saved before these settings existed stay readable
    (see parse_plan). A new run without them would finish with no threshold,
    or no target opponent, to judge its training by, so preflight refuses
    it.
    """

    trained = dict.fromkeys(job.experiment for job in jobs if job.kind == "train")
    problems = []
    for experiment in trained:
        if plan.answers[experiment]["progress_min_extra_wins"] is None:
            problems.append(f"[[train]] {experiment}: progress_min_extra_wins is required")
        if plan.answers[experiment]["progress_against"] is None:
            problems.append(f"[[train]] {experiment}: progress_against is required, because "
                            "diagnose_against has more than one label")
    return problems


def check_plan_and_build_tools(plan: Plan, jobs: Sequence[Job], bin_dir: Path) -> tuple[list[str], Path | None]:
    """Run every check. Returns the problems found and opponentcheck's path.

    The order matters only for speed: file checks need nothing built; the
    spec check needs opponentcheck, so it runs after the build. If the build
    fails there is no tool to check specs with, so that check is skipped.
    """

    problems = check_required_settings(plan, jobs) + check_input_files(plan, jobs)
    try:
        tool = build_go_tools(bin_dir)
    except PlanError as error:
        return [*problems, str(error)], None
    problems += check_specs_canonical(unique_specs(jobs), tool)
    return problems, tool


# ---------------------------------------------------------------------------
# Stages 2 and 3: run each job and record it.
#
# Each job runs other programs (opponentcheck, training, diagnose) as
# separate processes, waits for them, and records the outcome in its folder.
# ---------------------------------------------------------------------------


def utc_timestamp() -> str:
    """The current time in UTC, to the second, as text. Example:
    "2026-11-26T02:33:02+00:00". UTC keeps times comparable between a
    laptop and a cloud machine in another time zone."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomically(path: Path, value: Any) -> None:
    """Write JSON to a temporary file, then rename it into place, so a reader
    never sees half a file.

    A rename within one folder happens all at once: anyone reading the file
    sees either the old version or the new one, never a half-written mix.
    That matters for done.json, which promises "this job finished", and for
    session.json, which `status` may read while a session is updating it.
    sort_keys gives the same key order every time, so two files are easy to
    compare.
    """

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def format_toml(values: Mapping[str, Any]) -> str:
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
    """Runs jobs, one at a time, each as separate processes.

    A process is a separate running program. Each job's commands
    (opponentcheck, training, diagnose) run as processes the runner starts
    and waits for, so a crash in one ends only that job, and each starts
    with a clean slate (for example, its GPU memory is freed when it ends).
    """

    def __init__(self, plan: Plan, out: Path, tool: Path) -> None:
        self.plan = plan
        # The session folder.
        self.out = out
        # The opponentcheck binary built by preflight.
        self.tool = tool

    def run_command(self, arguments: Sequence[str], log: Path, *, cwd: Path, gomaxprocs: int) -> int:
        """Run one command, appending its output to the job's log.

        Returns the command's exit code: 0 means it succeeded, anything else
        that it failed.

        Ctrl-C: Python turns it into a KeyboardInterrupt in this program
        while it waits. The command is then stopped too (a terminal's Ctrl-C
        usually reaches it as well, but a signal sent to the runner alone
        wouldn't), and the KeyboardInterrupt carries on up to run_session,
        which records the session and stops. The job is left without a
        result, so the next run reruns it.
        """

        # The child gets a copy of this program's environment variables, with
        # its own GOMAXPROCS (Go's thread limit) for this job.
        env = {**os.environ, "GOMAXPROCS": str(gomaxprocs)}
        # "ab" appends in binary, so the log keeps every earlier command's
        # output, and the child's bytes are written exactly as it printed them.
        with log.open("ab") as handle:
            handle.write(f"$ {' '.join(arguments)}\n".encode())
            handle.flush()
            process = subprocess.Popen(arguments, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
            try:
                return process.wait()
            except KeyboardInterrupt:
                # terminate() asks the command to end (the SIGTERM signal);
                # wait() then makes sure it has, so nothing is left running.
                process.terminate()
                process.wait()
                raise

    def run_job(self, job: Job) -> str:
        """Run one job in its folder and return "done" or "failed".

        The stages are: set aside any unfinished folder (or, for a train job
        whose training finished, keep it); create the folder and record what
        it runs; run the job's commands; write done.json last, or
        failed.json. A job that raises an error is recorded as failed and
        never stops the other jobs. Ctrl-C is the exception: it stops the
        whole session, leaving this job without a result.
        """

        folder = self.out / job.experiment / job.label
        # Stage 1: a finished job is never run again. Any other folder is
        # left from an interrupted or failed try: rename it out of the way
        # (kept, never deleted), so the new try starts from an empty folder
        # and its files can't mix with the old ones. The exception is a train
        # job whose training finished: training is the expensive part, so
        # its folder is kept and the retry redoes only the steps that didn't
        # finish (see keep_finished_training). Reusing it is safe because a
        # resume runs the same code on the same inputs (the session
        # fingerprint), and training-done.json is written only after
        # training succeeded.
        if (folder / "done.json").exists():
            return "done"
        if folder.exists():
            if job.kind == "train" and (folder / TRAINING_DONE).exists():
                keep_finished_training(folder)
            else:
                set_aside_unfinished_folder(folder)
        # Stage 2: the folder, and job.json first, so even a job that dies
        # at once says what it was trying to run. monotonic() is a clock
        # that only moves forward, so the duration is right even if the
        # computer's clock is changed during the job.
        folder.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        record = {"job": job.id, "kind": job.kind, "settings": dict(job.settings), "started": utc_timestamp()}
        write_json_atomically(folder / "job.json", record)
        log = folder / "log.txt"
        # Stage 3: run the commands. Any error is caught and recorded, so one
        # broken job can't stop the jobs after it. KeyboardInterrupt (Ctrl-C)
        # is not an Exception in Python, so it isn't caught here: it passes
        # straight up to run_session, before anything below is written.
        try:
            failed = self._run_job_commands(job, folder, log)
        except Exception as error:  # recorded, so one broken job can't stop the session
            failed = f"{type(error).__name__}: {error}"
        record.update(finished=utc_timestamp(), seconds=round(time.monotonic() - started, 1))
        # Stage 4: record the outcome. done.json is written last of all, so
        # its presence promises everything before it finished.
        if failed:
            write_json_atomically(folder / "failed.json", {**record, "error": failed, "log": "log.txt"})
            return "failed"
        write_json_atomically(folder / "done.json", record)
        return "done"

    def _run_job_commands(self, job: Job, folder: Path, log: Path) -> str | None:
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
            code = self.run_command(arguments, log, cwd=REPOSITORY_ROOT, gomaxprocs=session.cpu_workers)
            return None if code == 0 else f"opponentcheck exited with {code}"

        # A diagnose job: one checkpoint plays the session's maps against
        # one opponent.
        if job.kind == "diagnose":
            run_dir = resolve_repository_path(job.settings["run_dir"])
            return self._run_diagnosis(run_dir, full_checkpoint_path(job.settings["run_dir"], job.settings["update"]),
                                  job.settings["opponent"], folder / "diagnosis", maps, log)

        # A train job runs in steps, and each step records its own success:
        # every diagnosis writes done.json in its folder, and training writes
        # training-done.json in the job folder. A retry of a job whose
        # training finished keeps that work (see run_job) and skips every
        # step already marked done, so a diagnosis failing after hours of
        # training costs a rerun of that diagnosis, not of the training.

        # Part 1: diagnose the starting checkpoint against every
        # diagnose_against opponent. The analysis compares the final
        # checkpoint with this, so both play the same maps, seats and
        # opponent with the same code. It runs before training, so the
        # baseline exists even if training then fails.
        start = resolve_repository_path(starting_run_dir(job.settings["init_from"]))
        for label, spec in job.settings["diagnose_against"].items():
            out = folder / "diagnoses" / f"start__{label}"
            if (out / "done.json").exists():
                continue  # finished on an earlier try
            failed = self._run_diagnosis(start, resolve_repository_path(job.settings["init_from"]), spec,
                                         out, maps, log)
            if failed:
                return failed

        # Part 2: write the job's own full config and train, unless an
        # earlier try already finished training.
        run_dir = folder / "run"
        training_done = folder / TRAINING_DONE
        if not training_done.exists():
            config = folder / "config.toml"
            config.write_text(format_toml(build_training_config(job, session)))
            # sys.executable is the Python running this script, so training
            # uses the same Python and installed packages.
            code = self.run_command([
                sys.executable, "-m", "orrery.train", "--config", str(config), "--run-dir", str(run_dir),
                "--init-from", str(resolve_repository_path(job.settings["init_from"])),
                "--max-updates", str(job.settings["max_updates"]),
            ], log, cwd=PYTHON_ROOT, gomaxprocs=session.gomaxprocs)
            if code != 0:
                return f"training exited with {code}"
            # Written only after training exited successfully, so its
            # presence promises a complete training run in run/.
            write_json_atomically(training_done, {"finished": utc_timestamp()})

        # Part 3: diagnose every checkpoint the run saved against every
        # diagnose_against opponent, skipping those an earlier try finished.
        # The checkpoint files actually saved, not the ones expected: names
        # are zero-padded (update-000030.pt), so sorting them as text puts
        # them in update order.
        checkpoints = sorted((run_dir / "checkpoints").glob("update-*.pt"))
        if not checkpoints:
            return "training saved no checkpoint"
        for checkpoint in checkpoints:
            for label, spec in job.settings["diagnose_against"].items():
                out = folder / "diagnoses" / f"{checkpoint.stem}__{label}"
                if (out / "done.json").exists():
                    continue  # finished on an earlier try
                failed = self._run_diagnosis(run_dir, checkpoint, spec, out, maps, log)
                if failed:
                    return failed
        return None

    def _run_diagnosis(self, run_dir: Path, checkpoint: Path, opponent: str, out: Path,
                  maps: list[str], log: Path) -> str | None:
        """Run diagnose.py on one checkpoint; return why it failed, or None.

        --no-charts skips the pictures: the analysis reads only report.json,
        and the charts would need the optional plotting packages. An earlier,
        unfinished attempt left in ``out`` is archived first (see
        archive_attempt). On success it writes done.json in ``out``, last,
        so a retry of a train job can tell a finished diagnosis from a
        half-written one.
        """

        archive_attempt(out)
        code = self.run_command([
            sys.executable, "-m", "scripts.diagnose", "--run-dir", str(run_dir), "--checkpoint", str(checkpoint),
            "--opponent", opponent, *maps, "--device", self.plan.session.device, "--no-charts",
            "--out", str(out),
        ], log, cwd=PYTHON_ROOT, gomaxprocs=self.plan.session.gomaxprocs)
        if code != 0:
            return f"diagnose of {checkpoint.name} against {opponent} exited with {code}"
        write_json_atomically(out / "done.json", {"finished": utc_timestamp()})
        return None


def set_aside_unfinished_folder(folder: Path) -> Path:
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


def keep_finished_training(folder: Path) -> None:
    """Get a train job's folder ready for a retry that reuses its finished
    training.

    The folder stays where it is, with its run/ (the trained model and
    checkpoints) and every diagnosis, finished or not. Only failed.json is
    moved, never deleted: it becomes failed.previous-<n>.json, so the job
    no longer reads as failed while it retries, and the earlier error stays
    readable. An unfinished diagnosis is archived later, just before it
    runs again (see archive_attempt).

    Example: diagnoses/update-000153__v2 failed after training finished.
    The retry keeps run/ and the five finished diagnoses, and runs only that
    diagnosis again, after archiving its failed attempt.
    """

    failed = folder / "failed.json"
    if failed.exists():
        n = 1
        while (target := folder / f"failed.previous-{n}.json").exists():
            n += 1
        failed.rename(target)


def archive_attempt(out: Path) -> Path | None:
    """Move an earlier, unfinished attempt at one diagnosis out of the way,
    just before that diagnosis runs again. Returns where it went, or None
    when there was nothing to move.

    diagnose.py needs an empty output folder, so a failed or interrupted
    attempt (a folder without done.json) must go first. It is moved into a
    hidden .partial folder beside it, one subfolder per diagnosis, so
    archived attempts never share a name with a real diagnosis folder,
    whatever the labels: a label must start with a letter or digit, so no
    diagnosis is ever named ".partial". Only the diagnosis about to run is
    touched; nothing is deleted. Use ``ls -a`` to see the archive.

    Example: a failed diagnoses/update-000153__v2 becomes
    diagnoses/.partial/update-000153__v2/attempt-1, and a second failure
    later becomes attempt-2.
    """

    if not out.exists():
        return None
    archive = out.parent / ".partial" / out.name
    archive.mkdir(parents=True, exist_ok=True)
    n = 1
    while (target := archive / f"attempt-{n}").exists():
        n += 1
    out.rename(target)
    return target


# ---------------------------------------------------------------------------
# Stage 4 and the session record: session.json, status and the job loop.
# ---------------------------------------------------------------------------


def read_job_state(out: Path, job: Job) -> str:
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


def read_command_output(arguments: Sequence[str]) -> str | None:
    """What a command prints, run from the repository root, or None if it
    can't run or fails (no git on the machine, say). Example:
    ["go", "version"] gives "go version go1.24.1 linux/amd64"."""

    try:
        result = subprocess.run(arguments, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def record_versions() -> dict[str, Any]:
    """The code and tools a session ran with, for replay.

    git_dirty is True when tracked files have changes not yet committed:
    then the commit alone doesn't say exactly what code ran. A value that
    can't be found (no git, say) is recorded as None, never guessed.
    """

    dirty = read_command_output(["git", "status", "--porcelain", "--untracked-files=no"])
    return {
        "git_commit": read_command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "go": read_command_output(["go", "version"]),
        "python": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# The session fingerprint: the code and input files a session's results came
# from.
#
# Results from different jobs are only comparable if every job ran the same
# code on the same input files. The first run saves a fingerprint of both in
# session.json, and every resume computes it again and refuses to run if
# anything differs. A fingerprint is made of SHA-256 hashes: a hash is a
# short text computed from a file's bytes, which changes if even one byte of
# the file changes.
# ---------------------------------------------------------------------------

# The file that git archive writes the commit into (see its own comments).
# It is left out of the code hash, because a Vast.ai bundle and a laptop
# checkout of the same commit differ only in this file.
RELEASE_FILE = "cmd/env/release.go"

# Top-level folders skipped when searching for Go code: runs/, which holds
# training output and can be large, and python/, whose files are listed
# separately. Hidden folders (.git, .build) are skipped too.
GO_SKIPPED = {"runs", "python"}


def hash_file(path: Path) -> str | None:
    """The SHA-256 hash of a file's bytes, or None when there is no file."""

    if not path.is_file():
        return None
    with path.open("rb") as handle:
        # file_digest reads the file in pieces, so a large checkpoint is
        # never held in memory all at once.
        return hashlib.file_digest(handle, "sha256").hexdigest()


def find_runtime_source_files(root: Path = REPOSITORY_ROOT) -> list[Path]:
    """The runtime source: every file whose content decides what a job does.

    That is the Go code without its tests, the Go module files, the Python
    packages the jobs run (orrery and scripts), and the Python dependency
    files. The files are found by searching the folder, not by asking Git,
    because the bundle copied to Vast.ai has no .git folder. They are sorted
    by name, so every machine lists them in the same order.
    """

    files: list[Path] = []
    for folder, subfolders, names in os.walk(root):
        # Changing subfolders in place tells os.walk which ones to enter.
        top_level = Path(folder) == root
        subfolders[:] = [
            name for name in subfolders
            if not name.startswith(".") and not (top_level and name in GO_SKIPPED)
        ]
        files += [Path(folder) / name for name in names if name.endswith(".go") and not name.endswith("_test.go")]
    files += [root / name for name in ("go.mod", "go.sum")]
    for package in ("python/orrery", "python/scripts"):
        files += (root / package).rglob("*.py")
    files += [root / "python" / name for name in ("pyproject.toml", "uv.lock")]
    kept = [path for path in files if path.is_file() and path != root / RELEASE_FILE]
    return sorted(kept, key=lambda path: path.relative_to(root).as_posix())


def installed_package_version(name: str) -> str | None:
    """An installed Python package's version, without importing it. Example:
    "torch" gives "2.5.1+cu124"; None when it isn't installed."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def fingerprint_code(root: Path = REPOSITORY_ROOT) -> dict[str, str | None]:
    """The code a session runs: one hash of the runtime source, plus the
    versions of the tools that run it.

    The source hash covers every file's name and bytes, so editing, adding,
    removing or renaming a runtime file changes it. The tool versions are
    recorded too, because the same code can play differently with another
    Go, Python, Torch or NumPy.
    """

    digest = hashlib.sha256()
    for path in find_runtime_source_files(root):
        # "\0" ends each file's name and "\n" ends its hash, so two
        # different lists of files can never give the same text to hash.
        digest.update(f"{path.relative_to(root).as_posix()}\0{hash_file(path)}\n".encode())
    return {
        "source_sha256": digest.hexdigest(),
        "go": read_command_output(["go", "version"]),
        "python": platform.python_version(),
        "torch": installed_package_version("torch"),
        "numpy": installed_package_version("numpy"),
    }


def list_input_files(jobs: Sequence[Job]) -> list[str]:
    """Every file the jobs read, as paths from the repository root, each
    once and sorted.

    A diagnose job reads the files that diagnosing its checkpoint needs
    (see files_read_by_diagnosis, the same list preflight checks). A train job
    reads the same files for its starting checkpoint, which it diagnoses
    first, plus its base config. Its own settings are in the experiment
    file's text, which a resume already compares. A bot job reads no files:
    its specs are in that text too.
    """

    found: set[str] = set()
    for job in jobs:
        if job.kind == "diagnose":
            run_dir = job.settings["run_dir"]
            found.update(files_read_by_diagnosis(run_dir, relative_checkpoint_path(run_dir, job.settings["update"])))
        elif job.kind == "train":
            start = job.settings["init_from"]
            found.update(files_read_by_diagnosis(starting_run_dir(start), start))
            found.add(job.settings["base_config"])
    return sorted(found)


def fingerprint_session(plan: Plan) -> dict[str, Any]:
    """The whole plan's fingerprint: its code and every input file.

    It covers every job in the file, not only the ones a run's --only picks:
    the analysis reads all finished jobs together, so all of them must come
    from the same code and inputs. A missing input is recorded as None.
    """

    return {
        "code": fingerprint_code(),
        "inputs": {name: hash_file(resolve_repository_path(name)) for name in list_input_files(plan.jobs)},
    }


def compare_fingerprints(saved: Mapping[str, Any] | None, current: Mapping[str, Any]) -> list[str]:
    """How the current fingerprint differs from the one a session saved; an
    empty list when they match.

    A session saved without a fingerprint (by a runner older than this
    check) can't be matched, so that counts as a difference too. Example
    result: ["input runs/x/checkpoints/update-000153.pt changed"].
    """

    if saved is None:
        return ["session.json has no fingerprint (an older runner made it), so its code and inputs are unknown"]
    changes = []
    for part in ("code", "inputs"):
        old, new = saved.get(part) or {}, current[part]
        # Look at every key either fingerprint has, so a key that only one
        # of them has also counts as a change.
        for key in sorted(set(old) | set(new)):
            before, after = old.get(key), new.get(key)
            if before == after:
                continue
            if part == "inputs":
                changes.append(f"input {key} {'is missing now' if after is None else 'changed'}")
            elif key == "source_sha256":
                changes.append("the runtime source code changed")
            else:
                changes.append(f"{key} was {before!r}, now {after!r}")
    return changes


def run_session(experiment: Path, out: Path, only: str | None = None) -> int:
    """Preflight, then run every job not yet done, one at a time.

    Returns 0 when every selected job is done, 1 when any failed or the run
    was interrupted, and 2 when preflight found problems or the session
    folder holds a different experiment file, code or inputs.

    The stages are: read the plan and choose the jobs; check the session
    folder belongs to this file, code and inputs; preflight; record the run
    in session.json; run the jobs in order; record every job's final state.
    """

    # Stage 1: the plan, and the jobs this run covers (--only picks one
    # experiment by name).
    text = experiment.read_text()
    plan = parse_plan(text)
    jobs = run_order([job for job in plan.jobs if only is None or job.experiment == only])
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
    # The same goes for the code and the input files: a resume must run what
    # the finished jobs ran. This is checked before preflight, so a refused
    # resume changes nothing in the session folder (preflight rebuilds the
    # Go tools in it).
    fingerprint = fingerprint_session(plan)
    if record:
        changes = compare_fingerprints(record.get("fingerprint"), fingerprint)
        if changes:
            print(f"{out} holds a session made with different code or inputs; use a new --out:")
            for change in changes:
                print(f"- {change}")
            return 2

    # Stage 3: preflight, with the Go tools built into the session folder.
    problems, tool = check_plan_and_build_tools(plan, jobs, out / "bin")
    for line in summarize_plan(plan, jobs):
        print(line)
    if problems:
        print("preflight found problems; nothing was run:")
        for problem in problems:
            print(f"- {problem}")
        return 2
    assert tool is not None

    # Stage 4: record this run. setdefault keeps what an earlier run already
    # saved; "runs" gets one entry per run, so a resumed session shows each
    # attempt with its own code version and times. The fingerprint is saved
    # once, by the first run, before any job starts; a resume has just
    # matched it.
    record.setdefault("experiment_file", str(experiment))
    record.setdefault("experiment_text", text)
    record.setdefault("fingerprint", fingerprint)
    record.setdefault("runs", []).append({"started": utc_timestamp(), "only": only, **record_versions()})

    def save(job: Job | None = None, state: str | None = None) -> None:
        """Set one job's state (if given) and write session.json."""

        if job is not None:
            record.setdefault("jobs", {})[job.id] = state
        write_json_atomically(session_file, record)

    for job in jobs:
        record.setdefault("jobs", {})[job.id] = read_job_state(out, job)
    save()

    # Stage 5: run the jobs, one at a time, skipping any already done (from
    # an earlier run of this session). session.json is saved before and after
    # each job, so `status` can follow along.
    runner = Runner(plan, out, tool)
    try:
        for number, job in enumerate(jobs, start=1):
            if read_job_state(out, job) == "done":
                continue
            print(f"[{number}/{len(jobs)}] start {job.id}", flush=True)
            save(job, "running")
            state = runner.run_job(job)
            save(job, state)
            print(f"[{number}/{len(jobs)}] {state} {job.id}", flush=True)
    except KeyboardInterrupt:
        # Ctrl-C: the running command has already been stopped (see
        # Runner.run_command) and its job left without a result. Record every
        # job's state and stop; the same command resumes later.
        print("interrupted: run the same command again to resume", flush=True)
        for job in jobs:
            record["jobs"][job.id] = read_job_state(out, job)
        record["runs"][-1]["finished"] = utc_timestamp()
        save()
        return 1

    # Stage 6: every job's final state, read from its folder, and a summary.
    for job in jobs:
        record["jobs"][job.id] = read_job_state(out, job)
    record["runs"][-1]["finished"] = utc_timestamp()
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
        state = read_job_state(out, job)
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
        problems, _tool = check_plan_and_build_tools(plan, jobs, Path(scratch))
    for line in summarize_plan(plan, jobs):
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
