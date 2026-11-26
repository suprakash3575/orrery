"""The experiment runner must expand, check and resume a session exactly."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_experiments
from scripts.run_experiments import (
    PlanError,
    Runner,
    check_input_files,
    check_specs_canonical,
    parse_plan,
    run_order,
    unique_specs,
)

V2 = "heuristic-v2:candidates=4,enemy_bonus=1,margin=0.1"
WEAK = "heuristic-v2:candidates=4,enemy_bonus=1,epsilon=0.5,margin=0.1"

# One section of each kind: 2 testers, 2 opponents x 1 run x 2 updates, and
# 1 training opponent x 2 starting checkpoints.
SMALL = f"""
[session]
name = "small"
maps = {{ start = 1000000, count = 2 }}
device = "cpu"
cpu_workers = 2

[[botcheck]]
name = "bots"
opponent = "{V2}"
check_decode = ["weak"]
[botcheck.testers]
normal = "{V2}"
weak = "{WEAK}"

[[diagnose]]
name = "probe"
run_dirs = ["runs/b-s1"]
updates = [30, 60]
[diagnose.opponents]
v2 = "{V2}"
"v2-eps0.5" = "{WEAK}"

[[train]]
name = "train"
base_config = "configs/rl-v2-4p.toml"
init_from = ["runs/b-s1/checkpoints/update-000153.pt", "runs/b-s2/checkpoints/update-000153.pt"]
training_opponents = {{ weak = "{WEAK}" }}
max_updates = 4
checkpoint_every_updates = 2
progress_min_extra_wins = 1
diagnose_against = {{ v2 = "{V2}" }}

[[compare]]
question = "weak minus normal"
baseline = "bots/normal"
candidate = "bots/weak"
rule = "restricted-bot"
"""


def test_a_small_file_expands_into_one_job_per_combination() -> None:
    """Each list or table of choices gives one job per combination, each with
    its own folder named after what it varies."""

    plan = parse_plan(SMALL)

    assert [job.id for job in plan.jobs] == [
        "bots/normal",
        "bots/weak",
        "probe/v2__b-s1__u000030",
        "probe/v2__b-s1__u000060",
        "probe/v2-eps0.5__b-s1__u000030",
        "probe/v2-eps0.5__b-s1__u000060",
        "train/weak__b-s1",
        "train/weak__b-s2",
    ]
    # Jobs run bot jobs first, then diagnose, then train, whatever order they
    # come in; within a kind they keep the order given. Reversing the list
    # shows both: the kinds are regrouped, and the two bot jobs stay reversed.
    ordered = run_order(list(reversed(plan.jobs)))
    assert [job.kind for job in ordered] == ["botcheck"] * 2 + ["diagnose"] * 4 + ["train"] * 2
    assert [job.id for job in ordered][:2] == ["bots/weak", "bots/normal"]
    assert plan.jobs[1].settings == {"tester": WEAK, "opponent": V2, "check_decode": True}
    assert unique_specs(plan.jobs) == [V2, WEAK]
    with pytest.raises(PlanError, match="not a bot job"):
        parse_plan(SMALL.replace('candidate = "bots/weak"', 'candidate = "bots/missing"'))
    # A progress threshold of 0 would count a run that did no better than
    # its start, or need no run at all.
    with pytest.raises(PlanError, match="progress_min_extra_wins must be at least 1, got 0"):
        parse_plan(SMALL.replace("progress_min_extra_wins = 1", "progress_min_extra_wins = 0"))
    with pytest.raises(PlanError, match="progress_min_runs must be at least 1, got 0"):
        parse_plan(SMALL.replace("progress_min_extra_wins = 1", "progress_min_extra_wins = 1\nprogress_min_runs = 0"))


def test_preflight_refuses_a_missing_checkpoint_and_a_non_canonical_spec() -> None:
    """Both mistakes are found before anything runs. A fake opponentcheck
    answers -canonical as Go would: parameters in alphabetical order."""

    plan = parse_plan(SMALL)
    missing = check_input_files(plan, [job for job in plan.jobs if job.kind == "diagnose"])
    assert len(missing) == 4
    # Every file diagnosing the checkpoint reads is checked, the two files
    # beside the checkpoint and its run's files included.
    assert missing[0] == (
        "probe/v2__b-s1__u000030: these files do not exist: runs/b-s1/checkpoints/update-000030.pt, "
        "runs/b-s1/checkpoints/update-000030.pt.sha256, runs/b-s1/checkpoints/update-000030.pt.metadata.json, "
        "runs/b-s1/run-config.json, runs/b-s1/manifest.json, runs/b-s1/metrics.csv"
    )

    def fake_opponentcheck(arguments, **_kwargs):
        forms = [V2 if spec.startswith("heuristic-v2:margin") else spec for spec in arguments[2:]]
        return SimpleNamespace(returncode=0, stdout="\n".join(forms) + "\n", stderr="")

    misspelled = "heuristic-v2:margin=0.1,candidates=4,enemy_bonus=1"
    problems = check_specs_canonical([V2, misspelled], Path("opponentcheck"), runner=fake_opponentcheck)
    assert problems == [f"spec is not canonical: {misspelled!r}\n    write it as: {V2!r}"]


def test_a_rerun_skips_finished_jobs_and_sets_aside_unfinished_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job with done.json is never run again. A folder without one is
    renamed <name>.partial-<n>, kept, and the job runs afresh."""

    plan = parse_plan(SMALL)
    normal, weak = plan.jobs[0], plan.jobs[1]
    ran = []
    monkeypatch.setattr(Runner, "_run_job_commands", lambda self, job, folder, log: ran.append(job.id))
    runner = Runner(plan, tmp_path, Path("opponentcheck"))

    finished = tmp_path / "bots" / "normal"
    finished.mkdir(parents=True)
    (finished / "done.json").write_text("{}")
    interrupted = tmp_path / "bots" / "weak"
    interrupted.mkdir(parents=True)
    (interrupted / "log.txt").write_text("half a run")

    assert runner.run_job(normal) == "done"
    assert runner.run_job(weak) == "done"

    assert ran == ["bots/weak"]
    assert (tmp_path / "bots" / "weak.partial-1" / "log.txt").read_text() == "half a run"
    assert (tmp_path / "bots" / "weak" / "done.json").exists()
    assert run_experiments.read_job_state(tmp_path, weak) == "done"


def test_a_resume_refuses_a_checkpoint_changed_at_the_same_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A finished diagnose job's checkpoint is replaced by different bytes at
    the same path. Resuming only the bot experiment must still refuse, before
    preflight, and leave every saved record as it was."""

    run_dir = tmp_path / "b-s1"
    checkpoint = run_dir / "checkpoints" / "update-000030.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"first model")
    for name in ("run-config.json", "manifest.json", "metrics.csv"):
        (run_dir / name).write_text(name)
    experiment = tmp_path / "small.toml"
    experiment.write_text(f"""
[session]
name = "small"
maps = {{ start = 1000000, count = 2 }}
device = "cpu"
cpu_workers = 2

[[botcheck]]
name = "bots"
opponent = "{V2}"
[botcheck.testers]
normal = "{V2}"

[[diagnose]]
name = "probe"
run_dirs = ["{run_dir}"]
updates = [30]
[diagnose.opponents]
v2 = "{V2}"
""")
    out = tmp_path / "session"
    # Only the inputs change in this test, so the code part is held fixed,
    # and no job really runs: each one just succeeds.
    monkeypatch.setattr(run_experiments, "fingerprint_code", lambda: {"source_sha256": "same"})
    monkeypatch.setattr(run_experiments, "check_plan_and_build_tools", lambda plan, jobs, bin_dir: ([], Path("opponentcheck")))
    monkeypatch.setattr(Runner, "_run_job_commands", lambda self, job, folder, log: None)
    assert run_experiments.run_session(experiment, out, only="probe") == 0
    session_before = (out / "session.json").read_text()
    done_before = (out / "probe" / "v2__b-s1__u000030" / "done.json").read_text()

    checkpoint.write_bytes(b"other model")

    def preflight_must_not_run(*_arguments):
        raise AssertionError("preflight ran on a refused resume")

    monkeypatch.setattr(run_experiments, "check_plan_and_build_tools", preflight_must_not_run)
    assert run_experiments.run_session(experiment, out, only="bots") == 2
    assert f"input {checkpoint.as_posix()} changed" in capsys.readouterr().out
    assert (out / "session.json").read_text() == session_before
    assert (out / "probe" / "v2__b-s1__u000030" / "done.json").read_text() == done_before
    assert not (out / "bots").exists()
