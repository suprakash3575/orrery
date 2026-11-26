"""The analysis must turn a session folder's results into the right numbers
and answers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_experiments import analyze, answer_training, measure_training_progress, render_markdown
from scripts.run_experiments import Runner, check_required_settings, parse_plan, print_status

V2 = "heuristic-v2:candidates=4,enemy_bonus=1,margin=0.1"
LIMITED = "heuristic-v2:candidates=4,enemy_bonus=1,garrison_cost=needed,learner_aim=1,margin=0.1,no_aim_check=1,whole_garrison=1"

EXPERIMENT = f"""
[session]
name = "hand-made"
maps = {{ start = 1000000, count = 2 }}
device = "cpu"
cpu_workers = 2

[[botcheck]]
name = "bots"
opponent = "{V2}"
check_decode = ["limited"]
[botcheck.testers]
normal = "{V2}"
limited = "{LIMITED}"

[[compare]]
question = "How much strength does the fully limited bot keep?"
baseline = "bots/normal"
candidate = "bots/limited"
rule = "restricted-bot"

[[dial]]
name = "dial"
order = ["bots/normal", "bots/limited"]
"""


def _bot_job(out: Path, label: str, winners: dict[int, int], calibration_seeds: int) -> None:
    """A finished bot job on 2 maps x 4 seats: winners maps a map seed to the
    one seat that won it, if any."""

    folder = out / "bots" / label
    (folder / "opponentcheck").mkdir(parents=True)
    games = [
        {"seed": seed, "seat": seat, "status": "completed", "result": 1 if winners.get(seed) == seat else -1}
        for seed in (1_000_000, 1_000_001)
        for seat in range(4)
    ]
    (folder / "opponentcheck" / "games.jsonl").write_text("".join(json.dumps(g) + "\n" for g in games))
    wins = sum(game["result"] > 0 for game in games)
    (folder / "opponentcheck" / "summary.json").write_text(json.dumps({
        "run": {"status": "complete"},
        "calibration": {"seeds": calibration_seeds, "passed": True},
        "testers": [{
            "completed": 8, "planned": 8, "wins": wins, "win_rate": wins / 8, "eliminated": 0,
            "length_p50": 500, "aim_failure_rate": None, "fallback_rate": None,
            "random_moves_per_game": 0.0, "decode_checked_launches": 40,
        }],
    }))
    (folder / "done.json").write_text("{}")


TRAIN_EXPERIMENT = f"""
[session]
name = "train-only"
maps = {{ start = 1000000, count = 2 }}
device = "cpu"
cpu_workers = 2

[[train]]
name = "train"
base_config = "configs/rl-v2-4p.toml"
init_from = ["runs/b-s1/checkpoints/update-000153.pt"]
max_updates = 2
checkpoint_every_updates = 1
progress_min_runs = 1
progress_min_extra_wins = 1
training_opponents = {{ weak = "heuristic-v2:candidates=4,enemy_bonus=1,epsilon=0.75,margin=0.1" }}
diagnose_against = {{ v2 = "{V2}" }}
"""


def diagnosis_report(wins: int, *, seeds: tuple[int, int] = (1000000, 1000002), rejected: int = 0) -> dict:
    """A clean diagnose report.json of 4 players against V2: every seed in
    ``seeds`` from every seat once, ``wins`` of them won, and ``rejected``
    learner actions refused by Go."""

    results = [{"seed": seed, "seat": seat} for seed in range(*seeds) for seat in range(4)]
    return {
        "format": "diagnosis-v2",
        "identity": {"players": 4, "contract": "rl-v2", "generator_version": "generator-v2-4p"},
        "games": {
            "seeds": list(seeds), "opponent": V2, "engine": "engine", "evaluation": {"device": "cpu"},
            "rejected_actions": rejected, "game_results": results,
            "measurements": {"games": len(results), "wins": wins},
        },
    }


def test_a_train_job_trains_diagnoses_every_checkpoint_and_is_analysed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A train job diagnoses its starting checkpoint, writes its config,
    trains, then diagnoses each checkpoint.

    Training and diagnosis are replaced by fakes that write what the real
    commands write. Training wins 1 of 4 games in update 1 and 3 of 4 in
    update 2, so its win rate is 25% then 75% and it rose. The diagnoses win
    0 of 8 at the start, 0 of 8 at checkpoint 1 and 1 of 8 at checkpoint 2,
    the final one: 1 extra win, which reaches this file's threshold of 1, so
    experiment 4's answer is "observed progress".
    """

    plan = parse_plan(TRAIN_EXPERIMENT)
    (job,) = plan.jobs
    diagnosis_wins = iter([0, 0, 1])

    def fake_command(self, arguments, log, *, cwd, gomaxprocs):
        if "orrery.train" in arguments:
            run = Path(arguments[arguments.index("--run-dir") + 1])
            (run / "checkpoints").mkdir(parents=True)
            for update in (1, 2):
                (run / "checkpoints" / f"update-{update:06d}.pt").write_bytes(b"weights")
            (run / "metrics.csv").write_text("update,wins,losses,draws\n1,1,3,0\n2,3,1,0\n")
            assert "--init-from" in arguments and gomaxprocs == plan.session.gomaxprocs
        else:
            out = Path(arguments[arguments.index("--out") + 1])
            out.mkdir(parents=True)
            (out / "report.json").write_text(json.dumps(diagnosis_report(next(diagnosis_wins))))
        return 0

    monkeypatch.setattr(Runner, "run_command", fake_command)
    assert Runner(plan, tmp_path, Path("opponentcheck")).run_job(job) == "done"
    folder = tmp_path / job.experiment / job.label
    assert (folder / "diagnoses" / "start__v2" / "report.json").exists()
    assert 'training_opponent = "heuristic-v2:candidates=4,enemy_bonus=1,epsilon=0.75,margin=0.1"' in (
        folder / "config.toml"
    ).read_text()

    (tmp_path / "session.json").write_text(json.dumps({"experiment_text": TRAIN_EXPERIMENT, "runs": []}))
    run = analyze(tmp_path)["train"]["train"]["runs"][0]
    assert [c["training_win_rate"] for c in run["checkpoints"]] == [0.25, 0.75]
    assert [c["diagnosis_win_rates"]["v2"] for c in run["checkpoints"]] == [0.0, 0.125]
    assert run["training_rose"] is True
    assert run["progress"] == {"against": "v2", "final_update": 2, "start": {"wins": 0, "games": 8},
                               "final": {"wins": 1, "games": 8}, "invalid": None, "extra_wins": 1}
    assert analyze(tmp_path)["train"]["train"]["answer"].startswith("Observed progress")

    # The final checkpoint's file is there but its report is not: the
    # analysis must not crash, and the answer waits.
    (folder / "diagnoses" / "update-000002__v2" / "report.json").unlink()
    analysis = analyze(tmp_path)
    run = analysis["train"]["train"]["runs"][0]
    assert run["checkpoints"][1]["diagnosis_win_rates"]["v2"] is None
    assert run["progress"]["extra_wins"] is None
    assert analysis["train"]["train"]["answer"].startswith("Pending")
    assert "| 2 | 75.0% | missing |" in render_markdown(analysis)


def test_progress_needs_extra_wins_in_enough_runs_and_waits_for_every_result() -> None:
    """Three runs against one training opponent gain 4, 4 and 0 wins over
    their starting checkpoints: two reach 4, so with min_runs 2 the answer
    is observed progress. If one of the two had no final diagnosis, the
    answer would wait rather than judge on part of the evidence."""

    def run(job: str, extra_wins: int | None) -> dict:
        return {"job": job, "done": True, "opponent": "weak", "training_rose": False,
                "progress": {"against": "v2", "invalid": None, "extra_wins": extra_wins}}

    runs = [run("train/weak__s1", 4), run("train/weak__s2", 4), run("train/weak__s3", 0)]
    assert answer_training(runs, min_runs=2, min_extra_wins=4, against="v2").startswith("Observed progress")
    runs[1] = run("train/weak__s2", None)
    assert answer_training(runs, min_runs=2, min_extra_wins=4, against="v2") == (
        "Pending: no final-versus-start result yet for train/weak__s2."
    )


def test_a_session_saved_before_the_progress_threshold_stays_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A session.json saved before progress_min_extra_wins existed still
    parses for status and the analysis, which give no progress verdict. A
    new run of the same file is refused by preflight."""

    old_text = TRAIN_EXPERIMENT.replace("progress_min_extra_wins = 1\n", "")
    (tmp_path / "session.json").write_text(json.dumps({"experiment_text": old_text, "runs": []}))

    assert print_status(tmp_path) == 0
    assert "waiting    train/weak__b-s1" in capsys.readouterr().out
    assert analyze(tmp_path)["train"]["train"]["answer"].startswith("Unavailable")
    plan = parse_plan(old_text)
    assert check_required_settings(plan, plan.jobs) == ["[[train]] train: progress_min_extra_wins is required"]


@pytest.mark.parametrize(("final", "reason"), [
    (diagnosis_report(5, seeds=(1000002, 1000004)), "differ in: seeds, map-and-seat assignments"),
    (diagnosis_report(5, rejected=1), "final report has 1 rejected actions"),
])
def test_progress_refuses_diagnoses_that_did_not_play_the_same_clean_games(
    tmp_path: Path, final: dict, reason: str
) -> None:
    """The final checkpoint wins 5 more games than the start, but on other
    maps, or with a turn the model didn't really play. Neither may count as
    progress: the run is invalid evidence, with the reason."""

    (job,) = parse_plan(TRAIN_EXPERIMENT).jobs
    for name, report in (("start__v2", diagnosis_report(0)), ("update-000002__v2", final)):
        (tmp_path / "diagnoses" / name).mkdir(parents=True)
        (tmp_path / "diagnoses" / name / "report.json").write_text(json.dumps(report))

    progress = measure_training_progress(tmp_path, job, "v2")

    assert progress["extra_wins"] is None and reason in progress["invalid"]
    run = {"job": job.id, "done": True, "opponent": "weak", "training_rose": True, "progress": progress}
    assert answer_training([run], min_runs=1, min_extra_wins=1, against="v2").startswith("Invalid evidence")


def test_bot_results_compare_and_answer_as_worked_out_by_hand(tmp_path: Path) -> None:
    """normal wins one seat of each map, 1/4 = 25% per map; limited wins
    none. So every map's difference is -25 points, the range is -25 to -25,
    strength kept is 0%, and experiment 1's rule answers "an obstacle"
    (-25 is at most -10 and the whole range is below 0). The dial falls from
    25% to 0%, so it is usable."""

    (tmp_path / "session.json").write_text(json.dumps({"experiment_text": EXPERIMENT, "runs": []}))
    _bot_job(tmp_path, "normal", {1_000_000: 0, 1_000_001: 2}, calibration_seeds=2)
    _bot_job(tmp_path, "limited", {}, calibration_seeds=0)

    analysis = analyze(tmp_path)

    normal, limited = analysis["botcheck"]["bots"]["testers"]
    assert (normal["win_rate"], normal["calibration"]) == (0.25, "passed")
    assert (limited["win_rate"], limited["calibration"]) == (0.0, "not run")
    compare = analysis["compare"][0]
    assert compare["result"] == {
        "maps": 2, "baseline_rate": 0.25, "candidate_rate": 0.0, "difference": -0.25,
        "range": [-0.25, -0.25], "strength_kept": 0.0,
    }
    assert compare["answer"].startswith("The learner's moves are an obstacle.")
    assert analysis["dial"][0]["win_rates"] == [0.25, 0.0]
    assert analysis["dial"][0]["answer"].startswith("Usable")
    assert "| normal | 8/8 | 2 | 25.0% |" in render_markdown(analysis)
