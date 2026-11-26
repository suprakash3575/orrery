"""The analysis must turn a session folder's results into the right numbers
and answers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_experiments import analyze, markdown
from scripts.run_experiments import Runner, parse_plan

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
training_opponents = {{ weak = "heuristic-v2:candidates=4,enemy_bonus=1,epsilon=0.75,margin=0.1" }}
diagnose_against = {{ v2 = "{V2}" }}
"""


def test_a_train_job_trains_diagnoses_every_checkpoint_and_is_analysed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A train job writes its config, trains, then diagnoses each checkpoint.

    Training and diagnosis are replaced by fakes that write what the real
    commands write. Training wins 1 of 4 games in update 1 and 3 of 4 in
    update 2, so its win rate is 25% then 75% and it rose. The diagnoses win
    0 of 8 at checkpoint 1 and 1 of 8 at checkpoint 2, above 0%, which is
    experiment 4's "first real progress".
    """

    plan = parse_plan(TRAIN_EXPERIMENT)
    (job,) = plan.jobs
    diagnosis_wins = iter([0, 1])

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
            report = {"games": {"measurements": {"games": 8, "wins": next(diagnosis_wins)}}}
            (out / "report.json").write_text(json.dumps(report))
        return 0

    monkeypatch.setattr(Runner, "command", fake_command)
    assert Runner(plan, tmp_path, Path("opponentcheck")).run_job(job) == "done"
    folder = tmp_path / job.experiment / job.label
    assert 'training_opponent = "heuristic-v2:candidates=4,enemy_bonus=1,epsilon=0.75,margin=0.1"' in (
        folder / "config.toml"
    ).read_text()

    (tmp_path / "session.json").write_text(json.dumps({"experiment_text": TRAIN_EXPERIMENT, "runs": []}))
    run = analyze(tmp_path)["train"]["train"]["runs"][0]
    assert [c["training_win_rate"] for c in run["checkpoints"]] == [0.25, 0.75]
    assert [c["diagnosis_win_rates"]["v2"] for c in run["checkpoints"]] == [0.0, 0.125]
    assert run["training_rose"] is True
    assert analyze(tmp_path)["train"]["train"]["answer"].startswith("First real progress")


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
    assert "| normal | 8/8 | 2 | 25.0% |" in markdown(analysis)
