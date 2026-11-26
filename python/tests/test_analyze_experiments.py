"""The analysis must turn a session folder's results into the right numbers
and answers."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_experiments import analyze, markdown

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
