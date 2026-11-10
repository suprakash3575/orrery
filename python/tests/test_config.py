"""Run-configuration contract tests for the rl-v2 training runs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from orrery.config import (
    CONTRACT,
    SeedRange,
    load_run_config,
    run_config_from_mapping,
)
from orrery.jsonio import json_value


CONFIGS = Path(__file__).resolve().parents[2] / "configs"
CONFIG = CONFIGS / "rl-v2-2p.toml"
FOUR_PLAYER_CONFIG = CONFIGS / "rl-v2-4p.toml"


def test_two_player_config_captures_the_training_contract() -> None:
    config = load_run_config(CONFIG)

    assert config.contract == CONTRACT == "rl-v2"
    assert config.players == 2
    assert config.device == "cuda"
    assert config.learner_seed == 1501
    assert config.environment_count == 512
    assert config.rollout_steps == 64
    assert config.batch_size == 32_768
    assert config.minibatch_size == 4_096
    assert config.ppo_epochs == 4
    assert config.weight_decay == 0.01
    assert config.gamma == 0.999
    assert config.gae_lambda == 0.95
    assert config.checkpoint_every_updates == 50
    assert config.review_steps == 25_000_000
    assert config.initial_step_ceiling == 50_000_000
    assert config.cumulative_step_ceiling == 500_000_000
    assert config.release_validation_seeds == 500
    assert config.release_required_wins == 900
    assert config.release_confidence_level == 0.95
    assert config.release_minimum_lower_bound == 0.85
    assert config.held_out_games == 2_000
    assert config.held_out_required_wins == 1_800
    assert config.training_seeds == SeedRange(0, 1_000_000)
    assert config.validation_seeds == SeedRange(1_000_000, 1_010_000)
    assert config.evaluation_pool_names == (
        "eval-1", "eval-2", "eval-3", "eval-4", "eval-5"
    )
    assert config.training_schedule == "train-v2"
    assert config.protocol_version == "env-v3"
    assert config.observation_schema == "obs-v2"
    assert config.generator_version == "generator-v2"
    assert config.checkpoint_format == "checkpoint-v2"


def test_four_player_config_differs_only_in_players_generator_and_win_counts() -> None:
    """Both files share every training setting and seed partition.

    The required wins double because every seed is played from four seats
    instead of two, which keeps the same win fractions: 320/400, 1,800/2,000,
    and a derived 3,600/4,000 held out.
    """

    two_player = load_run_config(CONFIG)
    four_player = load_run_config(FOUR_PLAYER_CONFIG)

    assert four_player == replace(
        two_player,
        players=4,
        generator_version="generator-v2-4p",
        pipeline_required_wins=320,
        release_required_wins=1_800,
    )
    assert (four_player.pipeline_games, four_player.release_games) == (400, 2_000)
    assert (four_player.held_out_required_wins, four_player.held_out_games) == (
        3_600,
        4_000,
    )


def test_config_serializes_nested_seed_ranges() -> None:
    serialized = load_run_config(CONFIG).to_dict()

    assert serialized["players"] == 2
    assert serialized["training_seeds"] == {"start": 0, "end": 1_000_000}
    assert serialized["evaluation_seeds"][0] == {
        "start": 1_010_000,
        "end": 1_011_000,
    }
    # Compatibility is described by explicit semantic fields. It does not
    # depend on Git being installed or on a particular source-tree layout.
    assert "git_commit" not in serialized


@pytest.mark.parametrize("path", [CONFIG, FOUR_PLAYER_CONFIG])
def test_recorded_config_round_trips_through_the_mapping_loader(path: Path) -> None:
    """A JSON record, as a checkpoint envelope stores it, rebuilds the same config."""

    config = load_run_config(path)

    assert run_config_from_mapping(json_value(config.to_dict())) == config


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"contract": "rl-v1"}, "contract must be 'rl-v2'"),
        (
            {"environment_count": 64, "rollout_steps": 512},
            "requires environment_count=512 and rollout_steps=64",
        ),
        ({"players": 3}, "players must be 2 or 4, got 3"),
        ({"players": True}, "players must be 2 or 4, got True"),
        (
            {"players": 4},
            "generator_version must be 'generator-v2-4p' for rl-v2 with 4 players",
        ),
        (
            {"generator_version": "generator-v2-4p"},
            "generator_version must be 'generator-v2' for rl-v2 with 2 players",
        ),
    ),
)
def test_run_config_rejects_malformed_contract_identity(
    changes: dict[str, object], message: str
) -> None:
    config = load_run_config(CONFIG)

    with pytest.raises(ValueError, match=message):
        replace(config, **changes)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("device", "mps", "device must be"),
        ("minibatch_size", 5_000, "divisible"),
        ("review_steps", 50_000_000, "review < initial"),
        ("pipeline_required_wins", 201, "pipeline game count"),
        ("release_required_wins", 1_001, "release game count"),
        ("release_confidence_level", 1.0, "must be in"),
        ("release_minimum_lower_bound", -0.01, "must be in"),
        ("weight_decay", -0.01, "non-negative"),
        ("launch_entropy_final", 0.06, "must not increase"),
        ("learner_seed", 2**32, "unsigned 32-bit"),
        ("training_schedule", "train-v1", "must be 'train-v2'"),
        # Each removed V1 identity must fail validation rather than start Go
        # or a model with a layout this code no longer speaks.
        ("protocol_version", "env-v2", "must be 'env-v3'"),
        ("observation_schema", "obs-v1", "must be 'obs-v2'"),
        ("generator_version", "generator-v1", "must be 'generator-v2'"),
        # Checkpoints and archives recorded with the CPU evaluator are refused
        # instead of being validated under different decoding rules.
        ("evaluation_policy", "eval-policy-v1", "must be 'eval-policy-v2'"),
        # Runs and checkpoints that trained against heuristic-v1 are refused,
        # so they cannot be resumed or released under the heuristic-v2 gate.
        (
            "training_opponent",
            "heuristic-v1:aggression=1.2,defense=0.1,economy=1,efficiency=1",
            "must be 'heuristic-v2:candidates=4,enemy_bonus=1,margin=0.1'",
        ),
    ),
)
def test_run_config_rejects_inconsistent_values(
    field: str, value: object, message: str
) -> None:
    config = load_run_config(CONFIG)

    with pytest.raises(ValueError, match=message):
        replace(config, **{field: value})


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("pipeline_required_wins", "pipeline game count"),
        ("release_required_wins", "release game count"),
    ),
)
def test_four_player_required_wins_are_capped_at_seeds_times_players(
    field: str, message: str
) -> None:
    """On four players the cap is seeds × 4: 400 and 2,000 games."""

    config = load_run_config(FOUR_PLAYER_CONFIG)
    games = config.pipeline_games if field == "pipeline_required_wins" else config.release_games

    replace(config, **{field: games})
    with pytest.raises(ValueError, match=message):
        replace(config, **{field: games + 1})


def test_run_config_rejects_overlapping_seed_ranges() -> None:
    config = load_run_config(CONFIG)

    with pytest.raises(ValueError, match="must be disjoint"):
        replace(config, validation_seeds=SeedRange(999_999, 1_010_000))


def test_run_config_rejects_shifted_disjoint_seed_ranges() -> None:
    config = load_run_config(CONFIG)

    with pytest.raises(ValueError, match="frozen seed partitions"):
        replace(config, training_seeds=SeedRange(10, 999_999))


@pytest.mark.parametrize(
    ("extra_line", "message"),
    (
        ("unexpected = 1\n", "unknown keys: unexpected"),
        ("", "missing keys: contract"),
    ),
)
def test_loader_rejects_unknown_and_missing_keys(
    tmp_path: Path, extra_line: str, message: str
) -> None:
    text = CONFIG.read_text()
    if not extra_line:
        text = text.replace('contract = "rl-v2"\n', "")
    path = tmp_path / "config.toml"
    path.write_text(text + extra_line)

    with pytest.raises(ValueError, match=message):
        load_run_config(path)
