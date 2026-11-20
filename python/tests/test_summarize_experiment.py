"""Tests for the training-opponent experiment summary."""

from __future__ import annotations

import pytest

from scripts.summarize_experiment import reached_two_planets_by


@pytest.mark.parametrize(
    ("game", "counts"),
    (
        # The deadline turn itself counts.
        ({"first_two_planets_turn": 50, "turns": 500}, True),
        # One turn late does not.
        ({"first_two_planets_turn": 51, "turns": 500}, False),
        # Expansion counts however the game ended: reaching two planets on
        # turn 20 still counts when the game was over by turn 40.
        ({"first_two_planets_turn": 20, "turns": 40}, True),
        # A game that never held two planets at once never counts.
        ({"first_two_planets_turn": None, "turns": 30}, False),
    ),
)
def test_reached_two_planets_by_turn_50(game: dict[str, int | None], counts: bool) -> None:
    assert reached_two_planets_by([game]) == (int(counts), 1)


def test_reached_two_planets_by_counts_every_game() -> None:
    games = [
        {"first_two_planets_turn": 12},
        {"first_two_planets_turn": None},
        {"first_two_planets_turn": 50},
        {"first_two_planets_turn": 77},
    ]

    assert reached_two_planets_by(games) == (2, 4)
