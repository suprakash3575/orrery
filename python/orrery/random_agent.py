"""Uniform legal-action controller used to verify the Python boundary."""

from __future__ import annotations

import numpy as np

from .env_client import BODY_SLOTS, Observation, SlotAction


def sample(obs: Observation, rng: np.random.Generator) -> list[SlotAction]:
    """Sample one legal launch/no-launch decision for every body slot.

    Each owned source independently launches with probability one half. When it
    launches, its target is uniform over the source row's legal target mask.
    Sources with no legal target always submit no-launch.
    """

    actions = [SlotAction() for _ in range(BODY_SLOTS)]
    for source in np.flatnonzero(obs.owned_mask):
        targets = np.flatnonzero(obs.target_mask[source])
        if targets.size == 0 or rng.integers(2) == 0:
            continue
        target = int(rng.choice(targets))
        actions[int(source)] = SlotAction(launch=True, target=target)
    return actions
