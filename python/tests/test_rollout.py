"""Checks for reusable fixed-policy rollout collection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from conftest import OPPONENT
from orrery.env_client import (
    BODY_FEATURES,
    BODY_SLOTS,
    END_GAME_FINISHED,
    END_LEARNER_ELIMINATED,
    END_ONGOING,
    PLAYER_FEATURES,
    PLAYER_SLOTS,
    EnvClient,
    Episode,
    Observation,
    StepResult,
)
from orrery.model import (
    ObservationBatch,
    OrbitPolicy,
    PolicyOutput,
    SampledActionBatch,
    policy_device,
    run_policy,
)
from orrery.ppo import joint_log_prob
from orrery.profiling import PhaseProfiler
from orrery.rollout import (
    CollectedRollout,
    RolloutBatch,
    _ordered_reset_episodes,
    _ordered_results,
    _require_generator_device,
    _transfer_collection_payload,
    _validate_episode_identity,
    calculate_bootstrap_values,
    collect_transition,
    collect_rollout,
)


@pytest.mark.parametrize(
    ("flags", "expected_passed"),
    [
        ((True, True, True), True),
        ((True, False, True), False),
    ],
)
def test_collection_payload_transfers_actions_and_flags_together(
    flags: tuple[bool, ...],
    expected_passed: bool,
) -> None:
    """One flat payload must preserve action rows and the validation decision."""

    launches = torch.zeros((2, BODY_SLOTS), dtype=torch.bool)
    launches[0, 0] = True
    launches[1, 3] = True
    targets = torch.zeros((2, BODY_SLOTS), dtype=torch.long)
    targets[0, 0] = 5
    targets[1, 3] = 7
    actions = SampledActionBatch(launches=launches, targets=targets)

    packed, passed = _transfer_collection_payload(
        actions,
        torch.tensor(flags, dtype=torch.bool),
    )

    assert packed[0] == (5, *([-1] * (BODY_SLOTS - 1)))
    assert packed[1] == (-1, -1, -1, 7, *([-1] * (BODY_SLOTS - 4)))
    assert passed is expected_passed


class _MarkerPolicy(nn.Module):
    """Use one observation feature directly as a predictable critic value."""

    def __init__(self) -> None:
        super().__init__()
        # A real parameter proves @no_grad collection disconnects stored data
        # from the policy even though the outputs normally depend on it.
        self.launch_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, observation: ObservationBatch) -> PolicyOutput:
        """Return one optional launch and the state's marker as its value."""

        launch_logits = self.launch_logit.expand_as(observation.owned_mask).masked_fill(
            ~observation.launchable_mask,
            -torch.inf,
        )
        target_logits = torch.zeros(
            (
                observation.batch_size,
                BODY_SLOTS,
                BODY_SLOTS,
            ),
            device=observation.bodies.device,
        ).masked_fill(~observation.target_mask, -torch.inf)
        return PolicyOutput(
            launch_logits=launch_logits,
            target_logits=target_logits,
            values=observation.bodies[:, 0, 0] + self.launch_logit * 0.0,
        )


class _BoundaryEnvironment:
    """Script two fake environment slots through every rollout boundary.

    This is a small test double, not the Go Orrery environment. It avoids
    game rules and returns predetermined ``StepResult`` values so the test can
    isolate ``collect_rollout()``. Each observation contains a visible marker
    such as 10, 11, or 20; ``_MarkerPolicy`` reads that marker as its critic
    value, making an incorrect bootstrap or row assignment easy to see.

    Across three collection steps, environment 1 truncates and resets first,
    environment 0 terminates and resets second, and the replacement episodes
    then continue to the rollout boundary. This single sequence checks that
    the collector keeps the two slots independent, bootstraps truncations but
    not true terminals, and stores rows in stable environment-ID order.
    """

    REPLAY_METADATA = {
        "engine": "test-engine",
        "schema": "obs-v1",
        "generator": "boundary-test-v1",
        "opponent": "scripted-test-v1",
    }

    def __init__(self, initial: Sequence[Episode]) -> None:
        # ``active`` is the episode currently occupying each reusable
        # environment slot. A boundary replaces only its own slot; the other
        # slot must continue its existing episode without being reset.
        self.metadata = dict(self.REPLAY_METADATA)
        self.active = {episode.env_id: episode for episode in initial}
        # One call to step() represents one collection turn for both slots.
        # This index selects the corresponding scripted row below.
        self.step_index = 0
        # Recording reset requests lets the test prove that truncation resets
        # slot 1 first and termination resets slot 0 on the following turn.
        self.reset_calls: list[tuple[int, ...]] = []

    def step_packed(
        self,
        action_targets: Mapping[int, Sequence[int]],
    ) -> list[StepResult]:
        """Return the next scripted result for both environment slots.

        Packed targets are accepted to match the collection interface, but this
        fixture does not simulate them. The test is about rollout storage and
        episode boundaries, not whether an Orrery action wins a game.
        """

        # The collector should send exactly one action list for each active
        # environment ID, regardless of the order in which episodes arrived.
        assert tuple(sorted(action_targets)) == (0, 1)
        assert all(len(row) == BODY_SLOTS for row in action_targets.values())
        transitions = (
            # At turn zero, environment 1 truncates. Its returned marker 11
            # must supply the bootstrap before that slot resets to marker 20.
            {
                0: (1.0, 0.0, False, False, 1),
                1: (11.0, 0.0, False, True, 1),
            },
            # Next, environment 0 truly terminates. Its marker 2 must not be
            # bootstrapped; environment 1 continues independently.
            {
                0: (2.0, 1.0, True, False, 2),
                1: (21.0, 0.0, False, False, 1),
            },
            # Both reset histories are now visible and both games continue at
            # the ordinary final rollout boundary.
            {
                0: (31.0, 0.0, False, False, 1),
                1: (22.0, 0.0, False, False, 2),
            },
        )
        row = transitions[self.step_index]
        self.step_index += 1
        results: list[StepResult] = []
        for env_id in (0, 1):
            episode = self.active[env_id]
            marker, reward, terminated, truncated, turn = row[env_id]
            # Preserve the current episode's identity while attaching this
            # turn's scripted observation, reward, and boundary flags.
            result = StepResult(
                env_id=env_id,
                seed=episode.seed,
                seat=episode.seat,
                explicit=episode.explicit,
                observation=_observation(marker),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                end_reason=END_GAME_FINISHED if terminated else END_ONGOING,
                turn=turn,
            )
            results.append(result)
            # A continuing StepResult becomes the slot's next active state.
            # A boundary stays in place only until collect_rollout() calls
            # reset() and installs the replacement episode below.
            if not (terminated or truncated):
                self.active[env_id] = result

        # Deliberately reverse transport order. collect_rollout must restore
        # the stable ascending B-axis before storing this turn.
        return list(reversed(results))

    def reset(self, ids: Sequence[int]) -> list[Episode]:
        """Install the predetermined new episode for the requested slot only.

        Marker 20 starts environment 1's replacement after its truncation;
        marker 30 starts environment 0's replacement after its termination.
        Their new seeds make accidental reuse of the finished episode visible
        in the collected replay fields as well as in critic values.
        """

        requested = tuple(ids)
        self.reset_calls.append(requested)
        replacements = {
            (1,): Episode(
                env_id=1,
                seed=201,
                seat=1,
                explicit=True,
                observation=_observation(20.0),
            ),
            (0,): Episode(
                env_id=0,
                seed=101,
                seat=0,
                explicit=True,
                observation=_observation(30.0),
            ),
        }
        episode = replacements[requested]
        self.active[episode.env_id] = episode
        return [episode]


def test_collect_rollout_preserves_boundaries_order_and_selective_resets() -> None:
    """Every stored row and bootstrap must belong to the right game fragment."""

    initial = (
        Episode(0, 100, 0, False, _observation(0.0)),
        Episode(1, 200, 1, False, _observation(10.0)),
    )
    environment = _BoundaryEnvironment(initial)
    policy = _MarkerPolicy().eval()

    collected = collect_rollout(
        environment,
        policy,
        # Supplying reversed input also proves that environment IDs, rather
        # than caller sequence order, define the stable B axis.
        tuple(reversed(initial)),
        steps=3,
        gamma=1.0,
        lam=1.0,
        generator=torch.Generator().manual_seed(14),
    )
    rollout = collected.batch

    assert rollout.time_steps == 3
    assert rollout.environment_count == 2
    assert rollout.transition_count == 6
    assert [
        indices.tolist()
        for indices in rollout.collection_groups()
    ] == [[0, 1], [2, 3], [4, 5]]
    assert rollout.environment_ids == (0, 1)
    assert environment.reset_calls == [(1,), (0,)]
    assert [episode.env_id for episode in collected.next_episodes] == [0, 1]
    assert [episode.seed for episode in collected.next_episodes] == [101, 201]

    # Markers independently witness the flattened time-major order. In
    # particular, reset state 20 replaces only environment 1 on turn one, and
    # reset state 30 replaces only environment 0 on turn two.
    markers = rollout.time_major(rollout.observations.bodies[:, 0, 0])
    next_markers = rollout.time_major(rollout.next_observations.bodies[:, 0, 0])
    torch.testing.assert_close(
        markers,
        torch.tensor([[0.0, 10.0], [1.0, 20.0], [30.0, 21.0]]),
    )
    torch.testing.assert_close(
        next_markers,
        torch.tensor([[1.0, 11.0], [2.0, 21.0], [31.0, 22.0]]),
    )

    torch.testing.assert_close(
        rollout.time_major(rollout.rewards),
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
    )
    assert torch.equal(
        rollout.time_major(rollout.terminated),
        torch.tensor([[False, False], [True, False], [False, False]]),
    )
    assert torch.equal(
        rollout.time_major(rollout.truncated),
        torch.tensor([[False, True], [False, False], [False, False]]),
    )

    # With value(marker), gamma=1, and lambda=1:
    # - truncation at [0,1] uses marker 11 but stops recurrence into its reset;
    # - termination at [1,0] uses zero future value;
    # - both final rows bootstrap from markers 31 and 22.
    torch.testing.assert_close(
        rollout.time_major(rollout.next_values),
        torch.tensor([[1.0, 11.0], [0.0, 21.0], [31.0, 22.0]]),
    )
    torch.testing.assert_close(
        rollout.time_major(rollout.advantages),
        torch.tensor([[1.0, 1.0], [0.0, 2.0], [1.0, 1.0]]),
    )
    torch.testing.assert_close(
        rollout.time_major(rollout.return_targets),
        torch.tensor([[1.0, 11.0], [1.0, 22.0], [31.0, 22.0]]),
    )

    assert torch.equal(
        rollout.time_major(rollout.seeds),
        torch.tensor([[100, 200], [100, 201], [101, 201]]),
    )
    assert torch.equal(
        rollout.time_major(rollout.turns),
        torch.tensor([[1, 1], [2, 1], [1, 2]]),
    )
    assert torch.equal(
        rollout.time_major(rollout.explicit),
        torch.tensor([[False, False], [False, True], [True, True]]),
    )
    assert rollout.replay_metadata == environment.metadata
    environment.metadata["engine"] = "mutated-after-collection"
    assert rollout.replay_metadata["engine"] == "test-engine"
    with pytest.raises(TypeError):
        rollout.replay_metadata["engine"] = "cannot-rewrite"  # type: ignore[index]

    # Collection must not create a graph or modify a policy parameter. The
    # update step owns both of those responsibilities later.
    assert policy.launch_logit.grad is None
    assert float(policy.launch_logit) == 0.0
    for value in vars(rollout).values():
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad


class _EliminationEnvironment:
    """Eliminate environment 0's learner on the first turn while 1 plays on.

    In a four-player game the learner can be knocked out while other players
    continue. Go then ends the learner's episode with end reason 2 and reward
    -1, and returns an observation in which the learner owns nothing.
    """

    def __init__(self, initial: Sequence[Episode]) -> None:
        self.metadata = dict(_BoundaryEnvironment.REPLAY_METADATA)
        self.active = {episode.env_id: episode for episode in initial}
        self.step_index = 0
        self.reset_calls: list[tuple[int, ...]] = []

    def step_packed(
        self,
        action_targets: Mapping[int, Sequence[int]],
    ) -> list[StepResult]:
        assert tuple(sorted(action_targets)) == (0, 1)
        results: list[StepResult] = []
        for env_id in (0, 1):
            episode = self.active[env_id]
            eliminated = self.step_index == 0 and env_id == 0
            if eliminated:
                # The eliminated learner's last state: no owned body, so no
                # source can launch and no target row is open.
                observation = _observation(2.0)
                observation.owned_mask[:] = False
                observation.target_mask[:] = False
            else:
                observation = _observation(float(10 * (env_id + 1) + self.step_index + 1))
            result = StepResult(
                env_id=env_id,
                seed=episode.seed,
                seat=episode.seat,
                explicit=episode.explicit,
                observation=observation,
                reward=-1.0 if eliminated else 0.0,
                terminated=eliminated,
                truncated=False,
                end_reason=END_LEARNER_ELIMINATED if eliminated else END_ONGOING,
                turn=self.step_index + 1,
            )
            results.append(result)
            if not eliminated:
                self.active[env_id] = result
        self.step_index += 1
        return results

    def reset(self, ids: Sequence[int]) -> list[Episode]:
        requested = tuple(ids)
        self.reset_calls.append(requested)
        assert requested == (0,)
        episode = Episode(
            env_id=0, seed=101, seat=3, explicit=False, observation=_observation(30.0)
        )
        self.active[0] = episode
        return [episode]


def test_learner_elimination_ends_only_its_episode_with_a_terminal_loss() -> None:
    """End reason 2 is a true terminal: -1, no bootstrap, and a selective reset.

    Environment 0's learner is eliminated on turn one while environment 1 keeps
    playing. The completed transition must keep the eliminated state (marker
    2), not the replacement episode's first state (marker 30), and its future
    value must be zero rather than the critic's estimate of either state.
    """

    initial = (
        Episode(0, 100, 2, False, _observation(0.0)),
        Episode(1, 200, 1, False, _observation(10.0)),
    )
    environment = _EliminationEnvironment(initial)

    rollout = collect_rollout(
        environment,
        _MarkerPolicy().eval(),
        initial,
        steps=2,
        gamma=1.0,
        lam=1.0,
        generator=torch.Generator().manual_seed(15),
    ).batch

    assert environment.reset_calls == [(0,)]
    torch.testing.assert_close(
        rollout.time_major(rollout.rewards),
        torch.tensor([[-1.0, 0.0], [0.0, 0.0]]),
    )
    assert torch.equal(
        rollout.time_major(rollout.terminated),
        torch.tensor([[True, False], [False, False]]),
    )
    # The completed transition ends in the eliminated state, and the
    # replacement starts the next row of the same slot.
    next_markers = rollout.time_major(rollout.next_observations.bodies[:, 0, 0])
    markers = rollout.time_major(rollout.observations.bodies[:, 0, 0])
    assert float(next_markers[0, 0]) == 2.0
    assert float(markers[1, 0]) == 30.0
    assert not bool(rollout.time_major(rollout.next_observations.owned_mask)[0, 0].any())
    # A terminal has no future value, so the eliminated row's return is just -1.
    assert float(rollout.time_major(rollout.next_values)[0, 0]) == 0.0
    assert float(rollout.time_major(rollout.return_targets)[0, 0]) == -1.0
    # The replacement episode's identity appears only from its own first row.
    assert rollout.time_major(rollout.seeds).tolist() == [[100, 200], [101, 200]]
    assert rollout.time_major(rollout.seats).tolist() == [[2, 1], [3, 1]]


def test_profiling_does_not_change_seeded_rollout_tensors() -> None:
    """Timing boundaries must observe a rollout without changing its results."""

    initial = (
        Episode(0, 100, 0, False, _observation(0.0)),
        Episode(1, 200, 1, False, _observation(10.0)),
    )

    # Both collectors start from independent copies of the same scripted games
    # and the same sampling seed. The second collector adds CPU phase timing;
    # every stored tensor must remain bit-identical to the ordinary path.
    ordinary = collect_rollout(
        _BoundaryEnvironment(initial),
        _MarkerPolicy().eval(),
        initial,
        steps=3,
        gamma=1.0,
        lam=1.0,
        generator=torch.Generator().manual_seed(14),
    )
    profiler = PhaseProfiler("cpu")
    profiled = collect_rollout(
        _BoundaryEnvironment(initial),
        _MarkerPolicy().eval(),
        initial,
        steps=3,
        gamma=1.0,
        lam=1.0,
        generator=torch.Generator().manual_seed(14),
        profiler=profiler,
    )

    for ordinary_tensor, profiled_tensor in zip(
        _rollout_tensors(ordinary.batch),
        _rollout_tensors(profiled.batch),
        strict=True,
    ):
        assert torch.equal(ordinary_tensor, profiled_tensor)
    assert ordinary.batch.environment_ids == profiled.batch.environment_ids
    assert ordinary.batch.time_steps == profiled.batch.time_steps
    assert ordinary.batch.replay_metadata == profiled.batch.replay_metadata
    assert [
        (episode.env_id, episode.seed, episode.seat, episode.explicit)
        for episode in ordinary.next_episodes
    ] == [
        (episode.env_id, episode.seed, episode.seat, episode.explicit)
        for episode in profiled.next_episodes
    ]

    # A non-empty snapshot also proves the profiled call crossed the intended
    # policy, sampling, transfer, bootstrap, and bulk-build boundaries.
    phase_seconds = profiler.snapshot_and_reset()
    for phase in (
        "policy",
        "sampling",
        "log_probability",
        "action_encode",
        "tensor_transfer",
        "bootstrap",
        "rollout_build",
    ):
        assert phase_seconds[phase] > 0.0


def _rollout_tensors(rollout: RolloutBatch) -> tuple[torch.Tensor, ...]:
    """Return every tensor whose bytes profiling must leave unchanged."""

    # Observation and action dataclasses hold tensors inside RolloutBatch, so
    # list them alongside its direct tensor fields rather than checking only
    # the top-level values returned by vars().
    return (
        *vars(rollout.observations).values(),
        *vars(rollout.next_observations).values(),
        *vars(rollout.actions).values(),
        rollout.old_log_probabilities,
        rollout.values,
        rollout.next_values,
        rollout.rewards,
        rollout.terminated,
        rollout.truncated,
        rollout.advantages,
        rollout.return_targets,
        rollout.seeds,
        rollout.seats,
        rollout.explicit,
        rollout.turns,
    )


def test_collect_rollout_reuses_returned_observation_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continuing slots reuse Go's result instead of rebuilding it next turn."""

    initial = (
        Episode(0, 100, 0, False, _observation(0.0)),
        Episode(1, 200, 1, False, _observation(10.0)),
    )
    original = ObservationBatch.from_observations.__func__
    converted_markers: list[list[float]] = []

    def tracked_conversion(
        cls: type[ObservationBatch],
        observations: Sequence[Observation],
        *,
        device: torch.device | str | None = None,
    ) -> ObservationBatch:
        # Record which NumPy rows cross into Torch at every call. A reset should
        # convert only its replacement row; rebuilding the complete current
        # batch would show [1, 20] and [30, 21] here instead.
        converted_markers.append(
            [float(observation.bodies[0, 0]) for observation in observations]
        )
        return original(cls, observations, device=device)

    monkeypatch.setattr(
        ObservationBatch,
        "from_observations",
        classmethod(tracked_conversion),
    )
    collect_rollout(
        _BoundaryEnvironment(initial),
        _MarkerPolicy().eval(),
        initial,
        steps=3,
        gamma=1.0,
        lam=1.0,
        generator=torch.Generator().manual_seed(14),
    )

    assert converted_markers == [
        [0.0, 10.0],
        [1.0, 11.0],
        [20.0],
        [2.0, 21.0],
        [30.0],
        [31.0, 22.0],
    ]


def test_shared_transition_matches_the_first_stored_rollout_row() -> None:
    """The teaching seam and full collector must retain identical step artifacts."""

    initial = (
        Episode(0, 100, 0, False, _observation(0.0)),
        Episode(1, 200, 1, False, _observation(10.0)),
    )
    direct = collect_transition(
        _BoundaryEnvironment(initial),
        _MarkerPolicy().eval(),
        initial,
        generator=torch.Generator().manual_seed(14),
    )
    rollout = collect_rollout(
        _BoundaryEnvironment(initial),
        _MarkerPolicy().eval(),
        initial,
        steps=1,
        generator=torch.Generator().manual_seed(14),
    ).batch

    assert torch.equal(direct.observation.bodies, rollout.observations.bodies)
    assert torch.equal(
        direct.next_observation.bodies,
        rollout.next_observations.bodies,
    )
    assert torch.equal(direct.actions.launches, rollout.actions.launches)
    assert torch.equal(direct.actions.targets, rollout.actions.targets)
    assert torch.equal(
        direct.old_log_probabilities,
        rollout.old_log_probabilities,
    )
    assert torch.equal(direct.policy_output.values, rollout.values)
    assert [result.reward for result in direct.results] == rollout.rewards.tolist()
    assert [result.terminated for result in direct.results] == (
        rollout.terminated.tolist()
    )
    assert [result.truncated for result in direct.results] == (
        rollout.truncated.tolist()
    )


def test_failed_deferred_validation_never_calls_the_environment() -> None:
    """Malformed policy scores must stop before the Go protocol boundary."""

    class NonfiniteLaunchPolicy(nn.Module):
        """Return NaN launch scores so the deferred finite-score flag fails."""

        def forward(self, observation: ObservationBatch) -> PolicyOutput:
            return PolicyOutput(
                launch_logits=torch.full(
                    (observation.batch_size, BODY_SLOTS),
                    torch.nan,
                    device=observation.bodies.device,
                ),
                target_logits=torch.zeros(
                    (observation.batch_size, BODY_SLOTS, BODY_SLOTS),
                    device=observation.bodies.device,
                ),
                values=torch.zeros(
                    observation.batch_size,
                    device=observation.bodies.device,
                ),
            )

    initial = (Episode(0, 100, 0, False, _observation(0.0)),)
    environment = _BoundaryEnvironment(initial)

    with pytest.raises(
        ValueError,
        match="launchable sources must have finite launch logits",
    ):
        collect_transition(
            environment,
            NonfiniteLaunchPolicy().eval(),
            initial,
            generator=torch.Generator().manual_seed(14),
        )

    # step_index changes only inside step_packed(). Keeping it at zero proves
    # the temporary safe action was rejected before any request reached Go.
    assert environment.step_index == 0


def test_calculate_bootstrap_values_detaches_direct_calls() -> None:
    """A standalone bootstrap calculation must not retain a critic graph."""

    policy = _MarkerPolicy().eval()
    next_observations = ObservationBatch.from_observations([_observation(3.0)])
    bootstrap = calculate_bootstrap_values(
        policy,
        next_observations,
        values=torch.tensor([[0.0]], requires_grad=True),
        terminated=torch.tensor([[False]]),
        truncated=torch.tensor([[False]]),
    )

    assert torch.equal(bootstrap, torch.tensor([[3.0]]))
    assert not bootstrap.requires_grad


def test_collect_rollout_round_trips_probabilities_through_real_go(
    env_command: tuple[str, ...],
) -> None:
    """A real Go rollout should retain exactly recomputable action data."""

    torch.manual_seed(1401)
    policy = OrbitPolicy().eval()
    generator = torch.Generator().manual_seed(1402)

    with EnvClient(env_command, players=2) as environment:
        metadata = environment.create(2, (0, 1_000_000), OPPONENT)
        initial = environment.reset([0, 1])
        collected = collect_rollout(
            environment,
            policy,
            initial,
            steps=2,
            generator=generator,
        )

        rollout = collected.batch
        assert rollout.transition_count == 4
        assert rollout.environment_ids == (0, 1)
        assert rollout.replay_metadata == metadata
        assert tuple(episode.env_id for episode in collected.next_episodes) == (0, 1)
        assert environment.error_count == 0

        recomputed_rows: list[torch.Tensor] = []
        with torch.no_grad():
            for indices in rollout.collection_groups():
                observation = rollout.select_observations(indices)
                output = policy(observation)
                recomputed_rows.append(
                    joint_log_prob(
                        output,
                        rollout.select_actions(indices),
                        observation.launchable_mask,
                        observation.target_mask,
                    )
                )
        # Reproduce the same B=2 calls collection made instead of flattening
        # all turns into a new batch shape with different reduction rounding.
        assert torch.equal(
            torch.cat(recomputed_rows),
            rollout.old_log_probabilities,
        )
        assert torch.isfinite(rollout.advantages).all()
        assert torch.isfinite(rollout.return_targets).all()


def test_rollout_batch_rejects_corrupt_storage_and_replay_identity() -> None:
    """Public rollout construction must reject data unsafe for optimization."""

    collected, _, _ = _collect_boundary_rollout()
    rollout = collected.batch
    carrying_gradient = rollout.advantages.clone().requires_grad_()
    incomplete_metadata = {"engine": "test-engine"}
    empty_engine = dict(rollout.replay_metadata)
    empty_engine["engine"] = ""
    cases = (
        ({"time_steps": 0}, "time_steps must be a positive integer"),
        (
            {"environment_ids": (1, 0)},
            "environment_ids must be unique and in ascending order",
        ),
        (
            {"rewards": rollout.rewards[:-1]},
            r"rewards has shape .* want \(6,\)",
        ),
        (
            {"terminated": rollout.terminated.to(torch.float32)},
            "terminated must use torch.bool",
        ),
        ({"seeds": rollout.seeds.to(torch.float32)}, "seeds must use torch.long"),
        (
            {"advantages": carrying_gradient},
            "rollout tensors must be detached",
        ),
        (
            {"replay_metadata": incomplete_metadata},
            "replay_metadata is missing required fields",
        ),
        (
            {"replay_metadata": empty_engine},
            "field 'engine' must be a non-empty string",
        ),
    )

    for changes, error in cases:
        with pytest.raises((TypeError, ValueError), match=error):
            replace(rollout, **changes)


@pytest.mark.parametrize(
    ("arguments", "error"),
    (
        ({"steps": 0}, "steps must be a positive integer"),
        ({"steps": 1, "gamma": -0.1}, r"gamma must be in \[0, 1\]"),
        ({"steps": 1, "gamma": 1.1}, r"gamma must be in \[0, 1\]"),
        ({"steps": 1, "lam": -0.1}, r"lam must be in \[0, 1\]"),
        ({"steps": 1, "lam": 1.1}, r"lam must be in \[0, 1\]"),
    ),
)
def test_collect_rollout_rejects_invalid_math_before_stepping(
    arguments: dict[str, float | int],
    error: str,
) -> None:
    """Invalid collection math must fail before mutating an environment."""

    initial = (
        Episode(0, 100, 0, False, _observation(0.0)),
        Episode(1, 200, 1, False, _observation(10.0)),
    )
    environment = _BoundaryEnvironment(initial)

    with pytest.raises(ValueError, match=error):
        collect_rollout(environment, _MarkerPolicy().eval(), initial, **arguments)

    assert environment.step_index == 0
    assert environment.reset_calls == []


def test_collect_rollout_rejects_training_mode_policy_before_stepping() -> None:
    """A stochastic training-mode policy cannot define stable old probabilities."""

    initial = (
        Episode(0, 100, 0, False, _observation(0.0)),
        Episode(1, 200, 1, False, _observation(10.0)),
    )
    environment = _BoundaryEnvironment(initial)

    with pytest.raises(ValueError, match="policy must be in evaluation mode"):
        collect_rollout(environment, _MarkerPolicy(), initial, steps=1)

    assert environment.step_index == 0


def test_generator_and_policy_devices_must_be_unambiguous() -> None:
    """Device mistakes should fail at the rollout boundary, not during sampling."""

    # A configured device can omit its index even though Torch later reports a
    # concrete index for tensors on that device. Equivalent backend forms must
    # not stop rollout collection.
    _require_generator_device(torch.Generator(), torch.device("cpu:0"))

    with pytest.raises(ValueError, match="generator uses cpu, but policy uses cuda"):
        _require_generator_device(torch.Generator(), torch.device("cuda"))

    mixed = nn.Module()
    mixed.register_parameter("cpu_parameter", nn.Parameter(torch.zeros(1)))
    mixed.register_buffer("meta_buffer", torch.empty(1, device="meta"))
    with pytest.raises(ValueError, match="parameters and buffers span devices"):
        policy_device(mixed)


def test_policy_and_environment_results_must_match_rollout_contracts() -> None:
    """Wrong model outputs and environment identities must fail immediately."""

    observation = ObservationBatch.from_observations([_observation(0.0)])
    wrong_policy = nn.Identity().eval()
    with pytest.raises(TypeError, match="policy returned ObservationBatch"):
        run_policy(wrong_policy, observation)

    episode = Episode(0, 100, 0, False, _observation(0.0))
    result = _step_result(episode, marker=1.0)
    duplicate = _step_result(episode, marker=2.0)
    with pytest.raises(RuntimeError, match="duplicate step-result IDs"):
        _ordered_results([result, duplicate], (0, 1))
    with pytest.raises(
        RuntimeError,
        match=r"step-result IDs \(0,\), want \(0, 1\)",
    ):
        _ordered_results([result], (0, 1))

    wrong_reset = Episode(0, 101, 0, True, _observation(3.0))
    with pytest.raises(
        RuntimeError,
        match=r"reset IDs \(0,\), want finished IDs \(1,\)",
    ):
        _ordered_reset_episodes([wrong_reset], (1,))

    changed_seed = replace(result, seed=999)
    with pytest.raises(RuntimeError, match="changed episode identity"):
        _validate_episode_identity([episode], [changed_seed])
    conflicting_flags = replace(result, terminated=True, truncated=True)
    with pytest.raises(RuntimeError, match="returned both boundary flags"):
        _validate_episode_identity([episode], [conflicting_flags])


def _collect_boundary_rollout() -> tuple[
    CollectedRollout,
    _BoundaryEnvironment,
    _MarkerPolicy,
]:
    """Collect the shared deterministic boundary fixture for guard tests."""

    initial = (
        Episode(0, 100, 0, False, _observation(0.0)),
        Episode(1, 200, 1, False, _observation(10.0)),
    )
    environment = _BoundaryEnvironment(initial)
    policy = _MarkerPolicy().eval()
    collected = collect_rollout(
        environment,
        policy,
        initial,
        steps=3,
        gamma=1.0,
        lam=1.0,
        generator=torch.Generator().manual_seed(14),
    )
    return collected, environment, policy


def _step_result(
    episode: Episode,
    *,
    marker: float,
) -> StepResult:
    """Build one ordinary result that retains an episode's replay identity."""

    return StepResult(
        env_id=episode.env_id,
        seed=episode.seed,
        seat=episode.seat,
        explicit=episode.explicit,
        observation=_observation(marker),
        reward=0.0,
        terminated=False,
        truncated=False,
        end_reason=END_ONGOING,
        turn=1,
    )


def _observation(marker: float) -> Observation:
    """Build one legal two-body state carrying a test-visible marker."""

    bodies = np.zeros((BODY_SLOTS, BODY_FEATURES), dtype=np.float32)
    bodies[0, 0] = marker
    players = np.zeros((PLAYER_SLOTS, PLAYER_FEATURES), dtype=np.float32)
    body_mask = np.zeros(BODY_SLOTS, dtype=np.bool_)
    body_mask[:2] = True
    owned_mask = np.zeros(BODY_SLOTS, dtype=np.bool_)
    owned_mask[0] = True
    player_mask = np.zeros(PLAYER_SLOTS, dtype=np.bool_)
    player_mask[:2] = True
    target_mask = np.zeros((BODY_SLOTS, BODY_SLOTS), dtype=np.bool_)
    target_mask[0, 1] = True
    slot_body_id = np.full(BODY_SLOTS, -1, dtype=np.int32)
    slot_body_id[:2] = (1, 2)
    return Observation(
        bodies=bodies,
        players=players,
        body_mask=body_mask,
        owned_mask=owned_mask,
        player_mask=player_mask,
        target_mask=target_mask,
        slot_body_id=slot_body_id,
    )
