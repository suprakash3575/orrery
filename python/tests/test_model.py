"""Readable checks for the tensors, transformer, policy heads, and action sampler."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from orrery.env_client import (
    BODY_FEATURES,
    BODY_SLOTS,
    PLAYER_FEATURES,
    PLAYER_SLOTS,
    Observation,
)
from orrery.model import (
    MODEL_DIM,
    ObservationBatch,
    OrbitEncoder,
    OrbitPolicy,
    PolicyOutput,
    main,
    parameter_count,
    policy_validation_flags,
    run_policy_for_collection,
    sample_actions,
    sample_actions_for_collection,
)


def test_observation_batch_stacks_client_observations() -> None:
    """Client observations should become one correctly typed Torch batch."""

    # Arrange two observations with different numbers and feature values so we
    # can prove they remain distinct after batching.
    first = _observation(real_bodies=3, real_players=2, feature_offset=0.0)
    second = _observation(real_bodies=5, real_players=2, feature_offset=10.0)

    # Act by converting the two NumPy observations at the Python-client
    # boundary into the Torch representation consumed by the model.
    batch = ObservationBatch.from_observations([first, second])

    # The first dimension must contain two games; the remaining dimensions must
    # preserve the fixed slots and feature counts defined by the protocol.
    assert batch.bodies.shape == (2, BODY_SLOTS, BODY_FEATURES)
    assert batch.players.shape == (2, PLAYER_SLOTS, PLAYER_FEATURES)
    # Torch uses dtype to describe the kind of value stored in a tensor.
    # Features stay 32-bit decimals and masks stay true/false values.
    assert batch.bodies.dtype == torch.float32
    assert batch.players.dtype == torch.float32
    assert batch.body_mask.dtype == torch.bool
    assert batch.target_mask.dtype == torch.bool
    # Tokenization concatenates bodies before players, so the combined mask
    # must follow that exact order rather than merely contain the same values.
    # torch.equal requires every value and shape to match exactly.
    assert torch.equal(batch.valid_token_mask[:, :BODY_SLOTS], batch.body_mask)
    assert torch.equal(batch.valid_token_mask[:, BODY_SLOTS:], batch.player_mask)
    # This known value came from the second observation. Finding it in batch row
    # 1 proves batching did not reorder or combine the two games.
    assert batch.bodies[1, 0, 0].item() == pytest.approx(10.0)


def test_observation_batch_rejects_empty_and_malformed_inputs() -> None:
    """Bad input should fail at the client-to-Torch boundary with a clear reason."""

    # No observations means there is no batch dimension for the model to use.
    with pytest.raises(ValueError, match="at least one observation"):
        ObservationBatch.from_observations([])

    # Start from one valid batch, then deliberately remove a body slot. The
    # constructor must name the incorrect field instead of letting attention
    # fail later with an unrelated matrix error.
    valid = _tensor_batch(real_bodies=3, real_players=2)
    with pytest.raises(ValueError, match="bodies has shape"):
        ObservationBatch(
            bodies=valid.bodies[:, :-1],
            players=valid.players,
            body_mask=valid.body_mask,
            owned_mask=valid.owned_mask,
            player_mask=valid.player_mask,
            target_mask=valid.target_mask,
        )

    # Masks made from decimal numbers are ambiguous even when they contain only
    # zero and one, so the constructor requires a real true/false dtype.
    with pytest.raises(TypeError, match="body_mask must use torch.bool"):
        ObservationBatch(
            bodies=valid.bodies,
            players=valid.players,
            body_mask=valid.body_mask.to(torch.float32),
            owned_mask=valid.owned_mask,
            player_mask=valid.player_mask,
            target_mask=valid.target_mask,
        )


def test_encoder_returns_one_contextual_embedding_per_token() -> None:
    """The encoder should return one learned vector for every fixed entity slot."""

    # A manual seed makes Torch create the same random test weights every run.
    torch.manual_seed(11)
    encoder = OrbitEncoder().eval()
    # Three games prove the encoder accepts a batch rather than requiring a
    # separate Python call for each environment.
    batch = _tensor_batch(batch_size=3, real_bodies=6, real_players=2)

    # This test only reads predictions. no_grad skips training-history storage,
    # making the forward call faster and using less memory.
    with torch.no_grad():
        encoded = encoder(batch)

    # Every game must return all 16 body rows and 4 player rows, with one
    # MODEL_DIM-sized learned description per row.
    assert encoded.shape == (3, BODY_SLOTS + PLAYER_SLOTS, MODEL_DIM)
    # isfinite rejects NaN and positive/negative infinity in real embeddings.
    assert torch.isfinite(encoded).all()
    # This budget applies to the encoder now and remains below the PR's 3M
    # ceiling after the comparatively small actor and critic heads are added.
    # Use the same helper as the full-policy check below so both tests always
    # count learned parameters according to one shared rule.
    encoder_parameter_count = parameter_count(encoder)
    assert 1_000_000 <= encoder_parameter_count <= 3_000_000


def test_encoder_ignores_and_clears_padding_tokens() -> None:
    """Changing unused rows must not influence real entity representations."""

    torch.manual_seed(23)
    encoder = OrbitEncoder().eval()
    original = _tensor_batch(batch_size=2, real_bodies=4, real_players=2)

    # clone makes independent tensors so changing padding cannot accidentally
    # modify the original comparison input too.
    changed_bodies = original.bodies.clone()
    changed_players = original.players.clone()
    # Padding is allowed to contain arbitrary transport data. Large changes
    # make this test sensitive to either body or player padding leaking through.
    changed_bodies[~original.body_mask] = 10_000.0
    changed_players[~original.player_mask] = -10_000.0
    changed = ObservationBatch(
        bodies=changed_bodies,
        players=changed_players,
        body_mask=original.body_mask,
        owned_mask=original.owned_mask,
        player_mask=original.player_mask,
        target_mask=original.target_mask,
    )

    # No gradients are needed because the test compares outputs without learning.
    with torch.no_grad():
        expected = encoder(original)
        actual = encoder(changed)

    # Floating-point calculations may differ by tiny rounding amounts, so
    # assert_close compares them using explicit relative and absolute limits.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    # Identical complete outputs prove the extreme padding values were ignored.
    # Key padding alone does not clear padded query rows; OrbitEncoder does so
    # explicitly to give every unused slot one canonical all-zero output.
    assert torch.count_nonzero(actual[~original.valid_token_mask]) == 0


def test_encoder_is_equivariant_to_body_slot_permutation() -> None:
    """Moving a body to another slot should move its encoding, not change it."""

    torch.manual_seed(37)
    encoder = OrbitEncoder().eval()
    original = _tensor_batch(real_bodies=7, real_players=2)
    # randperm creates every slot index exactly once in a shuffled order.
    permutation = torch.randperm(BODY_SLOTS)
    # Body features, ownership, and both axes of target legality must move
    # together because they all describe the same physical bodies.
    permuted = _permute_bodies(original, permutation)

    with torch.no_grad():
        expected = encoder(original)
        shuffled = encoder(permuted)

    # bodies[:, permutation] means new slot i contains old slot permutation[i].
    # Assigning with the same permutation restores each output to its old slot.
    restored_bodies = torch.empty_like(shuffled[:, :BODY_SLOTS])
    restored_bodies[:, permutation] = shuffled[:, :BODY_SLOTS]
    restored = torch.cat((restored_bodies, shuffled[:, BODY_SLOTS:]), dim=1)

    # After restoring slot order, every encoded value should match the run that
    # was never shuffled, within normal floating-point rounding tolerance.
    torch.testing.assert_close(restored, expected, rtol=1e-5, atol=1e-6)


def test_encoder_rejects_batch_entry_without_valid_tokens() -> None:
    """An observation containing only padding has no board for attention to read."""

    encoder = OrbitEncoder().eval()
    batch = _tensor_batch(real_bodies=0, real_players=0)

    # Attention cannot calculate meaningful comparisons when every row is
    # padding, so this malformed input must fail explicitly.
    with pytest.raises(ValueError, match="at least one valid token"):
        encoder(batch)


def test_policy_returns_one_batched_decision_set_and_value() -> None:
    """One policy call should produce complete outputs for several games at once."""

    torch.manual_seed(41)
    policy = OrbitPolicy().eval()
    batch = _tensor_batch(batch_size=3, real_bodies=6, real_players=2)

    # Prediction-only tests do not need Torch to remember a backward pass.
    with torch.no_grad():
        output = policy(batch)

    # Each game needs one launch score per body, one target score per possible
    # source/target pair, and one overall critic value.
    assert output.launch_logits.shape == (3, BODY_SLOTS)
    assert output.target_logits.shape == (3, BODY_SLOTS, BODY_SLOTS)
    assert output.values.shape == (3,)

    launchable = batch.launchable_mask
    # Legal choices need ordinary numeric scores. Illegal choices deliberately
    # use negative infinity, which becomes zero probability during sampling.
    assert torch.isfinite(output.launch_logits[launchable]).all()
    assert torch.isneginf(output.launch_logits[~launchable]).all()
    assert torch.isfinite(output.target_logits[batch.target_mask]).all()
    assert torch.isneginf(output.target_logits[~batch.target_mask]).all()
    assert torch.isfinite(output.values).all()

    # The architecture must stay inside the small V1 model budget.
    assert 1_000_000 <= parameter_count(policy) <= 3_000_000


def test_policy_rejects_a_missing_observing_player() -> None:
    """The critic cannot read its learner summary when player slot zero is empty."""

    policy = OrbitPolicy().eval()
    batch = _tensor_batch(real_bodies=4, real_players=0)

    # The policy-level validation runs before the transformer. Keeping this
    # exact message matters because deferred collection will rerun the same
    # pure check only after its combined validation flag reaches Python.
    with pytest.raises(ValueError, match="player slot 0 must contain"):
        policy(batch)


def test_policy_validation_flags_stay_as_device_booleans() -> None:
    """Deferred checks must remain tensors until actions cross to the CPU."""

    valid = _tensor_batch(real_bodies=4, real_players=2)
    malformed = _tensor_batch(real_bodies=0, real_players=0)

    valid_flags = policy_validation_flags(valid)
    malformed_flags = policy_validation_flags(malformed)

    assert valid_flags.dtype == torch.bool
    assert valid_flags.device == valid.bodies.device
    assert valid_flags.shape == (2,)
    assert valid_flags.tolist() == [True, True]
    # With no bodies or players, attention has no real token and player slot
    # zero cannot contain the learner. The helper records both failures rather
    # than raising while the results are still on the model device.
    assert malformed_flags.tolist() == [False, False]


def test_collection_policy_forward_matches_the_eager_forward() -> None:
    """Deferring policy checks must not change scores for a valid observation."""

    torch.manual_seed(43)
    policy = OrbitPolicy().eval()
    batch = _tensor_batch(batch_size=3, real_bodies=6, real_players=2)

    with torch.no_grad():
        eager = policy(batch)
        deferred = run_policy_for_collection(policy, batch)

    assert torch.equal(deferred.launch_logits, eager.launch_logits)
    assert torch.equal(deferred.target_logits, eager.target_logits)
    assert torch.equal(deferred.values, eager.values)
    assert policy_validation_flags(batch).tolist() == [True, True]


def test_collection_policy_rejects_an_incompatible_output_type() -> None:
    """Collection should explain a policy that breaks the actor-critic contract."""

    class TensorPolicy(nn.Module):
        """Return a plain tensor instead of the required PolicyOutput."""

        def forward(self, observation: ObservationBatch) -> torch.Tensor:
            return observation.bodies[:, 0, 0]

    batch = _tensor_batch(real_bodies=4, real_players=2)

    with pytest.raises(TypeError, match="policy returned Tensor, want PolicyOutput"):
        run_policy_for_collection(TensorPolicy().eval(), batch)


def test_collection_policy_respects_an_orbit_policy_forward_override() -> None:
    """A subclass's custom forward method must remain authoritative."""

    class OverriddenPolicy(OrbitPolicy):
        """Record whether collection entered this subclass's forward method."""

        def __init__(self) -> None:
            super().__init__()
            self.forward_called = False

        def forward(self, observation: ObservationBatch) -> PolicyOutput:
            self.forward_called = True
            return super().forward(observation)

    torch.manual_seed(47)
    policy = OverriddenPolicy().eval()
    batch = _tensor_batch(real_bodies=4, real_players=2)

    output = run_policy_for_collection(policy, batch)

    assert policy.forward_called
    assert isinstance(output, PolicyOutput)


def test_model_program_prints_parameter_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The human-facing model command should report the obs-v2 model size."""

    # main returns zero on success, following normal command-line convention.
    assert main() == 0

    output = capsys.readouterr().out.strip()
    # Lock the exact count so an accidental architecture change is immediately
    # visible rather than merely remaining somewhere inside the broad budget.
    # obs-v2 widens only the input projections: (27 - 16) x 128 body weights
    # plus (6 - 3) x 128 player weights adds 1,792 to V1's 1,355,906.
    assert output == "OrbitPolicy parameters: 1,357,698"


def test_policy_ignores_padding_in_every_output_head() -> None:
    """Unused rows must not change launch, target, or critic predictions."""

    torch.manual_seed(43)
    policy = OrbitPolicy().eval()
    original = _tensor_batch(batch_size=2, real_bodies=5, real_players=2)

    # Give only unused rows enormous different values. Real rows and every mask
    # remain identical between the two inputs.
    changed_bodies = original.bodies.clone()
    changed_players = original.players.clone()
    changed_bodies[~original.body_mask] = 50_000.0
    changed_players[~original.player_mask] = -50_000.0
    changed = ObservationBatch(
        bodies=changed_bodies,
        players=changed_players,
        body_mask=original.body_mask,
        owned_mask=original.owned_mask,
        player_mask=original.player_mask,
        target_mask=original.target_mask,
    )

    with torch.no_grad():
        expected = policy(original)
        actual = policy(changed)

    # All three heads must agree, proving padding cannot affect an action or the
    # critic's assessment of the board.
    torch.testing.assert_close(actual.launch_logits, expected.launch_logits)
    torch.testing.assert_close(actual.target_logits, expected.target_logits)
    torch.testing.assert_close(actual.values, expected.values)


def test_policy_probabilities_follow_body_slot_permutation() -> None:
    """Each body's action probabilities must follow it when slots are shuffled."""

    torch.manual_seed(47)
    policy = OrbitPolicy().eval()
    original = _tensor_batch(batch_size=2, real_bodies=7, real_players=2)
    permutation = torch.randperm(BODY_SLOTS)
    permuted = _permute_bodies(original, permutation)

    with torch.no_grad():
        expected = policy(original)
        shuffled = policy(permuted)

    # argsort gives the reverse mapping needed to put shuffled outputs back in
    # their original physical-body order.
    inverse = torch.argsort(permutation)
    # Apply the inverse order to both the source and target dimensions before
    # comparing the shuffled policy with the original policy.
    restored_launch_logits = shuffled.launch_logits[:, inverse]
    restored_target_logits = shuffled.target_logits[:, inverse][:, :, inverse]

    # sigmoid turns each launch score into a probability between zero and one.
    torch.testing.assert_close(
        torch.sigmoid(restored_launch_logits),
        torch.sigmoid(expected.launch_logits),
    )
    torch.testing.assert_close(
        _target_probabilities(restored_target_logits, original.target_mask),
        _target_probabilities(expected.target_logits, original.target_mask),
    )
    # The critic summarizes the whole board, so reordering bodies must leave its
    # single value unchanged rather than merely move it to another slot.
    torch.testing.assert_close(shuffled.values, expected.values)


def test_policy_sampling_enforces_launch_and_target_masks() -> None:
    """Sampling must never select an unowned source or forbidden target."""

    batch = _tensor_batch(batch_size=2, real_bodies=5, real_players=2)
    # Keep slot 1 owned but remove its targets, modeling an empty garrison. It
    # must be forced to no-launch even when its supplied launch logit is large.
    target_mask = batch.target_mask.clone()
    target_mask[:, 1] = False
    # Even an inconsistent target row cannot make an unowned body launch: the
    # sampler treats ownership and pairwise target legality as separate gates.
    target_mask[:, 2, 0] = True
    batch = ObservationBatch(
        bodies=batch.bodies,
        players=batch.players,
        body_mask=batch.body_mask,
        owned_mask=batch.owned_mask,
        player_mask=batch.player_mask,
        target_mask=target_mask,
    )
    # A score of 100 makes every otherwise legal launch effectively certain.
    # Zero target scores make legal targets equally likely.
    output = PolicyOutput(
        launch_logits=torch.full((2, BODY_SLOTS), 100.0),
        target_logits=torch.zeros((2, BODY_SLOTS, BODY_SLOTS)),
        values=torch.zeros(2),
    )

    # Act by sampling from deliberately strong launch scores. Mask enforcement,
    # not model quality, is what this test is isolating.
    sampled = sample_actions(
        output,
        batch,
        # A dedicated generator makes the random choices repeatable without
        # changing random state used by another test.
        generator=torch.Generator().manual_seed(53),
    )
    launchable = batch.launchable_mask

    # With near-certain launch scores, every and only launchable row should fire.
    assert torch.equal(sampled.launches, launchable)
    # Each chosen target must be true in the exact Go-provided source row.
    for batch_index, source in sampled.launches.nonzero().tolist():
        assert batch.target_mask[
            batch_index,
            source,
            sampled.targets[batch_index, source],
        ]

    # Finally verify the tensor decisions become the fixed 16-entry action list
    # expected by the JSON client, and the empty owned body remains no-launch.
    wire_actions = sampled.to_slot_actions()
    assert len(wire_actions) == batch.batch_size
    assert all(len(actions) == BODY_SLOTS for actions in wire_actions)
    assert all(not actions[1].launch for actions in wire_actions)


def test_sampling_rejects_a_launch_without_a_finite_target_score() -> None:
    """A sampled launch needs at least one numeric legal target score."""

    batch = _tensor_batch(batch_size=1, real_bodies=5, real_players=2)
    output = PolicyOutput(
        # Large finite scores make every launchable source launch, allowing the
        # target validator to inspect the deliberately malformed target rows.
        launch_logits=torch.full((1, BODY_SLOTS), 100.0),
        target_logits=torch.full(
            (1, BODY_SLOTS, BODY_SLOTS),
            torch.nan,
        ),
        values=torch.zeros(1),
    )

    with pytest.raises(ValueError, match="finite legal target logit"):
        sample_actions(
            output,
            batch,
            generator=torch.Generator().manual_seed(59),
        )


@pytest.mark.parametrize("force_every_source_idle", [False, True])
def test_collection_sampling_matches_eager_sampling_and_generator_state(
    force_every_source_idle: bool,
) -> None:
    """Safe collection sampling must not perturb valid replayable decisions."""

    batch = _tensor_batch(batch_size=3, real_bodies=6, real_players=2)
    if force_every_source_idle:
        # Removing ownership forces every launch probability to zero. This
        # covers the deferred path's empty target sampling operation, which
        # must consume no extra random values compared with the eager return.
        batch = ObservationBatch(
            bodies=batch.bodies,
            players=batch.players,
            body_mask=batch.body_mask,
            owned_mask=torch.zeros_like(batch.owned_mask),
            player_mask=batch.player_mask,
            target_mask=batch.target_mask,
        )
    output = PolicyOutput(
        launch_logits=torch.linspace(-1.5, 1.5, 3 * BODY_SLOTS).reshape(
            3,
            BODY_SLOTS,
        ),
        target_logits=torch.arange(
            3 * BODY_SLOTS * BODY_SLOTS,
            dtype=torch.float32,
        ).reshape(3, BODY_SLOTS, BODY_SLOTS)
        / 100.0,
        values=torch.zeros(3),
    )
    eager_generator = torch.Generator().manual_seed(61)
    collection_generator = torch.Generator().manual_seed(61)

    eager = sample_actions(output, batch, generator=eager_generator)
    deferred = sample_actions_for_collection(
        output,
        batch,
        generator=collection_generator,
    )

    assert torch.equal(deferred.actions.launches, eager.launches)
    assert torch.equal(deferred.actions.targets, eager.targets)
    assert torch.equal(collection_generator.get_state(), eager_generator.get_state())


def test_collection_sampling_keeps_malformed_scores_cuda_safe() -> None:
    """Temporary fallback probabilities must let later validation report errors."""

    batch = _tensor_batch(batch_size=1, real_bodies=5, real_players=2)

    # NaN launch scores would make Bernoulli reject its probability tensor.
    # The collection path substitutes probability zero without modifying the
    # original output that later validation inspects.
    invalid_launch_output = PolicyOutput(
        launch_logits=torch.full((1, BODY_SLOTS), torch.nan),
        target_logits=torch.zeros((1, BODY_SLOTS, BODY_SLOTS)),
        values=torch.zeros(1),
    )
    launch_actions = sample_actions_for_collection(
        invalid_launch_output,
        batch,
        generator=torch.Generator().manual_seed(67),
    )
    assert not launch_actions.actions.launches.any()
    assert torch.isnan(invalid_launch_output.launch_logits).all()

    # Certain launches combined with NaN target rows would normally make
    # multinomial fail. The temporary uniform legal rows keep every sampled
    # index safe until the deferred flags can be checked.
    invalid_target_output = PolicyOutput(
        launch_logits=torch.full((1, BODY_SLOTS), 100.0),
        target_logits=torch.full(
            (1, BODY_SLOTS, BODY_SLOTS),
            torch.nan,
        ),
        values=torch.zeros(1),
    )
    target_actions = sample_actions_for_collection(
        invalid_target_output,
        batch,
        generator=torch.Generator().manual_seed(71),
    )
    selected_targets = target_actions.sampled_targets
    assert selected_targets.numel() > 0
    assert torch.all((selected_targets >= 0) & (selected_targets < BODY_SLOTS))
    assert torch.isnan(invalid_target_output.target_logits).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_empty_target_sampling_preserves_generator_state() -> None:
    """An empty CUDA multinomial must not consume a Philox random-number block."""

    batch = _tensor_batch(batch_size=3, real_bodies=6, real_players=2)
    batch = ObservationBatch(
        bodies=batch.bodies,
        players=batch.players,
        body_mask=batch.body_mask,
        owned_mask=torch.zeros_like(batch.owned_mask),
        player_mask=batch.player_mask,
        target_mask=batch.target_mask,
    ).to("cuda")
    output = PolicyOutput(
        launch_logits=torch.zeros((3, BODY_SLOTS), device="cuda"),
        target_logits=torch.zeros(
            (3, BODY_SLOTS, BODY_SLOTS),
            device="cuda",
        ),
        values=torch.zeros(3, device="cuda"),
    )
    eager_generator = torch.Generator(device="cuda").manual_seed(83)
    collection_generator = torch.Generator(device="cuda").manual_seed(83)

    eager = sample_actions(output, batch, generator=eager_generator)
    deferred = sample_actions_for_collection(
        output,
        batch,
        generator=collection_generator,
    )

    assert torch.equal(deferred.actions.launches, eager.launches)
    assert torch.equal(deferred.actions.targets, eager.targets)
    assert torch.equal(collection_generator.get_state(), eager_generator.get_state())


def _observation(
    *,
    real_bodies: int,
    real_players: int,
    feature_offset: float,
) -> Observation:
    """Build a small, predictable NumPy observation for model tests."""

    # np.zeros creates an array of the requested shape with every value set to
    # zero. float32 means each feature is stored as the same 32-bit decimal type
    # used by the environment protocol.
    bodies = np.zeros((BODY_SLOTS, BODY_FEATURES), dtype=np.float32)
    players = np.zeros((PLAYER_SLOTS, PLAYER_FEATURES), dtype=np.float32)

    # np.arange creates counting values 0, 1, 2, and so on. reshape lays that
    # one-dimensional sequence out as [real rows, features per row]. Adding the
    # offset gives each synthetic observation an easy-to-recognize starting
    # value; it has no gameplay meaning. The [:real_bodies] slice fills only
    # real rows and leaves all later padding rows at zero.
    bodies[:real_bodies] = feature_offset + np.arange(
        real_bodies * BODY_FEATURES,
        dtype=np.float32,
    ).reshape(real_bodies, BODY_FEATURES)
    players[:real_players] = feature_offset + np.arange(
        real_players * PLAYER_FEATURES,
        dtype=np.float32,
    ).reshape(real_players, PLAYER_FEATURES)

    # bool_ arrays contain true/false switches rather than numbers. Start every
    # slot as false, then mark only the requested leading rows as real or owned.
    body_mask = np.zeros(BODY_SLOTS, dtype=np.bool_)
    body_mask[:real_bodies] = True
    player_mask = np.zeros(PLAYER_SLOTS, dtype=np.bool_)
    player_mask[:real_players] = True
    owned_mask = np.zeros(BODY_SLOTS, dtype=np.bool_)
    owned_mask[: min(real_bodies, 2)] = True
    # The two-dimensional target mask starts with every source/target pair
    # forbidden. Rows are sources and columns are possible targets.
    target_mask = np.zeros((BODY_SLOTS, BODY_SLOTS), dtype=np.bool_)
    # Every owned source can target every other real body in this synthetic
    # board. The source itself is removed because self-targeting is illegal.
    # flatnonzero returns the numeric positions whose mask value is true. Here
    # that gives the owned source slots whose legal target rows must be filled.
    for source in np.flatnonzero(owned_mask):
        target_mask[source, :real_bodies] = True
        target_mask[source, source] = False

    return Observation(
        bodies=bodies,
        players=players,
        body_mask=body_mask,
        owned_mask=owned_mask,
        player_mask=player_mask,
        target_mask=target_mask,
        # Give each slot its own counting ID. int32 matches the protocol's
        # 32-bit whole-number body-ID representation.
        slot_body_id=np.arange(BODY_SLOTS, dtype=np.int32),
    )


def _tensor_batch(
    *,
    batch_size: int = 1,
    real_bodies: int,
    real_players: int,
) -> ObservationBatch:
    """Create one or more synthetic observations and convert them to Torch."""

    # A different feature offset makes each game distinguishable while keeping
    # its shapes and masks identical.
    observations = [
        _observation(
            real_bodies=real_bodies,
            real_players=real_players,
            feature_offset=float(index),
        )
        for index in range(batch_size)
    ]
    return ObservationBatch.from_observations(observations)


def _permute_bodies(
    observation: ObservationBatch,
    permutation: torch.Tensor,
) -> ObservationBatch:
    """Move body data and both source/target mask axes in the same way."""

    # The source and target dimensions both name body slots, so a physical
    # permutation must reorder both axes of the pairwise legality mask.
    target_mask = observation.target_mask[:, permutation]
    target_mask = target_mask[:, :, permutation]
    return ObservationBatch(
        bodies=observation.bodies[:, permutation],
        players=observation.players,
        body_mask=observation.body_mask[:, permutation],
        owned_mask=observation.owned_mask[:, permutation],
        player_mask=observation.player_mask,
        target_mask=target_mask,
    )


def _target_probabilities(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Convert legal target scores to probabilities and leave no-choice rows zero.

    softmax makes every legal row sum to one. Applying it to a row containing
    only negative infinity is undefined, so rows with no choice stay zero.
    """

    probabilities = torch.zeros_like(logits)
    legal_rows = mask.any(dim=-1)
    probabilities[legal_rows] = torch.softmax(logits[legal_rows], dim=-1)
    return probabilities
