"""Tensor contracts shared by the Orrery policy and rollout code."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .env_client import (
    BODY_FEATURES,
    BODY_SLOTS,
    PLAYER_FEATURES,
    PLAYER_SLOTS,
    Observation,
    SlotAction,
)


# Number of learned features used to describe each body or player. A larger
# value gives the model more capacity but increases memory and computation.
MODEL_DIM = 128

# Number of different attention patterns calculated in parallel within each
# layer. The 128 model features are divided evenly across these four heads.
ATTENTION_HEADS = 4

# Number of times every entity can examine the other entities and update its
# learned description before the actor and critic make their predictions.
ENCODER_LAYERS = 4

# Width of the private feed-forward calculation applied to each entity inside
# a transformer layer. Naming it avoids PyTorch's much larger default.
FEEDFORWARD_DIM = 1024

# Fraction of intermediate values randomly hidden during training. V1 uses
# zero so the network itself produces repeatable scores for the same input.
DROPOUT = 0.0

# Index used to select the shared learned label added to every body row. Every
# body uses the same label, so its numeric slot does not gain special meaning.
BODY_TOKEN_TYPE = 0

# Index used to select the shared learned label added to every player row. It
# tells the transformer that player features differ from body features.
PLAYER_TOKEN_TYPE = 1

# Total number of row categories understood by the type-embedding table:
# bodies and players.
TOKEN_TYPES = 2


@dataclass(frozen=True)
class ObservationBatch:
    """Several game observations packaged for one model call.

    The Go client returns one NumPy observation per game. This class combines
    those observations into Torch tensors whose first dimension identifies the
    game. Keeping that conversion here gives the rest of the learning code one
    consistent meaning for shapes, data types, masks, and devices.
    """

    # Feature tensors keep the slot dimensions supplied by Go; the model may
    # project them, but must never reinterpret or reorder them implicitly.
    bodies: Tensor
    players: Tensor
    # These masks serve different purposes: attention sees real tokens,
    # sampling sees owned sources, and target selection sees legal pairs.
    body_mask: Tensor
    owned_mask: Tensor
    player_mask: Tensor
    target_mask: Tensor

    def __post_init__(self) -> None:
        """Reject an incomplete or inconsistent batch as soon as it is created.

        A bad shape or data type would otherwise fail much later inside the
        transformer with an error that does not identify the faulty input.
        """

        # Validate at the NumPy/Torch boundary. A malformed batch otherwise
        # tends to fail later inside attention with an unhelpful shape error.
        batch_size = _batch_size(self.bodies, "bodies")
        _require_shape(self.bodies, "bodies", (batch_size, BODY_SLOTS, BODY_FEATURES))
        _require_shape(
            self.players,
            "players",
            (batch_size, PLAYER_SLOTS, PLAYER_FEATURES),
        )
        _require_shape(self.body_mask, "body_mask", (batch_size, BODY_SLOTS))
        _require_shape(self.owned_mask, "owned_mask", (batch_size, BODY_SLOTS))
        _require_shape(self.player_mask, "player_mask", (batch_size, PLAYER_SLOTS))
        _require_shape(
            self.target_mask,
            "target_mask",
            (batch_size, BODY_SLOTS, BODY_SLOTS),
        )

        # Features are measurements the model learns from, so they use decimal
        # numbers. Masks are yes/no switches that include or exclude rows, so
        # they must contain true/false values rather than numeric 0s and 1s.
        for name in ("bodies", "players"):
            tensor = getattr(self, name)
            if not tensor.is_floating_point():
                raise TypeError(
                    f"{name} must use a floating-point dtype, got {tensor.dtype}"
                )
        for name in ("body_mask", "owned_mask", "player_mask", "target_mask"):
            tensor = getattr(self, name)
            if tensor.dtype != torch.bool:
                raise TypeError(f"{name} must use torch.bool, got {tensor.dtype}")

        # Mixed CPU/GPU fields would fail only when they are concatenated or
        # used for masking, so reject that inconsistent state immediately.
        devices = {
            tensor.device
            for tensor in (
                self.bodies,
                self.players,
                self.body_mask,
                self.owned_mask,
                self.player_mask,
                self.target_mask,
            )
        }
        if len(devices) != 1:
            raise ValueError(f"observation tensors must share one device, got {devices}")

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[Observation],
        *,
        device: torch.device | str | None = None,
    ) -> ObservationBatch:
        """Combine individual client observations into one model-ready batch.

        Body and player slots stay in the exact order supplied by Go. The
        optional device chooses where Torch stores the result, such as the CPU
        or a GPU.
        """

        if not observations:
            raise ValueError("at least one observation is required")

        # Each list below collects the same field from every game observation.
        # np.stack places those equally shaped arrays on top of one another and
        # adds a new first dimension: [game, ...original field dimensions].
        # torch.from_numpy then exposes the stacked values as a Torch tensor
        # without changing their order or converting decimal/bool data types.
        batch = cls(
            bodies=torch.from_numpy(
                np.stack([observation.bodies for observation in observations])
            ),
            players=torch.from_numpy(
                np.stack([observation.players for observation in observations])
            ),
            body_mask=torch.from_numpy(
                np.stack([observation.body_mask for observation in observations])
            ),
            owned_mask=torch.from_numpy(
                np.stack([observation.owned_mask for observation in observations])
            ),
            player_mask=torch.from_numpy(
                np.stack([observation.player_mask for observation in observations])
            ),
            target_mask=torch.from_numpy(
                np.stack([observation.target_mask for observation in observations])
            ),
        )
        return batch if device is None else batch.to(device)

    @property
    def batch_size(self) -> int:
        """Return how many separate games are present in this batch."""

        return self.bodies.shape[0]

    @property
    def valid_token_mask(self) -> Tensor:
        """Return one mask telling attention which body/player rows are real.

        ``True`` means the row describes a real game entity. ``False`` means it
        is unused padding and must not influence any model decision.
        """

        # This body-then-player order is part of the model contract: the later
        # tokenizer concatenates projected tokens in exactly the same order.
        return torch.cat((self.body_mask, self.player_mask), dim=1)

    @property
    def launchable_mask(self) -> Tensor:
        """Return the body slots from which the learner may launch.

        A source must be owned by the learner and have at least one legal
        target in the pairwise mask supplied by Go. Keeping that rule here
        ensures policy scoring and action sampling always agree.
        """

        return self.owned_mask & self.target_mask.any(dim=-1)

    def to(self, device: torch.device | str) -> ObservationBatch:
        """Move every tensor together to the requested CPU or GPU device."""

        # Reconstructing the frozen dataclass also reruns its invariants after
        # the move, rather than returning a partially moved object.
        return ObservationBatch(
            bodies=self.bodies.to(device),
            players=self.players.to(device),
            body_mask=self.body_mask.to(device),
            owned_mask=self.owned_mask.to(device),
            player_mask=self.player_mask.to(device),
            target_mask=self.target_mask.to(device),
        )


class OrbitEncoder(nn.Module):
    """Turn raw game rows into learned descriptions of bodies and players.

    Each output row describes one entity after it has considered every other
    real entity on the board. These descriptions feed the policy's action and
    value heads; the encoder does not choose an action itself.
    """

    def __init__(self) -> None:
        """Build the projections, entity-type labels, and transformer layers."""

        super().__init__()

        # Bodies and players have different feature schemas, so each needs its
        # own learned projection before both can share one attention space.
        self.body_projection = nn.Linear(BODY_FEATURES, MODEL_DIM)
        self.player_projection = nn.Linear(PLAYER_FEATURES, MODEL_DIM)
        self.token_type_embedding = nn.Embedding(TOKEN_TYPES, MODEL_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=ATTENTION_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Each transformer layer normalizes values before its attention and
        # feed-forward calculations. Its final skip connection happens afterward,
        # so add one last LayerNorm to give the policy heads consistently scaled
        # values.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=ENCODER_LAYERS,
            norm=nn.LayerNorm(MODEL_DIM),
            # V1 already uses fixed dense tensors and a padding mask. Disabling
            # nested-tensor conversion avoids a second representation with
            # different support and performance behavior across Torch versions.
            enable_nested_tensor=False,
        )

        # Store type IDs as a buffer so moving the module to a GPU moves these
        # IDs too, while keeping them out of the learned parameter count.
        token_type_ids = torch.cat(
            (
                torch.full((BODY_SLOTS,), BODY_TOKEN_TYPE, dtype=torch.long),
                torch.full((PLAYER_SLOTS,), PLAYER_TOKEN_TYPE, dtype=torch.long),
            )
        )
        self.register_buffer("token_type_ids", token_type_ids, persistent=False)

    def forward(self, observation: ObservationBatch) -> Tensor:
        """Encode a whole batch and return one learned vector per entity row.

        The result has shape ``[games, 48, MODEL_DIM]``: 44 body rows followed
        by 4 player rows. Unused rows are returned as zeros and cannot affect
        real rows.
        """

        return self._forward(observation, defer_validation=False)

    def forward_for_collection(self, observation: ObservationBatch) -> Tensor:
        """Encode while leaving data-dependent validation as GPU flags."""

        return self._forward(observation, defer_validation=True)

    def _forward(
        self,
        observation: ObservationBatch,
        *,
        defer_validation: bool,
    ) -> Tensor:
        """Share encoder math between ordinary and deferred-validation calls."""

        body_tokens = self.body_projection(observation.bodies)
        player_tokens = self.player_projection(observation.players)
        tokens = torch.cat((body_tokens, player_tokens), dim=1)

        # Every valid body receives the same body-type vector and every valid
        # player receives the same player-type vector. No absolute slot
        # position is introduced, so permuting bodies still permutes outputs.
        type_embedding = self.token_type_embedding(self.token_type_ids).unsqueeze(0)
        tokens = tokens + type_embedding

        valid_tokens = observation.valid_token_mask
        if not defer_validation:
            _validate_attention_tokens(valid_tokens)

        # PyTorch's key-padding convention is True = ignored, the inverse of
        # the environment's masks. Zeroing inputs as well ensures arbitrary
        # values hidden in padding cannot survive through residual connections.
        tokens = tokens.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
        encoded = self.transformer(
            tokens,
            src_key_padding_mask=~valid_tokens,
        )

        # Key padding prevents real queries from reading padding, but it does
        # not suppress padded query rows. Clear those rows before actor heads
        # are added so every masked output has one canonical representation.
        return encoded.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)


@dataclass(frozen=True)
class PolicyOutput:
    """The policy's action scores and board-value estimate for each game.

    These are scores rather than final actions. Sampling later turns launch
    scores into yes/no decisions and target scores into legal target choices.
    """

    # Store the actor's raw decision scores rather than converted probabilities.
    # Torch can use these scores directly to sample launch and target choices and
    # later calculate PPO probabilities without avoidable rounding errors. The
    # critic value is not a probability, so it remains an unrestricted number.
    launch_logits: Tensor
    target_logits: Tensor
    values: Tensor

    def __post_init__(self) -> None:
        """Check that every output describes the same number of games."""

        # All three heads must describe the same environments in the same
        # leading order or later action/log-probability pairing becomes wrong.
        batch_size = _batch_size(self.launch_logits, "launch_logits")
        _require_shape(
            self.launch_logits,
            "launch_logits",
            (batch_size, BODY_SLOTS),
        )
        _require_shape(
            self.target_logits,
            "target_logits",
            (batch_size, BODY_SLOTS, BODY_SLOTS),
        )
        _require_shape(self.values, "values", (batch_size,))

        for name in ("launch_logits", "target_logits", "values"):
            tensor = getattr(self, name)
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must use a floating-point dtype, got {tensor.dtype}")

        devices = {
            self.launch_logits.device,
            self.target_logits.device,
            self.values.device,
        }
        if len(devices) != 1:
            raise ValueError(f"policy outputs must share one device, got {devices}")


class OrbitPolicy(nn.Module):
    """Choose action scores and estimate board strength for many games at once.

    The actor side scores launches and targets. The critic side estimates how
    favorable each board is for the learner. Body slot numbers carry no learned
    meaning, so rearranging bodies does not change the physical decision.
    """

    def __init__(self) -> None:
        """Build the shared encoder plus the launch, target, and value heads."""

        super().__init__()
        self.encoder = OrbitEncoder()

        # Launch decisions are independent Bernoulli choices, one per body.
        # Target scores use shared query/key projections instead of a fixed
        # output column per slot, preserving body-permutation equivariance.
        self.launch_head = nn.Linear(MODEL_DIM, 1)
        self.target_query = nn.Linear(MODEL_DIM, MODEL_DIM)
        self.target_key = nn.Linear(MODEL_DIM, MODEL_DIM)

        # Player slot 0 is always the observing learner. Reading its contextual
        # token gives the critic a player-relative view without pooling padding.
        self.value_head = nn.Linear(MODEL_DIM, 1)

    def forward(self, observation: ObservationBatch) -> PolicyOutput:
        """Score every possible V1 decision and each game's value in one call.

        Illegal launch sources and target pairs receive negative infinity, so
        their probability becomes zero during sampling. Player row 0 supplies
        the learner-relative summary used by the critic.
        """

        return self._forward(observation, defer_validation=False)

    def forward_for_collection(self, observation: ObservationBatch) -> PolicyOutput:
        """Score a rollout turn while leaving validation results on the GPU.

        Collection transfers those flags with the sampled actions. Evaluators,
        PPO replay, and direct callers keep using ``forward`` and therefore
        retain immediate validation with the original error messages.
        """

        return self._forward(observation, defer_validation=True)

    def _forward(
        self,
        observation: ObservationBatch,
        *,
        defer_validation: bool,
    ) -> PolicyOutput:
        """Share policy calculations between eager and collection calls."""

        # Ordinary policy calls need an immediate, precise error if player slot
        # zero does not contain the learner. Rollout collection sets
        # defer_validation=True because it records this same condition as a GPU
        # flag and reads that flag later with the packed actions. Skipping the
        # eager read here avoids one GPU-to-CPU wait; it does not skip the check.
        if not defer_validation:
            _validate_observing_player(observation.player_mask)

        encoded = (
            self.encoder.forward_for_collection(observation)
            if defer_validation
            else self.encoder(observation)
        )
        body_tokens = encoded[:, :BODY_SLOTS]

        launch_logits = self.launch_head(body_tokens).squeeze(-1)
        # The shared launchability rule also excludes owned bodies with no
        # legal targets, so their launch probability becomes zero.
        launch_logits = launch_logits.masked_fill(
            ~observation.launchable_mask,
            -torch.inf,
        )

        queries = self.target_query(body_tokens)
        keys = self.target_key(body_tokens)
        target_logits = torch.matmul(queries, keys.transpose(-2, -1))
        target_logits = target_logits / math.sqrt(MODEL_DIM)
        target_logits = target_logits.masked_fill(
            ~observation.target_mask,
            -torch.inf,
        )

        observer_token = encoded[:, BODY_SLOTS]
        values = self.value_head(observer_token).squeeze(-1)
        return PolicyOutput(
            launch_logits=launch_logits,
            target_logits=target_logits,
            values=values,
        )


def policy_device(policy: nn.Module) -> torch.device:
    """Return the one device shared by a policy's parameters and buffers."""

    devices = {
        tensor.device
        for tensor in (*tuple(policy.parameters()), *tuple(policy.buffers()))
    }
    if len(devices) > 1:
        raise ValueError(f"policy parameters and buffers span devices {devices}")
    return next(iter(devices), torch.device("cpu"))


def require_evaluation_mode(policy: nn.Module, operation: str) -> None:
    """Require deterministic module behavior for a PPO lifecycle operation."""

    training_modules = [
        name or "<policy>"
        for name, module in policy.named_modules()
        if module.training
    ]
    if training_modules:
        raise ValueError(
            f"policy must be in evaluation mode during {operation}; "
            f"training modules: {training_modules}"
        )


def run_policy(policy: nn.Module, observation: ObservationBatch) -> PolicyOutput:
    """Run a policy and enforce the shared actor-critic output contract."""

    output = policy(observation)
    if not isinstance(output, PolicyOutput):
        raise TypeError(f"policy returned {type(output).__name__}, want PolicyOutput")
    return output


def run_policy_for_collection(
    policy: nn.Module,
    observation: ObservationBatch,
) -> PolicyOutput:
    """Run the production policy without forcing GPU validation reads.

    During rollout collection, Python calls the policy once for every batch of
    current game observations. The ordinary ``OrbitPolicy.forward`` validates
    two tensor conditions by reading their true/false results immediately.
    Reading a CUDA result makes Python wait for the GPU, so doing that twice on
    every turn adds many small pauses over a 512-turn rollout.

    ``OrbitPolicy.forward_for_collection`` performs the same transformer and
    actor-critic calculations but leaves those two validation results on the
    GPU. They are transferred later with the packed actions. Validation is
    therefore delayed, not removed: a failed flag is rerun through the eager
    validators before any action is sent to Go.
    """

    # The exact production OrbitPolicy owns the special collection path
    # described above. A subclass may override forward() to change scoring or
    # add instrumentation; calling an inherited forward_for_collection() would
    # silently bypass that override. Subclasses, small test policies, and other
    # caller-supplied modules therefore keep their ordinary forward method.
    output = (
        policy.forward_for_collection(observation)
        if type(policy) is OrbitPolicy
        else policy(observation)
    )
    # Both branches must still return the same structured actor-critic result.
    # Rejecting another type here prevents later sampling code from failing
    # with an unrelated missing-attribute error.
    if not isinstance(output, PolicyOutput):
        raise TypeError(f"policy returned {type(output).__name__}, want PolicyOutput")
    return output


@dataclass(frozen=True)
class SampledActionBatch:
    """The sampled launch flags and target slots for a batch of games.

    Decisions remain tensors here so later PPO code can retain a clean batched
    representation. They are converted to protocol objects only when sent to
    the Go environment.
    """

    launches: Tensor
    targets: Tensor

    def __post_init__(self) -> None:
        """Check that launch flags and targets align game-by-game and slot-by-slot."""

        batch_size = _batch_size(self.launches, "launches")
        _require_shape(self.launches, "launches", (batch_size, BODY_SLOTS))
        _require_shape(self.targets, "targets", (batch_size, BODY_SLOTS))
        if self.launches.dtype != torch.bool:
            raise TypeError(f"launches must use torch.bool, got {self.launches.dtype}")
        if self.targets.dtype != torch.long:
            raise TypeError(f"targets must use torch.long, got {self.targets.dtype}")
        if self.launches.device != self.targets.device:
            raise ValueError("sampled launches and targets must share one device")

    def to_slot_actions(self) -> list[list[SlotAction]]:
        """Create the per-game ``SlotAction`` lists expected by ``EnvClient``.

        A target is meaningful only for a launching source. No-launch entries
        therefore use target 0 as an ignored placeholder.
        """

        launches = self.launches.detach().cpu().tolist()
        targets = self.targets.detach().cpu().tolist()
        return [
            [
                SlotAction(launch=launch, target=target if launch else 0)
                for launch, target in zip(batch_launches, batch_targets, strict=True)
            ]
            for batch_launches, batch_targets in zip(launches, targets, strict=True)
        ]


@dataclass(frozen=True)
class CollectionActionSample:
    """Actions plus the selections reused by collection log-probability math.

    ``nonzero`` determines the row order once during sampling. Retaining those
    indices and their selected logits prevents the deferred validation and
    log-probability paths from repeating dynamic boolean-mask indexing on the
    GPU. The ordinary ``actions`` field remains the representation stored in a
    rollout and eventually encoded for Go.
    """

    actions: SampledActionBatch
    # Each index tensor has columns [game, source body]. The first covers every
    # optional launch decision; the second covers only sources that launched.
    launchable_indices: Tensor
    launched_indices: Tensor
    # These rows follow the exact order of their corresponding index tensor.
    # They retain the original policy scores so validation cannot be hidden by
    # the temporary safe values used to sample malformed input.
    launch_logits: Tensor
    legal_targets: Tensor
    target_logits: Tensor
    sampled_targets: Tensor


def sample_actions(
    output: PolicyOutput,
    observation: ObservationBatch,
    *,
    generator: torch.Generator | None = None,
) -> SampledActionBatch:
    """Turn policy scores into random but legal actions for every game.

    Launch choices are sampled together. Targets are then sampled together only
    for sources that launched. Both ownership and Go's target mask are checked
    again here so malformed scores cannot create an illegal protocol action.
    """

    return _sample_actions(
        output,
        observation,
        generator=generator,
        defer_validation=False,
    ).actions


def sample_actions_for_collection(
    output: PolicyOutput,
    observation: ObservationBatch,
    *,
    generator: torch.Generator | None = None,
) -> CollectionActionSample:
    """Sample with temporary safe values while GPU validation stays deferred.

    Valid scores follow the exact same calculations and random draws as
    ``sample_actions``. Malformed scores are replaced only in temporary
    sampling tensors so Bernoulli, softmax, and multinomial cannot fail before
    their GPU validation flags reach Python. The original ``PolicyOutput`` is
    never changed; later eager validation can therefore report the real input
    error, and the sampled action is never authoritative by itself.
    """

    return _sample_actions(
        output,
        observation,
        generator=generator,
        defer_validation=True,
    )


def _sample_actions(
    output: PolicyOutput,
    observation: ObservationBatch,
    *,
    generator: torch.Generator | None,
    defer_validation: bool,
) -> CollectionActionSample:
    """Implement eager and deferred sampling with one shared valid-input path."""

    # Shape and device checks inspect tensor metadata, so they do not wait for
    # CUDA and remain eager in both paths.
    _validate_sampling_alignment(output, observation)

    # Check launch legality again before creating actions. This prevents a test
    # or another caller from bypassing the masks by supplying its own scores
    # instead of obtaining them from OrbitPolicy.
    # masked_fill replaces every forbidden launch score with negative infinity.
    # Sigmoid below converts negative infinity to probability zero.
    launch_logits = output.launch_logits.masked_fill(
        ~observation.launchable_mask,
        -torch.inf,
    )
    # ``nonzero`` returns indices in row-major order, matching the order used
    # by the previous boolean selections. Save this result because collection
    # log-probability calculation needs the same source rows.
    launchable_indices = observation.launchable_mask.nonzero()
    launchable_games = launchable_indices[:, 0]
    launchable_sources = launchable_indices[:, 1]
    selected_launch_logits = output.launch_logits[
        launchable_games,
        launchable_sources,
    ]
    # Sigmoid converts each unrestricted launch score into a probability from
    # zero to one without changing the batch or body-slot dimensions.
    launch_probabilities = torch.sigmoid(launch_logits)
    if defer_validation:
        # A NaN probability makes torch.bernoulli raise before the combined
        # validation result can cross to Python. Use zero only in the temporary
        # sampling tensor. Positive and negative infinity logits already map
        # safely to probabilities one and zero, while later flags still reject
        # every non-finite launch logit from the untouched PolicyOutput.
        launch_probabilities = torch.nan_to_num(
            launch_probabilities,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
    # Bernoulli performs one yes/no random draw for every launch probability.
    # Supplying a seeded generator makes those draws replayable. The result is
    # converted from numeric zero/one values into a true/false launch mask.
    launches = torch.bernoulli(
        launch_probabilities,
        generator=generator,
    ).to(torch.bool)

    # Determine the actual launch rows once. Target sampling and the later
    # collection log-probability calculation both reuse this stable ordering.
    launched_indices = launches.nonzero()
    launched_games = launched_indices[:, 0]
    launched_sources = launched_indices[:, 1]

    # Start every body with target slot zero as an ignored placeholder. Only
    # bodies whose launch flag is true receive a meaningful target below.
    targets = torch.zeros_like(launches, dtype=torch.long)
    # Explicit [game, source] indices gather the selected target rows without a
    # second boolean-mask search. Empty index tensors naturally produce empty
    # [0, 44] rows when no source launched.
    legal_targets = observation.target_mask[launched_games, launched_sources]
    # A caller-supplied PolicyOutput may contain scores for forbidden targets,
    # so force those entries to negative infinity again.
    selected_logits = output.target_logits[
        launched_games,
        launched_sources,
    ].masked_fill(
        ~legal_targets,
        -torch.inf,
    )
    if defer_validation:
        # A target row is safe only when it contains at least one legal slot and
        # every legal score is finite. If it is malformed, sample temporarily
        # from a uniform legal row. A completely missing legal row uses slot
        # zero solely as an indexing-safe fallback; validation still fails and
        # a later substep will prevent that placeholder action from reaching Go.
        has_legal_target = legal_targets.any(dim=-1)
        finite_legal_scores = (
            torch.isfinite(selected_logits) | ~legal_targets
        ).all(dim=-1)
        valid_target_rows = has_legal_target & finite_legal_scores

        fallback_targets = torch.zeros_like(legal_targets)
        fallback_targets[:, 0] = True
        safe_target_mask = torch.where(
            has_legal_target.unsqueeze(1),
            legal_targets,
            fallback_targets,
        )
        uniform_legal_logits = torch.zeros_like(selected_logits).masked_fill(
            ~safe_target_mask,
            -torch.inf,
        )
        logits_for_sampling = torch.where(
            valid_target_rows.unsqueeze(1),
            selected_logits,
            uniform_legal_logits,
        )
    else:
        # Every launching source must retain at least one ordinary finite score.
        # Otherwise softmax would have no legal choice from which to sample.
        _validate_sampled_target_logits(selected_logits)
        logits_for_sampling = selected_logits

    # Softmax turns each launch row's target scores into probabilities that
    # sum to one; negative-infinity entries become probability zero.
    target_probabilities = torch.softmax(logits_for_sampling, dim=-1)
    # Multinomial chooses one target index from each probability row. It also
    # accepts an empty set of launch rows, which keeps deferred collection free
    # from an extra GPU-to-CPU "did anything launch?" decision.
    sampled_targets = torch.multinomial(
        target_probabilities,
        num_samples=1,
        generator=generator,
    ).squeeze(-1)
    # Write the compact list of sampled indices back into its original
    # game/body positions. No-launch positions keep placeholder target zero.
    targets[launched_games, launched_sources] = sampled_targets

    # Keep tensor form for later PPO calculations; conversion to JSON-friendly
    # SlotAction objects happens only at the environment boundary.
    return CollectionActionSample(
        actions=SampledActionBatch(launches=launches, targets=targets),
        launchable_indices=launchable_indices,
        launched_indices=launched_indices,
        launch_logits=selected_launch_logits,
        legal_targets=legal_targets,
        target_logits=selected_logits,
        sampled_targets=sampled_targets,
    )


def policy_validation_flags(observation: ObservationBatch) -> Tensor:
    """Return policy-input checks as boolean tensors on the input device.

    The two entries mean, in order:

    1. every game contains at least one real transformer token;
    2. player slot zero contains the observing learner in every game.

    Nothing in this function converts a tensor to a Python boolean. On CUDA,
    the checks therefore remain queued GPU work until the collection path
    deliberately transfers them together with the sampled actions.
    """

    return torch.stack(
        (
            _attention_tokens_are_valid(observation.valid_token_mask),
            _observing_player_is_valid(observation.player_mask),
        )
    )


def validate_policy_observation(observation: ObservationBatch) -> None:
    """Rerun policy checks in their normal forward-pass order.

    The learner-slot check normally belongs to ``OrbitPolicy`` and therefore
    runs before the encoder's valid-token check. Preserving that order ensures
    a malformed batch receives the same first detailed error after deferred
    flags report failure.
    """

    _validate_observing_player(observation.player_mask)
    _validate_attention_tokens(observation.valid_token_mask)


def validate_collection_sample(
    output: PolicyOutput,
    observation: ObservationBatch,
    sample: CollectionActionSample,
) -> None:
    """Rerun sampling checks without drawing another random action."""

    _validate_sampling_alignment(output, observation)
    _validate_sampled_target_logits(sample.target_logits)


def sampled_target_validation_flag(selected_logits: Tensor) -> Tensor:
    """Return whether every sampled launch has a finite legal target score.

    Illegal targets have already been replaced by negative infinity before
    this function is called. A row passes when at least one remaining target
    score is finite. The scalar result stays on the logits' device.
    """

    return torch.isfinite(selected_logits).any(dim=-1).all()


def _attention_tokens_are_valid(valid_tokens: Tensor) -> Tensor:
    """Build the GPU flag used by the eager and deferred token checks."""

    return valid_tokens.any(dim=1).all()


def _validate_attention_tokens(valid_tokens: Tensor) -> None:
    """Reject a game whose transformer input contains only padding.

    This helper performs validation only: it consumes no randomness and does
    not modify the supplied mask. Keeping the check separate from the encoder
    calculation lets the later collection fast path postpone reading its
    boolean result without maintaining a second definition of valid input.
    """

    if not _attention_tokens_are_valid(valid_tokens):
        # Attention over a row with no valid keys is undefined and produces
        # NaNs. Real observations always contain seated players, but this
        # check also makes malformed synthetic batches fail clearly.
        raise ValueError("every observation must contain at least one valid token")


def _observing_player_is_valid(player_mask: Tensor) -> Tensor:
    """Build the GPU flag used by the eager and deferred learner-slot checks."""

    return player_mask[:, 0].all()


def _validate_observing_player(player_mask: Tensor) -> None:
    """Require player slot zero to identify the learner in every game.

    The critic reads this exact slot as its learner-relative summary. Like the
    other extracted helpers, this function only validates existing data and
    therefore can be safely rerun when a deferred GPU flag reports failure.
    """

    if not _observing_player_is_valid(player_mask):
        raise ValueError("player slot 0 must contain the observing player")


def _validate_sampling_alignment(
    output: PolicyOutput,
    observation: ObservationBatch,
) -> None:
    """Require policy scores and observations to describe the same batch.

    Sampling combines scores and masks with matching tensor indices. A device
    or batch-size mismatch would otherwise associate unrelated games or fail
    later inside a Torch operation with a less useful error.
    """

    if output.launch_logits.device != observation.bodies.device:
        raise ValueError("policy output and observation batch must share one device")
    if output.launch_logits.shape[0] != observation.batch_size:
        raise ValueError(
            "policy output and observation batch must contain the same batch size"
        )


def _validate_sampled_target_logits(selected_logits: Tensor) -> None:
    """Require each sampled launch to retain a finite legal target score.

    ``selected_logits`` already contains only source rows that launched and
    has illegal targets masked to negative infinity. The helper does not draw
    another target, so rerunning it cannot advance the rollout generator.
    """

    if not sampled_target_validation_flag(selected_logits):
        raise ValueError("every sampled launch must have a finite legal target logit")


def parameter_count(module: nn.Module) -> int:
    """Count every learned number in a model for the V1 size check."""

    return sum(parameter.numel() for parameter in module.parameters())


def _batch_size(tensor: Tensor, name: str) -> int:
    """Read the game count from a tensor and reject a missing or empty batch."""

    # Empty batches have no useful model meaning and make the other expected
    # dimensions ambiguous, so every public contract rejects them.
    if tensor.ndim == 0:
        raise ValueError(f"{name} must include a batch dimension")
    if tensor.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one batch entry")
    return tensor.shape[0]


def _require_shape(tensor: Tensor, name: str, expected: tuple[int, ...]) -> None:
    """Explain clearly when a tensor does not have the exact required shape."""

    # Exact fixed shapes are deliberate in V1: accepting broadcast-compatible
    # shapes here could silently associate an action with the wrong body slot.
    actual = tuple(tensor.shape)
    if actual != expected:
        raise ValueError(f"{name} has shape {actual}, want {expected}")


def main() -> int:
    """Build a fresh policy and print its size for a quick human check."""

    policy = OrbitPolicy()
    print(f"OrbitPolicy parameters: {parameter_count(policy):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
