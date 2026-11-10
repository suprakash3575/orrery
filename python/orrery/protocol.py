"""Encode and decode the fixed-layout ``env-v3`` wire protocol.

This module owns the binary representation that crosses the Python/Go process
boundary. It currently implements frame handling, all request payloads,
structured error responses, and all four success-response payloads. Keeping
binary offsets and NumPy buffer handling here avoids mixing them into
:mod:`orrery.env_client`, whose job is to own the subprocess and enforce the
create/reset/step/close lifecycle.

It is also the one Python copy of the obs-v2 dimensions and feature scale,
which the Go side defines in ``mlenv/observation.go``.

The codec must not implement game or training behavior. Go remains authoritative
for game rules, while ``EnvClient`` remains responsible for deciding when to
send each request.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
import math
import struct
from typing import Any, BinaryIO, Final

import numpy as np
from numpy.typing import NDArray


# The protocol and observation names are persisted in run configuration and
# checkpoints. env-v3 carries obs-v2 observations and nothing else. The version
# numbers stay in these names because existing checkpoints record them.
PROTOCOL_VERSION: Final = "env-v3"
OBSERVATION_SCHEMA: Final = "obs-v2"

# Every frame begins with this fixed header. The magic differs from the old
# env-v2's "OWE2", so a reader speaking that protocol fails on the header. The
# maximum prevents a corrupted length field from making Python allocate an
# unbounded response buffer.
FRAME_MAGIC: Final = b"OWE3"
FRAME_HEADER_SIZE: Final = 12
MAX_PAYLOAD_SIZE: Final = 64 * 1024 * 1024

# General response strings may contain diagnostics, while the opponent and
# schedule fields are short control-plane identities with tighter limits.
MAX_STRING_SIZE: Final = 1024 * 1024
MAX_OPPONENT_SIZE: Final = 1024
MAX_SCHEDULE_SIZE: Final = 64


class MessageType(IntEnum):
    """Identify how the payload following an env-v3 header is laid out."""

    CREATE_REQUEST = 0x01
    RESET_REQUEST = 0x02
    STEP_REQUEST = 0x03
    CLOSE_REQUEST = 0x04

    CREATE_RESPONSE = 0x81
    RESET_RESPONSE = 0x82
    STEP_RESPONSE = 0x83
    CLOSE_RESPONSE = 0x84
    ERROR_RESPONSE = 0xFF


# NumPy reads these little-endian scalar layouts directly from response bytes.
# A wire Boolean is intentionally uint8 at first: the decoder must reject any
# byte other than 0 or 1 before exposing the array as np.bool_.
UINT8_DTYPE: Final = np.dtype("u1")
INT8_DTYPE: Final = np.dtype("i1")
UINT16_DTYPE: Final = np.dtype("<u2")
UINT32_DTYPE: Final = np.dtype("<u4")
INT32_DTYPE: Final = np.dtype("<i4")
INT64_DTYPE: Final = np.dtype("<i8")
FLOAT32_DTYPE: Final = np.dtype("<f4")
FLOAT64_DTYPE: Final = np.dtype("<f8")
BOOL_WIRE_DTYPE: Final = UINT8_DTYPE

# Named byte widths keep later offset calculations readable and derive from
# NumPy itself, avoiding a second handwritten description of each dtype.
UINT8_BYTES: Final = UINT8_DTYPE.itemsize
INT8_BYTES: Final = INT8_DTYPE.itemsize
UINT16_BYTES: Final = UINT16_DTYPE.itemsize
UINT32_BYTES: Final = UINT32_DTYPE.itemsize
INT32_BYTES: Final = INT32_DTYPE.itemsize
INT64_BYTES: Final = INT64_DTYPE.itemsize
FLOAT32_BYTES: Final = FLOAT32_DTYPE.itemsize
FLOAT64_BYTES: Final = FLOAT64_DTYPE.itemsize
BOOL_WIRE_BYTES: Final = BOOL_WIRE_DTYPE.itemsize

# These dimensions are fixed by obs-v2. They repeat BodySlots, BodyFeatures,
# PlayerSlots, and PlayerFeatures in mlenv/observation.go and must change with
# them: 40 planet slots plus 4 comet
# slots, 27 body columns, and four player rows holding two totals plus a
# four-way role one-hot. The codec needs them to calculate byte boundaries
# before constructing any NumPy view over an untrusted response.
BODY_SLOTS: Final = 44
BODY_FEATURES: Final = 27
PLAYER_SLOTS: Final = 4
PLAYER_FEATURES: Final = 6

# Ship columns hold log1p(ships) / log1p(SHIP_SCALE), mirroring FeatureShips
# and PlayerFeatureShips in mlenv/observation.go. The scale is a reference
# rather than a cap, so a column above 1 is legitimate and still decodes: real
# ships = expm1(value * log1p(SHIP_SCALE)).
SHIP_SCALE: Final = 1000

# Column 0 of every player row is that player's total ships.
PLAYER_FEATURE_SHIPS: Final = 0

# One observation is stored as seven consecutive columns. Calculate its wire
# size from the documented shapes so future decoder checks use one source of
# truth instead of scattering the literal 7,052 throughout the module.
BODY_FEATURE_BYTES_PER_ENVIRONMENT: Final = (
    BODY_SLOTS * BODY_FEATURES * FLOAT32_BYTES
)
PLAYER_FEATURE_BYTES_PER_ENVIRONMENT: Final = (
    PLAYER_SLOTS * PLAYER_FEATURES * FLOAT32_BYTES
)
BODY_MASK_BYTES_PER_ENVIRONMENT: Final = BODY_SLOTS * BOOL_WIRE_BYTES
OWNED_MASK_BYTES_PER_ENVIRONMENT: Final = BODY_SLOTS * BOOL_WIRE_BYTES
PLAYER_MASK_BYTES_PER_ENVIRONMENT: Final = PLAYER_SLOTS * BOOL_WIRE_BYTES
TARGET_MASK_BYTES_PER_ENVIRONMENT: Final = (
    BODY_SLOTS * BODY_SLOTS * BOOL_WIRE_BYTES
)
SLOT_BODY_ID_BYTES_PER_ENVIRONMENT: Final = BODY_SLOTS * INT32_BYTES

OBSERVATION_BYTES_PER_ENVIRONMENT: Final = (
    BODY_FEATURE_BYTES_PER_ENVIRONMENT
    + PLAYER_FEATURE_BYTES_PER_ENVIRONMENT
    + BODY_MASK_BYTES_PER_ENVIRONMENT
    + OWNED_MASK_BYTES_PER_ENVIRONMENT
    + PLAYER_MASK_BYTES_PER_ENVIRONMENT
    + TARGET_MASK_BYTES_PER_ENVIRONMENT
    + SLOT_BODY_ID_BYTES_PER_ENVIRONMENT
)

# Fixed portions of the high-volume messages. Variable-length create and reset
# request fields will be checked while they are encoded in later steps.
STEP_REQUEST_PREFIX_BYTES: Final = UINT32_BYTES
STEP_REQUEST_BYTES_PER_ENVIRONMENT: Final = UINT32_BYTES + BODY_SLOTS * INT8_BYTES
RESET_RESPONSE_PREFIX_BYTES: Final = INT64_BYTES + UINT32_BYTES
RESET_IDENTITY_BYTES_PER_ENVIRONMENT: Final = (
    UINT32_BYTES + INT64_BYTES + UINT8_BYTES + BOOL_WIRE_BYTES
)
STEP_RESPONSE_PREFIX_BYTES: Final = UINT32_BYTES
# Reward, terminated flag, truncated flag, end reason, and turn follow the
# identity columns in every step result.
STEP_RESULT_BYTES_PER_ENVIRONMENT: Final = (
    RESET_IDENTITY_BYTES_PER_ENVIRONMENT
    + FLOAT32_BYTES
    + BOOL_WIRE_BYTES
    + BOOL_WIRE_BYTES
    + UINT8_BYTES
    + UINT32_BYTES
)
CLOSE_REQUEST_BYTES: Final = 0
CLOSE_RESPONSE_BYTES: Final = BOOL_WIRE_BYTES

# The step response is the largest message per environment, so it sets the
# most environments one 64 MiB frame can answer for: 9,482. Go refuses a larger
# batch at create; checking here reports the error before anything is sent.
# This is a wire limit, not a sensible training size.
MAX_BATCH_SIZE: Final = (MAX_PAYLOAD_SIZE - STEP_RESPONSE_PREFIX_BYTES) // (
    STEP_RESULT_BYTES_PER_ENVIRONMENT + OBSERVATION_BYTES_PER_ENVIRONMENT
)

# Values of the step response's end-reason byte. Anything other than ongoing
# must coincide with the terminated flag.
END_ONGOING: Final = 0
END_GAME_FINISHED: Final = 1
END_LEARNER_ELIMINATED: Final = 2


# The format spells out every header field instead of relying on native Python
# or platform alignment: four magic bytes, two one-byte fields, one u16, and
# one u32, all little-endian. Its calculated size must match the frozen 12-byte
# contract above.
_FRAME_HEADER = struct.Struct("<4sBBHI")
if _FRAME_HEADER.size != FRAME_HEADER_SIZE:
    raise RuntimeError(
        f"env-v3 header format is {_FRAME_HEADER.size} bytes, "
        f"want {FRAME_HEADER_SIZE}"
    )

_RESPONSE_FOR_REQUEST: Final = {
    MessageType.CREATE_REQUEST: MessageType.CREATE_RESPONSE,
    MessageType.RESET_REQUEST: MessageType.RESET_RESPONSE,
    MessageType.STEP_REQUEST: MessageType.STEP_RESPONSE,
    MessageType.CLOSE_REQUEST: MessageType.CLOSE_RESPONSE,
}

# These fixed structs encode request fields without native alignment padding.
# Variable-length strings are appended separately after their u32 byte length.
_CREATE_REQUEST_PREFIX = struct.Struct("<Iqqq")
_RESET_OVERRIDE = struct.Struct("<IqB")
_UINT32 = struct.Struct("<I")


class ProtocolError(RuntimeError):
    """Report bytes or stream behavior that violate the env-v3 contract."""


class RemoteProtocolError(ProtocolError):
    """Carry one structured rejection returned by the Go environment."""

    def __init__(
        self,
        request_type: MessageType,
        kind: str,
        message: str,
        env_id: int | None,
    ) -> None:
        """Preserve stable fields so EnvClient can report and count the error."""

        location = "" if env_id is None else f" for environment {env_id}"
        super().__init__(
            f"environment rejected {request_type.name}{location}: {kind}: {message}"
        )
        self.request_type = request_type
        self.kind = kind
        self.remote_message = message
        self.env_id = env_id


@dataclass(frozen=True)
class Frame:
    """One complete, validated message read from the Go process.

    ``read_frame()`` has already removed and checked the 12-byte header before
    constructing this value. The message type tells the next decoder which
    payload layout to use. The payload contains exactly the number of bytes
    declared by the header; it does not include the header itself.

    The payload remains immutable so NumPy views created by later response
    decoders cannot mutate the received bytes underneath another view. Keeping
    the complete payload alive also keeps all such zero-copy views valid.
    """

    # Selects create, reset, step, close, or structured-error decoding.
    message_type: MessageType
    # Raw command-specific bytes following the validated frame header.
    payload: bytes


@dataclass(frozen=True)
class ResetOverride:
    """Request a specific episode assignment for one environment slot.

    A normal reset lets Go choose the next seed and learner seat from its
    deterministic schedule. Supplying this value instead asks Go to recreate a
    named seed-and-seat assignment, which is useful for replay and paired
    evaluation. The environment ID must also appear in the reset request's ID
    list. Go remains responsible for deciding whether the requested seat is
    valid for the configured game.
    """

    # Reusable game slot that will receive this explicit assignment.
    env_id: int
    # Scenario-generator seed used to reconstruct the starting game state.
    seed: int
    # Player position controlled by the learner in the reconstructed game.
    seat: int


@dataclass(frozen=True)
class ObservationDimensions:
    """The four obs-v2 array sizes Go reports when a batch is created.

    obs-v2 fixes its normalization scales in the schema, so Go sends no scales.
    It sends these sizes instead, and the decoder requires each to equal the
    Python constant, so Python learns of a layout mismatch at create rather than
    by misreading the first observation.
    """

    # Body rows per observation: 40 planet slots plus 4 comet slots.
    body_slots: int
    # Numeric columns in each body row.
    body_features: int
    # Player rows per observation: me, clockwise, opposite, counter-clockwise.
    player_slots: int
    # Numeric columns in each player row.
    player_features: int


@dataclass(frozen=True)
class DecodedCreateResponse:
    """Replay and compatibility metadata decoded from a create response.

    No games have started when this value is produced. EnvClient validates the
    identities against its requested configuration before marking the child
    process as created.
    """

    # Go build identity recorded so a run can be reproduced with the same code.
    engine: str
    # Fixed model-facing observation layout, always ``obs-v2``.
    schema: str
    # Seed-to-board generator identity used for every reset.
    generator: str
    # Canonical identity of the scripted opponent Go actually constructed.
    opponent: str
    # Inclusive start and exclusive end of the configured scenario seeds.
    seed_range: tuple[int, int]
    # Deterministic rule that maps the cursor to seed-and-seat assignments.
    schedule: str
    # First unconsumed assignment in that schedule.
    cursor: int
    # Observation sizes, already checked against the Python constants.
    dimensions: ObservationDimensions


@dataclass(frozen=True)
class DecodedObservationBatch:
    """Seven columnar NumPy views describing a batch of game states.

    Axis zero is the environment row in every field. A decoder therefore reads
    each wire column once and later code can select one environment by taking
    the same row from all seven arrays. The arrays are expected to be read-only
    views over one immutable response payload; freezing this dataclass prevents
    field replacement but does not itself make arbitrary NumPy arrays immutable.
    """

    # Shape: [environment, 44 body slots, 27 numeric features].
    bodies: NDArray[np.float32]
    # Shape: [environment, 4 player slots, 6 numeric features].
    players: NDArray[np.float32]
    # Shape: [environment, 44]; true rows contain real bodies.
    body_mask: NDArray[np.bool_]
    # Shape: [environment, 44]; true bodies belong to the learner.
    owned_mask: NDArray[np.bool_]
    # Shape: [environment, 4]; true rows contain seated players.
    player_mask: NDArray[np.bool_]
    # Shape: [environment, 44 sources, 44 targets]; true pairs are legal.
    target_mask: NDArray[np.bool_]
    # Shape: [environment, 44]; maps observation slots back to Go body IDs.
    slot_body_id: NDArray[np.int32]


@dataclass(frozen=True)
class DecodedResetResponse:
    """Columnar episode identities and initial observations returned by reset.

    Every array uses the same strictly increasing environment-ID row order.
    This retains one NumPy allocation/view per wire column rather than creating
    a dictionary and nested observation lists for every environment.
    """

    # First schedule assignment that remains after this reset.
    cursor: int
    # Reusable game-slot IDs, shape [environment].
    env_ids: NDArray[np.uint32]
    # Scenario seed assigned to each row, shape [environment].
    seeds: NDArray[np.int64]
    # Learner player position for each row, shape [environment].
    seats: NDArray[np.uint8]
    # True where the caller supplied a ResetOverride, shape [environment].
    explicit: NDArray[np.bool_]
    # Initial states aligned row-for-row with the identity arrays above.
    observations: DecodedObservationBatch


@dataclass(frozen=True)
class DecodedStepResponse:
    """Columnar identities, turn results, and next observations from one step.

    The response describes the state after Go processed each learner action and
    every scripted opponent's action. Every field shares the ascending
    environment order in ``env_ids`` so callers never need a dictionary lookup
    to align it.
    """

    # Reusable game-slot IDs, shape [environment].
    env_ids: NDArray[np.uint32]
    # Active episode seed for each row, shape [environment].
    seeds: NDArray[np.int64]
    # Learner player position for each row, shape [environment].
    seats: NDArray[np.uint8]
    # True where the active episode came from an explicit reset assignment.
    explicit: NDArray[np.bool_]
    # Reward emitted by this turn, shape [environment].
    rewards: NDArray[np.float32]
    # True where game rules ended the episode, shape [environment].
    terminated: NDArray[np.bool_]
    # True where an external limit ended the episode, shape [environment].
    truncated: NDArray[np.bool_]
    # Why the episode ended, shape [environment]: END_ONGOING (0),
    # END_GAME_FINISHED (1), or END_LEARNER_ELIMINATED (2). Nonzero exactly
    # where ``terminated`` is true.
    end_reasons: NDArray[np.uint8]
    # Authoritative completed turn number, shape [environment].
    turns: NDArray[np.uint32]
    # Post-turn states aligned row-for-row with every result column above.
    observations: DecodedObservationBatch


@dataclass(frozen=True)
class _DecodedEpisodeIdentityColumns:
    """Group the four identity arrays reused by reset and step decoding."""

    env_ids: NDArray[np.uint32]
    seeds: NDArray[np.int64]
    seats: NDArray[np.uint8]
    explicit: NDArray[np.bool_]


class _ColumnDecoder:
    """Advance through one size-checked response without copying its columns.

    The public decoder verifies the complete payload size before constructing
    this helper. Each ``array`` call can therefore create a NumPy view at the
    current byte offset, move to the next documented column, and reshape the
    view without allocating nested Python values.
    """

    def __init__(self, payload: bytes, offset: int) -> None:
        """Retain the immutable payload that owns every returned array's bytes."""

        self._payload = payload
        self._offset = offset

    def array(
        self,
        dtype: np.dtype[Any],
        shape: tuple[int, ...],
        field: str,
    ) -> NDArray[Any]:
        """Expose the next fixed-shape column and advance past its byte range."""

        element_count = math.prod(shape)
        byte_count = element_count * dtype.itemsize
        end = self._offset + byte_count
        if end > len(self._payload):
            # This indicates a decoder-layout bug because callers check the
            # exact complete size before any column is exposed.
            raise RuntimeError(
                f"env-v3 {field} decoder needs bytes through {end}, "
                f"payload has {len(self._payload)}"
            )
        result = np.frombuffer(
            self._payload,
            dtype=dtype,
            count=element_count,
            offset=self._offset,
        ).reshape(shape)
        self._offset = end
        return result

    def boolean_array(
        self,
        shape: tuple[int, ...],
        field: str,
    ) -> NDArray[np.bool_]:
        """Validate Boolean storage, then expose it without copying its bytes.

        NumPy treats every nonzero byte as true. Without this explicit check, a
        corrupt value such as 7 would look like a valid ``True`` and its wire
        corruption would be hidden from the caller.
        """

        wire_values = self.array(BOOL_WIRE_DTYPE, shape, field)
        valid = wire_values <= 1
        if not bool(np.all(valid)):
            flat_index = int(np.flatnonzero(~valid)[0])
            bad_value = int(wire_values.reshape(-1)[flat_index])
            raise ProtocolError(
                f"env-v3 {field} contains Boolean byte {bad_value} "
                f"at flat index {flat_index}, want 0 or 1"
            )
        return wire_values.view(np.bool_)

    def finite_float_array(
        self,
        dtype: np.dtype[Any],
        shape: tuple[int, ...],
        field: str,
    ) -> NDArray[Any]:
        """Expose a floating-point column only when every value is finite.

        NaN and infinity can silently spread through policy calculations and
        poison an optimizer update. Validate the complete NumPy column before
        returning its zero-copy view to training code.
        """

        values = self.array(dtype, shape, field)
        finite = np.isfinite(values)
        if not bool(np.all(finite)):
            flat_index = int(np.flatnonzero(~finite)[0])
            bad_value = values.reshape(-1)[flat_index]
            raise ProtocolError(
                f"env-v3 {field} contains non-finite value {bad_value!r} "
                f"at flat index {flat_index}"
            )
        return values

    def finish(self) -> None:
        """Assert that the declared layout consumed the complete payload."""

        if self._offset != len(self._payload):
            raise RuntimeError(
                f"env-v3 column decoder left "
                f"{len(self._payload) - self._offset} bytes"
            )


def encode_create_payload(
    batch: int,
    seed_range: tuple[int, int],
    opponent: str,
    schedule: str,
    *,
    cursor: int = 0,
) -> bytes:
    """Encode the values that configure one reusable Go environment batch.

    This function checks only representation constraints. Semantic rules such
    as a positive batch size and an increasing seed range remain authoritative
    in Go, so Python and Go cannot disagree about which values are valid. The
    one exception is ``MAX_BATCH_SIZE``: it is a wire limit rather than a game
    rule, and refusing it here names the problem before anything is sent.
    """

    if not isinstance(seed_range, tuple) or len(seed_range) != 2:
        raise TypeError("env-v3 seed_range must be a two-integer tuple")
    checked_batch = _checked_uint32(batch, "batch size")
    if checked_batch > MAX_BATCH_SIZE:
        raise ValueError(
            f"env-v3 batch size {checked_batch} does not fit one step "
            f"response, maximum is {MAX_BATCH_SIZE}"
        )
    seed_start = _checked_int64(seed_range[0], "seed-range start")
    seed_end = _checked_int64(seed_range[1], "seed-range end")
    checked_cursor = _checked_int64(cursor, "schedule cursor")
    opponent_bytes = _encode_string(opponent, MAX_OPPONENT_SIZE, "opponent")
    schedule_bytes = _encode_string(schedule, MAX_SCHEDULE_SIZE, "schedule")
    payload = (
        _CREATE_REQUEST_PREFIX.pack(
            checked_batch,
            seed_start,
            seed_end,
            checked_cursor,
        )
        + opponent_bytes
        + schedule_bytes
    )
    return _checked_payload_size(payload, "create request")


def encode_reset_payload(
    env_ids: Sequence[int] | NDArray[np.integer[Any]],
    overrides: Sequence[ResetOverride] = (),
) -> bytes:
    """Encode a deterministic reset request in ascending environment-ID order.

    Overrides are sorted independently by environment ID. The encoder rejects
    duplicate IDs because they cannot have one canonical row order. Go decides
    whether each override belongs to this reset and whether its seat is valid,
    and reports a structured validation response when it is not.
    """

    sorted_ids = _checked_sorted_env_ids(env_ids, "reset")
    checked_overrides: list[tuple[int, int, int]] = []
    seen_overrides: set[int] = set()
    for index, override in enumerate(overrides):
        if not isinstance(override, ResetOverride):
            raise TypeError(
                f"env-v3 reset override {index} is "
                f"{type(override).__name__}, want ResetOverride"
            )
        env_id = _checked_uint32(override.env_id, f"reset override {index} env_id")
        if env_id in seen_overrides:
            raise ValueError(
                f"env-v3 reset overrides repeat environment ID {env_id}"
            )
        seen_overrides.add(env_id)
        seed = _checked_int64(override.seed, f"reset override {env_id} seed")
        # A u8 can express 0 through 255. Go still decides which of those seats
        # exist, so a representable value such as 2 on a two-player map
        # reaches its semantic validator and produces a structured rejection.
        seat = _checked_uint8(override.seat, f"reset override {env_id} seat")
        checked_overrides.append((env_id, seed, seat))
    checked_overrides.sort(key=lambda item: item[0])

    payload = bytearray()
    payload.extend(_UINT32.pack(len(sorted_ids)))
    if sorted_ids:
        payload.extend(np.asarray(sorted_ids, dtype=UINT32_DTYPE).tobytes())
    override_count = _checked_uint32(
        len(checked_overrides), "reset override count"
    )
    payload.extend(_UINT32.pack(override_count))
    for env_id, seed, seat in checked_overrides:
        payload.extend(_RESET_OVERRIDE.pack(env_id, seed, seat))
    return _checked_payload_size(bytes(payload), "reset request")


def encode_step_payload(
    action_targets: Mapping[int, Sequence[int]] | NDArray[np.integer[Any]],
    *,
    env_ids: Sequence[int] | NDArray[np.integer[Any]] | None = None,
) -> bytes:
    """Encode packed launch targets without building per-slot dictionaries.

    Mapping input retains the defensive per-element checks needed for Python
    values, where ``True`` would otherwise be silently converted to target 1.
    A two-dimensional integer NumPy array can use the faster ``env_ids`` path:
    its dtype already proves that no Python Boolean is hidden in a row, so the
    complete matrix is range-checked without visiting every element in Python.
    ``-1`` becomes the signed byte ``0xff`` and slots 0 through 43 retain their
    numeric byte values.
    """

    if isinstance(action_targets, np.ndarray):
        return _encode_step_array_payload(action_targets, env_ids)
    if env_ids is not None:
        raise TypeError("env-v3 step env_ids are only used with a NumPy target array")
    if not isinstance(action_targets, Mapping):
        raise TypeError(
            "env-v3 step actions must be a mapping or an integer NumPy array"
        )
    checked_rows: list[tuple[int, Sequence[int]]] = []
    seen_ids: set[int] = set()
    for raw_env_id, row in action_targets.items():
        env_id = _checked_uint32(raw_env_id, "step environment ID")
        if env_id in seen_ids:
            raise ValueError(f"env-v3 step repeats environment ID {env_id}")
        seen_ids.add(env_id)
        checked_rows.append((env_id, row))
    checked_rows.sort(key=lambda item: item[0])

    env_ids = [item[0] for item in checked_rows]
    environment_count = _checked_uint32(
        len(env_ids), "step environment count"
    )
    if checked_rows:
        # NumPy would coerce a mixed row such as [True, 2, ...] to integers and
        # erase the fact that a Boolean crossed the API boundary. Check element
        # types first, then let one array conversion handle shapes and values.
        for row_index, (_, row) in enumerate(checked_rows):
            try:
                row_length = len(row)
            except TypeError as error:
                raise TypeError(
                    f"env-v3 environment {env_ids[row_index]} target row "
                    "has no length"
                ) from error
            if row_length != BODY_SLOTS:
                raise ValueError(
                    f"env-v3 environment {env_ids[row_index]} has {row_length} "
                    f"packed targets, want {BODY_SLOTS}"
                )
            for source_slot, target in enumerate(row):
                _checked_integer(
                    target,
                    f"environment {env_ids[row_index]} "
                    f"source slot {source_slot} target",
                )
        try:
            targets = np.asarray([item[1] for item in checked_rows])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "env-v3 step target rows must form one rectangular integer array"
            ) from error
        expected_shape = (len(checked_rows), BODY_SLOTS)
        if targets.shape != expected_shape:
            raise ValueError(
                f"env-v3 step targets have shape {targets.shape}, "
                f"want {expected_shape}"
            )
        if targets.dtype.kind not in {"i", "u"}:
            raise TypeError(
                f"env-v3 step targets have dtype {targets.dtype}, want integers"
            )
        if targets.dtype.kind == "u":
            valid_targets = targets < BODY_SLOTS
        else:
            valid_targets = (targets >= -1) & (targets < BODY_SLOTS)
        if not bool(np.all(valid_targets)):
            row_index, source_slot = np.argwhere(~valid_targets)[0]
            bad_target = targets[row_index, source_slot]
            raise ValueError(
                f"env-v3 environment {env_ids[int(row_index)]} source slot "
                f"{int(source_slot)} has invalid packed target {bad_target!r}"
            )
        packed_targets = targets.astype(INT8_DTYPE, copy=False)
    else:
        packed_targets = np.empty((0, BODY_SLOTS), dtype=INT8_DTYPE)

    payload = bytearray()
    payload.extend(_UINT32.pack(environment_count))
    if env_ids:
        payload.extend(np.asarray(env_ids, dtype=UINT32_DTYPE).tobytes())
    payload.extend(packed_targets.tobytes(order="C"))
    expected_size = (
        STEP_REQUEST_PREFIX_BYTES
        + len(env_ids) * STEP_REQUEST_BYTES_PER_ENVIRONMENT
    )
    if len(payload) != expected_size:
        raise RuntimeError(
            f"env-v3 step encoder produced {len(payload)} bytes, "
            f"expected {expected_size}"
        )
    return _checked_payload_size(bytes(payload), "step request")


def _encode_step_array_payload(
    action_targets: NDArray[np.integer[Any]],
    env_ids: Sequence[int] | NDArray[np.integer[Any]] | None,
) -> bytes:
    """Encode an already columnar action batch without a per-target Python loop."""

    if env_ids is None:
        raise TypeError("env-v3 step NumPy targets require env_ids")
    if action_targets.ndim != 2:
        raise ValueError(
            f"env-v3 step targets have shape {action_targets.shape}, "
            f"want (environments, {BODY_SLOTS})"
        )
    if action_targets.shape[1] != BODY_SLOTS:
        raise ValueError(
            f"env-v3 step targets have shape {action_targets.shape}, "
            f"want ({action_targets.shape[0]}, {BODY_SLOTS})"
        )
    if action_targets.dtype.kind not in {"i", "u"}:
        raise TypeError(
            f"env-v3 step targets have dtype {action_targets.dtype}, want integers"
        )

    # Keep IDs in their input order until the same permutation can be applied
    # to both arrays. Sorting IDs alone would attach action rows to the wrong
    # game whenever a caller supplies IDs such as [10, 2].
    checked_ids = _checked_env_ids_in_input_order(env_ids, "step")
    expected_shape = (len(checked_ids), BODY_SLOTS)
    if action_targets.shape != expected_shape:
        raise ValueError(
            f"env-v3 step targets have shape {action_targets.shape}, "
            f"want {expected_shape}"
        )
    environment_count = _checked_uint32(
        len(checked_ids), "step environment count"
    )

    if action_targets.dtype.kind == "u":
        valid_targets = action_targets < BODY_SLOTS
    else:
        valid_targets = (action_targets >= -1) & (action_targets < BODY_SLOTS)
    if not bool(np.all(valid_targets)):
        row_index, source_slot = np.argwhere(~valid_targets)[0]
        bad_target = action_targets[row_index, source_slot]
        raise ValueError(
            f"env-v3 environment {checked_ids[int(row_index)]} source slot "
            f"{int(source_slot)} has invalid packed target {bad_target!r}"
        )

    order = np.argsort(np.asarray(checked_ids, dtype=UINT32_DTYPE), kind="stable")
    sorted_ids = np.asarray(checked_ids, dtype=UINT32_DTYPE)[order]

    # Sorting places equal environment IDs next to each other. Comparing every
    # ID except the first with every ID except the last therefore produces True
    # wherever two neighboring rows contain the same ID. The size check avoids
    # doing that comparison when the batch has zero or one environment.
    if sorted_ids.size > 1 and bool(np.any(sorted_ids[1:] == sorted_ids[:-1])):
        # flatnonzero returns the positions where the neighboring comparison is
        # True. Select the first one so the error consistently reports the first
        # repeated ID in numeric order.
        duplicate_index = int(np.flatnonzero(sorted_ids[1:] == sorted_ids[:-1])[0])
        raise ValueError(
            f"env-v3 step repeats environment ID {int(sorted_ids[duplicate_index])}"
        )
    packed_targets = action_targets[order].astype(INT8_DTYPE, copy=False)

    payload = (
        _UINT32.pack(environment_count)
        + sorted_ids.tobytes(order="C")
        + packed_targets.tobytes(order="C")
    )
    expected_size = (
        STEP_REQUEST_PREFIX_BYTES
        + len(checked_ids) * STEP_REQUEST_BYTES_PER_ENVIRONMENT
    )
    if len(payload) != expected_size:
        raise RuntimeError(
            f"env-v3 step encoder produced {len(payload)} bytes, "
            f"expected {expected_size}"
        )
    return _checked_payload_size(payload, "step request")


def encode_close_payload() -> bytes:
    """Return the required empty close-request payload."""

    return b""


def decode_create_response(payload: bytes) -> DecodedCreateResponse:
    """Decode and validate the metadata returned by a successful create.

    The frame header has already established ``env-v3``. This payload must
    identify the ``obs-v2`` layout, contain valid UTF-8 strings, and report
    exactly the four dimensions this module decodes before EnvClient may store
    it.
    """

    checked_payload = _checked_response_payload(payload, "create response")
    offset = 0
    engine, offset = _decode_string(checked_payload, offset, "create engine")
    schema, offset = _decode_string(checked_payload, offset, "create schema")
    generator, offset = _decode_string(
        checked_payload, offset, "create generator"
    )
    opponent, offset = _decode_string(checked_payload, offset, "create opponent")
    seed_start, offset = _decode_int64(
        checked_payload, offset, "create seed-range start"
    )
    seed_end, offset = _decode_int64(
        checked_payload, offset, "create seed-range end"
    )
    schedule, offset = _decode_string(
        checked_payload, offset, "create schedule"
    )
    cursor, offset = _decode_int64(checked_payload, offset, "create cursor")

    # Four u32 sizes close the payload, in ObservationDimensions field order.
    # Each must equal the size this module was written for: decoding a later
    # observation with different sizes would misplace every column after it.
    dimension_fields = (
        ("body slots", BODY_SLOTS),
        ("body features", BODY_FEATURES),
        ("player slots", PLAYER_SLOTS),
        ("player features", PLAYER_FEATURES),
    )
    dimension_values: list[int] = []
    for field, expected in dimension_fields:
        value, offset = _decode_uint32(
            checked_payload, offset, f"create {field}"
        )
        if value != expected:
            raise ProtocolError(
                f"env-v3 create reports {value} {field}, want {expected}"
            )
        dimension_values.append(value)
    _require_complete_payload(checked_payload, offset, "create response")

    if schema != OBSERVATION_SCHEMA:
        raise ProtocolError(
            f"env-v3 observation schema is {schema!r}, "
            f"want {OBSERVATION_SCHEMA!r}"
        )
    return DecodedCreateResponse(
        engine=engine,
        schema=schema,
        generator=generator,
        opponent=opponent,
        seed_range=(seed_start, seed_end),
        schedule=schedule,
        cursor=cursor,
        dimensions=ObservationDimensions(*dimension_values),
    )


def decode_reset_response(
    payload: bytes,
    *,
    expected_env_ids: Sequence[int] | NDArray[np.integer[Any]],
) -> DecodedResetResponse:
    """Decode reset identities and observations as zero-copy NumPy columns.

    The response size is checked from ``N`` before the first call to
    ``numpy.frombuffer()``. A truncated or overlong payload therefore cannot
    expose a partially valid set of arrays to the caller.

    Malformed response bytes raise :class:`ProtocolError`. Invalid
    ``expected_env_ids`` raise ``TypeError`` or ``ValueError`` because those
    IDs came from the local Python caller rather than from Go.
    """

    checked_payload = _checked_response_payload(payload, "reset response")
    if len(checked_payload) < RESET_RESPONSE_PREFIX_BYTES:
        raise ProtocolError(
            f"env-v3 reset response has {len(checked_payload)} bytes, "
            f"needs at least {RESET_RESPONSE_PREFIX_BYTES}"
        )
    cursor = struct.unpack_from("<q", checked_payload, 0)[0]
    environment_count = struct.unpack_from("<I", checked_payload, INT64_BYTES)[0]
    expected_size = RESET_RESPONSE_PREFIX_BYTES + environment_count * (
        RESET_IDENTITY_BYTES_PER_ENVIRONMENT
        + OBSERVATION_BYTES_PER_ENVIRONMENT
    )
    _require_exact_payload_size(checked_payload, expected_size, "reset response")

    columns = _ColumnDecoder(checked_payload, RESET_RESPONSE_PREFIX_BYTES)
    identities = _decode_episode_identity_columns(
        columns, environment_count
    )
    _validate_episode_identity_columns(
        identities,
        expected_env_ids,
        "reset response",
    )
    observations = _decode_observation_columns(columns, environment_count)
    columns.finish()
    return DecodedResetResponse(
        cursor=cursor,
        env_ids=identities.env_ids,
        seeds=identities.seeds,
        seats=identities.seats,
        explicit=identities.explicit,
        observations=observations,
    )


def decode_step_response(
    payload: bytes,
    *,
    expected_env_ids: Sequence[int] | NDArray[np.integer[Any]],
) -> DecodedStepResponse:
    """Decode one post-turn result batch as zero-copy NumPy columns.

    Result scalars appear between the shared episode-identity columns and the
    shared observation block. All arrays retain the same axis-zero row order.

    Malformed response bytes raise :class:`ProtocolError`. Invalid
    ``expected_env_ids`` raise ``TypeError`` or ``ValueError`` because those
    IDs came from the local Python caller rather than from Go.
    """

    checked_payload = _checked_response_payload(payload, "step response")
    if len(checked_payload) < STEP_RESPONSE_PREFIX_BYTES:
        raise ProtocolError(
            f"env-v3 step response has {len(checked_payload)} bytes, "
            f"needs at least {STEP_RESPONSE_PREFIX_BYTES}"
        )
    environment_count = struct.unpack_from("<I", checked_payload, 0)[0]
    expected_size = STEP_RESPONSE_PREFIX_BYTES + environment_count * (
        STEP_RESULT_BYTES_PER_ENVIRONMENT
        + OBSERVATION_BYTES_PER_ENVIRONMENT
    )
    _require_exact_payload_size(checked_payload, expected_size, "step response")

    columns = _ColumnDecoder(checked_payload, STEP_RESPONSE_PREFIX_BYTES)
    identities = _decode_episode_identity_columns(
        columns, environment_count
    )
    _validate_episode_identity_columns(
        identities,
        expected_env_ids,
        "step response",
    )
    rewards = columns.finite_float_array(
        FLOAT32_DTYPE, (environment_count,), "step rewards"
    )
    terminated = columns.boolean_array(
        (environment_count,), "step terminated flags"
    )
    truncated = columns.boolean_array(
        (environment_count,), "step truncated flags"
    )
    end_reasons = columns.array(
        UINT8_DTYPE, (environment_count,), "step end reasons"
    )
    _validate_end_reasons(identities.env_ids, end_reasons, terminated)
    turns = columns.array(UINT32_DTYPE, (environment_count,), "step turns")
    observations = _decode_observation_columns(columns, environment_count)
    columns.finish()
    return DecodedStepResponse(
        env_ids=identities.env_ids,
        seeds=identities.seeds,
        seats=identities.seats,
        explicit=identities.explicit,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        end_reasons=end_reasons,
        turns=turns,
        observations=observations,
    )


def decode_close_response(payload: bytes) -> bool:
    """Require the one-byte acknowledgement sent after Go closes its batch."""

    checked_payload = _checked_response_payload(payload, "close response")
    if len(checked_payload) != CLOSE_RESPONSE_BYTES:
        raise ProtocolError(
            f"env-v3 close response has {len(checked_payload)} bytes, "
            f"want exactly {CLOSE_RESPONSE_BYTES}"
        )
    if checked_payload[0] != 1:
        raise ProtocolError(
            f"env-v3 close response is {checked_payload[0]}, want 1"
        )
    return True


def _decode_episode_identity_columns(
    columns: _ColumnDecoder,
    environment_count: int,
) -> _DecodedEpisodeIdentityColumns:
    """Decode the four identity columns shared by reset and step responses."""

    shape = (environment_count,)
    env_ids = columns.array(UINT32_DTYPE, shape, "environment IDs")
    seeds = columns.array(INT64_DTYPE, shape, "episode seeds")
    seats = columns.array(UINT8_DTYPE, shape, "learner seats")
    explicit = columns.boolean_array(shape, "explicit-assignment flags")
    return _DecodedEpisodeIdentityColumns(
        env_ids=env_ids,
        seeds=seeds,
        seats=seats,
        explicit=explicit,
    )


def _validate_episode_identity_columns(
    identities: _DecodedEpisodeIdentityColumns,
    expected_env_ids: Sequence[int] | NDArray[np.integer[Any]],
    field: str,
) -> None:
    """Require safe seats and the exact sorted IDs named by the request.

    Reset and step responses store related values in separate columns. Row 0
    of ``seeds`` and row 0 of ``seats``, for example, both describe row 0 of
    ``env_ids``. These checks ensure those rows can be matched safely to the
    games Python requested before any observation reaches the model.
    """

    env_ids = identities.env_ids
    # Go promises to return environment rows in increasing numeric ID order.
    # Compare each ID with the ID immediately before it. For [2, 5, 9], this
    # evaluates 5 > 2 and 9 > 5. Equality catches duplicates, while a smaller
    # next value catches an out-of-order response.
    if env_ids.size > 1:
        increasing = env_ids[1:] > env_ids[:-1]
        if not bool(np.all(increasing)):
            # Report the first bad neighboring pair so the protocol failure
            # identifies where the returned ordering stopped being valid.
            previous_index = int(np.flatnonzero(~increasing)[0])
            raise ProtocolError(
                f"env-v3 {field} environment IDs are not strictly increasing: "
                f"{int(env_ids[previous_index + 1])} follows "
                f"{int(env_ids[previous_index])}"
            )

    # A caller may request IDs in any order, but both the encoder and Go use
    # numeric order on the wire. Validate and sort the request IDs so [9, 3]
    # is compared with the legal response order [3, 9]. NumPy request IDs take
    # a vectorized path because the hot step loop already owns them as an array.
    expected = _checked_expected_env_id_array(
        expected_env_ids,
        f"expected {field}",
    )

    # A response with too few or too many rows belongs to a different request
    # or is incomplete. Check the count before comparing arrays so NumPy does
    # not broadcast arrays of different lengths.
    if env_ids.size != expected.size:
        raise ProtocolError(
            f"env-v3 {field} returned {env_ids.size} environments, "
            f"request named {expected.size}"
        )

    # Counts alone are insufficient: [3, 8] and [3, 9] both contain two IDs.
    # Compare each returned row with the sorted request and report the first
    # mismatch, since every later identity and observation column uses this
    # same row order.
    mismatched = env_ids != expected
    if bool(np.any(mismatched)):
        row = int(np.flatnonzero(mismatched)[0])
        raise ProtocolError(
            f"env-v3 {field} environment ID at row {row} is "
            f"{int(env_ids[row])}, want {int(expected[row])}"
        )

    # env-v3 stores a seat in one unsigned byte, which could represent 0..255.
    # A map seats at most four players, so only seats 0 through 3 may be
    # returned. Go checks the seat against each map's actual roster; Python only
    # bounds it. Use the already-validated environment ID in the error message.
    valid_seats = identities.seats < PLAYER_SLOTS
    if not bool(np.all(valid_seats)):
        row = int(np.flatnonzero(~valid_seats)[0])
        raise ProtocolError(
            f"env-v3 {field} environment {int(env_ids[row])} has seat "
            f"{int(identities.seats[row])}, want 0 through {PLAYER_SLOTS - 1}"
        )


def _validate_end_reasons(
    env_ids: NDArray[np.uint32],
    end_reasons: NDArray[np.uint8],
    terminated: NDArray[np.bool_],
) -> None:
    """Require every end reason to be known and to agree with ``terminated``.

    The two columns describe the same fact, so a reader trusting either alone
    must be told the same thing: a nonzero reason exactly where the episode
    terminated. Go refuses to encode a disagreement; this catches corruption.
    """

    known = end_reasons <= END_LEARNER_ELIMINATED
    if not bool(np.all(known)):
        row = int(np.flatnonzero(~known)[0])
        raise ProtocolError(
            f"env-v3 step environment {int(env_ids[row])} has end reason "
            f"{int(end_reasons[row])}, want 0 through {END_LEARNER_ELIMINATED}"
        )
    agrees = (end_reasons != END_ONGOING) == terminated
    if not bool(np.all(agrees)):
        row = int(np.flatnonzero(~agrees)[0])
        raise ProtocolError(
            f"env-v3 step environment {int(env_ids[row])} has end reason "
            f"{int(end_reasons[row])} but terminated={bool(terminated[row])}"
        )


def _decode_observation_columns(
    columns: _ColumnDecoder,
    environment_count: int,
) -> DecodedObservationBatch:
    """Create the seven obs-v2 array views in their documented wire order.

    Floats are checked for being finite but not for range: obs-v2 ship columns
    use a reference scale rather than a cap, so values above 1 are legitimate.
    """

    bodies = columns.finite_float_array(
        FLOAT32_DTYPE,
        (environment_count, BODY_SLOTS, BODY_FEATURES),
        "observation bodies",
    )
    players = columns.finite_float_array(
        FLOAT32_DTYPE,
        (environment_count, PLAYER_SLOTS, PLAYER_FEATURES),
        "observation players",
    )
    body_mask = columns.boolean_array(
        (environment_count, BODY_SLOTS), "observation body mask"
    )
    owned_mask = columns.boolean_array(
        (environment_count, BODY_SLOTS), "observation owned-body mask"
    )
    player_mask = columns.boolean_array(
        (environment_count, PLAYER_SLOTS), "observation player mask"
    )
    target_mask = columns.boolean_array(
        (environment_count, BODY_SLOTS, BODY_SLOTS),
        "observation target mask",
    )
    slot_body_id = columns.array(
        INT32_DTYPE,
        (environment_count, BODY_SLOTS),
        "observation slot-to-body IDs",
    )
    return DecodedObservationBatch(
        bodies=bodies,
        players=players,
        body_mask=body_mask,
        owned_mask=owned_mask,
        player_mask=player_mask,
        target_mask=target_mask,
        slot_body_id=slot_body_id,
    )


def _checked_response_payload(payload: object, field: str) -> bytes:
    """Require immutable, frame-bounded storage for zero-copy response views."""

    if not isinstance(payload, bytes):
        raise TypeError(
            f"env-v3 {field} payload is {type(payload).__name__}, want bytes"
        )
    if len(payload) > MAX_PAYLOAD_SIZE:
        raise ProtocolError(
            f"env-v3 {field} has {len(payload)} bytes, "
            f"maximum is {MAX_PAYLOAD_SIZE}"
        )
    return payload


def _require_exact_payload_size(
    payload: bytes,
    expected_size: int,
    field: str,
) -> None:
    """Reject truncated and trailing data before exposing any column views."""

    if len(payload) != expected_size:
        raise ProtocolError(
            f"env-v3 {field} has {len(payload)} bytes, "
            f"count requires exactly {expected_size}"
        )


def _checked_integer(value: object, field: str) -> int:
    """Return one Python integer while rejecting Boolean lookalikes."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"env-v3 {field} is {value!r}, want an integer")
    return int(value)


def _checked_integer_range(
    value: object,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    """Require one integer to fit the exact scalar used on the wire."""

    checked = _checked_integer(value, field)
    if checked < minimum or checked > maximum:
        raise ValueError(
            f"env-v3 {field} is {checked}, want {minimum} through {maximum}"
        )
    return checked


def _checked_uint8(value: object, field: str) -> int:
    """Require a value representable by one unsigned byte."""

    return _checked_integer_range(value, field, 0, (1 << 8) - 1)


def _checked_uint32(value: object, field: str) -> int:
    """Require a value representable by a little-endian u32."""

    return _checked_integer_range(value, field, 0, (1 << 32) - 1)


def _checked_int64(value: object, field: str) -> int:
    """Require a value representable by a little-endian i64."""

    return _checked_integer_range(value, field, -(1 << 63), (1 << 63) - 1)


def _checked_expected_env_id_array(
    env_ids: Sequence[int] | NDArray[np.integer[Any]],
    field: str,
) -> NDArray[np.uint32]:
    """Return validated request IDs in the order Go must return them.

    The rollout hot path already keeps environment IDs in a NumPy array. Check
    that complete array with NumPy operations rather than converting 1,024 IDs
    to Python integers and validating them one at a time. Ordinary sequences
    retain the detailed existing checks because they are used only by colder
    paths and tests.
    """

    if not isinstance(env_ids, np.ndarray):
        return np.asarray(
            _checked_sorted_env_ids(env_ids, field),
            dtype=UINT32_DTYPE,
        )
    if env_ids.ndim != 1:
        raise ValueError(
            f"env-v3 {field} environment IDs have shape {env_ids.shape}, "
            "want one dimension"
        )
    if env_ids.dtype.kind not in {"i", "u"}:
        raise TypeError(
            f"env-v3 {field} environment IDs have dtype {env_ids.dtype}, "
            "want integers"
        )

    # Signed integer arrays can contain negative IDs. Both signed and unsigned
    # arrays may also use a wider dtype than u32, so check both wire boundaries
    # before narrowing the values to the four-byte protocol representation.
    if env_ids.dtype.kind == "i":
        representable = (env_ids >= 0) & (env_ids <= (1 << 32) - 1)
    else:
        representable = env_ids <= (1 << 32) - 1
    if not bool(np.all(representable)):
        row = int(np.flatnonzero(~representable)[0])
        raise ValueError(
            f"env-v3 {field} environment ID {row} is {int(env_ids[row])}, "
            f"want 0 through {(1 << 32) - 1}"
        )
    checked = env_ids.astype(UINT32_DTYPE, copy=False)

    # Most training requests already contain [0, 1, 2, ...]. Return that array
    # directly when it is strictly increasing. Unsorted input is legal, so only
    # that less-common case pays for a sort. Sorting also places duplicates next
    # to each other, allowing one vectorized equality check to reject them.
    if checked.size < 2 or bool(np.all(checked[1:] > checked[:-1])):
        return checked
    sorted_ids = np.sort(checked, kind="stable")
    duplicates = sorted_ids[1:] == sorted_ids[:-1]
    if bool(np.any(duplicates)):
        row = int(np.flatnonzero(duplicates)[0])
        raise ValueError(
            f"env-v3 {field} repeats environment ID {int(sorted_ids[row])}"
        )
    return sorted_ids


def _checked_env_ids_in_input_order(
    env_ids: Sequence[int] | NDArray[np.integer[Any]],
    command: str,
) -> list[int]:
    """Validate one ID vector while retaining its action-row alignment."""

    if isinstance(env_ids, np.ndarray):
        if env_ids.ndim != 1:
            raise ValueError(
                f"env-v3 {command} environment IDs have shape {env_ids.shape}, "
                "want one dimension"
            )
        if env_ids.dtype.kind not in {"i", "u"}:
            raise TypeError(
                f"env-v3 {command} environment IDs have dtype {env_ids.dtype}, "
                "want integers"
            )
        values: Sequence[object] = env_ids.tolist()
    elif isinstance(env_ids, Sequence) and not isinstance(
        env_ids, (str, bytes, bytearray)
    ):
        values = env_ids
    else:
        raise TypeError(f"env-v3 {command} environment IDs must be a sequence")

    return [
        _checked_uint32(value, f"{command} environment ID {index}")
        for index, value in enumerate(values)
    ]


def _checked_sorted_env_ids(
    env_ids: Sequence[int] | NDArray[np.integer[Any]],
    command: str,
) -> list[int]:
    """Validate IDs, reject duplicates, and return deterministic numeric order."""

    checked = _checked_env_ids_in_input_order(env_ids, command)
    seen: set[int] = set()
    for env_id in checked:
        if env_id in seen:
            raise ValueError(f"env-v3 {command} repeats environment ID {env_id}")
        seen.add(env_id)
    _checked_uint32(len(checked), f"{command} environment count")
    checked.sort()
    return checked


def _encode_string(value: object, maximum: int, field: str) -> bytes:
    """Encode one bounded UTF-8 string with its little-endian u32 length."""

    if not isinstance(value, str):
        raise TypeError(
            f"env-v3 {field} is {type(value).__name__}, want a string"
        )
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"env-v3 {field} is not valid UTF-8") from error
    if len(encoded) > maximum:
        raise ValueError(
            f"env-v3 {field} has {len(encoded)} UTF-8 bytes, maximum is {maximum}"
        )
    return _UINT32.pack(len(encoded)) + encoded


def _checked_payload_size(payload: bytes, field: str) -> bytes:
    """Reject a request payload that cannot fit inside one env-v3 frame."""

    if len(payload) > MAX_PAYLOAD_SIZE:
        raise ValueError(
            f"env-v3 {field} has {len(payload)} bytes, "
            f"maximum is {MAX_PAYLOAD_SIZE}"
        )
    return payload


def encode_frame(
    message_type: MessageType,
    payload: bytes | bytearray | memoryview,
) -> bytes:
    """Return one complete frame ready to write to the Go process.

    The flags and reserved fields are always zero. Callers supply only the
    message type and its already encoded payload, which prevents ordinary
    requests from accidentally setting unsupported header options.
    """

    checked_type = _checked_message_type(message_type)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("env-v3 payload must be a contiguous bytes-like value")
    try:
        contiguous = not isinstance(payload, memoryview) or payload.contiguous
        payload_bytes = bytes(payload)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "env-v3 payload must be a contiguous bytes-like value"
        ) from error
    if not contiguous:
        raise TypeError("env-v3 payload must be a contiguous bytes-like value")
    payload_size = len(payload_bytes)
    if payload_size > MAX_PAYLOAD_SIZE:
        raise ProtocolError(
            f"env-v3 payload has {payload_size} bytes, maximum is {MAX_PAYLOAD_SIZE}"
        )
    header = _FRAME_HEADER.pack(
        FRAME_MAGIC,
        int(checked_type),
        0,  # Flags are reserved for a future protocol version.
        0,  # Reserved header bytes must also remain zero.
        payload_size,
    )
    return header + payload_bytes


def write_frame(
    output: BinaryIO,
    message_type: MessageType,
    payload: bytes | bytearray | memoryview,
) -> None:
    """Write and flush one frame, handling streams that accept partial writes.

    The Go process cannot begin decoding until buffered Python output reaches
    its stdin pipe. Flushing here keeps that delivery rule next to the write and
    prevents a caller from starting a response read while its request is still
    waiting in Python's buffer.
    """

    frame = encode_frame(message_type, payload)
    remaining = memoryview(frame)
    while remaining:
        try:
            written = output.write(remaining)
        except (OSError, TypeError, ValueError) as error:
            raise ProtocolError("failed to write env-v3 frame") from error
        if written is None or isinstance(written, bool) or written <= 0:
            raise ProtocolError("env-v3 stream accepted no frame bytes")
        if written > len(remaining):
            raise ProtocolError(
                f"env-v3 stream reported writing {written} bytes from a "
                f"{len(remaining)}-byte buffer"
            )
        remaining = remaining[written:]
    try:
        output.flush()
    except (OSError, ValueError) as error:
        raise ProtocolError("failed to flush env-v3 frame") from error


def read_frame(input_stream: BinaryIO) -> Frame:
    """Read and validate exactly one frame from the Go process.

    End-of-file is an error here because the Python client calls this function
    only after sending a request that requires one response. The caller must
    discard the process after any failure because its next byte boundary can no
    longer be trusted.
    """

    header = _read_exact(input_stream, FRAME_HEADER_SIZE, "frame header")
    magic, message_value, flags, reserved, payload_size = _FRAME_HEADER.unpack(
        header
    )
    if magic != FRAME_MAGIC:
        raise ProtocolError(
            f"env-v3 frame magic is {magic!r}, want {FRAME_MAGIC!r}"
        )
    if flags != 0:
        raise ProtocolError(f"env-v3 frame flags are {flags}, want 0")
    if reserved != 0:
        raise ProtocolError(
            f"env-v3 frame reserved field is {reserved}, want 0"
        )
    try:
        message_type = MessageType(message_value)
    except ValueError as error:
        raise ProtocolError(
            f"env-v3 frame has unknown message type {message_value:#x}"
        ) from error
    if payload_size > MAX_PAYLOAD_SIZE:
        raise ProtocolError(
            f"env-v3 payload has {payload_size} bytes, maximum is {MAX_PAYLOAD_SIZE}"
        )
    payload = _read_exact(input_stream, payload_size, "frame payload")
    return Frame(message_type=message_type, payload=payload)


def read_response_frame(
    input_stream: BinaryIO,
    request_type: MessageType,
) -> Frame:
    """Read the one response paired with ``request_type``.

    A structured error frame is a complete response, but callers should handle
    it as an exception rather than trying to decode it as the expected success
    payload. Any other response type mismatch means Python and Go disagree
    about the request/response sequence, so continuing would be unsafe.
    """

    checked_request = _checked_request_type(request_type)
    frame = read_frame(input_stream)
    if frame.message_type is MessageType.ERROR_RESPONSE:
        raise _decode_remote_error(frame.payload, checked_request)
    expected = _RESPONSE_FOR_REQUEST[checked_request]
    if frame.message_type is not expected:
        raise ProtocolError(
            f"env-v3 {checked_request.name} received {frame.message_type.name}, "
            f"want {expected.name}"
        )
    return frame


def _checked_message_type(value: MessageType) -> MessageType:
    """Reject Boolean and unknown numeric values before writing a header."""

    if isinstance(value, bool):
        raise TypeError("env-v3 message type must not be Boolean")
    try:
        return MessageType(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown env-v3 message type {value!r}") from error


def _checked_request_type(value: MessageType) -> MessageType:
    """Require one of the four message types Python is allowed to send."""

    checked = _checked_message_type(value)
    if checked not in _RESPONSE_FOR_REQUEST:
        raise ValueError(f"{checked.name} is not an env-v3 request type")
    return checked


def _read_exact(input_stream: BinaryIO, size: int, field: str) -> bytes:
    """Collect exactly ``size`` bytes even when a pipe returns short reads."""

    chunks: list[bytes] = []
    received = 0
    while received < size:
        try:
            chunk = input_stream.read(size - received)
        except (OSError, ValueError) as error:
            raise ProtocolError(f"failed to read env-v3 {field}") from error
        if chunk is None:
            raise ProtocolError(f"env-v3 stream returned no value for {field}")
        if not isinstance(chunk, bytes):
            raise ProtocolError(
                f"env-v3 stream returned {type(chunk).__name__} for {field}, "
                "want bytes"
            )
        if not chunk:
            raise ProtocolError(
                f"truncated env-v3 {field}: received {received} of {size} bytes"
            )
        if len(chunk) > size - received:
            raise ProtocolError(
                f"env-v3 stream returned {len(chunk)} bytes while {size - received} "
                f"bytes remained in {field}"
            )
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _decode_remote_error(
    payload: bytes,
    expected_request: MessageType,
) -> RemoteProtocolError:
    """Validate the stable fields in one Go error-response payload."""

    # The fixed prefix contains the failed u8 request type and an i32
    # environment ID. Two length-prefixed strings follow it.
    if len(payload) < UINT8_BYTES + INT32_BYTES:
        raise ProtocolError("env-v3 error payload is truncated before its fields")
    failed_value = payload[0]
    env_id_value = struct.unpack_from("<i", payload, UINT8_BYTES)[0]
    offset = UINT8_BYTES + INT32_BYTES
    kind, offset = _decode_string(payload, offset, "error kind")
    message, offset = _decode_string(payload, offset, "error message")
    if offset != len(payload):
        raise ProtocolError(
            f"env-v3 error payload has {len(payload) - offset} trailing bytes"
        )
    if failed_value != int(expected_request):
        raise ProtocolError(
            f"env-v3 error answers request type {failed_value:#x}, "
            f"want {int(expected_request):#x}"
        )
    if env_id_value < -1:
        raise ProtocolError(
            f"env-v3 error environment ID is {env_id_value}, want -1 or greater"
        )
    env_id = None if env_id_value == -1 else env_id_value
    return RemoteProtocolError(
        request_type=expected_request,
        kind=kind,
        message=message,
        env_id=env_id,
    )


def _decode_string(
    payload: bytes,
    offset: int,
    field: str,
) -> tuple[str, int]:
    """Decode one bounded, length-prefixed UTF-8 string from a payload."""

    if len(payload) - offset < UINT32_BYTES:
        raise ProtocolError(f"env-v3 {field} is missing its length")
    length = struct.unpack_from("<I", payload, offset)[0]
    offset += UINT32_BYTES
    if length > MAX_STRING_SIZE:
        raise ProtocolError(
            f"env-v3 {field} has {length} bytes, maximum is {MAX_STRING_SIZE}"
        )
    end = offset + length
    if end > len(payload):
        raise ProtocolError(
            f"env-v3 {field} needs {length} bytes, "
            f"only {len(payload) - offset} remain"
        )
    try:
        value = payload[offset:end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProtocolError(f"env-v3 {field} is not valid UTF-8") from error
    return value, end


def _decode_int64(payload: bytes, offset: int, field: str) -> tuple[int, int]:
    """Read one little-endian signed integer without crossing the payload end."""

    end = offset + INT64_BYTES
    if end > len(payload):
        raise ProtocolError(
            f"env-v3 {field} needs {INT64_BYTES} bytes, "
            f"only {len(payload) - offset} remain"
        )
    return struct.unpack_from("<q", payload, offset)[0], end


def _decode_uint32(payload: bytes, offset: int, field: str) -> tuple[int, int]:
    """Read one little-endian unsigned integer without crossing the payload end."""

    end = offset + UINT32_BYTES
    if end > len(payload):
        raise ProtocolError(
            f"env-v3 {field} needs {UINT32_BYTES} bytes, "
            f"only {len(payload) - offset} remain"
        )
    return struct.unpack_from("<I", payload, offset)[0], end


def _require_complete_payload(payload: bytes, offset: int, field: str) -> None:
    """Reject bytes left after a variable-layout decoder reads every field."""

    if offset != len(payload):
        raise ProtocolError(
            f"env-v3 {field} has {len(payload) - offset} trailing bytes"
        )
