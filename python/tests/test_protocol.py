"""Focused tests for env-v3 framing, stream handling, and error responses."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

import orrery.protocol as protocol


# obs-v2 sizes written out independently of the production constants, so a
# fixture built from them checks the codec rather than repeating it.
BODIES = 44
BODY_COLUMNS = 27
PLAYERS = 4
PLAYER_COLUMNS = 6


class ChunkedReader:
    """Return only a few bytes per read to imitate a fragmented OS pipe."""

    def __init__(self, data: bytes, chunk_size: int = 1) -> None:
        """Keep unread bytes and the largest fragment returned per call."""

        self._data = data
        self._offset = 0
        self._chunk_size = chunk_size

    def read(self, size: int) -> bytes:
        """Return at most ``chunk_size`` bytes without crossing the input end."""

        count = min(size, self._chunk_size, len(self._data) - self._offset)
        start = self._offset
        self._offset += count
        return self._data[start : self._offset]


class ChunkedWriter:
    """Accept only a few bytes per write and retain the complete result."""

    def __init__(self, chunk_size: int = 2) -> None:
        """Start with an empty output buffer and a fixed partial-write size."""

        self.data = bytearray()
        self._chunk_size = chunk_size
        self.flushed = False

    def write(self, data: bytes | memoryview) -> int:
        """Save one prefix and report exactly how many bytes were accepted."""

        count = min(len(data), self._chunk_size)
        self.data.extend(data[:count])
        return count

    def flush(self) -> None:
        """Record that buffered request bytes would now cross the pipe."""

        self.flushed = True


def _frame(
    message_type: int,
    payload: bytes = b"",
    *,
    magic: bytes = b"OWE3",
    flags: int = 0,
    reserved: int = 0,
    declared_size: int | None = None,
) -> bytes:
    """Build test bytes independently from the production frame encoder."""

    size = len(payload) if declared_size is None else declared_size
    return (
        magic
        + bytes((message_type, flags))
        + reserved.to_bytes(2, "little")
        + size.to_bytes(4, "little")
        + payload
    )


def _wire_string(value: bytes) -> bytes:
    """Build one raw length-prefixed string without using codec helpers."""

    return len(value).to_bytes(4, "little") + value


def _u32(value: int) -> bytes:
    """Encode a test u32 independently from the production request encoder."""

    return value.to_bytes(4, "little")


def _i64(value: int) -> bytes:
    """Encode a test i64 independently from the production request encoder."""

    return value.to_bytes(8, "little", signed=True)


def _create_response_wire(
    *,
    schema: bytes = b"obs-v2",
    dimensions: tuple[int, ...] = (BODIES, BODY_COLUMNS, PLAYERS, PLAYER_COLUMNS),
) -> bytes:
    """Build a complete create response in the documented field order."""

    return (
        _wire_string(b"engine-commit")
        + _wire_string(schema)
        + _wire_string(b"generator-v2")
        + _wire_string(b"idle-v1")
        + _i64(0)
        + _i64(1_000_000)
        + _wire_string(b"train-v2")
        + _i64(17)
        + b"".join(_u32(value) for value in dimensions)
    )


def _observation_wire(
    count: int,
) -> tuple[bytes, dict[str, np.ndarray]]:
    """Build all seven observation columns in documented order."""

    bodies = np.arange(count * BODIES * BODY_COLUMNS, dtype="<f4").reshape(
        count, BODIES, BODY_COLUMNS
    )
    players = (
        np.arange(count * PLAYERS * PLAYER_COLUMNS, dtype="<f4").reshape(
            count, PLAYERS, PLAYER_COLUMNS
        )
        + 1_000
    )
    body_mask = (np.arange(count * BODIES, dtype="u1") % 2).reshape(count, BODIES)
    owned_mask = (1 - body_mask).astype("u1", copy=False)
    player_mask = (np.arange(count * PLAYERS, dtype="u1") % 2).reshape(
        count, PLAYERS
    )
    target_mask = (np.arange(count * BODIES * BODIES, dtype="u1") % 2).reshape(
        count, BODIES, BODIES
    )
    slot_body_id = (
        np.arange(count * BODIES, dtype="<i4").reshape(count, BODIES) - 1
    )
    columns = {
        "bodies": bodies,
        "players": players,
        "body_mask": body_mask,
        "owned_mask": owned_mask,
        "player_mask": player_mask,
        "target_mask": target_mask,
        "slot_body_id": slot_body_id,
    }
    return b"".join(array.tobytes(order="C") for array in columns.values()), columns


def _step_response_wire(
    env_ids: tuple[int, ...] = (3,),
    *,
    terminated: tuple[int, ...] | None = None,
    end_reasons: tuple[int, ...] | None = None,
) -> bytes:
    """Build a step payload that corruption tests can alter by offset.

    By default every episode is still running: not terminated, end reason 0.
    """

    count = len(env_ids)
    observation_wire, _ = _observation_wire(count)
    return (
        _u32(count)
        + np.asarray(env_ids, dtype="<u4").tobytes()
        + np.arange(100, 100 + count, dtype="<i8").tobytes()
        + bytes(index % 2 for index in range(count))
        + bytes(index % 2 for index in range(count))
        + np.zeros(count, dtype="<f4").tobytes()
        + (bytes(count) if terminated is None else bytes(terminated))
        + bytes(count)
        + (bytes(count) if end_reasons is None else bytes(end_reasons))
        + np.arange(7, 7 + count, dtype="<u4").tobytes()
        + observation_wire
    )


def _assert_payload_backed(array: np.ndarray, payload: bytes) -> None:
    """Require a decoded array to retain the original immutable payload."""

    owner: object = array
    while isinstance(getattr(owner, "base", None), np.ndarray):
        owner = owner.base
    assert getattr(owner, "base", None) is payload
    assert array.flags.owndata is False
    assert array.flags.writeable is False


def _error_frame(
    request_type: int,
    *,
    env_id: int = -1,
    kind: bytes = b"validation",
    message: bytes = b"bad action",
    suffix: bytes = b"",
) -> bytes:
    """Build one complete error frame from the documented scalar layout."""

    payload = (
        bytes((request_type,))
        + env_id.to_bytes(4, "little", signed=True)
        + _wire_string(kind)
        + _wire_string(message)
        + suffix
    )
    return _frame(0xFF, payload)


def test_encode_create_payload_matches_documented_field_order() -> None:
    """Create uses fixed integers followed by two length-prefixed UTF-8 names."""

    encoded = protocol.encode_create_payload(
        64,
        (10, 1_000_010),
        "héuristic-v1",
        "train-v1",
        cursor=17,
    )
    expected = (
        _u32(64)
        + _i64(10)
        + _i64(1_000_010)
        + _i64(17)
        + _wire_string("héuristic-v1".encode())
        + _wire_string(b"train-v1")
    )
    assert encoded == expected


@pytest.mark.parametrize(
    ("arguments", "error_type", "message"),
    [
        ((True, (0, 1), "idle-v1", "train-v1"), TypeError, "batch size"),
        ((1, [0, 1], "idle-v1", "train-v1"), TypeError, "two-integer tuple"),
        ((1, (0, 1 << 63), "idle-v1", "train-v1"), ValueError, "seed-range end"),
        ((1, (0, 1), 7, "train-v1"), TypeError, "opponent is int"),
        ((1, (0, 1), "idle-v1", "bad\ud800"), ValueError, "not valid UTF-8"),
        (
            (9_483, (0, 1), "idle-v1", "train-v2"),
            ValueError,
            "batch size 9483 does not fit one step response, maximum is 9482",
        ),
    ],
)
def test_encode_create_payload_rejects_unrepresentable_values(
    arguments: tuple[object, object, object, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Create inputs fail clearly before struct packing can coerce them."""

    with pytest.raises(error_type, match=message):
        protocol.encode_create_payload(*arguments)  # type: ignore[arg-type]


def test_encode_create_payload_accepts_the_largest_batch_one_frame_can_answer() -> None:
    """9,482 is the most environments one env-v3 step response can carry.

    A step response is 4 + N x 7,077 bytes and a frame payload is at most
    64 MiB, so N = 9,482 fits and 9,483 does not.
    """

    assert protocol.MAX_BATCH_SIZE == 9_482
    assert 4 + 9_482 * 7_077 <= protocol.MAX_PAYLOAD_SIZE
    assert 4 + 9_483 * 7_077 > protocol.MAX_PAYLOAD_SIZE
    encoded = protocol.encode_create_payload(9_482, (0, 1), "idle-v1", "train-v2")
    assert encoded[:4] == _u32(9_482)


def test_encode_create_payload_counts_utf8_bytes_for_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multibyte name is limited by wire bytes rather than Python characters."""

    monkeypatch.setattr(protocol,"MAX_OPPONENT_SIZE", 1)
    with pytest.raises(ValueError, match="2 UTF-8 bytes, maximum is 1"):
        protocol.encode_create_payload(1, (0, 1), "é", "train-v1")


def test_encode_reset_payload_sorts_ids_and_overrides() -> None:
    """Equivalent reset inputs produce identical ascending wire order."""

    encoded = protocol.encode_reset_payload(
        [10, 2],
        [
            protocol.ResetOverride(env_id=10, seed=100, seat=1),
            protocol.ResetOverride(env_id=2, seed=-5, seat=0),
        ],
    )
    expected = (
        _u32(2)
        + _u32(2)
        + _u32(10)
        + _u32(2)
        + _u32(2)
        + _i64(-5)
        + b"\x00"
        + _u32(10)
        + _i64(100)
        + b"\x01"
    )
    assert encoded == expected


def test_encode_reset_payload_accepts_numpy_environment_ids() -> None:
    """The columnar caller may provide IDs without first making a Python list."""

    encoded = protocol.encode_reset_payload(np.asarray([10, 2], dtype="<u4"))

    assert encoded == _u32(2) + _u32(2) + _u32(10) + _u32(0)


@pytest.mark.parametrize(
    ("ids", "overrides", "error_type", "message"),
    [
        ([1, 1], (), ValueError, "repeats environment ID 1"),
        ([True], (), TypeError, "environment ID 0"),
        ([1 << 32], (), ValueError, "4294967296"),
        (
            [1],
            (
                protocol.ResetOverride(1, 10, 0),
                protocol.ResetOverride(1, 20, 1),
            ),
            ValueError,
            "overrides repeat environment ID 1",
        ),
        ([1], (object(),), TypeError, "want ResetOverride"),
        (
            [1],
            (protocol.ResetOverride(1, 10, True),),
            TypeError,
            "seat is True",
        ),
        (
            [1],
            (protocol.ResetOverride(1, 10, -1),),
            ValueError,
            "want 0 through 255",
        ),
        (
            [1],
            (protocol.ResetOverride(1, 10, 256),),
            ValueError,
            "want 0 through 255",
        ),
    ],
)
def test_encode_reset_payload_rejects_invalid_values(
    ids: object,
    overrides: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reset rejects ambiguous IDs and values that cannot fit their fields."""

    with pytest.raises(error_type, match=message):
        protocol.encode_reset_payload(  # type: ignore[arg-type]
            ids,
            overrides,
        )


def test_encode_reset_payload_leaves_representable_seat_validation_to_go() -> None:
    """Seat 2 fits u8 and reaches Go's shared semantic seat validator."""

    encoded = protocol.encode_reset_payload(
        [1], [protocol.ResetOverride(1, 10, 2)]
    )
    assert encoded[-1] == 2


def test_encode_reset_payload_leaves_override_membership_validation_to_go() -> None:
    """An unrequested override reaches Go so both protocols return one EnvError."""

    encoded = protocol.encode_reset_payload(
        [1], [protocol.ResetOverride(2, 10, 0)]
    )

    # The final override row proves that Python preserved the representable
    # request instead of replacing Go's structured validation with ValueError.
    assert encoded[-13:] == _u32(2) + _i64(10) + b"\x00"


def test_encode_step_payload_sorts_ids_and_packs_signed_targets() -> None:
    """Packed step rows follow numeric ID order and use 0xff for no launch."""

    row_two = [-1] * BODIES
    row_two[5] = 0
    row_ten = [-1] * BODIES
    row_ten[3] = 7
    # The last body slot, a comet slot, is a valid target.
    row_ten[40] = 43
    encoded = protocol.encode_step_payload({10: row_ten, 2: row_two})
    expected = (
        _u32(2)
        + _u32(2)
        + _u32(10)
        + bytes(0xFF if target == -1 else target for target in row_two)
        + bytes(0xFF if target == -1 else target for target in row_ten)
    )
    assert encoded == expected


def test_encode_step_payload_fast_array_path_sorts_rows_with_ids() -> None:
    """Integer NumPy input avoids Python target checks without misaligning rows."""

    row_ten = np.full(BODIES, -1, dtype=np.int64)
    row_ten[3] = 7
    row_ten[40] = 43
    row_two = np.full(BODIES, -1, dtype=np.int64)
    row_two[5] = 0
    targets = np.stack((row_ten, row_two))

    encoded = protocol.encode_step_payload(targets, env_ids=[10, 2])
    expected = protocol.encode_step_payload(
        {10: row_ten.tolist(), 2: row_two.tolist()}
    )

    assert encoded == expected


@pytest.mark.parametrize(
    ("targets", "env_ids", "error_type", "message"),
    [
        (
            np.zeros((1, BODIES), dtype=np.bool_),
            [0],
            TypeError,
            "dtype bool, want integers",
        ),
        (
            np.zeros((1, BODIES - 1), dtype=np.int8),
            [0],
            ValueError,
            r"want \(1, 44\)",
        ),
        (
            np.zeros((2, BODIES), dtype=np.int8),
            [0],
            ValueError,
            r"want \(1, 44\)",
        ),
        (
            np.zeros((2, BODIES), dtype=np.int8),
            [0, 0],
            ValueError,
            "repeats environment ID 0",
        ),
        (
            np.full((1, BODIES), BODIES, dtype=np.int8),
            [0],
            ValueError,
            "invalid packed target 44",
        ),
    ],
)
def test_encode_step_payload_fast_array_path_rejects_invalid_values(
    targets: np.ndarray,
    env_ids: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """The fast path retains shape, dtype, ID, and target safety checks."""

    with pytest.raises(error_type, match=message):
        protocol.encode_step_payload(
            targets,
            env_ids=env_ids,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("actions", "error_type", "message"),
    [
        ([], TypeError, "must be a mapping"),
        ({True: [-1] * BODIES}, TypeError, "environment ID"),
        ({1 << 32: [-1] * BODIES}, ValueError, "4294967296"),
        ({0: [-1] * (BODIES - 1)}, ValueError, "43 packed targets, want 44"),
        ({0: [True, *([-1] * (BODIES - 1))]}, TypeError, "target is True"),
        ({0: [1.0, *([-1] * (BODIES - 1))]}, TypeError, "want an integer"),
        ({0: [-2, *([-1] * (BODIES - 1))]}, ValueError, "invalid packed target"),
        ({0: [44, *([-1] * (BODIES - 1))]}, ValueError, "invalid packed target 44"),
    ],
)
def test_encode_step_payload_rejects_invalid_values(
    actions: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Invalid rows fail before any malformed action bytes reach Go."""

    with pytest.raises(error_type, match=message):
        protocol.encode_step_payload(actions)  # type: ignore[arg-type]


def test_encode_empty_step_and_close_payloads() -> None:
    """Zero-row step retains its count; close carries no payload at all."""

    assert protocol.encode_step_payload({}) == b"\x00\x00\x00\x00"
    assert protocol.encode_close_payload() == b""


def test_complete_request_frames_match_the_frozen_wire_contract() -> None:
    """Every command combines its exact payload with the documented header."""

    # Build expected payloads with the small independent test helpers above.
    # This prevents the assertion from merely framing bytes produced by the
    # same production payload encoder that the test is meant to verify.
    create_payload = (
        _u32(2)
        + _i64(10)
        + _i64(20)
        + _i64(3)
        + _wire_string(b"idle-v1")
        + _wire_string(b"train-v1")
    )
    reset_payload = _u32(1) + _u32(2) + _u32(0)
    step_payload = _u32(1) + _u32(2) + bytes((0xFF,)) * BODIES

    assert protocol.encode_frame(
        protocol.MessageType.CREATE_REQUEST,
        protocol.encode_create_payload(
            2,
            (10, 20),
            "idle-v1",
            "train-v1",
            cursor=3,
        ),
    ) == _frame(0x01, create_payload)
    assert protocol.encode_frame(
        protocol.MessageType.RESET_REQUEST,
        protocol.encode_reset_payload([2]),
    ) == _frame(0x02, reset_payload)
    assert protocol.encode_frame(
        protocol.MessageType.STEP_REQUEST,
        protocol.encode_step_payload({2: [-1] * BODIES}),
    ) == _frame(0x03, step_payload)
    assert protocol.encode_frame(
        protocol.MessageType.CLOSE_REQUEST,
        protocol.encode_close_payload(),
    ) == _frame(0x04)


def test_decode_create_response_validates_and_names_metadata() -> None:
    """Create decoding retains each replay identity and the four dimensions."""

    decoded = protocol.decode_create_response(_create_response_wire())

    assert decoded.engine == "engine-commit"
    assert decoded.schema == "obs-v2"
    assert decoded.generator == "generator-v2"
    assert decoded.opponent == "idle-v1"
    assert decoded.seed_range == (0, 1_000_000)
    assert decoded.schedule == "train-v2"
    assert decoded.cursor == 17
    assert decoded.dimensions == protocol.ObservationDimensions(
        body_slots=44,
        body_features=27,
        player_slots=4,
        player_features=6,
    )


def test_decode_create_response_rejects_incompatible_or_malformed_metadata() -> None:
    """Schema, UTF-8, and complete consumption are mandatory."""

    with pytest.raises(protocol.ProtocolError, match="schema is 'obs-v1'"):
        protocol.decode_create_response(
            _create_response_wire(schema=b"obs-v1")
        )
    with pytest.raises(protocol.ProtocolError, match="not valid UTF-8"):
        protocol.decode_create_response(
            _create_response_wire(schema=b"\xff")
        )
    with pytest.raises(protocol.ProtocolError, match="1 trailing bytes"):
        protocol.decode_create_response(_create_response_wire() + b"x")


@pytest.mark.parametrize(
    ("dimensions", "message"),
    [
        # obs-v1's body sizes: a create from a server still built for them.
        ((16, 16, 4, 3), "reports 16 body slots, want 44"),
        ((44, 26, 4, 6), "reports 26 body features, want 27"),
        ((44, 27, 2, 6), "reports 2 player slots, want 4"),
        ((44, 27, 4, 7), "reports 7 player features, want 6"),
    ],
)
def test_decode_create_response_requires_the_obs_v2_dimensions(
    dimensions: tuple[int, ...],
    message: str,
) -> None:
    """Any size other than the decoder's own stops before an observation is read."""

    with pytest.raises(protocol.ProtocolError, match=message):
        protocol.decode_create_response(
            _create_response_wire(dimensions=dimensions)
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_u32(5) + b"abc", "create engine needs 5 bytes, only 3 remain"),
        (
            _create_response_wire()[:-1],
            "player features needs 4 bytes, only 3 remain",
        ),
    ],
)
def test_decode_create_response_rejects_truncated_fields(
    payload: bytes,
    message: str,
) -> None:
    """A partial string or scalar cannot produce incomplete create metadata."""

    with pytest.raises(protocol.ProtocolError, match=message):
        protocol.decode_create_response(payload)


def test_decode_reset_response_creates_columnar_buffer_views() -> None:
    """Reset fields keep fixed dtypes, shapes, values, and payload ownership."""

    count = 2
    observation_wire, expected_observations = _observation_wire(count)
    payload = (
        _i64(25)
        + _u32(count)
        + np.asarray([2, 10], dtype="<u4").tobytes()
        + np.asarray([-5, 99], dtype="<i8").tobytes()
        + bytes((0, 1))
        + bytes((1, 0))
        + observation_wire
    )

    decoded = protocol.decode_reset_response(
        payload,
        expected_env_ids=[10, 2],
    )

    assert decoded.cursor == 25
    np.testing.assert_array_equal(decoded.env_ids, [2, 10])
    np.testing.assert_array_equal(decoded.seeds, [-5, 99])
    np.testing.assert_array_equal(decoded.seats, [0, 1])
    np.testing.assert_array_equal(decoded.explicit, [True, False])
    assert decoded.env_ids.dtype == np.dtype("<u4")
    assert decoded.seeds.dtype == np.dtype("<i8")
    assert decoded.seats.dtype == np.dtype("u1")
    assert decoded.explicit.dtype == np.dtype("bool")

    for name, expected in expected_observations.items():
        actual = getattr(decoded.observations, name)
        if expected.dtype == np.dtype("u1"):
            expected = expected.view(np.bool_)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        _assert_payload_backed(actual, payload)
    assert decoded.observations.bodies.shape == (2, 44, 27)
    assert decoded.observations.players.shape == (2, 4, 6)
    assert decoded.observations.target_mask.shape == (2, 44, 44)

    # Selecting one environment creates another view; it does not copy that
    # environment's 4,752 body-feature bytes into separate storage.
    body_row = decoded.observations.bodies[1]
    assert np.shares_memory(body_row, decoded.observations.bodies)
    _assert_payload_backed(body_row, payload)


def test_decode_step_response_creates_columnar_buffer_views() -> None:
    """Step result columns remain aligned with their post-turn observations."""

    count = 2
    observation_wire, _ = _observation_wire(count)
    payload = (
        _u32(count)
        + np.asarray([3, 9], dtype="<u4").tobytes()
        + np.asarray([100, 200], dtype="<i8").tobytes()
        + bytes((1, 0))
        + bytes((0, 1))
        + np.asarray([-1.0, 1.0], dtype="<f4").tobytes()
        + bytes((1, 0))
        + bytes((0, 1))
        # Environment 3 ended because its learner was eliminated.
        + bytes((2, 0))
        + np.asarray([7, 11], dtype="<u4").tobytes()
        + observation_wire
    )

    decoded = protocol.decode_step_response(
        payload,
        expected_env_ids=[9, 3],
    )

    np.testing.assert_array_equal(decoded.env_ids, [3, 9])
    np.testing.assert_array_equal(decoded.seeds, [100, 200])
    np.testing.assert_array_equal(decoded.seats, [1, 0])
    np.testing.assert_array_equal(decoded.explicit, [False, True])
    np.testing.assert_array_equal(decoded.rewards, [-1.0, 1.0])
    np.testing.assert_array_equal(decoded.terminated, [True, False])
    np.testing.assert_array_equal(decoded.truncated, [False, True])
    np.testing.assert_array_equal(
        decoded.end_reasons, [protocol.END_LEARNER_ELIMINATED, protocol.END_ONGOING]
    )
    np.testing.assert_array_equal(decoded.turns, [7, 11])
    assert decoded.rewards.dtype == np.dtype("<f4")
    assert decoded.end_reasons.dtype == np.dtype("u1")
    assert decoded.turns.dtype == np.dtype("<u4")

    for array in (
        decoded.env_ids,
        decoded.seeds,
        decoded.seats,
        decoded.explicit,
        decoded.rewards,
        decoded.terminated,
        decoded.truncated,
        decoded.end_reasons,
        decoded.turns,
        decoded.observations.bodies,
    ):
        _assert_payload_backed(array, payload)


def test_decode_step_response_requires_increasing_expected_environment_ids() -> None:
    """Returned rows must be sorted and match the IDs sent in the request."""

    with pytest.raises(
        protocol.ProtocolError,
        match="environment IDs are not strictly increasing",
    ):
        protocol.decode_step_response(
            _step_response_wire((9, 3)),
            expected_env_ids=[3, 9],
        )

    with pytest.raises(
        protocol.ProtocolError,
        match="3 follows 3",
    ):
        protocol.decode_step_response(
            _step_response_wire((3, 3)),
            expected_env_ids=[3, 9],
        )

    with pytest.raises(
        protocol.ProtocolError,
        match="environment ID at row 1 is 9, want 8",
    ):
        protocol.decode_step_response(
            _step_response_wire((3, 9)),
            expected_env_ids=[3, 8],
        )

    with pytest.raises(
        protocol.ProtocolError,
        match="returned 2 environments, request named 1",
    ):
        protocol.decode_step_response(
            _step_response_wire((3, 9)),
            expected_env_ids=[3],
        )


def test_decode_step_response_vectorizes_numpy_expected_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NumPy ID batch never enters the slower per-Python-integer validator."""

    def fail_if_called(*_args: object, **_kwargs: object) -> list[int]:
        raise AssertionError("NumPy expected IDs used the sequence validator")

    monkeypatch.setattr(protocol,"_checked_sorted_env_ids", fail_if_called)
    decoded = protocol.decode_step_response(
        _step_response_wire((3, 9)),
        expected_env_ids=np.asarray([9, 3], dtype=np.int64),
    )

    np.testing.assert_array_equal(decoded.env_ids, [3, 9])


def test_decode_step_response_treats_invalid_expected_ids_as_caller_errors() -> None:
    """Local expected-ID mistakes remain TypeError rather than wire failures."""

    with pytest.raises(TypeError, match="dtype bool, want integers"):
        protocol.decode_step_response(
            _step_response_wire(),
            expected_env_ids=np.asarray([True], dtype=np.bool_),
        )


def test_decode_step_response_rejects_invalid_seats_and_rewards() -> None:
    """Invalid learner seats and non-finite rewards stop before model use."""

    seat_payload = bytearray(_step_response_wire())
    seat_offset = (
        protocol.STEP_RESPONSE_PREFIX_BYTES
        + protocol.UINT32_BYTES
        + protocol.INT64_BYTES
    )
    # Seat 3 is the highest a four-player map can assign.
    seat_payload[seat_offset] = 3
    decoded = protocol.decode_step_response(
        bytes(seat_payload),
        expected_env_ids=[3],
    )
    np.testing.assert_array_equal(decoded.seats, [3])

    seat_payload[seat_offset] = 4
    with pytest.raises(
        protocol.ProtocolError,
        match="environment 3 has seat 4, want 0 through 3",
    ):
        protocol.decode_step_response(
            bytes(seat_payload),
            expected_env_ids=[3],
        )

    reward_payload = bytearray(_step_response_wire())
    reward_offset = (
        protocol.STEP_RESPONSE_PREFIX_BYTES
        + protocol.RESET_IDENTITY_BYTES_PER_ENVIRONMENT
    )
    reward_payload[
        reward_offset : reward_offset + protocol.FLOAT32_BYTES
    ] = np.asarray([float("nan")], dtype="<f4").tobytes()
    with pytest.raises(
        protocol.ProtocolError,
        match="step rewards contains non-finite value",
    ):
        protocol.decode_step_response(
            bytes(reward_payload),
            expected_env_ids=[3],
        )


@pytest.mark.parametrize(
    ("terminated", "end_reasons", "message"),
    [
        ((0, 1, 1), (0, 1, 2), None),
        ((0,), (3,), "environment 3 has end reason 3, want 0 through 2"),
        ((1,), (0,), "environment 3 has end reason 0 but terminated=True"),
        ((0,), (1,), "environment 3 has end reason 1 but terminated=False"),
        ((0,), (2,), "environment 3 has end reason 2 but terminated=False"),
    ],
)
def test_decode_step_response_requires_end_reasons_to_agree_with_terminated(
    terminated: tuple[int, ...],
    end_reasons: tuple[int, ...],
    message: str | None,
) -> None:
    """An end reason is known and nonzero exactly where the episode terminated."""

    env_ids = (3, 5, 9)[: len(terminated)]
    payload = _step_response_wire(
        env_ids, terminated=terminated, end_reasons=end_reasons
    )
    if message is None:
        decoded = protocol.decode_step_response(payload, expected_env_ids=env_ids)
        np.testing.assert_array_equal(decoded.end_reasons, end_reasons)
        return
    with pytest.raises(protocol.ProtocolError, match=message):
        protocol.decode_step_response(payload, expected_env_ids=env_ids)


@pytest.mark.parametrize(
    ("offset", "bad_value", "message"),
    [
        (
            protocol.STEP_RESPONSE_PREFIX_BYTES
            + protocol.RESET_IDENTITY_BYTES_PER_ENVIRONMENT
            + protocol.FLOAT32_BYTES,
            2,
            "step terminated flags contains Boolean byte 2",
        ),
        (
            protocol.STEP_RESPONSE_PREFIX_BYTES
            + protocol.STEP_RESULT_BYTES_PER_ENVIRONMENT
            + protocol.BODY_FEATURE_BYTES_PER_ENVIRONMENT
            + protocol.PLAYER_FEATURE_BYTES_PER_ENVIRONMENT,
            7,
            "observation body mask contains Boolean byte 7",
        ),
    ],
)
def test_decode_step_response_rejects_non_boolean_wire_bytes(
    offset: int,
    bad_value: int,
    message: str,
) -> None:
    """Bytes other than zero or one cannot be disguised as NumPy true values."""

    payload = bytearray(_step_response_wire())
    payload[offset] = bad_value

    with pytest.raises(protocol.ProtocolError, match=message):
        protocol.decode_step_response(
            bytes(payload),
            expected_env_ids=[3],
        )


@pytest.mark.parametrize(
    ("observation_offset", "bad_value", "message"),
    [
        (0, float("nan"), "observation bodies contains non-finite value"),
        (
            protocol.BODY_FEATURE_BYTES_PER_ENVIRONMENT,
            float("inf"),
            "observation players contains non-finite value",
        ),
    ],
)
def test_decode_step_response_rejects_non_finite_observations(
    observation_offset: int,
    bad_value: float,
    message: str,
) -> None:
    """NaN and infinity are rejected before observations can reach the model."""

    payload = bytearray(_step_response_wire())
    value_offset = (
        protocol.STEP_RESPONSE_PREFIX_BYTES
        + protocol.STEP_RESULT_BYTES_PER_ENVIRONMENT
        + observation_offset
    )
    payload[value_offset : value_offset + protocol.FLOAT32_BYTES] = np.asarray(
        [bad_value], dtype="<f4"
    ).tobytes()

    with pytest.raises(protocol.ProtocolError, match=message):
        protocol.decode_step_response(
            bytes(payload),
            expected_env_ids=[3],
        )


@pytest.mark.parametrize(
    ("decoder", "payload", "message"),
    [
        (
            protocol.decode_reset_response,
            b"",
            "reset response has 0 bytes, needs at least 12",
        ),
        (
            protocol.decode_reset_response,
            _i64(0) + _u32(1),
            "count requires exactly 7078",
        ),
        (
            protocol.decode_step_response,
            b"",
            "step response has 0 bytes, needs at least 4",
        ),
        (
            protocol.decode_step_response,
            _u32(1),
            "count requires exactly 7081",
        ),
    ],
)
def test_fixed_column_decoders_reject_truncated_payloads_before_arrays(
    decoder: object,
    payload: bytes,
    message: str,
) -> None:
    """A declared row count cannot expose partial identity or observation data."""

    with pytest.raises(protocol.ProtocolError, match=message):
        decoder(payload, expected_env_ids=[])  # type: ignore[operator]


def test_fixed_column_decoders_reject_trailing_and_mutable_payloads() -> None:
    """Exact sizes and immutable bytes protect all later zero-copy views."""

    with pytest.raises(protocol.ProtocolError, match="exactly 12"):
        protocol.decode_reset_response(
            _i64(0) + _u32(0) + b"x",
            expected_env_ids=[],
        )
    with pytest.raises(protocol.ProtocolError, match="exactly 4"):
        protocol.decode_step_response(
            _u32(0) + b"x",
            expected_env_ids=[],
        )
    with pytest.raises(TypeError, match="payload is bytearray, want bytes"):
        protocol.decode_step_response(
            bytearray(_u32(0)),  # type: ignore[arg-type]
            expected_env_ids=[],
        )


def test_fixed_column_decoder_rejects_oversized_payload_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Payload size remains guarded even when decoding bypasses frame reading."""

    # Lower the cap instead of allocating more than 64 MiB for this focused
    # unit test. A valid empty step payload is four bytes and now exceeds it.
    monkeypatch.setattr(protocol,"MAX_PAYLOAD_SIZE", 3)
    with pytest.raises(protocol.ProtocolError, match="maximum is 3"):
        protocol.decode_step_response(
            _u32(0),
            expected_env_ids=[],
        )


def test_fixed_column_decoders_support_empty_batches() -> None:
    """A valid zero-row payload produces every documented empty shape."""

    reset = protocol.decode_reset_response(
        _i64(8) + _u32(0),
        expected_env_ids=[],
    )
    step = protocol.decode_step_response(
        _u32(0),
        expected_env_ids=[],
    )
    assert reset.cursor == 8
    assert reset.env_ids.shape == (0,)
    assert reset.observations.bodies.shape == (0, 44, 27)
    assert step.rewards.shape == (0,)
    assert step.end_reasons.shape == (0,)
    assert step.observations.target_mask.shape == (0, 44, 44)


@pytest.mark.parametrize("payload", [b"", b"\x00", b"\x02", b"\x01\x00"])
def test_decode_close_response_rejects_invalid_acknowledgement(
    payload: bytes,
) -> None:
    """Missing, false, malformed, or overlong acknowledgements are rejected."""

    with pytest.raises(protocol.ProtocolError):
        protocol.decode_close_response(payload)


def test_decode_close_response_accepts_true_acknowledgement() -> None:
    """The complete one-byte value 1 confirms that Go closed its batch."""

    assert protocol.decode_close_response(b"\x01") is True


def test_fixed_wire_sizes_match_the_frozen_go_contract() -> None:
    """Python layout arithmetic stays pinned to Go's four per-row byte sizes.

    These are the env-v3 sizes documented in mlenv/README.md: a step request
    is 4 + N x 48, a reset response 12 + N x 7,066, a step response
    4 + N x 7,077.
    """

    assert protocol.OBSERVATION_BYTES_PER_ENVIRONMENT == 7_052
    assert protocol.RESET_IDENTITY_BYTES_PER_ENVIRONMENT == 14
    assert protocol.STEP_RESULT_BYTES_PER_ENVIRONMENT == 25
    assert protocol.STEP_REQUEST_BYTES_PER_ENVIRONMENT == 48


def test_encode_frame_matches_documented_bytes() -> None:
    """The encoder writes a zeroed 12-byte header followed by the payload."""

    payload = b"abc"
    encoded = protocol.encode_frame(protocol.MessageType.STEP_REQUEST, payload)
    assert encoded == b"OWE3\x03\x00\x00\x00\x03\x00\x00\x00abc"


def test_encode_frame_rejects_invalid_type_and_oversized_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad message IDs and payload lengths fail before bytes reach the pipe."""

    with pytest.raises(TypeError, match="must not be Boolean"):
        protocol.encode_frame(True, b"")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown env-v3 message type"):
        protocol.encode_frame(0x55, b"")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="contiguous bytes-like"):
        protocol.encode_frame(
            protocol.MessageType.CLOSE_REQUEST,
            5,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="contiguous bytes-like"):
        protocol.encode_frame(
            protocol.MessageType.CLOSE_REQUEST,
            memoryview(b"abcd")[::2],
        )

    # Lower the limit for this test instead of allocating a 64 MiB object.
    monkeypatch.setattr(protocol,"MAX_PAYLOAD_SIZE", 2)
    with pytest.raises(protocol.ProtocolError, match="maximum is 2"):
        protocol.encode_frame(protocol.MessageType.RESET_REQUEST, b"abc")


def test_write_frame_retries_partial_writes() -> None:
    """A stream accepting two bytes at a time still receives one whole frame."""

    writer = ChunkedWriter()
    protocol.write_frame(
        writer,  # type: ignore[arg-type]
        protocol.MessageType.CLOSE_REQUEST,
        b"",
    )
    assert bytes(writer.data) == _frame(0x04)
    assert writer.flushed is True


def test_write_frame_rejects_a_stream_that_makes_no_progress() -> None:
    """A zero-byte write cannot loop forever while the child is unresponsive."""

    writer = ChunkedWriter(chunk_size=0)
    with pytest.raises(protocol.ProtocolError, match="accepted no frame bytes"):
        protocol.write_frame(
            writer,  # type: ignore[arg-type]
            protocol.MessageType.CLOSE_REQUEST,
            b"",
        )


def test_read_frame_collects_partial_reads() -> None:
    """Header and payload decoding does not assume one read returns everything."""

    frame = protocol.read_frame(  # type: ignore[arg-type]
        ChunkedReader(_frame(0x83, b"result"), chunk_size=2)
    )
    assert frame == protocol.Frame(
        message_type=protocol.MessageType.STEP_RESPONSE,
        payload=b"result",
    )


@pytest.mark.parametrize(
    ("wire", "message"),
    [
        (b"", "truncated env-v3 frame header"),
        (b"OWE3", "received 4 of 12 bytes"),
        (_frame(0x81, magic=b"NOPE"), "frame magic"),
        (_frame(0x81, flags=1), "frame flags are 1"),
        (_frame(0x81, reserved=1), "reserved field is 1"),
        (_frame(0x55), "unknown message type 0x55"),
        (
            _frame(0x81, declared_size=3),
            "truncated env-v3 frame payload: received 0 of 3 bytes",
        ),
        (
            _frame(0x81, b"a", declared_size=3),
            "truncated env-v3 frame payload: received 1 of 3 bytes",
        ),
    ],
)
def test_read_frame_rejects_malformed_or_truncated_input(
    wire: bytes,
    message: str,
) -> None:
    """Every header and length invariant fails with an actionable error."""

    with pytest.raises(protocol.ProtocolError, match=message):
        protocol.read_frame(BytesIO(wire))


def test_read_frame_rejects_oversized_declared_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declared size is checked before Python attempts to read its payload."""

    monkeypatch.setattr(protocol,"MAX_PAYLOAD_SIZE", 2)
    with pytest.raises(protocol.ProtocolError, match="maximum is 2"):
        protocol.read_frame(BytesIO(_frame(0x81, declared_size=3)))


def test_read_response_frame_requires_the_matching_response_type() -> None:
    """A response for another request cannot shift the conversation silently."""

    matched = protocol.read_response_frame(
        BytesIO(_frame(0x82, b"reset")),
        protocol.MessageType.RESET_REQUEST,
    )
    assert matched.payload == b"reset"

    with pytest.raises(protocol.ProtocolError, match="want RESET_RESPONSE"):
        protocol.read_response_frame(
            BytesIO(_frame(0x83, b"step")),
            protocol.MessageType.RESET_REQUEST,
        )


def test_read_response_frame_decodes_structured_error() -> None:
    """A Go rejection retains its kind, environment, and human-readable text."""

    with pytest.raises(protocol.RemoteProtocolError) as caught:
        protocol.read_response_frame(
            BytesIO(_error_frame(0x03, env_id=7)),
            protocol.MessageType.STEP_REQUEST,
        )
    error = caught.value
    assert error.request_type is protocol.MessageType.STEP_REQUEST
    assert error.kind == "validation"
    assert error.remote_message == "bad action"
    assert error.env_id == 7


@pytest.mark.parametrize(
    ("wire", "message"),
    [
        (_error_frame(0x02), "answers request type 0x2"),
        (_error_frame(0x03, env_id=-2), "environment ID is -2"),
        (_error_frame(0x03, kind=b"\xff"), "error kind is not valid UTF-8"),
        (_error_frame(0x03, suffix=b"x"), "has 1 trailing bytes"),
    ],
)
def test_read_response_frame_rejects_malformed_error_payload(
    wire: bytes,
    message: str,
) -> None:
    """Malformed error details cannot masquerade as a valid Go rejection."""

    with pytest.raises(protocol.ProtocolError, match=message):
        protocol.read_response_frame(
            BytesIO(wire), protocol.MessageType.STEP_REQUEST
        )


def test_read_response_frame_requires_a_request_message_type() -> None:
    """Callers cannot ask for the response paired with another response."""

    with pytest.raises(ValueError, match="is not an env-v3 request type"):
        protocol.read_response_frame(
            BytesIO(b""), protocol.MessageType.STEP_RESPONSE
        )
