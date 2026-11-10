"""Focused client contract tests using in-memory and real cmd/env processes."""

from __future__ import annotations

from io import BytesIO
from itertools import count
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import orrery.env_client as env_client
import orrery.protocol as protocol
from orrery.profiling import PhaseProfiler


class RecordingBinaryStdin:
    """Retain every frame byte even after client cleanup closes the stream."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, value: bytes | bytearray | memoryview) -> int:
        """Accept one whole or partial frame write into the test buffer."""

        self.data.extend(value)
        return len(value)

    def flush(self) -> None:
        """Match a pipe whose recorded bytes are immediately available."""

        pass

    def close(self) -> None:
        """Record closure without discarding bytes needed by assertions."""

        self.closed = True


class InterruptingBinaryStdout:
    """Imitate a user interruption while Python awaits a binary response."""

    def read(self, _size: int = -1) -> bytes:
        """Interrupt before any response byte establishes a frame boundary."""

        raise KeyboardInterrupt


class FakeProcess:
    """Provide only the subprocess behavior exercised by EnvClient.

    Its stdout replays caller-supplied response frames in order, so a test can
    script a complete Go conversation without starting a real process.
    """

    def __init__(
        self,
        responses: bytes = b"",
        *,
        stdout: BytesIO | InterruptingBinaryStdout | None = None,
    ) -> None:
        """Assemble fake byte streams and observable process-lifecycle flags."""

        self.stdin = RecordingBinaryStdin()
        self.stdout = BytesIO(responses) if stdout is None else stdout
        self.stderr = BytesIO()
        self.return_code: int | None = None
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        """Return None while running or the saved code after termination."""

        return self.return_code

    def kill(self) -> None:
        """Record the forced stop used when protocol alignment is uncertain."""

        self.killed = True
        self.return_code = -9

    def wait(self, timeout: int | None = None) -> int:
        """Record that cleanup reaped the process and return its exit code."""

        del timeout
        self.waited = True
        if self.return_code is None:
            self.return_code = 0
        return self.return_code

    def written_request_types(self) -> list[protocol.MessageType]:
        """Return the type of every complete frame the client wrote, in order."""

        stream = BytesIO(bytes(self.stdin.data))
        types: list[protocol.MessageType] = []
        while stream.tell() < len(self.stdin.data):
            types.append(protocol.read_frame(stream).message_type)
        return types


class BinaryShutdownProcess(FakeProcess):
    """Return one binary close response followed by selected shutdown output."""

    def __init__(
        self,
        *,
        return_code: int,
        stderr: bytes = b"",
        extra_stdout: bytes = b"",
    ) -> None:
        """Prepare byte streams and the status produced when close waits."""

        super().__init__(_close_response_frame() + extra_stdout)
        self.stderr = BytesIO(stderr)
        self._wait_return_code = return_code

    def wait(self, timeout: int | None = None) -> int:
        """Finish with the status selected by this shutdown test."""

        del timeout
        self.waited = True
        self.return_code = self._wait_return_code
        return self.return_code


def _wire_string(value: str) -> bytes:
    """Encode one length-prefixed UTF-8 string as Go writes it."""

    encoded = value.encode()
    return len(encoded).to_bytes(4, "little") + encoded


def _create_response_frame(
    schedule: str,
    cursor: int,
    *,
    generator: str = "generator-v2",
) -> bytes:
    """Build a complete create response with the given schedule and cursor."""

    payload = (
        _wire_string("engine-commit")
        + _wire_string(env_client.OBSERVATION_SCHEMA)
        + _wire_string(generator)
        + _wire_string("idle-v1")
        + (0).to_bytes(8, "little", signed=True)
        + (10).to_bytes(8, "little", signed=True)
        + _wire_string(schedule)
        + cursor.to_bytes(8, "little", signed=True)
        + np.asarray([44, 27, 4, 6], dtype="<u4").tobytes()
    )
    return protocol.encode_frame(protocol.MessageType.CREATE_RESPONSE, payload)


def _observation_wire(count: int) -> bytes:
    """Build the seven observation columns for ``count`` all-zero games."""

    return b"".join(
        np.zeros(shape, dtype=dtype).tobytes()
        for shape, dtype in (
            ((count, 44, 27), "<f4"),
            ((count, 4, 6), "<f4"),
            ((count, 44), "u1"),
            ((count, 44), "u1"),
            ((count, 4), "u1"),
            ((count, 44, 44), "u1"),
            ((count, 44), "<i4"),
        )
    )


def _identity_wire(env_ids: tuple[int, ...]) -> bytes:
    """Build the env ID, seed, seat, and explicit columns for each game."""

    count = len(env_ids)
    return (
        np.asarray(env_ids, dtype="<u4").tobytes()
        + np.arange(100, 100 + count, dtype="<i8").tobytes()
        + bytes(count)
        + bytes(count)
    )


def _reset_response_frame(cursor: int, env_ids: tuple[int, ...]) -> bytes:
    """Build a complete reset response for the named environments."""

    payload = (
        cursor.to_bytes(8, "little", signed=True)
        + len(env_ids).to_bytes(4, "little")
        + _identity_wire(env_ids)
        + _observation_wire(len(env_ids))
    )
    return protocol.encode_frame(protocol.MessageType.RESET_RESPONSE, payload)


def _step_response_frame(env_ids: tuple[int, ...]) -> bytes:
    """Build a complete non-terminal step response at turn 1."""

    count = len(env_ids)
    payload = (
        count.to_bytes(4, "little")
        + _identity_wire(env_ids)
        + np.zeros(count, dtype="<f4").tobytes()
        + bytes(count)
        + bytes(count)
        + bytes(count)
        + np.ones(count, dtype="<u4").tobytes()
        + _observation_wire(count)
    )
    return protocol.encode_frame(protocol.MessageType.STEP_RESPONSE, payload)


def _error_response_frame(request_type: protocol.MessageType) -> bytes:
    """Build a structured Go rejection of one request."""

    payload = (
        bytes((int(request_type),))
        + (0).to_bytes(4, "little", signed=True)
        + _wire_string("not-reset")
        + _wire_string("environment must be reset before stepping")
    )
    return protocol.encode_frame(protocol.MessageType.ERROR_RESPONSE, payload)


def _close_response_frame() -> bytes:
    """Build the one-byte close acknowledgement."""

    return protocol.encode_frame(protocol.MessageType.CLOSE_RESPONSE, b"\x01")


def test_constructor_starts_go_with_binary_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the player count on the command line and open byte-exact pipes.

    cmd/env chooses its maps from --players for the whole process. The
    caller's command sequence must not be modified in place.
    """

    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(
        command: list[str],
        **options: object,
    ) -> FakeProcess:
        calls.append((command, options))
        return FakeProcess()

    monkeypatch.setattr(env_client.subprocess, "Popen", fake_popen)
    original_command = ["fake-env", "serve"]

    for players in (2, 4):
        client = env_client.EnvClient(original_command, players=players)
        client.close()
        assert client.players == players

    assert original_command == ["fake-env", "serve"]
    assert [command for command, _ in calls] == [
        ["fake-env", "serve", "--players", "2"],
        ["fake-env", "serve", "--players", "4"],
    ]
    _, options = calls[0]
    assert options.get("text", False) is False
    assert options["bufsize"] == 0


@pytest.mark.parametrize("players", [0, 1, 3, 5, True])
def test_constructor_rejects_unsupported_player_count_before_starting_process(
    monkeypatch: pytest.MonkeyPatch,
    players: object,
) -> None:
    """Only two- and four-player maps exist, so any other count fails first."""

    monkeypatch.setattr(
        env_client.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen must not be called"),
    )

    with pytest.raises(ValueError, match="players must be 2 or 4"):
        env_client.EnvClient(("fake-env",), players=players)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("return_code", "stderr", "extra_stdout", "expected_message"),
    [
        (2, b"panic: boom\n", b"", "environment exited with status 2: panic: boom"),
        (
            0,
            b"warning from Go\n",
            b"",
            "environment wrote unexpected stderr: 'warning from Go\\n'",
        ),
        (
            0,
            b"",
            b"stray response\n",
            "environment wrote extra stdout after close: 'stray response\\n'",
        ),
    ],
)
def test_close_decodes_binary_process_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    return_code: int,
    stderr: bytes,
    extra_stdout: bytes,
    expected_message: str,
) -> None:
    """Shutdown failures should display Go text without a byte-string prefix."""

    process = BinaryShutdownProcess(
        return_code=return_code,
        stderr=stderr,
        extra_stdout=extra_stdout,
    )
    monkeypatch.setattr(
        env_client.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    client = env_client.EnvClient(("fake-env",), players=2)
    # The test starts at shutdown, so mark create as completed without needing
    # to place an unrelated create frame before the close response.
    client._created = True

    with pytest.raises(env_client.ProtocolFailure) as caught:
        client.close()

    message = str(caught.value)
    assert expected_message in message
    assert "b'" not in message


def _decoded_binary_observations(count: int) -> protocol.DecodedObservationBatch:
    """Build validated-style NumPy columns for the public-boundary adapters."""

    return protocol.DecodedObservationBatch(
        bodies=np.arange(count * 44 * 27, dtype=np.float32).reshape(
            count, 44, 27
        ),
        players=np.arange(count * 4 * 6, dtype=np.float32).reshape(count, 4, 6),
        body_mask=np.ones((count, 44), dtype=np.bool_),
        owned_mask=np.zeros((count, 44), dtype=np.bool_),
        player_mask=np.ones((count, 4), dtype=np.bool_),
        target_mask=np.zeros((count, 44, 44), dtype=np.bool_),
        slot_body_id=np.arange(count * 44, dtype=np.int32).reshape(count, 44),
    )


def test_binary_reset_adapter_creates_public_objects_from_row_views() -> None:
    """Reset conversion creates Python scalars while observation arrays stay shared."""

    observations = _decoded_binary_observations(2)
    response = protocol.DecodedResetResponse(
        cursor=12,
        env_ids=np.asarray([2, 10], dtype="<u4"),
        seeds=np.asarray([100, 200], dtype="<i8"),
        seats=np.asarray([0, 1], dtype="u1"),
        explicit=np.asarray([False, True], dtype=np.bool_),
        observations=observations,
    )

    episodes = env_client._episodes_from_binary(response)

    assert len(episodes) == 2
    assert episodes[1].env_id == 10
    assert episodes[1].seed == 200
    assert episodes[1].seat == 1
    assert episodes[1].explicit is True
    assert type(episodes[1].env_id) is int
    assert type(episodes[1].explicit) is bool
    for field in (
        "bodies",
        "players",
        "body_mask",
        "owned_mask",
        "player_mask",
        "target_mask",
        "slot_body_id",
    ):
        assert np.shares_memory(
            getattr(episodes[1].observation, field),
            getattr(observations, field),
        )


def test_binary_step_adapter_converts_only_public_result_scalars() -> None:
    """Step conversion adds result fields while retaining wire-backed row arrays."""

    observations = _decoded_binary_observations(2)
    response = protocol.DecodedStepResponse(
        env_ids=np.asarray([3, 9], dtype="<u4"),
        seeds=np.asarray([300, 900], dtype="<i8"),
        seats=np.asarray([1, 0], dtype="u1"),
        explicit=np.asarray([False, False], dtype=np.bool_),
        rewards=np.asarray([-1.0, 1.0], dtype="<f4"),
        terminated=np.asarray([True, False], dtype=np.bool_),
        truncated=np.asarray([False, True], dtype=np.bool_),
        end_reasons=np.asarray([2, 0], dtype="u1"),
        turns=np.asarray([7, 11], dtype="<u4"),
        observations=observations,
    )

    results = env_client._step_results_from_binary(response)

    assert len(results) == 2
    assert results[0].env_id == 3
    assert results[0].reward == -1.0
    assert results[0].terminated is True
    assert results[0].truncated is False
    assert results[0].end_reason == env_client.END_LEARNER_ELIMINATED
    assert results[1].end_reason == env_client.END_ONGOING
    assert results[0].turn == 7
    assert type(results[0].reward) is float
    assert type(results[0].end_reason) is int
    assert type(results[0].turn) is int
    assert np.shares_memory(results[0].observation.bodies, observations.bodies)
    assert np.shares_memory(
        results[0].observation.target_mask,
        observations.target_mask,
    )


def test_create_defaults_to_train_v2_and_retains_replay_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default request and retained metadata must name train-v2 exactly."""

    process = FakeProcess(
        _create_response_frame("train-v2", 7) + _close_response_frame()
    )
    monkeypatch.setattr(env_client.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with env_client.EnvClient(("fake-env",), players=2) as client:
        metadata = client.create(2, (0, 10), "idle-v1", cursor=7)
        assert metadata["v"] == "env-v3"
        assert metadata["schema"] == "obs-v2"
        assert metadata["generator"] == "generator-v2"
        assert metadata["dimensions"] == {
            "body_slots": 44,
            "body_features": 27,
            "player_slots": 4,
            "player_features": 6,
        }
        assert "caps" not in metadata
        assert metadata["schedule"] == env_client.SCHEDULE_TRAIN_V2
        assert metadata["cursor"] == 7
        assert client.schedule == env_client.SCHEDULE_TRAIN_V2
        assert client.cursor == 7

    # The request the client wrote must carry the same schedule and cursor.
    request = protocol.read_frame(BytesIO(bytes(process.stdin.data)))
    assert request.payload == protocol.encode_create_payload(
        2, (0, 10), "idle-v1", env_client.SCHEDULE_TRAIN_V2, cursor=7
    )


def test_create_accepts_rotate_measurement_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finite rotate schedule is a supported client contract."""

    process = FakeProcess(
        _create_response_frame("rotate", 4) + _close_response_frame()
    )
    monkeypatch.setattr(env_client.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with env_client.EnvClient(("fake-env",), players=2) as client:
        metadata = client.create(
            1,
            (100, 110),
            "idle-v1",
            schedule=env_client.SCHEDULE_ROTATE,
            cursor=4,
        )
        assert metadata["schedule"] == env_client.SCHEDULE_ROTATE
        assert client.schedule == env_client.SCHEDULE_ROTATE


def test_packed_step_writes_the_same_bytes_as_slot_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The faster collection input must not change the step frame Go receives.

    Go rejects each step so the test needs no valid step response; the request
    bytes are fully written before that rejection arrives.
    """

    responses = (
        _create_response_frame("train-v2", 0)
        + _error_response_frame(protocol.MessageType.STEP_REQUEST)
        + _close_response_frame()
    )
    ordinary_process = FakeProcess(responses)
    packed_process = FakeProcess(responses)
    processes = iter((ordinary_process, packed_process))
    monkeypatch.setattr(
        env_client.subprocess,
        "Popen",
        lambda *_args, **_kwargs: next(processes),
    )

    idle = env_client.SlotAction()
    ordinary_actions = {
        10: [idle] * 43 + [env_client.SlotAction(launch=True, target=1)],
        2: [env_client.SlotAction(launch=True, target=43)] + [idle] * 43,
    }
    packed_actions = {
        10: [-1] * 43 + [1],
        2: [43] + [-1] * 43,
    }

    with env_client.EnvClient(("fake-env",), players=2) as client:
        client.create(2, (0, 10), "idle-v1")
        with pytest.raises(env_client.ProtocolFailure, match="not-reset"):
            client.step(ordinary_actions)
    with env_client.EnvClient(("fake-env",), players=2) as client:
        client.create(2, (0, 10), "idle-v1")
        with pytest.raises(env_client.ProtocolFailure, match="not-reset"):
            client.step_packed(packed_actions)

    assert bytes(packed_process.stdin.data) == bytes(ordinary_process.stdin.data)
    assert ordinary_process.written_request_types() == [
        protocol.MessageType.CREATE_REQUEST,
        protocol.MessageType.STEP_REQUEST,
        protocol.MessageType.CLOSE_REQUEST,
    ]


@pytest.mark.parametrize(
    ("targets", "error", "message"),
    [
        ([-1] * 43, ValueError, "43 packed targets"),
        ([-1] * 43 + [44], ValueError, "invalid packed target 44"),
        ([-1] * 43 + [True], TypeError, "True, want an integer"),
    ],
)
def test_packed_step_rejects_malformed_rows_before_sending_to_go(
    monkeypatch: pytest.MonkeyPatch,
    targets: list[object],
    error: type[Exception],
    message: str,
) -> None:
    """An internal packing defect must fail before Go advances a game."""

    process = FakeProcess(
        _create_response_frame("train-v2", 0) + _close_response_frame()
    )
    monkeypatch.setattr(env_client.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(error, match=message):
        with env_client.EnvClient(("fake-env",), players=2) as client:
            client.create(1, (0, 10), "idle-v1")
            client.step_packed({0: targets})

    assert process.written_request_types() == [
        protocol.MessageType.CREATE_REQUEST,
        protocol.MessageType.CLOSE_REQUEST,
    ]


def test_profiled_client_separates_reset_and_step_protocol_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record encoding, Go wait, and decoding under the correct operation."""

    process = FakeProcess(
        _create_response_frame("train-v2", 0)
        + _reset_response_frame(1, (0,))
        + _step_response_frame((0,))
        + _close_response_frame()
    )
    monkeypatch.setattr(env_client.subprocess, "Popen", lambda *_args, **_kwargs: process)
    # Advancing this clock by one for every read makes each measured interval
    # exactly one second without depending on machine speed.
    ticks = count()
    profiler = PhaseProfiler("cpu", clock=lambda: float(next(ticks)))

    with env_client.EnvClient(("fake-env",), players=2, profiler=profiler) as client:
        client.create(1, (0, 10), "idle-v1")
        client.reset([0])
        client.step({0: [env_client.SlotAction()] * env_client.BODY_SLOTS})

    phases = profiler.snapshot_and_reset()
    assert phases["reset_encode"] == 1.0
    assert phases["reset_go_wait"] == 1.0
    assert phases["reset_decode"] == 1.0
    assert phases["reset_observation_decode"] == 1.0
    assert phases["action_encode"] == 1.0
    assert phases["go_wait"] == 1.0
    assert phases["observation_decode"] == 1.0
    # Only the removed env-v1 JSON transport ever entered this phase.
    assert phases["json_decode"] == 0.0


def test_importing_environment_client_does_not_import_torch() -> None:
    """Keep the standalone protocol client lightweight outside training.

    This test uses a fresh Python interpreter because this test module imports
    PhaseProfiler and therefore already has Torch loaded. The child imports
    only Orrery's public package and verifies that resolving EnvClient's
    optional profiler annotation did not load Torch as a side effect.
    """

    python_directory = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import orrery.env_client; "
                "raise SystemExit('torch' in sys.modules)"
            ),
        ],
        cwd=python_directory,
        check=False,
    )

    assert completed.returncode == 0


@pytest.mark.parametrize("schedule", ["train", "future-v2"])
def test_create_rejects_removed_and_unknown_schedule_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    schedule: str,
) -> None:
    """A stale name must not be sent and silently interpreted by the server."""

    process = FakeProcess()
    monkeypatch.setattr(env_client.subprocess, "Popen", lambda *_args, **_kwargs: process)
    client = env_client.EnvClient(("fake-env",), players=2)

    with pytest.raises(ValueError, match="unsupported schedule"):
        client.create(1, (0, 10), "idle-v1", schedule=schedule)
    client.close()

    assert bytes(process.stdin.data) == b""


@pytest.mark.parametrize(
    ("returned_schedule", "returned_cursor", "message"),
    [
        ("train", 3, "unsupported schedule"),
        # paired was replaced by rotate and is no longer a known schedule.
        ("paired", 3, "unsupported schedule"),
        ("rotate", 3, "requested 'train-v2'"),
        ("train-v2", 4, "starting cursor 4"),
    ],
)
def test_create_rejects_incompatible_replay_metadata(
    monkeypatch: pytest.MonkeyPatch,
    returned_schedule: str,
    returned_cursor: int,
    message: str,
) -> None:
    """The client must not retain a schedule or cursor different from its request."""

    process = FakeProcess(_create_response_frame(returned_schedule, returned_cursor))
    monkeypatch.setattr(env_client.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(env_client.ProtocolFailure, match=message):
        with env_client.EnvClient(("fake-env",), players=2) as client:
            client.create(1, (0, 10), "idle-v1", cursor=3)


@pytest.mark.parametrize(
    ("players", "returned_generator"),
    [(2, "generator-v2-4p"), (4, "generator-v2"), (2, "generator-v1")],
)
def test_create_rejects_a_generator_for_another_player_count(
    monkeypatch: pytest.MonkeyPatch,
    players: int,
    returned_generator: str,
) -> None:
    """Maps for the wrong player count would seat the learner unexpectedly."""

    process = FakeProcess(
        _create_response_frame("train-v2", 0, generator=returned_generator)
    )
    monkeypatch.setattr(env_client.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(
        env_client.ProtocolFailure,
        match=f"generator '{returned_generator}', want .* for {players} players",
    ):
        with env_client.EnvClient(("fake-env",), players=players) as client:
            client.create(1, (0, 10), "idle-v1")


def test_go_client_preserves_train_v2_schedule_and_cursor(
    env_command: tuple[str, ...],
) -> None:
    """Real Go create/reset responses must retain Python's replay identity."""

    with env_client.EnvClient(env_command, players=2) as client:
        metadata = client.create(2, (0, 10), "idle-v1", cursor=3)
        assert metadata["schedule"] == env_client.SCHEDULE_TRAIN_V2
        assert metadata["cursor"] == 3
        episodes = client.reset([1, 0])
        assert [episode.env_id for episode in episodes] == [0, 1]
        assert client.schedule == env_client.SCHEDULE_TRAIN_V2
        assert client.cursor == 5


def test_go_supports_the_complete_public_client_lifecycle(
    env_command: tuple[str, ...],
) -> None:
    """Use create, reset, both step forms, and close through real binary pipes.

    Codec unit tests establish individual byte layouts. This process test adds
    the missing integration guarantee: EnvClient sends those frames in the
    expected lifecycle order and exposes the public Python objects used by
    rollout and evaluation.
    """

    with env_client.EnvClient(env_command, players=2) as client:
        metadata = client.create(2, (0, 10), "idle-v1", cursor=3)
        assert metadata["v"] == "env-v3"
        assert metadata["schema"] == env_client.OBSERVATION_SCHEMA
        assert metadata["generator"] == "generator-v2"
        assert metadata["seed_range"] == [0, 10]
        assert metadata["schedule"] == env_client.SCHEDULE_TRAIN_V2
        assert metadata["cursor"] == 3
        assert metadata["dimensions"]["body_slots"] == env_client.BODY_SLOTS

        # Reset accepts IDs in caller order, while the binary response and the
        # public objects must use deterministic numeric environment-ID order.
        episodes = client.reset([1, 0])
        assert [episode.env_id for episode in episodes] == [0, 1]
        assert client.cursor == 5
        assert all(type(episode.seed) is int for episode in episodes)

        ordinary_results = client.step(
            {
                episode.env_id: [env_client.SlotAction()] * env_client.BODY_SLOTS
                for episode in episodes
            }
        )
        assert [result.env_id for result in ordinary_results] == [0, 1]
        assert [result.turn for result in ordinary_results] == [1, 1]

        packed_results = client.step_packed(
            {
                result.env_id: [-1] * env_client.BODY_SLOTS
                for result in ordinary_results
            }
        )
        assert [result.env_id for result in packed_results] == [0, 1]
        assert [result.turn for result in packed_results] == [2, 2]

        # The first call performs the close exchange. The second call and the
        # context manager's later cleanup must both be harmless no-ops.
        client.close()
        client.close()

    assert client.error_count == 0


def test_go_counts_a_structured_rejection(
    env_command: tuple[str, ...],
) -> None:
    """A rejected action remains counted without desynchronizing later work."""

    with env_client.EnvClient(env_command, players=2) as client:
        client.create(1, (0, 10), "idle-v1")

        # The target bytes are valid, but environment 0 has not been reset yet.
        # Go therefore returns a complete structured not-reset error frame.
        with pytest.raises(env_client.ProtocolFailure, match="not-reset"):
            client.step_packed({0: [-1] * env_client.BODY_SLOTS})
        assert client.error_count == 1
        assert client._exchange_in_flight is False

        # A valid reset after the rejection proves both sides still agree about
        # which response belongs to the next request.
        episodes = client.reset([0])
        assert [episode.env_id for episode in episodes] == [0]


def test_interrupted_exchange_aborts_without_masking_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted binary read must kill Go and preserve KeyboardInterrupt."""

    process = FakeProcess(stdout=InterruptingBinaryStdout())
    monkeypatch.setattr(
        env_client.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    with pytest.raises(KeyboardInterrupt):
        with env_client.EnvClient(("fake-env",), players=2) as client:
            client.create(1, (0, 10), "idle-v1")

    assert process.killed
    assert process.waited
    # Only the create frame was written. Cleanup must not append close bytes
    # after an interruption leaves the response boundary unknown.
    assert process.written_request_types() == [protocol.MessageType.CREATE_REQUEST]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (b"OW", "truncated env-v3 frame header"),
        (
            protocol.encode_frame(
                protocol.MessageType.CREATE_RESPONSE,
                b"abc",
            )[:-1],
            "truncated env-v3 frame payload",
        ),
        (
            b"BAD!" + protocol.encode_frame(
                protocol.MessageType.CREATE_RESPONSE,
                b"",
            )[4:],
            "frame magic",
        ),
    ],
)
def test_malformed_response_aborts_without_sending_close(
    monkeypatch: pytest.MonkeyPatch,
    response: bytes,
    message: str,
) -> None:
    """An uncertain frame boundary must terminate rather than attempt recovery."""

    process = FakeProcess(response)
    monkeypatch.setattr(
        env_client.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    with pytest.raises(env_client.ProtocolFailure, match=message):
        with env_client.EnvClient(("fake-env",), players=2) as client:
            client.create(1, (0, 10), "idle-v1")

    assert process.killed
    assert process.waited
    assert client._exchange_in_flight is True
    assert process.written_request_types() == [protocol.MessageType.CREATE_REQUEST]
