"""Run the Go environment and expose its responses as Python values.

EnvClient exchanges env-v3 binary frames with the Go process. This module also
contains the narrow adapters that expose validated env-v3 binary columns
through the public Episode, StepResult, and Observation types. Keeping that
conversion at this public boundary prevents protocol code from depending on
client classes and prevents the client from rebuilding already-decoded NumPy
observation arrays.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from . import protocol
from .paths import REPOSITORY_ROOT

if TYPE_CHECKING:
    # EnvClient only calls methods on a profiler supplied by training. Keeping
    # this import out of normal execution prevents lightweight environment
    # clients from importing Torch merely to resolve a type annotation.
    from .profiling import PhaseProfiler
    from .protocol import (
        DecodedObservationBatch,
        DecodedResetResponse,
        DecodedStepResponse,
    )


# A disabled profile still enters a ``with`` block so the operation itself has
# one implementation. This shared context performs no clock reads or CUDA waits.
_NO_PROFILE_PHASE = nullcontext()


def _profile_phase(
    profiler: PhaseProfiler | None,
    phase: str,
) -> AbstractContextManager[None]:
    """Select a measured context when profiling is enabled, else a no-op."""

    if profiler is None:
        return _NO_PROFILE_PHASE
    return profiler.phase(phase)


# Name of the request/response transport spoken by the Go process. env-v3 is
# the only one, and cmd/env takes no flag to choose it.
PROTOCOL_VERSION = protocol.PROTOCOL_VERSION

# Name of the fixed observation layout returned inside environment messages.
OBSERVATION_SCHEMA = protocol.OBSERVATION_SCHEMA

# These are the only schedule identities the client accepts. The training name
# includes its version because changing the cursor mapping would otherwise make
# old checkpoint cursors silently select different episodes and seats.
# rotate plays every seed once from each seat on the map, seat 0 first, and
# is what every validation and evaluation path uses.
SCHEDULE_TRAIN_V2 = "train-v2"
SCHEDULE_ROTATE = "rotate"
SUPPORTED_SCHEDULES = frozenset((SCHEDULE_TRAIN_V2, SCHEDULE_ROTATE))

# cmd/env serves two-player generator-v2 maps or four-player generator-v2-4p
# maps for the whole process, chosen by --players. The create response names
# the generator, and the client requires it to match the count it asked for.
GENERATOR_FOR_PLAYERS = {2: "generator-v2", 4: "generator-v2-4p"}

# Every observation reserves 44 body rows: slots 0-39 for planets and 40-43
# for comets. Boards with fewer bodies fill the remaining rows with padding and
# mark them in body_mask.
BODY_SLOTS = protocol.BODY_SLOTS

# Number of numeric measurements Go supplies for each body row.
BODY_FEATURES = protocol.BODY_FEATURES

# Every observation reserves four player rows: me, clockwise, opposite, and
# counter-clockwise. A two-player game uses rows 0 and 2.
PLAYER_SLOTS = protocol.PLAYER_SLOTS

# Number of numeric measurements Go supplies for each player row.
PLAYER_FEATURES = protocol.PLAYER_FEATURES

# The four sizes above as create metadata records them under "dimensions".
# Training, resume, and evaluation compare recorded metadata with this value.
OBSERVATION_DIMENSIONS = {
    "body_slots": BODY_SLOTS,
    "body_features": BODY_FEATURES,
    "player_slots": PLAYER_SLOTS,
    "player_features": PLAYER_FEATURES,
}

# Why an episode ended, as reported in StepResult.end_reason.
END_ONGOING = protocol.END_ONGOING
END_GAME_FINISHED = protocol.END_GAME_FINISHED
END_LEARNER_ELIMINATED = protocol.END_LEARNER_ELIMINATED

class ProtocolFailure(RuntimeError):
    """Report that Python and the Go environment could not communicate safely."""


@dataclass(frozen=True)
class SlotAction:
    """One body's launch/no-launch decision using observation slot numbers."""

    # False leaves this source idle. True asks Go to launch from it.
    launch: bool = False
    # Destination body slot. Go ignores this placeholder when launch is False.
    target: int = 0


@dataclass(frozen=True)
class Observation:
    """One game state represented by fixed-shape NumPy arrays."""

    # Decimal features for every reserved body and player row. Masks below say
    # which rows contain real entities rather than unused padding.
    bodies: NDArray[np.float32]
    players: NDArray[np.float32]
    # True for real bodies, learner-owned bodies, and real player rows.
    body_mask: NDArray[np.bool_]
    owned_mask: NDArray[np.bool_]
    player_mask: NDArray[np.bool_]
    # target_mask[source, target] is True exactly when Go says that pair is a
    # legal choice for the learner on this turn.
    target_mask: NDArray[np.bool_]
    # Maps each body slot back to Go's body ID. Padding slots carry a placeholder.
    slot_body_id: NDArray[np.int32]


@dataclass(frozen=True)
class Episode:
    """A newly reset episode and the identity needed to replay it."""

    # Reusable slot inside the batch of concurrently running games.
    env_id: int
    # Board generator seed and learner position needed to recreate the game.
    seed: int
    seat: int
    # True means the caller requested this seed/seat directly instead of taking
    # the next assignment from the configured schedule.
    explicit: bool
    # First state from which the controller must choose an action.
    observation: Observation


@dataclass(frozen=True)
class StepResult(Episode):
    """One environment after a complete game turn."""

    # Reward produced by this turn. Training pays only at the end of an episode.
    reward: float
    # terminated means the episode ended by the rules: the game finished, or
    # the learner was eliminated while others play on. truncated means an
    # external limit stopped it while it could otherwise have continued.
    terminated: bool
    truncated: bool
    # Why the episode ended: END_ONGOING, END_GAME_FINISHED, or
    # END_LEARNER_ELIMINATED. Nonzero exactly when terminated is True.
    end_reason: int
    # Authoritative Go turn number after the submitted action was processed.
    turn: int


class EnvClient:
    """Own one Go server process speaking env-v3 binary frames.

    Use this class as a context manager so the child process is always closed
    or killed, even when a test or controller raises an exception.
    """

    def __init__(
        self,
        command: Sequence[str] = ("go", "run", "./cmd/env"),
        *,
        players: int,
        cwd: Path | str = REPOSITORY_ROOT,
        profiler: PhaseProfiler | None = None,
    ) -> None:
        """Start the configured environment command and open its three pipes.

        `command` is a sequence rather than one shell string, so no extra shell
        interprets it. `players` chooses two- or four-player maps for the whole
        process; it has no default, so every caller states which game it plays.
        `cwd` defaults to the repository root required by the Go command.
        `profiler` records protocol phases only when explicitly supplied;
        ordinary clients perform no synchronization or timer calls.
        """

        # bool is a subclass of int, so True would otherwise pass as 1 player
        # and reach the check below with a confusing message.
        if isinstance(players, bool) or players not in GENERATOR_FOR_PLAYERS:
            supported = " or ".join(str(count) for count in GENERATOR_FOR_PLAYERS)
            raise ValueError(f"players must be {supported}, got {players!r}")
        # The player count create must see reflected in the generator Go names.
        self.players = players

        # Pass the player count as a flag; cmd/env refuses any other count
        # before reading stdin. Build a new list instead of modifying a
        # caller-owned command sequence.
        process_command = [*command, "--players", str(players)]

        # Binary frames must preserve every byte exactly, so text transcoding
        # is disabled and the unbuffered streams hand framing control to
        # protocol.
        self._process = subprocess.Popen(
            process_command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        # Popen should supply all requested pipes. Check anyway so the client
        # never stores a missing stream and fails during a later game turn.
        if (
            self._process.stdin is None
            or self._process.stdout is None
            or self._process.stderr is None
        ):
            self._process.kill()
            raise ProtocolFailure("failed to open environment process pipes")
        self._stdin: BinaryIO = self._process.stdin
        self._stdout: BinaryIO = self._process.stdout
        self._stderr: BinaryIO = self._process.stderr
        # create must succeed before close sends a protocol close command.
        self._created = False
        # Once closed or aborted, no later request may reuse this client.
        self._closed = False
        # True means a request may have reached Go but its matching response
        # has not been consumed. Sending another command in that state would
        # make the two sides disagree about which response belongs to which
        # request, so cleanup must terminate the process instead.
        self._exchange_in_flight = False
        # Public evidence used by acceptance tests: every structured rejection
        # from Go increments this count before the exception is raised.
        self.error_count = 0
        # Number of scheduled episode assignments consumed so far. Reset
        # responses replace it with Go's authoritative value.
        self.cursor = 0
        # Accepted schedule identity, recorded with cursor in replay metadata.
        # None means create has not yet completed successfully.
        self.schedule: str | None = None
        # Replay and compatibility information returned by create, including
        # protocol, engine, generator, opponent, and seed-range identities.
        self.metadata: dict[str, Any] = {}
        # The trainer and client share one accumulator so action encoding, Go
        # waiting, and response decoding appear beside policy phases for the
        # same update. None keeps the normal hot path free of timer work.
        self._profiler = profiler

    def __enter__(self) -> EnvClient:
        """Return this client when entering a `with EnvClient(...)` block."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: object,
    ) -> bool:
        """Clean up the child while preserving any exception from user code."""

        # If user code was interrupted while waiting for a response, Python no
        # longer knows whether Go consumed the request. A graceful close would
        # send another message into that uncertain conversation, so kill it.
        if exc_type is not None and self._exchange_in_flight:
            self._abort()
            return False
        try:
            self.close()
        except BaseException:
            # Without an earlier error, cleanup failure is the useful error and
            # must be reported. With one, preserve the original failure instead.
            if exc_type is None:
                raise
            # Cleanup must not replace the exception already leaving the with
            # block. The original exception remains the useful failure.
            self._abort()
        return False

    def create(
        self,
        batch: int,
        seed_range: tuple[int, int],
        opponent: str,
        *,
        schedule: str = SCHEDULE_TRAIN_V2,
        cursor: int = 0,
    ) -> Mapping[str, Any]:
        """Configure the reusable game batch and return its replay metadata.

        `batch` chooses how many games can run concurrently. `seed_range`,
        `schedule`, and `cursor` tell Go which deterministic episode assignments
        to make. `opponent` identifies the scripted agent used in every game.
        """

        if schedule not in SUPPORTED_SCHEDULES:
            supported = ", ".join(sorted(SUPPORTED_SCHEDULES))
            raise ValueError(
                f"unsupported schedule {schedule!r}; supported schedules: {supported}"
            )

        # Create sends configuration only; individual games begin later when
        # reset names the environment slots to start.
        payload = protocol.encode_create_payload(
            batch,
            seed_range,
            opponent,
            schedule,
            cursor=cursor,
        )
        response_payload = self._exchange_binary(
            protocol.MessageType.CREATE_REQUEST,
            payload,
        )
        try:
            decoded = protocol.decode_create_response(response_payload)
        except protocol.ProtocolError as error:
            raise ProtocolFailure(str(error)) from error
        response = _binary_create_metadata(decoded)
        # A matching protocol version is not enough: the observation fields
        # must also follow the exact layout this Python client decodes.
        if response.get("schema") != OBSERVATION_SCHEMA:
            raise ProtocolFailure(
                f"observation schema is {response.get('schema')!r}, "
                f"want {OBSERVATION_SCHEMA!r}"
            )
        # cmd/env chose its generator from the --players flag this client
        # passed. A different generator means a stale binary or a wiring
        # mistake, and its maps would not have the seats this client expects.
        expected_generator = GENERATOR_FOR_PLAYERS[self.players]
        returned_generator = _string(response, "generator")
        if returned_generator != expected_generator:
            raise ProtocolFailure(
                f"environment returned generator {returned_generator!r}, "
                f"want {expected_generator!r} for {self.players} players"
            )
        returned_schedule = _string(response, "schedule")
        if returned_schedule not in SUPPORTED_SCHEDULES:
            raise ProtocolFailure(
                f"environment returned unsupported schedule {returned_schedule!r}"
            )
        if returned_schedule != schedule:
            raise ProtocolFailure(
                f"environment returned schedule {returned_schedule!r}, "
                f"requested {schedule!r}"
            )
        returned_cursor = _integer(response, "cursor")
        if returned_cursor != cursor:
            raise ProtocolFailure(
                f"environment returned starting cursor {returned_cursor}, "
                f"requested {cursor}"
            )
        # Store only after every compatibility check above has passed. This
        # prevents close from treating a rejected create as a live environment.
        self._created = True
        self.schedule = returned_schedule
        self.cursor = returned_cursor
        self.metadata = dict(response)
        return self.metadata

    def reset(self, ids: Sequence[int]) -> list[Episode]:
        """Start fresh episodes in the named reusable environment slots."""

        # Encode and decode use the same requested IDs. The decoder checks that
        # Go returned every requested slot exactly once and in numeric order
        # before any observation becomes public.
        with _profile_phase(self._profiler, "reset_encode"):
            payload = protocol.encode_reset_payload(ids)
        response_payload = self._exchange_binary(
            protocol.MessageType.RESET_REQUEST,
            payload,
            wait_phase="reset_go_wait",
        )
        try:
            with _profile_phase(self._profiler, "reset_decode"):
                decoded = protocol.decode_reset_response(
                    response_payload,
                    expected_env_ids=ids,
                )
        except protocol.ProtocolError as error:
            raise ProtocolFailure(str(error)) from error
        # The server assigns episodes in schedule order and owns this cursor.
        # Retaining it lets later checkpoint code resume at the next assignment.
        self.cursor = decoded.cursor
        with _profile_phase(self._profiler, "reset_observation_decode"):
            return _episodes_from_binary(decoded)

    def step(
        self, actions: Mapping[int, Sequence[SlotAction]]
    ) -> list[StepResult]:
        """Advance every named environment by one turn using its action list."""

        # env-v3 represents an idle source as -1 and a launch as its target
        # slot. Convert the public SlotAction objects into that compact form
        # before the shared binary step path writes one frame.
        with _profile_phase(self._profiler, "action_encode"):
            packed_actions = {
                env_id: [action.target if action.launch else -1 for action in row]
                for env_id, row in actions.items()
            }
            payload = protocol.encode_step_payload(packed_actions)
        return self._step_binary(payload, tuple(actions))

    def step_packed(
        self,
        action_targets: Mapping[int, Sequence[int]],
    ) -> list[StepResult]:
        """Advance games from compact target rows produced by collection.

        Every row contains one integer for each body slot. ``-1`` means the
        body does not launch; values from 0 through 43 mean launch at that
        target slot. This representation arrives from Torch in one CPU transfer
        and avoids constructing 22,528 temporary ``SlotAction`` objects per
        turn in the standard 512-environment batch. The same values become the
        fixed 44 signed target bytes defined by the binary protocol.
        """

        with _profile_phase(self._profiler, "action_encode"):
            payload = protocol.encode_step_payload(action_targets)
        return self._step_binary(payload, tuple(action_targets))

    def _step_binary(
        self,
        payload: bytes,
        expected_env_ids: Sequence[int],
    ) -> list[StepResult]:
        """Exchange one encoded step frame and expose its validated rows."""

        response_payload = self._exchange_binary(
            protocol.MessageType.STEP_REQUEST,
            payload,
            wait_phase="go_wait",
        )
        try:
            # Fixed columns need no separate parsing into nested Python lists.
            # Decode the columns and create the public result objects in the
            # observation_decode profiling phase.
            with _profile_phase(self._profiler, "observation_decode"):
                decoded = protocol.decode_step_response(
                    response_payload,
                    expected_env_ids=expected_env_ids,
                )
                return _step_results_from_binary(decoded)
        except protocol.ProtocolError as error:
            raise ProtocolFailure(str(error)) from error

    def close(self) -> None:
        """Ask a synchronized Go process to exit, then verify a clean shutdown."""

        # Repeated close calls are harmless, which simplifies context-manager
        # cleanup and callers that also close explicitly.
        if self._closed:
            return
        if self._exchange_in_flight:
            self._abort()
            raise ProtocolFailure(
                "cannot close cleanly while a protocol response is pending"
            )
        try:
            if self._created and self._process.poll() is None:
                # Close is a normal protocol exchange only when create succeeded
                # and the child is still running.
                response_payload = self._exchange_binary(
                    protocol.MessageType.CLOSE_REQUEST,
                    protocol.encode_close_payload(),
                )
                try:
                    protocol.decode_close_response(response_payload)
                except protocol.ProtocolError as error:
                    raise ProtocolFailure(str(error)) from error
                # No more requests will be written. Wait for Go, then require
                # both a successful status and completely clean output streams.
                self._stdin.close()
                return_code = self._process.wait(timeout=30)
                # Decode the binary pipes before building diagnostics so
                # failures read as ordinary text instead of Python's ``b'...'``
                # repr.
                diagnostics = _process_output_text(self._stderr.read())
                extra_stdout = _process_output_text(self._stdout.read())
                if return_code != 0:
                    raise ProtocolFailure(
                        f"environment exited with status {return_code}: "
                        f"{diagnostics.strip()}"
                    )
                if diagnostics:
                    raise ProtocolFailure(
                        f"environment wrote unexpected stderr: {diagnostics!r}"
                    )
                if extra_stdout:
                    raise ProtocolFailure(
                        f"environment wrote extra stdout after close: {extra_stdout!r}"
                    )
        finally:
            # Mark the client closed in every path and kill a child that did not
            # complete the graceful shutdown above.
            self._abort()

    def _exchange_binary(
        self,
        request_type: protocol.MessageType,
        payload: bytes,
        *,
        wait_phase: str | None = None,
    ) -> bytes:
        """Send one env-v3 frame and return its matched response payload.

        A successful frame or structured Go rejection consumes one complete
        response, so the next request remains aligned. Any other I/O or framing
        failure leaves ``_exchange_in_flight`` set: context-manager cleanup then
        kills the process instead of risking a close request at the wrong byte.
        """

        if self._closed:
            raise ProtocolFailure("environment client is closed")
        if self._exchange_in_flight:
            raise ProtocolFailure("another protocol response is still pending")

        self._exchange_in_flight = True
        try:
            # The optional phase starts before the write and ends only after the
            # full response arrives. It therefore measures pipe transfer plus
            # Go's request decoding, game work, and response encoding.
            with _profile_phase(
                self._profiler if wait_phase is not None else None,
                wait_phase or "go_wait",
            ):
                protocol.write_frame(self._stdin, request_type, payload)
                frame = protocol.read_response_frame(self._stdout, request_type)
        except protocol.RemoteProtocolError as error:
            # Go encoded a complete structured response, so the stream remains
            # safe for a later close even though this operation was rejected.
            self._exchange_in_flight = False
            self.error_count += 1
            raise ProtocolFailure(str(error)) from error
        except protocol.ProtocolError as error:
            # A partial write, truncated read, or malformed frame makes the next
            # byte boundary uncertain. Leave the in-flight marker set so cleanup
            # forcibly terminates the child.
            raise ProtocolFailure(str(error)) from error

        self._exchange_in_flight = False
        return frame.payload

    def _abort(self) -> None:
        """Stop the child immediately without sending another protocol message."""

        self._closed = True
        # poll returns None only while the process is still alive. kill stops it;
        # wait collects its exit status so no zombie child is left behind.
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()


def _binary_create_metadata(
    response: protocol.DecodedCreateResponse,
) -> dict[str, Any]:
    """Present env-v3 create fields as the public metadata dictionary.

    Training, checkpoints, and evaluation consume this mapping and record it
    in run artifacts. Rebuilding only these few control-plane values keeps
    those callers simple; the much larger reset and step observations remain
    zero-copy NumPy views.
    """

    return {
        "v": PROTOCOL_VERSION,
        "engine": response.engine,
        "schema": response.schema,
        "generator": response.generator,
        "opponent": response.opponent,
        # Recorded run metadata stores this pair as a list, so keep that shape
        # for comparisons against previously saved runs.
        "seed_range": list(response.seed_range),
        "schedule": response.schedule,
        "cursor": response.cursor,
        # ObservationDimensions fields use the snake-case names recorded in run
        # metadata, so asdict produces the expected representation.
        "dimensions": asdict(response.dimensions),
    }


def _process_output_text(value: bytes) -> str:
    """Return readable diagnostics from a binary process pipe.

    Go normally writes UTF-8 diagnostics. Replacement decoding keeps an
    unexpected non-UTF-8 byte from hiding the original process failure behind
    a secondary ``UnicodeDecodeError`` during cleanup.
    """

    return value.decode("utf-8", errors="replace")


def _episodes_from_binary(response: DecodedResetResponse) -> list[Episode]:
    """Expose decoded reset rows through the public Episode API.

    The binary decoder has already checked every dtype, shape, value, and row
    relationship. This boundary therefore creates the public Python objects
    without converting the complete NumPy columns a second time.
    """

    # ``tolist`` converts each complete scalar column in NumPy's compiled code.
    # Reading ``response.env_ids[row]`` inside the loop would cross the
    # NumPy-to-Python boundary once per environment and once per field. The
    # lists already contain the ordinary Python int and bool values promised by
    # Episode, so the loop below only assembles the public objects.
    env_ids = response.env_ids.tolist()
    seeds = response.seeds.tolist()
    seats = response.seats.tolist()
    explicit_values = response.explicit.tolist()

    episodes: list[Episode] = []
    for row, env_id in enumerate(env_ids):
        episodes.append(
            Episode(
                env_id=env_id,
                seed=seeds[row],
                seat=seats[row],
                explicit=explicit_values[row],
                observation=_observation_from_binary_row(
                    response.observations,
                    row,
                ),
            )
        )
    return episodes


def _step_results_from_binary(response: DecodedStepResponse) -> list[StepResult]:
    """Expose decoded step rows without rebuilding their observation arrays."""

    # These nine arrays contain only one scalar per environment. Converting
    # each whole column once avoids thousands of individual NumPy scalar reads
    # in a large batch. Observation columns stay as arrays because converting
    # their 3,284 values per environment would discard the binary transport's
    # main allocation and performance benefit.
    env_ids = response.env_ids.tolist()
    seeds = response.seeds.tolist()
    seats = response.seats.tolist()
    explicit_values = response.explicit.tolist()
    rewards = response.rewards.tolist()
    terminated_values = response.terminated.tolist()
    truncated_values = response.truncated.tolist()
    end_reasons = response.end_reasons.tolist()
    turns = response.turns.tolist()

    results: list[StepResult] = []
    for row, env_id in enumerate(env_ids):
        results.append(
            StepResult(
                env_id=env_id,
                seed=seeds[row],
                seat=seats[row],
                explicit=explicit_values[row],
                observation=_observation_from_binary_row(
                    response.observations,
                    row,
                ),
                reward=rewards[row],
                terminated=terminated_values[row],
                truncated=truncated_values[row],
                end_reason=end_reasons[row],
                turn=turns[row],
            )
        )
    return results


def _observation_from_binary_row(
    batch: DecodedObservationBatch,
    row: int,
) -> Observation:
    """Wrap one environment's existing NumPy row views as an Observation.

    Indexing the first axis returns views into the decoded response buffer. It
    does not call ``np.asarray``, allocate replacement arrays, or walk through
    the 3,284 individual wire values that make up one observation.
    """

    return Observation(
        bodies=batch.bodies[row],
        players=batch.players[row],
        body_mask=batch.body_mask[row],
        owned_mask=batch.owned_mask[row],
        player_mask=batch.player_mask[row],
        target_mask=batch.target_mask[row],
        slot_body_id=batch.slot_body_id[row],
    )


def _field(data: Mapping[str, Any], name: str) -> Any:
    """Return one required metadata field instead of silently defaulting it."""

    if name not in data:
        raise ProtocolFailure(f"response is missing {name!r}")
    return data[name]


def _integer(data: Mapping[str, Any], name: str) -> int:
    """Read a required whole-number field while rejecting true/false values."""

    value = _field(data, name)
    # bool inherits from int in Python, so `isinstance(True, int)` is true even
    # though a protocol seed, cursor, seat, ID, or turn must not be a Boolean.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolFailure(f"response field {name!r} is not an integer: {value!r}")
    return value


def _string(data: Mapping[str, Any], name: str) -> str:
    """Read a required field and require an actual string value."""

    value = _field(data, name)
    if not isinstance(value, str):
        raise ProtocolFailure(f"response field {name!r} is not a string: {value!r}")
    return value
