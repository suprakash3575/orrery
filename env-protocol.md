# Orrery Environment Protocol (`env-v3`)

## What this protocol does

The Python trainer chooses actions; the Go environment runs the game rules.
They run as separate processes and exchange binary messages through pipes:

```text
Python client  -- requests on Go's stdin -->  Go environment (cmd/env)
Python client  <-- responses on stdout --  Go environment (cmd/env)
```

A **protocol** is the agreement about what messages mean and how their bytes
are arranged. Orrery uses **`env-v3`**, which carries observations using
the **`obs-v2`** schema. These names describe different things:

- `env-v3` defines the messages: create, reset, step, close, and their responses.
- `obs-v2` defines the game-state arrays inside reset and step responses.

`cmd/env` speaks this protocol without a protocol-selection flag. Its
`--players 2` or `--players 4` flag chooses the map's player count for the
whole process. Standard output contains only protocol responses; startup and
fatal diagnostics go to standard error.

For normal Python use, start with
[`EnvClient`](python/orrery/env_client.py). It starts Go, handles the bytes,
and returns Python objects containing NumPy arrays. The binary reference
below is for readers working on the encoder or decoder. Go remains
authoritative for game rules, action legality, rewards, and episode endings.

## Environment IDs and schedules

An **environment ID** identifies a reusable game slot, not a scenario seed.
For example, slots 0 and 1 can play seed 10 from different learner seats.
Resetting a slot replaces its episode while keeping its environment ID.

The **schedule cursor** counts scheduled episode assignments, not game turns.
With `rotate` on a two-player map, assignments are `(seed 10, seat 0)`,
`(seed 10, seat 1)`, `(seed 11, seat 0)`, and so on. Reset assigns them in
ascending environment-ID order, so requesting `[1, 0]` still returns IDs
`[0, 1]`. Stepping does not move the schedule cursor.

The default schedule, `train-v2`, repeatedly walks the seed range and chooses
seats deterministically from the seed and pass number. `rotate` makes one pass
through every seed and seat, then reports `seed-range-exhausted`.

## The conversation lifecycle

1. **Create** once to configure reusable slots. This does not start their games.
2. **Reset** selected slots to start episodes and receive initial observations.
3. **Step** selected active slots to advance each by one complete game turn.
   Use the returned observation for the next decision. Reset completed slots
   before stepping them again.
4. **Close** when finished. Go acknowledges the request and exits.

Reset, step, and close require a successful create. A second successful create
is not allowed. Requests can name a subset of the batch; other slots stay as
they are. Valid environment IDs are `0` through `batch_size - 1`.

Python sends one request and waits for its complete response before sending
another. There are no request IDs and no overlapping requests on one connection.

## Reading observations and choosing actions

A **body slot** is an array index for a planet or comet. It differs from the
engine's body ID. Each observation reserves 44 slots: 0–39 for planets and
40–43 for comets. Unused slots are padding. Consult the masks when reading
rows or choosing actions.

| Python observation field | Shape for one environment | Meaning |
| --- | --- | --- |
| `bodies` | `(44, 27)` | Numeric features for each body slot |
| `players` | `(4, 6)` | Numeric features for each player role |
| `body_mask` | `(44,)` | True where a real body occupies the slot |
| `owned_mask` | `(44,)` | True where the learner owns that body |
| `player_mask` | `(4,)` | True for roles seated at the start of the episode |
| `target_mask` | `(44, 44)` | True where a source-to-target launch is legal |
| `slot_body_id` | `(44,)` | Engine body ID for the slot; 0 for padding |

Player rows describe roles relative to the learner: me, clockwise, opposite,
and counter-clockwise. Two-player games use rows 0 and 2. An eliminated
player remains marked in `player_mask` because they participated in the episode.

Features include positions, ship counts, ownership, and predicted incoming
fleets. Their scales are fixed by `obs-v2`; values are not all restricted to
0–1. For example, ship features use `log1p(ships) / log1p(1000)` and can exceed
1. Feature indices and meanings are documented in
[`mlenv/observation.go`](mlenv/observation.go).

A packed action has **44 signed integers**, one for each source slot:

- `-1`: do not launch from this source.
- `0` through `43`: launch from this source toward that target slot.

Targets are slot numbers. A number in range is only legal when the current
observation's `target_mask[source, target]` is true. Ownership alone is not
enough: an owned body may have no ships available to launch.

Use a fresh observation after every step; an earlier mask may no longer
represent legal actions. `client.step()` also accepts 44 `SlotAction` objects
per environment and converts them to the same packed representation.

## Binary reference: basic encodings

A **frame** is one message: a 12-byte header followed by its **payload**, the
message-specific data. All offsets and sizes below are in bytes, with no
padding between fields.

| Name | Bytes | Encoding |
| --- | ---: | --- |
| `u8` | 1 | Unsigned integer |
| `i8` | 1 | Signed two's-complement integer |
| `u16` | 2 | Little-endian unsigned integer |
| `u32` | 4 | Little-endian unsigned integer |
| `i32` | 4 | Little-endian signed two's-complement integer |
| `i64` | 8 | Little-endian signed two's-complement integer |
| `f32` | 4 | IEEE 754 binary32, little-endian |
| `bool` | 1 | `0` is false, `1` is true; all other values are invalid |
| `string` | Variable | `u32` byte length followed by that many UTF-8 bytes |

**Little-endian** means the least significant byte comes first. For example,
`u32(44)` is `2c 00 00 00` in hexadecimal. An `i8(-1)` is `ff`.

Strings have no terminating byte. For example, `"rotate"` is encoded as:

```text
06 00 00 00  72 6f 74 61 74 65
length = 6   r  o  t  a  t  e
```

Lengths count UTF-8 bytes, not characters. General strings are limited to
1 MiB (1,048,576 bytes); opponent identities have a 1,024-byte limit and
schedule identities a 64-byte limit. Invalid UTF-8 or a length extending past
the payload is malformed.

Encode fields individually. Serializing a native Go or C struct could
introduce compiler-dependent padding and change the wire layout.

## Frame header and message types

Every request and response starts with this header:

| Offset | Size | Field | Required value |
| ---: | ---: | --- | --- |
| 0 | 4 | Magic: identifies the protocol | ASCII `OWE3` |
| 4 | 1 | Message type | One value from the table below |
| 5 | 1 | Flags | `0` |
| 6 | 2 | Reserved | `0` |
| 8 | 4 | Payload length | `u32` count of bytes after the header |

The payload limit is **64 MiB (67,108,864 bytes)**, excluding the header.
This bounds how much memory a reader may allocate for one message.

| Operation | Request type | Success response type |
| --- | ---: | ---: |
| Create | `0x01` | `0x81` |
| Reset | `0x02` | `0x82` |
| Step | `0x03` | `0x83` |
| Close | `0x04` | `0x84` |

An error response uses `0xff` and may answer any request. A success response
must match its request's operation; a different response type is a protocol
failure even if its payload could be decoded.

For example, a close request has no payload. Its entire frame is:

```text
4f 57 45 33  04    00     00 00     00 00 00 00
O  W  E  3   type  flags  reserved  payload length = 0
```

Its successful response has type `0x84`, payload length 1, and payload `01`.

## Create messages

### Request payload

| Order | Field | Type |
| ---: | --- | --- |
| 1 | Batch size | `u32` |
| 2 | Seed-range start, inclusive | `i64` |
| 3 | Seed-range end, exclusive | `i64` |
| 4 | Starting schedule cursor | `i64` |
| 5 | Opponent identity | `string` |
| 6 | Schedule identity | `string` |

Batch size must be positive and at most **9,482**. This is the largest batch
whose full step response fits in one frame; it is a wire limit, not a
recommended training size. Names select supported opponent and schedule
implementations. The schedules are `train-v2` and `rotate`.

### Response payload

| Order | Field | Type or required value |
| ---: | --- | --- |
| 1 | Engine version | `string` |
| 2 | Observation schema | `string`, `"obs-v2"` |
| 3 | Scenario generator | `string` |
| 4 | Canonical opponent identity | `string` |
| 5 | Accepted seed-range start | `i64` |
| 6 | Accepted seed-range end | `i64` |
| 7 | Accepted schedule identity | `string` |
| 8 | Accepted starting cursor | `i64` |
| 9 | Body slots | `u32`, `44` |
| 10 | Body features | `u32`, `27` |
| 11 | Player slots | `u32`, `4` |
| 12 | Player features | `u32`, `6` |

Dimensions let a reader check its expected shapes before decoding observations.
The generator is `generator-v2` for two-player maps and `generator-v2-4p` for
four-player maps.

The Python client exposes this information as metadata, adding
`"v": "env-v3"` locally. The protocol name is not a separate payload field;
the frame magic already identifies it. Python groups the four shape values
under `metadata["dimensions"]`.

## Reset messages

### Request payload

| Order | Field | Type or shape |
| ---: | --- | --- |
| 1 | Environment count `N` | `u32` |
| 2 | Requested environment IDs | `u32[N]` |
| 3 | Explicit-override count `E` | `u32` |
| 4 | Explicit overrides | `E` entries of `(u32 env_id, i64 seed, u8 seat)` |

An **explicit override** requests an exact seed and learner seat for debugging
or replay instead of taking the next scheduled assignment. Its ID must appear
in the requested-ID list. Requested IDs and override IDs must each be unique.
Explicit assignments do not advance the schedule cursor.

Seats must be seated on the selected map: `0..1` for two players or `0..3`
for four players. This is the learner's actual seat; its observation role is
always "me".

Go sorts IDs before assigning scheduled episodes. The Python encoder sorts
IDs and overrides so equivalent requests have identical bytes.
`EnvClient.reset(ids)` sends no overrides; the lower-level
`protocol.encode_reset_payload()` supports them.

For example, resetting IDs `[0, 1]` without overrides encodes:

```text
u32(2), u32(0), u32(1), u32(0)
count   env IDs         override count
```

### Response payload

| Order | Field | Type or shape |
| ---: | --- | --- |
| 1 | Next schedule cursor | `i64` |
| 2 | Environment count `N` | `u32` |
| 3 | Environment IDs | `u32[N]` |
| 4 | Seeds | `i64[N]` |
| 5 | Seats | `u8[N]` |
| 6 | Explicit-assignment flags | `bool[N]` |
| 7 | Observations | Observation block for `N` environments |

IDs are strictly increasing. All following columns use the same order. An
explicit-assignment flag is true when the episode used an override. The
cursor points to the next scheduled assignment.

## Step messages

### Request payload

| Order | Field | Type or shape |
| ---: | --- | --- |
| 1 | Environment count `N` | `u32` |
| 2 | Environment IDs | `u32[N]` |
| 3 | Packed action targets | `i8[N, 44]` |

Every environment needs exactly 44 targets, including `-1` for sources that
stay idle. Values outside `-1..43` are invalid. Go validates source ownership
and target legality before advancing any game.

IDs may arrive in any order and may not repeat. Target rows must follow the
same order as the ID array. Python sorts IDs and action rows together; Go
returns results in ascending environment-ID order.

For one environment, ID 0, taking no launch actions, the payload is:

```text
u32(1), u32(0), 44 copies of i8(-1)
count   env ID  targets, each encoded as ff
```

### Response payload

| Order | Field | Type or shape |
| ---: | --- | --- |
| 1 | Environment count `N` | `u32` |
| 2 | Environment IDs | `u32[N]` |
| 3 | Seeds | `i64[N]` |
| 4 | Seats | `u8[N]` |
| 5 | Explicit-assignment flags | `bool[N]` |
| 6 | Rewards | `f32[N]` |
| 7 | Terminated flags | `bool[N]` |
| 8 | Truncated flags | `bool[N]` |
| 9 | End reasons | `u8[N]` |
| 10 | Completed turn numbers | `u32[N]` |
| 11 | Observations | Observation block for `N` environments |

`terminated` means the episode ended by game rules. `truncated` represents an
external cutoff; the current environment leaves it false. End reasons are:

| Value | Meaning | Required `terminated` value |
| ---: | --- | --- |
| 0 | Episode is still playing | False |
| 1 | Game finished | True |
| 2 | Learner eliminated while the game continues for other players | True |

Both encoder and decoder reject unknown reasons or disagreement with
`terminated`. A terminal response includes the final observation. Step does
not automatically reset the episode.

## Observation block on the wire

For `N` environments, the block stores these seven columns in this exact order:

| Order | Field | Type and shape | Bytes per environment |
| ---: | --- | --- | ---: |
| 1 | Bodies | `f32[N, 44, 27]` | 4,752 |
| 2 | Players | `f32[N, 4, 6]` | 96 |
| 3 | Body mask | `bool[N, 44]` | 44 |
| 4 | Owned-body mask | `bool[N, 44]` | 44 |
| 5 | Player mask | `bool[N, 4]` | 4 |
| 6 | Target mask | `bool[N, 44, 44]` | 1,936 |
| 7 | Slot-to-body IDs | `i32[N, 44]` | 176 |
| | **Total** | | **7,052** |

`N` is the number of environments in this response, which may be smaller
than the created batch. Python returns one observation per result, so its
arrays omit the leading `N` dimension.

The layout is **columnar**: all environments' values for one field come before
the next field. For two environments, the observation block is arranged as:

```text
[bodies for env 0][bodies for env 1]
[players for env 0][players for env 1]
[body mask for env 0][body mask for env 1]
...then the remaining four columns in table order...
```

Line breaks are only for illustration; the bytes are consecutive. Identity
and result fields are columnar too: all IDs, then all seeds, then all seats,
and so on. Fields are not interleaved into complete records per environment.

Arrays use **row-major order**: the last index changes fastest. For example,
all 27 features of body slot 0 precede those of slot 1. This matches Go's
nested arrays and NumPy's default C order. Python can view contiguous bytes
as typed arrays without constructing an object for every number.

There are no per-array lengths or alignment bytes. `N` and the fixed schema
determine every boundary. Before exposing arrays, the decoder checks the
exact payload length, Boolean bytes, and finite numeric values (no NaN or
infinity).

## Close messages

The request has an empty payload. A successful response contains one `bool`
named `closed`, whose required value is `1`. Go then ends the session without
writing more protocol bytes. With `EnvClient`, leaving the `with` block
performs cleanup automatically.

## Error responses and recovery

An error response has type `0xff` and this payload:

| Order | Field | Type |
| ---: | --- | --- |
| 1 | Failed request type | `u8` |
| 2 | Environment ID, or `-1` for a general error | `i32` |
| 3 | Stable error kind | `string` |
| 4 | Human-readable message | `string` |

For example, stepping slot 0 before resetting it returns an error identifying
request type `0x03`, environment ID `0`, and kind `not-reset`. Reset that slot
before trying to step it.

| Error kind | Meaning |
| --- | --- |
| `malformed-binary` | Payload shape or scalar encoding is invalid |
| `unknown-command` | Message type is not a supported request |
| `invalid-state` | Operation is out of lifecycle order |
| `validation` | Configuration, IDs, or other constraints are invalid |
| `decode` | Learner actions are illegal for the current observation |
| `terminated` | Episode must be reset before stepping again |
| `seed-range-exhausted` | No scheduled assignments remain |
| `poisoned` | A previous stateful step failure requires a reset |

Environment failures can also report kinds such as `scenario`, `opponent`,
`engine`, or `observation`. Messages are valid UTF-8 and at most 1 MiB. Go
repairs invalid UTF-8 and shortens oversized messages at a UTF-8 boundary,
adding `... [truncated]`.

A complete, bounded frame lets Go send an error and stay synchronized with the
next request. The operation may still need recovery: a step failure after
opponents begin acting requires resetting every slot in that step, because
some game or opponent state may have changed. A failure while closing ends
the session after its error response.

These framing failures are fatal because the next message boundary cannot be
trusted:

- Wrong magic.
- Nonzero flags or reserved bytes.
- Payload length greater than 64 MiB.
- A truncated header or payload.

Go may send no structured response, exits nonzero, and may write a diagnostic
to standard error. Python must discard that process. Response-encoding or I/O
failures also end the session: retrying could leave Python and Go disagreeing
about whether a game turn already happened.

## Determinism and replay

- Each successful request produces exactly one matching response.
- Responses contain exactly the requested environments in ascending ID order.
- Duplicate or out-of-range IDs and illegal learner actions are rejected
  before any game advances.
- Reset prepares all replacements before publishing episodes or moving the
  cursor, so setup failure does not partially apply a reset.
- Decoders consume every payload byte; trailing bytes are malformed.
- Integer conversions are range-checked before conversion to narrower values
  or Go `int`.
- Map iteration and worker completion order never choose wire order or change
  assignments or game results.

Run configurations, checkpoints, and evaluation metadata record `env-v3` along
with the engine, `obs-v2` schema, generator, opponent, schedule, seed range,
and cursor. Preserve these identities when resuming. A matching transport
name alone does not establish replay compatibility.

Protocol tests compare seeded wire sessions against direct Go batch calls,
including observations, rewards, episode endings, turns, metadata, and order.
Repeating an identical session must produce identical response bytes.

## Payload size checks

These sizes exclude the 12-byte header. `N` counts environments; `E` counts
reset overrides.

| Message | Payload bytes | Example |
| --- | --- | --- |
| Create request | `36 + opponent UTF-8 bytes + schedule UTF-8 bytes` | 49 for `idle-v1` and `rotate` |
| Reset request | `8 + 4*N + 13*E` | 16 for `N=2, E=0` |
| Step request | `4 + 48*N` | 52 for `N=1` |
| Reset response | `12 + 7,066*N` | 7,078 for `N=1` |
| Step response | `4 + 7,077*N` | 7,081 for `N=1` |
| Close request | `0` | 0 |
| Close response | `1` | 1 |

For example, a step result needs 25 identity/result bytes plus 7,052
observation bytes: `25 + 7,052 = 7,077`. Create responses and error responses
contain variable-length strings, so their lengths depend on actual values.

The Go wire implementation is
[`mlenv/protocol_codec.go`](mlenv/protocol_codec.go); the Python counterpart is
[`python/orrery/protocol.py`](python/orrery/protocol.py).
