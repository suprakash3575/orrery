# Orrery arena

The arena runs headless Orrery matches between agents. It builds on the
authoritative `game` engine, gives every agent an isolated state view, runs
paired seat-swapped matches concurrently, and can save deterministic NDJSON
records for replay.

## Run the CLI

From the repository root:

```bash
go run ./cmd/arena \
  -scenarios 50 \
  -workers 8 \
  -seed 100000 \
  -p0 "heuristic-v1:aggression=1.2,economy=1,efficiency=1,defense=0.1" \
  -p1 "random-v1:launch_probability=0.3,min_fraction=0.2,max_fraction=0.8" \
  -record runs/heuristic-vs-random.ndjson
```

The command creates missing parent directories and then creates or overwrites
the named file after all matches finish. Omit `-record` to print the summary
without writing match records.

Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-scenarios` | `10` | Number of scenario seeds. Each seed produces two games. |
| `-workers` | `GOMAXPROCS` | Maximum number of matches running concurrently. |
| `-seed` | `1` | First scenario seed; scenario `i` uses `seed+i`. |
| `-scenario` | `fixed-v1` | Which map factory builds each scenario. |
| `-p0` | `heuristic-v1:...` | Contestant `p0` agent specification. |
| `-p1` | `random-v1:...` | Contestant `p1` agent specification. |
| `-record` | empty | Optional NDJSON result path. |

The summary reports overall and seat-specific wins, losses, draws and rates,
average finishing turn, elimination and turn-limit counts, and agent, engine,
other, and cleanup failure counts.

## Scenarios

`-scenario` chooses the map factory. Each name is versioned, because a seed
only identifies a board together with the factory that built it — the same seed
means a different map under a different factory.

| Name | Map |
| --- | --- |
| `fixed-v1` | One explicit six-planet layout, identical for every seed. The seed still drives rule-level randomness such as scheduled comet spawns. |
| `generator-v1` | A seeded random map: 8, 10, or 12 planets with 2-fold rotational symmetry about the sun, the two homes diagonally opposite, and every neutral planet mirrored by a twin of equal production. |
| `generator-v2` | A seeded random map: 5–10 groups of four symmetric planets, including at least three static groups and one orbiting group. The sun hazard and scheduled comets are enabled. |

`generator-v1` builds the static two-player variant used by the reinforcement
learning environment: planets do not orbit, no comets ever spawn, and the sun
hazard is disabled. Homes start with 10 ships at production 5; a neutral planet
starts with ten ships per point of its production, so a more valuable planet is
also better defended.

`generator-v2` chooses one group for the two homes: player 0 gets its first
planet and player 1 its fourth, with 10 ships each and the group's rolled
production. Other planets remain neutral. The first three static groups draw
their shared garrisons as the lower of two rolls from 5–99; later groups draw
from 5–30. Planet surfaces have at least seven empty units between them,
including throughout a rotating-versus-static encounter; planets leave ten
empty units around the sun. This arena option does not change the static
`generator-v1` used by the RL environment.

`generator-v2-4p` (`scenario.GeneratorV2{FourPlayer: true}`) builds the same
map for a seed, but gives all four planets of the home group to players 0–3,
with 10 ships each. It is not yet selectable from `cmd/arena`, which schedules
two seats per match.

All three factories are deterministic. Repeating a command with the same flags
reproduces the same matches, and `-workers` never changes a result:

```bash
go run ./cmd/arena -scenario generator-v1 -scenarios 50 -workers 8 -seed 100000 \
  -p0 "heuristic-v1:aggression=1.2,economy=1,efficiency=1,defense=0.1" \
  -p1 "random-v1:launch_probability=0.3,min_fraction=0.2,max_fraction=0.8"
```

To run the full-rules generated variant:

```bash
go run ./cmd/arena -scenario generator-v2 -scenarios 10 -seed 1
```

## The seed budget

The reinforcement-learning work divides the seed space into seven
non-overlapping intervals serving three purposes so that a reported result
cannot have been measured on a map the policy was already shaped by. The
authoritative declarations live in `scenario.StandardRanges()`.

| Range | Seeds | Used for |
| --- | --- | --- |
| Training | `[0, 1000000)` | Sampled freely and repeatedly during a run. |
| Validation | `[1000000, 1010000)` | Deciding when a run is finished. Read often, so choosing a checkpoint by its validation score makes these seeds part of the selection. |
| Evaluation `eval-1` | `[1010000, 1011000)` | First one-shot evaluation of an already-frozen checkpoint. |
| Evaluation `eval-2` | `[1011000, 1012000)` | Second protected one-shot pool, used only after `eval-1` is spent. |
| Evaluation `eval-3` | `[1012000, 1013000)` | Third protected one-shot pool. |
| Evaluation `eval-4` | `[1013000, 1014000)` | Fourth protected one-shot pool. |
| Evaluation `eval-5` | `[1014000, 1015000)` | Fifth protected one-shot pool. |

Each evaluation pool may be spent at most once, in ascending order, on a
checkpoint frozen before that pool is opened. A spent pool is never reused for
debugging, tuning, or another evaluation; otherwise it is no longer held out.

`-seed` picks where a run starts and scenario `i` uses `seed+i`, so a run stays
inside a range by starting at its first seed and keeping `-scenarios` below its
width. Each seed is played twice, once per seat, so `-scenarios 500` covers
1,000 games:

```bash
# 100 training seeds
go run ./cmd/arena -scenario generator-v1 -seed 0       -scenarios 100

# 100 validation seeds
go run ./cmd/arena -scenario generator-v1 -seed 1000000 -scenarios 100
```

**The arena does not enforce any of this.** `-seed 999950 -scenarios 100` walks
out of training and into validation without complaint. `scenario.Disjoint` and
`Ranges.Validate` only check that a declared budget is internally coherent —
no range inverted, empty, or overlapping another — not what a given run
actually draws. Enforcing the boundary at run time belongs to the environment
process that serves the learner, which takes both an explicit seed range and a
schedule. The finite `paired` schedule fails after playing both seats for every
seed and must be used for validation and evaluation. The training-only
`train-v1` schedule instead wraps to a new epoch and deliberately reuses its
configured range forever; pointing it at held-out seeds would recycle them
without an exhaustion error.

For scripted agents the distinction is mostly bookkeeping: no policy learns
from a `heuristic-v1` against `random-v1` comparison, so running one over
evaluation seeds spends nothing. The budget starts to bind once a trained
checkpoint is the thing being measured.

## Agent specifications

Supported in-process agents are:

```text
idle-v1
random-v1:launch_probability=0.3,min_fraction=0.2,max_fraction=0.8
heuristic-v1:aggression=1.2,economy=1,efficiency=1,defense=0.1
```

`random-v1` requires:

- `launch_probability`, `min_fraction`, and `max_fraction` in `[0,1]`.
- `min_fraction <= max_fraction`.

`heuristic-v1` requires:

- Non-negative `aggression`, `economy`, and `efficiency` weights.
- `defense` in `[0,1]`.

Every configured agent requires its complete parameter set. Parsing rejects
unknown or duplicate keys, missing values, malformed numbers, NaN, and
infinity. Input parameter order and equivalent numeric spellings are
normalized into one canonical agent identity.

## Pairing and determinism

Every scenario is played twice:

```text
leg 0: p0 in player seat 0, p1 in player seat 1
leg 1: p1 in player seat 0, p0 in player seat 1
```

Both legs use the same scenario seed. A contestant's private agent seed is
derived from the scenario seed, contestant ID, and canonical agent
specification. It does not include seat or leg, so swapping seats does not
change that contestant's random stream.

Each match owns its game, agents, and random-number generators. Workers write
only to predetermined result indexes; aggregation and NDJSON encoding happen
sequentially after the workers finish. With the same code, inputs, and Go
runtime behavior, changing `-workers` does not change match results or NDJSON
bytes.

## Records and replay

Each NDJSON line is one versioned `arena.MatchResult` in deterministic schedule
order. When recording is enabled, the result includes the accepted action
batch for every completed turn. These batches—not a second agent run—are the
authoritative replay input, so a recorded match remains replayable even if its
original agent was nondeterministic or unavailable.

Replay one decoded result in Go:

```go
file, err := os.Open("runs/heuristic-vs-random.ndjson")
if err != nil {
	return err
}
defer file.Close()

decoder := json.NewDecoder(file)
var result arena.MatchResult
if err := decoder.Decode(&result); err != nil {
	return err
}
if result.Actions == nil {
	return errors.New("result has no action record")
}

replayed, err := arena.Replay(
	context.Background(),
	arena.FixedScenarioFactory{},
	*result.Actions,
)
if err != nil {
	return err
}
```

`Replay` validates the record schema, scenario identity, turn sequence, and
every action batch through the game engine.

## Go API

The main boundaries are:

- `ScenarioFactory`: constructs an independent seeded game.
- `SchedulePairedMatches`: creates the deterministic seat-swapped job list.
- `RunMatch`: runs one match and returns a complete result or structured
  failure.
- `RunExperiment`: executes a paired schedule through an indexed worker pool.
- `Summarize`: validates and aggregates ordered results.
- `WriteNDJSON`: serializes ordered match results sequentially.
- `Replay`: reconstructs a game from a scenario seed and accepted actions.

`RunMatch` creates an independent pre-turn view for every seat. All agents
decide from the same facts, their decisions are combined in ascending player
ID order, and the engine receives exactly one simultaneous action batch.
Agents that implement `io.Closer` are closed on completion, failure, or
cancellation.

## How agents and the arena play a match

An agent is the decision-maker for one player seat. In the current paired
arena, each match has one agent for player 0 and one for player 1. The same
two contestants then play a second, seat-swapped leg.

For each match:

1. The arena creates a fresh game from the scenario seed and creates one
   agent instance per player seat. Each instance receives its match metadata,
   assigned player ID, and private deterministic seed.
2. At the start of each game turn, the arena deep-copies the current game into
   a separate `agent.View` for every agent. Each agent sees the same pre-turn
   facts but cannot change the live game or another agent's view.
3. The arena calls `Decide` once per agent with a `TurnRequest` containing the
   match ID, that seat's player ID, and its view. The agent returns a
   `Decision` containing zero or more launch actions.
4. A launch action names the source planet or comet, the number of ships, and
   the heading angle. Every action's `PlayerID` must equal the
   `TurnRequest.PlayerID` the seat was assigned. The arena checks this before
   combining decisions, because the engine trusts each action's own
   `PlayerID`; a seat that submits an action for another player fails the match
   as an `agent-decision` failure attributed to that seat.
5. The arena combines the two decisions, in ascending player-ID order, into a
   single action batch and passes it to `game.AdvanceTurn`. Neither agent gets
   a full game turn before the other: their choices are submitted against the
   same snapshot and resolved simultaneously.
6. The engine expires comets, validates the full batch, spawns any scheduled
   comets, applies launches, produces ships, moves fleets, moves planets and
   comets, queues collisions, resolves all queued combat, and increments the
   turn. Validation happens before spawning so retrying an invalid turn cannot
   consume random state or duplicate a comet wave. One invalid action rejects
   the whole batch; the match then fails as an `engine` failure attributed to
   the seat whose action the engine's `game.ActionError` names.
7. The arena repeats until the engine reaches elimination or its turn limit,
   then records the final state, standings, winners, termination reason, and,
   when enabled, the accepted action batch for every completed turn.

## Current first-release limits

- Exactly two contestants per experiment.
- One fixed, symmetric scenario layout using normal scheduled comet spawning.
- In-process scripted agents only.
- No external or agentic-AI subprocess protocol yet.
- No generated maps, manifests, tournaments, or parameter sweeps yet.

These later capabilities are intended to reuse the existing agent, scenario,
match, replay, and experiment boundaries.

## Verification

```bash
go test ./...
go test -race ./...
go vet ./...
```
