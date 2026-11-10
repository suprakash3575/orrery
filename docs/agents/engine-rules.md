# Engine rules

Read this before changing `game/` or anything that depends on engine behavior.
It extends [AGENTS.md](../../AGENTS.md).

## Go engine invariants

Preserve these invariants unless a versioned rules change explicitly replaces
them. This list is current behavior, verified against `game/`, not the target
state — for what's still pending, see "Rules-engine gate before ML" below and
`IMPLEMENTATION-PLAN.md`'s gap list.

- Player IDs are `0` through `3`; neutral is `-1`.
- Planet production is in `[1, 5]`.
- Planet radius is derived as `1 + ln(production)`.
- Comets use radius `1` and production `1`.
- Actions for one turn are validated completely before any garrison is
  mutated.
- Multiple actions are simultaneous.
- Fleets travel in straight lines at speed derived from fleet size.
- Production occurs before fleet movement and arrival combat.
- Collision tests use the complete movement segment, not only its endpoint.
- A segment is tested against all four hazards together — board edge, sun,
  planets, comets — and the earliest intersection wins.
- A newly launched outward fleet does not collide with its own source. Two
  things are needed for this, not one: it spawns at `radius + spawnEpsilon`,
  strictly outside the source, and collision detection skips
  `Fleet.FromPlanetID` for the single turn named by `Fleet.LaunchTurn` —
  bodies move before fleets do, so a moving source outruns any fixed margin.
  The exclusion lasts exactly that one turn.
- Arrivals are collected during movement and resolved only once every fleet
  has moved, so combat never depends on fleet order.
- Combat groups arriving fleets by owner, independent of slice or map
  iteration order, in the two-stage form: attackers fight each other first
  with the garrison excluded, then the survivor reinforces or attacks it.
- A body that moves over a fleet drags it into that turn's combat, on the same
  footing as a fleet that flew in. A fleet is claimed by at most one body per
  turn: earliest intersection wins, ties fall to the lower body ID.
- Planet orbit rates are engine-assigned, not caller-supplied. `New()` rolls
  `Orbit.RadiansTurn` from `Config.AngularVelocity` over `[0.5x, 1.5x)` with a
  random direction, in ascending planet-ID order; a nil `Orbit` draws no
  randomness. The formula is part of the replay contract — changing it
  invalidates existing seeded replays.
- All state affecting a replay must be deterministic from the initial state,
  configuration, seed, and submitted actions.

The following are **not yet true** of `game/` — they are required before the
engine gate passes (see below) and must not be assumed as already-enforced
behavior when reading or reviewing code:

- Comets expire, spawn on a schedule, and produce ships. Expiry currently
  happens mid-turn rather than at the start of one, there is no spawn schedule
  at all, and `produce()` credits only planets.
- Comet motion covers `Config.CometSpeed` units per turn. Comets currently
  advance by a fixed angle instead, which is not a constant distance along an
  ellipse.
- The turn runs in the order under "Target turn order" below. `AdvanceTurn`
  currently omits the comet expire and spawn phases entirely.
- Scoring counts ships on owned planets, active fleets, and comets according
  to the rules. `Standings()` currently excludes comet ships, and
  `Finished()` only checks the turn limit — no elimination check yet.

When floating-point comparisons need tolerance, define and document the
tolerance once. Do not scatter unexplained epsilon values through the engine.

## Target turn order

The completed engine turn order, as `AdvanceTurn` runs it, is:

1. Expire any comet that has already finished its path (a safeguard for
   comets placed in that state by hand; normal expiry happens in step 8).
2. Validate every launch before anything is mutated.
3. Spawn scheduled comets. This runs after validation so a rejected turn,
   retried, cannot spawn a second wave.
4. Apply all launches.
5. Produce ships on every owned planet and comet.
6. Move orbiting planets and comets, recording each body's path for the turn.
7. Move fleets, testing each against every body's path in relative motion so
   fleet and body are compared at the same instant, and queue arrivals.
   Comets spawned this turn take no part in collision until the next turn.
8. Expire comets that stepped past their last point this turn, and discard
   arrivals queued for them, so their ships cannot affect combat or scoring.
9. Resolve all queued combat.
10. Increment the turn.

Changes that affect this sequence require focused turn-order tests.

## Rules-engine gate before ML

No ML training rollout or meaningful checkpoint may be created until the
following V1 engine gate passes:

1. Continuous planet-arrival collision.
2. Body-first hit priority for paths crossing several hazards: any planet or
   comet the path touches claims the fleet ahead of the sun and the board
   edge, and among several touched bodies the lowest body ID wins.
3. Outbound launch-source collision exclusion.
4. Two-stage combat:
   - Arriving owners fight each other first. Only the two largest forces
     count: the largest survives with its size minus the second largest's,
     smaller forces are destroyed, and a tie between the top two leaves no
     survivor.
   - The surviving force then reinforces or attacks the garrison.
5. Elimination termination in addition to the turn limit.
6. A real sun-hazard switch that is disabled for ML V1.
7. Collision-prediction parity between the engine and incoming-fleet
   observations on boards where no body moves, which is every ML V1 board
   (`FirstHit` treats bodies as stationary).

Do not work around an engine defect in Python observations. Fix or expose the
authoritative Go behavior first.

## Engine-gate regression coverage

Minimum focused regression coverage for the engine gate includes:

- A fast fleet tunneling through a small planet.
- One movement segment intersecting two planets, selecting the first.
- A new outward fleet not colliding with its source.
- Three-force combat proving the two-stage rule.
- Friendly reinforcement.
- Exact attacker ties.
- Elimination with planets and fleets considered.
- A fleet crossing the sun when the hazard is enabled.
- The same crossing surviving when the hazard is disabled.
- Engine/observation first-target and arrival-turn parity over randomized legal
  static states.
