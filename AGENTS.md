# AGENTS.md

## Project mission

Orrery is a deterministic, rules-only game simulator written in Go, with a
Python reinforcement-learning trainer that plays it through the Go environment.

Keep the project focused:

- No UI is required.
- The core engine must not contain ML framework dependencies.
- The core engine accepts explicit game state; randomized ML scenario
  generation belongs in a separate environment package.
- Correct, replayable game behavior is more important than premature
  optimization.
- Explanatory comments are encouraged because the project is also being used
  to learn Go and understand the game rules.

## Sources of truth

Use these sources in this order:

1. Go tests and Python tests (`python/tests/`) define verified behavior already
   implemented.
2. Go source describes current behavior, including documented known gaps.
   Python source (`python/orrery/`, `python/scripts/`) describes current
   training, evaluation, and diagnostic behavior; for game rules, masks, and
   physics, Go is authoritative.
3. `README.md` is an overview and may lag behind detailed plans.

Do not treat existing behavior as correct merely because it compiles. Several
source comments intentionally document behavior that still needs to be
replaced.

## Repository layout

- `game/types.go`: public game types and configuration.
- `game/game.go`: construction, validation, turn coordination, launching,
  production, and orbital movement.
- `game/resolution.go`: fleet movement, collision detection, arrivals, and
  combat.
- `game/scoring.go`: scores, standings, and winner calculation.
- `game/*_test.go`: rules and regression tests.
- `cmd/sim/main.go`: small deterministic CLI demonstration.
- `python/orrery/`: training, PPO, the model, the Go environment client, and
  checkpoint evaluation.
- `python/scripts/`: diagnostics and other command-line tools, such as
  `diagnose.py` and `validate_checkpoint.py`.
- `python/tests/`: Python tests for training, evaluation, and diagnostics.
- `docs/agents/`: task-specific agent rules that AGENTS.md points to.

Keep rule logic in `game`. The CLI should call the engine rather than duplicate
its rules.

## Task-specific rules

Read the matching file before starting that kind of task:

- Changing `game/` or engine behaviour: `docs/agents/engine-rules.md`
  (invariants, target turn order, the ML engine gate, required regression tests).
- Branching, pushing, PR summaries, or code reviews: `docs/agents/pull-requests.md`.
- Writing docs, reports, or learner-facing diagnostics: `docs/agents/documentation.md`.

## Required working practices

Before changing files:

1. Read `git status --short`.
2. Inspect relevant tests and known-gap comments.
3. Preserve unrelated user changes.
4. Confirm whether the request is for review, diagnosis, planning, or
   implementation.
5. Never delete a file without explicit user approval first — including a file
   you created or that a subagent created earlier in the same session.
   Surface what you'd delete and why, and wait for confirmation before
   removing it.

For review or diagnosis requests, do not modify files unless the user also asks
for a fix.

For implementation requests:

- Make the smallest coherent change that completes the requested step.
- Use `gofmt` on touched Go files.
- Add or update tests with every rule change.
- Explain why non-obvious code exists, especially geometry, turn ordering,
  probability, and combat math.
- Document struct fields whose meaning, units, zero value, ownership, lifecycle,
  or replay role is not obvious. This applies to important private structs as
  well as exported API types.
- Put explanatory comments at multi-stage code paths where ordering, atomicity,
  cleanup, or failure handling matters. State the invariant each stage protects
  close to the code that enforces it.
- Prefer table-driven tests for multiple rule cases.
- Prefer modern Go syntax supported by the version in `go.mod`.
- Avoid nondeterministic map iteration when order can affect results.
- Do not introduce a new dependency unless it materially simplifies the task.
- Commit messages use `type(scope): description`, e.g.
  `test(core-model): cover Config field rejection in New()`. `type` is one of
  `feat`, `fix`, `refactor`, `chore`, `docs`, or `test`; `scope` is the PR or
  area being touched (e.g. `core-model`, `combat`, `collision`).
- Choosing between `docs` and `chore`: see `docs/agents/pull-requests.md`.
- Commit each logical change as it lands rather than batching a whole task
  into one commit. In practice that is a separate commit for the engine
  change, for its tests, and for the docs update, plus one per review fix.
- Do not add a `Co-Authored-By:` trailer.

## Pull requests

One PR at a time, each requiring explicit approval before the next is started.
Branch from a freshly fetched `main`, keep the branch to one PR's scope, push
only when the user asks, and never open, merge, or close a PR unasked. The full
steps, PR summary format, and review expectations are in
`docs/agents/pull-requests.md`.

## Determinism requirements

- Prefer explicit seeds over time-seeded randomness.
- Do not share one mutable random-number generator across concurrently running
  games.
- A scenario seed and generator version must always produce the same state.
- Sort map-derived keys before they affect actions, state, logs, hashes, or
  tie-breaking.
- Record schema, generator, engine, opponent, and checkpoint versions needed
  to replay a run.

Concurrency must not change a match result.

## Verification

For Go source changes, normally run:

```bash
gofmt -w <touched-go-files>
go test ./...
go vet ./...
```

Also run this when CLI-visible behavior changes:

```bash
go run ./cmd/sim
```

Use `git diff --check` for every change.

Do not claim tests passed unless they were actually run. If sandbox or
dependency restrictions prevent a check, state exactly which check was not
run.

## Definition of done

A change is complete when:

- The requested behavior is implemented without unrelated edits.
- Relevant invariants and turn order remain correct.
- Focused regression tests exist.
- Formatting and applicable test commands pass.
- Documentation matches the implemented behavior.
- Determinism and replayability are preserved.
- The final response summarizes the result, tests, and any remaining known
  gaps.
