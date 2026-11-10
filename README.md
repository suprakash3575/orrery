# Orrery

Orrery is a turn-based space strategy game with no graphics: the game is
just a set of rules in code. This repository contains a deterministic game
engine written in Go and a Python reinforcement-learning (RL) trainer that
teaches a neural network to play it.

"Deterministic" means the same starting map, seed, and moves always produce
exactly the same game, so any match can be replayed and any test result
reproduced.

This is a personal learning project for understanding Go, RL, and transformer
models. It is a work in progress: commands, file formats, and configurations
may change without notice.

## The game in brief

The board is 100×100 with a sun in the middle. Planets orbit the sun or sit
still, and comets cross the board on elliptical paths. Two to four players
start with a few planets each. Every turn, each owned planet produces ships,
and players send fleets from planets they own to capture neutral or enemy
planets. A fleet that flies into the sun or off the board is destroyed.

A match ends after 500 turns, or earlier when only one player still owns
anything. The player with the most ships, counting those on planets, on comets,
and in flight, wins.

## What is in the repository

| Path | What it holds |
| --- | --- |
| [`game/`](game) | The rules engine: setup, launches, production, orbits, fleet movement, collisions, combat, and scoring. |
| [`scenario/`](scenario) | Seeded map generators, so a seed always produces the same map. |
| [`agent/`](agent) | Hand-written (scripted) players: random and heuristic opponents. |
| [`arena/`](arena), [`cmd/arena`](cmd/arena) | Runs matches between scripted players and summarizes the results. |
| [`mlenv/`](mlenv), [`cmd/env`](cmd/env) | The RL environment: turns game state into model inputs and runs many games at once behind a binary protocol. |
| [`cmd/sim`](cmd/sim) | A small demonstration that plays one fixed game. |
| [`python/orrery/`](python/orrery) | The Python side: environment client, transformer policy, PPO training, and evaluation. |
| [`python/scripts/`](python/scripts) | Diagnostics, run comparisons, and the control-center menu. |
| [`configs/`](configs) | Training configurations for two-player and four-player runs. |

The Go engine has no machine-learning dependencies. Python talks to it by
starting the Go environment as a separate process and exchanging messages
through that process's standard input and output.

## Requirements

- Go 1.27.3, as declared in [`go.mod`](go.mod).
- `make` for the convenience commands below.
- Python 3.11 or newer for the RL client and trainer.
- [`uv`](https://docs.astral.sh/uv/) to create the isolated Python environment
  and install its locked dependencies.
- A CUDA GPU for training. Everything else runs on a CPU.

Run the following commands from the repository root. See every available
command with:

```bash
make help
```

## Quick start

### Play one deterministic game

```bash
make sim
```

This runs one complete game from a fixed setup and seed. It prints turn events
and the final result, and produces the same output on repeated runs. It uses
the Go rules engine directly, not the RL environment.

### Run scripted players against each other

```bash
make arena
```

The default arena runs paired matches between the built-in heuristic and
random players. Each map is played twice with the players swapped, so the
summary can separate player strength from starting position.

To use static generated maps explicitly:

```bash
make arena ARENA_ARGS="-scenario generator-v1 -scenarios 10 -seed 1"
```

For generated maps with orbiting planets, the sun hazard, and scheduled
comets:

```bash
make arena ARENA_ARGS="-scenario generator-v2 -scenarios 10 -seed 1"
```

See every arena option with:

```bash
make arena-help
```

### Start the RL environment process

```bash
make env
```

This builds `cmd/env` as `.build/orrery-env`, then starts it as a
long-lived process speaking the `env-v3` protocol. Each request and response
is one binary frame: a small fixed header followed by a payload. The process
does not show a prompt or play a game by itself; a client must send `create`,
`reset`, `step`, and `close` frames through standard input. The Make target
prints startup guidance to standard error, so it is clear when the process is
waiting. Press Ctrl-C to stop an idle process.

`--players` chooses the maps for the whole process: `--players 2` (the
default) serves `generator-v2`, and `--players 4` serves `generator-v2-4p`.
Both place the same planets for a seed; in a two-player game the other two
planets of the home group stay neutral. Any other count exits with status 2
before reading any input.

See [`mlenv/README.md`](mlenv/README.md) for how the environment works.

### Check the Python-to-Go connection

```bash
make smoke
```

On the first run, `uv` creates `python/.venv` and installs the versions locked
in `python/uv.lock`. The script then starts `cmd/env`, creates eight
environments, resets them, sends "no launch" moves until every game finishes,
closes the process, and reports the total number of environment steps. It
checks the connection and the game lifecycle; it does not train or evaluate a
model.

## Train a policy

```bash
make train      # two players
make train-4p   # four players
```

Training uses PPO (Proximal Policy Optimization, a common RL algorithm) with a
transformer policy. The two targets select
[`configs/rl-v2-2p.toml`](configs/rl-v2-2p.toml) and
[`configs/rl-v2-4p.toml`](configs/rl-v2-4p.toml). Each PPO update collects 64
consecutive turns from each of 512 games, 32,768 game steps in all.

Each run writes its configuration, `metrics.csv`, `timings.csv`, checkpoints,
and validation results to a directory under `runs/`, which Git ignores.
Training regularly plays validation games against the built-in `heuristic-v1`
opponent. The first checkpoint that wins often enough is copied into
`runs/archive/` and training stops.

Pass extra options through `TRAIN_ARGS`. For example, run a single update as a
quick check, or resume from a saved checkpoint:

```bash
make train TRAIN_ARGS="--max-updates 1"
make train TRAIN_ARGS="--resume ../runs/<run>/checkpoints/<checkpoint>.pt"
```

If a run has not passed its first validation check by 25 million steps,
training pauses. Resuming past that point also needs a written reason, such as
`--review-reason 'validation results are still improving'`.
### Inspect and evaluate runs

```bash
make control-center
```

This opens a terminal menu to start or resume training, validate a checkpoint,
compare runs, and read reports. To chart a run and play its latest checkpoint
on validation maps:

```bash
make diagnose DIAG_RUN=runs/<run> DIAG_OUT=runs/<report-dir>
```

The final held-out test of an archived checkpoint is a one-time step, because
each pool of test maps may be used only once:

```bash
cd python
uv run python -m orrery.evaluate \
    --archive ../runs/archive/<identifier> \
    --spend-eval eval-1
```

It plays 2,000 games against `heuristic-v1` and records every result inside the
archive. `runs/evaluation-pools-v1.json` records which pools have been used;
keep it with your runs, because deleting it would let a used pool be offered
again.

## Run the checks

```bash
make check
```

`make check` verifies formatting, builds every Go package, runs `go vet`, and
runs every Go test plus the fast Python suite. The Python tests can also be run
separately:

```bash
make python-test
```

Two longer checks are kept separate:

```bash
make random-baseline   # 100 games: random controller vs heuristic-v1
make race              # Go tests with the race detector
```

## Further reading

- [arena/README.md](arena/README.md): running and reading scripted matches.
- [mlenv/README.md](mlenv/README.md): how the RL environment works.

## License

No licence has been chosen yet. Until one is added, the code can be viewed but
is not licensed for reuse.
