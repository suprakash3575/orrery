.DEFAULT_GOAL := help

GO ?= go
GOFMT ?= gofmt
UV ?= uv
ARENA_ARGS ?=
TRAIN_ARGS ?=
# Go threads for the env process during training. On a Vast.ai 5090 container
# limited to 46 CPUs, 4-player smoke runs measured about 2,900-3,000
# env-steps/s at 12 and 16 against 2,291 at 32 and 2,234 at Go's own default:
# more Go threads made each turn slower. Re-measure on a machine with a very
# different CPU limit; `make train-4p GOMAXPROCS=16` overrides it per run.
GOMAXPROCS ?= 12
ENV_BINARY ?= .build/orrery-env
ARENA_LINUX_BINARY ?= .build/linux-amd64/arena
TURNS ?= 1
BATCH ?= 1
TABLE ?=
DIAG_ARGS ?=
# diagnose reads one rl-v2 training run, named by DIAG_RUN. It refuses a
# non-empty output folder, so each run writes to its own timestamped folder
# unless DIAG_OUT names one.
DIAG_RUN ?=
DIAG_OUT ?= runs/diagnosis-$(shell date +%Y%m%d-%H%M%S)
# Experiment sessions (run_experiments.py). EXP is the experiment file and
# EXP_OUT the session folder; running the same exp-run again resumes it.
# EXP_ONLY limits a run or preflight to one experiment by name.
EXP ?=
EXP_OUT ?=
EXP_ONLY ?=
EXP_SMOKE_OUT ?= runs/stage1-smoke-$(shell date +%Y%m%d-%H%M%S)
EXP_SMOKE_GPU_OUT ?= runs/stage1-smoke-gpu-$(shell date +%Y%m%d-%H%M%S)
BOT_ARGS ?=

.PHONY: help sim arena arena-help arena-linux env smoke train train-4p random-baseline diagnose control-center exp-preflight exp-smoke exp-smoke-gpu exp-run exp-status exp-analyze botcheck python-test format-check build vet test race check

help:
	@printf '%s\n' \
		'Usage: make <target>' \
		'' \
		'Run Orrery:' \
		'  sim           Run the deterministic engine demonstration' \
		'  arena         Run scripted-agent matches (pass ARENA_ARGS="...")' \
		'  arena-help    Show every arena command-line option' \
		'  env           Start the long-lived env-v3 binary process' \
		'  smoke         Run eight complete episodes through the env process' \
		'  train         Start two-player rl-v2 CUDA training with GOMAXPROCS=12 (TRAIN_ARGS="...")' \
		'  train-4p      Start four-player rl-v2 CUDA training with GOMAXPROCS=12 (TRAIN_ARGS="...")' \
		'  random-baseline Run the 100-game random-controller acceptance baseline' \
		'  diagnose      Chart an rl-v2 run and play its latest checkpoint (DIAG_RUN=run DIAG_ARGS="--seeds 25")' \
		'  control-center Browse runs and reports; launch training and evaluations' \
		'' \
		'Run experiment sessions:' \
		'  exp-preflight Check an experiment file and count its work; run nothing (EXP=file [EXP_ONLY=name])' \
		'  exp-smoke     Run bot, probe and analysis jobs on 2 maps on the CPU (no training)' \
		'  exp-smoke-gpu Run every job type, training included, on 2 maps and 2 updates (Vast.ai)' \
		'  exp-run       Run or resume a session (EXP=file EXP_OUT=folder [EXP_ONLY=name])' \
		'  exp-status    Show each job: done, failed, unfinished or waiting (EXP_OUT=folder)' \
		'  exp-analyze   Write analysis.md and analysis.json into a session (EXP_OUT=folder)' \
		'  botcheck      Run opponentcheck by hand with any flags (BOT_ARGS="...")' \
		'' \
		'Package Orrery:' \
		'  arena-linux   Build a static linux/amd64 arena binary into .build/linux-amd64/' \
		'' \
		'Verify the repository:' \
		'  python-test   Run the Python client and controller tests' \
		'  format-check  List unformatted Go files and fail if any exist' \
		'  build         Build every Go package' \
		'  vet           Run Go static analysis' \
		'  test          Run every Go test' \
		'  race          Run every Go test with the race detector' \
		'  check         Run format-check, build, vet, and test'

sim:
	$(GO) run ./cmd/sim

arena:
	$(GO) run ./cmd/arena $(ARENA_ARGS)

arena-help:
	$(GO) run ./cmd/arena -h

# A static Linux binary for container images and VM disks, built from any
# host OS. CGO_ENABLED=0 drops the C-library dependency, so the binary runs in
# a FROM scratch image with no other files. GOARCH=amd64 matches the x86-64
# sandbox host. -trimpath strips this machine's file paths from the binary.
arena-linux:
	@mkdir -p $(dir $(ARENA_LINUX_BINARY))
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 $(GO) build -trimpath -o $(ARENA_LINUX_BINARY) ./cmd/arena

env:
	@printf '%s\n' 'Building the Orrery env-v2 executable...' >&2
	@mkdir -p $(dir $(ENV_BINARY))
	$(GO) build -o $(ENV_BINARY) ./cmd/env
	@printf '%s\n' \
		'Starting the Orrery env-v2 server...' \
		'Waiting for binary env-v2 frames on stdin; no prompt will appear.' \
		'Commands: create, reset, step, close. Press Ctrl-C to stop.' >&2
	$(ENV_BINARY)

smoke:
	cd python && $(UV) run python -m scripts.smoke

train:
	cd python && GOMAXPROCS=$(GOMAXPROCS) $(UV) run python -m orrery.train --config ../configs/rl-v2-2p.toml $(TRAIN_ARGS)

train-4p:
	cd python && GOMAXPROCS=$(GOMAXPROCS) $(UV) run python -m orrery.train --config ../configs/rl-v2-4p.toml $(TRAIN_ARGS)

random-baseline:
	cd python && $(UV) run pytest -s -m acceptance

# abspath resolves DIAG_RUN and DIAG_OUT from the repository root before the
# recipe moves into python/, so relative and absolute folders both work.
# DIAG_ARGS="--no-games" keeps only the training charts: no GPU or Go needed.
diagnose:
	@test -n "$(DIAG_RUN)" || { printf '%s\n' 'diagnose needs DIAG_RUN=<rl-v2 run directory>' >&2; exit 2; }
	cd python && $(UV) run --group analysis python -m scripts.diagnose --run-dir $(abspath $(DIAG_RUN)) --out $(abspath $(DIAG_OUT)) $(DIAG_ARGS)

control-center:
	cd python && $(UV) run --group analysis python -m scripts.control_center

# The experiment targets resolve EXP and EXP_OUT from the repository root with
# abspath before moving into python/, as diagnose does. The runner sets each
# job's own GOMAXPROCS, so none is passed here.
exp-preflight:
	@test -n "$(EXP)" || { printf '%s\n' 'exp-preflight needs EXP=<experiment file>' >&2; exit 2; }
	cd python && $(UV) run python -m scripts.run_experiments preflight $(abspath $(EXP)) $(if $(EXP_ONLY),--only $(EXP_ONLY))

# exp-smoke has no training job: an rl-v2 update is always full size, too
# slow for a laptop CPU. exp-smoke-gpu adds it, for the first step on Vast.ai.
exp-smoke:
	cd python && $(UV) run python -m scripts.run_experiments run $(abspath experiments/stage1-smoke.toml) --out $(abspath $(EXP_SMOKE_OUT))
	cd python && $(UV) run python -m scripts.analyze_experiments $(abspath $(EXP_SMOKE_OUT))

exp-smoke-gpu:
	cd python && $(UV) run python -m scripts.run_experiments run $(abspath experiments/stage1-smoke-gpu.toml) --out $(abspath $(EXP_SMOKE_GPU_OUT))
	cd python && $(UV) run python -m scripts.analyze_experiments $(abspath $(EXP_SMOKE_GPU_OUT))

exp-run:
	@test -n "$(EXP)" -a -n "$(EXP_OUT)" || { printf '%s\n' 'exp-run needs EXP=<experiment file> EXP_OUT=<session folder>' >&2; exit 2; }
	cd python && $(UV) run python -m scripts.run_experiments run $(abspath $(EXP)) --out $(abspath $(EXP_OUT)) $(if $(EXP_ONLY),--only $(EXP_ONLY))

exp-status:
	@test -n "$(EXP_OUT)" || { printf '%s\n' 'exp-status needs EXP_OUT=<session folder>' >&2; exit 2; }
	cd python && $(UV) run python -m scripts.run_experiments status $(abspath $(EXP_OUT))

exp-analyze:
	@test -n "$(EXP_OUT)" || { printf '%s\n' 'exp-analyze needs EXP_OUT=<session folder>' >&2; exit 2; }
	cd python && $(UV) run python -m scripts.analyze_experiments $(abspath $(EXP_OUT))

botcheck:
	$(GO) run ./cmd/opponentcheck $(BOT_ARGS)

python-test:
	cd python && $(UV) run pytest

format-check:
	@files="$$($(GOFMT) -l .)"; \
	if [ -n "$$files" ]; then \
		printf '%s\n' "$$files"; \
		exit 1; \
	fi

build:
	$(GO) build ./...

vet:
	$(GO) vet ./...

test:
	$(GO) test ./... -count=1

race:
	$(GO) test -race ./...

check: format-check build vet test python-test
