.DEFAULT_GOAL := help

GO ?= go
GOFMT ?= gofmt
UV ?= uv
ARENA_ARGS ?=
TRAIN_ARGS ?=
GOMAXPROCS ?= 32
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

.PHONY: help sim arena arena-help arena-linux env smoke train train-4p random-baseline diagnose control-center python-test format-check build vet test race check

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
		'  train         Start two-player rl-v2 CUDA training with GOMAXPROCS=32 (TRAIN_ARGS="...")' \
		'  train-4p      Start four-player rl-v2 CUDA training with GOMAXPROCS=32 (TRAIN_ARGS="...")' \
		'  random-baseline Run the 100-game random-controller acceptance baseline' \
		'  diagnose      Chart an rl-v2 run and play its latest checkpoint (DIAG_RUN=run DIAG_ARGS="--seeds 25")' \
		'  control-center Browse runs and reports; launch training and evaluations' \
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
