"""Make validation inference repeatable on the GPU, and record how it ran.

A GPU can add the same numbers in a different order from one run to the
next, so a launch score of 0.0000001 may round to -0.0000001 and flip a
launch decision. Two pieces of setup prevent that:

1. ``configure_cuda_workspace`` fixes cuBLAS's workspace once per process.
   cuBLAS reads the setting when it first starts, so it must run before any
   CUDA work. Every command-line entry point calls it first in ``main()``.
   Python code or tests that use CUDA directly must call it themselves
   before their first CUDA use: setting it inside validation would be too
   late for a process that has already trained on the GPU.
2. ``evaluation_context`` switches PyTorch to full float32 matrix precision
   and deterministic algorithms only while validation runs, then restores
   training's settings.

The settings actually in force are recorded beside every result, because an
exact replay needs the same device, software versions and settings.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Iterator, MutableMapping

import torch


# The cuBLAS workspace settings NVIDIA documents as deterministic. ":4096:8"
# is the default chosen here; ":16:8" uses less memory and is accepted when a
# machine already sets it.
CUBLAS_WORKSPACE_VARIABLE = "CUBLAS_WORKSPACE_CONFIG"
DEFAULT_CUBLAS_WORKSPACE = ":4096:8"
DETERMINISTIC_CUBLAS_WORKSPACES = (DEFAULT_CUBLAS_WORKSPACE, ":16:8")


def configure_cuda_workspace(
    environ: MutableMapping[str, str] = os.environ,
) -> str:
    """Set a deterministic cuBLAS workspace for this process, and return it.

    An unset variable becomes ``:4096:8``; an existing ``:16:8`` is kept.
    Any other value is refused with an error, because PyTorch's deterministic
    mode would later refuse GPU matrix multiplication with a less clear one.

    The setting lasts for the whole process, so training's cuBLAS work uses
    it too. It has no effect on CPU runs, but is checked there as well so the
    same command behaves the same on every machine. Calling it again is safe.
    ``environ`` is replaceable for tests.
    """

    value = environ.get(CUBLAS_WORKSPACE_VARIABLE)
    if value is None:
        environ[CUBLAS_WORKSPACE_VARIABLE] = DEFAULT_CUBLAS_WORKSPACE
        return DEFAULT_CUBLAS_WORKSPACE
    if value not in DETERMINISTIC_CUBLAS_WORKSPACES:
        accepted = " or ".join(DETERMINISTIC_CUBLAS_WORKSPACES)
        raise ValueError(
            f"{CUBLAS_WORKSPACE_VARIABLE}={value!r} is not deterministic; "
            f"unset it or use {accepted}"
        )
    return value


# Devices a validation or evaluation command may be asked to use.
EVALUATION_DEVICES = ("cpu", "cuda")

# Whether PyTorch's fused multi-head-attention "fast path" may run during
# evaluation. It is a faster inference-only implementation of the same
# transformer layer, and in eval mode PyTorch chooses it on its own whenever
# it can. Evaluation sets it to this fixed value instead of inheriting
# whatever the process last chose. True keeps PyTorch's default; the early
# GPU compatibility check decides whether deterministic mode requires False
# on the target GPU.
EVALUATION_MHA_FASTPATH = True


def _mha_backend() -> object | None:
    """Return PyTorch's fast-path switch, or None where this version lacks it.

    ``torch.backends.mha`` exists from PyTorch 2.3. Older versions (the
    Intel-Mac 2.2 build used locally) cannot switch the fast path, and their
    settings record it as None: not controlled rather than on or off.
    """

    return getattr(torch.backends, "mha", None)


def resolve_evaluation_device(requested: str | None, configured: str) -> torch.device:
    """Choose where a command evaluates: ``--device`` if given, else the config.

    There is no automatic fallback to the CPU. A silent switch would change
    the recorded execution settings without anyone choosing it, and a
    protected held-out pool can be spent only once. Asking for CUDA on a
    machine without it is therefore an error that names the explicit
    alternative.
    """

    name = requested if requested is not None else configured
    if name not in EVALUATION_DEVICES:
        raise ValueError(f"evaluation device must be cpu or cuda, got {name!r}")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is not available on this machine; pass --device cpu to evaluate on the CPU"
        )
    return torch.device(name)


def execution_settings(device: torch.device) -> dict[str, object]:
    """Describe the device, software and numeric settings now in force.

    Called while ``evaluation_context`` is active, so it records the
    evaluation settings rather than training's. Only facts that apply to the
    device are included: a CPU run records its thread count, a CUDA run its
    GPU name, CUDA version and cuBLAS workspace.
    """

    mha = _mha_backend()
    settings: dict[str, object] = {
        "device": device.type,
        "torch_version": torch.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "mha_fastpath": mha.get_fastpath_enabled() if mha is not None else None,
    }
    if device.type == "cuda":
        settings["device_name"] = torch.cuda.get_device_name(device)
        settings["cuda_version"] = torch.version.cuda
        settings["cublas_workspace"] = os.environ.get(CUBLAS_WORKSPACE_VARIABLE)
    else:
        settings["cpu_threads"] = torch.get_num_threads()
    return settings


@contextmanager
def evaluation_context(device: torch.device) -> Iterator[dict[str, object]]:
    """Run validation with full float32 precision and deterministic algorithms.

    On entry the context:

    - sets float32 matrix multiplication to ``"highest"``, so the GPU does not
      use the faster but less precise TF32 format;
    - turns on deterministic algorithms with ``warn_only=False``, so an
      operation without a deterministic version raises instead of quietly
      running a non-repeatable one;
    - sets the attention fast path to ``EVALUATION_MHA_FASTPATH``, where
      this PyTorch version can switch it.

    It yields ``execution_settings(device)``, captured while these settings
    are active, for the caller to store with its results. On exit, after
    success or an exception, it restores training's previous settings.
    """

    # Read everything that will change before changing anything, so the
    # finally block can always put back exactly what training had.
    mha = _mha_backend()
    previous_precision = torch.get_float32_matmul_precision()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_fastpath = mha.get_fastpath_enabled() if mha is not None else None
    try:
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True, warn_only=False)
        if mha is not None:
            mha.set_fastpath_enabled(EVALUATION_MHA_FASTPATH)
        yield execution_settings(device)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.use_deterministic_algorithms(
            previous_deterministic, warn_only=previous_warn_only
        )
        if mha is not None:
            mha.set_fastpath_enabled(previous_fastpath)
