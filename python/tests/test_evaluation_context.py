"""Startup cuBLAS setup and the scoped evaluation settings."""

from __future__ import annotations

from typing import Callable

import pytest
import torch

import orrery.evaluate as evaluate_module
import orrery.evaluation_context as evaluation_context_module
import orrery.train as train_module
import scripts.validate_checkpoint as validate_checkpoint_module
from orrery.evaluation_context import (
    CUBLAS_WORKSPACE_VARIABLE,
    EVALUATION_MHA_FASTPATH,
    configure_cuda_workspace,
    evaluation_context,
    execution_settings,
    resolve_evaluation_device,
)


@pytest.mark.parametrize(
    ("before", "expected"),
    [(None, ":4096:8"), (":4096:8", ":4096:8"), (":16:8", ":16:8")],
    ids=["unset becomes default", "default kept", "smaller workspace kept"],
)
def test_cuda_workspace_defaults_and_keeps_deterministic_values(
    before: str | None, expected: str
) -> None:
    """Set the default when unset, and keep either deterministic value.

    Calling it a second time must change nothing, because more than one
    entry point can run in the same process during tests.
    """

    environ = {} if before is None else {CUBLAS_WORKSPACE_VARIABLE: before}

    assert configure_cuda_workspace(environ) == expected
    assert configure_cuda_workspace(environ) == expected
    assert environ == {CUBLAS_WORKSPACE_VARIABLE: expected}


def test_cuda_workspace_refuses_a_non_deterministic_value() -> None:
    """Refuse a value deterministic mode cannot use, and leave it untouched."""

    environ = {CUBLAS_WORKSPACE_VARIABLE: ":0:0"}

    with pytest.raises(ValueError, match="not deterministic"):
        configure_cuda_workspace(environ)
    assert environ == {CUBLAS_WORKSPACE_VARIABLE: ":0:0"}


@pytest.mark.parametrize(
    ("module", "main"),
    [
        (train_module, train_module.main),
        (evaluate_module, evaluate_module.main),
        (validate_checkpoint_module, validate_checkpoint_module.main),
    ],
    ids=["train", "evaluate", "validate_checkpoint"],
)
def test_entry_points_configure_the_workspace_before_anything_else(
    monkeypatch: pytest.MonkeyPatch, module: object, main: Callable[..., int]
) -> None:
    """Each command fixes the workspace before it even parses its arguments.

    The replacement setup raises, so reaching argument parsing (which would
    exit for the empty argument list) would mean the setup ran too late.
    """

    class Configured(RuntimeError):
        """Stop the command as soon as the workspace setup is reached."""

    def configured() -> str:
        raise Configured

    monkeypatch.setattr(module, "configure_cuda_workspace", configured)

    with pytest.raises(Configured):
        main([])


def test_evaluation_context_applies_and_records_settings_then_restores_them() -> None:
    """Use evaluation settings inside the context and training's afterwards.

    Training's settings are made different from the evaluation ones first,
    so restoration is visible. The yielded record must describe the settings
    in force inside the context, not the ones restored after it.
    """

    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(False)
    try:
        with evaluation_context(torch.device("cpu")) as settings:
            assert torch.get_float32_matmul_precision() == "highest"
            assert torch.are_deterministic_algorithms_enabled()

        assert settings["device"] == "cpu"
        assert settings["float32_matmul_precision"] == "highest"
        assert settings["deterministic_algorithms"] is True
        assert settings["deterministic_warn_only"] is False
        assert settings["cpu_threads"] == torch.get_num_threads()
        assert "cublas_workspace" not in settings
        assert torch.get_float32_matmul_precision() == "high"
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.set_float32_matmul_precision("highest")


def test_evaluation_context_restores_settings_after_a_failure() -> None:
    """An error during validation still puts training's settings back."""

    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with pytest.raises(RuntimeError, match="validation failed"):
            with evaluation_context(torch.device("cpu")):
                raise RuntimeError("validation failed")

        assert torch.get_float32_matmul_precision() == "high"
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(False)


@pytest.mark.parametrize(
    ("requested", "configured", "expected"),
    [("cpu", "cuda", "cpu"), (None, "cpu", "cpu"), ("cuda", "cpu", "cuda")],
    ids=["option beats config", "config is the default", "cuda when available"],
)
def test_evaluation_device_prefers_the_option_over_the_config(
    monkeypatch: pytest.MonkeyPatch, requested: str | None, configured: str, expected: str
) -> None:
    """``--device`` wins; without it, the run config's device is used.

    CUDA availability is replaced so the CUDA case runs on any machine.
    """

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_evaluation_device(requested, configured) == torch.device(expected)


def test_evaluation_device_refuses_unknown_names() -> None:
    """Only the CPU and CUDA are evaluation devices."""

    with pytest.raises(ValueError, match="must be cpu or cuda, got 'mps'"):
        resolve_evaluation_device("mps", "cuda")


class _FakeFastPath:
    """Stand-in for ``torch.backends.mha``, which older PyTorch lacks."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def get_fastpath_enabled(self) -> bool:
        return self.enabled

    def set_fastpath_enabled(self, value: bool) -> None:
        self.enabled = value


@pytest.mark.parametrize("fail", [False, True], ids=["completes", "fails"])
def test_evaluation_context_fixes_records_and_restores_the_attention_fast_path(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """Evaluation uses the fixed fast-path choice, records it, then puts training's back.

    Training's process has the fast path off here, so the evaluation value
    is visible inside the context and the restoration is visible after it,
    whether validation completes or raises.
    """

    # Replace the lookup rather than torch.backends.mha itself: newer PyTorch
    # versions manage that attribute internally, so patching it is unreliable.
    switch = _FakeFastPath(enabled=not EVALUATION_MHA_FASTPATH)
    monkeypatch.setattr(evaluation_context_module, "_mha_backend", lambda: switch)

    def run() -> dict[str, object]:
        with evaluation_context(torch.device("cpu")) as settings:
            assert switch.enabled is EVALUATION_MHA_FASTPATH
            if fail:
                raise RuntimeError("validation failed")
        return settings

    if fail:
        with pytest.raises(RuntimeError, match="validation failed"):
            run()
    else:
        assert run()["mha_fastpath"] is EVALUATION_MHA_FASTPATH
    assert switch.enabled is not EVALUATION_MHA_FASTPATH


def test_execution_settings_record_an_uncontrollable_fast_path_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PyTorch without the switch records None: not controlled, not on or off.

    The lookup is replaced to imitate an older PyTorch on any version: on
    PyTorch that has ``torch.backends.mha``, deleting the attribute does not
    reliably hide it.
    """

    monkeypatch.setattr(evaluation_context_module, "_mha_backend", lambda: None)

    with evaluation_context(torch.device("cpu")) as settings:
        pass

    assert settings["mha_fastpath"] is None


def test_cuda_execution_settings_record_gpu_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CUDA run records its GPU name and cuBLAS workspace, not CPU threads.

    The GPU name lookup is replaced, so this needs no GPU.
    """

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")
    monkeypatch.setenv(CUBLAS_WORKSPACE_VARIABLE, ":16:8")

    settings = execution_settings(torch.device("cuda"))

    assert settings["device"] == "cuda"
    assert settings["device_name"] == "Test GPU"
    assert settings["cublas_workspace"] == ":16:8"
    assert settings["cuda_version"] == torch.version.cuda
    assert "cpu_threads" not in settings
