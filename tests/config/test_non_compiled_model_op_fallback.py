# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for eager IR-op / CustomOp fallback when a model is not compiled."""

from types import SimpleNamespace
from unittest.mock import patch

import torch.nn as nn

import vllm.ir
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    KernelConfig,
    VllmConfig,
    get_cached_compilation_config,
    set_current_vllm_config,
)
from vllm.config.kernel import IrOpPriorityConfig
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms.cuda import CudaPlatformBase
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@CustomOp.register("test_fallback_dummy_op")
class _DummyCustomOp(CustomOp):
    def forward_native(self, x):
        return ("native", x)

    def forward_cuda(self, x):
        return ("cuda", x)

    def forward_cpu(self, x):
        return ("cpu", x)


class _DummyUncompiledModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.op = _DummyCustomOp()


class _DummyCompiledSubmodule(nn.Module, TorchCompileWithNoGuardsWrapper):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.do_not_compile = False
        self.op = _DummyCustomOp()

    def forward(self, x):
        return self.op(x)


def _make_vllm_config(
    *,
    custom_ops: list[str] | None = None,
    ir_enable_torch_wrap: bool | None = None,
    ir_op_priority: IrOpPriorityConfig | None = None,
) -> VllmConfig:
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        backend="inductor",
        custom_ops=list(custom_ops) if custom_ops is not None else [],
        ir_enable_torch_wrap=ir_enable_torch_wrap,  # type: ignore[arg-type]
    )
    kc = KernelConfig(
        ir_op_priority=ir_op_priority
        if ir_op_priority is not None
        else IrOpPriorityConfig()
    )
    with patch(
        "vllm.platforms.current_platform.get_default_ir_op_priority",
        side_effect=CudaPlatformBase.get_default_ir_op_priority,
    ):
        return VllmConfig(compilation_config=cc, kernel_config=kc)


def _make_runner_stub(vllm_config: VllmConfig, model: nn.Module) -> SimpleNamespace:
    return SimpleNamespace(
        vllm_config=vllm_config,
        compilation_config=vllm_config.compilation_config,
        model=model,
        speculator=None,
    )


def test_uncompiled_model_falls_back_to_eager_ir_and_custom_ops():
    cfg = _make_vllm_config()
    assert cfg.kernel_config.ir_op_priority.rms_norm == ["native"]
    assert cfg.compilation_config.ir_enable_torch_wrap is True
    assert "none" in cfg.compilation_config.custom_ops

    assert cfg.compilation_config._auto_disabled_custom_ops is True
    assert cfg.compilation_config._auto_enabled_ir_torch_wrap is True

    get_cached_compilation_config.cache_clear()
    with set_current_vllm_config(cfg):
        model = _DummyUncompiledModel()
        assert model.op(1) == ("native", 1)

    runner = _make_runner_stub(cfg, model)
    with (
        patch(
            "vllm.platforms.current_platform.get_default_ir_op_priority",
            side_effect=CudaPlatformBase.get_default_ir_op_priority,
        ),
        patch.object(cfg.kernel_config.ir_op_priority, "set_default") as mock_set_def,
        patch.object(vllm.ir, "set_default_torch_wrap") as mock_wrap,
    ):
        GPUModelRunner._maybe_fallback_eager_ops_for_uncompiled_model(runner)  # type: ignore[arg-type]

    assert cfg.kernel_config.ir_op_priority.rms_norm == ["vllm_c", "native"]
    assert cfg.kernel_config.ir_op_priority.fused_add_rms_norm == ["vllm_c", "native"]
    assert cfg.kernel_config.ir_op_priority.rotary_embedding == ["vllm_c", "native"]
    mock_set_def.assert_called_once()
    mock_wrap.assert_called_once_with(False)
    assert cfg.compilation_config.ir_enable_torch_wrap is False
    assert "all" in cfg.compilation_config.custom_ops
    assert "none" not in cfg.compilation_config.custom_ops
    assert model.op(1)[0] != "native"


def test_user_overrides_are_preserved_on_uncompiled_model():
    cfg = _make_vllm_config(
        custom_ops=["none"],
        ir_enable_torch_wrap=True,
        ir_op_priority=IrOpPriorityConfig(rms_norm=["native"]),
    )
    get_cached_compilation_config.cache_clear()
    with set_current_vllm_config(cfg):
        model = _DummyUncompiledModel()

    runner = _make_runner_stub(cfg, model)
    with (
        patch(
            "vllm.platforms.current_platform.get_default_ir_op_priority",
            side_effect=CudaPlatformBase.get_default_ir_op_priority,
        ),
        patch.object(cfg.kernel_config.ir_op_priority, "set_default"),
        patch.object(vllm.ir, "set_default_torch_wrap") as mock_wrap,
    ):
        GPUModelRunner._maybe_fallback_eager_ops_for_uncompiled_model(runner)  # type: ignore[arg-type]

    # User-specified rms_norm stays native; unspecified fused_add_rms_norm falls back.
    assert cfg.kernel_config.ir_op_priority.rms_norm == ["native"]
    assert cfg.kernel_config.ir_op_priority.fused_add_rms_norm == ["vllm_c", "native"]
    # Explicit user custom_ops=["none"] and ir_enable_torch_wrap=True are kept.
    assert cfg.compilation_config.custom_ops == ["none"]
    assert cfg.compilation_config.ir_enable_torch_wrap is True
    mock_wrap.assert_not_called()
    assert model.op(1) == ("native", 1)


def test_compiled_model_keeps_inductor_defaults():
    cfg = _make_vllm_config()
    get_cached_compilation_config.cache_clear()
    with set_current_vllm_config(cfg):
        model = _DummyCompiledSubmodule()

    runner = _make_runner_stub(cfg, model)
    with (
        patch(
            "vllm.platforms.current_platform.get_default_ir_op_priority",
            side_effect=CudaPlatformBase.get_default_ir_op_priority,
        ),
        patch.object(cfg.kernel_config.ir_op_priority, "set_default") as mock_set_def,
        patch.object(vllm.ir, "set_default_torch_wrap") as mock_wrap,
    ):
        GPUModelRunner._maybe_fallback_eager_ops_for_uncompiled_model(runner)  # type: ignore[arg-type]

    assert cfg.kernel_config.ir_op_priority.rms_norm == ["native"]
    assert "none" in cfg.compilation_config.custom_ops
    assert cfg.compilation_config.ir_enable_torch_wrap is True
    mock_set_def.assert_not_called()
    mock_wrap.assert_not_called()
    assert model.op(1) == ("native", 1)
