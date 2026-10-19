# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.workspace as workspace
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import _num_workspace_lanes


class _SpecConfig:
    def __init__(self, dspark: bool) -> None:
        self._dspark = dspark

    def use_dspark(self) -> bool:
        return self._dspark


class _VllmConfig:
    def __init__(self, spec_config: _SpecConfig | None) -> None:
        self.speculative_config = spec_config


@pytest.mark.parametrize(
    ("use_v2_model_runner", "spec_config", "expected"),
    [
        (True, _SpecConfig(True), 2),
        (False, _SpecConfig(True), 1),
        (True, _SpecConfig(False), 1),
        (True, None, 1),
    ],
)
def test_workspace_lane_count_is_dspark_only(
    use_v2_model_runner: bool,
    spec_config: _SpecConfig | None,
    expected: int,
) -> None:
    config = cast(VllmConfig, _VllmConfig(spec_config))
    assert _num_workspace_lanes(config, use_v2_model_runner) == expected


def test_workspace_lanes_do_not_alias_and_restore_context(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    assert manager._current_workspaces == [None, None, None, None]

    (target,) = manager.get_simultaneous(((512,), torch.uint8))
    with workspace.use_workspace_lane(1):
        (draft,) = manager.get_simultaneous(((256,), torch.uint8))
        (draft_reused,) = manager.get_simultaneous(((8,), torch.uint8))
    (target_reused,) = manager.get_simultaneous(((8,), torch.uint8))

    assert manager._current_workspaces[0].numel() == 512  # type: ignore[union-attr]
    assert manager._current_workspaces[1].numel() == 256  # type: ignore[union-attr]
    assert manager._current_workspaces[2:] == [None, None]
    assert target.data_ptr() != draft.data_ptr()
    assert draft.data_ptr() == draft_reused.data_ptr()
    assert target.data_ptr() == target_reused.data_ptr()


def test_workspace_lanes_compose_with_ubatches(monkeypatch) -> None:
    active_ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch[0])
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    pointers = set()
    for ubatch_id in range(2):
        active_ubatch[0] = ubatch_id
        for lane in range(2):
            with workspace.use_workspace_lane(lane):
                (buffer,) = manager.get_simultaneous(((16,), torch.uint8))
                pointers.add(buffer.data_ptr())

    assert len(pointers) == 4


def test_workspace_lock_blocks_growth_and_unlock_restores(monkeypatch) -> None:
    """Once locked, oversized requests fail loudly instead of reallocating the
    buffer that captured CUDA graphs point at; unlock restores growth."""
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=1)

    (buf,) = manager.get_simultaneous(((256,), torch.uint8))
    manager.lock()
    assert manager.is_locked()

    # Requests within the reserved size still reuse the same buffer.
    (same,) = manager.get_simultaneous(((256,), torch.uint8))
    (smaller,) = manager.get_simultaneous(((8,), torch.uint8))
    assert same.data_ptr() == buf.data_ptr()
    assert smaller.data_ptr() == buf.data_ptr()

    with pytest.raises(AssertionError, match="Workspace is locked"):
        manager.get_simultaneous(((512,), torch.uint8))

    manager.unlock()
    (grown,) = manager.get_simultaneous(((512,), torch.uint8))
    assert grown.numel() == 512


def test_workspace_lane_validation(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=1)

    with (
        pytest.raises(ValueError, match="non-negative"),
        workspace.use_workspace_lane(-1),
    ):
        pass

    with (
        workspace.use_workspace_lane(1),
        pytest.raises(RuntimeError, match="is not configured"),
    ):
        manager.get_simultaneous(((1,), torch.uint8))

    with pytest.raises(ValueError, match="at least one"):
        workspace.WorkspaceManager(torch.device("cpu"), num_lanes=0)


def test_persistent_resources_are_lazy_and_isolated(monkeypatch) -> None:
    """Only the requesting slot allocates; each ubatch/lane owns its resource."""
    active_ubatch = 0
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )
    factory = Mock(side_effect=lambda: {"buffer": torch.zeros(8)})
    resources = []
    for ubatch in range(2):
        active_ubatch = ubatch
        for lane in range(2):
            with workspace.use_workspace_lane(lane):
                resource = manager.get_persistent_resource("scratch", factory)
                resources.append(resource)
                resource["buffer"].fill_(len(resources))
                assert manager.get_persistent_resource("scratch", factory) is resource
                assert factory.call_count == len(resources)

    assert len({r["buffer"].data_ptr() for r in resources}) == 4
    for i, resource in enumerate(resources, 1):
        assert torch.all(resource["buffer"] == i)


def test_persistent_resource_lock_applies_to_each_slot(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=2)
    factory = Mock(side_effect=object)
    first = manager.get_persistent_resource("scratch", factory)
    manager.lock()
    assert manager.get_persistent_resource("scratch", factory) is first
    with pytest.raises(AssertionError, match="was not allocated during warmup"):
        manager.get_persistent_resource("new", factory)
    with workspace.use_workspace_lane(1):
        with pytest.raises(AssertionError, match="was not allocated during warmup"):
            manager.get_persistent_resource("scratch", factory)
        assert factory.call_count == 1
        manager.unlock()
        assert manager.get_persistent_resource("scratch", factory) is not first
    assert manager.get_persistent_resource("scratch", factory) is first


def test_persistent_tensor_preserves_contents_and_rejects_changes() -> None:
    """Initialization happens once and transient scratch cannot overwrite locks."""
    manager = workspace.WorkspaceManager(torch.device("cpu"))
    first = manager.get_persistent("locks", (8,), torch.int32, zero_init=True)
    assert torch.count_nonzero(first) == 0
    first.fill_(1)
    (scratch,) = manager.get_simultaneous(((1024,), torch.uint8))
    scratch.zero_()
    assert manager.get_persistent("locks", (8,), torch.int32, zero_init=True) is first
    assert torch.all(first == 1)
    with pytest.raises(ValueError, match="requested shape"):
        manager.get_persistent("locks", (16,), torch.int32)
    with pytest.raises(ValueError, match="requested shape"):
        manager.get_persistent("locks", (8,), torch.int64)
    manager.lock()
    assert manager.get_persistent("locks", (8,), torch.int32) is first
    with pytest.raises(AssertionError, match="was not allocated during warmup"):
        manager.get_persistent("new", (8,), torch.int32)


@pytest.mark.parametrize("model_type", ["kimi", "glm"])
@pytest.mark.parametrize("metadata", [None, {}])
@torch.inference_mode()
def test_kda_warmup_reserves_larger_eager_batch(
    monkeypatch: pytest.MonkeyPatch, model_type: str, metadata
):
    """A batch above graph warmup size must fit after the workspace is locked."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.ops import kda_recoverssm as shared
    from vllm.models.kimi_k3.nvidia import kda as nvidia_kda
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import kda_recoverssm_verify
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    # Exercise allocation and model warmup on CPU; CUDA arithmetic is covered
    # by the model kernel suites. Only the device trait and launches are faked.
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda: DeviceCapability(9, 0)
    )

    class NoLaunch:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(shared, "_kda_factored_prep_kernel", NoLaunch())
    monkeypatch.setattr(shared, "_kda_factored_verify_kernel", NoLaunch())
    from vllm.models.glm5next.common import kda as glm_kda
    from vllm.v1.worker import workspace

    module = nvidia_kda if model_type == "kimi" else glm_kda
    cls = (
        KimiK3DeltaAttention
        if model_type == "kimi"
        else glm_kda.Glm5NextLinearAttention
    )
    layer = object.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.prefix = "model.layers.0.attn"
    layer.use_recoverssm = True
    layer.local_num_heads, layer.head_dim = 64, 128
    layer.spec_query_len, layer.gate_lower_bound = 12, -5.0
    layer._max_num_seqs = 16
    layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
    layer.cache_config = SimpleNamespace(
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="float32",
        use_kda_recoverssm=True,
    )
    monkeypatch.setattr(
        module, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )
    manager = workspace.WorkspaceManager(torch.device("cpu"))
    monkeypatch.setattr(workspace, "_manager", manager)
    empty = torch.empty(0, device="cpu")
    if model_type == "kimi":
        layer._forward(empty, empty, empty, empty, empty)
    else:
        layer._forward(empty, empty, empty, empty)

    batch, heads, dim, window = 16, 64, 128, 12
    q = torch.zeros(1, batch * window, heads, dim, dtype=torch.bfloat16, device="cpu")
    v = torch.ones_like(q)
    g = torch.full_like(q, 50.0)
    beta = torch.zeros(1, batch * window, heads, dtype=q.dtype, device="cpu")
    state = torch.zeros(batch + 1, heads, dim, dim, device="cpu")
    correction = torch.empty(batch + 1, heads, window, dim, device="cpu")
    kg = torch.empty(batch + 1, heads, window, 2 * dim, dtype=q.dtype, device="cpu")
    cu = torch.arange(0, (batch + 1) * window, window, dtype=torch.int32, device="cpu")
    blocks = torch.arange(1, batch + 1, dtype=torch.int32, device="cpu")
    a_log = torch.zeros(heads, device="cpu")
    dt_bias = torch.zeros(heads * dim, device="cpu")

    def verify(size):
        tokens = size * window
        return kda_recoverssm_verify(
            q[:, :tokens],
            q[:, :tokens],
            v[:, :tokens],
            g[:, :tokens],
            beta[:, :tokens],
            a_log,
            dt_bias,
            -5.0,
            state,
            correction,
            kg,
            cu[: size + 1],
            blocks[:size],
            window,
        )

    verify(9)
    manager.lock()
    verify(batch)
