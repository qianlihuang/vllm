# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hopper NVFP4 weight conversion and execution through FlashInfer MegaMoE."""

from __future__ import annotations

from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import torch

from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available

if TYPE_CHECKING:
    from flashinfer.megamoe import MegaMoESymmBuffer

    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep import (
        FlashInferMoeEpEpilogue,
        FlashInferMoeEpWeights,
    )

_BUFFERS: WeakValueDictionary[tuple, MegaMoESymmBuffer] = WeakValueDictionary()
# Use fused kernels while split-family serving hangs are investigated.
_KERNEL_FAMILY = "fused"


def nvfp4_to_marlin(packed: torch.Tensor) -> torch.Tensor:
    # Low-nibble-first checkpoint bytes -> K[b] (high), K[b + 4] (low).
    words = packed.view(torch.uint8).reshape(*packed.shape[:-1], -1, 4)
    codes = torch.stack((words & 15, words >> 4), dim=-1).flatten(-2)
    return ((codes[..., :4] << 4) | codes[..., 4:]).reshape(packed.shape)


def rebalance_nvfp4_scales(
    scale: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move powers of two into the globals without changing decoded weights."""
    local = scale.view(torch.float8_e4m3fn).float()
    # The decoder represents FP4 * local scale in one E4M3 byte.
    shift = torch.exp2(
        torch.ceil(torch.log2(local.amax(dim=(1, 2)) / (448.0 / 6))).clamp_min(0)
    )
    local = local / shift[:, None, None]
    encoded = local.to(torch.float8_e4m3fn)
    if not torch.equal(encoded.float(), local):
        raise ValueError(
            "NVFP4 scales cannot be rebalanced into the SM90 decoder range exactly"
        )
    return encoded.view(torch.uint8), (global_scale.float() * shift).contiguous()


def _get_buffer(group, num_experts, capacity, top_k, hidden, intermediate, clamp):
    key = (group, num_experts, capacity, top_k, hidden, intermediate, clamp)
    if (buffer := _BUFFERS.get(key)) is not None:
        return buffer
    try:
        from flashinfer.megamoe import MegaMoESymmBuffer
    except ImportError as err:
        raise RuntimeError(
            "SM90 NVFP4 requires a FlashInfer source build with flashinfer.megamoe"
        ) from err
    buffer = MegaMoESymmBuffer(
        group, num_experts, capacity, top_k, hidden, intermediate, clamp
    )
    buffer.prewarm(kernel_family=_KERNEL_FAMILY)
    _BUFFERS[key] = buffer
    return buffer


class FlashInferSm90Nvfp4MoeEp:
    def __init__(
        self,
        moe: FusedMoEConfig,
        weights: FlashInferMoeEpWeights,
        epilogue: FlashInferMoeEpEpilogue,
        *,
        apply_topk_in_fc1: bool,
    ) -> None:
        from flashinfer.megamoe import transform_nvfp4_weights_for_mega_moe

        if apply_topk_in_fc1:
            raise ValueError("SM90 NVFP4 MegaMoE does not weight expert inputs")
        assert weights.w13_scale is not None and weights.w2_scale is not None
        assert epilogue.fc1_alpha is not None and epilogue.fc2_alpha is not None
        assert moe.swiglu_limit is not None
        s1, self.g1 = rebalance_nvfp4_scales(weights.w13_scale, epilogue.fc1_alpha)
        s2, self.g2 = rebalance_nvfp4_scales(weights.w2_scale, epilogue.fc2_alpha)
        self.w1, self.w2 = transform_nvfp4_weights_for_mega_moe(
            (nvfp4_to_marlin(weights.w13), s1),
            (nvfp4_to_marlin(weights.w2), s2),
        )
        self._empty_scale = torch.empty(0, dtype=torch.uint8, device=self.w1.device)
        self.moe = moe
        self.buffer = _get_buffer(
            get_ep_group().device_group,
            moe.num_experts,
            moe.max_num_tokens,
            moe.experts_per_token,
            moe.hidden_dim,
            moe.intermediate_size_per_partition,
            float(moe.swiglu_limit),
        )

    def __call__(self, hidden_states, topk_ids, topk_weights):
        from flashinfer.megamoe import nvfp4_mega_moe

        n = hidden_states.shape[0]
        m = n
        if is_forward_context_available():
            metadata = get_forward_context().dp_metadata
            if metadata is not None:
                if metadata.local_sizes is not None:
                    m = max(m, max(metadata.local_sizes))
                else:
                    total = int(metadata.num_tokens_across_dp_cpu.max())
                    m = max(m, (total + self.moe.sp_size - 1) // self.moe.sp_size)
        if m == 0:
            return hidden_states.new_empty((0, self.moe.hidden_dim))
        if m > self.moe.max_num_tokens:
            raise ValueError(f"SM90 MegaMoE M={m} exceeds {self.moe.max_num_tokens}")
        b = self.buffer
        if n:
            torch.ops._C.per_token_group_fp8_quant(
                hidden_states.contiguous(),
                b.x[:n],
                b.x_sf[:n],
                128,
                1e-10,
                -448.0,
                448.0,
                False,
                False,
                False,
            )
            b.topk_idx[:n].copy_(topk_ids)
            b.topk_weights[:n].copy_(topk_weights)
        if n < m:
            b.x[n:m].zero_()
            b.x_sf[n:m].fill_(1)
            b.topk_idx[n:m].fill_(-1)
            b.topk_weights[n:m].zero_()
        b.commit_staged(m)
        output = hidden_states.new_empty((m, self.moe.hidden_dim))
        nvfp4_mega_moe(
            output,
            self.w1,
            self.w2,
            b,
            num_tokens=m,
            l1_global_scales=self.g1,
            l2_global_scales=self.g2,
            kernel_family=_KERNEL_FAMILY,
        )
        return output[:n]

    def warmup(self) -> None:
        pass  # The shared buffer prewarms all selectable plans once at creation.

    def kernel_weights(self):
        # Scales are embedded in the packed weight tensors.
        return self.w1, self._empty_scale, self.w2, self._empty_scale
