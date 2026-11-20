# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal window indexing over engine-managed sliding MLA pages."""

from typing import Any

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.models.hy_v4.nvidia.flashmla_sparse import HYV4FlashMLASparseBackend
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadataBuilder


class FlashSparseMetadataBuilder(FlashMLASparseMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        layer = vllm_config.compilation_config.static_forward_context[layer_names[0]]
        assert layer.topk_buffer_width is not None
        self.topk_tokens = layer.topk_buffer_width
        self.topk_tokens_tensor.fill_(self.topk_tokens)
        # The direct paged kernel never gathers evicted SWA history.
        self.fp8_use_mixed_batch = True


class FlashSparseBackend(HYV4FlashMLASparseBackend):
    @staticmethod
    def get_builder_cls():
        return FlashSparseMetadataBuilder

    @classmethod
    def supports_pcp(cls):
        return False

    @classmethod
    def supports_dcp(cls):
        return False


class FlashSWABackend(FlashSparseBackend):
    @classmethod
    def supports_sliding_window(cls):
        return True


@triton.jit(do_not_specialize=["num_actual_tokens", "num_reqs"])
def _fill_window_indices_kernel(
    Out,
    QueryStartLoc,
    SeqLens,
    num_actual_tokens,
    num_reqs,
    OUT_STRIDE,
    WINDOW: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0)
    total_tokens = tl.load(QueryStartLoc + num_reqs)
    live = (row < num_actual_tokens) & (row < total_tokens)
    logical = -1
    if live:
        lo = 0
        hi = num_reqs
        while lo < hi:
            mid = (lo + hi) >> 1
            start_mid = tl.load(QueryStartLoc + mid)
            if start_mid <= row:
                lo = mid + 1
            else:
                hi = mid
        req = lo - 1
        q_start = tl.load(QueryStartLoc + req)
        q_end = tl.load(QueryStartLoc + req + 1)
        seq_len = tl.load(SeqLens + req)
        logical = seq_len - (q_end - q_start) + (row - q_start)

    out_row = Out + row * OUT_STRIDE
    offs = tl.arange(0, BLOCK_W)
    for base in tl.static_range(0, WIDTH, BLOCK_W):
        cols = base + offs
        idx = logical - WINDOW + 1 + cols
        valid = live & (logical >= 0) & (cols < WINDOW) & (idx >= 0)
        tl.store(
            out_row + cols,
            tl.where(valid, idx, -1).to(tl.int32),
            mask=cols < WIDTH,
        )


def window_indices(positions: torch.Tensor, window: int, width: int) -> torch.Tensor:
    if window < 1 or width < window:
        raise ValueError("SWA index width must cover its causal window")
    columns = torch.arange(width, device=positions.device)
    indices = positions[:, None] - window + 1 + columns
    live = (positions[:, None] >= 0) & (columns < window) & (indices >= 0)
    return torch.where(live, indices, -1).int()


class FlashWindowIndexer(nn.Module):
    def __init__(
        self,
        window: int,
        topk_indices_buffer,
        attention_prefix: str,
        *,
        reuse_cached_indices: bool = False,
    ):
        super().__init__()
        self.window = window
        self.attention_prefix = attention_prefix
        self.reuse_cached_indices = reuse_cached_indices
        # Logical-to-physical remapping uses 128-column tiles.
        width = round_up(window, 128)
        if window < 1 or topk_indices_buffer.shape[1] < width:
            raise ValueError("SWA index width must cover its causal window")
        rows = topk_indices_buffer.shape[0]
        # Reuse the shared scratch storage with a packed, window-sized stride.
        self.topk_indices_buffer = topk_indices_buffer.view(-1)[: rows * width].view(
            rows, width
        )

    def forward(self, hidden_states, q_c, positions, rope=None):
        output = self.topk_indices_buffer[: hidden_states.shape[0]]
        if self.reuse_cached_indices:
            return output
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return output.fill_(-1)
        if not isinstance(metadata, dict):
            raise TypeError("Flash SWA requires per-layer metadata")
        meta: Any = metadata[self.attention_prefix]
        if meta.num_actual_tokens == 0 or meta.seq_lens.numel() == 0:
            return output.fill_(-1)
        starts = meta.query_start_loc
        num_reqs = starts.numel() - 1
        if (
            output.is_cuda
            and starts.is_cuda
            and meta.seq_lens.is_cuda
            and output.dtype == torch.int32
            and output.stride(-1) == 1
            and starts.stride(0) == 1
            and meta.seq_lens.stride(0) == 1
            and num_reqs > 0
            and output.shape[0] > 0
        ):
            width = output.shape[1]
            _fill_window_indices_kernel[(output.shape[0],)](
                output,
                starts,
                meta.seq_lens,
                meta.num_actual_tokens,
                num_reqs,
                output.stride(0),
                WINDOW=self.window,
                WIDTH=width,
                BLOCK_W=min(512, triton.next_power_of_2(width)),
            )
            return output
        rows = torch.arange(output.shape[0], device=output.device)
        requests = torch.searchsorted(starts[1:], rows, right=True)
        requests = requests.clamp_max(starts.numel() - 2).long()
        logical = meta.seq_lens[requests] - (starts[requests + 1] - starts[requests])
        logical = logical + rows - starts[requests]
        live = (rows < meta.num_actual_tokens) & (rows < starts[-1])
        output.copy_(
            window_indices(torch.where(live, logical, -1), self.window, output.shape[1])
        )
        return output
