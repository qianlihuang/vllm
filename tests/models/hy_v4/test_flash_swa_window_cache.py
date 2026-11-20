# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HY4-Flash SWA window index caching and Triton kernel."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.models.hy_v4_flash.nvidia.swa import FlashWindowIndexer, window_indices
from vllm.platforms import current_platform


def _make_meta(starts: list[int], seq_lens: list[int], num_actual_tokens: int, device):
    return SimpleNamespace(
        query_start_loc=torch.tensor(starts, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        num_actual_tokens=num_actual_tokens,
    )


def test_flash_window_indexer_reuse_cached_indices():
    buf = torch.full((16, 512), -99, dtype=torch.int32)
    first = FlashWindowIndexer(512, buf, "layer.2.attn", reuse_cached_indices=False)
    second = FlashWindowIndexer(512, buf, "layer.3.attn", reuse_cached_indices=True)

    meta = _make_meta([0, 4, 8], [100, 3], num_actual_tokens=8, device="cpu")
    hidden = torch.zeros((10, 8))
    positions = torch.zeros((10,), dtype=torch.int64)

    with patch(
        "vllm.models.hy_v4_flash.nvidia.swa.get_forward_context",
        return_value=SimpleNamespace(attn_metadata={"layer.2.attn": meta}),
    ):
        out1 = first(hidden, None, positions)

    # Second SWA layer reuses the cached buffer without needing metadata.
    with patch(
        "vllm.models.hy_v4_flash.nvidia.swa.get_forward_context",
        side_effect=AssertionError("cached SWA layer should not query metadata"),
    ):
        out2 = second(hidden, None, positions)

    assert out2.data_ptr() == out1.data_ptr()
    assert torch.equal(out2, out1)
    # Rows 8..9 are past num_actual_tokens -> all -1.
    assert (out1[8:] == -1).all()
    # Row 0 (req 0, seq_len=100, q_len=4) -> logical pos 96 -> right-aligned [0..96].
    assert (out1[0, : 512 - 97] == -1).all()
    assert out1[0, 512 - 97 :].tolist() == list(range(97))


@pytest.mark.skipif(
    not (current_platform.is_cuda() and torch.accelerator.is_available()),
    reason="CUDA required",
)
@pytest.mark.parametrize("window", [128, 512, 1024])
def test_flash_window_indexer_triton_matches_cpu_reference(window: int):
    width = ((window + 127) // 128) * 128
    # Include variable lengths, zero-length request, short seq < window,
    # long seq > window, and trailing padded rows > num_actual_tokens.
    starts = [0, 0, 1, 5, 11, 15]
    seq_lens = [0, 2, 10, 1500, 600]
    num_actual_tokens = 15
    total_rows = 20

    cpu_buf = torch.full((total_rows, width), -77, dtype=torch.int32)
    gpu_buf = torch.full((total_rows, width), -77, dtype=torch.int32, device="cuda")

    cpu_idx = FlashWindowIndexer(window, cpu_buf, "attn", reuse_cached_indices=False)
    gpu_idx = FlashWindowIndexer(window, gpu_buf, "attn", reuse_cached_indices=False)

    cpu_meta = _make_meta(starts, seq_lens, num_actual_tokens, device="cpu")
    gpu_meta = _make_meta(starts, seq_lens, num_actual_tokens, device="cuda")

    with patch(
        "vllm.models.hy_v4_flash.nvidia.swa.get_forward_context",
        return_value=SimpleNamespace(attn_metadata={"attn": cpu_meta}),
    ):
        ref = cpu_idx(torch.zeros((total_rows, 4)), None, torch.zeros(total_rows))

    with patch(
        "vllm.models.hy_v4_flash.nvidia.swa.get_forward_context",
        return_value=SimpleNamespace(attn_metadata={"attn": gpu_meta}),
    ):
        got = gpu_idx(
            torch.zeros((total_rows, 4), device="cuda"),
            None,
            torch.zeros(total_rows, device="cuda"),
        )

    assert torch.equal(got.cpu(), ref)
    # Cross-check row 7 (req 3, q_start=5, q_end=11, seq_len=1500 -> logical=1496)
    expected_row7 = window_indices(torch.tensor([1496]), window, width)[0]
    assert torch.equal(got[7].cpu(), expected_row7)
