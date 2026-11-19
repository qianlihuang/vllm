# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HY4-Flash IndexPool select_groups and expand_groups kernels."""

import pytest
import torch

from vllm.models.hy_v4_flash.nvidia.indexpool_ops import expand_groups, select_groups
from vllm.platforms import current_platform


def _canonicalize_indices(indices: torch.Tensor) -> torch.Tensor:
    sentinel = torch.iinfo(torch.int32).max
    ordered = torch.where(indices >= 0, indices, sentinel).sort(dim=-1).values
    return torch.where(ordered == sentinel, -1, ordered)


def test_select_groups_cpu_and_sorted_fallback():
    logits = torch.tensor(
        [
            [0.1, 3.0, 2.0, 5.0, 99.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        dtype=torch.float32,
    )
    upper = torch.tensor([4, 2, 0], dtype=torch.int64)
    got = select_groups(logits, upper, budget=3)
    expected = torch.tensor(
        [
            [1, 2, 3],
            [0, 1, -1],
            [-1, -1, -1],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(got, expected)


def test_expand_groups_cpu_and_out_buffer():
    group_ids = torch.tensor(
        [
            [2, 0, -1],
            [5, 1, 0],
            [0, 1, 2],
        ],
        dtype=torch.int32,
    )
    positions = torch.tensor([13, 6, -1], dtype=torch.int64)
    active = torch.tensor([True, True, True], dtype=torch.bool)
    out = torch.full((3, 16), -99, dtype=torch.int32)

    result = expand_groups(group_ids, positions, ratio=4, active=active, out=out)
    assert result is out
    ref = expand_groups(group_ids, positions, ratio=4, active=active)
    assert torch.equal(result, ref)
    # Row 0: groups 2 and 0 (upper=3) + tail [12, 13] -> 10 valid tokens, 6 -1s
    assert result[0].tolist() == [
        8,
        9,
        10,
        11,
        0,
        1,
        2,
        3,
        12,
        13,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    ]
    # Row 2: dead position -> all -1
    assert (result[2] == -1).all()


@pytest.mark.skipif(
    not (current_platform.is_cuda() and torch.accelerator.is_available()),
    reason="CUDA required",
)
@pytest.mark.parametrize("width", [64, 4096, 32768, 262144])
def test_select_groups_cuda_matches_reference(width: int):
    device = torch.device("cuda")
    torch.manual_seed(0)
    bounds_list = [0, 1, 7, 100, 510, 511, 512, 1000, min(width, 32768), width]
    bounds_list = [min(b, width) for b in bounds_list]
    rows = len(bounds_list)
    logits = torch.randn((rows, width), device=device, dtype=torch.float32)
    for i, b in enumerate(bounds_list):
        if b < width:
            logits[i, b:] = 1e9
    upper = torch.tensor(bounds_list, device=device, dtype=torch.int64)

    ref = select_groups(logits, upper, budget=511, sorted_output=True)
    got = select_groups(logits, upper, budget=511, sorted_output=False)
    assert got.dtype == torch.int32
    assert got.shape == (rows, 511)
    assert torch.equal(_canonicalize_indices(got), _canonicalize_indices(ref))


@pytest.mark.skipif(
    not (current_platform.is_cuda() and torch.accelerator.is_available()),
    reason="CUDA required",
)
def test_expand_groups_triton_matches_cpu_reference():
    device = torch.device("cuda")
    torch.manual_seed(0)
    rows, budget, ratio = 12, 511, 4
    group_ids = torch.randint(-2, 600, (rows, budget), device=device, dtype=torch.int32)
    positions = torch.tensor(
        [-1, 0, 1, 2, 3, 4, 5, 100, 2047, 2048, 131071, 131072],
        device=device,
        dtype=torch.int64,
    )
    active = torch.tensor(
        [True, True, False, True, True, True, True, True, True, True, True, True],
        device=device,
        dtype=torch.bool,
    )
    out = torch.full(
        (rows, (budget + 1) * ratio), -77, device=device, dtype=torch.int32
    )

    got = expand_groups(group_ids, positions, ratio, active=active, out=out)
    assert got is out
    ref = expand_groups(group_ids.cpu(), positions.cpu(), ratio, active=active.cpu())
    assert torch.equal(got.cpu(), ref)
