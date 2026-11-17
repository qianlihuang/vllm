# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-row helpers at the boundaries of a sequence-parallel residual stream."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.common.ops import sequence_parallel as sp_ops


def _two_tp_ranks(monkeypatch, rank: int) -> None:
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_rank", lambda: rank)


def test_sp_all_gather_packed_uses_one_collective_and_trims(monkeypatch):
    gathered = []

    def all_gather(shard: torch.Tensor) -> torch.Tensor:
        gathered.append(shard)
        return torch.cat([shard, shard + 100])

    monkeypatch.setattr(sp_ops, "sp_all_gather", all_gather)
    hidden = torch.arange(4, dtype=torch.float32).view(2, 2)
    aux = torch.arange(6, dtype=torch.float32).view(2, 3) + 10

    out_hidden, out_aux = sp_ops.sp_all_gather_packed([hidden, aux], 3)

    assert len(gathered) == 1
    assert gathered[0].shape == (2, 5)
    torch.testing.assert_close(out_hidden, torch.cat([hidden, hidden + 100])[:3])
    torch.testing.assert_close(out_aux, torch.cat([aux, aux + 100])[:3])


def test_sp_all_gather_packed_gathers_a_single_shard_in_place(monkeypatch):
    gathered = []

    def all_gather(shard: torch.Tensor) -> torch.Tensor:
        gathered.append(shard)
        return torch.cat([shard, shard])

    monkeypatch.setattr(sp_ops, "sp_all_gather", all_gather)
    hidden = torch.arange(4, dtype=torch.float32).view(2, 2)

    (out,) = sp_ops.sp_all_gather_packed([hidden], 3)

    assert gathered[0] is hidden
    torch.testing.assert_close(out, torch.cat([hidden, hidden])[:3])


@pytest.mark.parametrize(
    ("is_partial", "sequence_parallel", "completion"),
    [
        (True, True, "reduce_scatter"),
        (True, False, "all_reduce"),
        (False, True, "shard"),
        (False, False, None),
    ],
)
def test_sp_complete_output_completes_exactly_once(
    monkeypatch, is_partial: bool, sequence_parallel: bool, completion: str | None
):
    calls = {
        "reduce_scatter": Mock(return_value=torch.zeros(2, 2)),
        "all_reduce": Mock(return_value=torch.zeros(4, 2)),
        "shard": Mock(return_value=torch.zeros(2, 2)),
    }
    monkeypatch.setattr(sp_ops, "sp_reduce_scatter", calls["reduce_scatter"])
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_all_reduce", calls["all_reduce"])
    monkeypatch.setattr(sp_ops, "sp_shard", calls["shard"])
    output = torch.ones(4, 2)

    completed = sp_ops.sp_complete_output(
        output, is_partial=is_partial, sequence_parallel=sequence_parallel
    )

    for name, call in calls.items():
        assert call.call_count == (name == completion), name
    if completion is None:
        assert completed is output
    else:
        calls[completion].assert_called_once_with(output)
        assert completed is calls[completion].return_value


def test_sp_shard_forward_padding_keeps_this_rank_rows(monkeypatch):
    _two_tp_ranks(monkeypatch, rank=1)
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    context = SimpleNamespace(is_padding=torch.tensor([False, True, False]))
    monkeypatch.setattr(sp_ops, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(sp_ops, "get_forward_context", lambda: context)

    sp_ops.sp_shard_forward_padding(torch.empty(3, 2))

    # Three rows pad to four: rank 1 owns the third row and the padding row.
    assert context.is_padding.tolist() == [False, True]


@pytest.mark.parametrize(("skip_padding", "has_context"), [("0", True), ("1", False)])
def test_sp_shard_forward_padding_leaves_the_mask_alone(
    monkeypatch, skip_padding: str, has_context: bool
):
    _two_tp_ranks(monkeypatch, rank=0)
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", skip_padding)
    is_padding = torch.tensor([False, True, False])
    context = SimpleNamespace(is_padding=is_padding)
    monkeypatch.setattr(sp_ops, "is_forward_context_available", lambda: has_context)
    monkeypatch.setattr(sp_ops, "get_forward_context", lambda: context)

    sp_ops.sp_shard_forward_padding(torch.empty(3, 2))

    assert context.is_padding is is_padding
