# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HY4-Flash ShardQ (SWA indexer, IndexPool metadata, workspaces)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.models.hy_v4_flash.nvidia.indexpool_cache import IndexPoolMetadataBuilder
from vllm.models.hy_v4_flash.nvidia.swa import FlashWindowIndexer
from vllm.v1.attention.ops import shardq as shardq_mod
from vllm.v1.attention.ops.shardq import ShardQMetadata, zigzag_split


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("decode_lens", [[], [1, 1, 1]])
@pytest.mark.parametrize("prefill_lens", [[2048], [65, 2049, 1023]])
def test_flash_window_indexer_shardq_matches_full_batch(
    monkeypatch, world, decode_lens, prefill_lens
):
    """FlashWindowIndexer with ShardQ writes the exact decode rows followed by
    this TP rank's zigzag prefill rows into the prefix of topk_indices_buffer."""
    query_lens = decode_lens + prefill_lens
    seq_lens = [q + 128 for q in decode_lens] + [q + 64 for q in prefill_lens]
    total_tokens = sum(query_lens)
    num_decode = sum(decode_lens)
    starts = torch.tensor([0] + np.cumsum(query_lens).tolist(), dtype=torch.int32)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    hidden = torch.zeros(total_tokens, 8)
    positions = torch.zeros(total_tokens, dtype=torch.int64)

    buf_full = torch.full((total_tokens, 512), -9, dtype=torch.int32)
    indexer_full = FlashWindowIndexer(
        window=256, topk_indices_buffer=buf_full, attention_prefix="layer0.attn"
    )
    meta_full = SimpleNamespace(
        num_actual_tokens=total_tokens,
        seq_lens=seq_lens_t,
        query_start_loc=starts,
        shardq=None,
    )
    monkeypatch.setattr(
        "vllm.models.hy_v4_flash.nvidia.swa.get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer0.attn": meta_full}),
    )
    expected_full = indexer_full(hidden, None, positions).clone()

    token_ids_np, counts_np = zigzag_split(np.array(prefill_lens), world)
    shardq_meta = ShardQMetadata(
        num_decode_tokens=num_decode,
        num_tokens_per_rank=counts_np.tolist(),
        token_ids=torch.from_numpy(token_ids_np),
        rank_offsets=torch.from_numpy(np.r_[0, np.cumsum(counts_np)]),
    )
    for rank in range(world):
        monkeypatch.setattr(
            shardq_mod,
            "get_tp_group",
            lambda r=rank, w=world: SimpleNamespace(world_size=w, rank_in_group=r),
        )
        buf_sq = torch.full((total_tokens, 512), -9, dtype=torch.int32)
        indexer_sq = FlashWindowIndexer(
            window=256, topk_indices_buffer=buf_sq, attention_prefix="layer0.attn"
        )
        meta_sq = SimpleNamespace(
            num_actual_tokens=total_tokens,
            seq_lens=seq_lens_t,
            query_start_loc=starts,
            shardq=shardq_meta,
        )
        monkeypatch.setattr(
            "vllm.models.hy_v4_flash.nvidia.swa.get_forward_context",
            lambda m=meta_sq: SimpleNamespace(attn_metadata={"layer0.attn": m}),
        )
        out_sq = indexer_sq(hidden, None, positions)
        local_ids = shardq_meta.local_token_ids
        if num_decode > 0:
            assert torch.equal(out_sq[:num_decode], expected_full[:num_decode])
        n_local = local_ids.numel()
        assert torch.equal(
            out_sq[num_decode : num_decode + n_local],
            expected_full[num_decode:][local_ids],
        )


def test_indexpool_builder_builds_shardq_gather_indices(monkeypatch):
    """IndexPoolMetadataBuilder populates shardq_gather_indices only when
    use_shardq is enabled and the step splits under build_shardq_split."""
    monkeypatch.setattr("vllm.v1.utils.PIN_MEMORY", False)
    monkeypatch.setattr(
        shardq_mod,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=1),
    )
    layer_name = "layers.1.self_attn.indexer.k_cache"
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=8192,
            hf_text_config=SimpleNamespace(index_topk=2048),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        attention_config=SimpleNamespace(
            sparse_mla_shard_q_min_tokens=1024,
            sparse_mla_force_mqa=True,
        ),
        speculative_config=None,
        compilation_config=SimpleNamespace(
            static_forward_context={layer_name: SimpleNamespace(use_shardq=True)}
        ),
    )
    builder = IndexPoolMetadataBuilder(
        kv_cache_spec=SimpleNamespace(),
        layer_names=[layer_name],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )
    cm = create_common_attn_metadata(
        BatchSpec(seq_lens=[129, 2048], query_lens=[1, 2048]),
        block_size=256,
        device=torch.device("cpu"),
    )
    meta = builder.build(0, cm)
    assert meta.shardq_gather_indices is not None
    token_ids, counts = zigzag_split(np.array([2048]), 2)
    expected = torch.cat(
        (
            torch.tensor([0], dtype=torch.int64),
            torch.from_numpy(token_ids[counts[0] : counts[0] + counts[1]]) + 1,
        )
    )
    assert torch.equal(meta.shardq_gather_indices, expected)

    # Decode-only batch must not produce shardq_gather_indices.
    cm_decode = create_common_attn_metadata(
        BatchSpec(seq_lens=[129, 256], query_lens=[1, 1]),
        block_size=256,
        device=torch.device("cpu"),
    )
    assert builder.build(0, cm_decode).shardq_gather_indices is None


def test_shardq_workspaces_keyed_by_head_count(monkeypatch):
    """Heterogeneous local head counts (e.g. 16 sparse vs 32 SWA heads at TP2)
    must keep independent symmetric-memory workspaces."""
    created: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []

    class _FakeWorkspace:
        def __init__(self, group, device, num_ubatches, q, out):
            created.append((num_ubatches, q[0], out[0]))

    monkeypatch.setattr(shardq_mod, "_workspaces", {})
    monkeypatch.setattr(shardq_mod, "ShardQWorkspace", _FakeWorkspace)
    monkeypatch.setattr(shardq_mod, "direct_cp_enabled", lambda *a: True)
    monkeypatch.setattr(
        shardq_mod,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0, device_group=None),
    )
    monkeypatch.setattr(shardq_mod.torch.accelerator, "current_device_index", lambda: 0)
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(sparse_mla_shard_q_min_tokens=1024),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=16),
        parallel_config=SimpleNamespace(num_ubatches=1),
    )
    shardq_mod.allocate_shardq_buffers(
        vllm_config, 16, 576, 512, torch.bfloat16, torch.bfloat16
    )
    shardq_mod.allocate_shardq_buffers(
        vllm_config, 32, 576, 512, torch.bfloat16, torch.bfloat16
    )
    # Re-allocating an existing head count is a no-op.
    shardq_mod.allocate_shardq_buffers(
        vllm_config, 16, 576, 512, torch.bfloat16, torch.bfloat16
    )
    assert set(shardq_mod._workspaces.keys()) == {16, 32}
    assert len(created) == 2
    assert created[0][1][1] == 32  # world * 16
    assert created[1][1][1] == 64  # world * 32
