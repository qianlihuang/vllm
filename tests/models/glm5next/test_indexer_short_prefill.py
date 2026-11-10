# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import Glm5NextConfig, Glm5NextTextConfig

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import (
    CacheConfig,
    CUDAGraphMode,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.models.glm5next.common.attention import Indexer
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
)
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA producer")


@torch.inference_mode()
def test_short_prefill_preserves_caches_and_refreshes_graph_metadata(
    dist_init, tmp_path
):
    torch.manual_seed(11)
    text_config = Glm5NextTextConfig(
        hidden_size=256,
        q_lora_rank=64,
        index_n_heads=16,
        index_head_dim=128,
        index_topk=64,
        index_kpool=4,
        qk_rope_head_dim=0,
    )
    Glm5NextConfig(
        text_config=text_config.to_dict(),
        architectures=["Glm5NextForConditionalGeneration"],
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=4096,
        ),
        cache_config=CacheConfig(block_size=128),
        kernel_config={"sparse_indexer_topk_backend": "torch"},
    )
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        indexer = Indexer(
            config,
            text_config,
            256,
            64,
            None,
            config.cache_config,
            torch.empty(8, 68, dtype=torch.int32),
            prefix="indexer",
        )
        for parameter in indexer.parameters():
            parameter.normal_(std=0.1)
        indexer.process_weights_after_loading()
        indexer.k_cache.kv_cache = torch.zeros(4, 32, 132, dtype=torch.uint8)
        indexer.tail_cache.kv_cache = torch.zeros(4, 2, 4, 128)
        hidden = torch.randn(8, 256)
        qr = torch.randn(8, 64)
        positions = torch.arange(8, dtype=torch.int64)

    def metadata(length, *, known=True, shardq=False):
        pos = torch.arange(length - 8, length, device="cuda")
        compressed = length // 4
        cumulative = torch.tensor([0, compressed], dtype=torch.int32, device="cuda")
        table = torch.tensor([[1]], dtype=torch.int32, device="cuda")
        chunk = DeepseekV32IndexerPrefillChunkMetadata(
            block_table=table,
            cu_seqlen_ks=torch.zeros(8, dtype=torch.int32, device="cuda"),
            cu_seqlen_ke=((pos + 1) // 4).int(),
            cu_seq_lens=cumulative,
            token_to_seq=torch.zeros(compressed, dtype=torch.int32, device="cuda"),
            total_seq_lens=compressed,
            max_seqlen_k=compressed,
            token_start=0,
            token_end=8,
            num_reqs=1,
            local_cu_seq_lens=cumulative,
            local_total_seq_lens=compressed,
            max_local_total_seq_lens=compressed,
        )
        batch = DeepseekV32IndexerMetadata(
            seq_lens=torch.tensor([compressed], dtype=torch.int32, device="cuda"),
            max_seq_len=length,
            slot_mapping=torch.where(pos % 4 == 3, 32 + pos // 4, -1),
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=1,
            num_prefill_tokens=8,
            prefill=DeepseekV32IndexerPrefillMetadata(
                chunks=[chunk],
                max_prefill_seq_len=length if known else -1,
                shardq_gather_indices=torch.arange(8, device="cuda")
                if shardq
                else None,
            ),
        )
        tail = DeepseekV32IndexerMetadata(
            seq_lens=batch.seq_lens,
            max_seq_len=length,
            slot_mapping=torch.where(pos >= length - 4, 4 + pos % 4, -1),
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=1,
            num_prefill_tokens=8,
        )
        return {indexer.k_cache.prefix: batch, indexer.tail_cache.prefix: tail}

    def run(batch, mode=CUDAGraphMode.NONE):
        with (
            set_current_vllm_config(config),
            set_forward_context(
                batch,
                config,
                cudagraph_runtime_mode=mode,
            ),
        ):
            return indexer(hidden, qr, positions, None)

    q_calls = []
    handle = indexer.wq_b.register_forward_hook(lambda *_: q_calls.append(1))
    reset_workspace_manager()
    init_workspace_manager(torch.device("cuda"))
    try:
        short = metadata(8, shardq=True)
        reference = run(short, CUDAGraphMode.FULL).clone()
        expected_k = indexer.k_cache.kv_cache.clone()
        expected_tail = indexer.tail_cache.kv_cache.clone()
        indexer.k_cache.kv_cache.zero_()
        indexer.tail_cache.kv_cache.zero_()
        q_calls.clear()
        actual = run(short)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(indexer.k_cache.kv_cache, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(
            indexer.tail_cache.kv_cache, expected_tail, rtol=0, atol=0
        )
        assert not q_calls

        # Populate the cached prefix with actual keys, then reuse an eight-row
        # capture for a chunk whose complete context exceeds top-k.
        for length in range(8, 65, 8):
            positions.copy_(torch.arange(length - 8, length, device="cuda"))
            run(metadata(length))
        initial_k = indexer.k_cache.kv_cache.clone()
        initial_tail = indexer.tail_cache.kv_cache.clone()
        batches = [short, metadata(72), metadata(8, known=False)]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with set_current_vllm_config(config), torch.cuda.stream(stream):
            for batch in batches:
                length = batch[indexer.k_cache.prefix].max_seq_len
                positions.copy_(torch.arange(length - 8, length, device="cuda"))
                run(batch)
            positions.copy_(torch.arange(8, device="cuda"))
            capture = BreakableCUDAGraphCapture()
            with (
                set_forward_context(
                    short,
                    config,
                    cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                ),
                capture,
            ):
                captured_out = indexer(hidden, qr, positions, None).clone()
            assert capture.num_eager_breaks == 1
            for batch in batches[1:] + batches[:1]:
                length = batch[indexer.k_cache.prefix].max_seq_len
                positions.copy_(torch.arange(length - 8, length, device="cuda"))
                indexer.k_cache.kv_cache.copy_(initial_k)
                indexer.tail_cache.kv_cache.copy_(initial_tail)
                expected = run(batch, CUDAGraphMode.FULL).clone()
                expected_k = indexer.k_cache.kv_cache.clone()
                expected_tail = indexer.tail_cache.kv_cache.clone()
                indexer.k_cache.kv_cache.copy_(initial_k)
                indexer.tail_cache.kv_cache.copy_(initial_tail)
                q_calls.clear()
                with set_forward_context(
                    batch,
                    config,
                    cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                ):
                    capture.replay()
                torch.testing.assert_close(captured_out, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    indexer.k_cache.kv_cache, expected_k, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    indexer.tail_cache.kv_cache, expected_tail, rtol=0, atol=0
                )
                assert len(q_calls) == (0 if batch is short else 1)

            # FULL capture keeps the original producer, with no eager breaks.
            positions.copy_(torch.arange(8, device="cuda"))
            run(short, CUDAGraphMode.FULL)
            full = BreakableCUDAGraphCapture()
            with (
                set_forward_context(
                    short, config, cudagraph_runtime_mode=CUDAGraphMode.FULL
                ),
                full,
            ):
                full_out = indexer(hidden, qr, positions, None).clone()
            assert full.num_eager_breaks == 0
            hidden.mul_(0.5)
            indexer.k_cache.kv_cache.copy_(initial_k)
            indexer.tail_cache.kv_cache.copy_(initial_tail)
            expected = run(short, CUDAGraphMode.FULL).clone()
            expected_k = indexer.k_cache.kv_cache.clone()
            expected_tail = indexer.tail_cache.kv_cache.clone()
            indexer.k_cache.kv_cache.copy_(initial_k)
            indexer.tail_cache.kv_cache.copy_(initial_tail)
            full.replay()
            torch.testing.assert_close(full_out, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                indexer.k_cache.kv_cache, expected_k, rtol=0, atol=0
            )
            torch.testing.assert_close(
                indexer.tail_cache.kv_cache, expected_tail, rtol=0, atol=0
            )
        torch.cuda.current_stream().wait_stream(stream)
    finally:
        handle.remove()
        reset_workspace_manager()
