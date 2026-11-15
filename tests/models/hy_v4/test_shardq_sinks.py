# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.multiprocessing as mp

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import finalize_model_weights
from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention, Indexer
from vllm.models.hy_v4.nvidia.flashmla_sparse import HYV4FlashMLASparseImpl
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.hy_v4 import HYV4Config
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager


def _check_sinks(rank, world, num_heads, offload, port, model_path):
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)
    config = VllmConfig(
        model_config=ModelConfig(
            model=model_path, dtype="bfloat16", skip_tokenizer_init=True
        ),
        cache_config=CacheConfig(cache_dtype="fp8_ds_mla"),
        parallel_config=ParallelConfig(tensor_parallel_size=world),
        scheduler_config=SchedulerConfig(
            max_model_len=8192,
            is_encoder_decoder=False,
            max_num_batched_tokens=128,
            max_num_seqs=8,
        ),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend="nccl",
        )
        ensure_model_parallel_initialized(world, 1)
        init_workspace_manager(device)
        heads = num_heads // world
        with set_default_torch_dtype(torch.bfloat16), torch.device(device):
            layer = HYV4MLAAttention(
                vllm_config=config,
                config=config.model_config.hf_text_config,
                hidden_size=256,
                num_heads=num_heads,
                q_lora_rank=128,
                kv_lora_rank=512,
                qk_nope_head_dim=192,
                qk_rope_head_dim=64,
                v_head_dim=256,
                cache_config=config.cache_config,
                prefix="model.layers.0.self_attn",
                topk_indices_buffer=torch.empty(128, 2048, dtype=torch.int32),
            )
        sinks = layer.learnable_sink_param
        impl = layer.mla_attn.impl
        assert isinstance(impl, HYV4FlashMLASparseImpl)
        assert layer.mla_attn.use_shardq == (world > 1)
        local_query = torch.zeros(2, 1, heads, 576, device=device)
        full_query = torch.zeros(2, 1, num_heads, 576, device=device)
        graph = None
        for offset in (0, 2):
            if offload:
                layer.cpu()
            checkpoint_sinks = torch.arange(num_heads, device=device).float() + offset
            sinks.weight_loader(sinks, checkpoint_sinks.to(sinks.device))
            finalize_model_weights(layer, device)
            layer.to(device)
            local = impl._sinks_for_query(local_query, 2, 64)
            assert local is not None
            torch.testing.assert_close(
                local[:heads], checkpoint_sinks[rank * heads : (rank + 1) * heads]
            )
            assert torch.isneginf(local[heads:]).all()
            full = impl._sinks_for_query(full_query, 2, 64)
            assert full is not None
            torch.testing.assert_close(full[:num_heads], checkpoint_sinks)
            assert torch.isneginf(full[num_heads:]).all()
            if graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_sinks = impl._sinks_for_query(full_query, 2, 64)
                    assert graph_sinks is not None
                    captured_sinks = graph_sinks.clone()
            graph.replay()
            torch.testing.assert_close(captured_sinks[:num_heads], checkpoint_sinks)
            assert torch.isneginf(captured_sinks[num_heads:]).all()
        with pytest.raises(ValueError, match="head count must match"):
            impl._sinks_for_query(full_query[:, :, :3], 2, 64)
        reset_workspace_manager()
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize(
    "world,num_heads,offload", [(1, 64, False), (2, 32, True), (8, 64, False)]
)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_local_and_shardq_sinks_follow_checkpoint_reload(
    world, num_heads, offload, tmp_path
):
    if torch.accelerator.device_count() < world:
        pytest.skip(f"Requires {world} GPUs")
    config = HYV4Config(
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        q_lora_rank=128,
        max_position_embeddings=8192,
        layer_types=["deepseek_sparse_attention"],
    )
    config.architectures = ["HYV4ForCausalLM"]
    config.save_pretrained(tmp_path)
    mp.spawn(
        _check_sinks,
        args=(world, num_heads, offload, get_open_port(), str(tmp_path)),
        nprocs=world,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("native", [False, True])
@torch.inference_mode()
def test_indexer_preparation_matches_rotary_backend(
    tmp_path, dist_init, scaled, native
):
    hf_config = HYV4Config(
        hidden_size=256,
        num_hidden_layers=1,
        q_lora_rank=128,
        index_n_heads=32,
        index_head_dim=128,
        qk_rope_head_dim=64,
        max_position_embeddings=4096,
    )
    hf_config.architectures = ["HYV4ForCausalLM"]
    hf_config.save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True
        ),
        cache_config=CacheConfig(cache_dtype="fp8_ds_mla"),
        compilation_config=CompilationConfig(custom_ops=["none" if native else "all"]),
        scheduler_config=SchedulerConfig(
            max_model_len=4096,
            is_encoder_decoder=False,
            max_num_batched_tokens=32,
            max_num_seqs=4,
        ),
    )
    torch.manual_seed(42)
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        indexer = Indexer(
            config,
            hf_config,
            256,
            128,
            None,
            config.cache_config,
            None,
            prefix="indexer",
        )
        rotary_kwargs = dict(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=4096,
            base=10000.0,
            is_neox_style=False,
            dtype=torch.bfloat16,
        )
        if scaled:
            rotary = DeepseekScalingRotaryEmbedding(**rotary_kwargs, scaling_factor=2.0)
        else:
            rotary = RotaryEmbedding(**rotary_kwargs)
        for parameter in indexer.parameters():
            parameter.normal_(std=0.02)
        hidden = torch.randn(3, 256)
        qr = torch.randn(3, 128)
        positions = torch.tensor([0, 7, 31], dtype=torch.int64)
        q = indexer.wq_b(qr)[0].view(3, 32, 128)
        kw = indexer.wk_weights_proj(hidden)[0]
        k = indexer.k_norm(kw[:, :128])
        q_pe, k_pe = rotary(positions, q[..., 64:], k[:, None, 64:])
        q = torch.cat([q[..., :64], q_pe], dim=-1)
        expected_k = torch.cat([k[:, :64], k_pe.squeeze(1)], dim=-1)
        expected_q, scale = per_token_group_quant_fp8(
            q.view(-1, 128), 128, use_ue8m0=True
        )
        expected_weights = kw[:, 128:].float() * scale.view(3, 32)
        expected_weights = expected_weights * 128**-0.5 * 32**-0.5
        _, actual_q, actual_k, actual_weights = indexer.prepare_inputs(
            hidden, qr, positions, rotary
        )
    assert torch.equal(
        actual_q.view(torch.int8), expected_q.view(3, 32, 128).view(torch.int8)
    )
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
