# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HY4-Flash iHC residual SP, FlashEngram SP, and passthrough MoE SP."""

import pytest
import torch
from torch import nn

from vllm.models.hy_v4_flash.nvidia import engram as engram_mod
from vllm.models.hy_v4_flash.nvidia import model as flash_model_mod
from vllm.models.hy_v4_flash.nvidia import moe as flash_moe_mod


def _fake_sp_shard(x: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    pad = (-x.shape[0]) % world
    if pad:
        x = torch.nn.functional.pad(x, (0, 0) * (x.ndim - 1) + (0, pad))
    chunk = x.shape[0] // world
    return x[rank * chunk : (rank + 1) * chunk]


def _make_norm(dim: int) -> nn.Module:
    class _RMS(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (
                x
                * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt().to(x.dtype)
                * self.weight
            )

    return _RMS(dim)


@pytest.mark.parametrize("num_tokens", [8, 13])
def test_flash_engram_sp_matches_replicated(monkeypatch, num_tokens):
    """FlashEngram SP slices invalid-hash masks and reduce-scatters key/value
    to match replicated execution on both even and odd token counts."""
    torch.manual_seed(0)
    world, hc_mult, hidden_size, head_dim = 2, 4, 16, 4
    num_heads = 2
    full_hidden = torch.randn(num_tokens, hc_mult, hidden_size)
    hashes = torch.randint(0, 32, (num_tokens, num_heads))
    hashes[0] = -1
    hashes[-1] = -1
    # Full (tp=1) lookup rows of shape [T, num_heads, head_dim]
    full_rows = torch.randn(num_tokens, num_heads, head_dim)
    key_weights = [
        torch.randn(hidden_size, num_heads * head_dim) for _ in range(hc_mult)
    ]
    key_biases = [torch.randn(hidden_size) for _ in range(hc_mult)]
    val_weight = torch.randn(hidden_size, num_heads * head_dim)
    val_bias = torch.randn(hidden_size)
    head_gammas = [torch.rand(head_dim) + 0.5 for _ in range(num_heads)]
    pre_gamma = torch.rand(hidden_size) + 0.5
    per_hc_scales = [torch.tensor([0.25 * (c + 1)]) for c in range(hc_mult)]

    def build_engram(rank: int, tp: int) -> engram_mod.FlashEngram:
        mod = engram_mod.FlashEngram.__new__(engram_mod.FlashEngram)
        nn.Module.__init__(mod)
        mod.rank = rank
        mod.hc_mult = hc_mult
        mod.hidden_size = hidden_size
        heads_per_rank = num_heads // tp
        h0, h1 = rank * heads_per_rank, (rank + 1) * heads_per_rank
        c0, c1 = h0 * head_dim, h1 * head_dim

        class _Embed(nn.Module):
            def __init__(self):
                super().__init__()
                self.tp_size = tp
                self.head_start = h0
                self.head_end = h1

            def forward(self, h):
                return full_rows[:, h0:h1, :]

        mod.multi_head_embedding = _Embed()
        mod.lookup_per_head_norms = nn.ModuleList()
        for h in range(h0, h1):
            n = nn.Module()
            n.weight = nn.Parameter(head_gammas[h].clone())
            mod.lookup_per_head_norms.append(n)
        mod.key_projs = nn.ModuleList()
        for c in range(hc_mult):
            lin = nn.Linear(c1 - c0, hidden_size, bias=True)
            lin.weight.data.copy_(key_weights[c][:, c0:c1])
            lin.bias.data.copy_(
                key_biases[c] if rank == 0 else torch.zeros(hidden_size)
            )
            mod.key_projs.append(lin)
        mod.value_proj = nn.Linear(c1 - c0, hidden_size, bias=True)
        mod.value_proj.weight.data.copy_(val_weight[:, c0:c1])
        mod.value_proj.bias.data.copy_(
            val_bias if rank == 0 else torch.zeros(hidden_size)
        )
        mod.norm_key = nn.ModuleList([_make_norm(hidden_size) for _ in range(hc_mult)])
        mod.norm_query = nn.ModuleList(
            [_make_norm(hidden_size) for _ in range(hc_mult)]
        )
        mod.unified_pre_gamma = nn.Parameter(pre_gamma.clone())
        mod.unified_per_hc_scales = nn.ParameterList(
            [nn.Parameter(s.clone()) for s in per_hc_scales]
        )
        return mod

    ref_mod = build_engram(0, 1)
    expected = ref_mod(full_hidden, hashes)

    mods = [build_engram(r, world) for r in range(world)]
    # Precompute partial sums across both TP ranks to simulate reduce_scatter.
    captured_keys: list[list[torch.Tensor]] = [[], []]
    captured_vals: list[torch.Tensor] = []
    for r, m in enumerate(mods):
        rows = m.multi_head_embedding(hashes)
        eps = torch.finfo(torch.float32).eps
        gamma = torch.stack([n.weight for n in m.lookup_per_head_norms]).float()
        lh = hashes[
            :, m.multi_head_embedding.head_start : m.multi_head_embedding.head_end
        ]
        x = rows.masked_fill(lh.unsqueeze(-1) < 0, 0)
        x = (x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * gamma).flatten(1)
        for c in range(hc_mult):
            captured_keys[r].append(m.key_projs[c](x))
        captured_vals.append(m.value_proj(x))

    summed_keys = [captured_keys[0][c] + captured_keys[1][c] for c in range(hc_mult)]
    summed_val = captured_vals[0] + captured_vals[1]

    rank_outputs = []
    for r, m in enumerate(mods):
        call_idx = {"i": 0}

        def _fake_rs(tensor: torch.Tensor, rank=r, counter=call_idx) -> torch.Tensor:
            idx = counter["i"]
            counter["i"] += 1
            full = summed_keys[idx] if idx < hc_mult else summed_val
            return _fake_sp_shard(full, rank, world)

        monkeypatch.setattr(engram_mod, "sp_reduce_scatter", _fake_rs)
        sharded_hidden = _fake_sp_shard(full_hidden, r, world)
        rank_outputs.append(m(sharded_hidden, hashes))

    gathered = torch.cat(rank_outputs, dim=0)[:num_tokens]
    torch.testing.assert_close(gathered, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_tokens", [8, 13])
def test_passthrough_moe_sp_matches_replicated(monkeypatch, num_tokens):
    """HYV4FlashMoE.forward_sp_passthrough matches replicated forward while
    running routed experts on local [T_local, D] shards."""
    torch.manual_seed(1)
    world, dim = 2, 16
    x_full = torch.randn(num_tokens, dim)
    routed_w = torch.randn(dim, dim)
    shared_w = torch.randn(dim, dim)
    norm = _make_norm(dim)

    def make_moe(rank: int) -> flash_moe_mod.HYV4FlashMoE:
        m = flash_moe_mod.HYV4FlashMoE.__new__(flash_moe_mod.HYV4FlashMoE)
        nn.Module.__init__(m)
        m.tp_size = world
        m.apply_post_norm = True
        m.separate_shared_experts = True
        m.combine_output_norm = norm
        # Simulate TP-sharded shared_experts whose partial outputs sum across ranks.
        m.shared_experts = lambda h, r=rank: (h @ shared_w.T) * (0.4 if r == 0 else 0.6)
        m.gate = lambda h: (h, None)
        m.experts = lambda hidden_states, router_logits: hidden_states @ routed_w.T
        return m

    expected = norm(x_full @ routed_w.T + x_full @ shared_w.T)

    # Simulate rank 0 & rank 1 running forward_sp_passthrough.
    routed_shapes = []
    out_shards = []
    for r in range(world):
        m = make_moe(r)
        orig_experts = m.experts

        def _track_experts(hidden_states, router_logits, fn=orig_experts):
            routed_shapes.append(hidden_states.shape[0])
            return fn(hidden_states, router_logits)

        m.experts = _track_experts
        monkeypatch.setattr(flash_moe_mod, "sp_all_gather", lambda h: x_full)
        monkeypatch.setattr(
            flash_moe_mod,
            "sp_reduce_scatter",
            lambda h, rank=r: _fake_sp_shard(x_full @ shared_w.T, rank, world),
        )
        x_local = _fake_sp_shard(x_full, r, world)
        out_shards.append(m.forward_sp_passthrough(x_local, num_tokens))

    expected_local = (num_tokens + world - 1) // world
    assert routed_shapes == [expected_local, expected_local]
    gathered = torch.cat(out_shards, dim=0)[:num_tokens]
    torch.testing.assert_close(gathered, expected, atol=1e-5, rtol=1e-5)


def test_decoder_layer_ihc_sp_runs_post_norms_on_sharded_rows(monkeypatch):
    """HYV4FlashDecoderLayer._forward_ihc applies post-norms after
    sp_reduce_scatter so post-norms execute on T_local rows."""
    torch.manual_seed(2)
    world, full_tokens, dim = 2, 10, 8
    local_tokens = full_tokens // world
    observed_rows: list[tuple[str, int]] = []

    class _IdentityHC(nn.Module):
        def prepare_input(self, x):
            return x

        def pre(self, x):
            observed_rows.append(("hc_pre", x.shape[0]))
            return x, None, x

        def post(self, x, residual, gates):
            observed_rows.append(("hc_post", x.shape[0]))
            return x + residual

    def _recording_norm(tag: str):
        def _fn(x):
            observed_rows.append((tag, x.shape[0]))
            return x

        return _fn

    layer = flash_model_mod.HYV4FlashDecoderLayer.__new__(
        flash_model_mod.HYV4FlashDecoderLayer
    )
    nn.Module.__init__(layer)
    layer.defer_attn_reduction = True
    layer.defer_mlp_reduction = True
    layer.use_passthrough_moe = False
    layer.hc_attn_layer = _IdentityHC()
    layer.hc_mlp_layer = _IdentityHC()
    layer.input_layernorm = _recording_norm("input_ln")
    layer.post_attention_layernorm = _recording_norm("post_attn_ln")
    layer.self_attn = lambda positions, hidden_states: hidden_states
    layer.self_attn.post_proj_norm = _recording_norm("attn_post_norm")
    layer.mlp = lambda hidden_states: hidden_states
    layer.mlp.combine_output_norm = _recording_norm("mlp_post_norm")

    monkeypatch.setattr(
        flash_model_mod,
        "sp_all_gather",
        lambda x: torch.cat([x, x], dim=0),
    )
    monkeypatch.setattr(
        flash_model_mod,
        "sp_reduce_scatter",
        lambda x: x[:local_tokens],
    )

    positions = torch.arange(full_tokens)
    hidden_local = torch.randn(local_tokens, dim)
    out, _ = layer._forward_ihc(positions, hidden_local)
    assert out.shape == (local_tokens, dim)
    for tag, rows in observed_rows:
        assert rows == local_tokens, (
            f"{tag} received {rows} rows, expected {local_tokens}"
        )
