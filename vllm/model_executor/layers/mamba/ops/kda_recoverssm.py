# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factored KDA RecoverSSM verification shared by Kimi-K3 and GLM.

With D_t the per-channel decay accumulated before token t, the state after the
first t draft tokens is

    S_t = S_0 * D_t + sum_{i<t} u_i (x) (nk_i * D_t / D_{i+1}),

so every checkpoint read collapses into one contraction S_0 @ [e | f] with
e_t = D_{t+1} * nk_t and f_t = D_{t+1} * nq_t, and the corrections u_t follow
from a T x T triangular solve with A[i][t] = c_i . e_t and B[i][t] = c_i . f_t,
c_i = nk_i / D_{i+1}. The window vectors and triangular weights are batch-row
workspaces; only the correction and raw key/gate records survive until commit.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.worker.workspace import current_workspace_manager

# H200-tuned: the checkpoint slab is read in K chunks, so it never sits in
# registers whole.
VERIFY_K_CHUNK = 32
# Prep launch shape by padded window length. Up to BT=8 one K chunk (the whole
# tile) is fastest; at BT=16 the tile no longer fits and is walked in chunks.
PREP_K_CHUNK = {2: 128, 4: 128, 8: 128, 16: 64}
PREP_NUM_WARPS = {2: 1, 4: 1, 8: 1, 16: 2}
PREP_NUM_STAGES = 2


def _is_factored_kda_config_supported(
    state_dtype: torch.dtype,
    record_dtype: torch.dtype,
    head_dim: int,
    query_len: int,
    lower_bound: float | None,
) -> bool:
    capability = current_platform.get_device_capability()
    return (
        current_platform.is_cuda()
        and capability is not None
        and capability.major == 9
        and state_dtype == torch.float32
        and record_dtype == torch.bfloat16
        and head_dim == 128
        # Beyond 12 tokens at the -5 bound, the tf32x3 low half of e_t underflows.
        and 2 <= query_len <= 12
        and lower_bound is not None
        and -5.0 <= lower_bound < 0.0
    )


def is_factored_kda_supported(
    checkpoint: torch.Tensor,
    kg_cache: torch.Tensor,
    query_len: int,
    lower_bound: float | None,
) -> bool:
    return (
        checkpoint.shape[-2:] == (128, 128)
        and checkpoint.stride(-1) == 1
        and kg_cache.stride(-1) == 1
        and _is_factored_kda_config_supported(
            checkpoint.dtype,
            kg_cache.dtype,
            checkpoint.shape[-1],
            query_len,
            lower_bound,
        )
    )


def _factored_workspace_specs(batch: int, heads: int, dim: int, block_t: int):
    return (
        ((batch, heads, 2 * block_t, dim), torch.float32),
        ((batch, heads, block_t + 1, 2 * block_t), torch.float32),
    )


def reserve_factored_kda_workspace(
    max_batch: int,
    heads: int,
    dim: int,
    query_len: int,
    state_dtype: torch.dtype,
    record_dtype: torch.dtype,
    lower_bound: float | None,
) -> None:
    """Reserve eager-batch capacity before CUDA graph capture locks the workspace."""
    if max_batch > 1 and _is_factored_kda_config_supported(
        state_dtype, record_dtype, dim, query_len, lower_bound
    ):
        current_workspace_manager().get_simultaneous(
            *_factored_workspace_specs(
                max_batch, heads, dim, triton.next_power_of_2(query_len)
            )
        )


@triton.jit
def _kda_factored_prep_kernel(
    q_ptr,
    k_ptr,
    raw_g_ptr,
    raw_beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    kg_cache_ptr,
    ef_cache_ptr,
    w_cache_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    scale,
    lower_bound,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_g_token,
    stride_beta_token,
    stride_query_start_loc,
    stride_state_indices,
    stride_kg_block,
    stride_kg_head,
    stride_kg_pos,
    stride_ef_block,
    stride_ef_head,
    stride_ef_tok,
    stride_w_block,
    stride_w_head,
    stride_w_tok,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    KC: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if state_idx <= null_block_id:
        return
    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(
        tl.int64
    )
    offs_t = tl.arange(0, BT)
    offs_c = tl.arange(0, KC)
    valid_t = offs_t < eos - bos
    tokens = (bos + offs_t)[:, None]
    a = tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))
    kg_ptr = (
        kg_cache_ptr
        + state_idx * stride_kg_block
        + pid_h * stride_kg_head
        + offs_t[:, None] * stride_kg_pos
    )
    # Every row is written so padding tokens read back zeros, not stale scratch.
    ef_ptr = (
        ef_cache_ptr
        + pid_b * stride_ef_block
        + pid_h * stride_ef_head
        + offs_t[:, None] * stride_ef_tok
    )

    # One pass over K. The l2 norms are per-token scalars, so e / f / c are kept
    # unnormalized here; the norms scale A / B below and the projection columns
    # in the verify.
    k_sq = tl.zeros([BT], dtype=tl.float32)
    q_sq = tl.zeros([BT], dtype=tl.float32)
    k_dot_q = tl.zeros([BT], dtype=tl.float32)
    a_raw = tl.zeros([BT, BT], dtype=tl.float32)
    b_raw = tl.zeros([BT, BT], dtype=tl.float32)
    for kc in tl.range(0, BK, KC, num_stages=NUM_STAGES):
        cols = kc + offs_c
        mask_c = cols < K
        mask = valid_t[:, None] & mask_c[None, :]
        k = tl.load(
            k_ptr + tokens * stride_k_token + pid_h * K + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        raw_g = tl.load(
            raw_g_ptr + tokens * stride_g_token + pid_h * K + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        q = tl.load(
            q_ptr + tokens * stride_q_token + pid_h * K + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + pid_h * K + cols, mask=mask_c, other=0.0).to(
            tl.float32
        )
        tl.store(kg_ptr + cols[None, :], k, mask=mask)
        tl.store(kg_ptr + K + cols[None, :], raw_g, mask=mask)
        k_sq += tl.sum(k * k, axis=1)
        q_sq += tl.sum(q * q, axis=1)
        k_dot_q += tl.sum(k * q, axis=1)
        gate = lower_bound / (1.0 + tl.exp(-(a * (raw_g + dt_bias[None, :]))))
        # log D_{t+1}: token t's own decay is applied before its update.
        log_decay = tl.cumsum(tl.where(mask, gate, 0.0), axis=0)
        e = tl.exp(log_decay) * k
        f = tl.exp(log_decay) * q
        c = tl.exp(-log_decay) * k
        tl.store(ef_ptr + cols[None, :], e, mask=mask_c[None, :])
        tl.store(ef_ptr + BT * stride_ef_tok + cols[None, :], f, mask=mask_c[None, :])
        if BT >= 16:
            a_raw += tl.dot(c, tl.trans(e), input_precision="tf32x3")
            b_raw += tl.dot(c, tl.trans(f), input_precision="tf32x3")
        else:
            a_raw += tl.sum(c[:, None, :] * e[None, :, :], axis=2)
            b_raw += tl.sum(c[:, None, :] * f[None, :, :], axis=2)

    k_scale = 1.0 / tl.sqrt(k_sq + 1e-6)
    q_scale = scale / tl.sqrt(q_sq + 1e-6)
    raw_beta = tl.load(
        raw_beta_ptr + (bos + offs_t) * stride_beta_token + pid_h,
        mask=valid_t,
        other=0.0,
    ).to(tl.float32)
    beta = tl.sigmoid(raw_beta)
    a_mat = a_raw * k_scale[:, None] * k_scale[None, :]
    b_mat = b_raw * k_scale[:, None] * q_scale[None, :]
    # u_t is added after the decay, so its own output weight is nk_t . nq_t.
    diagonal = k_dot_q * k_scale * q_scale
    row = offs_t[:, None]
    col = offs_t[None, :]
    # Rows t < BT of W are [A[t] | B[t]], with sigmoid(beta_t) on A's unused
    # diagonal; row BT holds the column scales of the projection.
    w_ptr = w_cache_ptr + pid_b * stride_w_block + pid_h * stride_w_head
    tl.store(
        w_ptr + row * stride_w_tok + col,
        tl.where(row < col, a_mat, tl.where(row == col, beta[:, None], 0.0)),
    )
    tl.store(
        w_ptr + row * stride_w_tok + BT + col,
        tl.where(row < col, b_mat, tl.where(row == col, diagonal[:, None], 0.0)),
    )
    tl.store(w_ptr + BT * stride_w_tok + offs_t, k_scale)
    tl.store(w_ptr + BT * stride_w_tok + BT + offs_t, q_scale)


@triton.jit
def _tf32_hi(x):
    return tl.inline_asm_elementwise(
        "cvt.rna.tf32.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _tf32_lo(x):
    # x - tf32(x) as a single op, so the split stays in the dot operand layout.
    return tl.inline_asm_elementwise(
        "{ .reg .b32 t; cvt.rna.tf32.f32 t, $1; sub.f32 $0, $1, t; }",
        "=r,r",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _kda_factored_verify_kernel(
    v_ptr,
    state_ptr,
    correction_cache_ptr,
    ef_cache_ptr,
    w_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    null_block_id,
    stride_v_token,
    stride_query_start_loc,
    stride_state_indices,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_correction_block,
    stride_correction_head,
    stride_correction_pos,
    stride_ef_block,
    stride_ef_head,
    stride_ef_tok,
    stride_w_block,
    stride_w_head,
    stride_w_tok,
    stride_out_token,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    EF_WIDTH: tl.constexpr,
    BKC: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)

    offs_e = tl.arange(0, EF_WIDTH)
    offs_t = tl.arange(0, BT)
    offs_v = pid_v * BV + tl.arange(0, BV)
    # Column t < BT is token t's e-projection, column BT + t its f-projection.
    in_window = offs_e < 2 * BT
    if state_idx <= null_block_id:
        for t in tl.static_range(SPEC_QUERY_LEN):
            tl.store(
                out_ptr + (bos + t) * stride_out_token + pid_h * V + offs_v,
                tl.zeros([BV], dtype=tl.float32),
                mask=(t < query_len) & (offs_v < V),
            )
        return

    state_base = state_ptr + state_idx * stride_state_block + pid_h * stride_state_head
    ef_base = ef_cache_ptr + pid_b * stride_ef_block + pid_h * stride_ef_head
    mask_v = offs_v < V
    v = tl.load(
        v_ptr + (bos + offs_e)[None, :] * stride_v_token + pid_h * V + offs_v[:, None],
        mask=mask_v[:, None] & (offs_e < query_len)[None, :],
        other=0.0,
    ).to(tl.float32)
    w_base = w_cache_ptr + pid_b * stride_w_block + pid_h * stride_w_head
    w_all = tl.load(
        w_base + offs_t[:, None] * stride_w_tok + offs_e[None, :],
        mask=in_window[None, :],
        other=0.0,
    )
    # e / f are stored unnormalized; this applies the per-token l2 normalization
    # (and q's 1/sqrt(K) scale) to their projection columns.
    column_scale = tl.load(
        w_base + BT * stride_w_tok + offs_e, mask=in_window, other=0.0
    )
    # S_0 @ [e | f] in K chunks as tf32x3, split by hand. With
    # input_precision="tf32x3", or with a software-pipelined loop, Triton 3.8
    # stages the split halves through extra shared memory, which doubles the
    # verify time on H200.
    offs_kc = tl.arange(0, BKC)
    projection = tl.zeros([BV, EF_WIDTH], dtype=tl.float32)
    # Full unrolling fits short windows once their unused columns are removed.
    for kc in tl.range(
        0,
        BK,
        BKC,
        num_stages=1,
        loop_unroll_factor=4 if BT <= 4 else (2 if BT <= 8 else 1),
    ):
        state = tl.load(
            state_base + offs_v[:, None] * stride_state_v + (kc + offs_kc)[None, :],
            mask=mask_v[:, None] & ((kc + offs_kc) < K)[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        # Token-major in memory so prep stores stay coalesced along K.
        ef = tl.load(
            ef_base + offs_e[None, :] * stride_ef_tok + (kc + offs_kc)[:, None],
            mask=((kc + offs_kc) < K)[:, None] & in_window[None, :],
            other=0.0,
        )
        state_hi = _tf32_hi(state)
        ef_hi = _tf32_hi(ef)
        projection = tl.dot(_tf32_lo(state), ef_hi, projection, input_precision="tf32")
        projection = tl.dot(state_hi, _tf32_lo(ef), projection, input_precision="tf32")
        projection = tl.dot(state_hi, ef_hi, projection, input_precision="tf32")
    correction_ptr = (
        correction_cache_ptr
        + state_idx * stride_correction_block
        + pid_h * stride_correction_head
    )
    # residual[:, t] = v_t - S_t' . nk_t and residual[:, BT + t] = -o_t once every
    # earlier u_i has been folded in through its row of W, so the triangular
    # solve never goes back to memory.
    residual = v - projection * column_scale[None, :]
    for t in tl.static_range(SPEC_QUERY_LEN):
        valid = t < query_len
        w = tl.sum(tl.where(offs_t[:, None] == t, w_all, 0.0), axis=0)
        # w[t] is sigmoid(beta_t).
        u = tl.sum(tl.where(offs_e[None, :] == t, residual * w[None, :], 0.0), axis=1)
        u = tl.where(valid, u, 0.0)
        residual -= u[:, None] * w[None, :]
        out = -tl.sum(tl.where(offs_e[None, :] == BT + t, residual, 0.0), axis=1)
        tl.store(
            out_ptr + (bos + t) * stride_out_token + pid_h * V + offs_v,
            out,
            mask=valid & mask_v,
        )
        tl.store(
            correction_ptr + t * stride_correction_pos + offs_v,
            u,
            mask=valid & mask_v,
        )


def factored_kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify from a read-only checkpoint using batch-sized scratch.

    Persistent records remain raw key/gate in activation dtype, as in
    the shared Kimi-K3/GLM RecoverSSM interface.
    """
    _, total_tokens, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    batch = state_indices.shape[0]
    if out is None:
        out = torch.empty_like(v)
    if total_tokens == 0 or batch == 0:
        return out
    assert correction_cache.dtype == torch.float32
    assert kg_cache.shape[-2:] == (spec_query_len, 2 * key_dim)
    assert checkpoint_state.stride(-1) == 1 and kg_cache.stride(-1) == 1
    assert correction_cache.stride(-1) == 1
    block_t = triton.next_power_of_2(spec_query_len)
    ef_width = 2 * block_t
    ef_cache, w_cache = current_workspace_manager().get_simultaneous(
        *_factored_workspace_specs(batch, num_heads, key_dim, block_t)
    )
    block_k = triton.next_power_of_2(key_dim)
    _kda_factored_prep_kernel[(batch, num_heads)](
        q,
        k,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        kg_cache,
        ef_cache,
        w_cache,
        query_start_loc,
        state_indices,
        key_dim**-0.5,
        lower_bound,
        NULL_BLOCK_ID,
        q.stride(1),
        k.stride(1),
        raw_g.stride(1),
        raw_beta.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
        kg_cache.stride(0),
        kg_cache.stride(1),
        kg_cache.stride(2),
        ef_cache.stride(0),
        ef_cache.stride(1),
        ef_cache.stride(2),
        w_cache.stride(0),
        w_cache.stride(1),
        w_cache.stride(2),
        K=key_dim,
        BK=block_k,
        BT=block_t,
        KC=min(PREP_K_CHUNK[block_t], block_k),
        NUM_STAGES=PREP_NUM_STAGES,
        num_warps=PREP_NUM_WARPS[block_t],
    )
    # Wider V tiles once there are enough programs to fill the GPU; longer
    # windows amortize their larger triangular solve over more rows.
    block_v = (64 if block_t >= 8 else 32) if batch * num_heads >= 384 else 16
    _kda_factored_verify_kernel[(triton.cdiv(value_dim, block_v), batch, num_heads)](
        v,
        checkpoint_state,
        correction_cache,
        ef_cache,
        w_cache,
        out,
        query_start_loc,
        state_indices,
        NULL_BLOCK_ID,
        v.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        correction_cache.stride(0),
        correction_cache.stride(1),
        correction_cache.stride(2),
        ef_cache.stride(0),
        ef_cache.stride(1),
        ef_cache.stride(2),
        w_cache.stride(0),
        w_cache.stride(1),
        w_cache.stride(2),
        out.stride(1),
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        BT=block_t,
        SPEC_QUERY_LEN=spec_query_len,
        EF_WIDTH=ef_width,
        BKC=min(VERIFY_K_CHUNK, block_k),
        num_warps=1,
    )
    return out
