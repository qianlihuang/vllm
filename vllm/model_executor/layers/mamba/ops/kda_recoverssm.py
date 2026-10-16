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

MIN_DOT_COLS = 16
_PREP_NUM_WARPS = 1
_VERIFY_BKC = 32
_VERIFY_NUM_STAGES = 3


def is_factored_kda_supported(
    checkpoint: torch.Tensor,
    kg_cache: torch.Tensor,
    query_len: int,
    lower_bound: float | None,
) -> bool:
    capability = current_platform.get_device_capability()
    return (
        current_platform.is_cuda()
        and capability is not None
        and capability.major == 9
        and checkpoint.dtype == torch.float32
        and checkpoint.shape[-2:] == (128, 128)
        and checkpoint.stride(-1) == 1
        and kg_cache.dtype == torch.bfloat16
        and kg_cache.stride(-1) == 1
        and 2 <= query_len <= 8
        and lower_bound is not None
        and -5.0 <= lower_bound < 0.0
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
    offs_k = tl.arange(0, BK)
    mask_k = offs_k < K
    valid_t = offs_t < eos - bos
    mask = valid_t[:, None] & mask_k[None, :]
    tokens = (bos + offs_t)[:, None]
    k = tl.load(
        k_ptr + tokens * stride_k_token + pid_h * K + offs_k[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    raw_g = tl.load(
        raw_g_ptr + tokens * stride_g_token + pid_h * K + offs_k[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    q = tl.load(
        q_ptr + tokens * stride_q_token + pid_h * K + offs_k[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    raw_beta = tl.load(
        raw_beta_ptr + (bos + offs_t) * stride_beta_token + pid_h,
        mask=valid_t,
        other=0.0,
    ).to(tl.float32)
    a = tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0).to(
        tl.float32
    )

    kg_ptr = (
        kg_cache_ptr
        + state_idx * stride_kg_block
        + pid_h * stride_kg_head
        + offs_t[:, None] * stride_kg_pos
    )
    tl.store(kg_ptr + offs_k[None, :], k, mask=mask)
    tl.store(kg_ptr + K + offs_k[None, :], raw_g, mask=mask)

    normalized_k = k / tl.sqrt(tl.sum(k * k, axis=1) + 1e-6)[:, None]
    normalized_q = q / tl.sqrt(tl.sum(q * q, axis=1) + 1e-6)[:, None] * scale
    gate = lower_bound / (1.0 + tl.exp(-(a * (raw_g + dt_bias[None, :]))))
    # log D_{t+1}: token t's own decay is applied before its update.
    log_decay = tl.cumsum(tl.where(mask, gate, 0.0), axis=0)
    e = tl.exp(log_decay) * normalized_k
    f = tl.exp(log_decay) * normalized_q
    c = tl.exp(-log_decay) * normalized_k
    # Every row is written so padding tokens read back zeros, not stale scratch.
    ef_ptr = (
        ef_cache_ptr
        + pid_b * stride_ef_block
        + pid_h * stride_ef_head
        + offs_t[:, None] * stride_ef_tok
        + offs_k[None, :]
    )
    tl.store(ef_ptr, e, mask=mask_k[None, :])
    tl.store(ef_ptr + BT * stride_ef_tok, f, mask=mask_k[None, :])

    a_mat = tl.sum(c[:, None, :] * e[None, :, :], axis=2)
    b_mat = tl.sum(c[:, None, :] * f[None, :, :], axis=2)
    # u_t is added after the decay, so its own output weight is nk_t . nq_t.
    diagonal = tl.sum(normalized_k * normalized_q, axis=1)
    beta = tl.sigmoid(raw_beta)
    row = offs_t[:, None]
    col = offs_t[None, :]
    # Row t of W is [A[t] | B[t]]; A's unused diagonal carries sigmoid(beta_t).
    w_ptr = (
        w_cache_ptr
        + pid_b * stride_w_block
        + pid_h * stride_w_head
        + row * stride_w_tok
    )
    tl.store(
        w_ptr + col,
        tl.where(row < col, a_mat, tl.where(row == col, beta[:, None], 0.0)),
    )
    tl.store(
        w_ptr + BT + col,
        tl.where(row < col, b_mat, tl.where(row == col, diagonal[:, None], 0.0)),
    )


@triton.jit
def _kda_factored_solve_tile(
    projection,
    v,
    w_all,
    out_ptr,
    correction_ptr,
    bos,
    query_len,
    pid_h,
    offs_v,
    offs_e,
    offs_t,
    stride_out_token,
    stride_correction_pos,
    V: tl.constexpr,
    BT: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
):
    mask_v = offs_v < V
    # residual[:, t] = v_t - S_t' . nk_t and residual[:, BT + t] = -o_t once every
    # earlier u_i has been folded in through its row of W, so the triangular
    # solve never goes back to memory.
    residual = v - projection
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
    NUM_STAGES: tl.constexpr,
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
    w_all = tl.load(
        w_cache_ptr
        + pid_b * stride_w_block
        + pid_h * stride_w_head
        + offs_t[:, None] * stride_w_tok
        + offs_e[None, :],
        mask=in_window[None, :],
        other=0.0,
    )
    # S_0 @ [e | f] in K chunks: the checkpoint slab streams through shared
    # memory instead of sitting in registers, which keeps occupancy up.
    offs_kc = tl.arange(0, BKC)
    projection = tl.zeros([BV, EF_WIDTH], dtype=tl.float32)
    for kc in tl.range(0, BK, BKC, num_stages=NUM_STAGES):
        state = tl.load(
            state_base + offs_v[:, None] * stride_state_v + (kc + offs_kc)[None, :],
            mask=mask_v[:, None] & ((kc + offs_kc) < K)[None, :],
            other=0.0,
        ).to(tl.float32)
        # Token-major in memory so prep stores stay coalesced along K.
        ef = tl.load(
            ef_base + offs_e[None, :] * stride_ef_tok + (kc + offs_kc)[:, None],
            mask=((kc + offs_kc) < K)[:, None] & in_window[None, :],
            other=0.0,
        )
        projection = tl.dot(state, ef, projection, input_precision="tf32x3")
    correction_ptr = (
        correction_cache_ptr
        + state_idx * stride_correction_block
        + pid_h * stride_correction_head
    )
    _kda_factored_solve_tile(
        projection,
        v,
        w_all,
        out_ptr,
        correction_ptr,
        bos,
        query_len,
        pid_h,
        offs_v,
        offs_e,
        offs_t,
        stride_out_token,
        stride_correction_pos,
        V,
        BT,
        SPEC_QUERY_LEN,
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
    ef_width = max(MIN_DOT_COLS, 2 * block_t)
    ef_cache, w_cache = current_workspace_manager().get_simultaneous(
        ((batch, num_heads, 2 * block_t, key_dim), torch.float32),
        ((batch, num_heads, block_t, 2 * block_t), torch.float32),
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
        num_warps=_PREP_NUM_WARPS,
    )
    block_v = 16
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
        BKC=min(_VERIFY_BKC, block_k),
        NUM_STAGES=_VERIFY_NUM_STAGES,
        num_warps=1,
    )
    return out


@triton.jit
def _commit_factored_kda_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    correction_cache_ref_ptr,
    correction_cache_base_addrs_ptr,
    correction_cache_block_strides_ptr,
    kg_cache_ref_ptr,
    kg_cache_base_addrs_ptr,
    kg_cache_block_strides_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    lower_bound,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_cache_head,
    stride_correction_cache_pos,
    stride_correction_cache_dim,
    stride_kg_cache_head,
    stride_kg_cache_pos,
    stride_kg_cache_dim,
    stride_A_layer,
    stride_A_head,
    stride_dt_bias_layer,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS

    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(
        tl.int64
    )
    if source_state_idx <= null_block_id:
        return
    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    state_base_addr = tl.load(state_base_addrs_ptr + pid_l)
    state_block_stride = tl.load(state_block_strides_ptr + pid_l)
    state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    source_state_ptr = (
        state_ptr + source_state_idx * state_block_stride + pid_h * stride_state_head
    )

    correction_cache_base_addr = tl.load(correction_cache_base_addrs_ptr + pid_l)
    correction_cache_block_stride = tl.load(correction_cache_block_strides_ptr + pid_l)
    correction_cache_ptr = correction_cache_base_addr.to(
        tl.pointer_type(correction_cache_ref_ptr.dtype.element_ty)
    )
    correction_cache_ptr += (
        source_state_idx * correction_cache_block_stride
        + pid_h * stride_correction_cache_head
    )
    kg_cache_base_addr = tl.load(kg_cache_base_addrs_ptr + pid_l)
    kg_cache_block_stride = tl.load(kg_cache_block_strides_ptr + pid_l)
    kg_cache_ptr = kg_cache_base_addr.to(
        tl.pointer_type(kg_cache_ref_ptr.dtype.element_ty)
    )
    kg_cache_ptr += (
        source_state_idx * kg_cache_block_stride + pid_h * stride_kg_cache_head
    )

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    state_ptrs = (
        source_state_ptr
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    initial_state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)
    A = tl.exp(
        tl.load(A_log_ptr + pid_l * stride_A_layer + pid_h * stride_A_head).to(
            tl.float32
        )
    )

    dt_bias = tl.load(
        dt_bias_ptr
        + pid_l * stride_dt_bias_layer
        + pid_h * stride_dt_bias_head
        + offs_k * stride_dt_bias_dim,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    state = initial_state
    for token_offset in tl.static_range(SPEC_QUERY_LEN):
        valid = token_offset < commit_len
        if ALIGN_MODE:
            boundary_ptrs = (
                state_ptr
                + boundary_state_idx * state_block_stride
                + pid_h * stride_state_head
                + offs_v[:, None] * stride_state_v
                + offs_k[None, :] * stride_state_k
            )
            tl.store(
                boundary_ptrs,
                state,
                mask=mask_state
                & (boundary_state_idx > null_block_id)
                & (token_offset == boundary_recovery_len),
            )
        kg_ptr = kg_cache_ptr + token_offset * stride_kg_cache_pos
        raw_k = tl.load(
            kg_ptr + offs_k * stride_kg_cache_dim,
            mask=valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        normalized_k = raw_k / tl.sqrt(tl.sum(raw_k * raw_k) + 1e-6)
        raw_g = tl.load(
            kg_ptr + (K + offs_k) * stride_kg_cache_dim,
            mask=valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        decay = tl.where(
            valid,
            tl.exp(lower_bound / (1.0 + tl.exp(-(A * (raw_g + dt_bias))))),
            1.0,
        )
        correction = tl.load(
            correction_cache_ptr
            + token_offset * stride_correction_cache_pos
            + offs_v * stride_correction_cache_dim,
            mask=valid & mask_v,
            other=0.0,
        ).to(tl.float32)
        state = state * decay[None, :] + correction[:, None] * normalized_k[None, :]
    if ALIGN_MODE:
        tl.store(
            boundary_ptrs,
            state,
            mask=mask_state
            & (boundary_state_idx > null_block_id)
            & (boundary_recovery_len == commit_len),
        )
    final_ptrs = (
        state_ptr
        + final_state_idx * state_block_stride
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    tl.store(final_ptrs, state, mask=mask_state)
