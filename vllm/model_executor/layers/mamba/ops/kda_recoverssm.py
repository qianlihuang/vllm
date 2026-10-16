# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factored KDA RecoverSSM verification shared by Kimi-K3 and GLM.

The checkpoint contraction is S_0 @ [D*k, D*q]. A short triangular
recurrence forms corrections and outputs without updating a full state tile.
Window vectors and triangular weights are temporary batch workspaces; only
the correction and original key/gate records survive until commit.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.worker.workspace import current_workspace_manager

MIN_DOT_COLS = 16


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
    c_cache_ptr,
    ab_cache_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    scale,
    lower_bound,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_g_token,
    stride_kd_block,
    stride_kd_head,
    stride_kd_pos,
    stride_ef_block,
    stride_ef_head,
    stride_ef_tok,
    stride_c_block,
    stride_c_head,
    stride_c_tok,
    stride_ab_block,
    stride_ab_head,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    bos = tl.load(query_start_loc_ptr + pid_b).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + pid_b + 1).to(tl.int64)
    state_idx = tl.load(state_indices_ptr + pid_b).to(tl.int64)
    if state_idx <= null_block_id:
        return
    offs_k = tl.arange(0, BK)
    mask_k = offs_k < K
    a_log = tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0).to(
        tl.float32
    )
    kd_ptr = kg_cache_ptr + state_idx * stride_kd_block + pid_h * stride_kd_head
    ef_ptr = ef_cache_ptr + pid_b * stride_ef_block + pid_h * stride_ef_head
    c_ptr = c_cache_ptr + pid_b * stride_c_block + pid_h * stride_c_head
    ab_ptr = ab_cache_ptr + pid_b * stride_ab_block + pid_h * stride_ab_head

    decay_after = tl.full([BK], 1.0, dtype=tl.float32)
    for t in tl.static_range(SPEC_QUERY_LEN):
        valid = t < eos - bos
        token = bos + t
        k = tl.load(
            k_ptr + token * stride_k_token + pid_h * K + offs_k,
            mask=valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        raw_g = tl.load(
            raw_g_ptr + token * stride_g_token + pid_h * K + offs_k,
            mask=valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        q = tl.load(
            q_ptr + token * stride_q_token + pid_h * K + offs_k,
            mask=valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        normalized_k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        gate = lower_bound / (1.0 + tl.exp(-(a_log * (raw_g + dt_bias))))
        decay = tl.exp(gate)
        normalized_q = q / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
        # D_{t+1}; c_t is normalised by it because the u_t rank-1 term is only
        # decayed from S_{t+1} onwards.
        decay_after = decay_after * decay
        tl.store(kd_ptr + t * stride_kd_pos + offs_k, k, mask=valid & mask_k)
        tl.store(kd_ptr + t * stride_kd_pos + K + offs_k, raw_g, mask=valid & mask_k)
        tl.store(
            ef_ptr + t * stride_ef_tok + offs_k,
            decay_after * normalized_k,
            mask=valid & mask_k,
        )
        tl.store(
            ef_ptr + (SPEC_QUERY_LEN + t) * stride_ef_tok + offs_k,
            decay_after * normalized_q,
            mask=valid & mask_k,
        )
        tl.store(
            c_ptr + t * stride_c_tok + offs_k,
            tl.where(valid, normalized_k / decay_after, 0.0),
            mask=mask_k,
        )
        # The u_t term is added after the decay, so its output weight is plain
        # nk_t . nq_t rather than c_t . f_t.
        tl.store(
            ab_ptr + SPEC_QUERY_LEN * SPEC_QUERY_LEN + t * SPEC_QUERY_LEN + t,
            tl.sum(tl.where(valid, normalized_k * normalized_q, 0.0)),
        )

    # A = C^T E and B = C^T F, one broadcast reduction over [BT, BK] tiles.
    offs_t = tl.arange(0, BT)
    mask_t = (offs_t < SPEC_QUERY_LEN)[:, None] & mask_k[None, :]
    c_tile = tl.load(
        c_ptr + offs_t[:, None] * stride_c_tok + offs_k[None, :],
        mask=mask_t,
        other=0.0,
    )
    e_tile = tl.load(
        ef_ptr + offs_t[:, None] * stride_ef_tok + offs_k[None, :],
        mask=mask_t,
        other=0.0,
    )
    f_tile = tl.load(
        ef_ptr + (SPEC_QUERY_LEN + offs_t)[:, None] * stride_ef_tok + offs_k[None, :],
        mask=mask_t,
        other=0.0,
    )
    a_mat = tl.sum(c_tile[:, None, :] * e_tile[None, :, :], axis=2)
    b_mat = tl.sum(c_tile[:, None, :] * f_tile[None, :, :], axis=2)
    row = offs_t[:, None]
    col = offs_t[None, :]
    in_range = (offs_t < SPEC_QUERY_LEN)[:, None] & (offs_t < SPEC_QUERY_LEN)[None, :]
    tl.store(
        ab_ptr + row * SPEC_QUERY_LEN + col,
        tl.where(row < col, a_mat, 0.0),
        mask=in_range,
    )
    # Strictly upper for B too, so the diagonal written in the token loop holds.
    tl.store(
        ab_ptr + SPEC_QUERY_LEN * SPEC_QUERY_LEN + row * SPEC_QUERY_LEN + col,
        tl.where(row < col, b_mat, 0.0),
        mask=in_range & (row != col),
    )


@triton.jit
def _kda_factored_verify_kernel(
    q_ptr,
    v_ptr,
    raw_beta_ptr,
    state_ptr,
    correction_cache_ptr,
    ef_cache_ptr,
    ab_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    null_block_id,
    stride_q_token,
    stride_v_token,
    stride_beta_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_correction_block,
    stride_correction_head,
    stride_correction_pos,
    stride_ef_block,
    stride_ef_head,
    stride_ef_tok,
    stride_ab_block,
    stride_ab_head,
    stride_out_token,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    EF_WIDTH: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    bos = tl.load(query_start_loc_ptr + pid_b).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + pid_b + 1).to(tl.int64)
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    offs_e = tl.arange(0, EF_WIDTH)
    mask_k = offs_k < K
    mask_v = offs_v < V
    if state_idx <= null_block_id:
        for pos in tl.static_range(SPEC_QUERY_LEN):
            tl.store(
                out_ptr + (bos + pos) * stride_out_token + pid_h * V + offs_v,
                tl.zeros([BV], dtype=tl.float32),
                mask=(pos < query_len) & mask_v,
            )
        return

    state = tl.load(
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :],
        mask=mask_v[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)
    # Stored token-major so the prep stores and this load stay coalesced along
    # K; transposed here for the contraction.
    ef = tl.trans(
        tl.load(
            ef_cache_ptr
            + pid_b * stride_ef_block
            + pid_h * stride_ef_head
            + offs_e[:, None] * stride_ef_tok
            + offs_k[None, :],
            mask=(offs_e[:, None] < 2 * SPEC_QUERY_LEN) & mask_k[None, :],
            other=0.0,
        )
    )
    # Columns [0, L) are S_0 . e_t, [L, 2L) are S_0 . f_t.
    proj = tl.dot(state, ef, input_precision="tf32x3")
    ab_ptr = ab_cache_ptr + pid_b * stride_ab_block + pid_h * stride_ab_head
    correction_ptr = (
        correction_cache_ptr
        + state_idx * stride_correction_block
        + pid_h * stride_correction_head
    )

    for t in tl.static_range(SPEC_QUERY_LEN):
        valid = t < query_len
        token = bos + t
        acc = tl.sum(tl.where(offs_e[None, :] == t, proj, 0.0), axis=1)
        acc_out = tl.sum(
            tl.where(offs_e[None, :] == SPEC_QUERY_LEN + t, proj, 0.0), axis=1
        )
        for i in tl.static_range(SPEC_QUERY_LEN):
            if i < t:
                # The correction record is written below, so it doubles as the
                # store for the triangular solve's earlier u_i.
                u_i = tl.load(
                    correction_ptr + i * stride_correction_pos + offs_v,
                    mask=mask_v,
                    other=0.0,
                )
                acc += tl.load(ab_ptr + i * SPEC_QUERY_LEN + t) * u_i
                acc_out += (
                    tl.load(
                        ab_ptr
                        + SPEC_QUERY_LEN * SPEC_QUERY_LEN
                        + i * SPEC_QUERY_LEN
                        + t
                    )
                    * u_i
                )
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * V + offs_v,
            mask=valid & mask_v,
            other=0.0,
        ).to(tl.float32)
        beta = tl.sigmoid(
            tl.load(
                raw_beta_ptr + token * stride_beta_token + pid_h,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
        )
        correction = (v - acc) * beta
        acc_out += (
            tl.load(ab_ptr + SPEC_QUERY_LEN * SPEC_QUERY_LEN + t * SPEC_QUERY_LEN + t)
            * correction
        )
        tl.store(
            out_ptr + token * stride_out_token + pid_h * V + offs_v,
            acc_out,
            mask=valid & mask_v,
        )
        tl.store(
            correction_ptr + t * stride_correction_pos + offs_v,
            correction,
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
    ef_cache, c_cache, ab_cache = current_workspace_manager().get_simultaneous(
        ((batch, num_heads, 2 * spec_query_len, key_dim), torch.float32),
        ((batch, num_heads, spec_query_len, key_dim), torch.float32),
        ((batch, num_heads, 2, spec_query_len * spec_query_len), torch.float32),
    )
    ef_width = max(MIN_DOT_COLS, triton.next_power_of_2(2 * spec_query_len))
    assert 2 * spec_query_len <= ef_width
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
        c_cache,
        ab_cache,
        query_start_loc,
        state_indices,
        key_dim**-0.5,
        lower_bound,
        NULL_BLOCK_ID,
        q.stride(1),
        k.stride(1),
        raw_g.stride(1),
        kg_cache.stride(0),
        kg_cache.stride(1),
        kg_cache.stride(2),
        ef_cache.stride(0),
        ef_cache.stride(1),
        ef_cache.stride(2),
        c_cache.stride(0),
        c_cache.stride(1),
        c_cache.stride(2),
        ab_cache.stride(0),
        ab_cache.stride(1),
        K=key_dim,
        BK=block_k,
        BT=triton.next_power_of_2(spec_query_len),
        SPEC_QUERY_LEN=spec_query_len,
        num_warps=1,
    )
    block_v = 16
    _kda_factored_verify_kernel[(triton.cdiv(value_dim, block_v), batch, num_heads)](
        q,
        v,
        raw_beta,
        checkpoint_state,
        correction_cache,
        ef_cache,
        ab_cache,
        out,
        query_start_loc,
        state_indices,
        NULL_BLOCK_ID,
        q.stride(1),
        v.stride(1),
        raw_beta.stride(1),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        correction_cache.stride(0),
        correction_cache.stride(1),
        correction_cache.stride(2),
        ef_cache.stride(0),
        ef_cache.stride(1),
        ef_cache.stride(2),
        ab_cache.stride(0),
        ab_cache.stride(1),
        out.stride(1),
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        SPEC_QUERY_LEN=spec_query_len,
        EF_WIDTH=ef_width,
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
