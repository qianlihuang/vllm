# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Single-warp Triton kernel for MiniMax-M2 sigmoid top-k routing.

On H100/H200 the existing CUDA fused topk_sigmoid kernel takes ~6 µs.
For prefill-disaggregated decode servers where batch sizes are small
(often 1–4 tokens), a single-warp Triton kernel reduces launch overhead
and keeps all work within one warp per token, avoiding cross-warp sync.

The kernel fuses: sigmoid → correction bias → top-k selection → renormalize

Environment variables
---------------------
``VLLM_TRITON_MINIMAX_ROUTING``
    ``"auto"`` (default) — enable for batches ≤
    ``VLLM_TRITON_ROUTING_MAX_TOKENS`` tokens (default 32).
    ``"1"`` / ``"true"`` — always use Triton when shape-compatible.
    ``"0"`` / ``"false"`` — never use Triton.

``VLLM_TRITON_ROUTING_MAX_TOKENS``
    Batch-size threshold for auto mode (default 32).
"""

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

_ENABLE_ENV_VAR = "VLLM_TRITON_MINIMAX_ROUTING"
_TRITON_MAX_TOKENS_DEFAULT = 32


def _use_triton_routing(num_tokens: int) -> bool:
    """Return True if the Triton fast path should be used for this batch."""
    if not HAS_TRITON:
        return False
    import os

    flag = os.getenv(_ENABLE_ENV_VAR, "auto")
    if flag == "0" or flag.lower() == "false":
        return False
    if flag == "1" or flag.lower() == "true":
        return True
    # auto: enable for decode-sized batches
    threshold = int(
        os.getenv("VLLM_TRITON_ROUTING_MAX_TOKENS", str(_TRITON_MAX_TOKENS_DEFAULT))
    )
    return num_tokens <= threshold


def _can_use_triton_minimax_routing(
    gating_output: torch.Tensor,
    scoring_func: str,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
) -> bool:
    """Check whether the Triton single-warp kernel can handle this call."""
    if not HAS_TRITON:
        return False
    if scoring_func != "sigmoid":
        return False
    if not renormalize:
        return False
    if e_score_correction_bias is None:
        return False
    if topk != 8:
        return False
    num_experts = gating_output.shape[-1]
    if num_experts > 256:
        return False
    if num_experts & (num_experts - 1) != 0:
        return False  # not a power of 2
    if gating_output.ndim != 2:
        return False
    if gating_output.dtype != torch.float32:
        return False
    return _use_triton_routing(gating_output.shape[0])


# ---------------------------------------------------------------------------
# Triton kernel — only defined when Triton is available to avoid import
# errors on platforms without Triton (e.g. some CPU-only builds).
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _minimax_sigmoid_topk_kernel(
        gating_output_ptr,
        correction_bias_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        token_expert_indices_ptr,
        stride_gm,
        stride_ge,
        stride_wm,
        stride_wk,
        stride_im,
        stride_ik,
        stride_tem,
        stride_tek,
        num_tokens,
        num_experts: tl.constexpr,
        BLOCK_E: tl.constexpr,
        TOPK: tl.constexpr,
    ):
        """Single-warp per-token kernel: sigmoid + bias + topk + renormalize.

        Each program processes exactly one token row.  With ``num_warps=1``
        every warp works independently so there is zero cross-warp
        synchronisation, minimising latency for small decode batches.
        """
        token_id = tl.program_id(0)
        if token_id >= num_tokens:
            return

        offs_e = tl.arange(0, BLOCK_E)
        expert_mask = offs_e < num_experts

        # ---- load gating logits --------------------------------------------
        logits = tl.load(
            gating_output_ptr + token_id * stride_gm + offs_e * stride_ge,
            mask=expert_mask,
            other=-float("inf"),
        ).to(tl.float32)

        # ---- load correction bias ------------------------------------------
        bias = tl.load(
            correction_bias_ptr + offs_e,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)

        # ---- sigmoid -------------------------------------------------------
        scores = tl.sigmoid(logits.to(tl.float32))

        # Bias is added only for *selection*; stored weights are raw sigmoid.
        choice_scores = tl.where(expert_mask, scores + bias, -float("inf"))

        weights_sum = 0.0

        for k in tl.static_range(0, TOPK):
            # argmax across the row — warp-level reduction
            best_choice_score = tl.max(choice_scores, axis=0)
            # tie-break: lowest expert index wins
            best_expert = tl.min(
                tl.where(choice_scores == best_choice_score, offs_e, BLOCK_E),
                axis=0,
            )
            # Retrieve the raw sigmoid score for the winning expert.
            best_weight = tl.max(
                tl.where(offs_e == best_expert, scores, 0.0), axis=0
            )

            weights_sum += best_weight

            tl.store(
                topk_ids_ptr + token_id * stride_im + k * stride_ik,
                best_expert.to(tl.int32),
            )
            tl.store(
                topk_weights_ptr + token_id * stride_wm + k * stride_wk,
                best_weight,
            )
            tl.store(
                token_expert_indices_ptr
                + token_id * stride_tem
                + k * stride_tek,
                k * num_tokens + token_id,
            )

            # Blank the winner so the next iteration picks the next best.
            choice_scores = tl.where(
                offs_e == best_expert, -float("inf"), choice_scores
            )

        # ---- renormalize ---------------------------------------------------
        denom = tl.where(weights_sum != 0.0, weights_sum, 1.0)
        for k in tl.static_range(0, TOPK):
            weight = tl.load(
                topk_weights_ptr + token_id * stride_wm + k * stride_wk
            )
            tl.store(
                topk_weights_ptr + token_id * stride_wm + k * stride_wk,
                weight / denom,
            )


def minimax_triton_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton single-warp topk_sigmoid — fast path for decode-sized batches.

    Args:
        topk_weights: output ``[num_tokens, topk]`` float32
        topk_ids: output ``[num_tokens, topk]`` int32
        token_expert_indices: output ``[num_tokens, topk]`` int32
        gating_output: ``[num_tokens, num_experts]`` float32
        renormalize: must be True
        e_score_correction_bias: ``[num_experts]`` float32

    Returns:
        (topk_weights, topk_ids, token_expert_indices)
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "minimax_triton_topk_sigmoid requires Triton, "
            "but it is not installed."
        )

    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[-1]
    block_e = triton.next_power_of_2(num_experts)

    _minimax_sigmoid_topk_kernel[(num_tokens,)](
        gating_output,
        e_score_correction_bias,
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output.stride(0),
        gating_output.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        token_expert_indices.stride(0),
        token_expert_indices.stride(1),
        num_tokens,
        num_experts=num_experts,
        BLOCK_E=block_e,
        TOPK=topk,
        num_warps=1,
    )

    return topk_weights, topk_ids, token_expert_indices
