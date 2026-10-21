# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import has_deep_select
from vllm.model_executor.layers.indexer_topk import (
    deep_select_topk,
    get_deep_select_stride_requirement,
    is_deep_select_supported,
)
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


# Row strides follow the logits width, which changes with every batch, and the
# row bounds are scalar loads; neither may pick a Triton specialization, or the
# first long request recompiles mid-inference.
@triton.jit(
    do_not_specialize=["stride_row", "stride_scores", "width", "nblocks"],
    do_not_specialize_on_alignment=["starts", "ends"],
)
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    valid_blocks,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    stride_scores,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
    FULL_WIDTH: tl.constexpr,
):
    # Writes each row's valid block count; FULL_WIDTH also scores the -inf tail.
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    col_end = tl.minimum(end, width)
    num_valid = tl.cdiv(tl.maximum(col_end - start, 0), BLOCK_SIZE)
    if tl.program_id(1) == 0:
        tl.store(valid_blocks + row, num_valid)
    num_blocks = nblocks if FULL_WIDTH else num_valid
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    num_tiles = tl.cdiv(num_blocks, TILE)
    for tile_id in range(tl.program_id(1), num_tiles, tl.num_programs(1)):
        blocks = tile_id * TILE + tl.arange(0, TILE)
        cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
        values = tl.load(
            logits + row * stride_row + cols * stride_col,
            (offsets[None, :] < BLOCK_SIZE) & (cols < col_end),
            other=-float("inf"),
        ).to(tl.float32)
        reduced = tl.reduce(values, 1, _max_with_nan)
        # Pin the newest block. DeepSelect leaves NaN rows partially written,
        # so also pin NaN blocks (torch.topk ranks NaN first too).
        pinned = (end > start) & (blocks == (end - start - 1) // BLOCK_SIZE)
        if not FULL_WIDTH:
            pinned |= reduced != reduced
        reduced = tl.where(pinned, float("inf"), reduced)
        tl.store(scores + row * stride_scores + blocks, reduced, blocks < num_blocks)


@triton.jit(
    do_not_specialize=["k"],
    do_not_specialize_on_alignment=["output"],
)
def _store_candidates_kernel(
    values,
    indices,
    output,
    out_stride_row,
    out_stride_col,
    k,
    OUT_K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    value = tl.load(values + row * k + cols, cols < k, other=-float("inf"))
    index = tl.load(indices + row * k + cols, cols < k, other=-1)
    # NaN scores can occur during warmup; only -inf denotes padding.
    tl.store(
        output + row * out_stride_row + cols * out_stride_col,
        tl.where(value != -float("inf"), index, -1),
        cols < OUT_K,
    )


@triton.jit(
    do_not_specialize=["width", "nblocks"],
    do_not_specialize_on_alignment=["candidates", "starts"],
)
def _candidate_flags_kernel(
    candidates,
    starts,
    flags,
    stride_row,
    stride_col,
    stride_start,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(nblocks + 1, 1024)):
        slots = tile * 1024 + offsets
        tl.store(flags + row * (nblocks + 1) + slots, 0, slots <= nblocks)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    cols = tl.arange(0, triton.next_power_of_2(K))
    block = tl.load(
        candidates + row * stride_row + cols * stride_col, cols < K, other=-1
    ).to(tl.int64)
    # Preserve the packed-column clamp for candidates beyond the logits width.
    block = tl.where(start + block * BLOCK_SIZE >= width, nblocks, block)
    tl.debug_barrier()
    tl.store(flags + row * (nblocks + 1) + block, 1, (cols < K) & (block >= 0))


@triton.jit(
    do_not_specialize=["stride_row", "width", "nblocks"],
    do_not_specialize_on_alignment=["starts", "ends"],
)
def _mask_candidates_kernel(
    logits,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    valid = (cols >= start) & (cols < end) & (cols < width)
    block = (cols - start) // BLOCK_SIZE
    keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
    tl.store(
        logits + row * stride_row + cols * stride_col,
        -float("inf"),
        (cols < width) & ~(valid & keep),
    )


def _block_scores_warmup_inputs(
    *,
    block_size: int,
    row_repeats: tuple[int, ...],
    logits_dtypes: tuple[torch.dtype, ...],
) -> list[dict[str, object]]:
    return [
        dict(
            logits=TritonWarmupTensor(dtype, shape=(1, block_size)),
            starts=starts,
            ends=TritonWarmupTensor(torch.int32),
            scores=TritonWarmupTensor(dtype),
            valid_blocks=TritonWarmupTensor(torch.int32),
            block_size=block_size,
            row_repeat=row_repeat,
            full_width=full_width,
        )
        for dtype in logits_dtypes
        for row_repeat in row_repeats
        for full_width in (False, True)
        # CUDA prefill starts every row at column zero; the ROCm and XPU
        # paths pass explicit row starts instead.
        for starts in (None, TritonWarmupTensor(torch.int32))
    ]


@triton_kernel_dispatcher_with_warmup(
    kernel=_block_scores_kernel, warmup_inputs=_block_scores_warmup_inputs
)
def _launch_block_scores(
    logits: torch.Tensor,
    starts: torch.Tensor | None,
    ends: torch.Tensor,
    scores: torch.Tensor,
    valid_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int,
    full_width: bool,
) -> DispatchSpec:
    rows, width = logits.shape
    return (rows, 32), dict(
        stride_row=logits.stride(0),
        stride_col=logits.stride(1),
        stride_start=starts.stride(0) if starts is not None else 0,
        stride_end=ends.stride(0),
        stride_scores=scores.stride(0),
        width=width,
        nblocks=triton.cdiv(width, block_size),
        BLOCK_SIZE=block_size,
        HAS_STARTS=starts is not None,
        ROW_REPEAT=row_repeat,
        TILE=128,
        FULL_WIDTH=full_width,
    )


def _store_candidates_warmup_inputs(
    *, topk_blocks: int, logits_dtypes: tuple[torch.dtype, ...]
) -> list[dict[str, object]]:
    return [
        dict(
            values=TritonWarmupTensor(dtype),
            indices=TritonWarmupTensor(torch.int64),
            output=TritonWarmupTensor(torch.int32, shape=(1, topk_blocks)),
            k=1,
            topk_blocks=topk_blocks,
        )
        for dtype in logits_dtypes
    ]


@triton_kernel_dispatcher_with_warmup(
    kernel=_store_candidates_kernel, warmup_inputs=_store_candidates_warmup_inputs
)
def _launch_store_candidates(
    values: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    k: int,
    topk_blocks: int,
) -> DispatchSpec:
    return (values.shape[0], triton.cdiv(topk_blocks, 256)), dict(
        out_stride_row=output.stride(0),
        out_stride_col=output.stride(1),
        k=k,
        OUT_K=topk_blocks,
        TILE=256,
    )


def _candidate_flags_warmup_inputs(
    *, block_size: int, topk_blocks: int, row_repeats: tuple[int, ...]
) -> list[dict[str, object]]:
    return [
        dict(
            candidates=TritonWarmupTensor(torch.int32, shape=(1, topk_blocks)),
            starts=starts,
            flags=TritonWarmupTensor(torch.uint8),
            width=block_size,
            block_size=block_size,
            row_repeat=row_repeat,
        )
        for row_repeat in row_repeats
        for starts in (None, TritonWarmupTensor(torch.int32))
    ]


@triton_kernel_dispatcher_with_warmup(
    kernel=_candidate_flags_kernel, warmup_inputs=_candidate_flags_warmup_inputs
)
def _launch_candidate_flags(
    candidates: torch.Tensor,
    starts: torch.Tensor | None,
    flags: torch.Tensor,
    width: int,
    block_size: int,
    row_repeat: int,
) -> DispatchSpec:
    return (candidates.shape[0],), dict(
        stride_row=candidates.stride(0),
        stride_col=candidates.stride(1),
        stride_start=starts.stride(0) if starts is not None else 0,
        width=width,
        nblocks=triton.cdiv(width, block_size),
        BLOCK_SIZE=block_size,
        K=candidates.shape[1],
        HAS_STARTS=starts is not None,
        ROW_REPEAT=row_repeat,
    )


def _mask_candidates_warmup_inputs(
    *,
    block_size: int,
    row_repeats: tuple[int, ...],
    logits_dtypes: tuple[torch.dtype, ...],
) -> list[dict[str, object]]:
    return [
        dict(
            logits=TritonWarmupTensor(dtype, shape=(1, block_size)),
            starts=starts,
            ends=TritonWarmupTensor(torch.int32),
            flags=TritonWarmupTensor(torch.uint8),
            block_size=block_size,
            row_repeat=row_repeat,
        )
        for dtype in logits_dtypes
        for row_repeat in row_repeats
        for starts in (None, TritonWarmupTensor(torch.int32))
    ]


@triton_kernel_dispatcher_with_warmup(
    kernel=_mask_candidates_kernel, warmup_inputs=_mask_candidates_warmup_inputs
)
def _launch_mask_candidates(
    logits: torch.Tensor,
    starts: torch.Tensor | None,
    ends: torch.Tensor,
    flags: torch.Tensor,
    block_size: int,
    row_repeat: int,
) -> DispatchSpec:
    rows, width = logits.shape
    return (rows, triton.cdiv(width, 1024)), dict(
        stride_row=logits.stride(0),
        stride_col=logits.stride(1),
        stride_start=starts.stride(0) if starts is not None else 0,
        stride_end=ends.stride(0),
        width=width,
        nblocks=triton.cdiv(width, block_size),
        BLOCK_SIZE=block_size,
        HAS_STARTS=starts is not None,
        ROW_REPEAT=row_repeat,
        TILE=1024,
    )


def register_candidate_block_warmup(
    *,
    select: bool,
    block_size: int,
    topk_blocks: int,
    row_repeats: tuple[int, ...],
    logits_dtypes: tuple[torch.dtype, ...],
) -> None:
    """Warm the kernels one indexer's candidate step launches.

    The candidate source (``select``) scores and publishes blocks; the other
    indexers only mask their logits with them. Short startup batches reach
    neither the DeepSelect path nor the spec-decode row repeat, so every
    reachable specialization is compiled here instead.
    """
    if select:
        _launch_block_scores.register_warmup(
            block_size=block_size,
            row_repeats=row_repeats,
            logits_dtypes=logits_dtypes,
        )
        _launch_store_candidates.register_warmup(
            topk_blocks=topk_blocks, logits_dtypes=logits_dtypes
        )
    else:
        _launch_candidate_flags.register_warmup(
            block_size=block_size, topk_blocks=topk_blocks, row_repeats=row_repeats
        )
        _launch_mask_candidates.register_warmup(
            block_size=block_size,
            row_repeats=row_repeats,
            logits_dtypes=logits_dtypes,
        )


def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
) -> None:
    """Select local block IDs by maximum score, pinning each row's newest block.

    Row bounds are in packed column space; absent starts mean zero.
    Decode rows share bounds in groups of ``row_repeat``. Output is -1 padded.
    """
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return
    nblocks = triton.cdiv(width, block_size)
    # Logits are max_model_len wide (CUDA-graph static) but rows are shorter:
    # DeepSelect stops at each row's valid block count. Order does not matter.
    use_deep_select = has_deep_select() and nblocks >= topk_blocks
    if use_deep_select:
        in_align, out_align = get_deep_select_stride_requirement()
        padded = round_up(nblocks, in_align // logits.element_size())
        scores = logits.new_empty((rows, padded))[:, :nblocks]
        use_deep_select = (
            out.stride(1) == 1
            and out.stride(0) * out.element_size() % out_align == 0
            and is_deep_select_supported(scores, topk_blocks)
        )
    else:
        scores = logits.new_empty((rows, nblocks))
    valid_blocks = torch.empty(rows, dtype=torch.int32, device=logits.device)
    _launch_block_scores(
        logits,
        row_ks,
        row_ke,
        scores,
        valid_blocks,
        block_size,
        row_repeat,
        full_width=not use_deep_select,
    )
    if use_deep_select:
        deep_select_topk(scores, topk_blocks, end=valid_blocks, output_idx=out)
        return
    # Keep the existing top-k tie behavior.
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    _launch_store_candidates(
        top.values, top.indices, out, top.values.shape[1], topk_blocks
    )


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
) -> None:
    """Mask packed logits outside causal bounds and request-local candidates."""
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows or not width:
        return
    nblocks = triton.cdiv(width, block_size)
    flags = torch.empty((rows, nblocks + 1), device=logits.device, dtype=torch.uint8)
    _launch_candidate_flags(
        candidate_blocks, row_ks, flags, width, block_size, row_repeat
    )
    _launch_mask_candidates(logits, row_ks, row_ke, flags, block_size, row_repeat)
