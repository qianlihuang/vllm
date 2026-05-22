# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from dataclasses import dataclass

import torch
from tabulate import tabulate

from vllm import _custom_ops as ops
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    create_kv_caches_with_random_flash,
    set_random_seed,
)


@dataclass(frozen=True)
class MiniMaxShape:
    num_q_heads: int
    num_kv_heads: int
    head_size: int = 128
    rotary_dim: int = 64


MINIMAX_SHAPES = {
    "tp4": MiniMaxShape(num_q_heads=12, num_kv_heads=2),
    "tp8": MiniMaxShape(num_q_heads=6, num_kv_heads=1),
}


def set_device(device: str) -> None:
    torch.set_default_device(device)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))


def make_cos_sin_cache(
    max_position: int,
    rotary_dim: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    positions = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(
        dtype=dtype, device=device
    )


def clone_preserve_strides(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_strided(
        tuple(x.shape), tuple(x.stride()), dtype=x.dtype, device=x.device
    )
    y.copy_(x)
    return y


def make_slot_mapping(
    num_tokens: int,
    num_actual_tokens: int,
    block_size: int,
    num_blocks: int,
    device: str,
    include_negative_slot: bool,
) -> torch.Tensor:
    if num_actual_tokens > num_tokens:
        raise ValueError("num_actual_tokens cannot exceed num_tokens")

    num_slots = block_size * num_blocks
    if num_actual_tokens > num_slots:
        raise ValueError("num_actual_tokens cannot exceed total cache slots")

    slot_mapping = torch.tensor(
        random.sample(range(num_slots), num_actual_tokens),
        dtype=torch.long,
        device=device,
    )
    if include_negative_slot and num_actual_tokens > 1:
        slot_mapping[1] = -1
    return slot_mapping


def make_case(
    shape_name: str,
    num_tokens: int,
    num_actual_tokens: int | None,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    cos_sin_dtype: torch.dtype,
    kv_cache_layout: str,
    max_position: int,
    include_negative_slot: bool,
    device: str,
):
    shape = MINIMAX_SHAPES[shape_name]
    if num_actual_tokens is None:
        num_actual_tokens = num_tokens

    query = torch.randn(
        num_tokens, shape.num_q_heads, shape.head_size, dtype=dtype, device=device
    )
    key = torch.randn(
        num_tokens, shape.num_kv_heads, shape.head_size, dtype=dtype, device=device
    )
    value = torch.randn_like(key)
    positions = torch.randint(
        0, max_position, (num_tokens,), dtype=torch.long, device=device
    )
    cos_sin_cache = make_cos_sin_cache(
        max_position, shape.rotary_dim, cos_sin_dtype, device
    )
    slot_mapping = make_slot_mapping(
        num_tokens,
        num_actual_tokens,
        block_size,
        num_blocks,
        device,
        include_negative_slot,
    )

    key_caches, value_caches = create_kv_caches_with_random_flash(
        num_blocks,
        block_size,
        1,
        shape.num_kv_heads,
        shape.head_size,
        "auto",
        dtype,
        device=device,
        cache_layout=kv_cache_layout,
    )
    key_cache = key_caches[0]
    value_cache = value_caches[0]
    del key_caches, value_caches

    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    v_scale = torch.ones(1, dtype=torch.float32, device=device)
    return (
        shape,
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        slot_mapping,
        key_cache,
        value_cache,
        k_scale,
        v_scale,
    )


def max_abs_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    return (x.float() - y.float()).abs().max().item()


@torch.inference_mode()
def run_accuracy(args) -> None:
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    cos_sin_dtype = STR_DTYPE_TO_TORCH_DTYPE[args.cos_sin_dtype]
    set_device(args.device)
    set_random_seed(args.seed)

    rows = []
    for shape_name in args.shapes:
        for layout in args.layouts:
            (
                shape,
                query,
                key,
                value,
                positions,
                cos_sin_cache,
                slot_mapping,
                key_cache,
                value_cache,
                k_scale,
                v_scale,
            ) = make_case(
                shape_name,
                args.accuracy_num_tokens,
                args.accuracy_actual_tokens,
                args.block_size,
                args.num_blocks,
                dtype,
                cos_sin_dtype,
                layout,
                args.max_position,
                args.include_negative_slot,
                args.device,
            )

            q_ref = query.clone()
            k_ref = key.clone()
            key_cache_ref = clone_preserve_strides(key_cache)
            value_cache_ref = clone_preserve_strides(value_cache)
            ops.rotary_embedding(
                positions, q_ref, k_ref, shape.head_size, cos_sin_cache, True
            )
            ops.reshape_and_cache_flash(
                k_ref,
                value,
                key_cache_ref,
                value_cache_ref,
                slot_mapping,
                "auto",
                k_scale,
                v_scale,
            )

            q_fused = query.clone()
            k_fused = key.clone()
            key_cache_fused = clone_preserve_strides(key_cache)
            value_cache_fused = clone_preserve_strides(value_cache)
            ops.fused_rope_and_cache_flash(
                q_fused,
                k_fused,
                value,
                key_cache_fused,
                value_cache_fused,
                slot_mapping,
                positions,
                cos_sin_cache,
                True,
                "auto",
                k_scale,
                v_scale,
            )
            torch.accelerator.synchronize()

            atol = args.atol
            rtol = args.rtol
            torch.testing.assert_close(q_fused, q_ref, atol=atol, rtol=rtol)
            torch.testing.assert_close(k_fused, k_ref, atol=atol, rtol=rtol)
            torch.testing.assert_close(
                key_cache_fused, key_cache_ref, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                value_cache_fused, value_cache_ref, atol=atol, rtol=rtol
            )

            rows.append(
                [
                    shape_name,
                    layout,
                    args.accuracy_num_tokens,
                    slot_mapping.numel(),
                    f"{max_abs_diff(q_fused, q_ref):.3e}",
                    f"{max_abs_diff(k_fused, k_ref):.3e}",
                    f"{max_abs_diff(key_cache_fused, key_cache_ref):.3e}",
                    f"{max_abs_diff(value_cache_fused, value_cache_ref):.3e}",
                    "PASS",
                ]
            )

    print(
        tabulate(
            rows,
            headers=[
                "shape",
                "layout",
                "tokens",
                "actual",
                "q max",
                "k max",
                "k cache max",
                "v cache max",
                "status",
            ],
        )
    )


def cuda_event_bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


@torch.inference_mode()
def run_perf(args) -> None:
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    cos_sin_dtype = STR_DTYPE_TO_TORCH_DTYPE[args.cos_sin_dtype]
    set_device(args.device)
    set_random_seed(args.seed)

    rows = []
    for shape_name in args.shapes:
        for layout in args.layouts:
            for num_tokens in args.num_tokens:
                (
                    shape,
                    query,
                    key,
                    value,
                    positions,
                    cos_sin_cache,
                    slot_mapping,
                    key_cache,
                    value_cache,
                    k_scale,
                    v_scale,
                ) = make_case(
                    shape_name,
                    num_tokens,
                    args.actual_tokens,
                    args.block_size,
                    args.num_blocks,
                    dtype,
                    cos_sin_dtype,
                    layout,
                    args.max_position,
                    False,
                    args.device,
                )

                q_rope = query.clone()
                k_rope = key.clone()
                rope_fn = lambda: ops.rotary_embedding(
                    positions, q_rope, k_rope, shape.head_size, cos_sin_cache, True
                )

                k_cache_only = key.clone()
                key_cache_only = clone_preserve_strides(key_cache)
                value_cache_only = clone_preserve_strides(value_cache)
                cache_fn = lambda: ops.reshape_and_cache_flash(
                    k_cache_only,
                    value,
                    key_cache_only,
                    value_cache_only,
                    slot_mapping,
                    "auto",
                    k_scale,
                    v_scale,
                )

                q_unfused = query.clone()
                k_unfused = key.clone()
                key_cache_unfused = clone_preserve_strides(key_cache)
                value_cache_unfused = clone_preserve_strides(value_cache)

                def unfused_fn():
                    ops.rotary_embedding(
                        positions,
                        q_unfused,
                        k_unfused,
                        shape.head_size,
                        cos_sin_cache,
                        True,
                    )
                    ops.reshape_and_cache_flash(
                        k_unfused,
                        value,
                        key_cache_unfused,
                        value_cache_unfused,
                        slot_mapping,
                        "auto",
                        k_scale,
                        v_scale,
                    )

                q_fused = query.clone()
                k_fused = key.clone()
                key_cache_fused = clone_preserve_strides(key_cache)
                value_cache_fused = clone_preserve_strides(value_cache)

                def fused_fn():
                    ops.fused_rope_and_cache_flash(
                        q_fused,
                        k_fused,
                        value,
                        key_cache_fused,
                        value_cache_fused,
                        slot_mapping,
                        positions,
                        cos_sin_cache,
                        True,
                        "auto",
                        k_scale,
                        v_scale,
                    )

                rope_us = cuda_event_bench(rope_fn, args.warmup, args.iters)
                cache_us = cuda_event_bench(cache_fn, args.warmup, args.iters)
                unfused_us = cuda_event_bench(unfused_fn, args.warmup, args.iters)
                fused_us = cuda_event_bench(fused_fn, args.warmup, args.iters)
                sum_us = rope_us + cache_us

                rows.append(
                    [
                        shape_name,
                        layout,
                        num_tokens,
                        slot_mapping.numel(),
                        f"{rope_us:.3f}",
                        f"{cache_us:.3f}",
                        f"{sum_us:.3f}",
                        f"{unfused_us:.3f}",
                        f"{fused_us:.3f}",
                        f"{sum_us / fused_us:.2f}x",
                        f"{unfused_us / fused_us:.2f}x",
                    ]
                )

                del query, key, value, key_cache, value_cache
                torch.accelerator.empty_cache()

    print(
        tabulate(
            rows,
            headers=[
                "shape",
                "layout",
                "tokens",
                "actual",
                "rope us",
                "cache us",
                "sum us",
                "seq us",
                "fused us",
                "sum/fused",
                "seq/fused",
            ],
        )
    )


def parse_args():
    parser = FlexibleArgumentParser(
        description="Microbenchmark fused RoPE + KV cache for MiniMax M2.x shapes."
    )
    parser.add_argument("--mode", choices=["accuracy", "perf", "all"], default="all")
    parser.add_argument("--shapes", nargs="+", choices=["tp4", "tp8"], default=["tp4"])
    parser.add_argument("--layouts", nargs="+", choices=["NHD", "HND"], default=["NHD"])
    parser.add_argument(
        "--num-tokens",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="Token counts for perf mode.",
    )
    parser.add_argument(
        "--actual-tokens",
        type=int,
        default=None,
        help="Optional slot_mapping length for perf mode; defaults to num_tokens.",
    )
    parser.add_argument("--accuracy-num-tokens", type=int, default=7)
    parser.add_argument(
        "--accuracy-actual-tokens",
        type=int,
        default=5,
        help="slot_mapping length for accuracy mode; <= accuracy-num-tokens.",
    )
    parser.add_argument("--include-negative-slot", action="store_true")
    parser.add_argument("--block-size", type=int, choices=[16, 32, 64], default=16)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--max-position", type=int, default=8192)
    parser.add_argument(
        "--dtype",
        choices=["half", "bfloat16", "float"],
        default="bfloat16",
    )
    parser.add_argument(
        "--cos-sin-dtype",
        choices=["half", "bfloat16", "float"],
        default="bfloat16",
    )
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode in ("accuracy", "all"):
        run_accuracy(args)
    if args.mode in ("perf", "all"):
        run_perf(args)
