#!/usr/bin/env python3
"""
Micro-benchmark: Triton single-warp vs CUDA topk_sigmoid kernel.

Sweeps num_tokens from 1 to 1024 with MiniMax-M2 shapes
(num_experts=256, topk=8, float32 gating + float32 correction_bias).

Usage:
    python benchmarks/kernels/benchmark_minimax_routing.py
    python benchmarks/kernels/benchmark_minimax_routing.py --num-expert 128 --topk 4
"""

import argparse
import itertools
import time

import torch

from vllm._custom_ops import topk_sigmoid as cuda_topk_sigmoid
from vllm.model_executor.layers.fused_moe.router.minimax_routing import (
    _can_use_triton_minimax_routing,
    minimax_triton_topk_sigmoid,
)
from vllm.triton_utils import HAS_TRITON

_WARMUP_ITERS = 50
_BENCH_ITERS = 200


def _run_cuda(
    gating_output: torch.Tensor,
    topk: int,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device="cuda")
    topk_ids = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")
    token_expert_indices = torch.empty(
        num_tokens, topk, dtype=torch.int32, device="cuda"
    )
    cuda_topk_sigmoid(
        topk_weights, topk_ids, token_expert_indices,
        gating_output, True, correction_bias,
    )
    return topk_weights, topk_ids, token_expert_indices


def _run_triton(
    gating_output: torch.Tensor,
    topk: int,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device="cuda")
    topk_ids = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")
    token_expert_indices = torch.empty(
        num_tokens, topk, dtype=torch.int32, device="cuda"
    )
    return minimax_triton_topk_sigmoid(
        topk_weights, topk_ids, token_expert_indices,
        gating_output, True, correction_bias,
    )


def benchmark_shape(
    num_tokens: int,
    num_experts: int,
    topk: int,
) -> dict:
    device = torch.device("cuda")
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    correction_bias = torch.zeros(num_experts, dtype=torch.float32, device=device)

    can_triton = _can_use_triton_minimax_routing(
        gating_output=gating_output,
        scoring_func="sigmoid",
        topk=topk,
        renormalize=True,
        e_score_correction_bias=correction_bias,
    )

    # ---- correctness check ------------------------------------------------
    if can_triton:
        w_cuda, ids_cuda, tei_cuda = _run_cuda(gating_output, topk, correction_bias)
        w_triton, ids_triton, tei_triton = _run_triton(gating_output, topk, correction_bias)

        ids_match = torch.equal(ids_cuda, ids_triton)
        weights_close = torch.allclose(w_cuda, w_triton, atol=1e-5, rtol=1e-3)
        tei_match = torch.equal(tei_cuda, tei_triton)

        if not ids_match:
            # count mismatched rows
            mismatch_mask = (ids_cuda != ids_triton).any(dim=1)
            num_mismatch = mismatch_mask.sum().item()
        else:
            num_mismatch = 0
    else:
        ids_match = weights_close = tei_match = num_mismatch = None

    # ---- CUDA benchmark ---------------------------------------------------
    torch.cuda.synchronize()
    for _ in range(_WARMUP_ITERS):
        _run_cuda(gating_output, topk, correction_bias)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(_BENCH_ITERS):
        _run_cuda(gating_output, topk, correction_bias)
    torch.cuda.synchronize()
    elapsed_cuda = (time.perf_counter() - start) / _BENCH_ITERS * 1e6  # µs

    # ---- Triton benchmark -------------------------------------------------
    if can_triton:
        torch.cuda.synchronize()
        for _ in range(_WARMUP_ITERS):
            _run_triton(gating_output, topk, correction_bias)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(_BENCH_ITERS):
            _run_triton(gating_output, topk, correction_bias)
        torch.cuda.synchronize()
        elapsed_triton = (time.perf_counter() - start) / _BENCH_ITERS * 1e6
    else:
        elapsed_triton = None

    return {
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "topk": topk,
        "triton_eligible": can_triton,
        "cuda_us": elapsed_cuda,
        "triton_us": elapsed_triton,
        "speedup": elapsed_cuda / elapsed_triton if elapsed_triton else None,
        "ids_match": ids_match,
        "weights_close": weights_close,
        "tei_match": tei_match,
        "num_mismatch_rows": num_mismatch,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark MiniMax routing kernels")
    parser.add_argument("--num-expert", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--topk", type=int, nargs="+", default=[8])
    parser.add_argument("--num-tokens", type=str, default="1,2,4,8,16,32,64,128,256,512,1024",
                        help="Comma-separated list of num_tokens")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    num_tokens_list = [int(x) for x in args.num_tokens.split(",")]

    print(f"{'Tokens':>8} {'E':>4} {'K':>3} {'Triton?':>8} {'CUDA(µs)':>10} {'Triton(µs)':>10} {'Speedup':>8} {'IDs OK':>7} {'W OK':>6}")
    print("-" * 85)

    results = []
    for num_tokens, num_experts, topk in itertools.product(
        num_tokens_list, args.num_expert, args.topk
    ):
        if topk > num_experts:
            continue
        r = benchmark_shape(num_tokens, num_experts, topk)
        results.append(r)

        triton_str = f"{r['triton_us']:.1f}" if r["triton_us"] else "N/A"
        speedup_str = f"{r['speedup']:.2f}x" if r["speedup"] else "-"
        ids_str = "PASS" if r["ids_match"] else f"FAIL({r['num_mismatch_rows']})"
        w_str = "PASS" if r["weights_close"] else "FAIL"

        print(
            f"{r['num_tokens']:>8} {r['num_experts']:>4} {r['topk']:>3} "
            f"{'YES' if r['triton_eligible'] else 'NO':>8} "
            f"{r['cuda_us']:>10.1f} {triton_str:>10} {speedup_str:>8} "
            f"{ids_str:>7} {w_str:>6}"
        )

    if args.json:
        import json
        # make results JSON-serializable
        for r in results:
            for k in ("cuda_us", "triton_us", "speedup"):
                if r[k] is not None:
                    r[k] = round(r[k], 2)
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
