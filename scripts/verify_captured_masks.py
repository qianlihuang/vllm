"""Compare the retained distributions across the full untimed capture pool."""

import argparse
import json
from pathlib import Path

import torch

from vllm.v1.sample.ops.topk_topp_cake import apply_top_k_top_p_cake
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = torch.load(args.logits, map_location="cuda", weights_only=True).float()
    assert torch.isfinite(pool).any(-1).all()
    totals = {name: {"mask_diff_rows": 0, "tv_sum": 0.0, "tv_max": 0.0}
              for name in ("cake_vs_triton", "cake_vs_pytorch", "triton_vs_pytorch")}
    for rows in pool.split(64):
        k = torch.full((len(rows),), 20, device="cuda", dtype=torch.int32)
        p = torch.full((len(rows),), 0.95, device="cuda")
        masked = {
            "cake": apply_top_k_top_p_cake(rows.clone(), k, p, 20),
            "triton": apply_top_k_top_p_triton(rows.clone(), k, p),
            "pytorch": apply_top_k_top_p_pytorch(rows.clone(), k, p),
        }
        for logits in masked.values():
            assert torch.isfinite(logits).any(-1).all()
        assert (masked["cake"] > -torch.inf).sum(-1).max() <= 20
        for name, total in totals.items():
            left, right = (masked[x] for x in name.split("_vs_"))
            total["mask_diff_rows"] += int(((left > -torch.inf) != (right > -torch.inf)).any(-1).sum())
            tv = (left.double().softmax(-1) - right.double().softmax(-1)).abs().sum(-1) / 2
            total["tv_sum"] += float(tv.sum())
            total["tv_max"] = max(total["tv_max"], float(tv.max()))
    for total in totals.values():
        total["tv_mean"] = total.pop("tv_sum") / len(pool)
    result = {"captured_rows": len(pool), "vocab": pool.shape[-1], "chunk_rows": 64,
              "top_k": 20, "top_p": 0.95, "comparisons": totals}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
