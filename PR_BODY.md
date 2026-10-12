## Overview

Add an opt-in FlashInfer Cake path (`VLLM_USE_FLASHINFER_CAKE_SAMPLER=1`) for
MRV2 top-k/top-p sampling and masked-logit filtering. Related to #59725. It
needs FlashInfer ≥ 0.7.1 (#60948); without it, the existing paths are used.

## Claims

- Batches where every request sets `1 <= top_k <= 1024` use Cake. The route is
  chosen from host-side metadata, with no GPU sync.
- Faster kernels on captured Qwen3.8-Flash-Next logits (V=248,320, k=20,
  p=0.95) on H200:

  | Rows | Triton mask → Cake (µs) | FlashInfer sampling → Cake (µs) |
  | --- | --- | --- |
  | 1 | 241 → 59 | 183 → 73 |
  | 8 | 246 → 63 | 191 → 76 |
  | 64 | 334 → 174 | 403 → 146 |

- The retained set matches Triton on all 512 captured rows.
- End to end with MTP3: +0.8–1.4% throughput at concurrency 1. Above that the
  expected gain (<2%) is below the benchmark noise. Cake therefore stays opt-in.

## Validation

```bash
pytest tests/v1/sample/test_topk_topp_cake.py tests/v1/sample/test_topk_topp_sampler.py -q
# H200, FlashInfer 0.7.1: 176 passed, 1 skipped
```

E2E setup:
- Qwen3.8-Flash-Next on H200 with MTP3.
- SPEED-Bench `throughput_1k` through `vllm bench serve`.
- Requests carry a fixed sampling seed. Acceptance matches exactly at
  concurrency 1. Above that the arms batch differently and vLLM is not
  batch-invariant, so outputs still diverge slightly.
- Four rounds alternate which arm runs first and toggle only the env var.

E2E result, output throughput change with Cake (mean ± SE):

| Config | Concurrency 1 | Higher concurrency |
| --- | --- | --- |
| BF16, TP4+EP | +1.4% ± 0.2% | not resolved (C2–C32) |
| FP8, TP2 | +1.3% ± 0.3% | not resolved (C2–C32) |
| FP8, TP8+EP | +0.8% ± 0.3% | not resolved (C4–C256) |

How Cake applies with MTP3:
- Only the target verification mask uses top-k/top-p; draft sampling ignores
  them.
- This PR sends that mask to Cake for up to 64 rows (16 requests). The cutoff is
  conservative, not a Cake limit.

Why the gain above concurrency 1 is not resolved:
- The expected gain falls from ~2% at concurrency 1 to <1% at 8–16 requests.
  Per-point SE is 1–3%.
- Pooled over C2–C16, the measured change is +0.3% ± 0.5%.
- Non-speculative decoding uses Cake's direct sampling at any batch size. It was
  not measured end to end.

Results, plots and reproduction:
[benchmarks/cake-sampling-h200-20261010](https://github.com/qianlihuang/vllm/tree/benchmarks/cake-sampling-h200-20261010).

## Details

- Filtering keeps the masked-logits interface, so the Gumbel and rejection
  kernels are unchanged.
- The mask path uses Cake for up to 64 rows. Direct sampling has no cutoff.
- The existing paths are used on non-CUDA platforms, without Cake in
  FlashInfer, with top-k > 1024, or when any row lacks top-k.
- Cake is exact-k and breaks ties by lower token ID. Like Triton today, it can
  differ from the PyTorch reference at the kth logit: 23 of 512 rows, max total
  variation 0.031.
- No open PR integrates Cake sampling. I checked #59725 and the PRs that
  reference it; #58907 is about min-p.
- AI assistance: Claude Code and OpenAI Codex.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [ ] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [ ] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [ ] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [ ] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
