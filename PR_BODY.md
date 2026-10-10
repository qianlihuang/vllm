## Overview

Add an opt-in FlashInfer Cake path for MRV2 top-k/top-p sampling and
masked-logit filtering, with fallback to the existing paths. Related to the Cake
tracker #59725. It complements the FlashInfer 0.7.1 upgrade #60948; this branch
does not change the dependency pin.

## Claims

- With `VLLM_USE_FLASHINFER_CAKE_SAMPLER=1`, CUDA batches in which every request
  sets `1 <= top_k <= 1024` use Cake. The route is chosen from host-side
  metadata, so there is no GPU-to-host read.
- Lower sampling latency on captured Qwen3.8-Flash-Next logits (vocabulary
  248,320, top-k 20, top-p 0.95). H200, medians of three rounds:

  | Sampling rows | Triton mask → Cake (µs) | FlashInfer sampling → Cake (µs) |
  | --- | --- | --- |
  | 1 | 241.4 → 58.8 | 182.8 → 73.0 |
  | 8 | 246.3 → 62.5 | 190.7 → 75.8 |
  | 64 | 333.6 → 174.4 | 403.2 → 146.3 |

- Same retained distribution as the Triton mask on all 512 captured
  target/draft rows: 0 differing rows.
- **No resolvable end-to-end change.** Real Flash-Next serving with MTP3 gives
  median throughput deltas of −1.1% to +7.4% across concurrency 1–64. Paired
  A/B differences within a round vary in sign at every concurrency. That is
  why Cake stays opt-in.

## Validation

[Results, per-run data, plots and reproduction commands](https://github.com/qianlihuang/vllm/tree/benchmarks/cake-sampling-h200-20261010).
The serving client is upstream `vllm bench serve`.

```bash
.venv/bin/python -m pytest tests/v1/sample/test_topk_topp_cake.py \
  tests/v1/sample/test_topk_topp_sampler.py -q
# H200, FlashInfer 0.7.1: 176 passed, 1 skipped (final head)

CUDA_VISIBLE_DEVICES=7 .venv/bin/python \
  benchmarks/kernels/benchmark_cake_sampling.py \
  --logits flash_next_logits.pt --rounds 3 --repeats 50 \
  --output microbench_flash_next.json
```

**Microbenchmark.**
- It checks retained support and reports probability differences against the
  PyTorch reference. It also checks Cake's exact-k contract separately; the
  largest exact-k total variation error is 2.6e-8.
- Masking is 1.9–4.2× faster at up to 64 rows. Direct sampling is 2.3–3.0×
  faster from 1 to 256 rows.
- Timing excludes compilation, allocation, copies and restoration.
  - Masking: CUDA graph events and an untimed L2 flush.
  - Direct sampling: FlashInfer's CUPTI helper with CUDA graphs and cold L2.

**End-to-end setup.**
- Model and hardware: real BF16 Qwen3.8-Flash-Next on four H200s, TP4 + EP, MRV2.
- Speculation: native MTP3 with probabilistic draft sampling and standard
  rejection sampling.
- Workload: SPEED-Bench `throughput_1k`, thinking enabled, 512 outputs, T=1,
  k=20, p=0.95.
- Runs: three alternating A/B rounds over concurrency 1–64, on the same loaded
  model, toggling only `VLLM_USE_FLASHINFER_CAKE_SAMPLER`.
- Route probes confirm Cake calls in the enabled arm and none in the baseline.
  The probe wrappers are removed before timing.
- Serving runs with `VLLM_GPU_SYNC_CHECK=error`.

**End-to-end results.**

| Concurrency | Output tok/s, baseline → Cake (median) | Δ |
| --- | --- | --- |
| 1 | 300.5 → 297.4 | −1.1% |
| 2 | 481.8 → 480.6 | −0.2% |
| 4 | 731.5 → 741.0 | +1.3% |
| 8 | 1064.3 → 1074.3 | +0.9% |
| 16 | 1567.9 → 1594.0 | +1.7% |
| 32 | 2352.1 → 2525.5 | +7.4% |
| 64 | 3858.5 → 3865.6 | +0.2% |

- Within a round, paired differences range from −12% to +23%.
- The kernel savings predict about 1–3% of step time, below this variation.
- All runs completed with the expected output token counts.
- The two arms' acceptance lengths are within 0.06 of each other.

**Checks and CI.**
- Changed files pass pre-commit, including lint, formatting and test tethering.
- The existing GPU sampling CI collects the tests. The Cake cases need a
  FlashInfer release that exposes the API.
- Real-model validation was done outside CI on H200, because it needs the
  checkpoint and the hardware.

## Details

- **Mask path.** Filtering keeps the existing masked-logits interface, so the
  downstream Gumbel and rejection-sampling kernels are unchanged.
- **Row cutoff.** Mask dispatch limits Cake to 64 sampling rows, a conservative
  measured cutoff; the crossover depends on the logit distribution. Direct
  sampling has no cutoff.
- **Caching.** Route probes are cached. Workspaces are cached by power-of-two
  batch capacity, so the cache stays bounded and captured CUDA graphs keep valid
  views.
- **Fallbacks.** The existing path is kept for: CPU and non-CUDA platforms;
  missing metadata or API; unsupported routes; top-k above 1024; and batches
  with any row that has no top-k.
- **Behavior differences.** Cake uses exact-k and resolves ties by lower token
  ID. The PyTorch reference keeps all ties at the kth logit, and softmax
  rounding can also move boundaries. On the captured rows, Cake and the
  existing Triton mask both differ from the PyTorch reference on the same 23 of
  512 rows (maximum total variation 0.031). This is not a claim of bitwise-equal
  tokens, the same RNG stream, or downstream task quality.
- **Measurement conditions.** GPU clocks were unlocked, and the model used the
  default Triton MoE configuration without H200-specific tuning.

**Duplicate check.**
- Covered: #59725's discussion and the open PRs that reference it, plus Cake
  sampling and top-k work.
- The sparse-indexer and attention PRs address other operations.
- #58907 changes min-p. #60948 is the separate dependency upgrade.
- No open PR addresses this integration.

**AI assistance.**
- Claude Code implemented the integration and bounded the workspace cache.
- OpenAI Codex reviewed the code and prepared the H200 validation and
  benchmark artifacts.

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
