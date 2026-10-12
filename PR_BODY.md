## Overview<!-- markdownlint-disable -->

Add an opt-in FlashInfer Cake path for MRV2 top-k/top-p sampling and masked-logit filtering, with fallback to the existing paths. Related to the Cake tracker #59725. It needs FlashInfer ≥ 0.7.1 (#60948); without it, the existing paths are used.

## Claims



- Batches where every request sets `1 <= top_k <= 1024` use Cake. The route is chosen from host-side metadata, with no GPU sync.
- Faster kernels on captured Qwen3.8-Flash-Next logits (V=248,320, k=20, p=0.95) on H200:

  | Rows | Triton mask → Cake (µs) | FlashInfer sampling → Cake (µs) |
  | --- | --- | --- |
  | 1 | 241 → 59 | 183 → 73 |
  | 8 | 246 → 63 | 191 → 76 |
  | 64 | 334 → 174 | 403 → 146 |

- The retained set matches Triton on all 512 captured rows.

## Validation



```bash
pytest tests/v1/sample/test_topk_topp_cake.py tests/v1/sample/test_topk_topp_sampler.py -q
# H200, FlashInfer 0.7.1: 176 passed, 1 skipped
```

E2E on Qwen3.8-Flash-Next with MTP3 (BF16 TP4+EP, FP8 TP2, FP8 TP8+EP): +0.8–1.4% throughput at concurrency 1. Above that, the expected gain is below run-to-run noise.

Results, plots and reproduction:
[benchmarks/cake-sampling-h200-20261010](https://github.com/qianlihuang/vllm/tree/benchmarks/cake-sampling-h200-20261010).

## Details

No open PR integrates Cake sampling. I checked #59725 and the PRs that reference it; #58907 is about min-p.

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


