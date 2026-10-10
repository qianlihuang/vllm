# Cake sampling: Qwen3.8-Flash-Next on H200

This directory contains the measured results and reproduction commands for the
Cake top-k/top-p integration. The benchmark client is the upstream Python
`vllm bench serve`; the helper only alternates sampling backends and collects its
JSON results.

## Configuration

- Source: `11ccf19a170508a4fb404e1739c8a1ba37414e4b` on
  `feat/flashinfer-cake-topk-topp`, based on
  `a98247ab4db686ee03c66d5feb3c761e52a2f8ab`. Every measurement below used this
  revision. Later commits on the branch:
  - `5855bfc921` clarifies a comment.
  - `93d4108b7e` makes Cake opt-in. Both A/B arms set
    `VLLM_USE_FLASHINFER_CAKE_SAMPLER` explicitly, so this does not affect the
    measurements.
  - `2cc12a48fb` only changes how the workspace cache allocates its buffers.

  At the final head the unit suites were re-run on H200: 176 passed, 1 skipped.
- Model: `Qwen/Qwen3.8-Flash-Next`, BF16, vocabulary 248,320. Downloaded with
  ModelScope; all 131 safetensors shards passed header/file-size checks. The
  model index matches the Hugging Face snapshot
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`; index SHA256:
  `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`.
- Four H200 141 GB GPUs, TP4 + EP, native MTP with three draft tokens,
  probabilistic draft sampling and the default **standard** rejection sampler.
- Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0, FlashInfer 0.7.1,
  CUPTI Python 13.4.0, Apache TVM FFI 0.1.11,
  vLLM `0.31.1rc1.dev260+ga98247ab4`. The environment
  uses the FlashInfer 0.7.1 wheel on `PYTHONPATH` and official vLLM compiled
  extensions and companion FlashAttention modules from the exact source base
  commit `a98247ab4`. This is a Python-only patch; no native vLLM source changed.
  Image/video inputs are disabled for this text-only benchmark. The container's
  older vLLM 0.30.0 native extensions were incompatible and were replaced
  before any serving measurement.
- SPEED-Bench `throughput_1k`, thinking enabled, 512 output tokens,
  temperature 1.0, top-k 20, top-p 0.95. The prepared dataset has 1,536 rows
  (512 each of low, mixed and high entropy). The fixed local prepared snapshot
  SHA256 is `a00378d0343074d6273266972de2f0e9bf6f7275a7dba6e845207cba8db15516`.
  This is an existing prepared snapshot, not a claim that today's upstream
  dataset is byte-identical. Dataset contents are not redistributed here.
- A/B compares **the same source branch**, Cake disabled versus enabled, on
  the same loaded model and physical GPUs. It does not compare two independent
  installations of patched and unpatched vLLM.
- GPU clocks were unlocked. Microbenchmark clock/power samples are included.
- The sync checker remains enabled for both e2e arms. These are measurements
  under that diagnostic configuration; its host overhead is not removed.
  Torch profiling is configured but stopped throughout timed requests.
- Unquantized MoE uses Triton with its default configuration; no H200-specific
  tuning file was available for `E=128,N=640`. This is a comparison of the
  sampling change under a fixed configuration, not a tuned serving maximum.

## Results

**Kernels.** Captured Flash-Next logits, median GPU time in µs over three rounds
of 50 repeats ([JSON](results/microbench_flash_next.json),
[plot](results/microbench.png)).

| Sampling rows | Triton mask | Cake mask | FlashInfer sampling | Cake sampling |
| --- | --- | --- | --- | --- |
| 1 | 241.4 | 58.8 | 182.8 | 73.0 |
| 2 | 246.7 | 58.7 | 186.5 | 73.1 |
| 4 | 246.9 | 59.1 | 187.2 | 74.1 |
| 8 | 246.3 | 62.5 | 190.7 | 75.8 |
| 16 | 245.7 | 72.9 | 236.7 | 79.7 |
| 32 | 253.4 | 122.7 | 315.7 | 116.1 |
| 64 | 333.6 | 174.4 | 403.2 | 146.3 |
| 128 | 332.3 | (294.9, not dispatched) | 525.9 | 227.4 |
| 256 | 601.5 | (519.1, not dispatched) | 847.5 | 356.2 |

Masking is 1.9–4.2× faster at up to 64 rows, where Cake is dispatched. Direct
sampling is 2.3–3.0× faster at every measured size.

On all 512 captured rows, Cake's mask is identical to Triton's: 0 differing
rows, total variation 0 ([JSON](results/captured_mask_comparison.json)). Both
differ from the PyTorch reference on the same 23 rows (maximum total variation
0.031). The cause is tie handling at the kth logit and softmax rounding; see
"Correctness and scope" below. The largest exact-k total variation error is
2.6e-8.

**Serving.** Medians of three alternating A/B rounds, with min–max in
parentheses ([per-run CSV](results/pareto_runs.csv), [plot](results/pareto.png)).

| Concurrency | Baseline output tok/s | Cake output tok/s | Median Δ | Mean TPOT, ms (baseline → Cake) | Acceptance length (baseline / Cake) |
| --- | --- | --- | --- | --- | --- |
| 1 | 300.5 (278.4–305.1) | 297.4 (283.6–298.0) | −1.1% | 3.05 → 3.05 | 2.67 / 2.61 |
| 2 | 481.8 (473.3–493.5) | 480.6 (471.5–501.7) | −0.2% | 3.76 → 3.69 | 2.63 / 2.66 |
| 4 | 731.5 (726.9–804.7) | 741.0 (729.2–741.0) | +1.3% | 4.90 → 4.81 | 2.62 / 2.62 |
| 8 | 1064.3 (1039.2–1089.3) | 1074.3 (1069.3–1115.6) | +0.9% | 6.68 → 6.57 | 2.61 / 2.56 |
| 16 | 1567.9 (1472.6–1796.5) | 1594.0 (1577.8–1815.7) | +1.7% | 9.24 → 9.00 | 2.62 / 2.59 |
| 32 | 2352.1 (2272.2–2596.7) | 2525.5 (2480.9–2707.9) | +7.4% | 12.26 → 11.45 | 2.62 / 2.63 |
| 64 | 3858.5 (3466.3–3989.2) | 3865.6 (3676.1–4090.6) | +0.2% | 14.91 → 14.98 | 2.64 / 2.64 |

**This setup does not resolve an end-to-end difference.** Within one round,
the paired Cake − baseline throughput difference ranges from −12% to +23%, and
its sign varies at every concurrency. The rounds at C32 also disagree in sign.
From the kernel savings, the expected gain is about 1–3% of step time, which is
below this run-to-run variation. Every run completed all requests with the
expected output token count. The two arms' acceptance lengths are within 0.06
of each other.

## Reproduction

Use a vLLM environment that supports this model, with FlashInfer 0.7.1 or newer.
For an editable Python-only install, follow the repository's precompiled-build
instructions, pinning `VLLM_PRECOMPILED_WHEEL_COMMIT` to the source base commit.
Install `cupti-python` for the microbenchmark. Prepare
SPEED-Bench using the upstream
[preparation script](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/speed-bench/prepare.py).
Some source datasets require access permission. Use the same prepared file for
both arms and record its SHA256.

```bash
git clone --single-branch --branch feat/flashinfer-cake-topk-topp \
  https://github.com/qianlihuang/vllm.git vllm-cake
git clone --single-branch --branch benchmarks/cake-sampling-h200-20261010 \
  https://github.com/qianlihuang/vllm.git benchmark-artifacts
export CAKE_ARTIFACTS="$PWD/benchmark-artifacts"
cd vllm-cake
# Create .venv and install dependencies following vLLM's editable-build guide.
export CAKE_RUN_ROOT="$PWD/cake-results"
export CAKE_MODEL=/path/to/Qwen3.8-Flash-Next
mkdir -p "$CAKE_RUN_ROOT/results/pareto" "$CAKE_RUN_ROOT/datasets/speed_bench"
# Place the prepared throughput_1k.jsonl in the dataset directory above.
export PYTHONPATH="$CAKE_ARTIFACTS/scripts:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_GPU_SYNC_CHECK=error
export VLLM_USE_FLASHINFER_CAKE_SAMPLER=0  # set to 1 for the candidate

.venv/bin/python -m vllm.entrypoints.cli.main serve "$CAKE_MODEL" \
  --served-model-name flashnext --host 127.0.0.1 --port 8500 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$CAKE_RUN_ROOT/profiles\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}"
```

The client command can run without any helper or development endpoint:

```bash
.venv/bin/python -m vllm.entrypoints.cli.main bench serve \
  --backend vllm --base-url http://127.0.0.1:8500 \
  --model flashnext --tokenizer "$CAKE_MODEL" \
  --dataset-name speed_bench \
  --dataset-path "$CAKE_RUN_ROOT/datasets/speed_bench" \
  --speed-bench-dataset-subset throughput_1k --speed-bench-output-len 512 \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --ignore-eos --request-rate inf --num-warmups 4 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
  --extra-body '{"temperature":1.0,"top_k":20,"top_p":0.95}' \
  --seed 100 --num-prompts 64 --max-concurrency 16 \
  --save-result --result-dir "$CAKE_RUN_ROOT/results/pareto" \
  --result-filename baseline_r0_c16.json --disable-tqdm
```

Sweep concurrency 1, 2, 4, 8, 16, 32, 64, using `max(8, 4 * concurrency)`
prompts, seeds 100/101/102, and three rounds. Reverse concurrency and A/B order
in the second round. Every run includes the client's endpoint check and four
warmups; they are excluded from its timed request interval. Assert all requests
complete, output tokens equal `num_prompts * 512`, and speculative acceptance
is nonzero with acceptance length greater than one.
Request-level p95 at concurrency 1 and 2 has only eight samples per run; these
short sweeps are not production tail-latency estimates.

For the measured same-process A/B, add
`--worker-extension-cls cake_bench_extension.CakeBenchExtension` to the server
and set `VLLM_SERVER_DEV_MODE=1` while binding to localhost. Then run:

```bash
.venv/bin/python -u "$CAKE_ARTIFACTS/scripts/run_pareto.py" \
  --root "$CAKE_RUN_ROOT" --model "$CAKE_MODEL"
```

Its debug probes verify distinct routes, capture 512
logit rows, and restore the original functions before timed requests. This
capture uses 16 concurrent requests with the single-request check and warmups
disabled, so the pool is not filled by the endpoint check alone. This
debug extension is only benchmark tooling; it is not part of the PR. Steady
state serving runs with `VLLM_GPU_SYNC_CHECK=error`.

The sweep waits after the probes. In another terminal, run the microbenchmark
on a separate idle H200 once `results/probes.json` exists:

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python \
  benchmarks/kernels/benchmark_cake_sampling.py \
  --logits "$CAKE_RUN_ROOT/results/flash_next_logits.pt" \
  --rounds 3 --repeats 50 \
  --output "$CAKE_RUN_ROOT/results/microbench_flash_next.json"
echo 0 > "$CAKE_RUN_ROOT/results/microbench_flash_next.exit"
```

This command measures captured model logits, not random synthetic logits.
Mask timing uses CUDA graph events with input restoration and a 128 MiB L2
flush outside the timed interval. Sampling uses FlashInfer's CUPTI helper with
CUDA graphs and cold L2. Allocation, compilation, transfers and restoration
are excluded. Each row count uses the first N captured rows (cycling the pool
if necessary); rounds repeat the same inputs. The pool mixes target and draft
sampling calls. Model weights, logits, prompts and generated text are not
redistributed.

## Correctness and scope

Cake uses exact-k selection with lower token ID resolving ties. The existing
PyTorch reference retains all tokens tied at the kth logit. The microbenchmark
reports mask differences and total variation against that reference, and
separately verifies Cake's exact-k probability contract. Backend changes do
not promise bitwise identical tokens or RNG streams.

Cake is opt-in: set `VLLM_USE_FLASHINFER_CAKE_SAMPLER=1`. Mask dispatch uses
Cake for at most 64 sampling rows; larger mask cases fall back to Triton. The
graph shows raw Cake timing above that threshold for comparison. Direct
sampling has no 64-row cutoff. Its workspace cache is keyed by power-of-two
batch capacity, so it stays bounded. API availability, supported routes and
top-k limits are checked before dispatch. Unsupported configurations keep the
existing backend, including FlashInfer versions without Cake.

Validation performed on H200:

```bash
.venv/bin/python -m pytest tests/v1/sample/test_topk_topp_cake.py \
  tests/v1/sample/test_topk_topp_sampler.py -q
# Matched-wheel environment, tested revision 11ccf19a17: 176 passed, 1 skipped.
# Same environment, final head 2cc12a48fb: 176 passed, 1 skipped in 28.01 s.
```

The changed files passed pre-commit, including formatting, linting and the
Buildkite test-tethering check. A separate three-round 32k-vocabulary synthetic
microbenchmark covers the smaller-vocabulary default path; it is not presented
as real-model e2e evidence.
