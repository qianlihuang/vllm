# vLLM #64817: convolution + FlashKDA profiles

Operator-chain traces on one H200, BF16 Q/K/V, FP32 convolution weights and state, D128, 8192 query tokens, 96 local heads, four sequences of 2048 tokens. These are synthetic operator inputs, not a full-model serving trace or a measured serving speedup.

- `three-conv.pt.trace.json.gz`: the original three-convolution producer from vLLM `59d58fd261b1e0726bcc9030b756c1e923183583`.
- `packed-conv.pt.trace.json.gz`: the packed producer from PR #64817 at `cdd8c35258f06fc45106f67a9bd366ec3a9624f2`.

Both arms use the same FlashKDA `47fc9fec6661d04aa9a42e9e5e069a51ad3b649b` Tensor entry point. The traces contain convolution and FlashKDA only. They show three convolution launches becoming one; the native beta transpose remains one copy in both arms. The old Q/K/V outputs were already dense.

The later raw-projection/checkpoint correction does not change the timed operator chain. These traces do not validate checkpoint export, the complete Kimi K3 model, or Kimi Linear 48B. Kimi Linear 48B selects another KDA implementation and does not exercise this change.

Open the gzip-compressed Chrome traces in Perfetto or a compatible torch-profiler viewer. Trace timing is diagnostic; use unprofiled measurements for latency comparisons.
