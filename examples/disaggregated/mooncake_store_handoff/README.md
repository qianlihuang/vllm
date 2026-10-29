# Decode handoff through Mooncake Store

This experiment continues active requests on another decode engine using the
existing shared prefix cache. It adds no engine pause mode or KV transport
protocol. It uses the scale-out token API and supports one completion per
request. It is an experimental controller, not a production request router.

1. A prefill engine optionally provides ordinary NIXL P/D transfer parameters
   for each initial request. The source D saves decode KV to Mooncake.
2. `POST /retire` stops selecting the source and calls its existing
   `/pause?mode=abort&clear_cache=false` endpoint.
3. The proxy retains each request's generated tokens. After the pause returns,
   it sends the prompt plus those tokens and the remaining output budget
   directly to a target D, without calling P or forwarding NIXL parameters.
4. The target loads the store prefix and recomputes whatever suffix was not
   saved. The client sees the original output followed by the continuation.

The pause must drain pending connector jobs. Job completion means the GPU
blocks are no longer being read; failed stores can still cause cache misses.
Check the measured target hits and replay counts instead of assuming that
all computed KV was saved. No fixed store-settling sleep is used here.
Initial source admission waits for its first upstream event before allowing
retirement to pause the engine. A request still awaiting P when retirement
starts is sent to a target. Disconnected clients cancel their upstream request.

## Deployment

Use compatible model, KV layout, block-hash settings and a shared Mooncake
master/store on the instances. The source needs `save_decode_cache: true`:

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
      {
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"save_decode_cache": true}
      }
    ]
  }
}
```

Enable `--enable-scale-out`, prefix caching and prompt-token details. The
`/pause` control endpoint additionally requires `VLLM_SERVER_DEV_MODE=1`.
On NVIDIA hosts without `nvidia-peermem`, Mooncake can use DMA-BUF through
`WITH_NVIDIA_PEERMEM=0`; validate actual GPU registration before serving.

The store must remain available when the source exits. One option is zero
Mooncake global-segment capacity on the retiring source and nonzero capacity
on surviving workers. Do not reset the external store during the handoff.
Use `mode: "standalone-store"` for a zero-capacity client; `embedded` requires
positive global-segment capacity.

```bash
python examples/disaggregated/mooncake_store_handoff/proxy.py \
  --prefill http://P:9000 --source http://D1:9001 \
  --targets http://D2:9002 http://D3:9003 --port 8400 \
  --stats-file handoff-stats.json
```

Submit `/inference/v1/generate` requests to the proxy using `token_ids` and
`sampling_params`. Both streaming and nonstreaming responses are supported.
The proxy's `/retire`, `/resume` and `/stats` endpoints control and inspect
the experiment. Bind it only to a trusted experiment network: these control
endpoints intentionally have no production authentication.

```bash
python examples/disaggregated/mooncake_store_handoff/benchmark.py \
  --prefill http://P:9000 --source http://D1:9001 \
  --targets http://D2:9002 http://D3:9003 \
  --handoff-tokens 2048 --background 8 --out result-2k.json
```

The driver waits for generated-token progress, retires the source, and
records the full request outputs, source/target metrics, cache coverage,
uncached suffix length, target TTFT, and existing target requests' chunk-normalized
output gaps before/during/after handoff. These client gaps include network
and scheduler effects; they are not measurements of GPU step duration.
Handoff and background requests use distinct cache salts. Target local APC
is reset before each trial. The pre-retirement metrics snapshot excludes
background admission; external-transfer token deltas must match handoff cache
counts, local-hit deltas must be zero, and Mooncake load RPCs must succeed.
The per-request cache counts alone combine local and external cache hits.
The uncached suffix is context length minus reported cached tokens. It excludes
any additional bounded replay of a cached sliding window. Current DSV4.1 with
SWA bounded replay enabled rebuilds that window after a store hit; record the
model's window and serving setting separately. Total replayed GPU work is not
measured by this driver. TTFT and output gaps include its latency cost.
Use warmed servers and several repeated trials. Run with `--background 0`
for isolated handoff and around 2K/6K generated tokens for different contexts.
Add `--control` to keep requests on the source and measure background decode
without admitting the handoff burst. Both arms use the same main engine;
the experimental controller changes request placement, not model execution.
After each handoff run, a separately salted cold recompute checks the first
continuation token. Exact agreement is reported, not assumed: vLLM kernels
are not batch invariant. Cold checks run outside the measured handoff window.

Generated tokens become prompt tokens on the target. Sampler/RNG, penalties
and structured-output state are not transferred. This example neither adds
SSM decode-state offload nor promises zero replay or batch-invariant tokens.
Use dedicated source instances because `/pause` aborts every source request.

Tests use external HTTP services to verify drain ordering, direct target
dispatch, token-budget preservation and failure propagation:

```bash
pytest tests/entrypoints/mooncake_store_handoff/test_proxy.py
```
