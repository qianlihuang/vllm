# Experimental decode handoff through Mooncake

The [Mooncake store handoff example](../examples/disaggregated/mooncake_store_handoff.md)
continues an active generation on another decode instance using saved decode KV.
It is an opt-in experiment using existing scale-out and pause APIs; serving
behavior is unchanged unless the example proxy is run.

Enable `save_decode_cache` in `MooncakeStoreConnector`, share a compatible
store across instances, and run the example proxy in front of a dedicated source
decode instance. On retirement, the proxy aborts and drains the source, then sends
the original prompt plus generated tokens directly to a target decode instance.
The target reads the available prefix from the store and computes the suffix.

The example includes a benchmark for cache coverage, remaining replay, the
output pause and the impact on existing target requests. The store must survive
source retirement. Sampler and structured-output state are not transferred;
linear-attention/SSM decode-state offload is not implemented by this example.
See the example for configuration, limitations and reproduction commands.
