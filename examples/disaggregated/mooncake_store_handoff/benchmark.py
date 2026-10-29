# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure store handoff and its effect on requests already decoding on D."""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path

import aiohttp
from prometheus_client.parser import text_string_to_metric_families


def summary(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "p95": values[int((len(values) - 1) * 0.95)],
        "max": values[-1],
        "n": len(values),
    }


async def post(session, url, **kwargs):
    async with session.post(url, **kwargs) as response:
        text = await response.text()
        if response.status != 200:
            raise RuntimeError(f"{url}: HTTP {response.status}: {text[:400]}")
        return json.loads(text)


async def metrics(session, base):
    async with session.get(base + "/metrics") as response:
        response.raise_for_status()
        return await response.text()


def counter(text, name, **labels):
    return sum(
        sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == name
        and all(sample.labels.get(key) == value for key, value in labels.items())
    )


def store_evidence(before, after, targets):
    def delta(name, **labels):
        return sum(
            counter(after[base], name, **labels) - counter(before[base], name, **labels)
            for base in targets
        )

    return {
        "external_tokens": delta(
            "vllm:prompt_tokens_by_source_total", source="external_kv_transfer"
        ),
        "local_hit_tokens": delta(
            "vllm:prompt_tokens_by_source_total", source="local_cache_hit"
        ),
        "load_ok": delta(
            "vllm:mooncake_store_operation_total", operation="load_get", status="ok"
        ),
        "load_keys": delta(
            "vllm:mooncake_store_operation_keys_total", operation="load_get"
        ),
        "load_bytes": delta(
            "vllm:mooncake_store_operation_bytes_total", operation="load_get"
        ),
        "load_failed_keys": delta(
            "vllm:mooncake_store_operation_failed_keys_total", operation="load_get"
        ),
        "load_errors": sum(
            delta(
                "vllm:mooncake_store_operation_total",
                operation="load_get",
                status=status,
            )
            for status in ("error", "partial_failure")
        ),
    }


async def stream(session, base, payload, record):
    record.update(tokens=[], arrivals=[], started=time.perf_counter())
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    async with session.post(base + "/inference/v1/generate", json=payload) as response:
        if response.status != 200:
            raise RuntimeError(
                f"{base}: HTTP {response.status}: {(await response.text())[:400]}"
            )
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            item = json.loads(line[5:])
            if item.get("error"):
                raise RuntimeError(item["error"])
            if item.get("usage"):
                record["usage"] = item["usage"]
            for choice in item.get("choices") or []:
                ids = choice.get("token_ids") or []
                if ids:
                    record["tokens"].extend(ids)
                    record["arrivals"].append(
                        [time.perf_counter(), len(record["tokens"])]
                    )
                if choice.get("finish_reason"):
                    record["finish_reason"] = choice["finish_reason"]
    record["finished"] = time.perf_counter()


async def wait_tokens(records, count, tasks, timeout=300):
    async with asyncio.timeout(timeout):
        while not all(len(r.get("tokens", [])) >= count for r in records):
            for task in tasks:
                if task.done():
                    await task
                    raise RuntimeError("A request ended before the handoff boundary")
            await asyncio.sleep(0.05)


def token_gaps(records, start, end):
    values = []
    for record in records:
        for (t0, n0), (t1, n1) in zip(record["arrivals"], record["arrivals"][1:]):
            if start <= t1 <= end:
                values.append(1000 * (t1 - t0) / (n1 - n0))
    return summary(values)


async def run(args):
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        nonce = uuid.uuid4().hex
        tokenized = await post(
            session,
            args.prefill + "/tokenize",
            json={
                "model": args.model,
                "prompt": f"Experiment {nonce}. Explain how a database works. ",
            },
        )
        stem = tokenized["tokens"]
        prompts = []
        for i in range(args.requests):
            prefix = (
                await post(
                    session,
                    args.prefill + "/tokenize",
                    json={"model": args.model, "prompt": f"{nonce}-{i}: "},
                )
            )["tokens"]
            prompts.append(
                (prefix + stem * (args.prompt_tokens // len(stem) + 1))[
                    : args.prompt_tokens
                ]
            )

        for target in args.targets:
            await post(
                session,
                target + "/reset_prefix_cache",
                params={"reset_connector": "false"},
            )
        before_stats = await (await session.get(args.proxy + "/stats")).json()
        offset = len(before_stats["requests"])
        before = {
            base: await metrics(session, base) for base in [args.source, *args.targets]
        }
        records = [{} for _ in prompts]
        request_ids = [f"{nonce}-{i}" for i in range(len(prompts))]
        tasks = [
            asyncio.create_task(
                stream(
                    session,
                    args.proxy,
                    {
                        "model": args.model,
                        "request_id": request_ids[i],
                        "cache_salt": request_ids[i],
                        "token_ids": prompt,
                        "sampling_params": {
                            "max_tokens": args.max_tokens,
                            "temperature": 0,
                            "ignore_eos": True,
                        },
                    },
                    record,
                )
            )
            for i, (prompt, record) in enumerate(zip(prompts, records))
        ]
        background, bg_tasks = [], []
        try:
            await wait_tokens(records, max(1, args.handoff_tokens - 512), tasks)
            for i in range(args.background):
                base = args.targets[i % len(args.targets)]
                prompt = prompts[i % len(prompts)][:128]
                bg = {"target": base}
                background.append(bg)
                payload = {
                    "model": args.model,
                    "token_ids": prompt,
                    "cache_salt": f"background-{nonce}-{i}",
                    "sampling_params": {
                        "max_tokens": args.background_tokens,
                        "temperature": 0,
                        "ignore_eos": True,
                    },
                }
                p_result = await post(
                    session,
                    args.prefill + "/inference/v1/generate",
                    json=dict(
                        payload,
                        sampling_params={"max_tokens": 1},
                        kv_transfer_params={"do_remote_decode": True},
                    ),
                )
                payload["kv_transfer_params"] = p_result["kv_transfer_params"]
                bg_tasks.append(asyncio.create_task(stream(session, base, payload, bg)))
            if background:
                await wait_tokens(background, 64, bg_tasks)
            await wait_tokens(records, args.handoff_tokens, tasks)
            if background:
                async with asyncio.timeout(30):
                    while True:
                        snapshots = {
                            base: await metrics(session, base) for base in args.targets
                        }
                        if all(
                            counter(snapshots[base], "vllm:generation_tokens_total")
                            - counter(before[base], "vllm:generation_tokens_total")
                            >= 64
                            * sum(
                                i % len(args.targets) == j
                                for i in range(args.background)
                            )
                            for j, base in enumerate(args.targets)
                        ):
                            break
                        await asyncio.sleep(0.1)
            before_handoff = {
                base: await metrics(session, base)
                for base in [args.source, *args.targets]
            }
            handoff_at = time.perf_counter()
            retirement = (
                None
                if args.control
                else await post(session, args.proxy + "/retire", json={})
            )
            retired_at = time.perf_counter()
            await asyncio.gather(*tasks, *bg_tasks)
        finally:
            for task in tasks + bg_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, *bg_tasks, return_exceptions=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(
                    {"partial": True, "requests": records, "background": background},
                    indent=2,
                )
            )

        stats = await (await session.get(args.proxy + "/stats")).json()
        handoffs = stats["requests"][offset:]
        after = {
            base: await metrics(session, base) for base in [args.source, *args.targets]
        }
        cached = sum(h.get("target_cached_tokens") or 0 for h in handoffs)
        evidence = store_evidence(before_handoff, after, args.targets)
        snapshot = {
            "partial": True,
            "config": vars(args) | {"out": str(args.out)},
            "retirement": retirement,
            "pause_ms": 1000 * (retired_at - handoff_at),
            "handoffs": handoffs,
            "requests": records,
            "background": background,
            "metrics_before": before,
            "metrics_before_handoff": before_handoff,
            "metrics_after": after,
            "store_evidence": evidence,
        }
        args.out.write_text(json.dumps(snapshot, indent=2))
        if not args.control:
            async with asyncio.timeout(30):
                while evidence["external_tokens"] < cached or evidence["load_ok"] == 0:
                    after = {
                        base: await metrics(session, base)
                        for base in [args.source, *args.targets]
                    }
                    evidence = store_evidence(before_handoff, after, args.targets)
                    await asyncio.sleep(0.1)
        snapshot.update(metrics_after=after, store_evidence=evidence)
        args.out.write_text(json.dumps(snapshot, indent=2))
        cold_checks = []
        if not args.control and not args.skip_cold_reference:
            by_id = {h["request_id"]: h for h in handoffs}

            async def reference(i):
                count = by_id[request_ids[i]]["source_tokens"]
                context = prompts[i] + records[i]["tokens"][:count]
                cold = await post(
                    session,
                    by_id[request_ids[i]]["target"] + "/inference/v1/generate",
                    json={
                        "model": args.model,
                        "token_ids": context,
                        "cache_salt": f"cold-{nonce}-{i}",
                        "sampling_params": {
                            "max_tokens": 1,
                            "temperature": 0,
                            "ignore_eos": True,
                        },
                    },
                )
                cached = (
                    cold.get("usage", {})
                    .get("prompt_tokens_details", {})
                    .get("cached_tokens", 0)
                )
                return {
                    "request_id": request_ids[i],
                    "cached_tokens": cached,
                    "actual": records[i]["tokens"][count],
                    "reference": cold["choices"][0]["token_ids"][0],
                }

            cold_checks = await asyncio.gather(
                *(reference(i) for i in range(len(prompts)))
            )
            assert all(c["cached_tokens"] == 0 for c in cold_checks), (
                "Cold reference unexpectedly hit cache"
            )
        uncached_suffix = [
            h["continuation_tokens"] - h["target_cached_tokens"]
            for h in handoffs
            if h.get("target_cached_tokens") is not None
        ]
        result = {
            "partial": False,
            "config": vars(args) | {"out": str(args.out)},
            "retirement": retirement,
            "pause_ms": 1000 * (retired_at - handoff_at),
            "summary": {
                "requests": len(records),
                "handoffs": sum(h.get("source_tokens", 0) > 0 for h in handoffs),
                "store_hits": sum(
                    h.get("target_cached_tokens", 0) > 0 for h in handoffs
                ),
                "decode_kv_hits": sum(
                    (h.get("target_cached_tokens") or 0) > h["prompt_tokens"]
                    for h in handoffs
                ),
                "store_evidence": evidence,
                "uncached_suffix_tokens": uncached_suffix,
                "cold_next_token_matches": sum(
                    c["actual"] == c["reference"] for c in cold_checks
                ),
                "cold_next_token_checks": len(cold_checks),
                "target_ttft_ms": summary(
                    [h["target_ttft_ms"] for h in handoffs if "target_ttft_ms" in h]
                ),
                "handoff_gap_ms": summary(
                    [h["handoff_gap_ms"] for h in handoffs if "handoff_gap_ms" in h]
                ),
                "background_tpot_before_ms": token_gaps(
                    background, handoff_at - 3, handoff_at
                ),
                "background_tpot_during_ms": token_gaps(
                    background, handoff_at, handoff_at + 3
                ),
                "background_tpot_after_ms": token_gaps(
                    background, handoff_at + 3, handoff_at + 6
                ),
            },
            "handoffs": handoffs,
            "cold_reference": cold_checks,
            "requests": records,
            "background": background,
            "metrics_before": before,
            "metrics_before_handoff": before_handoff,
            "metrics_after": after,
            "by_target": {
                base: {
                    "target_ttft_ms": summary(
                        [
                            h["target_ttft_ms"]
                            for h in handoffs
                            if h.get("target") == base and "target_ttft_ms" in h
                        ]
                    ),
                    "background_tpot_before_ms": token_gaps(
                        [r for r in background if r["target"] == base],
                        handoff_at - 3,
                        handoff_at,
                    ),
                    "background_tpot_during_ms": token_gaps(
                        [r for r in background if r["target"] == base],
                        handoff_at,
                        handoff_at + 3,
                    ),
                    "background_tpot_after_ms": token_gaps(
                        [r for r in background if r["target"] == base],
                        handoff_at + 3,
                        handoff_at + 6,
                    ),
                }
                for base in args.targets
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result["summary"], indent=2), flush=True)
        assert len(handoffs) == args.requests, "Unexpected proxy request count"
        if not args.control:
            assert result["summary"]["handoffs"] == args.requests, (
                "Some requests did not migrate in flight"
            )
            assert result["summary"]["store_hits"] == args.requests, (
                "Cache miss: inspect before claiming full store coverage"
            )
            assert result["summary"]["decode_kv_hits"] == args.requests, (
                "Only original prompt KV was reused; decode-KV offload is not proven"
            )
            assert (
                evidence["external_tokens"] == cached
                and evidence["local_hit_tokens"] == 0
            ), "Cache attribution failed: cache counts are not proven external reads"
            assert (
                evidence["load_ok"] > 0
                and evidence["load_failed_keys"] == 0
                and evidence["load_errors"] == 0
            ), "Mooncake loads failed or were not observed"
        assert all(len(r["tokens"]) == args.max_tokens for r in records)
        await post(session, args.proxy + "/resume", json={})
        for target in args.targets:
            await post(
                session,
                target + "/pause",
                params={"mode": "wait", "clear_cache": "false"},
            )
            await post(session, target + "/resume")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default="http://127.0.0.1:8400")
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--model", default="dsv41")
    parser.add_argument("--skip-cold-reference", action="store_true")
    parser.add_argument(
        "--control",
        action="store_true",
        help="Keep source requests in place; measure target background without handoff",
    )
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--handoff-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--background", type=int, default=8)
    parser.add_argument("--background-tokens", type=int, default=4096)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
