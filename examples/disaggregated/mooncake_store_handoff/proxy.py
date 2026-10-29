# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental token-API proxy for retirement through a shared KV store."""

import argparse
import asyncio
import itertools
import json
import time
from pathlib import Path

import aiohttp
from aiohttp import web


class StoreHandoffProxy:
    def __init__(self, source, targets, prefill=None, mode="store"):
        if not targets:
            raise ValueError("At least one target decode URL is required")
        if mode not in ("store", "migrate"):
            raise ValueError(f"Unknown handoff mode: {mode}")
        self.source = source.rstrip("/")
        self.targets = itertools.cycle(t.rstrip("/") for t in targets)
        self.prefill = prefill.rstrip("/") if prefill else None
        self.mode = mode
        self.session = None
        self.retirement = None
        self.retire_received = None
        self.admission = asyncio.Lock()
        self.records = []
        self.pauses = []

    async def lifecycle(self, app):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3600),
            connector=aiohttp.TCPConnector(limit=0),
        )
        yield
        if self.retirement is not None and not self.retirement.done():
            self.retirement.cancel()
            await asyncio.gather(self.retirement, return_exceptions=True)
        await self.session.close()

    async def post(self, base, path, **kwargs):
        async with self.session.post(base + path, **kwargs) as response:
            text = await response.text()
            if response.status != 200:
                raise RuntimeError(
                    f"{base}{path}: HTTP {response.status}: {text[:400]}"
                )
            result = json.loads(text)
            if result.get("error"):
                raise RuntimeError(f"{base}{path}: {result['error']}")
            return result

    async def prefill_params(self, body):
        if self.prefill is None:
            return None
        params = dict(body.get("sampling_params", {}), max_tokens=1)
        payload = dict(body, sampling_params=params, stream=False)
        payload.pop("stream_options", None)
        payload["kv_transfer_params"] = {"do_remote_decode": True}
        result = await self.post(self.prefill, "/inference/v1/generate", json=payload)
        return result.get("kv_transfer_params")

    async def chunks(self, base, body):
        payload = dict(body, stream=True, stream_options={"include_usage": True})
        async with self.session.post(
            base + "/inference/v1/generate", json=payload
        ) as r:
            if r.status != 200:
                raise RuntimeError(f"{base}: HTTP {r.status}: {(await r.text())[:400]}")
            async for raw in r.content:
                line = raw.decode().strip()
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    item = json.loads(line[5:])
                    if item.get("error"):
                        raise RuntimeError(f"{base}: {item['error']}")
                    yield item

    async def generate(self, request):
        body = await request.json()
        prompt = list(body["token_ids"])
        params = dict(body.get("sampling_params", {}))
        if params.get("extra_args"):
            extra = dict(params["extra_args"])
            extra.pop("kv_transfer_params", None)
            params["extra_args"] = extra
        budget = int(params.get("max_tokens", 16))
        if budget < 1 or params.get("n", 1) != 1:
            raise web.HTTPBadRequest(text="This experiment needs n=1 and max_tokens>0")
        record = {
            "id": len(self.records),
            "request_id": body.get("request_id", str(len(self.records))),
            "phase": "prefill",
            "prompt_tokens": len(prompt),
        }
        self.records.append(record)
        outgoing = dict(body, sampling_params=dict(params, max_tokens=budget))
        tokens, text, usage, finish = [], "", {}, None
        streaming = bool(body.get("stream", False))
        response = (
            web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            if streaming
            else None
        )

        async def attempt(url, payload, resumed=False, admission=None):
            nonlocal text, usage, finish
            finish = None
            started = time.perf_counter()
            try:
                async for chunk in self.chunks(url, payload):
                    if admission is not None:
                        admission.release()
                        admission = None
                    choices = chunk.get("choices") or []
                    if chunk.get("metrics"):
                        record["target_metrics" if resumed else "source_metrics"] = (
                            chunk["metrics"]
                        )
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                        if resumed:
                            record["target_usage"] = usage
                            record["target_cached_tokens"] = usage.get(
                                "prompt_tokens_details", {}
                            ).get("cached_tokens")
                    if not resumed and chunk.get("kv_transfer_params"):
                        record["source_kv_params"] = chunk["kv_transfer_params"]
                    for choice in choices:
                        new_tokens = choice.get("token_ids") or []
                        if new_tokens:
                            now = time.perf_counter()
                            if resumed and "target_ttft_ms" not in record:
                                record["target_ttft_ms"] = 1000 * (now - started)
                                if self.retire_received is not None:
                                    record["retire_to_first_token_ms"] = 1000 * (
                                        now - self.retire_received
                                    )
                                last = record.get(
                                    "source_last_token_at", record["handoff_at"]
                                )
                                record["handoff_gap_ms"] = 1000 * (now - last)
                            if not resumed:
                                record["source_last_token_at"] = now
                            tokens.extend(new_tokens)
                            text += choice.get("text") or ""
                            if response is not None:
                                visible = dict(
                                    chunk, choices=[dict(choice, finish_reason=None)]
                                )
                                visible.pop("usage", None)
                                await response.write(
                                    f"data: {json.dumps(visible)}\n\n".encode()
                                )
                        if choice.get("finish_reason") is not None:
                            finish = choice["finish_reason"]
            finally:
                if admission is not None:
                    admission.release()
            if finish is None:
                raise RuntimeError(f"{url}: stream ended without a terminal outcome")

        try:
            kv_params = await self.prefill_params(body)
            outgoing.pop("kv_transfer_params", None)
            if kv_params is not None:
                outgoing["kv_transfer_params"] = kv_params
            if response is not None:
                await response.prepare(request)
            await self.admission.acquire()
            use_source = self.retirement is None
            record["phase"] = "source" if use_source else "target"
            if use_source:
                await attempt(self.source, outgoing, admission=self.admission)
            else:
                self.admission.release()
                await attempt(next(self.targets), outgoing)
            initial_usage = usage
            if (
                use_source
                and finish == "abort"
                and self.retirement is not None
                and len(tokens) < budget
            ):
                record.update(
                    phase="waiting_for_drain",
                    source_tokens=len(tokens),
                    handoff_at=time.perf_counter(),
                )
                if self.mode == "migrate":
                    record["phase"] = "dispatching"
                else:
                    await asyncio.shield(self.retirement)
                base = next(self.targets)
                context = prompt + tokens
                record.update(
                    phase="target", target=base, continuation_tokens=len(context)
                )
                outgoing = dict(
                    body,
                    token_ids=context,
                    sampling_params=dict(params, max_tokens=budget - len(tokens)),
                )
                outgoing.pop("kv_transfer_params", None)
                if self.mode == "migrate":
                    resumed_from_source = record.get("source_kv_params")
                    if resumed_from_source is None:
                        record["migrated_without_kv"] = True
                        raise RuntimeError(
                            "Migrate response has no exported KV parameters"
                        )
                    outgoing["kv_transfer_params"] = resumed_from_source
                await attempt(base, outgoing, resumed=True)
            record.update(
                phase="complete", completion_tokens=len(tokens), finish_reason=finish
            )
            final = {
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "token_ids": tokens,
                        "finish_reason": finish,
                    }
                ],
                "usage": dict(
                    initial_usage,
                    prompt_tokens=len(prompt),
                    completion_tokens=len(tokens),
                    total_tokens=len(prompt) + len(tokens),
                ),
            }
            if response is None:
                return web.json_response(final)
            terminal = dict(
                final, choices=[dict(final["choices"][0], text="", token_ids=[])]
            )
            await response.write(
                f"data: {json.dumps(terminal)}\n\ndata: [DONE]\n\n".encode()
            )
            await response.write_eof()
            return response
        except (asyncio.CancelledError, ConnectionError):
            record["phase"] = "cancelled"
            raise
        except (RuntimeError, aiohttp.ClientError, TimeoutError) as exc:
            record.update(phase="failed", error=str(exc))
            if response is None or not response.prepared:
                return web.json_response({"error": str(exc)}, status=502)
            await response.write(
                f"data: {json.dumps({'error': str(exc)})}\n\ndata: [DONE]\n\n".encode()
            )
        await response.write_eof()
        return response

    async def drain(self):
        started = time.perf_counter()
        mode = "migrate" if self.mode == "migrate" else "abort"
        result = await self.post(
            self.source, "/pause", params={"mode": mode, "clear_cache": "false"}
        )
        self.pauses.append({"seconds": time.perf_counter() - started, "status": result})
        return result

    async def retire(self, request):
        self.retire_received = time.perf_counter()
        async with self.admission:
            if self.retirement is None:
                self.retirement = asyncio.create_task(self.drain())
        try:
            await asyncio.shield(self.retirement)
            return web.json_response({"status": "retired", "pauses": self.pauses})
        except (RuntimeError, aiohttp.ClientError, TimeoutError) as exc:
            return web.json_response({"error": str(exc)}, status=502)

    async def resume(self, request):
        if self.retirement is not None and not self.retirement.done():
            raise web.HTTPConflict(text="Wait for the source drain before resetting")
        if any(
            r["phase"] not in ("complete", "failed", "cancelled") for r in self.records
        ):
            raise web.HTTPConflict(
                text="Wait for all continuations before resetting the experiment"
            )
        try:
            await self.post(self.source, "/resume")
        except (RuntimeError, aiohttp.ClientError, TimeoutError) as exc:
            return web.json_response({"error": str(exc)}, status=502)
        self.retirement = None
        return web.json_response({"status": "resumed"})

    async def stats(self, request):
        return web.json_response(
            {"mode": self.mode, "requests": self.records, "pauses": self.pauses}
        )


PROXY_KEY = web.AppKey("store_handoff_proxy", StoreHandoffProxy)


def create_app(source, targets, prefill=None, mode="store"):
    proxy = StoreHandoffProxy(source, targets, prefill, mode)
    app = web.Application(handler_args={"handler_cancellation": True})
    app[PROXY_KEY] = proxy
    app.cleanup_ctx.append(proxy.lifecycle)
    app.router.add_post("/inference/v1/generate", proxy.generate)
    app.router.add_post("/retire", proxy.retire)
    app.router.add_post("/resume", proxy.resume)
    app.router.add_get("/stats", proxy.stats)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--prefill")
    parser.add_argument(
        "--mode",
        choices=("store", "migrate"),
        default="store",
        help=(
            "store: wait for the source drain and reload the saved prefix; "
            "migrate: forward the source's kv_transfer_params as soon as the "
            "aborted response arrives"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8400)
    parser.add_argument("--stats-file", type=Path)
    args = parser.parse_args()
    app = create_app(args.source, args.targets, args.prefill, args.mode)
    if args.stats_file:

        async def save_stats(app):
            proxy = app[PROXY_KEY]
            args.stats_file.write_text(
                json.dumps(
                    {"requests": proxy.records, "pauses": proxy.pauses}, indent=2
                )
            )

        app.on_shutdown.append(save_stats)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
