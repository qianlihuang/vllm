# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the handoff proxy against real external HTTP services."""

import asyncio
import importlib
import json
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

pytestmark = [pytest.mark.cpu_test, pytest.mark.asyncio]


def proxy_app(source, target, prefill=None, mode="store"):
    module = importlib.import_module(
        "examples.disaggregated.mooncake_store_handoff.proxy"
    )
    return module.create_app(source, [target], prefill=prefill, mode=mode)


@pytest.mark.parametrize("status", ["ok", "partial_failure", "error"])
async def test_p2p_evidence_rejects_store_reads_including_failed_loads(status):
    benchmark = importlib.import_module(
        "examples.disaggregated.mooncake_store_handoff.benchmark"
    )
    target = "http://target"
    evidence = benchmark.store_evidence(
        {target: ""},
        {
            target: 'vllm:mooncake_store_operation_total{operation="load_get",'
            f'status="{status}"}} 1\n'
        },
        [target],
    )
    nixl = {"transfers": 1, "bytes": 1024, "errors": {"num_failed_transfers": 0}}
    with pytest.raises(AssertionError, match="another cache path"):
        benchmark.validate_migration(
            [{"source_kv_params": {"remote": "kv"}}], 1, 1, evidence, nixl
        )


async def emit(
    response, tokens, finish=None, cached=0, prompt_tokens=0, completion_tokens=0
):
    payload = {
        "choices": [
            {"token_ids": tokens, "text": "x" * len(tokens), "finish_reason": finish}
        ],
    }
    if finish:
        payload["usage"] = {
            "prompt_tokens_details": {"cached_tokens": cached},
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    await response.write(f"data: {json.dumps(payload)}\n\n".encode())


async def collect(client, url, body):
    async with client.post(url, json=body) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            return response.status, await response.json()
        tokens: list[int] = []
        terminal, done, errors, usage = [], 0, [], {}
        async for raw in response.content:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                done += 1
            elif line.startswith("data:"):
                chunk = json.loads(line[5:])
                if chunk.get("error"):
                    errors.append(chunk["error"])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    tokens.extend(choice.get("token_ids") or [])
                    if choice.get("finish_reason"):
                        terminal.append(choice["finish_reason"])
        return response.status, {
            "choices": [
                {
                    "token_ids": tokens,
                    "finish_reason": terminal[-1] if terminal else None,
                }
            ],
            "terminal": terminal,
            "done": done,
            "errors": errors,
            "usage": usage,
        }


async def wait_phase(client, base, phase):
    async with asyncio.timeout(5):
        while True:
            async with client.get(base + "/stats") as response:
                data = await response.json()
            if any(r["phase"] == phase for r in data["requests"]):
                return data
            await asyncio.sleep(0)


@pytest.mark.parametrize("streaming", [False, True])
async def test_handoff_waits_for_store_drain_and_resumes_without_prefill(streaming):
    """A source abort cannot race its pending saves; replay goes straight to D
    without stale P/D params, preserving generated tokens and the token budget."""
    started, aborted, pause_started, release = [asyncio.Event() for _ in range(4)]
    seen: dict[str, list[dict[str, Any]]] = {
        "prefill": [],
        "source": [],
        "target": [],
        "pause": [],
    }

    async def prefill(request):
        seen["prefill"].append(await request.json())
        return web.json_response({"kv_transfer_params": {"from_p": True}})

    async def source(request):
        seen["source"].append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await emit(response, [10, 11])
        started.set()
        await aborted.wait()
        await emit(response, [], "abort")
        await response.write_eof()
        return response

    async def pause(request):
        seen["pause"].append(dict(request.query))
        aborted.set()
        pause_started.set()
        await release.wait()
        return web.json_response({"status": "paused"})

    async def target(request):
        body = await request.json()
        seen["target"].append(body)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await emit(
            response,
            [12, 13, 14],
            "length",
            cached=3,
            prompt_tokens=5,
            completion_tokens=3,
        )
        await response.write_eof()
        return response

    backend = web.Application()
    backend.router.add_post("/inference/v1/generate", source)
    backend.router.add_post("/pause", pause)
    p_app, d_app = web.Application(), web.Application()
    p_app.router.add_post("/inference/v1/generate", prefill)
    d_app.router.add_post("/inference/v1/generate", target)
    async with (
        TestServer(backend) as src,
        TestServer(p_app) as p,
        TestServer(d_app) as d,
    ):
        app = proxy_app(
            str(src.make_url("")).rstrip("/"),
            str(d.make_url("")).rstrip("/"),
            str(p.make_url("")).rstrip("/"),
        )
        async with TestServer(app) as proxy, aiohttp.ClientSession() as client:
            base = str(proxy.make_url("")).rstrip("/")
            body = {
                "model": "test",
                "token_ids": [1, 2, 3],
                "sampling_params": {
                    "max_tokens": 5,
                    "extra_args": {"kv_transfer_params": {"stale": True}, "keep": 1},
                },
                "stream": streaming,
            }
            generation = asyncio.create_task(
                collect(client, base + "/inference/v1/generate", body)
            )
            retirement = None
            try:
                await asyncio.wait_for(started.wait(), 5)
                retirement = asyncio.create_task(collect(client, base + "/retire", {}))
                await asyncio.wait_for(pause_started.wait(), 5)
                await wait_phase(client, base, "waiting_for_drain")
                assert seen["target"] == []
                assert not generation.done()
                release.set()
                assert (await asyncio.wait_for(retirement, 5))[0] == 200
                status, result = await asyncio.wait_for(generation, 5)
            finally:
                release.set()
                aborted.set()
                for task in (generation, retirement):
                    if task is not None and not task.done():
                        task.cancel()
                await asyncio.gather(
                    *(t for t in (generation, retirement) if t), return_exceptions=True
                )
            assert status == 200
            assert result["choices"][0]["token_ids"] == [10, 11, 12, 13, 14]
            assert result["choices"][0]["finish_reason"] == "length"
            assert result["usage"]["prompt_tokens"] == 3
            assert result["usage"]["completion_tokens"] == 5
            assert result["usage"]["total_tokens"] == 8
            if streaming:
                assert result["terminal"] == ["length"]
                assert result["done"] == 1
            assert seen["pause"] == [{"mode": "abort", "clear_cache": "false"}]
            assert len(seen["prefill"]) == 1
            assert seen["source"][0]["kv_transfer_params"] == {"from_p": True}
            resumed = seen["target"][0]
            assert resumed["token_ids"] == [1, 2, 3, 10, 11]
            assert resumed["sampling_params"]["max_tokens"] == 3
            assert "kv_transfer_params" not in resumed
            assert resumed["sampling_params"]["extra_args"] == {"keep": 1}


@pytest.mark.parametrize("export_kv", [True, False])
async def test_migrate_requires_source_params_without_waiting_for_drain(export_kv):
    """The migrate arm continues as soon as the aborted response arrives,
    carrying the source's transfer params; it must not block on the pause."""
    started, aborted, pause_started, release = [asyncio.Event() for _ in range(4)]
    seen: dict[str, list[dict[str, Any]]] = {"source": [], "target": [], "pause": []}
    params = {"remote_engine_id": "src", "remote_num_tokens": 7}

    async def source(request):
        seen["source"].append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await emit(response, [10, 11])
        started.set()
        await aborted.wait()
        payload = {
            "choices": [{"token_ids": [], "text": "", "finish_reason": "abort"}],
        }
        if export_kv and seen["pause"][-1]["mode"] == "migrate":
            payload["kv_transfer_params"] = params
        await response.write(f"data: {json.dumps(payload)}\n\n".encode())
        await response.write_eof()
        return response

    async def pause(request):
        seen["pause"].append(dict(request.query))
        aborted.set()
        pause_started.set()
        await release.wait()
        return web.json_response({"status": "paused"})

    async def target(request):
        seen["target"].append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await emit(response, [12, 13, 14], "length")
        await response.write_eof()
        return response

    backend = web.Application()
    backend.router.add_post("/inference/v1/generate", source)
    backend.router.add_post("/pause", pause)
    d_app = web.Application()
    d_app.router.add_post("/inference/v1/generate", target)
    async with TestServer(backend) as src, TestServer(d_app) as d:
        app = proxy_app(
            str(src.make_url("")).rstrip("/"),
            str(d.make_url("")).rstrip("/"),
            mode="migrate",
        )
        async with TestServer(app) as proxy, aiohttp.ClientSession() as client:
            base = str(proxy.make_url("")).rstrip("/")
            body = {
                "model": "test",
                "token_ids": [1, 2, 3],
                "sampling_params": {"max_tokens": 5},
                "stream": True,
            }
            generation = asyncio.create_task(
                collect(client, base + "/inference/v1/generate", body)
            )
            retirement = None
            try:
                await asyncio.wait_for(started.wait(), 5)
                retirement = asyncio.create_task(collect(client, base + "/retire", {}))
                await asyncio.wait_for(pause_started.wait(), 5)
                status, result = await asyncio.wait_for(generation, 5)
                assert not release.is_set()
                if export_kv:
                    assert seen["target"], "continuation must not wait for the drain"
                else:
                    assert seen["target"] == []
                    assert result["errors"]
                    await wait_phase(client, base, "failed")
            finally:
                release.set()
                aborted.set()
                for task in (generation, retirement):
                    if task is not None and not task.done():
                        task.cancel()
                await asyncio.gather(
                    *(t for t in (generation, retirement) if t), return_exceptions=True
                )
            assert status == 200
            assert seen["pause"] == [{"mode": "migrate", "clear_cache": "false"}]
            if not export_kv:
                return
            assert result["choices"][0]["token_ids"] == [10, 11, 12, 13, 14]
            resumed = seen["target"][0]
            assert resumed["token_ids"] == [1, 2, 3, 10, 11]
            assert resumed["sampling_params"]["max_tokens"] == 3
            assert resumed["kv_transfer_params"] == params


@pytest.mark.parametrize("finish", ["stop", "length", "abort"])
async def test_natural_finish_does_not_create_a_continuation(finish):
    async def source(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await emit(response, [8], finish)
        await response.write_eof()
        return response

    backend = web.Application()
    backend.router.add_post("/inference/v1/generate", source)
    async with TestServer(backend) as src:
        app = proxy_app(str(src.make_url("")).rstrip("/"), "http://127.0.0.1:1")
        async with TestServer(app) as proxy, aiohttp.ClientSession() as client:
            status, data = await collect(
                client,
                str(proxy.make_url("/inference/v1/generate")),
                {"token_ids": [1], "sampling_params": {"max_tokens": 5}},
            )
            assert status == 200
            assert data["choices"][0]["token_ids"] == [8]
            assert data["choices"][0]["finish_reason"] == finish


async def test_failed_retirement_fails_the_request_without_reading_store():
    aborted, started = asyncio.Event(), asyncio.Event()

    async def source(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await emit(response, [8])
        started.set()
        await aborted.wait()
        await emit(response, [], "abort")
        await response.write_eof()
        return response

    async def pause(request):
        aborted.set()
        return web.json_response({"error": "store drain failed"}, status=500)

    backend = web.Application()
    backend.router.add_post("/inference/v1/generate", source)
    backend.router.add_post("/pause", pause)
    async with TestServer(backend) as src:
        app = proxy_app(str(src.make_url("")).rstrip("/"), "http://127.0.0.1:1")
        async with TestServer(app) as proxy, aiohttp.ClientSession() as client:
            base = str(proxy.make_url("")).rstrip("/")
            task = asyncio.create_task(
                collect(
                    client,
                    base + "/inference/v1/generate",
                    {"token_ids": [1], "sampling_params": {"max_tokens": 5}},
                )
            )
            await asyncio.wait_for(started.wait(), 5)
            status, _ = await collect(client, base + "/retire", {})
            assert status == 502
            status, data = await asyncio.wait_for(task, 5)
            assert status == 502
            assert "store drain failed" in data["error"]


@pytest.mark.parametrize("where", ["source", "target"])
@pytest.mark.parametrize("failure", ["sse_error", "missing_terminal"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_upstream_failure_cannot_be_reported_as_success(
    where, failure, streaming
):
    started, aborted = asyncio.Event(), asyncio.Event()

    async def failing(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await emit(response, [9])
        if failure == "sse_error":
            await response.write(b'data: {"error":{"message":"engine failed"}}\n\n')
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def source(request):
        if where == "source":
            return await failing(request)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await emit(response, [8])
        started.set()
        await aborted.wait()
        await emit(response, [], "abort")
        await response.write_eof()
        return response

    async def pause(request):
        aborted.set()
        return web.json_response({"status": "paused"})

    backend, target = web.Application(), web.Application()
    backend.router.add_post("/inference/v1/generate", source)
    backend.router.add_post("/pause", pause)
    target.router.add_post("/inference/v1/generate", failing)
    async with TestServer(backend) as src, TestServer(target) as dst:
        app = proxy_app(
            str(src.make_url("")).rstrip("/"), str(dst.make_url("")).rstrip("/")
        )
        async with TestServer(app) as proxy, aiohttp.ClientSession() as client:
            base = str(proxy.make_url("")).rstrip("/")
            task = asyncio.create_task(
                collect(
                    client,
                    base + "/inference/v1/generate",
                    {
                        "token_ids": [1],
                        "sampling_params": {"max_tokens": 5},
                        "stream": streaming,
                    },
                )
            )
            if where == "target":
                await asyncio.wait_for(started.wait(), 5)
                assert (await collect(client, base + "/retire", {}))[0] == 200
            status, data = await asyncio.wait_for(task, 5)
            assert status == (200 if streaming else 502)
            if streaming:
                assert data["errors"]
                assert data["terminal"] == []
                assert data["done"] == 1
            else:
                assert data["error"]
            await wait_phase(client, base, "failed")


async def test_retirement_during_prefill_routes_initial_request_to_target():
    entered, release, retired = asyncio.Event(), asyncio.Event(), asyncio.Event()
    seen = []

    async def prefill(request):
        entered.set()
        await release.wait()
        return web.json_response({"kv_transfer_params": {"from_p": True}})

    async def source(request):
        return web.json_response({"error": "source paused"}, status=503)

    async def pause(request):
        retired.set()
        return web.json_response({"status": "paused"})

    async def target(request):
        seen.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await emit(response, [9], "length")
        await response.write_eof()
        return response

    p_app, src_app, dst_app = [web.Application() for _ in range(3)]
    p_app.router.add_post("/inference/v1/generate", prefill)
    src_app.router.add_post("/inference/v1/generate", source)
    src_app.router.add_post("/pause", pause)
    dst_app.router.add_post("/inference/v1/generate", target)
    async with (
        TestServer(p_app) as p,
        TestServer(src_app) as src,
        TestServer(dst_app) as dst,
    ):
        app = proxy_app(
            str(src.make_url("")).rstrip("/"),
            str(dst.make_url("")).rstrip("/"),
            str(p.make_url("")).rstrip("/"),
        )
        async with TestServer(app) as proxy, aiohttp.ClientSession() as client:
            base = str(proxy.make_url("")).rstrip("/")
            task = asyncio.create_task(
                collect(
                    client,
                    base + "/inference/v1/generate",
                    {
                        "token_ids": [1, 2],
                        "sampling_params": {"max_tokens": 1},
                    },
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), 5)
                assert (await collect(client, base + "/retire", {}))[0] == 200
                assert retired.is_set()
                release.set()
                status, data = await asyncio.wait_for(task, 5)
                assert status == 200
                assert data["choices"][0]["token_ids"] == [9]
                assert seen[0]["token_ids"] == [1, 2]
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)


async def test_disconnected_client_does_not_resume_and_releases_active_record():
    aborted, disconnected = asyncio.Event(), asyncio.Event()

    async def source(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await emit(response, [8])
        try:
            await aborted.wait()
            await emit(response, [], "abort")
        finally:
            disconnected.set()
        return response

    async def resume(request):
        return web.json_response({"status": "resumed"})

    backend = web.Application()
    backend.router.add_post("/inference/v1/generate", source)
    backend.router.add_post("/resume", resume)
    async with TestServer(backend, handler_cancellation=True) as src:
        app = proxy_app(str(src.make_url("")).rstrip("/"), "http://127.0.0.1:1")
        async with TestServer(app) as proxy, aiohttp.ClientSession() as client:
            base = str(proxy.make_url("")).rstrip("/")
            try:
                response = await client.post(
                    base + "/inference/v1/generate",
                    json={
                        "token_ids": [1],
                        "sampling_params": {"max_tokens": 5},
                        "stream": True,
                    },
                )
                assert b"8" in await asyncio.wait_for(response.content.readline(), 5)
                response.close()
                await asyncio.wait_for(disconnected.wait(), 5)
                await wait_phase(client, base, "cancelled")
                assert (await collect(client, base + "/resume", {}))[0] == 200
            finally:
                aborted.set()


async def test_prefill_failure_releases_the_request_record():
    async def prefill(request):
        return web.json_response({"error": "prefill failed"}, status=500)

    async def resume(request):
        return web.json_response({"status": "resumed"})

    backend = web.Application()
    backend.router.add_post("/inference/v1/generate", prefill)
    backend.router.add_post("/resume", resume)
    async with TestServer(backend) as p:
        url = str(p.make_url("")).rstrip("/")
        app = proxy_app(url, "http://127.0.0.1:1", url)
        async with TestServer(app) as proxy, aiohttp.ClientSession() as client:
            base = str(proxy.make_url("")).rstrip("/")
            async with client.post(
                base + "/inference/v1/generate", json={"token_ids": [1]}
            ) as response:
                assert response.status == 502
                assert "prefill failed" in (await response.json())["error"]
            await wait_phase(client, base, "failed")
            assert (await collect(client, base + "/resume", {}))[0] == 200
