"""Lossless capture proxy for the lab vLLM endpoint.

Sits in front of a vLLM server and records every OpenAI-compatible call so the raw
model traffic is recoverable for later analysis. It is a transparent relay: clients
point at this port instead of the server and need no other change.

Records, one JSON object per line:
  ts, method, path, model, stream, status, latency_s, request, response,
  usage, error

Streaming replies are relayed chunk-by-chunk to the client while being accumulated,
so a streamed response is still logged whole once it completes.

Env: PROXY_LOG (required), PROXY_PORT (8000), PROXY_UPSTREAM (http://127.0.0.1:8000),
     PROXY_HOST (127.0.0.1)
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

LOG_PATH = os.environ.get("PROXY_LOG", "api.jsonl")
HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("PROXY_PORT", "8000"))
UPSTREAM = os.environ.get("PROXY_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")

_client: httpx.AsyncClient | None = None
_log = None


def _write(record: dict[str, Any]) -> None:
    _log.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    _log.flush()


def _first_present(message: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty one of several aliases.

    vLLM reports the reasoning trace as ``reasoning`` in some versions and
    ``reasoning_content`` in others; the chat template can also emit ``reasoning_details``.
    Capture whichever exists so a trace is never silently dropped.
    """
    for key in keys:
        value = message.get(key)
        if value:
            return value
    return ""


REASONING_KEYS = ("reasoning_content", "reasoning", "reasoning_details")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client, _log
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    _log = open(LOG_PATH, "a", encoding="utf-8")           # noqa: SIM115 - lives for process
    # Long timeout: hard LCB problems can think for many minutes.
    _client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=30.0))
    print(f"[proxy] listening {HOST}:{PORT} -> {UPSTREAM}  log={LOG_PATH}", flush=True)
    try:
        yield
    finally:
        await _client.aclose()
        _log.close()


app = FastAPI(lifespan=lifespan)


@app.get("/_proxy/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "upstream": UPSTREAM, "log": LOG_PATH}


EXTRA_HEADERS = {"content-encoding", "content-length", "transfer-encoding", "connection"}


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def relay(path: str, request: Request):
    """Transparently forward a request upstream, logging request + response."""
    assert _client is not None
    started = time.time()
    raw_body = await request.body()

    payload: Any = None
    if raw_body:
        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = raw_body.decode("utf-8", "replace")

    model = payload.get("model") if isinstance(payload, dict) else None
    stream = bool(payload.get("stream")) if isinstance(payload, dict) else False

    upstream_url = f"{UPSTREAM}/{path}"
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"
    # Drop hop-by-hop AND body-framing headers. The body is re-serialized by httpx from
    # `payload`, so a forwarded Content-Length/Transfer-Encoding would describe the ORIGINAL
    # bytes and trip "Too little data for declared Content-Length".
    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "content-encoding", "transfer-encoding", "connection"}
    }

    base = {
        "ts": time.time(),
        "method": request.method,
        "path": path,
        "model": model,
        "stream": stream,
        "request": payload,
    }

    if stream:
        async def body_iter():
            chunks: list[str] = []
            status = 0
            error: str | None = None
            try:
                async with _client.stream(
                    request.method, upstream_url, json=payload, headers=forward_headers
                ) as upstream:
                    status = upstream.status_code
                    async for chunk in upstream.aiter_text():
                        chunks.append(chunk)
                        yield chunk
            except Exception as exc:                      # noqa: BLE001 - must log all failures
                error = f"{type(exc).__name__}: {exc}"
                yield f"data: {json.dumps({'error': error})}\n\n"
            finally:
                text = "".join(chunks)
                completion: Any = None
                usage: Any = None
                finish: Any = None
                reasoning_parts: list[str] = []
                content_parts: list[str] = []
                for line in text.splitlines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        # vLLM/Qwen may expose the trace as either field depending on
                        # version and parser config; capture both or the trace is lost.
                        for key in ("reasoning_content", "reasoning"):
                            if delta.get(key):
                                reasoning_parts.append(delta[key])
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                if content_parts or reasoning_parts:
                    trace = "".join(reasoning_parts)
                    completion = {
                        "content": "".join(content_parts),
                        "reasoning_content": trace,
                        "reasoning_text": trace,   # unified key, matches non-streaming
                        "finish_reason": finish,
                    }
                _write(
                    {
                        **base,
                        "status": status,
                        "latency_s": round(time.time() - started, 3),
                        "response": completion,
                        "raw_sse": text,
                        "usage": usage,
                        "error": error,
                    }
                )

        return StreamingResponse(body_iter(), media_type="text/event-stream")

    # ---- non-streaming ----
    try:
        upstream = await _client.request(
            request.method, upstream_url, json=payload, headers=forward_headers
        )
        try:
            response_body: Any = upstream.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            response_body = upstream.text
        # Normalise the reasoning field so downstream analysis has one key to read.
        if isinstance(response_body, dict):
            for choice in response_body.get("choices") or []:
                message = choice.get("message")
                if isinstance(message, dict):
                    trace = _first_present(message, REASONING_KEYS)
                    if trace:
                        message["reasoning_text"] = trace
        _write(
            {
                **base,
                "status": upstream.status_code,
                "latency_s": round(time.time() - started, 3),
                "response": response_body,
                "usage": (response_body or {}).get("usage")
                if isinstance(response_body, dict)
                else None,
                "error": None,
            }
        )
        return JSONResponse(
            content=response_body,
            status_code=upstream.status_code,
            headers={k: v for k, v in upstream.headers.items() if k.lower() not in EXTRA_HEADERS},
        )
    except Exception as exc:                              # noqa: BLE001
        _write(
            {
                **base,
                "status": 0,
                "latency_s": round(time.time() - started, 3),
                "response": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return JSONResponse(
            content={"error": f"proxy upstream failure: {type(exc).__name__}: {exc}"},
            status_code=502,
        )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
