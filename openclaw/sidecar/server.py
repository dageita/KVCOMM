#!/usr/bin/env python3
"""
KVCOMM OpenClaw sidecar: OpenAI-compatible HTTP front for HF KVCOMM engine or vLLM proxy.

OpenClaw provider baseUrl points here (e.g. http://127.0.0.1:8100/v1).
When inference_mode=kv_reuse, requests carry X-KVCOMM-Mode: kv_reuse (or extra_body.kvcomm).

Set KVCOMM_HF_MODEL (or KVCOMM_HF_MODEL_PATH) to enable the real HF KVCOMM engine.
Use a local path (e.g. /models/Qwen3-32B) or Hub id Qwen/Qwen3-32B (auto-resolves to /models/Qwen3-32B if present).
Multi-GPU: KVCOMM_HF_DEVICE=2,3 (device_map=auto). Single 44GB GPU is too small for Qwen3-32B inference.
Otherwise dense_prefill proxies to vLLM.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# Ensure KVCOMM package is importable when launched from openclaw/sidecar.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sidecar.kvcomm_adapter import (
    SIDECAR_VERSION,
    _engine_enabled,
    _kv_reuse_engine_label,
    configure_hf_engine,
    engine_loaded,
    get_adapter,
    pending_context_depth,
    register_pending_context,
    release_adapter,
    resolve_hf_model_path,
    resolve_request_mode,
)
from sidecar.openclaw_prefix import PrefixOverflowError

UPSTREAM_BASE = os.environ.get("KVCOMM_VLLM_UPSTREAM", "http://127.0.0.1:8001/v1").rstrip("/")
LISTEN_HOST = os.environ.get("KVCOMM_SIDECAR_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("KVCOMM_SIDECAR_PORT", "8100"))
DEFAULT_MODE = os.environ.get("KVCOMM_MODE", "dense_prefill")


def _env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _dense_via_hf_enabled() -> bool:
    """Route dense_prefill through HF when engine is configured (default: yes)."""
    raw = os.environ.get("KVCOMM_DENSE_VIA_HF", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return _engine_enabled()


def _vllm_fallback_allowed() -> bool:
    """When HF engine is configured, do not silently proxy to vLLM unless explicitly allowed."""
    if not _engine_enabled():
        return True
    return _env_truthy("KVCOMM_ALLOW_VLLM_FALLBACK")


def _reject_vllm_proxy(message: str, *, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "invalid_request_error"}},
        headers={"X-KVCOMM-Route": "hf", "X-KVCOMM-VLLM-Fallback": "denied"},
    )


def _sidecar_access_log_enabled() -> bool:
    """Uvicorn per-request access lines (noisy during tool-call agent loops)."""
    return _env_truthy("KVCOMM_SIDECAR_ACCESS_LOG", default=False)

app = FastAPI(title="KVCOMM OpenClaw Sidecar", version=SIDECAR_VERSION)
_logger = logging.getLogger("kvcomm-sidecar")

_metrics: dict[str, Any] = {
    "requests_total": 0,
    "requests_by_mode": {"dense_prefill": 0, "kv_reuse": 0},
    "requests_by_route": {"hf": 0, "proxy": 0},
    "last_request_ms": None,
}


def _resolve_mode(request: Request) -> str:
    header = request.headers.get("x-kvcomm-mode", "").strip()
    if header in ("dense_prefill", "kv_reuse"):
        return header
    return DEFAULT_MODE if DEFAULT_MODE in ("dense_prefill", "kv_reuse") else "dense_prefill"


def _wants_stream_usage(body: dict[str, Any]) -> bool:
    opts = body.get("stream_options")
    if isinstance(opts, dict):
        return bool(opts.get("include_usage"))
    return False


def _completion_to_sse_body(payload: dict[str, Any], *, include_usage: bool) -> str:
    """Convert a non-streaming chat.completion into OpenAI SSE chunks for OpenClaw."""
    from sidecar.tool_bridge import completion_payload_to_sse

    return completion_payload_to_sse(payload, include_usage=include_usage)


def _format_engine_error(exc: Exception) -> str:
    """Unwrap tenacity RetryError so gateway logs show the root cause."""
    try:
        from tenacity import RetryError
    except ImportError:
        return str(exc)
    if isinstance(exc, RetryError):
        last = exc.last_attempt
        if last is not None and last.failed:
            inner = last.exception()
            if inner is not None:
                return f"{type(inner).__name__}: {inner}"
    return str(exc)


def _hf_load_plan_label() -> str | None:
    if not _engine_enabled():
        return None
    if not engine_loaded():
        device = os.environ.get("KVCOMM_HF_DEVICE", "").strip()
        if device:
            return f"pending_load selected_gpus={device} (engine not loaded)"
        return "pending_load (engine not loaded)"
    try:
        from KVCOMM.llm.gpt_chat import LLMChat

        return LLMChat.describe_hf_load_plan(resolve_hf_model_path())
    except Exception:
        return None


@app.get("/health")
async def health() -> dict[str, Any]:
    dense_via_hf = _dense_via_hf_enabled()
    try:
        from sidecar.openclaw_prefix import prefix_max_tokens

        prefix_cap = prefix_max_tokens()
    except Exception:
        prefix_cap = None
    return {
        "status": "ok",
        "upstream": UPSTREAM_BASE,
        "default_mode": DEFAULT_MODE,
        "dense_via_hf": dense_via_hf,
        "prefix_max_tokens": prefix_cap,
        "sidecar": "kvcomm-openclaw",
        "sidecar_version": SIDECAR_VERSION,
        "kv_reuse_engine": _kv_reuse_engine_label(),
        "hf_model": resolve_hf_model_path() or None,
        "hf_model_env": os.environ.get("KVCOMM_HF_MODEL", "").strip() or None,
        "hf_device": os.environ.get("KVCOMM_HF_DEVICE", "").strip() or "cuda:0",
        "hf_load_plan": _hf_load_plan_label(),
        "engine_loaded": engine_loaded() if _engine_enabled() else False,
    }


@app.get("/diagnostics")
async def diagnostics(run_id: str | None = None, agent_index: str | None = None) -> dict[str, Any]:
    base: dict[str, Any] = {
        "upstream": UPSTREAM_BASE,
        "default_mode": DEFAULT_MODE,
        "metrics": _metrics,
        "kv_reuse_engine": _kv_reuse_engine_label(),
    }
    if _engine_enabled():
        adapter_diag = get_adapter().diagnostics()
        base["engine_diagnostics"] = adapter_diag
        if run_id is not None and agent_index is not None:
            row = get_adapter().metrics_for(run_id, agent_index)
            if row:
                base["agent_metrics"] = row
    else:
        base["note"] = "Set KVCOMM_HF_MODEL to enable HF KVCOMM engine"
    return base


def _sync_metrics_from_adapter() -> None:
    """Mirror adapter counters into top-level /diagnostics metrics."""
    adapter = get_adapter()
    _metrics["requests_total"] = adapter.requests_total
    _metrics["requests_by_mode"] = dict(adapter.requests_by_mode)
    if adapter.last_request_ms is not None:
        _metrics["last_request_ms"] = adapter.last_request_ms


async def _hf_sse_with_metrics(
    body: dict[str, Any],
    headers: dict[str, str],
    mode: str,
    *,
    include_usage: bool,
):
    """Stream HF tokens once and sync /diagnostics counters when the stream finishes."""
    try:
        async for chunk in get_adapter().generate_stream_sse(
            body,
            headers,
            mode,
            include_usage=include_usage,
        ):
            yield chunk
    finally:
        _sync_metrics_from_adapter()


async def _handle_chat_completions(request: Request, body: dict[str, Any], mode: str) -> Response:
    started = time.perf_counter()
    headers = {k: v for k, v in request.headers.items()}
    effective_mode = resolve_request_mode(body, headers, mode)

    dense_via_hf = _dense_via_hf_enabled()
    use_hf = _engine_enabled() and (
        effective_mode == "kv_reuse" or (effective_mode == "dense_prefill" and dense_via_hf)
    )
    route = "hf" if use_hf else "proxy"
    _metrics["requests_by_route"][route] = _metrics["requests_by_route"].get(route, 0) + 1
    if use_hf:
        try:
            if body.get("stream"):
                await get_adapter().check_request(body, headers, mode)
                stream_headers = {
                    "Content-Type": "text/event-stream; charset=utf-8",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-KVCOMM-Route": route,
                    "X-KVCOMM-Mode": effective_mode,
                }
                return StreamingResponse(
                    _hf_sse_with_metrics(
                        body,
                        headers,
                        mode,
                        include_usage=_wants_stream_usage(body),
                    ),
                    media_type="text/event-stream",
                    headers=stream_headers,
                )
            payload, resp_headers = await get_adapter().generate(body, headers, mode)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            _sync_metrics_from_adapter()
            resp_headers["X-KVCOMM-Proxy-Latency-Ms"] = str(elapsed_ms)
            resp_headers["X-KVCOMM-Route"] = route
            resp_headers["X-KVCOMM-Mode"] = effective_mode
            return JSONResponse(content=payload, headers=resp_headers)
        except ValueError as exc:
            if "missing kvcomm context" in str(exc):
                if not _vllm_fallback_allowed():
                    return _reject_vllm_proxy(
                        f"{exc} (HF-only sidecar; set KVCOMM_ALLOW_VLLM_FALLBACK=1 to permit vLLM upstream)",
                    )
                return await _proxy(
                    request,
                    "chat/completions",
                    body_override=body,
                    mode=mode,
                    started=started,
                )
            return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error"}})
        except PrefixOverflowError as exc:
            if not _vllm_fallback_allowed():
                return _reject_vllm_proxy(f"prefix overflow: {exc}", status_code=507)
            proxy_resp = await _proxy(
                request,
                "chat/completions",
                body_override=body,
                mode="dense_prefill",
                started=started,
            )
            proxy_resp.headers["X-KVCOMM-Route"] = "proxy"
            proxy_resp.headers["X-KVCOMM-Prefix-Fallback"] = "dense_prefill"
            proxy_resp.headers["X-KVCOMM-Prefix-Fallback-Reason"] = str(exc)[:200]
            return proxy_resp
        except Exception as exc:
            detail = _format_engine_error(exc)
            _logger.error("KVCOMM engine error: %s\n%s", detail, traceback.format_exc())
            if "out of memory" in detail.lower() and _vllm_fallback_allowed():
                proxy_resp = await _proxy(
                    request,
                    "chat/completions",
                    body_override=body,
                    mode="dense_prefill",
                    started=started,
                )
                proxy_resp.headers["X-KVCOMM-Route"] = "proxy"
                proxy_resp.headers["X-KVCOMM-Prefix-Fallback"] = "dense_prefill"
                proxy_resp.headers["X-KVCOMM-Prefix-Fallback-Reason"] = detail[:200]
                return proxy_resp
            return JSONResponse(
                status_code=500,
                content={"error": {"message": f"KVCOMM engine error: {detail}", "type": "server_error"}},
            )

    if _engine_enabled() and not _vllm_fallback_allowed():
        return _reject_vllm_proxy(
            "HF engine is configured but this request would route to vLLM upstream "
            "(set KVCOMM_DENSE_VIA_HF=0 only when intentionally using vLLM fallback)",
            status_code=503,
        )

    proxy_resp = await _proxy(
        request,
        "chat/completions",
        body_override=body,
        mode=effective_mode,
        started=started,
    )
    proxy_resp.headers["X-KVCOMM-Route"] = route
    return proxy_resp


@app.post("/v1/kvcomm/configure")
async def kvcomm_configure(request: Request) -> JSONResponse:
    """Register HF model/device at runtime without loading GPU (loads on first kv_reuse)."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body"}})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": {"message": "body must be a JSON object"}})
    try:
        result = configure_hf_engine(body)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error"}})
    return JSONResponse(content={"status": "ok", **result})


@app.post("/v1/kvcomm/release")
async def kvcomm_release() -> JSONResponse:
    """Unload HF engine weights and free GPU memory (no-op if not loaded)."""
    if not _engine_enabled():
        return JSONResponse(
            content={
                "status": "ok",
                "released": False,
                "engine_loaded": False,
                "note": "HF engine disabled (KVCOMM_HF_MODEL not set)",
            }
        )
    result = release_adapter()
    try:
        from KVCOMM.llm.gpt_chat import LLMChat

        result["gpu_pool"] = LLMChat.configured_gpu_pool()
    except Exception:
        pass
    return JSONResponse(content={"status": "ok", **result})


@app.post("/v1/kvcomm/register")
async def kvcomm_register(request: Request) -> JSONResponse:
    """Bench driver registers per-agent kvcomm context before sequential spawn."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body"}})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": {"message": "body must be a JSON object"}})
    ctx = register_pending_context(body)
    return JSONResponse(
        content={
            "status": "ok",
            "run_id": ctx.run_id,
            "agent_index": ctx.agent_index,
            "mode": ctx.mode,
            "queue_depth": pending_context_depth(),
        }
    )


async def _proxy(
    request: Request,
    path: str,
    body_override: bytes | dict[str, Any] | None = None,
    mode: str | None = None,
    started: float | None = None,
) -> Response:
    mode = mode or _resolve_mode(request)
    started = started if started is not None else time.perf_counter()
    _metrics["requests_total"] += 1
    _metrics["requests_by_mode"][mode] = _metrics["requests_by_mode"].get(mode, 0) + 1

    if body_override is None:
        body = await request.body()
    elif isinstance(body_override, dict):
        body = json.dumps(body_override).encode("utf-8")
    else:
        body = body_override

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    headers["X-KVCOMM-Mode"] = mode

    url = f"{UPSTREAM_BASE}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=600.0) as client:
        upstream = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
            params=dict(request.query_params),
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    _metrics["last_request_ms"] = elapsed_ms

    resp_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
    }
    resp_headers["X-KVCOMM-Mode"] = mode
    resp_headers["X-KVCOMM-Proxy-Latency-Ms"] = str(elapsed_ms)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def v1_proxy(request: Request, path: str) -> Response:
    if request.method == "POST" and path == "chat/completions":
        raw = await request.body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body"}})
        default_mode = _resolve_mode(request)
        return await _handle_chat_completions(request, body, default_mode)
    return await _proxy(request, path)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def root_proxy(request: Request, path: str) -> Response:
    if path in ("health", "diagnostics", "docs", "openapi.json", "redoc"):
        return JSONResponse(status_code=404, content={"detail": "not found"})
    return await _proxy(request, path)


def main() -> None:
    engine = _kv_reuse_engine_label()
    print(
        f"[kvcomm-sidecar] upstream={UPSTREAM_BASE} listen={LISTEN_HOST}:{LISTEN_PORT} "
        f"mode={DEFAULT_MODE} engine={engine}"
    )
    uvicorn.run(
        app,
        host=LISTEN_HOST,
        port=LISTEN_PORT,
        log_level="info",
        access_log=_sidecar_access_log_enabled(),
    )


if __name__ == "__main__":
    main()
