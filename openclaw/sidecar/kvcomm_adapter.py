"""OpenAI chat/completions adapter for KVCOMM HF engine (kv_reuse + dense_prefill)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sidecar.openclaw_prefix import (
    PrefixBuildResult,
    PrefixOverflowError,
    build_prefix_from_openclaw_messages,
    count_assistant_turns,
    normalize_run_specific_paths,
    use_openclaw_prefix,
)

KVCOMM_META_RE = re.compile(r"<!--KVCOMM_META:(\{.*?\})-->", re.DOTALL)
SIDECAR_VERSION = "0.2.0-kvcomm-engine"


def _anchor_pool_key(node_id: str, message_key: str) -> str:
    return f"{node_id}:{message_key}"

# Bench registers context before each sequential spawn (OpenClaw does not forward extra_body.kvcomm).
_pending_context_by_key: dict[str, "KvcommContext"] = {}
_active_registered_context: "KvcommContext | None" = None
_run_task_profiles: dict[str, str] = {}
_node_task_profiles: dict[str, str] = {}


def _context_key(run_id: str, agent_index: str) -> str:
    return f"{run_id}:{agent_index}"


def _extract_kvcomm_hints(
    body: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    hints: dict[str, Any] = {}

    extra = body.get("extra_body") or {}
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            extra = {}
    if isinstance(extra, dict):
        kvcomm = extra.get("kvcomm")
        if isinstance(kvcomm, dict):
            hints.update(kvcomm)

    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        embedded, _ = _extract_embedded_kvcomm(_message_content(msg))
        if embedded:
            hints = {**embedded, **hints}

    header_vars_raw = headers.get("x-kvcomm-vars") or headers.get("X-KVCOMM-Vars") or ""
    if header_vars_raw:
        try:
            parsed = json.loads(header_vars_raw)
            if isinstance(parsed, dict):
                hints.setdefault("vars", parsed)
        except json.JSONDecodeError:
            pass

    if headers.get("x-kvcomm-request-uid") or headers.get("X-KVCOMM-Request-Uid"):
        hints.setdefault(
            "run_id",
            headers.get("x-kvcomm-request-uid") or headers.get("X-KVCOMM-Request-Uid"),
        )
    if headers.get("x-kvcomm-mode") or headers.get("X-KVCOMM-Mode"):
        hints.setdefault("mode", headers.get("x-kvcomm-mode") or headers.get("X-KVCOMM-Mode"))

    return hints


@dataclass
class KvcommContext:
    run_id: str
    agent_index: str
    mode: str
    message_key: str
    vars: dict[str, str] = field(default_factory=dict)
    system_prompt: str = ""
    user_prompt: str = ""
    bench_user_prompt: str = ""
    task_profile: str = "copy"
    max_tokens: int = 512
    temperature: float = 0.0


def _extract_embedded_kvcomm(text: str) -> tuple[dict[str, Any], str]:
    match = KVCOMM_META_RE.search(text or "")
    if not match:
        return {}, text
    try:
        embedded = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}, text
    cleaned = (text[: match.start()] + text[match.end() :]).strip()
    if not isinstance(embedded, dict):
        return {}, text
    return embedded, cleaned


def register_pending_context(payload: dict[str, Any]) -> KvcommContext:
    """Register kvcomm context from bench driver before sequential spawn."""
    global _active_registered_context
    ctx = KvcommContext(
        run_id=str(payload.get("run_id") or uuid.uuid4()),
        agent_index=str(payload.get("agent_index") if payload.get("agent_index") is not None else "0"),
        mode=str(payload.get("mode") or "dense_prefill"),
        message_key=str(payload.get("message_key") or payload.get("task_body") or "default"),
        vars={str(k): "" if v is None else str(v) for k, v in (payload.get("vars") or {}).items()},
        system_prompt=str(payload.get("system_prompt") or ""),
        user_prompt=str(payload.get("user_prompt") or ""),
        bench_user_prompt=str(payload.get("user_prompt") or payload.get("bench_user_prompt") or ""),
        task_profile=str(payload.get("task_profile") or payload.get("taskProfile") or "copy"),
        max_tokens=int(payload.get("max_tokens") or 512),
        temperature=float(payload.get("temperature") if payload.get("temperature") is not None else 0.0),
    )
    if ctx.task_profile not in ("copy", "clawbench"):
        ctx.task_profile = "copy"
    if ctx.mode not in ("dense_prefill", "kv_reuse"):
        ctx.mode = "dense_prefill"
    ctx = _normalize_task_profile(ctx)
    _pending_context_by_key[_context_key(ctx.run_id, ctx.agent_index)] = ctx
    _active_registered_context = ctx
    return ctx


def resolve_registered_context(
    body: dict[str, Any],
    headers: dict[str, str],
) -> KvcommContext | None:
    """Lookup bench-registered context by run_id:agent_index or latest sequential register."""
    hints = _extract_kvcomm_hints(body, headers)
    run_id = str(hints.get("run_id") or "").strip()
    agent_index = hints.get("agent_index")
    if agent_index is None and hints.get("node_id") is not None:
        agent_index = hints.get("node_id")
    agent_index_str = str(agent_index).strip() if agent_index is not None else ""
    if run_id and agent_index_str:
        hit = _pending_context_by_key.get(_context_key(run_id, agent_index_str))
        if hit is not None:
            return hit
    return _active_registered_context


def consume_registered_context(
    body: dict[str, Any],
    headers: dict[str, str],
) -> KvcommContext | None:
    global _active_registered_context
    ctx = resolve_registered_context(body, headers)
    if ctx is None:
        return None
    key = _context_key(ctx.run_id, ctx.agent_index)
    consumed = _pending_context_by_key.pop(key, None)
    if _active_registered_context is not None and _context_key(
        _active_registered_context.run_id,
        _active_registered_context.agent_index,
    ) == key:
        _active_registered_context = None
    return consumed or ctx


def pending_context_depth() -> int:
    return len(_pending_context_by_key)


def resolve_request_mode(
    body: dict[str, Any],
    headers: dict[str, str],
    default_mode: str,
) -> str:
    """Resolve inference mode for HF vs vLLM routing (bench register / embedded meta)."""
    hints = _extract_kvcomm_hints(body, headers)
    registered = resolve_registered_context(body, headers)

    for source in (
        hints.get("mode"),
        registered.mode if registered else None,
        headers.get("x-kvcomm-mode") or headers.get("X-KVCOMM-Mode"),
        default_mode,
    ):
        candidate = str(source or "").strip()
        if candidate in ("dense_prefill", "kv_reuse"):
            return candidate
    return "dense_prefill"


def _bench_no_think_enabled() -> bool:
    raw = os.environ.get("KVCOMM_BENCH_NO_THINK", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _append_no_think_to_body(body: dict[str, Any]) -> None:
    if not _bench_no_think_enabled():
        return
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and "/no_think" not in content:
            msg["content"] = f"{content.rstrip()}\n/no_think"
        break


def _load_clawbench_role() -> str:
    """ClawBench chain role (matches experiments/bench clawbench_chain.role.txt)."""
    explicit = os.environ.get("KVCOMM_CLAWBENCH_ROLE", "").strip()
    if explicit:
        return explicit
    role_path = os.environ.get(
        "KVCOMM_CLAWBENCH_ROLE_PATH",
        str(Path(__file__).resolve().parents[2] / "experiments/bench/prompts/clawbench_chain.role.txt"),
    )
    path = Path(role_path)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return (
        "You are one agent in a fixed multi-agent chain. Follow your role instructions precisely.\n"
        "Respond in plain language unless your role requires code or structured output.\n"
        "Do not spawn subagents or delegate to other agents."
    )


def _load_copy_role() -> str:
    """COPY bench constraint text (matches experiments/bench copy_machine.role.txt)."""
    explicit = os.environ.get("KVCOMM_COPY_ROLE", "").strip()
    if explicit:
        return explicit
    role_path = os.environ.get(
        "KVCOMM_COPY_ROLE_PATH",
        str(Path(__file__).resolve().parents[2] / "experiments/bench/prompts/copy_machine.role.txt"),
    )
    path = Path(role_path)
    if path.is_file():
        out_length = os.environ.get("COPY_OUT_LENGTH", "128")
        prefix_repeats = int(os.environ.get("COPY_PREFIX_REPEATS", "64") or "64")
        body = path.read_text(encoding="utf-8").replace("{{out_length}}", out_length).strip()
        prefix = " Ω" * max(0, prefix_repeats)
        return f"{prefix}\n{body}".strip()
    return "Do NOT use any tools. Output ONLY Ω and Δ characters."


def _agent_role_label(ctx: KvcommContext, agent_index: int) -> str:
    role_key = f"agent_{agent_index}_role"
    label = (ctx.vars.get(role_key) or "").strip()
    if label:
        return label
    roles_raw = (ctx.vars.get("agent_roles") or "").strip()
    if roles_raw:
        try:
            roles = json.loads(roles_raw)
            if isinstance(roles, list) and 0 <= agent_index < len(roles):
                return str(roles[agent_index])
        except json.JSONDecodeError:
            parts = [part.strip() for part in roles_raw.split(",") if part.strip()]
            if 0 <= agent_index < len(parts):
                return parts[agent_index]
    return f"Agent {agent_index}"


def _build_kvcomm_user_prompt(ctx: KvcommContext, *, copy_layout: bool = True) -> str:
    """Build prefix template with KVCOMM placeholders."""
    user_prompt = "The task is: {user_question}\n"
    try:
        agent_index = int(ctx.agent_index)
    except (TypeError, ValueError):
        agent_index = 0

    if agent_index > 0:
        spatial_parts: list[str] = []
        for i in range(agent_index):
            key = f"agent_{i}_current"
            role_label = "Copy Machine" if copy_layout else _agent_role_label(ctx, i)
            spatial_parts.append(
                f"Agent {i}, role is {role_label}, output is:\n\n {{{key}}}\n\n"
            )
        if spatial_parts:
            user_prompt += (
                "At the same time, the outputs of other agents are as follows:\n\n"
                + "".join(spatial_parts)
                + "\n"
            )
    return user_prompt


def _infer_task_profile(ctx: KvcommContext) -> str:
    if ctx.task_profile == "clawbench":
        return "clawbench"
    if ctx.run_id and ctx.run_id in _run_task_profiles:
        cached = _run_task_profiles[ctx.run_id]
        if cached == "clawbench":
            return "clawbench"
    combined = f"{ctx.system_prompt}\n{ctx.user_prompt}"
    if "Output ONLY Ω" in combined or "Copy Machine" in combined:
        return "copy"
    if ctx.vars.get("task_body") or "Your job (Agent" in combined:
        return "clawbench"
    return ctx.task_profile if ctx.task_profile in ("copy", "clawbench") else "copy"


def _normalize_task_profile(ctx: KvcommContext) -> KvcommContext:
    ctx.task_profile = _infer_task_profile(ctx)
    if ctx.run_id:
        _run_task_profiles[ctx.run_id] = ctx.task_profile
    return ctx


def _reset_prefix_node(node_id: str) -> None:
    from KVCOMM.llm.gpt_chat import LLMChat

    LLMChat._initialization[node_id] = False
    bucket = LLMChat._shared_kv_cache_memory.get(node_id)
    if isinstance(bucket, dict):
        bucket.pop("prefix", None)
        bucket.pop("placeholder_info", None)
        bucket.pop("token_ids", None)
        bucket.pop("turn", None)
        bucket.pop("turn_count", None)
        bucket.pop("user_template", None)
        bucket.pop("system_prompt", None)


def _clear_copy_input_cache(message: str) -> None:
    from KVCOMM.llm.gpt_chat import LLMChat

    shared = LLMChat._shared_kv_cache_memory
    if not isinstance(shared, dict):
        return
    for key in ("input", "input_ids", "input_drop_num"):
        bucket = shared.get(key)
        if isinstance(bucket, dict):
            bucket.pop(message, None)


def _resolve_prefix_prompts(ctx: KvcommContext) -> tuple[str, str]:
    """Choose system/user prefix templates by bench workload (legacy fallback)."""
    if ctx.task_profile == "clawbench":
        system_prompt = _load_clawbench_role()
        user_prompt = normalize_run_specific_paths(
            (ctx.bench_user_prompt or _build_kvcomm_user_prompt(ctx, copy_layout=False)).strip()
        )
        return system_prompt, user_prompt
    return _load_copy_role(), _build_kvcomm_user_prompt(ctx, copy_layout=True)


def _build_openclaw_prefix(
    ctx: KvcommContext,
    body: dict[str, Any],
) -> PrefixBuildResult:
    """Parse OpenClaw messages into prefix template (A+E) with bench fallback."""
    if not use_openclaw_prefix(ctx.task_profile):
        system_prompt, user_prompt = _resolve_prefix_prompts(ctx)
        return PrefixBuildResult(
            system_prompt=system_prompt,
            user_template=user_prompt,
            static_text=user_prompt,
            placeholders=[],
            turn_count=0,
            use_openclaw=False,
            fallback_reason="profile_disabled",
        )

    messages = [
        msg for msg in (body.get("messages") or []) if isinstance(msg, dict)
    ]
    try:
        return build_prefix_from_openclaw_messages(
            messages,
            bench_user_prompt=normalize_run_specific_paths(ctx.bench_user_prompt),
            clawbench_role=_load_clawbench_role(),
            task_profile=ctx.task_profile,
        )
    except PrefixOverflowError:
        raise
    except Exception as exc:
        system_prompt, user_prompt = _resolve_prefix_prompts(ctx)
        return PrefixBuildResult(
            system_prompt=system_prompt,
            user_template=user_prompt,
            static_text=user_prompt,
            placeholders=[],
            turn_count=0,
            use_openclaw=False,
            fallback_reason=f"parse_error:{exc}",
        )


def _turn_segment_template(turn_index: int) -> str:
    return f"\n\n{{turn_{turn_index}_assistant}}\n\n{{turn_{turn_index}_tool}}\n"


_UPSTREAM_AGENT_PLACEHOLDER_RE = re.compile(r"\{agent_(\d+)_current\}")


def _required_upstream_kv_placeholders(user_template: str, agent_index: int) -> list[str]:
    """Return upstream chain placeholders like '{agent_0_current}' for this agent."""
    required: list[str] = []
    for match in _UPSTREAM_AGENT_PLACEHOLDER_RE.finditer(user_template or ""):
        try:
            upstream = int(match.group(1))
        except ValueError:
            continue
        if upstream < agent_index:
            required.append(match.group(0))
    return required


def _prefix_missing_upstream_kv_placeholders(
    stored_user_template: str,
    expected_user_template: str,
    agent_index: int,
) -> bool:
    """Detect warmup strict-render prefix that inlined upstream text (no KV placeholders)."""
    required = _required_upstream_kv_placeholders(expected_user_template, agent_index)
    if not required:
        return False
    stored = stored_user_template or ""
    return any(token not in stored for token in required)


def _trim_completed_agent_prefixes(ctx: KvcommContext) -> None:
    """Drop prefix KV for upstream agents once the chain advances (clawbench only)."""
    if ctx.task_profile != "clawbench":
        return
    try:
        current = int(ctx.agent_index)
    except (TypeError, ValueError):
        return
    for agent_id in range(current):
        _reset_prefix_node(str(agent_id))


def _engine_enabled() -> bool:
    if os.environ.get("KVCOMM_ENGINE", "").strip() == "stub_forward":
        return False
    if os.environ.get("KVCOMM_STUB", "").strip() in ("1", "true", "yes"):
        return False
    return bool(resolve_hf_model_path())


def resolve_hf_model_path() -> str:
    """Resolve HF model id to a local directory when possible (no Hub download).

    Priority: KVCOMM_HF_MODEL_PATH > existing KVCOMM_HF_MODEL dir >
    /models/{basename} for Hub-style ids (e.g. Qwen/Qwen3-32B -> /models/Qwen3-32B).
    """
    local_override = os.environ.get("KVCOMM_HF_MODEL_PATH", "").strip()
    if local_override:
        return os.path.expanduser(local_override)

    explicit = os.environ.get("KVCOMM_HF_MODEL", "").strip()
    if not explicit:
        return ""

    expanded = os.path.expanduser(explicit)
    if os.path.isdir(expanded):
        return expanded

    basename = explicit.split("/")[-1]
    local_candidate = os.path.join("/models", basename)
    if os.path.isdir(local_candidate):
        return local_candidate

    return explicit


def parse_kvcomm_context(body: dict[str, Any], headers: dict[str, str], default_mode: str) -> KvcommContext | None:
    """Extract bench kvcomm metadata from extra_body, headers, or task prefix."""
    extra = body.get("extra_body") or {}
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            extra = {}
    kvcomm = extra.get("kvcomm") if isinstance(extra, dict) else None
    if not isinstance(kvcomm, dict):
        kvcomm = {}

    header_vars_raw = headers.get("x-kvcomm-vars") or headers.get("X-KVCOMM-Vars") or ""
    if header_vars_raw:
        try:
            kvcomm.setdefault("vars", json.loads(header_vars_raw))
        except json.JSONDecodeError:
            pass

    messages = body.get("messages") or []
    system_parts: list[str] = []
    user_parts: list[str] = []
    embedded_from_messages: dict[str, Any] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        content = _message_content(msg)
        embedded, cleaned = _extract_embedded_kvcomm(content)
        if embedded:
            embedded_from_messages = {**embedded_from_messages, **embedded}
            content = cleaned
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)

    if embedded_from_messages:
        kvcomm = {**embedded_from_messages, **kvcomm}

    combined_user = "\n\n".join(part for part in user_parts if part).strip()

    run_id = (
        str(kvcomm.get("run_id") or headers.get("x-kvcomm-request-uid") or headers.get("X-KVCOMM-Request-Uid") or "")
        .strip()
    )
    agent_index = str(kvcomm.get("agent_index") if kvcomm.get("agent_index") is not None else "").strip()
    if not agent_index:
        node = kvcomm.get("node_id")
        if node is not None:
            agent_index = str(node).strip()

    if not run_id and not agent_index and not kvcomm.get("message_key"):
        return None

    mode = (
        str(kvcomm.get("mode") or headers.get("x-kvcomm-mode") or headers.get("X-KVCOMM-Mode") or default_mode)
        .strip()
    )
    if mode not in ("dense_prefill", "kv_reuse"):
        mode = default_mode if default_mode in ("dense_prefill", "kv_reuse") else "dense_prefill"

    vars_map = kvcomm.get("vars") or {}
    if not isinstance(vars_map, dict):
        vars_map = {}
    vars_map = {str(k): "" if v is None else str(v) for k, v in vars_map.items()}

    message_key = str(kvcomm.get("message_key") or kvcomm.get("user_question") or vars_map.get("user_question") or "").strip()
    if not message_key:
        message_key = combined_user[:256] if combined_user else "default"

    user_prompt = combined_user or "\n".join(user_parts)
    meta_in_user = "<!--KVCOMM_META:" in user_prompt
    if meta_in_user:
        _, user_prompt = _extract_embedded_kvcomm(user_prompt)

    task_profile = str(kvcomm.get("task_profile") or kvcomm.get("taskProfile") or "copy")
    if task_profile not in ("copy", "clawbench"):
        task_profile = "copy"

    return KvcommContext(
        run_id=run_id or str(uuid.uuid4()),
        agent_index=agent_index or "0",
        mode=mode,
        message_key=message_key,
        vars=vars_map,
        system_prompt="\n\n".join(system_parts).strip(),
        user_prompt=user_prompt,
        bench_user_prompt=user_prompt,
        task_profile=task_profile,
        max_tokens=int(body.get("max_tokens") or 512),
        temperature=float(body.get("temperature") if body.get("temperature") is not None else 0.0),
    )


def _message_content(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts)
    return ""


def _metrics_from_result(
    result: Any,
    *,
    effective_mode: str,
    ctx: "KvcommContext",
    prefix_build: PrefixBuildResult,
    turn_index: int,
    elapsed_ms: float,
    input_anchor_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = result.metadata or {}
    input_anchor_meta = input_anchor_meta or {}
    kvcomm_latency_ms = metadata.get("kvcomm_latency")
    if kvcomm_latency_ms is not None:
        kvcomm_latency_ms = round(float(kvcomm_latency_ms) * 1000, 2)
    generation_ttft_ms = metadata.get("generation_ttft")
    if generation_ttft_ms is not None:
        generation_ttft_ms = round(float(generation_ttft_ms) * 1000, 2)
    preprocess_latency_ms = metadata.get("preprocess_latency")
    if preprocess_latency_ms is not None:
        preprocess_latency_ms = round(float(preprocess_latency_ms) * 1000, 2)
    reuse_rate = 1.0 if result.mode == "kv_reuse" else 0.0
    anchor_prediction = metadata.get("anchor_prediction")
    if anchor_prediction is not None:
        anchor_prediction = bool(anchor_prediction)
    anchor_pooled_tokens = metadata.get("anchor_pooled_tokens")
    if anchor_pooled_tokens is not None:
        anchor_pooled_tokens = int(anchor_pooled_tokens)
    input_anchor_pooled_tokens = input_anchor_meta.get("input_anchor_pooled_tokens")
    if input_anchor_pooled_tokens is not None:
        input_anchor_pooled_tokens = int(input_anchor_pooled_tokens)
    return {
        "mode": result.mode,
        "effective_mode": effective_mode,
        "input_routing_mode": input_anchor_meta.get("input_routing_mode"),
        "ttft_ms": round(float(result.ttft) * 1000, 2),
        "generation_ttft_ms": generation_ttft_ms,
        "preprocess_latency_ms": preprocess_latency_ms,
        "kvcomm_latency_ms": kvcomm_latency_ms,
        "reuse_rate": reuse_rate,
        "anchor_prediction": anchor_prediction,
        "anchor_pooled_tokens": anchor_pooled_tokens,
        "input_anchor_pooled_tokens": input_anchor_pooled_tokens,
        "reuse_kv_text": metadata.get("reuse_kv_text"),
        "reuse_kv_segments": metadata.get("reuse_kv_segments"),
        "run_id": ctx.run_id,
        "agent_index": ctx.agent_index,
        "message_key_hash": ctx.message_key[:32],
        "turn_index": turn_index,
        "turn_count": prefix_build.turn_count,
        "prefix_estimated_tokens": prefix_build.estimated_tokens,
        "use_openclaw_prefix": prefix_build.use_openclaw,
        "prefix_fallback_reason": prefix_build.fallback_reason,
        "bench_no_think": _bench_no_think_enabled(),
        "elapsed_ms": elapsed_ms,
    }


def _openai_completion(content: str, model: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-kvcomm-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(content.split()),
            "total_tokens": len(content.split()),
        },
        "kvcomm": metrics,
    }


class KvcommEngineAdapter:
    """Lazy HF KVCOMM engine with per-request metrics for bench bridge."""

    def __init__(self) -> None:
        self.model_name = resolve_hf_model_path() or "Qwen/Qwen3-32B"
        self._llm = None
        self._sessions: dict[str, dict[str, Any]] = {}
        self._request_metrics: dict[str, dict[str, Any]] = {}
        self._anchor_pool: dict[str, dict[str, Any]] = {}
        self.requests_total = 0
        self.requests_by_mode: dict[str, int] = {"dense_prefill": 0, "kv_reuse": 0}
        self.last_request_ms: float | None = None

    @property
    def engine_id(self) -> str:
        if not _engine_enabled():
            return "stub_forward"
        return f"hf_kvcomm/{self.model_name}"

    def _assert_hf_runtime(self) -> None:
        """Fail fast with a clear message when transformers cannot load this checkpoint."""
        import transformers
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        local_kwargs: dict[str, Any] = {}
        if os.path.isdir(os.path.expanduser(self.model_name)):
            local_kwargs["local_files_only"] = True

        try:
            cfg = AutoConfig.from_pretrained(
                self.model_name, trust_remote_code=True, **local_kwargs
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load HF config from {self.model_name}: {exc}"
            ) from exc

        if cfg.model_type not in CONFIG_MAPPING:
            raise RuntimeError(
                f"transformers {transformers.__version__} does not support "
                f"model_type={cfg.model_type!r} (checkpoint requires >=4.51.0). "
                "Fix: pip install 'transformers>=4.51.0,<4.52' in the sidecar Python env."
            ) from None

    def _get_llm(self):
        if self._llm is None:
            from KVCOMM.llm.config import KVCommConfig
            from KVCOMM.llm.llm_registry import LLMRegistry

            self._assert_hf_runtime()
            config = KVCommConfig.from_env().validate()
            self._llm = LLMRegistry.get(self.model_name, llm_config=config)
        return self._llm

    def _restore_anchors(self, llm, request_uid: str, node_id: str, message_key: str) -> None:
        snapshot = self._anchor_pool.get(_anchor_pool_key(node_id, message_key))
        if not snapshot:
            return
        state = llm.kv_engine.resolve_request_state(request_uid)
        for key, value in snapshot.items():
            if key == "anchors":
                for ph_id, bucket in value.items():
                    state.anchors.setdefault(ph_id, {}).update(bucket)
            elif key == "anchor_dict":
                for ph_id, bucket in value.items():
                    state.anchor_dict.setdefault(ph_id, {}).update(bucket)
            elif key == "anchor_len_dict":
                for ph_id, bucket in value.items():
                    state.anchor_len_dict.setdefault(ph_id, {}).update(bucket)
            elif key == "anchor_info_dict":
                for ph_id, bucket in value.items():
                    state.anchor_info_dict.setdefault(ph_id, {}).update(bucket)
            elif key == "global_anchor_info":
                for ph_id, bucket in value.items():
                    state.global_anchor_info.setdefault(ph_id, {}).update(bucket)

    def _snapshot_anchors(self, llm, request_uid: str, node_id: str, message_key: str) -> None:
        state = llm.kv_engine.resolve_request_state(request_uid)
        self._anchor_pool[_anchor_pool_key(node_id, message_key)] = {
            "anchors": {k: dict(v) for k, v in state.anchors.items()},
            "anchor_dict": {k: dict(v) for k, v in state.anchor_dict.items()},
            "anchor_len_dict": {k: dict(v) for k, v in state.anchor_len_dict.items()},
            "anchor_info_dict": {k: dict(v) for k, v in state.anchor_info_dict.items()},
            "global_anchor_info": {k: dict(v) for k, v in state.global_anchor_info.items()},
        }

    def _enrich_context_from_request(self, ctx: KvcommContext, body: dict[str, Any]) -> KvcommContext:
        messages = body.get("messages") or []
        system_parts: list[str] = []
        user_parts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "")
            content = _message_content(msg)
            _, content = _extract_embedded_kvcomm(content)
            if role == "system" and content:
                system_parts.append(content)
            elif role == "user" and content:
                user_parts.append(content)
        if system_parts:
            ctx.system_prompt = "\n\n".join(system_parts).strip()
        if user_parts:
            ctx.user_prompt = user_parts[-1]
        if body.get("max_tokens") is not None:
            ctx.max_tokens = int(body.get("max_tokens") or ctx.max_tokens)
        if body.get("temperature") is not None:
            ctx.temperature = float(body.get("temperature"))
        return ctx

    async def _ensure_prefix(
        self,
        llm,
        ctx: KvcommContext,
        body: dict[str, Any],
        prefix_build: PrefixBuildResult,
    ) -> None:
        node_id = ctx.agent_index
        llm.set_id(node_id, f"agent_{node_id}")
        prev_profile = _node_task_profiles.get(node_id)
        if prev_profile and prev_profile != ctx.task_profile and llm.has_prefix_initialized(node_id):
            _reset_prefix_node(node_id)
        _node_task_profiles[node_id] = ctx.task_profile

        stored_turns = llm.get_prefix_turn_count(node_id)
        turn_index = max(0, count_assistant_turns(body.get("messages") or []) - 1)
        stored_user = ""
        if llm.has_prefix_initialized(node_id):
            from KVCOMM.llm.gpt_chat import LLMChat

            bucket = LLMChat._shared_kv_cache_memory.get(node_id, {})
            stored_user = str(bucket.get("user_template") or "")

        try:
            agent_idx = int(node_id)
        except (TypeError, ValueError):
            agent_idx = 0
        needs_kvreuse_placeholder_rebuild = (
            ctx.task_profile == "clawbench"
            and ctx.mode == "kv_reuse"
            and llm.has_prefix_initialized(node_id)
            and _prefix_missing_upstream_kv_placeholders(
                stored_user,
                prefix_build.user_template,
                agent_idx,
            )
        )
        needs_rebuild = (
            not llm.has_prefix_initialized(node_id)
            or stored_turns != prefix_build.turn_count
            or needs_kvreuse_placeholder_rebuild
        )

        if needs_rebuild:
            if (
                llm.has_prefix_initialized(node_id)
                and prefix_build.turn_count > stored_turns
                and prefix_build.turn_count == stored_turns + 1
                and use_openclaw_prefix(ctx.task_profile)
                and not needs_kvreuse_placeholder_rebuild
            ):
                await llm.append_prefix_segment(
                    node_id,
                    _turn_segment_template(prefix_build.turn_count),
                    system_prompt=prefix_build.system_prompt,
                )
            else:
                if llm.has_prefix_initialized(node_id):
                    _reset_prefix_node(node_id)
                await llm.prepare_prefix_kv_segments(
                    node_id,
                    prefix_build.system_prompt,
                    prefix_build.user_template,
                )
            llm.set_prefix_turn_count(node_id, prefix_build.turn_count)

        await llm.materialize_turn_placeholders(
            node_id,
            ctx.message_key,
            prefix_build.turn_content,
        )

    async def _maybe_update_input_anchor(self, llm, ctx: KvcommContext) -> str:
        """Ensure user_question input KV exists in shared_memory['input']; pick mode."""
        if not llm.has_prefix_initialized(ctx.agent_index):
            return "dense_prefill"

        if ctx.task_profile == "clawbench":
            # ClawBench embeds task text in the per-agent prefix user prompt.
            # Keep shared input KV across warmup/measure agents so kv_reuse can hit it.
            user_content = ctx.message_key
            prefix_text = "The task is: "
            preferred = llm.update_input_anchor(
                request_uid=ctx.run_id,
                agent_id=ctx.agent_index,
                message=ctx.message_key,
                user_content=f"{prefix_text}{user_content}",
                prefix_text=prefix_text,
                test_time=False,
            )
            if ctx.mode == "dense_prefill":
                return "dense_prefill"
            return preferred

        user_content = ctx.vars.get("user_question") or ctx.message_key
        prefix_text = "The task is: "
        # Required for both dense_prefill and kv_reuse: agen_kvcomm reads shared_memory["input"].
        preferred = llm.update_input_anchor(
            request_uid=ctx.run_id,
            agent_id=ctx.agent_index,
            message=ctx.message_key,
            user_content=f"{prefix_text}{user_content}",
            prefix_text=prefix_text,
            test_time=False,
        )
        if ctx.mode == "dense_prefill":
            return "dense_prefill"
        return preferred if ctx.mode == "kv_reuse" else "dense_prefill"

    async def _prepare_generation(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
        default_mode: str,
    ) -> tuple[Any, "KvcommContext", PrefixBuildResult, int, Any]:
        _append_no_think_to_body(body)
        ctx = consume_registered_context(body, headers)
        if ctx is None:
            ctx = parse_kvcomm_context(body, headers, default_mode)
        if ctx is None:
            raise ValueError("missing kvcomm context (run_id/agent_index/message_key)")

        ctx = self._enrich_context_from_request(ctx, body)
        ctx = _normalize_task_profile(ctx)
        _trim_completed_agent_prefixes(ctx)

        prefix_build = _build_openclaw_prefix(ctx, body)
        turn_index = max(0, count_assistant_turns(body.get("messages") or []) - 1)

        llm = self._get_llm()
        try:
            await self._ensure_prefix(llm, ctx, body, prefix_build)
        except PrefixOverflowError:
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise PrefixOverflowError(str(exc)) from exc
            raise
        llm.set_id(ctx.agent_index, f"agent_{ctx.agent_index}")

        if ctx.mode == "kv_reuse":
            self._restore_anchors(llm, ctx.run_id, ctx.agent_index, ctx.message_key)

        effective_mode = await self._maybe_update_input_anchor(llm, ctx)
        return llm, ctx, prefix_build, turn_index, effective_mode

    async def _finalize_generation(
        self,
        llm: Any,
        ctx: "KvcommContext",
        effective_mode: str,
        result: Any,
        prefix_build: PrefixBuildResult,
        turn_index: int,
        started: float,
        *,
        model: str,
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        if ctx.task_profile == "clawbench" or effective_mode == "dense_prefill":
            self._snapshot_anchors(llm, ctx.run_id, ctx.agent_index, ctx.message_key)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        self.last_request_ms = elapsed_ms
        self.requests_by_mode[effective_mode] = self.requests_by_mode.get(effective_mode, 0) + 1
        self.requests_total += 1

        try:
            state = llm.get_request_state(ctx.run_id)
            input_anchor_meta = getattr(state, "input_anchor_metrics", None) or {}
        except Exception:
            input_anchor_meta = {}
        metrics = _metrics_from_result(
            result,
            effective_mode=effective_mode,
            ctx=ctx,
            prefix_build=prefix_build,
            turn_index=turn_index,
            elapsed_ms=elapsed_ms,
            input_anchor_meta=input_anchor_meta,
        )
        metric_key = f"{ctx.run_id}:{ctx.agent_index}"
        self._request_metrics[metric_key] = metrics
        self._sessions.setdefault(ctx.run_id, {})[ctx.agent_index] = metrics

        kvcomm_latency_ms = metrics.get("kvcomm_latency_ms")
        resp_headers = {
            "X-KVCOMM-Mode": result.mode,
            "X-KVCOMM-Latency-Ms": str(kvcomm_latency_ms if kvcomm_latency_ms is not None else elapsed_ms),
            "X-KVCOMM-Reuse-Rate": str(metrics.get("reuse_rate")),
            "X-KVCOMM-TTFT-Ms": str(metrics.get("ttft_ms")),
            "X-KVCOMM-Generation-TTFT-Ms": str(metrics.get("generation_ttft_ms") or ""),
        }
        return _openai_completion(result.text, model, metrics), resp_headers, metrics

    async def generate(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
        default_mode: str,
        *,
        on_token: Any = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        started = time.perf_counter()
        model = str(body.get("model") or f"kvcomm/{self.model_name.split('/')[-1]}")
        llm, ctx, prefix_build, turn_index, effective_mode = await self._prepare_generation(
            body,
            headers,
            default_mode,
        )
        result = await llm.generate_for_agent(
            request_uid=ctx.run_id,
            message=ctx.message_key,
            preferred_mode=effective_mode,
            max_tokens=ctx.max_tokens,
            temperature=ctx.temperature,
            agent_id=ctx.agent_index,
            agent_name=f"Agent{ctx.agent_index}",
            agent_role=f"agent_{ctx.agent_index}",
            on_token=on_token,
        )
        payload, resp_headers, _metrics = await self._finalize_generation(
            llm,
            ctx,
            effective_mode,
            result,
            prefix_build,
            turn_index,
            started,
            model=model,
        )
        return payload, resp_headers

    async def generate_stream_sse(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
        default_mode: str,
        *,
        include_usage: bool = False,
    ):
        """Yield OpenAI SSE bytes with incremental HF token deltas."""
        started = time.perf_counter()
        model = str(body.get("model") or f"kvcomm/{self.model_name.split('/')[-1]}")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        chunk_id = f"chatcmpl-kvcomm-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def chunk_obj(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        async def run_generate() -> None:
            def on_token(token: str) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, ("token", token))

            try:
                payload, resp_headers = await self.generate(
                    body,
                    headers,
                    default_mode,
                    on_token=on_token,
                )
                await queue.put(("done", (payload, resp_headers)))
            except Exception as exc:
                await queue.put(("error", exc))

        task = asyncio.create_task(run_generate())
        content_streamed = False
        try:
            yield f"data: {json.dumps(chunk_obj({'role': 'assistant'}), ensure_ascii=False)}\n\n".encode("utf-8")
            while True:
                kind, value = await queue.get()
                if kind == "token":
                    token = str(value)
                    if token:
                        content_streamed = True
                        yield f"data: {json.dumps(chunk_obj({'content': token}), ensure_ascii=False)}\n\n".encode("utf-8")
                elif kind == "error":
                    raise value
                elif kind == "done":
                    payload, resp_headers = value
                    if not content_streamed:
                        message = (payload.get("choices") or [{}])[0].get("message") or {}
                        content = message.get("content") or ""
                        if content:
                            yield f"data: {json.dumps(chunk_obj({'content': content}), ensure_ascii=False)}\n\n".encode(
                                "utf-8",
                            )
                            content_streamed = True
                    finish_reason = (payload.get("choices") or [{}])[0].get("finish_reason") or "stop"
                    yield f"data: {json.dumps(chunk_obj({}, finish_reason), ensure_ascii=False)}\n\n".encode("utf-8")
                    if include_usage and isinstance(payload.get("usage"), dict):
                        usage_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": payload["usage"],
                        }
                        yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"
                    break
        finally:
            if not task.done():
                task.cancel()

    def diagnostics(self) -> dict[str, Any]:
        reuse_calls = sum(1 for m in self._request_metrics.values() if m.get("mode") == "kv_reuse")
        total = len(self._request_metrics)
        return {
            "engine": self.engine_id,
            "sidecar_version": SIDECAR_VERSION,
            "model": self.model_name,
            "requests_total": self.requests_total,
            "requests_by_mode": self.requests_by_mode,
            "last_request_ms": self.last_request_ms,
            "reuse_rate": round(reuse_calls / total, 4) if total else 0.0,
            "anchor_pool_tasks": len(self._anchor_pool),
            "recent_metrics": list(self._request_metrics.values())[-10:],
        }

    def metrics_for(self, run_id: str, agent_index: str) -> dict[str, Any] | None:
        return self._request_metrics.get(f"{run_id}:{agent_index}")

    def release(self) -> dict[str, Any]:
        """Unload HF weights and clear per-run sidecar state."""
        from KVCOMM.llm.gpt_chat import LLMChat

        released = LLMChat.release_shared_resources()
        self._llm = None
        self._sessions.clear()
        self._request_metrics.clear()
        self._anchor_pool.clear()
        global _active_registered_context, _run_task_profiles, _node_task_profiles
        _pending_context_by_key.clear()
        _active_registered_context = None
        _run_task_profiles.clear()
        _node_task_profiles.clear()
        return {
            "released": released,
            "engine_loaded": False,
            "engine": self.engine_id,
        }


_adapter: KvcommEngineAdapter | None = None


def get_adapter() -> KvcommEngineAdapter:
    global _adapter
    if _adapter is None:
        _adapter = KvcommEngineAdapter()
    return _adapter


def release_adapter() -> dict[str, Any]:
    global _adapter
    if _adapter is None:
        return {
            "released": False,
            "engine_loaded": False,
            "engine": _kv_reuse_engine_label(),
        }
    result = _adapter.release()
    return result


def engine_loaded() -> bool:
    return _adapter is not None and _adapter._llm is not None


def _kv_reuse_engine_label() -> str:
    if not _engine_enabled():
        return "stub_forward"
    if _adapter is not None:
        return _adapter.engine_id
    model = resolve_hf_model_path()
    return f"hf_kvcomm/{model}" if model else "stub_forward"


def _maybe_apply_cuda_visible_devices(hf_device: str) -> tuple[str | None, str]:
    """Pin sidecar to selected physical GPUs before first torch import.

    When CUDA_VISIBLE_DEVICES is applied, remap KVCOMM_HF_DEVICE to logical 0..n-1.
    """
    import sys

    device = hf_device.strip()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return visible, os.environ.get("KVCOMM_HF_DEVICE", device).strip() or device
    if not device:
        return None, device
    if "torch" in sys.modules:
        return None, device
    parts = [part.strip() for part in device.split(",") if part.strip().isdigit()]
    if not parts:
        return None, device
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(parts)
    logical = ",".join(str(index) for index in range(len(parts)))
    os.environ["KVCOMM_HF_DEVICE"] = logical
    return ",".join(parts), logical


def configure_hf_engine(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply HF engine settings at runtime (lightweight sidecar; GPU loads on first kv_reuse)."""
    global _adapter

    hf_model = str(payload.get("hf_model") or payload.get("KVCOMM_HF_MODEL") or "").strip()
    hf_model_path = str(payload.get("hf_model_path") or payload.get("KVCOMM_HF_MODEL_PATH") or "").strip()
    hf_device = str(payload.get("hf_device") or payload.get("KVCOMM_HF_DEVICE") or "").strip()
    dense_via_hf = payload.get("dense_via_hf", payload.get("KVCOMM_DENSE_VIA_HF"))

    if not hf_model and not hf_model_path:
        raise ValueError("hf_model or hf_model_path is required")

    prev_path = resolve_hf_model_path()

    if hf_model_path:
        os.environ["KVCOMM_HF_MODEL_PATH"] = hf_model_path
    if hf_model:
        os.environ["KVCOMM_HF_MODEL"] = hf_model
    if hf_device:
        cuda_visible, effective_device = _maybe_apply_cuda_visible_devices(hf_device)
        if effective_device:
            os.environ["KVCOMM_HF_DEVICE"] = effective_device
    else:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES") or None
        effective_device = os.environ.get("KVCOMM_HF_DEVICE", "").strip()
    if dense_via_hf is not None:
        enabled = dense_via_hf in (True, 1, "1", "true", "yes", "on")
        os.environ["KVCOMM_DENSE_VIA_HF"] = "1" if enabled else "0"

    new_path = resolve_hf_model_path()
    if _adapter is not None:
        loaded = _adapter._llm is not None
        model_changed = bool(prev_path and new_path and prev_path != new_path)
        name_stale = bool(new_path and _adapter.model_name != new_path)
        if loaded and model_changed:
            _adapter.release()
            _adapter = None
        elif name_stale:
            _adapter = None

    return {
        "hf_model": new_path or None,
        "hf_device": os.environ.get("KVCOMM_HF_DEVICE", "").strip() or None,
        "hf_device_physical": cuda_visible or hf_device or None,
        "cuda_visible_devices": cuda_visible,
        "engine_enabled": _engine_enabled(),
        "engine_loaded": engine_loaded(),
        "kv_reuse_engine": _kv_reuse_engine_label(),
    }
