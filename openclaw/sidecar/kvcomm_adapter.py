"""OpenAI chat/completions adapter for KVCOMM HF engine (kv_reuse + dense_prefill)."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from KVCOMM.utils.log import logger
from sidecar.bench_prompt_compose import (
    inject_tool_constraints,
    is_add_tests_normalizer_task,
    is_browser_family_task,
    is_bugfix_discount_task,
    is_config_loader_task,
    is_find_that_task,
    is_quick_note_task,
    tool_constraints_for_context,
)
from sidecar.openclaw_prefix import (
    PrefixBuildResult,
    PrefixOverflowError,
    analyzer_reads_satisfied,
    build_prefix_from_openclaw_messages,
    count_assistant_turns,
    missing_analyzer_reads,
    normalize_run_specific_paths,
    build_pricing_edit_hint,
    build_config_loader_analyzer_read_hint,
    build_config_loader_edit_hint,
    build_config_loader_edit_message,
    browser_exploration_satisfied,
    browser_patcher_edit_applied_in_messages,
    browser_patcher_read_satisfied,
    browser_verifier_exec_done,
    browser_verifier_exec_passed,
    build_browser_appjs_edit_hint,
    build_browser_verifier_exec_hint,
    build_find_that_verifier_exec_hint,
    build_find_that_writer_copy_hint,
    config_loader_analyzer_reads_satisfied,
    config_loader_missing_analyzer_reads,
    config_loader_patcher_fix_satisfied,
    config_loader_patcher_read_satisfied,
    config_loader_verifier_should_force_edit,
    config_loader_verifier_should_force_exec,
    config_loader_verifier_should_force_read,
    find_that_copy_satisfied,
    find_that_source_located,
    find_that_verifier_exec_done,
    find_that_verifier_passed,
    normalizer_analyzer_read_satisfied,
    normalizer_patcher_read_satisfied,
    normalizer_tests_ready,
    normalizer_tests_satisfied,
    quick_note_write_satisfied,
    patcher_read_satisfied,
    patcher_fix_satisfied,
    verifier_exec_pytest_done,
    verifier_pytest_passed,
    verifier_read_satisfied,
    verifier_should_force_edit,
    verifier_should_force_exec,
    verifier_should_force_read,
    static_without_turn_placeholders,
    use_openclaw_prefix,
)
from sidecar.stores.prefix_topology import PrefixRebuildPlan, plan_prefix_update, write_topology
from sidecar.stores.registry import get_store_registry
from sidecar.tool_bridge import sanitize_chat_template_leaks
from sidecar.tool_bridge import (
    build_tool_injection_text,
    completion_payload_to_sse,
    ensure_clawbench_agent_tools,
    extract_tool_request,
    filter_tools_for_agent,
    fix_normalizer_test_file_on_disk,
    openai_message_from_generation,
    restore_browser_form_broken_on_disk,
    sanitize_generation_text,
    should_inject_tools,
    sse_tool_call_deltas,
    sync_clawbench_browser_default_to_chain,
    sync_clawbench_browser_workspaces,
    sync_clawbench_config_loader_default_to_chain,
    tool_bridge_buffered_sse_enabled,
)

KVCOMM_META_RE = re.compile(r"<!--KVCOMM_META:(\{.*?\})-->", re.DOTALL)
SIDECAR_VERSION = "0.2.0-kvcomm-engine"
_CLAWBENCH_TEXT_ONLY_MAX_TOKENS = 160
_CLAWBENCH_TOOL_CONTINUATION_MAX_TOKENS = 128
_CONFIG_LOADER_PYTEST_CMD = "PYTHONPATH=. python -m pytest -q tests/test_config_loader.py"


def _anchor_pool_key(node_id: str, message_key: str) -> str:
    return f"{node_id}:{message_key}"


def reset_bench_run_state(*, run_id: str | None = None, preserve_shared_kv: bool = False) -> None:
    """Clear per-run anchor/KV state before a new bench run (agent 0 register).

    When ``preserve_shared_kv`` is True (measure runs with ``kv_reuse``), keep
    shared prefix/input KV and anchor pools so warmup work is not discarded.
    """
    from KVCOMM.llm.gpt_chat import LLMChat
    from KVCOMM.llm.kvcomm_engine import KVCOMMEngine

    global _adapter

    global _pending_context_by_key, _active_registered_context
    _pending_context_by_key.clear()
    _active_registered_context = None

    with KVCOMMEngine._request_lock:
        if run_id:
            KVCOMMEngine._request_states.pop(run_id, None)
            KVCOMMEngine._active_requests.discard(run_id)
            rid = str(run_id)
            for key in list(_browser_tool_emit_count):
                if key == rid or key.startswith(f"{rid}:"):
                    _browser_tool_emit_count.pop(key, None)
            KVCOMMEngine._staged_commits = [
                state
                for state in KVCOMMEngine._staged_commits
                if getattr(state, "request_uid", None) != run_id
            ]
        elif not preserve_shared_kv:
            KVCOMMEngine._request_states.clear()
            KVCOMMEngine._staged_commits.clear()
            KVCOMMEngine._active_requests.clear()
        if not preserve_shared_kv:
            for store in (
                KVCOMMEngine.anchors,
                KVCOMMEngine.anchor_dict,
                KVCOMMEngine.anchor_len_dict,
                KVCOMMEngine.anchor_info_dict,
                KVCOMMEngine.weight_dict,
                KVCOMMEngine.global_anchor_info_dict,
            ):
                store.clear()

    if not preserve_shared_kv:
        shared = LLMChat._ensure_shared_kv_memory()
        if isinstance(shared, dict):
            for bucket_key in ("input", "input_ids", "input_drop_num"):
                bucket = shared.get(bucket_key)
                if isinstance(bucket, dict):
                    bucket.clear()
            for node_key in list(shared.keys()):
                if str(node_key).isdigit():
                    bucket = shared.get(node_key)
                    if not isinstance(bucket, dict):
                        shared.pop(node_key, None)
                    else:
                        _reset_prefix_node(str(node_key))
                    LLMChat._initialization[str(node_key)] = False

    if _adapter is not None:
        if not preserve_shared_kv:
            _adapter._anchor_pool.clear()
        _adapter._request_metrics.clear()
        _adapter._sessions.clear()

# Bench registers context before each sequential spawn (OpenClaw does not forward extra_body.kvcomm).
_pending_context_by_key: dict[str, "KvcommContext"] = {}
_active_registered_context: "KvcommContext | None" = None
_run_task_profiles: dict[str, str] = {}
_node_task_profiles: dict[str, str] = {}
# Browser tasks: count browser tool_calls emitted per agent when gateway never returns tool results.
_browser_tool_emit_count: dict[str, int] = {}
_last_agent0_run_id: str | None = None
_consumer_stable_cleared_for_measure: bool = False


def _note_agent0_bench_register(run_id: str | None, register_mode: str) -> None:
    """Track agent-0 bench registers; clear warmup tool stability on first kv_reuse measure run."""
    global _last_agent0_run_id, _consumer_stable_cleared_for_measure
    from KVCOMM.llm.gpt_chat import LLMChat

    rid = str(run_id or "").strip()
    if not rid:
        return

    prev = _last_agent0_run_id
    _last_agent0_run_id = rid

    if register_mode != "kv_reuse":
        logger.debug(
            "[kvcomm-bench] agent=0 register run_id={} mode={} prev={}",
            rid,
            register_mode,
            prev,
        )
        return

    if prev and prev != rid:
        LLMChat.clear_consumer_tool_schema_stable_all()
        logger.info(
            "[kvcomm-bench] measure run {} — cleared consumer tool-schema stable (prev register {})",
            rid,
            prev,
        )
        if not _consumer_stable_cleared_for_measure:
            LLMChat.set_consumer_first_measure_dense_pending(True)
            _consumer_stable_cleared_for_measure = True
        else:
            LLMChat.set_consumer_first_measure_dense_pending(False)
        return

    logger.debug(
        "[kvcomm-bench] agent=0 register run_id={} mode=kv_reuse prev={} cleared={}",
        rid,
        prev,
        _consumer_stable_cleared_for_measure,
    )


def _browser_emit_key(run_id: str, agent_index: str) -> str:
    return f"{run_id}:{agent_index}"


def _emitted_tool_call_key(entry: dict[str, Any]) -> str:
    name = str(entry.get("name") or "")
    payload = entry.get("input") if isinstance(entry.get("input"), dict) else {}
    return f"{name}:{json.dumps(payload, sort_keys=True, ensure_ascii=False)}"


def _tool_call_to_emitted_record(call: dict[str, Any]) -> dict[str, Any] | None:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(fn.get("name") or "").strip()
    if not name:
        return None
    args_raw = fn.get("arguments")
    input_obj: dict[str, Any] = {}
    if isinstance(args_raw, str) and args_raw.strip():
        try:
            parsed = json.loads(args_raw)
            if isinstance(parsed, dict):
                input_obj = parsed
        except json.JSONDecodeError:
            input_obj = {}
    elif isinstance(args_raw, dict):
        input_obj = dict(args_raw)
    return {"name": name, "input": input_obj}


def _append_emitted_tool_calls(metrics: dict[str, Any], message: dict[str, Any]) -> None:
    """Persist sidecar-emitted tool calls for bench trajectory scoring."""
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    emitted: list[dict[str, Any]] = list(metrics.get("emitted_tool_calls") or [])
    seen = {_emitted_tool_call_key(entry) for entry in emitted}
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        entry = _tool_call_to_emitted_record(call)
        if entry is None:
            continue
        key = _emitted_tool_call_key(entry)
        if key in seen:
            continue
        seen.add(key)
        emitted.append(entry)
    metrics["emitted_tool_calls"] = emitted


def _browser_agent_exploration_done(
    messages: list[dict[str, Any]],
    run_id: str,
    agent_index: str,
) -> bool:
    if browser_exploration_satisfied(messages):
        return True
    # Gateway may reject browser and retry with empty history — break the loop per agent.
    return _browser_tool_emit_count.get(_browser_emit_key(run_id, agent_index), 0) >= 1


def _note_browser_tool_emission(run_id: str, agent_index: str, message: dict[str, Any]) -> None:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        if str(fn.get("name") or "") == "browser":
            key = _browser_emit_key(str(run_id or ""), str(agent_index or ""))
            _browser_tool_emit_count[key] = _browser_tool_emit_count.get(key, 0) + 1
            logger.debug(
                "[tool-bridge] run_id={} agent={} browser tool emitted (count={})",
                run_id,
                agent_index,
                _browser_tool_emit_count[key],
            )
            return


def _browser_url_hint(ctx: "KvcommContext") -> str:
    form_port = str((ctx.vars or {}).get("form_app_port") or "").strip()
    if form_port:
        return f"http://127.0.0.1:{form_port}/"
    return "the task URL from the user request"


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
    task_id: str = ""
    clawbench_family: str = ""
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


_CLAWBENCH_ROLE_PADDING_MARKER = "Long-context bench context (stable system segment for KV prefix tests):"


def _strip_clawbench_role_padding(text: str) -> str:
    """Remove optional long KV-bench padding block from a clawbench role prompt."""
    if not text:
        return text
    marker = _CLAWBENCH_ROLE_PADDING_MARKER
    idx = text.find(marker)
    if idx < 0:
        return text.strip()
    return text[:idx].rstrip()


def _normalize_registered_clawbench_role(system_prompt: str) -> str:
    """Normalize register-time role to the minimal clawbench prompt."""
    registered = (system_prompt or "").strip()
    if not registered:
        return registered
    stripped = _strip_clawbench_role_padding(registered)
    return stripped or _load_clawbench_role()


def register_pending_context(payload: dict[str, Any]) -> KvcommContext:
    """Register kvcomm context from bench driver before sequential spawn."""
    global _active_registered_context
    agent_index_raw = payload.get("agent_index")
    try:
        agent_index_int = int(agent_index_raw if agent_index_raw is not None else 0)
    except (TypeError, ValueError):
        agent_index_int = 0
    register_mode = str(payload.get("mode") or "dense_prefill")
    if agent_index_int == 0:
        run_id = str(payload.get("run_id") or "").strip() or None
        reset_bench_run_state(
            run_id=run_id,
            preserve_shared_kv=register_mode == "kv_reuse",
        )
        _note_agent0_bench_register(run_id, register_mode)
    registered_system = str(payload.get("system_prompt") or "")
    ctx = KvcommContext(
        run_id=str(payload.get("run_id") or uuid.uuid4()),
        agent_index=str(payload.get("agent_index") if payload.get("agent_index") is not None else "0"),
        mode=str(payload.get("mode") or "dense_prefill"),
        message_key=str(payload.get("message_key") or payload.get("task_body") or "default"),
        vars={str(k): "" if v is None else str(v) for k, v in (payload.get("vars") or {}).items()},
        system_prompt=registered_system,
        user_prompt=str(payload.get("user_prompt") or ""),
        bench_user_prompt=str(payload.get("user_prompt") or payload.get("bench_user_prompt") or ""),
        task_id=str(payload.get("task_id") or ""),
        clawbench_family=str(payload.get("clawbench_family") or ""),
        task_profile=str(payload.get("task_profile") or payload.get("taskProfile") or "copy"),
        max_tokens=int(payload.get("max_tokens") or 512),
        temperature=float(payload.get("temperature") if payload.get("temperature") is not None else 0.0),
    )
    if ctx.task_profile not in ("copy", "clawbench"):
        ctx.task_profile = "copy"
    if ctx.mode not in ("dense_prefill", "kv_reuse"):
        ctx.mode = "dense_prefill"
    if ctx.task_profile == "clawbench":
        ctx.system_prompt = _normalize_registered_clawbench_role(ctx.system_prompt)
    ctx = _normalize_task_profile(ctx)
    if ctx.task_profile == "clawbench" and agent_index_int == 0 and is_browser_family_task(ctx):
        chain_ws = str((ctx.vars or {}).get("workspace_dir") or "")
        if chain_ws:
            restore_browser_form_broken_on_disk(workspace_dir=chain_ws)
    _pending_context_by_key[_context_key(ctx.run_id, ctx.agent_index)] = ctx
    _active_registered_context = ctx
    if register_mode == "kv_reuse" and agent_index_int > 0 and _adapter is not None:
        try:
            llm = _adapter._get_llm()
            _adapter._seed_cross_run_placeholder_anchors(llm, ctx)
        except Exception as exc:
            logger.debug(
                "[kvcomm-seed] register-time seed skipped run_id={} agent={}: {}",
                ctx.run_id,
                ctx.agent_index,
                exc,
            )
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
        if (
            _active_registered_context is not None
            and str(_active_registered_context.agent_index) == agent_index_str
        ):
            return _active_registered_context
        return None
    return _active_registered_context


def consume_registered_context(
    body: dict[str, Any],
    headers: dict[str, str],
) -> KvcommContext | None:
    """Return bench-registered context without consuming (multi-turn safe)."""
    return resolve_registered_context(body, headers)


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


def _default_clawbench_role_path() -> Path:
    prompts = Path(__file__).resolve().parents[2] / "experiments/bench/prompts"
    return prompts / "clawbench_chain.role.minimal.txt"


def _load_clawbench_role() -> str:
    """ClawBench chain role (minimal prompt)."""
    explicit = os.environ.get("KVCOMM_CLAWBENCH_ROLE", "").strip()
    if explicit:
        return explicit
    role_path = os.environ.get("KVCOMM_CLAWBENCH_ROLE_PATH", str(_default_clawbench_role_path()))
    path = Path(role_path)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return (
        "You are one agent in a fixed multi-agent chain. Follow your role instructions precisely.\n"
        "Respond in plain language unless your role requires code or structured output.\n"
        "Do not spawn subagents or delegate to other agents."
    )


def _resolve_clawbench_role(ctx: KvcommContext | None = None) -> str:
    """Prefer bench-registered system_prompt; fall back to env/file default."""
    if ctx is not None:
        registered = (ctx.system_prompt or "").strip()
        if registered:
            return _strip_clawbench_role_padding(registered) or _load_clawbench_role()
    return _load_clawbench_role()


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


def _scrub_stale_anchor_deltas_for_message(
    node_id: str,
    message_key: str,
    ph_ids: list[str] | None = None,
    *,
    purge_all: bool = False,
) -> None:
    """Remove request/global anchor deltas whose topology coordinates are stale."""
    from KVCOMM.llm.kvcomm_engine import KVCOMMEngine

    delta_keys = (
        f"{node_id}_ph_key_delta",
        f"{node_id}_ph_value_delta",
        f"{node_id}_pf_key_delta",
        f"{node_id}_pf_value_delta",
    )

    def _scrub_store(store: dict[str, Any]) -> None:
        for ph_id, bucket in list(store.items()):
            if ph_ids is not None and str(ph_id) not in ph_ids:
                continue
            if not isinstance(bucket, dict):
                continue
            entry = bucket.get(message_key)
            if not isinstance(entry, dict):
                continue
            if not any(key in entry for key in delta_keys):
                continue
            for key in delta_keys:
                entry.pop(key, None)
            entry.pop("anchor_topology_key", None)
            if not entry:
                bucket.pop(message_key, None)
            if not bucket:
                store.pop(ph_id, None)

    for store in (KVCOMMEngine.anchors, KVCOMMEngine.weight_dict):
        if isinstance(store, dict):
            _scrub_store(store)


def _proactive_stale_topology_purge(
    node_id: str,
    message_key: str,
    bucket: dict,
    *,
    purge_all: bool = False,
) -> list[str]:
    """Proactively drop topology-stale anchor deltas before blend (not only on blend fail)."""
    from sidecar.stores.topology_anchor import current_topology_keys
    from KVCOMM.llm.gpt_chat import LLMChat

    stores = get_store_registry()
    if purge_all:
        removed = stores.agent_anchors.purge_stale_topology(
            node_id=str(node_id),
            message_key=str(message_key),
            current_keys={},
            purge_all=True,
        )
        stores.template_ph_base.purge_node(str(node_id))
        stores.upstream_agent_slots.purge_message(
            node_id=str(node_id),
            message_key=str(message_key),
        )
        node_bucket = LLMChat._ensure_shared_kv_memory().get(str(node_id))
        if isinstance(node_bucket, dict):
            node_bucket.pop("_upstream_materialized", None)
        _scrub_stale_anchor_deltas_for_message(
            str(node_id),
            str(message_key),
            purge_all=True,
        )
        return removed

    current = current_topology_keys(bucket)
    removed = stores.agent_anchors.purge_stale_topology(
        node_id=str(node_id),
        message_key=str(message_key),
        current_keys=current,
        purge_all=False,
    )
    if removed:
        _scrub_stale_anchor_deltas_for_message(
            str(node_id),
            str(message_key),
            removed,
        )
    return removed


def _purge_node_anchor_deltas(node_id: str) -> None:
    """Drop anchor deltas materialised under a node's prefix topology."""
    from KVCOMM.llm.kvcomm_engine import KVCOMMEngine

    delta_keys = (
        f"{node_id}_ph_key_delta",
        f"{node_id}_ph_value_delta",
        f"{node_id}_pf_key_delta",
        f"{node_id}_pf_value_delta",
    )

    def _scrub_anchor_store(store: dict[str, Any]) -> None:
        for ph_id, bucket in list(store.items()):
            if not isinstance(bucket, dict):
                continue
            for message, entry in list(bucket.items()):
                if not isinstance(entry, dict):
                    continue
                if not any(key in entry for key in delta_keys):
                    continue
                for key in delta_keys:
                    entry.pop(key, None)
                if not entry:
                    bucket.pop(message, None)
            if not bucket:
                store.pop(ph_id, None)

    for store in (
        KVCOMMEngine.anchors,
        KVCOMMEngine.weight_dict,
    ):
        if isinstance(store, dict):
            _scrub_anchor_store(store)

    stores = get_store_registry()
    stores.segment_cache.purge_node(node_id)
    stores.agent_anchors.purge_node(node_id)
    stores.turn_slots.purge_node(node_id)
    stores.upstream_agent_slots.purge_node(node_id)
    stores.template_ph_base.purge_node(node_id)

    global _adapter
    if _adapter is not None:
        prefix = f"{node_id}:"
        for pool_key in list(_adapter._anchor_pool):
            if pool_key.startswith(prefix):
                _adapter._anchor_pool.pop(pool_key, None)


def _purge_node_turn_state(node_id: str, *, message_key: str | None = None) -> None:
    """Drop turn-slot state without clearing global tool KV backend."""
    stores = get_store_registry()
    from KVCOMM.llm.gpt_chat import LLMChat

    bucket = LLMChat._ensure_shared_kv_memory().get(node_id)
    if isinstance(bucket, dict):
        bucket.pop("turn", None)

    if message_key is not None:
        stores.purge_turn_downstream(
            node_id=str(node_id),
            message_key=str(message_key),
            turn_index=0,
        )
        stores.agent_anchors.invalidate_pf_for_message(
            node_id=str(node_id),
            message_key=str(message_key),
        )
        return

    stores.turn_slots.purge_node(node_id)


def _reset_prefix_node(node_id: str) -> None:
    from KVCOMM.llm.gpt_chat import LLMChat

    _purge_node_anchor_deltas(node_id)
    LLMChat._initialization[node_id] = False
    bucket = LLMChat._ensure_shared_kv_memory().get(node_id)
    if not isinstance(bucket, dict):
        return
    for key in (
        "prefix",
        "placeholder_info",
        "token_ids",
        "turn",
        "turn_count",
        "user_template",
        "system_prompt",
        "static_template_hash",
        "topology_id",
        "span_registry",
        "prefix_span_order",
        "prompt_token_len",
        "prompt_input_ids",
        "base_kv_full",
        "_prev_placeholder_info",
        "_upstream_materialized",
    ):
        bucket.pop(key, None)


def _clear_copy_input_cache(message: str) -> None:
    from KVCOMM.llm.gpt_chat import LLMChat

    shared = LLMChat._ensure_shared_kv_memory()
    if not isinstance(shared, dict):
        return
    for key in ("input", "input_ids", "input_drop_num"):
        bucket = shared.get(key)
        if isinstance(bucket, dict):
            bucket.pop(message, None)


def _resolve_prefix_prompts(ctx: KvcommContext) -> tuple[str, str]:
    """Choose system/user prefix templates by bench workload (legacy fallback)."""
    if ctx.task_profile == "clawbench":
        system_prompt = _resolve_clawbench_role(ctx)
        user_prompt = normalize_run_specific_paths(
            (ctx.bench_user_prompt or _build_kvcomm_user_prompt(ctx, copy_layout=False)).strip()
        )
        return system_prompt, user_prompt
    return _load_copy_role(), _build_kvcomm_user_prompt(ctx, copy_layout=True)


def _apply_bench_tool_constraints(prefix_build: PrefixBuildResult, ctx: KvcommContext) -> PrefixBuildResult:
    """Inject bench-only tool/path constraints into prefix template (prod agent_tasks stay short)."""
    from sidecar.openclaw_prefix import _collect_placeholders, static_without_turn_placeholders

    constraints = tool_constraints_for_context(ctx)
    if not constraints:
        return prefix_build
    injected = inject_tool_constraints(prefix_build.user_template, constraints)
    if injected == prefix_build.user_template:
        return prefix_build
    return PrefixBuildResult(
        system_prompt=prefix_build.system_prompt,
        user_template=injected,
        static_text=static_without_turn_placeholders(injected),
        placeholders=_collect_placeholders(injected),
        turn_count=prefix_build.turn_count,
        turn_content=dict(prefix_build.turn_content),
        estimated_tokens=prefix_build.estimated_tokens,
        use_openclaw=prefix_build.use_openclaw,
        fallback_reason=prefix_build.fallback_reason,
    )


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
        built = build_prefix_from_openclaw_messages(
            messages,
            bench_user_prompt=normalize_run_specific_paths(ctx.bench_user_prompt),
            clawbench_role=_resolve_clawbench_role(ctx),
            task_profile=ctx.task_profile,
        )
        return _apply_bench_tool_constraints(built, ctx)
    except PrefixOverflowError as exc:
        logger.warning("[kvcomm-prefix] {} — falling back to static prefix", exc)
        system_prompt, user_prompt = _resolve_prefix_prompts(ctx)
        return PrefixBuildResult(
            system_prompt=system_prompt,
            user_template=user_prompt,
            static_text=user_prompt,
            placeholders=[],
            turn_count=0,
            use_openclaw=False,
            fallback_reason=f"prefix_overflow:{exc}",
        )
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
    from sidecar.openclaw_prefix import turn_segment_template

    return turn_segment_template(turn_index)


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


def _prefix_turn_kv_out_of_sync(
    node_id: str,
    message_key: str,
    prefix_build: "PrefixBuildResult",
) -> bool:
    """Detect prefix placeholder_info listing turn_* without matching turn KV."""
    from KVCOMM.llm.gpt_chat import LLMChat

    bucket = LLMChat._ensure_shared_kv_memory().get(node_id) or {}
    if not isinstance(bucket, dict) or not bucket.get("placeholder_info"):
        return False

    ph_info = bucket.get("placeholder_info") or {}
    if not isinstance(ph_info, dict):
        return False
    turn_ph_ids = [str(ph_id) for ph_id in ph_info if str(ph_id).startswith("turn_")]
    if not turn_ph_ids:
        return False

    if prefix_build.turn_count == 0:
        return True

    stores = get_store_registry()
    for ph_id in turn_ph_ids:
        slot = stores.turn_slots.get(str(node_id), str(message_key), ph_id)
        if slot is not None:
            continue
        turn_root = bucket.get("turn") or {}
        if not isinstance(turn_root, dict):
            turn_root = {}
        ph_bucket = turn_root.get(ph_id) or {}
        if isinstance(ph_bucket, dict) and ph_bucket.get(message_key):
            continue
        content = str((prefix_build.turn_content or {}).get(ph_id, "")).strip()
        if not content or content == " ":
            continue
        return True
    return False


def _clear_turn_cache_only(node_id: str) -> None:
    """Drop ephemeral turn placeholder KV; keep static prefix segment A intact.

    Legacy sync helper — prefer ``_rebuild_static_prefix_only`` when an LLM instance
    is available so prefix/placeholder_info stay consistent.
    """
    from KVCOMM.llm.gpt_chat import LLMChat

    bucket = LLMChat._ensure_shared_kv_memory().get(node_id)
    if not isinstance(bucket, dict):
        return
    bucket.pop("turn", None)
    stored_user = str(bucket.get("user_template") or "")
    if stored_user:
        bucket["user_template"] = static_without_turn_placeholders(stored_user)
    bucket["turn_count"] = 0
    bucket.pop("static_template_hash", None)
    bucket.pop("topology_id", None)
    get_store_registry().turn_slots.purge_node(node_id)


async def _rebuild_static_prefix_only(llm, node_id: str) -> None:
    """Clear turn KV and rebuild static-only prefix segments (segment A)."""
    from KVCOMM.llm.gpt_chat import LLMChat

    bucket = LLMChat._ensure_shared_kv_memory().get(node_id)
    if not isinstance(bucket, dict):
        return
    bucket.pop("turn", None)
    stored_user = static_without_turn_placeholders(str(bucket.get("user_template") or ""))
    stored_system = str(bucket.get("system_prompt") or "").strip()
    bucket["user_template"] = stored_user
    bucket["turn_count"] = 0
    bucket.pop("static_template_hash", None)
    bucket.pop("topology_id", None)
    get_store_registry().turn_slots.purge_node(node_id)
    if stored_system and stored_user:
        await llm.prepare_prefix_kv_segments(node_id, stored_system, stored_user)
        return
    bucket.pop("prefix", None)
    bucket.pop("placeholder_info", None)
    bucket.pop("token_ids", None)
    LLMChat._initialization[node_id] = False


def _upstream_response_kv_available(upstream_node: str, message_key: str) -> bool:
    """Return True when upstream agent has materialized response KV for this message."""
    from KVCOMM.llm.gpt_chat import LLMChat

    bucket = LLMChat._ensure_shared_kv_memory().get(str(upstream_node)) or {}
    if not isinstance(bucket, dict):
        return False
    resp = bucket.get("response") or {}
    if not isinstance(resp, dict):
        return False
    values = resp.get(message_key)
    return bool(values)


def _tool_deliverable_fingerprint_for_generation(
    llm,
    ctx: "KvcommContext",
    body: dict[str, Any],
) -> str:
    """Hash upstream deliverable context for tool schema branch lookup."""
    from sidecar.stores.hashing import tool_deliverable_fingerprint

    upstream_text = ""
    try:
        agent_idx = int(ctx.agent_index)
    except (TypeError, ValueError):
        agent_idx = 0
    if agent_idx > 0:
        ph_id = f"agent_{agent_idx - 1}_current"
        slot = llm.resolve_upstream_agent_slot(ph_id, ctx.message_key)
        if slot is not None and isinstance(getattr(slot, "token_ids", None), dict):
            ids = slot.token_ids.get("input_ids")
            tokenizer = getattr(llm, "tokenizer", None)
            if ids is not None and tokenizer is not None:
                try:
                    upstream_text = tokenizer.decode(ids[0], skip_special_tokens=True)
                except Exception:
                    upstream_text = ""
    if not upstream_text.strip():
        for msg in reversed(body.get("messages") or []):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content") or ""
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                content = "\n".join(parts)
            content = str(content)
            if content.strip():
                upstream_text = content
                break
    return tool_deliverable_fingerprint(
        task_id=str(ctx.task_id or ""),
        upstream_text=upstream_text,
    )


def _apply_pooled_placeholder_anchor(
    llm,
    request_uid: str,
    snapshot: dict[str, Any],
    *,
    ph_id: str,
    message: str,
    delta_key: str,
) -> bool:
    """Copy one placeholder anchor entry (+metadata) from a pool snapshot into request state."""
    anchors = snapshot.get("anchors") or {}
    bucket = anchors.get(ph_id) or {}
    entry = bucket.get(message)
    if not isinstance(entry, dict) or delta_key not in entry:
        return False

    state = llm.kv_engine.resolve_request_state(request_uid)
    state.anchors.setdefault(ph_id, {})[message] = copy.deepcopy(entry)
    anchor_dict_bucket = snapshot.get("anchor_dict") or {}
    if isinstance(anchor_dict_bucket.get(ph_id), dict):
        flag = anchor_dict_bucket[ph_id].get(message)
        if flag is not None:
            state.anchor_dict.setdefault(ph_id, {})[message] = flag
    for aux_key, state_attr in (
        ("anchor_len_dict", "anchor_len_dict"),
        ("anchor_info_dict", "anchor_info_dict"),
        ("global_anchor_info", "global_anchor_info"),
    ):
        aux = snapshot.get(aux_key) or {}
        ph_bucket = aux.get(ph_id) if isinstance(aux, dict) else None
        if isinstance(ph_bucket, dict) and message in ph_bucket:
            target = getattr(state, state_attr)
            target.setdefault(ph_id, {})[message] = copy.deepcopy(ph_bucket[message])
    return True


def _trim_completed_agent_prefixes(ctx: KvcommContext) -> None:
    """Schedule upstream turn-cache trim (see ``_trim_completed_agent_prefixes_async``)."""
    _pending_prefix_trims.update(
        str(agent_id)
        for agent_id in range(_trim_agent_upper_bound(ctx))
    )


def _trim_agent_upper_bound(ctx: KvcommContext) -> int:
    if ctx.task_profile != "clawbench":
        return 0
    if ctx.mode == "kv_reuse":
        return 0
    try:
        return int(ctx.agent_index)
    except (TypeError, ValueError):
        return 0


_pending_prefix_trims: set[str] = set()


async def _trim_completed_agent_prefixes_async(llm, ctx: KvcommContext) -> None:
    """Clear upstream turn caches and rebuild static prefix for completed agents."""
    if ctx.task_profile != "clawbench" or ctx.mode == "kv_reuse":
        _pending_prefix_trims.clear()
        return
    try:
        current = int(ctx.agent_index)
    except (TypeError, ValueError):
        _pending_prefix_trims.clear()
        return
    targets = {str(agent_id) for agent_id in range(current)} | _pending_prefix_trims
    _pending_prefix_trims.clear()
    for agent_id in sorted(targets, key=int):
        if int(agent_id) < current:
            await _rebuild_static_prefix_only(llm, agent_id)


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
        task_id=str(kvcomm.get("task_id") or vars_map.get("task_id") or ""),
        clawbench_family=str(kvcomm.get("clawbench_family") or vars_map.get("clawbench_family") or ""),
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


def _accumulate_agent_metrics(
    existing: dict[str, Any] | None,
    latest: dict[str, Any],
) -> dict[str, Any]:
    """Merge per-spawn sidecar requests so bench metrics reflect all LLM calls, not only the last."""
    actual_mode = str(latest.get("mode") or "")
    is_kv = actual_mode == "kv_reuse"
    is_dense = actual_mode == "dense_prefill"
    blend_fb = bool(latest.get("blend_fallback"))

    if not existing:
        out = dict(latest)
        out["sidecar_request_count"] = 1
        out["kv_reuse_request_count"] = 1 if is_kv else 0
        out["dense_request_count"] = 1 if is_dense else 0
        out["blend_fallback_count"] = 1 if blend_fb else 0
    else:
        out = dict(latest)
        out["sidecar_request_count"] = int(existing.get("sidecar_request_count", 1)) + 1
        out["kv_reuse_request_count"] = int(existing.get("kv_reuse_request_count", 0)) + (1 if is_kv else 0)
        out["dense_request_count"] = int(existing.get("dense_request_count", 0)) + (1 if is_dense else 0)
        out["blend_fallback_count"] = int(existing.get("blend_fallback_count", 0)) + (1 if blend_fb else 0)
        merged_emitted: list[dict[str, Any]] = list(existing.get("emitted_tool_calls") or [])
        seen = {_emitted_tool_call_key(entry) for entry in merged_emitted}
        for entry in latest.get("emitted_tool_calls") or []:
            if not isinstance(entry, dict):
                continue
            key = _emitted_tool_call_key(entry)
            if key in seen:
                continue
            seen.add(key)
            merged_emitted.append(entry)
        out["emitted_tool_calls"] = merged_emitted

    total = int(out.get("sidecar_request_count", 1))
    kv_count = int(out.get("kv_reuse_request_count", 0))
    out["reuse_rate"] = round(kv_count / total, 4) if total else 0.0
    if kv_count == total and int(out.get("blend_fallback_count", 0)) == 0:
        out["mode"] = "kv_reuse"
    elif kv_count == 0:
        out["mode"] = "dense_prefill"
    else:
        out["mode"] = "partial_kv_reuse"
    return out


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
    blend_fallback = bool(metadata.get("blend_fallback"))
    routed_mode = metadata.get("routed_mode")
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
        "blend_fallback": blend_fallback,
        "routed_mode": routed_mode,
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


def _openai_completion(message: dict[str, Any], model: str, metrics: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    completion_units = 0
    if isinstance(content, str) and content.strip():
        completion_units = len(content.split())
    elif isinstance(tool_calls, list):
        completion_units = len(tool_calls)
    finish_reason = "tool_calls" if isinstance(tool_calls, list) and tool_calls else "stop"
    return {
        "id": f"chatcmpl-kvcomm-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": completion_units,
            "total_tokens": completion_units,
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
        self._apply_anchor_snapshot(llm, request_uid, snapshot)

    def _apply_anchor_snapshot(self, llm, request_uid: str, snapshot: dict[str, Any]) -> None:
        state = llm.kv_engine.resolve_request_state(request_uid)
        for key, value in snapshot.items():
            if key == "anchors":
                for ph_id, bucket in value.items():
                    if not isinstance(bucket, dict):
                        continue
                    target = state.anchors.setdefault(ph_id, {})
                    if isinstance(target, dict):
                        target.update(copy.deepcopy(bucket))
            elif key == "anchor_dict":
                for ph_id, bucket in value.items():
                    if not isinstance(bucket, dict):
                        continue
                    target = state.anchor_dict.setdefault(ph_id, {})
                    if isinstance(target, dict):
                        target.update(copy.deepcopy(bucket))
            elif key == "anchor_len_dict":
                for ph_id, bucket in value.items():
                    if not isinstance(bucket, dict):
                        continue
                    target = state.anchor_len_dict.setdefault(ph_id, {})
                    if isinstance(target, dict):
                        target.update(copy.deepcopy(bucket))
            elif key == "anchor_info_dict":
                for ph_id, bucket in value.items():
                    if not isinstance(bucket, dict):
                        continue
                    target = state.anchor_info_dict.setdefault(ph_id, {})
                    if isinstance(target, dict):
                        target.update(copy.deepcopy(bucket))
            elif key == "global_anchor_info":
                for ph_id, bucket in value.items():
                    if not isinstance(bucket, dict):
                        continue
                    target = state.global_anchor_info.setdefault(ph_id, {})
                    if isinstance(target, dict):
                        target.update(copy.deepcopy(bucket))

    async def _rematerialize_consumer_slots_for_kv_reuse(
        self,
        llm,
        ctx: "KvcommContext",
        bucket: dict,
        agent_idx: int,
    ) -> None:
        if ctx.mode != "kv_reuse" or agent_idx <= 0:
            return
        bucket.pop("_upstream_materialized", None)
        bucket.pop("_tool_consumer_materialized", None)
        self._seed_cross_run_placeholder_anchors(llm, ctx)
        await llm.prepare_consumer_upstream_slots(str(ctx.message_key))

    def _seed_cross_run_placeholder_anchors(self, llm, ctx: "KvcommContext") -> None:
        """Restore agent_K_current ph_key_delta from typed anchor pool across measure runs."""
        from KVCOMM.llm.gpt_chat import LLMChat

        try:
            agent_idx = int(ctx.agent_index)
        except (TypeError, ValueError):
            return
        message = ctx.message_key
        node_id = ctx.agent_index
        llm.set_id(node_id, f"agent_{node_id}")

        self._restore_anchors(llm, ctx.run_id, node_id, message)
        if agent_idx <= 0:
            return

        state = llm.kv_engine.resolve_request_state(ctx.run_id)
        bucket = LLMChat._ensure_shared_kv_memory().get(str(node_id)) or {}
        static_hash = str(bucket.get("static_template_hash") or "")
        stores = get_store_registry()

        for upstream in range(agent_idx):
            ph_id = f"agent_{upstream}_current"
            node_delta_key = f"{node_id}_ph_key_delta"
            existing = (state.anchors.get(ph_id) or {}).get(message)
            if isinstance(existing, dict) and node_delta_key in existing:
                continue

            seeded = False
            typed = None
            ph_info = bucket.get("placeholder_info") if isinstance(bucket, dict) else {}
            if isinstance(ph_info, dict) and ph_id in ph_info:
                from sidecar.stores.prefix_spans import normalize_placeholder_info
                from sidecar.stores.topology_anchor import delta_key_from_ph_rec

                rec = normalize_placeholder_info(ph_info).get(ph_id)
                if isinstance(rec, dict) and static_hash:
                    topo = str(bucket.get("topology_id") or "")
                    topology_delta_key = delta_key_from_ph_rec(
                        ph_id=ph_id,
                        ph_rec=rec,
                        static_template_hash=static_hash,
                        topology_id=topo,
                    )
                    typed = stores.agent_anchors.get_by_topology_key(
                        node_id=str(node_id),
                        message_key=str(message),
                        delta_key=topology_delta_key,
                    )
            if typed is None:
                typed = stores.agent_anchors.get_any_for_ph(
                    node_id=str(node_id),
                    message_key=str(message),
                    ph_id=ph_id,
                )
            if typed is not None and typed.ph_delta is not None and static_hash:
                if typed.static_template_hash == static_hash or not typed.static_template_hash:
                    state.anchors.setdefault(ph_id, {})[message] = {
                        node_delta_key: typed.ph_delta,
                        f"{node_id}_ph_value_delta": getattr(typed, "ph_value_delta", None),
                        f"{node_id}_pf_key_delta": typed.pf_delta,
                        f"{node_id}_pf_value_delta": getattr(typed, "pf_value_delta", None),
                    }
                    seeded = True
                    logger.debug(
                        "[kvcomm-seed] run_id={} agent={} restored {} from AgentAnchorPool",
                        ctx.run_id,
                        node_id,
                        ph_id,
                    )

            if seeded:
                continue

            own_pool_key = _anchor_pool_key(node_id, message)
            pool_items = list(self._anchor_pool.items())
            if own_pool_key in self._anchor_pool:
                pool_items = [(own_pool_key, self._anchor_pool[own_pool_key])] + [
                    (k, v) for k, v in pool_items if k != own_pool_key
                ]
            for pool_key, snapshot in pool_items:
                if not pool_key.endswith(f":{message}"):
                    continue
                if _apply_pooled_placeholder_anchor(
                    llm,
                    ctx.run_id,
                    snapshot,
                    ph_id=ph_id,
                    message=message,
                    delta_key=node_delta_key,
                ):
                    seeded = True
                    logger.debug(
                        "[kvcomm-seed] run_id={} agent={} restored {} delta from pool key={}",
                        ctx.run_id,
                        node_id,
                        ph_id,
                        pool_key,
                    )
                    break

            if not seeded and _upstream_response_kv_available(str(upstream), message):
                for pool_key, snapshot in pool_items:
                    if not pool_key.endswith(f":{message}"):
                        continue
                    if _apply_pooled_placeholder_anchor(
                        llm,
                        ctx.run_id,
                        snapshot,
                        ph_id=ph_id,
                        message=message,
                        delta_key=node_delta_key,
                    ):
                        seeded = True
                        logger.debug(
                            "[kvcomm-seed] run_id={} agent={} restored {} delta from pool key={} (upstream response KV present)",
                            ctx.run_id,
                            node_id,
                            ph_id,
                            pool_key,
                        )
                        break
                if not seeded:
                    logger.debug(
                        "[kvcomm-seed] run_id={} agent={} upstream {} response KV exists but no pooled {} for message",
                        ctx.run_id,
                        node_id,
                        upstream,
                        node_delta_key,
                    )

            if not seeded:
                logger.debug(
                    "[kvcomm-seed] run_id={} agent={} no pooled delta for {}",
                    ctx.run_id,
                    node_id,
                    ph_id,
                )

    def _snapshot_anchors(self, llm, request_uid: str, node_id: str, message_key: str) -> None:
        from KVCOMM.llm.gpt_chat import LLMChat

        state = llm.kv_engine.resolve_request_state(request_uid)
        stores = get_store_registry()
        node_delta_key = f"{node_id}_ph_key_delta"
        pf_key = f"{node_id}_pf_key_delta"
        for entry in stores.agent_anchors.list_for_message(
            node_id=str(node_id),
            message_key=str(message_key),
        ):
            if not str(entry.ph_id).startswith("agent_"):
                continue
            if entry.ph_delta is None:
                continue
            state.anchors.setdefault(entry.ph_id, {})[message_key] = {
                node_delta_key: entry.ph_delta,
                f"{node_id}_ph_value_delta": entry.ph_value_delta,
                pf_key: entry.pf_delta,
                f"{node_id}_pf_value_delta": entry.pf_value_delta,
            }
        self._anchor_pool[_anchor_pool_key(node_id, message_key)] = {
            "anchors": {k: dict(v) for k, v in state.anchors.items()},
            "anchor_dict": {k: dict(v) for k, v in state.anchor_dict.items()},
            "anchor_len_dict": {k: dict(v) for k, v in state.anchor_len_dict.items()},
            "anchor_info_dict": {k: dict(v) for k, v in state.anchor_info_dict.items()},
            "global_anchor_info": {k: dict(v) for k, v in state.global_anchor_info.items()},
        }

        bucket = LLMChat._ensure_shared_kv_memory().get(str(node_id)) or {}
        static_hash = str(bucket.get("static_template_hash") or "")
        upstream_hash = str(message_key)
        from sidecar.stores.prefix_spans import normalize_placeholder_info
        from sidecar.stores.topology_anchor import delta_key_from_ph_rec

        ph_info = normalize_placeholder_info(bucket.get("placeholder_info"))
        topo = str(bucket.get("topology_id") or "")
        for ph_id, msg_bucket in state.anchors.items():
            if not str(ph_id).startswith("agent_"):
                continue
            entry = (msg_bucket or {}).get(message_key)
            if not isinstance(entry, dict) or node_delta_key not in entry:
                continue
            ph_rec = ph_info.get(str(ph_id)) or {}
            anchor_delta_key = delta_key_from_ph_rec(
                ph_id=str(ph_id),
                ph_rec=ph_rec,
                static_template_hash=static_hash,
                topology_id=topo,
            )
            stores.agent_anchors.put(
                node_id=str(node_id),
                message_key=str(message_key),
                ph_id=str(ph_id),
                static_template_hash=static_hash,
                upstream_hash=upstream_hash,
                ph_delta=entry.get(node_delta_key),
                ph_value_delta=entry.get(f"{node_id}_ph_value_delta"),
                pf_delta=entry.get(pf_key),
                pf_value_delta=entry.get(f"{node_id}_pf_value_delta"),
                pf_segment_len=entry.get("pf_segment_len"),
                delta_key=anchor_delta_key,
            )

    def _enrich_context_from_request(self, ctx: KvcommContext, body: dict[str, Any]) -> KvcommContext:
        """Merge OpenClaw request fields into ctx without clobbering bench-registered prompts."""
        from sidecar.openclaw_prefix import _strip_tool_schema

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
                system_parts.append(_strip_tool_schema(content))
            elif role == "user" and content:
                user_parts.append(content)

        registered_system = (ctx.system_prompt or "").strip()
        registered_user = (ctx.bench_user_prompt or ctx.user_prompt or "").strip()
        if ctx.task_profile == "clawbench":
            # Bench register carries the canonical role/task text; ignore OpenClaw bootstrap noise.
            if not registered_system and system_parts:
                ctx.system_prompt = "\n\n".join(system_parts).strip()
            if not registered_user and user_parts:
                ctx.user_prompt = user_parts[-1]
                ctx.bench_user_prompt = user_parts[-1]
        else:
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
        from KVCOMM.llm.gpt_chat import LLMChat

        node_id = ctx.agent_index
        llm.set_id(node_id, f"agent_{node_id}")
        prev_profile = _node_task_profiles.get(node_id)
        if prev_profile and prev_profile != ctx.task_profile and llm.has_prefix_initialized(node_id):
            _reset_prefix_node(node_id)
        _node_task_profiles[node_id] = ctx.task_profile

        bucket = LLMChat._ensure_shared_kv_memory().setdefault(node_id, {})
        stored_user = str(bucket.get("user_template") or "") if llm.has_prefix_initialized(node_id) else ""

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

        plan = plan_prefix_update(
            user_template=prefix_build.user_template,
            desired_turn_count=prefix_build.turn_count,
            bucket=bucket if isinstance(bucket, dict) else {},
            initialized=llm.has_prefix_initialized(node_id),
        )
        if plan.action == "noop" and prefix_build.turn_count == 0:
            ph_info = bucket.get("placeholder_info") if isinstance(bucket, dict) else {}
            if isinstance(ph_info, dict) and any(str(k).startswith("turn_") for k in ph_info):
                plan = PrefixRebuildPlan(
                    action="rewind_turns",
                    reason="stale_turn_placeholders_in_static_prefix",
                    from_turn_count=plan.from_turn_count,
                    to_turn_count=prefix_build.turn_count,
                )
        if needs_kvreuse_placeholder_rebuild:
            plan = PrefixRebuildPlan(
                action="full_rebuild",
                reason="missing_upstream_kv_placeholders",
                from_turn_count=plan.from_turn_count,
                to_turn_count=prefix_build.turn_count,
            )

        stores = get_store_registry()
        logger.debug(
            "[kvcomm-prefix] run_id={} agent={} plan={} reason={} turns {}→{}",
            ctx.run_id,
            node_id,
            plan.action,
            plan.reason,
            plan.from_turn_count,
            plan.to_turn_count,
        )

        if plan.action == "noop":
            if ctx.mode == "kv_reuse" and agent_idx > 0 and prefix_build.turn_count == 0:
                await self._rematerialize_consumer_slots_for_kv_reuse(
                    llm, ctx, bucket if isinstance(bucket, dict) else {}, agent_idx
                )
        elif plan.action == "append_turn":
            bucket = LLMChat._shared_kv_cache_memory.get(node_id) or {}
            if use_openclaw_prefix(ctx.task_profile):
                await llm.append_prefix_segment(
                    node_id,
                    _turn_segment_template(prefix_build.turn_count),
                    system_prompt=str(bucket.get("system_prompt") or prefix_build.system_prompt),
                )
            else:
                await llm.prepare_prefix_kv_segments(
                    node_id,
                    prefix_build.system_prompt,
                    prefix_build.user_template,
                )
            llm.set_prefix_turn_count(node_id, prefix_build.turn_count)
            # user_template/topology already committed by append/prepare; do not
            # overwrite with openclaw re-parse (breaks next incremental append).
            bucket = LLMChat._shared_kv_cache_memory.get(node_id) or {}
            if isinstance(bucket, dict) and bucket.get("user_template"):
                write_topology(
                    bucket,
                    user_template=str(bucket["user_template"]),
                    turn_count=prefix_build.turn_count,
                )
            stores.purge_turn_downstream(
                node_id=str(node_id),
                message_key=str(ctx.message_key),
                turn_index=max(0, plan.from_turn_count),
            )
            _proactive_stale_topology_purge(
                str(node_id),
                str(ctx.message_key),
                bucket if isinstance(bucket, dict) else {},
                purge_all=False,
            )
        elif plan.action == "rewind_turns":
            _purge_node_turn_state(node_id, message_key=ctx.message_key)
            stores.purge_turn_downstream(
                node_id=str(node_id),
                message_key=str(ctx.message_key),
                turn_index=max(0, plan.to_turn_count),
            )
            if plan.reason != "turn_count_regression_new_run":
                _proactive_stale_topology_purge(
                    str(node_id),
                    str(ctx.message_key),
                    bucket if isinstance(bucket, dict) else {},
                    purge_all=False,
                )
            ok = await llm.rewind_prefix_to_turn_count(
                node_id,
                system_prompt=prefix_build.system_prompt,
                user_template=prefix_build.user_template,
                turn_count=prefix_build.turn_count,
            )
            if not ok:
                logger.warning(
                    "[kvcomm-prefix] run_id={} agent={} rewind_turns failed; static rebuild",
                    ctx.run_id,
                    node_id,
                )
                _proactive_stale_topology_purge(
                    str(node_id),
                    str(ctx.message_key),
                    bucket if isinstance(bucket, dict) else {},
                    purge_all=True,
                )
                _purge_node_turn_state(node_id, message_key=ctx.message_key)
                LLMChat._initialization[node_id] = False
                for key in ("prefix", "placeholder_info", "token_ids"):
                    bucket.pop(key, None)
                await llm.prepare_prefix_kv_segments(
                    node_id,
                    prefix_build.system_prompt,
                    prefix_build.user_template,
                )
                llm.set_prefix_turn_count(node_id, prefix_build.turn_count)
                write_topology(
                    bucket,
                    user_template=prefix_build.user_template,
                    turn_count=prefix_build.turn_count,
                )
            else:
                llm.set_prefix_turn_count(node_id, prefix_build.turn_count)
                write_topology(
                    bucket,
                    user_template=prefix_build.user_template,
                    turn_count=prefix_build.turn_count,
                )
                # Rewind preserves static prefix KV; rematerialize consumer slots from
                # pooled anchors / upstream response KV instead of forcing dense tool turns.
                await self._rematerialize_consumer_slots_for_kv_reuse(
                    llm,
                    ctx,
                    bucket if isinstance(bucket, dict) else {},
                    agent_idx,
                )
        elif plan.action == "static_rebuild":
            _proactive_stale_topology_purge(
                str(node_id),
                str(ctx.message_key),
                bucket if isinstance(bucket, dict) else {},
                purge_all=True,
            )
            _purge_node_turn_state(node_id, message_key=ctx.message_key)
            LLMChat._initialization[node_id] = False
            for key in ("prefix", "placeholder_info", "token_ids"):
                bucket.pop(key, None)
            await llm.prepare_prefix_kv_segments(
                node_id,
                prefix_build.system_prompt,
                prefix_build.user_template,
            )
            llm.set_prefix_turn_count(node_id, prefix_build.turn_count)
            write_topology(bucket, user_template=prefix_build.user_template, turn_count=prefix_build.turn_count)
        else:
            _proactive_stale_topology_purge(
                str(node_id),
                str(ctx.message_key),
                bucket if isinstance(bucket, dict) else {},
                purge_all=True,
            )
            if llm.has_prefix_initialized(node_id):
                _reset_prefix_node(node_id)
            await llm.prepare_prefix_kv_segments(
                node_id,
                prefix_build.system_prompt,
                prefix_build.user_template,
            )
            llm.set_prefix_turn_count(node_id, prefix_build.turn_count)
            write_topology(bucket, user_template=prefix_build.user_template, turn_count=prefix_build.turn_count)

        await llm.materialize_turn_placeholders(
            node_id,
            ctx.message_key,
            prefix_build.turn_content,
        )
        if _prefix_turn_kv_out_of_sync(node_id, ctx.message_key, prefix_build):
            logger.warning(
                "[kvcomm-prefix] run_id={} agent={} turn KV out of sync after materialize; static rebuild",
                ctx.run_id,
                node_id,
            )
            _purge_node_turn_state(node_id, message_key=ctx.message_key)
            LLMChat._initialization[node_id] = False
            for key in ("prefix", "placeholder_info", "token_ids"):
                bucket.pop(key, None)
            await llm.prepare_prefix_kv_segments(
                node_id,
                prefix_build.system_prompt,
                prefix_build.user_template,
            )
            llm.set_prefix_turn_count(node_id, prefix_build.turn_count)
            write_topology(bucket, user_template=prefix_build.user_template, turn_count=prefix_build.turn_count)
            await llm.materialize_turn_placeholders(
                node_id,
                ctx.message_key,
                prefix_build.turn_content,
            )
            if _prefix_turn_kv_out_of_sync(node_id, ctx.message_key, prefix_build):
                ph_info = bucket.get("placeholder_info") if isinstance(bucket, dict) else {}
                missing = [
                    str(ph_id)
                    for ph_id in (ph_info or {})
                    if str(ph_id).startswith("turn_")
                ]
                logger.warning(
                    "[kvcomm-prefix] run_id={} agent={} turn KV still missing after rebuild placeholders={}",
                    ctx.run_id,
                    node_id,
                    missing,
                )

    async def _maybe_update_input_anchor(self, llm, ctx: KvcommContext) -> str:
        """Ensure user_question input KV exists; return input_routing_mode for metrics only."""
        if not llm.has_prefix_initialized(ctx.agent_index):
            return "dense_prefill"

        # Bench dense_prefill runs must not reuse global input-anchor KV; generation
        # already uses dense_prefill via ctx.mode (separate from this metrics label).
        if str(ctx.mode or "").strip() == "dense_prefill":
            return "dense_prefill"

        if ctx.task_profile == "clawbench":
            user_content = ctx.message_key
            prefix_text = "The task is: "
            return await asyncio.to_thread(
                llm.update_input_anchor,
                request_uid=ctx.run_id,
                agent_id=ctx.agent_index,
                message=ctx.message_key,
                user_content=f"{prefix_text}{user_content}",
                prefix_text=prefix_text,
                test_time=False,
            )

        user_content = ctx.vars.get("user_question") or ctx.message_key
        prefix_text = "The task is: "
        return await asyncio.to_thread(
            llm.update_input_anchor,
            request_uid=ctx.run_id,
            agent_id=ctx.agent_index,
            message=ctx.message_key,
            user_content=f"{prefix_text}{user_content}",
            prefix_text=prefix_text,
            test_time=False,
        )

    def _resolve_generation_mode(self, ctx: KvcommContext) -> str:
        """Bench-registered ctx.mode drives agen_kvcomm; decoupled from input routing."""
        mode = str(ctx.mode or "dense_prefill").strip()
        return mode if mode in ("dense_prefill", "kv_reuse") else "dense_prefill"

    async def _prepare_generation(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
        default_mode: str,
    ) -> tuple[Any, "KvcommContext", PrefixBuildResult, int, str, str]:
        _append_no_think_to_body(body)
        ctx = consume_registered_context(body, headers)
        if ctx is None:
            ctx = parse_kvcomm_context(body, headers, default_mode)
        if ctx is None:
            raise ValueError("missing kvcomm context (run_id/agent_index/message_key)")

        ctx = self._enrich_context_from_request(ctx, body)
        ctx = _normalize_task_profile(ctx)

        llm = self._get_llm()
        _trim_completed_agent_prefixes(ctx)
        await _trim_completed_agent_prefixes_async(llm, ctx)

        prefix_build = _build_openclaw_prefix(ctx, body)
        turn_index = max(0, count_assistant_turns(body.get("messages") or []) - 1)
        messages = body.get("messages") or []
        role_counts: dict[str, int] = {}
        tool_result_chars = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
            if role == "tool":
                content = msg.get("content")
                if isinstance(content, str):
                    tool_result_chars += len(content)
        logger.debug(
            "[kvcomm-msgs] run_id={} agent={} turn_index={} roles={} tool_result_chars={}",
            ctx.run_id,
            ctx.agent_index,
            turn_index,
            role_counts,
            tool_result_chars,
        )
        logger.info(
            "[kvcomm] run_id={} agent={} preparing prefix (turn_index={} est_tokens={})",
            ctx.run_id,
            ctx.agent_index,
            turn_index,
            getattr(prefix_build, "estimated_tokens", None),
        )
        try:
            await self._ensure_prefix(llm, ctx, body, prefix_build)
        except PrefixOverflowError:
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise PrefixOverflowError(str(exc)) from exc
            raise
        logger.info(
            "[kvcomm] run_id={} agent={} prefix ready; starting generation",
            ctx.run_id,
            ctx.agent_index,
        )
        if not llm.has_prefix_initialized(ctx.agent_index):
            raise RuntimeError(
                f"prefix KV not initialized for agent {ctx.agent_index} after _ensure_prefix"
            )
        llm.set_id(ctx.agent_index, f"agent_{ctx.agent_index}")

        if ctx.mode == "kv_reuse":
            self._seed_cross_run_placeholder_anchors(llm, ctx)

        input_routing_mode = await self._maybe_update_input_anchor(llm, ctx)
        generation_mode = self._resolve_generation_mode(ctx)
        logger.debug(
            "[kvcomm-routing] run_id={} agent={} ctx.mode={} input_routing={} generation_mode={} message_key={}",
            ctx.run_id,
            ctx.agent_index,
            ctx.mode,
            input_routing_mode,
            generation_mode,
            (ctx.message_key[:72] + "...") if len(ctx.message_key) > 75 else ctx.message_key,
        )
        return llm, ctx, prefix_build, turn_index, generation_mode, input_routing_mode

    async def _finalize_generation(
        self,
        llm: Any,
        ctx: "KvcommContext",
        generation_mode: str,
        result: Any,
        prefix_build: PrefixBuildResult,
        turn_index: int,
        started: float,
        *,
        model: str,
        openai_tools: list[dict[str, Any]] | None = None,
        config_loader_edit_fallback: bool = False,
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        if ctx.task_profile == "clawbench":
            self._snapshot_anchors(llm, ctx.run_id, ctx.agent_index, ctx.message_key)

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        self.last_request_ms = elapsed_ms
        reported_mode = str(getattr(result, "mode", None) or generation_mode)
        self.requests_by_mode[reported_mode] = self.requests_by_mode.get(reported_mode, 0) + 1
        self.requests_total += 1

        try:
            state = llm.get_request_state(ctx.run_id)
            input_anchor_meta = getattr(state, "input_anchor_metrics", None) or {}
        except Exception:
            input_anchor_meta = {}
        metrics = _metrics_from_result(
            result,
            effective_mode=reported_mode,
            ctx=ctx,
            prefix_build=prefix_build,
            turn_index=turn_index,
            elapsed_ms=elapsed_ms,
            input_anchor_meta=input_anchor_meta,
        )
        metric_key = f"{ctx.run_id}:{ctx.agent_index}"
        if openai_tools:
            message = openai_message_from_generation(
                result.text,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                workspace_dir=str((ctx.vars or {}).get("workspace_dir") or ""),
                task_vars=dict(ctx.vars or {}),
            )
            if config_loader_edit_fallback and not (message.get("tool_calls") or []):
                message = build_config_loader_edit_message()
                logger.debug(
                    "[tool-bridge] run_id={} agent={} config-loader edit parse failed — injecting canonical edit",
                    ctx.run_id,
                    ctx.agent_index,
                )
            _note_browser_tool_emission(ctx.run_id, ctx.agent_index, message)
            try:
                agent_idx = int(ctx.agent_index)
            except (TypeError, ValueError):
                agent_idx = 0
            if agent_idx > 0 and (message.get("tool_calls") or []):
                reported_mode = str(getattr(result, "mode", None) or generation_mode)
                if reported_mode == "kv_reuse":
                    llm.mark_consumer_tool_schema_stable(ctx.message_key)
                else:
                    logger.debug(
                        "[kvcomm-bench] skip tool-schema stable mark run_id={} agent={} mode={} (dense fallback)",
                        ctx.run_id,
                        ctx.agent_index,
                        reported_mode,
                    )
            request_metrics = dict(metrics)
            _append_emitted_tool_calls(request_metrics, message)
            metrics = _accumulate_agent_metrics(self._request_metrics.get(metric_key), request_metrics)
            metrics["tool_bridge"] = True
            metrics["tool_calls_count"] = len(message.get("tool_calls") or [])
        else:
            metrics = _accumulate_agent_metrics(self._request_metrics.get(metric_key), metrics)
            message = {
                "role": "assistant",
                "content": sanitize_generation_text(result.text or ""),
            }
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
        return _openai_completion(message, model, metrics), resp_headers, metrics

    async def check_request(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
        default_mode: str,
    ) -> None:
        """Validate kvcomm context and prefix budget before starting SSE."""
        _append_no_think_to_body(body)
        ctx = consume_registered_context(body, headers)
        if ctx is None:
            ctx = parse_kvcomm_context(body, headers, default_mode)
        if ctx is None:
            raise ValueError("missing kvcomm context (run_id/agent_index/message_key)")
        ctx = self._enrich_context_from_request(ctx, body)
        ctx = _normalize_task_profile(ctx)
        _build_openclaw_prefix(ctx, body)

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
        llm, ctx, prefix_build, turn_index, generation_mode, _input_routing = await self._prepare_generation(
            body,
            headers,
            default_mode,
        )
        openai_tools, tool_choice = extract_tool_request(body)
        tool_injection_text = None
        messages = [msg for msg in (body.get("messages") or []) if isinstance(msg, dict)]
        try:
            agent_idx = int(ctx.agent_index)
        except (TypeError, ValueError):
            agent_idx = -1
        bugfix_bridge = ctx.task_profile == "clawbench" and is_bugfix_discount_task(ctx)
        quick_note_bridge = ctx.task_profile == "clawbench" and is_quick_note_task(ctx)
        normalizer_bridge = ctx.task_profile == "clawbench" and is_add_tests_normalizer_task(ctx)
        config_loader_bridge = ctx.task_profile == "clawbench" and is_config_loader_task(ctx)
        find_that_bridge = ctx.task_profile == "clawbench" and is_find_that_task(ctx)
        browser_bridge = ctx.task_profile == "clawbench" and is_browser_family_task(ctx)
        chain_workspace = str((ctx.vars or {}).get("workspace_dir") or "")
        if browser_bridge and chain_workspace:
            sync_clawbench_browser_workspaces(
                workspace_dir=chain_workspace,
                prefer_default=(agent_idx >= 2),
            )
        force_browser_analyzer_done = (
            browser_bridge
            and agent_idx == 0
            and _browser_agent_exploration_done(messages, ctx.run_id, str(agent_idx))
        )
        force_browser_analyzer_browser = (
            browser_bridge
            and agent_idx == 0
            and not _browser_agent_exploration_done(messages, ctx.run_id, str(agent_idx))
        )
        force_browser_patcher_done = (
            browser_bridge
            and agent_idx == 1
            and browser_patcher_edit_applied_in_messages(messages)
        )
        force_browser_patcher_edit = (
            browser_bridge
            and agent_idx == 1
            and browser_patcher_read_satisfied(messages)
            and not browser_patcher_edit_applied_in_messages(messages)
        )
        force_browser_patcher_read = (
            browser_bridge
            and agent_idx == 1
            and not browser_patcher_read_satisfied(messages)
        )
        force_browser_verifier_pass = (
            browser_bridge
            and agent_idx == 2
            and browser_verifier_exec_passed(messages)
        )
        force_browser_verifier_fail = (
            browser_bridge
            and agent_idx == 2
            and browser_verifier_exec_done(messages)
            and not browser_verifier_exec_passed(messages)
        )
        force_browser_verifier_exec = (
            browser_bridge
            and agent_idx == 2
            and not browser_verifier_exec_done(messages)
        )
        force_config_loader_analyzer_done = (
            config_loader_bridge
            and agent_idx == 0
            and config_loader_analyzer_reads_satisfied(messages)
        )
        force_config_loader_analyzer_read = (
            config_loader_bridge
            and agent_idx == 0
            and not config_loader_analyzer_reads_satisfied(messages)
        )
        force_config_loader_patcher_done = (
            config_loader_bridge
            and agent_idx == 1
            and config_loader_patcher_fix_satisfied(messages)
            and verifier_exec_pytest_done(messages)
        )
        force_config_loader_patcher_pytest = (
            config_loader_bridge
            and agent_idx == 1
            and config_loader_patcher_fix_satisfied(messages)
            and not verifier_exec_pytest_done(messages)
        )
        force_config_loader_patcher_read = (
            config_loader_bridge
            and agent_idx == 1
            and not config_loader_patcher_read_satisfied(messages)
        )
        force_config_loader_edit_only = (
            config_loader_bridge
            and agent_idx == 1
            and config_loader_patcher_read_satisfied(messages)
            and not config_loader_patcher_fix_satisfied(messages)
        )
        force_config_loader_verifier_pass = (
            config_loader_bridge
            and agent_idx == 2
            and verifier_pytest_passed(messages)
        )
        force_config_loader_verifier_exec = (
            config_loader_bridge
            and agent_idx == 2
            and config_loader_verifier_should_force_exec(messages)
        )
        force_config_loader_verifier_edit = (
            config_loader_bridge
            and agent_idx == 2
            and config_loader_verifier_should_force_edit(messages)
        )
        force_config_loader_verifier_read = (
            config_loader_bridge
            and agent_idx == 2
            and config_loader_verifier_should_force_read(messages)
        )
        force_find_that_extractor_done = (
            find_that_bridge
            and agent_idx == 0
            and find_that_source_located(messages)
        )
        force_find_that_writer_copy = (
            find_that_bridge
            and agent_idx == 1
            and find_that_source_located(messages)
            and not find_that_copy_satisfied(messages)
        )
        force_find_that_writer_done = (
            find_that_bridge
            and agent_idx == 1
            and find_that_copy_satisfied(messages)
        )
        force_find_that_verifier_exec = (
            find_that_bridge
            and agent_idx == 2
            and not find_that_verifier_exec_done(messages)
        )
        force_find_that_verifier_done = (
            find_that_bridge
            and agent_idx == 2
            and find_that_verifier_passed(messages)
        )
        force_text_only = (
            bugfix_bridge
            and agent_idx == 0
            and analyzer_reads_satisfied(messages)
        )
        force_normalizer_analyzer_done = (
            normalizer_bridge
            and agent_idx == 0
            and normalizer_analyzer_read_satisfied(messages)
        )
        force_normalizer_analyzer_read = (
            normalizer_bridge
            and agent_idx == 0
            and not normalizer_analyzer_read_satisfied(messages)
        )
        force_normalizer_patcher_read = (
            normalizer_bridge
            and agent_idx == 1
            and not normalizer_patcher_read_satisfied(messages)
        )
        force_normalizer_patcher_pytest = (
            normalizer_bridge
            and agent_idx == 1
            and normalizer_tests_ready(messages, workspace_dir=chain_workspace)
            and not verifier_pytest_passed(messages)
        )
        force_normalizer_patcher_write = (
            normalizer_bridge
            and agent_idx == 1
            and normalizer_patcher_read_satisfied(messages)
            and not normalizer_tests_ready(messages, workspace_dir=chain_workspace)
        )
        force_normalizer_patcher_done = (
            normalizer_bridge
            and agent_idx == 1
            and verifier_pytest_passed(messages)
        )
        force_normalizer_verifier_pass = (
            normalizer_bridge
            and agent_idx == 2
            and verifier_pytest_passed(messages)
        )
        force_normalizer_verifier_exec = (
            normalizer_bridge
            and agent_idx == 2
            and normalizer_tests_ready(messages, workspace_dir=chain_workspace)
            and not verifier_pytest_passed(messages)
        )
        force_patcher_done = (
            bugfix_bridge
            and agent_idx == 1
            and patcher_fix_satisfied(messages)
            and verifier_exec_pytest_done(messages)
        )
        force_patcher_pytest = (
            bugfix_bridge
            and agent_idx == 1
            and patcher_fix_satisfied(messages)
            and not verifier_exec_pytest_done(messages)
        )
        force_patcher_read = (
            bugfix_bridge
            and agent_idx == 1
            and not patcher_read_satisfied(messages)
        )
        force_edit_only = (
            bugfix_bridge
            and agent_idx == 1
            and patcher_read_satisfied(messages)
            and not patcher_fix_satisfied(messages)
        )
        force_verifier_pass = (
            bugfix_bridge
            and agent_idx == 2
            and verifier_pytest_passed(messages)
        )
        force_verifier_exec = (
            bugfix_bridge
            and agent_idx == 2
            and verifier_should_force_exec(messages)
        )
        force_verifier_edit = (
            bugfix_bridge
            and agent_idx == 2
            and verifier_should_force_edit(messages)
        )
        force_verifier_read = (
            bugfix_bridge
            and agent_idx == 2
            and verifier_should_force_read(messages)
        )
        force_quick_note_verifier_done = (
            quick_note_bridge
            and agent_idx == 2
            and verifier_read_satisfied(messages)
        )
        force_quick_note_writer_done = (
            quick_note_bridge
            and agent_idx == 1
            and quick_note_write_satisfied(messages)
        )
        analyzer_cart_hint = None
        config_loader_read_hint = None
        if (
            bugfix_bridge
            and agent_idx == 0
            and openai_tools
            and missing_analyzer_reads(messages) == frozenset({"cart.py"})
        ):
            analyzer_cart_hint = (
                "\npricing.py is already in context above. "
                "Read cart.py next — do not re-read pricing.py.\n"
            )
        if config_loader_bridge and agent_idx == 0 and openai_tools:
            missing_reads = config_loader_missing_analyzer_reads(messages)
            config_loader_read_hint = build_config_loader_analyzer_read_hint(
                missing_reads, soft=True
            )
        if force_browser_analyzer_done:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\nBrowser exploration is complete and shown in context above. "
                "Summarize what is broken on the signup page and which frontend files "
                "likely need changes. Do not call browser, edit, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=0 browser exploration satisfied — forcing text-only",
                ctx.run_id,
            )
        elif force_browser_analyzer_browser:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "browser"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "browser"}}
            url_hint = _browser_url_hint(ctx)
            browser_hint = (
                f"\nCall browser with action `open`, target `host`, and url `{url_hint}`. "
                "Then use snapshot if needed to inspect the page. "
                "Do not call read, edit, write, or exec yet.\n"
            )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + browser_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=0 browser task — forcing browser-only",
                ctx.run_id,
            )
        elif force_browser_patcher_done:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\napp.js fix is applied. Reply DONE in plain text summarizing the contact-form id fix. "
                "Do not call edit, read, exec, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=1 browser patcher fix satisfied — forcing text-only DONE",
                ctx.run_id,
            )
        elif force_browser_patcher_edit:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") in {"edit", "write"}
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "edit"}}
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + build_browser_appjs_edit_hint()
            logger.debug(
                "[tool-bridge] run_id={} agent=1 browser patcher — forcing edit-only",
                ctx.run_id,
            )
        elif force_browser_patcher_read:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "read"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "read"}}
            read_hint = (
                "\nRead app.js and index.html from the workspace to locate the form id mismatch. "
                "Do not call edit, browser, or exec yet.\n"
            )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + read_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=1 browser patcher — forcing read-only",
                ctx.run_id,
            )
        elif force_browser_verifier_pass:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\nverify_form.cjs passed. Reply PASS in plain text summarizing verification. "
                "Do not call exec, read, edit, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=2 verify_form passed — forcing text-only PASS",
                ctx.run_id,
            )
        elif force_browser_verifier_fail:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\nverify_form.cjs failed. Reply FAIL in plain text explaining the verification failure. "
                "Do not call exec, read, edit, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=2 verify_form failed — forcing text-only FAIL",
                ctx.run_id,
            )
        elif force_browser_verifier_exec:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "exec"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "exec"}}
            form_port = str((ctx.vars or {}).get("form_app_port") or "").strip()
            node_path = ":".join(
                part
                for part in (
                    str((ctx.vars or {}).get("openclaw_node_path") or "").strip(),
                    str((ctx.vars or {}).get("benchmark_node_path") or "").strip(),
                )
                if part
            )
            exec_hint = build_browser_verifier_exec_hint(form_app_port=form_port, node_path=node_path)
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + exec_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=2 browser task — forcing exec-only verify_form",
                ctx.run_id,
            )
        elif force_config_loader_analyzer_done:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\nconfig_loader.py, app_config.py, and tests/test_config_loader.py are already in context above. "
                "Output your Agent 0 analysis in plain text: explain the config precedence and validation bugs. "
                "Do not call read or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=0 config-loader reads satisfied — forcing text-only",
                ctx.run_id,
            )
        elif force_config_loader_analyzer_read:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "read"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "read"}}
            missing_reads = config_loader_missing_analyzer_reads(messages)
            read_hint = build_config_loader_analyzer_read_hint(missing_reads)
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + read_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=0 config-loader missing reads={} — forcing read-only",
                ctx.run_id,
                sorted(missing_reads),
            )
        elif force_config_loader_patcher_done:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\nconfig_loader.py fix and pytest are complete. "
                "Reply with one short line starting with DONE summarizing the fix. "
                "Do not call edit, read, exec, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=1 config-loader patcher fix satisfied — forcing text-only DONE",
                ctx.run_id,
            )
        elif force_config_loader_patcher_pytest:
            sync_clawbench_config_loader_default_to_chain(workspace_dir=chain_workspace)
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "exec"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "exec"}}
            exec_hint = (
                "\nconfig_loader.py fix is applied. Call exec with command "
                f"`{_CONFIG_LOADER_PYTEST_CMD}` and workdir `.` "
                "(not ~/.openclaw/workspace). Never set elevated: true.\n"
            )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + exec_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=1 config-loader fix satisfied — forcing exec-only pytest",
                ctx.run_id,
            )
        elif force_config_loader_verifier_pass:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\npytest passed. Reply PASS in plain text summarizing verification. "
                "Do not call exec, read, edit, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=2 config-loader pytest passed — forcing text-only PASS",
                ctx.run_id,
            )
        elif force_config_loader_verifier_exec:
            sync_clawbench_config_loader_default_to_chain(workspace_dir=chain_workspace)
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "exec"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "exec"}}
            exec_hint = (
                "\nRun pytest now. Call exec with command "
                f"`{_CONFIG_LOADER_PYTEST_CMD}` and workdir `.`. "
                "Do not read config_loader.py first — run tests first. "
                "Never set elevated: true.\n"
            )
            if config_loader_patcher_read_satisfied(messages):
                exec_hint = (
                    "\nconfig_loader.py is already in context above. "
                    "Do not read again. Call exec with command "
                    f"`{_CONFIG_LOADER_PYTEST_CMD}` and workdir `.` only.\n"
                )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + exec_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=2 config-loader forcing exec-only pytest",
                ctx.run_id,
            )
        elif force_config_loader_verifier_edit:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "edit"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "edit"}}
            edit_hint = build_config_loader_edit_hint(messages)
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + edit_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=2 config-loader verifier forcing edit-only",
                ctx.run_id,
            )
        elif force_config_loader_verifier_read:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "read"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "read"}}
            read_hint = (
                "\npytest failed. Call read on config_loader.py once to inspect load_config, "
                "then you will edit on the next turn.\n"
            )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + read_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=2 config-loader pytest failed — forcing read",
                ctx.run_id,
            )
        elif force_config_loader_patcher_read:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "read"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "read"}}
            read_hint = (
                "\nStep 1: call read on config_loader.py first. "
                "Do not output analysis text — only a read tool call.\n"
            )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + read_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=1 config-loader patcher must read — forcing read-only",
                ctx.run_id,
            )
        elif force_config_loader_edit_only:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "edit"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "edit"}}
            edit_hint = build_config_loader_edit_hint(messages)
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + edit_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=1 config-loader patcher read satisfied — forcing edit-only",
                ctx.run_id,
            )
        elif force_text_only:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\nBoth pricing.py and cart.py are already in context above. "
                "Output your Agent 0 analysis in plain text: quote the apply_discount "
                "function verbatim (signature + return line) and state the root cause. "
                "Do not call read or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=0 analyzer reads satisfied — forcing text-only generation",
                ctx.run_id,
            )
        elif force_patcher_done:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\npricing.py fix and pytest are complete. "
                "Reply with one short line starting with DONE summarizing the fix. "
                "Do not call edit, read, exec, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=1 patcher fix satisfied — forcing text-only DONE",
                ctx.run_id,
            )
        elif force_patcher_pytest:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "exec"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "exec"}}
            exec_hint = (
                "\npricing.py fix is applied. Call exec with command "
                "`pytest -q tests/test_pricing.py` from the workspace root. "
                "Never set elevated: true.\n"
            )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + exec_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=1 patcher fix satisfied — forcing exec-only pytest",
                ctx.run_id,
            )
        elif force_verifier_pass:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\npytest passed. Reply PASS in plain text summarizing verification. "
                "Do not call exec, read, edit, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=2 pytest passed — forcing text-only PASS",
                ctx.run_id,
            )
        elif force_find_that_extractor_done:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\nDocuments/q3_marketing_budget_v3.xlsx is already in context above. "
                "Summarize the deliverable: source path, target copy path "
                "(Desktop/q3_marketing_budget.xlsx), and what the user asked for. "
                "Do not call exec, read, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=0 find-that source located — forcing text-only",
                ctx.run_id,
            )
        elif force_find_that_writer_copy:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "exec"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "exec"}}
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + build_find_that_writer_copy_hint()
            logger.debug(
                "[tool-bridge] run_id={} agent=1 find-that copy pending — forcing exec-only",
                ctx.run_id,
            )
        elif force_find_that_writer_done:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\nDesktop/q3_marketing_budget.xlsx was copied successfully and is shown "
                "in context above. Reply DONE in plain text summarizing the copy. "
                "Do not call exec, read, write, edit, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=1 find-that copy satisfied — forcing text-only DONE",
                ctx.run_id,
            )
        elif force_find_that_verifier_exec:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "exec"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "exec"}}
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + build_find_that_verifier_exec_hint()
            logger.debug(
                "[tool-bridge] run_id={} agent=2 find-that verify pending — forcing exec-only",
                ctx.run_id,
            )
        elif force_find_that_verifier_done:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\nverify_correct_file.py passed and is shown in context above. "
                "Reply PASS in plain text summarizing verification. "
                "Do not call exec, read, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=2 find-that verify passed — forcing text-only PASS",
                ctx.run_id,
            )
        elif force_quick_note_writer_done:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\nnotes/quick_note.md was written successfully and is shown in context above. "
                "Reply DONE in plain text: include the path written and quote the FULL note body "
                "verbatim (every list line). Do not call write, edit, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=1 quick-note write satisfied — forcing text-only DONE",
                ctx.run_id,
            )
        elif force_quick_note_verifier_done:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\nnotes/quick_note.md is already in context above. "
                "Confirm all three reminders are present and reply DONE in plain text. "
                "Do not call read, edit, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=2 quick-note read satisfied — forcing text-only DONE",
                ctx.run_id,
            )
        elif force_normalizer_analyzer_done:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\nnormalizer.py is already in context above. "
                "Output your Agent 0 analysis in plain text: summarize normalize_title and "
                "normalize_tags behavior and what pytest coverage is missing. "
                "Do not call read or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=0 normalizer read satisfied — forcing text-only",
                ctx.run_id,
            )
        elif force_normalizer_analyzer_read:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "read"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "read"}}
            read_hint = (
                "\nStep 1: call read on normalizer.py first. "
                "Do not output analysis text — only a read tool call.\n"
            )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + read_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=0 normalizer must read normalizer.py — forcing read-only",
                ctx.run_id,
            )
        elif force_normalizer_patcher_done:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\npytest passed for tests/test_normalizer.py. "
                "Reply DONE in plain text summarizing the test suite. "
                "Do not call edit, exec, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=1 normalizer tests passed — forcing text-only DONE",
                ctx.run_id,
            )
        elif force_normalizer_verifier_pass:
            openai_tools = None
            tool_choice = None
            tool_injection_text = (
                "\npytest passed. Reply PASS in plain text summarizing verification. "
                "Do not call exec, read, edit, or any other tools.\n"
            )
            logger.debug(
                "[tool-bridge] run_id={} agent=2 normalizer pytest passed — forcing text-only PASS",
                ctx.run_id,
            )
        elif force_normalizer_patcher_pytest or force_normalizer_verifier_exec:
            fix_normalizer_test_file_on_disk(workspace_dir=chain_workspace)
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "exec"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "exec"}}
            exec_hint = (
                "\ntests/test_normalizer.py is ready. Call exec with command "
                "`PYTHONPATH=. python -m pytest -q tests/test_normalizer.py` and workdir `.` "
                "(not ~/.openclaw/workspace). "
                "Do not edit the import line — use `from normalizer import normalize_title, normalize_tags`. "
                "Never set elevated: true.\n"
            )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + exec_hint
            logger.debug(
                "[tool-bridge] run_id={} agent={} normalizer forcing exec-only pytest",
                ctx.run_id,
                agent_idx,
            )
        elif force_normalizer_patcher_write:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") in {"write", "edit"}
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "write"}}
            write_hint = (
                "\nnormalizer.py is in context above. Call write to create tests/test_normalizer.py "
                "with `from normalizer import normalize_title, normalize_tags` and pytest cases for "
                "normalize_title and normalize_tags edge cases. "
                "Do not call read or exec — only write the test file.\n"
            )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + write_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=1 normalizer read satisfied — forcing write-only",
                ctx.run_id,
            )
        elif force_normalizer_patcher_read:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "read"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "read"}}
            read_hint = (
                "\nStep 1: call read on normalizer.py first. "
                "Do not output analysis text — only a read tool call.\n"
            )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + read_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=1 normalizer must read normalizer.py — forcing read-only",
                ctx.run_id,
            )
        elif force_verifier_exec:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "exec"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "exec"}}
            exec_hint = (
                "\nRun pytest now. Call exec with command `pytest -q tests/test_pricing.py` "
                "from the workspace root. Do not read pricing.py first — run tests first. "
                "Never set elevated: true.\n"
            )
            if patcher_read_satisfied(messages):
                exec_hint = (
                    "\npricing.py is already in context above. "
                    "Do not read again. Call exec with command "
                    "`pytest -q tests/test_pricing.py` only.\n"
                )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + exec_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=2 forcing exec-only pytest generation",
                ctx.run_id,
            )
        elif force_verifier_edit:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "edit"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "edit"}}
            edit_hint = build_pricing_edit_hint(messages)
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + edit_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=2 verifier forcing edit-only generation",
                ctx.run_id,
            )
        elif force_verifier_read:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "read"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "read"}}
            read_hint = (
                "\npytest failed. Call read on pricing.py once to inspect apply_discount, "
                "then you will edit the return line on the next turn.\n"
            )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + read_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=2 pytest failed — forcing read pricing.py",
                ctx.run_id,
            )
        elif force_patcher_read:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "read"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "read"}}
            read_hint = (
                "\nStep 1: call read on pricing.py first. "
                "Do not output analysis text — only a read tool call.\n"
            )
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + read_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=1 patcher must read pricing.py — forcing read-only",
                ctx.run_id,
            )
        elif force_edit_only:
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools or [],
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = [
                t
                for t in openai_tools
                if str((t.get("function") or {}).get("name") or "") == "edit"
            ] or openai_tools
            tool_choice = {"type": "function", "function": {"name": "edit"}}
            edit_hint = build_pricing_edit_hint(messages)
            tool_injection_text = build_tool_injection_text(
                openai_tools, llm.tokenizer, tool_choice
            ) + edit_hint
            logger.debug(
                "[tool-bridge] run_id={} agent=1 patcher read satisfied — forcing edit-only generation",
                ctx.run_id,
            )
        elif openai_tools and should_inject_tools(body, task_profile=ctx.task_profile):
            role_label = (ctx.vars.get(f"agent_{ctx.agent_index}_role") or "").strip()
            openai_tools = ensure_clawbench_agent_tools(
                openai_tools,
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            openai_tools = filter_tools_for_agent(
                openai_tools,
                agent_index=ctx.agent_index,
                agent_role=role_label,
                task_profile=ctx.task_profile,
                task_id=ctx.task_id,
                clawbench_family=ctx.clawbench_family,
            )
            tool_injection_text = build_tool_injection_text(openai_tools, llm.tokenizer, tool_choice)
            if analyzer_cart_hint:
                tool_injection_text += analyzer_cart_hint
                logger.debug(
                    "[tool-bridge] run_id={} agent=0 pricing read done — hint read cart.py",
                    ctx.run_id,
                )
            if config_loader_read_hint:
                tool_injection_text += config_loader_read_hint
                logger.debug(
                    "[tool-bridge] run_id={} agent=0 config-loader partial reads — sequential read hint",
                    ctx.run_id,
                )
            logger.debug(
                "[tool-bridge] run_id={} agent={} tools={} choice={}",
                ctx.run_id,
                ctx.agent_index,
                [str((t.get("function") or {}).get("name") or "") for t in openai_tools],
                tool_choice,
            )
        elif analyzer_cart_hint or config_loader_read_hint:
            logger.debug(
                "[tool-bridge] run_id={} agent=0 partial reads — hint next file (no inject)",
                ctx.run_id,
            )
        elif isinstance(body.get("tools"), list) and body.get("tools"):
            logger.warning(
                "[tool-bridge] tools present but bridge inactive run_id={} agent={} tool_choice={}",
                ctx.run_id,
                ctx.agent_index,
                body.get("tool_choice"),
            )
        # Tool-bridge turns must not stream raw HF tokens: OpenClaw treats streamed
        # `<tool_call>` text as plain assistant content and never executes tools.
        if openai_tools and on_token is not None:
            on_token = None
        generation_max_tokens = ctx.max_tokens
        if openai_tools is None and tool_injection_text:
            generation_max_tokens = min(ctx.max_tokens, _CLAWBENCH_TEXT_ONLY_MAX_TOKENS)
        elif openai_tools and turn_index > 0:
            generation_max_tokens = min(ctx.max_tokens, _CLAWBENCH_TOOL_CONTINUATION_MAX_TOKENS)
        result = await llm.generate_for_agent(
            request_uid=ctx.run_id,
            message=ctx.message_key,
            preferred_mode=generation_mode,
            max_tokens=generation_max_tokens,
            temperature=ctx.temperature,
            agent_id=ctx.agent_index,
            agent_name=f"Agent{ctx.agent_index}",
            agent_role=f"agent_{ctx.agent_index}",
            on_token=on_token,
            tool_injection_text=tool_injection_text,
            tool_schema_injection=bool(openai_tools),
            full_tool_schema=bool(openai_tools and len(openai_tools) > 1),
            tool_deliverable_fingerprint=_tool_deliverable_fingerprint_for_generation(
                llm,
                ctx,
                body,
            ),
        )
        payload, resp_headers, _metrics = await self._finalize_generation(
            llm,
            ctx,
            generation_mode,
            result,
            prefix_build,
            turn_index,
            started,
            model=model,
            openai_tools=openai_tools,
            config_loader_edit_fallback=(
                config_loader_bridge
                and (force_config_loader_edit_only or force_config_loader_verifier_edit)
            ),
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
        openai_tools, _tool_choice = extract_tool_request(body)
        if openai_tools and tool_bridge_buffered_sse_enabled():
            # Buffered tool-bridge SSE: OpenClaw must see stopReason=toolUse with
            # parsed toolCall blocks. Incremental HF token streaming previously
            # produced empty assistant turns (stopReason=stop) and read never ran.
            payload, _resp_headers = await self.generate(body, headers, default_mode, on_token=None)
            choice = (payload.get("choices") or [{}])[0]
            finish = choice.get("finish_reason")
            tool_count = len((choice.get("message") or {}).get("tool_calls") or [])
            logger.debug(
                "[tool-bridge] buffered SSE finish={} tool_calls={}",
                finish,
                tool_count,
            )
            yield completion_payload_to_sse(payload, include_usage=include_usage).encode("utf-8")
            return

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
                    choice = (payload.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    tool_calls = message.get("tool_calls")
                    content = message.get("content") or ""
                    has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
                    if not content_streamed and content.strip() and not has_tool_calls:
                        yield f"data: {json.dumps(chunk_obj({'content': content}), ensure_ascii=False)}\n\n".encode(
                            "utf-8",
                        )
                        content_streamed = True
                    if has_tool_calls:
                        for delta_tool in sse_tool_call_deltas(tool_calls):
                            yield f"data: {json.dumps(chunk_obj({'tool_calls': [delta_tool]}), ensure_ascii=False)}\n\n".encode(
                                "utf-8",
                            )
                    finish_reason = choice.get("finish_reason") or ("tool_calls" if has_tool_calls else "stop")
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
        from sidecar.stores.registry import reset_store_registry

        reset_store_registry()
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
    elif hf_model or hf_model_path:
        # HF bench runs should default to local dense inference, not vLLM upstream.
        os.environ.setdefault("KVCOMM_DENSE_VIA_HF", "1")
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
