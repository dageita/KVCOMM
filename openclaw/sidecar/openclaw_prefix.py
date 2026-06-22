"""Parse OpenClaw chat/completions messages into KVCOMM prefix templates (A+E)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from sidecar.tool_bridge import sanitize_chat_template_leaks

KVCOMM_META_RE = re.compile(r"<!--KVCOMM_META:(\{.*?\})-->", re.DOTALL)
_TOOL_SCHEMA_BLOCK_RE = re.compile(
    r"(?:^|\n)(?:#+\s*)?(?:Available tools|Tool definitions|tools?\s*:\s*\[).*$",
    re.IGNORECASE | re.DOTALL,
)
_JSON_TOOL_ARRAY_RE = re.compile(r"\[\s*\{[^\]]*\"type\"\s*:\s*\"function\"", re.DOTALL)
_OPENCLAW_BOOTSTRAP_RE = re.compile(
    r"<!--\s*openclaw[^>]*-->.*?<!--\s*/openclaw[^>]*-->",
    re.IGNORECASE | re.DOTALL,
)

DEFAULT_PREFIX_MAX_TOKENS = 4096
DEFAULT_TOOL_RESULT_MAX_CHARS = 2000


class PrefixOverflowError(RuntimeError):
    """Raised when parsed prefix exceeds the configured token budget."""


@dataclass
class PrefixBuildResult:
    system_prompt: str
    user_template: str
    static_text: str
    placeholders: list[str] = field(default_factory=list)
    turn_count: int = 0
    turn_content: dict[str, str] = field(default_factory=dict)
    estimated_tokens: int = 0
    use_openclaw: bool = True
    fallback_reason: str | None = None


def prefix_max_tokens() -> int:
    raw = os.environ.get("KVCOMM_PREFIX_MAX_TOKENS", "").strip()
    if not raw:
        return DEFAULT_PREFIX_MAX_TOKENS
    try:
        return max(256, int(raw))
    except ValueError:
        return DEFAULT_PREFIX_MAX_TOKENS


def tool_result_max_chars() -> int:
    raw = os.environ.get("KVCOMM_TOOL_RESULT_MAX_CHARS", "").strip()
    if not raw:
        return DEFAULT_TOOL_RESULT_MAX_CHARS
    try:
        return max(128, int(raw))
    except ValueError:
        return DEFAULT_TOOL_RESULT_MAX_CHARS


def use_openclaw_prefix(task_profile: str) -> bool:
    if task_profile != "clawbench":
        return False
    raw = os.environ.get("KVCOMM_USE_OPENCLAW_PREFIX", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _message_content(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_result":
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "\n".join(parts)
    return ""


def _strip_kvcomm_meta(text: str) -> str:
    match = KVCOMM_META_RE.search(text or "")
    if not match:
        return (text or "").strip()
    return (text[: match.start()] + text[match.end() :]).strip()


def _strip_tool_schema(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    cleaned = _OPENCLAW_BOOTSTRAP_RE.sub("", cleaned).strip()
    cleaned = _TOOL_SCHEMA_BLOCK_RE.sub("", cleaned).strip()
    if _JSON_TOOL_ARRAY_RE.search(cleaned):
        idx = _JSON_TOOL_ARRAY_RE.search(cleaned).start()
        head = cleaned[:idx].strip()
        if head:
            cleaned = head
    lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped.startswith('{"type":') and "function" in stripped:
            continue
        if stripped.startswith("[{") and "function" in stripped:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _assistant_text(msg: dict[str, Any]) -> str:
    text = _message_content(msg).strip()
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        call_summaries: list[str] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(fn.get("name") or call.get("name") or "tool")
            args = fn.get("arguments") or call.get("arguments") or ""
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            call_summaries.append(f"[tool_call {name}: {str(args)[:200]}]")
        if text:
            return sanitize_chat_template_leaks(f"{text}\n" + "\n".join(call_summaries))
        return sanitize_chat_template_leaks("\n".join(call_summaries))
    return sanitize_chat_template_leaks(text)


def _tool_text(msg: dict[str, Any]) -> str:
    name = str(msg.get("name") or msg.get("tool_call_id") or "tool")
    body = sanitize_chat_template_leaks(_message_content(msg).strip())
    max_chars = tool_result_max_chars()
    if len(body) > max_chars:
        body = body[:max_chars] + "\n...[truncated]"
    return f"[{name}]\n{body}" if body else f"[{name}]"


def count_assistant_turns(messages: list[dict[str, Any]]) -> int:
    return sum(1 for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant")


def _extract_turn_pairs(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    seen_first_user = False
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict):
            i += 1
            continue
        role = str(msg.get("role") or "")
        if role == "user":
            if not seen_first_user:
                seen_first_user = True
                i += 1
                continue
        if role != "assistant":
            i += 1
            continue
        assistant = _assistant_text(msg)
        tool_parts: list[str] = []
        j = i + 1
        while j < len(messages):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            tool_parts.append(_tool_text(nxt))
            j += 1
        turns.append(
            {
                "assistant": assistant[: tool_result_max_chars()],
                "tool": "\n".join(tool_parts)[: tool_result_max_chars()],
            }
        )
        i = j
    return turns


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _strip_kvcomm_meta(_message_content(msg))
    return ""


def _first_system_text(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            return _strip_tool_schema(_message_content(msg))
    return ""


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 3)


def _collect_placeholders(user_template: str) -> list[str]:
    pattern = re.compile(
        r"\{((?:agent|condition)_\w+_(?:current|history)|user_question|turn_\d+_(?:assistant|tool))\}"
    )
    return [m.group(1) for m in pattern.finditer(user_template)]


def _static_without_turn_placeholders(user_template: str) -> str:
    return re.sub(r"\{turn_\d+_(?:assistant|tool)\}", "", user_template).strip()


def static_without_turn_placeholders(user_template: str) -> str:
    """Public helper: user_template with turn_N placeholders removed (static segment A)."""
    return _static_without_turn_placeholders(user_template)


_RUN_WORKSPACE_PATH_RE = re.compile(
    r"(?:at\s+)?/[\S]*/run-[a-f0-9]{6,}/notes/quick_note\.md",
    re.IGNORECASE,
)
_RUN_UID_SEGMENT_RE = re.compile(r"/run-[a-f0-9]{6,}\b", re.IGNORECASE)


def normalize_run_specific_paths(text: str) -> str:
    """Canonicalize per-measure workspace paths so prefix KV is stable across runs."""
    if not text:
        return text
    cleaned = _RUN_WORKSPACE_PATH_RE.sub("at notes/quick_note.md", text)
    cleaned = _RUN_UID_SEGMENT_RE.sub("", cleaned)
    return cleaned


def _align_static_user(parsed_user: str, bench_user_prompt: str) -> tuple[str, str | None]:
    bench = normalize_run_specific_paths((bench_user_prompt or "").strip())
    parsed = normalize_run_specific_paths((parsed_user or "").strip())
    if bench and parsed:
        bench_norm = re.sub(r"\s+", " ", bench)
        parsed_norm = re.sub(r"\s+", " ", parsed)
        if bench_norm == parsed_norm:
            return bench, None
        if bench_norm in parsed_norm or parsed_norm in bench_norm:
            return bench, None
        bench_tokens = set(bench_norm.split())
        parsed_tokens = set(parsed_norm.split())
        overlap = len(bench_tokens & parsed_tokens) / max(1, len(bench_tokens))
        if overlap < 0.5:
            return bench, "static_user_mismatch"
    if bench:
        return bench, None
    return parsed, None if parsed else "empty_user"


def build_prefix_from_openclaw_messages(
    messages: list[dict[str, Any]],
    *,
    bench_user_prompt: str,
    clawbench_role: str,
    task_profile: str = "clawbench",
) -> PrefixBuildResult:
    """Build prefix template from OpenClaw messages with static (A) + turn (E) segments."""
    if not use_openclaw_prefix(task_profile):
        static = (bench_user_prompt or "").strip()
        return PrefixBuildResult(
            system_prompt=clawbench_role,
            user_template=static,
            static_text=static,
            placeholders=_collect_placeholders(static),
            turn_count=0,
            use_openclaw=False,
            fallback_reason="disabled",
        )

    system_prompt = clawbench_role

    parsed_user = normalize_run_specific_paths(_first_user_text(messages))
    static_user, warn = _align_static_user(parsed_user, bench_user_prompt)
    if not static_user:
        static_user = (bench_user_prompt or parsed_user).strip()

    turns = _extract_turn_pairs(messages)
    turn_suffix = ""
    turn_content: dict[str, str] = {}
    for idx, turn in enumerate(turns, start=1):
        turn_suffix += f"\n\n{{turn_{idx}_assistant}}\n\n{{turn_{idx}_tool}}\n"
        turn_content[f"turn_{idx}_assistant"] = turn["assistant"].strip() or " "
        turn_content[f"turn_{idx}_tool"] = turn["tool"].strip() or " "

    user_template = f"{static_user.rstrip()}{turn_suffix}".strip()
    placeholders = _collect_placeholders(user_template)
    static_text = _static_without_turn_placeholders(user_template)

    estimated = _estimate_tokens(system_prompt) + _estimate_tokens(user_template)
    for content in turn_content.values():
        estimated += _estimate_tokens(content)

    max_tokens = prefix_max_tokens()
    if estimated > max_tokens:
        raise PrefixOverflowError(
            f"prefix estimated_tokens={estimated} exceeds KVCOMM_PREFIX_MAX_TOKENS={max_tokens}"
        )

    return PrefixBuildResult(
        system_prompt=system_prompt,
        user_template=user_template,
        static_text=static_text,
        placeholders=placeholders,
        turn_count=len(turns),
        turn_content=turn_content,
        estimated_tokens=estimated,
        use_openclaw=True,
        fallback_reason=warn,
    )
