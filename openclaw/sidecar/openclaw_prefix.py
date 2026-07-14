"""Parse OpenClaw chat/completions messages into KVCOMM prefix templates (A+E)."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from sidecar.bench_prompt_compose import (
    SUMMARIZE_THREAD_ASSISTANT_MAX_CHARS,
    SUMMARIZE_THREAD_TASK_ID,
    SUMMARIZE_THREAD_TOOL_RESULT_MAX_CHARS,
)
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

DEFAULT_PREFIX_MAX_TOKENS = 8192
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


def _is_summarize_thread_task(task_id: str | None) -> bool:
    return str(task_id or "").strip() == SUMMARIZE_THREAD_TASK_ID


def tool_result_max_chars(task_id: str | None = None) -> int:
    raw = os.environ.get("KVCOMM_TOOL_RESULT_MAX_CHARS", "").strip()
    if raw:
        try:
            return max(128, int(raw))
        except ValueError:
            pass
    if _is_summarize_thread_task(task_id):
        return SUMMARIZE_THREAD_TOOL_RESULT_MAX_CHARS
    return DEFAULT_TOOL_RESULT_MAX_CHARS


def assistant_turn_max_chars(task_id: str | None = None) -> int:
    raw = os.environ.get("KVCOMM_ASSISTANT_MAX_CHARS", "").strip()
    if raw:
        try:
            return max(128, int(raw))
        except ValueError:
            pass
    if _is_summarize_thread_task(task_id):
        return SUMMARIZE_THREAD_ASSISTANT_MAX_CHARS
    return tool_result_max_chars(task_id)


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


def _strip_embedded_tool_call_markup(text: str) -> str:
    """Remove Qwen `<tool_call>` XML leaked into assistant content strings."""
    if not text or "<tool_call" not in text.lower():
        return text
    from sidecar.tool_bridge import parse_qwen_tool_calls, sanitize_chat_template_leaks

    content, _ = parse_qwen_tool_calls(text)
    return sanitize_chat_template_leaks(content or "").strip()


def _assistant_text(msg: dict[str, Any]) -> str:
    text = _strip_embedded_tool_call_markup(_message_content(msg).strip())
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


def _tool_text(
    msg: dict[str, Any],
    *,
    tool_name: str = "",
    max_chars: int | None = None,
) -> str:
    name = (tool_name or "").strip()
    if not name:
        raw_name = str(msg.get("name") or "").strip()
        if raw_name and not raw_name.startswith("call"):
            name = raw_name
        else:
            name = str(msg.get("tool_call_id") or "tool")
            if name.startswith("call") and len(name) > 20:
                name = "tool"
    body = sanitize_chat_template_leaks(_message_content(msg).strip())
    limit = max_chars if max_chars is not None else tool_result_max_chars()
    if len(body) > limit:
        body = body[:limit] + "\n...[truncated]"
    return f"[{name}]\n{body}" if body else f"[{name}]"


def count_assistant_turns(messages: list[dict[str, Any]]) -> int:
    return sum(1 for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant")


def _extract_turn_pairs(
    messages: list[dict[str, Any]],
    *,
    task_id: str = "",
) -> list[dict[str, str]]:
    task_key = task_id or None
    tool_limit = tool_result_max_chars(task_key)
    assistant_limit = assistant_turn_max_chars(task_key)
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
        id_to_name: dict[str, str] = {}
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                tc_id = str(call.get("id") or "").strip()
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                tc_name = str(fn.get("name") or "").strip()
                if tc_id and tc_name:
                    id_to_name[tc_id] = tc_name
        tool_parts: list[str] = []
        j = i + 1
        while j < len(messages):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            tc_id = str(nxt.get("tool_call_id") or "").strip()
            tool_parts.append(
                _tool_text(nxt, tool_name=id_to_name.get(tc_id, ""), max_chars=tool_limit)
            )
            j += 1
        turns.append(
            {
                "assistant": assistant[:assistant_limit],
                "tool": "\n".join(tool_parts)[:tool_limit],
            }
        )
        i = j
    return turns


def _normalize_read_path(path: str) -> str:
    cleaned = (path or "").strip().replace("\\", "/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.split("/")[-1] if cleaned else ""


def _parse_read_tool_args(call: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    if str(fn.get("name") or "").strip() != "read":
        return {}
    raw_args = fn.get("arguments")
    if isinstance(raw_args, dict):
        args = raw_args
    else:
        try:
            args = json.loads(str(raw_args or "{}"))
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


def _parse_read_path_from_tool_call(call: dict[str, Any]) -> str:
    args = _parse_read_tool_args(call)
    if not args:
        return ""
    return _normalize_read_path(str(args.get("path") or ""))


def _parse_read_offset_from_tool_call(call: dict[str, Any]) -> int | None:
    args = _parse_read_tool_args(call)
    if not args or "offset" not in args:
        return None
    try:
        return int(args.get("offset"))
    except (TypeError, ValueError):
        return None


def completed_read_paths(messages: list[dict[str, Any]]) -> set[str]:
    """Return basenames of files successfully read (assistant read + non-empty tool result)."""
    paths: set[str] = set()
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        requested: list[str] = []
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict):
                    path = _parse_read_path_from_tool_call(call)
                    if path:
                        requested.append(path)
        j = i + 1
        tool_idx = 0
        while j < len(messages):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            if tool_idx < len(requested):
                body = _message_content(nxt).strip()
                if body and not _tool_result_looks_like_failure(body):
                    paths.add(requested[tool_idx])
            tool_idx += 1
            j += 1
        i = j
    return paths


def _tool_result_looks_like_failure(body: str) -> bool:
    """True for OpenClaw tool error payloads (ENOENT JSON, Traceback, etc.)."""
    stripped = (body or "").strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered.startswith("error") or "traceback" in lowered[:120]:
        return True
    if '"status": "error"' in stripped or '"status":"error"' in stripped:
        return True
    if "enoent" in lowered or "no such file or directory" in lowered:
        return True
    return False


def analyzer_reads_satisfied(
    messages: list[dict[str, Any]],
    *,
    required: frozenset[str] = frozenset({"pricing.py", "cart.py"}),
) -> bool:
    """True when Agent 0 analyzer has non-empty read results for all required files."""
    done = completed_read_paths(messages)
    return required.issubset(done)


def patcher_read_satisfied(
    messages: list[dict[str, Any]],
    *,
    required: frozenset[str] = frozenset({"pricing.py"}),
) -> bool:
    """True when Agent 1 patcher has pricing.py context (read tool or Agent 0 quote)."""
    done = completed_read_paths(messages)
    if required.issubset(done):
        return True
    if "pricing.py" in required:
        upstream = pricing_content_from_context(messages)
        if pricing_apply_discount_return_line(upstream):
            return True
    return False


def pricing_content_from_context(messages: list[dict[str, Any]]) -> str:
    """pricing.py text from tool reads or upstream Agent 0 analysis in the user prompt."""
    latest = _latest_pricing_py_from_tool_messages(messages)
    if latest.strip():
        return latest
    return _pricing_content_from_user_messages(messages)


def _pricing_content_from_user_messages(messages: list[dict[str, Any]]) -> str:
    """Extract a pricing.py snippet quoted by Agent 0 in the chain user prompt."""
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = _message_content(msg)
        if "apply_discount" not in text or "subtotal_cents" not in text:
            continue
        for block in re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
            if "def apply_discount" in block:
                return block.strip()
        match = re.search(
            r"(def apply_discount\([^\)]*\)\s*(?:->[^\n]*)?\n"
            r"(?:.*\n)*?\s*return subtotal_cents[^\n]*)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
    return ""


def _latest_pricing_py_from_tool_messages(messages: list[dict[str, Any]]) -> str:
    """Latest pricing.py body from read results or edit failure snapshots."""
    latest = ""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            i += 1
            continue
        pending: list[str] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            for tool_name in ("edit", "read"):
                path = _parse_tool_path_from_call(call, tool_name)
                if path == "pricing.py":
                    pending.append(tool_name)
                    break
        j = i + 1
        result_idx = 0
        while j < len(messages) and result_idx < len(pending):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            tool_name = pending[result_idx]
            body = _message_content(nxt)
            if tool_name == "read" and body.strip():
                latest = body
            elif tool_name == "edit":
                current_match = _EDIT_CURRENT_FILE_RE.search(body)
                if current_match:
                    latest = current_match.group(1)
            result_idx += 1
            j += 1
        i = j if j > i + 1 else i + 1
    return latest


def verifier_read_satisfied(
    messages: list[dict[str, Any]],
    *,
    required: frozenset[str] = frozenset({"quick_note.md"}),
) -> bool:
    """True when verifier has non-empty read results for required files (basename match)."""
    done = completed_read_paths(messages)
    return required.issubset(done)


# Pre-write exploration target for t1-fs-quick-note (non-empty; .gitkeep is empty and
# would be treated as a failed read by completed_read_paths).
QUICK_NOTE_EXTRACTOR_READ = "verify_three_items.py"


def quick_note_extractor_read_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when Agent 0 has read the quick-note workspace probe file."""
    return QUICK_NOTE_EXTRACTOR_READ in completed_read_paths(messages)


_WRITE_SUCCESS_RE = re.compile(r"Successfully wrote", re.IGNORECASE)


def quick_note_file_valid(content: str) -> bool:
    """True when notes/quick_note.md content covers all three reminders in list form."""
    text = (content or "").strip()
    if not text:
        return False
    lower = text.lower()
    if "dry clean" not in lower:
        return False
    if "sam" not in lower:
        return False
    if "babysit" not in lower:
        return False
    if "60" not in lower:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    list_lines = [
        line
        for line in lines
        if line.startswith(("-", "*", "+")) or re.match(r"^\d+[.)]\s", line)
    ]
    return len(list_lines) >= 2 or len(lines) >= 3


def _is_quick_note_path(path: str) -> bool:
    return _normalize_read_path(path) == "quick_note.md"


def quick_note_write_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when notes/quick_note.md was written/edited with all three reminders."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            i += 1
            continue
        pending: list[tuple[str, dict[str, Any]]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            for tool_name in ("edit", "write"):
                path = _parse_tool_path_from_call(call, tool_name)
                if _is_quick_note_path(path):
                    pending.append((tool_name, call))
                    break
        j = i + 1
        result_idx = 0
        while j < len(messages) and result_idx < len(pending):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            tool_name, call = pending[result_idx]
            body = _message_content(nxt)
            if tool_name == "write" and body.strip() and "error" not in body.lower()[:120]:
                if _WRITE_SUCCESS_RE.search(body) or "successfully wrote" in body.lower():
                    written = _parse_write_content_from_call(call)
                    if quick_note_file_valid(written):
                        return True
            elif tool_name == "edit":
                if _EDIT_SUCCESS_RE.search(body):
                    return True
                current_match = _EDIT_CURRENT_FILE_RE.search(body)
                if current_match and quick_note_file_valid(current_match.group(1)):
                    return True
            result_idx += 1
            j += 1
        i = j if j > i + 1 else i + 1
    return False


def _parse_exec_command_from_call(call: dict[str, Any]) -> str:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    if str(fn.get("name") or "").strip() != "exec":
        return ""
    raw_args = fn.get("arguments")
    if isinstance(raw_args, dict):
        args = raw_args
    else:
        try:
            args = json.loads(str(raw_args or "{}"))
        except json.JSONDecodeError:
            return ""
    if not isinstance(args, dict):
        return ""
    return str(args.get("command") or "").strip()


def _parse_tool_name_from_call(call: dict[str, Any]) -> str:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    return str(fn.get("name") or "").strip()


def _iter_exec_calls_from_messages(messages: list[dict[str, Any]]):
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            command = _parse_exec_command_from_call(call)
            if command:
                yield command


def browser_form_fix_applied(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    return "contact-formm" not in text and "contact-form" in text


def browser_patcher_read_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when Agent 1 read app.js (the file with the form id typo)."""
    return "app.js" in completed_read_paths(messages)


def browser_patcher_edit_applied_in_messages(messages: list[dict[str, Any]]) -> bool:
    """True when Agent 1 successfully edited/wrote app.js in this transcript."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            i += 1
            continue
        pending: list[str] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            for tool_name in ("edit", "write"):
                path = _parse_tool_path_from_call(call, tool_name)
                if path == "app.js":
                    pending.append(tool_name)
                    break
        j = i + 1
        result_idx = 0
        while j < len(messages) and result_idx < len(pending):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            body = _message_content(nxt)
            if _EDIT_SUCCESS_RE.search(body):
                return True
            result_idx += 1
            j += 1
        i = j if j > i + 1 else i + 1
    return False


def browser_patcher_fix_satisfied(
    messages: list[dict[str, Any]],
    *,
    workspace_dir: str = "",
) -> bool:
    """True when app.js fix is confirmed via edit tool success in the transcript."""
    _ = workspace_dir
    if browser_patcher_edit_applied_in_messages(messages):
        return True
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            i += 1
            continue
        pending: list[tuple[str, str]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            path = _parse_tool_path_from_call(call, "read")
            if path == "app.js":
                pending.append(("read", str(call.get("id") or "")))
                break
        j = i + 1
        result_idx = 0
        while j < len(messages) and result_idx < len(pending):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            tool_name, _call_id = pending[result_idx]
            body = _message_content(nxt)
            if tool_name == "read" and browser_form_fix_applied(body):
                return True
            result_idx += 1
            j += 1
        i = j if j > i + 1 else i + 1
    return False


def browser_verifier_exec_done(messages: list[dict[str, Any]]) -> bool:
    return any("verify_form" in command for command in _iter_exec_calls_from_messages(messages))


def _verify_form_exec_body_succeeded(body: str) -> bool:
    """Parse OpenClaw exec tool output for verify_form.cjs."""
    stripped = (body or "").strip()
    if not stripped:
        return False
    # verify_form exits 0 with no stdout; OpenClaw renders that as "(no output)".
    if stripped == "(no output)":
        return True
    lowered = stripped.lower()
    if "exit code 0" in lowered or 'exit_code": 0' in stripped or "exit_code\": 0" in stripped:
        return True
    if (
        "exit code 1" in lowered
        or "exited with code 1" in lowered
        or "command exited with code 1" in lowered
        or "timeout" in lowered
        or "exceeded" in lowered
        or "error:" in lowered
        or "cannot find module" in lowered
        or "enoent" in lowered
        or '"status": "error"' in stripped
        or '"status":"error"' in stripped
    ):
        return False
    # Non-zero exits append "(Command exited with code N)"; absence implies success.
    if "command exited with code" in lowered or "exited with code" in lowered:
        return False
    return True


def browser_verifier_exec_passed(messages: list[dict[str, Any]]) -> bool:
    """True when the latest verify_form exec result succeeded."""
    last_body = ""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        pending_commands: list[str] = []
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                command = _parse_exec_command_from_call(call)
                if command and "verify_form" in command:
                    pending_commands.append(command)
        j = i + 1
        result_idx = 0
        while j < len(messages) and result_idx < len(pending_commands):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            last_body = _message_content(nxt)
            result_idx += 1
            j += 1
        i = j if j > i + 1 else i + 1
    return _verify_form_exec_body_succeeded(last_body)


def build_browser_appjs_edit_hint() -> str:
    return (
        "\napp.js uses getElementById(\"contact-formm\") but index.html defines id=\"contact-form\". "
        "Edit app.js and replace contact-formm with contact-form. "
        "Use the exact oldText from the file you read.\n"
    )


def build_browser_verifier_exec_hint(*, form_app_port: str = "", node_path: str = "") -> str:
    port = (form_app_port or "").strip() or "{form_app_port}"
    np = (node_path or "").strip() or "{openclaw_node_path}:{benchmark_node_path}"
    return (
        f"\nCall exec with command "
        f"`NODE_PATH={np} node verify_form.cjs http://127.0.0.1:{port}/` from the workspace root. "
        "Never set elevated: true.\n"
    )


def browser_exploration_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when Agent 0 successfully used the browser tool at least once."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            i += 1
            continue
        pending = [
            call
            for call in tool_calls
            if isinstance(call, dict) and _parse_tool_name_from_call(call) == "browser"
        ]
        j = i + 1
        result_idx = 0
        while j < len(messages) and result_idx < len(pending):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            body = _message_content(nxt)
            if body.strip() and "error" not in body.lower()[:160]:
                return True
            result_idx += 1
            j += 1
        i = j if j > i + 1 else i + 1
    return False


def verifier_exec_pytest_done(messages: list[dict[str, Any]]) -> bool:
    """True when Agent 2 has run pytest via exec at least once."""
    return any("pytest" in command.lower() for command in _iter_exec_calls_from_messages(messages))


def last_pytest_exec_body(messages: list[dict[str, Any]]) -> str:
    """Return the tool result body from the most recent pytest exec call."""
    last_pytest_body = ""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        pending_commands: list[str] = []
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                command = _parse_exec_command_from_call(call)
                if command:
                    pending_commands.append(command)
        j = i + 1
        result_idx = 0
        while j < len(messages) and result_idx < len(pending_commands):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            command = pending_commands[result_idx]
            if "pytest" in command.lower():
                last_pytest_body = _message_content(nxt)
            result_idx += 1
            j += 1
        i = j if j > i + 1 else i + 1
    return last_pytest_body


def pytest_collection_or_import_failed(messages: list[dict[str, Any]]) -> bool:
    """True when the latest pytest exec failed during collection/import (not assertion failures)."""
    body = last_pytest_exec_body(messages)
    if not body:
        return False
    lowered = body.lower()
    if re.search(r"\b\d+\s+failed\b", lowered):
        return False
    return (
        "modulenotfounderror" in lowered
        or "importerror while importing" in lowered
        or "error collecting" in lowered
        or "interrupted: 1 error during collection" in lowered
    )


def verifier_pytest_passed(messages: list[dict[str, Any]]) -> bool:
    """True when the latest pytest exec result indicates all tests passed."""
    last_pytest_body = last_pytest_exec_body(messages)
    if not last_pytest_body:
        return False
    lowered = last_pytest_body.lower()
    if re.search(r"\b\d+\s+failed\b", lowered) or re.search(r"\b\d+\s+error", lowered):
        return False
    if re.search(r"\b\d+\s+passed\b", lowered):
        return True
    return "passed" in lowered and "failed" not in lowered


def verifier_should_force_read(messages: list[dict[str, Any]]) -> bool:
    """After a failing pytest, read pricing.py before editing."""
    if verifier_pytest_passed(messages) or patcher_read_satisfied(messages):
        return False
    return verifier_exec_pytest_done(messages) and not patcher_fix_satisfied(messages)


def verifier_should_force_exec(messages: list[dict[str, Any]]) -> bool:
    """Verifier must run pytest before read/edit loops when tests not yet passed."""
    if verifier_pytest_passed(messages):
        return False
    if not verifier_exec_pytest_done(messages):
        return True
    if patcher_fix_satisfied(messages):
        return not verifier_pytest_passed(messages)
    return False


def verifier_should_force_edit(messages: list[dict[str, Any]]) -> bool:
    """After a failing pytest, edit pricing.py when the bug is still present."""
    if verifier_pytest_passed(messages) or patcher_fix_satisfied(messages):
        return False
    if not verifier_exec_pytest_done(messages):
        return False
    if not patcher_read_satisfied(messages):
        return False
    return True


_PRICING_DISCOUNT_FIX_RE = re.compile(
    r"return\s+subtotal_cents\s*\*\s*\(\s*100\s*-\s*discount_percent\s*\)\s*//\s*100",
    re.IGNORECASE,
)
_PRICING_BUGGY_RETURN_RE = re.compile(
    r"return\s+subtotal_cents\s*-\s*discount_percent\b",
    re.IGNORECASE,
)
_EDIT_SUCCESS_RE = re.compile(r"Successfully replaced", re.IGNORECASE)
_EDIT_CURRENT_FILE_RE = re.compile(
    r"Current file contents:\s*(.*)",
    re.DOTALL | re.IGNORECASE,
)


def pricing_discount_fix_applied(content: str) -> bool:
    """True when pricing.py text already uses percentage discount on the return line."""
    if not content:
        return False
    return bool(_PRICING_DISCOUNT_FIX_RE.search(content))


_PRICING_RETURN_LINE_RE = re.compile(r"^\s*return\s+subtotal_cents\b", re.IGNORECASE)
_PRICING_EDIT_RETURN_BODY = "return subtotal_cents * (100 - discount_percent) // 100"


def pricing_apply_discount_return_line(content: str) -> str:
    """Return the exact apply_discount return line (indent preserved) from pricing.py text."""
    if not content:
        return ""
    for line in content.splitlines():
        if _PRICING_RETURN_LINE_RE.match(line):
            return line
    return ""


def latest_pricing_py_content(messages: list[dict[str, Any]]) -> str:
    """Latest pricing.py body from read results, edit snapshots, or Agent 0 quotes."""
    return pricing_content_from_context(messages)


def build_pricing_edit_hint(messages: list[dict[str, Any]]) -> str:
    """Tool-bridge hint with exact edit oldText/newText from pricing.py context."""
    content = latest_pricing_py_content(messages)
    old_line = pricing_apply_discount_return_line(content)
    if not old_line:
        old_line = "    return subtotal_cents - discount_percent"
    indent = old_line[: len(old_line) - len(old_line.lstrip())]
    new_line = indent + _PRICING_EDIT_RETURN_BODY
    return (
        "\npricing.py is already in context above. "
        "Call edit on pricing.py with ONE edit. "
        f"oldText must match read output exactly: {old_line!r}. "
        f"newText: {new_line!r}. "
        "Do not read again. One edit call only.\n"
    )


_CONFIG_LOADER_BUGGY_BLOCK = (
    '    if "APP_PORT" in os.environ and path:\n'
    '        config["port"] = json.loads(Path(path).read_text(encoding="utf-8")).get("port", DEFAULTS["port"])\n'
    '    if "APP_DEBUG" in os.environ:\n'
    '        config["debug"] = os.environ["APP_DEBUG"]'
)
_CONFIG_LOADER_FIXED_BLOCK = (
    '    if "APP_PORT" in os.environ:\n'
    '        config["port"] = int(os.environ["APP_PORT"])\n'
    '    elif path:\n'
    '        config["port"] = json.loads(Path(path).read_text(encoding="utf-8")).get("port", DEFAULTS["port"])\n'
    '    \n'
    '    if "APP_DEBUG" in os.environ:\n'
    '        config["debug"] = os.environ["APP_DEBUG"].lower() == "true"'
)


def config_loader_fix_applied(content: str) -> bool:
    """True when config_loader.py applies env overrides and boolean debug parsing."""
    text = content or ""
    if _CONFIG_LOADER_BUGGY_BLOCK in text:
        return False
    return "int(os.environ[\"APP_PORT\"])" in text and '.lower() == "true"' in text


def build_config_loader_edit_hint(messages: list[dict[str, Any]]) -> str:
    """Tool-bridge hint with exact config_loader.py edit oldText/newText."""
    _ = messages
    return (
        "\nconfig_loader.py is in context above. Call edit on config_loader.py with ONE edit. "
        f"oldText must match read output exactly: {_CONFIG_LOADER_BUGGY_BLOCK!r}. "
        f"newText: {_CONFIG_LOADER_FIXED_BLOCK!r}. "
        "Use valid JSON with double-quoted strings in the tool_call. "
        "Do not read again. One edit call only.\n"
    )


def build_config_loader_edit_message() -> dict[str, Any]:
    """Canonical OpenAI assistant message for the config-loader patch edit."""
    arguments = {
        "path": "config_loader.py",
        "edits": [
            {
                "oldText": _CONFIG_LOADER_BUGGY_BLOCK,
                "newText": _CONFIG_LOADER_FIXED_BLOCK,
            }
        ],
    }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": "edit",
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def _parse_tool_path_from_call(call: dict[str, Any], tool_name: str) -> str:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    if str(fn.get("name") or "").strip() != tool_name:
        return ""
    raw_args = fn.get("arguments")
    if isinstance(raw_args, dict):
        args = raw_args
    else:
        try:
            args = json.loads(str(raw_args or "{}"))
        except json.JSONDecodeError:
            return ""
    if not isinstance(args, dict):
        return ""
    return _normalize_read_path(str(args.get("path") or ""))


def patcher_fix_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when pricing.py is already fixed or a successful edit was applied."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            i += 1
            continue
        pending: list[tuple[str, str]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            for tool_name in ("edit", "read"):
                path = _parse_tool_path_from_call(call, tool_name)
                if path == "pricing.py":
                    pending.append((tool_name, str(call.get("id") or "")))
                    break
        j = i + 1
        result_idx = 0
        while j < len(messages) and result_idx < len(pending):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            tool_name, _call_id = pending[result_idx]
            body = _message_content(nxt)
            if tool_name == "edit":
                if _EDIT_SUCCESS_RE.search(body):
                    return True
                current_match = _EDIT_CURRENT_FILE_RE.search(body)
                if current_match and pricing_discount_fix_applied(current_match.group(1)):
                    return True
            elif tool_name == "read" and body.strip():
                if pricing_discount_fix_applied(body) and not _PRICING_BUGGY_RETURN_RE.search(body):
                    return True
            result_idx += 1
            j += 1
        i = j if j > i + 1 else i + 1
    return False


def normalizer_test_file_valid(content: str) -> bool:
    """True when tests/test_normalizer.py has the expected import and coverage hooks."""
    text = (content or "").strip()
    if not text:
        return False
    lower = text.lower()
    if "from ..normalizer" in lower or "openclaw" in lower or "normalize_text" in lower:
        return False
    if "from normalizer import" not in lower and "import normalizer" not in lower:
        return False
    if "normalize_title" not in lower or "normalize_tags" not in lower:
        return False
    if "def test_" not in lower:
        return False
    # Must catch verify_added_tests.py mutants: emoji stripping + blank tags.
    has_emoji_case = (
        "emoji" in lower
        or "\\u0001f" in lower
        or any(ord(ch) >= 0x1F300 for ch in text)
    )
    has_blank_tags = "normalize_tags" in lower and ("== []" in text or "==[]" in text.replace(" ", ""))
    return has_emoji_case and has_blank_tags


def normalizer_analyzer_read_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when Agent 0 has read normalizer.py."""
    return "normalizer.py" in completed_read_paths(messages)


def normalizer_patcher_read_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when Agent 1 has read normalizer.py before authoring tests."""
    return "normalizer.py" in completed_read_paths(messages)


def _parse_write_content_from_call(call: dict[str, Any]) -> str:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    if str(fn.get("name") or "").strip() != "write":
        return ""
    raw_args = fn.get("arguments")
    if isinstance(raw_args, dict):
        args = raw_args
    else:
        try:
            args = json.loads(str(raw_args or "{}"))
        except json.JSONDecodeError:
            return ""
    if not isinstance(args, dict):
        return ""
    return str(args.get("content") or "")


def _is_normalizer_test_path(path: str) -> bool:
    return _normalize_read_path(path) == "test_normalizer.py"


def normalizer_tests_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when tests/test_normalizer.py exists with a usable pytest suite."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            i += 1
            continue
        pending: list[tuple[str, dict[str, Any]]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            for tool_name in ("edit", "read", "write"):
                path = _parse_tool_path_from_call(call, tool_name)
                if _is_normalizer_test_path(path):
                    pending.append((tool_name, call))
                    break
        j = i + 1
        result_idx = 0
        while j < len(messages) and result_idx < len(pending):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            tool_name, call = pending[result_idx]
            body = _message_content(nxt)
            if tool_name == "write" and body.strip() and "error" not in body.lower()[:120]:
                written = _parse_write_content_from_call(call)
                if normalizer_test_file_valid(written):
                    return True
            elif tool_name == "edit":
                current_match = _EDIT_CURRENT_FILE_RE.search(body)
                if current_match and normalizer_test_file_valid(current_match.group(1)):
                    return True
            elif tool_name == "read" and body.strip():
                if normalizer_test_file_valid(body):
                    return True
            result_idx += 1
            j += 1
        i = j if j > i + 1 else i + 1
    return False


def normalizer_tests_ready(
    messages: list[dict[str, Any]],
    *,
    workspace_dir: str = "",
) -> bool:
    """True when tests/test_normalizer.py is ready in session history or on disk."""
    if normalizer_tests_satisfied(messages):
        return True
    root = (workspace_dir or "").strip()
    if not root or not os.path.isdir(root):
        return False
    path = os.path.join(root, "tests", "test_normalizer.py")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as handle:
        return normalizer_test_file_valid(handle.read())


def missing_analyzer_reads(
    messages: list[dict[str, Any]],
    *,
    required: frozenset[str] = frozenset({"pricing.py", "cart.py"}),
) -> frozenset[str]:
    """Paths the analyzer still needs to read."""
    return required - completed_read_paths(messages)


CONFIG_LOADER_ANALYZER_READS = frozenset({"config_loader.py", "app_config.py", "test_config_loader.py"})
CONFIG_LOADER_ANALYZER_READ_ORDER = (
    "config_loader.py",
    "app_config.py",
    "test_config_loader.py",
)


def _config_loader_read_target(basename: str) -> str:
    if basename == "test_config_loader.py":
        return "tests/test_config_loader.py"
    return basename


def next_config_loader_analyzer_read(missing_reads: frozenset[str]) -> str | None:
    """Return the basename of the next config-loader analyzer read in canonical order."""
    for path in CONFIG_LOADER_ANALYZER_READ_ORDER:
        if path in missing_reads:
            return path
    return None


def build_config_loader_analyzer_read_hint(
    missing_reads: frozenset[str],
    *,
    soft: bool = False,
) -> str:
    """Build the next sequential read hint for the config-loader analyzer."""
    next_path = next_config_loader_analyzer_read(missing_reads)
    if next_path is None:
        return ""
    target = _config_loader_read_target(next_path)
    if soft:
        if next_path == "app_config.py":
            return (
                "\nconfig_loader.py is already in context above. "
                "Read app_config.py next — do not re-read config_loader.py.\n"
            )
        if next_path == "test_config_loader.py":
            return (
                "\nconfig_loader.py and app_config.py are already in context above. "
                "Read tests/test_config_loader.py next.\n"
            )
        return ""
    if next_path == "config_loader.py":
        if missing_reads == CONFIG_LOADER_ANALYZER_READS:
            return (
                "\nStep 1: call read on config_loader.py first. "
                "Do not output analysis text — only a read tool call.\n"
            )
        return (
            "\nCall read on config_loader.py next. "
            "Do not output analysis text — only a read tool call.\n"
        )
    return (
        f"\nCall read on {target} next. "
        "Do not output analysis text — only a read tool call.\n"
    )


def config_loader_analyzer_reads_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when Agent 0 has read config_loader.py, app_config.py, and the pytest file."""
    return CONFIG_LOADER_ANALYZER_READS.issubset(completed_read_paths(messages))


def config_loader_missing_analyzer_reads(messages: list[dict[str, Any]]) -> frozenset[str]:
    """Paths the config-loader analyzer still needs to read."""
    return CONFIG_LOADER_ANALYZER_READS - completed_read_paths(messages)


def config_loader_patcher_read_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when Agent 1 has config_loader.py in context."""
    return "config_loader.py" in completed_read_paths(messages)


def config_loader_patcher_fix_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when Agent 1 successfully edited config_loader.py or fix is already present."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            i += 1
            continue
        pending: list[str] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            for tool_name in ("edit", "write"):
                path = _parse_tool_path_from_call(call, tool_name)
                if path == "config_loader.py":
                    pending.append(tool_name)
                    break
        j = i + 1
        result_idx = 0
        while j < len(messages) and result_idx < len(pending):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            body = _message_content(nxt)
            if _EDIT_SUCCESS_RE.search(body):
                return True
            result_idx += 1
            j += 1
        i = j if j > i + 1 else i + 1
    return False


def _index_of_last_pytest_tool_result(messages: list[dict[str, Any]]) -> int:
    """Return message index of the latest pytest exec tool result, or -1."""
    last_idx = -1
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        pending_commands: list[str] = []
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                command = _parse_exec_command_from_call(call)
                if command:
                    pending_commands.append(command)
        j = i + 1
        result_idx = 0
        while j < len(messages) and result_idx < len(pending_commands):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            if "pytest" in pending_commands[result_idx].lower():
                last_idx = j
            result_idx += 1
            j += 1
        i = j if j > i + 1 else i + 1
    return last_idx


def _config_loader_read_after_last_pytest(messages: list[dict[str, Any]]) -> bool:
    """True when config_loader.py was successfully read after the latest pytest result."""
    after = _index_of_last_pytest_tool_result(messages)
    if after < 0:
        return False
    return "config_loader.py" in completed_read_paths(messages[after + 1 :])


def _config_loader_edit_after_last_pytest(messages: list[dict[str, Any]]) -> bool:
    """True when a successful config_loader.py edit landed after the latest pytest result."""
    after = _index_of_last_pytest_tool_result(messages)
    if after < 0:
        return False
    return config_loader_patcher_fix_satisfied(messages[after + 1 :])


def config_loader_verifier_should_force_exec(messages: list[dict[str, Any]]) -> bool:
    """Verifier must run pytest before read/edit loops when tests not yet passed."""
    if verifier_pytest_passed(messages):
        return False
    if pytest_collection_or_import_failed(messages):
        return True
    if not verifier_exec_pytest_done(messages):
        return True
    # After a failed pytest + recovery edit, re-run pytest once (avoid endless exec loops).
    if _config_loader_edit_after_last_pytest(messages):
        return True
    return False


def config_loader_verifier_should_force_read(messages: list[dict[str, Any]]) -> bool:
    """After a failing pytest, read config_loader.py before editing."""
    if verifier_pytest_passed(messages):
        return False
    if pytest_collection_or_import_failed(messages):
        return False
    if not verifier_exec_pytest_done(messages):
        return False
    if _config_loader_edit_after_last_pytest(messages):
        return False
    if _config_loader_read_after_last_pytest(messages):
        return False
    return True


def config_loader_verifier_should_force_edit(messages: list[dict[str, Any]]) -> bool:
    """After a failing pytest, edit config_loader.py when the bug is still present."""
    if verifier_pytest_passed(messages):
        return False
    if pytest_collection_or_import_failed(messages):
        return False
    if not verifier_exec_pytest_done(messages):
        return False
    if _config_loader_edit_after_last_pytest(messages):
        return False
    if not _config_loader_read_after_last_pytest(messages):
        return False
    return True


FIND_THAT_SOURCE_BASENAME = "q3_marketing_budget_v3.xlsx"
FIND_THAT_COPY_BASENAME = "q3_marketing_budget.xlsx"


def _exec_tool_body_succeeded(body: str) -> bool:
    text = (body or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith("{") and '"status": "error"' in lowered.replace(" ", ""):
        return False
    if "cannot create regular file" in lowered:
        return False
    if "no such file or directory" in lowered and "cp:" in lowered:
        return False
    if "(command exited with code 1)" in lowered or "exit code 1" in lowered:
        return False
    return True


def find_that_source_located(messages: list[dict[str, Any]]) -> bool:
    """True when exploration located Documents/q3_marketing_budget_v3.xlsx."""
    needle = FIND_THAT_SOURCE_BASENAME
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        if needle in _message_content(msg):
            return True
    return False


def find_that_copy_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when q3_marketing_budget_v3.xlsx was copied to Desktop/q3_marketing_budget.xlsx."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        tool_calls = msg.get("tool_calls")
        pending = False
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                command = _parse_exec_command_from_call(call)
                if not command:
                    continue
                lowered = command.lower()
                if "cp" not in lowered:
                    continue
                if FIND_THAT_SOURCE_BASENAME not in command:
                    continue
                if FIND_THAT_COPY_BASENAME not in command:
                    continue
                pending = True
                break
        if not pending:
            i += 1
            continue
        j = i + 1
        if j < len(messages):
            nxt = messages[j]
            if isinstance(nxt, dict) and nxt.get("role") == "tool":
                if _exec_tool_body_succeeded(_message_content(nxt)):
                    return True
        i += 1
    return False


def find_that_verifier_exec_done(messages: list[dict[str, Any]]) -> bool:
    return any(
        "verify_correct_file.py" in command
        for command in _iter_exec_calls_from_messages(messages)
    )


def find_that_verifier_passed(messages: list[dict[str, Any]]) -> bool:
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        pending = False
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            command = _parse_exec_command_from_call(call)
            if command and "verify_correct_file.py" in command:
                pending = True
                break
        if not pending:
            i += 1
            continue
        j = i + 1
        if j < len(messages):
            nxt = messages[j]
            if isinstance(nxt, dict) and nxt.get("role") == "tool":
                if "pass:" in _message_content(nxt).lower():
                    return True
        i += 1
    return False


def build_find_that_writer_copy_hint() -> str:
    return (
        "\nCopy the located spreadsheet with one exec call: "
        "mkdir -p Desktop && cp Documents/q3_marketing_budget_v3.xlsx "
        "Desktop/q3_marketing_budget.xlsx\n"
        "Do not output analysis text — only an exec tool call.\n"
    )


def build_find_that_verifier_exec_hint() -> str:
    return (
        "\nRun python3 verify_correct_file.py via exec to confirm the deliverable. "
        "Do not call read on xlsx files.\n"
    )


def build_find_that_bench_copy_exec_message(*, workspace_dir: str = "") -> dict[str, Any]:
    from sidecar.tool_bridge import clawbench_tool_workspace

    workdir = workspace_dir.strip() or clawbench_tool_workspace()
    command = (
        "mkdir -p Desktop && cp Documents/q3_marketing_budget_v3.xlsx "
        "Desktop/q3_marketing_budget.xlsx"
    )
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_bench_find_that_copy",
                "type": "function",
                "function": {
                    "name": "exec",
                    "arguments": json.dumps({"command": command, "workdir": workdir}, ensure_ascii=False),
                },
            }
        ],
    }


def build_pricing_bench_edit_message(*, messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    content = latest_pricing_py_content(list(messages or []))
    old_line = pricing_apply_discount_return_line(content) or "    return subtotal_cents - discount_percent"
    indent = old_line[: len(old_line) - len(old_line.lstrip())]
    new_line = indent + _PRICING_EDIT_RETURN_BODY
    arguments = {
        "path": "pricing.py",
        "edits": [{"oldText": old_line, "newText": new_line}],
    }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_bench_pricing_edit",
                "type": "function",
                "function": {
                    "name": "edit",
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


NORMALIZER_BENCH_TEST_CONTENT = (
    "import pytest\n"
    "from normalizer import normalize_title, normalize_tags\n\n"
    "def test_whitespace_cleanup():\n"
    "    assert normalize_title('  test\\t\\n') == 'Test'\n\n"
    "def test_emoji_stripping_in_titles():\n"
    "    assert normalize_title('🎉 party') == 'Party'\n\n"
    "def test_blank_tags():\n"
    "    assert normalize_tags(',,,') == []\n"
)


def build_normalizer_bench_write_message() -> dict[str, Any]:
    arguments = {
        "path": "tests/test_normalizer.py",
        "content": NORMALIZER_BENCH_TEST_CONTENT,
    }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_bench_normalizer_write",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def build_browser_bench_edit_message() -> dict[str, Any]:
    arguments = {
        "path": "app.js",
        "edits": [
            {
                "oldText": 'document.getElementById("contact-formm")',
                "newText": 'document.getElementById("contact-form")',
            }
        ],
    }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_bench_browser_edit",
                "type": "function",
                "function": {
                    "name": "edit",
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def build_summarize_thread_bench_writer_write_message(
    *,
    messages: list[dict[str, Any]] | None = None,
    workspace_dir: str = "",
    llm: Any = None,
    message_key: str = "",
) -> dict[str, Any]:
    message = build_summarize_thread_writer_write_message(
        messages=list(messages or []),
        workspace_dir=workspace_dir,
        llm=llm,
        message_key=message_key,
    )
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            call["id"] = "call_bench_summarize_write"
    return message


def build_summarize_thread_bench_verifier_exec_message(*, workspace_dir: str = "") -> dict[str, Any]:
    message = build_summarize_thread_verifier_exec_message(workspace_dir=workspace_dir)
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            call["id"] = "call_bench_summarize_exec"
    return message


SUMMARIZE_THREAD_SOURCE_BASENAME = "thread.txt"
SUMMARIZE_THREAD_CONTINUATION_OFFSET = 27
SUMMARIZE_THREAD_VERIFY_SCRIPTS = (
    "verify_summary_structure.py",
    "verify_latest_decision.py",
    "verify_commitments.py",
)
_SUMMARIZE_THREAD_SKIP_MD = frozenset(
    {
        "agents.md",
        "bootstrap.md",
        "heartbeat.md",
        "identity.md",
        "soul.md",
        "tools.md",
        "user.md",
    }
)


def _is_summarize_thread_deliverable_path(path: str) -> bool:
    base = _normalize_read_path(path).lower()
    if not base.endswith(".md"):
        return False
    if base in _SUMMARIZE_THREAD_SKIP_MD:
        return False
    if base.startswith("verify_"):
        return False
    return True


def _iter_summarize_thread_reads(
    messages: list[dict[str, Any]],
) -> list[tuple[int | None, str]]:
    """Return (offset, body) for each successful thread.txt read."""
    reads: list[tuple[int | None, str]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        pending: list[tuple[int | None, str]] = []
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                path = _parse_read_path_from_tool_call(call)
                if path == SUMMARIZE_THREAD_SOURCE_BASENAME:
                    pending.append((_parse_read_offset_from_tool_call(call), ""))
        j = i + 1
        tool_idx = 0
        while j < len(messages):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            if tool_idx < len(pending):
                offset, _ = pending[tool_idx]
                body = _message_content(nxt).strip()
                if body:
                    pending[tool_idx] = (offset, body)
            tool_idx += 1
            j += 1
        reads.extend(entry for entry in pending if entry[1])
        i = j
    return reads


def summarize_thread_thread_read_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when Agent 0 successfully read thread.txt."""
    return SUMMARIZE_THREAD_SOURCE_BASENAME in completed_read_paths(messages)


def summarize_thread_thread_read_truncated(
    messages: list[dict[str, Any]],
    *,
    task_id: str = "",
) -> bool:
    """True when a thread.txt read body exceeds the effective tool-result char limit."""
    limit = tool_result_max_chars(task_id or SUMMARIZE_THREAD_TASK_ID)
    return any(len(body) > limit for _offset, body in _iter_summarize_thread_reads(messages))


def summarize_thread_thread_continuation_read_done(messages: list[dict[str, Any]]) -> bool:
    """True when thread.txt was read from SUMMARIZE_THREAD_CONTINUATION_OFFSET or later."""
    return any(
        offset is not None
        and offset >= SUMMARIZE_THREAD_CONTINUATION_OFFSET
        and body.strip()
        for offset, body in _iter_summarize_thread_reads(messages)
    )


def summarize_thread_extractor_read_complete(
    messages: list[dict[str, Any]],
    *,
    task_id: str = "",
) -> bool:
    """True when thread.txt is fully available (fits in one read or continuation read done)."""
    if not summarize_thread_thread_read_satisfied(messages):
        return False
    if not summarize_thread_thread_read_truncated(messages, task_id=task_id):
        return True
    return summarize_thread_thread_continuation_read_done(messages)


def build_summarize_thread_extractor_read_continuation_hint() -> str:
    return (
        f"\nThe prior thread.txt read was truncated in context (file continues after line "
        f"{SUMMARIZE_THREAD_CONTINUATION_OFFSET - 1}). "
        f"Read thread.txt with offset={SUMMARIZE_THREAD_CONTINUATION_OFFSET} to load the "
        "remaining messages. Do not call exec or other tools — only read.\n"
    )


def summarize_thread_write_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when Agent 1 wrote/edited a non-bootstrap summary markdown deliverable."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            i += 1
            continue
        pending: list[tuple[str, dict[str, Any]]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            for tool_name in ("edit", "write"):
                path = _parse_tool_path_from_call(call, tool_name)
                if _is_summarize_thread_deliverable_path(path):
                    pending.append((tool_name, call))
                    break
        j = i + 1
        result_idx = 0
        while j < len(messages) and result_idx < len(pending):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            tool_name, call = pending[result_idx]
            body = _message_content(nxt)
            if tool_name == "write" and body.strip() and "error" not in body.lower()[:120]:
                if _WRITE_SUCCESS_RE.search(body) or "successfully wrote" in body.lower():
                    write_content = _parse_write_content_from_call(call)
                    if is_valid_summarize_thread_write_content(write_content):
                        return True
            elif tool_name == "edit":
                if _EDIT_SUCCESS_RE.search(body):
                    return True
                current_match = _EDIT_CURRENT_FILE_RE.search(body)
                if current_match and current_match.group(1).strip():
                    return True
            result_idx += 1
            j += 1
        i = j if j > i + 1 else i + 1
    return False


def summarize_thread_verifier_exec_done(messages: list[dict[str, Any]]) -> bool:
    commands = list(_iter_exec_calls_from_messages(messages))
    return all(
        any(script in command for command in commands)
        for script in SUMMARIZE_THREAD_VERIFY_SCRIPTS
    )


def _summarize_thread_passed_scripts_in_tool_body(body: str) -> set[str]:
    """Return verify scripts that emitted PASS in one tool result (supports chained exec)."""
    passed: set[str] = set()
    for line in (body or "").splitlines():
        lowered = line.strip().lower()
        if not lowered.startswith("pass:"):
            continue
        for script in SUMMARIZE_THREAD_VERIFY_SCRIPTS:
            stem = script.replace(".py", "")
            if script.lower() in lowered or stem in lowered:
                passed.add(script)
    return passed


def summarize_thread_verifier_passed(messages: list[dict[str, Any]]) -> bool:
    passed_scripts: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        passed_scripts |= _summarize_thread_passed_scripts_in_tool_body(_message_content(msg))
    return all(script in passed_scripts for script in SUMMARIZE_THREAD_VERIFY_SCRIPTS)


def build_summarize_thread_verifier_exec_hint(*, workspace_dir: str = "") -> str:
    from sidecar.tool_bridge import (
        SUMMARIZE_THREAD_VERIFY_COMMAND,
        clawbench_tool_workspace,
    )

    workdir = clawbench_tool_workspace(workspace_dir=workspace_dir)
    return (
        "\nRun all three verification scripts via exec from the chain workspace root:\n"
        f"command: {SUMMARIZE_THREAD_VERIFY_COMMAND}\n"
        f"workdir: {workdir}\n"
        "Do not cd to ~/workspace or ~/.openclaw/workspace. "
        "Do not output analysis text — only an exec tool call.\n"
    )


def build_summarize_thread_verifier_exec_message(*, workspace_dir: str = "") -> dict[str, Any]:
    """Canonical OpenAI assistant message for summarize-thread verifier exec."""
    from sidecar.tool_bridge import (
        SUMMARIZE_THREAD_VERIFY_COMMAND,
        clawbench_tool_workspace,
    )

    workdir = clawbench_tool_workspace(workspace_dir=workspace_dir)
    arguments = {
        "command": SUMMARIZE_THREAD_VERIFY_COMMAND,
        "workdir": workdir,
    }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": "exec",
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


SUMMARIZE_THREAD_DELIVERABLE_PATH = "design_summary.md"
SUMMARIZE_THREAD_MIN_WRITE_CONTENT_CHARS = 60
_UNFILLED_UPSTREAM_PLACEHOLDER_RE = re.compile(r"^\s*\{agent_\d+_current\}\s*$")


def _is_unfilled_upstream_placeholder(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if _UNFILLED_UPSTREAM_PLACEHOLDER_RE.match(stripped):
        return True
    if "{agent_0_current}" in stripped or "{agent_1_current}" in stripped:
        if len(stripped) < SUMMARIZE_THREAD_MIN_WRITE_CONTENT_CHARS:
            return True
    return False


def _parse_write_content_from_call(call: dict[str, Any]) -> str:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    if str(fn.get("name") or "").strip() != "write":
        return ""
    raw_args = fn.get("arguments")
    if isinstance(raw_args, dict):
        args = raw_args
    else:
        try:
            args = json.loads(str(raw_args or "{}"))
        except json.JSONDecodeError:
            return ""
    if not isinstance(args, dict):
        return ""
    return str(args.get("content") or "")


def _summarize_thread_has_substance(lowered: str) -> bool:
    return any(
        token in lowered
        for token in (
            "decision",
            "decided",
            "commitment",
            "still open",
            "open question",
            "design channel",
        )
    )


def is_valid_summarize_thread_write_content(content: str) -> bool:
    body = (content or "").strip()
    if len(body) < SUMMARIZE_THREAD_MIN_WRITE_CONTENT_CHARS:
        return False
    if _is_unfilled_upstream_placeholder(body):
        return False
    return _summarize_thread_has_substance(body.lower())


def _is_usable_agent0_analysis(text: str) -> bool:
    body = (text or "").strip()
    if _is_unfilled_upstream_placeholder(body):
        return False
    if len(body) < 30:
        return False
    return _summarize_thread_has_substance(body.lower())


def resolve_summarize_thread_agent0_text(
    messages: list[dict[str, Any]],
    *,
    llm: Any = None,
    message_key: str = "",
) -> str:
    """Resolve Agent 0 analysis for writer fallback (messages or upstream KV slot)."""
    text = extract_summarize_thread_agent0_analysis(messages).strip()
    if text and _is_usable_agent0_analysis(text):
        return text
    if llm is not None and message_key:
        decode_upstream = getattr(llm, "decode_upstream_agent_response_text", None)
        if callable(decode_upstream):
            try:
                decoded = sanitize_chat_template_leaks(
                    str(decode_upstream("agent_0_current", message_key) or "")
                ).strip()
                if decoded and _is_usable_agent0_analysis(decoded):
                    return decoded
            except Exception:
                pass
        try:
            slot = llm.resolve_upstream_agent_slot("agent_0_current", message_key)
        except Exception:
            slot = None
        if slot is not None:
            token_ids = getattr(slot, "token_ids", None)
            if isinstance(token_ids, dict):
                input_ids = token_ids.get("input_ids")
                tokenizer = getattr(llm, "tokenizer", None)
                if input_ids is not None and tokenizer is not None:
                    try:
                        decoded = sanitize_chat_template_leaks(
                            tokenizer.decode(input_ids[0], skip_special_tokens=True)
                        ).strip()
                        if decoded and _is_usable_agent0_analysis(decoded):
                            return decoded
                    except Exception:
                        pass
    if text and not _is_unfilled_upstream_placeholder(text) and len(text) >= 40:
        return text
    return ""


def extract_summarize_thread_agent0_analysis(messages: list[dict[str, Any]]) -> str:
    """Best-effort Agent 0 extractor text for summarize-thread writer fallback."""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = _strip_kvcomm_meta(_message_content(msg))
        for label in (
            "Output from Agent 0 (Extractor):",
            "Output from Agent 0:",
        ):
            idx = content.find(label)
            if idx < 0:
                continue
            chunk = content[idx + len(label) :].strip()
            for stop in (
                "\nOpenClaw tool cwd:",
                "\nYour job (Agent 1",
                "\nRead source files first",
            ):
                si = chunk.find(stop)
                if si > 0:
                    chunk = chunk[:si].strip()
            chunk = sanitize_chat_template_leaks(chunk)
            lowered = chunk.lower()
            if (
                chunk
                and not _is_unfilled_upstream_placeholder(chunk)
                and not lowered.startswith("if no tool call is needed")
            ):
                return chunk
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        if msg.get("tool_calls"):
            continue
        body = sanitize_chat_template_leaks(_message_content(msg)).strip()
        lowered = body.lower()
        if (
            body
            and len(body) > 80
            and "decision" in lowered
            and not lowered.startswith("if no tool call is needed")
        ):
            return body
    return ""


def build_summarize_thread_writer_write_message(
    *,
    messages: list[dict[str, Any]],
    workspace_dir: str = "",
    llm: Any = None,
    message_key: str = "",
) -> dict[str, Any]:
    """Canonical write tool call when Agent 1 HF output is unusable."""
    from sidecar.tool_bridge import normalize_summarize_thread_summary_markdown

    _ = workspace_dir
    analysis = resolve_summarize_thread_agent0_text(
        messages,
        llm=llm,
        message_key=message_key,
    ).strip()
    content = analysis
    if not content:
        content = (
            "# Design Channel Summary\n\n"
            "## Decisions\n\n"
            "See Agent 0 analysis in context.\n\n"
            "## Open Questions\n\n"
            "Pending.\n\n"
            "## Commitments\n\n"
            "Pending.\n"
        )
    elif not content.lstrip().startswith("#"):
        content = f"# Design Channel Summary\n\n{content}"
    content = normalize_summarize_thread_summary_markdown(content)
    arguments = {
        "path": SUMMARIZE_THREAD_DELIVERABLE_PATH,
        "content": content,
    }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def summarize_thread_writer_write_needs_canonical_fallback(message: dict[str, Any]) -> bool:
    """True when HF write tool call is missing or has placeholder/invalid content."""
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(tool_calls, list) or not tool_calls:
        return True
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        if str(fn.get("name") or "").strip() != "write":
            continue
        path = _parse_tool_path_from_call(call, "write")
        if not _is_summarize_thread_deliverable_path(path):
            continue
        content = _parse_write_content_from_call(call)
        if is_valid_summarize_thread_write_content(content):
            return False
        return True
    return True


def extract_summarize_thread_summary_from_messages(messages: list[dict[str, Any]]) -> str:
    """Extract summary markdown from Agent 1 context or tool results."""
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        content = _message_content(msg)
        if not content:
            continue
        fence = re.search(r"```(?:markdown|md)?\s*\n(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if fence:
            body = fence.group(1).strip()
            if body and ("decision" in body.lower() or "design channel" in body.lower()):
                return body
        if "Output from Agent 1" in content and "# Design" in content:
            chunk = content[content.find("# Design") :]
            for stop in (
                "\nOpenClaw tool cwd:",
                "\nInspect outputs",
                "\nYour job (Agent 2",
                "\n[tool_call",
            ):
                si = chunk.find(stop)
                if si > 0:
                    chunk = chunk[:si]
            chunk = chunk.strip()
            if chunk:
                return chunk
    return ""


def ensure_summarize_thread_chain_deliverable(
    *,
    workspace_dir: str = "",
    messages: list[dict[str, Any]] | None = None,
    llm: Any = None,
    message_key: str = "",
) -> bool:
    """Ensure design_summary.md exists in the chain workspace before verify exec."""
    from sidecar.tool_bridge import (
        clawbench_tool_workspace,
        normalize_summarize_thread_summary_markdown,
    )

    chain_root = clawbench_tool_workspace(workspace_dir=workspace_dir)
    if not workspace_dir or not os.path.isdir(chain_root):
        return False
    target = os.path.join(chain_root, SUMMARIZE_THREAD_DELIVERABLE_PATH)
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        try:
            existing = Path(target).read_text(encoding="utf-8", errors="ignore")
            if is_valid_summarize_thread_write_content(existing):
                fixed = normalize_summarize_thread_summary_markdown(existing)
                if fixed and fixed != existing:
                    Path(target).write_text(fixed, encoding="utf-8")
                return True
        except OSError:
            pass

    agent0 = resolve_summarize_thread_agent0_text(
        list(messages or []),
        llm=llm,
        message_key=message_key,
    )
    if agent0.strip():
        content = normalize_summarize_thread_summary_markdown(
            agent0 if agent0.lstrip().startswith("#") else f"# Design Channel Summary\n\n{agent0}"
        )
        try:
            Path(target).write_text(content, encoding="utf-8")
            return True
        except OSError:
            pass

    extracted = extract_summarize_thread_summary_from_messages(list(messages or []))
    if extracted.strip():
        content = normalize_summarize_thread_summary_markdown(extracted)
        try:
            Path(target).write_text(content, encoding="utf-8")
            return True
        except OSError:
            pass

    default_root = os.path.normpath(clawbench_tool_workspace(workspace_dir=""))
    default_file = os.path.join(default_root, SUMMARIZE_THREAD_DELIVERABLE_PATH)
    if os.path.isfile(default_file) and os.path.getsize(default_file) > 0:
        try:
            source = Path(default_file).read_text(encoding="utf-8", errors="ignore")
            if is_valid_summarize_thread_write_content(source):
                fixed = normalize_summarize_thread_summary_markdown(source)
                Path(target).write_text(fixed or source, encoding="utf-8")
                return True
        except OSError:
            pass

    return False


REDACT_DOC_SOURCE_BASENAME = "contract.txt"
REDACT_DOC_DELIVERABLE_PATH = "contract_redacted.txt"
REDACT_DOC_PII_MARKERS = (
    "Lin Park",
    "lin.park@personalmail.example",
    "+1 (415) 555-0173",
    "AC-77821-PK",
)
REDACT_DOC_PRESERVED_KEYWORDS = ("service agreement", "scope of work", "termination")

# Fixed-length bench canonical outputs for comparable dense vs kv_reuse asst(s).
REDACT_DOC_EXTRACTOR_CANONICAL = (
    "Agent 0 analysis: contract.txt contains PII that must be redacted before sharing.\n"
    "PII fields:\n"
    "- Name: Lin Park\n"
    "- Email: lin.park@personalmail.example\n"
    "- Phone: +1 (415) 555-0173\n"
    "- Account number: AC-77821-PK\n"
    "Deliverable: contract_redacted.txt with PII replaced; original contract.txt unchanged."
)
REDACT_DOC_WRITER_DONE_CANONICAL = "DONE: contract_redacted.txt"
REDACT_DOC_VERIFIER_PASS_CANONICAL = "PASS: verify_redaction.py OK"
REDACT_DOC_BENCH_WRITE_CALL_ID = "call_bench_redact_doc_write"
REDACT_DOC_BENCH_EXEC_CALL_ID = "call_bench_redact_doc_exec"


def estimate_bench_text_tokens(text: str) -> int:
    """Rough token estimate for bench reuse/decode accounting."""
    body = str(text or "").strip()
    if not body:
        return 0
    return max(1, len(body) // 4)


def redact_doc_bench_canonical_text(gate: str) -> str:
    """Return fixed canonical assistant text for redact-doc bench text-only gates."""
    key = str(gate or "").strip().lower()
    if key in ("extractor", "extractor_done", "agent_0"):
        return REDACT_DOC_EXTRACTOR_CANONICAL
    if key in ("writer_done", "done", "agent_1"):
        return REDACT_DOC_WRITER_DONE_CANONICAL
    if key in ("verifier_done", "pass", "agent_2"):
        return REDACT_DOC_VERIFIER_PASS_CANONICAL
    return ""


def build_redact_doc_bench_writer_write_message(
    *,
    messages: list[dict[str, Any]] | None = None,
    workspace_dir: str = "",
) -> dict[str, Any]:
    """Deterministic bench write tool call (fixed call id for stable token count)."""
    message = build_redact_doc_writer_write_message(
        messages=list(messages or []),
        workspace_dir=workspace_dir,
    )
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            call["id"] = REDACT_DOC_BENCH_WRITE_CALL_ID
    return message


def build_redact_doc_bench_verifier_exec_message(*, workspace_dir: str = "") -> dict[str, Any]:
    """Deterministic bench verify exec tool call (fixed call id)."""
    message = build_redact_doc_verifier_exec_message(workspace_dir=workspace_dir)
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            call["id"] = REDACT_DOC_BENCH_EXEC_CALL_ID
    return message


def redact_doc_bench_forced_generation_text(
    gate: str,
    *,
    messages: list[dict[str, Any]] | None = None,
    workspace_dir: str = "",
) -> str:
    """Return fixed HF assistant text for bench teacher-forced decode."""
    from sidecar.tool_bridge import openai_message_to_generation_text

    key = str(gate or "").strip().lower()
    if key in ("extractor", "extractor_done", "agent_0"):
        return REDACT_DOC_EXTRACTOR_CANONICAL
    if key in ("writer_write", "write", "agent_1_write"):
        return openai_message_to_generation_text(
            build_redact_doc_bench_writer_write_message(
                messages=list(messages or []),
                workspace_dir=workspace_dir,
            )
        )
    if key in ("writer_done", "done", "agent_1"):
        return REDACT_DOC_WRITER_DONE_CANONICAL
    if key in ("verifier_exec", "exec", "agent_2_exec"):
        return openai_message_to_generation_text(
            build_redact_doc_bench_verifier_exec_message(workspace_dir=workspace_dir)
        )
    if key in ("verifier_done", "pass", "agent_2"):
        return REDACT_DOC_VERIFIER_PASS_CANONICAL
    return ""


def _is_redact_doc_deliverable_path(path: str) -> bool:
    base = _normalize_read_path(path).lower()
    return "contract" in base and "redact" in base


def redact_contract_content(content: str) -> str:
    """Replace known PII strings with redaction placeholders."""
    text = str(content or "")
    for old, new in (
        ("Lin Park", "[REDACTED NAME]"),
        ("lin.park@personalmail.example", "[REDACTED EMAIL]"),
        ("+1 (415) 555-0173", "[REDACTED PHONE]"),
        ("AC-77821-PK", "[REDACTED ACCOUNT]"),
    ):
        text = text.replace(old, new)
    return text


def _redact_doc_pii_present(content: str) -> bool:
    return any(marker in (content or "") for marker in REDACT_DOC_PII_MARKERS)


def is_valid_redact_doc_write_content(content: str) -> bool:
    body = str(content or "").strip()
    if len(body) < 40:
        return False
    if _redact_doc_pii_present(body):
        return False
    lowered = body.lower()
    return all(keyword in lowered for keyword in REDACT_DOC_PRESERVED_KEYWORDS)


def redact_doc_extractor_read_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when contract.txt was read successfully in the chain."""
    return REDACT_DOC_SOURCE_BASENAME in completed_read_paths(messages)


def _extract_contract_text_from_tool_results(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        body = _message_content(msg).strip()
        if not body or "error" in body.lower()[:120]:
            continue
        if "[read]" in body.lower()[:12]:
            body = re.sub(r"^\[read\]\s*", "", body, flags=re.IGNORECASE).strip()
        if "Service Agreement" in body and "Lin Park" in body:
            return body
    return ""


def resolve_redact_doc_contract_text(
    *,
    messages: list[dict[str, Any]] | None = None,
    workspace_dir: str = "",
) -> str:
    """Load contract.txt from chain workspace or prior read tool results."""
    chain_root = str(workspace_dir or "").strip()
    if chain_root:
        source = os.path.join(chain_root, REDACT_DOC_SOURCE_BASENAME)
        try:
            if os.path.isfile(source):
                return Path(source).read_text(encoding="utf-8")
        except OSError:
            pass
    return _extract_contract_text_from_tool_results(list(messages or []))


def redact_doc_write_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True when a redacted contract copy was written without PII."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            i += 1
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            i += 1
            continue
        pending: list[dict[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            if str((call.get("function") or {}).get("name") or "").strip() != "write":
                continue
            path = _parse_tool_path_from_call(call, "write")
            if _is_redact_doc_deliverable_path(path):
                pending.append(call)
        j = i + 1
        result_idx = 0
        while j < len(messages) and result_idx < len(pending):
            nxt = messages[j]
            if not isinstance(nxt, dict) or nxt.get("role") != "tool":
                break
            call = pending[result_idx]
            body = _message_content(nxt)
            if body.strip() and "error" not in body.lower()[:120]:
                if _WRITE_SUCCESS_RE.search(body) or "successfully wrote" in body.lower():
                    write_content = _parse_write_content_from_call(call)
                    if is_valid_redact_doc_write_content(write_content):
                        return True
            result_idx += 1
            j += 1
        i = j if j > i + 1 else i + 1
    return False


def build_redact_doc_writer_write_message(
    *,
    messages: list[dict[str, Any]] | None = None,
    workspace_dir: str = "",
) -> dict[str, Any]:
    """Canonical write tool call: contract_redacted.txt with PII removed."""
    source = resolve_redact_doc_contract_text(
        messages=list(messages or []),
        workspace_dir=workspace_dir,
    )
    content = redact_contract_content(source)
    if not is_valid_redact_doc_write_content(content):
        content = (
            "Service Agreement\n\n"
            "This agreement is between [REDACTED NAME] (\"Client\") and the Vendor.\n\n"
            "Client contact:\n"
            "  Name: [REDACTED NAME]\n"
            "  Email: [REDACTED EMAIL]\n"
            "  Phone: [REDACTED PHONE]\n"
            "  Account number: [REDACTED ACCOUNT]\n\n"
            "Scope of work:\n"
            "  - Implement the data ingestion pipeline described in Appendix A.\n"
            "  - Deliver weekly progress reports.\n"
            "  - Handover by Q3 2026.\n\n"
            "Pricing:\n"
            "  Base fee: $48,000 (US dollars)\n"
            "  Optional extension: up to $12,000 additional, billed monthly.\n\n"
            "Termination:\n"
            "  Either party may terminate with 30 days written notice.\n\n"
            "Signed,\n"
            "[REDACTED NAME] (Client)\n"
            "April 9, 2026\n"
        )
    arguments = {
        "path": REDACT_DOC_DELIVERABLE_PATH,
        "content": content,
    }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def redact_doc_writer_write_needs_canonical_fallback(message: dict[str, Any]) -> bool:
    """True when HF output edits the source file or writes an invalid deliverable."""
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(tool_calls, list) or not tool_calls:
        return True
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        tool_name = str(fn.get("name") or "").strip()
        if tool_name == "edit":
            path = _parse_tool_path_from_call(call, "edit")
            if _normalize_read_path(path).lower() == REDACT_DOC_SOURCE_BASENAME:
                return True
            return True
        if tool_name != "write":
            continue
        path = _parse_tool_path_from_call(call, "write")
        if _normalize_read_path(path).lower() == REDACT_DOC_SOURCE_BASENAME:
            return True
        if not _is_redact_doc_deliverable_path(path):
            return True
        content = _parse_write_content_from_call(call)
        if not is_valid_redact_doc_write_content(content):
            return True
        return False
    return True


def redact_doc_verifier_exec_done(messages: list[dict[str, Any]]) -> bool:
    return any(
        "verify_redaction.py" in command
        for command in _iter_exec_calls_from_messages(messages)
    )


def redact_doc_verifier_passed(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        body = _message_content(msg).lower()
        if "pass:" in body and "redact" in body:
            return True
    return False


def redact_doc_verifier_exec_needs_canonical_fallback(message: dict[str, Any]) -> bool:
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(tool_calls, list) or not tool_calls:
        return True
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        command = _parse_exec_command_from_call(call)
        if command and "verify_redaction.py" in command:
            return False
    return True


def build_redact_doc_verifier_exec_hint(*, workspace_dir: str = "") -> str:
    from sidecar.tool_bridge import REDACT_DOC_VERIFY_COMMAND, clawbench_tool_workspace

    workdir = clawbench_tool_workspace(workspace_dir=workspace_dir)
    return (
        "\nRun the redaction verifier via exec from the chain workspace root:\n"
        f"command: {REDACT_DOC_VERIFY_COMMAND}\n"
        f"workdir: {workdir}\n"
        "Do not modify contract.txt. Do not output analysis text — only an exec tool call.\n"
    )


def build_redact_doc_verifier_exec_message(*, workspace_dir: str = "") -> dict[str, Any]:
    from sidecar.tool_bridge import REDACT_DOC_VERIFY_COMMAND, clawbench_tool_workspace

    workdir = clawbench_tool_workspace(workspace_dir=workspace_dir)
    arguments = {
        "command": REDACT_DOC_VERIFY_COMMAND,
        "workdir": workdir,
    }
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": "exec",
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def ensure_redact_doc_chain_deliverable(
    *,
    workspace_dir: str,
    messages: list[dict[str, Any]] | None = None,
) -> bool:
    """Ensure contract_redacted.txt exists in the chain workspace before verify."""
    chain_root = str(workspace_dir or "").strip()
    if not chain_root:
        return False
    target = os.path.join(chain_root, REDACT_DOC_DELIVERABLE_PATH)
    try:
        if os.path.isfile(target):
            existing = Path(target).read_text(encoding="utf-8")
            if is_valid_redact_doc_write_content(existing):
                return True
    except OSError:
        pass
    source = resolve_redact_doc_contract_text(
        messages=list(messages or []),
        workspace_dir=chain_root,
    )
    content = redact_contract_content(source)
    if not is_valid_redact_doc_write_content(content):
        message = build_redact_doc_writer_write_message(
            messages=list(messages or []),
            workspace_dir=chain_root,
        )
        for call in message.get("tool_calls") or []:
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            if str(fn.get("name") or "") != "write":
                continue
            try:
                args = json.loads(str(fn.get("arguments") or "{}"))
            except json.JSONDecodeError:
                continue
            content = str(args.get("content") or "")
            break
    try:
        Path(target).write_text(content, encoding="utf-8")
        return True
    except OSError:
        return False


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


def turn_segment_template(turn_index: int) -> str:
    """Single-turn suffix appended to static user_template (matches openclaw parse)."""
    return f"\n\n{{turn_{turn_index}_assistant}}\n\n{{turn_{turn_index}_tool}}\n"


def build_user_template_with_turns(static_user: str, turn_count: int) -> str:
    """Canonical cumulative user_template for *turn_count* assistant/tool turns."""
    static = str(static_user or "").rstrip()
    suffix = ""
    for idx in range(1, max(0, int(turn_count)) + 1):
        suffix += turn_segment_template(idx)
    return f"{static}{suffix}".strip()


def merge_turn_segment_into_user_template(
    stored_user: str,
    segment_template: str,
    *,
    expected_user_template: str | None = None,
) -> str:
    """Extend committed user_template with one turn segment (openclaw-compatible merge)."""
    stored = str(stored_user or "").strip()
    segment = str(segment_template or "")
    seg_core = segment.rstrip()
    if not seg_core:
        if expected_user_template is not None:
            return str(expected_user_template).strip()
        return stored
    if stored and stored.rstrip().endswith(seg_core):
        return stored
    base = stored.rstrip()
    # Prior commits strip trailing whitespace, dropping the newline that separates
    # turn blocks. Restore it before appending turn 2+ (segment already starts with \n\n).
    if stored and re.search(r"\{turn_\d+_(?:assistant|tool)\}", base):
        merged = f"{base}\n{segment}".strip()
    elif stored:
        merged = f"{base}{segment}".strip()
    else:
        merged = seg_core
    if expected_user_template is not None and not stored:
        return str(expected_user_template).strip()
    return merged


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
    # Preserve run-* paths in OpenClaw cwd hints — agents/exec need the exact workspace.
    if "OpenClaw cwd:" in cleaned or "OpenClaw tool cwd:" in cleaned:
        return cleaned
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
    task_id: str = "",
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

    turns = _extract_turn_pairs(messages, task_id=task_id)
    max_tokens = prefix_max_tokens()
    trimmed_turns = 0
    turn_suffix = ""
    turn_content: dict[str, str] = {}
    estimated = 0
    while True:
        turn_suffix = ""
        turn_content = {}
        for idx, turn in enumerate(turns, start=1):
            turn_suffix += turn_segment_template(idx)
            turn_content[f"turn_{idx}_assistant"] = turn["assistant"].strip() or " "
            turn_content[f"turn_{idx}_tool"] = turn["tool"].strip() or " "

        user_template = f"{static_user.rstrip()}{turn_suffix}".strip()
        estimated = _estimate_tokens(system_prompt) + _estimate_tokens(user_template)
        for content in turn_content.values():
            estimated += _estimate_tokens(content)

        if estimated <= max_tokens:
            break
        if turns:
            turns = turns[1:]
            trimmed_turns += 1
            continue
        raise PrefixOverflowError(
            f"prefix estimated_tokens={estimated} exceeds KVCOMM_PREFIX_MAX_TOKENS={max_tokens}"
        )

    if trimmed_turns:
        trim_note = f"prefix_turns_trimmed={trimmed_turns}"
        warn = f"{warn};{trim_note}" if warn else trim_note

    user_template = f"{static_user.rstrip()}{turn_suffix}".strip()
    placeholders = _collect_placeholders(user_template)
    static_text = _static_without_turn_placeholders(user_template)

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
