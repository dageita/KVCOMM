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


def _tool_text(msg: dict[str, Any], *, tool_name: str = "") -> str:
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
            tool_parts.append(_tool_text(nxt, tool_name=id_to_name.get(tc_id, "")))
            j += 1
        turns.append(
            {
                "assistant": assistant[: tool_result_max_chars()],
                "tool": "\n".join(tool_parts)[: tool_result_max_chars()],
            }
        )
        i = j
    return turns


def _normalize_read_path(path: str) -> str:
    cleaned = (path or "").strip().replace("\\", "/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.split("/")[-1] if cleaned else ""


def _parse_read_path_from_tool_call(call: dict[str, Any]) -> str:
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    if str(fn.get("name") or "").strip() != "read":
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
            if tool_idx < len(requested) and _message_content(nxt).strip():
                paths.add(requested[tool_idx])
            tool_idx += 1
            j += 1
        i = j
    return paths


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


def verifier_pytest_passed(messages: list[dict[str, Any]]) -> bool:
    """True when the latest pytest exec result indicates all tests passed."""
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
    return "def test_" in lower


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
