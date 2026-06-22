"""Bridge OpenAI tool_calls API to Qwen3 HF generation for the KVCOMM sidecar."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any

_CHAT_TEMPLATE_LEAK_RE = re.compile(
    r"<\|im_start\|>\s*|<\|im_end\|>\s*|<\|redacted_im_end\|>\s*",
    re.IGNORECASE,
)

_WRITER_TOOL_NAMES = frozenset({"write", "edit"})
_VERIFIER_TOOL_NAMES = frozenset({"read", "write", "edit"})


def sanitize_chat_template_leaks(text: str) -> str:
    """Strip Qwen/OpenClaw chat control tokens that leaked into generated text."""
    if not text:
        return text
    cleaned = _CHAT_TEMPLATE_LEAK_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


_QUICK_NOTE_ALIASES = {
    "notes.txt": "notes/quick_note.md",
    "workspace/notes.txt": "notes/quick_note.md",
    "notes/notes.txt": "notes/quick_note.md",
}


def normalize_tool_file_path(path: str) -> str:
    """Fix common relative-path mistakes before OpenClaw resolves against workspace."""
    if not path:
        return path
    normalized = path.strip().replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized in _QUICK_NOTE_ALIASES:
        return _QUICK_NOTE_ALIASES[normalized]
    if normalized.startswith("workspace/"):
        stripped = normalized[len("workspace/") :]
        return _QUICK_NOTE_ALIASES.get(stripped, stripped)
    return normalized


def _normalize_tool_arguments(name: str, arguments: Any) -> Any:
    if not isinstance(arguments, dict):
        return arguments
    if "path" not in arguments:
        return arguments
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        return arguments
    fixed = normalize_tool_file_path(path)
    if fixed == path:
        return arguments
    updated = dict(arguments)
    updated["path"] = fixed
    return updated


def filter_tools_for_agent(
    tools: list[dict[str, Any]],
    *,
    agent_index: str | int | None = None,
    agent_role: str = "",
) -> list[dict[str, Any]]:
    """Keep only role-relevant tools to shrink generation-boundary injection."""
    if os.environ.get("KVCOMM_TOOL_BRIDGE_MINIMAL", "1").strip().lower() in ("0", "false", "no", "off"):
        return tools
    role = (agent_role or "").strip().lower()
    try:
        idx = int(agent_index) if agent_index is not None else -1
    except (TypeError, ValueError):
        idx = -1
    if "writer" in role or idx == 1:
        allowed = _WRITER_TOOL_NAMES
    elif "verifier" in role or idx == 2:
        allowed = _VERIFIER_TOOL_NAMES
    else:
        return tools
    filtered: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = str(fn.get("name") or tool.get("name") or "").strip()
        if name in allowed:
            filtered.append(tool)
    return filtered or tools


def should_inject_tools(body: dict[str, Any]) -> bool:
    """Inject tool schemas only on the first assistant turn of a session."""
    raw = os.environ.get("KVCOMM_TOOL_INJECT_ON_TURNS", "first_only").strip().lower()
    if raw in ("0", "false", "no", "off", "never"):
        return False
    if raw in ("always", "all", "every"):
        return True
    from sidecar.openclaw_prefix import count_assistant_turns

    return count_assistant_turns(body.get("messages") or []) == 0


def tool_bridge_enabled() -> bool:
    raw = os.environ.get("KVCOMM_TOOL_BRIDGE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def extract_tool_request(body: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, Any]:
    """Return (normalized_tools, tool_choice) when the request expects tool calling."""
    if not tool_bridge_enabled():
        return None, None
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return None, None

    choice = body.get("tool_choice")
    extra = body.get("extra_body")
    if choice is None and isinstance(extra, dict):
        choice = extra.get("tool_choice")
    if choice in ("none", None):
        return None, None
    if isinstance(choice, dict) and choice.get("type") == "none":
        return None, None

    normalized = normalize_openai_tools(tools)
    if not normalized:
        return None, None
    return normalized, choice if choice is not None else "auto"


def normalize_openai_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Normalize OpenClaw/OpenAI tool definitions for HF chat templates."""
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(fn.get("description") or ""),
                        "parameters": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {},
                    },
                }
            )
            continue
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description") or ""),
                    "parameters": tool.get("input_schema")
                    if isinstance(tool.get("input_schema"), dict)
                    else (tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}),
                },
            }
        )
    return normalized


def _manual_qwen_tools_text(tools: list[dict[str, Any]]) -> str:
    """Qwen3-style tools preamble when tokenizer chat template is unavailable."""
    parts = [
        "\n# Tools\n\n",
        "You may call one or more functions to assist with the user query.\n\n",
        "You are provided with function signatures within <tools></tools> XML tags:\n",
        "<tools>",
    ]
    for tool in tools:
        parts.append("\n")
        parts.append(json.dumps(tool, ensure_ascii=False))
    parts.append(
        "\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>\n"
    )
    return "".join(parts)


def build_tool_injection_text(
    tools: list[dict[str, Any]],
    tokenizer: Any = None,
    tool_choice: Any = "auto",
) -> str:
    """Text appended at generation boundary so HF sees tool schemas without polluting prefix KV."""
    _ = tokenizer  # reserved for future tokenizer-specific formatting
    text = _manual_qwen_tools_text(tools).strip()
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else {}
        forced = str(fn.get("name") or tool_choice.get("name") or "").strip()
        if forced:
            text += f"\nYou must call the `{forced}` tool for this turn.\n"
    return f"\n{text}\n"


def _parse_arguments(raw: Any) -> str:
    if raw is None:
        return "{}"
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return "{}"
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            return json.dumps({"raw": stripped}, ensure_ascii=False)
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    return json.dumps(raw, ensure_ascii=False)


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def parse_qwen_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse Qwen `<tool_call>` blocks into OpenAI `tool_calls` payloads."""
    if not text:
        return "", []

    tool_calls: list[dict[str, Any]] = []
    content_parts: list[str] = []
    last_end = 0

    for match in _TOOL_CALL_RE.finditer(text):
        content_parts.append(text[last_end : match.start()])
        payload_raw = match.group(1).strip()
        last_end = match.end()
        if not payload_raw:
            continue
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name") or payload.get("function") or "").strip()
        if not name:
            continue
        arguments = _normalize_tool_arguments(name, payload.get("arguments"))
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _parse_arguments(arguments),
                },
            }
        )

    content_parts.append(text[last_end:])
    content = "".join(content_parts).strip()
    if not content:
        content = None if tool_calls else ""
    return content or "", tool_calls


def openai_message_from_generation(raw: str) -> dict[str, Any]:
    """Convert raw HF assistant text into an OpenAI chat completion message."""
    content, tool_calls = parse_qwen_tool_calls(sanitize_chat_template_leaks(raw or ""))
    message: dict[str, Any] = {"role": "assistant"}
    if content:
        message["content"] = content
    else:
        message["content"] = None
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def sse_tool_call_deltas(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build OpenAI SSE delta payloads for each tool call."""
    deltas: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        deltas.append(
            {
                "index": index,
                "id": tool_call.get("id"),
                "type": tool_call.get("type") or "function",
                "function": tool_call.get("function") or {},
            }
        )
    return deltas
