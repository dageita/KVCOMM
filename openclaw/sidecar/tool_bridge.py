"""Bridge OpenAI tool_calls API to Qwen3 HF generation for the KVCOMM sidecar."""

from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from typing import Any

from sidecar.bench_prompt_compose import BUGFIX_DISCOUNT_TASK_ID

_CHAT_TEMPLATE_LEAK_RE = re.compile(
    r"<\|im_start\|>\s*|<\|im_end\|>\s*|<\|redacted_im_end\|>\s*",
    re.IGNORECASE,
)

_WRITER_TOOL_NAMES = frozenset({"write", "edit"})
_VERIFIER_TOOL_NAMES = frozenset({"read", "write", "edit"})

# ClawBench / OpenClaw capability chain roles (by index or role label).
_ANALYZER_TOOLS = frozenset({"read"})
_PATCHER_TOOLS = frozenset({"read", "edit", "write", "apply_patch"})
_VERIFIER_TOOLS = frozenset({"read", "edit", "exec", "process"})
_CLAWBENCH_REQUIRED_BY_INDEX = {
    0: _ANALYZER_TOOLS,
    1: _PATCHER_TOOLS,
    2: _VERIFIER_TOOLS,
}

_FALLBACK_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "read": {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file (relative or absolute path).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                    "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                    "limit": {"type": "number", "description": "Maximum number of lines to read"},
                },
                "required": ["path"],
            },
        },
    },
    "edit": {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit a file by replacing text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "edits": {
                        "type": "array",
                        "description": "List of search/replace edits",
                        "items": {
                            "type": "object",
                            "properties": {
                                "search": {"type": "string"},
                                "replace": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    "write": {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    "apply_patch": {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff patch to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to patch"},
                    "patch": {"type": "string", "description": "Unified diff patch text"},
                },
                "required": ["path", "patch"],
            },
        },
    },
    "exec": {
        "type": "function",
        "function": {
            "name": "exec",
            "description": "Run a shell command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    "process": {
        "type": "function",
        "function": {
            "name": "process",
            "description": "Manage background processes started by exec.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Process action (list, kill, etc.)"},
                },
            },
        },
    },
}

_TOOL_NAME_ALIASES = {
    "read_file": "read",
    "write_file": "write",
    "edit_file": "edit",
    "apply_patch_file": "apply_patch",
    "run_terminal_cmd": "exec",
    "bash": "exec",
    "shell": "exec",
    "terminal": "exec",
}


def canonical_tool_name(name: str) -> str:
    normalized = (name or "").strip()
    if not normalized:
        return normalized
    return _TOOL_NAME_ALIASES.get(normalized, normalized)


def sanitize_chat_template_leaks(text: str) -> str:
    """Strip Qwen/OpenClaw chat control tokens that leaked into generated text."""
    if not text:
        return text
    cleaned = _CHAT_TEMPLATE_LEAK_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


_THINKING_BLOCK_RE = re.compile(
    r"<\|redacted_thinking\|>.*?<\|/redacted_thinking\|>\s*",
    re.DOTALL | re.IGNORECASE,
)
_THINKING_OPEN_RE = re.compile(
    r"^<\|redacted_thinking\|>.*?(?:<\|/redacted_thinking\|>\s*|$)",
    re.DOTALL | re.IGNORECASE,
)


def _collapse_line_repetition(text: str, *, min_repeats: int = 3) -> str:
    """Trim degenerate loops like repeated 'Do not call functions…' tails."""
    lines = text.splitlines()
    if len(lines) < min_repeats * 2:
        return text
    run_line = lines[-1]
    run_len = 1
    idx = len(lines) - 2
    while idx >= 0 and lines[idx] == run_line:
        run_len += 1
        idx -= 1
    if run_len >= min_repeats and run_line.strip():
        keep = max(1, len(lines) - run_len + 1)
        return "\n".join(lines[:keep]).strip()
    return text


_NEVER_CALL_LINE_RE = re.compile(r"^Never call a function\b", re.IGNORECASE)
_RESPONSE_BLOCK_LEAK_RE = re.compile(
    r"(?:After all tool calls|If no tools are needed)[^\n]*<response>[^\n]*\n?",
    re.IGNORECASE,
)
_NOW_PROCEED_LEAK_RE = re.compile(
    r"^Now proceed to solve the problem\.\s*\n?",
    re.MULTILINE | re.IGNORECASE,
)
_TOOL_PREAMBLE_LEAK_RE = re.compile(
    r"^(?:If multiple actions are needed.*?\n)?(?:Never call a function.*?\n){3,}",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_GUIDELINE_PREAMBLE_RE = re.compile(
    r"^(?:If multiple actions are needed[^\n]*\n)"
    r"(?:[A-Z][^\n]{0,120}\n){5,}",
    re.MULTILINE,
)


def _collapse_never_call_spam(text: str, *, min_lines: int = 3) -> str:
    """Drop OpenClaw tool-discipline loops that Qwen regurgitates under long prompts."""
    lines = text.splitlines()
    never_idx = [i for i, line in enumerate(lines) if _NEVER_CALL_LINE_RE.match(line.strip())]
    if len(never_idx) >= min_lines:
        first = never_idx[0]
        prefix = "\n".join(lines[:first]).strip()
        if prefix.lower().startswith("if multiple actions are needed"):
            return ""
        return prefix
    return text


def sanitize_generation_text(text: str) -> str:
    """Normalize HF assistant output before OpenClaw delivery or upstream KV storage."""
    if not text:
        return ""
    cleaned = sanitize_chat_template_leaks(text)
    cleaned = _RESPONSE_BLOCK_LEAK_RE.sub("", cleaned)
    cleaned = _NOW_PROCEED_LEAK_RE.sub("", cleaned)
    cleaned = _THINKING_BLOCK_RE.sub("", cleaned)
    cleaned = _THINKING_OPEN_RE.sub("", cleaned)
    cleaned = _collapse_never_call_spam(cleaned)
    cleaned = _TOOL_PREAMBLE_LEAK_RE.sub("", cleaned)
    cleaned = _TOOL_GUIDELINE_PREAMBLE_RE.sub("", cleaned)
    cleaned = _collapse_line_repetition(cleaned)
    cleaned = _collapse_never_call_spam(cleaned)
    if _NEVER_CALL_LINE_RE.match(cleaned.strip()):
        return ""
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


def clawbench_tool_workspace() -> str:
    """OpenClaw read/edit/exec resolve coding task files against this directory."""
    state = os.environ.get("OPENCLAW_STATE_DIR", os.path.expanduser("~/.openclaw"))
    return os.path.join(state, "workspace")


_CLAWBENCH_PYTEST_TARGET = "tests/test_pricing.py"


def _normalize_clawbench_pytest_command(command: str) -> str:
    """Scope bare `pytest -q` to canonical tests/ path (avoids collecting stale run-* trees)."""
    cmd = command.strip()
    if not cmd or "pytest" not in cmd.lower():
        return command
    if re.search(r"\btests/", cmd) or re.search(r"\S+\.py\b", cmd):
        return command
    return f"{cmd} {_CLAWBENCH_PYTEST_TARGET}"


def _normalize_tool_arguments(
    name: str,
    arguments: Any,
    *,
    task_profile: str = "",
    task_id: str = "",
) -> Any:
    if not isinstance(arguments, dict):
        return arguments
    if "path" in arguments:
        path = arguments.get("path")
        if isinstance(path, str) and path.strip():
            fixed = normalize_tool_file_path(path)
            if fixed != path:
                updated = dict(arguments)
                updated["path"] = fixed
                arguments = updated
    if name == "exec" and (task_profile or "").strip().lower() == "clawbench":
        updated = dict(arguments)
        workdir = str(updated.get("workdir") or ".").strip()
        if workdir in (".", "", "./"):
            updated["workdir"] = clawbench_tool_workspace()
        command = str(updated.get("command") or "").strip()
        if command and str(task_id or "").strip() == BUGFIX_DISCOUNT_TASK_ID:
            normalized_cmd = _normalize_clawbench_pytest_command(command)
            if normalized_cmd != command:
                updated["command"] = normalized_cmd
        return updated
    return arguments


def _required_tools_for_agent(
    *,
    agent_index: str | int | None = None,
    agent_role: str = "",
    task_id: str = "",
) -> frozenset[str] | None:
    role = (agent_role or "").strip().lower()
    try:
        idx = int(agent_index) if agent_index is not None else -1
    except (TypeError, ValueError):
        idx = -1
    if str(task_id or "").strip() == BUGFIX_DISCOUNT_TASK_ID and idx in _CLAWBENCH_REQUIRED_BY_INDEX:
        return _CLAWBENCH_REQUIRED_BY_INDEX[idx]
    if any(tag in role for tag in ("extractor", "analyzer")):
        return _ANALYZER_TOOLS
    if any(tag in role for tag in ("patcher",)):
        return _PATCHER_TOOLS
    if any(tag in role for tag in ("formatter", "writer")):
        return _WRITER_TOOL_NAMES
    if any(tag in role for tag in ("reviewer", "verifier")):
        return _VERIFIER_TOOL_NAMES
    return None


def ensure_clawbench_agent_tools(
    tools: list[dict[str, Any]],
    *,
    agent_index: str | int | None = None,
    agent_role: str = "",
    task_profile: str = "",
    task_id: str = "",
) -> list[dict[str, Any]]:
    """Merge role-required tool schemas when OpenClaw sends an incomplete tools array."""
    if (task_profile or "").strip().lower() != "clawbench":
        return tools
    required = _required_tools_for_agent(
        agent_index=agent_index,
        agent_role=agent_role,
        task_id=task_id,
    )
    if not required:
        return tools
    by_name: dict[str, dict[str, Any]] = {}
    for tool in tools:
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = str(fn.get("name") or tool.get("name") or "").strip()
        if name:
            by_name[name] = tool
    for name in required:
        if name not in by_name and name in _FALLBACK_TOOL_SCHEMAS:
            by_name[name] = copy.deepcopy(_FALLBACK_TOOL_SCHEMAS[name])
    return list(by_name.values())


def filter_tools_for_agent(
    tools: list[dict[str, Any]],
    *,
    agent_index: str | int | None = None,
    agent_role: str = "",
    task_profile: str = "",
    task_id: str = "",
) -> list[dict[str, Any]]:
    """Keep only role-relevant tools to shrink generation-boundary injection."""
    if os.environ.get("KVCOMM_TOOL_BRIDGE_MINIMAL", "1").strip().lower() in ("0", "false", "no", "off"):
        return tools
    allowed = _required_tools_for_agent(
        agent_index=agent_index,
        agent_role=agent_role,
        task_id=task_id,
    )
    if not allowed:
        return tools
    filtered: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = str(fn.get("name") or tool.get("name") or "").strip()
        if name in allowed:
            filtered.append(tool)
    if (task_profile or "").strip().lower() == "clawbench":
        return filtered
    return filtered or tools


def should_inject_tools(body: dict[str, Any], *, task_profile: str = "") -> bool:
    """Inject tool schemas at the generation boundary."""
    raw = os.environ.get("KVCOMM_TOOL_INJECT_ON_TURNS", "first_only").strip().lower()
    if raw in ("0", "false", "no", "off", "never"):
        return False
    if raw in ("always", "all", "every"):
        return True
    if (task_profile or "").strip().lower() == "clawbench":
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
    if choice == "none":
        return None, None
    if isinstance(choice, dict) and choice.get("type") == "none":
        return None, None

    normalized = normalize_openai_tools(tools)
    if not normalized:
        return None, None
    # OpenAI default: omitted tool_choice means "auto", not "disabled".
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
        "Use the exact function names from the schema above (e.g. `read`, not `read_file`). "
        "One tool call per <tool_call> block.\n"
        "Do not output instructions, guidelines, or commentary. "
        "Start your response immediately with <tool_call> if a tool call is needed.\n"
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
_LOOSE_TOOL_JSON_RE = re.compile(
    r'\{\s*"name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*"arguments"\s*:\s*(?P<args>\{.*?\})\s*\}',
    re.DOTALL,
)


def _append_tool_call(
    tool_calls: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    task_profile: str = "",
    task_id: str = "",
) -> None:
    name = canonical_tool_name(str(payload.get("name") or payload.get("function") or "").strip())
    if not name:
        return
    arguments = _normalize_tool_arguments(
        name,
        payload.get("arguments"),
        task_profile=task_profile,
        task_id=task_id,
    )
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


def parse_qwen_tool_calls(
    text: str,
    *,
    task_profile: str = "",
    task_id: str = "",
) -> tuple[str, list[dict[str, Any]]]:
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
        for piece in re.split(r"(?=\{)", payload_raw):
            piece = piece.strip()
            if not piece:
                continue
            try:
                payload = json.loads(piece)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                _append_tool_call(tool_calls, payload, task_profile=task_profile, task_id=task_id)

    if not tool_calls:
        for match in _LOOSE_TOOL_JSON_RE.finditer(text):
            try:
                payload = {
                    "name": match.group("name"),
                    "arguments": json.loads(match.group("args")),
                }
            except json.JSONDecodeError:
                continue
            _append_tool_call(tool_calls, payload, task_profile=task_profile, task_id=task_id)

    content_parts.append(text[last_end:])
    content = _TOOL_CALL_RE.sub("", "".join(content_parts)).strip()
    if not content:
        content = None if tool_calls else ""
    return content or "", tool_calls


def openai_message_from_generation(
    raw: str,
    *,
    task_profile: str = "",
    task_id: str = "",
) -> dict[str, Any]:
    """Convert raw HF assistant text into an OpenAI chat completion message."""
    content, tool_calls = parse_qwen_tool_calls(
        sanitize_generation_text(raw or ""),
        task_profile=task_profile,
        task_id=task_id,
    )
    message: dict[str, Any] = {"role": "assistant"}
    if tool_calls:
        # Structured tool_calls only — never leak Qwen `<tool_call>` markup as content.
        message["content"] = None
        message["tool_calls"] = tool_calls
    elif content:
        message["content"] = content
    else:
        message["content"] = None
    return message


def sse_tool_call_deltas(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build OpenAI SSE delta payloads for each tool call.

    OpenClaw's openai-transport-stream accumulates ``function.arguments`` across
    chunks; emit id+name first, then arguments, matching native OpenAI streaming.
    """
    deltas: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = str(fn.get("name") or "").strip()
        arguments = fn.get("arguments")
        if isinstance(arguments, dict):
            args_str = json.dumps(arguments, ensure_ascii=False)
        elif arguments is None:
            args_str = ""
        else:
            args_str = str(arguments)
        deltas.append(
            {
                "index": index,
                "id": tool_call.get("id"),
                "type": tool_call.get("type") or "function",
                "function": {"name": name, "arguments": ""},
            }
        )
        if args_str:
            deltas.append({"index": index, "function": {"arguments": args_str}})
    return deltas


def completion_payload_to_sse(
    payload: dict[str, Any],
    *,
    include_usage: bool = False,
) -> str:
    """Convert a buffered chat.completion into OpenAI SSE for OpenClaw."""
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls")
    chunk_id = payload.get("id") or f"chatcmpl-kvcomm-{uuid.uuid4().hex[:12]}"
    model = payload.get("model") or "kvcomm"
    created = payload.get("created") or int(time.time())
    usage = payload.get("usage")
    finish_reason = choice.get("finish_reason") or "stop"

    def chunk_obj(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    events: list[str] = []
    events.append(f"data: {json.dumps(chunk_obj({'role': 'assistant'}), ensure_ascii=False)}\n\n")

    has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
    if content and not has_tool_calls:
        events.append(f"data: {json.dumps(chunk_obj({'content': content}), ensure_ascii=False)}\n\n")

    if has_tool_calls:
        for delta_tool in sse_tool_call_deltas(tool_calls):
            events.append(
                f"data: {json.dumps(chunk_obj({'tool_calls': [delta_tool]}), ensure_ascii=False)}\n\n"
            )

    events.append(f"data: {json.dumps(chunk_obj({}, finish_reason), ensure_ascii=False)}\n\n")

    if include_usage and isinstance(usage, dict):
        usage_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": usage,
        }
        events.append(f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n")

    events.append("data: [DONE]\n\n")
    return "".join(events)


def tool_bridge_buffered_sse_enabled() -> bool:
    raw = os.environ.get("KVCOMM_TOOL_BRIDGE_BUFFERED_SSE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")
