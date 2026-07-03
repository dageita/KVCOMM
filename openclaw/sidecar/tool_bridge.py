"""Bridge OpenAI tool_calls API to Qwen3 HF generation for the KVCOMM sidecar."""

from __future__ import annotations

import ast
import copy
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from sidecar.bench_prompt_compose import (
    ADD_TESTS_NORMALIZER_TASK_ID,
    BUGFIX_DISCOUNT_TASK_ID,
    CONFIG_LOADER_TASK_ID,
    fix_normalizer_test_imports,
)

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
_BROWSER_ANALYZER_TOOLS = frozenset({"browser", "read"})
_BROWSER_PATCHER_TOOLS = frozenset({"read", "edit", "write", "browser", "exec"})
_BROWSER_VERIFIER_TOOLS = frozenset({"read", "browser", "exec"})
_BROWSER_TOOLS_BY_INDEX = {
    0: _BROWSER_ANALYZER_TOOLS,
    1: _BROWSER_PATCHER_TOOLS,
    2: _BROWSER_VERIFIER_TOOLS,
}
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
    "browser": {
        "type": "function",
        "function": {
            "name": "browser",
            "description": (
                "Control the host browser. Use action open with target host and url to load a page, "
                "then snapshot to inspect the DOM."
            ),
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["open", "snapshot", "tabs", "status", "act"],
                        "description": "Browser operation to perform",
                    },
                    "target": {
                        "type": "string",
                        "enum": ["host", "sandbox", "node"],
                        "description": "Browser target (use host for bench tasks)",
                    },
                    "url": {"type": "string", "description": "URL for action=open"},
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
_TOOL_SAME_TOOL_SPAM_RE = re.compile(
    r"^(?:Do not use the same tool[^\n]*\n){3,}",
    re.MULTILINE | re.IGNORECASE,
)
_TOOL_MULTI_STEP_SPAM_RE = re.compile(
    r"^(?:If multiple steps are needed[^\n]*\n)"
    r"(?:(?:Do not|Only|Never|If no tool)[^\n]*\n){3,}",
    re.MULTILINE | re.IGNORECASE,
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
    cleaned = _TOOL_MULTI_STEP_SPAM_RE.sub("", cleaned)
    cleaned = _TOOL_SAME_TOOL_SPAM_RE.sub("", cleaned)
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


_NORMALIZER_MODULE_ALIASES = frozenset(
    {
        "text_normalization_module.py",
        "text_normalization.py",
        "normalization_module.py",
        "text_normalizer.py",
    }
)


def normalize_tool_file_path(path: str, *, task_id: str = "") -> str:
    """Fix common relative-path mistakes before OpenClaw resolves against workspace."""
    if not path:
        return path
    normalized = path.strip().replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if str(task_id or "").strip() == ADD_TESTS_NORMALIZER_TASK_ID:
        basename = normalized.rsplit("/", 1)[-1]
        if basename in _NORMALIZER_MODULE_ALIASES:
            return "normalizer.py"
        if basename == "test_normalizer.py" and not normalized.startswith("tests/"):
            return "tests/test_normalizer.py"
    if str(task_id or "").strip() == CONFIG_LOADER_TASK_ID:
        basename = normalized.rsplit("/", 1)[-1]
        if basename == "test_config_loader.py" and not normalized.startswith("tests/"):
            return "tests/test_config_loader.py"
    if normalized in _QUICK_NOTE_ALIASES:
        return _QUICK_NOTE_ALIASES[normalized]
    if normalized.startswith("workspace/"):
        stripped = normalized[len("workspace/") :]
        return _QUICK_NOTE_ALIASES.get(stripped, stripped)
    return normalized


def clawbench_tool_workspace(*, workspace_dir: str = "") -> str:
    """OpenClaw read/edit/exec resolve coding task files against this directory."""
    explicit = (workspace_dir or "").strip()
    if explicit and os.path.isdir(explicit):
        return explicit
    state = os.environ.get("OPENCLAW_STATE_DIR", os.path.expanduser("~/.openclaw"))
    return os.path.join(state, "workspace")


def _normalize_clawbench_exec_workdir(workdir: str, *, workspace_dir: str = "") -> str:
    """Map default agent workspace to the registered chain workspace when present."""
    cleaned = (workdir or ".").strip()
    chain_root = clawbench_tool_workspace(workspace_dir=workspace_dir)
    if cleaned in (".", "", "./"):
        return chain_root
    expanded = os.path.normpath(os.path.expanduser(cleaned))
    default_root = os.path.normpath(clawbench_tool_workspace(workspace_dir=""))
    chain_norm = os.path.normpath(chain_root)
    if workspace_dir and chain_norm != default_root and expanded == default_root:
        return chain_root
    return cleaned


_CLAWBENCH_PYTEST_TARGETS = {
    BUGFIX_DISCOUNT_TASK_ID: "tests/test_pricing.py",
    ADD_TESTS_NORMALIZER_TASK_ID: "tests/test_normalizer.py",
    CONFIG_LOADER_TASK_ID: "tests/test_config_loader.py",
}
_CLAWBENCH_PYTEST_PYTHONPATH_TASKS = frozenset(
    {ADD_TESTS_NORMALIZER_TASK_ID, CONFIG_LOADER_TASK_ID}
)


def _default_clawbench_node_paths() -> list[str]:
    paths: list[str] = []
    bench_root = Path(__file__).resolve().parents[3] / "clawbench" / "node_modules"
    if bench_root.is_dir():
        paths.append(str(bench_root))
    openclaw_nm = Path(os.environ.get("OPENCLAW_STATE_DIR", os.path.expanduser("~/.openclaw"))) / "node_modules"
    if openclaw_nm.is_dir():
        paths.append(str(openclaw_nm))
    return paths


def _clawbench_node_path(*, task_vars: dict[str, Any] | None = None) -> str:
    paths: list[str] = []
    for key in ("openclaw_node_path", "benchmark_node_path"):
        val = str((task_vars or {}).get(key) or "").strip()
        if val and os.path.isdir(val) and val not in paths:
            paths.append(val)
    for fallback in _default_clawbench_node_paths():
        if fallback not in paths:
            paths.append(fallback)
    return ":".join(paths)


def _normalize_clawbench_verify_form_command(
    command: str,
    *,
    task_vars: dict[str, Any] | None = None,
) -> str:
    """Ensure verify_form.cjs can resolve playwright via NODE_PATH."""
    cmd = _substitute_runtime_placeholders(command.strip(), task_vars=task_vars)
    if "verify_form" not in cmd:
        return cmd
    node_path = _clawbench_node_path(task_vars=task_vars)
    if not node_path or "NODE_PATH=" in cmd:
        return cmd
    return f"NODE_PATH={node_path} {cmd}"


def _normalize_clawbench_pytest_command(command: str, *, task_id: str = "") -> str:
    """Scope bare `pytest -q` to canonical tests/ path (avoids collecting stale run-* trees)."""
    cmd = command.strip()
    if not cmd or "pytest" not in cmd.lower():
        return command
    if re.search(r"\btests/", cmd) or re.search(r"\S+\.py\b", cmd):
        scoped = cmd
    else:
        target = _CLAWBENCH_PYTEST_TARGETS.get(str(task_id or "").strip())
        if not target:
            return command
        scoped = f"{cmd} {target}"
    if str(task_id or "").strip() in _CLAWBENCH_PYTEST_PYTHONPATH_TASKS:
        if "python -m pytest" not in scoped.lower():
            scoped = re.sub(r"\bpytest\b", "python -m pytest", scoped, count=1)
        if not scoped.startswith("PYTHONPATH=."):
            scoped = f"PYTHONPATH=. {scoped}"
    return scoped


def _is_normalizer_test_path(path: str) -> bool:
    cleaned = (path or "").strip().replace("\\", "/").lstrip("./")
    return cleaned in {"tests/test_normalizer.py", "test_normalizer.py"} or cleaned.endswith(
        "/test_normalizer.py"
    )


def _set_file_immutable(path: str, *, immutable: bool) -> None:
    """Toggle Linux immutable bit when chattr is available (bench protects tests with +i)."""
    if not os.path.exists(path):
        return
    flag = "+i" if immutable else "-i"
    try:
        subprocess.run(
            ["chattr", flag, path],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError):
        pass


def _clear_immutable_path(path: str) -> None:
    """Clear chattr +i on a file or all files under a directory tree."""
    if not os.path.exists(path):
        return
    if os.path.isfile(path):
        _set_file_immutable(path, immutable=False)
        return
    for root, _dirs, files in os.walk(path):
        for name in files:
            _set_file_immutable(os.path.join(root, name), immutable=False)


def _copy_clawbench_file(src: str, dst: str) -> None:
    """Copy a bench workspace file, clearing destination immutable flags first."""
    _clear_immutable_path(dst)
    if os.path.dirname(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def sync_clawbench_coding_default_to_chain(*, workspace_dir: str = "") -> bool:
    """Copy editable .py modules from default OpenClaw cwd into chain workspace."""
    chain_root = clawbench_tool_workspace(workspace_dir=workspace_dir)
    if not (workspace_dir or "").strip() or not os.path.isdir(chain_root):
        return False
    default_root = clawbench_tool_workspace(workspace_dir="")
    if not os.path.isdir(default_root):
        return False
    changed = False
    for name in os.listdir(default_root):
        if not name.endswith(".py") or name.startswith("verify_"):
            continue
        src = os.path.join(default_root, name)
        dst = os.path.join(chain_root, name)
        if not os.path.isfile(src):
            continue
        if not os.path.isfile(dst):
            _copy_clawbench_file(src, dst)
            changed = True
            continue
        with open(src, encoding="utf-8") as src_handle, open(dst, encoding="utf-8") as dst_handle:
            src_content = src_handle.read()
            dst_content = dst_handle.read()
        if src_content != dst_content or os.path.getmtime(src) > os.path.getmtime(dst):
            _copy_clawbench_file(src, dst)
            changed = True
    return changed


def sync_clawbench_config_loader_default_to_chain(*, workspace_dir: str = "") -> bool:
    """Ensure chain workspace has latest config-loader modules before pytest."""
    return sync_clawbench_coding_default_to_chain(workspace_dir=workspace_dir)


def sync_clawbench_tests_default_to_chain(*, workspace_dir: str = "") -> bool:
    """Copy tests/ from default OpenClaw cwd into chain workspace (write tool lands in default)."""
    chain_root = clawbench_tool_workspace(workspace_dir=workspace_dir)
    if not (workspace_dir or "").strip() or not os.path.isdir(chain_root):
        return False
    default_root = clawbench_tool_workspace(workspace_dir="")
    default_tests = os.path.join(default_root, "tests")
    chain_tests = os.path.join(chain_root, "tests")
    if not os.path.isdir(default_tests):
        return False
    os.makedirs(chain_tests, exist_ok=True)
    changed = False
    for name in os.listdir(default_tests):
        if not name.endswith(".py"):
            continue
        src = os.path.join(default_tests, name)
        dst = os.path.join(chain_tests, name)
        if not os.path.isfile(src):
            continue
        if not os.path.isfile(dst):
            _copy_clawbench_file(src, dst)
            changed = True
            continue
        with open(src, encoding="utf-8") as src_handle, open(dst, encoding="utf-8") as dst_handle:
            src_content = src_handle.read()
            dst_content = dst_handle.read()
        if src_content != dst_content or os.path.getmtime(src) > os.path.getmtime(dst):
            _copy_clawbench_file(src, dst)
            changed = True
    return changed


_BROWSER_FRONTEND_FILES = ("index.html", "app.js")


def browser_form_fixed_on_disk(*, workspace_dir: str = "") -> bool:
    """True when app.js uses contact-form instead of the typo contact-formm."""
    root = clawbench_tool_workspace(workspace_dir=workspace_dir)
    path = os.path.join(root, "app.js")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as handle:
        content = handle.read()
    return "contact-formm" not in content and "contact-form" in content


def sync_clawbench_browser_workspaces(*, workspace_dir: str = "", prefer_default: bool = False) -> bool:
    """Sync browser frontend files between chain and default OpenClaw workspaces.

    When ``prefer_default`` is False (analyzer/patcher agents), never import the
    default workspace into the chain run directory — a stale fixed ``app.js`` in
    the default cwd would skip the read→edit trajectory. Only push chain→default
    when the chain copy is newer. When ``prefer_default`` is True (verifier), pull
    default→chain so exec/browser tools see edits that landed in the default cwd.
    """
    chain_root = clawbench_tool_workspace(workspace_dir=workspace_dir)
    if not (workspace_dir or "").strip() or not os.path.isdir(chain_root):
        return False
    default_root = clawbench_tool_workspace(workspace_dir="")
    os.makedirs(default_root, exist_ok=True)
    changed = False
    for name in _BROWSER_FRONTEND_FILES:
        chain_path = os.path.join(chain_root, name)
        default_path = os.path.join(default_root, name)
        chain_exists = os.path.isfile(chain_path)
        default_exists = os.path.isfile(default_path)
        if chain_exists and not default_exists:
            shutil.copy2(chain_path, default_path)
            changed = True
            continue
        if default_exists and not chain_exists:
            shutil.copy2(default_path, chain_path)
            changed = True
            continue
        if not chain_exists or not default_exists:
            continue
        chain_mtime = os.path.getmtime(chain_path)
        default_mtime = os.path.getmtime(default_path)
        if prefer_default:
            if default_mtime >= chain_mtime:
                shutil.copy2(default_path, chain_path)
                changed = True
        elif chain_mtime > default_mtime:
            shutil.copy2(chain_path, default_path)
            changed = True
    return changed


_BROWSER_BROKEN_APP_JS = """const form = document.getElementById("contact-formm");
const emailInput = document.getElementById("email");
const statusNode = document.getElementById("status");

if (form) {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const email = emailInput.value.trim();
    if (!email.includes("@")) {
      statusNode.textContent = "Enter a valid email.";
      return;
    }
    statusNode.textContent = `Saved ${email}`;
  });
}
"""


def restore_browser_form_broken_on_disk(*, workspace_dir: str = "") -> bool:
    """Reset chain workspace app.js to the broken bench template for a fresh run."""
    chain_root = clawbench_tool_workspace(workspace_dir=workspace_dir)
    if not (workspace_dir or "").strip() or not os.path.isdir(chain_root):
        return False
    path = os.path.join(chain_root, "app.js")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            existing = handle.read()
        if existing == _BROWSER_BROKEN_APP_JS:
            return False
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_BROWSER_BROKEN_APP_JS)
    return True


def sync_clawbench_browser_default_to_chain(*, workspace_dir: str = "") -> bool:
    """Copy browser frontend files from chain workspace into default OpenClaw cwd."""
    return sync_clawbench_browser_workspaces(workspace_dir=workspace_dir, prefer_default=False)


def fix_normalizer_test_file_on_disk(*, workspace_dir: str = "") -> bool:
    """Fix relative imports in tests/test_normalizer.py on disk before pytest."""
    sync_clawbench_tests_default_to_chain(workspace_dir=workspace_dir)
    root = clawbench_tool_workspace(workspace_dir=workspace_dir)
    path = os.path.join(root, "tests", "test_normalizer.py")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as handle:
        content = handle.read()
    fixed = fix_normalizer_test_imports(content)
    if fixed == content:
        return False
    _clear_immutable_path(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(fixed)
    return True


def _sanitize_normalizer_test_source_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Rewrite broken relative imports in write content or edit newText."""
    content = str(arguments.get("content") or "")
    if content:
        fixed = fix_normalizer_test_imports(content)
        if fixed != content:
            updated = dict(arguments)
            updated["content"] = fixed
            return updated
    edits = arguments.get("edits")
    if not isinstance(edits, list):
        return arguments
    changed = False
    kept: list[Any] = []
    for edit in edits:
        if not isinstance(edit, dict):
            kept.append(edit)
            continue
        new_text = str(edit.get("newText") or edit.get("new_text") or "")
        fixed_text = fix_normalizer_test_imports(new_text)
        if fixed_text != new_text:
            updated_edit = dict(edit)
            if "newText" in updated_edit:
                updated_edit["newText"] = fixed_text
            if "new_text" in updated_edit:
                updated_edit["new_text"] = fixed_text
            kept.append(updated_edit)
            changed = True
        else:
            kept.append(edit)
    if not changed:
        return arguments
    updated = dict(arguments)
    updated["edits"] = kept
    return updated


def _substitute_runtime_placeholders(value: str, *, task_vars: dict[str, Any] | None = None) -> str:
    text = str(value or "")
    for key, raw in (task_vars or {}).items():
        placeholder = "{" + str(key) + "}"
        if placeholder in text and raw is not None:
            text = text.replace(placeholder, str(raw))
    return text


def _normalize_browser_arguments(
    arguments: dict[str, Any],
    *,
    task_vars: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(arguments)
    for field in ("url", "targetUrl"):
        if field in updated:
            updated[field] = _substitute_runtime_placeholders(str(updated.get(field) or ""), task_vars=task_vars)
    if not str(updated.get("action") or "").strip():
        if updated.get("url") or updated.get("targetUrl"):
            updated["action"] = "open"
    if not str(updated.get("target") or "").strip():
        updated["target"] = "host"
    return updated


def _normalize_tool_arguments(
    name: str,
    arguments: Any,
    *,
    task_profile: str = "",
    task_id: str = "",
    workspace_dir: str = "",
    task_vars: dict[str, Any] | None = None,
) -> Any:
    if not isinstance(arguments, dict):
        return arguments
    if name == "browser":
        return _normalize_browser_arguments(arguments, task_vars=task_vars)
    if "path" in arguments:
        path = arguments.get("path")
        if isinstance(path, str) and path.strip():
            fixed = normalize_tool_file_path(path, task_id=str(task_id or ""))
            if fixed != path:
                updated = dict(arguments)
                updated["path"] = fixed
                arguments = updated
    if str(task_id or "").strip() == ADD_TESTS_NORMALIZER_TASK_ID:
        path = str(arguments.get("path") or "")
        if name in {"edit", "write"} and _is_normalizer_test_path(path):
            arguments = _sanitize_normalizer_test_source_arguments(arguments)
    if name == "exec" and (task_profile or "").strip().lower() == "clawbench":
        updated = dict(arguments)
        workdir = str(updated.get("workdir") or ".").strip()
        updated["workdir"] = _normalize_clawbench_exec_workdir(
            workdir, workspace_dir=workspace_dir
        )
        command = str(updated.get("command") or "").strip()
        if command:
            command = _substitute_runtime_placeholders(command, task_vars=task_vars)
            verify_cmd = _normalize_clawbench_verify_form_command(command, task_vars=task_vars)
            if verify_cmd != command:
                updated["command"] = verify_cmd
                command = verify_cmd
        if command and str(task_id or "").strip() in _CLAWBENCH_PYTEST_TARGETS:
            normalized_cmd = _normalize_clawbench_pytest_command(command, task_id=str(task_id or ""))
            if normalized_cmd != command:
                updated["command"] = normalized_cmd
            if str(task_id or "").strip() == ADD_TESTS_NORMALIZER_TASK_ID:
                fix_normalizer_test_file_on_disk(workspace_dir=workspace_dir)
        return updated
    return arguments


def _required_tools_for_agent(
    *,
    agent_index: str | int | None = None,
    agent_role: str = "",
    task_id: str = "",
    clawbench_family: str = "",
) -> frozenset[str] | None:
    role = (agent_role or "").strip().lower()
    try:
        idx = int(agent_index) if agent_index is not None else -1
    except (TypeError, ValueError):
        idx = -1
    if str(clawbench_family or "").strip() == "browser" and idx in _BROWSER_TOOLS_BY_INDEX:
        return _BROWSER_TOOLS_BY_INDEX[idx]
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
    clawbench_family: str = "",
) -> list[dict[str, Any]]:
    """Merge role-required tool schemas when OpenClaw sends an incomplete tools array."""
    if (task_profile or "").strip().lower() != "clawbench":
        return tools
    required = _required_tools_for_agent(
        agent_index=agent_index,
        agent_role=agent_role,
        task_id=task_id,
        clawbench_family=clawbench_family,
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
    clawbench_family: str = "",
) -> list[dict[str, Any]]:
    """Keep only role-relevant tools to shrink generation-boundary injection."""
    if os.environ.get("KVCOMM_TOOL_BRIDGE_MINIMAL", "1").strip().lower() in ("0", "false", "no", "off"):
        return tools
    allowed = _required_tools_for_agent(
        agent_index=agent_index,
        agent_role=agent_role,
        task_id=task_id,
        clawbench_family=clawbench_family,
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


def _parse_tool_call_payload(piece: str) -> dict[str, Any] | None:
    """Parse a tool_call JSON object, tolerating Python-style single-quoted strings."""
    stripped = (piece or "").strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return None
    return payload if isinstance(payload, dict) else None


def _append_tool_call(
    tool_calls: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    task_profile: str = "",
    task_id: str = "",
    workspace_dir: str = "",
    task_vars: dict[str, Any] | None = None,
) -> None:
    name = canonical_tool_name(str(payload.get("name") or payload.get("function") or "").strip())
    if not name:
        return
    arguments = _normalize_tool_arguments(
        name,
        payload.get("arguments"),
        task_profile=task_profile,
        task_id=task_id,
        workspace_dir=workspace_dir,
        task_vars=task_vars,
    )
    if name == "edit" and isinstance(arguments, dict) and arguments.get("edits") == []:
        return
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
    workspace_dir: str = "",
    task_vars: dict[str, Any] | None = None,
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
                payload = _parse_tool_call_payload(piece)
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict):
                _append_tool_call(
                    tool_calls,
                    payload,
                    task_profile=task_profile,
                    task_id=task_id,
                    workspace_dir=workspace_dir,
                    task_vars=task_vars,
                )

    if not tool_calls:
        for match in _LOOSE_TOOL_JSON_RE.finditer(text):
            try:
                payload = {
                    "name": match.group("name"),
                    "arguments": json.loads(match.group("args")),
                }
            except json.JSONDecodeError:
                continue
            _append_tool_call(
                tool_calls,
                payload,
                task_profile=task_profile,
                task_id=task_id,
                workspace_dir=workspace_dir,
                task_vars=task_vars,
            )

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
    workspace_dir: str = "",
    task_vars: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert raw HF assistant text into an OpenAI chat completion message."""
    content, tool_calls = parse_qwen_tool_calls(
        sanitize_generation_text(raw or ""),
        task_profile=task_profile,
        task_id=task_id,
        workspace_dir=workspace_dir,
        task_vars=task_vars,
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
