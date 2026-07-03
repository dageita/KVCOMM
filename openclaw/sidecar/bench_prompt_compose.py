"""Compose bench prefix templates: short agent_tasks + sidecar-injected tool_constraints."""

from __future__ import annotations

import re

_TOOL_CONSTRAINTS_MARKER = "{{tool_constraints}}"
_JOB_MARKER_RE = re.compile(r"\nYour job \(Agent ", re.IGNORECASE)

BUGFIX_DISCOUNT_TASK_ID = "t1-bugfix-discount"
QUICK_NOTE_TASK_ID = "t1-fs-quick-note"
ADD_TESTS_NORMALIZER_TASK_ID = "t2-add-tests-normalizer"
CONFIG_LOADER_TASK_ID = "t2-config-loader"
BROWSER_FORM_FIX_TASK_ID = "t2-browser-form-fix"
QUICK_NOTE_VERIFIER_READ = frozenset({"quick_note.md"})


def fix_normalizer_test_imports(content: str) -> str:
    """Rewrite broken package-relative imports to flat `from normalizer import ...`."""
    text = content or ""
    text = re.sub(
        r"from\s+\.\.normalizer\s+import",
        "from normalizer import",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"from\s+\.\.\s+import\s+normalize_title,\s*normalize_tags",
        "from normalizer import normalize_title, normalize_tags",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"from\s+\.\.\s+import\s+normalizer\b",
        "from normalizer import",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"from\s+openclaw\.normalizer\s+import\s+normalize_text\b",
        "from normalizer import normalize_title, normalize_tags",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"from\s+openclaw\.normalizer\s+import",
        "from normalizer import",
        text,
        flags=re.IGNORECASE,
    )
    return text


def is_bugfix_discount_task(ctx) -> bool:
    """True when the active bench row is the tier1 coding bugfix task."""
    if ctx is None:
        return False
    return str(getattr(ctx, "task_id", "") or "").strip() == BUGFIX_DISCOUNT_TASK_ID


def is_quick_note_task(ctx) -> bool:
    """True when the active bench row is the tier1 quick-note filesystem task."""
    if ctx is None:
        return False
    return str(getattr(ctx, "task_id", "") or "").strip() == QUICK_NOTE_TASK_ID


def is_add_tests_normalizer_task(ctx) -> bool:
    """True when the active bench row is the tier2 normalizer test-authoring task."""
    if ctx is None:
        return False
    return str(getattr(ctx, "task_id", "") or "").strip() == ADD_TESTS_NORMALIZER_TASK_ID


def is_config_loader_task(ctx) -> bool:
    """True when the active bench row is the tier2 config-loader repo task."""
    if ctx is None:
        return False
    return str(getattr(ctx, "task_id", "") or "").strip() == CONFIG_LOADER_TASK_ID


def is_browser_family_task(ctx) -> bool:
    """True when the active bench row is a browser-family ClawBench task."""
    if ctx is None:
        return False
    return str(getattr(ctx, "clawbench_family", "") or "").strip() == "browser"


def inject_tool_constraints(user_template: str, tool_constraints: str | None) -> str:
    """Insert bench-only tool/path constraints without changing prod agent_tasks text."""
    constraints = (tool_constraints or "").strip()
    if not constraints:
        return (user_template or "").strip()

    text = (user_template or "").strip()
    if _TOOL_CONSTRAINTS_MARKER in text:
        return text.replace(_TOOL_CONSTRAINTS_MARKER, constraints)

    match = _JOB_MARKER_RE.search(text)
    if match:
        before = text[: match.start()].rstrip()
        after = text[match.start() :].lstrip("\n")
        return f"{before}\n\n{constraints}\n\n{after}"

    return f"{text}\n\n{constraints}"


def tool_constraints_for_context(ctx) -> str:
    """Resolve tool_constraints from KvcommContext vars (set by bench register)."""
    if ctx is None:
        return ""
    raw = (ctx.vars or {}).get("tool_constraints") or (ctx.vars or {}).get("toolConstraints")
    return str(raw or "").strip()
