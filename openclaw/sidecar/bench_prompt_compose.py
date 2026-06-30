"""Compose bench prefix templates: short agent_tasks + sidecar-injected tool_constraints."""

from __future__ import annotations

import re

_TOOL_CONSTRAINTS_MARKER = "{{tool_constraints}}"
_JOB_MARKER_RE = re.compile(r"\nYour job \(Agent ", re.IGNORECASE)

BUGFIX_DISCOUNT_TASK_ID = "t1-bugfix-discount"
QUICK_NOTE_TASK_ID = "t1-fs-quick-note"
QUICK_NOTE_VERIFIER_READ = frozenset({"quick_note.md"})


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
