"""Tests for bench prompt composition (tool_constraints injection)."""

from __future__ import annotations

import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from sidecar.bench_prompt_compose import BUGFIX_DISCOUNT_TASK_ID, QUICK_NOTE_TASK_ID, inject_tool_constraints, is_bugfix_discount_task, is_quick_note_task
from sidecar.kvcomm_adapter import KvcommContext
from sidecar.tool_bridge import _required_tools_for_agent


def test_inject_before_your_job_marker() -> None:
    base = (
        "{{role_prompt}}\n\nUser request:\n{{task_body}}\n\n"
        "Output from Agent 0:\n\n{agent_0_current}\n\n"
        "Your job (Agent 1 - Patcher): Fix pricing.py.\n"
    )
    constraints = "Workspace: edit pricing.py ONLY."
    out = inject_tool_constraints(base, constraints)
    assert "Workspace: edit pricing.py ONLY." in out
    assert out.index("Workspace") < out.index("Your job (Agent 1")


def test_inject_explicit_placeholder() -> None:
    base = "User request:\n{{task_body}}\n\n{{tool_constraints}}\n\nYour job (Agent 0): analyze.\n"
    out = inject_tool_constraints(base, "Use relative paths only.")
    assert "{{tool_constraints}}" not in out
    assert "Use relative paths only." in out


def test_is_bugfix_discount_task_matches_task_id() -> None:
    ctx = KvcommContext(
        run_id="r",
        agent_index="0",
        mode="dense_prefill",
        message_key="m",
        task_id=BUGFIX_DISCOUNT_TASK_ID,
        task_profile="clawbench",
    )
    assert is_bugfix_discount_task(ctx) is True
    ctx.task_id = QUICK_NOTE_TASK_ID
    assert is_bugfix_discount_task(ctx) is False
    assert is_quick_note_task(ctx) is True


def test_required_tools_use_role_for_non_bugfix_tasks() -> None:
    writer_tools = _required_tools_for_agent(
        agent_index=1,
        agent_role="Writer",
        task_id="t1-fs-quick-note",
    )
    assert writer_tools == frozenset({"write", "edit"})
    verifier_tools = _required_tools_for_agent(
        agent_index=2,
        agent_role="Verifier",
        task_id="t1-fs-quick-note",
    )
    assert verifier_tools == frozenset({"read", "write", "edit"})
    assert "exec" not in verifier_tools
    bugfix_tools = _required_tools_for_agent(
        agent_index=1,
        agent_role="Writer",
        task_id=BUGFIX_DISCOUNT_TASK_ID,
    )
    assert "read" in bugfix_tools


def test_required_tools_tools_family_includes_exec() -> None:
    extractor = _required_tools_for_agent(
        agent_index=0,
        agent_role="Extractor",
        task_id="t2-fs-find-that-thing",
        clawbench_family="tools",
    )
    assert extractor == frozenset({"read", "exec"})
    writer = _required_tools_for_agent(
        agent_index=1,
        agent_role="Writer",
        task_id="t2-fs-find-that-thing",
        clawbench_family="tools",
    )
    assert writer == frozenset({"read", "write", "edit", "exec"})
    verifier = _required_tools_for_agent(
        agent_index=2,
        agent_role="Verifier",
        task_id="t2-fs-find-that-thing",
        clawbench_family="tools",
    )
    assert verifier == frozenset({"read", "exec"})
