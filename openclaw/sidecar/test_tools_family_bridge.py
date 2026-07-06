"""Tests for tools-family clawbench tool bridge (search/copy rewrite)."""

from __future__ import annotations

import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from sidecar.tool_bridge import filter_tools_for_agent, parse_qwen_tool_calls


def test_filter_tools_family_extractor_gets_read_and_exec() -> None:
    tools = [
        {"type": "function", "function": {"name": "read", "parameters": {}}},
        {"type": "function", "function": {"name": "write", "parameters": {}}},
        {"type": "function", "function": {"name": "exec", "parameters": {}}},
    ]
    filtered = filter_tools_for_agent(
        tools,
        agent_index=0,
        agent_role="Extractor",
        task_profile="clawbench",
        task_id="t2-fs-find-that-thing",
        clawbench_family="tools",
    )
    names = {str((t.get("function") or {}).get("name") or "") for t in filtered}
    assert names == {"read", "exec"}


def test_parse_search_tool_call_rewrites_to_exec() -> None:
    raw = (
        '<tool_call>{"name": "search", "arguments": '
        '{"query": "Q3 marketing budget regional breakdown"}}</tool_call>'
    )
    _content, calls = parse_qwen_tool_calls(raw, task_profile="clawbench")
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "exec"
    assert "rg -l -i" in fn["arguments"]
    assert "Q3 marketing budget regional breakdown" in fn["arguments"]


def test_parse_copy_tool_call_rewrites_to_exec() -> None:
    raw = (
        '<tool_call>{"name": "copy", "arguments": '
        '{"source": "docs/budget.xlsx", "destination": "desktop/out.xlsx"}}</tool_call>'
    )
    _content, calls = parse_qwen_tool_calls(raw, task_profile="clawbench")
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "exec"
    assert "cp " in fn["arguments"]
    assert "docs/budget.xlsx" in fn["arguments"]
    assert "desktop/out.xlsx" in fn["arguments"]
