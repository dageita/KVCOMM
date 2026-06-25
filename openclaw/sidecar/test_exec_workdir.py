"""Tests for clawbench exec workdir normalization."""

from __future__ import annotations

import json

from sidecar.tool_bridge import clawbench_tool_workspace, openai_message_from_generation


def test_exec_workdir_defaults_to_openclaw_workspace_for_clawbench() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q tests/test_pricing.py", "workdir": "."}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(raw, task_profile="clawbench")
    tool_calls = message.get("tool_calls") or []
    assert len(tool_calls) == 1
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["workdir"] == clawbench_tool_workspace()
    assert args["command"] == "pytest -q tests/test_pricing.py"


def test_exec_workdir_unchanged_for_non_clawbench() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q", "workdir": "."}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(raw, task_profile="copy")
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == "."
    assert args["command"] == "pytest -q"


def test_exec_pytest_scoped_to_tests_path_for_clawbench() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q", "workdir": "."}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(raw, task_profile="clawbench")
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == clawbench_tool_workspace()
    assert args["command"] == "pytest -q tests/test_pricing.py"
