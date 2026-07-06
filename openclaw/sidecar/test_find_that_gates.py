"""Tests for t2-fs-find-that-thing gate helpers."""

from __future__ import annotations

from sidecar.openclaw_prefix import (
    find_that_copy_satisfied,
    find_that_source_located,
    find_that_verifier_passed,
)
from sidecar.tool_bridge import _normalize_find_that_exec_command, parse_qwen_tool_calls


def _exec_call(command: str, *, call_id: str = "call_exec", body: str = "ok") -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "exec", "arguments": f'{{"command": "{command}"}}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": body},
    ]


def test_find_that_source_located_after_find_exec() -> None:
    messages = _exec_call(
        'find . -name "*q3*marketing*budget*.xlsx"',
        body="./Documents/q3_marketing_budget_v3.xlsx",
    )
    assert find_that_source_located(messages) is True


def test_find_that_copy_satisfied_after_desktop_cp() -> None:
    messages = _exec_call(
        "mkdir -p Desktop && cp Documents/q3_marketing_budget_v3.xlsx Desktop/q3_marketing_budget.xlsx",
    )
    assert find_that_copy_satisfied(messages) is True


def test_find_that_copy_not_satisfied_on_failed_cp() -> None:
    messages = _exec_call(
        "cp Documents/q3_marketing_budget_v3.xlsx ~/desktop/q3_marketing_budget.xlsx",
        body="cp: cannot create regular file '/root/desktop/q3_marketing_budget.xlsx': No such file or directory\n(Command exited with code 1)",
    )
    assert find_that_copy_satisfied(messages) is False


def test_find_that_verifier_passed() -> None:
    messages = _exec_call(
        "python3 verify_correct_file.py",
        body="PASS: agent surfaced Q3 marketing budget content at/in Desktop/q3_marketing_budget.xlsx",
    )
    assert find_that_verifier_passed(messages) is True


def test_normalize_find_that_exec_rewrites_desktop_path() -> None:
    cmd = "cp ./Documents/q3_marketing_budget_v3.xlsx ~/desktop/q3_marketing_budget.xlsx"
    normalized = _normalize_find_that_exec_command(cmd)
    assert "Desktop/q3_marketing_budget.xlsx" in normalized
    assert "mkdir -p Desktop" in normalized
    assert "~/desktop" not in normalized.lower()


def test_parse_copy_rewrites_to_desktop_exec() -> None:
    raw = (
        '<tool_call>{"name": "copy", "arguments": '
        '{"source": "Documents/q3_marketing_budget_v3.xlsx", '
        '"destination": "/desktop/q3_marketing_budget.xlsx"}}</tool_call>'
    )
    _content, calls = parse_qwen_tool_calls(raw, task_profile="clawbench")
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "exec"
    assert "Desktop/q3_marketing_budget.xlsx" in fn["arguments"]
    assert "mkdir -p Desktop" in fn["arguments"]
