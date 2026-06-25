"""Tests for Agent 2 verifier gate detection."""

from __future__ import annotations

from sidecar.openclaw_prefix import (
    patcher_read_satisfied,
    verifier_exec_pytest_done,
    verifier_pytest_passed,
    verifier_should_force_edit,
    verifier_should_force_exec,
    verifier_should_force_read,
)


def _exec_pytest_turn(*, passed: bool) -> list[dict]:
    body = "1 passed in 0.01s" if passed else "1 failed, 0 passed"
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_x",
                    "function": {"name": "exec", "arguments": '{"command":"pytest -q"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_x", "content": body},
    ]


def _pricing_read_turn() -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_r",
                    "function": {"name": "read", "arguments": '{"path":"pricing.py"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_r",
            "content": "def apply_discount(...):\n    return subtotal_cents - discount_percent",
        },
    ]


def test_verifier_should_force_exec_on_turn_zero() -> None:
    messages = [{"role": "user", "content": "task"}]
    assert verifier_should_force_exec(messages) is True
    assert verifier_exec_pytest_done(messages) is False


def test_verifier_should_force_exec_after_read_without_pytest() -> None:
    messages = [{"role": "user", "content": "task"}, *_pricing_read_turn()]
    assert patcher_read_satisfied(messages) is True
    assert verifier_should_force_exec(messages) is True


def test_verifier_pytest_passed_detects_success() -> None:
    messages = [{"role": "user", "content": "task"}, *_exec_pytest_turn(passed=True)]
    assert verifier_pytest_passed(messages) is True
    assert verifier_should_force_exec(messages) is False


def test_verifier_should_force_read_after_failed_pytest() -> None:
    messages = [{"role": "user", "content": "task"}, *_exec_pytest_turn(passed=False)]
    assert verifier_should_force_read(messages) is True
    assert verifier_should_force_edit(messages) is False


def test_verifier_should_force_edit_after_failed_pytest_and_read() -> None:
    messages = [
        {"role": "user", "content": "task"},
        *_exec_pytest_turn(passed=False),
        *_pricing_read_turn(),
    ]
    assert verifier_should_force_read(messages) is False
    assert verifier_should_force_edit(messages) is True
    assert verifier_should_force_exec(messages) is False
