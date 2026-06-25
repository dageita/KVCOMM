"""Tests for analyzer read-completion detection."""

from __future__ import annotations

from sidecar.openclaw_prefix import (
    analyzer_reads_satisfied,
    completed_read_paths,
    missing_analyzer_reads,
    patcher_fix_satisfied,
    patcher_read_satisfied,
    pricing_discount_fix_applied,
)


def _pricing_read_turn() -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_a",
                    "function": {"name": "read", "arguments": '{"path":"pricing.py"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_a",
            "content": "def apply_discount(subtotal_cents, discount_percent):\n    return subtotal_cents - discount_percent",
        },
    ]


def _cart_read_turn() -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_b",
                    "function": {"name": "read", "arguments": '{"path":"cart.py"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_b",
            "content": "from pricing import apply_discount\n\ndef checkout_total(...): ...",
        },
    ]


def test_completed_read_paths_tracks_successful_reads() -> None:
    messages = [{"role": "user", "content": "task"}, *_pricing_read_turn()]
    assert completed_read_paths(messages) == {"pricing.py"}


def test_analyzer_reads_satisfied_when_both_files_read() -> None:
    messages = [{"role": "user", "content": "task"}, *_pricing_read_turn(), *_cart_read_turn()]
    assert analyzer_reads_satisfied(messages) is True


def test_analyzer_reads_not_satisfied_with_pricing_only() -> None:
    messages = [{"role": "user", "content": "task"}, *_pricing_read_turn()]
    assert analyzer_reads_satisfied(messages) is False


def test_missing_analyzer_reads_after_pricing() -> None:
    messages = [{"role": "user", "content": "task"}, *_pricing_read_turn()]
    assert missing_analyzer_reads(messages) == frozenset({"cart.py"})


def test_patcher_read_satisfied_after_pricing() -> None:
    messages = [{"role": "user", "content": "task"}, *_pricing_read_turn()]
    assert patcher_read_satisfied(messages) is True


def test_pricing_discount_fix_applied_detects_target_return() -> None:
    fixed = "return subtotal_cents * (100 - discount_percent) // 100"
    assert pricing_discount_fix_applied(fixed) is True
    assert pricing_discount_fix_applied("return subtotal_cents - discount_percent") is False


def _pricing_edit_success_turn() -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_e",
                    "function": {
                        "name": "edit",
                        "arguments": '{"path":"pricing.py","edits":[{"oldText":"x","newText":"y"}]}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_e",
            "content": "Successfully replaced 1 block(s) in pricing.py.",
        },
    ]


def test_patcher_fix_satisfied_after_successful_edit() -> None:
    messages = [
        {"role": "user", "content": "task"},
        *_pricing_read_turn(),
        *_pricing_edit_success_turn(),
    ]
    assert patcher_fix_satisfied(messages) is True


def test_patcher_fix_satisfied_when_edit_fails_but_file_already_fixed() -> None:
    messages = [
        {"role": "user", "content": "task"},
        *_pricing_read_turn(),
        *_pricing_edit_success_turn(),
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_f",
                    "function": {
                        "name": "edit",
                        "arguments": '{"path":"pricing.py","edits":[{"oldText":"bad","newText":"good"}]}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_f",
            "content": (
                'Could not find the exact text in pricing.py.\n'
                "Current file contents:\n"
                "def apply_discount(subtotal_cents: int, discount_percent: int) -> int:\n"
                "    return subtotal_cents * (100 - discount_percent) // 100\n"
            ),
        },
    ]
    assert patcher_fix_satisfied(messages) is True
