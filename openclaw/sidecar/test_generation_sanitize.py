"""Tests for generation text sanitization."""

from __future__ import annotations

from sidecar.tool_bridge import sanitize_generation_text


def test_strips_thinking_and_template_leaks() -> None:
    raw = (
        "<|redacted_thinking|>hidden reasoning<|/redacted_thinking|>\n"
        "The analysis text."
    )
    assert sanitize_generation_text(raw) == "The analysis text."


def test_collapses_repetition_tail() -> None:
    line = "Do not call functions with incorrect parameters."
    tail = "\n".join([line] * 8)
    cleaned = sanitize_generation_text(tail)
    assert cleaned == line


def test_strips_never_call_spam() -> None:
    lines = [
        "If multiple actions are needed, make separate tool calls in sequence.",
        "Never call a function that isn't listed in the tools section.",
        "Never call a function twice in the same response.",
        "Never call a function with parameters that don't match its schema.",
        "Never call a function with invalid JSON.",
    ]
    raw = "\n".join(lines)
    assert sanitize_generation_text(raw) == ""


def test_strips_response_block_template_leak() -> None:
    raw = (
        "After all tool calls, provide the final response in a <response> block.\n"
        "If no tools are needed, respond directly in the <response> block.\n\n"
        "Now proceed to solve the problem.\n"
        'tool_call_start\n{"name": "read", "arguments": {"path": "pricing.py"}}\ntool_call_end'
    )
    cleaned = sanitize_generation_text(raw)
    assert "response> block" not in cleaned
    assert "read" in cleaned


def test_strips_tool_guideline_preamble() -> None:
    raw = (
        "If multiple actions are needed, make separate tool calls in sequence.\n"
        "Only call functions that are explicitly defined in the tools section.\n"
        "Do not make assumptions about file contents or structure beyond what's visible.\n"
        "Do not make multiple overlapping edits in the same file in a single call.\n"
        "Merge adjacent or overlapping edits into a single edit when possible.\n"
        "When editing files, ensure oldText matches exactly what exists in the file.\n"
        "Do not add extra text outside the edits unless necessary for context.\n"
        "For new files, use write() rather than edit().\n"
        'tool_call_start\n{"name": "write", "arguments": {"path": "notes/quick_note.md", "content": "test"}}\ntool_call_end'
    )
    cleaned = sanitize_generation_text(raw)
    assert "If multiple actions are needed" not in cleaned
    assert "Only call functions" not in cleaned
    assert "write" in cleaned
