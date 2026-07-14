"""Tests for generation text sanitization."""

from __future__ import annotations

from sidecar.tool_bridge import (
    is_degenerate_tool_guideline_spam,
    is_unusable_tool_generation_text,
    sanitize_generation_text,
    strip_tool_call_preamble,
    tool_guideline_spam_without_tool_call,
)


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


def test_strips_same_tool_spam() -> None:
    line = "Do not use the same tool more than once unless necessary."
    raw = (
        "If multiple steps are needed, make one tool call per step. Do not combine multiple actions in one tool call.\n"
        + "\n".join([line] * 10)
    )
    cleaned = sanitize_generation_text(raw)
    assert cleaned.count(line) <= 2


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


def test_detects_if_no_tool_call_spam() -> None:
    lines = [
        "If no tool call is needed, respond directly to the user's question with the final answer inside <answer> tags.",
        "If multiple steps are needed, make one tool call at a time and wait for the result before proceeding.",
        "If you're unsure about the correct action, ask for clarification from the user.",
        "If an error occurs during execution, handle it gracefully and inform the user.",
        "If the task is completed successfully, provide the final answer to the user.",
        "If the task requires multiple actions, perform them sequentially and update the user after each step.",
    ]
    raw = "\n".join(lines)
    assert is_degenerate_tool_guideline_spam(raw)
    cleaned = sanitize_generation_text(raw)
    assert "If no tool call is needed" not in cleaned
    assert "If multiple steps are needed" not in cleaned


def test_does_not_flag_valid_tool_call() -> None:
    raw = (
        "If no tool call is needed, respond directly.\n"
        '<tool_call>\n{"name": "exec", "arguments": {"command": "cp a b"}}\n</tool_call>'
    )
    assert not is_degenerate_tool_guideline_spam(raw)


def test_empty_string_spam_without_tool_call() -> None:
    raw = "If no action is needed, respond with an empty string."
    assert is_degenerate_tool_guideline_spam(raw)
    assert tool_guideline_spam_without_tool_call(raw)


def test_preamble_spam_with_tool_call_is_not_without_tool_call() -> None:
    lines = [
        "If no tool call is needed, respond directly to the user with the final answer.",
        "If multiple steps are needed, make sure to call the appropriate functions in sequence.",
        "Make sure to use the correct parameters for each function.",
        "Make sure to use the correct JSON format for the arguments.",
        "Now proceed to solve the problem.",
        "Only use functions listed in the tools section.",
    ]
    raw = "\n".join(lines) + '\n<tool_call>\n{"name": "exec", "arguments": {"command": "ls"}}\n</tool_call>'
    assert is_degenerate_tool_guideline_spam(raw)
    assert not tool_guideline_spam_without_tool_call(raw)
    stripped = strip_tool_call_preamble(raw)
    assert "If no tool call is needed" not in stripped
    assert stripped.lstrip().startswith("<tool_call>")


def test_strip_tool_call_preamble_keeps_short_prefix() -> None:
    raw = (
        "I'll list the directory.\n"
        '<tool_call>\n{"name": "exec", "arguments": {"command": "ls"}}\n</tool_call>'
    )
    assert strip_tool_call_preamble(raw) == raw


def test_openai_message_strips_preamble_before_tool_call() -> None:
    from sidecar.tool_bridge import openai_message_from_generation

    lines = [
        "If no action is needed, respond with an empty string.",
        "If multiple steps are needed, make one tool call at a time.",
        "Make sure to use the correct JSON format for the arguments.",
    ]
    raw = "\n".join(lines) + '\n<tool_call>\n{"name": "exec", "arguments": {"command": "ls"}}\n</tool_call>'
    message = openai_message_from_generation(raw)
    assert message.get("tool_calls")
    assert message.get("content") is None


def test_detects_make_sure_correct_spam() -> None:
    lines = [
        "If no tool call is needed, respond directly to the user with the final answer.",
        "If multiple steps are needed, make sure to call the appropriate functions in sequence.",
        "Make sure to use the correct parameters for each function.",
        "Make sure to use the correct JSON format for the arguments.",
        "Make sure to use the correct syntax for the JSON object.",
    ]
    raw = "\n".join(lines)
    assert is_degenerate_tool_guideline_spam(raw)


def test_detects_make_sure_follow_spam() -> None:
    lines = [
        "If no tool call is needed, respond directly to the user with the final answer inside <answer> tags.",
        "If multiple steps are needed, make sure to call the appropriate functions in sequence.",
        "Make sure to follow the parameters' requirements and constraints.",
        "Make sure to use the correct JSON syntax and structure.",
        "Make sure to use the correct function names and parameter names.",
    ]
    raw = "\n".join(lines)
    assert is_degenerate_tool_guideline_spam(raw)


def test_strips_kv_reuse_corporate_guideline_spam_without_tool_call() -> None:
    lines = [
        "If no tool call is needed, respond directly to the user question without using XML blocks.",
        "Only use the functions listed in the tools section. Do not make up new functions.",
        "Make sure all parameters are correctly formatted according to their types.",
        "Avoid making multiple unrelated tool calls in sequence. Group related operations together when possible.",
        "When writing files, ensure proper formatting and structure based on the file type.",
        "Communicate effectively with team members throughout the development lifecycle.",
        "Maintain flexibility and resilience in the face of uncertainty.",
    ]
    raw = "\n".join(lines)
    assert is_degenerate_tool_guideline_spam(raw)
    assert tool_guideline_spam_without_tool_call(raw)
    assert sanitize_generation_text(raw) == ""


def test_strips_corporate_do_not_use_functions_spam() -> None:
    lines = [
        "If no tool call is needed, respond directly to the user's question.",
        "Only use the functions provided in the list above. Do not make up new functions.",
        "Do not use markdown formatting in your responses.",
        "Do not use any external libraries or APIs beyond the provided functions.",
        "Do not use any functions outside the scope of the provided tools.",
        "Do not use any functions that require additional parameters not listed in the schema.",
        "Do not use any functions that would cause unintended side effects.",
        "Do not use any functions that would compromise system security.",
        "Do not use any functions that would violate user privacy.",
        "Do not use any functions that would interfere with other processes.",
    ]
    raw = "\n".join(lines)
    assert is_degenerate_tool_guideline_spam(raw)
    assert tool_guideline_spam_without_tool_call(raw)
    assert sanitize_generation_text(raw) == ""
    message = __import__(
        "sidecar.tool_bridge", fromlist=["openai_message_from_generation"]
    ).openai_message_from_generation(raw)
    assert message.get("content") is None
    assert not message.get("tool_calls")


def test_strips_single_line_if_no_tool_call_spam() -> None:
    raw = "If no tool call is needed, respond directly to the user question without using XML blocks."
    assert is_degenerate_tool_guideline_spam(raw)
    assert tool_guideline_spam_without_tool_call(raw)
    assert sanitize_generation_text(raw) == ""


def test_strips_redacted_thinking_close_tag() -> None:
    tag = "<" + "/" + "red" + "acted_" + "thinking" + ">"
    assert sanitize_generation_text(tag) == ""
    assert is_unusable_tool_generation_text(tag) is True


def test_strips_single_line_preamble_before_write_tool_call() -> None:
    raw = (
        "If no tool call is needed, respond directly to the user's question.\n"
        "A: \n\n"
        '<tool_call>\n{"name": "write", "arguments": {"path": "design_summary.md", "content": "x"}}\n</tool_call>'
    )
    cleaned = sanitize_generation_text(raw)
    assert "If no tool call is needed" not in cleaned
    assert cleaned.lstrip().startswith("<tool_call>")
    message = __import__("sidecar.tool_bridge", fromlist=["openai_message_from_generation"]).openai_message_from_generation(raw)
    assert message.get("tool_calls")
    assert message.get("content") is None
