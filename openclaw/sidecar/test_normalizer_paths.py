"""Tests for t2-add-tests-normalizer path aliases."""

from __future__ import annotations

import json

from sidecar.bench_prompt_compose import ADD_TESTS_NORMALIZER_TASK_ID, BUGFIX_DISCOUNT_TASK_ID
from sidecar.tool_bridge import normalize_tool_file_path, openai_message_from_generation


def test_normalizer_read_path_aliases() -> None:
    assert (
        normalize_tool_file_path("text_normalization_module.py", task_id=ADD_TESTS_NORMALIZER_TASK_ID)
        == "normalizer.py"
    )
    assert normalize_tool_file_path("normalizer.py", task_id=ADD_TESTS_NORMALIZER_TASK_ID) == "normalizer.py"
    assert (
        normalize_tool_file_path("text_normalization_module.py", task_id=BUGFIX_DISCOUNT_TASK_ID)
        == "text_normalization_module.py"
    )


def test_normalizer_test_path_aliases() -> None:
    assert (
        normalize_tool_file_path("test_normalizer.py", task_id=ADD_TESTS_NORMALIZER_TASK_ID)
        == "tests/test_normalizer.py"
    )


def test_normalizer_pytest_scoped_to_test_normalizer_py() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q", "workdir": "."}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=ADD_TESTS_NORMALIZER_TASK_ID,
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["command"] == "PYTHONPATH=. python -m pytest -q tests/test_normalizer.py"


def test_normalizer_pytest_uses_python_m_and_pythonpath() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q tests/test_normalizer.py", "workdir": "."}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=ADD_TESTS_NORMALIZER_TASK_ID,
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["command"].startswith("PYTHONPATH=. python -m pytest")


def test_read_tool_rewrites_wrong_normalizer_module_path() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "read", "arguments": {"path": "text_normalization_module.py"}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=ADD_TESTS_NORMALIZER_TASK_ID,
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "normalizer.py"


def test_normalizer_edit_rewrites_relative_import_rewrite() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "edit", "arguments": {"path": "test_normalizer.py", "edits": [{"oldText":'
        '"from normalizer import normalize_title, normalize_tags",'
        '"newText":"from ..normalizer import normalize_title, normalize_tags"}]}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=ADD_TESTS_NORMALIZER_TASK_ID,
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    new_text = args["edits"][0]["newText"]
    assert new_text == "from normalizer import normalize_title, normalize_tags"


def test_normalizer_write_rewrites_relative_import() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "write", "arguments": {"path": "tests/test_normalizer.py", "content":'
        '"from ..normalizer import normalize_title, normalize_tags\\n'
        'def test_x(): pass"}}'
        "\n</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=ADD_TESTS_NORMALIZER_TASK_ID,
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["content"].startswith("from normalizer import")
