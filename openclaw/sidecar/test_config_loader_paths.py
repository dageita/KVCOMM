"""Tests for t2-config-loader path aliases."""

from __future__ import annotations

import json

from sidecar.bench_prompt_compose import BUGFIX_DISCOUNT_TASK_ID, CONFIG_LOADER_TASK_ID
from sidecar.openclaw_prefix import build_config_loader_edit_message
from sidecar.tool_bridge import normalize_tool_file_path, openai_message_from_generation


def test_config_loader_test_path_aliases() -> None:
    assert (
        normalize_tool_file_path("test_config_loader.py", task_id=CONFIG_LOADER_TASK_ID)
        == "tests/test_config_loader.py"
    )
    assert (
        normalize_tool_file_path("tests/test_config_loader.py", task_id=CONFIG_LOADER_TASK_ID)
        == "tests/test_config_loader.py"
    )
    assert (
        normalize_tool_file_path("test_config_loader.py", task_id=BUGFIX_DISCOUNT_TASK_ID)
        == "test_config_loader.py"
    )


def test_config_loader_pytest_scoped_to_test_file() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q", "workdir": "."}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=CONFIG_LOADER_TASK_ID,
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["command"] == "PYTHONPATH=. python -m pytest -q tests/test_config_loader.py"


def test_config_loader_read_rewrites_wrong_test_path() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "read", "arguments": {"path": "test_config_loader.py"}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=CONFIG_LOADER_TASK_ID,
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "tests/test_config_loader.py"


def test_config_loader_canonical_edit_message() -> None:
    message = build_config_loader_edit_message()
    tool_calls = message.get("tool_calls") or []
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "edit"
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["path"] == "config_loader.py"
    assert len(args["edits"]) == 1
    assert "int(os.environ[\"APP_PORT\"])" in args["edits"][0]["newText"]
