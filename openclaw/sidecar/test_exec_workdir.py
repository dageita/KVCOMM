"""Tests for clawbench exec workdir normalization."""

from __future__ import annotations

import json

from sidecar.bench_prompt_compose import ADD_TESTS_NORMALIZER_TASK_ID, BUGFIX_DISCOUNT_TASK_ID
from sidecar.tool_bridge import (
    clawbench_tool_workspace,
    openai_message_from_generation,
    sync_clawbench_tests_default_to_chain,
)


def test_exec_workdir_defaults_to_openclaw_workspace_for_clawbench() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q tests/test_pricing.py", "workdir": "."}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=BUGFIX_DISCOUNT_TASK_ID,
    )
    tool_calls = message.get("tool_calls") or []
    assert len(tool_calls) == 1
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["workdir"] == clawbench_tool_workspace()
    assert args["command"] == "pytest -q tests/test_pricing.py"


def test_exec_workdir_uses_registered_chain_workspace_when_present(tmp_path) -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q tests/test_normalizer.py", "workdir": "."}}\n'
        "</tool_call>"
    )
    chain_workspace = str(tmp_path / "run-deadbeef")
    chain_workspace_path = tmp_path / "run-deadbeef"
    chain_workspace_path.mkdir()
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=ADD_TESTS_NORMALIZER_TASK_ID,
        workspace_dir=str(chain_workspace_path),
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == str(chain_workspace_path)
    assert args["command"] == "PYTHONPATH=. python -m pytest -q tests/test_normalizer.py"


def test_exec_workdir_redirects_default_openclaw_workspace_to_chain(tmp_path) -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q tests/test_normalizer.py", '
        '"workdir": "~/.openclaw/workspace"}}\n'
        "</tool_call>"
    )
    chain_workspace_path = tmp_path / "run-deadbeef"
    chain_workspace_path.mkdir()
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=ADD_TESTS_NORMALIZER_TASK_ID,
        workspace_dir=str(chain_workspace_path),
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == str(chain_workspace_path)


def test_sync_clawbench_tests_default_to_chain(tmp_path, monkeypatch) -> None:
    default_root = tmp_path / "default-workspace"
    chain_root = tmp_path / "chain-workspace"
    default_tests = default_root / "tests"
    default_tests.mkdir(parents=True)
    (default_tests / "test_normalizer.py").write_text(
        "from normalizer import normalize_title, normalize_tags\n\ndef test_x():\n    assert True\n",
        encoding="utf-8",
    )
    chain_tests = chain_root / "tests"
    chain_tests.mkdir(parents=True)
    (chain_tests / "test_normalizer.py").write_text(
        "from openclaw.normalizer import normalize_text\n",
        encoding="utf-8",
    )

    def fake_workspace(*, workspace_dir: str = "") -> str:
        explicit = (workspace_dir or "").strip()
        if explicit:
            return explicit
        return str(default_root)

    monkeypatch.setattr("sidecar.tool_bridge.clawbench_tool_workspace", fake_workspace)
    changed = sync_clawbench_tests_default_to_chain(workspace_dir=str(chain_root))
    assert changed is True
    synced = (chain_tests / "test_normalizer.py").read_text(encoding="utf-8")
    assert "from normalizer import normalize_title, normalize_tags" in synced
    assert "openclaw" not in synced


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
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=BUGFIX_DISCOUNT_TASK_ID,
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == clawbench_tool_workspace()
    assert args["command"] == "pytest -q tests/test_pricing.py"
