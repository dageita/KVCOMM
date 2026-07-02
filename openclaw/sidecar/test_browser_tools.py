"""Tests for browser-family ClawBench tool bridge."""

from __future__ import annotations

import json

from sidecar.tool_bridge import (
    ensure_clawbench_agent_tools,
    filter_tools_for_agent,
    openai_message_from_generation,
)


def test_browser_family_agent0_tools_include_browser() -> None:
    tools = ensure_clawbench_agent_tools(
        [],
        agent_index=0,
        agent_role="Analyzer",
        task_profile="clawbench",
        clawbench_family="browser",
    )
    tools = filter_tools_for_agent(
        tools,
        agent_index=0,
        agent_role="Analyzer",
        task_profile="clawbench",
        clawbench_family="browser",
    )
    names = {str((t.get("function") or {}).get("name") or "") for t in tools}
    assert "browser" in names
    assert "read" in names


def test_browser_tool_call_gets_action_open_and_port_substitution() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "browser", "arguments": {"target": "host", '
        '"url": "http://127.0.0.1:{form_app_port}/"}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_vars={"form_app_port": "8765"},
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["action"] == "open"
    assert args["target"] == "host"
    assert args["url"] == "http://127.0.0.1:8765/"


def test_verify_form_exec_gets_node_path() -> None:
    from sidecar.tool_bridge import _normalize_tool_arguments

    args = _normalize_tool_arguments(
        "exec",
        {"command": "node verify_form.cjs http://127.0.0.1:8765/"},
        task_profile="clawbench",
        task_id="t2-browser-form-fix",
        task_vars={"form_app_port": "8765"},
    )
    command = str(args["command"])
    assert "NODE_PATH=" in command
    assert "verify_form.cjs" in command
    assert "8765" in command
    assert "/clawbench/node_modules" in command or "node_modules" in command


def test_browser_verifier_exec_passed_accepts_no_output() -> None:
    from sidecar.openclaw_prefix import browser_verifier_exec_passed

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_e",
                    "function": {
                        "name": "exec",
                        "arguments": '{"command":"node verify_form.cjs http://127.0.0.1:8765/"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_e", "content": "(no output)"},
    ]
    assert browser_verifier_exec_passed(messages) is True


def test_browser_verifier_exec_passed_rejects_timeout() -> None:
    from sidecar.openclaw_prefix import browser_verifier_exec_passed

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_e",
                    "function": {
                        "name": "exec",
                        "arguments": '{"command":"node verify_form.cjs http://127.0.0.1:8765/"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_e",
            "content": "page.waitForFunction: Timeout 3000ms exceeded.\n\n(Command exited with code 1)",
        },
    ]
    assert browser_verifier_exec_passed(messages) is False


def test_sync_clawbench_browser_workspaces_prefers_default(tmp_path, monkeypatch) -> None:
    import os
    import time

    from sidecar.tool_bridge import sync_clawbench_browser_workspaces

    default_root = tmp_path / "default-workspace"
    chain_root = tmp_path / "chain-workspace"
    default_root.mkdir()
    chain_root.mkdir()
    (chain_root / "app.js").write_text('getElementById("contact-formm")', encoding="utf-8")
    (default_root / "app.js").write_text('getElementById("contact-form")', encoding="utf-8")
    time.sleep(0.01)
    os.utime(default_root / "app.js", None)

    def fake_workspace(*, workspace_dir: str = "") -> str:
        explicit = (workspace_dir or "").strip()
        if explicit:
            return explicit
        return str(default_root)

    monkeypatch.setattr("sidecar.tool_bridge.clawbench_tool_workspace", fake_workspace)
    sync_clawbench_browser_workspaces(workspace_dir=str(chain_root), prefer_default=True)
    assert (chain_root / "app.js").read_text(encoding="utf-8") == 'getElementById("contact-form")'


def test_sync_clawbench_browser_workspaces_does_not_import_stale_default(tmp_path, monkeypatch) -> None:
    import os
    import time

    from sidecar.tool_bridge import sync_clawbench_browser_workspaces

    default_root = tmp_path / "default-workspace"
    chain_root = tmp_path / "chain-workspace"
    default_root.mkdir()
    chain_root.mkdir()
    (chain_root / "app.js").write_text('getElementById("contact-formm")', encoding="utf-8")
    (default_root / "app.js").write_text('getElementById("contact-form")', encoding="utf-8")
    time.sleep(0.01)
    os.utime(default_root / "app.js", None)

    def fake_workspace(*, workspace_dir: str = "") -> str:
        explicit = (workspace_dir or "").strip()
        if explicit:
            return explicit
        return str(default_root)

    monkeypatch.setattr("sidecar.tool_bridge.clawbench_tool_workspace", fake_workspace)
    sync_clawbench_browser_workspaces(workspace_dir=str(chain_root), prefer_default=False)
    assert (chain_root / "app.js").read_text(encoding="utf-8") == 'getElementById("contact-formm")'


def test_browser_form_fixed_on_disk() -> None:
    import os
    import tempfile

    from sidecar.tool_bridge import browser_form_fixed_on_disk

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "app.js")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('const form = document.getElementById("contact-formm");\n')
        assert browser_form_fixed_on_disk(workspace_dir=tmp) is False
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('const form = document.getElementById("contact-form");\n')
        assert browser_form_fixed_on_disk(workspace_dir=tmp) is True


def test_verify_form_tool_call_from_generation_includes_node_path() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command":"node verify_form.cjs http://127.0.0.1:8765/"}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id="t2-browser-form-fix",
        task_vars={"form_app_port": "8765"},
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert "NODE_PATH=" in args["command"]
    assert "verify_form.cjs" in args["command"]


def test_browser_patcher_and_verifier_gates(tmp_path) -> None:
    from sidecar.openclaw_prefix import (
        browser_patcher_edit_applied_in_messages,
        browser_patcher_fix_satisfied,
        browser_patcher_read_satisfied,
        browser_verifier_exec_done,
        browser_verifier_exec_passed,
    )

    chain_ws = tmp_path / "chain"
    chain_ws.mkdir()
    (chain_ws / "app.js").write_text('getElementById("contact-formm")', encoding="utf-8")

    read_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_r",
                    "function": {"name": "read", "arguments": '{"path":"app.js"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_r", "content": 'getElementById("contact-formm")'},
    ]
    assert browser_patcher_read_satisfied(read_messages) is True
    assert browser_patcher_edit_applied_in_messages(read_messages) is False
    assert browser_patcher_fix_satisfied(read_messages, workspace_dir=str(chain_ws)) is False

    (chain_ws / "app.js").write_text('getElementById("contact-form")', encoding="utf-8")
    assert browser_patcher_fix_satisfied(read_messages, workspace_dir=str(chain_ws)) is False

    edit_messages = read_messages + [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_ed",
                    "function": {
                        "name": "edit",
                        "arguments": '{"path":"app.js","old_string":"contact-formm","new_string":"contact-form"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_ed", "content": "Successfully replaced text in app.js"},
    ]
    assert browser_patcher_edit_applied_in_messages(edit_messages) is True
    assert browser_patcher_fix_satisfied(edit_messages, workspace_dir=str(chain_ws)) is True

    exec_messages = read_messages + [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_e",
                    "function": {
                        "name": "exec",
                        "arguments": '{"command":"node verify_form.cjs http://127.0.0.1:8765/"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_e", "content": "Exit code 0"},
    ]
    assert browser_verifier_exec_done(exec_messages) is True
    assert browser_verifier_exec_passed(exec_messages) is True

    from sidecar.openclaw_prefix import browser_exploration_satisfied

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_b",
                    "function": {
                        "name": "browser",
                        "arguments": '{"action":"open","target":"host","url":"http://127.0.0.1:8765/"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_b",
            "content": "Opened page title: Newsletter Signup",
        },
    ]
    assert browser_exploration_satisfied(messages) is True
