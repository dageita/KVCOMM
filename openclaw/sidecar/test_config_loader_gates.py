"""Tests for t2-config-loader analyzer/patcher gate helpers."""

from __future__ import annotations

from sidecar.openclaw_prefix import (
    CONFIG_LOADER_ANALYZER_READS,
    build_config_loader_analyzer_read_hint,
    config_loader_analyzer_reads_satisfied,
    config_loader_missing_analyzer_reads,
    config_loader_patcher_fix_satisfied,
    config_loader_verifier_should_force_exec,
    config_loader_verifier_should_force_read,
    next_config_loader_analyzer_read,
    pytest_collection_or_import_failed,
)


def _read_call(path: str, *, call_id: str = "call_1") -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read", "arguments": f'{{"path": "{path}"}}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "file body"},
    ]


def _edit_call(path: str, *, call_id: str = "call_edit") -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "edit",
                        "arguments": f'{{"path": "{path}", "edits": [{{"oldText": "a", "newText": "b"}}]}}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "Successfully replaced 1 occurrence"},
    ]


def test_config_loader_analyzer_read_hint_order() -> None:
    all_missing = CONFIG_LOADER_ANALYZER_READS
    assert next_config_loader_analyzer_read(all_missing) == "config_loader.py"
    hint = build_config_loader_analyzer_read_hint(all_missing)
    assert "config_loader.py first" in hint

    after_tests = all_missing - {"test_config_loader.py"}
    assert next_config_loader_analyzer_read(after_tests) == "config_loader.py"
    hint = build_config_loader_analyzer_read_hint(after_tests)
    assert "config_loader.py next" in hint
    assert "test_config_loader" not in hint

    after_config = frozenset({"app_config.py", "test_config_loader.py"})
    assert next_config_loader_analyzer_read(after_config) == "app_config.py"
    hint = build_config_loader_analyzer_read_hint(after_config)
    assert "app_config.py next" in hint
    assert "test_config_loader" not in hint

    after_config_and_app = frozenset({"test_config_loader.py"})
    assert next_config_loader_analyzer_read(after_config_and_app) == "test_config_loader.py"
    hint = build_config_loader_analyzer_read_hint(after_config_and_app)
    assert "tests/test_config_loader.py next" in hint


def test_config_loader_analyzer_reads_gate() -> None:
    messages: list[dict] = []
    assert config_loader_analyzer_reads_satisfied(messages) is False
    assert config_loader_missing_analyzer_reads(messages) == frozenset(
        {"config_loader.py", "app_config.py", "test_config_loader.py"}
    )

    messages.extend(_read_call("config_loader.py"))
    assert config_loader_missing_analyzer_reads(messages) == frozenset(
        {"app_config.py", "test_config_loader.py"}
    )

    messages.extend(_read_call("app_config.py", call_id="call_2"))
    messages.extend(_read_call("tests/test_config_loader.py", call_id="call_3"))
    assert config_loader_analyzer_reads_satisfied(messages) is True


def test_config_loader_patcher_fix_gate() -> None:
    messages = _edit_call("config_loader.py")
    assert config_loader_patcher_fix_satisfied(messages) is True


def test_config_loader_import_error_retries_exec_not_read() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_exec",
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "arguments": '{"command": "pytest -q tests/test_config_loader.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_exec",
            "content": (
                "ERROR collecting tests/test_config_loader.py\n"
                "ModuleNotFoundError: No module named 'config_loader'\n"
                "Interrupted: 1 error during collection"
            ),
        },
    ]
    assert pytest_collection_or_import_failed(messages) is True
    assert config_loader_verifier_should_force_exec(messages) is True
    assert config_loader_verifier_should_force_read(messages) is False
