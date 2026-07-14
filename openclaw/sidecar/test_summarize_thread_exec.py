"""Tests for summarize-thread verifier exec normalization and fallbacks."""

from __future__ import annotations

import json

from sidecar.openclaw_prefix import (
    build_summarize_thread_verifier_exec_hint,
    build_summarize_thread_verifier_exec_message,
)
from sidecar.tool_bridge import (
    SUMMARIZE_THREAD_VERIFY_COMMAND,
    _normalize_summarize_thread_exec_command,
    is_unusable_tool_generation_text,
    parse_qwen_tool_calls,
)


def test_normalize_summarize_thread_exec_strips_bad_cd() -> None:
    cmd = (
        "cd ~/workspace && python3 verify_summary_structure.py && "
        "python3 verify_latest_decision.py && python3 verify_commitments.py"
    )
    assert _normalize_summarize_thread_exec_command(cmd) == SUMMARIZE_THREAD_VERIFY_COMMAND


def test_normalize_summarize_thread_exec_empty_defaults() -> None:
    assert _normalize_summarize_thread_exec_command("") == SUMMARIZE_THREAD_VERIFY_COMMAND


def test_normalize_summarize_thread_exec_strips_kvcomm_chain_cd() -> None:
    cmd = (
        "cd /root/.openclaw/workspace/kvcomm-chain/t2-msg-summarize-thread/run-abc && "
        "python3 verify_summary_structure.py && python3 verify_latest_decision.py && "
        "python3 verify_commitments.py"
    )
    assert _normalize_summarize_thread_exec_command(cmd) == SUMMARIZE_THREAD_VERIFY_COMMAND


def test_normalize_summarize_thread_summary_markdown_adds_decision_keyword() -> None:
    from sidecar.tool_bridge import normalize_summarize_thread_summary_markdown

    raw = "## Decided ✅\n- Option B selected.\n\n## Still Open ❓\n- favicon\n"
    fixed = normalize_summarize_thread_summary_markdown(raw)
    assert "decision" in fixed.lower()
    assert "open" in fixed.lower()


def test_summarize_thread_exec_needs_canonical_fallback() -> None:
    from sidecar.tool_bridge import summarize_thread_exec_needs_canonical_fallback

    assert summarize_thread_exec_needs_canonical_fallback({"role": "assistant"}) is True
    bad = {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "exec",
                    "arguments": json.dumps(
                        {
                            "command": (
                                "cd /root/.openclaw/workspace/kvcomm-chain/run-1 && "
                                "python3 verify_summary_structure.py"
                            )
                        }
                    ),
                },
            }
        ],
    }
    assert summarize_thread_exec_needs_canonical_fallback(bad) is True
    good = {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "exec",
                    "arguments": json.dumps(
                        {
                            "command": SUMMARIZE_THREAD_VERIFY_COMMAND,
                            "workdir": "/tmp/chain",
                        }
                    ),
                },
            }
        ],
    }
    assert summarize_thread_exec_needs_canonical_fallback(good) is False


def test_exec_normalize_injects_workdir_for_summarize_thread(tmp_path) -> None:
    from sidecar.tool_bridge import _normalize_tool_arguments

    chain = tmp_path / "run-1"
    chain.mkdir()
    args = _normalize_tool_arguments(
        "exec",
        {
            "command": (
                "cd /root/.openclaw/workspace/kvcomm-chain/run-1 && "
                "python3 verify_summary_structure.py && python3 verify_latest_decision.py && "
                "python3 verify_commitments.py"
            )
        },
        task_profile="clawbench",
        task_id="t2-msg-summarize-thread",
        workspace_dir=str(chain),
    )
    assert args["command"] == SUMMARIZE_THREAD_VERIFY_COMMAND
    assert args["workdir"] == str(chain)


def test_verifier_exec_hint_includes_chain_workdir(tmp_path) -> None:
    chain = tmp_path / "run-abc123"
    chain.mkdir()
    hint = build_summarize_thread_verifier_exec_hint(workspace_dir=str(chain))
    assert str(chain) in hint
    assert SUMMARIZE_THREAD_VERIFY_COMMAND in hint
    assert "~/workspace" in hint


def test_verifier_exec_message_uses_chain_workdir(tmp_path) -> None:
    chain = tmp_path / "run-abc123"
    chain.mkdir()
    message = build_summarize_thread_verifier_exec_message(workspace_dir=str(chain))
    tool_calls = message.get("tool_calls") or []
    assert len(tool_calls) == 1
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["command"] == SUMMARIZE_THREAD_VERIFY_COMMAND
    assert args["workdir"] == str(chain)


def test_unusable_tool_generation_detects_fence_only() -> None:
    assert is_unusable_tool_generation_text("```") is True
    assert is_unusable_tool_generation_text("") is True


def test_unusable_tool_generation_accepts_tool_call() -> None:
    raw = (
        '<tool_call>\n{"name": "exec", "arguments": {"command": "python3 verify_summary_structure.py"}}'
        "\n</tool_call>"
    )
    assert is_unusable_tool_generation_text(raw) is False
    _, tool_calls = parse_qwen_tool_calls(raw)
    assert tool_calls
