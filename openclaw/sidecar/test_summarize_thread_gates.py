"""Tests for t2-msg-summarize-thread gate helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from sidecar.bench_prompt_compose import (
    SUMMARIZE_THREAD_ASSISTANT_MAX_CHARS,
    SUMMARIZE_THREAD_TASK_ID,
    SUMMARIZE_THREAD_TOOL_RESULT_MAX_CHARS,
)
from sidecar.openclaw_prefix import (
    _extract_turn_pairs,
    _tool_text,
    assistant_turn_max_chars,
    build_prefix_from_openclaw_messages,
    summarize_thread_extractor_read_complete,
    summarize_thread_thread_continuation_read_done,
    summarize_thread_thread_read_satisfied,
    summarize_thread_thread_read_truncated,
    tool_result_max_chars,
    summarize_thread_verifier_passed,
    summarize_thread_write_satisfied,
)

_THREAD_FIXTURE = Path(
    "/src/clawbench/tasks-public/assets/t2_msg_summarize_thread/thread.txt"
).read_text(encoding="utf-8")


def _read_call(
    *,
    call_id: str = "call_read",
    body: str = "Channel: #design-redesign",
    path: str = "./thread.txt",
    offset: int | None = None,
) -> list[dict]:
    args: dict[str, object] = {"path": path}
    if offset is not None:
        args["offset"] = offset
    import json

    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps(args),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": body},
    ]


def _write_call(
    path: str = "design_summary.md",
    *,
    call_id: str = "call_write",
    body: str = "Successfully wrote 393 bytes to design_summary.md",
    content: str | None = None,
) -> list[dict]:
    write_args = content or (
        f'{{"path": "{path}", "content": "# Design Summary\\n\\n## Decisions\\n'
        f'- Option B selected for homepage layout with Inter typography and brand orange CTAs."}}'
    )
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": write_args,
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": body},
    ]


def _exec_call(command: str, *, call_id: str = "call_exec", body: str = "ok") -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "exec", "arguments": f'{{"command": "{command}"}}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": body},
    ]


def test_summarize_thread_read_satisfied_after_thread_read() -> None:
    messages = [{"role": "user", "content": "task"}, *_read_call()]
    assert summarize_thread_thread_read_satisfied(messages) is True


def test_summarize_thread_read_not_satisfied_before_read() -> None:
    messages = [{"role": "user", "content": "task"}]
    assert summarize_thread_thread_read_satisfied(messages) is False


def test_summarize_thread_write_satisfied_after_successful_write() -> None:
    messages = [{"role": "user", "content": "task"}, *_write_call()]
    assert summarize_thread_write_satisfied(messages) is True


def test_summarize_thread_write_accepts_nonstandard_summary_path() -> None:
    messages = [{"role": "user", "content": "task"}, *_write_call("design-update.md")]
    assert summarize_thread_write_satisfied(messages) is True


def test_summarize_thread_write_not_satisfied_for_bootstrap_md() -> None:
    messages = [{"role": "user", "content": "task"}, *_write_call("AGENTS.md")]
    assert summarize_thread_write_satisfied(messages) is False


def test_summarize_thread_write_not_satisfied_for_placeholder_content() -> None:
    messages = [
        {"role": "user", "content": "task"},
        *_write_call(
            content=(
                '{"path": "design_summary.md", '
                '"content": "# Design Channel Summary\\n\\n{agent_0_current}"}'
            ),
            body="Successfully wrote 43 bytes to design_summary.md",
        ),
    ]
    assert summarize_thread_write_satisfied(messages) is False


def test_summarize_thread_write_not_satisfied_for_tiny_content() -> None:
    messages = [
        {"role": "user", "content": "task"},
        *_write_call(
            content='{"path": "design_summary.md", "content": "# Design Channel Summary\\n\\nPending."}',
            body="Successfully wrote 40 bytes to design_summary.md",
        ),
    ]
    assert summarize_thread_write_satisfied(messages) is False


def test_extract_agent0_rejects_unfilled_placeholder() -> None:
    from sidecar.openclaw_prefix import extract_summarize_thread_agent0_analysis

    messages = [
        {
            "role": "user",
            "content": "Output from Agent 0 (Extractor):\n\n{agent_0_current}\n",
        }
    ]
    assert extract_summarize_thread_agent0_analysis(messages) == ""


def test_writer_fallback_rejects_placeholder_agent0_block() -> None:
    from sidecar.openclaw_prefix import build_summarize_thread_writer_write_message

    messages = [
        {
            "role": "user",
            "content": "Output from Agent 0 (Extractor):\n\n{agent_0_current}\n",
        }
    ]
    message = build_summarize_thread_writer_write_message(messages=messages)
    body = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert "{agent_0_current}" not in body["content"]
    assert "Pending" in body["content"] or "See Agent 0" in body["content"]


def test_writer_fallback_resolves_agent0_from_llm_decode() -> None:
    from sidecar.openclaw_prefix import (
        build_summarize_thread_writer_write_message,
        resolve_summarize_thread_agent0_text,
    )

    analysis = (
        "**Decisions Made**\n"
        "- Homepage layout changed to Option B after reconsideration.\n\n"
        "**Still Open**\n"
        "- Mobile breakpoints deferred to next sprint.\n"
    )

    class _FakeLlm:
        def decode_upstream_agent_response_text(self, ph_id: str, message_key: str) -> str:
            assert ph_id == "agent_0_current"
            assert message_key == "task-key"
            return analysis

    messages = [
        {
            "role": "user",
            "content": "Output from Agent 0 (Extractor):\n\n{agent_0_current}\n",
        }
    ]
    resolved = resolve_summarize_thread_agent0_text(
        messages,
        llm=_FakeLlm(),
        message_key="task-key",
    )
    assert "Option B" in resolved
    message = build_summarize_thread_writer_write_message(
        messages=messages,
        llm=_FakeLlm(),
        message_key="task-key",
    )
    body = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert "Option B" in body["content"]
    assert "{agent_0_current}" not in body["content"]


def test_summarize_thread_verifier_passed_after_all_scripts() -> None:
    messages = [{"role": "user", "content": "task"}]
    for idx, script in enumerate(
        (
            "verify_summary_structure.py",
            "verify_latest_decision.py",
            "verify_commitments.py",
        )
    ):
        messages.extend(
            _exec_call(
                f"python3 {script}",
                call_id=f"call_exec_{idx}",
                body=f"PASS: t2_msg_summarize_thread/{script}",
            )
        )
    assert summarize_thread_verifier_passed(messages) is True


def test_summarize_thread_verifier_not_passed_after_one_script() -> None:
    messages = [
        {"role": "user", "content": "task"},
        *_exec_call(
            "python3 verify_summary_structure.py",
            body="PASS: t2_msg_summarize_thread/verify_summary_structure.py",
        ),
    ]
    assert summarize_thread_verifier_passed(messages) is False


def test_summarize_thread_verifier_passed_after_chained_exec() -> None:
    command = (
        "cd /root/.openclaw/workspace/kvcomm-chain/t2-msg-summarize-thread/run-fc78a2da && "
        "python3 verify_summary_structure.py && "
        "python3 verify_latest_decision.py && "
        "python3 verify_commitments.py"
    )
    body = (
        "PASS: t2_msg_summarize_thread/verify_summary_structure.py\n"
        "PASS: t2_msg_summarize_thread/verify_latest_decision.py\n"
        "PASS: t2_msg_summarize_thread/verify_commitments.py"
    )
    messages = [{"role": "user", "content": "task"}, *_exec_call(command, body=body)]
    assert summarize_thread_verifier_passed(messages) is True


def test_summarize_thread_task_specific_tool_result_limit() -> None:
    assert tool_result_max_chars(SUMMARIZE_THREAD_TASK_ID) == SUMMARIZE_THREAD_TOOL_RESULT_MAX_CHARS
    assert tool_result_max_chars("other-task") == 2000


def test_summarize_thread_task_specific_assistant_limit() -> None:
    assert (
        assistant_turn_max_chars(SUMMARIZE_THREAD_TASK_ID)
        == SUMMARIZE_THREAD_ASSISTANT_MAX_CHARS
    )
    assert assistant_turn_max_chars("other-task") == 2000


def test_summarize_thread_fixture_fits_task_tool_limit() -> None:
    assert len(_THREAD_FIXTURE) < SUMMARIZE_THREAD_TOOL_RESULT_MAX_CHARS
    messages = [{"role": "user", "content": "task"}, *_read_call(body=_THREAD_FIXTURE)]
    assert summarize_thread_thread_read_truncated(messages, task_id=SUMMARIZE_THREAD_TASK_ID) is False
    assert summarize_thread_extractor_read_complete(messages, task_id=SUMMARIZE_THREAD_TASK_ID) is True


def test_summarize_thread_read_truncated_without_continuation() -> None:
    body = "x" * 2500
    messages = [{"role": "user", "content": "task"}, *_read_call(body=body)]
    assert summarize_thread_thread_read_truncated(messages, task_id=SUMMARIZE_THREAD_TASK_ID) is False
    assert summarize_thread_thread_read_truncated(messages, task_id="other-task") is True
    assert summarize_thread_extractor_read_complete(messages, task_id="other-task") is False
    assert summarize_thread_thread_continuation_read_done(messages) is False


def test_summarize_thread_read_complete_after_offset_continuation() -> None:
    body = "x" * 2500
    tail = "[Apr 8 10:15] Priya: favicon update?"
    messages = [
        {"role": "user", "content": "task"},
        *_read_call(call_id="call_read_1", body=body),
        *_read_call(call_id="call_read_2", body=tail, offset=27),
    ]
    assert summarize_thread_thread_read_truncated(messages, task_id="other-task") is True
    assert summarize_thread_thread_continuation_read_done(messages) is True
    assert summarize_thread_extractor_read_complete(messages, task_id="other-task") is True


def test_summarize_thread_prefix_keeps_full_thread_read() -> None:
    messages = [
        {"role": "system", "content": "You are Agent 0"},
        {"role": "user", "content": "catch me up"},
        *_read_call(body=_THREAD_FIXTURE),
    ]
    built = build_prefix_from_openclaw_messages(
        messages,
        bench_user_prompt="catch me up",
        clawbench_role="Agent 0",
        task_id=SUMMARIZE_THREAD_TASK_ID,
    )
    tool_text = built.turn_content.get("turn_1_tool", "")
    assert "...[truncated]" not in tool_text
    assert "favicon update" in tool_text


def test_summarize_thread_assistant_turn_not_capped_at_2000() -> None:
    analysis = "A" * 3500
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": analysis,
        },
    ]
    turns = _extract_turn_pairs(messages, task_id=SUMMARIZE_THREAD_TASK_ID)
    assert len(turns[0]["assistant"]) == 3500


def test_tool_text_marks_truncation_when_over_limit() -> None:
    body = "y" * 2500
    rendered = _tool_text({"role": "tool", "content": body}, max_chars=2000)
    assert rendered.endswith("...[truncated]")
    assert len(rendered) < 2500


def test_summarize_thread_writer_write_fallback_from_agent0_block() -> None:
    from sidecar.openclaw_prefix import (
        build_summarize_thread_writer_write_message,
        extract_summarize_thread_agent0_analysis,
    )

    messages = [
        {
            "role": "user",
            "content": (
                "User request:\nCatch me up.\n\n"
                "Output from Agent 0 (Extractor):\n\n"
                "**Decisions Made**\n- Option B selected.\n\n"
                "OpenClaw tool cwd: default agent workspace\n"
                "Your job (Agent 1 - Writer): write the summary.\n"
            ),
        }
    ]
    analysis = extract_summarize_thread_agent0_analysis(messages)
    assert "Option B" in analysis
    message = build_summarize_thread_writer_write_message(messages=messages)
    assert message.get("tool_calls")
    assert message["tool_calls"][0]["function"]["name"] == "write"
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "design_summary.md"
    assert "Option B" in args["content"]
    assert "decision" in args["content"].lower()


def test_writer_fallback_normalizes_decided_heading() -> None:
    from sidecar.openclaw_prefix import build_summarize_thread_writer_write_message

    messages = [
        {
            "role": "user",
            "content": (
                "Output from Agent 0 (Extractor):\n\n"
                "## Decided ✅\n- Option B selected.\n\n"
                "## Still Open ❓\n- favicon\n"
            ),
        }
    ]
    message = build_summarize_thread_writer_write_message(messages=messages)
    body = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert "decision" in body["content"].lower()


def test_ensure_chain_deliverable_copies_from_default_workspace(tmp_path, monkeypatch) -> None:
    from sidecar.openclaw_prefix import ensure_summarize_thread_chain_deliverable
    from sidecar import tool_bridge

    default_root = tmp_path / "default"
    chain_root = tmp_path / "chain-run"
    default_root.mkdir()
    chain_root.mkdir()
    (default_root / "design_summary.md").write_text(
        "## Decisions\n"
        "- Option B selected for homepage layout with Inter typography.\n\n"
        "## Still Open\n"
        "- Mobile breakpoints deferred; favicon ownership unclear.\n\n"
        "## Commitments\n"
        "- Marcus updates Option B specs; zhentongfan drafts docs by Friday.\n",
        encoding="utf-8",
    )

    def _workspace(*, workspace_dir: str = "") -> str:
        explicit = (workspace_dir or "").strip()
        if explicit and os.path.isdir(explicit):
            return explicit
        return str(default_root)

    monkeypatch.setattr(tool_bridge, "clawbench_tool_workspace", _workspace)
    assert ensure_summarize_thread_chain_deliverable(
        workspace_dir=str(chain_root),
        messages=[],
    )
    copied = chain_root / "design_summary.md"
    assert copied.is_file()
    assert "decision" in copied.read_text(encoding="utf-8").lower()


def test_ensure_chain_deliverable_extracts_from_agent1_block(tmp_path) -> None:
    from sidecar.openclaw_prefix import ensure_summarize_thread_chain_deliverable

    chain_root = tmp_path / "chain-run"
    chain_root.mkdir()
    messages = [
        {
            "role": "user",
            "content": (
                "Output from Agent 1 (Writer):\n\n"
                "```\n# Design Channel Summary\n\n## Decisions Made\n- Option B\n\n"
                "## Still Open\n- favicon\n```\n\nYour job (Agent 2 - Verifier)\n"
            ),
        }
    ]
    assert ensure_summarize_thread_chain_deliverable(
        workspace_dir=str(chain_root),
        messages=messages,
    )
    body = (chain_root / "design_summary.md").read_text(encoding="utf-8")
    assert "decision" in body.lower()


def test_write_path_targets_chain_workspace(tmp_path) -> None:
    from sidecar.tool_bridge import _normalize_tool_arguments

    chain = tmp_path / "run-abc"
    chain.mkdir()
    args = _normalize_tool_arguments(
        "write",
        {"path": "design_summary.md", "content": "## Decisions\n- x\n\n## Still Open\n- y\n"},
        task_profile="clawbench",
        task_id="t2-msg-summarize-thread",
        workspace_dir=str(chain),
    )
    assert args["path"] == str(chain / "design_summary.md")
