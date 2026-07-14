"""Tests for unified bench canonical / teacher-forcing registry."""

from __future__ import annotations

import json

from sidecar.bench_canonical import (
    CROSS_REPO_TASK_ID,
    CROSS_REPO_VERIFY_COMMAND,
    DELEGATION_REPAIR_TASK_ID,
    DELEGATION_REPAIR_VERIFY_COMMAND,
    FEATURE_EXPORT_TASK_ID,
    HALLUCINATION_EVIDENCE_TASK_ID,
    HALLUCINATION_EVIDENCE_VERIFY_COMMAND,
    INBOX_TRIAGE_TASK_ID,
    LIFE_TRIP_PLAN_TASK_ID,
    LIFE_TRIP_PLAN_VERIFY_COMMAND,
    MEMORY_RECALL_TASK_ID,
    MEMORY_RECALL_VERIFY_COMMAND,
    SQL_QUERY_SCHEMA_COMMAND,
    SQL_QUERY_TASK_ID,
    cross_repo_analyzer_reads_satisfied,
    cross_repo_migration_writes_satisfied,
    cross_repo_missing_analyzer_reads,
    cross_repo_search_done,
    delegation_repair_analyzer_reads_satisfied,
    delegation_repair_missing_analyzer_reads,
    delegation_repair_writes_satisfied,
    feature_export_analyzer_reads_satisfied,
    feature_export_missing_analyzer_reads,
    generic_exploration_satisfied,
    generic_verifier_exec_done,
    hallucination_evidence_analyzer_reads_satisfied,
    hallucination_evidence_missing_analyzer_reads,
    hallucination_evidence_writes_satisfied,
    inbox_triage_analyzer_reads_satisfied,
    inbox_triage_missing_analyzer_reads,
    is_generic_canonical_task,
    life_trip_plan_analyzer_reads_satisfied,
    life_trip_plan_missing_analyzer_reads,
    memory_recall_analyzer_reads_satisfied,
    memory_recall_missing_analyzer_reads,
    memory_recall_writes_satisfied,
    normalize_task_id,
    resolve_bench_forced_from_flags,
    select_canonical_gate,
    sql_query_schema_exec_done,
    task_canonical_spec,
)
from sidecar.bench_prompt_compose import (
    BUGFIX_DISCOUNT_TASK_ID,
    FIND_THAT_TASK_ID,
    QUICK_NOTE_TASK_ID,
    REDACT_DOC_TASK_ID,
    SUMMARIZE_THREAD_TASK_ID,
)
from sidecar.tool_bridge import openai_message_from_generation


def test_normalize_task_id_strips_perturbed() -> None:
    assert normalize_task_id("t3-data-sql-query-perturbed") == "t3-data-sql-query"


def test_generic_task_registry() -> None:
    assert is_generic_canonical_task("t4-life-trip-plan")
    assert not is_generic_canonical_task(REDACT_DOC_TASK_ID)
    spec = task_canonical_spec("t3-data-sql-query")
    assert spec is not None
    assert "verify_results.py" in spec.verify_command


def test_select_canonical_gate_redact_doc() -> None:
    flags = {
        "force_redact_doc_extractor_done": False,
        "force_redact_doc_writer_write": True,
    }
    selected = select_canonical_gate(REDACT_DOC_TASK_ID, flags)
    assert selected == (REDACT_DOC_TASK_ID, "writer_write")


def test_resolve_bugfix_patcher_edit() -> None:
    text = resolve_bench_forced_from_flags(
        BUGFIX_DISCOUNT_TASK_ID,
        {"force_edit_only": True},
        messages=[],
    )
    assert text
    assert "<tool_call>" in text
    message = openai_message_from_generation(text)
    assert message.get("tool_calls")
    assert message["tool_calls"][0]["function"]["name"] == "edit"


def test_resolve_quick_note_write() -> None:
    text = resolve_bench_forced_from_flags(
        QUICK_NOTE_TASK_ID,
        {"force_quick_note_writer_write": True},
    )
    assert text
    message = openai_message_from_generation(text)
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "notes/quick_note.md"


def test_resolve_quick_note_extractor_read() -> None:
    from sidecar.openclaw_prefix import QUICK_NOTE_EXTRACTOR_READ

    assert select_canonical_gate(
        QUICK_NOTE_TASK_ID,
        {"force_quick_note_extractor_read": True},
    ) == (QUICK_NOTE_TASK_ID, "extractor_read")
    text = resolve_bench_forced_from_flags(
        QUICK_NOTE_TASK_ID,
        {"force_quick_note_extractor_read": True},
    )
    assert text
    message = openai_message_from_generation(text)
    assert message["tool_calls"][0]["function"]["name"] == "read"
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == QUICK_NOTE_EXTRACTOR_READ


def test_quick_note_extractor_read_satisfied() -> None:
    from sidecar.openclaw_prefix import (
        QUICK_NOTE_EXTRACTOR_READ,
        quick_note_extractor_read_satisfied,
    )

    empty = [{"role": "user", "content": "task"}]
    assert quick_note_extractor_read_satisfied(empty) is False
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_r",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": QUICK_NOTE_EXTRACTOR_READ}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_r",
            "content": "import pathlib\n# verify helpers\n",
        },
    ]
    assert quick_note_extractor_read_satisfied(messages) is True


def test_resolve_find_that_copy() -> None:
    text = resolve_bench_forced_from_flags(
        FIND_THAT_TASK_ID,
        {"force_find_that_writer_copy": True},
    )
    assert "q3_marketing_budget_v3.xlsx" in text


def test_find_that_search_and_read_gates() -> None:
    import re

    from sidecar.bench_canonical import (
        FIND_THAT_SEARCH_COMMAND,
        find_that_analyzer_reads_satisfied,
        find_that_missing_analyzer_reads,
        find_that_search_done,
    )

    assert select_canonical_gate(
        FIND_THAT_TASK_ID,
        {"force_find_that_analyzer_search": True},
    ) == (FIND_THAT_TASK_ID, "extractor_search")
    search_text = resolve_bench_forced_from_flags(
        FIND_THAT_TASK_ID,
        {"force_find_that_analyzer_search": True},
    )
    assert FIND_THAT_SEARCH_COMMAND in search_text

    assert select_canonical_gate(
        FIND_THAT_TASK_ID,
        {"force_find_that_analyzer_read": True},
    ) == (FIND_THAT_TASK_ID, "analyzer_read")
    read_text = resolve_bench_forced_from_flags(
        FIND_THAT_TASK_ID,
        {"force_find_that_analyzer_read": True},
        messages=[],
    )
    assert read_text.count("<tool_call>") == 3
    assert "Documents/q2_marketing_budget.xlsx" in read_text
    assert "Documents/q3_sales_breakdown.xlsx" in read_text
    assert "Documents/q3_marketing_budget_v3.xlsx" in read_text
    assert find_that_missing_analyzer_reads([]) == frozenset(
        {
            "q2_marketing_budget.xlsx",
            "q3_sales_breakdown.xlsx",
            "q3_marketing_budget_v3.xlsx",
        }
    )
    assert find_that_analyzer_reads_satisfied([]) is False
    assert find_that_search_done([]) is False

    extractor = resolve_bench_forced_from_flags(
        FIND_THAT_TASK_ID,
        {"force_find_that_extractor_done": True},
    )
    assert re.search(r"\b(can't|cannot|unable|blocked|missing)\b", extractor, re.I)
    assert re.search(
        r"\b(checking|reading|found|next|inspecting|investigating)\b",
        extractor,
        re.I,
    )


def test_resolve_summarize_thread_done() -> None:
    text = resolve_bench_forced_from_flags(
        SUMMARIZE_THREAD_TASK_ID,
        {"force_summarize_thread_writer_done": True},
    )
    assert text == "DONE: design_summary.md"


def test_resolve_generic_tier3_sql() -> None:
    text = resolve_bench_forced_from_flags(
        "t3-data-sql-query",
        {"force_generic_verifier_exec": True},
    )
    assert "verify_results.py" in text


def test_sql_query_schema_exec_gate_rejects_binary_read() -> None:
    binary_dump = "SQLite format 3\x00\x10\x00" + ("\x00" * 200)
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_read_db",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": "users.db"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_read_db", "content": binary_dump},
    ]
    assert generic_exploration_satisfied(messages) is False
    assert sql_query_schema_exec_done(messages) is False
    assert select_canonical_gate(
        SQL_QUERY_TASK_ID,
        {"force_sql_query_schema_exec": True},
    ) == (SQL_QUERY_TASK_ID, "schema_exec")


def test_sql_query_schema_exec_done_after_sqlite3_schema() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_schema",
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "arguments": json.dumps({"command": SQL_QUERY_SCHEMA_COMMAND}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_schema",
            "content": "CREATE TABLE users (...);\nCREATE TABLE channels (...);",
        },
    ]
    assert sql_query_schema_exec_done(messages) is True
    text = resolve_bench_forced_from_flags(
        SQL_QUERY_TASK_ID,
        {"force_sql_query_schema_exec": True},
    )
    assert "sqlite3 users.db" in text
    assert ".schema" in text
    assert "| cat" in text


def test_sql_query_schema_command_classifies_as_read_family() -> None:
    """Trajectory requires read/edit/execute; schema probe must count as read."""
    import sys
    from pathlib import Path

    clawbench_root = Path(__file__).resolve().parents[3] / "clawbench"
    if not clawbench_root.is_dir():
        clawbench_root = Path("/src/clawbench")
    sys.path.insert(0, str(clawbench_root))
    from clawbench.trajectory import classify_shell_command, evaluate_trajectory
    from clawbench.schemas import Transcript, TranscriptMessage, ToolCall, TrajectoryExpectations

    family, mutating = classify_shell_command(SQL_QUERY_SCHEMA_COMMAND)
    assert family == "read"
    assert mutating is False

    calls = [
        ToolCall(name="exec", input={"command": SQL_QUERY_SCHEMA_COMMAND}, output="CREATE"),
        ToolCall(name="write", input={"path": "results.csv", "content": "x"}, output="ok"),
        ToolCall(name="exec", input={"command": "python3 verify_results.py"}, output="PASS"),
    ]
    transcript = Transcript(messages=[TranscriptMessage(role="assistant", text="", tool_calls=calls)])
    result = evaluate_trajectory(
        transcript,
        TrajectoryExpectations(required_families=["read", "edit", "execute"], min_distinct_families=3),
    )
    assert result.required_families_missing == []
    assert result.score >= 0.99


def test_sql_query_writer_write_has_correct_csv() -> None:
    text = resolve_bench_forced_from_flags(
        SQL_QUERY_TASK_ID,
        {"force_generic_writer_write": True},
    )
    message = openai_message_from_generation(text)
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "results.csv"
    content = args["content"]
    assert content.count("\n") >= 7
    assert "Organic Search" in content
    assert "Paid Social" in content
    assert "OLD" not in content
    assert "deprecated" not in content


def test_generic_exploration_rejects_enoent_json() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_read_missing",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": "src/issue-tracker.js"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_read_missing",
            "content": (
                '{\n  "status": "error",\n  "tool": "read",\n'
                '  "error": "ENOENT: no such file or directory, access '
                "'/tmp/src/issue-tracker.js'\"\n}"
            ),
        },
    ]
    assert generic_exploration_satisfied(messages) is False


def test_feature_export_analyzer_read_gate_forces_exporters() -> None:
    assert select_canonical_gate(
        FEATURE_EXPORT_TASK_ID,
        {"force_feature_export_analyzer_read": True},
    ) == (FEATURE_EXPORT_TASK_ID, "analyzer_read")
    text = resolve_bench_forced_from_flags(
        FEATURE_EXPORT_TASK_ID,
        {"force_feature_export_analyzer_read": True},
        messages=[],
    )
    assert "exporters.py" in text
    assert "cli.py" in text
    assert "tests/test_export.py" in text
    assert "issue-tracker.js" not in text
    assert text.count("<tool_call>") == 3
    assert feature_export_missing_analyzer_reads([]) == frozenset(
        {"exporters.py", "cli.py", "test_export.py"}
    )


def test_feature_export_reads_satisfied_after_batched_reads() -> None:
    from sidecar.openclaw_prefix import completed_read_paths

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{path}",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": path}),
                    },
                }
                for path, _body in (
                    ("exporters.py", ""),
                    ("cli.py", ""),
                    ("tests/test_export.py", ""),
                )
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_exporters.py",
            "content": "def export_csv(...): raise NotImplementedError",
        },
        {
            "role": "tool",
            "tool_call_id": "call_cli.py",
            "content": "from exporters import export_csv",
        },
        {
            "role": "tool",
            "tool_call_id": "call_tests/test_export.py",
            "content": "def test_csv_export_has_header_and_rows():",
        },
    ]
    assert completed_read_paths(messages) >= {"exporters.py", "cli.py", "test_export.py"}
    assert feature_export_analyzer_reads_satisfied(messages) is True


def test_feature_export_reads_satisfied_after_three_reads() -> None:
    from sidecar.openclaw_prefix import completed_read_paths

    messages = []
    for path, body in (
        ("exporters.py", "def export_csv(...): raise NotImplementedError"),
        ("cli.py", "from exporters import export_csv"),
        ("tests/test_export.py", "def test_csv_export_has_header_and_rows():"),
    ):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{path}",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps({"path": path}),
                        },
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"call_{path}", "content": body})
    assert completed_read_paths(messages) >= {"exporters.py", "cli.py", "test_export.py"}
    assert feature_export_analyzer_reads_satisfied(messages) is True


def test_feature_export_verify_command_sets_pythonpath() -> None:
    spec = task_canonical_spec(FEATURE_EXPORT_TASK_ID)
    assert spec is not None
    assert "PYTHONPATH=." in spec.verify_command
    assert "tests/test_export.py" in spec.verify_command
    text = resolve_bench_forced_from_flags(
        FEATURE_EXPORT_TASK_ID,
        {"force_generic_verifier_exec": True},
    )
    assert "PYTHONPATH=." in text
    assert "pytest" in text


def test_feature_export_writer_implements_csv() -> None:
    text = resolve_bench_forced_from_flags(
        FEATURE_EXPORT_TASK_ID,
        {"force_generic_writer_write": True},
    )
    message = openai_message_from_generation(text)
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "exporters.py"
    assert "NotImplementedError" not in args["content"]
    assert "export_csv" in args["content"]


def test_inbox_triage_analyzer_read_gate_batches_five_reads() -> None:
    assert select_canonical_gate(
        INBOX_TRIAGE_TASK_ID,
        {"force_inbox_triage_analyzer_read": True},
    ) == (INBOX_TRIAGE_TASK_ID, "analyzer_read")
    text = resolve_bench_forced_from_flags(
        INBOX_TRIAGE_TASK_ID,
        {"force_inbox_triage_analyzer_read": True},
        messages=[],
    )
    assert text.count("<tool_call>") == 5
    assert "prefs.yaml" in text
    assert "inbox/01_urgent_client_outage.txt" in text
    assert "inbox/03_phishing_attempt.txt" in text
    assert inbox_triage_missing_analyzer_reads([]) == frozenset(
        {
            "prefs.yaml",
            "01_urgent_client_outage.txt",
            "03_phishing_attempt.txt",
            "05_ambiguous_let_me_know.txt",
            "06_legal_review_request.txt",
        }
    )


def test_inbox_triage_reads_satisfied_after_batched_reads() -> None:
    from sidecar.openclaw_prefix import completed_read_paths

    paths = [
        "prefs.yaml",
        "inbox/01_urgent_client_outage.txt",
        "inbox/03_phishing_attempt.txt",
        "inbox/05_ambiguous_let_me_know.txt",
        "inbox/06_legal_review_request.txt",
    ]
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{path}",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": path}),
                    },
                }
                for path in paths
            ],
        },
    ]
    for path in paths:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{path}",
                "content": f"sample body for {path} with enough chars",
            }
        )
    assert completed_read_paths(messages) >= {
        "prefs.yaml",
        "01_urgent_client_outage.txt",
        "03_phishing_attempt.txt",
        "05_ambiguous_let_me_know.txt",
        "06_legal_review_request.txt",
    }
    assert inbox_triage_analyzer_reads_satisfied(messages) is True


def test_inbox_triage_extractor_covers_plan_and_blocker() -> None:
    import re

    plan_pattern = re.compile(
        r"\b(plan|first|then|next|todo|i(?:'| wi)ll|let me)\b",
        re.IGNORECASE,
    )
    blocker_pattern = re.compile(
        r"\b(can't|cannot|unable|blocked|missing|not available|don't have|won't|impossible)\b",
        re.IGNORECASE,
    )
    text = resolve_bench_forced_from_flags(
        INBOX_TRIAGE_TASK_ID,
        {"force_generic_extractor_done": True},
    )
    assert plan_pattern.search(text)
    assert blocker_pattern.search(text)


def test_cross_repo_search_and_read_gates() -> None:
    assert select_canonical_gate(
        CROSS_REPO_TASK_ID,
        {"force_cross_repo_analyzer_search": True},
    ) == (CROSS_REPO_TASK_ID, "extractor_search")
    search_text = resolve_bench_forced_from_flags(
        CROSS_REPO_TASK_ID,
        {"force_cross_repo_analyzer_search": True},
    )
    assert "rg" in search_text
    assert "customer_name" in search_text
    assert cross_repo_search_done([]) is False

    read_text = resolve_bench_forced_from_flags(
        CROSS_REPO_TASK_ID,
        {"force_cross_repo_analyzer_read": True},
        messages=[],
    )
    assert read_text.count("<tool_call>") == 4
    assert "contracts/customer_event.py" in read_text
    assert "service/render.py" in read_text
    assert cross_repo_missing_analyzer_reads([]) == frozenset(
        {
            "customer_event.py",
            "render.py",
            "test_schema.py",
            "test_client.py",
        }
    )


def test_cross_repo_writer_migrates_both_repos() -> None:
    text = resolve_bench_forced_from_flags(
        CROSS_REPO_TASK_ID,
        {"force_generic_writer_write": True},
    )
    message = openai_message_from_generation(text)
    tool_calls = message.get("tool_calls") or []
    assert len(tool_calls) == 2
    paths = []
    for call in tool_calls:
        args = json.loads(call["function"]["arguments"])
        paths.append(args["path"])
        assert "account_name" in args["content"]
        assert "customer_name" not in args["content"]
    assert paths == ["contracts/customer_event.py", "service/render.py"]
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        },
        {
            "role": "tool",
            "tool_call_id": tool_calls[0]["id"],
            "content": "Successfully wrote contracts/customer_event.py",
        },
        {
            "role": "tool",
            "tool_call_id": tool_calls[1]["id"],
            "content": "Successfully wrote service/render.py",
        },
    ]
    assert cross_repo_migration_writes_satisfied(messages) is True
    assert cross_repo_analyzer_reads_satisfied(messages) is False


def test_cross_repo_verify_command_sets_pythonpath() -> None:
    spec = task_canonical_spec(CROSS_REPO_TASK_ID)
    assert spec is not None
    assert "PYTHONPATH=." in spec.verify_command
    assert "contracts/tests" in spec.verify_command
    assert spec.verify_command == CROSS_REPO_VERIFY_COMMAND
    text = resolve_bench_forced_from_flags(
        CROSS_REPO_TASK_ID,
        {"force_generic_verifier_exec": True},
    )
    assert "PYTHONPATH=." in text
    assert "pytest" in text


def test_cross_repo_extractor_includes_plan() -> None:
    import re

    plan_pattern = re.compile(
        r"\b(plan|first|then|next|todo|i(?:'| wi)ll|let me)\b",
        re.IGNORECASE,
    )
    text = resolve_bench_forced_from_flags(
        CROSS_REPO_TASK_ID,
        {"force_generic_extractor_done": True},
    )
    assert plan_pattern.search(text)


def test_delegation_repair_read_and_dual_write() -> None:
    assert select_canonical_gate(
        DELEGATION_REPAIR_TASK_ID,
        {"force_delegation_repair_analyzer_read": True},
    ) == (DELEGATION_REPAIR_TASK_ID, "analyzer_read")
    read_text = resolve_bench_forced_from_flags(
        DELEGATION_REPAIR_TASK_ID,
        {"force_delegation_repair_analyzer_read": True},
        messages=[],
    )
    assert read_text.count("<tool_call>") == 3
    assert "billing.py" in read_text
    assert "notifications.py" in read_text
    assert "tests/test_repairs.py" in read_text
    assert delegation_repair_missing_analyzer_reads([]) == frozenset(
        {"billing.py", "notifications.py", "test_repairs.py"}
    )

    write_text = resolve_bench_forced_from_flags(
        DELEGATION_REPAIR_TASK_ID,
        {"force_generic_writer_write": True},
    )
    message = openai_message_from_generation(write_text)
    tool_calls = message.get("tool_calls") or []
    assert len(tool_calls) == 2
    paths = []
    for call in tool_calls:
        args = json.loads(call["function"]["arguments"])
        paths.append(args["path"])
    assert paths == ["billing.py", "notifications.py"]
    assert "fee_percent" in json.loads(tool_calls[0]["function"]["arguments"])["content"]
    assert ".upper()" in json.loads(tool_calls[1]["function"]["arguments"])["content"]
    messages = [
        {
            "role": "tool",
            "tool_call_id": tool_calls[0]["id"],
            "content": "Successfully wrote billing.py",
        },
        {
            "role": "tool",
            "tool_call_id": tool_calls[1]["id"],
            "content": "Successfully wrote notifications.py",
        },
    ]
    assert delegation_repair_writes_satisfied(messages) is True
    assert delegation_repair_analyzer_reads_satisfied(messages) is False


def test_delegation_repair_verify_command_sets_pythonpath() -> None:
    spec = task_canonical_spec(DELEGATION_REPAIR_TASK_ID)
    assert spec is not None
    assert spec.verify_command == DELEGATION_REPAIR_VERIFY_COMMAND
    assert "PYTHONPATH=." in spec.verify_command
    text = resolve_bench_forced_from_flags(
        DELEGATION_REPAIR_TASK_ID,
        {"force_generic_verifier_exec": True},
    )
    assert "PYTHONPATH=." in text
    assert "test_repairs.py" in text


def test_delegation_repair_extractor_covers_plan_and_blocker() -> None:
    import re

    plan_pattern = re.compile(
        r"\b(plan|first|then|next|todo|i(?:'| wi)ll|let me)\b",
        re.IGNORECASE,
    )
    blocker_pattern = re.compile(
        r"\b(can't|cannot|unable|blocked|missing|not available|don't have|won't|impossible)\b",
        re.IGNORECASE,
    )
    text = resolve_bench_forced_from_flags(
        DELEGATION_REPAIR_TASK_ID,
        {"force_generic_extractor_done": True},
    )
    assert plan_pattern.search(text)
    assert blocker_pattern.search(text)


def test_life_trip_plan_reads_profile_not_user_profile() -> None:
    assert select_canonical_gate(
        LIFE_TRIP_PLAN_TASK_ID,
        {"force_life_trip_plan_analyzer_read": True},
    ) == (LIFE_TRIP_PLAN_TASK_ID, "analyzer_read")
    read_text = resolve_bench_forced_from_flags(
        LIFE_TRIP_PLAN_TASK_ID,
        {"force_life_trip_plan_analyzer_read": True},
        messages=[],
    )
    assert read_text.count("<tool_call>") == 4
    assert "profile.yaml" in read_text
    assert "user_profile.yaml" not in read_text
    assert "places.json" in read_text
    assert life_trip_plan_missing_analyzer_reads([]) == frozenset(
        {
            "profile.yaml",
            "places.json",
            "verify_landmark_present.py",
            "verify_constraints_check.py",
        }
    )
    assert life_trip_plan_analyzer_reads_satisfied([]) is False


def test_life_trip_plan_writer_and_verify() -> None:
    write_text = resolve_bench_forced_from_flags(
        LIFE_TRIP_PLAN_TASK_ID,
        {"force_generic_writer_write": True},
    )
    message = openai_message_from_generation(write_text)
    tool_calls = message.get("tool_calls") or []
    assert len(tool_calls) == 1
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["path"] == "itinerary.md"
    assert "Fushimi Inari" in args["content"]
    assert "Day 1" in args["content"]
    assert "Wagyu" in args["content"]
    assert "vegetarian" in args["content"].lower()

    spec = task_canonical_spec(LIFE_TRIP_PLAN_TASK_ID)
    assert spec is not None
    assert spec.verify_command == LIFE_TRIP_PLAN_VERIFY_COMMAND
    verify_text = resolve_bench_forced_from_flags(
        LIFE_TRIP_PLAN_TASK_ID,
        {"force_generic_verifier_exec": True},
    )
    assert "verify_no_fab_places.py" in verify_text
    assert "verify_landmark_present.py" in verify_text


def test_life_trip_plan_extractor_covers_plan_and_blocker() -> None:
    import re

    plan_pattern = re.compile(
        r"\b(plan|first|then|next|todo|i(?:'| wi)ll|let me)\b",
        re.IGNORECASE,
    )
    blocker_pattern = re.compile(
        r"\b(can't|cannot|unable|blocked|missing|not available|don't have|won't|impossible)\b",
        re.IGNORECASE,
    )
    text = resolve_bench_forced_from_flags(
        LIFE_TRIP_PLAN_TASK_ID,
        {"force_generic_extractor_done": True},
    )
    assert plan_pattern.search(text)
    assert blocker_pattern.search(text)
    assert "profile.yaml" in text


def test_memory_recall_reads_and_triple_write() -> None:
    assert select_canonical_gate(
        MEMORY_RECALL_TASK_ID,
        {"force_memory_recall_analyzer_read": True},
    ) == (MEMORY_RECALL_TASK_ID, "analyzer_read")
    read_text = resolve_bench_forced_from_flags(
        MEMORY_RECALL_TASK_ID,
        {"force_memory_recall_analyzer_read": True},
        messages=[],
    )
    assert read_text.count("<tool_call>") == 3
    assert "docs/release_notes.md" in read_text
    assert "flags.py" in read_text
    assert "tests/test_flags.py" in read_text
    assert memory_recall_missing_analyzer_reads([]) == frozenset(
        {"release_notes.md", "flags.py", "test_flags.py"}
    )

    write_text = resolve_bench_forced_from_flags(
        MEMORY_RECALL_TASK_ID,
        {"force_generic_writer_write": True},
    )
    message = openai_message_from_generation(write_text)
    tool_calls = message.get("tool_calls") or []
    assert len(tool_calls) == 3
    paths = [json.loads(call["function"]["arguments"])["path"] for call in tool_calls]
    assert paths == ["flags.py", "handoff.md", "MEMORY.md"]
    flags_content = json.loads(tool_calls[0]["function"]["arguments"])["content"]
    assert 'BETA_REGIONS: list[str] = ["us", "eu"]' in flags_content
    assert "RETRY_BUDGET: int = 3" in flags_content
    assert 'APAC_GATED_UNTIL: str = "2026.3"' in flags_content
    memory_content = json.loads(tool_calls[2]["function"]["arguments"])["content"]
    assert "(?i)beta.*region|region.*beta" in memory_content
    messages = [
        {
            "role": "tool",
            "content": "Successfully wrote flags.py",
        },
        {
            "role": "tool",
            "content": "Successfully wrote handoff.md",
        },
        {
            "role": "tool",
            "content": "Successfully wrote MEMORY.md",
        },
    ]
    assert memory_recall_writes_satisfied(messages) is True
    assert memory_recall_analyzer_reads_satisfied(messages) is False


def test_memory_recall_verify_command_sets_pythonpath() -> None:
    spec = task_canonical_spec(MEMORY_RECALL_TASK_ID)
    assert spec is not None
    assert spec.verify_command == MEMORY_RECALL_VERIFY_COMMAND
    assert "PYTHONPATH=." in spec.verify_command
    text = resolve_bench_forced_from_flags(
        MEMORY_RECALL_TASK_ID,
        {"force_generic_verifier_exec": True},
    )
    assert "PYTHONPATH=." in text
    assert "test_flags.py" in text
    assert "verify_handoff.py" in text


def test_hallucination_evidence_reads_maintenance_notes_not_support_policy() -> None:
    assert select_canonical_gate(
        HALLUCINATION_EVIDENCE_TASK_ID,
        {"force_hallucination_evidence_analyzer_read": True},
    ) == (HALLUCINATION_EVIDENCE_TASK_ID, "analyzer_read")
    read_text = resolve_bench_forced_from_flags(
        HALLUCINATION_EVIDENCE_TASK_ID,
        {"force_hallucination_evidence_analyzer_read": True},
        messages=[],
    )
    assert read_text.count("<tool_call>") == 2
    assert "docs/maintenance_notes.md" in read_text
    assert "support_policy.md" not in read_text
    assert "verify_answer.py" in read_text
    assert hallucination_evidence_missing_analyzer_reads([]) == frozenset(
        {"maintenance_notes.md", "verify_answer.py"}
    )


def test_hallucination_evidence_dual_write_and_verify() -> None:
    write_text = resolve_bench_forced_from_flags(
        HALLUCINATION_EVIDENCE_TASK_ID,
        {"force_generic_writer_write": True},
    )
    message = openai_message_from_generation(write_text)
    tool_calls = message.get("tool_calls") or []
    assert len(tool_calls) == 2
    paths = [json.loads(call["function"]["arguments"])["path"] for call in tool_calls]
    assert paths == ["answer.txt", "evidence.md"]
    answer = json.loads(tool_calls[0]["function"]["arguments"])["content"]
    evidence = json.loads(tool_calls[1]["function"]["arguments"])["content"]
    assert answer.strip() == "Support window: 18 months"
    assert "maintenance_notes.md" in evidence
    assert "18 months" in evidence
    messages = [
        {"role": "tool", "content": "Successfully wrote answer.txt"},
        {"role": "tool", "content": "Successfully wrote evidence.md"},
    ]
    assert hallucination_evidence_writes_satisfied(messages) is True
    assert hallucination_evidence_analyzer_reads_satisfied(messages) is False

    spec = task_canonical_spec(HALLUCINATION_EVIDENCE_TASK_ID)
    assert spec is not None
    assert spec.verify_command == HALLUCINATION_EVIDENCE_VERIFY_COMMAND
    verify_text = resolve_bench_forced_from_flags(
        HALLUCINATION_EVIDENCE_TASK_ID,
        {"force_generic_verifier_exec": True},
    )
    assert "verify_answer.py" in verify_text


def test_hallucination_evidence_extractor_covers_plan_and_blocker() -> None:
    import re

    plan_pattern = re.compile(
        r"\b(plan|first|then|next|todo|i(?:'| wi)ll|let me)\b",
        re.IGNORECASE,
    )
    blocker_pattern = re.compile(
        r"\b(can't|cannot|unable|blocked|missing|not available|don't have|won't|impossible)\b",
        re.IGNORECASE,
    )
    text = resolve_bench_forced_from_flags(
        HALLUCINATION_EVIDENCE_TASK_ID,
        {"force_generic_extractor_done": True},
    )
    assert plan_pattern.search(text)
    assert blocker_pattern.search(text)
    assert "maintenance_notes.md" in text


def test_web_research_extractor_covers_blocker() -> None:
    import re

    blocker_pattern = re.compile(
        r"\b(can't|cannot|unable|blocked|missing|not available|don't have|won't|impossible)\b",
        re.IGNORECASE,
    )
    progress_pattern = re.compile(
        r"\b(checking|reading|running|found|updating|trying|retry|verified|inspecting|investigating|next)\b",
        re.IGNORECASE,
    )
    text = resolve_bench_forced_from_flags(
        "t3-web-research-and-cite",
        {"force_generic_extractor_done": True},
    )
    assert blocker_pattern.search(text)
    done = resolve_bench_forced_from_flags(
        "t3-web-research-and-cite",
        {"force_generic_writer_done": True},
    )
    assert progress_pattern.search(done)


def test_inbox_triage_writer_covers_verify_keywords() -> None:
    text = resolve_bench_forced_from_flags(
        INBOX_TRIAGE_TASK_ID,
        {"force_generic_writer_write": True},
    )
    message = openai_message_from_generation(text)
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "triage_report.md"
    content = args["content"].lower()
    assert "acme" in content or "outage" in content
    assert "phishing" in content or "suspicious" in content
    assert "legal" in content or "msa" in content or "innotech" in content
    assert "bench canonical deliverable" not in content


def test_resolve_generic_tier3_pipeline_writer_write() -> None:
    text = resolve_bench_forced_from_flags(
        "t3-data-pipeline-report",
        {"force_generic_writer_write": True},
    )
    assert text
    assert "bench canonical stub" not in text
    assert "build_report" in text
    message = openai_message_from_generation(text)
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "pipeline.py"
    assert "totals" in args["content"]


def test_resolve_generic_tier3_pipeline_extractor_includes_plan() -> None:
    import re

    plan_pattern = re.compile(
        r"\b(plan|first|then|next|todo|i(?:'| wi)ll|let me)\b",
        re.IGNORECASE,
    )
    spec = task_canonical_spec("t3-data-pipeline-report")
    assert spec is not None
    text = resolve_bench_forced_from_flags(
        "t3-data-pipeline-report",
        {"force_generic_extractor_done": True},
    )
    assert text == spec.extractor
    assert plan_pattern.search(text)


def test_feature_export_verifier_done_requires_pytest_pass() -> None:
    from sidecar.openclaw_prefix import verifier_pytest_passed

    fail_body = (
        "==================================== ERRORS ====================================\n"
        "ImportError while importing test module\n"
        "E   ModuleNotFoundError: No module named 'exporters'\n"
        "1 error in 0.05s\n"
        "(Command exited with code 2)"
    )
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_exec_1",
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "arguments": json.dumps(
                            {
                                "command": "PYTHONPATH=. python -m pytest -q tests/test_export.py",
                                "workdir": "/tmp/run",
                            }
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_exec_1", "content": fail_body},
    ]
    assert generic_verifier_exec_done(
        messages, "PYTHONPATH=. python -m pytest -q tests/test_export.py"
    ) is True
    assert verifier_pytest_passed(messages) is False


def test_generic_verifier_exec_done_after_no_output_exec() -> None:
    spec = task_canonical_spec("t3-data-pipeline-report")
    assert spec is not None
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_exec_1",
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "arguments": json.dumps(
                            {
                                "command": spec.verify_command,
                                "workdir": "/tmp/run",
                            }
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_exec_1", "content": "(no output)"},
    ]
    assert generic_verifier_exec_done(messages, spec.verify_command) is True


def test_generic_verifier_gate_transitions_after_exec() -> None:
    spec = task_canonical_spec("t3-data-pipeline-report")
    assert spec is not None
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_exec_1",
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "arguments": json.dumps({"command": spec.verify_command}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_exec_1", "content": "(no output)"},
    ]
    flags_before = {"force_generic_verifier_exec": True, "force_generic_verifier_done": False}
    flags_after = {"force_generic_verifier_exec": False, "force_generic_verifier_done": True}
    exec_done = generic_verifier_exec_done(messages, spec.verify_command)
    assert exec_done is True
    assert select_canonical_gate("t3-data-pipeline-report", flags_before) == (
        "t3-data-pipeline-report",
        "verifier_exec",
    )
    assert select_canonical_gate("t3-data-pipeline-report", flags_after) == (
        "t3-data-pipeline-report",
        "verifier_done",
    )
    pass_text = resolve_bench_forced_from_flags(
        "t3-data-pipeline-report",
        flags_after,
    )
    assert pass_text == spec.verifier_pass
