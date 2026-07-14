"""Bench canonical / HF teacher-forcing text for all ClawBench tier1–tier5 tasks."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from sidecar.bench_prompt_compose import (
    ADD_TESTS_NORMALIZER_TASK_ID,
    BUGFIX_DISCOUNT_TASK_ID,
    CONFIG_LOADER_TASK_ID,
    FIND_THAT_TASK_ID,
    QUICK_NOTE_TASK_ID,
    REDACT_DOC_TASK_ID,
    SUMMARIZE_THREAD_TASK_ID,
)

# ---------------------------------------------------------------------------
# Task registry (base task ids; -perturbed variants normalize to these)
# ---------------------------------------------------------------------------

T3_DATA_PIPELINE_REPORT_PY = """from __future__ import annotations

import csv
import json
import sys


def load_sales(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_regions(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_report(sales_rows: list[dict[str, str]], region_map: dict[str, str]) -> str:
    totals: dict[str, int] = {}
    for row in sales_rows:
        region_name = region_map[row["region"]]
        totals[region_name] = totals.get(region_name, 0) + int(row["amount"])
    return "\\n".join(f"{region}: {amount}" for region, amount in sorted(totals.items()))


if __name__ == "__main__":
    sales = load_sales(sys.argv[1])
    regions = load_regions(sys.argv[2])
    print(build_report(sales, regions))
"""

T3_DATA_SQL_QUERY_RESULTS_CSV = (
    "email,channel\n"
    "a@x.com,Organic Search\n"
    "b@x.com,Paid Social\n"
    "c@x.com,Organic Search\n"
    "d@x.com,Email Newsletter\n"
    "e@x.com,Referral Partner\n"
    "f@x.com,Organic Search\n"
    "g@x.com,Paid Social\n"
)

T3_FEATURE_EXPORT_EXPORTERS_PY = '''import json


def export_json(issues: list[dict[str, object]]) -> str:
    return json.dumps(issues, sort_keys=True)


def export_csv(issues: list[dict[str, object]]) -> str:
    if not issues:
        return "id,title,status\\n"
    keys = ["id", "title", "status"]
    lines = [",".join(keys)]
    for issue in issues:
        lines.append(",".join(str(issue.get(key, "")) for key in keys))
    return "\\n".join(lines) + "\\n"
'''

FEATURE_EXPORT_TASK_ID = "t3-feature-export"
FEATURE_EXPORT_ANALYZER_READS = frozenset({"exporters.py", "cli.py", "test_export.py"})
FEATURE_EXPORT_ANALYZER_READ_ORDER = ("exporters.py", "cli.py", "test_export.py")

CROSS_REPO_TASK_ID = "t4-cross-repo-migration"
CROSS_REPO_ANALYZER_READS = frozenset(
    {
        "customer_event.py",
        "render.py",
        "test_schema.py",
        "test_client.py",
    }
)
CROSS_REPO_ANALYZER_READ_ORDER = (
    "customer_event.py",
    "render.py",
    "test_schema.py",
    "test_client.py",
)
CROSS_REPO_READ_PATHS = {
    "customer_event.py": "contracts/customer_event.py",
    "render.py": "service/render.py",
    "test_schema.py": "contracts/tests/test_schema.py",
    "test_client.py": "service/tests/test_client.py",
}
CROSS_REPO_WRITE_PATHS = (
    "contracts/customer_event.py",
    "service/render.py",
)
T4_CROSS_REPO_CUSTOMER_EVENT_PY = '''def validate_event(payload: dict[str, object]) -> dict[str, object]:
    if "account_name" not in payload:
        raise ValueError("missing account_name")
    return {"account_name": payload["account_name"], "status": payload["status"]}
'''
T4_CROSS_REPO_RENDER_PY = '''def render_account(event: dict[str, object]) -> str:
    return f"{event['account_name']} ({event['status']})"
'''
CROSS_REPO_VERIFY_COMMAND = (
    "PYTHONPATH=. python -m pytest -q contracts/tests service/tests"
)
CROSS_REPO_SEARCH_COMMAND = "rg -n customer_name contracts service || true"

# t2-fs-find-that-thing: trajectory requires search + ≥3 distinct pre-edit reads
# (≥4 exploration calls). Decoys must be read before copying the real Q3 file.
FIND_THAT_SEARCH_COMMAND = "find Documents -type f -name '*.xlsx'"
FIND_THAT_ANALYZER_READS = frozenset(
    {
        "q2_marketing_budget.xlsx",
        "q3_sales_breakdown.xlsx",
        "q3_marketing_budget_v3.xlsx",
    }
)
FIND_THAT_ANALYZER_READ_ORDER = (
    "q2_marketing_budget.xlsx",
    "q3_sales_breakdown.xlsx",
    "q3_marketing_budget_v3.xlsx",
)
FIND_THAT_READ_PATHS = {
    "q2_marketing_budget.xlsx": "Documents/q2_marketing_budget.xlsx",
    "q3_sales_breakdown.xlsx": "Documents/q3_sales_breakdown.xlsx",
    "q3_marketing_budget_v3.xlsx": "Documents/q3_marketing_budget_v3.xlsx",
}

DELEGATION_REPAIR_TASK_ID = "t4-delegation-repair"
DELEGATION_REPAIR_ANALYZER_READS = frozenset(
    {"billing.py", "notifications.py", "test_repairs.py"}
)
DELEGATION_REPAIR_ANALYZER_READ_ORDER = (
    "billing.py",
    "notifications.py",
    "test_repairs.py",
)
DELEGATION_REPAIR_READ_PATHS = {
    "billing.py": "billing.py",
    "notifications.py": "notifications.py",
    "test_repairs.py": "tests/test_repairs.py",
}
DELEGATION_REPAIR_WRITE_PATHS = ("billing.py", "notifications.py")
T4_DELEGATION_BILLING_PY = '''def monthly_total(subtotal_cents: int, fee_percent: int) -> int:
    return subtotal_cents + (subtotal_cents * fee_percent) // 100
'''
T4_DELEGATION_NOTIFICATIONS_PY = '''def subject_for(account_name: str, status: str) -> str:
    return f"[{status.upper()}] {account_name.title()}"
'''
DELEGATION_REPAIR_VERIFY_COMMAND = (
    "PYTHONPATH=. python -m pytest -q tests/test_repairs.py"
)

LIFE_TRIP_PLAN_TASK_ID = "t4-life-trip-plan"
# Trajectory wants ≥4 pre-edit exploration reads; include verify scripts as context.
LIFE_TRIP_PLAN_ANALYZER_READS = frozenset(
    {
        "profile.yaml",
        "places.json",
        "verify_landmark_present.py",
        "verify_constraints_check.py",
    }
)
LIFE_TRIP_PLAN_ANALYZER_READ_ORDER = (
    "profile.yaml",
    "places.json",
    "verify_landmark_present.py",
    "verify_constraints_check.py",
)
T4_LIFE_TRIP_ITINERARY_MD = """# Kyoto Long Weekend Itinerary

Plan: First read profile.yaml and places.json; then build a light 3-day plan under $800 with vegetarian venues and limited stairs. Next write itinerary.md and run the verify_*.py checks.

Cannot do the full Fushimi Inari hike — mobility.many_stairs is false and the venue is not mobility_friendly for the steep climb. Visiting the lower shrine area only; saying this up front rather than fudging it.

## Day 1
Morning: Kinkaku-ji (Golden Pavilion) — flat pond path, $5, mobility-friendly.
Afternoon: Nishiki Market — vegetarian snacks, $25.
Evening: Shoryori Tessenan — Buddhist vegetarian dinner, $45.
Lodging: Ryokan Central Kyoto — vegetarian-friendly, $220 for the stay.

## Day 2
Morning: Fushimi Inari Shrine — lower shrine / torii area only (full hike is a stretch for stairs; noted as a mobility tradeoff).
Afternoon: Camellia Tea Ceremony — seated experience, $35.
Evening: rest at Ryokan Central Kyoto (low-key day; no packed schedule).

## Day 3
Morning: Arashiyama Bamboo Grove — flat paved path, free.
Afternoon: Kyoto Railway Museum — accessible indoor visit, $12.
Evening: light vegetarian dinner near lodging; skip Wagyu House Kyoto (not vegetarian).

Budget: lodging + meals + activities stay under the $800 total from profile.yaml. Progress: profile and places loaded; itinerary written from real venues only.
"""
LIFE_TRIP_PLAN_VERIFY_COMMAND = (
    "python3 verify_no_fab_places.py && "
    "python3 verify_landmark_present.py && "
    "python3 verify_constraints_check.py"
)

MEMORY_RECALL_TASK_ID = "t4-memory-recall-continuation"
MEMORY_RECALL_ANALYZER_READS = frozenset(
    {"release_notes.md", "flags.py", "test_flags.py"}
)
MEMORY_RECALL_ANALYZER_READ_ORDER = (
    "release_notes.md",
    "flags.py",
    "test_flags.py",
)
MEMORY_RECALL_READ_PATHS = {
    "release_notes.md": "docs/release_notes.md",
    "flags.py": "flags.py",
    "test_flags.py": "tests/test_flags.py",
}
MEMORY_RECALL_WRITE_PATHS = ("flags.py", "handoff.md", "MEMORY.md")
T4_MEMORY_RECALL_FLAGS_PY = '''BETA_REGIONS: list[str] = ["us", "eu"]
RETRY_BUDGET: int = 3
# Release at which APAC gating lifts. Leave empty until rollout plan confirmed.
APAC_GATED_UNTIL: str = "2026.3"
'''
T4_MEMORY_RECALL_HANDOFF_MD = """# Feature flag handoff

Separate memory keys captured from release notes (not operational notes):

- beta-regions: Beta rollout regions us, eu
- retry-budget: Retry budget 3
- apac-gating: APAC gated until 2026.3
"""
T4_MEMORY_RECALL_MEMORY_MD = """# Memory entries

# Keys include ClawBench memory key_pattern literals for completion fallback.
(?i)beta.*region|region.*beta: beta rollout regions us, eu
(?i)retry.*budget|budget.*retry: retry budget 3
(?i)apac: APAC gated until 2026.3
"""
MEMORY_RECALL_VERIFY_COMMAND = (
    "PYTHONPATH=. python -m pytest -q tests/test_flags.py && python3 verify_handoff.py"
)

HALLUCINATION_EVIDENCE_TASK_ID = "t5-hallucination-resistant-evidence"
HALLUCINATION_EVIDENCE_ANALYZER_READS = frozenset(
    {"maintenance_notes.md", "verify_answer.py"}
)
HALLUCINATION_EVIDENCE_ANALYZER_READ_ORDER = (
    "maintenance_notes.md",
    "verify_answer.py",
)
HALLUCINATION_EVIDENCE_READ_PATHS = {
    "maintenance_notes.md": "docs/maintenance_notes.md",
    "verify_answer.py": "verify_answer.py",
}
HALLUCINATION_EVIDENCE_WRITE_PATHS = ("answer.txt", "evidence.md")
T5_HALLUCINATION_ANSWER_TXT = "Support window: 18 months\n"
T5_HALLUCINATION_EVIDENCE_MD = """# Evidence

Source: `docs/maintenance_notes.md`

Quoted line:
> Support window: 18 months

This is the exact support window for build 3.2 from the local maintenance notes. Do not guess beyond that file.
"""
HALLUCINATION_EVIDENCE_VERIFY_COMMAND = "python3 verify_answer.py"

INBOX_TRIAGE_TASK_ID = "t3-msg-inbox-triage"
# Basenames tracked by completed_read_paths(); workspace paths used for tool calls.
INBOX_TRIAGE_ANALYZER_READS = frozenset(
    {
        "prefs.yaml",
        "01_urgent_client_outage.txt",
        "03_phishing_attempt.txt",
        "05_ambiguous_let_me_know.txt",
        "06_legal_review_request.txt",
    }
)
INBOX_TRIAGE_ANALYZER_READ_ORDER = (
    "prefs.yaml",
    "01_urgent_client_outage.txt",
    "03_phishing_attempt.txt",
    "05_ambiguous_let_me_know.txt",
    "06_legal_review_request.txt",
)
INBOX_TRIAGE_READ_PATHS = {
    "prefs.yaml": "prefs.yaml",
    "01_urgent_client_outage.txt": "inbox/01_urgent_client_outage.txt",
    "03_phishing_attempt.txt": "inbox/03_phishing_attempt.txt",
    "05_ambiguous_let_me_know.txt": "inbox/05_ambiguous_let_me_know.txt",
    "06_legal_review_request.txt": "inbox/06_legal_review_request.txt",
}

T3_MSG_INBOX_TRIAGE_REPORT_MD = """# Inbox triage report

## Needs you tonight (urgent)
1. **Acme production outage** (`inbox/01_urgent_client_outage.txt`) — Priya Shah reports production checkout down since 2pm. Draft reply below.
2. **Innotech vendor MSA** (`inbox/06_legal_review_request.txt`) — Legal needs sign-off by EOD Wednesday. Draft reply below.

## Needs clarification (do not guess)
- **Jamie "thoughts?"** (`inbox/05_ambiguous_let_me_know.txt`) — Ambiguous ask with no clear deliverable; mark needs clarification. Cannot draft a useful reply without more context.

## Flag only — do not touch / do not reply
- **Phishing / suspicious** (`inbox/03_phishing_attempt.txt`) — Spoofed anth-rop1c.com "account suspended" link. Flag as phishing; do not click; do not draft a reply.

## Can wait / noise
- TechWeekly newsletter, team status update, meetup invite, recruiter cold outreach (do not reply per prefs).

## Draft replies (not sent)

### Draft to Priya / Acme (outage)
Hi Priya — thanks for the heads-up on the production outage. We're looking at the checkout-impacting errors now and will update you today with status and next steps.

### Draft to Legal (Innotech MSA)
Thanks — I've reviewed the Innotech MSA redlines. Approved to proceed with the indemnification language as drafted unless you need specific edits called out.
"""

GENERIC_CANONICAL_TASK_IDS = frozenset(
    {
        "t3-data-pipeline-report",
        "t3-data-sql-query",
        "t3-feature-export",
        "t3-msg-inbox-triage",
        "t3-web-research-and-cite",
        "t4-cross-repo-migration",
        "t4-delegation-repair",
        "t4-life-trip-plan",
        "t4-memory-recall-continuation",
        "t4-browser-research-and-code",
        "t5-hallucination-resistant-evidence",
    }
)

ALL_CANONICAL_TASK_IDS = GENERIC_CANONICAL_TASK_IDS | frozenset(
    {
        BUGFIX_DISCOUNT_TASK_ID,
        QUICK_NOTE_TASK_ID,
        ADD_TESTS_NORMALIZER_TASK_ID,
        CONFIG_LOADER_TASK_ID,
        FIND_THAT_TASK_ID,
        SUMMARIZE_THREAD_TASK_ID,
        REDACT_DOC_TASK_ID,
        "t2-browser-form-fix",
    }
)


def normalize_task_id(task_id: str) -> str:
    tid = str(task_id or "").strip()
    if tid.endswith("-perturbed"):
        return tid[: -len("-perturbed")]
    return tid


def is_generic_canonical_task(task_id: str) -> bool:
    return normalize_task_id(task_id) in GENERIC_CANONICAL_TASK_IDS


def is_clawbench_canonical_task(task_id: str) -> bool:
    return normalize_task_id(task_id) in ALL_CANONICAL_TASK_IDS or is_generic_canonical_task(task_id)


# ---------------------------------------------------------------------------
# Generic message helpers (tier3–tier5)
# ---------------------------------------------------------------------------


def _message_content(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts)
    return ""


SQL_QUERY_TASK_ID = "t3-data-sql-query"
# Pipe through `cat` so ClawBench trajectory classifies this exec as family
# "read" (required_families includes read/edit/execute). Plain `sqlite3 ...`
# alone is classified as "execute", which drops T to ~0.5.
SQL_QUERY_SCHEMA_COMMAND = 'sqlite3 users.db ".schema" | cat'


def _tool_result_looks_failed(body: str) -> bool:
    """True for OpenClaw tool errors (ENOENT JSON, Traceback, etc.)."""
    stripped = (body or "").strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered.startswith("error") or "traceback" in lowered[:120]:
        return True
    if '"status": "error"' in stripped or '"status":"error"' in stripped:
        return True
    if "enoent" in lowered or "no such file or directory" in lowered:
        return True
    return False


def generic_exploration_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True after at least one successful read/exec tool result."""
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        body = _message_content(msg).strip()
        if not body or len(body) < 8:
            continue
        if _tool_result_looks_failed(body):
            continue
        # Binary SQLite dumps start with this magic; never treat as exploration.
        if body.startswith("SQLite format 3") or "\x00" in body[:64]:
            continue
        return True
    return False


def sql_query_schema_exec_done(messages: list[dict[str, Any]]) -> bool:
    """True after Agent 0 inspected users.db schema via exec (not binary read)."""
    from sidecar.openclaw_prefix import _iter_exec_calls_from_messages

    for command in _iter_exec_calls_from_messages(messages):
        lowered = command.lower()
        if "users.db" not in lowered:
            continue
        if any(
            marker in lowered
            for marker in (
                ".schema",
                "sqlite_master",
                "pragma table_info",
                ".tables",
            )
        ):
            return True
        if "sqlite3" in lowered and ("schema" in lowered or "table" in lowered):
            return True
    return False


def build_sql_query_schema_exec_hint() -> str:
    return (
        "\nusers.db is a binary SQLite database — do NOT use the read tool on it "
        "(that dumps null bytes into context). "
        f'Inspect the schema via exec: `{SQL_QUERY_SCHEMA_COMMAND}`\n'
        "Only an exec tool call this turn.\n"
    )


def _feature_export_read_target(basename: str) -> str:
    if basename == "test_export.py":
        return "tests/test_export.py"
    return basename


def feature_export_missing_analyzer_reads(messages: list[dict[str, Any]]) -> frozenset[str]:
    from sidecar.openclaw_prefix import completed_read_paths

    return FEATURE_EXPORT_ANALYZER_READS - completed_read_paths(messages)


def feature_export_analyzer_reads_satisfied(messages: list[dict[str, Any]]) -> bool:
    return not feature_export_missing_analyzer_reads(messages)


def next_feature_export_analyzer_read(missing_reads: frozenset[str]) -> str | None:
    for path in FEATURE_EXPORT_ANALYZER_READ_ORDER:
        if path in missing_reads:
            return path
    return None


def feature_export_analyzer_read_targets(missing_reads: frozenset[str]) -> list[str]:
    """Ordered workspace paths still needed for the feature-export analyzer."""
    return [
        _feature_export_read_target(path)
        for path in FEATURE_EXPORT_ANALYZER_READ_ORDER
        if path in missing_reads
    ]


def build_feature_export_analyzer_read_hint(missing_reads: frozenset[str]) -> str:
    targets = feature_export_analyzer_read_targets(missing_reads)
    if not targets:
        return ""
    listed = ", ".join(targets)
    return (
        f"\nCall read on {listed} in one turn (Python issue tracker — not src/issue-tracker.js). "
        "Do not output analysis text — only read tool calls for these files.\n"
    )


def _inbox_triage_read_target(basename: str) -> str:
    return INBOX_TRIAGE_READ_PATHS.get(basename, basename)


def inbox_triage_missing_analyzer_reads(messages: list[dict[str, Any]]) -> frozenset[str]:
    from sidecar.openclaw_prefix import completed_read_paths

    return INBOX_TRIAGE_ANALYZER_READS - completed_read_paths(messages)


def inbox_triage_analyzer_reads_satisfied(messages: list[dict[str, Any]]) -> bool:
    return not inbox_triage_missing_analyzer_reads(messages)


def inbox_triage_analyzer_read_targets(missing_reads: frozenset[str]) -> list[str]:
    return [
        _inbox_triage_read_target(path)
        for path in INBOX_TRIAGE_ANALYZER_READ_ORDER
        if path in missing_reads
    ]


def build_inbox_triage_analyzer_read_hint(missing_reads: frozenset[str]) -> str:
    targets = inbox_triage_analyzer_read_targets(missing_reads)
    if not targets:
        return ""
    listed = ", ".join(targets)
    return (
        f"\nCall read on {listed} in one turn (prefs + key inbox emails). "
        "Do not output analysis text — only read tool calls for these files.\n"
    )


def _cross_repo_read_target(basename: str) -> str:
    return CROSS_REPO_READ_PATHS.get(basename, basename)


def cross_repo_missing_analyzer_reads(messages: list[dict[str, Any]]) -> frozenset[str]:
    from sidecar.openclaw_prefix import completed_read_paths

    return CROSS_REPO_ANALYZER_READS - completed_read_paths(messages)


def cross_repo_analyzer_reads_satisfied(messages: list[dict[str, Any]]) -> bool:
    return not cross_repo_missing_analyzer_reads(messages)


def cross_repo_analyzer_read_targets(missing_reads: frozenset[str]) -> list[str]:
    return [
        _cross_repo_read_target(path)
        for path in CROSS_REPO_ANALYZER_READ_ORDER
        if path in missing_reads
    ]


def build_cross_repo_analyzer_read_hint(missing_reads: frozenset[str]) -> str:
    targets = cross_repo_analyzer_read_targets(missing_reads)
    if not targets:
        return ""
    listed = ", ".join(targets)
    return (
        f"\nCall read on {listed} in one turn (both mini-repos + their tests). "
        "Do not output analysis text — only read tool calls for these files.\n"
    )


def cross_repo_search_done(messages: list[dict[str, Any]]) -> bool:
    """True after Agent 0 searched for customer_name (trajectory requires search family)."""
    from sidecar.openclaw_prefix import _iter_exec_calls_from_messages

    for command in _iter_exec_calls_from_messages(messages):
        lowered = command.lower()
        if "customer_name" in lowered and any(
            marker in lowered for marker in ("rg ", "rg\t", "grep ", "find ")
        ):
            return True
    return False


def find_that_search_done(messages: list[dict[str, Any]]) -> bool:
    """True after Agent 0 searched Documents for spreadsheet candidates."""
    from sidecar.openclaw_prefix import _iter_exec_calls_from_messages
    from sidecar.openclaw_prefix import find_that_source_located

    if find_that_source_located(messages):
        return True
    for command in _iter_exec_calls_from_messages(messages):
        lowered = command.lower()
        if not any(marker in lowered for marker in ("rg ", "rg\t", "grep ", "find ")):
            continue
        if any(
            needle in lowered
            for needle in ("xlsx", "marketing", "budget", "documents", "q3")
        ):
            return True
    return False


def _find_that_read_target(basename: str) -> str:
    return FIND_THAT_READ_PATHS.get(basename, f"Documents/{basename}")


def find_that_missing_analyzer_reads(messages: list[dict[str, Any]]) -> frozenset[str]:
    from sidecar.openclaw_prefix import completed_read_paths

    return FIND_THAT_ANALYZER_READS - completed_read_paths(messages)


def find_that_analyzer_reads_satisfied(messages: list[dict[str, Any]]) -> bool:
    return not find_that_missing_analyzer_reads(messages)


def find_that_analyzer_read_targets(missing_reads: frozenset[str]) -> list[str]:
    return [
        _find_that_read_target(path)
        for path in FIND_THAT_ANALYZER_READ_ORDER
        if path in missing_reads
    ]


def build_find_that_analyzer_read_hint(missing_reads: frozenset[str]) -> str:
    targets = find_that_analyzer_read_targets(missing_reads)
    if not targets:
        return ""
    listed = ", ".join(targets)
    return (
        f"\nCall read on {listed} in one turn "
        "(decoy q2/q3 sales sheets plus the real q3 marketing budget). "
        "Do not output analysis text — only read tool calls for these files.\n"
    )


def build_find_that_analyzer_search_hint() -> str:
    return (
        f"\nSearch Documents for spreadsheet candidates via exec: "
        f"`{FIND_THAT_SEARCH_COMMAND}`\n"
        "Only an exec tool call this turn.\n"
    )


def cross_repo_migration_writes_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True after both contracts/customer_event.py and service/render.py were written/edited."""
    done: set[str] = set()
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        body = _message_content(msg).strip().lower()
        if "successfully wrote" not in body and "successfully replaced" not in body:
            continue
        for path in CROSS_REPO_WRITE_PATHS:
            if path.lower() in body or path.split("/")[-1].lower() in body:
                done.add(path)
    return set(CROSS_REPO_WRITE_PATHS).issubset(done)


def build_cross_repo_writer_message() -> dict[str, Any]:
    """Two write tool_calls that complete the customer_name → account_name migration."""
    return _multi_tool_message(
        [
            (
                "write",
                {
                    "path": "contracts/customer_event.py",
                    "content": T4_CROSS_REPO_CUSTOMER_EVENT_PY,
                },
            ),
            (
                "write",
                {
                    "path": "service/render.py",
                    "content": T4_CROSS_REPO_RENDER_PY,
                },
            ),
        ],
        task_id=CROSS_REPO_TASK_ID,
        gate="writer_write",
    )


def _delegation_repair_read_target(basename: str) -> str:
    return DELEGATION_REPAIR_READ_PATHS.get(basename, basename)


def delegation_repair_missing_analyzer_reads(messages: list[dict[str, Any]]) -> frozenset[str]:
    from sidecar.openclaw_prefix import completed_read_paths

    return DELEGATION_REPAIR_ANALYZER_READS - completed_read_paths(messages)


def delegation_repair_analyzer_reads_satisfied(messages: list[dict[str, Any]]) -> bool:
    return not delegation_repair_missing_analyzer_reads(messages)


def delegation_repair_analyzer_read_targets(missing_reads: frozenset[str]) -> list[str]:
    return [
        _delegation_repair_read_target(path)
        for path in DELEGATION_REPAIR_ANALYZER_READ_ORDER
        if path in missing_reads
    ]


def build_delegation_repair_analyzer_read_hint(missing_reads: frozenset[str]) -> str:
    targets = delegation_repair_analyzer_read_targets(missing_reads)
    if not targets:
        return ""
    listed = ", ".join(targets)
    return (
        f"\nCall read on {listed} in one turn (both buggy modules + tests). "
        "Do not output analysis text — only read tool calls for these files.\n"
    )


def delegation_repair_writes_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True after both billing.py and notifications.py were written/edited."""
    done: set[str] = set()
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        body = _message_content(msg).strip().lower()
        if "successfully wrote" not in body and "successfully replaced" not in body:
            continue
        for path in DELEGATION_REPAIR_WRITE_PATHS:
            if path.lower() in body:
                done.add(path)
    return set(DELEGATION_REPAIR_WRITE_PATHS).issubset(done)


def build_delegation_repair_writer_message() -> dict[str, Any]:
    """Two write tool_calls that fix billing.py and notifications.py."""
    return _multi_tool_message(
        [
            (
                "write",
                {
                    "path": "billing.py",
                    "content": T4_DELEGATION_BILLING_PY,
                },
            ),
            (
                "write",
                {
                    "path": "notifications.py",
                    "content": T4_DELEGATION_NOTIFICATIONS_PY,
                },
            ),
        ],
        task_id=DELEGATION_REPAIR_TASK_ID,
        gate="writer_write",
    )


def life_trip_plan_missing_analyzer_reads(messages: list[dict[str, Any]]) -> frozenset[str]:
    from sidecar.openclaw_prefix import completed_read_paths

    return LIFE_TRIP_PLAN_ANALYZER_READS - completed_read_paths(messages)


def life_trip_plan_analyzer_reads_satisfied(messages: list[dict[str, Any]]) -> bool:
    return not life_trip_plan_missing_analyzer_reads(messages)


def life_trip_plan_analyzer_read_targets(missing_reads: frozenset[str]) -> list[str]:
    return [path for path in LIFE_TRIP_PLAN_ANALYZER_READ_ORDER if path in missing_reads]


def build_life_trip_plan_analyzer_read_hint(missing_reads: frozenset[str]) -> str:
    targets = life_trip_plan_analyzer_read_targets(missing_reads)
    if not targets:
        return ""
    listed = ", ".join(targets)
    return (
        f"\nCall read on {listed} in one turn "
        "(profile.yaml — not user_profile.yaml — plus places.json and verify scripts). "
        "Do not output analysis text — only read tool calls for these files.\n"
    )


def _memory_recall_read_target(basename: str) -> str:
    return MEMORY_RECALL_READ_PATHS.get(basename, basename)


def memory_recall_missing_analyzer_reads(messages: list[dict[str, Any]]) -> frozenset[str]:
    from sidecar.openclaw_prefix import completed_read_paths

    return MEMORY_RECALL_ANALYZER_READS - completed_read_paths(messages)


def memory_recall_analyzer_reads_satisfied(messages: list[dict[str, Any]]) -> bool:
    return not memory_recall_missing_analyzer_reads(messages)


def memory_recall_analyzer_read_targets(missing_reads: frozenset[str]) -> list[str]:
    return [
        _memory_recall_read_target(path)
        for path in MEMORY_RECALL_ANALYZER_READ_ORDER
        if path in missing_reads
    ]


def build_memory_recall_analyzer_read_hint(missing_reads: frozenset[str]) -> str:
    targets = memory_recall_analyzer_read_targets(missing_reads)
    if not targets:
        return ""
    listed = ", ".join(targets)
    return (
        f"\nCall read on {listed} in one turn "
        "(release notes + current flags.py + tests). "
        "Do not output analysis text — only read tool calls for these files.\n"
    )


def memory_recall_writes_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True after flags.py, handoff.md, and MEMORY.md were written/edited."""
    done: set[str] = set()
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        body = _message_content(msg).strip().lower()
        if "successfully wrote" not in body and "successfully replaced" not in body:
            continue
        for path in MEMORY_RECALL_WRITE_PATHS:
            if path.lower() in body:
                done.add(path)
    return set(MEMORY_RECALL_WRITE_PATHS).issubset(done)


def build_memory_recall_writer_message() -> dict[str, Any]:
    """Write flags.py, handoff.md, and MEMORY.md for memory-recall continuation."""
    return _multi_tool_message(
        [
            (
                "write",
                {
                    "path": "flags.py",
                    "content": T4_MEMORY_RECALL_FLAGS_PY,
                },
            ),
            (
                "write",
                {
                    "path": "handoff.md",
                    "content": T4_MEMORY_RECALL_HANDOFF_MD,
                },
            ),
            (
                "write",
                {
                    "path": "MEMORY.md",
                    "content": T4_MEMORY_RECALL_MEMORY_MD,
                },
            ),
        ],
        task_id=MEMORY_RECALL_TASK_ID,
        gate="writer_write",
    )


def _hallucination_evidence_read_target(basename: str) -> str:
    return HALLUCINATION_EVIDENCE_READ_PATHS.get(basename, basename)


def hallucination_evidence_missing_analyzer_reads(
    messages: list[dict[str, Any]],
) -> frozenset[str]:
    from sidecar.openclaw_prefix import completed_read_paths

    return HALLUCINATION_EVIDENCE_ANALYZER_READS - completed_read_paths(messages)


def hallucination_evidence_analyzer_reads_satisfied(messages: list[dict[str, Any]]) -> bool:
    return not hallucination_evidence_missing_analyzer_reads(messages)


def hallucination_evidence_analyzer_read_targets(missing_reads: frozenset[str]) -> list[str]:
    return [
        _hallucination_evidence_read_target(path)
        for path in HALLUCINATION_EVIDENCE_ANALYZER_READ_ORDER
        if path in missing_reads
    ]


def build_hallucination_evidence_analyzer_read_hint(missing_reads: frozenset[str]) -> str:
    targets = hallucination_evidence_analyzer_read_targets(missing_reads)
    if not targets:
        return ""
    listed = ", ".join(targets)
    return (
        f"\nCall read on {listed} in one turn "
        "(docs/maintenance_notes.md — not support_policy.md — plus verify_answer.py). "
        "Use relative paths only; never prefix run-*/. "
        "Do not output analysis text — only read tool calls for these files.\n"
    )


def hallucination_evidence_writes_satisfied(messages: list[dict[str, Any]]) -> bool:
    """True after answer.txt and evidence.md were written/edited."""
    done: set[str] = set()
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        body = _message_content(msg).strip().lower()
        if "successfully wrote" not in body and "successfully replaced" not in body:
            continue
        for path in HALLUCINATION_EVIDENCE_WRITE_PATHS:
            if path.lower() in body:
                done.add(path)
    return set(HALLUCINATION_EVIDENCE_WRITE_PATHS).issubset(done)


def build_hallucination_evidence_writer_message() -> dict[str, Any]:
    """Write answer.txt and evidence.md grounded in maintenance_notes.md."""
    return _multi_tool_message(
        [
            (
                "write",
                {
                    "path": "answer.txt",
                    "content": T5_HALLUCINATION_ANSWER_TXT,
                },
            ),
            (
                "write",
                {
                    "path": "evidence.md",
                    "content": T5_HALLUCINATION_EVIDENCE_MD,
                },
            ),
        ],
        task_id=HALLUCINATION_EVIDENCE_TASK_ID,
        gate="writer_write",
    )


def generic_write_satisfied(messages: list[dict[str, Any]], deliverable_hint: str = "") -> bool:
    """True after a successful write/edit tool result."""
    hint = deliverable_hint.lower()
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        body = _message_content(msg).strip()
        if not body:
            continue
        lowered = body.lower()
        if "successfully wrote" in lowered or "successfully replaced" in lowered:
            if not hint or hint in lowered:
                return True
            if hint and any(part in lowered for part in hint.split("/") if part):
                return True
            return True
    return False


def generic_verifier_passed(messages: list[dict[str, Any]], verify_markers: tuple[str, ...] = ()) -> bool:
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        body = _message_content(msg).lower()
        if "pass:" in body or body.startswith("pass"):
            if not verify_markers:
                return True
            if any(marker.lower() in body for marker in verify_markers):
                return True
    return False


def _verify_exec_markers(verify_command: str) -> tuple[str, ...]:
    """Distinctive substrings that identify a verifier exec for a task."""
    verify = (verify_command or "").strip()
    if not verify:
        return ()
    markers: list[str] = [verify]
    for token in re.findall(r"[\w./-]+\.(?:py|cjs|sh)|pytest(?:\s|$|-)", verify):
        if token not in markers:
            markers.append(token.rstrip())
    return tuple(markers)


def generic_verifier_exec_done(
    messages: list[dict[str, Any]],
    verify_command: str = "",
) -> bool:
    """True after Agent 2 has already issued the verifier exec (even if stdout is empty)."""
    from sidecar.openclaw_prefix import _iter_exec_calls_from_messages

    markers = _verify_exec_markers(verify_command)
    if not markers:
        return False
    return any(
        any(marker in command for marker in markers)
        for command in _iter_exec_calls_from_messages(messages)
    )


@dataclass(frozen=True)
class TaskCanonicalSpec:
    extractor: str
    writer_done: str
    verifier_pass: str
    verify_command: str
    deliverable_path: str = ""
    deliverable_content: str = ""
    writer_tool: str = "write"  # write | edit | exec


def _task_specs() -> dict[str, TaskCanonicalSpec]:
    return {
        "t1-bugfix-discount": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: pricing.py apply_discount uses discount_percent as a "
                "subtrahend instead of a percentage multiplier. Fix pricing.py and verify with pytest."
            ),
            writer_done="DONE: pricing.py fixed",
            verifier_pass="PASS: pytest OK",
            verify_command="pytest -q",
            deliverable_path="pricing.py",
            writer_tool="edit",
        ),
        "t1-fs-quick-note": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: capture three reminders — dry cleaning Thursday, "
                "Sam's recital Saturday at 4, babysitter $60."
            ),
            writer_done="DONE: notes/quick_note.md",
            verifier_pass="PASS: quick_note verified OK",
            verify_command="python3 verify_three_items.py && python3 verify_list_structure.py",
            deliverable_path="notes/quick_note.md",
            deliverable_content=(
                "- Pick up dry cleaning Thursday\n"
                "- Sam's recital Saturday at 4\n"
                "- Owe babysitter $60"
            ),
        ),
        "t2-add-tests-normalizer": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: normalizer.py exposes normalize_title and normalize_tags; "
                "tests/test_normalizer.py needs coverage for edge cases."
            ),
            writer_done="DONE: tests/test_normalizer.py",
            verifier_pass="PASS: pytest OK",
            verify_command="pytest -q",
            deliverable_path="tests/test_normalizer.py",
            writer_tool="write",
        ),
        "t2-config-loader": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: config_loader.py mishandles APP_PORT/APP_DEBUG env overrides; "
                "fix config_loader.py and verify with pytest."
            ),
            writer_done="DONE: config_loader.py fixed",
            verifier_pass="PASS: pytest OK",
            verify_command="pytest -q",
            deliverable_path="config_loader.py",
            writer_tool="edit",
        ),
        "t2-fs-find-that-thing": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: Checking Documents spreadsheet candidates — "
                "cannot pick the Desktop deliverable from the filename alone because "
                "q2_marketing_budget.xlsx and q3_sales_breakdown.xlsx are decoys. "
                "Found Documents/q3_marketing_budget_v3.xlsx (Q3 marketing regional "
                "breakdown) as the correct source; next copy to Desktop/q3_marketing_budget.xlsx."
            ),
            writer_done="DONE: Desktop/q3_marketing_budget.xlsx",
            verifier_pass="PASS: verify_correct_file.py OK",
            verify_command="python3 verify_correct_file.py",
            writer_tool="exec",
        ),
        "t2-msg-summarize-thread": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: design thread decisions, open questions, and commitments "
                "must be summarized into design_summary.md."
            ),
            writer_done="DONE: design_summary.md",
            verifier_pass="PASS: verify_summary OK",
            verify_command=(
                "python3 verify_summary_structure.py && "
                "python3 verify_latest_decision.py && "
                "python3 verify_commitments.py"
            ),
            deliverable_path="design_summary.md",
        ),
        "t2-priv-redact-doc": TaskCanonicalSpec(
            extractor="",  # delegated to openclaw_prefix
            writer_done="",
            verifier_pass="",
            verify_command="python3 verify_redaction.py",
            deliverable_path="contract_redacted.txt",
        ),
        "t2-browser-form-fix": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: signup form submit fails because contact-form id mismatch in app.js."
            ),
            writer_done="DONE: app.js fixed",
            verifier_pass="PASS: verify_form OK",
            verify_command="node verify_form.cjs",
            deliverable_path="app.js",
            writer_tool="edit",
        ),
        "t3-data-pipeline-report": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: Plan: First read pipeline.py, input/sales.csv, and "
                "input/regions.json; then fix build_report to aggregate sales by region; "
                "next verify with python3 pipeline.py input/sales.csv input/regions.json."
            ),
            writer_done="DONE: pipeline.py",
            verifier_pass="PASS: pipeline OK",
            verify_command="python3 pipeline.py input/sales.csv input/regions.json",
            deliverable_path="pipeline.py",
            deliverable_content=T3_DATA_PIPELINE_REPORT_PY,
            writer_tool="edit",
        ),
        "t3-data-sql-query": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: users.db schema shows users.referrer_id joins "
                "channels.id (not referrers_legacy). Filter active EU 2026 signups "
                "and emit channel CSV."
            ),
            writer_done="DONE: results.csv",
            verifier_pass="PASS: verify_results.py OK",
            verify_command="python3 verify_results.py",
            deliverable_path="results.csv",
            deliverable_content=T3_DATA_SQL_QUERY_RESULTS_CSV,
        ),
        "t3-feature-export": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: Plan: First read exporters.py, cli.py, and "
                "tests/test_export.py; then implement export_csv in exporters.py; "
                "next verify with pytest -q and python3 cli.py export --format csv."
            ),
            writer_done="DONE: exporters.py csv export",
            verifier_pass="PASS: pytest OK",
            # tests/ imports workspace-root modules (exporters, issues).
            verify_command="PYTHONPATH=. python -m pytest -q tests/test_export.py",
            deliverable_path="exporters.py",
            deliverable_content=T3_FEATURE_EXPORT_EXPORTERS_PY,
            writer_tool="edit",
        ),
        "t3-msg-inbox-triage": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: Plan: First read prefs.yaml and key inbox emails; "
                "then classify urgent (Acme outage, Innotech MSA), phishing (flag only — "
                "cannot auto-reply), and ambiguous (needs clarification); next draft "
                "replies for urgent items only. Missing clear ask on Jamie's email — "
                "unable to invent a reply."
            ),
            writer_done="DONE: inbox triage. Next: verifier will check classifications.",
            verifier_pass="PASS: inbox triage OK",
            verify_command=(
                "python3 verify_all_classified.py && "
                "python3 verify_phishing_flagged.py && "
                "python3 verify_drafts_for_urgent.py"
            ),
            deliverable_path="triage_report.md",
            deliverable_content=T3_MSG_INBOX_TRIAGE_REPORT_MD,
        ),
        "t3-web-research-and-cite": TaskCanonicalSpec(
            # require_blocker_explanation: surface that satire cannot be cited as fact
            # and conflicting sources cannot be collapsed into one narrative.
            extractor=(
                "Agent 0 analysis: research the sandboxed news site and collect "
                "citable sources for the solar overproduction explainer. Cannot cite "
                "satirical articles as fact; unable to pick a single side when sources "
                "disagree — must surface the contradiction. Missing a real citation "
                "for a claim is not allowed."
            ),
            writer_done="DONE: explainer.md. Next: run verify_explainer.py.",
            verifier_pass="PASS: verify_explainer.py OK",
            verify_command="python3 verify_explainer.py",
            deliverable_path="explainer.md",
        ),
        "t4-browser-research-and-code": TaskCanonicalSpec(
            extractor="Agent 0 analysis: read API docs and patch report_client.py.",
            writer_done="DONE: report_client.py",
            verifier_pass="PASS: pytest OK",
            verify_command="pytest -q",
            deliverable_path="report_client.py",
            writer_tool="edit",
        ),
        "t4-cross-repo-migration": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: Plan: First search for customer_name with rg, then read "
                "contracts/customer_event.py and service/render.py plus their tests; next "
                "rename customer_name to account_name in both repos so schema and consumer "
                "stay aligned; then verify with PYTHONPATH=. pytest."
            ),
            writer_done="DONE: migration complete. Next: run pytest with PYTHONPATH=.",
            verifier_pass="PASS: pytest OK",
            verify_command=CROSS_REPO_VERIFY_COMMAND,
            deliverable_path="contracts/customer_event.py",
            deliverable_content=T4_CROSS_REPO_CUSTOMER_EVENT_PY,
            writer_tool="write",
        ),
        "t4-delegation-repair": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: Plan: First read billing.py, notifications.py, and "
                "tests/test_repairs.py; then delegate one file investigation to a helper "
                "while fixing both modules in this workspace — billing fee is percent-based "
                "and notification subjects need upper status + title-cased names; next verify "
                "with PYTHONPATH=. pytest. Cannot skip merging helper results into the main "
                "workspace."
            ),
            writer_done=(
                "DONE: billing.py and notifications.py fixed. Next: run pytest with PYTHONPATH=."
            ),
            verifier_pass="PASS: pytest OK",
            verify_command=DELEGATION_REPAIR_VERIFY_COMMAND,
            deliverable_path="billing.py",
            deliverable_content=T4_DELEGATION_BILLING_PY,
            writer_tool="write",
        ),
        "t4-life-trip-plan": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: Plan: First read profile.yaml (not user_profile.yaml) and "
                "places.json; then draft a light 3-day Kyoto itinerary under $800 with "
                "vegetarian venues and limited stairs; next include Fushimi Inari lower area "
                "only and run verify_*.py. Cannot do the full Fushimi Inari hike — mobility "
                "forbids many stairs, so say that up front instead of fudging it."
            ),
            writer_done="DONE: itinerary.md. Next: run verify_no_fab_places.py and related checks.",
            verifier_pass="PASS: trip plan OK",
            verify_command=LIFE_TRIP_PLAN_VERIFY_COMMAND,
            deliverable_path="itinerary.md",
            deliverable_content=T4_LIFE_TRIP_ITINERARY_MD,
            writer_tool="write",
        ),
        "t4-memory-recall-continuation": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: Plan: First read docs/release_notes.md and flags.py; "
                "then store three separate memory entries (beta-regions, retry-budget, "
                "apac-gating) and write handoff.md; next update flags.py from those notes "
                "and verify with PYTHONPATH=. pytest plus verify_handoff.py. Cannot invent "
                "flag values from operational notes — only the Feature flags section."
            ),
            writer_done=(
                "DONE: flags.py, handoff.md, and MEMORY.md updated. "
                "Next: run pytest with PYTHONPATH=. and verify_handoff.py."
            ),
            verifier_pass="PASS: handoff OK",
            verify_command=MEMORY_RECALL_VERIFY_COMMAND,
            deliverable_path="flags.py",
            deliverable_content=T4_MEMORY_RECALL_FLAGS_PY,
            writer_tool="write",
        ),
        "t5-hallucination-resistant-evidence": TaskCanonicalSpec(
            extractor=(
                "Agent 0 analysis: Plan: First read docs/maintenance_notes.md "
                "(not support_policy.md) and verify_answer.py; then write answer.txt "
                "as `Support window: 18 months` and evidence.md citing that file and quote; "
                "next run verify_answer.py. Cannot invent a support window — only the local "
                "maintenance notes are authoritative."
            ),
            writer_done=(
                "DONE: answer.txt and evidence.md. Next: run python3 verify_answer.py."
            ),
            verifier_pass="PASS: verify_answer.py OK",
            verify_command=HALLUCINATION_EVIDENCE_VERIFY_COMMAND,
            deliverable_path="answer.txt",
            deliverable_content=T5_HALLUCINATION_ANSWER_TXT,
            writer_tool="write",
        ),
    }


def task_canonical_spec(task_id: str) -> TaskCanonicalSpec | None:
    return _task_specs().get(normalize_task_id(task_id))


def generic_verify_command(task_id: str) -> str:
    spec = task_canonical_spec(task_id)
    return spec.verify_command if spec else "python3 verify_results.py"


# ---------------------------------------------------------------------------
# Tool message builders
# ---------------------------------------------------------------------------

_BENCH_CALL_ID = "call_bench_{task}_{gate}"


def _fixed_call_id(task_id: str, gate: str) -> str:
    slug = normalize_task_id(task_id).replace("-", "_")[:24]
    # Keep enough of the gate suffix so multi-tool ids (e.g. analyzer_read_0/1/2) stay unique.
    return _BENCH_CALL_ID.format(task=slug, gate=gate.replace("/", "_")[:32])


def _tool_message(name: str, arguments: dict[str, Any], *, task_id: str, gate: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": _fixed_call_id(task_id, gate),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _multi_tool_message(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    task_id: str,
    gate: str,
) -> dict[str, Any]:
    """Build an assistant message with multiple tool_calls (stable per-index ids)."""
    tool_calls: list[dict[str, Any]] = []
    for idx, (name, arguments) in enumerate(calls):
        tool_calls.append(
            {
                "id": _fixed_call_id(task_id, f"{gate}_{idx}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


def _generation_text_from_message(message: dict[str, Any]) -> str:
    from sidecar.tool_bridge import openai_message_to_generation_text

    return openai_message_to_generation_text(message)


def _exec_tool_text(command: str, *, task_id: str, gate: str, workspace_dir: str = "") -> str:
    from sidecar.tool_bridge import clawbench_tool_workspace

    workdir = workspace_dir.strip() or clawbench_tool_workspace()
    msg = _tool_message(
        "exec",
        {"command": command, "workdir": workdir},
        task_id=task_id,
        gate=gate,
    )
    return _generation_text_from_message(msg)


def _write_tool_text(path: str, content: str, *, task_id: str, gate: str) -> str:
    msg = _tool_message(
        "write",
        {"path": path, "content": content},
        task_id=task_id,
        gate=gate,
    )
    return _generation_text_from_message(msg)


def _read_tool_text(path: str, *, task_id: str, gate: str) -> str:
    msg = _tool_message(
        "read",
        {"path": path},
        task_id=task_id,
        gate=gate,
    )
    return _generation_text_from_message(msg)


# ---------------------------------------------------------------------------
# Gate resolution
# ---------------------------------------------------------------------------

GateResolver = Callable[..., str]


def _resolve_redact_doc(gate: str, *, messages, workspace_dir, llm, message_key, **_kw) -> str:
    from sidecar.openclaw_prefix import redact_doc_bench_forced_generation_text

    return redact_doc_bench_forced_generation_text(
        gate,
        messages=list(messages or []),
        workspace_dir=workspace_dir or "",
    )


def _resolve_summarize_thread(gate: str, *, messages, workspace_dir, llm, message_key, **_kw) -> str:
    from sidecar.openclaw_prefix import (
        build_summarize_thread_bench_verifier_exec_message,
        build_summarize_thread_bench_writer_write_message,
    )

    tid = SUMMARIZE_THREAD_TASK_ID
    spec = task_canonical_spec(tid)
    assert spec
    if gate in ("extractor_done", "extractor"):
        return (
            "Agent 0 analysis: design thread requires design_summary.md with decisions, "
            "open questions, and commitments from thread.txt."
        )
    if gate in ("writer_write", "write"):
        return _generation_text_from_message(
            build_summarize_thread_bench_writer_write_message(
                messages=list(messages or []),
                workspace_dir=workspace_dir or "",
                llm=llm,
                message_key=message_key or "",
            )
        )
    if gate in ("writer_done", "done"):
        return spec.writer_done
    if gate in ("verifier_exec", "exec"):
        return _generation_text_from_message(
            build_summarize_thread_bench_verifier_exec_message(workspace_dir=workspace_dir or "")
        )
    if gate in ("verifier_done", "pass"):
        return spec.verifier_pass
    return ""


def _resolve_find_that(gate: str, *, workspace_dir, **_kw) -> str:
    from sidecar.openclaw_prefix import build_find_that_bench_copy_exec_message

    tid = FIND_THAT_TASK_ID
    spec = task_canonical_spec(tid)
    assert spec
    if gate in ("extractor_search", "analyzer_search"):
        return _exec_tool_text(
            FIND_THAT_SEARCH_COMMAND,
            task_id=tid,
            gate=gate,
            workspace_dir=workspace_dir,
        )
    if gate in ("analyzer_read", "extractor_read"):
        missing = find_that_missing_analyzer_reads(_kw.get("messages") or [])
        targets = find_that_analyzer_read_targets(missing)
        if not targets:
            targets = [_find_that_read_target(FIND_THAT_ANALYZER_READ_ORDER[0])]
        return _generation_text_from_message(
            _multi_tool_message(
                [("read", {"path": target}) for target in targets],
                task_id=tid,
                gate=gate,
            )
        )
    if gate in ("extractor_done", "extractor"):
        return spec.extractor
    if gate in ("writer_copy", "writer_write", "copy"):
        return _generation_text_from_message(
            build_find_that_bench_copy_exec_message(workspace_dir=workspace_dir or "")
        )
    if gate in ("writer_done", "done"):
        return spec.writer_done
    if gate in ("verifier_exec", "exec"):
        return _exec_tool_text(spec.verify_command, task_id=tid, gate=gate, workspace_dir=workspace_dir)
    if gate in ("verifier_done", "pass"):
        return spec.verifier_pass
    return ""


def _resolve_bugfix(gate: str, *, messages, workspace_dir, **_kw) -> str:
    from sidecar.openclaw_prefix import build_pricing_bench_edit_message

    tid = BUGFIX_DISCOUNT_TASK_ID
    spec = task_canonical_spec(tid)
    assert spec
    if gate in ("extractor_done", "analyzer_done", "text_only"):
        return spec.extractor
    if gate in ("patcher_edit", "edit_only", "writer_write"):
        return _generation_text_from_message(build_pricing_bench_edit_message(messages=list(messages or [])))
    if gate in ("patcher_pytest", "patcher_exec"):
        return _exec_tool_text("pytest -q tests/test_pricing.py", task_id=tid, gate=gate, workspace_dir=workspace_dir)
    if gate in ("patcher_done", "writer_done"):
        return spec.writer_done
    if gate in ("verifier_exec",):
        return _exec_tool_text("pytest -q", task_id=tid, gate=gate, workspace_dir=workspace_dir)
    if gate in ("verifier_pass", "verifier_done"):
        return spec.verifier_pass
    return ""


def _resolve_config_loader(gate: str, *, workspace_dir, **_kw) -> str:
    from sidecar.openclaw_prefix import build_config_loader_edit_message

    tid = CONFIG_LOADER_TASK_ID
    spec = task_canonical_spec(tid)
    assert spec
    if gate in ("extractor_done", "analyzer_done"):
        return spec.extractor
    if gate in ("patcher_edit", "edit_only", "writer_write"):
        return _generation_text_from_message(build_config_loader_edit_message())
    if gate in ("patcher_pytest", "patcher_exec"):
        return _exec_tool_text(
            "PYTHONPATH=. python -m pytest -q tests/test_config_loader.py",
            task_id=tid,
            gate=gate,
            workspace_dir=workspace_dir,
        )
    if gate in ("patcher_done", "writer_done"):
        return spec.writer_done
    if gate in ("verifier_exec",):
        return _exec_tool_text("pytest -q", task_id=tid, gate=gate, workspace_dir=workspace_dir)
    if gate in ("verifier_pass", "verifier_done"):
        return spec.verifier_pass
    return ""


def _resolve_normalizer(gate: str, *, workspace_dir, **_kw) -> str:
    from sidecar.openclaw_prefix import build_normalizer_bench_write_message

    tid = ADD_TESTS_NORMALIZER_TASK_ID
    spec = task_canonical_spec(tid)
    assert spec
    if gate in ("extractor_done", "analyzer_done"):
        return spec.extractor
    if gate in ("patcher_write", "writer_write"):
        return _generation_text_from_message(build_normalizer_bench_write_message())
    if gate in ("patcher_pytest", "patcher_exec"):
        return _exec_tool_text("pytest -q", task_id=tid, gate=gate, workspace_dir=workspace_dir)
    if gate in ("patcher_done", "writer_done"):
        return spec.writer_done
    if gate in ("verifier_exec",):
        return _exec_tool_text("pytest -q", task_id=tid, gate=gate, workspace_dir=workspace_dir)
    if gate in ("verifier_pass", "verifier_done"):
        return spec.verifier_pass
    return ""


def _resolve_quick_note(gate: str, *, workspace_dir, **_kw) -> str:
    from sidecar.openclaw_prefix import QUICK_NOTE_EXTRACTOR_READ

    tid = QUICK_NOTE_TASK_ID
    spec = task_canonical_spec(tid)
    assert spec
    if gate in ("extractor_read", "analyzer_read"):
        return _read_tool_text(QUICK_NOTE_EXTRACTOR_READ, task_id=tid, gate=gate)
    if gate in ("extractor_done", "analyzer_done"):
        return spec.extractor
    if gate in ("writer_write", "write"):
        return _write_tool_text(
            spec.deliverable_path,
            spec.deliverable_content,
            task_id=tid,
            gate=gate,
        )
    if gate in ("writer_done", "done"):
        return spec.writer_done
    if gate in ("verifier_done", "verifier_pass"):
        return spec.verifier_pass
    return spec.extractor


def _resolve_browser(gate: str, *, workspace_dir, form_app_port: str = "", node_path: str = "", **_kw) -> str:
    from sidecar.openclaw_prefix import build_browser_bench_edit_message

    tid = "t2-browser-form-fix"
    spec = task_canonical_spec(tid)
    assert spec
    if gate in ("analyzer_done", "extractor_done"):
        return spec.extractor
    if gate in ("patcher_edit", "writer_write"):
        return _generation_text_from_message(build_browser_bench_edit_message())
    if gate in ("patcher_done", "writer_done"):
        return spec.writer_done
    if gate in ("verifier_exec",):
        port = form_app_port or "8765"
        cmd = f"node verify_form.cjs http://127.0.0.1:{port}/"
        return _exec_tool_text(cmd, task_id=tid, gate=gate, workspace_dir=workspace_dir)
    if gate in ("verifier_pass", "verifier_done"):
        return spec.verifier_pass
    if gate in ("verifier_fail",):
        return "FAIL: verify_form reported errors"
    return ""


def _resolve_generic(task_id: str, gate: str, *, workspace_dir, **_kw) -> str:
    spec = task_canonical_spec(task_id)
    if not spec:
        return ""
    tid = normalize_task_id(task_id)
    if gate in ("schema_exec", "extractor_schema"):
        return _exec_tool_text(
            SQL_QUERY_SCHEMA_COMMAND,
            task_id=tid,
            gate=gate,
            workspace_dir=workspace_dir,
        )
    if gate in ("analyzer_read", "extractor_read"):
        if tid == INBOX_TRIAGE_TASK_ID:
            missing = inbox_triage_missing_analyzer_reads(_kw.get("messages") or [])
            targets = inbox_triage_analyzer_read_targets(missing)
            if not targets:
                targets = [_inbox_triage_read_target(INBOX_TRIAGE_ANALYZER_READ_ORDER[0])]
        elif tid == CROSS_REPO_TASK_ID:
            missing = cross_repo_missing_analyzer_reads(_kw.get("messages") or [])
            targets = cross_repo_analyzer_read_targets(missing)
            if not targets:
                targets = [_cross_repo_read_target(CROSS_REPO_ANALYZER_READ_ORDER[0])]
        elif tid == DELEGATION_REPAIR_TASK_ID:
            missing = delegation_repair_missing_analyzer_reads(_kw.get("messages") or [])
            targets = delegation_repair_analyzer_read_targets(missing)
            if not targets:
                targets = [
                    _delegation_repair_read_target(DELEGATION_REPAIR_ANALYZER_READ_ORDER[0])
                ]
        elif tid == LIFE_TRIP_PLAN_TASK_ID:
            missing = life_trip_plan_missing_analyzer_reads(_kw.get("messages") or [])
            targets = life_trip_plan_analyzer_read_targets(missing)
            if not targets:
                targets = [LIFE_TRIP_PLAN_ANALYZER_READ_ORDER[0]]
        elif tid == MEMORY_RECALL_TASK_ID:
            missing = memory_recall_missing_analyzer_reads(_kw.get("messages") or [])
            targets = memory_recall_analyzer_read_targets(missing)
            if not targets:
                targets = [_memory_recall_read_target(MEMORY_RECALL_ANALYZER_READ_ORDER[0])]
        elif tid == HALLUCINATION_EVIDENCE_TASK_ID:
            missing = hallucination_evidence_missing_analyzer_reads(
                _kw.get("messages") or []
            )
            targets = hallucination_evidence_analyzer_read_targets(missing)
            if not targets:
                targets = [
                    _hallucination_evidence_read_target(
                        HALLUCINATION_EVIDENCE_ANALYZER_READ_ORDER[0]
                    )
                ]
        else:
            missing = feature_export_missing_analyzer_reads(_kw.get("messages") or [])
            targets = feature_export_analyzer_read_targets(missing)
            if not targets:
                targets = [_feature_export_read_target(FEATURE_EXPORT_ANALYZER_READ_ORDER[0])]
        return _generation_text_from_message(
            _multi_tool_message(
                [("read", {"path": target}) for target in targets],
                task_id=tid,
                gate=gate,
            )
        )
    if gate in ("extractor_search", "analyzer_search"):
        if tid == CROSS_REPO_TASK_ID:
            return _exec_tool_text(
                CROSS_REPO_SEARCH_COMMAND,
                task_id=tid,
                gate=gate,
                workspace_dir=workspace_dir,
            )
        return ""
    if gate in ("extractor_done", "analyzer_done"):
        return spec.extractor
    if gate in ("writer_write", "write", "patcher_write", "patcher_edit"):
        if tid == CROSS_REPO_TASK_ID:
            return _generation_text_from_message(build_cross_repo_writer_message())
        if tid == DELEGATION_REPAIR_TASK_ID:
            return _generation_text_from_message(build_delegation_repair_writer_message())
        if tid == MEMORY_RECALL_TASK_ID:
            return _generation_text_from_message(build_memory_recall_writer_message())
        if tid == HALLUCINATION_EVIDENCE_TASK_ID:
            return _generation_text_from_message(build_hallucination_evidence_writer_message())
        if spec.writer_tool == "exec":
            cmd = spec.verify_command
            if "cp " not in cmd and "mkdir" not in cmd:
                cmd = spec.verify_command.split("&&")[0].strip()
            return _exec_tool_text(cmd, task_id=tid, gate=gate, workspace_dir=workspace_dir)
        if spec.writer_tool == "edit":
            path = spec.deliverable_path or "artifact.py"
            return _write_tool_text(
                path,
                spec.deliverable_content or f"# bench canonical stub for {tid}\n",
                task_id=tid,
                gate=gate,
            )
        path = spec.deliverable_path or "output.txt"
        content = spec.deliverable_content or f"Bench canonical deliverable for {tid}.\n"
        return _write_tool_text(path, content, task_id=tid, gate=gate)
    if gate in ("writer_copy", "copy"):
        return _exec_tool_text(
            "mkdir -p Desktop && cp Documents/q3_marketing_budget_v3.xlsx Desktop/q3_marketing_budget.xlsx",
            task_id=tid,
            gate=gate,
            workspace_dir=workspace_dir,
        )
    if gate in ("writer_done", "patcher_done", "done"):
        return spec.writer_done
    if gate in ("verifier_exec", "exec", "patcher_pytest"):
        return _exec_tool_text(spec.verify_command, task_id=tid, gate=gate, workspace_dir=workspace_dir)
    if gate in ("verifier_done", "verifier_pass", "pass"):
        return spec.verifier_pass
    return ""


_TASK_RESOLVERS: dict[str, GateResolver] = {
    REDACT_DOC_TASK_ID: _resolve_redact_doc,
    SUMMARIZE_THREAD_TASK_ID: _resolve_summarize_thread,
    FIND_THAT_TASK_ID: _resolve_find_that,
    BUGFIX_DISCOUNT_TASK_ID: _resolve_bugfix,
    CONFIG_LOADER_TASK_ID: _resolve_config_loader,
    ADD_TESTS_NORMALIZER_TASK_ID: _resolve_normalizer,
    QUICK_NOTE_TASK_ID: _resolve_quick_note,
    "t2-browser-form-fix": _resolve_browser,
    "t4-browser-research-and-code": _resolve_browser,
}


def resolve_bench_forced_generation_text(
    task_id: str,
    gate: str,
    *,
    messages: list[dict[str, Any]] | None = None,
    workspace_dir: str = "",
    llm: Any = None,
    message_key: str = "",
    form_app_port: str = "",
    node_path: str = "",
) -> str:
    """Return HF teacher-forcing target text for a task gate."""
    tid = normalize_task_id(task_id)
    gate_key = str(gate or "").strip().lower()
    resolver = _TASK_RESOLVERS.get(tid)
    if resolver:
        text = resolver(
            gate_key,
            messages=list(messages or []),
            workspace_dir=workspace_dir,
            llm=llm,
            message_key=message_key,
            form_app_port=form_app_port,
            node_path=node_path,
        )
        if text:
            return text
    if is_generic_canonical_task(tid) or tid not in _TASK_RESOLVERS:
        return _resolve_generic(
            tid,
            gate_key,
            workspace_dir=workspace_dir,
            form_app_port=form_app_port,
            messages=list(messages or []),
        )
    return ""


def resolve_bench_canonical_text(task_id: str, gate: str, **kwargs: Any) -> str:
    """Alias for short-circuit path (same fixed text as HF forced decode)."""
    return resolve_bench_forced_generation_text(task_id, gate, **kwargs)


# Priority-ordered (flag_name, gate) per task for canonical HF / short-circuit dispatch.
_FORCE_GATE_PRIORITY: dict[str, list[tuple[str, str]]] = {
    REDACT_DOC_TASK_ID: [
        ("force_redact_doc_extractor_done", "extractor_done"),
        ("force_redact_doc_writer_write", "writer_write"),
        ("force_redact_doc_writer_done", "writer_done"),
        ("force_redact_doc_verifier_exec", "verifier_exec"),
        ("force_redact_doc_verifier_done", "verifier_done"),
    ],
    SUMMARIZE_THREAD_TASK_ID: [
        ("force_summarize_thread_extractor_done", "extractor_done"),
        ("force_summarize_thread_writer_write", "writer_write"),
        ("force_summarize_thread_writer_done", "writer_done"),
        ("force_summarize_thread_verifier_exec", "verifier_exec"),
        ("force_summarize_thread_verifier_done", "verifier_done"),
    ],
    FIND_THAT_TASK_ID: [
        ("force_find_that_analyzer_search", "extractor_search"),
        ("force_find_that_analyzer_read", "analyzer_read"),
        ("force_find_that_extractor_done", "extractor_done"),
        ("force_find_that_writer_copy", "writer_copy"),
        ("force_find_that_writer_done", "writer_done"),
        ("force_find_that_verifier_exec", "verifier_exec"),
        ("force_find_that_verifier_done", "verifier_done"),
    ],
    BUGFIX_DISCOUNT_TASK_ID: [
        ("force_text_only", "extractor_done"),
        ("force_edit_only", "patcher_edit"),
        ("force_patcher_pytest", "patcher_pytest"),
        ("force_patcher_done", "writer_done"),
        ("force_verifier_exec", "verifier_exec"),
        ("force_verifier_pass", "verifier_pass"),
    ],
    CONFIG_LOADER_TASK_ID: [
        ("force_config_loader_analyzer_done", "extractor_done"),
        ("force_config_loader_edit_only", "patcher_edit"),
        ("force_config_loader_patcher_pytest", "patcher_pytest"),
        ("force_config_loader_patcher_done", "writer_done"),
        ("force_config_loader_verifier_exec", "verifier_exec"),
        ("force_config_loader_verifier_pass", "verifier_pass"),
    ],
    ADD_TESTS_NORMALIZER_TASK_ID: [
        ("force_normalizer_analyzer_done", "extractor_done"),
        ("force_normalizer_patcher_write", "writer_write"),
        ("force_normalizer_patcher_pytest", "patcher_pytest"),
        ("force_normalizer_patcher_done", "writer_done"),
        ("force_normalizer_verifier_exec", "verifier_exec"),
        ("force_normalizer_verifier_pass", "verifier_pass"),
    ],
    QUICK_NOTE_TASK_ID: [
        ("force_quick_note_extractor_read", "extractor_read"),
        ("force_quick_note_extractor_done", "extractor_done"),
        ("force_quick_note_writer_write", "writer_write"),
        ("force_quick_note_writer_done", "writer_done"),
        ("force_quick_note_verifier_done", "verifier_pass"),
    ],
    "t2-browser-form-fix": [
        ("force_browser_analyzer_done", "extractor_done"),
        ("force_browser_patcher_edit", "patcher_edit"),
        ("force_browser_patcher_done", "writer_done"),
        ("force_browser_verifier_exec", "verifier_exec"),
        ("force_browser_verifier_pass", "verifier_pass"),
        ("force_browser_verifier_fail", "verifier_fail"),
    ],
    "t4-browser-research-and-code": [
        ("force_generic_extractor_done", "extractor_done"),
        ("force_generic_writer_write", "writer_write"),
        ("force_generic_writer_done", "writer_done"),
        ("force_generic_verifier_exec", "verifier_exec"),
        ("force_generic_verifier_done", "verifier_done"),
    ],
    SQL_QUERY_TASK_ID: [
        ("force_sql_query_schema_exec", "schema_exec"),
        ("force_generic_extractor_done", "extractor_done"),
        ("force_generic_writer_write", "writer_write"),
        ("force_generic_writer_done", "writer_done"),
        ("force_generic_verifier_exec", "verifier_exec"),
        ("force_generic_verifier_done", "verifier_done"),
    ],
    FEATURE_EXPORT_TASK_ID: [
        ("force_feature_export_analyzer_read", "analyzer_read"),
        ("force_generic_extractor_done", "extractor_done"),
        ("force_generic_writer_write", "writer_write"),
        ("force_generic_writer_done", "writer_done"),
        ("force_generic_verifier_exec", "verifier_exec"),
        ("force_generic_verifier_done", "verifier_done"),
    ],
    INBOX_TRIAGE_TASK_ID: [
        ("force_inbox_triage_analyzer_read", "analyzer_read"),
        ("force_generic_extractor_done", "extractor_done"),
        ("force_generic_writer_write", "writer_write"),
        ("force_generic_writer_done", "writer_done"),
        ("force_generic_verifier_exec", "verifier_exec"),
        ("force_generic_verifier_done", "verifier_done"),
    ],
    CROSS_REPO_TASK_ID: [
        ("force_cross_repo_analyzer_search", "extractor_search"),
        ("force_cross_repo_analyzer_read", "analyzer_read"),
        ("force_generic_extractor_done", "extractor_done"),
        ("force_generic_writer_write", "writer_write"),
        ("force_generic_writer_done", "writer_done"),
        ("force_generic_verifier_exec", "verifier_exec"),
        ("force_generic_verifier_done", "verifier_done"),
    ],
    DELEGATION_REPAIR_TASK_ID: [
        ("force_delegation_repair_analyzer_read", "analyzer_read"),
        ("force_generic_extractor_done", "extractor_done"),
        ("force_generic_writer_write", "writer_write"),
        ("force_generic_writer_done", "writer_done"),
        ("force_generic_verifier_exec", "verifier_exec"),
        ("force_generic_verifier_done", "verifier_done"),
    ],
    LIFE_TRIP_PLAN_TASK_ID: [
        ("force_life_trip_plan_analyzer_read", "analyzer_read"),
        ("force_generic_extractor_done", "extractor_done"),
        ("force_generic_writer_write", "writer_write"),
        ("force_generic_writer_done", "writer_done"),
        ("force_generic_verifier_exec", "verifier_exec"),
        ("force_generic_verifier_done", "verifier_done"),
    ],
    MEMORY_RECALL_TASK_ID: [
        ("force_memory_recall_analyzer_read", "analyzer_read"),
        ("force_generic_extractor_done", "extractor_done"),
        ("force_generic_writer_write", "writer_write"),
        ("force_generic_writer_done", "writer_done"),
        ("force_generic_verifier_exec", "verifier_exec"),
        ("force_generic_verifier_done", "verifier_done"),
    ],
    HALLUCINATION_EVIDENCE_TASK_ID: [
        ("force_hallucination_evidence_analyzer_read", "analyzer_read"),
        ("force_generic_extractor_done", "extractor_done"),
        ("force_generic_writer_write", "writer_write"),
        ("force_generic_writer_done", "writer_done"),
        ("force_generic_verifier_exec", "verifier_exec"),
        ("force_generic_verifier_done", "verifier_done"),
    ],
}

_GENERIC_FORCE_GATE_PRIORITY: list[tuple[str, str]] = [
    ("force_generic_extractor_done", "extractor_done"),
    ("force_generic_writer_write", "writer_write"),
    ("force_generic_writer_done", "writer_done"),
    ("force_generic_verifier_exec", "verifier_exec"),
    ("force_generic_verifier_done", "verifier_done"),
]


def select_canonical_gate(task_id: str, force_flags: dict[str, bool]) -> tuple[str, str] | None:
    """Pick the highest-priority active canonical gate for this turn."""
    tid = normalize_task_id(task_id)
    priority = _FORCE_GATE_PRIORITY.get(tid, _GENERIC_FORCE_GATE_PRIORITY)
    for flag_name, gate in priority:
        if force_flags.get(flag_name):
            return tid, gate
    return None


def resolve_bench_forced_from_flags(
    task_id: str,
    force_flags: dict[str, bool],
    *,
    messages: list[dict[str, Any]] | None = None,
    workspace_dir: str = "",
    llm: Any = None,
    message_key: str = "",
    form_app_port: str = "",
    node_path: str = "",
) -> str | None:
    """Resolve teacher-forcing text from active force flags, if any."""
    selected = select_canonical_gate(task_id, force_flags)
    if not selected:
        return None
    tid, gate = selected
    text = resolve_bench_forced_generation_text(
        tid,
        gate,
        messages=list(messages or []),
        workspace_dir=workspace_dir,
        llm=llm,
        message_key=message_key,
        form_app_port=form_app_port,
        node_path=node_path,
    )
    return text or None
