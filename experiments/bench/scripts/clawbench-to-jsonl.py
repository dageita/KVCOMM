#!/usr/bin/env python3
"""Convert ClawBench YAML tasks to KVCOMM bench JSONL (Chain pipeline templates)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROLE_PROMPT_SLOT = "{{role_prompt}}"

WORKSPACE_HINT = (
    "OpenClaw tool cwd: default agent workspace (~/.openclaw/workspace). "
    "Use relative paths only. Never use absolute paths or other run-* directories."
)

FAMILY_ROLES = {
    "tools": ["Extractor", "Writer", "Verifier"],
    "coding": ["Analyzer", "Patcher", "Verifier"],
    "repo": ["Analyzer", "Patcher", "Verifier"],
    "browser": ["Analyzer", "Patcher", "Verifier"],
}

FAMILY_AGENT_0_JOB = {
    "tools": (
        "Your job (Agent 0 - Extractor): Read the user request and explore the workspace "
        "with read/search tools. Summarize what must be delivered and which files or data matter. "
        "Do not write or edit files yet."
    ),
    "coding": (
        "Your job (Agent 0 - Analyzer): Read relevant source and test files. "
        "Summarize the problem, key files, and what needs to change. Do not edit files yet."
    ),
    "repo": (
        "Your job (Agent 0 - Analyzer): Read the relevant modules and tests. "
        "Explain the bug or missing behavior and which files are involved. Do not edit yet."
    ),
    "browser": (
        "Your job (Agent 0 - Analyzer): Use the browser tool (target: host) to reproduce the issue. "
        "Summarize what is broken and which frontend files likely need changes."
    ),
}

FAMILY_AGENT_1_JOB = {
    "tools": (
        "Your job (Agent 1 - Writer): Use edit/write/exec tools to produce the deliverable "
        "described in the user request. Build on Agent 0's analysis."
    ),
    "coding": (
        "Your job (Agent 1 - Patcher): Read first, then edit only the files needed to satisfy "
        "the user request. Run pytest or other verification via exec after changes."
    ),
    "repo": (
        "Your job (Agent 1 - Patcher): Fix the code across the relevant files. "
        "Run the project's pytest suite via exec after editing."
    ),
    "browser": (
        "Your job (Agent 1 - Patcher): Fix the frontend code in the workspace, then re-check "
        "with the browser tool or exec."
    ),
}

FAMILY_AGENT_2_JOB = {
    "tools": (
        "Your job (Agent 2 - Verifier): Inspect the workspace outputs, run checks if helpful, "
        "and confirm the deliverable meets the user request. Output a final summary."
    ),
    "coding": (
        "Your job (Agent 2 - Verifier): Run pytest -q via exec. If tests pass, report PASS and stop. "
        "If tests fail, read the relevant files and fix only what is needed."
    ),
    "repo": (
        "Your job (Agent 2 - Verifier): Run pytest -q via exec from the workspace root. "
        "Report PASS or apply minimal fixes until verification passes."
    ),
    "browser": (
        "Your job (Agent 2 - Verifier): Verify the page works in the browser or via execution checks. "
        "Report whether the task is complete."
    ),
}

TEXT_AGENT_0 = (
    f"{ROLE_PROMPT_SLOT}\n\n"
    "User request:\n{{task_body}}\n\n"
    "Your job (Agent 0 - Extractor): List all items or key facts the user needs. "
    "Output plain text only. Do not use tools.\n"
)


def resolve_tasks_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        Path(__file__).resolve().parents[4] / "clawbench" / "tasks-public",
        Path.cwd() / "tasks-public",
        Path.cwd().parent / "clawbench" / "tasks-public",
    ]
    for candidate in candidates:
        if (candidate / "tier1").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate clawbench/tasks-public")


def load_task_yaml(tasks_dir: Path, task_id: str) -> dict:
    for tier_dir in sorted(tasks_dir.glob("tier*")):
        path = tier_dir / f"{task_id}.yaml"
        if path.is_file():
            import yaml

            with path.open(encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
    raise FileNotFoundError(f"Task YAML not found for {task_id}")


def discover_task_ids(tasks_dir: Path, *, tier: str, exclude_perturbed: bool) -> list[str]:
    tier_dir = tasks_dir / tier
    if not tier_dir.is_dir():
        raise FileNotFoundError(f"Tier directory not found: {tier_dir}")
    task_ids: list[str] = []
    for path in sorted(tier_dir.glob("*.yaml")):
        task_id = path.stem
        if exclude_perturbed and task_id.endswith("-perturbed"):
            continue
        task_ids.append(task_id)
    if not task_ids:
        raise FileNotFoundError(f"No tasks found under {tier_dir}")
    return task_ids


def all_task_prompt(task: dict) -> str:
    user = task.get("user") or {}
    turns = user.get("turns") or []
    messages = [str(turn.get("message") or "").strip() for turn in turns if turn.get("message")]
    if messages:
        if len(messages) == 1:
            return messages[0]
        parts = [messages[0]]
        for index, message in enumerate(messages[1:], start=2):
            parts.append(f"Follow-up {index}: {message}")
        return "\n\n".join(parts)
    phases = task.get("phases") or []
    if phases:
        phase_user = phases[0].get("user") or {}
        phase_turns = phase_user.get("turns") or []
        if phase_turns and phase_turns[0].get("message"):
            return str(phase_turns[0]["message"])
    return str(task.get("name") or task.get("id") or "")


def roles_for_family(family: str, agent_count: int) -> list[str]:
    base = list(FAMILY_ROLES.get(family, FAMILY_ROLES["tools"]))
    roles = base[:agent_count]
    while len(roles) < agent_count:
        roles.append(f"Worker{len(roles)}")
    return roles


def has_pytest_checks(task: dict) -> bool:
    completion = task.get("completion") or {}
    for check in completion.get("execution_checks") or []:
        command = str(check.get("command") or "").lower()
        if "pytest" in command:
            return True
    return False


def build_tool_constraints(task: dict, agent_index: int, roles: list[str]) -> str:
    family = str(task.get("family") or "tools").lower()
    trajectory = task.get("trajectory") or {}
    required = [str(item).lower() for item in (trajectory.get("required_families") or [])]
    lines = [WORKSPACE_HINT]
    role = roles[agent_index].lower()

    if agent_index == 0:
        if "browser" in required or family == "browser":
            lines.append("Use the browser tool (target: host) to reproduce the issue before editing files.")
        if "search" in required:
            lines.append("Use read and search tools to explore the workspace before making changes.")
        elif "read" in required:
            lines.append("Use read tools on relevant files before making changes.")
        lines.append("Do NOT edit, write, or exec unless your role requires it.")
    elif agent_index == 1:
        if family in {"coding", "repo"} or "edit" in required:
            lines.append("Read source files first, then edit only what is needed.")
        if has_pytest_checks(task):
            lines.append("After editing, run `pytest -q` via exec to verify.")
        elif family == "browser":
            lines.append("Fix frontend files, then re-check in the browser or via exec.")
        elif "edit" in required or "write" in required:
            lines.append("Use edit/write/exec as needed to produce the deliverable.")
    else:
        if has_pytest_checks(task):
            lines.append("Run `pytest -q` via exec. If tests pass, report PASS and stop.")
        elif family == "browser":
            lines.append("Verify the fix with browser or execution checks before finishing.")
        else:
            lines.append("Inspect outputs and confirm the user request is fully satisfied.")

    if "verifier" in role and trajectory.get("require_self_verification"):
        lines.append("Run self-verification before declaring the task complete.")

    return "\n\n".join(lines)


def build_text_agent_tasks(roles: list[str]) -> dict[str, str]:
    tasks: dict[str, str] = {"agent_0": TEXT_AGENT_0}
    for index in range(1, len(roles)):
        prev_index = index - 1
        tasks[f"agent_{index}"] = (
            f"{ROLE_PROMPT_SLOT}\n\n"
            "User request:\n{{task_body}}\n\n"
            f"Output from Agent {prev_index} ({roles[prev_index]}):\n\n"
            f"{{{{agent_{prev_index}_current}}}}\n\n"
            f"Your job (Agent {index} - {roles[index]}): Build on the previous agent's output. "
            "Output plain text only. Do not use tools.\n"
        )
    return tasks


def build_capability_agent_tasks(task: dict, roles: list[str], agent_count: int) -> dict[str, str]:
    if agent_count < 3:
        raise ValueError("capability templates require at least 3 agents")
    family = str(task.get("family") or "tools").lower()
    job0 = FAMILY_AGENT_0_JOB.get(family, FAMILY_AGENT_0_JOB["tools"])
    job1 = FAMILY_AGENT_1_JOB.get(family, FAMILY_AGENT_1_JOB["tools"])
    job2 = FAMILY_AGENT_2_JOB.get(family, FAMILY_AGENT_2_JOB["tools"])

    tasks: dict[str, str] = {
        "agent_0": (
            f"{ROLE_PROMPT_SLOT}\n\n"
            "User request:\n{{task_body}}\n\n"
            f"{job0}\n"
        ),
        "agent_1": (
            f"{ROLE_PROMPT_SLOT}\n\n"
            "User request:\n{{task_body}}\n\n"
            f"Output from Agent 0 ({roles[0]}):\n\n{{agent_0_current}}\n\n"
            f"{job1}\n"
        ),
        "agent_2": (
            f"{ROLE_PROMPT_SLOT}\n\n"
            "User request:\n{{task_body}}\n\n"
            f"Output from Agent 1 ({roles[1]}):\n\n{{agent_1_current}}\n\n"
            f"{job2}\n"
        ),
    }
    for index in range(3, agent_count):
        prev_index = index - 1
        tasks[f"agent_{index}"] = (
            f"{ROLE_PROMPT_SLOT}\n\n"
            "User request:\n{{task_body}}\n\n"
            f"Output from Agent {prev_index} ({roles[prev_index]}):\n\n"
            f"{{agent_{prev_index}_current}}\n\n"
            f"Your job (Agent {index} - {roles[index]}): Continue verification and produce the final summary.\n"
        )
    return tasks


def build_tool_constraints_map(task: dict, roles: list[str], agent_count: int) -> dict[str, str]:
    return {
        f"agent_{index}": build_tool_constraints(task, index, roles)
        for index in range(agent_count)
    }


ADD_TESTS_NORMALIZER_TASK_ID = "t2-add-tests-normalizer"


def _add_tests_normalizer_overrides() -> dict:
    return {
        "tool_constraints": {
            "agent_0": (
                f"{WORKSPACE_HINT}\n\n"
                "Read normalizer.py at the workspace root first. "
                "The module is NOT named text_normalization_module.py. Do not edit yet."
            ),
            "agent_1": (
                f"{WORKSPACE_HINT}\n\n"
                "Read normalizer.py first. Create tests/test_normalizer.py with pytest cases for "
                "whitespace cleanup, emoji stripping in titles, and blank tag handling. "
                "Use `from normalizer import normalize_title, normalize_tags` (not a relative import). "
                "If edit fails because oldText is missing but the file already looks correct, run "
                "`pytest -q tests/test_normalizer.py` instead of retrying the same edit."
            ),
            "agent_2": (
                f"{WORKSPACE_HINT}\n\n"
                "Run `pytest -q tests/test_normalizer.py` via exec. "
                "If tests pass, report PASS and stop."
            ),
        },
        "capability_agent_tasks": {
            "agent_0": (
                f"{ROLE_PROMPT_SLOT}\n\n"
                "User request:\n{{task_body}}\n\n"
                "Your job (Agent 0 - Analyzer): Read normalizer.py at the workspace root. "
                "Summarize normalize_title and normalize_tags behavior and what tests are missing. "
                "Do not edit files yet."
            ),
            "agent_1": (
                f"{ROLE_PROMPT_SLOT}\n\n"
                "User request:\n{{task_body}}\n\n"
                "Output from Agent 0 (Analyzer):\n\n{{agent_0_current}}\n\n"
                "Your job (Agent 1 - Patcher): Read normalizer.py, then write tests/test_normalizer.py "
                "with focused pytest coverage for whitespace cleanup, emoji stripping in titles, "
                "and blank tag handling. Use `from normalizer import normalize_title, normalize_tags`. "
                "If the test file already has the correct import, run `pytest -q tests/test_normalizer.py` "
                "via exec — do not retry identical edits."
            ),
            "agent_2": (
                f"{ROLE_PROMPT_SLOT}\n\n"
                "User request:\n{{task_body}}\n\n"
                "Output from Agent 1 (Patcher):\n\n{{agent_1_current}}\n\n"
                "Your job (Agent 2 - Verifier): Run `pytest -q tests/test_normalizer.py` via exec. "
                "If tests pass, report PASS and stop. Otherwise fix tests or normalizer.py minimally."
            ),
        },
    }


TASK_OVERRIDES: dict[str, dict] = {
    ADD_TESTS_NORMALIZER_TASK_ID: _add_tests_normalizer_overrides(),
}


def apply_task_overrides(row: dict) -> dict:
    override = TASK_OVERRIDES.get(row["task_id"])
    if not override:
        return row
    if override.get("tool_constraints"):
        row["tool_constraints"] = override["tool_constraints"]
    if override.get("capability_agent_tasks"):
        row["capability_agent_tasks"] = override["capability_agent_tasks"]
    return row


def task_row_from_yaml(task: dict, *, agent_count: int, template: str) -> dict:
    task_id = str(task["id"])
    task_body = all_task_prompt(task)
    family = str(task.get("family") or "tools").lower()
    roles = roles_for_family(family, agent_count)

    setup = task.get("setup") or {}
    row = {
        "task_id": task_id,
        "task_body": task_body,
        "clawbench_ref": {
            "yaml_id": task_id,
            "asset_packs": list(setup.get("asset_packs") or []),
            "tier": str(task.get("tier") or ""),
            "family": family,
        },
        "agent_roles": roles,
        "agent_tasks": build_text_agent_tasks(roles),
    }
    if template in {"capability", "all"}:
        row["capability_agent_tasks"] = build_capability_agent_tasks(task, roles, agent_count)
        row["tool_constraints"] = build_tool_constraints_map(task, roles, agent_count)
    return apply_task_overrides(row)


def resolve_task_ids(args: argparse.Namespace, tasks_dir: Path) -> list[str]:
    if args.task_ids:
        return [item.strip() for item in args.task_ids.split(",") if item.strip()]
    if args.tier:
        return discover_task_ids(tasks_dir, tier=args.tier, exclude_perturbed=not args.include_perturbed)
    raise SystemExit("Provide --task-ids or --tier")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ClawBench YAML tasks to bench JSONL")
    parser.add_argument(
        "--task-ids",
        default="",
        help="Comma-separated ClawBench task ids (e.g. t2-fs-find-that-thing)",
    )
    parser.add_argument(
        "--tier",
        default="",
        help="Discover all tasks under clawbench/tasks-public/<tier>/ (e.g. tier2)",
    )
    parser.add_argument(
        "--include-perturbed",
        action="store_true",
        help="When using --tier, include *-perturbed task variants",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "datasets" / "tier1_clawbench.jsonl",
    )
    parser.add_argument("--tasks-dir", type=Path, default=None)
    parser.add_argument("--agent-count", type=int, default=3)
    parser.add_argument(
        "--template",
        default="capability",
        choices=["chain-pipeline-v1", "capability", "all"],
    )
    args = parser.parse_args()

    tasks_dir = resolve_tasks_dir(args.tasks_dir)
    task_ids = resolve_task_ids(args, tasks_dir)
    template = args.template if args.template != "chain-pipeline-v1" else "chain-pipeline-v1"

    rows = []
    for task_id in task_ids:
        task_yaml = load_task_yaml(tasks_dir, task_id)
        rows.append(
            task_row_from_yaml(
                task_yaml,
                agent_count=args.agent_count,
                template=template if template != "chain-pipeline-v1" else "chain-pipeline-v1",
            )
        )
        if args.template == "all":
            rows[-1]["capability_agent_tasks"] = build_capability_agent_tasks(
                task_yaml,
                rows[-1]["agent_roles"],
                args.agent_count,
            )
            rows[-1]["tool_constraints"] = build_tool_constraints_map(
                task_yaml,
                rows[-1]["agent_roles"],
                args.agent_count,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} tasks -> {args.output}")
    for row in rows:
        print(f"  - {row['task_id']} ({row['clawbench_ref']['family']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
