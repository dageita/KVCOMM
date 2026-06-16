#!/usr/bin/env python3
"""Convert ClawBench YAML tasks to KVCOMM bench JSONL (Chain pipeline templates)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve_tasks_dir() -> Path:
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


def first_task_prompt(task: dict) -> str:
    user = task.get("user") or {}
    turns = user.get("turns") or []
    if turns and turns[0].get("message"):
        return str(turns[0]["message"])
    phases = task.get("phases") or []
    if phases:
        phase_user = phases[0].get("user") or {}
        phase_turns = phase_user.get("turns") or []
        if phase_turns and phase_turns[0].get("message"):
            return str(phase_turns[0]["message"])
    return str(task.get("name") or task.get("id") or "")


ROLE_PROMPT_SLOT = "{{role_prompt}}"

PIPELINE_ROLES = ["Extractor", "Formatter", "Reviewer"]

TEXT_AGENT_0 = (
    f"{ROLE_PROMPT_SLOT}\n\n"
    "User request:\n{{task_body}}\n\n"
    "Your job (Agent 0 - Extractor): List all items or key facts the user needs. "
    "Output plain text only. Do not use tools.\n"
)

TEXT_AGENT_N = (
    f"{ROLE_PROMPT_SLOT}\n\n"
    "User request:\n{{task_body}}\n\n"
    "Output from Agent {prev_index} ({prev_role}):\n\n{{agent_{prev_index}_current}}\n\n"
    "Your job (Agent {index} - {role}): Build on the previous agent's output. "
    "Output plain text only. Do not use tools.\n"
)

CAPABILITY_AGENT_0 = (
    f"{ROLE_PROMPT_SLOT}\n\n"
    "User request:\n{{task_body}}\n\n"
    "Your job (Agent 0 - Extractor): Read the user request and list all items or key facts. "
    "You may use read tools on the workspace if helpful. Output a clear plain-text summary.\n"
)

CAPABILITY_AGENT_1 = (
    f"{ROLE_PROMPT_SLOT}\n\n"
    "User request:\n{{task_body}}\n\n"
    "Output from Agent 0 (Extractor):\n\n{{agent_0_current}}\n\n"
    "Your job (Agent 1 - Executor): Use the workspace tools (read, edit, write, exec) to "
    "produce the deliverable described in the user request. Build on Agent 0's analysis.\n"
)

CAPABILITY_AGENT_2 = (
    f"{ROLE_PROMPT_SLOT}\n\n"
    "User request:\n{{task_body}}\n\n"
    "Output from Agent 1 (Executor):\n\n{{agent_1_current}}\n\n"
    "Your job (Agent 2 - Verifier): Inspect the workspace, run any needed checks, "
    "and output a final summary of what was done and whether the task is complete.\n"
)


def build_text_agent_tasks(roles: list[str]) -> dict[str, str]:
    tasks: dict[str, str] = {"agent_0": TEXT_AGENT_0}
    for index in range(1, len(roles)):
        prev_index = index - 1
        tasks[f"agent_{index}"] = TEXT_AGENT_N.format(
            prev_index=prev_index,
            prev_role=roles[prev_index],
            index=index,
            role=roles[index],
        )
    return tasks


def build_capability_agent_tasks(agent_count: int) -> dict[str, str]:
    if agent_count < 3:
        raise ValueError("capability templates require at least 3 agents")
    tasks = {"agent_0": CAPABILITY_AGENT_0, "agent_1": CAPABILITY_AGENT_1}
    for index in range(2, agent_count):
        prev_index = index - 1
        tasks[f"agent_{index}"] = (
            f"{ROLE_PROMPT_SLOT}\n\n"
            "User request:\n{{task_body}}\n\n"
            f"Output from Agent {prev_index}:\n\n{{agent_{prev_index}_current}}\n\n"
            f"Your job (Agent {index} - Verifier): Continue verification and produce the final summary.\n"
        )
    return tasks


def task_row_from_yaml(task: dict, *, agent_count: int, template: str) -> dict:
    task_id = str(task["id"])
    task_body = first_task_prompt(task)
    roles = PIPELINE_ROLES[:agent_count]
    while len(roles) < agent_count:
        roles.append(f"Worker{len(roles)}")

    setup = task.get("setup") or {}
    row = {
        "task_id": task_id,
        "task_body": task_body,
        "clawbench_ref": {
            "yaml_id": task_id,
            "asset_packs": list(setup.get("asset_packs") or []),
            "tier": str(task.get("tier") or ""),
            "family": str(task.get("family") or ""),
        },
        "agent_roles": roles,
        "agent_tasks": build_text_agent_tasks(roles),
    }
    if template in {"capability", "all"}:
        row["capability_agent_tasks"] = build_capability_agent_tasks(agent_count)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ClawBench YAML tasks to bench JSONL")
    parser.add_argument(
        "--task-ids",
        default="t1-fs-quick-note,t1-bugfix-discount",
        help="Comma-separated ClawBench task ids",
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
        default="chain-pipeline-v1",
        choices=["chain-pipeline-v1", "capability", "all"],
    )
    args = parser.parse_args()

    tasks_dir = args.tasks_dir or resolve_tasks_dir()
    task_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]

    rows = []
    for task_id in task_ids:
        task_yaml = load_task_yaml(tasks_dir, task_id)
        rows.append(
            task_row_from_yaml(
                task_yaml,
                agent_count=args.agent_count,
                template=args.template if args.template != "chain-pipeline-v1" else "chain-pipeline-v1",
            )
        )
        if args.template == "all":
            rows[-1]["capability_agent_tasks"] = build_capability_agent_tasks(args.agent_count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} tasks -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
