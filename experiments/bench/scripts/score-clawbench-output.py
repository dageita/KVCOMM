#!/usr/bin/env python3
"""Score final chain agent text output with ClawBench judge (Phase 1 helper)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def add_clawbench_to_path() -> Path:
    root = Path(__file__).resolve().parents[4] / "clawbench"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_final_output(rows: list[dict], task_id: str | None) -> tuple[str, str]:
    agent_rows = [
        row
        for row in rows
        if row.get("type") != "run_summary"
        and row.get("output_text")
        and (task_id is None or row.get("task_id") == task_id)
    ]
    if not agent_rows:
        raise ValueError("No agent rows with output_text found in jsonl")
    agent_rows.sort(key=lambda row: (row.get("run_id", ""), row.get("agent_index", 0)))
    final = agent_rows[-1]
    return str(final.get("task_id") or task_id or ""), str(final.get("output_text") or "")


async def score_output(task_id: str, output_text: str, judge_model: str) -> dict:
    add_clawbench_to_path()
    from clawbench.client import GatewayClient, GatewayConfig
    from clawbench.schemas import CompletionResult, Transcript, TranscriptMessage
    from clawbench.tasks import load_all_tasks
    from clawbench.judge import judge_task_run

    tasks = {task.id: task for task in load_all_tasks()}
    if task_id not in tasks:
        raise KeyError(f"Unknown ClawBench task id: {task_id}")
    task = tasks[task_id]

    transcript = Transcript(
        messages=[
            TranscriptMessage(role="user", text=task.user.turns[0].message if task.user and task.user.turns else ""),
            TranscriptMessage(role="assistant", text=output_text),
        ],
        assistant_text=output_text,
    )

    client = GatewayClient(GatewayConfig())
    await client.connect()
    try:
        judge_result = await judge_task_run(
            task=task,
            transcript=transcript,
            workspace=Path("/tmp"),
            client=client,
            judge_model=judge_model,
            completion_result=CompletionResult(),
        )
    finally:
        await client.close()

    return {
        "task_id": task_id,
        "judge_enabled": judge_result.enabled,
        "judge_score": judge_result.score,
        "judge_passing_threshold": task.judge.passing_threshold if task.judge else None,
        "judge_reason": judge_result.reason,
        "judge_error": judge_result.error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score kvcomm bench output with ClawBench judge")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--judge-model", default="", help="OpenClaw model ref for judge (optional)")
    parser.add_argument("--text", default="", help="Score explicit assistant text instead of jsonl")
    args = parser.parse_args()

    if args.text:
        task_id = args.task_id
        output_text = args.text
    else:
        rows = load_jsonl(args.jsonl)
        task_id, output_text = pick_final_output(rows, args.task_id or None)

    result = {
        "task_id": task_id,
        "output_chars": len(output_text),
        "judge_enabled": False,
        "judge_score": None,
        "judge_reason": "Judge skipped (no --judge-model)",
    }

    if args.judge_model:
        result = asyncio.run(score_output(task_id, output_text, args.judge_model))

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
