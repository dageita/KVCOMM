#!/usr/bin/env python3
"""Score a kvcomm clawbench capability chain run (workspace + transcript)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def add_clawbench_to_path() -> None:
    root = Path(__file__).resolve().parents[4] / "clawbench"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


async def score_chain_run(
    *,
    task_id: str,
    workspace: Path,
    transcript_path: Path,
    judge_model: str,
) -> dict:
    add_clawbench_to_path()
    from clawbench.client import GatewayClient, GatewayConfig
    from clawbench.schemas import Transcript, TranscriptMessage
    from clawbench.scorer import score_task_run
    from clawbench.tasks import load_all_tasks

    tasks = {task.id: task for task in load_all_tasks()}
    if task_id not in tasks:
        raise KeyError(f"Unknown ClawBench task id: {task_id}")
    task = tasks[task_id]

    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    messages = []
    for item in payload.get("messages", []):
        messages.append(TranscriptMessage(role=item["role"], text=item.get("text", "")))
    assistant_text = payload.get("assistant_text") or ""
    if not assistant_text and messages:
        for message in reversed(messages):
            if message.role == "assistant" and message.text:
                assistant_text = message.text
                break
    transcript = Transcript(messages=messages, assistant_text=assistant_text, tool_calls=payload.get("tool_calls", []))

    client = GatewayClient(
        GatewayConfig(token=os.environ.get("OPENCLAW_GATEWAY_TOKEN", "").strip())
    )
    await client.connect()
    try:
        result = await score_task_run(
            task=task,
            transcript=transcript,
            workspace=workspace,
            client=client,
            session_key="kvcomm-chain-scoring",
            agent_id=None,
            duration_ms=int(payload.get("duration_ms") or 0),
            runtime_values={},
            judge_model=judge_model,
            judge_affects_score=False,
        )
    finally:
        await client.close()

    return {
        "task_id": task_id,
        "workspace": str(workspace),
        "run_score": result.run_score,
        "completion_score": result.completion_result.score,
        "trajectory_score": result.trajectory_result.score,
        "behavior_score": result.behavior_result.score,
        "judge_score": result.judge_result.score if result.judge_result.enabled else None,
        "judge_error": result.judge_result.error,
        "failed_assertions": result.completion_result.failed_assertions[:5],
        "trajectory_violations": result.trajectory_result.forbidden_violations[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score kvcomm clawbench capability chain run")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", default="")
    args = parser.parse_args()

    result = asyncio.run(
        score_chain_run(
            task_id=args.task_id,
            workspace=args.workspace.resolve(),
            transcript_path=args.transcript,
            judge_model=args.judge_model,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
