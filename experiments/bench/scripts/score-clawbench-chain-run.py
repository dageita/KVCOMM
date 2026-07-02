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


def _build_transcript(payload: dict) -> "Transcript":
    """Build a ClawBench Transcript; attach chain tool_calls for trajectory scoring."""
    add_clawbench_to_path()
    from clawbench.schemas import ToolCall, Transcript, TranscriptMessage

    messages: list[TranscriptMessage] = []
    for item in payload.get("messages", []):
        tool_calls = [_parse_tool_call(raw) for raw in item.get("tool_calls", [])]
        messages.append(
            TranscriptMessage(
                role=item["role"],
                text=item.get("text", ""),
                tool_calls=[call for call in tool_calls if call is not None],
            )
        )

    chain_tool_calls = [
        call
        for raw in payload.get("tool_calls", [])
        if (call := _parse_tool_call(raw)) is not None
    ]
    if chain_tool_calls:
        messages.append(TranscriptMessage(role="assistant", text="", tool_calls=chain_tool_calls))

    return Transcript(messages=messages)


def _parse_tool_call(raw: object) -> "ToolCall | None":
    add_clawbench_to_path()
    from clawbench.schemas import ToolCall

    if not isinstance(raw, dict) or not str(raw.get("name") or "").strip():
        return None
    return ToolCall(
        name=str(raw["name"]),
        input=dict(raw.get("input") or {}),
        output=str(raw.get("output") or ""),
        success=raw.get("success"),
    )


async def score_chain_run(
    *,
    task_id: str,
    workspace: Path,
    transcript_path: Path,
    judge_model: str,
    runtime_values: dict | None = None,
) -> dict:
    add_clawbench_to_path()
    from clawbench.client import GatewayClient, GatewayConfig
    from clawbench.scorer import score_task_run
    from clawbench.services import build_runtime_values
    from clawbench.tasks import load_all_tasks

    tasks = {task.id: task for task in load_all_tasks()}
    if task_id not in tasks:
        raise KeyError(f"Unknown ClawBench task id: {task_id}")
    task = tasks[task_id]

    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript = _build_transcript(payload)

    clawbench_root = Path(__file__).resolve().parents[4] / "clawbench"
    merged_runtime_values = build_runtime_values(
        workspace=workspace,
        repo_root=clawbench_root,
        extra=runtime_values or {},
    )

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
            runtime_values=merged_runtime_values,
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
    parser.add_argument("--runtime-values", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", default="")
    args = parser.parse_args()

    runtime_values = {}
    if args.runtime_values and args.runtime_values.is_file():
        runtime_values = json.loads(args.runtime_values.read_text(encoding="utf-8"))

    result = asyncio.run(
        score_chain_run(
            task_id=args.task_id,
            workspace=args.workspace.resolve(),
            transcript_path=args.transcript,
            judge_model=args.judge_model,
            runtime_values=runtime_values,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
