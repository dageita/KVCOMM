#!/usr/bin/env python3
"""Prepare a ClawBench asset workspace for kvcomm chain capability runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path


def resolve_tasks_public_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates = [
        Path(__file__).resolve().parents[4] / "clawbench" / "tasks-public",
        Path.cwd() / "tasks-public",
        Path.cwd().parent / "clawbench" / "tasks-public",
    ]
    for candidate in candidates:
        if (candidate / "assets").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate clawbench/tasks-public")


def copy_into_workspace(source: Path, workspace: Path) -> None:
    if source.is_file():
        target = workspace / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = workspace / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def setup_workspace(*, task_id: str, asset_packs: list[str], run_id: str | None, tasks_dir: Path) -> Path:
    assets_dir = tasks_dir / "assets"
    state_dir = Path(os.environ.get("OPENCLAW_STATE_DIR", os.path.expanduser("~/.openclaw")))
    run_suffix = run_id or uuid.uuid4().hex[:8]
    workspace = state_dir / "workspace" / "kvcomm-chain" / task_id / f"run-{run_suffix}"
    workspace.mkdir(parents=True, exist_ok=True)

    for pack in asset_packs:
        source = assets_dir / pack
        if not source.exists():
            raise FileNotFoundError(f"Missing asset pack: {source}")
        copy_into_workspace(source, workspace)

    return workspace.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup ClawBench workspace for kvcomm chain runs")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--asset-packs", required=True, help="Comma-separated asset pack names")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--tasks-dir", type=Path, default=None)
    parser.add_argument("--output-json", action="store_true", help="Print JSON with workspace path")
    args = parser.parse_args()

    asset_packs = [item.strip() for item in args.asset_packs.split(",") if item.strip()]
    tasks_dir = resolve_tasks_public_dir(args.tasks_dir)
    workspace = setup_workspace(
        task_id=args.task_id,
        asset_packs=asset_packs,
        run_id=args.run_id or None,
        tasks_dir=tasks_dir,
    )

    if args.output_json:
        print(json.dumps({"workspace": str(workspace), "task_id": args.task_id}))
    else:
        print(workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
