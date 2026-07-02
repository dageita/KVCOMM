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
        if any(part == "__pycache__" for part in relative.parts):
            continue
        if item.suffix == ".pyc":
            continue
        target = workspace / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _bench_pristine_root(state_dir: Path, asset_pack: str) -> Path:
    return state_dir / "bench-pristine" / asset_pack


def _snapshot_pristine_fixtures(workspace: Path, *, state_dir: Path, asset_packs: list[str]) -> None:
    """Cache read-only fixtures outside the agent workspace (agents must not mutate this)."""
    for pack in asset_packs:
        pristine = _bench_pristine_root(state_dir, pack)
        for name in ("cart.py", "normalizer.py"):
            src = workspace / name
            if src.is_file():
                dst = pristine / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        tests = workspace / "tests"
        if tests.is_dir():
            dst_tests = pristine / "tests"
            if dst_tests.exists():
                shutil.rmtree(dst_tests)
            shutil.copytree(tests, dst_tests, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


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

    _snapshot_pristine_fixtures(workspace, state_dir=state_dir, asset_packs=asset_packs)
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
