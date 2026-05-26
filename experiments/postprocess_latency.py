import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_records(latency_path: Path) -> List[Dict[str, Any]]:
    with open(latency_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, list):
        raise ValueError(f"{latency_path} is not a JSON list.")
    return loaded


def _dump_records(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def _sorted_by_timestamp(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: (
            float(r.get("timestamp", 0.0)),
            str(r.get("request_uid", "")),
            str(r.get("agent_id", "")),
        ),
    )


def _sorted_by_request_agent(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: (
            str(r.get("request_uid", "")),
            str(r.get("agent_id", "")),
            float(r.get("timestamp", 0.0)),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process KVCOMM Latency.json ordering.")
    parser.add_argument(
        "--latency-json",
        type=str,
        default=None,
        help="Path to Latency.json. If omitted, uses <output-dir>/Latency.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory that contains Latency.json.",
    )
    args = parser.parse_args()

    if args.latency_json is None and args.output_dir is None:
        raise ValueError("Provide --latency-json or --output-dir.")

    latency_path = (
        Path(args.latency_json)
        if args.latency_json is not None
        else Path(args.output_dir) / "Latency.json"
    )
    if not latency_path.exists():
        raise FileNotFoundError(f"Latency.json not found: {latency_path}")

    records = _load_records(latency_path)
    target_dir = latency_path.parent / "latency_sorted"
    _dump_records(target_dir / "Latency_by_timestamp.json", _sorted_by_timestamp(records))
    _dump_records(
        target_dir / "Latency_by_request_agent.json",
        _sorted_by_request_agent(records),
    )
    print(f"Wrote sorted latency views to: {target_dir}")


if __name__ == "__main__":
    main()
