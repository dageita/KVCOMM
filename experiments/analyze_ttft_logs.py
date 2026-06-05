import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


TTFT_LINE_PATTERN = re.compile(r"\[TTFT:([^\]]+)\]\s+(\{.*\})")

DEFAULT_LOG_METRICS = ["ttft", "others_latency"]
KV_REUSE_LOG_METRICS = ["preprocess_latency", "generation_ttft", "others_latency"]
KV_REUSE_LATENCY_JSON_METRICS = ["ttft_ratio_dense_over_kvcomm"]

METRIC_FALLBACKS: Dict[str, List[str]] = {
    "others_latency": ["others_e2e"],
    "preprocess_latency": ["kvcomm_latency"],
    "generation_ttft": ["first_token_decode"],
}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _parse_agent_ids(agent_ids_text: Optional[str]) -> Optional[List[str]]:
    if agent_ids_text is None:
        return None
    parts = [p.strip() for p in agent_ids_text.split(",")]
    ids = [p for p in parts if p]
    if not ids:
        raise ValueError("No valid agent ids provided.")
    return ids


def _iter_ttft_records(
    log_path: Path,
    *,
    mode_tag: Optional[str] = None,
) -> Iterable[Dict[str, Any]]:
    """Yield TTFT JSON payloads; optionally filter by exact log tag (e.g. default, kv_reuse)."""
    with open(log_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            match = TTFT_LINE_PATTERN.search(line)
            if not match:
                continue
            mode_from_tag = match.group(1).strip()
            if mode_tag is not None and mode_from_tag != mode_tag:
                continue
            payload_text = match.group(2).strip()
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            payload.setdefault("_line_no", line_no)
            payload.setdefault("_mode_tag", mode_from_tag)
            yield payload


def _collect_agent_ids(records: Iterable[Dict[str, Any]]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for rec in records:
        aid = str(rec.get("agent_id", "")).strip()
        if not aid or aid in seen:
            continue
        seen.add(aid)
        ordered.append(aid)
    return sorted(ordered, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))


def _resolve_metric_value(
    rec: Dict[str, Any],
    metric: str,
    fallback_fields: Dict[str, List[str]],
) -> Optional[float]:
    value = _safe_float(rec.get(metric))
    if value is not None:
        return value
    for fallback_key in fallback_fields.get(metric, []):
        value = _safe_float(rec.get(fallback_key))
        if value is not None:
            return value
    return None


def _aggregate_by_agent(
    records: Iterable[Dict[str, Any]],
    agent_ids: List[str],
    metric_fields: List[str],
    fallback_fields: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    fallback_fields = fallback_fields or {}
    metric_sums: Dict[str, Dict[str, float]] = {aid: {m: 0.0 for m in metric_fields} for aid in agent_ids}
    metric_counts: Dict[str, Dict[str, int]] = {aid: {m: 0 for m in metric_fields} for aid in agent_ids}
    sample_counts: Dict[str, int] = {aid: 0 for aid in agent_ids}

    for rec in records:
        aid = str(rec.get("agent_id", "")).strip()
        if aid not in sample_counts:
            continue
        sample_counts[aid] += 1
        for metric in metric_fields:
            value = _resolve_metric_value(rec, metric, fallback_fields)
            if value is None:
                continue
            metric_sums[aid][metric] += value
            metric_counts[aid][metric] += 1

    result: Dict[str, Dict[str, Any]] = {}
    for aid in agent_ids:
        row: Dict[str, Any] = {"samples": sample_counts[aid]}
        for metric in metric_fields:
            cnt = metric_counts[aid][metric]
            row[f"{metric}_avg"] = (metric_sums[aid][metric] / cnt) if cnt > 0 else None
            row[f"{metric}_count"] = cnt
        result[aid] = row
    return result


def _print_section(title: str, rows: Dict[str, Dict[str, Any]], metrics: List[str]) -> None:
    print(f"\n=== {title} ===")
    headers = ["agent_id", "samples"] + [f"{m}_avg" for m in metrics]
    print("\t".join(headers))
    for aid, row in rows.items():
        values = [aid, str(row["samples"])]
        for metric in metrics:
            value = row.get(f"{metric}_avg")
            values.append("N/A" if value is None else f"{value:.6f}")
        print("\t".join(values))


def _resolve_log_paths(args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    if args.log:
        log_path = Path(args.log)
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        return log_path, None

    baseline_path = Path(args.baseline_log)
    kvreuse_path = Path(args.kvreuse_log)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline log not found: {baseline_path}")
    if not kvreuse_path.exists():
        raise FileNotFoundError(f"KV reuse log not found: {kvreuse_path}")
    return baseline_path, kvreuse_path


def _resolve_latency_json_path(
    kvreuse_log: Path,
    explicit_path: Optional[str],
) -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None
    if kvreuse_log.parent.name == "logs":
        candidate = kvreuse_log.parent.parent / "Latency.json"
    else:
        candidate = kvreuse_log.parent / "Latency.json"
    return candidate if candidate.exists() else None


def _load_latency_json_records(
    latency_path: Path,
    *,
    mode: str = "kv_reuse",
) -> List[Dict[str, Any]]:
    with open(latency_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, list):
        return []
    return [
        rec
        for rec in loaded
        if isinstance(rec, dict) and str(rec.get("mode", "")).strip() == mode
    ]


def _analyze_records(
    records: List[Dict[str, Any]],
    agent_ids: Optional[List[str]],
    metrics: List[str],
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    if not records:
        return agent_ids or [], {}
    resolved_agent_ids = agent_ids or _collect_agent_ids(records)
    if not resolved_agent_ids:
        return [], {}
    rows = _aggregate_by_agent(
        records,
        agent_ids=resolved_agent_ids,
        metric_fields=metrics,
        fallback_fields=METRIC_FALLBACKS,
    )
    return resolved_agent_ids, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-agent TTFT metrics from benchmark logs. "
            "default: avg ttft, others_latency from [TTFT:default]. "
            "kv_reuse: avg preprocess_latency, generation_ttft, others_latency from [TTFT:kv_reuse], "
            "plus ttft_ratio_dense_over_kvcomm from Latency.json."
        )
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Single log file (e.g. runs/.../logs/log.txt).",
    )
    parser.add_argument(
        "--baseline-log",
        type=str,
        default="bench_baseline.log",
        help="Path to default-mode log containing [TTFT:default] lines.",
    )
    parser.add_argument(
        "--kvreuse-log",
        type=str,
        default="bench_kvreuse.log",
        help="Path to kv_reuse log containing [TTFT:kv_reuse] lines.",
    )
    parser.add_argument(
        "--latency-json",
        type=str,
        default=None,
        help="Path to Latency.json for ttft_ratio_dense_over_kvcomm (auto-detected from kvreuse log dir).",
    )
    parser.add_argument(
        "--agent-ids",
        type=str,
        default=None,
        help='Comma separated agent ids, e.g. "0,1,2". Auto-detect when omitted.',
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output path to save aggregated JSON.",
    )
    args = parser.parse_args()

    agent_ids = _parse_agent_ids(args.agent_ids)
    primary_log, secondary_log = _resolve_log_paths(args)

    if secondary_log is None:
        default_log = primary_log
        kvreuse_log = primary_log
    else:
        default_log = primary_log
        kvreuse_log = secondary_log

    default_records = list(_iter_ttft_records(default_log, mode_tag="default"))
    kvreuse_log_records = list(_iter_ttft_records(kvreuse_log, mode_tag="kv_reuse"))

    default_agent_ids, default_rows = _analyze_records(
        default_records,
        agent_ids,
        DEFAULT_LOG_METRICS,
    )

    shared_agent_ids = default_agent_ids or agent_ids
    kvreuse_agent_ids, kvreuse_log_rows = _analyze_records(
        kvreuse_log_records,
        shared_agent_ids,
        KV_REUSE_LOG_METRICS,
    )
    if not default_agent_ids and kvreuse_agent_ids:
        default_agent_ids = kvreuse_agent_ids

    latency_path = _resolve_latency_json_path(kvreuse_log, args.latency_json)
    latency_records: List[Dict[str, Any]] = []
    kvreuse_ratio_rows: Dict[str, Dict[str, Any]] = {}
    ratio_agent_ids: List[str] = []

    if latency_path is not None:
        latency_records = _load_latency_json_records(latency_path, mode="kv_reuse")
        ratio_agent_ids, kvreuse_ratio_rows = _analyze_records(
            latency_records,
            kvreuse_agent_ids or shared_agent_ids,
            KV_REUSE_LATENCY_JSON_METRICS,
        )

    if default_records:
        _print_section("default ([TTFT:default])", default_rows, DEFAULT_LOG_METRICS)
    else:
        print("\n=== default ([TTFT:default]) ===")
        print("No [TTFT:default] records found.")

    if kvreuse_log_records:
        _print_section(
            "kv_reuse log ([TTFT:kv_reuse])",
            kvreuse_log_rows,
            KV_REUSE_LOG_METRICS,
        )
    else:
        print("\n=== kv_reuse log ([TTFT:kv_reuse]) ===")
        print("No [TTFT:kv_reuse] records found.")

    if latency_path is None:
        print("\n=== kv_reuse Latency.json (ttft_ratio_dense_over_kvcomm) ===")
        print("Latency.json not found (use --latency-json to specify path).")
    elif latency_records:
        _print_section(
            f"kv_reuse Latency.json ({latency_path.name})",
            kvreuse_ratio_rows,
            KV_REUSE_LATENCY_JSON_METRICS,
        )
    else:
        print("\n=== kv_reuse Latency.json (ttft_ratio_dense_over_kvcomm) ===")
        print(f"No mode=kv_reuse records in: {latency_path}")

    output_obj: Dict[str, Any] = {
        "default": {
            "source": "log",
            "mode_tag": "default",
            "metrics": DEFAULT_LOG_METRICS,
            "agent_ids": default_agent_ids,
            "rows": default_rows,
            "records_total": len(default_records),
        },
        "kv_reuse": {
            "log": {
                "source": "log",
                "mode_tag": "kv_reuse",
                "metrics": KV_REUSE_LOG_METRICS,
                "agent_ids": kvreuse_agent_ids,
                "rows": kvreuse_log_rows,
                "records_total": len(kvreuse_log_records),
            },
            "latency_json": {
                "source": "Latency.json",
                "path": str(latency_path) if latency_path else None,
                "metrics": KV_REUSE_LATENCY_JSON_METRICS,
                "agent_ids": ratio_agent_ids,
                "rows": kvreuse_ratio_rows,
                "records_total": len(latency_records),
            },
        },
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(output_obj, handle, ensure_ascii=False, indent=2)
        print(f"\nWrote aggregated results to: {output_path}")


if __name__ == "__main__":
    main()
