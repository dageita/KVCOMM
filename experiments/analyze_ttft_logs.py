import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TTFT_LINE_PATTERN = re.compile(r"\[TTFT:(default|kv_reuse)\]\s+(\{.*\})")


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


def _parse_agent_ids(agent_ids_text: str) -> List[str]:
    parts = [p.strip() for p in agent_ids_text.split(",")]
    ids = [p for p in parts if p]
    if not ids:
        raise ValueError("No valid agent ids provided.")
    return ids


def _iter_ttft_records(log_path: Path, expected_mode: str) -> Iterable[Dict[str, Any]]:
    with open(log_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            match = TTFT_LINE_PATTERN.search(line)
            if not match:
                continue
            mode_from_tag = match.group(1)
            if mode_from_tag != expected_mode:
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


def _aggregate_by_agent(
    records: Iterable[Dict[str, Any]],
    agent_ids: List[str],
    metric_fields: List[str],
    fallback_fields: Optional[Dict[str, str]] = None,
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
            value = _safe_float(rec.get(metric))
            if value is None and metric in fallback_fields:
                value = _safe_float(rec.get(fallback_fields[metric]))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate TTFT-related averages by agent_id from benchmark logs.")
    parser.add_argument(
        "--baseline-log",
        type=str,
        default="bench_baseline.log",
        help="Path to baseline log file containing [TTFT:default] lines.",
    )
    parser.add_argument(
        "--kvreuse-log",
        type=str,
        default="bench_kvreuse.log",
        help="Path to kv_reuse log file containing [TTFT:kv_reuse] lines.",
    )
    parser.add_argument(
        "--agent-ids",
        type=str,
        default="0,1,2,3,4",
        help='Comma separated agent ids, e.g. "0,1,2,3,4".',
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output path to save aggregated JSON.",
    )
    args = parser.parse_args()

    agent_ids = _parse_agent_ids(args.agent_ids)
    baseline_path = Path(args.baseline_log)
    kvreuse_path = Path(args.kvreuse_log)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline log not found: {baseline_path}")
    if not kvreuse_path.exists():
        raise FileNotFoundError(f"KV reuse log not found: {kvreuse_path}")

    baseline_records = list(_iter_ttft_records(baseline_path, expected_mode="default"))
    kvreuse_records = list(_iter_ttft_records(kvreuse_path, expected_mode="kv_reuse"))

    baseline_metrics = ["first_token_decode", "others_e2e"]
    kvreuse_metrics = ["kvcomm_latency", "first_token_decode", "others_e2e"]

    baseline_rows = _aggregate_by_agent(
        baseline_records,
        agent_ids=agent_ids,
        metric_fields=baseline_metrics,
        fallback_fields={"others_e2e": "others_latency"},
    )
    kvreuse_rows = _aggregate_by_agent(
        kvreuse_records,
        agent_ids=agent_ids,
        metric_fields=kvreuse_metrics,
        fallback_fields={"others_e2e": "others_latency"},
    )

    _print_section("baseline(default)", baseline_rows, baseline_metrics)
    _print_section("kv_reuse", kvreuse_rows, kvreuse_metrics)

    output_obj = {
        "baseline": {
            "mode": "default",
            "metrics": baseline_metrics,
            "rows": baseline_rows,
            "records_total": len(baseline_records),
        },
        "kv_reuse": {
            "mode": "kv_reuse",
            "metrics": kvreuse_metrics,
            "rows": kvreuse_rows,
            "records_total": len(kvreuse_records),
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
