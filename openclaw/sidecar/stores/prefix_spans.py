"""Explicit ph↔pf span layout for prefix KV (replaces reversed zip pairing)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sidecar.stores.hashing import sha256_text


@dataclass
class TextSpanRecord:
    span_id: str
    text: str
    text_hash: str
    token_start: int
    token_end: int
    kv_index: int = -1


@dataclass
class PlaceholderRecord:
    ph_id: str
    start: int
    end: int
    pf_span_id: str | None = None


@dataclass
class PrefixLayout:
    text_spans: list[TextSpanRecord] = field(default_factory=list)
    placeholders: list[PlaceholderRecord] = field(default_factory=list)
    span_registry: dict[str, dict[str, Any]] = field(default_factory=dict)
    placeholder_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    prefix_span_order: list[str] = field(default_factory=list)
    prompt_token_len: int = 0


def normalize_placeholder_entry(entry: Any) -> dict[str, Any]:
    """Accept legacy [start, end] or enriched {start, end, pf_span_id}."""
    if isinstance(entry, dict):
        start = int(entry.get("start", entry.get("token_start", 0)))
        end = int(entry.get("end", entry.get("token_end", 0)))
        pf = entry.get("pf_span_id")
        return {"start": start, "end": end, "pf_span_id": pf}
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        out: dict[str, Any] = {"start": int(entry[0]), "end": int(entry[1])}
        if len(entry) >= 3 and entry[2]:
            out["pf_span_id"] = str(entry[2])
        return out
    return {"start": 0, "end": 0, "pf_span_id": None}


def normalize_placeholder_info(ph_info: dict | None) -> dict[str, dict[str, Any]]:
    if not isinstance(ph_info, dict):
        return {}
    return {str(ph_id): normalize_placeholder_entry(entry) for ph_id, entry in ph_info.items()}


def placeholder_token_span(ph_info: dict | None, ph_id: str) -> tuple[int, int]:
    rec = normalize_placeholder_info(ph_info).get(str(ph_id), {})
    return int(rec.get("start", 0)), int(rec.get("end", 0))


def build_layout_from_segments(segments: list[tuple]) -> PrefixLayout:
    """Build explicit ph→pf_span_id layout from locate_placeholder segments."""
    ordered = sorted(segments, key=lambda seg: seg[3])
    layout = PrefixLayout()
    pending_ph: list[tuple[str, int, int]] = []
    text_idx = 0

    for seg in ordered:
        kind = seg[0]
        if kind == "text":
            txt = str(seg[1] or "")
            token_start, token_end = int(seg[3]), int(seg[4])
            span_id = "T0" if text_idx == 0 else f"text_{text_idx}"
            record = TextSpanRecord(
                span_id=span_id,
                text=txt,
                text_hash=sha256_text(txt),
                token_start=token_start,
                token_end=token_end,
                kv_index=text_idx,
            )
            layout.text_spans.append(record)
            layout.prefix_span_order.append(span_id)
            layout.span_registry[span_id] = {
                "span_id": span_id,
                "text_hash": record.text_hash,
                "token_start": token_start,
                "token_end": token_end,
                "kv_index": text_idx,
            }
            for ph_id, ph_start, ph_end in pending_ph:
                layout.placeholder_info[ph_id] = {
                    "start": ph_start,
                    "end": ph_end,
                    "pf_span_id": span_id,
                }
                for ph in layout.placeholders:
                    if ph.ph_id == ph_id:
                        ph.pf_span_id = span_id
            pending_ph = []
            text_idx += 1
        elif kind == "placeholder":
            ph_id = str(seg[1])
            token_start, token_end = int(seg[3]), int(seg[4])
            ph = PlaceholderRecord(ph_id=ph_id, start=token_start, end=token_end)
            layout.placeholders.append(ph)
            layout.placeholder_info[ph_id] = {
                "start": token_start,
                "end": token_end,
                "pf_span_id": None,
            }
            pending_ph.append((ph_id, token_start, token_end))

    if ordered:
        layout.prompt_token_len = max(int(seg[4]) for seg in ordered)
    return layout


def ordered_placeholders(ph_info: dict | None) -> list[tuple[str, dict[str, Any]]]:
    normalized = normalize_placeholder_info(ph_info)
    return sorted(normalized.items(), key=lambda item: item[1]["start"])


def span_registry_entry(registry: dict | None, span_id: str | None) -> dict[str, Any] | None:
    if not span_id or not isinstance(registry, dict):
        return None
    return registry.get(str(span_id))


def resolve_pf_kv_index(
    *,
    ph_id: str,
    placeholder_info: dict | None,
    span_registry: dict | None,
) -> int | None:
    rec = normalize_placeholder_info(placeholder_info).get(str(ph_id), {})
    pf_span_id = rec.get("pf_span_id")
    entry = span_registry_entry(span_registry, pf_span_id)
    if entry is None:
        return None
    try:
        return int(entry.get("kv_index", -1))
    except (TypeError, ValueError):
        return None


def frozen_span_count(
    old_registry: dict[str, dict[str, Any]],
    old_span_order: list[str],
    new_layout: PrefixLayout,
) -> int:
    """Number of leading text spans with identical hash and token span (left freeze)."""
    count = 0
    for span_id in old_span_order:
        old = old_registry.get(span_id)
        new = new_layout.span_registry.get(span_id)
        if not old or not new:
            break
        if (
            old.get("text_hash") != new.get("text_hash")
            or int(old.get("token_start", -1)) != int(new.get("token_start", -1))
            or int(old.get("token_end", -1)) != int(new.get("token_end", -1))
        ):
            break
        count += 1
    return count


def shared_prefix_token_len(old_ids: Any, new_ids: Any) -> int:
    """Longest token prefix shared by two 1-D (or batched) id tensors/lists."""
    if old_ids is None or new_ids is None:
        return 0
    if hasattr(old_ids, "reshape"):
        old_seq = old_ids.reshape(-1).tolist()
    else:
        old_seq = list(old_ids)
    if hasattr(new_ids, "reshape"):
        new_seq = new_ids.reshape(-1).tolist()
    else:
        new_seq = list(new_ids)
    limit = min(len(old_seq), len(new_seq))
    for idx in range(limit):
        if old_seq[idx] != new_seq[idx]:
            return idx
    return limit


def legacy_forward_pf_kv_index(placeholder_info: dict | None, ph_id: str) -> int | None:
    """Legacy fallback: pf index = 1 + forward placeholder order (post-T0)."""
    ordered = ordered_placeholders(placeholder_info)
    for idx, (pid, _rec) in enumerate(ordered):
        if pid == str(ph_id):
            return idx + 1
    return None


def legacy_placeholder_bounds(ph_info: dict) -> dict[str, list[int]]:
    """Downstream helpers expecting ph_id -> [start, end] only."""
    out: dict[str, list[int]] = {}
    for ph_id, entry in normalize_placeholder_info(ph_info).items():
        out[ph_id] = [int(entry["start"]), int(entry["end"])]
    return out
