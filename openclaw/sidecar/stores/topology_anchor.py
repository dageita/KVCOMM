"""Topology anchoring — delta keys and proactive stale invalidation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sidecar.stores.prefix_spans import normalize_placeholder_info


@dataclass(frozen=True)
class DeltaAnchorKey:
    static_template_hash: str
    topology_id: str
    ph_id: str
    ph_token_start: int
    ph_token_end: int
    pf_span_id: str | None
    content_hash: str = ""

    def as_tuple(self) -> tuple[str, ...]:
        return (
            self.static_template_hash,
            self.topology_id,
            self.ph_id,
            str(self.ph_token_start),
            str(self.ph_token_end),
            str(self.pf_span_id or ""),
            self.content_hash,
        )


def delta_key_from_ph_rec(
    *,
    ph_id: str,
    ph_rec: dict[str, Any],
    static_template_hash: str,
    topology_id: str,
    content_hash: str = "",
) -> DeltaAnchorKey:
    return DeltaAnchorKey(
        static_template_hash=str(static_template_hash),
        topology_id=str(topology_id),
        ph_id=str(ph_id),
        ph_token_start=int(ph_rec.get("start", 0)),
        ph_token_end=int(ph_rec.get("end", 0)),
        pf_span_id=ph_rec.get("pf_span_id"),
        content_hash=str(content_hash or ""),
    )


def current_topology_keys(
    bucket: dict[str, Any],
    *,
    content_hash_by_ph: dict[str, str] | None = None,
) -> dict[str, DeltaAnchorKey]:
    """Build current coordinate keys for all placeholders on a node bucket."""
    static_hash = str(bucket.get("static_template_hash") or "")
    topo = str(bucket.get("topology_id") or "")
    ph_info = normalize_placeholder_info(bucket.get("placeholder_info"))
    content_hash_by_ph = content_hash_by_ph or {}
    out: dict[str, DeltaAnchorKey] = {}
    for ph_id, rec in ph_info.items():
        out[str(ph_id)] = delta_key_from_ph_rec(
            ph_id=str(ph_id),
            ph_rec=rec,
            static_template_hash=static_hash,
            topology_id=topo,
            content_hash=str(content_hash_by_ph.get(str(ph_id), "")),
        )
    return out


def stored_key_matches(stored: dict[str, Any] | None, current: DeltaAnchorKey) -> bool:
    if not isinstance(stored, dict):
        return False
    for field, attr in (
        ("static_template_hash", "static_template_hash"),
        ("topology_id", "topology_id"),
        ("ph_token_start", "ph_token_start"),
        ("ph_token_end", "ph_token_end"),
        ("pf_span_id", "pf_span_id"),
    ):
        if str(stored.get(field, "")) != str(getattr(current, attr, "") or ""):
            return False
    stored_hash = str(stored.get("content_hash") or "")
    if stored_hash and current.content_hash and stored_hash != current.content_hash:
        return False
    return True


def coordinate_shifted(
    stored: dict[str, Any] | None,
    current: DeltaAnchorKey,
    *,
    ignore_content: bool = False,
) -> bool:
    """True when topology/static/start/pf_span changed (geometry stale)."""
    if not isinstance(stored, dict):
        return False
    for field, attr in (
        ("static_template_hash", "static_template_hash"),
        ("topology_id", "topology_id"),
        ("ph_token_start", "ph_token_start"),
        ("ph_token_end", "ph_token_end"),
        ("pf_span_id", "pf_span_id"),
    ):
        if str(stored.get(field, "")) != str(getattr(current, attr, "") or ""):
            return True
    if not ignore_content:
        stored_hash = str(stored.get("content_hash") or "")
        if stored_hash and current.content_hash and stored_hash != current.content_hash:
            return True
    return False


def new_tail_placeholder_ids(
    old_ph_info: dict | None,
    new_ph_info: dict | None,
) -> set[str]:
    old_ids = set(normalize_placeholder_info(old_ph_info).keys())
    new_ids = set(normalize_placeholder_info(new_ph_info).keys())
    return new_ids - old_ids


def serialize_anchor_key(key: DeltaAnchorKey) -> dict[str, Any]:
    return {
        "static_template_hash": key.static_template_hash,
        "topology_id": key.topology_id,
        "ph_id": key.ph_id,
        "ph_token_start": key.ph_token_start,
        "ph_token_end": key.ph_token_end,
        "pf_span_id": key.pf_span_id,
        "content_hash": key.content_hash,
    }
