"""SegmentCache — frozen contextual text-segment absolute KV per node."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sidecar.stores.hashing import short_hash


@dataclass
class SegmentEntry:
    span_id: str
    template_hash: str
    absolute_kv: Any
    token_ids: dict[str, Any]
    token_len: int
    materialization: str = "contextual_template"


@dataclass
class NodeSegmentCache:
    node_id: str
    segments: dict[str, SegmentEntry] = field(default_factory=dict)
    prefix_kv_list: list[Any] = field(default_factory=list)
    prefix_token_ids: list[dict[str, Any]] = field(default_factory=list)
    placeholder_info: dict[str, Any] = field(default_factory=dict)
    static_template_hash: str = ""
    topology_id: str = ""
    span_registry: dict[str, dict[str, Any]] = field(default_factory=dict)
    prefix_span_order: list[str] = field(default_factory=list)
    turn_count: int = 0

    def segment_key(self, span_id: str, template_hash: str) -> str:
        return f"seg:{self.node_id}:{span_id}:{short_hash(template_hash)}"

    def put_prefix_blob(
        self,
        *,
        prefix_kv_list: list[Any],
        prefix_token_ids: list[dict[str, Any]],
        placeholder_info: dict[str, Any],
        static_template_hash: str,
        topology_id: str,
        turn_count: int,
        span_registry: dict[str, dict[str, Any]] | None = None,
        prefix_span_order: list[str] | None = None,
    ) -> None:
        self.prefix_kv_list = list(prefix_kv_list)
        self.prefix_token_ids = list(prefix_token_ids)
        self.placeholder_info = dict(placeholder_info)
        self.static_template_hash = static_template_hash
        self.topology_id = topology_id
        self.turn_count = int(turn_count)
        self.span_registry = dict(span_registry or {})
        self.prefix_span_order = list(prefix_span_order or [])

        self.segments.clear()
        registry = self.span_registry
        for idx, seg_kv in enumerate(prefix_kv_list):
            token_ids = prefix_token_ids[idx] if idx < len(prefix_token_ids) else {}
            token_len = 0
            if isinstance(token_ids, dict) and token_ids.get("input_ids") is not None:
                token_len = int(token_ids["input_ids"].shape[-1])
            reg_entry = None
            for entry in registry.values():
                if int(entry.get("kv_index", -1)) == idx:
                    reg_entry = entry
                    break
            span_id = (
                str(reg_entry["span_id"])
                if reg_entry and reg_entry.get("span_id")
                else ("T0" if idx == 0 else f"text_{idx}")
            )
            text_hash = str(reg_entry.get("text_hash", "")) if reg_entry else ""
            template_hash = text_hash or f"{static_template_hash}:{topology_id}:{idx}"
            key = self.segment_key(span_id, template_hash)
            self.segments[key] = SegmentEntry(
                span_id=span_id,
                template_hash=template_hash,
                absolute_kv=seg_kv,
                token_ids=token_ids,
                token_len=token_len,
            )

    def get_by_span_id(self, span_id: str) -> SegmentEntry | None:
        target = str(span_id)
        for entry in self.segments.values():
            if entry.span_id == target:
                return entry
        return None

    def resolve_pf(self, ph_id: str) -> tuple[Any, dict[str, Any]] | None:
        from sidecar.stores.prefix_spans import normalize_placeholder_info, span_registry_entry

        rec = normalize_placeholder_info(self.placeholder_info).get(str(ph_id))
        if not rec:
            return None
        pf_span_id = rec.get("pf_span_id")
        reg = span_registry_entry(self.span_registry, pf_span_id)
        if reg is not None:
            kv_index = int(reg.get("kv_index", -1))
            if 0 <= kv_index < len(self.prefix_kv_list):
                tok = self.prefix_token_ids[kv_index] if kv_index < len(self.prefix_token_ids) else {}
                return self.prefix_kv_list[kv_index], tok
        entry = self.get_by_span_id(str(pf_span_id)) if pf_span_id else None
        if entry is not None:
            return entry.absolute_kv, entry.token_ids
        return None

    def clear(self) -> None:
        self.segments.clear()
        self.prefix_kv_list.clear()
        self.prefix_token_ids.clear()
        self.placeholder_info.clear()
        self.span_registry.clear()
        self.prefix_span_order.clear()
        self.static_template_hash = ""
        self.topology_id = ""
        self.turn_count = 0


class SegmentCacheRegistry:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeSegmentCache] = {}

    def for_node(self, node_id: str) -> NodeSegmentCache:
        key = str(node_id)
        if key not in self._nodes:
            self._nodes[key] = NodeSegmentCache(node_id=key)
        return self._nodes[key]

    def purge_node(self, node_id: str) -> None:
        bucket = self._nodes.pop(str(node_id), None)
        if bucket is not None:
            bucket.clear()

    def clear(self) -> None:
        for bucket in self._nodes.values():
            bucket.clear()
        self._nodes.clear()
