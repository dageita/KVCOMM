"""Template placeholder KV sliced from base_kv_full (pure template tokens)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TemplatePhBaseRecord:
    ph_id: str
    static_template_hash: str
    topology_id: str
    ph_token_start: int
    ph_token_end: int
    pf_span_id: str | None
    absolute_kv: Any
    token_ids: dict[str, Any]


class TemplatePhBaseStore:
    """Per-node template ph KV indexed by topology anchor coordinates."""

    def __init__(self) -> None:
        self._by_node: dict[str, dict[str, TemplatePhBaseRecord]] = {}

    @staticmethod
    def record_key(
        *,
        static_template_hash: str,
        topology_id: str,
        ph_id: str,
        ph_token_start: int,
        pf_span_id: str | None,
    ) -> str:
        pf = pf_span_id or ""
        return f"{static_template_hash}:{topology_id}:{ph_id}:{ph_token_start}:{pf}"

    def put(self, node_id: str, record: TemplatePhBaseRecord) -> None:
        node_key = str(node_id)
        bucket = self._by_node.setdefault(node_key, {})
        key = self.record_key(
            static_template_hash=record.static_template_hash,
            topology_id=record.topology_id,
            ph_id=record.ph_id,
            ph_token_start=record.ph_token_start,
            pf_span_id=record.pf_span_id,
        )
        bucket[key] = record
        bucket[f"ph:{record.ph_id}"] = record

    def get_for_ph(self, node_id: str, ph_id: str) -> TemplatePhBaseRecord | None:
        node_bucket = self._by_node.get(str(node_id)) or {}
        rec = node_bucket.get(f"ph:{str(ph_id)}")
        return rec if isinstance(rec, TemplatePhBaseRecord) else None

    def get_exact(
        self,
        node_id: str,
        *,
        static_template_hash: str,
        topology_id: str,
        ph_id: str,
        ph_token_start: int,
        pf_span_id: str | None,
    ) -> TemplatePhBaseRecord | None:
        node_bucket = self._by_node.get(str(node_id)) or {}
        key = self.record_key(
            static_template_hash=static_template_hash,
            topology_id=topology_id,
            ph_id=ph_id,
            ph_token_start=ph_token_start,
            pf_span_id=pf_span_id,
        )
        return node_bucket.get(key)

    def replace_node(self, node_id: str, records: dict[str, TemplatePhBaseRecord]) -> None:
        self._by_node[str(node_id)] = dict(records)

    def purge_node(self, node_id: str) -> None:
        self._by_node.pop(str(node_id), None)

    def clear(self) -> None:
        self._by_node.clear()
