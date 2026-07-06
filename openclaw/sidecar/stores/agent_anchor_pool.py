"""AgentAnchorPool — topology-anchored ph/pf delta storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sidecar.stores.hashing import short_hash
from sidecar.stores.topology_anchor import DeltaAnchorKey, serialize_anchor_key


@dataclass
class AgentAnchorEntry:
    anchor_key: str
    node_id: str
    message_key: str
    ph_id: str
    static_template_hash: str
    topology_id: str
    upstream_hash: str
    ph_token_start: int = 0
    ph_token_end: int = 0
    pf_span_id: str | None = None
    content_hash: str = ""
    ph_key_embedding: Any | None = None
    ph_value_embedding: Any | None = None
    ph_delta: Any = None
    ph_value_delta: Any | None = None
    pf_delta: Any | None = None
    pf_value_delta: Any | None = None
    pf_segment_len: int | None = None
    materialization: str = "contextual_delta"


class AgentAnchorPool:
    def __init__(self) -> None:
        self._entries: dict[str, AgentAnchorEntry] = {}

    @staticmethod
    def make_key(
        *,
        node_id: str,
        message_key: str,
        ph_id: str,
        static_template_hash: str,
        upstream_hash: str,
    ) -> str:
        return (
            f"agent:{node_id}:{message_key}:{ph_id}:"
            f"static:{short_hash(static_template_hash)}:up:{short_hash(upstream_hash)}"
        )

    @staticmethod
    def make_topology_key(delta_key: DeltaAnchorKey, *, node_id: str, message_key: str) -> str:
        return (
            f"topo:{node_id}:{message_key}:{delta_key.ph_id}:"
            f"static:{short_hash(delta_key.static_template_hash)}:"
            f"topo:{short_hash(delta_key.topology_id)}:"
            f"pos:{delta_key.ph_token_start}:{delta_key.ph_token_end}:"
            f"pf:{short_hash(str(delta_key.pf_span_id or ''))}:"
            f"content:{short_hash(delta_key.content_hash or 'none')}"
        )

    def put(
        self,
        *,
        node_id: str,
        message_key: str,
        ph_id: str,
        static_template_hash: str,
        upstream_hash: str,
        ph_delta: Any,
        pf_delta: Any | None = None,
        pf_segment_len: int | None = None,
        ph_value_delta: Any | None = None,
        pf_value_delta: Any | None = None,
        ph_key_embedding: Any | None = None,
        ph_value_embedding: Any | None = None,
        delta_key: DeltaAnchorKey | None = None,
    ) -> AgentAnchorEntry:
        if delta_key is not None:
            anchor_key = self.make_topology_key(delta_key, node_id=node_id, message_key=message_key)
            static_template_hash = delta_key.static_template_hash
            ph_token_start = delta_key.ph_token_start
            ph_token_end = delta_key.ph_token_end
            pf_span_id = delta_key.pf_span_id
            content_hash = delta_key.content_hash
        else:
            anchor_key = self.make_key(
                node_id=node_id,
                message_key=message_key,
                ph_id=ph_id,
                static_template_hash=static_template_hash,
                upstream_hash=upstream_hash,
            )
            ph_token_start = 0
            ph_token_end = 0
            pf_span_id = None
            content_hash = ""

        entry = AgentAnchorEntry(
            anchor_key=anchor_key,
            node_id=str(node_id),
            message_key=str(message_key),
            ph_id=ph_id,
            static_template_hash=static_template_hash,
            topology_id=str(delta_key.topology_id if delta_key else ""),
            upstream_hash=upstream_hash,
            ph_token_start=int(ph_token_start),
            ph_token_end=int(ph_token_end),
            pf_span_id=pf_span_id,
            content_hash=str(content_hash),
            ph_key_embedding=ph_key_embedding,
            ph_value_embedding=ph_value_embedding,
            ph_delta=ph_delta,
            ph_value_delta=ph_value_delta,
            pf_delta=pf_delta,
            pf_value_delta=pf_value_delta,
            pf_segment_len=pf_segment_len,
        )
        self._entries[anchor_key] = entry
        return entry

    def get(
        self,
        *,
        node_id: str,
        message_key: str,
        ph_id: str,
        static_template_hash: str,
        upstream_hash: str,
    ) -> AgentAnchorEntry | None:
        key = self.make_key(
            node_id=node_id,
            message_key=message_key,
            ph_id=ph_id,
            static_template_hash=static_template_hash,
            upstream_hash=upstream_hash,
        )
        return self._entries.get(key)

    def get_by_topology_key(
        self,
        *,
        node_id: str,
        message_key: str,
        delta_key: DeltaAnchorKey,
    ) -> AgentAnchorEntry | None:
        key = self.make_topology_key(delta_key, node_id=node_id, message_key=message_key)
        return self._entries.get(key)

    def get_any_for_ph(self, *, node_id: str, message_key: str, ph_id: str) -> AgentAnchorEntry | None:
        prefix = f"topo:{node_id}:{message_key}:{ph_id}:"
        legacy_prefix = f"agent:{node_id}:{message_key}:{ph_id}:"
        matches = [
            entry
            for key, entry in self._entries.items()
            if key.startswith(prefix) or key.startswith(legacy_prefix)
        ]
        return matches[-1] if matches else None

    def list_for_message(self, *, node_id: str, message_key: str) -> list[AgentAnchorEntry]:
        prefix = f"topo:{node_id}:{message_key}:"
        legacy = f"agent:{node_id}:{message_key}:"
        return [
            entry
            for key, entry in self._entries.items()
            if key.startswith(prefix) or key.startswith(legacy)
        ]

    def purge_stale_topology(
        self,
        *,
        node_id: str,
        message_key: str,
        current_keys: dict[str, DeltaAnchorKey],
        purge_all: bool = False,
    ) -> list[str]:
        """Remove entries whose coordinates no longer match current topology."""
        removed: list[str] = []
        prefix = f"topo:{node_id}:{message_key}:"
        for key in list(self._entries.keys()):
            if not key.startswith(prefix):
                continue
            if purge_all:
                entry = self._entries.pop(key, None)
                if entry is not None:
                    removed.append(entry.ph_id)
                continue
            entry = self._entries.get(key)
            if entry is None:
                continue
            current = current_keys.get(entry.ph_id)
            if current is None:
                self._entries.pop(key, None)
                removed.append(entry.ph_id)
                continue
            if (
                entry.ph_token_start != current.ph_token_start
                or entry.ph_token_end != current.ph_token_end
                or str(entry.pf_span_id or "") != str(current.pf_span_id or "")
                or entry.static_template_hash != current.static_template_hash
                or entry.topology_id != current.topology_id
                or (
                    entry.content_hash
                    and current.content_hash
                    and entry.content_hash != current.content_hash
                )
            ):
                self._entries.pop(key, None)
                removed.append(entry.ph_id)
        return sorted(set(removed))

    def purge_node(self, node_id: str) -> None:
        prefix = f"topo:{node_id}:"
        legacy = f"agent:{node_id}:"
        for key in list(self._entries.keys()):
            if key.startswith(prefix) or key.startswith(legacy):
                self._entries.pop(key, None)

    def purge_message(self, *, node_id: str, message_key: str) -> None:
        prefix = f"topo:{node_id}:{message_key}:"
        legacy = f"agent:{node_id}:{message_key}:"
        for key in list(self._entries.keys()):
            if key.startswith(prefix) or key.startswith(legacy):
                self._entries.pop(key, None)

    def invalidate_pf_for_message(self, *, node_id: str, message_key: str) -> None:
        """Drop pf deltas when prefix topology grows; ph deltas may remain valid."""
        for entry in self.list_for_message(node_id=node_id, message_key=message_key):
            entry.pf_delta = None
            entry.pf_value_delta = None
            entry.pf_segment_len = None

    def export_for_request(self, *, node_id: str, message_key: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for entry in self.list_for_message(node_id=node_id, message_key=message_key):
            out[entry.ph_id] = {
                "ph": entry.ph_delta,
                "pf": entry.pf_delta,
                "pf_segment_len": entry.pf_segment_len,
                "topology_key": serialize_anchor_key(
                    DeltaAnchorKey(
                        static_template_hash=entry.static_template_hash,
                        topology_id=entry.topology_id,
                        ph_id=entry.ph_id,
                        ph_token_start=entry.ph_token_start,
                        ph_token_end=entry.ph_token_end,
                        pf_span_id=entry.pf_span_id,
                        content_hash=entry.content_hash,
                    )
                ),
            }
        return out

    def import_from_legacy_snapshot(
        self,
        *,
        node_id: str,
        message_key: str,
        static_template_hash: str,
        upstream_hash: str,
        snapshot: dict[str, Any],
    ) -> None:
        for ph_id, payload in snapshot.items():
            if not isinstance(payload, dict):
                continue
            if not str(ph_id).startswith("agent_"):
                continue
            self.put(
                node_id=node_id,
                message_key=message_key,
                ph_id=str(ph_id),
                static_template_hash=static_template_hash,
                upstream_hash=upstream_hash,
                ph_delta=payload.get("ph"),
                pf_delta=payload.get("pf"),
                pf_segment_len=payload.get("pf_segment_len"),
            )

    def clear(self) -> None:
        self._entries.clear()
