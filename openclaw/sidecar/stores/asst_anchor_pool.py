"""AsstAnchorPool — contextual ph_delta for turn_*_assistant placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sidecar.stores.hashing import left_fingerprint, short_hash


@dataclass
class AsstAnchorEntry:
    anchor_key: str
    node_id: str
    message_key: str
    ph_id: str
    content_hash: str
    left_fingerprint: str
    static_template_hash: str
    ph_delta: Any
    pf_delta: Any | None = None
    pf_segment_len: int | None = None
    materialization: str = "contextual_delta"


class AsstAnchorPool:
    def __init__(self) -> None:
        self._entries: dict[str, AsstAnchorEntry] = {}

    @staticmethod
    def make_key(
        *,
        node_id: str,
        message_key: str,
        ph_id: str,
        content_hash: str,
        left_fp: str,
        static_template_hash: str,
    ) -> str:
        return (
            f"asst:{node_id}:{message_key}:{ph_id}:"
            f"content:{short_hash(content_hash)}:left:{short_hash(left_fp)}:"
            f"static:{short_hash(static_template_hash)}"
        )

    def put(
        self,
        *,
        node_id: str,
        message_key: str,
        ph_id: str,
        content_hash: str,
        left_parts: list[str],
        static_template_hash: str,
        ph_delta: Any,
        pf_delta: Any | None = None,
        pf_segment_len: int | None = None,
    ) -> AsstAnchorEntry:
        left_fp = left_fingerprint(left_parts)
        anchor_key = self.make_key(
            node_id=node_id,
            message_key=message_key,
            ph_id=ph_id,
            content_hash=content_hash,
            left_fp=left_fp,
            static_template_hash=static_template_hash,
        )
        entry = AsstAnchorEntry(
            anchor_key=anchor_key,
            node_id=str(node_id),
            message_key=str(message_key),
            ph_id=ph_id,
            content_hash=content_hash,
            left_fingerprint=left_fp,
            static_template_hash=static_template_hash,
            ph_delta=ph_delta,
            pf_delta=pf_delta,
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
        content_hash: str,
        left_parts: list[str],
        static_template_hash: str,
    ) -> AsstAnchorEntry | None:
        left_fp = left_fingerprint(left_parts)
        key = self.make_key(
            node_id=node_id,
            message_key=message_key,
            ph_id=ph_id,
            content_hash=content_hash,
            left_fp=left_fp,
            static_template_hash=static_template_hash,
        )
        return self._entries.get(key)

    def invalidate_downstream_of_turn(self, *, node_id: str, message_key: str, turn_index: int) -> None:
        """Drop assistant anchors at or after turn_index when prefix grows."""
        needle = f"asst:{node_id}:{message_key}:turn_{turn_index}_"
        for key in list(self._entries.keys()):
            if not key.startswith(f"asst:{node_id}:{message_key}:"):
                continue
            ph_part = key.split(":", 3)[3] if key.count(":") >= 3 else ""
            if ph_part.startswith(f"turn_{turn_index}_") or self._turn_index_from_key(key) >= turn_index:
                self._entries.pop(key, None)

    @staticmethod
    def _turn_index_from_key(key: str) -> int:
        try:
            ph_id = key.split(":", 3)[3]
            if ph_id.startswith("turn_"):
                return int(ph_id.split("_")[1])
        except (IndexError, ValueError):
            pass
        return 10**9

    def purge_node(self, node_id: str) -> None:
        prefix = f"asst:{node_id}:"
        for key in list(self._entries.keys()):
            if key.startswith(prefix):
                self._entries.pop(key, None)

    def purge_message(self, *, node_id: str, message_key: str) -> None:
        prefix = f"asst:{node_id}:{message_key}:"
        for key in list(self._entries.keys()):
            if key.startswith(prefix):
                self._entries.pop(key, None)

    def export_for_request(self, *, node_id: str, message_key: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        prefix = f"asst:{node_id}:{message_key}:"
        for key, entry in self._entries.items():
            if not key.startswith(prefix):
                continue
            out[entry.ph_id] = {
                "ph": entry.ph_delta,
                "pf": entry.pf_delta,
                "pf_segment_len": entry.pf_segment_len,
            }
        return out

    def clear(self) -> None:
        self._entries.clear()
