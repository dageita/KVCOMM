"""ToolSchemaBranchSlot — cached schema+prefill KV splice for consumer tool turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolSchemaBranchSlot:
    """Schema segment KV (rotary-relative to prefix boundary) plus metadata for splice."""

    consumer_node_id: str
    message_key: str
    schema_hash: str
    deliverable_hash: str
    prefix_boundary_len: int
    schema_token_len: int
    absolute_kv: Any
    token_ids: dict[str, Any]
    tool_call_hash: str = ""
    prefix_token_fingerprint: str = ""


class ToolSchemaBranchRegistry:
    def __init__(self) -> None:
        self._branches: dict[str, ToolSchemaBranchSlot] = {}

    @staticmethod
    def branch_key(
        consumer_node_id: str,
        message_key: str,
        schema_hash: str,
        deliverable_hash: str,
    ) -> str:
        return (
            f"schemabr:{consumer_node_id}:{message_key}:"
            f"{schema_hash}:{deliverable_hash}"
        )

    def put(
        self,
        *,
        consumer_node_id: str,
        message_key: str,
        schema_hash: str,
        deliverable_hash: str,
        prefix_boundary_len: int,
        schema_token_len: int,
        absolute_kv: Any,
        token_ids: dict[str, Any],
        tool_call_hash: str = "",
        prefix_token_fingerprint: str = "",
    ) -> ToolSchemaBranchSlot:
        slot = ToolSchemaBranchSlot(
            consumer_node_id=str(consumer_node_id),
            message_key=str(message_key),
            schema_hash=str(schema_hash),
            deliverable_hash=str(deliverable_hash),
            prefix_boundary_len=int(prefix_boundary_len),
            schema_token_len=int(schema_token_len),
            absolute_kv=absolute_kv,
            token_ids=token_ids,
            tool_call_hash=str(tool_call_hash or ""),
            prefix_token_fingerprint=str(prefix_token_fingerprint or ""),
        )
        self._branches[
            self.branch_key(
                consumer_node_id,
                message_key,
                schema_hash,
                deliverable_hash,
            )
        ] = slot
        return slot

    def get(
        self,
        consumer_node_id: str,
        message_key: str,
        schema_hash: str,
        deliverable_hash: str,
    ) -> ToolSchemaBranchSlot | None:
        return self._branches.get(
            self.branch_key(
                consumer_node_id,
                message_key,
                schema_hash,
                deliverable_hash,
            )
        )

    def find_for_schema(
        self,
        consumer_node_id: str,
        message_key: str,
        schema_hash: str,
    ) -> ToolSchemaBranchSlot | None:
        prefix = f"schemabr:{consumer_node_id}:{message_key}:{schema_hash}:"
        matches = [slot for key, slot in self._branches.items() if key.startswith(prefix)]
        return matches[-1] if matches else None

    def purge_node(self, node_id: str) -> None:
        prefix = f"schemabr:{node_id}:"
        for key in list(self._branches.keys()):
            if key.startswith(prefix):
                self._branches.pop(key, None)

    def purge_message(self, *, node_id: str, message_key: str) -> None:
        prefix = f"schemabr:{node_id}:{message_key}:"
        for key in list(self._branches.keys()):
            if key.startswith(prefix):
                self._branches.pop(key, None)

    def clear(self) -> None:
        self._branches.clear()
