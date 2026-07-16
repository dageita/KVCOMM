"""ToolConsumerSlot — contextual TOOL_TO_LLM consume slots (AgentMesh dataflow)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sidecar.stores.kv_token_pair import require_paired_slot_payload

Materialization = Literal["consumer_contextual", "isolated_fallback"]


@dataclass
class ToolConsumerSlot:
    ph_id: str
    message_key: str
    content_hash: str
    kv_ref: str
    materialization: Materialization
    absolute_kv: Any
    token_ids: dict[str, Any]
    consumer_node_id: str = ""
    slot_token_start: int = 0
    turn_index: int = 0
    tool_call_hash: str = ""
    drop_num: int = 0


class ToolConsumerSlotRegistry:
    def __init__(self) -> None:
        self._consumer: dict[str, ToolConsumerSlot] = {}

    @staticmethod
    def consumer_key(
        consumer_node_id: str,
        message_key: str,
        ph_id: str,
        content_hash: str,
        slot_token_start: int,
    ) -> str:
        return (
            f"toolcons:{consumer_node_id}:{message_key}:{ph_id}:"
            f"{content_hash}:pos:{int(slot_token_start)}"
        )

    def put_consumer(
        self,
        *,
        consumer_node_id: str,
        message_key: str,
        ph_id: str,
        content_hash: str,
        kv_ref: str,
        absolute_kv: Any,
        token_ids: dict[str, Any],
        slot_token_start: int,
        turn_index: int = 0,
        tool_call_hash: str = "",
        drop_num: int = 0,
        materialization: Materialization = "consumer_contextual",
    ) -> ToolConsumerSlot:
        drop_num = require_paired_slot_payload(
            absolute_kv,
            token_ids,
            drop_num=drop_num,
            require_absolute_drop=(materialization == "consumer_contextual"),
            context=f"tool consumer {ph_id}",
        )
        slot = ToolConsumerSlot(
            ph_id=str(ph_id),
            message_key=str(message_key),
            content_hash=str(content_hash),
            kv_ref=str(kv_ref),
            materialization=materialization,
            absolute_kv=absolute_kv,
            token_ids=token_ids,
            consumer_node_id=str(consumer_node_id),
            slot_token_start=int(slot_token_start),
            turn_index=int(turn_index),
            tool_call_hash=str(tool_call_hash or ""),
            drop_num=int(drop_num),
        )
        key = self.consumer_key(
            consumer_node_id,
            message_key,
            ph_id,
            content_hash,
            slot_token_start,
        )
        self._consumer[key] = slot
        return slot

    def get_consumer(
        self,
        consumer_node_id: str,
        message_key: str,
        ph_id: str,
        content_hash: str,
        slot_token_start: int,
    ) -> ToolConsumerSlot | None:
        return self._consumer.get(
            self.consumer_key(
                consumer_node_id,
                message_key,
                ph_id,
                content_hash,
                slot_token_start,
            )
        )

    def find_consumer_for_ph(
        self,
        consumer_node_id: str,
        message_key: str,
        ph_id: str,
    ) -> ToolConsumerSlot | None:
        prefix = f"toolcons:{consumer_node_id}:{message_key}:{ph_id}:"
        matches = [slot for key, slot in self._consumer.items() if key.startswith(prefix)]
        return matches[-1] if matches else None

    def purge_node(self, node_id: str) -> None:
        prefix = f"toolcons:{node_id}:"
        for key in list(self._consumer.keys()):
            if key.startswith(prefix):
                self._consumer.pop(key, None)

    def purge_message(self, *, node_id: str, message_key: str) -> None:
        prefix = f"toolcons:{node_id}:{message_key}:"
        for key in list(self._consumer.keys()):
            if key.startswith(prefix):
                self._consumer.pop(key, None)

    def purge_turns_from(self, *, node_id: str, message_key: str, turn_index: int) -> None:
        prefix = f"toolcons:{node_id}:{message_key}:"
        for key in list(self._consumer.keys()):
            if not key.startswith(prefix):
                continue
            slot = self._consumer.get(key)
            if slot is not None and slot.turn_index >= int(turn_index):
                self._consumer.pop(key, None)

    def clear(self) -> None:
        self._consumer.clear()
