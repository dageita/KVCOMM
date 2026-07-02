"""TurnSlotRegistry — per-node turn placeholder resolution metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


SlotKind = Literal["tool", "assistant"]


@dataclass
class TurnPhSlot:
    node_id: str
    message_key: str
    ph_id: str
    slot_kind: SlotKind
    content_hash: str
    kv_ref: str | None = None
    consumer_slot_key: str | None = None
    branch_slot_key: str | None = None
    tool_call_hash: str = ""
    absolute_kv: Any | None = None
    token_ids: dict[str, Any] | None = None
    drop_num: int = 0
    turn_index: int = 0


class TurnSlotRegistry:
    def __init__(self) -> None:
        self._slots: dict[str, TurnPhSlot] = {}

    @staticmethod
    def slot_key(node_id: str, message_key: str, ph_id: str) -> str:
        return f"turnslot:{node_id}:{message_key}:{ph_id}"

    def put(self, slot: TurnPhSlot) -> TurnPhSlot:
        key = self.slot_key(slot.node_id, slot.message_key, slot.ph_id)
        self._slots[key] = slot
        return slot

    def get(self, node_id: str, message_key: str, ph_id: str) -> TurnPhSlot | None:
        return self._slots.get(self.slot_key(node_id, message_key, ph_id))

    def list_for_message(self, node_id: str, message_key: str) -> list[TurnPhSlot]:
        prefix = f"turnslot:{node_id}:{message_key}:"
        return [slot for key, slot in self._slots.items() if key.startswith(prefix)]

    def purge_node(self, node_id: str) -> None:
        prefix = f"turnslot:{node_id}:"
        for key in list(self._slots.keys()):
            if key.startswith(prefix):
                self._slots.pop(key, None)

    def purge_message(self, *, node_id: str, message_key: str) -> None:
        prefix = f"turnslot:{node_id}:{message_key}:"
        for key in list(self._slots.keys()):
            if key.startswith(prefix):
                self._slots.pop(key, None)

    def purge_turns_from(self, *, node_id: str, message_key: str, turn_index: int) -> None:
        prefix = f"turnslot:{node_id}:{message_key}:"
        for key in list(self._slots.keys()):
            if not key.startswith(prefix):
                continue
            slot = self._slots.get(key)
            if slot is not None and slot.turn_index >= turn_index:
                self._slots.pop(key, None)

    def clear(self) -> None:
        self._slots.clear()
