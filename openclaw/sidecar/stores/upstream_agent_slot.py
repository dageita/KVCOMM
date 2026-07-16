"""UpstreamAgentSlot — producer/consumer contextual KV for agent_*_current placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from sidecar.stores.kv_token_pair import require_paired_slot_payload

Materialization = Literal["producer_contextual", "consumer_contextual", "isolated"]


@dataclass
class UpstreamAgentSlot:
    ph_id: str
    message_key: str
    content_hash: str
    materialization: Materialization
    absolute_kv: Any
    token_ids: dict[str, Any]
    drop_num: int = 0
    producer_node_id: str = ""
    consumer_node_id: str = ""
    upstream_node_id: str = ""
    prefix_token_len: int = 0
    slot_token_start: int = 0


class UpstreamAgentSlotRegistry:
    def __init__(self) -> None:
        self._producer: dict[str, UpstreamAgentSlot] = {}
        self._consumer: dict[str, UpstreamAgentSlot] = {}

    @staticmethod
    def producer_key(producer_node_id: str, message_key: str, ph_id: str, content_hash: str) -> str:
        return f"upprod:{producer_node_id}:{message_key}:{ph_id}:{content_hash}"

    @staticmethod
    def consumer_key(consumer_node_id: str, message_key: str, ph_id: str, content_hash: str) -> str:
        return f"upcons:{consumer_node_id}:{message_key}:{ph_id}:{content_hash}"

    def put_producer(
        self,
        *,
        producer_node_id: str,
        message_key: str,
        ph_id: str,
        content_hash: str,
        absolute_kv: Any,
        token_ids: dict[str, Any],
        prefix_token_len: int,
        drop_num: int = 0,
    ) -> UpstreamAgentSlot:
        drop_num = require_paired_slot_payload(
            absolute_kv,
            token_ids,
            drop_num=drop_num,
            require_absolute_drop=True,
            context=f"upstream producer {ph_id}",
        )
        slot = UpstreamAgentSlot(
            ph_id=str(ph_id),
            message_key=str(message_key),
            content_hash=str(content_hash),
            materialization="producer_contextual",
            absolute_kv=absolute_kv,
            token_ids=token_ids,
            drop_num=int(drop_num),
            producer_node_id=str(producer_node_id),
            prefix_token_len=int(prefix_token_len),
        )
        key = self.producer_key(producer_node_id, message_key, ph_id, content_hash)
        self._producer[key] = slot
        return slot

    def put_consumer(
        self,
        *,
        consumer_node_id: str,
        message_key: str,
        ph_id: str,
        content_hash: str,
        absolute_kv: Any,
        token_ids: dict[str, Any],
        upstream_node_id: str,
        slot_token_start: int,
        drop_num: int = 0,
    ) -> UpstreamAgentSlot:
        drop_num = require_paired_slot_payload(
            absolute_kv,
            token_ids,
            drop_num=drop_num,
            require_absolute_drop=True,
            context=f"upstream consumer {ph_id}",
        )
        slot = UpstreamAgentSlot(
            ph_id=str(ph_id),
            message_key=str(message_key),
            content_hash=str(content_hash),
            materialization="consumer_contextual",
            absolute_kv=absolute_kv,
            token_ids=token_ids,
            drop_num=int(drop_num),
            consumer_node_id=str(consumer_node_id),
            upstream_node_id=str(upstream_node_id),
            slot_token_start=int(slot_token_start),
        )
        key = self.consumer_key(consumer_node_id, message_key, ph_id, content_hash)
        self._consumer[key] = slot
        return slot

    def get_producer(
        self,
        producer_node_id: str,
        message_key: str,
        ph_id: str,
        content_hash: str,
    ) -> UpstreamAgentSlot | None:
        return self._producer.get(
            self.producer_key(producer_node_id, message_key, ph_id, content_hash)
        )

    def find_producer_for_ph(
        self,
        producer_node_id: str,
        message_key: str,
        ph_id: str,
    ) -> UpstreamAgentSlot | None:
        prefix = f"upprod:{producer_node_id}:{message_key}:{ph_id}:"
        matches = [slot for key, slot in self._producer.items() if key.startswith(prefix)]
        return matches[-1] if matches else None

    def get_consumer(
        self,
        consumer_node_id: str,
        message_key: str,
        ph_id: str,
        content_hash: str,
    ) -> UpstreamAgentSlot | None:
        return self._consumer.get(
            self.consumer_key(consumer_node_id, message_key, ph_id, content_hash)
        )

    def find_consumer_for_ph(
        self,
        consumer_node_id: str,
        message_key: str,
        ph_id: str,
    ) -> UpstreamAgentSlot | None:
        prefix = f"upcons:{consumer_node_id}:{message_key}:{ph_id}:"
        matches = [slot for key, slot in self._consumer.items() if key.startswith(prefix)]
        return matches[-1] if matches else None

    def purge_node(self, node_id: str) -> None:
        prod_prefix = f"upprod:{node_id}:"
        cons_prefix = f"upcons:{node_id}:"
        for key in list(self._producer.keys()):
            if key.startswith(prod_prefix):
                self._producer.pop(key, None)
        for key in list(self._consumer.keys()):
            if key.startswith(cons_prefix):
                self._consumer.pop(key, None)

    def purge_message(self, *, node_id: str, message_key: str) -> None:
        prod_prefix = f"upprod:{node_id}:{message_key}:"
        cons_prefix = f"upcons:{node_id}:{message_key}:"
        for key in list(self._producer.keys()):
            if key.startswith(prod_prefix):
                self._producer.pop(key, None)
        for key in list(self._consumer.keys()):
            if key.startswith(cons_prefix):
                self._consumer.pop(key, None)

    def clear(self) -> None:
        self._producer.clear()
        self._consumer.clear()
