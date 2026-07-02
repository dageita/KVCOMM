"""Central registry wiring typed KVCOMM stores to sidecar / LLMChat."""

from __future__ import annotations

from sidecar.stores.agent_anchor_pool import AgentAnchorPool
from sidecar.stores.asst_anchor_pool import AsstAnchorPool
from sidecar.stores.llm_branch_slot import LlmBranchSlotRegistry
from sidecar.stores.segment_cache import SegmentCacheRegistry
from sidecar.stores.tool_consumer_slot import ToolConsumerSlotRegistry
from sidecar.stores.tool_kv_backend import ToolKVBackend
from sidecar.stores.tool_semantic_index import ToolSemanticIndex
from sidecar.stores.template_ph_base import TemplatePhBaseStore
from sidecar.stores.turn_slot_registry import TurnSlotRegistry
from sidecar.stores.upstream_agent_slot import UpstreamAgentSlotRegistry

_REGISTRY: "KvcommStoreRegistry | None" = None


class KvcommStoreRegistry:
    def __init__(self) -> None:
        self.segment_cache = SegmentCacheRegistry()
        self.agent_anchors = AgentAnchorPool()
        self.asst_anchors = AsstAnchorPool()
        self.tool_kv = ToolKVBackend()
        self.tool_semantic = ToolSemanticIndex()
        self.tool_consumer_slots = ToolConsumerSlotRegistry()
        self.llm_branch_slots = LlmBranchSlotRegistry()
        self.turn_slots = TurnSlotRegistry()
        self.upstream_agent_slots = UpstreamAgentSlotRegistry()
        self.template_ph_base = TemplatePhBaseStore()

    def purge_node(self, node_id: str, *, message_key: str | None = None) -> None:
        if message_key is not None:
            self.agent_anchors.purge_message(node_id=node_id, message_key=message_key)
            self.asst_anchors.purge_message(node_id=node_id, message_key=message_key)
            self.turn_slots.purge_message(node_id=node_id, message_key=message_key)
            self.tool_consumer_slots.purge_message(node_id=node_id, message_key=message_key)
            self.llm_branch_slots.purge_message(node_id=node_id, message_key=message_key)
            self.upstream_agent_slots.purge_message(node_id=node_id, message_key=message_key)
            return
        self.segment_cache.purge_node(node_id)
        self.agent_anchors.purge_node(node_id)
        self.asst_anchors.purge_node(node_id)
        self.turn_slots.purge_node(node_id)
        self.tool_consumer_slots.purge_node(node_id)
        self.llm_branch_slots.purge_node(node_id)
        self.upstream_agent_slots.purge_node(node_id)
        self.template_ph_base.purge_node(node_id)

    def purge_turn_downstream(self, *, node_id: str, message_key: str, turn_index: int) -> None:
        """Drop turn bindings from turn_index onward; global ToolKV pool is preserved."""
        self.turn_slots.purge_turns_from(
            node_id=node_id,
            message_key=message_key,
            turn_index=turn_index,
        )
        self.tool_consumer_slots.purge_turns_from(
            node_id=node_id,
            message_key=message_key,
            turn_index=turn_index,
        )
        self.llm_branch_slots.purge_turns_from(
            node_id=node_id,
            message_key=message_key,
            turn_index=turn_index,
        )

    def clear(self) -> None:
        self.segment_cache.clear()
        self.agent_anchors.clear()
        self.asst_anchors.clear()
        self.tool_kv.clear()
        self.tool_semantic.clear()
        self.tool_consumer_slots.clear()
        self.llm_branch_slots.clear()
        self.turn_slots.clear()
        self.upstream_agent_slots.clear()
        self.template_ph_base.clear()


def get_store_registry() -> KvcommStoreRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = KvcommStoreRegistry()
    return _REGISTRY


def reset_store_registry() -> None:
    global _REGISTRY
    if _REGISTRY is not None:
        _REGISTRY.clear()
    _REGISTRY = KvcommStoreRegistry()
