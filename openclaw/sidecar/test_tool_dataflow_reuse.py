"""Unit tests for AgentMesh-style tool/assistant dataflow reuse (no torch)."""

from __future__ import annotations

import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from sidecar.stores.hashing import sha256_text
from sidecar.stores.llm_branch_slot import LlmBranchSlotRegistry
from sidecar.stores.registry import get_store_registry, reset_store_registry
from sidecar.stores.tool_consumer_slot import ToolConsumerSlotRegistry
from sidecar.stores.tool_kv_backend import ToolKVBackend
from sidecar.stores.tool_semantic_index import ToolSemanticIndex
from sidecar.stores.turn_slot_registry import TurnPhSlot


def test_cross_consumer_shares_global_tool_kv_ref() -> None:
    """Two nodes can point at the same ToolKV produce entry."""
    reset_store_registry()
    stores = get_store_registry()
    backend = stores.tool_kv
    content = "tool result payload"
    entry = backend.get_or_create(content, lambda t: ({"kv": t}, {"input_ids": [[1, 2]]}))

    for node in ("1", "2"):
        stores.turn_slots.put(
            TurnPhSlot(
                node_id=node,
                message_key="msg",
                ph_id="turn_0_tool",
                slot_kind="tool",
                content_hash=entry.content_hash,
                kv_ref=entry.kv_ref,
                turn_index=0,
            )
        )

    slot_a = stores.turn_slots.get("1", "msg", "turn_0_tool")
    slot_b = stores.turn_slots.get("2", "msg", "turn_0_tool")
    assert slot_a is not None and slot_b is not None
    assert slot_a.kv_ref == slot_b.kv_ref == entry.kv_ref


def test_tool_consumer_slots_are_per_consumer_context() -> None:
    """Same kv_ref, different consumer nodes → distinct contextual slots."""
    registry = ToolConsumerSlotRegistry()
    kv_ref = "tool:kv:abc"
    content_hash = "abc"
    for node in ("1", "2"):
        registry.put_consumer(
            consumer_node_id=node,
            message_key="msg",
            ph_id="turn_0_tool",
            content_hash=content_hash,
            kv_ref=kv_ref,
            absolute_kv={"kv": f"ctx-{node}"},
            token_ids={"input_ids": [[1]]},
            slot_token_start=10,
            turn_index=0,
        )

    a = registry.get_consumer("1", "msg", "turn_0_tool", content_hash, 10)
    b = registry.get_consumer("2", "msg", "turn_0_tool", content_hash, 10)
    assert a is not None and b is not None
    assert a.kv_ref == b.kv_ref == kv_ref
    assert a.absolute_kv != b.absolute_kv


def test_semantic_lookup_reuses_tool_kv_across_turns() -> None:
    """Semantic index maps assistant query → existing tool kv_ref."""
    index = ToolSemanticIndex()
    backend = ToolKVBackend()
    tool_text = "file contents here"
    entry = backend.get_or_create(tool_text, lambda t: ({"kv": t}, {"input_ids": [[9]]}))
    asst_query = "please read the config file"
    index.upsert(
        query=asst_query,
        kv_ref=entry.kv_ref,
        content_hash=entry.content_hash,
        token_len=1,
    )
    lookup = index.lookup(asst_query, content_hash=entry.content_hash)
    assert lookup.hit
    assert lookup.kv_ref == entry.kv_ref


def test_purge_turn_downstream_preserves_global_tool_pool() -> None:
    """Append-turn purge drops bindings but keeps ToolKV produce pool."""
    reset_store_registry()
    stores = get_store_registry()
    entry = stores.tool_kv.get_or_create("payload", lambda t: ({"kv": t}, {"input_ids": [[3]]}))
    stores.turn_slots.put(
        TurnPhSlot(
            node_id="2",
            message_key="msg",
            ph_id="turn_1_tool",
            slot_kind="tool",
            content_hash=entry.content_hash,
            kv_ref=entry.kv_ref,
            turn_index=1,
        )
    )
    stores.tool_consumer_slots.put_consumer(
        consumer_node_id="2",
        message_key="msg",
        ph_id="turn_1_tool",
        content_hash=entry.content_hash,
        kv_ref=entry.kv_ref,
        absolute_kv={"kv": "ctx"},
        token_ids={"input_ids": [[3]]},
        slot_token_start=50,
        turn_index=1,
    )
    stores.llm_branch_slots.put_consumer(
        consumer_node_id="2",
        message_key="msg",
        ph_id="turn_1_assistant",
        content_hash=sha256_text("assistant text"),
        absolute_kv={"kv": "branch"},
        token_ids={"input_ids": [[4]]},
        slot_token_start=40,
        turn_index=1,
    )

    stores.purge_turn_downstream(node_id="2", message_key="msg", turn_index=1)

    assert stores.turn_slots.get("2", "msg", "turn_1_tool") is None
    assert stores.tool_consumer_slots.find_consumer_for_ph("2", "msg", "turn_1_tool") is None
    assert stores.llm_branch_slots.find_consumer_for_ph("2", "msg", "turn_1_assistant") is None
    assert stores.tool_kv.get(entry.kv_ref) is not None


def test_llm_branch_slot_registry_purge_turns() -> None:
    registry = LlmBranchSlotRegistry()
    registry.put_consumer(
        consumer_node_id="1",
        message_key="m",
        ph_id="turn_0_assistant",
        content_hash="h0",
        absolute_kv={"kv": 0},
        token_ids={"input_ids": [[1]]},
        slot_token_start=5,
        turn_index=0,
    )
    registry.put_consumer(
        consumer_node_id="1",
        message_key="m",
        ph_id="turn_1_assistant",
        content_hash="h1",
        absolute_kv={"kv": 1},
        token_ids={"input_ids": [[2]]},
        slot_token_start=20,
        turn_index=1,
    )
    registry.purge_turns_from(node_id="1", message_key="m", turn_index=1)
    assert registry.find_consumer_for_ph("1", "m", "turn_0_assistant") is not None
    assert registry.find_consumer_for_ph("1", "m", "turn_1_assistant") is None
