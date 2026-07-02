"""Unit tests for typed KVCOMM store scaffolding (no torch/HF required)."""

from __future__ import annotations

import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from sidecar.stores.agent_anchor_pool import AgentAnchorPool
from sidecar.stores.asst_anchor_pool import AsstAnchorPool
from sidecar.stores.hashing import static_template_hash, topology_id
from sidecar.stores.prefix_topology import plan_prefix_update, write_topology
from sidecar.stores.registry import reset_store_registry, get_store_registry
from sidecar.stores.tool_kv_backend import ToolKVBackend
from sidecar.stores.tool_semantic_index import ToolSemanticIndex
from sidecar.stores.topology_anchor import DeltaAnchorKey
from sidecar.stores.turn_slot_registry import TurnPhSlot, TurnSlotRegistry


def test_static_template_hash_ignores_turn_placeholders() -> None:
    base = "User task\n{agent_0_current}\n"
    with_turns = base + "{turn_0_assistant}\n{turn_0_tool}\n{turn_1_assistant}\n"
    assert static_template_hash(base) == static_template_hash(with_turns)


def test_topology_plan_append_vs_regression() -> None:
    user = "task\n{turn_0_assistant}\n{turn_0_tool}\n"
    static_hash = static_template_hash(user)
    bucket = {
        "static_template_hash": static_hash,
        "topology_id": topology_id(static_hash=static_hash, turn_count=1),
        "turn_count": 1,
        "prefix": ["seg"],
    }
    append = plan_prefix_update(
        user_template=user + "{turn_1_assistant}\n{turn_1_tool}\n",
        desired_turn_count=2,
        bucket=bucket,
        initialized=True,
    )
    assert append.action == "append_turn"

    regression = plan_prefix_update(
        user_template=user,
        desired_turn_count=1,
        bucket={
            **bucket,
            "topology_id": topology_id(static_hash=static_hash, turn_count=2),
            "turn_count": 2,
        },
        initialized=True,
    )
    assert regression.action == "static_rebuild"
    assert regression.reason == "turn_count_regression_new_run"


def test_tool_kv_backend_deduplicates_by_content() -> None:
    backend = ToolKVBackend()
    calls = {"n": 0}

    def forward(text: str):
        calls["n"] += 1
        return {"kv": text}, {"input_ids": [[1, 2, 3]]}

    e1 = backend.get_or_create("hello tool", forward)
    e2 = backend.get_or_create("hello tool", forward)
    assert calls["n"] == 1
    assert e1.kv_ref == e2.kv_ref
    assert e2.ref_count == 2


def test_tool_semantic_index_exact_hash_hit() -> None:
    index = ToolSemanticIndex()
    index.upsert(query="read file", kv_ref="tool:kv:abc", content_hash="abc", token_len=10)
    result = index.lookup("anything", content_hash="abc")
    assert result.hit
    assert result.match_mode == "exact_hash"
    assert result.kv_ref == "tool:kv:abc"


def test_turn_slot_registry_and_agent_pf_invalidation() -> None:
    reset_store_registry()
    stores = get_store_registry()
    stores.turn_slots.put(
        TurnPhSlot(
            node_id="1",
            message_key="msg",
            ph_id="turn_0_tool",
            slot_kind="tool",
            content_hash="h1",
            kv_ref="tool:kv:h1",
            turn_index=0,
        )
    )
    assert stores.turn_slots.get("1", "msg", "turn_0_tool") is not None

    pool = AgentAnchorPool()
    pool.put(
        node_id="1",
        message_key="msg",
        ph_id="agent_0_current",
        static_template_hash="static",
        upstream_hash="up",
        ph_delta={"k": 1},
        pf_delta={"k": 2},
        pf_segment_len=100,
    )
    pool.invalidate_pf_for_message(node_id="1", message_key="msg")
    entry = pool.get_any_for_ph(node_id="1", message_key="msg", ph_id="agent_0_current")
    assert entry is not None
    assert entry.ph_delta == {"k": 1}
    assert entry.pf_delta is None


def test_agent_anchor_pool_topology_purge_on_append() -> None:
    pool = AgentAnchorPool()
    static = static_template_hash("base")
    topo = topology_id(static_hash=static, turn_count=1)
    old_key = DeltaAnchorKey(
        static_template_hash=static,
        topology_id=topo,
        ph_id="agent_0_current",
        ph_token_start=0,
        ph_token_end=5,
        pf_span_id="T0",
    )
    pool.put(
        node_id="2",
        message_key="m",
        ph_id="agent_0_current",
        static_template_hash=static,
        upstream_hash="m",
        ph_delta={"old": True},
        delta_key=old_key,
    )
    new_topo = topology_id(static_hash=static, turn_count=2)
    current = {
        "agent_0_current": DeltaAnchorKey(
            static_template_hash=static,
            topology_id=new_topo,
            ph_id="agent_0_current",
            ph_token_start=0,
            ph_token_end=5,
            pf_span_id="T0",
        )
    }
    removed = pool.purge_stale_topology(
        node_id="2",
        message_key="m",
        current_keys=current,
    )
    assert removed == ["agent_0_current"]


def test_write_topology_updates_bucket() -> None:
    bucket: dict = {}
    user = "static\n{turn_0_assistant}\n"
    write_topology(bucket, user_template=user, turn_count=1)
    assert bucket["turn_count"] == 1
    assert bucket["user_template"] == user.strip()
    assert bucket["static_template_hash"] == static_template_hash(user)
    assert bucket["topology_id"].endswith(":turns:1")


def test_purge_turn_downstream_preserves_tool_kv() -> None:
    reset_store_registry()
    stores = get_store_registry()
    entry = stores.tool_kv.get_or_create("x", lambda t: ({"kv": t}, {"input_ids": [[1]]}))
    stores.turn_slots.put(
        TurnPhSlot(
            node_id="1",
            message_key="msg",
            ph_id="turn_0_tool",
            slot_kind="tool",
            content_hash=entry.content_hash,
            kv_ref=entry.kv_ref,
            turn_index=0,
        )
    )
    stores.purge_turn_downstream(node_id="1", message_key="msg", turn_index=0)
    assert stores.turn_slots.get("1", "msg", "turn_0_tool") is None
    assert stores.tool_kv.get(entry.kv_ref) is not None


def test_asst_anchor_pool_left_fingerprint() -> None:
    pool = AsstAnchorPool()
    pool.put(
        node_id="2",
        message_key="m",
        ph_id="turn_0_assistant",
        content_hash="c",
        left_parts=["static", "0"],
        static_template_hash="static",
        ph_delta={"d": 1},
    )
    hit = pool.get(
        node_id="2",
        message_key="m",
        ph_id="turn_0_assistant",
        content_hash="c",
        left_parts=["static", "0"],
        static_template_hash="static",
    )
    assert hit is not None
    assert hit.ph_delta == {"d": 1}
