"""Topology anchoring — delta keys, proactive stale purge, template ph base."""

from __future__ import annotations

import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from sidecar.stores.agent_anchor_pool import AgentAnchorPool
from sidecar.stores.hashing import static_template_hash, topology_id
from sidecar.stores.registry import reset_store_registry, get_store_registry
from sidecar.stores.template_ph_base import TemplatePhBaseRecord
from sidecar.stores.topology_anchor import (
    DeltaAnchorKey,
    coordinate_shifted,
    current_topology_keys,
    delta_key_from_ph_rec,
    new_tail_placeholder_ids,
    serialize_anchor_key,
)


def test_delta_key_from_ph_rec_includes_absolute_start_and_pf_span() -> None:
    static = static_template_hash("task\n{agent_0_current}\n")
    topo = topology_id(static_hash=static, turn_count=1)
    key = delta_key_from_ph_rec(
        ph_id="agent_0_current",
        ph_rec={"start": 42, "end": 50, "pf_span_id": "text_1"},
        static_template_hash=static,
        topology_id=topo,
        content_hash="abc",
    )
    assert key.ph_token_start == 42
    assert key.ph_token_end == 50
    assert key.pf_span_id == "text_1"
    assert key.content_hash == "abc"


def test_coordinate_shifted_detects_topology_geometry_change() -> None:
    static = "static"
    current = DeltaAnchorKey(
        static_template_hash=static,
        topology_id="topo:v2",
        ph_id="agent_0_current",
        ph_token_start=100,
        ph_token_end=110,
        pf_span_id="text_2",
    )
    stored = serialize_anchor_key(
        DeltaAnchorKey(
            static_template_hash=static,
            topology_id="topo:v1",
            ph_id="agent_0_current",
            ph_token_start=100,
            ph_token_end=110,
            pf_span_id="text_2",
        )
    )
    assert coordinate_shifted(stored, current) is True

    stored["topology_id"] = "topo:v2"
    stored["ph_token_start"] = 120
    assert coordinate_shifted(stored, current) is True


def test_new_tail_placeholder_ids_on_append_turn() -> None:
    old = {"agent_0_current": {"start": 0, "end": 5, "pf_span_id": "T0"}}
    new = {
        **old,
        "turn_1_assistant": {"start": 20, "end": 25, "pf_span_id": "text_1"},
    }
    assert new_tail_placeholder_ids(old, new) == {"turn_1_assistant"}


def test_agent_anchor_pool_topology_key_and_proactive_purge() -> None:
    pool = AgentAnchorPool()
    static = static_template_hash("x")
    topo = topology_id(static_hash=static, turn_count=0)
    key_v1 = DeltaAnchorKey(
        static_template_hash=static,
        topology_id=topo,
        ph_id="agent_0_current",
        ph_token_start=10,
        ph_token_end=20,
        pf_span_id="T0",
    )
    pool.put(
        node_id="1",
        message_key="msg",
        ph_id="agent_0_current",
        static_template_hash=static,
        upstream_hash="msg",
        ph_delta={"k": 1},
        delta_key=key_v1,
    )
    hit = pool.get_by_topology_key(node_id="1", message_key="msg", delta_key=key_v1)
    assert hit is not None
    assert hit.ph_delta == {"k": 1}

    key_v2 = DeltaAnchorKey(
        static_template_hash=static,
        topology_id=topo,
        ph_id="agent_0_current",
        ph_token_start=10,
        ph_token_end=20,
        pf_span_id="text_1",
    )
    removed = pool.purge_stale_topology(
        node_id="1",
        message_key="msg",
        current_keys={"agent_0_current": key_v2},
    )
    assert removed == ["agent_0_current"]
    assert pool.get_by_topology_key(node_id="1", message_key="msg", delta_key=key_v1) is None


def test_template_ph_base_store_slice_by_coordinates() -> None:
    reset_store_registry()
    stores = get_store_registry()
    static = static_template_hash("task")
    topo = topology_id(static_hash=static, turn_count=0)
    rec = TemplatePhBaseRecord(
        ph_id="agent_0_current",
        static_template_hash=static,
        topology_id=topo,
        ph_token_start=5,
        ph_token_end=10,
        pf_span_id="T0",
        absolute_kv={"slice": "5:10"},
        token_ids={"input_ids": [[1, 2, 3]]},
    )
    stores.template_ph_base.put("1", rec)
    got = stores.template_ph_base.get_for_ph("1", "agent_0_current")
    assert got is not None
    assert got.ph_token_start == 5
    assert got.absolute_kv == {"slice": "5:10"}


def test_current_topology_keys_from_bucket() -> None:
    static = static_template_hash("u")
    topo = topology_id(static_hash=static, turn_count=2)
    bucket = {
        "static_template_hash": static,
        "topology_id": topo,
        "placeholder_info": {
            "agent_1_current": {"start": 30, "end": 40, "pf_span_id": "text_1"},
        },
    }
    keys = current_topology_keys(bucket)
    assert "agent_1_current" in keys
    assert keys["agent_1_current"].ph_token_start == 30
    assert keys["agent_1_current"].pf_span_id == "text_1"
