"""Unit tests for UpstreamAgentSlot registry (no torch)."""

from __future__ import annotations

import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from sidecar.stores.upstream_agent_slot import UpstreamAgentSlotRegistry


def test_producer_and_consumer_slot_keys() -> None:
    reg = UpstreamAgentSlotRegistry()
    reg.put_producer(
        producer_node_id="0",
        message_key="msg",
        ph_id="agent_0_current",
        content_hash="h0",
        absolute_kv={"kv": 1},
        token_ids={"input_ids": [[1, 2]]},
        prefix_token_len=100,
    )
    reg.put_consumer(
        consumer_node_id="1",
        message_key="msg",
        ph_id="agent_0_current",
        content_hash="h0",
        absolute_kv={"kv": 2},
        token_ids={"input_ids": [[1, 2]]},
        upstream_node_id="0",
        slot_token_start=50,
    )

    prod = reg.get_producer("0", "msg", "agent_0_current", "h0")
    cons = reg.get_consumer("1", "msg", "agent_0_current", "h0")
    assert prod is not None and prod.materialization == "producer_contextual"
    assert cons is not None and cons.materialization == "consumer_contextual"
    assert reg.find_consumer_for_ph("1", "msg", "agent_0_current") is cons


def test_purge_message_clears_both_sides() -> None:
    reg = UpstreamAgentSlotRegistry()
    reg.put_producer(
        producer_node_id="0",
        message_key="msg",
        ph_id="agent_0_current",
        content_hash="h0",
        absolute_kv={},
        token_ids={},
        prefix_token_len=1,
    )
    reg.put_consumer(
        consumer_node_id="1",
        message_key="msg",
        ph_id="agent_0_current",
        content_hash="h0",
        absolute_kv={},
        token_ids={},
        upstream_node_id="0",
        slot_token_start=1,
    )
    reg.purge_message(node_id="1", message_key="msg")
    assert reg.get_consumer("1", "msg", "agent_0_current", "h0") is None
    assert reg.get_producer("0", "msg", "agent_0_current", "h0") is not None

    reg.purge_message(node_id="0", message_key="msg")
    assert reg.get_producer("0", "msg", "agent_0_current", "h0") is None
