"""Unit tests for KV/token length pair guards (no model)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SIDECAR_ROOT = Path(__file__).resolve().parents[1]
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from sidecar.stores.kv_token_pair import (
    check_kv_token_length_pair,
    require_paired_slot_payload,
    token_ids_seq_length,
)
from sidecar.stores.upstream_agent_slot import UpstreamAgentSlotRegistry


def test_require_paired_slot_rejects_empty_token_ids() -> None:
    with pytest.raises(ValueError, match="token_ids.input_ids required"):
        require_paired_slot_payload({"kv": 1}, {}, drop_num=0, context="t")


def test_require_paired_slot_rejects_nonzero_drop() -> None:
    with pytest.raises(ValueError, match="drop_num=0"):
        require_paired_slot_payload(
            {"kv": 1},
            {"input_ids": [[1, 2]]},
            drop_num=3,
            context="t",
        )


def test_require_paired_slot_allows_empty_placeholder_kv() -> None:
    assert require_paired_slot_payload({}, {}, drop_num=0) == 0


def test_upstream_put_consumer_requires_ids_and_absolute_drop() -> None:
    reg = UpstreamAgentSlotRegistry()
    with pytest.raises(ValueError, match="token_ids"):
        reg.put_consumer(
            consumer_node_id="1",
            message_key="msg",
            ph_id="agent_0_current",
            content_hash="h0",
            absolute_kv={"kv": 1},
            token_ids={},
            upstream_node_id="0",
            slot_token_start=0,
        )
    with pytest.raises(ValueError, match="drop_num=0"):
        reg.put_consumer(
            consumer_node_id="1",
            message_key="msg",
            ph_id="agent_0_current",
            content_hash="h0",
            absolute_kv={"kv": 1},
            token_ids={"input_ids": [[1]]},
            upstream_node_id="0",
            slot_token_start=0,
            drop_num=2,
        )


def test_check_kv_token_length_pair_helpers() -> None:
    assert token_ids_seq_length({"input_ids": [[1, 2, 3]]}) == 3
    assert token_ids_seq_length({}) == 0
    assert check_kv_token_length_pair(4, 4, drop_num=1) is None
    assert check_kv_token_length_pair(4, 0, drop_num=0) is not None
    assert "drop_num" in (check_kv_token_length_pair(4, 4, drop_num=5) or "")
    assert "kv_len" in (check_kv_token_length_pair(5, 4, drop_num=0) or "")
