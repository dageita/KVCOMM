"""Anchor routing and prefix-reset purge checks for clawbench kv_reuse."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    import torch
    from transformers.cache_utils import DynamicCache
except ImportError:  # pragma: no cover
    torch = None
    DynamicCache = None


def _fake_cache(seen_tokens: int) -> DynamicCache:
    layer = torch.zeros(1, 1, seen_tokens, 1)
    cache = DynamicCache()
    cache.key_cache = [layer]
    cache.value_cache = [layer.clone()]
    cache._seen_tokens = seen_tokens
    return cache


@pytest.fixture(autouse=True)
def _reset_engine_state():
    from KVCOMM.llm.gpt_chat import LLMChat
    from KVCOMM.llm.kvcomm_engine import KVCOMMEngine

    LLMChat._shared_kv_cache_memory.clear()
    LLMChat._initialization.clear()
    for store in (
        KVCOMMEngine.anchors,
        KVCOMMEngine.anchor_dict,
        KVCOMMEngine.weight_dict,
        KVCOMMEngine._request_states,
    ):
        store.clear()
    KVCOMMEngine._staged_commits.clear()
    KVCOMMEngine._active_requests.clear()
    yield


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_soft_anchor_gaps_reject_missing_agent_current() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat
    from KVCOMM.llm.kvcomm_engine import KVCOMMEngine

    chat = object.__new__(LLMChat)
    chat.node_id = "1"
    LLMChat._shared_kv_cache_memory["1"] = {
        "placeholder_info": {"agent_0_current": [10, 20]},
        "prefix": [_fake_cache(5), _fake_cache(100)],
    }

    state = KVCOMMEngine._get_request_state("route-test")
    state.anchor_dict.setdefault("agent_0_current", {})["msg"] = True
    LLMChat._shared_kv_cache_memory["0"] = {
        "response": {"msg": [_fake_cache(40)]},
        "response_ids": {"msg": [{"input_ids": torch.zeros(1, 40, dtype=torch.long)}]},
        "response_drop_num": {"msg": [0]},
    }

    assert chat.can_kv_reuse_with_soft_anchor_gaps("route-test", "msg") is True
    assert chat.resolve_generation_mode("route-test", "msg", "kv_reuse") == "kv_reuse"


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_soft_anchor_gaps_allow_missing_turn_only() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat
    from KVCOMM.llm.kvcomm_engine import KVCOMMEngine

    chat = object.__new__(LLMChat)
    chat.node_id = "1"
    LLMChat._shared_kv_cache_memory["1"] = {
        "placeholder_info": {"turn_1_assistant": [10, 20]},
        "prefix": [_fake_cache(5), _fake_cache(50)],
    }

    state = KVCOMMEngine._get_request_state("route-test")
    state.anchor_dict.setdefault("turn_1_assistant", {})["msg"] = True

    assert chat.can_kv_reuse_with_soft_anchor_gaps("route-test", "msg") is True


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_find_incompatible_detects_pf_length_mismatch() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat
    from KVCOMM.llm.kvcomm_engine import KVCOMMEngine

    chat = object.__new__(LLMChat)
    chat.node_id = "1"
    chat.kv_engine = KVCOMMEngine(chat)

    pf_len = 325
    LLMChat._shared_kv_cache_memory["1"] = {
        "placeholder_info": {"agent_0_current": [10, 20]},
        "prefix": [_fake_cache(5), _fake_cache(pf_len)],
    }
    LLMChat._shared_kv_cache_memory["0"] = {
        "response": {"msg": [_fake_cache(40)]},
        "response_ids": {"msg": [{"input_ids": torch.zeros(1, 40, dtype=torch.long)}]},
        "response_drop_num": {"msg": [0]},
    }

    stale_pf = torch.zeros(1, 1, 317, 1)
    KVCOMMEngine.anchors["agent_0_current"] = {
        "msg": {"1_pf_key_delta": stale_pf},
    }

    incompatible = chat.find_incompatible_anchor_deltas("route-test", "msg")
    assert incompatible == []


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_reset_prefix_node_purges_node_anchor_deltas() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat
    from KVCOMM.llm.kvcomm_engine import KVCOMMEngine
    from sidecar.kvcomm_adapter import _reset_prefix_node

    delta = torch.zeros(1, 1, 10, 1)
    KVCOMMEngine.anchors["agent_0_current"] = {
        "msg": {
            "1_pf_key_delta": delta,
            "1_ph_key_delta": delta.clone(),
        },
    }
    LLMChat._shared_kv_cache_memory["1"] = {
        "prefix": [_fake_cache(3)],
        "placeholder_info": {"agent_0_current": [0, 1]},
        "turn_count": 2,
    }
    LLMChat._initialization["1"] = True

    _reset_prefix_node("1")

    assert "prefix" not in LLMChat._shared_kv_cache_memory["1"]
    entry = KVCOMMEngine.anchors.get("agent_0_current", {}).get("msg", {})
    assert "1_pf_key_delta" not in entry
    assert "1_ph_key_delta" not in entry


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_offset_kv_cache_pair_reports_blend_failure() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat
    from KVCOMM.llm.kvcomm_engine import KVCOMMEngine

    chat = object.__new__(LLMChat)
    chat.node_id = "1"
    engine = KVCOMMEngine(chat)

    base_ph = _fake_cache(40)
    base_pf = _fake_cache(325)
    pf_delta = torch.zeros(1, 1, 317, 1)
    anchor = {
        "1_ph_key_delta": torch.zeros(1, 1, 40, 1),
        "1_ph_value_delta": torch.zeros(1, 1, 40, 1),
        "1_pf_key_delta": pf_delta,
        "1_pf_value_delta": torch.zeros(1, 1, 317, 1),
    }

    _ph, _pf, blended = engine.offset_kv_cache_pair(
        "agent_0_current",
        "msg",
        "req",
        base_ph,
        base_pf,
        [anchor],
    )
    assert blended is False


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_kv_reuse_anchors_skip_isolated_turn_and_upstream() -> None:
    from types import SimpleNamespace

    from KVCOMM.llm.gpt_chat import LLMChat

    chat = object.__new__(LLMChat)
    chat.node_id = "2"
    slot = SimpleNamespace(absolute_kv=_fake_cache(5), kv_ref=None, slot_kind="tool", content_hash="h1")
    chat.resolve_turn_ph_slot = lambda ph_id, _msg: slot if ph_id == "turn_1_tool" else None
    chat.resolve_tool_consumer_slot = lambda _ph_id, _msg: None
    chat.resolve_llm_branch_slot = lambda _ph_id, _msg: None
    chat.resolve_upstream_agent_slot = lambda _ph_id, _msg: None
    chat._resolve_upstream_consumer_slot = lambda ph_id, _msg: (
        type("S", (), {"materialization": "consumer_contextual"})()
        if ph_id == "agent_1_current"
        else None
    )

    LLMChat._shared_kv_cache_memory["1"] = {
        "response": {"msg": [_fake_cache(40)]},
    }
    stale = {"msg": {"2_ph_key_delta": torch.zeros(1)}}
    anchors = {"turn_1_tool": stale, "agent_1_current": stale, "agent_2_current": stale}

    assert chat._kv_reuse_anchors_for_ph("turn_1_tool", "msg", anchors) == []
    assert chat._kv_reuse_anchors_for_ph("agent_1_current", "msg", anchors) == []
    assert len(chat._kv_reuse_anchors_for_ph("agent_2_current", "msg", anchors)) == 1


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_turn_tool_dual_lock_allows_delta_only_after_contextual_materialize() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat
    from sidecar.stores.agent_anchor_pool import AgentAnchorPool
    from sidecar.stores.hashing import static_template_hash, topology_id
    from sidecar.stores.registry import reset_store_registry, get_store_registry
    from sidecar.stores.topology_anchor import DeltaAnchorKey

    reset_store_registry()
    chat = object.__new__(LLMChat)
    chat.node_id = "1"
    chat.resolve_turn_ph_slot = lambda _ph, _msg: type(
        "S", (), {"content_hash": "content-abc"}
    )()
    chat.resolve_tool_consumer_slot = lambda _ph, _msg: None
    chat.resolve_llm_branch_slot = lambda _ph, _msg: None

    static = static_template_hash("task\n{turn_0_assistant}\n")
    topo = topology_id(static_hash=static, turn_count=1)
    LLMChat._shared_kv_cache_memory["1"] = {
        "static_template_hash": static,
        "topology_id": topo,
        "placeholder_info": {"turn_0_assistant": {"start": 10, "end": 15, "pf_span_id": "T0"}},
    }

    pool = get_store_registry().agent_anchors
    delta_key = DeltaAnchorKey(
        static_template_hash=static,
        topology_id=topo,
        ph_id="turn_0_assistant",
        ph_token_start=10,
        ph_token_end=15,
        pf_span_id="T0",
        content_hash="content-abc",
    )
    pool.put(
        node_id="1",
        message_key="msg",
        ph_id="turn_0_assistant",
        static_template_hash=static,
        upstream_hash="msg",
        ph_key_embedding=torch.zeros(1, 1, 20, 1),
        ph_value_embedding=torch.zeros(1, 1, 20, 1),
        ph_delta=torch.zeros(1, 1, 20, 1),
        ph_value_delta=torch.zeros(1, 1, 20, 1),
        pf_delta=torch.zeros(1, 1, 50, 1),
        pf_value_delta=torch.zeros(1, 1, 50, 1),
        delta_key=delta_key,
    )

    assert chat._kv_reuse_anchors_for_ph("turn_0_assistant", "msg", {}) == []

    chat._mark_turn_tool_delta_retry_eligible(
        node_id="1",
        message_key="msg",
        ph_id="turn_0_assistant",
        content_hash="content-abc",
    )
    anchored = chat._kv_reuse_anchors_for_ph("turn_0_assistant", "msg", {})
    assert len(anchored) == 1
    assert "1_ph_key_delta" in anchored[0]


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_turn_tool_passes_through_when_consumer_slot_present() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat

    chat = object.__new__(LLMChat)
    chat.node_id = "1"
    chat.resolve_turn_ph_slot = lambda _ph, _msg: type(
        "S", (), {"content_hash": "content-abc"}
    )()
    chat.resolve_llm_branch_slot = lambda _ph, _msg: type("S", (), {})()
    chat.resolve_tool_consumer_slot = lambda _ph, _msg: None
    chat._mark_turn_tool_delta_retry_eligible(
        node_id="1",
        message_key="msg",
        ph_id="turn_0_assistant",
        content_hash="content-abc",
    )

    assert chat._kv_reuse_anchors_for_ph("turn_0_assistant", "msg", {}) == []


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_offset_kv_cache_pair_passes_through_without_anchors() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat
    from KVCOMM.llm.kvcomm_engine import KVCOMMEngine

    chat = object.__new__(LLMChat)
    chat.node_id = "0"
    engine = KVCOMMEngine(chat)

    base_ph = _fake_cache(40)
    base_pf = _fake_cache(80)
    _ph, _pf, blended = engine.offset_kv_cache_pair(
        "turn_1_tool",
        "msg",
        "req",
        base_ph,
        base_pf,
        [],
    )
    assert blended is True


def test_requires_dense_for_tool_injection_on_consumer_node() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat

    chat = object.__new__(LLMChat)
    chat.node_id = "1"
    LLMChat._shared_kv_cache_memory["1"] = {
        "placeholder_info": {"agent_0_current": {"start": 10, "end": 20}},
    }
    assert chat._requires_dense_for_tool_injection()

    chat.node_id = "0"
    LLMChat._shared_kv_cache_memory["0"] = {
        "placeholder_info": {"user_question": {"start": 0, "end": 5}},
    }
    assert not chat._requires_dense_for_tool_injection()


def test_should_force_dense_when_unstable_or_pending() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat

    chat = object.__new__(LLMChat)
    chat.node_id = "1"
    LLMChat._shared_kv_cache_memory["1"] = {
        "placeholder_info": {"agent_0_current": {"start": 10, "end": 20}},
        "_consumer_tool_schema_stable": {"msg": True},
    }
    LLMChat.set_consumer_first_measure_dense_pending(False)
    chat.placeholders_missing_anchor_delta = lambda _uid, _msg: []
    assert not chat._should_force_dense_consumer_tool_schema(
        message="msg",
        request_uid="run-stable",
        full_tool_schema=True,
    )

    LLMChat.set_consumer_first_measure_dense_pending(True)
    assert chat._should_force_dense_consumer_tool_schema(
        message="msg",
        request_uid="run-pending",
        full_tool_schema=True,
    )
    LLMChat.set_consumer_first_measure_dense_pending(False)

    LLMChat._shared_kv_cache_memory["1"].pop("_consumer_tool_schema_stable", None)
    assert chat._should_force_dense_consumer_tool_schema(
        message="msg",
        request_uid="run-full-unstable",
        full_tool_schema=True,
    )

    chat.placeholders_missing_anchor_delta = lambda _uid, _msg: ["agent_0_current"]
    assert chat._should_force_dense_consumer_tool_schema(
        message="msg",
        request_uid="run-exec-missing",
        full_tool_schema=False,
    )
    chat.placeholders_missing_anchor_delta = lambda _uid, _msg: []
    assert not chat._should_force_dense_consumer_tool_schema(
        message="msg",
        request_uid="run-exec-ok",
        full_tool_schema=False,
    )
    LLMChat._shared_kv_cache_memory.pop("1", None)


def test_should_force_dense_only_when_anchor_delta_missing() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat

    chat = object.__new__(LLMChat)
    chat.node_id = "1"
    LLMChat._shared_kv_cache_memory["1"] = {
        "placeholder_info": {"agent_0_current": {"start": 10, "end": 20}},
    }
    chat.get_request_state = lambda _uid: type("S", (), {"anchor_dict": {}})()
    chat.placeholders_missing_anchor_delta = lambda _uid, _msg: ["agent_0_current"]
    LLMChat._shared_kv_cache_memory["1"]["_consumer_tool_schema_stable"] = {"msg": True}
    LLMChat.set_consumer_first_measure_dense_pending(False)
    assert chat._should_force_dense_consumer_tool_schema(
        message="msg",
        request_uid="run-1",
        full_tool_schema=False,
    )

    chat.placeholders_missing_anchor_delta = lambda _uid, _msg: []
    assert not chat._should_force_dense_consumer_tool_schema(
        message="msg",
        request_uid="run-2",
        full_tool_schema=False,
    )
    LLMChat._shared_kv_cache_memory["1"].pop("_consumer_tool_schema_stable", None)
    assert chat._should_force_dense_consumer_tool_schema(
        message="msg",
        request_uid="run-3",
        full_tool_schema=True,
    )
    LLMChat._shared_kv_cache_memory.pop("1", None)


def test_should_not_force_dense_when_schema_branch_pooled() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat
    from sidecar.stores.registry import get_store_registry, reset_store_registry

    reset_store_registry()
    chat = object.__new__(LLMChat)
    chat.node_id = "1"
    LLMChat._shared_kv_cache_memory["1"] = {
        "placeholder_info": {"agent_0_current": {"start": 10, "end": 20}},
    }
    chat.placeholders_missing_anchor_delta = lambda _uid, _msg: []
    LLMChat.set_consumer_first_measure_dense_pending(False)

    schema_hash = "abc123"
    deliverable = "deliv456"
    stores = get_store_registry()
    stores.tool_schema_branches.put(
        consumer_node_id="1",
        message_key="msg",
        schema_hash=schema_hash,
        deliverable_hash=deliverable,
        prefix_boundary_len=50,
        schema_token_len=10,
        absolute_kv=None,
        token_ids={},
    )

    assert not chat._should_force_dense_consumer_tool_schema(
        message="msg",
        request_uid="run-branch",
        full_tool_schema=True,
        tool_schema_hash=schema_hash,
        deliverable_hash=deliverable,
    )
    LLMChat._shared_kv_cache_memory.pop("1", None)
