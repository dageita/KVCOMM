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
    slot = SimpleNamespace(absolute_kv=_fake_cache(5), kv_ref=None, slot_kind="tool")
    chat.resolve_turn_ph_slot = lambda ph_id, _msg: slot if ph_id == "turn_1_tool" else None
    chat.resolve_upstream_agent_slot = lambda _ph_id, _msg: None

    LLMChat._shared_kv_cache_memory["1"] = {
        "response": {"msg": [_fake_cache(40)]},
    }
    stale = {"msg": {"2_ph_key_delta": torch.zeros(1)}}
    anchors = {"turn_1_tool": stale, "agent_1_current": stale, "agent_2_current": stale}

    assert chat._kv_reuse_anchors_for_ph("turn_1_tool", "msg", anchors) == []
    assert chat._kv_reuse_anchors_for_ph("agent_1_current", "msg", anchors) == []
    assert len(chat._kv_reuse_anchors_for_ph("agent_2_current", "msg", anchors)) == 1


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
