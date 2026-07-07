"""Resolve upstream slots must not reuse producer KV on downstream consumer nodes."""

from __future__ import annotations

import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from sidecar.stores.registry import reset_store_registry, get_store_registry
from sidecar.stores.hashing import sha256_text


def test_downstream_node_uses_consumer_not_producer() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat

    reset_store_registry()
    stores = get_store_registry()
    content_hash = sha256_text("upstream text")
    stores.upstream_agent_slots.put_producer(
        producer_node_id="0",
        message_key="msg",
        ph_id="agent_0_current",
        content_hash=content_hash,
        absolute_kv={"kind": "producer"},
        token_ids={"input_ids": [[1, 2, 3]]},
        prefix_token_len=10,
    )
    stores.upstream_agent_slots.put_consumer(
        consumer_node_id="1",
        message_key="msg",
        ph_id="agent_0_current",
        content_hash=content_hash,
        absolute_kv={"kind": "consumer"},
        token_ids={"input_ids": [[1, 2, 3]]},
        upstream_node_id="0",
        slot_token_start=5,
    )

    LLMChat._shared_kv_cache_memory["0"] = {
        "response_ids": {"msg": [{"input_ids": [[1, 2, 3]]}]},
    }
    LLMChat._shared_kv_cache_memory["1"] = {
        "placeholder_info": {"agent_0_current": {"start": 5, "end": 15}},
    }

    chat = object.__new__(LLMChat)
    chat.node_id = "1"
    chat.tokenizer = type(
        "Tok",
        (),
        {"decode": staticmethod(lambda ids, skip_special_tokens=True: "upstream text")},
    )()

    slot = chat.resolve_upstream_agent_slot("agent_0_current", "msg")
    assert slot is not None
    assert slot.materialization == "consumer_contextual"
    assert slot.absolute_kv == {"kind": "consumer"}

    chat.node_id = "0"
    prod = chat.resolve_upstream_agent_slot("agent_0_current", "msg")
    assert prod is not None
    assert prod.materialization == "producer_contextual"
    assert prod.absolute_kv == {"kind": "producer"}

    LLMChat._shared_kv_cache_memory.clear()


def test_consumer_slot_rejected_when_placeholder_start_mismatch() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat

    reset_store_registry()
    stores = get_store_registry()
    content_hash = sha256_text("upstream text")
    stores.upstream_agent_slots.put_consumer(
        consumer_node_id="1",
        message_key="msg",
        ph_id="agent_0_current",
        content_hash=content_hash,
        absolute_kv={"kind": "consumer"},
        token_ids={"input_ids": [[1, 2, 3]]},
        upstream_node_id="0",
        slot_token_start=5,
    )

    LLMChat._shared_kv_cache_memory["0"] = {
        "response_ids": {"msg": [{"input_ids": [[1, 2, 3]]}]},
    }
    LLMChat._shared_kv_cache_memory["1"] = {
        "placeholder_info": {"agent_0_current": {"start": 99, "end": 109}},
    }

    chat = object.__new__(LLMChat)
    chat.node_id = "1"
    chat.tokenizer = type(
        "Tok",
        (),
        {"decode": staticmethod(lambda ids, skip_special_tokens=True: "upstream text")},
    )()

    assert chat._resolve_upstream_consumer_slot("agent_0_current", "msg") is None
    LLMChat._shared_kv_cache_memory.clear()


def test_upstream_response_token_ids_reads_producer_bucket() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat

    LLMChat._shared_kv_cache_memory["0"] = {
        "response_ids": {"msg": [{"input_ids": [[11, 12]]}]},
    }
    chat = object.__new__(LLMChat)
    entry = chat._upstream_response_token_ids("0", "msg")
    assert entry == {"input_ids": [[11, 12]]}
    LLMChat._shared_kv_cache_memory.clear()


def test_upstream_content_hash_uses_producer_response_not_stale_meta() -> None:
    from KVCOMM.llm.gpt_chat import LLMChat

    LLMChat._shared_kv_cache_memory["0"] = {
        "response_ids": {"msg": [{"input_ids": [[1, 2]]}]},
    }
    chat = object.__new__(LLMChat)
    chat.tokenizer = type(
        "Tok",
        (),
        {"decode": staticmethod(lambda ids, skip_special_tokens=True: "measure-run text")},
    )()
    h = chat._upstream_agent_content_hash("agent_0_current", "msg")
    assert h == sha256_text("measure-run text")
    LLMChat._shared_kv_cache_memory.clear()
