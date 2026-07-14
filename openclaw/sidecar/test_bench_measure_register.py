"""Bench register transitions (warmup dense -> measure kv_reuse)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import sidecar.kvcomm_adapter as adapter
from sidecar.kvcomm_adapter import _note_agent0_bench_register


def _reset_register_globals() -> None:
    adapter._last_agent0_run_id = None
    adapter._consumer_stable_cleared_for_measure = False


def _fake_gpt_chat_module(*, shared: dict | None = None, pending: list[bool] | None = None) -> ModuleType:
    mod = ModuleType("KVCOMM.llm.gpt_chat")
    mock_llm = MagicMock()
    mock_llm._shared_kv_cache_memory = shared if shared is not None else {}

    def _clear() -> None:
        bucket = mock_llm._shared_kv_cache_memory.get("1")
        if isinstance(bucket, dict):
            bucket.pop("_consumer_tool_schema_stable", None)

    store = pending if pending is not None else []

    mock_llm.clear_consumer_tool_schema_stable_all = _clear
    mock_llm.set_consumer_first_measure_dense_pending = lambda v: store.append(bool(v))
    mock_llm.consumer_first_measure_dense_pending = lambda: bool(store and store[-1])
    mod.LLMChat = mock_llm
    return mod


def test_dense_warmup_then_kvreuse_measure_clears_consumer_stable() -> None:
    _reset_register_globals()
    shared = {"1": {"_consumer_tool_schema_stable": {"task": True}}}
    pending: list[bool] = []
    fake = _fake_gpt_chat_module(shared=shared, pending=pending)

    with patch.dict(sys.modules, {"KVCOMM.llm.gpt_chat": fake}):
        _note_agent0_bench_register("run-a", "dense_prefill")
        assert shared["1"].get("_consumer_tool_schema_stable")
        assert not pending

        _note_agent0_bench_register("run-b", "kv_reuse")
        assert not shared["1"].get("_consumer_tool_schema_stable")
        assert pending == [True]
        assert adapter._last_agent0_run_id == "run-b"


def test_first_kvreuse_register_without_warmup_does_not_clear() -> None:
    _reset_register_globals()
    pending: list[bool] = []
    fake = _fake_gpt_chat_module(pending=pending)
    fake.LLMChat.clear_consumer_tool_schema_stable_all = MagicMock()

    with patch.dict(sys.modules, {"KVCOMM.llm.gpt_chat": fake}):
        _note_agent0_bench_register("run-only", "kv_reuse")
        fake.LLMChat.clear_consumer_tool_schema_stable_all.assert_not_called()
        assert not pending
        assert adapter._last_agent0_run_id == "run-only"


def test_each_measure_run_clears_consumer_stable() -> None:
    _reset_register_globals()
    shared = {"1": {"_consumer_tool_schema_stable": {"task": True}}}
    pending: list[bool] = []
    fake = _fake_gpt_chat_module(shared=shared, pending=pending)

    with patch.dict(sys.modules, {"KVCOMM.llm.gpt_chat": fake}), patch(
        "sidecar.kvcomm_adapter._purge_bench_tool_schema_branches"
    ) as purge:
        _note_agent0_bench_register("run-a", "dense_prefill")
        _note_agent0_bench_register("run-b", "kv_reuse")
        assert not shared["1"].get("_consumer_tool_schema_stable")
        assert pending == [True]
        purge.assert_not_called()

        shared["1"]["_consumer_tool_schema_stable"] = {"task": True}
        pending.clear()
        purge.reset_mock()
        _note_agent0_bench_register("run-c", "kv_reuse")
        assert not shared["1"].get("_consumer_tool_schema_stable")
        assert pending == [False]
        purge.assert_not_called()


def test_measure_register_retains_tool_schema_branches() -> None:
    _reset_register_globals()
    reset_store_registry = __import__(
        "sidecar.stores.registry", fromlist=["reset_store_registry"]
    ).reset_store_registry
    reset_store_registry()
    stores = __import__(
        "sidecar.stores.registry", fromlist=["get_store_registry"]
    ).get_store_registry()
    stores.tool_schema_branches.put(
        consumer_node_id="1",
        message_key="task",
        schema_hash="schema_warmup",
        deliverable_hash="deliv_warmup",
        prefix_boundary_len=50,
        schema_token_len=10,
        absolute_kv=None,
        token_ids={},
    )
    fake = _fake_gpt_chat_module()

    with patch.dict(sys.modules, {"KVCOMM.llm.gpt_chat": fake}):
        _note_agent0_bench_register("run-warmup", "dense_prefill")
        _note_agent0_bench_register("run-measure-1", "kv_reuse")
        hit = stores.tool_schema_branches.get("1", "task", "schema_warmup", "deliv_warmup")
        assert hit is not None


def test_register_pending_context_wires_agent0_note() -> None:
    _reset_register_globals()
    with patch("sidecar.kvcomm_adapter._note_agent0_bench_register") as note, patch(
        "sidecar.kvcomm_adapter.reset_bench_run_state"
    ):
        from sidecar.kvcomm_adapter import register_pending_context

        register_pending_context(
            {"run_id": "run-x", "agent_index": 0, "mode": "dense_prefill", "message_key": "task"}
        )
        note.assert_called_once_with("run-x", "dense_prefill")
