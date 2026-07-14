"""ToolSchemaBranchRegistry and deliverable fingerprint helpers."""

from __future__ import annotations

import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from sidecar.stores.hashing import (
    branch_fingerprint,
    prefix_boundary_token_fingerprint,
    producer_turn_branch_fingerprint,
    tool_deliverable_fingerprint,
    tool_schema_hash,
)
from sidecar.stores.registry import get_store_registry, reset_store_registry
from sidecar.stores.tool_schema_branch import ToolSchemaBranchRegistry


def test_tool_schema_hash_stable() -> None:
    text = "<tools>\n{\"name\": \"read\"}\n</tools>"
    other = "<tools>\n{\"name\": \"write\"}\n</tools>"
    assert tool_schema_hash(text) == tool_schema_hash(text)
    assert tool_schema_hash(text) != tool_schema_hash(other)


def test_deliverable_fingerprint_normalizes_paths() -> None:
    a = tool_deliverable_fingerprint(
        task_id="t2-fs-find-that-thing",
        upstream_text="Copy /workspace/run-abcdef123456/Documents/file.xlsx to Desktop/out.xlsx",
    )
    b = tool_deliverable_fingerprint(
        task_id="t2-fs-find-that-thing",
        upstream_text="Copy /workspace/run-fedcba654321/Documents/file.xlsx to Desktop/out.xlsx",
    )
    assert a == b


def test_branch_fingerprint_pairs_schema_and_deliverable() -> None:
    s = tool_schema_hash("schema")
    d = tool_deliverable_fingerprint(task_id="task", upstream_text="paths")
    assert branch_fingerprint(schema_hash=s, deliverable_hash=d) != branch_fingerprint(
        schema_hash=s,
        deliverable_hash="other",
    )


def test_registry_key_and_lookup() -> None:
    reg = ToolSchemaBranchRegistry()
    reg.put(
        consumer_node_id="1",
        message_key="msg",
        schema_hash="schema_a",
        deliverable_hash="deliv_a",
        prefix_boundary_len=100,
        schema_token_len=42,
        absolute_kv={"kv": 1},
        token_ids={"input_ids": [[1, 2]]},
        tool_call_hash="tc",
    )
    hit = reg.get("1", "msg", "schema_a", "deliv_a")
    assert hit is not None
    assert hit.schema_token_len == 42
    assert hit.prefix_boundary_len == 100
    assert reg.get("1", "msg", "schema_a", "deliv_b") is None
    fallback = reg.find_for_schema("1", "msg", "schema_a")
    assert fallback is not None
    assert fallback.deliverable_hash == "deliv_a"


def test_schema_fallback_lookup_after_deliverable_drift() -> None:
    """Cross-run reuse: schema-only lookup when deliverable hash differs."""
    reg = ToolSchemaBranchRegistry()
    reg.put(
        consumer_node_id="1",
        message_key="msg",
        schema_hash="schema_a",
        deliverable_hash="deliv_warmup",
        prefix_boundary_len=100,
        schema_token_len=42,
        absolute_kv={"kv": 1},
        token_ids={"input_ids": [[1, 2]]},
        prefix_token_fingerprint="fp_stable",
    )
    assert reg.get("1", "msg", "schema_a", "deliv_measure_new") is None
    fallback = reg.find_for_schema("1", "msg", "schema_a")
    assert fallback is not None
    assert fallback.deliverable_hash == "deliv_warmup"
    assert fallback.prefix_token_fingerprint == "fp_stable"


def test_registry_wired_in_store_registry() -> None:
    reset_store_registry()
    stores = get_store_registry()
    stores.tool_schema_branches.put(
        consumer_node_id="2",
        message_key="m",
        schema_hash="s",
        deliverable_hash="d",
        prefix_boundary_len=10,
        schema_token_len=5,
        absolute_kv=None,
        token_ids={},
        prefix_token_fingerprint="abc123",
    )
    hit = stores.tool_schema_branches.get("2", "m", "s", "d")
    assert hit is not None
    assert hit.prefix_token_fingerprint == "abc123"
    stores.purge_node("2")
    assert stores.tool_schema_branches.get("2", "m", "s", "d") is None


def test_prefix_boundary_token_fingerprint_stable() -> None:
    fp_a = prefix_boundary_token_fingerprint({"input_ids": [[1, 2, 3, 4, 5]]}, 3)
    fp_b = prefix_boundary_token_fingerprint({"input_ids": [[1, 2, 3, 9, 9]]}, 3)
    fp_c = prefix_boundary_token_fingerprint({"input_ids": [[1, 2, 4, 4, 5]]}, 3)
    assert fp_a == fp_b
    assert fp_a != fp_c


def test_prefix_fp_mismatch_falls_through_to_suffix_prefill_not_raise() -> None:
    """Regression: prefix_fp mismatch must not raise (that forced dense rematerialize)."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "KVCOMM" / "llm" / "gpt_chat.py"
    text = src.read_text(encoding="utf-8")
    assert "tool schema branch prefix_fp mismatch" not in text
    assert "fp_ok={} -> suffix prefill" in text
    # Ensure the mismatch path no longer raises RuntimeError for fingerprint alone.
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_build_kv_reuse_generation_inputs":
            continue
        body_src = ast.get_source_segment(text, node) or ""
        assert "raise RuntimeError(\n                        \"tool schema branch prefix_fp mismatch" not in body_src
        assert "-> suffix prefill" in body_src
        break
    else:
        raise AssertionError("_build_kv_reuse_generation_inputs not found")


def test_producer_turn_branch_fingerprint_separates_turns() -> None:
    turn0 = producer_turn_branch_fingerprint(
        task_id="t2-msg-summarize-thread",
        turn_index=0,
        tool_names=["exec", "read"],
    )
    turn1 = producer_turn_branch_fingerprint(
        task_id="t2-msg-summarize-thread",
        turn_index=1,
        tool_names=["exec", "read"],
    )
    assert turn0 != turn1


def test_purge_bench_tool_schema_branches_clears_nodes() -> None:
    from sidecar.kvcomm_adapter import _purge_bench_tool_schema_branches

    reset_store_registry()
    stores = get_store_registry()
    for node in ("0", "1", "2"):
        stores.tool_schema_branches.put(
            consumer_node_id=node,
            message_key="m",
            schema_hash="s",
            deliverable_hash=f"d-{node}",
            prefix_boundary_len=10,
            schema_token_len=5,
            absolute_kv=None,
            token_ids={},
        )
    _purge_bench_tool_schema_branches()
    for node in ("0", "1", "2"):
        assert stores.tool_schema_branches.get(node, "m", "s", f"d-{node}") is None
