"""ToolSchemaBranchRegistry and deliverable fingerprint helpers."""

from __future__ import annotations

import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from sidecar.stores.hashing import (
    branch_fingerprint,
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
    )
    assert stores.tool_schema_branches.get("2", "m", "s", "d") is not None
    stores.purge_node("2")
    assert stores.tool_schema_branches.get("2", "m", "s", "d") is None
