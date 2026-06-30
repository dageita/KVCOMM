"""Unit tests for explicit ph↔pf span layout (no torch)."""

from __future__ import annotations

import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from sidecar.stores.prefix_spans import (
    build_layout_from_segments,
    frozen_span_count,
    legacy_forward_pf_kv_index,
    normalize_placeholder_entry,
    normalize_placeholder_info,
    ordered_placeholders,
    resolve_pf_kv_index,
    shared_prefix_token_len,
)


def _segments_static_plus_one_turn() -> list[tuple]:
    # T0 [0,10), ph turn_1_assistant [10,15), text_1 [15,20), ph turn_1_tool [20,25), text_2 [25,30)
    return [
        ("text", "static prefix", {}, 0, 10),
        ("placeholder", "turn_1_assistant", {}, 10, 15),
        ("text", "\n\n", {}, 15, 20),
        ("placeholder", "turn_1_tool", {}, 20, 25),
        ("text", "\n", {}, 25, 30),
    ]


def test_build_layout_links_ph_to_following_text_span() -> None:
    layout = build_layout_from_segments(_segments_static_plus_one_turn())
    assert layout.prefix_span_order == ["T0", "text_1", "text_2"]
    assert layout.placeholder_info["turn_1_assistant"]["pf_span_id"] == "text_1"
    assert layout.placeholder_info["turn_1_tool"]["pf_span_id"] == "text_2"
    assert layout.span_registry["text_1"]["kv_index"] == 1


def test_ordered_placeholders_follows_token_start() -> None:
    layout = build_layout_from_segments(_segments_static_plus_one_turn())
    ids = [ph_id for ph_id, _ in ordered_placeholders(layout.placeholder_info)]
    assert ids == ["turn_1_assistant", "turn_1_tool"]


def test_resolve_pf_kv_index_via_span_registry() -> None:
    layout = build_layout_from_segments(_segments_static_plus_one_turn())
    idx = resolve_pf_kv_index(
        ph_id="turn_1_tool",
        placeholder_info=layout.placeholder_info,
        span_registry=layout.span_registry,
    )
    assert idx == 2


def test_frozen_span_count_stops_on_hash_mismatch() -> None:
    layout = build_layout_from_segments(_segments_static_plus_one_turn())
    old_registry = dict(layout.span_registry)
    old_order = list(layout.prefix_span_order)

    assert frozen_span_count(old_registry, old_order, layout) == 3

    mutated = build_layout_from_segments(
        [
            ("text", "CHANGED static", {}, 0, 10),
            ("placeholder", "turn_1_assistant", {}, 10, 15),
            ("text", "\n\n", {}, 15, 20),
        ]
    )
    assert frozen_span_count(old_registry, old_order, mutated) == 0


def test_normalize_legacy_placeholder_list() -> None:
    entry = normalize_placeholder_entry([10, 20])
    assert entry["start"] == 10 and entry["end"] == 20
    enriched = normalize_placeholder_info({"agent_0_current": {"start": 1, "end": 2, "pf_span_id": "text_1"}})
    assert enriched["agent_0_current"]["pf_span_id"] == "text_1"


def test_legacy_forward_pf_index() -> None:
    ph_info = {
        "turn_1_assistant": {"start": 10, "end": 15, "pf_span_id": "text_1"},
        "turn_1_tool": {"start": 20, "end": 25, "pf_span_id": "text_2"},
    }
    assert legacy_forward_pf_kv_index(ph_info, "turn_1_assistant") == 1
    assert legacy_forward_pf_kv_index(ph_info, "turn_1_tool") == 2


def test_shared_prefix_token_len() -> None:
    old = [1, 2, 3, 4, 5]
    new = [1, 2, 3, 9, 8]
    assert shared_prefix_token_len(old, new) == 3
    assert shared_prefix_token_len(old, old) == 5
    assert shared_prefix_token_len([], new) == 0
