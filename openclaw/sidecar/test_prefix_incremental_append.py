"""Tests for incremental prefix append merge logic."""

from __future__ import annotations

from sidecar.openclaw_prefix import (
    build_user_template_with_turns,
    merge_turn_segment_into_user_template,
    turn_segment_template,
)


def test_append_extends_committed_template() -> None:
    stored = "static\n{agent_0_current}\nYour job: fix"
    segment = turn_segment_template(1)
    merged = merge_turn_segment_into_user_template(stored, segment)
    assert merged.endswith("{turn_1_tool}")
    assert merged.startswith("static")
    assert "Your job: fix" in merged


def test_append_idempotent_when_segment_present() -> None:
    stored = "static\n\n{turn_1_assistant}\n\n{turn_1_tool}\n"
    segment = turn_segment_template(1)
    assert merge_turn_segment_into_user_template(stored, segment) == stored.strip()


def test_openclaw_expected_not_used_when_stored_present() -> None:
    stored = "committed-static-v1"
    expected = "openclaw-static-v2\n{turn_1_assistant}\n\n{turn_1_tool}"
    segment = turn_segment_template(1)
    merged = merge_turn_segment_into_user_template(
        stored, segment, expected_user_template=expected
    )
    assert merged.startswith("committed-static-v1")
    assert "openclaw-static-v2" not in merged


def test_merge_matches_openclaw_cumulative_build() -> None:
    static = (
        "User request: fix pricing bug\n\n"
        "Your job (Agent 1 - Patcher): read pricing.py\n"
        "OpenClaw tool cwd: workspace\n"
    )
    for turn_count in (1, 2, 3):
        incremental = static.rstrip()
        for idx in range(1, turn_count + 1):
            incremental = merge_turn_segment_into_user_template(
                incremental, turn_segment_template(idx)
            )
        canonical = build_user_template_with_turns(static, turn_count)
        assert incremental == canonical


def test_merge_does_not_insert_extra_newline_before_segment() -> None:
    stored = "static tail"
    segment = turn_segment_template(1)
    merged = merge_turn_segment_into_user_template(stored, segment)
    assert "\n\n\n{turn_1_assistant}" not in merged
    assert merged.startswith("static tail\n\n{turn_1_assistant}")
