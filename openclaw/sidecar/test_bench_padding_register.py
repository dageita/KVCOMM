"""Register-time bench padding normalization."""

from __future__ import annotations

import os

from sidecar.kvcomm_adapter import (
    _strip_clawbench_role_padding,
    register_pending_context,
    reset_bench_run_state,
)


def test_strip_clawbench_role_padding_removes_long_block() -> None:
    long_role = (
        "You are one agent in a fixed multi-agent chain.\n\n"
        "Long-context bench context (stable system segment for KV prefix tests):\n"
        "- filler line\n"
    )
    stripped = _strip_clawbench_role_padding(long_role)
    assert "Long-context bench context" not in stripped
    assert stripped.endswith("fixed multi-agent chain.")


def test_register_strips_long_role_when_bench_padding_off() -> None:
    os.environ["KVCOMM_BENCH_PADDING"] = "0"
    reset_bench_run_state()
    long_role = (
        "You are one agent in a fixed multi-agent chain.\n\n"
        "Long-context bench context (stable system segment for KV prefix tests):\n"
        "- filler line\n"
    )
    ctx = register_pending_context(
        {
            "run_id": "test-run",
            "agent_index": 0,
            "task_profile": "clawbench",
            "system_prompt": long_role,
            "bench_padding": False,
        }
    )
    assert "Long-context bench context" not in ctx.system_prompt
    assert os.environ.get("KVCOMM_BENCH_PADDING") == "0"
