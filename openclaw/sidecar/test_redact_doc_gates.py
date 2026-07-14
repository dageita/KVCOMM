"""Tests for t2-priv-redact-doc gate helpers."""

from __future__ import annotations

import json
from pathlib import Path

from sidecar.bench_prompt_compose import REDACT_DOC_TASK_ID
from sidecar.openclaw_prefix import (
    build_redact_doc_writer_write_message,
    estimate_bench_text_tokens,
    is_valid_redact_doc_write_content,
    redact_contract_content,
    redact_doc_bench_canonical_text,
    redact_doc_bench_forced_generation_text,
    redact_doc_extractor_read_satisfied,
    REDACT_DOC_EXTRACTOR_CANONICAL,
    REDACT_DOC_VERIFIER_PASS_CANONICAL,
    REDACT_DOC_WRITER_DONE_CANONICAL,
    redact_doc_verifier_passed,
    redact_doc_write_satisfied,
    redact_doc_writer_write_needs_canonical_fallback,
    resolve_redact_doc_contract_text,
)

_CONTRACT_FIXTURE = Path(
    "/src/clawbench/tasks-public/assets/t2_priv_redact_doc/contract.txt"
).read_text(encoding="utf-8")


def _read_call(
    *,
    call_id: str = "call_read",
    body: str | None = None,
    path: str = "./contract.txt",
) -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": path}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": body or _CONTRACT_FIXTURE,
        },
    ]


def _write_call(
    path: str = "contract_redacted.txt",
    *,
    call_id: str = "call_write",
    body: str = "Successfully wrote 512 bytes to contract_redacted.txt",
    content: str | None = None,
) -> list[dict]:
    redacted = content or redact_contract_content(_CONTRACT_FIXTURE)
    write_args = json.dumps({"path": path, "content": redacted})
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": write_args,
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": body},
    ]


def test_redact_contract_content_removes_pii() -> None:
    redacted = redact_contract_content(_CONTRACT_FIXTURE)
    assert "Lin Park" not in redacted
    assert "[REDACTED NAME]" in redacted
    assert is_valid_redact_doc_write_content(redacted)


def test_redact_doc_extractor_read_satisfied_after_contract_read() -> None:
    messages = _read_call()
    assert redact_doc_extractor_read_satisfied(messages) is True


def test_redact_doc_write_satisfied_after_redacted_write() -> None:
    messages = _write_call()
    assert redact_doc_write_satisfied(messages) is True


def test_redact_doc_write_not_satisfied_for_edit_on_original() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_edit",
                    "type": "function",
                    "function": {
                        "name": "edit",
                        "arguments": json.dumps(
                            {
                                "path": "contract.txt",
                                "edits": [
                                    {"oldText": "Lin Park", "newText": "[REDACTED NAME]"}
                                ],
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_edit",
            "content": '{"status":"error","error":"Found 3 occurrences"}',
        },
    ]
    assert redact_doc_write_satisfied(messages) is False


def test_redact_doc_writer_fallback_for_edit_on_source() -> None:
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_edit",
                "type": "function",
                "function": {
                    "name": "edit",
                    "arguments": json.dumps(
                        {
                            "path": "contract.txt",
                            "edits": [
                                {"oldText": "Lin Park", "newText": "[REDACTED NAME]"}
                            ],
                        }
                    ),
                },
            }
        ],
    }
    assert redact_doc_writer_write_needs_canonical_fallback(message) is True


def test_build_redact_doc_writer_write_message(tmp_path) -> None:
    source = tmp_path / "contract.txt"
    source.write_text(_CONTRACT_FIXTURE, encoding="utf-8")
    message = build_redact_doc_writer_write_message(workspace_dir=str(tmp_path))
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "contract_redacted.txt"
    assert is_valid_redact_doc_write_content(args["content"])


def test_resolve_redact_doc_contract_text_from_workspace(tmp_path) -> None:
    source = tmp_path / "contract.txt"
    source.write_text(_CONTRACT_FIXTURE, encoding="utf-8")
    resolved = resolve_redact_doc_contract_text(workspace_dir=str(tmp_path))
    assert resolved == _CONTRACT_FIXTURE


def test_redact_doc_verifier_passed() -> None:
    messages = [
        {
            "role": "tool",
            "content": "PASS: redacted contract found at contract_redacted.txt, PII removed",
        }
    ]
    assert redact_doc_verifier_passed(messages) is True


def test_redact_doc_task_id_constant() -> None:
    assert REDACT_DOC_TASK_ID == "t2-priv-redact-doc"


def test_redact_doc_bench_canonical_text_fixed_length() -> None:
    assert redact_doc_bench_canonical_text("extractor_done") == REDACT_DOC_EXTRACTOR_CANONICAL
    assert redact_doc_bench_canonical_text("writer_done") == REDACT_DOC_WRITER_DONE_CANONICAL
    assert redact_doc_bench_canonical_text("verifier_done") == REDACT_DOC_VERIFIER_PASS_CANONICAL
    assert 40 <= estimate_bench_text_tokens(REDACT_DOC_EXTRACTOR_CANONICAL) <= 120
    assert estimate_bench_text_tokens(REDACT_DOC_WRITER_DONE_CANONICAL) <= 10
    assert estimate_bench_text_tokens(REDACT_DOC_VERIFIER_PASS_CANONICAL) <= 10


def test_redact_doc_bench_forced_generation_text_roundtrip() -> None:
    from sidecar.tool_bridge import openai_message_from_generation

    done_text = redact_doc_bench_forced_generation_text("writer_done")
    assert done_text == REDACT_DOC_WRITER_DONE_CANONICAL
    exec_text = redact_doc_bench_forced_generation_text("verifier_exec")
    exec_message = openai_message_from_generation(exec_text, task_profile="clawbench")
    assert exec_message.get("tool_calls")
    assert exec_message["tool_calls"][0]["function"]["name"] == "exec"
    write_text = redact_doc_bench_forced_generation_text("writer_write")
    write_message = openai_message_from_generation(write_text, task_profile="clawbench")
    assert write_message.get("tool_calls")
    write_args = json.loads(write_message["tool_calls"][0]["function"]["arguments"])
    assert write_args["path"] == "contract_redacted.txt"
    assert "[REDACTED NAME]" in write_args["content"]


def test_accumulate_reuse_breakdown() -> None:
    from sidecar.kvcomm_adapter import _accumulate_agent_metrics

    first = {
        "mode": "kv_reuse",
        "prefix_estimated_tokens": 615,
        "tool_injection_tokens": 759,
        "anchor_pooled_tokens": 113,
        "input_anchor_pooled_tokens": 0,
        "response_decode_tokens": 80,
        "input_reuse_kind": "input_cache",
    }
    second = {
        "mode": "kv_reuse",
        "prefix_estimated_tokens": 472,
        "tool_injection_tokens": 0,
        "anchor_pooled_tokens": 0,
        "input_anchor_pooled_tokens": 0,
        "response_decode_tokens": 0,
        "input_reuse_kind": "anchor_delta",
        "short_circuit": "redact_doc_writer_done_text",
    }
    merged = _accumulate_agent_metrics(first, second)
    assert merged["prefix_tokens_max"] == 615
    assert merged["tool_schema_tokens_sum"] == 759
    assert merged["response_anchor_tokens_sum"] == 113
    assert merged["response_decode_tokens_sum"] == 80
    assert merged["short_circuit_count"] == 1
    assert set(merged["input_reuse_kinds"]) == {"input_cache", "anchor_delta"}


def test_accumulate_inference_ttft_preserved_after_short_circuit() -> None:
    from sidecar.kvcomm_adapter import _accumulate_agent_metrics

    hf_turn = {
        "mode": "dense_prefill",
        "generation_ttft_ms": 512.0,
        "kvcomm_latency_ms": 198.0,
        "ttft_ms": 710.0,
        "response_decode_tokens": 22,
    }
    short_turn = {
        "mode": "dense_prefill",
        "ttft_ms": 0.0,
        "response_decode_tokens": 0,
        "short_circuit": "redact_doc_extractor_text",
    }
    merged = _accumulate_agent_metrics(hf_turn, short_turn)
    assert merged["generation_ttft_ms"] == 512.0
    assert merged["kvcomm_latency_ms"] == 198.0
    assert merged["ttft_ms"] == 710.0
    assert merged["response_decode_tokens_sum"] == 22
    assert merged["short_circuit_count"] == 1


def test_accumulate_inference_ttft_keeps_coherent_max_triple() -> None:
    """Long edit then short DONE must not mix max(ttft) with last(gen/kvcomm)."""
    from sidecar.kvcomm_adapter import _accumulate_agent_metrics

    edit_turn = {
        "mode": "dense_prefill",
        "generation_ttft_ms": 800.0,
        "kvcomm_latency_ms": 500.0,
        "ttft_ms": 1300.0,
        "response_decode_tokens": 179,
    }
    done_turn = {
        "mode": "kv_reuse",
        "generation_ttft_ms": 270.0,
        "kvcomm_latency_ms": 280.0,
        "ttft_ms": 550.0,
        "response_decode_tokens": 7,
    }
    merged = _accumulate_agent_metrics(edit_turn, done_turn)
    assert merged["ttft_ms"] == 1300.0
    assert merged["generation_ttft_ms"] == 800.0
    assert merged["kvcomm_latency_ms"] == 500.0
    # Coherent: sidecar_ttft ≈ gen_ttft + kvcomm_ms
    assert abs(merged["ttft_ms"] - merged["generation_ttft_ms"] - merged["kvcomm_latency_ms"]) < 1.0


def test_accumulate_inference_ttft_all_short_circuit_stays_empty() -> None:
    from sidecar.kvcomm_adapter import _accumulate_agent_metrics

    first = {
        "mode": "dense_prefill",
        "ttft_ms": 0.0,
        "short_circuit": "redact_doc_writer_write",
    }
    second = {
        "mode": "dense_prefill",
        "ttft_ms": 0.0,
        "short_circuit": "redact_doc_writer_done_text",
    }
    merged = _accumulate_agent_metrics(first, second)
    assert merged.get("generation_ttft_ms") is None
    assert merged.get("kvcomm_latency_ms") is None
    assert merged["ttft_ms"] == 0.0
    assert merged["short_circuit_count"] == 2
