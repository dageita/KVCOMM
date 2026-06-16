#!/usr/bin/env bash
# Integration checks for Sidecar kv_reuse + run-clawbench fusion (no full HF load required).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MODULE_ROOT}/.." && pwd)"
BENCH_ROOT="${REPO_ROOT}/experiments/bench"

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${REPO_ROOT}:${MODULE_ROOT}:${PYTHONPATH:-}"

echo "=== validate-sidecar-kvreuse ==="

echo "[1/9] adapter parse"
"${PYTHON}" - <<'PY'
from sidecar.kvcomm_adapter import (
    parse_kvcomm_context,
    register_pending_context,
    resolve_request_mode,
    resolve_registered_context,
    consume_registered_context,
    SIDECAR_VERSION,
)
body = {
    "messages": [
        {"role": "user", "content": '<!--KVCOMM_META:{"run_id":"r","agent_index":2,"mode":"kv_reuse","message_key":"m"}-->\n{agent_1_current}'},
    ],
}
ctx = parse_kvcomm_context(body, {}, "dense_prefill")
assert ctx.agent_index == "2" and ctx.mode == "kv_reuse"
register_pending_context({"run_id": "run-a", "agent_index": 0, "mode": "dense_prefill", "message_key": "warm"})
register_pending_context({"run_id": "run-b", "agent_index": 0, "mode": "kv_reuse", "message_key": "meas"})
meta_body = {
    "messages": [
        {"role": "user", "content": '<!--KVCOMM_META:{"run_id":"run-b","agent_index":0,"mode":"kv_reuse"}-->\ntask'},
    ],
}
assert resolve_request_mode(meta_body, {}, "dense_prefill") == "kv_reuse"
assert resolve_registered_context(meta_body, {}).mode == "kv_reuse"
assert resolve_request_mode({"messages": [{"role": "user", "content": "hi"}]}, {}, "dense_prefill") == "kv_reuse"
consume_registered_context(meta_body, {})
print("  ok", SIDECAR_VERSION)
PY

echo "[2/9] anchor pool per-node + path normalization"
"${PYTHON}" - <<'PY'
from sidecar.kvcomm_adapter import KvcommEngineAdapter, _anchor_pool_key
from sidecar.openclaw_prefix import normalize_run_specific_paths, build_prefix_from_openclaw_messages

assert _anchor_pool_key("1", "task") == "1:task"
path_a = "/root/.openclaw/workspace/kvcomm-chain/t1-fs-quick-note/run-abc12345/notes/quick_note.md"
path_b = "/root/.openclaw/workspace/kvcomm-chain/t1-fs-quick-note/run-deadbeef/notes/quick_note.md"
norm_a = normalize_run_specific_paths(f"write at {path_a}")
norm_b = normalize_run_specific_paths(f"write at {path_b}")
assert norm_a == norm_b == "write at notes/quick_note.md"

adapter = KvcommEngineAdapter()
adapter._anchor_pool["1:msg"] = {"anchors": {"agent_0_current": {"m": {"1_ph_key_delta": 1}}}, "anchor_dict": {}}
adapter._anchor_pool["2:msg"] = {"anchors": {"agent_0_current": {"m": {"2_ph_key_delta": 1}}}, "anchor_dict": {}}
assert "1:msg" in adapter._anchor_pool and "2:msg" in adapter._anchor_pool

from sidecar.kvcomm_adapter import _prefix_missing_upstream_kv_placeholders

warmup_agent2 = (
    "Output from Agent 1 (Writer):\n\n"
    "1. Pick up dry cleaning on Thursday.\n"
    "File written to: notes/quick_note.md\n"
)
measure_agent2 = (
    "Output from Agent 1 (Writer):\n\n{agent_1_current}\n\n"
    "Your job (Agent 2 - Verifier): Read the workspace files.\n"
)
assert _prefix_missing_upstream_kv_placeholders(warmup_agent2, measure_agent2, 2) is True
assert _prefix_missing_upstream_kv_placeholders(measure_agent2, measure_agent2, 2) is False
print("  ok")
PY

echo "[3/9] openclaw prefix parser (A+E)"
"${PYTHON}" - <<'PY'
from sidecar.openclaw_prefix import (
    build_prefix_from_openclaw_messages,
    count_assistant_turns,
    PrefixOverflowError,
)

bench_user = (
    "User request:\nPick up dry cleaning.\n\n"
    "Your job (Agent 1 - Writer): Use edit tool.\n\n"
    "Output from Agent 0:\n\n{agent_0_current}\n"
)
messages = [
    {"role": "system", "content": "You are one agent.\n[tools json omitted]"},
    {
        "role": "user",
        "content": '<!--KVCOMM_META:{"run_id":"r"}-->\n' + bench_user,
    },
    {"role": "assistant", "content": "I'll write the note.", "tool_calls": [{"id": "1", "function": {"name": "write", "arguments": "{}"}}]},
    {"role": "tool", "content": "wrote quick_note.md", "name": "write"},
]
result = build_prefix_from_openclaw_messages(
    messages,
    bench_user_prompt=bench_user,
    clawbench_role="You are one agent.",
    task_profile="clawbench",
)
assert result.use_openclaw is True
assert "{agent_0_current}" in result.user_template
assert "{turn_1_assistant}" in result.user_template
assert "turn_1_assistant" in result.turn_content
assert count_assistant_turns(messages) == 1
import os
os.environ["KVCOMM_PREFIX_MAX_TOKENS"] = "300"
huge_bench = bench_user + ("x" * 5000)
try:
    build_prefix_from_openclaw_messages(
        messages,
        bench_user_prompt=huge_bench,
        clawbench_role="You are one agent.",
        task_profile="clawbench",
    )
    raise SystemExit("expected PrefixOverflowError")
except PrefixOverflowError:
    pass
finally:
    os.environ.pop("KVCOMM_PREFIX_MAX_TOKENS", None)
print("  ok")
PY

echo "[4/9] runtime configure + release shim"
"${PYTHON}" - <<'PY'
import os
from sidecar.kvcomm_adapter import configure_hf_engine, release_adapter, engine_loaded, _engine_enabled

os.environ.pop("KVCOMM_HF_MODEL", None)
os.environ.pop("KVCOMM_HF_MODEL_PATH", None)
assert _engine_enabled() is False
cfg = configure_hf_engine({"hf_model": "/models/Qwen3-32B", "hf_device": "2,3,4"})
assert cfg["engine_enabled"] is True
assert cfg["engine_loaded"] is False
assert engine_loaded() is False
result = release_adapter()
assert result["released"] is False
print("  ok")
PY

echo "[5/9] openai SSE stream shim"
"${PYTHON}" - <<'PY'
from sidecar.server import _completion_to_sse_body
payload = {
    "id": "chatcmpl-test",
    "model": "kvcomm/Qwen3-32B",
    "created": 1,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Ω"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
sse = _completion_to_sse_body(payload, include_usage=True)
assert "chat.completion.chunk" in sse
assert "Ω" in sse
assert "finish_reason" in sse
assert sse.strip().endswith("data: [DONE]")
print("  ok")
PY

echo "[6/9] template kv_reuse"
node -e "
import { renderTemplateKvReuse } from '${BENCH_ROOT}/lib/template.mjs';
const out = renderTemplateKvReuse('{{task_body}} {{agent_0_current}}', { task_body: 'T', agent_0_current: 'X' });
if (!out.includes('{agent_0_current}') || out.includes('X')) process.exit(1);
console.log('  ok');
"

echo "[7/9] profile clawbench-capability-sidecar"
test -f "${MODULE_ROOT}/config/openclaw.kvcomm.clawbench-capability-sidecar.json"
"${PYTHON}" "${SCRIPT_DIR}/apply-openclaw-profile.py" \
  "${MODULE_ROOT}/config/openclaw.kvcomm.clawbench-capability-sidecar.json" \
  /tmp/openclaw-kvcomm-test.json clawbench-capability-sidecar
"${PYTHON}" -c "
import json
d=json.load(open('/tmp/openclaw-kvcomm-test.json'))
assert d['models']['providers']['kvcomm']['baseUrl'].startswith('http')
assert d['agents']['defaults']['model']['primary']=='kvcomm/Qwen3-32B'
print('  ok')
"

echo "[8/9] bench dry-run (tier0 copy + clawbench)"
node "${BENCH_ROOT}/drivers/run-o0-pre-chain.mjs" --dry-run \
  --inference-mode kv_reuse --inference-backend kvcomm_sidecar --task-id micro-001 >/dev/null
node "${BENCH_ROOT}/drivers/run-clawbench-chain.mjs" --dry-run \
  --inference-mode kv_reuse --inference-backend kvcomm_sidecar --task-id t1-fs-quick-note >/dev/null
echo "  ok"

echo "[9/9] sidecar health (optional)"
SIDECAR_URL="${KVCOMM_SIDECAR_URL:-http://127.0.0.1:8100}"
if curl -sf --max-time 3 "${SIDECAR_URL}/health" >/dev/null 2>&1; then
  health="$(curl -sf "${SIDECAR_URL}/health")"
  echo "  sidecar: ${health}"
  if echo "${health}" | grep -q stub_forward; then
    echo "  note: kv_reuse_engine=stub_forward — set KVCOMM_HF_MODEL + full requirements for true kv_reuse"
  fi
else
  echo "  skip (sidecar not running)"
fi

echo "=== validate-sidecar-kvreuse OK ==="
