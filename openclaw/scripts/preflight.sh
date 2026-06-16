#!/usr/bin/env bash
# Preflight for KVCOMM OpenClaw bench (Gateway + optional sidecar + vLLM).
set -euo pipefail

GATEWAY_URL="${OPENCLAW_GATEWAY_URL:-ws://127.0.0.1:18789}"
GATEWAY_HTTP="${GATEWAY_URL/ws:/http:}"
GATEWAY_HTTP="${GATEWAY_HTTP/wss:/https:}"
STATE_DIR="${OPENCLAW_STATE_DIR:-${HOME}/.openclaw}"
INFERENCE_BACKEND="${KVCOMM_INFERENCE_BACKEND:-vllm_direct}"
SIDECAR_URL="${KVCOMM_SIDECAR_URL:-http://127.0.0.1:8100}"
VLLM_URL="${KVCOMM_VLLM_UPSTREAM:-http://127.0.0.1:8001/v1}"

fail=0

check_http() {
  local name="$1" url="$2"
  if curl -sf --max-time 5 "${url}" >/dev/null 2>&1; then
    echo "[ok] ${name}: ${url}"
  else
    echo "[fail] ${name} unreachable: ${url}" >&2
    fail=1
  fi
}

echo "=== KVCOMM OpenClaw preflight ==="

if [[ -f "${STATE_DIR}/openclaw.json" ]]; then
  echo "[ok] openclaw.json: ${STATE_DIR}/openclaw.json"
  if ! python3 -c "
import json, sys
d=json.load(open('${STATE_DIR}/openclaw.json'))
allow=d.get('gateway',{}).get('tools',{}).get('allow',[])
sys.exit(0 if 'sessions_spawn' in allow else 1)
" 2>/dev/null; then
    echo "[fail] gateway.tools.allow must include sessions_spawn" >&2
    fail=1
  else
    echo "[ok] sessions_spawn allowed in gateway.tools.allow"
  fi
else
  echo "[fail] missing ${STATE_DIR}/openclaw.json — run scripts/setup-openclaw.sh" >&2
  fail=1
fi

# Gateway health (HTTP probe if available)
if curl -sf --max-time 5 "${GATEWAY_HTTP}/health" >/dev/null 2>&1; then
  echo "[ok] gateway health: ${GATEWAY_HTTP}/health"
else
  echo "[warn] gateway health probe failed (is 'openclaw gateway run' active?)" >&2
fi

if [[ "${INFERENCE_BACKEND}" == "kvcomm_sidecar" ]]; then
  check_http "kvcomm sidecar" "${SIDECAR_URL}/health"
  engine="$(curl -sf "${SIDECAR_URL}/health" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('kv_reuse_engine',''))" 2>/dev/null || echo "")"
  if [[ -z "${engine}" || "${engine}" == "stub_forward" ]]; then
    echo "[fail] sidecar kv_reuse_engine is stub_forward — set KVCOMM_HF_MODEL and restart sidecar" >&2
    fail=1
  else
    echo "[ok] sidecar kv_reuse_engine: ${engine}"
    hf_device="$(curl -sf "${SIDECAR_URL}/health" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('hf_device',''))" 2>/dev/null || echo "")"
    if [[ -n "${hf_device}" ]]; then
      echo "[ok] sidecar hf_device: ${hf_device}"
    fi
    "${KVCOMM_PYTHON:-/opt/conda/envs/kvcomm/bin/python3}" -c "
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
if 'qwen3' not in CONFIG_MAPPING:
    raise SystemExit('transformers ' + transformers.__version__ + ' missing qwen3; pip install \"transformers>=4.51.0,<4.52\"')
print('[ok] transformers', transformers.__version__, 'supports qwen3')
" 2>/dev/null || {
      echo "[warn] sidecar Python env may lack transformers>=4.51.0 for Qwen3 (pip install 'transformers>=4.51.0,<4.52')" >&2
    }
  fi
  VLLM_URL="$(curl -sf "${SIDECAR_URL}/health" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('upstream',''))" 2>/dev/null || echo "")"
  if [[ -n "${VLLM_URL}" ]]; then
    check_http "vllm upstream (via sidecar)" "${VLLM_URL%/}/models"
  fi
else
  check_http "vllm" "${VLLM_URL%/}/models"
fi

if [[ "${fail}" -ne 0 ]]; then
  echo "=== preflight FAILED ===" >&2
  exit 1
fi

echo "=== preflight OK ==="
