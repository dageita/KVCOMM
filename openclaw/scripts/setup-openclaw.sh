#!/usr/bin/env bash
# Apply KVCOMM OpenClaw profile (dense direct vLLM or sidecar kvreuse path).
# Merges into existing openclaw.json — does not wipe meta/plugins/wizard.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${MODULE_ROOT}/config"

PROFILE="${1:-dense}"
STATE_DIR="${OPENCLAW_STATE_DIR:-${HOME}/.openclaw}"
TARGET="${STATE_DIR}/openclaw.json"

case "${PROFILE}" in
  dense)
    SRC="${CONFIG_DIR}/openclaw.kvcomm.dense.json"
    ;;
  sidecar|kv_reuse|kvreuse)
    SRC="${CONFIG_DIR}/openclaw.kvcomm.sidecar.json"
    PROFILE="sidecar"
    ;;
  clawbench-capability|clawbench)
    SRC="${CONFIG_DIR}/openclaw.kvcomm.clawbench-capability.json"
    PROFILE="clawbench-capability"
    ;;
  clawbench-capability-sidecar|clawbench-sidecar|capability-sidecar)
    SRC="${CONFIG_DIR}/openclaw.kvcomm.clawbench-capability-sidecar.json"
    PROFILE="clawbench-capability-sidecar"
    ;;
  dual|both)
    SRC="${CONFIG_DIR}/openclaw.kvcomm.dual.json"
    PROFILE="dual"
    ;;
  *)
    echo "Usage: $0 [dense|sidecar|dual|clawbench-capability|clawbench-capability-sidecar]" >&2
    exit 1
    ;;
esac

if [[ ! -f "${SRC}" ]]; then
  echo "Missing config template: ${SRC}" >&2
  exit 1
fi

mkdir -p "${STATE_DIR}"
python3 "${SCRIPT_DIR}/apply-openclaw-profile.py" "${SRC}" "${TARGET}" "${PROFILE}"

# Optional overrides from env
if [[ -n "${KVCOMM_VLLM_UPSTREAM:-}" && ("${PROFILE}" == "dense" || "${PROFILE}" == "dual") ]]; then
  python3 - <<PY
import json, os
path = "${TARGET}"
url = os.environ["KVCOMM_VLLM_UPSTREAM"].rstrip("/")
if not url.endswith("/v1"):
    url = url + "/v1"
with open(path) as f:
    data = json.load(f)
data["models"]["providers"]["vllm"]["baseUrl"] = url
with open(path, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
  echo "[setup] set vllm baseUrl from KVCOMM_VLLM_UPSTREAM"
fi

if [[ -n "${KVCOMM_SIDECAR_URL:-}" && ("${PROFILE}" == "sidecar" || "${PROFILE}" == "dual" || "${PROFILE}" == "clawbench-capability-sidecar") ]]; then
  python3 - <<PY
import json, os
path = "${TARGET}"
url = os.environ["KVCOMM_SIDECAR_URL"].rstrip("/")
if not url.endswith("/v1"):
    url = url + "/v1"
with open(path) as f:
    data = json.load(f)
data["models"]["providers"]["kvcomm"]["baseUrl"] = url
with open(path, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
  echo "[setup] set kvcomm sidecar baseUrl from KVCOMM_SIDECAR_URL"
fi

echo "[setup] applied profile=${PROFILE} -> ${TARGET}"
echo "[setup] note: ClawBench native tools.profile=coding is preserved when already set"
echo "[setup] for ClawBench smoke: bash /src/clawbench/scripts/setup_vllm_clawbench.sh (once), then KVCOMM setup is safe to re-run"
echo "[setup] validate: openclaw doctor"
echo "[setup] restart Gateway: openclaw gateway run"
echo "[setup] optional pre-bench cleanup: ./scripts/clean-bench-sessions.sh"
