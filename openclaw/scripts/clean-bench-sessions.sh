#!/usr/bin/env bash
# Purge bloated main-agent session transcripts before a long KVCOMM bench run.
# Recommended: stop Gateway first (openclaw gateway stop) to avoid races.
set -euo pipefail

STATE_DIR="${OPENCLAW_STATE_DIR:-${HOME}/.openclaw}"
SESSIONS_DIR="${STATE_DIR}/agents/main/sessions"
TS="$(date +%Y%m%d%H%M%S)"
BACKUP="${SESSIONS_DIR}.bak.${TS}"

if [[ ! -d "${SESSIONS_DIR}" ]]; then
  echo "[clean] no sessions dir: ${SESSIONS_DIR}"
  exit 0
fi

cp -a "${SESSIONS_DIR}" "${BACKUP}"
echo "[clean] backup -> ${BACKUP}"

export SESSIONS_DIR BACKUP
python3 <<'PY'
import json
import os
from pathlib import Path

sessions_dir = Path(os.environ["SESSIONS_DIR"])
store_path = sessions_dir / "sessions.json"
store: dict = {}
if store_path.exists():
    raw = json.loads(store_path.read_text(encoding="utf-8"))
    store = raw if isinstance(raw, dict) else {}

removed_ids: set[str] = set()
for key in list(store.keys()):
    entry = store.get(key) or {}
    sid = entry.get("sessionId")
    if isinstance(sid, str) and sid:
        removed_ids.add(sid)
    del store[key]

for sid in removed_ids:
    for name in (
        f"{sid}.jsonl",
        f"{sid}.trajectory.jsonl",
        f"{sid}.trajectory-path.json",
    ):
        path = sessions_dir / name
        if path.exists():
            path.unlink()

for pattern in ("*.jsonl", "*.trajectory-path.json"):
    for path in sessions_dir.glob(pattern):
        path.unlink()

store_path.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(
    f"[clean] cleared {len(removed_ids)} session store entries and orphan transcripts "
    f"under {sessions_dir}"
)
PY

echo "[clean] done — cold-restart Gateway before bench if it was running"
