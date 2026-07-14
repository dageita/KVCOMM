"""Content and topology hashing helpers for KVCOMM store keys."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

_TURN_PH_RE = re.compile(r"\{turn_\d+_(?:assistant|tool)\}")


def sha256_text(text: str) -> str:
    normalized = (text or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def short_hash(text: str, *, length: int = 16) -> str:
    return sha256_text(text)[:length]


def static_template_hash(user_template: str) -> str:
    """Hash static portion of user template (strip turn placeholders)."""
    static = _TURN_PH_RE.sub("", user_template or "")
    return sha256_text(static.strip())


def topology_id(*, static_hash: str, turn_count: int) -> str:
    return f"static:{static_hash}:turns:{int(turn_count)}"


def tool_schema_hash(tool_injection_text: str) -> str:
    """Stable hash for generation-boundary tool schema injection text."""
    return sha256_text(tool_injection_text or "")


def tool_deliverable_fingerprint(*, task_id: str, upstream_text: str) -> str:
    """Hash deliverable context for schema branch lookup (paths normalized)."""
    text = (upstream_text or "").strip()
    if not text and task_id:
        return sha256_text(f"task:{task_id}")
    try:
        from sidecar.openclaw_prefix import normalize_run_specific_paths

        text = normalize_run_specific_paths(text)
    except ImportError:
        pass
    return sha256_text(text)


def branch_fingerprint(*, schema_hash: str, deliverable_hash: str) -> str:
    return sha256_text(f"{schema_hash}|{deliverable_hash}")


def producer_turn_branch_fingerprint(
    *,
    task_id: str,
    turn_index: int,
    tool_names: Iterable[str],
) -> str:
    """Stable branch key for producer (agent 0) tool turns keyed by turn + tool set."""
    tools = ",".join(sorted(str(name).strip() for name in tool_names if str(name).strip()))
    return sha256_text(f"producer:{task_id or 'task'}:turn:{int(turn_index)}:tools:{tools}")


def prefix_boundary_token_fingerprint(token_ids: Any, prefix_len: int) -> str:
    """Hash token ids up to the tool-schema boundary for branch splice validation."""
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if isinstance(token_ids, dict):
        ids = token_ids.get("input_ids")
    else:
        ids = token_ids
    if ids is None:
        return ""
    if torch is not None and isinstance(ids, torch.Tensor):
        slice_ids = ids[0, : int(prefix_len)].detach().cpu().tolist()
    else:
        row = ids[0] if isinstance(ids, (list, tuple)) else ids
        slice_ids = list(row[: int(prefix_len)])
    return sha256_text(",".join(str(x) for x in slice_ids))

