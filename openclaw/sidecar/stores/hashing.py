"""Content and topology hashing helpers for KVCOMM store keys."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

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


def left_fingerprint(parts: Iterable[str]) -> str:
    joined = "|".join(str(part) for part in parts if part)
    return sha256_text(joined)
