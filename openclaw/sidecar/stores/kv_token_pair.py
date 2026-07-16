"""Guards for paired KV + token_ids payloads (no torch dependency)."""

from __future__ import annotations

from typing import Any, Optional


def token_ids_seq_length(token_ids: Any) -> int:
    """Return sequence length of tokenizer ``input_ids``, or 0 if missing."""
    if not isinstance(token_ids, dict):
        return 0
    input_ids = token_ids.get("input_ids")
    if input_ids is None:
        return 0
    try:
        return int(input_ids.shape[-1])
    except (AttributeError, TypeError, IndexError):
        try:
            return int(len(input_ids[0]))
        except (TypeError, IndexError):
            return 0


def check_kv_token_length_pair(
    kv_len: int,
    token_len: int,
    *,
    drop_num: int = 0,
    allow_empty: bool = False,
) -> Optional[str]:
    """Return an error message if KV/token lengths are inconsistent, else None.

    When ``drop_num`` is applied symmetrically, both sides must share the same
    full length before trim (``kv_len == token_len``) and ``0 <= drop_num <= len``.
    """
    kv_len = int(kv_len)
    token_len = int(token_len)
    drop_num = int(drop_num or 0)
    if kv_len < 0 or token_len < 0:
        return f"negative lengths kv={kv_len} tokens={token_len}"
    if kv_len == 0 and token_len == 0:
        return None if allow_empty else "empty KV and token_ids"
    if kv_len > 0 and token_len <= 0:
        return f"empty token_ids with non-empty KV (kv={kv_len})"
    if token_len > 0 and kv_len <= 0:
        return f"empty KV with non-empty token_ids (tokens={token_len})"
    if drop_num < 0 or drop_num > kv_len or drop_num > token_len:
        return f"drop_num={drop_num} out of range for kv={kv_len} tokens={token_len}"
    if kv_len != token_len:
        return f"kv_len={kv_len} != token_len={token_len} (drop_num={drop_num})"
    return None


def require_paired_slot_payload(
    absolute_kv: Any,
    token_ids: Any,
    *,
    drop_num: int = 0,
    require_absolute_drop: bool = True,
    context: str = "slot",
) -> int:
    """Validate slot write payload; return normalized ``drop_num``.

    New consumer/producer slots must store already-sliced absolute KV with
    ``drop_num=0``. Empty placeholder KV used in unit tests is allowed.
    """
    drop = int(drop_num or 0)
    if absolute_kv is None or absolute_kv == {} or absolute_kv == []:
        return drop
    if not isinstance(token_ids, dict) or token_ids.get("input_ids") is None:
        raise ValueError(f"{context}: token_ids.input_ids required when absolute_kv is set")
    if require_absolute_drop and drop != 0:
        raise ValueError(
            f"{context}: absolute sliced slots require drop_num=0 (got {drop})"
        )
    return drop
