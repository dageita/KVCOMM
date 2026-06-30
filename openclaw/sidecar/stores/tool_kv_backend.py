"""ToolKVBackend — content-addressed isolated tool-result absolute KV."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Any, Callable

from sidecar.stores.hashing import sha256_text


@dataclass
class ToolKVEntry:
    kv_ref: str
    content_hash: str
    absolute_kv: Any
    token_ids: dict[str, Any]
    token_len: int
    materialization: str = "isolated"
    ref_count: int = 0


class ToolKVBackend:
    def __init__(self, *, max_entries: int = 512) -> None:
        self.max_entries = max_entries
        self._by_ref: dict[str, ToolKVEntry] = {}
        self._by_content_hash: dict[str, str] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()

    @staticmethod
    def kv_ref_for_content(content_hash: str) -> str:
        return f"tool:kv:{content_hash}"

    def get(self, kv_ref: str) -> ToolKVEntry | None:
        entry = self._by_ref.get(kv_ref)
        if entry is not None:
            self._touch(kv_ref)
        return entry

    def get_by_content_hash(self, content_hash: str) -> ToolKVEntry | None:
        ref = self._by_content_hash.get(content_hash)
        if not ref:
            return None
        return self.get(ref)

    def get_or_create(
        self,
        result_text: str,
        forward_fn: Callable[[str], tuple[Any, dict[str, Any]]],
    ) -> ToolKVEntry:
        content_hash = sha256_text(result_text)
        existing = self.get_by_content_hash(content_hash)
        if existing is not None:
            existing.ref_count += 1
            return existing

        absolute_kv, token_ids = forward_fn(result_text)
        token_len = 0
        if isinstance(token_ids, dict) and token_ids.get("input_ids") is not None:
            input_ids = token_ids["input_ids"]
            if hasattr(input_ids, "shape"):
                token_len = int(input_ids.shape[-1])
            elif isinstance(input_ids, (list, tuple)):
                first = input_ids[0] if input_ids else []
                token_len = len(first) if isinstance(first, (list, tuple)) else len(input_ids)

        kv_ref = self.kv_ref_for_content(content_hash)
        entry = ToolKVEntry(
            kv_ref=kv_ref,
            content_hash=content_hash,
            absolute_kv=absolute_kv,
            token_ids=token_ids,
            token_len=token_len,
            ref_count=1,
        )
        self._insert(entry)
        return entry

    def _insert(self, entry: ToolKVEntry) -> None:
        while len(self._by_ref) >= self.max_entries and self._lru:
            evict_ref, _ = self._lru.popitem(last=False)
            evicted = self._by_ref.pop(evict_ref, None)
            if evicted is not None:
                self._by_content_hash.pop(evicted.content_hash, None)
        self._by_ref[entry.kv_ref] = entry
        self._by_content_hash[entry.content_hash] = entry.kv_ref
        self._touch(entry.kv_ref)

    def _touch(self, kv_ref: str) -> None:
        if kv_ref in self._lru:
            self._lru.move_to_end(kv_ref)
        else:
            self._lru[kv_ref] = None

    def clear(self) -> None:
        self._by_ref.clear()
        self._by_content_hash.clear()
        self._lru.clear()
