"""ToolSemanticIndex — semantic catalog over ToolKVBackend entries."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any

from sidecar.stores.hashing import sha256_text, short_hash


@dataclass
class ToolSemanticEntry:
    entry_id: str
    kv_ref: str
    content_hash: str
    query: str
    query_embedding: list[float]
    tool_name: str = ""
    path_hint: str = ""
    ph_id_hint: str = ""
    token_len: int = 0
    hit_count: int = 0


@dataclass
class ToolSemanticLookupResult:
    hit: bool
    s_max: float = 0.0
    match_mode: str = "miss"
    entry: ToolSemanticEntry | None = None
    kv_ref: str | None = None


class ToolSemanticIndex:
    def __init__(self, *, tau: float = 0.8) -> None:
        self.tau = tau
        self._entries: dict[str, ToolSemanticEntry] = {}
        self._by_content_hash: dict[str, list[str]] = {}

    @staticmethod
    def _embed(text: str, dim: int = 64) -> list[float]:
        """Lightweight deterministic embedding (scaffolding; swap for SentenceTransformer)."""
        vec = [0.0] * dim
        tokens = re.findall(r"[a-zA-Z0-9_./-]+", (text or "").lower())
        if not tokens:
            tokens = ["_empty_"]
        for tok in tokens:
            h = int(sha256_text(tok)[:8], 16)
            vec[h % dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def upsert(
        self,
        *,
        query: str,
        kv_ref: str,
        content_hash: str,
        tool_name: str = "",
        path_hint: str = "",
        ph_id_hint: str = "",
        token_len: int = 0,
    ) -> ToolSemanticEntry:
        entry_id = f"tse:{short_hash(content_hash)}:{short_hash(query)}"
        entry = ToolSemanticEntry(
            entry_id=entry_id,
            kv_ref=kv_ref,
            content_hash=content_hash,
            query=query,
            query_embedding=self._embed(query),
            tool_name=tool_name,
            path_hint=path_hint,
            ph_id_hint=ph_id_hint,
            token_len=token_len,
        )
        self._entries[entry_id] = entry
        self._by_content_hash.setdefault(content_hash, [])
        if entry_id not in self._by_content_hash[content_hash]:
            self._by_content_hash[content_hash].append(entry_id)
        return entry

    def lookup(self, query: str, *, content_hash: str | None = None) -> ToolSemanticLookupResult:
        if content_hash:
            for entry_id in self._by_content_hash.get(content_hash, []):
                entry = self._entries.get(entry_id)
                if entry is not None:
                    entry.hit_count += 1
                    return ToolSemanticLookupResult(
                        hit=True,
                        s_max=1.0,
                        match_mode="exact_hash",
                        entry=entry,
                        kv_ref=entry.kv_ref,
                    )

        v_in = self._embed(query)
        best: ToolSemanticEntry | None = None
        best_score = -1.0
        for entry in self._entries.values():
            score = self._dot(v_in, entry.query_embedding)
            if score > best_score:
                best_score = score
                best = entry

        if best is None or best_score < self.tau:
            return ToolSemanticLookupResult(hit=False, s_max=max(0.0, best_score))

        best.hit_count += 1
        return ToolSemanticLookupResult(
            hit=True,
            s_max=best_score,
            match_mode="semantic_full" if best_score >= 0.95 else "semantic_partial",
            entry=best,
            kv_ref=best.kv_ref,
        )

    def purge_node_hints(self, node_id: str) -> None:
        _ = node_id
        return

    def clear(self) -> None:
        self._entries.clear()
        self._by_content_hash.clear()
