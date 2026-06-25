"""Prefix segment parity checks for OpenClaw turn placeholders."""

from __future__ import annotations

import re

import pytest

try:
    import torch
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - optional in minimal CI
    torch = None
    AutoTokenizer = None


def _count_segments(original_text: str, tokenizer, *, keep_whitespace: bool) -> tuple[int, int]:
    placeholder_pattern = (
        r"\{((?:agent|condition)_\w+_(?:current|history)|user_question|turn_\d+_(?:assistant|tool))\}"
    )
    matches = list(re.finditer(placeholder_pattern, original_text))
    last_pos = 0
    text_segments = 0
    for m in matches:
        start, _ = m.span()
        if last_pos < start:
            txt = original_text[last_pos:start]
            if keep_whitespace:
                if txt:
                    text_segments += 1
            elif txt.strip():
                text_segments += 1
        last_pos = m.end()
    txt = original_text[last_pos:]
    if keep_whitespace:
        if txt:
            text_segments += 1
    elif txt.strip():
        text_segments += 1
    return len(matches), text_segments


@pytest.mark.skipif(AutoTokenizer is None, reason="transformers not installed")
def test_turn_placeholder_segment_parity() -> None:
    model_path = "/models/Qwen3-32B"
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    static = "User request: fix pricing bug\n\nYour job: read pricing.py"
    turn_suffix = "".join(
        f"\n\n{{turn_{idx}_assistant}}\n\n{{turn_{idx}_tool}}\n" for idx in range(1, 4)
    )
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": static + turn_suffix}],
        tokenize=False,
        add_generation_prompt=True,
    )

    ph_old, text_old = _count_segments(prompt, tok, keep_whitespace=False)
    ph_new, text_new = _count_segments(prompt, tok, keep_whitespace=True)

    assert ph_old == 6
    assert text_old - 1 < ph_old, "legacy whitespace skip drops turn-tool filler segments"
    assert text_new - 1 >= ph_new, "whitespace spans must pair every turn placeholder"
