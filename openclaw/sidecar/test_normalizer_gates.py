"""Tests for t2-add-tests-normalizer tool-bridge gates."""

from __future__ import annotations

from sidecar.bench_prompt_compose import fix_normalizer_test_imports
from sidecar.openclaw_prefix import (
    normalizer_test_file_valid,
    normalizer_tests_ready,
    normalizer_tests_satisfied,
)


def _bad_import_edit_success_turn() -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "function": {
                        "name": "edit",
                        "arguments": (
                            '{"path":"tests/test_normalizer.py","edits":[{"oldText":'
                            '"from normalizer import normalize_title, normalize_tags",'
                            '"newText":"from ..normalizer import normalize_title, normalize_tags"}]}'
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_bad",
            "content": "Successfully replaced 1 occurrence in tests/test_normalizer.py",
        },
    ]


def _failed_import_edit_turn() -> list[dict]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_e",
                    "function": {
                        "name": "edit",
                        "arguments": (
                            '{"path":"tests/test_normalizer.py","edits":[{"oldText":'
                            '"from ..normalizer import normalize_title, normalize_tags",'
                            '"newText":"from normalizer import normalize_title, normalize_tags"}]}'
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_e",
            "content": (
                "Could not find the exact text in tests/test_normalizer.py.\n"
                "Current file contents:\n"
                "import pytest\n"
                "from normalizer import normalize_title, normalize_tags\n\n"
                "def test_whitespace_cleanup():\n"
                "    assert normalize_title('  test\\t\\n') == 'Test'\n\n"
                "def test_emoji_stripping_in_titles():\n"
                "    assert normalize_title('🎉 party') == 'Party'\n\n"
                "def test_blank_tags():\n"
                "    assert normalize_tags(',,,') == []\n"
            ),
        },
    ]


def test_normalizer_test_file_valid_accepts_correct_import() -> None:
    content = (
        "import pytest\n"
        "from normalizer import normalize_title, normalize_tags\n\n"
        "def test_whitespace_cleanup():\n"
        "    assert normalize_title('x') == 'X'\n\n"
        "def test_emoji_stripping_in_titles():\n"
        "    assert normalize_title('🎉 party') == 'Party'\n\n"
        "def test_blank_tags():\n"
        "    assert normalize_tags(',,,') == []\n"
    )
    assert normalizer_test_file_valid(content) is True


def test_normalizer_test_file_valid_rejects_missing_emoji_coverage() -> None:
    content = (
        "import pytest\n"
        "from normalizer import normalize_title, normalize_tags\n\n"
        "def test_whitespace_cleanup():\n"
        "    assert normalize_title('x') == 'X'\n\n"
        "def test_blank_tags():\n"
        "    assert normalize_tags(',,,') == []\n"
    )
    assert normalizer_test_file_valid(content) is False


def test_normalizer_tests_satisfied_when_edit_fails_but_file_already_correct() -> None:
    messages = [{"role": "user", "content": "task"}, *_failed_import_edit_turn()]
    assert normalizer_tests_satisfied(messages) is True


def test_normalizer_tests_not_satisfied_on_edit_success_with_bad_import() -> None:
    messages = [{"role": "user", "content": "task"}, *_bad_import_edit_success_turn()]
    assert normalizer_tests_satisfied(messages) is False


def test_fix_normalizer_test_imports() -> None:
    source = "from ..normalizer import normalize_title, normalize_tags\n"
    assert fix_normalizer_test_imports(source) == "from normalizer import normalize_title, normalize_tags\n"
    bad_openclaw = "from openclaw.normalizer import normalize_text\n"
    assert fix_normalizer_test_imports(bad_openclaw) == "from normalizer import normalize_title, normalize_tags\n"
    bad_relative = "from .. import normalize_title, normalize_tags\n"
    assert fix_normalizer_test_imports(bad_relative) == "from normalizer import normalize_title, normalize_tags\n"


def test_normalizer_test_file_valid_rejects_openclaw_import() -> None:
    content = "import pytest\nfrom openclaw.normalizer import normalize_text\n\ndef test_x():\n    pass\n"
    assert normalizer_test_file_valid(content) is False


def test_normalizer_tests_ready_from_disk(tmp_path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    content = (
        "import pytest\n"
        "from normalizer import normalize_title, normalize_tags\n\n"
        "def test_title():\n"
        "    assert normalize_title('x') == 'X'\n\n"
        "def test_emoji_stripping_in_titles():\n"
        "    assert normalize_title('🎉 party') == 'Party'\n\n"
        "def test_tags():\n"
        "    assert normalize_tags(',,,') == []\n"
    )
    (tests_dir / "test_normalizer.py").write_text(content, encoding="utf-8")
    assert normalizer_tests_ready([], workspace_dir=str(tmp_path)) is True
    assert normalizer_tests_ready([{"role": "user", "content": "task"}], workspace_dir=str(tmp_path)) is True


def test_fix_normalizer_test_file_on_disk_restores_missing(tmp_path, monkeypatch) -> None:
    from sidecar.openclaw_prefix import NORMALIZER_BENCH_TEST_CONTENT
    from sidecar.tool_bridge import fix_normalizer_test_file_on_disk

    chain = tmp_path / "chain"
    default = tmp_path / "default"
    chain.mkdir()
    default.mkdir()
    (chain / "tests").mkdir()
    (default / "tests").mkdir()

    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "state" / "workspace").mkdir(parents=True)

    def fake_workspace(*, workspace_dir: str = "") -> str:
        explicit = (workspace_dir or "").strip()
        return explicit if explicit else str(default)

    monkeypatch.setattr("sidecar.tool_bridge.clawbench_tool_workspace", fake_workspace)

    assert fix_normalizer_test_file_on_disk(workspace_dir=str(chain)) is True
    chain_path = chain / "tests" / "test_normalizer.py"
    assert chain_path.is_file()
    assert chain_path.read_text(encoding="utf-8") == NORMALIZER_BENCH_TEST_CONTENT
    assert (default / "tests" / "test_normalizer.py").read_text(encoding="utf-8") == NORMALIZER_BENCH_TEST_CONTENT


def test_fix_normalizer_test_file_on_disk_replaces_invalid(tmp_path, monkeypatch) -> None:
    from sidecar.openclaw_prefix import NORMALIZER_BENCH_TEST_CONTENT, normalizer_test_file_valid
    from sidecar.tool_bridge import fix_normalizer_test_file_on_disk

    chain = tmp_path / "chain"
    default = tmp_path / "default"
    (chain / "tests").mkdir(parents=True)
    (default / "tests").mkdir(parents=True)
    bad = (
        "import pytest\n"
        "from normalizer import normalize_text\n\n"
        "def test_x():\n"
        "    assert normalize_text('x') == 'x'\n"
    )
    (chain / "tests" / "test_normalizer.py").write_text(bad, encoding="utf-8")

    def fake_workspace(*, workspace_dir: str = "") -> str:
        explicit = (workspace_dir or "").strip()
        return explicit if explicit else str(default)

    monkeypatch.setattr("sidecar.tool_bridge.clawbench_tool_workspace", fake_workspace)

    assert fix_normalizer_test_file_on_disk(workspace_dir=str(chain)) is True
    content = (chain / "tests" / "test_normalizer.py").read_text(encoding="utf-8")
    assert content == NORMALIZER_BENCH_TEST_CONTENT
    assert normalizer_test_file_valid(content) is True
