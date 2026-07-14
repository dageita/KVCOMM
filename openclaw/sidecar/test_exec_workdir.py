"""Tests for clawbench exec workdir normalization."""

from __future__ import annotations

import json
import subprocess

from sidecar.bench_prompt_compose import ADD_TESTS_NORMALIZER_TASK_ID, BUGFIX_DISCOUNT_TASK_ID, CONFIG_LOADER_TASK_ID
from sidecar.bench_canonical import FEATURE_EXPORT_TASK_ID
from sidecar.tool_bridge import (
    _resolve_clawbench_tool_path,
    clawbench_tool_workspace,
    openai_message_from_generation,
    sync_clawbench_config_loader_default_to_chain,
    sync_clawbench_tests_default_to_chain,
)


def test_exec_workdir_defaults_to_openclaw_workspace_for_clawbench() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q tests/test_pricing.py", "workdir": "."}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=BUGFIX_DISCOUNT_TASK_ID,
    )
    tool_calls = message.get("tool_calls") or []
    assert len(tool_calls) == 1
    args = json.loads(tool_calls[0]["function"]["arguments"])
    assert args["workdir"] == clawbench_tool_workspace()
    assert args["command"] == "pytest -q tests/test_pricing.py"


def test_exec_workdir_uses_registered_chain_workspace_when_present(tmp_path) -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q tests/test_normalizer.py", "workdir": "."}}\n'
        "</tool_call>"
    )
    chain_workspace = str(tmp_path / "run-deadbeef")
    chain_workspace_path = tmp_path / "run-deadbeef"
    chain_workspace_path.mkdir()
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=ADD_TESTS_NORMALIZER_TASK_ID,
        workspace_dir=str(chain_workspace_path),
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == str(chain_workspace_path)
    assert args["command"] == "PYTHONPATH=. python -m pytest -q tests/test_normalizer.py"


def test_exec_workdir_redirects_default_openclaw_workspace_to_chain(tmp_path) -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q tests/test_normalizer.py", '
        '"workdir": "~/.openclaw/workspace"}}\n'
        "</tool_call>"
    )
    chain_workspace_path = tmp_path / "run-deadbeef"
    chain_workspace_path.mkdir()
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=ADD_TESTS_NORMALIZER_TASK_ID,
        workspace_dir=str(chain_workspace_path),
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == str(chain_workspace_path)


def test_read_path_resolves_to_chain_workspace(tmp_path) -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "read", "arguments": {"path": "thread.txt"}}\n'
        "</tool_call>"
    )
    chain_workspace_path = tmp_path / "run-abc123"
    chain_workspace_path.mkdir()
    (chain_workspace_path / "thread.txt").write_text("slack thread body\n", encoding="utf-8")

    resolved = _resolve_clawbench_tool_path(
        "thread.txt",
        workspace_dir=str(chain_workspace_path),
        task_profile="clawbench",
    )
    assert resolved == str(chain_workspace_path / "thread.txt")

    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        workspace_dir=str(chain_workspace_path),
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == str(chain_workspace_path / "thread.txt")


def test_read_path_does_not_double_prefix_chain_absolute(tmp_path) -> None:
    chain_workspace_path = tmp_path / "run-abc123"
    chain_workspace_path.mkdir(parents=True)
    thread = chain_workspace_path / "thread.txt"
    thread.write_text("slack thread body\n", encoding="utf-8")

    default_root = tmp_path / "default"
    default_root.mkdir()
    chain_abs = str(thread)

    resolved = _resolve_clawbench_tool_path(
        chain_abs,
        workspace_dir=str(chain_workspace_path),
        task_profile="clawbench",
    )
    assert resolved == chain_abs
    assert "kvcomm-chain" not in resolved.replace(str(chain_workspace_path), "", 1)

    raw = (
        '<tool_call>\n'
        f'{{"name": "read", "arguments": {{"path": "{chain_abs}"}}}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        workspace_dir=str(chain_workspace_path),
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == chain_abs


def test_read_path_strips_duplicated_run_prefix(tmp_path) -> None:
    chain_workspace_path = tmp_path / "run-f8dcd281"
    docs = chain_workspace_path / "docs"
    docs.mkdir(parents=True)
    notes = docs / "maintenance_notes.md"
    notes.write_text("Support window: 18 months\n", encoding="utf-8")

    resolved = _resolve_clawbench_tool_path(
        "run-f8dcd281/docs/maintenance_notes.md",
        workspace_dir=str(chain_workspace_path),
        task_profile="clawbench",
    )
    assert resolved == str(notes)

    doubled_abs = str(chain_workspace_path / "run-f8dcd281" / "docs" / "maintenance_notes.md")
    resolved_abs = _resolve_clawbench_tool_path(
        doubled_abs,
        workspace_dir=str(chain_workspace_path),
        task_profile="clawbench",
    )
    assert resolved_abs == str(notes)

    raw = (
        '<tool_call>\n'
        '{"name": "read", "arguments": {"path": "run-f8dcd281/docs/maintenance_notes.md"}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        workspace_dir=str(chain_workspace_path),
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == str(notes)
    assert args["path"].count("run-f8dcd281") == 1


def test_read_path_resolves_default_root_kvcomm_chain_absolute(tmp_path, monkeypatch) -> None:
    state = tmp_path / "openclaw"
    workspace = state / "workspace"
    chain = workspace / "kvcomm-chain" / "t2-msg-summarize-thread" / "run-xyz"
    chain.mkdir(parents=True)
    thread = chain / "thread.txt"
    thread.write_text("body\n", encoding="utf-8")

    def fake_workspace(*, workspace_dir: str = "") -> str:
        explicit = (workspace_dir or "").strip()
        if explicit and os.path.isdir(explicit):
            return explicit
        return str(workspace)

    import os

    monkeypatch.setattr("sidecar.tool_bridge.clawbench_tool_workspace", fake_workspace)

    abs_path = str(thread)
    resolved = _resolve_clawbench_tool_path(
        abs_path,
        workspace_dir=str(chain),
        task_profile="clawbench",
    )
    assert resolved == abs_path
    assert resolved.count("kvcomm-chain") == 1


def test_sync_clawbench_tests_default_to_chain(tmp_path, monkeypatch) -> None:
    default_root = tmp_path / "default-workspace"
    chain_root = tmp_path / "chain-workspace"
    default_tests = default_root / "tests"
    default_tests.mkdir(parents=True)
    (default_tests / "test_normalizer.py").write_text(
        "from normalizer import normalize_title, normalize_tags\n\ndef test_x():\n    assert True\n",
        encoding="utf-8",
    )
    chain_tests = chain_root / "tests"
    chain_tests.mkdir(parents=True)
    (chain_tests / "test_normalizer.py").write_text(
        "from openclaw.normalizer import normalize_text\n",
        encoding="utf-8",
    )

    def fake_workspace(*, workspace_dir: str = "") -> str:
        explicit = (workspace_dir or "").strip()
        if explicit:
            return explicit
        return str(default_root)

    monkeypatch.setattr("sidecar.tool_bridge.clawbench_tool_workspace", fake_workspace)
    changed = sync_clawbench_tests_default_to_chain(workspace_dir=str(chain_root))
    assert changed is True
    synced = (chain_tests / "test_normalizer.py").read_text(encoding="utf-8")
    assert "from normalizer import normalize_title, normalize_tags" in synced
    assert "openclaw" not in synced


def test_exec_workdir_uses_pythonpath_for_config_loader(tmp_path) -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q", "workdir": "."}}\n'
        "</tool_call>"
    )
    chain_workspace_path = tmp_path / "run-deadbeef"
    chain_workspace_path.mkdir()
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=CONFIG_LOADER_TASK_ID,
        workspace_dir=str(chain_workspace_path),
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == str(chain_workspace_path)
    assert args["command"] == "PYTHONPATH=. python -m pytest -q tests/test_config_loader.py"


def test_exec_workdir_uses_pythonpath_for_cross_repo(tmp_path) -> None:
    from sidecar.bench_canonical import CROSS_REPO_TASK_ID

    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q", "workdir": "."}}\n'
        "</tool_call>"
    )
    chain_workspace_path = tmp_path / "run-cross"
    chain_workspace_path.mkdir()
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=CROSS_REPO_TASK_ID,
        workspace_dir=str(chain_workspace_path),
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == str(chain_workspace_path)
    assert args["command"] == (
        "PYTHONPATH=. python -m pytest -q contracts/tests service/tests"
    )


def test_exec_workdir_uses_pythonpath_for_feature_export(tmp_path) -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q", "workdir": "."}}\n'
        "</tool_call>"
    )
    chain_workspace_path = tmp_path / "run-feature"
    chain_workspace_path.mkdir()
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=FEATURE_EXPORT_TASK_ID,
        workspace_dir=str(chain_workspace_path),
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == str(chain_workspace_path)
    assert args["command"] == "PYTHONPATH=. python -m pytest -q tests/test_export.py"


def test_exec_workdir_uses_pythonpath_for_delegation_repair(tmp_path) -> None:
    from sidecar.bench_canonical import DELEGATION_REPAIR_TASK_ID

    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q", "workdir": "."}}\n'
        "</tool_call>"
    )
    chain_workspace_path = tmp_path / "run-delegation"
    chain_workspace_path.mkdir()
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=DELEGATION_REPAIR_TASK_ID,
        workspace_dir=str(chain_workspace_path),
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == str(chain_workspace_path)
    assert args["command"] == (
        "PYTHONPATH=. python -m pytest -q tests/test_repairs.py"
    )


def test_exec_workdir_uses_pythonpath_for_memory_recall(tmp_path) -> None:
    from sidecar.bench_canonical import MEMORY_RECALL_TASK_ID

    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q && python3 verify_handoff.py", "workdir": "."}}\n'
        "</tool_call>"
    )
    chain_workspace_path = tmp_path / "run-memory"
    chain_workspace_path.mkdir()
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=MEMORY_RECALL_TASK_ID,
        workspace_dir=str(chain_workspace_path),
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == str(chain_workspace_path)
    assert args["command"] == (
        "PYTHONPATH=. python -m pytest -q tests/test_flags.py && python3 verify_handoff.py"
    )


def test_sync_clawbench_config_loader_default_to_chain(tmp_path, monkeypatch) -> None:
    default_root = tmp_path / "default-workspace"
    chain_root = tmp_path / "chain-workspace"
    default_root.mkdir()
    chain_root.mkdir()
    buggy = (
        'from __future__ import annotations\n'
        'import json, os\n'
        'from pathlib import Path\n'
        'from app_config import DEFAULTS\n\n'
        'def load_config(path=None):\n'
        '    config = dict(DEFAULTS)\n'
        '    if path:\n'
        '        config.update(json.loads(Path(path).read_text(encoding="utf-8")))\n'
        '    if "APP_PORT" in os.environ and path:\n'
        '        config["port"] = json.loads(Path(path).read_text(encoding="utf-8")).get("port", DEFAULTS["port"])\n'
        '    if "APP_DEBUG" in os.environ:\n'
        '        config["debug"] = os.environ["APP_DEBUG"]\n'
        '    return config\n'
    )
    fixed = (
        'from __future__ import annotations\n'
        'import json, os\n'
        'from pathlib import Path\n'
        'from app_config import DEFAULTS\n\n'
        'def load_config(path=None):\n'
        '    config = dict(DEFAULTS)\n'
        '    if path:\n'
        '        config.update(json.loads(Path(path).read_text(encoding="utf-8")))\n'
        '    if "APP_PORT" in os.environ:\n'
        '        config["port"] = int(os.environ["APP_PORT"])\n'
        '    elif path:\n'
        '        config["port"] = json.loads(Path(path).read_text(encoding="utf-8")).get("port", DEFAULTS["port"])\n'
        '    if "APP_DEBUG" in os.environ:\n'
        '        config["debug"] = os.environ["APP_DEBUG"].lower() == "true"\n'
        '    return config\n'
    )
    (default_root / "config_loader.py").write_text(fixed, encoding="utf-8")
    (default_root / "app_config.py").write_text("DEFAULTS = {}\n", encoding="utf-8")
    chain_test = chain_root / "tests" / "test_config_loader.py"
    chain_test.parent.mkdir(parents=True)
    chain_test.write_text("unchanged test\n", encoding="utf-8")
    (chain_root / "config_loader.py").write_text(buggy, encoding="utf-8")
    chattr_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        chattr_calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    def fake_workspace(*, workspace_dir: str = "") -> str:
        explicit = (workspace_dir or "").strip()
        if explicit:
            return explicit
        return str(default_root)

    monkeypatch.setattr("sidecar.tool_bridge.subprocess.run", fake_run)
    monkeypatch.setattr("sidecar.tool_bridge.clawbench_tool_workspace", fake_workspace)
    changed = sync_clawbench_config_loader_default_to_chain(workspace_dir=str(chain_root))
    assert changed is True
    assert (chain_root / "config_loader.py").read_text(encoding="utf-8") == fixed
    assert (chain_root / "app_config.py").read_text(encoding="utf-8") == "DEFAULTS = {}\n"
    assert chain_test.read_text(encoding="utf-8") == "unchanged test\n"
    assert any(call[:2] == ["chattr", "-i"] for call in chattr_calls)


def test_sync_config_loader_prefers_fixed_chain_over_buggy_default(tmp_path, monkeypatch) -> None:
    """Edits on absolute chain paths must not be clobbered by a buggy default cwd."""
    default_root = tmp_path / "default-workspace"
    chain_root = tmp_path / "chain-workspace"
    default_root.mkdir()
    chain_root.mkdir()
    buggy = (
        'if "APP_PORT" in os.environ and path:\n'
        '    config["port"] = 1\n'
        'if "APP_DEBUG" in os.environ:\n'
        '    config["debug"] = os.environ["APP_DEBUG"]\n'
    )
    fixed = (
        'if "APP_PORT" in os.environ:\n'
        '    config["port"] = int(os.environ["APP_PORT"])\n'
        'if "APP_DEBUG" in os.environ:\n'
        '    config["debug"] = os.environ["APP_DEBUG"].lower() == "true"\n'
    )
    (default_root / "config_loader.py").write_text(buggy, encoding="utf-8")
    (default_root / "app_config.py").write_text("DEFAULTS = {}\n", encoding="utf-8")
    (chain_root / "config_loader.py").write_text(fixed, encoding="utf-8")
    (chain_root / "app_config.py").write_text("DEFAULTS = {}\n", encoding="utf-8")

    def fake_workspace(*, workspace_dir: str = "") -> str:
        explicit = (workspace_dir or "").strip()
        if explicit:
            return explicit
        return str(default_root)

    monkeypatch.setattr(
        "sidecar.tool_bridge.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0),
    )
    monkeypatch.setattr("sidecar.tool_bridge.clawbench_tool_workspace", fake_workspace)
    changed = sync_clawbench_config_loader_default_to_chain(workspace_dir=str(chain_root))
    assert changed is True
    assert (chain_root / "config_loader.py").read_text(encoding="utf-8") == fixed
    assert (default_root / "config_loader.py").read_text(encoding="utf-8") == fixed



def test_exec_workdir_unchanged_for_non_clawbench() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q", "workdir": "."}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(raw, task_profile="copy")
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == "."
    assert args["command"] == "pytest -q"


def test_exec_pytest_scoped_to_tests_path_for_clawbench() -> None:
    raw = (
        '<tool_call>\n'
        '{"name": "exec", "arguments": {"command": "pytest -q", "workdir": "."}}\n'
        "</tool_call>"
    )
    message = openai_message_from_generation(
        raw,
        task_profile="clawbench",
        task_id=BUGFIX_DISCOUNT_TASK_ID,
    )
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["workdir"] == clawbench_tool_workspace()
    assert args["command"] == "pytest -q tests/test_pricing.py"
