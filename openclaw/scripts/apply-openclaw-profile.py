#!/usr/bin/env python3
"""Merge KVCOMM OpenClaw profile into existing openclaw.json (no full overwrite)."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

# ClawBench native harness requires global tools.profile=coding for agents.create run agents.
CLAWBENCH_NATIVE_TOOLS_PROFILE = "coding"

KVCOMM_PROFILES = frozenset(
    {"dense", "sidecar", "dual", "clawbench-capability", "clawbench-capability-sidecar"}
)
CLAWBENCH_CAPABILITY_PROFILES = frozenset({"clawbench-capability", "clawbench-capability-sidecar"})

DEFAULT_CAPABILITY_SUBAGENT_TOOLS = [
    "read",
    "write",
    "edit",
    "apply_patch",
    "exec",
    "process",
    "session_status",
]


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in patch.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def strip_private_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_private_keys(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_private_keys(v) for v in obj]
    return obj


def strip_kvcomm_global_tools_profile(template: dict[str, Any]) -> dict[str, Any]:
    """KVCOMM lane configures orchestrator/subagents explicitly; do not set global tools.profile."""
    cleaned = deepcopy(template)
    tools = cleaned.get("tools")
    if isinstance(tools, dict):
        tools = deepcopy(tools)
        tools.pop("profile", None)
        if tools:
            cleaned["tools"] = tools
        else:
            cleaned.pop("tools", None)
    return cleaned


def ensure_main_agent_tools(merged: dict[str, Any], profile: str = "") -> None:
    """KVCOMM orchestrator (main) only: minimal + sessions_spawn. Does not touch global tools.profile."""
    agents = merged.setdefault("agents", {})
    agent_list = agents.setdefault("list", [])
    main_entry = None
    for entry in agent_list:
        if isinstance(entry, dict) and entry.get("id") == "main":
            main_entry = entry
            break
    if main_entry is None:
        main_entry = {"id": "main"}
        agent_list.append(main_entry)
    tools = main_entry.setdefault("tools", {})
    tools["profile"] = "coding" if profile in CLAWBENCH_CAPABILITY_PROFILES else "minimal"
    tools.pop("allow", None)
    also_allow = set(tools.get("alsoAllow") or [])
    also_allow.add("sessions_spawn")
    if profile in CLAWBENCH_CAPABILITY_PROFILES:
        subagent_allow = (
            (merged.get("tools") or {}).get("subagents", {}).get("tools", {}).get("allow") or []
        )
        also_allow.update(str(name).strip() for name in subagent_allow if str(name).strip())
    tools["alsoAllow"] = sorted(also_allow)


def preserve_provider_base_urls(existing: dict[str, Any], merged: dict[str, Any]) -> None:
    """Keep working upstream URLs when switching profiles."""
    existing_providers = (existing.get("models") or {}).get("providers") or {}
    merged_providers = (merged.get("models") or {}).get("providers") or {}
    for name, prov in existing_providers.items():
        if not isinstance(prov, dict):
            continue
        existing_url = prov.get("baseUrl")
        if existing_url and isinstance(merged_providers.get(name), dict):
            merged_providers[name]["baseUrl"] = existing_url


def merge_model_allowlists(existing: dict[str, Any], merged: dict[str, Any]) -> None:
    """Additive merge for agents.defaults.models — keep refs from prior profiles."""
    existing_models = (existing.get("agents") or {}).get("defaults", {}).get("models") or {}
    merged_models = (merged.get("agents") or {}).get("defaults", {}).setdefault("models", {})
    if not isinstance(existing_models, dict) or not isinstance(merged_models, dict):
        return
    for key, value in existing_models.items():
        merged_models.setdefault(key, deepcopy(value))


def merge_provider_model_catalogs(existing: dict[str, Any], merged: dict[str, Any]) -> None:
    """Additive merge for models.providers.*.models arrays."""
    existing_providers = (existing.get("models") or {}).get("providers") or {}
    merged_providers = (merged.get("models") or {}).setdefault("providers", {})
    for pname, eprov in existing_providers.items():
        if not isinstance(eprov, dict):
            continue
        if pname not in merged_providers:
            merged_providers[pname] = deepcopy(eprov)
            continue
        mprov = merged_providers[pname]
        if not isinstance(mprov, dict):
            continue
        e_models = eprov.get("models") or []
        m_models = mprov.setdefault("models", [])
        if not isinstance(e_models, list) or not isinstance(m_models, list):
            continue
        seen = {
            m.get("id")
            for m in m_models
            if isinstance(m, dict) and m.get("id")
        }
        for entry in e_models:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if model_id and model_id not in seen:
                m_models.append(deepcopy(entry))
                seen.add(model_id)


def merge_gateway_tool_allow(existing: dict[str, Any], merged: dict[str, Any]) -> None:
    """Additive merge for gateway HTTP tool allowlist — never replace ClawBench allows with KVCOMM-only list."""
    existing_allow = set((existing.get("gateway") or {}).get("tools", {}).get("allow") or [])
    merged_allow = set((merged.get("gateway") or {}).get("tools", {}).get("allow") or [])
    merged_allow.add("sessions_spawn")
    merged_allow.update(existing_allow)
    merged.setdefault("gateway", {}).setdefault("tools", {})["allow"] = sorted(merged_allow)


def preserve_clawbench_tools_profile(existing: dict[str, Any], merged: dict[str, Any]) -> None:
    existing_profile = (existing.get("tools") or {}).get("profile")
    if existing_profile == CLAWBENCH_NATIVE_TOOLS_PROFILE:
        merged.setdefault("tools", {})["profile"] = CLAWBENCH_NATIVE_TOOLS_PROFILE
        print(
            f"[apply-profile] preserved tools.profile={CLAWBENCH_NATIVE_TOOLS_PROFILE} "
            "(ClawBench native harness)"
        )


def ensure_capability_subagent_tools(
    merged: dict[str, Any],
    template: dict[str, Any],
    profile: str,
) -> None:
    """Force full coding-tool allowlist for capability lane subagents (do not inherit stale allow)."""
    if profile not in CLAWBENCH_CAPABILITY_PROFILES:
        return
    template_allow = (
        (template.get("tools") or {}).get("subagents", {}).get("tools", {}).get("allow")
    )
    allow = [str(name).strip() for name in (template_allow or DEFAULT_CAPABILITY_SUBAGENT_TOOLS) if str(name).strip()]
    merged.setdefault("tools", {}).setdefault("subagents", {}).setdefault("tools", {})["allow"] = allow
    print(f"[apply-profile] set tools.subagents.tools.allow={allow}")


def ensure_capability_elevated(merged: dict[str, Any], template: dict[str, Any], profile: str) -> None:
    """Allow elevated exec for local bench subagents (model sometimes passes elevated: true)."""
    if profile not in CLAWBENCH_CAPABILITY_PROFILES:
        return
    template_elevated = (template.get("tools") or {}).get("elevated")
    if not template_elevated:
        return
    merged.setdefault("tools", {})["elevated"] = deepcopy(template_elevated)
    print("[apply-profile] set tools.elevated for clawbench capability lane")


def ensure_capability_primary_model(merged: dict[str, Any], template: dict[str, Any], profile: str) -> None:
    """Capability lane: primary model follows profile template (vllm vs kvcomm)."""
    if profile not in CLAWBENCH_CAPABILITY_PROFILES:
        return
    primary = ((template.get("agents") or {}).get("defaults") or {}).get("model", {}).get("primary")
    if primary:
        merged.setdefault("agents", {}).setdefault("defaults", {}).setdefault("model", {})["primary"] = primary
        print(f"[apply-profile] set agents.defaults.model.primary={primary}")


def apply_profile(template_path: Path, target_path: Path, profile: str) -> None:
    template_raw = strip_private_keys(json.loads(template_path.read_text(encoding="utf-8")))
    existing: dict[str, Any] = {}
    if target_path.exists():
        existing = json.loads(target_path.read_text(encoding="utf-8"))

    token = (
        existing.get("gateway", {})
        .get("auth", {})
        .get("token")
    )
    if token and token != "CHANGE_ME_AFTER_SETUP":
        template_raw.setdefault("gateway", {}).setdefault("auth", {})["token"] = token

    template = template_raw
    if profile in KVCOMM_PROFILES:
        template = strip_kvcomm_global_tools_profile(template_raw)

    merged = deep_merge(existing, template)

    preserve_provider_base_urls(existing, merged)
    merge_model_allowlists(existing, merged)
    merge_provider_model_catalogs(existing, merged)
    merge_gateway_tool_allow(existing, merged)

    if profile in KVCOMM_PROFILES:
        ensure_capability_subagent_tools(merged, template_raw, profile)
        ensure_capability_elevated(merged, template_raw, profile)
        ensure_capability_primary_model(merged, template_raw, profile)
        ensure_main_agent_tools(merged, profile)
        preserve_clawbench_tools_profile(existing, merged)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        bak = target_path.with_suffix(f".json.bak.{profile}")
        bak.write_text(target_path.read_text(encoding="utf-8"), encoding="utf-8")

    target_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[apply-profile] merged profile={profile} -> {target_path}")


def main() -> None:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <template.json> <target.json> <profile>", file=sys.stderr)
        sys.exit(1)
    apply_profile(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])


if __name__ == "__main__":
    main()
