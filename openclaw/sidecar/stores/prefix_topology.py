"""Prefix topology decisions for turn append vs static rebuild."""

from __future__ import annotations

from dataclasses import dataclass

from sidecar.stores.hashing import static_template_hash, topology_id


@dataclass
class PrefixTopologyState:
    static_template_hash: str = ""
    topology_id: str = ""
    turn_count: int = 0


@dataclass
class PrefixRebuildPlan:
    action: str  # "noop" | "append_turn" | "rewind_turns" | "static_rebuild" | "full_rebuild"
    reason: str
    from_turn_count: int = 0
    to_turn_count: int = 0


def read_topology(bucket: dict) -> PrefixTopologyState:
    return PrefixTopologyState(
        static_template_hash=str(bucket.get("static_template_hash") or ""),
        topology_id=str(bucket.get("topology_id") or ""),
        turn_count=int(bucket.get("turn_count") or 0),
    )


def plan_prefix_update(
    *,
    user_template: str,
    desired_turn_count: int,
    bucket: dict,
    initialized: bool,
) -> PrefixRebuildPlan:
    static_hash = static_template_hash(user_template)
    stored = read_topology(bucket)
    desired = int(desired_turn_count)

    if not initialized or not bucket.get("prefix"):
        return PrefixRebuildPlan(
            action="full_rebuild",
            reason="uninitialized_or_missing_prefix",
            from_turn_count=stored.turn_count,
            to_turn_count=desired,
        )

    if stored.static_template_hash and stored.static_template_hash != static_hash:
        return PrefixRebuildPlan(
            action="static_rebuild",
            reason="static_template_hash_changed",
            from_turn_count=stored.turn_count,
            to_turn_count=desired,
        )

    if desired < stored.turn_count:
        return PrefixRebuildPlan(
            action="rewind_turns",
            reason="turn_count_regression_new_run",
            from_turn_count=stored.turn_count,
            to_turn_count=desired,
        )

    if desired == stored.turn_count:
        expected_topology = topology_id(static_hash=static_hash, turn_count=desired)
        if stored.topology_id and stored.topology_id != expected_topology:
            return PrefixRebuildPlan(
                action="static_rebuild",
                reason="topology_id_mismatch",
                from_turn_count=stored.turn_count,
                to_turn_count=desired,
            )
        return PrefixRebuildPlan(
            action="noop",
            reason="topology_current",
            from_turn_count=stored.turn_count,
            to_turn_count=desired,
        )

    if desired == stored.turn_count + 1:
        return PrefixRebuildPlan(
            action="append_turn",
            reason="incremental_turn_append",
            from_turn_count=stored.turn_count,
            to_turn_count=desired,
        )

    return PrefixRebuildPlan(
        action="static_rebuild",
        reason="turn_count_jump",
        from_turn_count=stored.turn_count,
        to_turn_count=desired,
    )


def write_topology(bucket: dict, *, user_template: str, turn_count: int) -> None:
    static_hash = static_template_hash(user_template)
    bucket["static_template_hash"] = static_hash
    bucket["topology_id"] = topology_id(static_hash=static_hash, turn_count=int(turn_count))
    bucket["turn_count"] = int(turn_count)
    bucket["user_template"] = str(user_template or "").strip()
