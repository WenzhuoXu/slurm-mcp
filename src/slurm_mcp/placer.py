"""Placement: target keys, candidate generation, hard feasibility and a provisional ranking
(design section 8; phase-2 minimal build).

What exists now (section 8 names kept):
- ``resolve_explicit(placement_or_target)`` -> ``Target`` for an explicit placement string / object.
- ``candidates(spec, caps_by_cluster, profiles, policy, placement="auto")`` -> ``list[Target]``: partition access
  rules (AllowGroups/AllowAccounts/assoc), one typed candidate per gres type (never untyped), joint candidates for
  ``profile.partition_groups``, ``policy.targets_allow/deny`` globs, ``target_overrides[*].enabled``, explicit
  placement strings/lists restricting the set, ``spec.cluster``/``spec.partition`` pinning.
- ``feasibility(target, spec, caps, snapshot, policy, my_running, ...)`` -> ``(ok, why)``: the hard rules that need
  only caps/snapshot: ``time_s <= effective max_wall``, ``gpus x nodes <= max_tres_pj gres/gpu``, per-node gres
  count, ``nodes <= MaxNodes``, cpus/mem <= node size, partition ``State=UP``, etiquette ``max_running_per_target``
  (policy + profile override), QOS ``max_jobs_pu``/``max_submit_pu``, the self-preemption rule, circuit-breaker and
  infeasible windows from ``target_stats``.
- ``queue_depth(snapshot, caps, target)`` -> ``(typed, untyped)`` pending rows per section 6.2 classification.
- ``rank(spec, caps_by_cluster, snapshots, profiles, policy, ...)`` -> ``list[PlanOption]`` ordered by the
  provisional ``score_h = est_wait_h + hours + etiquette + penalty`` with ``est_wait_h = min(unknown_wait_h,
  0.25 x queue_ahead)`` and ``est_wait_src="depth"``; infeasible rows (<= 3) last.

Phase 3 MUST ADD (section 8, unchanged names): ``--test-only`` estimates (top 4 by pre-score, one exec per
target, ``est_wait_src="test_only"``, never overridden downwards by depth), ``hist_p50`` from ``wait_history``
(``src="history"`` when ``< test_only/3``), the ``ahead`` subset by priority from ``::MINE``, the
``est_wait_h = 0`` shortcut only when ``depth == 0`` including untyped rows, ``cost_su``/``cost_worst_su`` (rate x
units x hours, worst case x (1 + max_restarts) when requeueable) and the ``su_balance - su_reserve`` feasibility
rule, maintenance windows (``::RESV`` MAINT overlapping ``now + est_wait + time``), preemption risk
(``risk_pct``/``risk_h`` with the GraceTime rule), the full score (``cost_su x su_to_hours``, staging penalty,
``cheapest`` free-cluster preference, tie-breaks), pending caps ("hold locally", ``QUEUED``), portable-input checks
for cross-cluster moves, rebalance proposals and ``why`` texts quoting the numbers.
"""
from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .config import ClusterProfile, target_override
from .errors import err
from .models import Charge, JobSpec, PlacementPolicy, PlanOption, Target
from .render import requeueable
from .slurm.discovery import charge_for, effective_limits, partition_accessible, qos_for_partition

MAX_INFEASIBLE_ROWS = 3
DEPTH_HOURS_PER_JOB = 0.25
STAGING_PENALTY_H = 0.5

_MEM_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kKmMgGtT]?)[bB]?$")
_HOSTLIST_RE = re.compile(r"([^,\[\]]+)(?:\[([^\]]*)\])?")


def _mem_mb(value: str | None) -> float | None:
    if not value:
        return None
    m = _MEM_RE.match(str(value).strip())
    if not m:
        return None
    num, unit = float(m.group(1)), m.group(2).upper()
    return num * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]


def expand_hostlist(expr: str | None) -> set[str]:
    """``trace[01-29],x[1,3]`` -> node names (a tiny SLURM hostlist expander for node-sharing checks)."""
    out: set[str] = set()
    if not expr:
        return out
    for m in _HOSTLIST_RE.finditer(expr):
        prefix, ranges = m.group(1), m.group(2)
        if ranges is None:
            out.add(prefix)
            continue
        for part in ranges.split(","):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                width = len(a)
                for i in range(int(a), int(b) + 1):
                    out.add(f"{prefix}{str(i).zfill(width)}")
            else:
                out.add(f"{prefix}{part}")
    return out


# --- targets -------------------------------------------------------------------------------------------------

def resolve_explicit(placement_or_target: Target | str | Sequence[str], profiles: Mapping[str, ClusterProfile] | None = None,
                     caps_by_cluster: Mapping[str, Mapping[str, Any] | None] | None = None) -> Target:
    """An explicit placement (``"trace:batch:a40"``, a ``Target``, or a one-element list) -> validated ``Target``.

    Validates the cluster against ``profiles`` and the partitions against the cluster's caps when given
    (``E_PARTITION``); a multi-element list is not an explicit target (``E_INVALID_SPEC``).
    """
    if isinstance(placement_or_target, Target):
        tgt = placement_or_target
    elif isinstance(placement_or_target, str):
        if placement_or_target.strip().lower() == "auto":
            raise err("E_INVALID_SPEC", "placement 'auto' is not an explicit target", fix="pass '<cluster>:<partition>'")
        tgt = Target.parse(placement_or_target)
    else:
        items = list(placement_or_target)
        if len(items) != 1:
            raise err("E_INVALID_SPEC", f"an explicit target needs exactly one target string, got {len(items)}",
                      fix="pass placement='<cluster>:<partition>[:<gres-type>][@<qos>]'")
        tgt = Target.parse(str(items[0]))
    if profiles is not None and tgt.cluster not in profiles:
        raise err("E_INVALID_SPEC", f"unknown cluster {tgt.cluster!r} in target {tgt.key}",
                  fix=f"known clusters: {', '.join(sorted(profiles)) or '(none)'}")
    if caps_by_cluster is not None and caps_by_cluster.get(tgt.cluster):
        parts = (caps_by_cluster[tgt.cluster] or {}).get("partitions") or {}
        for p in tgt.partitions:
            if parts and p not in parts:
                raise err("E_PARTITION", f"partition {p!r} does not exist on {tgt.cluster}", cluster=tgt.cluster)
    if tgt.account is None and profiles is not None:
        tgt = tgt.model_copy(update={"account": profiles[tgt.cluster].default_account})
    return tgt


def _placement_filters(placement: str | Sequence[str] | None) -> list[Target] | None:
    if placement is None:
        return None
    if isinstance(placement, str):
        if placement.strip().lower() == "auto":
            return None
        return [Target.parse(placement)]
    items = [Target.parse(str(p)) for p in placement]
    return items or None


def _matches_filter(tgt: Target, flt: Target) -> bool:
    if tgt.cluster != flt.cluster:
        return False
    if not set(tgt.partitions) <= set(flt.partitions):      # a joint candidate needs every member allowed
        return False
    if flt.gres_type and tgt.gres_type != flt.gres_type:
        return False
    if flt.qos and tgt.qos != flt.qos:
        return False
    return True


def candidates(spec: JobSpec, caps_by_cluster: Mapping[str, Mapping[str, Any] | None],
               profiles: Mapping[str, ClusterProfile], policy: PlacementPolicy | None = None,
               placement: str | Sequence[str] | None = "auto") -> list[Target]:
    """Candidate targets per section 8 (see the module docstring for the rules)."""
    policy = policy or PlacementPolicy()
    filters = _placement_filters(placement)
    clusters = [spec.cluster] if spec.cluster else list(caps_by_cluster)
    if policy.prefer_cluster in clusters:
        clusters = [policy.prefer_cluster] + [c for c in clusters if c != policy.prefer_cluster]
    out: list[Target] = []
    for cluster in clusters:
        caps = caps_by_cluster.get(cluster)
        profile = profiles.get(cluster)
        if not caps or profile is None:
            continue
        partitions: Mapping[str, Mapping[str, Any]] = caps.get("partitions") or {}
        account = spec.account or profile.default_account or caps.get("default_account")
        per_partition: dict[str, list[Target]] = {}
        for name, part in partitions.items():
            if spec.partition and name != spec.partition:
                continue
            if not spec.partition and not part.get("accessible", partition_accessible(part, caps)):
                continue
            cands = qos_for_partition(caps, profile, name, spec.qos)
            qos = cands[0] if cands else None
            if spec.resources.gpus > 0:
                types = [t for t in (part.get("gres_type_list") or sorted(t for t in (part.get("gres_types") or {}) if t))]
                if spec.resources.gpu_types:
                    types = [t for t in types if t in spec.resources.gpu_types]
                for gtype in types:
                    per_partition.setdefault(name, []).append(
                        Target(cluster=cluster, partitions=[name], gres_type=gtype, qos=qos, account=account))
            elif not part.get("has_gpu") or spec.partition:
                # A job that asks for no GPUs is not a candidate for a GPU partition unless the user named
                # that partition: GPUs are the scarce resource on both clusters and the GPU rate is what gets
                # charged, so a CPU-only job there wastes an accelerator and burns SUs for nothing.
                per_partition.setdefault(name, []).append(
                    Target(cluster=cluster, partitions=[name], gres_type=None, qos=qos, account=account))
        cluster_targets: list[Target] = [t for ts in per_partition.values() for t in ts]
        for group in profile.partition_groups:
            members = [m for m in group if m in per_partition]
            if len(members) != len(group) or len(group) < 2:
                continue
            first = per_partition[members[0]]
            for t in first:
                if all(any(o.gres_type == t.gres_type for o in per_partition[m]) for m in members[1:]):
                    cluster_targets.append(Target(cluster=cluster, partitions=list(group), gres_type=t.gres_type,
                                                  qos=t.qos, account=account))
        out.extend(cluster_targets)
    result: list[Target] = []
    seen: set[str] = set()
    for tgt in out:
        key = tgt.key
        if key in seen:
            continue
        if filters is not None and not any(_matches_filter(tgt, f) for f in filters):
            continue
        if policy.targets_allow and not any(fnmatch.fnmatchcase(key, g) for g in policy.targets_allow):
            continue
        if any(fnmatch.fnmatchcase(key, g) for g in policy.targets_deny):
            continue
        profile = profiles[tgt.cluster]
        if not target_override(profile, key).get("enabled", True):
            continue
        seen.add(key)
        result.append(tgt)
    return result


# --- feasibility -----------------------------------------------------------------------------------------------

def _count(mapping: Mapping[str, Any] | None, key: str) -> int:
    v = (mapping or {}).get(key)
    if isinstance(v, (list, tuple, set)):
        return len(v)
    return int(v or 0)


def _handles(mapping: Mapping[str, Any] | None, key: str) -> list[str]:
    v = (mapping or {}).get(key)
    return [str(x) for x in v] if isinstance(v, (list, tuple, set)) else []


def _cap_for(policy: PlacementPolicy, profile: ClusterProfile | None, key: str, field: str, override_key: str,
             ) -> int | None:
    caps = [v for g, v in getattr(policy, field).items() if fnmatch.fnmatchcase(key, g)]
    if profile is not None:
        ov = target_override(profile, key).get(override_key)
        if ov is not None:
            caps.append(int(ov))
    return min(caps) if caps else None


def feasibility(target: Target, spec: JobSpec, caps: Mapping[str, Any] | None, snapshot: Mapping[str, Any] | None = None,
                policy: PlacementPolicy | None = None, my_running: Mapping[str, Any] | None = None, *,
                my_pending: Mapping[str, Any] | None = None, target_stats: Mapping[str, Any] | None = None,
                profile: ClusterProfile | None = None, now_local: float | None = None) -> tuple[bool, str]:
    """Hard feasibility of section 8 that needs only caps/snapshot; ``(True, "")`` or ``(False, why)``.

    ``my_running``/``my_pending`` map partition -> count or list of handles (this user's jobs on the cluster);
    ``target_stats`` is the ``target_stats`` row of the target; ``now_local`` is ``time.time()`` (tests pin it).
    """
    import time as _time
    policy = policy or PlacementPolicy()
    now = _time.time() if now_local is None else now_local
    if not caps:
        return False, f"{target.cluster}: not discovered (unreachable or auth failed)"
    partitions: Mapping[str, Mapping[str, Any]] = caps.get("partitions") or {}
    r = spec.resources
    key = target.key
    if target_stats:
        until = target_stats.get("breaker_open_until_local")
        if until and float(until) > now:
            return False, f"circuit breaker open for {int(float(until) - now)} s ({target_stats.get('last_error') or 'repeated failures'})"
        until = target_stats.get("infeasible_until_local")
        if until and float(until) > now:
            return False, f"recently refused ({target_stats.get('infeasible_reason') or 'QOS/limit'}); retry in {int(float(until) - now)} s"
    for pname in target.partitions:
        part = partitions.get(pname)
        if part is None:
            return False, f"unknown partition {pname} on {target.cluster}"
        state = str(part.get("state") or "UP").upper()
        if not state.startswith("UP"):
            return False, f"partition {pname} is {state}"
        if not part.get("accessible", partition_accessible(part, caps)):
            return False, f"partition {pname} not allowed for this user/account"
        limits = effective_limits(caps, pname, target.qos)
        max_wall = limits.get("max_wall_s")
        if max_wall is not None and r.time_s > max_wall:
            return False, f"time {r.time} exceeds max wall {_hms(max_wall)} on {pname}"
        max_nodes = limits.get("max_nodes")
        if max_nodes is not None and r.nodes > max_nodes:
            return False, f"nodes {r.nodes} > MaxNodes {max_nodes} on {pname}"
        tres = limits.get("max_tres_pj") or {}
        if r.gpus > 0:
            types = part.get("gres_types") or {}
            gres_nodes = part.get("gres_nodes") or {}
            if target.gres_type is None:
                return False, "untyped GPU request (candidates are always typed)"
            if (types or gres_nodes) and target.gres_type not in types and target.gres_type not in gres_nodes:
                return False, f"no {target.gres_type} GPUs in {pname}"
            if not part.get("has_gpu", bool(types)):
                return False, f"partition {pname} has no GPUs"
            per_node = (gres_nodes.get(target.gres_type) or {}).get("per_node")
            if per_node and r.gpus > per_node:
                return False, f"gpus {r.gpus} per node > {per_node} {target.gres_type} per node in {pname}"
            total = r.gpus * r.nodes
            cap = tres.get(f"gres/gpu:{target.gres_type}", tres.get("gres/gpu"))
            if cap is not None and total > cap:
                return False, f"gpus x nodes {total} > per-job limit {int(cap)} on {pname}"
        elif tres.get("cpu") is not None and r.cpus and r.cpus * (r.tasks or 1) * r.nodes > tres["cpu"]:
            return False, f"cpus {r.cpus * (r.tasks or 1) * r.nodes} > per-job limit {int(tres['cpu'])} on {pname}"
        max_cpus = limits.get("max_cpus_node")
        if max_cpus and r.cpus and r.cpus * (r.tasks or 1) > max_cpus:
            return False, f"cpus per node {r.cpus * (r.tasks or 1)} > node size {max_cpus} in {pname}"
        max_mem = limits.get("max_mem_mb_node")
        mem_mb = _mem_mb(r.mem)
        if max_mem and mem_mb and mem_mb > float(max_mem):
            return False, f"mem {r.mem} > node memory {int(max_mem)} MB in {pname}"
        running_here = _count(my_running, pname)
        pending_here = _count(my_pending, pname)
        mj, ms = limits.get("max_jobs_pu"), limits.get("max_submit_pu")
        if mj is not None and running_here >= mj:
            return False, f"QOS max_jobs_pu {mj} reached on {pname} ({running_here} running)"
        if ms is not None and running_here + pending_here >= ms:
            return False, f"QOS max_submit_pu {ms} reached on {pname} ({running_here + pending_here} submitted)"
    cap = _cap_for(policy, profile, key, "max_running_per_target", "max_running")
    if cap is not None:
        mine = sum(_count(my_running, p) for p in target.partitions)
        if mine >= cap:
            return False, f"etiquette cap: {mine} running >= max_running {cap} for {key}"
    why = _self_preemption(target, spec, caps, snapshot, policy, my_running, profile)
    if why:
        return False, why
    return True, ""


def _self_preemption(target: Target, spec: JobSpec, caps: Mapping[str, Any], snapshot: Mapping[str, Any] | None,
                     policy: PlacementPolicy, my_running: Mapping[str, Any] | None, profile: ClusterProfile | None,
                     ) -> str | None:
    """Section 8: a higher-PriorityTier partition sharing nodes with a lower-tier one where I have running jobs is
    infeasible unless the target has an idle node carrying the requested gres, or self-preemption is allowed."""
    if policy.allow_self_preempt:
        return None
    if profile is not None and target_override(profile, target.key).get("allow_self_preempt"):
        return None
    partitions: Mapping[str, Mapping[str, Any]] = caps.get("partitions") or {}
    snap_parts = (snapshot or {}).get("partitions") or {}
    for pname in target.partitions:
        part = partitions.get(pname) or {}
        tier = part.get("priority_tier")
        if tier is None:
            continue
        my_nodes = expand_hostlist(part.get("nodes"))
        victims: list[str] = []
        victim_parts: list[str] = []
        for other, opart in partitions.items():
            if other == pname or opart.get("priority_tier") is None or opart["priority_tier"] >= tier:
                continue
            if my_nodes and not (my_nodes & expand_hostlist(opart.get("nodes"))):
                continue
            n = _count(my_running, other)
            if n:
                victim_parts.append(other)
                victims += _handles(my_running, other) or [f"{n} job(s)"]
        if not victim_parts:
            continue
        snap = snap_parts.get(pname) or {}
        if spec.resources.gpus > 0:
            idle = (snap.get("idle_gres") or {}).get(target.gres_type, 0) if snap else 0
        else:
            idle = (snap.get("nodes") or {}).get("idle", 0) if snap else 0
        if idle > 0:
            continue
        return (f"would preempt my own {'/'.join(victim_parts)} job(s) ({', '.join(victims)}); "
                f"set placement.allow_self_preempt or target_overrides['{target.key}'].allow_self_preempt")
    return None


def _hms(seconds: int) -> str:
    from .clock import format_duration
    return format_duration(seconds) or str(seconds)


# --- depth and provisional ranking ------------------------------------------------------------------------------

def queue_depth(snapshot: Mapping[str, Any] | None, caps: Mapping[str, Any] | None, target: Target,
                spec: JobSpec | None = None) -> tuple[int | None, int | None]:
    """``(depth_typed, depth_untyped)`` pending rows of the target's partitions (section 8 "Estimates"):
    typed = rows requesting this gres type (or CPU rows for a CPU job), untyped = rows with ``N/A`` in both TRES
    views of a GPU partition (counted against every type). ``(None, None)`` without a snapshot."""
    if not snapshot or "partitions" not in snapshot:
        return None, None
    gpus = spec.resources.gpus if spec is not None else (1 if target.gres_type else 0)
    typed = untyped = 0
    for pname in target.partitions:
        pending = ((snapshot.get("partitions") or {}).get(pname) or {}).get("pending") or {}
        if gpus > 0:
            typed += int(pending.get(target.gres_type, 0))
            untyped += int(pending.get(None, 0))
        else:
            typed += int(pending.get("cpu", 0))
            untyped += int(pending.get(None, 0))
    return typed, untyped


def rank(spec: JobSpec, caps_by_cluster: Mapping[str, Mapping[str, Any] | None],
         snapshots: Mapping[str, Mapping[str, Any] | None], profiles: Mapping[str, ClusterProfile],
         policy: PlacementPolicy | None = None, *, my_running: Mapping[str, Mapping[str, Any]] | None = None,
         my_pending: Mapping[str, Mapping[str, Any]] | None = None,
         target_stats: Mapping[str, Mapping[str, Any]] | None = None, placement: str | Sequence[str] | None = "auto",
         inputs_cluster: str | None = None, max_options: int = 6, now_local: float | None = None) -> list[PlanOption]:
    """Provisional ranking WITHOUT test-only/history/cost (see the module docstring): feasible options sorted by
    ``score_h``, then up to 3 infeasible rows with their ``why``. ``my_running``/``my_pending``/``target_stats``
    are keyed by cluster (then partition / target key)."""
    policy = policy or PlacementPolicy()
    hours = spec.resources.time_s / 3600.0
    feasible: list[PlanOption] = []
    infeasible: list[PlanOption] = []
    for tgt in candidates(spec, caps_by_cluster, profiles, policy, placement):
        caps = caps_by_cluster.get(tgt.cluster) or {}
        snap = (snapshots or {}).get(tgt.cluster)
        profile = profiles.get(tgt.cluster)
        running = (my_running or {}).get(tgt.cluster)
        pending = (my_pending or {}).get(tgt.cluster)
        stats = ((target_stats or {}).get(tgt.cluster) or {}).get(tgt.key)
        ok, why = feasibility(tgt, spec, caps, snap, policy, running, my_pending=pending, target_stats=stats,
                              profile=profile, now_local=now_local)
        typed, untyped = queue_depth(snap, caps, tgt, spec)
        p0 = tgt.partitions[0]
        pcaps = ((caps.get("partitions") or {}).get(p0)) or {}
        charge = charge_for(caps, profile, p0, tgt.gres_type) if profile is not None else "free"
        opt = PlanOption(target=tgt.key, feasible=ok, charge=charge, why=why,
                         requeueable=requeueable(spec, pcaps, caps) if pcaps else None)
        if typed is not None:
            ahead = typed + (untyped or 0)
            opt.queue_ahead = ahead
            opt.queue_ahead_untyped = untyped
            opt.est_wait_h = min(policy.unknown_wait_h, DEPTH_HOURS_PER_JOB * ahead)
            opt.est_wait_src = "depth"
        else:
            opt.est_wait_h = policy.unknown_wait_h
            opt.est_wait_src = "none"
        if not ok:
            infeasible.append(opt)
            continue
        etiquette = 0.0
        soft = _cap_for(policy, profile, tgt.key, "soft_caps", "soft_cap")
        if soft is not None and sum(_count(running, p) for p in tgt.partitions) >= soft:
            etiquette = policy.etiquette_h
        opt.etiquette_h = etiquette
        penalty = float(target_override(profile, tgt.key).get("penalty_h", 0.0)) if profile is not None else 0.0
        staging = STAGING_PENALTY_H if (inputs_cluster and inputs_cluster != tgt.cluster) else 0.0
        opt.score_h = round((opt.est_wait_h or 0.0) + hours + etiquette + penalty + staging, 3)
        parts = [f"queue ahead {opt.queue_ahead}" + (f" ({untyped} untyped)" if untyped else "")
                 if opt.queue_ahead is not None else "queue depth unknown",
                 f"est wait {opt.est_wait_h:.1f} h ({opt.est_wait_src})"]
        if etiquette:
            parts.append(f"etiquette +{etiquette:g} h")
        if staging:
            parts.append("inputs elsewhere +0.5 h")
        opt.why = "; ".join(parts)
        feasible.append(opt)
    feasible.sort(key=lambda o: (o.score_h if o.score_h is not None else 1e9, o.queue_ahead or 0, o.target))
    return feasible[: max(1, int(max_options))] + infeasible[:MAX_INFEASIBLE_ROWS]


def recommended(options: Iterable[PlanOption]) -> str | None:
    for o in options:
        if o.feasible:
            return o.target
    return None





# ======================================================================================================
# Phase 3 (design section 8): cost, worst case, preemption risk, history, the full score and rebalancing.
# ======================================================================================================

SU_TO_HOURS = {"balanced": 0.25, "fastest": 0.02, "cheapest": 2.0}
HIST_MIN_SAMPLES = 3
HIST_WINDOW_S = 30 * 86400
TIE_H = 0.25
RISK_BASE_PCT = 10.0
RISK_PER_PENDING_PCT = 5.0
RISK_MAX_PCT = 80.0
RISK_NODE_FAIL_PCT = 2.0


def cost_units(spec: JobSpec, caps: Mapping[str, Any], partition: str) -> float:
    """Billable units for one run (section 8 Cost): GPUs on a shared partition, whole-node GPUs on an exclusive
    one, else cores. ``OverSubscribe=EXCLUSIVE`` marks the whole-node partitions."""
    part = (caps.get("partitions") or {}).get(partition) or {}
    gpus = int(spec.resources.gpus or 0)
    nodes = max(1, int(spec.resources.nodes or 1))
    exclusive = str(part.get("over_subscribe") or "").upper().startswith("EXCLUSIVE")
    if gpus > 0:
        per_node = int(part.get("gres_per_node") or gpus)
        return float(per_node * nodes) if exclusive else float(gpus * nodes)
    cpus = int(spec.resources.cpus or spec.resources.tasks or 1)
    if exclusive:
        cpus = int(part.get("cpus_per_node") or cpus)
    return float(cpus * nodes)


def cost_su(spec: JobSpec, caps: Mapping[str, Any], profile: ClusterProfile | None, target: Target):
    """``(su_for_one_run, charge)``; ``su`` is None when the cluster does not charge (section 8 Cost)."""
    partition = target.partitions[0]
    charge = charge_for(caps, profile, partition, target.gres_type) if profile is not None else "free"
    if charge == "free" or not isinstance(charge, Mapping):
        return None, charge
    rate = float(charge.get("su_per_unit_h") or 0.0)
    hours = spec.resources.time_s / 3600.0
    return round(rate * cost_units(spec, caps, partition) * hours, 2), charge


def worst_case_su(one_run, spec: JobSpec, requeueable_flag):
    """``cost x (1 + max_restarts)`` when the attempt can be requeued, else the single-run cost (section 8)."""
    if one_run is None:
        return None
    return round(one_run * (1 + int(spec.max_restarts)), 2) if requeueable_flag else one_run


def preempt_risk(target: Target, spec: JobSpec, caps: Mapping[str, Any],
                 snapshot: Mapping[str, Any] | None, stats: Mapping[str, Any] | None = None):
    """``(risk_pct, expected_rework_hours)`` for a preemptible partition (section 8 Risk).

    No checkpoint credit unless the discovered ``GraceTime`` covers ``spec.grace_s`` or ``PreemptParameters``
    carries ``send_user_signal``: on both target clusters GraceTime is 0, so preemption is an immediate kill
    and no signal handler runs.
    """
    partition = target.partitions[0]
    part = (caps.get("partitions") or {}).get(partition) or {}
    modes = {str(m).upper() for m in (part.get("preempt_mode") or [])}
    if not modes & {"REQUEUE", "CANCEL"}:
        return 0.0, 0.0
    tier = int(part.get("priority_tier") or 1)
    nodes = expand_hostlist(part.get("nodes_expr") or part.get("nodes"))
    higher = []
    for name, other in (caps.get("partitions") or {}).items():
        if name == partition or int(other.get("priority_tier") or 1) <= tier:
            continue
        if not nodes or (expand_hostlist(other.get("nodes_expr") or other.get("nodes")) & nodes):
            higher.append(name)
    if not higher:
        return 0.0, 0.0
    pending = 0
    for name in higher:
        entry = ((snapshot or {}).get("by_partition") or {}).get(name) or {}
        pending += int(entry.get("pending_total") or 0)
    pct = min(RISK_MAX_PCT, RISK_BASE_PCT + RISK_PER_PENDING_PCT * pending)
    if stats and stats.get("last_node_fail_local"):
        pct = min(RISK_MAX_PCT, pct + RISK_NODE_FAIL_PCT)
    hours = spec.resources.time_s / 3600.0
    if spec.checkpoint_interval_h:
        lost = min(hours, float(spec.checkpoint_interval_h))
    else:
        lost = hours / 2.0
    return round(pct, 1), round(pct / 100.0 * lost, 3)


def history_wait_h(rows, target_key: str, now_ts=None):
    """Median observed wait for a target over the last 30 days (>= 3 samples), else None (section 8)."""
    if not rows:
        return None
    cutoff = (now_ts or 0) - HIST_WINDOW_S
    waits = sorted(float(r["wait_s"]) for r in rows
                   if r.get("target_key") == target_key and float(r.get("start_ts") or 0) >= cutoff)
    if len(waits) < HIST_MIN_SAMPLES:
        return None
    mid = len(waits) // 2
    median = waits[mid] if len(waits) % 2 else (waits[mid - 1] + waits[mid]) / 2.0
    return round(median / 3600.0, 3)


def apply_estimate(option: PlanOption, *, est_start_ts, now_ts, hist_h=None) -> PlanOption:
    """Fold a ``sbatch --test-only`` start time (and optional history) into an option (section 8 Estimates).

    ``--test-only`` wins over depth and is never overridden downwards; history replaces it only when far lower
    (under a third), and the test-only number is then quoted in ``why``.
    """
    if est_start_ts is not None and now_ts is not None:
        wait_h = max(0.0, (int(est_start_ts) - int(now_ts)) / 3600.0)
        option.est_start_ts = int(est_start_ts)
        option.est_wait_h = round(wait_h, 3)
        option.est_wait_src = "test_only"
        if hist_h is not None and hist_h < wait_h / 3.0:
            option.est_wait_h = hist_h
            option.est_wait_src = "history"
            option.why = (option.why + "; test-only said %.1f h, history %.1f h" % (wait_h, hist_h)).strip("; ")
    elif hist_h is not None:
        option.est_wait_h = hist_h
        option.est_wait_src = "history"
    return option


def score(option: PlanOption, spec: JobSpec, policy: PlacementPolicy, *, staging_h: float = 0.0,
          penalty_h: float = 0.0, risk_h: float = 0.0) -> float:
    """Section 8 Score: wait + wall clock + expected rework + SU-weighted cost + etiquette + penalties."""
    hours = spec.resources.time_s / 3600.0
    su_to_hours = policy.su_to_hours
    if su_to_hours is None:
        su_to_hours = SU_TO_HOURS.get(policy.objective, SU_TO_HOURS["balanced"])
    cost_h = float(option.cost_su or 0.0) * float(su_to_hours)
    return round((option.est_wait_h or 0.0) + hours + risk_h + cost_h + option.etiquette_h
                 + staging_h + penalty_h, 3)


def su_feasible(worst, caps: Mapping[str, Any], policy: PlacementPolicy):
    """The ``cost_worst_su <= balance - reserve`` rule (section 8 Feasibility); an unknown balance passes."""
    if worst is None:
        return True, ""
    balance = caps.get("su_balance")
    if balance is None:
        return True, ""
    left = float(balance) - float(policy.su_reserve)
    if worst > left:
        return False, ("worst case %g SU exceeds the usable balance (%g SU - %g reserved = %g SU)"
                       % (worst, float(balance), policy.su_reserve, left))
    return True, ""


def maintenance_conflict(caps: Mapping[str, Any], target: Target, spec: JobSpec, est_wait_h, now_ts) -> str:
    """Non-empty when a MAINT reservation covering the target starts before the job would finish (section 8)."""
    if now_ts is None:
        return ""
    finish = int(now_ts) + int((est_wait_h or 0.0) * 3600) + int(spec.resources.time_s)
    for resv in caps.get("reservations") or []:
        if not resv.get("maint"):
            continue
        parts = set(resv.get("partitions") or [])
        if parts and not (parts & set(target.partitions)):
            continue
        start = resv.get("start_ts")
        if start and int(now_ts) < int(start) < finish:
            return "maintenance reservation %s starts before the job would finish" % (resv.get("name"),)
    return ""


def enrich_options(options, spec: JobSpec, caps_by_cluster: Mapping[str, Any],
                   snapshots: Mapping[str, Any], profiles: Mapping[str, ClusterProfile],
                   policy: PlacementPolicy, *, history=None, target_stats=None, now_by_cluster=None,
                   inputs_cluster: str | None = None):
    """Add cost, worst-case cost, risk, history and the full score to provisional options (section 8).

    Applied after :func:`rank`; ``--test-only`` estimates are folded in separately by the caller (which owns
    the SSH budget of at most four passes) through :func:`apply_estimate`.
    """
    policy = policy or PlacementPolicy()
    out = []
    for opt in options:
        tgt = resolve_explicit(opt.target, profiles)
        caps = caps_by_cluster.get(tgt.cluster) or {}
        profile = profiles.get(tgt.cluster)
        snap = (snapshots or {}).get(tgt.cluster)
        stats = ((target_stats or {}).get(tgt.cluster) or {}).get(tgt.key)
        now_ts = (now_by_cluster or {}).get(tgt.cluster)
        one, charge = cost_su(spec, caps, profile, tgt)
        opt.cost_su = one
        # PlanOption.charge is the Charge model or the literal "free"; charge_for returns a plain mapping
        opt.charge = charge if charge == "free" else Charge.model_validate(dict(charge))
        opt.cost_worst_su = worst_case_su(one, spec, opt.requeueable)
        pct, risk_h = preempt_risk(tgt, spec, caps, snap, stats)
        opt.risk_pct = pct or None
        hist = history_wait_h((history or {}).get(tgt.cluster), tgt.key, now_ts)
        if opt.est_wait_src == "depth" and hist is not None:
            opt.est_wait_h = hist
            opt.est_wait_src = "history"
        if opt.feasible:
            ok, why = su_feasible(opt.cost_worst_su, caps, policy)
            if not ok:
                opt.feasible, opt.why = False, why
            else:
                clash = maintenance_conflict(caps, tgt, spec, opt.est_wait_h, now_ts)
                if clash:
                    opt.feasible, opt.why = False, clash
        if opt.feasible:
            penalty = float(target_override(profile, tgt.key).get("penalty_h", 0.0)) if profile else 0.0
            staging = STAGING_PENALTY_H if (inputs_cluster and inputs_cluster != tgt.cluster) else 0.0
            opt.score_h = score(opt, spec, policy, staging_h=staging, penalty_h=penalty, risk_h=risk_h)
            bits = [opt.why] if opt.why else []
            if one is not None:
                bits.append("%g SU" % one + (" (worst %g)" % opt.cost_worst_su
                                             if opt.cost_worst_su and opt.cost_worst_su != one else ""))
            if pct:
                bits.append("preempt risk %g%%" % pct)
            opt.why = "; ".join(b for b in bits if b)
        out.append(opt)
    feasible = [o for o in out if o.feasible]
    infeasible = [o for o in out if not o.feasible]
    feasible.sort(key=_tie_key(policy, profiles, inputs_cluster))
    return feasible + infeasible


def _tie_key(policy: PlacementPolicy, profiles: Mapping[str, ClusterProfile], inputs_cluster: str | None):
    """Sort by score, then the section 8 tie-breaks within ``TIE_H``: non-preemptible, data locality, the
    preferred cluster, then the shorter queue."""
    def key(o: PlanOption):
        bucket = round((o.score_h or 1e9) / TIE_H)
        cluster = o.target.split(":", 1)[0]
        return (bucket, float(o.risk_pct or 0.0), 0 if cluster == inputs_cluster else 1,
                0 if cluster == policy.prefer_cluster else 1, o.queue_ahead or 0, o.score_h or 1e9, o.target)
    return key


__all__ = ["MAX_INFEASIBLE_ROWS", "DEPTH_HOURS_PER_JOB", "STAGING_PENALTY_H", "SU_TO_HOURS", "TIE_H",
           "expand_hostlist", "resolve_explicit", "candidates", "feasibility", "queue_depth", "rank",
           "recommended", "cost_units", "cost_su", "worst_case_su", "preempt_risk", "history_wait_h",
           "apply_estimate", "score", "su_feasible", "maintenance_conflict", "enrich_options"]
