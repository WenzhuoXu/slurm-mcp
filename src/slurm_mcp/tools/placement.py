"""Placement tools: ``plan_job`` and ``rebalance`` (design section 4 "Jobs", section 5.4, section 8).

``plan_job`` is read-only: ``sbatch --test-only`` validates a script and returns an estimated start without
creating a job. ``rebalance`` proposes moves for pending auto-placed jobs and, with ``dry_run=False``, applies
them in the section 5.4 order (submit the new attempt, re-check the old one, cancel the loser).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .. import _mcp
from .._mcp import Context, MCPServer
from ..errors import err
from ..models import JobSpec, PlanResult, RebalanceProposal, RebalanceResult
from ..placer import apply_estimate, enrich_options, rank, recommended, resolve_explicit
from ..slurm.states import JobState
from . import BIG_RESULT_META, run_tool

log = logging.getLogger("slurm_mcp.tools.placement")

MAX_TEST_ONLY = 4               # section 8: at most four --test-only passes per decision
PLAN_TTL_S = 15 * 60            # section 4: a plan is valid 15 minutes

PLAN_DESC = (
    "Rank the places a job could run, without submitting it. Returns one row per candidate target "
    "('cluster:partition[:gpu-type][@qos]') with feasible, est_wait_h and its source (test_only from "
    "sbatch --test-only, history from your past waits, depth from the queue, none), est_start_ts, queue_ahead "
    "(queue_ahead_untyped counts pending jobs that asked for GPUs without naming a type), cost_su for one run "
    "and cost_worst_su if it can be requeued, risk_pct of preemption, etiquette_h, score_h (lower is better) "
    "and a why string quoting the numbers. Infeasible targets are listed last with the reason (walltime over "
    "the QOS limit, not enough GPUs of that type, SU balance too low, maintenance, a partition that would "
    "preempt your own jobs). Also returns plan_id (valid 15 min) to pass to submit_job, and a preview of the "
    "rendered script. Read-only: no job is created and nothing is charged."
)
REBALANCE_DESC = (
    "Re-evaluate pending auto-placed jobs against the queue right now and move the ones that would start "
    "meaningfully sooner elsewhere. dry_run=True (default) only proposes: each row gives the current and new "
    "target, est_wait_now_h vs est_wait_new_h, gain_h, cost_delta_su and why. dry_run=False applies them in "
    "the safe order -- submit the new attempt first, re-check that the old job is still pending, then cancel "
    "the old one -- so a job is never lost; if the re-check is inconclusive both are kept and retried on the "
    "next tick, and a job that started meanwhile keeps running while the new attempt is cancelled instead. "
    "Handles stay stable across a move (j17 is still j17 on the new cluster). Jobs you held, jobs with "
    "dependencies pinning them, and jobs that already moved max_moves_per_job times are skipped with a reason."
)


async def plan_job(service: Any, job: JobSpec | dict, placement: Any = "auto", max_options: int = 6,
                   progress: Any = None) -> PlanResult:
    """Section 4 ``plan_job``: rank targets, price them, and estimate the top few with ``--test-only``."""
    spec = job if isinstance(job, JobSpec) else JobSpec.model_validate(job)
    ctx = await _planning_context(service, spec)
    options = rank(spec, ctx["caps"], ctx["snapshots"], ctx["profiles"], ctx["policy"],
                   my_running=ctx["running"], my_pending=ctx["pending"], target_stats=ctx["stats"],
                   placement=placement, inputs_cluster=ctx["inputs_cluster"], max_options=max_options)
    options = enrich_options(options, spec, ctx["caps"], ctx["snapshots"], ctx["profiles"], ctx["policy"],
                             history=ctx["history"], target_stats=ctx["stats"],
                             now_by_cluster=ctx["now"], inputs_cluster=ctx["inputs_cluster"])
    options = await _estimate_top(service, spec, options, ctx, progress=progress)
    plan_id = await _store_plan(service, spec, options)
    best = recommended(options)
    preview = await _preview(service, spec, best, ctx)
    feasible = [o for o in options if o.feasible]
    if feasible:
        head = feasible[0]
        summary = (f"{len(feasible)} feasible target(s); best {head.target}: "
                   f"wait ~{head.est_wait_h:.1f} h ({head.est_wait_src})"
                   + (f", {head.cost_su:g} SU" if head.cost_su else ", free"))
    else:
        summary = f"no feasible target for {spec.name}: " + "; ".join(f"{o.target}: {o.why}" for o in options[:3])
    return PlanResult(summary=summary, plan_id=plan_id, options=options, recommended=best,
                      rendered_preview=preview, expires_ts=int(time.time() + PLAN_TTL_S),
                      unread_events=await service.unread(),
                      next=(f"submit_job(plan_id='{plan_id}')" if best else
                            "relax the spec (shorter time, fewer GPUs) or configure(placement=...)"))


async def rebalance(service: Any, ids: list[str] | None = None, dry_run: bool = True,
                    min_gain_h: float | None = None, wait_s: float = 90.0) -> RebalanceResult:
    """Section 4 ``rebalance`` / section 5.4: propose (and optionally apply) moves for pending jobs."""
    policy = await service.placement_policy()
    rp = policy.rebalance
    gain_needed = float(min_gain_h if min_gain_h is not None else rp.min_gain_h) + float(rp.hysteresis_h)
    store = service.store
    rows = await store.read(lambda c: store.list_jobs(c, states=[JobState.SUBMITTED], kind="job"))
    if ids:
        wanted = set(ids)
        rows = [r for r in rows if r["handle"] in wanted]
    proposals: list[RebalanceProposal] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        why = _ineligible(row, rp)
        if why:
            skipped.append({"handle": row["handle"], "why": why})
            continue
        spec = JobSpec.model_validate(_spec_of(row))
        ctx = await _planning_context(service, spec)
        current = row.get("target_json") and resolve_explicit(_target_key(row), ctx["profiles"])
        options = rank(spec, ctx["caps"], ctx["snapshots"], ctx["profiles"], ctx["policy"],
                       my_running=ctx["running"], my_pending=ctx["pending"], target_stats=ctx["stats"],
                       placement="auto", inputs_cluster=row.get("cluster"), max_options=4)
        options = enrich_options(options, spec, ctx["caps"], ctx["snapshots"], ctx["profiles"], ctx["policy"],
                                 history=ctx["history"], target_stats=ctx["stats"], now_by_cluster=ctx["now"],
                                 inputs_cluster=row.get("cluster"))
        now_wait = _current_wait_h(row, ctx)
        best = next((o for o in options if o.feasible and o.target != (current.key if current else None)), None)
        if best is None or best.est_wait_h is None:
            skipped.append({"handle": row["handle"], "why": "no better feasible target"})
            continue
        gain = round(now_wait - float(best.est_wait_h), 2)
        delta = round(float(best.cost_worst_su or 0.0) - float(row.get("cost_worst_su") or 0.0), 2)
        will = bool(gain >= gain_needed and delta <= float(rp.max_extra_su))
        proposals.append(RebalanceProposal(
            handle=row["handle"], from_target=current.key if current else (row.get("target_json") or ""),
            to_target=best.target, est_wait_now_h=round(now_wait, 2), est_wait_new_h=round(best.est_wait_h, 2),
            gain_h=gain, cost_delta_su=delta, will_move=will,
            why=(f"{best.target} would start ~{gain:.1f} h sooner" if will else
                 f"gain {gain:.1f} h is below the {gain_needed:.1f} h threshold"
                 if gain < gain_needed else f"would cost {delta:g} SU more than allowed")))
    moved: list[str] = []
    moving: list[str] = []
    if not dry_run:
        mover = service.components.get("submitter")
        for p in [x for x in proposals if x.will_move]:
            if mover is None or not hasattr(mover, "move"):
                skipped.append({"handle": p.handle, "why": "moves are not available in this build"})
                continue
            try:
                done = await mover.move(p.handle, p.to_target, wait_s=wait_s)
                (moved if done else moving).append(p.handle)
            except Exception as e:                      # a failed move never loses the old attempt (5.4)
                skipped.append({"handle": p.handle, "why": f"move failed: {e}"})
    verb = "would move" if dry_run else "moved"
    n = sum(1 for p in proposals if p.will_move) if dry_run else len(moved)
    return RebalanceResult(
        summary=(f"{len(proposals)} candidate(s), {verb} {n}"
                 + (f", {len(moving)} still moving" if moving else "")
                 + (f", {len(skipped)} skipped" if skipped else "")),
        proposals=proposals, skipped=skipped, moved=moved, moving=moving,
        unread_events=await service.unread(),
        next=("rebalance(dry_run=False) to apply" if dry_run and n else None))


# --- helpers ------------------------------------------------------------------------------------------------

async def _planning_context(service: Any, spec: JobSpec) -> dict[str, Any]:
    """Caps, snapshots, my running/pending counts, wait history, target stats and cluster clocks."""
    names = [spec.cluster] if spec.cluster else list(service.registry.names())
    caps: dict[str, Any] = {}
    snapshots: dict[str, Any] = {}
    now: dict[str, Any] = {}
    for name in names:
        try:
            caps[name] = await service.caps(name)
            snapshots[name] = await service.snapshot(name)
            client = service.client(name)
            now[name] = int(client.clock.remote_now())
        except Exception as e:                          # an unreachable cluster is simply not a candidate
            log.info("planning: skipping %s (%s)", name, e)
    store = service.store
    history = {}
    stats: dict[str, Any] = {}
    for name in caps:
        history[name] = await store.read(lambda c, n=name: store.wait_history_all(c, n)) \
            if hasattr(store, "wait_history_all") else []
        stats[name] = {}
    running, pending = await service.my_counts(names) if hasattr(service, "my_counts") else ({}, {})
    return {"caps": caps, "snapshots": snapshots, "now": now, "profiles": service.profiles(),
            "policy": await service.placement_policy(), "history": history, "stats": stats,
            "running": running, "pending": pending,
            "inputs_cluster": spec.cluster or (names[0] if len(names) == 1 else None)}


async def _estimate_top(service: Any, spec: JobSpec, options: list, ctx: dict[str, Any],
                        progress: Any = None) -> list:
    """One ``sbatch --test-only`` per target for the best few (section 8); failures leave the row untouched."""
    feasible = [o for o in options if o.feasible][:MAX_TEST_ONLY]
    if not feasible:
        return options
    submitter = service.components.get("submitter")
    estimator = getattr(submitter, "estimate_target", None) if submitter else None
    if estimator is None:
        return options
    for i, opt in enumerate(feasible):
        if progress:
            try:
                await progress((i + 1) / len(feasible), f"test-only {i + 1}/{len(feasible)}")
            except Exception:
                pass
        try:
            est = await estimator(spec, opt.target)
        except Exception as e:
            log.info("test-only failed for %s: %s", opt.target, e)
            continue
        if est is None:
            continue
        if est.get("infeasible"):
            opt.feasible, opt.why = False, str(est.get("reason") or "sbatch --test-only refused the job")
            continue
        cluster = opt.target.split(":", 1)[0]
        apply_estimate(opt, est_start_ts=est.get("est_start_ts"), now_ts=ctx["now"].get(cluster))
    ordered = [o for o in options if o.feasible]
    ordered.sort(key=lambda o: (o.score_h if o.score_h is not None else 1e9, o.est_wait_h or 0.0, o.target))
    return ordered + [o for o in options if not o.feasible]


async def _store_plan(service: Any, spec: JobSpec, options: list) -> str:
    store = service.store

    def fn(conn: Any) -> str:
        return store.insert_plan(conn, spec_json=spec.model_dump(mode="json"),
                                 options_json=[o.model_dump(mode="json") for o in options],
                                 recommended=recommended(options), ttl_s=PLAN_TTL_S)
    return await store.write(fn)


async def _preview(service: Any, spec: JobSpec, target: str | None, ctx: dict[str, Any]) -> str:
    """First lines of the rendered script for the recommended target (best effort)."""
    submitter = service.components.get("submitter")
    fn = getattr(submitter, "preview", None) if submitter else None
    if fn is None or target is None:
        return ""
    try:
        text = await fn(spec, target)
        return "\n".join(str(text).splitlines()[:25])
    except Exception as e:
        log.info("preview failed: %s", e)
        return ""


def _spec_of(row: Any) -> dict[str, Any]:
    import json
    raw = row.get("spec_json")
    if isinstance(raw, str):
        try:
            return dict(json.loads(raw))
        except ValueError:
            return {}
    return dict(raw or {})


def _target_key(row: Any) -> str:
    import json
    raw = row.get("target_json")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except ValueError:
            return raw
    else:
        data = raw or {}
    if isinstance(data, str):
        return data
    parts = ",".join(data.get("partitions") or [])
    key = f"{data.get('cluster')}:{parts}"
    if data.get("gres_type"):
        key += f":{data['gres_type']}"
    if data.get("qos"):
        key += f"@{data['qos']}"
    return key


def _ineligible(row: Any, rp: Any) -> str:
    """Section 5.4 eligibility, as a reason string (empty when the job may move)."""
    if row.get("placement_mode") != "auto":
        return "placement is pinned (hold or an explicit target)"
    if row.get("hold_reason"):
        return "job is held"
    if int(row.get("moves") or 0) >= int(rp.max_moves_per_job):
        return f"already moved {row.get('moves')} times"
    if row.get("depends_on_json") not in (None, "", "[]", []):
        return "dependencies pin the cluster"
    submit = row.get("submit_ts") or 0
    if submit and (time.time() - float(submit)) < float(rp.min_age_min) * 60:
        return f"submitted less than {rp.min_age_min} min ago"
    return ""


def _current_wait_h(row: Any, ctx: dict[str, Any]) -> float:
    """How much longer the job is expected to wait where it is: its own estimate, else the policy default."""
    est = row.get("est_start_ts")
    now = ctx["now"].get(row.get("cluster"))
    if est and now:
        return max(0.0, (int(est) - int(now)) / 3600.0)
    return float(ctx["policy"].unknown_wait_h)


def register(mcp: MCPServer, service: Any) -> None:
    """Register ``plan_job`` (read-only) and ``rebalance`` (mutating; destructive when applied)."""

    @mcp.tool(name="plan_job", description=PLAN_DESC, annotations=_mcp.read_only(), meta=BIG_RESULT_META)
    async def plan_tool(job: JobSpec, placement: Any = "auto", max_options: int = 6,
                        ctx: Context | None = None) -> PlanResult:
        return await run_tool(plan_job(service, job, placement, max_options, _progress(ctx)))

    @mcp.tool(name="rebalance", description=REBALANCE_DESC, annotations=_mcp.mutating())
    async def rebalance_tool(ids: Optional[list[str]] = None, dry_run: bool = True,
                             min_gain_h: Optional[float] = None, wait_s: float = 90.0) -> RebalanceResult:
        return await run_tool(rebalance(service, ids, dry_run, min_gain_h, wait_s))


def _progress(ctx: Context | None) -> Any:
    if ctx is None:
        return None

    async def cb(fraction: float, message: str) -> None:
        try:
            await ctx.report_progress(float(fraction), 1.0, message)
        except Exception:
            pass
    return cb


__all__ = ["plan_job", "rebalance", "register", "MAX_TEST_ONLY", "PLAN_TTL_S"]
