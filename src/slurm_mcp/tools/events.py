"""Events tool: ``wait_for_events`` (design section 4 "Events", section 5.6, section 11g) and the attachment of the
Monitor component.

``register(mcp, service)`` attaches ``Monitor(service)`` when the service is already bound (tests, the CLI mirror)
and registers the tool; with the server's ``ServiceProxy`` the lifespan attaches the Monitor itself
(``server.attach_optional_components``). ``wait_for_events_op`` is the Service-level operation (usable by the CLI
without MCP): deliver-then-ack on top of ``EventBus.wait`` with the section 4 result fields, the 600 s server cap,
progress every 30 s and the ``snapshot`` counts.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .. import _mcp
from .._mcp import Context, MCPServer
from ..clock import parse_duration
from ..events import DEFAULT_MAX_EVENTS, DEFAULT_POLL_S, EventBus
from ..models import EventRow, EventSnapshot, EventsResult
from ..slurm.states import JobState
from . import run_tool

log = logging.getLogger("slurm_mcp.tools.events")

MAX_TIMEOUT_S = 600
PROGRESS_EVERY_S = DEFAULT_POLL_S
PENDING_STATES = (JobState.UPLOADING, JobState.SUBMITTING, JobState.SUBMITTED)
RUNNING_STATES = (JobState.RUNNING, JobState.COMPLETING)
SUBMIT_STATES = (JobState.UPLOADING, JobState.SUBMITTING)

WAIT_DESC = (
    "Long-poll the durable event log: returns at once when unacknowledged events match, else blocks until one "
    "arrives or timeout_s (server cap 600; progress every 30 s so Claude Code backgrounds the call after 2 min "
    "and returns the result as a task notification). Kinds: queued, submitted, submit_failed, started, completed, "
    "failed, timeout, oom, cancelled, preempted, node_fail, lost, requeued, rebalanced, held, dependency_updated, "
    "needs_attention, alloc_ready, alloc_expiring, alloc_ended, cmd_done, transfer_done, transfer_failed, "
    "cluster_unreachable, cluster_recovered, quota_warning. Deliver-then-ack: returning never consumes an event; "
    "pass the previous result's next_seq as ack_seq on the NEXT call to acknowledge exactly that delivery (a "
    "lost/compacted result is simply replayed). since_seq without ack_seq is a pure re-read (timeout_s=0 lists). "
    "kinds/job_ids filter the delivery but never acknowledge or hide the rest: unread_events counts every "
    "unacknowledged event, unread_unmatched those hidden by the filters. client_id defaults to this server's "
    "session_id (see clusters()). snapshot gives queued/pending/running/alloc_ready/transfers_running/"
    "submits_running counts. Never poll faster than 30 s; use timeout_s >= 300 to wait for a job."
)


def _hms(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


async def _snapshot(service: Any) -> tuple[EventSnapshot, str]:
    """Counts for ``EventsResult.snapshot`` and the parenthetical of the progress message."""
    store = service.store

    def fn(conn: Any) -> tuple[EventSnapshot, list[dict[str, Any]]]:
        rows = store.list_jobs(conn, states=[JobState.QUEUED, *PENDING_STATES, *RUNNING_STATES],
                               order_by="start_ts, handle")
        snap = EventSnapshot()
        running: list[dict[str, Any]] = []
        for r in rows:
            st = r.get("state")
            if st == JobState.QUEUED.value:
                snap.queued += 1
            elif st in {s.value for s in PENDING_STATES}:
                snap.pending += 1
            else:
                snap.running += 1
                running.append(r)
            if st in {s.value for s in SUBMIT_STATES}:
                snap.submits_running += 1
            if r.get("kind") == "alloc" and r.get("alloc_ready"):
                snap.alloc_ready += 1
        try:
            snap.transfers_running = store.count(conn, "transfers", state="running")
        except Exception:
            snap.transfers_running = 0
        return snap, running

    try:
        snap, running = await store.read(fn)
    except Exception as e:
        log.debug("snapshot counts failed: %s", e)
        return EventSnapshot(), ""
    detail = ""
    for r in running:
        start = r.get("start_ts")
        if start is None:
            continue
        try:
            now = service.clock(r["cluster"]).remote_now()
        except Exception:
            now = int(time.time())
        limit = None
        try:
            import json
            spec = json.loads(r.get("spec_json") or "{}")
            limit = parse_duration(((spec.get("resources") or {}).get("time")))
        except Exception:
            limit = None
        detail = f" ({r['handle']} {_hms(now - int(start))}/{_hms(limit)})"
        break
    return snap, detail


def _next_hint(result: EventsResult, kinds: list[str] | None, job_ids: list[str] | None, timeout_s: int) -> str:
    args: list[str] = []
    if result.next_seq is not None:
        args.append(f"ack_seq={result.next_seq}")
    if kinds:
        args.append(f"kinds={kinds!r}")
    if job_ids:
        args.append(f"job_ids={job_ids!r}")
    args.append(f"timeout_s={max(300, min(int(timeout_s) or 300, MAX_TIMEOUT_S))}")
    return f"wait_for_events({', '.join(args)})"


async def wait_for_events_op(service: Any, *, timeout_s: int = 300, kinds: list[str] | None = None,
                             job_ids: list[str] | None = None, since_seq: int | None = None, ack_seq: int | None = None,
                             max_events: int = DEFAULT_MAX_EVENTS, client_id: str | None = None,
                             ctx: Any = None) -> EventsResult:
    """Section 4 ``wait_for_events`` on top of ``EventBus.wait`` (section 5.6 algorithm)."""
    events: EventBus = service.events
    cid = client_id or getattr(ctx, "client_id", None) or service.session_id
    timeout = max(0, min(int(timeout_s), MAX_TIMEOUT_S))
    max_events = max(1, int(max_events))
    kinds_l = [str(k) for k in kinds] if kinds is not None else None
    ids_l = [str(j) for j in job_ids] if job_ids is not None else None

    async def progress(i: int, elapsed: float) -> None:
        if ctx is None:
            return
        snap, detail = await _snapshot(service)
        message = f"waiting: {snap.pending} pending, {snap.running} running{detail}"
        try:
            await ctx.report_progress(float(i), None, message)
        except Exception as e:  # a client that ignores progress must not break the wait
            log.debug("report_progress failed: %s", e)

    res = await events.wait(cid, timeout_s=timeout, kinds=kinds_l, job_ids=ids_l, since_seq=since_seq, ack_seq=ack_seq,
                            max_events=max_events, progress_cb=progress, poll_s=PROGRESS_EVERY_S)
    snap, _detail = await _snapshot(service)
    rows = [e if isinstance(e, EventRow) else EventRow.model_validate(e) for e in res.events]
    if rows:
        kinds_seen: dict[str, int] = {}
        for e in rows:
            kinds_seen[e.kind] = kinds_seen.get(e.kind, 0) + 1
        summary = f"{len(rows)} event(s): " + ", ".join(f"{k} x{n}" if n > 1 else k for k, n in kinds_seen.items())
        summary += f"; {res.unread_events} unread"
        if res.unread_unmatched:
            summary += f" ({res.unread_unmatched} hidden by filters)"
    elif res.timed_out:
        summary = (f"no matching events within {timeout} s; {snap.pending} pending, {snap.running} running, "
                   f"{res.unread_events} unread")
    else:
        summary = f"no unacknowledged events; {snap.pending} pending, {snap.running} running"
    result = EventsResult(summary=summary, unread_events=res.unread_events, events=rows,
                          delivered_seqs=list(res.delivered_seqs), next_seq=res.next_seq, acked=res.acked,
                          unread_unmatched=res.unread_unmatched, timed_out=bool(res.timed_out), snapshot=snap,
                          client_id=cid, warnings=list(res.warnings))
    if rows:
        result.next = _next_hint(result, kinds_l, ids_l, timeout_s)
    elif snap.pending or snap.running or snap.queued:
        result.next = _next_hint(result, kinds_l, ids_l, timeout_s)
    else:
        result.next = "list_jobs() or submit_job(job)"
    return result


def register(mcp: MCPServer, service: Any) -> None:
    """Attach the Monitor (bound services only) and register ``wait_for_events``."""
    if getattr(service, "bound", True):
        try:
            components = getattr(service, "components", None)
            if isinstance(components, dict) and "monitor" not in components:
                from ..monitor import Monitor
                service.attach("monitor", Monitor(service))
        except Exception as e:  # pragma: no cover - a proxy that is not bound yet
            log.debug("monitor not attached at registration: %s", e)

    @mcp.tool(name="wait_for_events", description=WAIT_DESC, annotations=_mcp.read_only())
    async def wait_for_events(timeout_s: int = 300, kinds: Optional[list[str]] = None, job_ids: Optional[list[str]] = None,
                              since_seq: Optional[int] = None, ack_seq: Optional[int] = None, max_events: int = 50,
                              client_id: Optional[str] = None, ctx: Context = None) -> EventsResult:
        return await run_tool(wait_for_events_op(service, timeout_s=timeout_s, kinds=kinds, job_ids=job_ids,
                                                 since_seq=since_seq, ack_seq=ack_seq, max_events=max_events,
                                                 client_id=client_id, ctx=ctx))


__all__ = ["register", "wait_for_events_op", "WAIT_DESC", "MAX_TIMEOUT_S", "PROGRESS_EVERY_S"]
