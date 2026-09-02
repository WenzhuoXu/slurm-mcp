"""Monitor: per-cluster tick loops, the 12-step reconciliation of design section 5.2, the classification table
of section 5.3, lease fencing (sections 3.3, 5.8), restart recovery (section 5.8) and the cadence rules.

Layout:

* :func:`apply_observation` is the **pure** per-job reconciliation (steps 1-7 of section 5.2 plus the section
  5.3 table): ``(jobs_current row, attempts row, Observation, now, JobSpec) -> Outcome`` where ``Outcome`` lists
  the ``jobs``/``attempts`` field updates, the events to append and the side effects (``scancel``, ``scontrol
  hold``, ``wait_history`` rows, ...) the orchestrator must run. It touches neither SSH nor SQLite, so every
  row of the table is unit-testable; ``transition()`` from ``slurm/states.py`` guards every state change and
  events are produced once per transition.
* :class:`Monitor` owns one asyncio task per cluster (section 5.2 cadence: ``profile.poll`` base/min/max,
  +-10 % jitter, ``kick()`` within 5 s, backoff 30 s -> 5 min with ``cluster_unreachable``/``cluster_recovered``),
  renews the lease inside ``BEGIN IMMEDIATE`` before every tick (``LeaseLost`` stops it and the Service records
  ``needs_attention{lease_lost}``), re-acquires it after a clock jump, and runs ``run_tick()``: the composite
  probe (``SlurmClient.tick``), the per-job pure step, then steps 8-12 (untracked/unconfirmed confirmation via
  ``%o``/``%k``/``jobid``/``SubmitLine``, the INTENT sweep and QUEUED resume, dependency repointing/holding,
  enrichment) inside ``store.write_fenced``; SSH side effects run after the fenced write committed; finally
  ``notify.after_tick()``. The first tick after a (re)start is the section 5.8 sweep: every event it emits
  carries ``payload.observed_late = true``.

No cluster name appears in this module.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import asyncssh

from .clock import parse_duration
from .errors import SlurmMcpError
from .models import SIGNAL_NUMBERS, JobSpec, Target, parse_input
from .render import expand_pattern
from .slurm.client import TickFailed
from .slurm.discovery import charge_for
from .slurm.parse import IncompleteProbe
from .slurm.states import (
    LIVE, PRE_SLURM, TERMINAL, AttemptState, CmdState, JobState, classify_reason, map_slurm_state, transition,
)
from .store import LeaseLost, Store, loads_json
from .transport import CommandTimeout, ConnectionDropped

log = logging.getLogger("slurm_mcp.monitor")

# --- section 5.2 constants ----------------------------------------------------------------------------------
KICK_WITHIN_S = 5.0
JITTER = 0.10
UNREACHABLE_AFTER_FAILURES = 3
BACKOFF_MIN_S = 30.0
BACKOFF_MAX_S = 300.0
CLOCK_JUMP_S = 120.0
ATTEMPT_YOUNG_S = 120                  # step 3: dbd/ctld lag window
STALE_TICKS_LOST = 3                   # step 3: scontrol probe / LOST
COMPLETING_MAX_TICKS = 2               # step 2: terminal name in squeue waits <= 2 ticks for sacct
CANCEL_AGENT_WINDOW_S = 900            # section 5.3: cancelled{by:"agent"} when requested within 15 min
HEARTBEAT_STALE_S = 900                # step 7
ALLOC_EXPIRING_S = 600                 # step 6
UNCONFIRMED_DEADLINE_S = 900           # step 9: 15 min of healthy observation
INTENT_STUCK_S = 600                   # step 10: INTENT older than 10 min
REQUEUE_LOOP_N = 3                     # step 4 guard
REQUEUE_LOOP_WINDOW_S = 600
ENRICH_BATCH = 20                      # step 12
ENRICH_GIVE_UP_S = 3600                # stop asking sacct for a terminal job after 1 h without a row
STARTED_RECENT_S = 300                 # cadence: min_s while a job started < 5 min ago
TIME_LEFT_MIN_S = 600                  # cadence: min_s while a running job has < 10 min left
MOVE_RECENT_S = 300                    # cadence: min_s for 5 min after a rebalance move
SNAPSHOT_IDLE_S = 600                  # cadence: idle clusters refresh the snapshot every 10 min
OOM_RSS_RATIO = 0.95                   # section 5.3 OOM heuristic
RESTART_COST_RATIO = 1.1               # step 12 needs_attention{restart_cost}
UNTRACKED_KEY_PREFIX = "untracked."
MAX_UNTRACKED_ROWS = 200
LOST_HINT = "run_command('sacct -j {id}')"

TERMINAL_KINDS: frozenset[str] = frozenset(s.value.lower() for s in TERMINAL)


# --- observation / outcome ----------------------------------------------------------------------------------

@dataclass
class Observation:
    """Everything the tick learned about one job (section 5.2 ``O``), plus the ledger context the pure step needs.

    ``scontrol``: None = not queried, ``{}`` = ``Invalid job id specified``, a dict = ``parse_scontrol_job`` output
    (with ``*_ts`` fields). ``prior_kinds`` = event kinds already recorded for the handle (``needs_attention``
    entries as ``needs_attention:<why>``) for the "once" rules. ``requeue_history`` = ``(ts, prev_exit_code)`` of
    the earlier ``requeued`` events (requeue-loop guard). ``observed_late`` marks the first tick after a restart.
    """

    squeue: dict[str, Any] | None = None
    restart_cnt: int | None = None
    sacct: dict[str, Any] | None = None
    files: dict[str, Any] = field(default_factory=dict)
    scontrol: dict[str, Any] | None = None
    healthy: bool = True
    prior_kinds: set[str] = field(default_factory=set)
    requeue_history: list[tuple[int, int | None]] = field(default_factory=list)
    observed_late: bool = False


@dataclass
class Outcome:
    """What :func:`apply_observation` decided: field updates, events (``kind, summary, payload``) and actions.

    Actions (dicts with ``op``): ``scancel{id}``, ``hold{id, why}``, ``show_job{id}``, ``wait_history{...}``,
    ``abort_cmds{handle}``. ``terminal`` is the new terminal state when the job finalised in this step.
    """

    job: dict[str, Any] = field(default_factory=dict)
    attempt: dict[str, Any] = field(default_factory=dict)
    events: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    terminal: JobState | None = None

    def event(self, kind: str, summary: str, **payload: Any) -> None:
        self.events.append((kind, summary, payload))

    def action(self, op: str, **kw: Any) -> None:
        self.actions.append({"op": op, **kw})


def _state(value: Any) -> JobState | None:
    if value is None:
        return None
    try:
        return JobState(str(value))
    except ValueError:
        return None


def _status_json(files: Mapping[str, Any] | None) -> dict[str, Any] | None:
    st = (files or {}).get("status.json")
    return st if isinstance(st, dict) else None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def signum(name: str | None) -> int | None:
    """``"USR1"`` / ``"SIGTERM"`` -> Linux signal number (section 5.3 exit-code matching)."""
    if not name:
        return None
    s = str(name).strip().upper()
    if s.startswith("SIG"):
        s = s[3:]
    return SIGNAL_NUMBERS.get(s)


def spec_of(job_row: Mapping[str, Any]) -> JobSpec | None:
    """``jobs.spec_json`` -> ``JobSpec`` (None when the row holds something else, e.g. an adopted raw job)."""
    raw = job_row.get("spec_json")
    try:
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        return parse_input(JobSpec, data) if data else None
    except Exception:
        return None


def time_limit_s(spec: JobSpec | None) -> int | None:
    if spec is None:
        return None
    return parse_duration(spec.resources.time)


def target_of(row: Mapping[str, Any]) -> Target | None:
    """``attempts.target_json`` -> ``Target`` (None when malformed)."""
    raw = row.get("target_json")
    try:
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        if isinstance(data, str):
            return Target.parse(data)
        return Target.model_validate(data) if data else None
    except Exception:
        return None


# --- section 5.3 classification ---------------------------------------------------------------------------

def helper_timeout(cur: Mapping[str, Any], status: Mapping[str, Any] | None, spec: JobSpec | None) -> bool:
    """Section 5.3 (b): sacct FAILED/CANCELLED whose ExitCode is ``128+signum:0`` or ``0:signum`` for the
    forwarded signal (``child_signal`` when declared, TERM always), ``status.json.cause == "timeout"`` and
    ``ElapsedRaw >= TimelimitRaw*60 - grace_s - 60`` (skipped when either value is unknown)."""
    if not status or status.get("cause") != "timeout":
        return False
    ec = cur.get("exit_code")
    if not ec:
        return False
    sigs = {SIGNAL_NUMBERS["TERM"]}
    cs = signum(spec.child_signal) if spec is not None else None
    if cs:
        sigs.add(cs)
    if not any(tuple(ec) == (128 + s, 0) or tuple(ec) == (0, s) for s in sigs):
        return False
    el, tl = cur.get("elapsed_s"), cur.get("timelimit_s")
    grace = spec.grace_s if spec is not None else 120
    if el is not None and tl is not None and el < tl - grace - 60:
        return False
    return True


def cancelled_by(cur: Mapping[str, Any], job: Mapping[str, Any], now: int) -> str:
    """``agent`` (cancel_requested_ts within 15 min), ``scheduler`` (DependencyNeverSatisfied) or ``user/admin``."""
    reason = str(cur.get("reason") or "")
    if "DependencyNeverSatisfied" in reason or "DependencyNeverSatisfied" in str(job.get("reason") or ""):
        return "scheduler"
    req = _int(job.get("cancel_requested_ts"))
    if req is not None:
        end = _int(cur.get("end_ts")) or now
        if -60 <= end - req <= CANCEL_AGENT_WINDOW_S:
            return "agent"
    return "user/admin"


def classify_sacct(cur: Mapping[str, Any], steps: Iterable[Mapping[str, Any]], status: Mapping[str, Any] | None,
                   spec: JobSpec | None, job: Mapping[str, Any], now: int) -> tuple[JobState, dict[str, Any]]:
    """The section 5.3 table for a terminal sacct allocation row -> ``(state, details)``.

    ``details``: ``exit_code, exit_signal, elapsed_s, time_limit_s, cause, source, by, end_ts, start_ts, node``.
    """
    base = cur.get("job_state") or JobState.FAILED
    ec = cur.get("exit_code") or (None, None)
    rc, sig = (ec[0], ec[1]) if ec and ec[0] is not None else (None, None)
    new = base
    source = "sacct"
    by: str | None = None
    if base == JobState.TIMEOUT:
        new = JobState.TIMEOUT
    elif base in (JobState.FAILED, JobState.CANCELLED):
        if helper_timeout(cur, status, spec):
            new, source = JobState.TIMEOUT, "helper+sacct"
        elif base == JobState.CANCELLED:
            by = cancelled_by(cur, job, now)
    if base == JobState.OOM or any(s.get("job_state") == JobState.OOM for s in steps):
        new = JobState.OOM
    if status and new == base and status.get("phase") == "exited":
        source = "helper+sacct"
    return new, {
        "exit_code": rc, "exit_signal": sig, "elapsed_s": cur.get("elapsed_s"), "time_limit_s": cur.get("timelimit_s"),
        "cause": (status or {}).get("cause") or None, "source": source, "by": by, "end_ts": _int(cur.get("end_ts")),
        "start_ts": _int(cur.get("start_ts")), "node": cur.get("nodelist"), "slurm_state": cur.get("state_raw"),
        "reason": cur.get("reason"),
    }


def classify_helper(status: Mapping[str, Any], spec: JobSpec | None) -> tuple[JobState, dict[str, Any]]:
    """Section 5.2 step 3: ``status.json.phase == "exited"`` -> COMPLETED (rc 0) | TIMEOUT (cause timeout) |
    CANCELLED (cause cancel) | FAILED, ``source="helper"``."""
    rc = _int(status.get("rc"))
    cause = status.get("cause") or None
    if rc == 0:
        new = JobState.COMPLETED
    elif cause == "timeout":
        new = JobState.TIMEOUT
    elif cause == "cancel":
        new = JobState.CANCELLED
    else:
        new = JobState.FAILED
    start = _int(status.get("start"))
    end = _int(status.get("now"))
    limit = _int(status.get("limit_s")) or time_limit_s(spec)
    return new, {
        "exit_code": rc, "exit_signal": (rc - 128) if rc is not None and rc > 128 else None,
        "elapsed_s": (end - start) if start is not None and end is not None else None,
        "time_limit_s": limit or None, "cause": cause, "source": "helper", "by": None, "end_ts": end,
        "start_ts": start, "node": status.get("node") or None, "slurm_state": None, "reason": None,
    }


def oom_suspected(job: Mapping[str, Any], enrich: Mapping[str, Any] | None) -> bool:
    """Section 5.3 OOM heuristic on enrichment data: FAILED/CANCELLED with ``0:9``/``137`` and a step whose
    ``MaxRSS >= 0.95 x ReqMem``."""
    if not enrich or _state(job.get("state")) not in (JobState.FAILED, JobState.CANCELLED):
        return False
    rc, sig = _int(job.get("exit_code")), _int(job.get("exit_signal"))
    if not ((rc == 0 and sig == 9) or rc == 137 or (rc == 128 + 9)):
        return False
    rss, req = enrich.get("max_rss_bytes"), enrich.get("req_mem_bytes")
    if not isinstance(rss, (int, float)) or not isinstance(req, (int, float)) or req <= 0:
        return False
    return rss >= OOM_RSS_RATIO * req


def terminal_summary(handle: str, cluster: str | None, slurm_id: Any, state: JobState, det: Mapping[str, Any]) -> str:
    parts = [f"{handle} {state.value}"]
    if cluster:
        parts.append(f"on {cluster}")
    if slurm_id:
        parts.append(f"(slurm {slurm_id})")
    if det.get("exit_code") is not None:
        parts.append(f"rc={det['exit_code']}" + (f":{det['exit_signal']}" if det.get("exit_signal") else ""))
    if det.get("elapsed_s") is not None:
        parts.append(f"after {int(det['elapsed_s'])}s")
    if det.get("by"):
        parts.append(f"by {det['by']}")
    return " ".join(parts)


def terminal_hint(state: JobState, det: Mapping[str, Any], spec: JobSpec | None) -> str | None:
    """The actionable hint of the terminal events (section 5.3 "Result" column)."""
    if state == JobState.TIMEOUT:
        return ("declare child_signal + checkpoint_interval_h and on_timeout='requeue', raise resources.time, or pick a "
                "partition with a longer MaxWall")
    if state == JobState.OOM:
        rss = det.get("max_rss")
        if isinstance(rss, (int, float)):
            return f"raise resources.mem to >= {1.3 * rss / 2**30:.1f}G (1.3 x max_rss)"
        return "raise resources.mem to >= 1.3 x max_rss"
    if state == JobState.LOST:
        return LOST_HINT
    if state == JobState.FAILED:
        return "call job_logs(id, stream='err')"
    if state == JobState.PREEMPTED:
        return "resubmit on a non-preemptable target or set requeue=True"
    if state == JobState.NODE_FAIL:
        return "resubmit; the failed node is excluded for 2 h"
    return None


# --- the pure per-job step (section 5.2 steps 1-7) ----------------------------------------------------------

def apply_observation(job_row: Mapping[str, Any], attempt_row: Mapping[str, Any] | None, observation: Observation,
                      now: int, spec: JobSpec | None = None) -> Outcome:
    """Reconcile one tracked job against this tick's observation (design section 5.2 steps 1-7, section 5.3).

    Pure: reads ``job_row`` (a ``jobs_current`` row), ``attempt_row`` (the current ``attempts`` row, may be None)
    and ``observation``; returns an :class:`Outcome`. ``now`` is cluster epoch seconds. Applying the outcome and
    calling again with the same observation yields no events (transitions are emitted once).
    """
    job = dict(job_row)
    attempt = dict(attempt_row or {})
    out = Outcome()
    obs = observation
    handle = str(job.get("handle"))
    cluster = job.get("cluster")
    slurm_id = job.get("slurm_id") or attempt.get("slurm_id")
    kind = job.get("kind") or "job"
    state0 = _state(job.get("state"))
    cur_state = state0
    restarts0 = int(job.get("restarts") or 0)
    if spec is None:
        spec = spec_of(job)
    sq = obs.squeue
    sa = obs.sacct or None
    cur = sa.get("current") if sa else None
    rows: list[dict[str, Any]] = list(sa.get("rows") or []) if sa else []
    steps: list[dict[str, Any]] = list(sa.get("steps") or []) if sa else []
    status = _status_json(obs.files)
    sq_state: JobState | None = sq.get("job_state") if sq else None
    sq_live = sq is not None and sq_state is not None and sq_state not in TERMINAL
    cur_terminal = bool(cur and cur.get("job_state") in TERMINAL and cur.get("end_ts") is not None)

    def set_state(new: JobState) -> bool:
        nonlocal cur_state
        if transition(cur_state, new):
            cur_state = new
            out.job["state"] = new.value
            return True
        return False

    def finalise(new: JobState, det: Mapping[str, Any]) -> None:
        """Apply a terminal classification: fields, attempt, one event per transition (upgrades included)."""
        was_terminal = cur_state in TERMINAL
        moved = set_state(new)
        if not moved and cur_state != new:
            return                                   # illegal (a downgrade): keep what we have
        fill = {"end_ts": det.get("end_ts"), "exit_code": det.get("exit_code"), "exit_signal": det.get("exit_signal"),
                "slurm_state": det.get("slurm_state"), "reason": det.get("reason"), "node": det.get("node")}
        for k, v in fill.items():
            if v is not None and (moved or job.get(k) is None):
                if k == "node":
                    out.attempt["node"] = v
                else:
                    out.job[k] = v
        if det.get("start_ts") is not None and job.get("start_ts") is None:
            out.job["start_ts"] = det["start_ts"]
        out.job.setdefault("stale_ticks", 0)
        out.job["last_seen_ts"] = now
        out.job["cancel_hard_ts"] = None
        out.attempt["state"] = AttemptState.DONE.value
        out.attempt["final_state"] = new.value
        if det.get("exit_code") is not None:
            out.attempt["exit_code"] = det["exit_code"]
        if det.get("end_ts") is not None:
            out.attempt["end_ts"] = det["end_ts"]
        if det.get("reason") is not None:
            out.attempt["reason"] = det["reason"]
        if not moved:
            return
        if not was_terminal:
            out.job["terminal_ts"] = now
        out.terminal = new
        payload = {
            "exit_code": det.get("exit_code"), "exit_signal": det.get("exit_signal"), "elapsed_s": det.get("elapsed_s"),
            "time_limit_s": det.get("time_limit_s") if det.get("time_limit_s") is not None else time_limit_s(spec),
            "cost_su": job.get("cost_actual_su"), "stdout_path": job.get("stdout_path"), "last_line": job.get("last_line"),
            "cause": det.get("cause"), "source": det.get("source"), "restarts": out.job.get("restarts", restarts0),
            "by": det.get("by"), "state": new.value,
        }
        hint = terminal_hint(new, det, spec)
        if hint:
            payload["hint"] = hint
        out.event(new.value.lower(), terminal_summary(handle, cluster, slurm_id, new, det), **payload)
        if kind == "alloc":
            out.event("alloc_ended", f"{handle} allocation ended ({new.value})", state=new.value)
            out.action("abort_cmds", handle=handle)

    # -- step 4 first: requeue detection (RUNNING -> PENDING, RestartCnt, -D incarnations, status.json.restart)
    signals: list[int] = []
    if obs.restart_cnt is not None:
        signals.append(int(obs.restart_cnt))
    if sa:
        signals.append(int(sa.get("incarnations") or 1) - 1)
    if status and isinstance(status.get("restart"), int):
        signals.append(int(status["restart"]))
    new_restarts = max(signals) if signals else restarts0
    if state0 in (JobState.RUNNING, JobState.COMPLETING) and sq_state == JobState.SUBMITTED and new_restarts <= restarts0:
        new_restarts = restarts0 + 1
    prev_rc: int | None = None
    if new_restarts > restarts0:
        prev_rows = [r for r in rows if r is not cur]
        prev = prev_rows[-1] if prev_rows else None
        cause = "requeue"
        if prev is not None and prev.get("job_state") == JobState.PREEMPTED:
            cause = "preempted"
        elif prev is not None and prev.get("job_state") == JobState.NODE_FAIL:
            cause = "node_fail"
        elif status and status.get("cause") == "timeout":
            cause = "timeout"
        if prev is not None and prev.get("exit_code"):
            prev_rc = _int(prev["exit_code"][0])
        out.job["restarts"] = new_restarts
        out.job["start_ts"] = None
        out.attempt["node"] = None
        set_state(JobState.SUBMITTED)
        if sq is not None:
            out.job["slurm_state"] = sq.get("state")
            out.job["reason"] = sq.get("reason")
        if cause in ("preempted", "node_fail"):
            out.event(cause, f"{handle} {cause} on {cluster}; requeued by SLURM (restart {new_restarts})",
                      requeued=True, restarts=new_restarts, cause=cause, state=JobState.SUBMITTED.value)
        out.event("requeued", f"{handle} requeued ({cause}), restart {new_restarts} of {spec.max_restarts if spec else '?'}",
                  cause=cause, restarts=new_restarts, attempt_no=job.get("attempt_no"), prev_exit_code=prev_rc,
                  state=JobState.SUBMITTED.value)
        max_restarts = spec.max_restarts if spec is not None else 3
        if new_restarts > max_restarts and "needs_attention:max_restarts" not in obs.prior_kinds:
            out.action("hold", id=slurm_id, why="max_restarts")
            out.event("needs_attention", f"{handle} restarted {new_restarts} times (max {max_restarts}); held",
                      why="max_restarts", hint="job_control(release) after choosing another target, or cancel",
                      state=JobState.SUBMITTED.value)
        history = list(obs.requeue_history) + [(now, prev_rc)]
        recent = [h for h in history if now - int(h[0]) <= REQUEUE_LOOP_WINDOW_S and h[1] not in (0, None)]
        if len(recent) >= REQUEUE_LOOP_N and "needs_attention:requeue_loop" not in obs.prior_kinds:
            out.action("hold", id=slurm_id, why="requeue_loop")
            out.event("needs_attention", f"{handle} requeued {len(recent)} times in 10 min, each ending rc!=0; held",
                      why="requeue_loop", hint="inspect job_logs; job_control(release) or cancel",
                      state=JobState.SUBMITTED.value)

    # -- step 1: sacct terminal truth ----------------------------------------------------------------------
    if cur_terminal and not sq_live:
        new, det = classify_sacct(cur, steps, status, spec, job, now)
        finalise(new, det)
    # -- step 2: squeue live state -------------------------------------------------------------------------
    elif sq is not None and sq_state is not None:
        out.job["last_seen_ts"] = now
        if sq_state == JobState.SUBMITTED:
            set_state(JobState.SUBMITTED)
            out.job["slurm_state"] = sq.get("state")
            out.job["reason"] = sq.get("reason")
            out.job["est_start_ts"] = _int(sq.get("start_ts"))
            out.job["stale_ticks"] = 0
            if classify_reason(sq.get("reason")) == "held" and classify_reason(job.get("reason")) != "held" \
                    and cur_state in LIVE:
                out.event("held", f"{handle} held by SLURM ({sq.get('reason')})", reason=sq.get("reason"),
                          state=cur_state.value)
        elif sq_state == JobState.RUNNING:
            out.job["slurm_state"] = sq.get("state")
            out.job["reason"] = None
            out.job["stale_ticks"] = 0
            if cur_state != JobState.RUNNING and set_state(JobState.RUNNING):
                start_ts = _int(sq.get("start_ts")) or now
                submit_ts = _int(job.get("submit_ts")) or _int(sq.get("submit_ts")) or start_ts
                out.job["start_ts"] = start_ts
                out.attempt["node"] = sq.get("nodes")
                wait_s = max(0, start_ts - submit_ts)
                out.event("started", f"{handle} started on {cluster} node {sq.get('nodes') or '?'} after {wait_s}s wait",
                          node=sq.get("nodes"), wait_s=wait_s, state=JobState.RUNNING.value)
                out.action("wait_history", submit_ts=submit_ts, start_ts=start_ts)
            elif sq.get("nodes") and sq.get("nodes") != attempt.get("node"):
                out.attempt["node"] = sq.get("nodes")
        elif sq_state == JobState.COMPLETING:
            set_state(JobState.COMPLETING)
            out.job["slurm_state"] = sq.get("state")
            out.job["stale_ticks"] = 0
        else:                                                    # a terminal name inside squeue (MinJobAge)
            if cur_state not in TERMINAL:
                ticks = int(job.get("stale_ticks") or 0) + 1 if state0 == JobState.COMPLETING else 1
                set_state(JobState.COMPLETING)
                out.job["slurm_state"] = sq.get("state")
                out.job["stale_ticks"] = ticks
                if ticks > COMPLETING_MAX_TICKS:
                    det = {"exit_code": None, "exit_signal": None, "elapsed_s": sq.get("elapsed_s"),
                           "time_limit_s": sq.get("time_limit_s"), "cause": None, "source": "squeue", "by": None,
                           "end_ts": _int(sq.get("end_ts")) or now, "start_ts": _int(sq.get("start_ts")),
                           "node": sq.get("nodes"), "slurm_state": sq.get("state"), "reason": sq.get("reason")}
                    if sq_state == JobState.CANCELLED:
                        det["by"] = cancelled_by({"reason": sq.get("reason"), "end_ts": det["end_ts"]}, job, now)
                    finalise(sq_state, det)
    # -- step 3: in neither ---------------------------------------------------------------------------------
    elif cur_state in LIVE and cur_state not in PRE_SLURM and slurm_id:
        if status and status.get("phase") == "exited":
            new, det = classify_helper(status, spec)
            finalise(new, det)
        else:
            born = _int(attempt.get("submit_ts")) or _int(job.get("submit_ts")) or _int(job.get("last_seen_ts"))
            if born is not None and now - born < ATTEMPT_YOUNG_S:
                pass                                             # dbd/ctld lag: keep
            else:
                ticks = int(job.get("stale_ticks") or 0) + 1
                out.job["stale_ticks"] = ticks
                if ticks >= STALE_TICKS_LOST:
                    info = obs.scontrol
                    if info is None:
                        out.action("show_job", id=slurm_id)
                    elif not info:
                        finalise(JobState.LOST, {"exit_code": None, "exit_signal": None, "elapsed_s": None,
                                                 "time_limit_s": None, "cause": None, "source": "scontrol", "by": None,
                                                 "end_ts": now, "start_ts": None, "node": None, "slurm_state": None,
                                                 "reason": "Invalid job id specified"})
                    else:
                        st = info.get("state") or map_slurm_state(info.get("job_state"))
                        if st in TERMINAL:
                            ec = info.get("exit_code") or (None, None)
                            finalise(st, {"exit_code": ec[0], "exit_signal": ec[1], "elapsed_s": None,
                                          "time_limit_s": info.get("time_limit_s"), "cause": None, "source": "scontrol",
                                          "by": None, "end_ts": _int(info.get("end_time_ts")) or now,
                                          "start_ts": _int(info.get("start_time_ts")), "node": info.get("node_list"),
                                          "slurm_state": info.get("job_state"), "reason": info.get("reason")})
                        else:
                            out.job["reason"] = info.get("reason")
                            out.job["slurm_state"] = info.get("job_state")
                            if isinstance(info.get("restarts"), int) and info["restarts"] > new_restarts:
                                out.job["restarts"] = info["restarts"]
                            out.job["stale_ticks"] = 0
                            out.job["last_seen_ts"] = now

    # -- step 5: cancel grace -------------------------------------------------------------------------------
    hard = _int(job.get("cancel_hard_ts"))
    if hard is not None and now >= hard and cur_state not in TERMINAL:
        if sq_live and slurm_id:
            out.action("scancel", id=slurm_id, handle=handle)
        out.job["cancel_hard_ts"] = None

    # -- step 6: allocations --------------------------------------------------------------------------------
    if kind == "alloc":
        limit = time_limit_s(spec)
        # helpers/alloc-agent.sh writes phase "ready" exactly once, immediately before its event loop,
        # and "running" every second from then on. A monitor tick lands every ~30 s, so it practically
        # never observes that one-second "ready" -- keying readiness off it alone meant alloc_ready
        # never fired and alloc_run stayed E_ALLOC_NOT_READY for the life of the allocation. Any live
        # phase from the agent means the node is ours; only "exited" says otherwise.
        agent_live = bool(status) and status.get("phase") in ("ready", "running")
        if agent_live and not job.get("alloc_ready") and cur_state in LIVE:
            start = _int(out.job.get("start_ts")) or _int(job.get("start_ts")) or _int(status.get("start")) or now
            end = start + limit if limit else None
            out.job["alloc_ready"] = 1
            out.job["alloc_end_ts"] = end
            node = status.get("node") or attempt.get("node") or out.attempt.get("node")
            out.event("alloc_ready", f"{handle} allocation ready on {node or '?'}", node=node, end_ts=end,
                      state=cur_state.value)
        end_ts = _int(out.job.get("alloc_end_ts")) or _int(job.get("alloc_end_ts"))
        ready = bool(out.job.get("alloc_ready") or job.get("alloc_ready"))
        if ready and cur_state in LIVE and end_ts is not None and end_ts - now < ALLOC_EXPIRING_S \
                and "alloc_expiring" not in obs.prior_kinds:
            minutes = max(0, (end_ts - now) // 60)
            out.event("alloc_expiring", f"{handle} allocation ends in {minutes} min", minutes_left=minutes,
                      state=cur_state.value)

    # -- step 7: files --------------------------------------------------------------------------------------
    hb = _int(obs.files.get("heartbeat")) if obs.files else None
    if hb is not None:
        out.job["heartbeat_ts"] = hb
    prog = obs.files.get("progress.json") if obs.files else None
    if prog is not None:
        out.job["progress_json"] = json.dumps(prog, separators=(",", ":")) if isinstance(prog, (dict, list)) else str(prog)
    hb_ts = hb if hb is not None else _int(job.get("heartbeat_ts"))
    if cur_state == JobState.RUNNING and hb_ts is not None and now - hb_ts > HEARTBEAT_STALE_S \
            and "needs_attention:heartbeat_stale" not in obs.prior_kinds:
        out.event("needs_attention", f"{handle} heartbeat is {now - hb_ts}s old while RUNNING", why="heartbeat_stale",
                  hint="job_logs(id) to inspect; the node or the wrapper may be wedged", state=cur_state.value)
    return out


# --- the orchestrator -----------------------------------------------------------------------------------------

@dataclass
class TickPlan:
    """The ledger view one tick starts from (read in one transaction)."""

    jobs: list[dict[str, Any]] = field(default_factory=list)            # jobs_current rows to observe
    attempts: dict[str, dict[str, Any]] = field(default_factory=dict)   # handle -> current attempts row
    tracked_ids: set[int] = field(default_factory=set)                  # every attempts.slurm_id on the cluster
    unconfirmed: list[dict[str, Any]] = field(default_factory=list)     # attempts rows
    intents: list[dict[str, Any]] = field(default_factory=list)
    queued: list[dict[str, Any]] = field(default_factory=list)          # jobs_current rows (QUEUED)
    cmds: list[dict[str, Any]] = field(default_factory=list)            # outstanding alloc_cmds rows
    dependents: list[dict[str, Any]] = field(default_factory=list)      # live jobs with depends_on_json
    prior_kinds: dict[str, set[str]] = field(default_factory=dict)
    requeue_history: dict[str, list[tuple[int, int | None]]] = field(default_factory=dict)
    ids: list[int] = field(default_factory=list)
    ctrl_dirs: list[str] = field(default_factory=list)
    rc_paths: list[str] = field(default_factory=list)
    enrich: list[dict[str, Any]] = field(default_factory=list)          # terminal, un-enriched rows
    live_count: int = 0

    @property
    def idle(self) -> bool:
        return not (self.ids or self.unconfirmed or self.intents or self.queued or self.cmds or self.enrich)


@dataclass
class TickReport:
    cluster: str
    ok: bool
    now: int | None = None
    events: int = 0
    error: str | None = None
    skipped: bool = False
    actions: int = 0


def rc_path_for(out_path: str) -> str:
    """``cmds/002.out`` -> ``cmds/002.rc`` (the agent's rc file next to the command's output, section 7.3)."""
    return re.sub(r"\.out$", ".rc", out_path) if out_path.endswith(".out") else out_path + ".rc"


def prior_kind_keys(events: Iterable[Mapping[str, Any]]) -> set[str]:
    """Event rows -> ``{"held", "alloc_expiring", "needs_attention:max_restarts", ...}``."""
    keys: set[str] = set()
    for e in events:
        kind = e.get("kind")
        if not kind:
            continue
        keys.add(str(kind))
        if kind == "needs_attention":
            payload = e.get("payload") if isinstance(e.get("payload"), dict) else loads_json(e, "payload_json", {})
            why = (payload or {}).get("why")
            if why:
                keys.add(f"needs_attention:{why}")
    return keys


class Monitor:
    """The per-cluster tick loops (design section 5.2/5.8). Attach as ``service.attach("monitor", Monitor(service))``."""

    def __init__(self, service: Any, *, startup_sweep: bool = True, rng: Callable[[], float] | None = None) -> None:
        self.service = service
        self.store: Store = service.store
        self.events = service.events
        self.startup_sweep = startup_sweep
        self._rng = rng or random.random
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._kicks: dict[str, asyncio.Event] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.last_tick: dict[str, float] = {}          # local wall time of the last successful tick
        self.last_tick_now: dict[str, int] = {}        # cluster epoch of the last successful tick
        self.failures: dict[str, int] = {}
        self.unreachable: dict[str, bool] = {}
        self.first_tick: dict[str, bool] = {}
        self.unconfirmed_healthy_s: dict[int, float] = {}
        self.last_snapshot_local: dict[str, float] = {}
        self.recent_moves: dict[str, float] = {}
        self.stats: dict[str, dict[str, int]] = {}
        self.running = False
        self.stopped_reason: str | None = None

    # -- component protocol --------------------------------------------------------------------------------

    def start(self) -> None:
        """Start one loop per cluster (only when this process holds the lease; the Service restarts us on acquisition)."""
        if self.running:
            return
        if getattr(self.service, "lease_token", None) is None:
            log.info("monitor not started: no lease")
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self.running = True
        self.stopped_reason = None
        for name in self.service.registry.names():
            self.first_tick[name] = self.startup_sweep
            self._kicks.setdefault(name, asyncio.Event())
            self._locks.setdefault(name, asyncio.Lock())
            self._tasks[name] = asyncio.create_task(self._loop(name), name=f"slurm-mcp-monitor-{name}")

    async def stop(self) -> None:
        self.running = False
        tasks = list(self._tasks.values())
        self._tasks.clear()
        current = asyncio.current_task()
        for t in tasks:
            if t is not current:
                t.cancel()
        for t in tasks:
            if t is current:
                continue
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    def kick(self, cluster: str) -> None:
        """Force a tick within 5 s (section 5.2)."""
        ev = self._kicks.setdefault(cluster, asyncio.Event())
        ev.set()

    def note_move(self, cluster: str) -> None:
        """A rebalance move happened: min cadence for 5 min (section 5.2)."""
        self.recent_moves[cluster] = time.time()

    def last_tick_local(self, cluster: str) -> float | None:
        return self.last_tick.get(cluster)

    async def tick_if_stale(self, cluster: str, max_age_s: float = 20.0) -> bool:
        """Run a tick now when the last one is older than ``max_age_s`` (section 4 job_status/job_control)."""
        if getattr(self.service, "lease_token", None) is None:
            return False
        last = self.last_tick.get(cluster)
        if last is not None and time.time() - last < max_age_s:
            return False
        report = await self.tick_now(cluster)
        return report.ok

    async def tick_now(self, cluster: str) -> TickReport:
        """One tick outside the loop (serialised with it per cluster). Raises ``LeaseLost`` only via the loop."""
        lock = self._locks.setdefault(cluster, asyncio.Lock())
        async with lock:
            try:
                return await self.run_tick(cluster)
            except LeaseLost as e:
                await self._lease_lost(e)
                return TickReport(cluster, False, error=str(e))

    # -- cadence ---------------------------------------------------------------------------------------------

    def interval_for(self, cluster: str) -> float:
        """Section 5.2 cadence: min_s/base_s/max_s from the ledger, +-10 % jitter; backoff after failures."""
        n = self.failures.get(cluster, 0)
        if n > 0:
            return min(BACKOFF_MAX_S, BACKOFF_MIN_S * (2 ** (n - 1)))
        poll = self.service.profile(cluster).poll
        base, lo, hi = float(poll["base_s"]), float(poll["min_s"]), float(poll["max_s"])
        try:
            level = self.store.read_sync(lambda c: self._cadence_level(c, cluster))
        except Exception:
            level = "base"
        if level == "min":
            secs = lo
        elif level == "max":
            secs = hi
        else:
            secs = base
        return secs * (1.0 + JITTER * (2.0 * self._rng() - 1.0))

    def _cadence_level(self, conn: Any, cluster: str) -> str:
        rows = self.store.list_jobs(conn, cluster=cluster, states=list(LIVE))
        clock = self.service.clock(cluster)
        now = clock.remote_now()
        if time.time() - self.recent_moves.get(cluster, 0.0) < MOVE_RECENT_S:
            return "min"
        fast = False
        for r in rows:
            st = _state(r.get("state"))
            if st == JobState.COMPLETING or r.get("cancel_requested_ts") or st in (JobState.SUBMITTING, JobState.QUEUED) \
                    or r.get("attempt_state") == AttemptState.UNCONFIRMED.value:
                fast = True
            start = _int(r.get("start_ts"))
            if st == JobState.RUNNING and start is not None:
                if now - start < STARTED_RECENT_S:
                    fast = True
                limit = time_limit_s(spec_of(r))
                if limit is not None and start + limit - now < TIME_LEFT_MIN_S:
                    fast = True
            if r.get("kind") == "alloc" and st == JobState.RUNNING:
                if self.store.alloc_cmds_for(conn, r["handle"], states=[CmdState.queued, CmdState.running]):
                    fast = True
        if fast:
            return "min"
        unconfirmed = self.store.count(conn, "attempts", cluster=cluster, state=AttemptState.UNCONFIRMED.value)
        if unconfirmed:
            return "min"
        return "base" if rows else "max"

    async def _loop(self, cluster: str) -> None:
        ev = self._kicks.setdefault(cluster, asyncio.Event())
        first = True
        while self.running:
            try:
                if not first:
                    timeout = self.interval_for(cluster)
                    try:
                        await asyncio.wait_for(ev.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        pass
                    ev.clear()
                first = False
                if not self.running:
                    break
                async with self._locks.setdefault(cluster, asyncio.Lock()):
                    await self.run_tick(cluster)
            except asyncio.CancelledError:
                raise
            except LeaseLost as e:
                await self._lease_lost(e)
                return
            except Exception as e:  # pragma: no cover - defensive: the loop must survive anything
                log.exception("monitor %s: unexpected error: %s", cluster, e)
                self.failures[cluster] = self.failures.get(cluster, 0) + 1

    async def _lease_lost(self, exc: LeaseLost) -> None:
        if self.stopped_reason == "lease_lost":
            return
        log.warning("monitor stopping: %s", exc)
        self.running = False
        self.stopped_reason = "lease_lost"
        handler = getattr(self.service, "on_lease_lost", None)
        if callable(handler):
            asyncio.get_running_loop().create_task(_await_maybe(handler()), name="slurm-mcp-lease-lost")

    # -- lease (section 5.2 "Before every tick") --------------------------------------------------------------

    async def _renew_lease(self, cluster: str) -> int:
        svc = self.service
        token = getattr(svc, "lease_token", None)
        if token is None:
            raise LeaseLost(None, None)
        clock = svc.clock(cluster)
        if clock.jump_detected(CLOCK_JUMP_S):
            info = await svc.acquire_lease()
            if not info.acquired:
                raise LeaseLost(token, {"owner_pid": info.owner_pid})
            token = info.token
        ok = await self.store.write(lambda c: self.store.lease_renew(c, token))
        if not ok:
            row = await self.store.read(lambda c: self.store.lease_get(c))
            raise LeaseLost(token, row)
        return int(token)

    # -- the tick ---------------------------------------------------------------------------------------------

    async def run_tick(self, cluster: str) -> TickReport:
        """Section 5.2: renew the lease, probe, reconcile inside one fenced write, run side effects, notify."""
        svc = self.service
        token = await self._renew_lease(cluster)
        st = self.stats.setdefault(cluster, {"ticks": 0, "failed": 0, "events": 0})
        try:
            caps = await svc.caps(cluster)
        except (SlurmMcpError, CommandTimeout, ConnectionDropped, asyncssh.Error, OSError) as e:
            return await self._tick_failed(cluster, token, e)
        profile = svc.profile(cluster)
        client = svc.client(cluster)
        clock = svc.clock(cluster)
        plan = await self.store.read(lambda c: self._plan(c, cluster))
        first = self.first_tick.get(cluster, False)
        if plan.idle and not first:
            await self._refresh_snapshot_if_idle(cluster)
            self.last_tick[cluster] = time.time()
            self.last_tick_now[cluster] = clock.remote_now()
            return TickReport(cluster, True, now=clock.remote_now(), skipped=True)
        enrich_rows = plan.enrich[:ENRICH_BATCH]
        try:
            obs = await client.tick(plan.ids, plan.ctrl_dirs, plan.rc_paths, recover=bool(plan.unconfirmed),
                                    enrich_ids=[_int(r.get("slurm_id")) for r in enrich_rows if _int(r.get("slurm_id"))],
                                    stdout_paths=[r["stdout_path"] for r in enrich_rows if r.get("stdout_path")])
        except (TickFailed, IncompleteProbe, CommandTimeout, ConnectionDropped, SlurmMcpError, asyncssh.Error, OSError) as e:
            return await self._tick_failed(cluster, token, e)
        now = int(obs.get("now") or clock.remote_now())
        prev_now = self.last_tick_now.get(cluster)
        by_id: dict[int, dict[str, Any]] = {}
        for row in obs["squeue"]:
            sid = row.get("slurm_id")
            if sid is not None:
                by_id[int(sid)] = row
        # scontrol probes for jobs about to reach 3 stale ticks (step 3), before the pure step
        scontrol: dict[str, dict[str, Any] | None] = {}
        for job in plan.jobs:
            sid = _int(job.get("slurm_id"))
            if sid is None or _state(job.get("state")) not in LIVE or int(job.get("stale_ticks") or 0) < STALE_TICKS_LOST - 1:
                continue
            group = obs["sacct"].get(sid)
            cur = group.get("current") if group else None
            if sid in by_id or (cur and cur.get("job_state") in TERMINAL and cur.get("end_ts") is not None):
                continue
            try:
                info = await client.show_job(sid)
                scontrol[job["handle"]] = info if info is not None else {}
            except (SlurmMcpError, CommandTimeout, ConnectionDropped, asyncssh.Error, OSError) as e:
                log.warning("%s: scontrol show job %s failed: %s", cluster, sid, e)
        outcomes: dict[str, Outcome] = {}
        specs: dict[str, JobSpec | None] = {}
        for job in plan.jobs:
            sid = _int(job.get("slurm_id"))
            if sid is None:
                continue
            spec = spec_of(job)
            specs[job["handle"]] = spec
            sq = by_id.get(sid)
            if sq is None and job.get("array_size"):
                sq = _array_summary_row(obs["squeue"], sid)
            o = Observation(
                squeue=sq, restart_cnt=(obs["restarts"].get(sid) or {}).get("restarts"), sacct=obs["sacct"].get(sid),
                files=obs["files"].get(job.get("ctrl_dir") or "", {}), scontrol=scontrol.get(job["handle"]),
                healthy=bool(obs.get("healthy")), prior_kinds=plan.prior_kinds.get(job["handle"], set()),
                requeue_history=plan.requeue_history.get(job["handle"], []), observed_late=first)
            outcomes[job["handle"]] = apply_observation(job, plan.attempts.get(job["handle"]), o, now, spec)
        untracked = [r for r in obs["squeue"] if r.get("slurm_id") is not None and int(r["slurm_id"]) not in plan.tracked_ids
                     and (r.get("array_job_id") is None or int(r["array_job_id"]) not in plan.tracked_ids)]
        # healthy observation time for UNCONFIRMED attempts (step 9)
        if obs.get("healthy") and prev_now is not None:
            delta = max(0.0, float(now - prev_now))
            for a in plan.unconfirmed:
                self.unconfirmed_healthy_s[int(a["id"])] = self.unconfirmed_healthy_s.get(int(a["id"]), 0.0) + delta
        for a in plan.unconfirmed:
            self.unconfirmed_healthy_s.setdefault(int(a["id"]), 0.0)
        ctx = _TickContext(cluster=cluster, now=now, caps=caps, profile=profile, plan=plan, obs=obs, outcomes=outcomes,
                           specs=specs, untracked=untracked, observed_late=first,
                           healthy_s={int(a["id"]): self.unconfirmed_healthy_s.get(int(a["id"]), 0.0) for a in plan.unconfirmed},
                           enrich_rows=enrich_rows)
        actions, n_events = await self.store.write_fenced(token, lambda c: self._apply(c, ctx))
        st["ticks"] += 1
        st["events"] += n_events
        done_actions = await self._run_actions(cluster, token, actions)
        await self._resume_queued(cluster, ctx)
        self.failures[cluster] = 0
        self.first_tick[cluster] = False
        self.last_tick[cluster] = time.time()
        self.last_tick_now[cluster] = now
        if self.unreachable.get(cluster):
            self.unreachable[cluster] = False
            await self.events.emit("cluster_recovered", cluster=cluster, summary=f"{cluster} reachable again", ts=now,
                                   token=token)
        await self._after_tick(cluster)
        return TickReport(cluster, True, now=now, events=n_events, actions=done_actions)

    async def _after_tick(self, cluster: str) -> None:
        notify = self.service.components.get("notify")
        fn = getattr(notify, "after_tick", None)
        if callable(fn):
            try:
                await _await_maybe(fn())
            except Exception as e:  # pragma: no cover
                log.warning("notify.after_tick failed: %s", e)
        try:
            await self.events.notify_all()
        except Exception:  # pragma: no cover
            pass

    async def _refresh_snapshot_if_idle(self, cluster: str) -> None:
        last = self.last_snapshot_local.get(cluster, 0.0)
        if time.time() - last < SNAPSHOT_IDLE_S:
            return
        self.last_snapshot_local[cluster] = time.time()
        try:
            await self.service.snapshot(cluster, max_age_s=SNAPSHOT_IDLE_S)
        except Exception as e:
            log.debug("%s: idle snapshot refresh failed: %s", cluster, e)

    async def _tick_failed(self, cluster: str, token: int, exc: BaseException) -> TickReport:
        n = self.failures.get(cluster, 0) + 1
        self.failures[cluster] = n
        st = self.stats.setdefault(cluster, {"ticks": 0, "failed": 0, "events": 0})
        st["failed"] += 1
        log.warning("%s: tick failed (%d in a row): %s", cluster, n, exc)
        if n >= UNREACHABLE_AFTER_FAILURES and not self.unreachable.get(cluster):
            self.unreachable[cluster] = True
            profile = self.service.profile(cluster)
            hint = "check the network; the Monitor retries with backoff"
            try:
                transport = self.service.transport(cluster)
                if not await transport.tcp_probe():
                    hint = profile.requires_vpn_hint or f"{profile.host}:{profile.port or 22} does not accept TCP; VPN or outage?"
            except Exception:
                pass
            await self.events.emit("cluster_unreachable", cluster=cluster,
                                   summary=f"{cluster} unreachable after {n} failed ticks: {str(exc)[:160]}",
                                   payload={"error": str(exc)[:500], "hint": hint}, token=token)
        return TickReport(cluster, False, error=str(exc))

    # -- ledger reads -----------------------------------------------------------------------------------------

    def _plan(self, conn: Any, cluster: str) -> TickPlan:
        store = self.store
        plan = TickPlan()
        live = store.list_jobs(conn, cluster=cluster, states=list(LIVE), order_by="created_local, handle")
        plan.live_count = len(live)
        terminal_unenriched = [r for r in store.select(conn, "jobs_current", {"cluster": cluster, "enriched": 0,
                                                                                "state": [s.value for s in TERMINAL]},
                                                       order_by="terminal_ts")
                               if r.get("slurm_id")]
        plan.enrich = terminal_unenriched
        plan.jobs = [r for r in live if r.get("slurm_id")] + terminal_unenriched
        for r in plan.jobs:
            plan.attempts[r["handle"]] = store.current_attempt(conn, r["handle"]) or {}
            sid = _int(r.get("slurm_id"))
            if sid is not None and sid not in plan.ids:
                plan.ids.append(sid)
            st = _state(r.get("state"))
            wrapped = True
            spec = spec_of(r)
            if spec is not None:
                wrapped = bool(spec.wrap)
            if r.get("ctrl_dir") and ((st == JobState.RUNNING and (wrapped or r.get("kind") == "alloc"))
                                      or (r.get("kind") == "alloc" and st in LIVE)):
                if r["ctrl_dir"] not in plan.ctrl_dirs:
                    plan.ctrl_dirs.append(r["ctrl_dir"])
            if r.get("depends_on_json") and st in LIVE:
                plan.dependents.append(r)
            evs = self.events.events_for_sync(conn, r["handle"])
            plan.prior_kinds[r["handle"]] = prior_kind_keys(e.model_dump() for e in evs)
            plan.requeue_history[r["handle"]] = [
                (int(e.ts or 0), _int(e.payload.get("prev_exit_code"))) for e in evs if e.kind == "requeued"]
        for a in store.select(conn, "attempts", {"cluster": cluster}):
            sid = _int(a.get("slurm_id"))
            if sid is not None:
                plan.tracked_ids.add(sid)
            if a.get("state") == AttemptState.UNCONFIRMED.value:
                job = store.get_job_base(conn, a["handle"])
                if job and _state(job.get("state")) in LIVE and int(job.get("attempt_no") or 0) == int(a["attempt_no"]):
                    plan.unconfirmed.append(a)
                    if a.get("ctrl_dir") and a["ctrl_dir"] not in plan.ctrl_dirs:
                        plan.ctrl_dirs.append(a["ctrl_dir"])
            elif a.get("state") == AttemptState.INTENT.value:
                job = store.get_job_base(conn, a["handle"])
                if job and _state(job.get("state")) in LIVE and int(job.get("attempt_no") or 0) == int(a["attempt_no"]):
                    plan.intents.append(a)
        plan.queued = [r for r in live if _state(r.get("state")) == JobState.QUEUED]
        for r in live:
            if r.get("kind") == "alloc":
                for cmd in store.alloc_cmds_for(conn, r["handle"], states=[CmdState.queued, CmdState.running]):
                    plan.cmds.append(cmd)
                    plan.rc_paths.append(rc_path_for(cmd["out_path"]))
        return plan

    # -- the fenced write (steps 8-12 + applying the outcomes) -------------------------------------------------

    def _apply(self, conn: Any, ctx: "_TickContext") -> tuple[list[dict[str, Any]], int]:
        store = self.store
        actions: list[dict[str, Any]] = []
        n_events = 0
        now = ctx.now
        cluster = ctx.cluster
        user = (ctx.caps or {}).get("user") or ""

        def emit(kind: str, handle: str | None, slurm_id: Any, summary: str, payload: dict[str, Any],
                 state: Any = None) -> None:
            nonlocal n_events
            body = dict(payload)
            if ctx.observed_late:
                body["observed_late"] = True
            self.events.append(conn, kind, handle, cluster, str(slurm_id) if slurm_id is not None else None, summary,
                               body, ts=now, state=state)
            n_events += 1

        # -- outcomes of the pure step ------------------------------------------------------------------------
        for handle, o in ctx.outcomes.items():
            job = next((j for j in ctx.plan.jobs if j["handle"] == handle), None)
            if job is None:
                continue
            attempt = ctx.plan.attempts.get(handle) or {}
            if o.job:
                store.update_job(conn, handle, **o.job)
            if o.attempt and attempt.get("id") is not None:
                store.update_attempt(conn, int(attempt["id"]), **o.attempt)
            new_state = o.job.get("state", job.get("state"))
            for kind, summary, payload in o.events:
                emit(kind, handle, job.get("slurm_id"), summary, payload, state=payload.get("state", new_state))
            for act in o.actions:
                op = act.get("op")
                if op == "wait_history":
                    tgt = target_of(attempt or job)
                    spec = ctx.specs.get(handle)
                    if tgt is not None:
                        limit = time_limit_s(spec)
                        store.insert_wait_history(conn, cluster=cluster, target_key=tgt.key, submit_ts=int(act["submit_ts"]),
                                                  start_ts=int(act["start_ts"]), source="observed",
                                                  gpus=spec.resources.gpus if spec else None,
                                                  hours=round(limit / 3600.0, 3) if limit else None)
                elif op == "abort_cmds":
                    for cmd in store.alloc_cmds_for(conn, handle, states=[CmdState.queued, CmdState.running]):
                        store.update_alloc_cmd(conn, cmd["id"], state=CmdState.aborted, done_ts=now)
                elif op == "hold":
                    store.update_job(conn, handle, hold_reason=act.get("why"), placement_mode="explicit")
                    actions.append({**act, "handle": handle})
                elif op in ("scancel", "show_job"):
                    actions.append({**act, "handle": handle})
            if o.terminal is not None and o.terminal == JobState.NODE_FAIL:
                node = o.attempt.get("node") or attempt.get("node")
                tgt = target_of(attempt or job)
                if node and tgt is not None:
                    store.upsert_target_stats(conn, cluster, tgt.key, last_node_fail_node=node, last_node_fail_local=time.time())
        # -- step 6b: ::CMDS rc files -> cmd_done ------------------------------------------------------------
        for cmd in ctx.plan.cmds:
            rc = ctx.obs["cmds"].get(rc_path_for(cmd["out_path"]))
            if rc is None:
                continue
            state = CmdState.killed if cmd.get("kill_requested_local") else CmdState.done
            store.update_alloc_cmd(conn, cmd["id"], state=state, rc=_int(rc), done_ts=now)
            emit("cmd_done", cmd["id"], None, f"{cmd['id']} finished rc={rc}",
                 {"cmd_id": cmd["id"], "rc": _int(rc), "out_path": cmd["out_path"], "handle": cmd["handle"]},
                 state=state.value)
        # -- steps 8/9: confirm UNCONFIRMED attempts ------------------------------------------------------------
        confirmed_ids: set[int] = set()
        for a in ctx.plan.unconfirmed:
            handle = a["handle"]
            ctrl_dir = a.get("ctrl_dir") or ""
            token = a.get("token") or ""
            comment_prefix = f"slurm-mcp:{handle}:{a.get('attempt_no')}:{token}"
            script = f"{ctrl_dir}/job.sbatch"
            sq_matches = [r for r in ctx.untracked if r.get("command") == script
                          or str(r.get("comment") or "").startswith(comment_prefix)]
            files = ctx.obs["files"].get(ctrl_dir) or {}
            jobid_file = _int(str(files.get("jobid") or "").strip().split("_")[0] or None)
            rec_matches = [r for r in ctx.obs.get("recover") or [] if r.get("slurm_id") is not None
                           and (script in str(r.get("submit_line") or "") or (token and token in str(r.get("submit_line") or "")))]
            candidates: dict[int, dict[str, Any] | None] = {}
            for r in sq_matches:
                candidates[int(r["slurm_id"])] = r
            if jobid_file is not None:
                candidates.setdefault(jobid_file, None)
            for r in rec_matches:
                candidates.setdefault(int(r["slurm_id"]), None)
            if candidates:
                keep = min(candidates)
                source = "squeue" if keep in {int(r["slurm_id"]) for r in sq_matches} else \
                    ("jobid" if keep == jobid_file else "sacct")
                self._confirm(conn, a, keep, candidates.get(keep) or {}, now, source, user, emit)
                confirmed_ids.add(keep)
                for dup in sorted(candidates):
                    if dup == keep:
                        continue
                    actions.append({"op": "scancel", "id": dup, "handle": handle, "duplicate": True})
                    emit("needs_attention", handle, keep, f"{handle}: duplicate job {dup} cancelled, keeping {keep}",
                         {"why": "duplicate_cancelled", "hint": f"submit.sh ran twice; {dup} was scancel'ed",
                          "duplicate_id": dup}, state=JobState.SUBMITTED.value)
                continue
            healthy_s = ctx.healthy_s.get(int(a["id"]), 0.0)
            if healthy_s >= UNCONFIRMED_DEADLINE_S:
                lookalike = any(r.get("workdir") == a.get("workdir") and str(r.get("command") or "").startswith(a.get("ctrl_root") or "\0")
                                for r in ctx.untracked)
                if not lookalike:
                    store.update_attempt(conn, int(a["id"]), state=AttemptState.FAILED, reason="submit_unconfirmed",
                                         end_ts=now, final_state=JobState.FAILED.value)
                    job = store.get_job_base(conn, handle) or {}
                    if transition(job.get("state"), JobState.FAILED):
                        store.update_job(conn, handle, state=JobState.FAILED, terminal_ts=now, end_ts=now,
                                         reason="submit_unconfirmed")
                    emit("needs_attention", handle, None,
                         f"{handle}: submit never confirmed after {int(healthy_s)}s of healthy observation",
                         {"why": "submit_unconfirmed", "hint": "check squeue --me / sacct for a job in the workdir before resubmitting"},
                         state=JobState.FAILED.value)
                    actions.append({"op": "rm_lock", "path": f"{ctrl_dir}/.submit.lock", "handle": handle})
                    self.unconfirmed_healthy_s.pop(int(a["id"]), None)
        # -- step 8b: the rest -> kv.untracked.<cluster> --------------------------------------------------------
        rest = [r for r in ctx.untracked if int(r["slurm_id"]) not in confirmed_ids]
        store.kv_set(conn, UNTRACKED_KEY_PREFIX + cluster, {
            "ts": now, "rows": [{"slurm_id": r.get("slurm_id"), "state": r.get("state"), "partition": r.get("partition"),
                                 "reason": r.get("reason"), "workdir": r.get("workdir"), "command": r.get("command"),
                                 "submit_ts": r.get("submit_ts"), "nodes": r.get("nodes"), "display_id": r.get("display_id")}
                                for r in rest[:MAX_UNTRACKED_ROWS]]})
        # -- step 10a: INTENT sweep ------------------------------------------------------------------------------
        submitter = self.service.components.get("submitter")
        for a in ctx.plan.intents:
            age = time.time() - float(a.get("intent_local") or time.time())
            if age < INTENT_STUCK_S or _submit_active(submitter, a["handle"]):
                continue
            store.update_attempt(conn, int(a["id"]), state=AttemptState.FAILED, reason="submit_stuck", end_ts=now,
                                 final_state=JobState.FAILED.value)
            job = store.get_job_base(conn, a["handle"]) or {}
            if transition(job.get("state"), JobState.FAILED):
                store.update_job(conn, a["handle"], state=JobState.FAILED, terminal_ts=now, end_ts=now, reason="submit_stuck")
            emit("needs_attention", a["handle"], None, f"{a['handle']}: submit never started ({int(age)}s in INTENT)",
                 {"why": "submit_stuck", "hint": "submit_job again (nothing reached the cluster)"}, state=JobState.FAILED.value)
        # -- step 11: dependencies -------------------------------------------------------------------------------
        for job in ctx.plan.dependents:
            handle = job["handle"]
            current = store.get_job(conn, handle) or job
            if _state(current.get("state")) != JobState.SUBMITTED or not current.get("slurm_id"):
                continue
            deps = loads_json(current, "depends_on_json", []) or []
            changed = False
            unsat: dict[str, Any] | None = None
            new_list: list[str] = []
            for d in deps:
                if not isinstance(d, dict) or not d.get("handle"):
                    continue
                dep = store.get_job(conn, d["handle"])
                if dep is None:
                    continue
                dtype = d.get("type") or "afterok"
                dstate = _state(dep.get("state"))
                if dstate in LIVE and dep.get("slurm_id"):
                    if str(dep["slurm_id"]) != str(d.get("resolved_slurm_id")):
                        d["resolved_slurm_id"] = str(dep["slurm_id"])
                        changed = True
                    new_list.append(f"{dtype}:{dep['slurm_id']}")
                elif dstate in TERMINAL:
                    if (dtype == "afterok" and dstate != JobState.COMPLETED) or (dtype == "afternotok" and dstate == JobState.COMPLETED):
                        unsat = {"handle": d["handle"], "type": dtype, "state": dstate.value}
            if changed:
                store.update_job(conn, handle, depends_on_json=deps)
                dep_text = ",".join(new_list)
                actions.append({"op": "update_dependency", "id": current["slurm_id"], "deps": dep_text, "handle": handle})
                emit("dependency_updated", handle, current.get("slurm_id"), f"{handle}: dependency repointed to {dep_text or 'none'}",
                     {"dependent": handle, "new_dependency": dep_text}, state=JobState.SUBMITTED.value)
            prior = ctx.plan.prior_kinds.get(handle, set())
            if unsat is not None and "needs_attention:dependency_unsatisfiable" not in prior:
                store.update_job(conn, handle, hold_reason="dependency_unsatisfiable", placement_mode="explicit")
                actions.append({"op": "hold", "id": current["slurm_id"], "why": "dependency_unsatisfiable", "handle": handle})
                emit("needs_attention", handle, current.get("slurm_id"),
                     f"{handle}: dependency {unsat['type']}:{unsat['handle']} can never be satisfied ({unsat['state']}); held",
                     {"why": "dependency_unsatisfiable", "hint": "fix the dependency then job_control(release), or cancel",
                      "dependency": unsat}, state=JobState.SUBMITTED.value)
        # -- step 12: enrichment -----------------------------------------------------------------------------------
        enrich = ctx.obs.get("enrich") or {"jobs": {}, "last_lines": {}}
        for job in ctx.enrich_rows:
            handle = job["handle"]
            current = store.get_job(conn, handle) or job
            sid = _int(current.get("slurm_id"))
            ej = enrich["jobs"].get(sid) if sid is not None else None
            updates: dict[str, Any] = {}
            last = enrich["last_lines"].get(current.get("stdout_path") or "")
            if last:
                updates["last_line"] = last[:300]
            if ej and ej.get("alloc"):
                updates["enriched"] = 1
                cost = self._cost_actual(current, ej, ctx.caps, ctx.profile, ctx.specs.get(handle) or spec_of(current))
                if cost is not None:
                    updates["cost_actual_su"] = cost
                    est = current.get("cost_est_su")
                    prior = ctx.plan.prior_kinds.get(handle, set())
                    if int(current.get("restarts") or 0) > 0 and isinstance(est, (int, float)) and est > 0 \
                            and cost > est * RESTART_COST_RATIO and "needs_attention:restart_cost" not in prior:
                        emit("needs_attention", handle, sid, f"{handle}: restarts cost {cost:.1f} SU vs {est:.1f} estimated",
                             {"why": "restart_cost", "hint": "consider requeue=False or checkpointing", "cost_actual_su": cost,
                              "cost_est_su": est}, state=current.get("state"))
                if oom_suspected(current, ej) and transition(current.get("state"), JobState.OOM):
                    updates["state"] = JobState.OOM.value
                    attempt = store.current_attempt(conn, handle)
                    if attempt:
                        store.update_attempt(conn, int(attempt["id"]), final_state=JobState.OOM.value)
                    det = {"max_rss": ej.get("max_rss_bytes")}
                    emit("oom", handle, sid, f"{handle} OOM suspected (MaxRSS {ej.get('max_rss_bytes')} of {ej.get('req_mem_bytes')})",
                         {"exit_code": current.get("exit_code"), "exit_signal": current.get("exit_signal"), "max_rss": ej.get("max_rss_bytes"),
                          "req_mem": ej.get("req_mem_bytes"), "oom_suspected": True, "source": "sacct",
                          "hint": terminal_hint(JobState.OOM, det, None), "restarts": current.get("restarts")},
                         state=JobState.OOM.value)
            elif _int(current.get("terminal_ts")) is not None and now - int(current["terminal_ts"]) > ENRICH_GIVE_UP_S:
                updates["enriched"] = 1
            elif _state(current.get("state")) == JobState.LOST:
                updates["enriched"] = 1
            if updates:
                store.update_job(conn, handle, **updates)
        return actions, n_events

    def _confirm(self, conn: Any, attempt: Mapping[str, Any], slurm_id: int, row: Mapping[str, Any], now: int,
                 source: str, user: str, emit: Callable[..., None]) -> None:
        """Steps 8/9: an UNCONFIRMED attempt gets its SLURM id (``ACTIVE``, ``submitted`` event)."""
        store = self.store
        handle = attempt["handle"]
        job = store.get_job_base(conn, handle) or {}
        name = job.get("name") or handle
        upd: dict[str, Any] = {"slurm_id": str(slurm_id), "state": AttemptState.ACTIVE, "confirmed_local": time.time(),
                               "submit_ts": _int(row.get("submit_ts")) or now}
        for pat, col in (("stdout_pattern", "stdout_path"), ("stderr_pattern", "stderr_path")):
            pattern = attempt.get(pat)
            if pattern and not attempt.get(col):
                path = expand_pattern(pattern, slurm_id, name, user)
                if path:
                    upd[col] = path
        store.update_attempt(conn, int(attempt["id"]), **upd)
        job_upd: dict[str, Any] = {"submit_ts": upd["submit_ts"], "stale_ticks": 0, "last_seen_ts": now}
        if row:
            job_upd["slurm_state"] = row.get("state")
            job_upd["reason"] = row.get("reason")
            job_upd["est_start_ts"] = _int(row.get("start_ts"))
        if transition(job.get("state"), JobState.SUBMITTED):
            job_upd["state"] = JobState.SUBMITTED
        store.update_job(conn, handle, **job_upd)
        tgt = target_of(attempt)
        emit("submitted", handle, slurm_id, f"{handle} confirmed as {attempt.get('cluster')} job {slurm_id} (via {source})",
             {"target": tgt.key if tgt else None, "attempt_no": attempt.get("attempt_no"), "est_start_ts": job_upd.get("est_start_ts"),
              "stdout_path": upd.get("stdout_path") or attempt.get("stdout_path"), "workdir": attempt.get("workdir"),
              "ctrl_dir": attempt.get("ctrl_dir"), "injected": [], "warnings": [], "confirmed_by": source},
             state=JobState.SUBMITTED.value)

    @staticmethod
    def _cost_actual(job: Mapping[str, Any], enrich: Mapping[str, Any], caps: Mapping[str, Any] | None, profile: Any,
                     spec: JobSpec | None) -> float | None:
        """``cost_actual_su`` from the charge of the attempt's target and the sacct elapsed time (section 8 Cost)."""
        tgt = target_of(job)
        start, end = _int(job.get("start_ts")), _int(job.get("end_ts"))
        if tgt is None or start is None or end is None or end <= start or caps is None:
            return None
        charge = charge_for(caps, profile, tgt.partitions[0], tgt.gres_type)
        if not isinstance(charge, Mapping):
            return 0.0
        alloc = enrich.get("alloc") or {}
        tres = alloc.get("alloc_tres") if isinstance(alloc.get("alloc_tres"), dict) else {}
        unit = str(charge.get("unit") or "cpu")
        if unit.startswith("gpu"):
            n = tres.get("gres/gpu") or (spec.resources.gpus if spec else 1) or 1
        else:
            n = tres.get("cpu") or (spec.resources.cpus if spec and spec.resources.cpus else 1)
        try:
            return round((end - start) / 3600.0 * float(charge.get("su_per_unit_h") or 0.0) * float(n), 4)
        except (TypeError, ValueError):
            return None

    # -- SSH side effects after the fenced write ----------------------------------------------------------------

    async def _run_actions(self, cluster: str, token: int, actions: list[dict[str, Any]]) -> int:
        if not actions:
            return 0
        client = self.service.client(cluster)
        done = 0
        restore: list[dict[str, Any]] = []
        cancel_ids = sorted({int(a["id"]) for a in actions if a.get("op") == "scancel" and _int(a.get("id")) is not None})
        hold_ids = sorted({int(a["id"]) for a in actions if a.get("op") == "hold" and _int(a.get("id")) is not None})
        try:
            if cancel_ids:
                res = await client.cancel(cancel_ids)
                if res.get("ok"):
                    done += len(cancel_ids)
                else:
                    restore += [a for a in actions if a.get("op") == "scancel" and not a.get("duplicate")]
            if hold_ids:
                res = await client.hold(hold_ids)
                done += len(hold_ids) if res.get("ok") else 0
            for a in actions:
                if a.get("op") == "update_dependency":
                    res = await client.update_dependency(a["id"], a.get("deps") or "")
                    done += 1 if res.get("ok") else 0
                elif a.get("op") == "rm_lock":
                    await client.run(f"rm -rf {_q(a['path'])}", login_shell=False)
                    done += 1
        except (SlurmMcpError, CommandTimeout, ConnectionDropped, asyncssh.Error, OSError) as e:
            log.warning("%s: side effect failed: %s", cluster, e)
            restore += [a for a in actions if a.get("op") == "scancel" and not a.get("duplicate")]
        if restore:
            now = self.service.clock(cluster).remote_now()

            def fn(conn: Any) -> None:
                for a in restore:
                    self.store.update_job(conn, a["handle"], cancel_hard_ts=now)   # retry on the next tick
            try:
                await self.store.write_fenced(token, fn)
            except LeaseLost:
                raise
        return done

    async def _resume_queued(self, cluster: str, ctx: "_TickContext") -> None:
        """Step 10b: hand QUEUED jobs with a free slot to the Submitter (oldest first, at most the free slots)."""
        if not ctx.plan.queued:
            return
        submitter = self.service.components.get("submitter")
        resume = getattr(submitter, "resume_queued", None)
        if not callable(resume):
            return
        caps = ctx.caps or {}
        policy = self.service.placement_policy_cached()
        cap_part = caps.get("pending_cap_part")
        cap_all = caps.get("pending_cap")
        pending_rows = [r for r in ctx.obs["squeue"] if r.get("job_state") == JobState.SUBMITTED]
        used_all = len(pending_rows)
        used_part: dict[str, int] = {}
        for r in pending_rows:
            for p in r.get("partitions") or [r.get("partition")]:
                if p:
                    used_part[p] = used_part.get(p, 0) + 1
        for job in sorted(ctx.plan.queued, key=lambda r: (float(r.get("created_local") or 0), r["handle"])):
            tgt = target_of(job)
            part = tgt.partitions[0] if tgt else None
            cap = policy.max_pending_per_target
            if cap is None:
                cap = cap_part if (cap_part is not None and part) else cap_all
            if cap is not None:
                used = used_part.get(part, 0) if (cap_part is not None and part) else used_all
                if used >= int(cap):
                    continue
            try:
                await _await_maybe(resume(job["handle"]))
            except Exception as e:
                log.warning("%s: resume_queued(%s) failed: %s", cluster, job["handle"], e)
                continue
            used_all += 1
            if part:
                used_part[part] = used_part.get(part, 0) + 1


@dataclass
class _TickContext:
    cluster: str
    now: int
    caps: Mapping[str, Any] | None
    profile: Any
    plan: TickPlan
    obs: Mapping[str, Any]
    outcomes: dict[str, Outcome]
    specs: dict[str, JobSpec | None]
    untracked: list[dict[str, Any]]
    observed_late: bool
    healthy_s: dict[int, float]
    enrich_rows: list[dict[str, Any]]


def _array_summary_row(rows: Iterable[Mapping[str, Any]], base_id: int) -> dict[str, Any] | None:
    """For an array job tracked by its base id: the "most live" element row (RUNNING > PENDING > COMPLETING >
    terminal) so the job-level state follows the elements (section 6.2 squeue -r)."""
    order = {JobState.RUNNING: 0, JobState.SUBMITTED: 1, JobState.COMPLETING: 2}
    best: dict[str, Any] | None = None
    for r in rows:
        if r.get("array_job_id") != base_id:
            continue
        rank = order.get(r.get("job_state"), 3)
        if best is None or rank < order.get(best.get("job_state"), 3):
            best = dict(r)
    return best


def _submit_active(submitter: Any, handle: str) -> bool:
    if submitter is None:
        return False
    fn = getattr(submitter, "is_active", None)
    if callable(fn):
        try:
            return bool(fn(handle))
        except Exception:
            return False
    active = getattr(submitter, "active_handles", None)
    try:
        handles = active() if callable(active) else active
        return handle in set(handles or [])
    except Exception:
        return False


def _q(path: str) -> str:
    return "'" + path.replace("'", "'\\''") + "'"


async def _await_maybe(value: Any) -> Any:
    if asyncio.iscoroutine(value) or asyncio.isfuture(value):
        return await value
    return value


__all__ = [
    "Observation", "Outcome", "TickPlan", "TickReport", "Monitor", "apply_observation", "classify_sacct", "classify_helper",
    "helper_timeout", "cancelled_by", "oom_suspected", "signum", "spec_of", "target_of", "time_limit_s", "rc_path_for",
    "prior_kind_keys", "terminal_hint", "TERMINAL_KINDS", "LOST_HINT",
]
