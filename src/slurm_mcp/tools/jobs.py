"""Job tools: ``submit_job``, ``list_jobs``, ``job_status``, ``job_logs``, ``job_control`` (design section 4 "Jobs";
``plan_job`` is phase 3). The operations are module-level ``async`` functions taking the ``Service`` (CLI parity,
section 1 rule 8); ``register(mcp, service)`` wraps them as tools and attaches the ``submitter`` component lazily
(the tools are registered against a proxy before the lifespan binds the service).

Ids accepted by ``job_status``/``job_logs``/``job_control``: ``j17``, ``j18[7]`` (array element), ``a3``, ``a3.c2``
(allocation command), ``t4`` (transfer) and ``<cluster>:<slurm_id>`` (adoption of an untracked job of mine through
``scontrol -o show job``, section 6.3). ``job_control`` implements the per-state cancel table of section 4 and
changelog item 13 (``cancel.requested`` + ``scancel --signal=TERM --full`` + ``cancel_hard_ts = now + spec.grace_s``
for running jobs, plain ``scancel`` for pending ones, local cancellation before SLURM knows the job).
"""
from __future__ import annotations

import fnmatch
import json
import logging
import re
import secrets
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional

from .. import _mcp
from .._mcp import Context, MCPServer
from ..clock import format_duration
from ..errors import SlurmMcpError, err
from ..models import (
    AllocInfo, AttemptRow, CmdInfo, ControlOutcome, ControlResult, DependencyRow, ExitInfo, JobDetail, JobListResult,
    JobPaths, JobRow, JobSpec, JobStatusResult, LogResult, LogStream, NAME_RE, SubmitResult, TransferInfo,
)
from ..render import expand_pattern
from ..slurm.states import AttemptState, JobState, LIVE, TERMINAL, map_slurm_state
from ..store import loads_json, parse_transfer_handle
from ..submitter import Submitter, target_from_json
from . import BIG_RESULT_META, run_tool

log = logging.getLogger("slurm_mcp.tools.jobs")

CONFIRM_LIMIT = 10                      # section 4 job_control: more ids need confirm=True
ID_RE = re.compile(r"^(?P<base>[ja]\d+)(?:\[(?P<task>\d+)\]|\.c(?P<cmd>\d+))?$")
ADOPT_RE = re.compile(r"^(?P<cluster>[A-Za-z0-9_.-]+):(?P<sid>\d+)$")
STATE_FILTERS: dict[str, tuple[JobState, ...] | None] = {
    "active": tuple(LIVE),
    "pending": (JobState.QUEUED, JobState.UPLOADING, JobState.SUBMITTING, JobState.SUBMITTED),
    "running": (JobState.RUNNING, JobState.COMPLETING),
    "terminal": tuple(TERMINAL),
    "all": None,
}
PRE_SLURM_STATES = {str(JobState.QUEUED), str(JobState.UPLOADING), str(JobState.SUBMITTING)}

SUBMIT_DESC = (
    "Submit a job (JobSpec: name, one of command/script/script_path, resources{time,gpus,gpu_types,cpus,mem,nodes}, "
    "optional workdir/cluster/partition/qos/account, modules/setup/env, array, depends_on ['j12','afterany:j12'], "
    "outputs globs, grace_s, child_signal, max_restarts, stdout/stderr patterns) or a plan_id from plan_job. "
    "placement='auto' (default) ranks every accessible partition/GPU type across clusters and takes the first feasible "
    "target; a target string 'cluster:partition[:gres][@qos]' (or target=) pins it; a list restricts auto placement. "
    "User scripts have their #SBATCH lines converted into spec fields and stripped (see stripped_directives); every "
    "option the server adds is listed in injected. The intent is committed immediately (handle j<N>); helper deploy, "
    "render and sbatch run server-side and the call waits up to wait_s (default 90 s) for the result. state SUBMITTED "
    "carries slurm_id, submit_line, workdir, ctrl_dir and stdout_path; QUEUED means held locally by a pending cap; "
    "SUBMITTING with slurm_id=null means the reply was lost and the Monitor is confirming it (never resubmit). "
    "Definite sbatch errors raise E_QOS*/E_PARTITION*/E_ACCOUNT/... with a fix. hold=True submits with --hold. Then "
    "wait_for_events(job_ids=[handle]) or job_status([handle])."
)
LIST_DESC = (
    "List tracked jobs from the ledger (no SSH): handle, kind, name, cluster, slurm_id, state (QUEUED/UPLOADING/"
    "SUBMITTING/SUBMITTED/RUNNING/COMPLETING/COMPLETED/FAILED/TIMEOUT/OOM/CANCELLED/PREEMPTED/NODE_FAIL/LOST), reason, "
    "target, elapsed_s, time_limit_s, restarts, moves, est_start_ts. state selects active (default), pending, running, "
    "terminal or all; since_h limits to jobs created in the last N hours; name is an exact name or glob; kind job/alloc/"
    "all; include_untracked=True appends this user's squeue rows the server does not track (handle=null; adopt one with "
    "job_status(['<cluster>:<slurm_id>'])). ~15 tokens per row; counts_by_state summarises; truncated=true when more "
    "than limit exist."
)
STATUS_DESC = (
    "Detailed status of jobs, allocations, commands or transfers: ids like j17, j18[7] (array element), a3, a3.c2, t4 or "
    "'<cluster>:<slurm_id>' (adopts an untracked job of mine with wrap=False). Per job: the list row plus submit/start/"
    "end timestamps (cluster epoch), exit {rc, signal}, node, progress (from progress.json), heartbeat_age_s, "
    "last_log_line, cost_su/cost_worst_su, attempts_count, paths {cluster, workdir, ctrl_dir, stdout, stderr} of the "
    "current attempt, dependencies [{handle, type, status}], dependents, alloc/transfer/cmd blocks where relevant and "
    "next_action (the call to make next). detail='full' adds the attempt history and the raw scontrol fields of live "
    "jobs. Runs one monitor tick first when the last one is older than 20 s. Prefer wait_for_events to poll."
)
LOGS_DESC = (
    "Read a job's stdout/stderr (stream out, err or both): the last tail_lines (default 80), or grep -n -E matches, or a "
    "byte window from offset, capped at max_chars (default 12000; truncated=true with next_offset for paging). id may be "
    "j17, j18[7] (the element's file) or a3.c2 (the command's .out). Paths come from the current attempt (the cluster "
    "it runs on now). E_NO_LOG_YET names the state and what to wait for when the file does not exist yet (pending job, "
    "pattern needing the controller, adopted job without a known StdOut). For arbitrary files use remote_read."
)
CONTROL_DESC = (
    "Control jobs: action cancel|hold|release|requeue|signal on ids (j17, j18[7], a3, a3.c2, t4). cancel decides per id "
    "from the ledger: QUEUED/UPLOADING/SUBMITTING jobs are cancelled locally (the submit task stops; an unconfirmed "
    "sbatch is cancelled once its id is known: outcome cancel_pending_confirmation); pending SLURM jobs get a plain "
    "scancel (outcome cancelled); running jobs with graceful=True (default) get cancel.requested + "
    "scancel --signal=TERM --full and a hard scancel at hard_kill_ts = now + spec.grace_s (outcome terminating); "
    "graceful=False kills now. a3 releases the allocation (release file + scancel); a3.c2 writes the command's kill "
    "file; t4 cancels the transfer. hold/release/requeue map to scontrol; hold pins placement (the rebalancer leaves "
    "the job alone) and release restores it; signal sends 'scancel --signal=<sig> --full'. Partial failures are per-id "
    "outcomes, not errors. More than 10 ids require confirm=True (E_CONFIRM_REQUIRED). Runs a tick first when stale."
)


def _submitter(service: Any) -> Submitter:
    """The ``submitter`` component, attached on first use (the server registers tools before the lifespan)."""
    comp = service.components.get("submitter")
    if comp is None:
        comp = service.attach("submitter", Submitter(service))
    return comp


def parse_id(text: str) -> dict[str, Any]:
    """``j17`` / ``j18[7]`` / ``a3`` / ``a3.c2`` / ``t4`` / ``<cluster>:<slurm_id>`` -> ``{kind, base, task, cmd, ...}``."""
    s = (text or "").strip()
    m = ID_RE.match(s)
    if m:
        return {"kind": "job", "id": s, "base": m.group("base"),
                "task": int(m.group("task")) if m.group("task") is not None else None,
                "cmd": s if m.group("cmd") is not None else None}
    tid = parse_transfer_handle(s)
    if tid is not None:
        return {"kind": "transfer", "id": s, "transfer_id": tid}
    m = ADOPT_RE.match(s)
    if m:
        return {"kind": "adopt", "id": s, "cluster": m.group("cluster"), "slurm_id": m.group("sid")}
    raise err("E_UNKNOWN_ID", f"{s!r} is not a job id")


def _target_key(row: dict[str, Any]) -> str | None:
    tgt = target_from_json(row.get("target_json"))
    return tgt.key if tgt else None


def _spec(row: dict[str, Any]) -> dict[str, Any]:
    return loads_json(row, "spec_json", {}) or {}


def _time_limit_s(row: dict[str, Any]) -> int | None:
    from ..clock import parse_duration
    return parse_duration(((_spec(row).get("resources") or {}).get("time")))


def _elapsed_s(row: dict[str, Any], now_ts: int) -> int | None:
    start = row.get("start_ts")
    if start is None:
        return None
    end = row.get("end_ts") if row.get("state") in {str(s) for s in TERMINAL} else None
    return max(0, int((end if end is not None else now_ts) - int(start)))


def _job_row(row: dict[str, Any], now_ts: int) -> JobRow:
    return JobRow(handle=row["handle"], kind=row.get("kind") or "job", name=row.get("name"), cluster=row.get("cluster"),
                  slurm_id=row.get("slurm_id"), state=JobState(row["state"]), reason=row.get("reason"),
                  target=_target_key(row), elapsed_s=_elapsed_s(row, now_ts), time_limit_s=_time_limit_s(row),
                  restarts=int(row.get("restarts") or 0), moves=int(row.get("moves") or 0),
                  est_start_ts=row.get("est_start_ts"))


def _untracked_rows(rows: Any, cluster: str) -> list[JobRow]:
    """``kv.untracked.<cluster>`` -> rows with ``handle=None`` (section 4 ``list_jobs``).

    The Monitor stores ``{"ts": <cluster epoch>, "rows": [...]}`` (section 5.2 step 8b); a bare list is also
    accepted so an older ledger keeps working.
    """
    if isinstance(rows, Mapping):
        rows = rows.get("rows") or []
    out: list[JobRow] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        state = map_slurm_state(r.get("state")) if r.get("state") else None
        out.append(JobRow(handle=None, kind="job", name=r.get("name"), cluster=cluster, slurm_id=str(r.get("slurm_id") or ""),
                          state=state, reason=r.get("reason"), target=f"{cluster}:{r['partition']}" if r.get("partition") else None,
                          time_limit_s=r.get("time_limit_s"), est_start_ts=r.get("start_ts")))
    return out


def _next_action(row: dict[str, Any]) -> str:
    handle = row["handle"]
    state = row.get("state")
    ids = f"['{handle}']"
    if state in PRE_SLURM_STATES:
        return f"wait_for_events(kinds=['submitted','submit_failed','queued'], job_ids={ids})"
    if state == str(JobState.SUBMITTED):
        return f"wait_for_events(job_ids={ids}, timeout_s=600)"
    if state in (str(JobState.RUNNING), str(JobState.COMPLETING)):
        return f"job_logs('{handle}') or wait_for_events(job_ids={ids}, timeout_s=600)"
    if state == str(JobState.COMPLETED):
        return f"collect_results({ids})"
    if state in (str(JobState.FAILED), str(JobState.OOM), str(JobState.TIMEOUT)):
        rc = row.get("exit_code")
        return f"{state}{f' rc={rc}' if rc is not None else ''}: call job_logs('{handle}', stream='err')"
    if state == str(JobState.LOST):
        return f"run_command('{row.get('cluster')}', 'sacct -j {row.get('slurm_id')}')"
    return f"job_status({ids})"


# --- submit_job -------------------------------------------------------------------------------------------------------

async def submit_job(service: Any, job: JobSpec | dict[str, Any] | None = None, plan_id: str | None = None,
                     placement: str | Sequence[str] = "auto", target: str | None = None, hold: bool = False,
                     wait_s: int = 90, ctx: Context | None = None) -> SubmitResult:
    """Section 4 ``submit_job``: commit the intent, await the ``SubmitTask`` for ``wait_s`` with progress every 5 s."""
    sub = _submitter(service)
    handle, _task = await sub.submit(job, placement=placement, target=target, plan_id=plan_id, hold=hold)
    tick = {"n": 0}

    async def progress(message: str) -> None:
        tick["n"] += 1
        if ctx is not None:
            try:
                await ctx.report_progress(tick["n"], None, f"{handle}: {message}")
            except Exception:  # a closed client must never fail the submit
                pass
    result = await sub.await_result(handle, wait_s=max(0, int(wait_s)), progress_cb=progress)
    result.unread_events = await service.unread()
    return result


# --- list_jobs ---------------------------------------------------------------------------------------------------------

async def list_jobs(service: Any, cluster: str | None = None, state: str = "active", since_h: float | None = None,
                    name: str | None = None, kind: str = "all", include_untracked: bool = False, limit: int = 50,
                    ) -> JobListResult:
    """Section 4 ``list_jobs`` from ``jobs_current`` (no SSH, except the untracked refresh below)."""
    if state not in STATE_FILTERS:
        raise err("E_INVALID_SPEC", f"state must be one of {', '.join(STATE_FILTERS)}, got {state!r}")
    if kind not in ("job", "alloc", "all"):
        raise err("E_INVALID_SPEC", f"kind must be job|alloc|all, got {kind!r}")
    if cluster is not None:
        service.profile(cluster)
    states = STATE_FILTERS[state]
    store = service.store

    # ``kv.untracked.<cluster>`` is written by the Monitor tick (section 5.2 step 8). On a freshly started server
    # -- or when nothing is tracked, so the tick runs at its slowest cadence -- that key is empty or stale, and
    # "what is in my queue?" would answer "nothing" while the real queue is full. Refresh first (one squeue).
    if include_untracked:
        for name_ in ([cluster] if cluster else service.registry.names()):
            try:
                await service.tick_if_stale(name_, 20)
            except Exception as exc:  # an unreachable cluster must not fail the whole listing
                log.warning("list_jobs: tick for %s failed: %s", name_, exc)

    def fn(conn: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows = store.list_jobs(conn, states=list(states) if states is not None else None,
                               kind=None if kind == "all" else kind, cluster=cluster)
        untracked: dict[str, Any] = {}
        if include_untracked:
            for name_ in ([cluster] if cluster else service.registry.names()):
                untracked[name_] = store.kv_get(conn, f"untracked.{name_}") or []
        return rows, untracked
    rows, untracked = await store.read(fn)
    if since_h is not None:
        cutoff = time.time() - float(since_h) * 3600.0
        rows = [r for r in rows if float(r.get("created_local") or 0) >= cutoff]
    if name:
        rows = [r for r in rows if r.get("name") == name or fnmatch.fnmatchcase(str(r.get("name") or ""), name)]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    limit_n = max(1, int(limit))
    truncated = len(rows) > limit_n
    now_by: dict[str, int] = {}
    out: list[JobRow] = []
    for r in rows[:limit_n]:
        c = r.get("cluster")
        if c not in now_by:
            try:
                now_by[c] = service.clock(c).remote_now()
            except SlurmMcpError:
                now_by[c] = int(time.time())
        out.append(_job_row(r, now_by[c]))
    n_untracked = 0
    for name_, rows_u in untracked.items():
        extra = _untracked_rows(rows_u, name_)
        n_untracked += len(extra)
        out.extend(extra)
    parts = [f"{v} {k}" for k, v in sorted(counts.items())]
    summary = f"{len(rows)} {state} job(s)" + (f" ({', '.join(parts)})" if parts else "")
    if n_untracked:
        summary += f"; {n_untracked} untracked squeue row(s)"
    if truncated:
        summary += f"; showing {limit_n}"
    nxt = None
    if truncated:
        nxt = "raise limit or narrow with cluster/state/name"
    elif out:
        nxt = "job_status([handle]) for details; wait_for_events(timeout_s=300) to follow"
    return JobListResult(summary=summary, unread_events=await service.unread(), jobs=out, counts_by_state=counts,
                         truncated=truncated, next=nxt)


# --- job_status ------------------------------------------------------------------------------------------------------

def _dependencies(store: Any, conn: Any, row: dict[str, Any]) -> list[DependencyRow]:
    out: list[DependencyRow] = []
    for d in loads_json(row, "depends_on_json", []) or []:
        if not isinstance(d, dict):
            continue
        dep = store.get_job(conn, d.get("handle"))
        out.append(DependencyRow(handle=str(d.get("handle")), type=str(d.get("type") or "afterok"),
                                 status=dep["state"] if dep else "unknown"))
    return out


def _dependents(conn: Any, handle: str) -> list[str]:
    rows = conn.execute("SELECT handle, depends_on_json, state FROM jobs WHERE depends_on_json IS NOT NULL").fetchall()
    out: list[str] = []
    for r in rows:
        try:
            deps = json.loads(r["depends_on_json"]) if r["depends_on_json"] else []
        except ValueError:
            deps = []
        if any(isinstance(d, dict) and d.get("handle") == handle for d in deps):
            out.append(r["handle"])
    return out


def _detail_from_row(service: Any, conn: Any, row: dict[str, Any], now_ts: int, *, full: bool) -> JobDetail:
    store = service.store
    base = _job_row(row, now_ts)
    heartbeat = row.get("heartbeat_ts")
    progress = None
    if row.get("progress_json"):
        try:
            progress = json.loads(row["progress_json"])
        except ValueError:
            progress = row["progress_json"][:1024]
    attempts = store.attempts_for(conn, row["handle"])
    detail = JobDetail(
        **base.model_dump(), submit_ts=row.get("submit_ts"), start_ts=row.get("start_ts"), end_ts=row.get("end_ts"),
        exit=ExitInfo(rc=row.get("exit_code"), signal=row.get("exit_signal")), node=row.get("node"), progress=progress,
        heartbeat_age_s=(max(0.0, float(now_ts - int(heartbeat))) if heartbeat is not None else None),
        last_log_line=row.get("last_line"), cost_su=row.get("cost_actual_su") or row.get("cost_est_su"),
        cost_worst_su=row.get("cost_worst_su"), attempts_count=len(attempts),
        paths=JobPaths(cluster=row.get("cluster"), workdir=row.get("workdir"), ctrl_dir=row.get("ctrl_dir"),
                       stdout=row.get("stdout_path"), stderr=row.get("stderr_path")),
        dependencies=_dependencies(store, conn, row), dependents=_dependents(conn, row["handle"]),
        next_action=_next_action(row))
    if row.get("kind") == "alloc":
        outstanding = len(store.alloc_cmds_for(conn, row["handle"], states=["queued", "running"]))
        detail.alloc = AllocInfo(ready=bool(row.get("alloc_ready")), end_ts=row.get("alloc_end_ts"),
                                 cmds_outstanding=outstanding)
    if full:
        detail.attempts = [AttemptRow(attempt_no=int(a["attempt_no"]), state=AttemptState(a["state"]), cluster=a.get("cluster"),
                                      slurm_id=a.get("slurm_id"), target=_target_key(a), cause=a.get("cause"),
                                      submit_ts=a.get("submit_ts"), end_ts=a.get("end_ts"), workdir=a.get("workdir"))
                           for a in attempts]
    return detail


async def _adopt(service: Any, cluster: str, slurm_id: str) -> str:
    """``<cluster>:<slurm_id>`` -> an existing handle, or a new ``wrap=False`` job from ``scontrol show job``."""
    service.profile(cluster)
    store = service.store
    existing = await store.read(lambda c: store.select_one(c, "attempts", cluster=cluster, slurm_id=slurm_id))
    if existing is not None:
        return existing["handle"]
    caps = await service.caps(cluster)
    info = await service._guard(cluster, service.client(cluster).show_job(slurm_id))
    if not info:
        raise err("E_UNKNOWN_ID", f"{cluster}:{slurm_id} is not in the controller's memory")
    user = caps.get("user") or service.profile(cluster).user
    uid = str((info.get("raw") or {}).get("UserId") or "")
    if user and uid and not uid.startswith(f"{user}("):
        raise err("E_UNKNOWN_ID", f"{cluster}:{slurm_id} belongs to {uid}, not to {user}")
    raw_name = str(info.get("job_name") or f"job{slurm_id}")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_name)[:64] or f"job{slurm_id}"
    if not NAME_RE.match(name):
        name = f"job{slurm_id}"
    tl = info.get("time_limit_s")
    spec = JobSpec(name=name, command=f"# adopted from {cluster}:{slurm_id}", cluster=cluster,
                   partition=info.get("partition"), qos=info.get("qos"), account=info.get("account"),
                   resources={"time": format_duration(tl) or "01:00:00"}, wrap=False,
                   workdir=info.get("work_dir"), stdout=info.get("std_out"), stderr=info.get("std_err"))
    state = info.get("state") or JobState.SUBMITTED
    profile = service.profile(cluster)
    from ..submitter import derive_paths
    now_local = time.time()

    def fn(conn: Any) -> str:
        handle = store.next_handle(conn, "job")
        paths = derive_paths(profile, caps, spec, handle, 1)
        workdir = info.get("work_dir") or paths["workdir"]
        terminal = state in TERMINAL
        store.insert_job(conn, handle=handle, kind="job", name=name, state=state, spec_json=spec.model_dump(),
                         placement_mode="explicit", slurm_state=info.get("job_state"), reason=info.get("reason"),
                         submit_ts=info.get("submit_time_ts"), start_ts=info.get("start_time_ts"),
                         end_ts=info.get("end_time_ts") if terminal else None, restarts=int(info.get("restarts") or 0),
                         exit_code=(info.get("exit_code") or (None,))[0] if terminal else None,
                         terminal_ts=info.get("end_time_ts") if terminal else None, created_local=now_local,
                         updated_local=now_local)
        tgt = {"cluster": cluster, "partitions": [p for p in str(info.get("partition") or "").split(",") if p] or ["unknown"],
               "gres_type": None, "qos": info.get("qos"), "account": info.get("account")}
        store.insert_attempt(conn, handle=handle, attempt_no=1, cluster=cluster, token="t-" + secrets.token_hex(6),
                             ctrl_root=paths["ctrl_root"], ctrl_dir=paths["ctrl_dir"], workdir=workdir,
                             stdout_pattern=info.get("std_out"), stderr_pattern=info.get("std_err"),
                             stdout_path=info.get("std_out"), stderr_path=info.get("std_err"), node=info.get("batch_host"),
                             slurm_id=str(slurm_id), target_json=tgt, state=AttemptState.DONE if terminal else AttemptState.ACTIVE,
                             cause="user", confirmed_local=now_local, submit_ts=info.get("submit_time_ts"),
                             submit_line=None)
        return handle
    return await store.write(fn)


async def job_status(service: Any, ids: Sequence[str], detail: str = "normal") -> JobStatusResult:
    """Section 4 ``job_status`` (ledger + one tick when stale; adoption of ``<cluster>:<slurm_id>``)."""
    if detail not in ("normal", "full"):
        raise err("E_INVALID_SPEC", f"detail must be normal|full, got {detail!r}")
    if not ids:
        raise err("E_INVALID_SPEC", "ids is empty")
    parsed = [parse_id(i) for i in ids]
    store = service.store
    handles: list[tuple[dict[str, Any], str]] = []
    for p in parsed:
        if p["kind"] == "adopt":
            handles.append((p, await _adopt(service, p["cluster"], p["slurm_id"])))
        elif p["kind"] == "job":
            handles.append((p, p["base"]))
        else:
            handles.append((p, p["id"]))
    rows = await store.read(lambda c: {h: store.get_job(c, h) for _, h in handles if not h.startswith("t")})
    clusters = {r["cluster"] for r in rows.values() if r and r.get("cluster")}
    for cluster in clusters:
        try:
            await service.tick_if_stale(cluster, 20)
        except Exception:  # a failed tick never blocks a status read
            pass
    if clusters:
        rows = await store.read(lambda c: {h: store.get_job(c, h) for _, h in handles if not h.startswith("t")})
    full = detail == "full"
    now_by: dict[str, int] = {}
    details: list[JobDetail] = []

    def build(conn: Any) -> list[JobDetail]:
        out: list[JobDetail] = []
        for p, h in handles:
            if p["kind"] == "transfer":
                tr = store.get_transfer(conn, p["transfer_id"])
                if tr is None:
                    raise err("E_UNKNOWN_ID", f"no transfer {p['id']!r}")
                out.append(JobDetail(handle=p["id"], kind="job", name=f"{tr['kind']} {tr['remote']}", cluster=tr["cluster"],
                                     transfer=TransferInfo(state=tr["state"], files_done=int(tr.get("files_done") or 0),
                                                           files_total=tr.get("files_total"), bytes_done=int(tr.get("bytes_done") or 0),
                                                           bytes_total=tr.get("bytes_total"), error=tr.get("error")),
                                     next_action=f"wait_for_events(kinds=['transfer_done','transfer_failed'], job_ids=['{p['id']}'])"))
                continue
            row = rows.get(h)
            if row is None:
                raise err("E_UNKNOWN_ID", f"no job {h!r}")
            c = row.get("cluster")
            if c not in now_by:
                now_by[c] = service.clock(c).remote_now()
            d = _detail_from_row(service, conn, row, now_by[c], full=full)
            if p.get("cmd"):
                cmd = store.get_alloc_cmd(conn, p["cmd"])
                if cmd is None:
                    raise err("E_UNKNOWN_ID", f"no command {p['cmd']!r}")
                d.handle = p["cmd"]
                d.cmd = CmdInfo(state=cmd["state"], rc=cmd.get("rc"), started_ts=cmd.get("started_ts"), done_ts=cmd.get("done_ts"))
                d.next_action = (f"job_logs('{p['cmd']}')" if cmd["state"] in ("done", "killed")
                                 else f"wait_for_events(kinds=['cmd_done'], job_ids=['{p['cmd']}'])")
            elif p.get("task") is not None:
                task = next((t for t in store.array_tasks_for(conn, h) if int(t["task_id"]) == p["task"]), None)
                d.handle = p["id"]
                if task is not None:
                    d.state = JobState(task["state"]) if task.get("state") in JobState.__members__ else d.state
                    d.slurm_id = task.get("slurm_id") or (f"{row.get('slurm_id')}_{p['task']}" if row.get("slurm_id") else None)
                    d.exit = ExitInfo(rc=task.get("exit_code"))
                    d.start_ts, d.end_ts, d.node = task.get("start_ts"), task.get("end_ts"), task.get("node")
                elif row.get("slurm_id"):
                    d.slurm_id = f"{row['slurm_id']}_{p['task']}"
            out.append(d)
        return out
    details = await store.read(build)
    parts = [f"{d.handle} {d.state.value if d.state else (d.transfer.state if d.transfer else '?')}"
             + (f" ({d.reason})" if d.reason and d.state and d.state.value in ("SUBMITTED", "QUEUED") else "")
             for d in details]
    summary = "; ".join(parts)
    nxt = details[0].next_action if len(details) == 1 else "wait_for_events(timeout_s=300) to follow them"
    return JobStatusResult(summary=summary, unread_events=await service.unread(), jobs=details, next=nxt)


# --- job_logs ---------------------------------------------------------------------------------------------------------

async def job_logs(service: Any, id: str, stream: str = "out", tail_lines: int = 80, grep: str | None = None,
                   offset: int | None = None, max_chars: int = 12000) -> LogResult:
    """Section 4 ``job_logs``: paths from the current attempt; ``E_NO_LOG_YET`` when the file is not there yet."""
    if stream not in ("out", "err", "both"):
        raise err("E_INVALID_SPEC", f"stream must be out|err|both, got {stream!r}")
    p = parse_id(id)
    if p["kind"] != "job":
        raise err("E_INVALID_SPEC", f"job_logs takes a job id (j17, j18[7], a3.c2), got {id!r}",
                  fix="use job_status for transfers")
    store = service.store

    def load(conn: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        row = store.get_job(conn, p["base"])
        cmd = store.get_alloc_cmd(conn, p["cmd"]) if p.get("cmd") else None
        task = None
        if row is not None and p.get("task") is not None:
            task = next((t for t in store.array_tasks_for(conn, p["base"]) if int(t["task_id"]) == p["task"]), None)
        return row, cmd, task
    row, cmd, task = await store.read(load)
    if row is None:
        raise err("E_UNKNOWN_ID", f"no job {p['base']!r}")
    if p.get("cmd") and cmd is None:
        raise err("E_UNKNOWN_ID", f"no command {p['cmd']!r}")
    cluster = row["cluster"]
    state = str(row["state"])
    ids = f"['{id}']"
    wait_hint = f"wait_for_events(job_ids={ids}, timeout_s=600)"

    def no_log(why: str, hint: str | None = None) -> SlurmMcpError:
        return err("E_NO_LOG_YET", f"{id}: {why}", state=state, next=hint or wait_hint)

    if cmd is not None:
        out_path: str | None = cmd["out_path"]
        err_path: str | None = None
        if cmd.get("state") == "queued":
            raise no_log("the command has not started", f"wait_for_events(kinds=['cmd_done'], job_ids={ids})")
    else:
        if state in PRE_SLURM_STATES:
            raise no_log(f"the job is {state} (not in SLURM yet)",
                         f"wait_for_events(kinds=['submitted','submit_failed'], job_ids={ids})")
        out_path, err_path = row.get("stdout_path"), row.get("stderr_path")
        if p.get("task") is not None:
            caps = service.caps_cached(cluster) or {}
            user = caps.get("user") or service.profile(cluster).user
            base_id = row.get("slurm_id")
            if base_id:
                out_path = expand_pattern(row.get("stdout_pattern") or "", base_id, row["name"], user, array_index=p["task"]) \
                    if row.get("stdout_pattern") else None
                err_path = expand_pattern(row.get("stderr_pattern") or "", base_id, row["name"], user, array_index=p["task"]) \
                    if row.get("stderr_pattern") else None
        if out_path is None and err_path is None:
            why = "the output path is not known yet" if row.get("slurm_id") else "the job has no SLURM id yet"
            if (_spec(row).get("wrap") is False) and row.get("slurm_id"):
                raise no_log("adopted job without a known StdOut", f"remote_read('{cluster}', '<path>')")
            raise no_log(why + " (patterns with %N/%n/%t/%s resolve on the first tick after RUNNING)")
    client = service.client(cluster)
    await service.caps(cluster)

    async def read(path: str | None) -> LogStream | None:
        if path is None:
            return None
        try:
            out = await service._guard(cluster, client.read_file(path, tail_lines=tail_lines, grep=grep, offset=offset,
                                                                  max_chars=max_chars))
        except SlurmMcpError as e:
            if e.code == "E_INVALID_SPEC" and "no such file" in e.message:
                raise no_log(f"{path} does not exist yet" + ("" if state != str(JobState.SUBMITTED) else " (job pending)"))
            raise
        return LogStream(text=out["text"], size=out["size"], next_offset=out["next_offset"], path=path,
                         truncated=out["truncated"])
    out_s = await read(out_path) if stream in ("out", "both") else None
    if stream in ("err", "both"):
        if err_path is None or err_path == out_path:
            err_s = out_s if (err_path == out_path and out_s is not None) else await read(err_path or out_path)
        else:
            err_s = await read(err_path)
    else:
        err_s = None
    shown = out_s or err_s
    lines = (shown.text.count("\n") + (1 if shown.text and not shown.text.endswith("\n") else 0)) if shown else 0
    summary = f"{id} {state}: {lines} line(s) of {stream}" + (f" from {shown.path}" if shown else "")
    if shown and shown.truncated:
        summary += " [truncated]"
    nxt = None
    if shown and shown.next_offset is not None:
        nxt = f"job_logs('{id}', offset={shown.next_offset})"
    elif state in (str(JobState.SUBMITTED), str(JobState.RUNNING), str(JobState.COMPLETING)):
        nxt = wait_hint
    return LogResult(summary=summary, unread_events=await service.unread(), id=id, state=state, out=out_s, err=err_s, next=nxt)


# --- job_control -----------------------------------------------------------------------------------------------------

def _slurm_target_id(row: dict[str, Any], task: int | None) -> str | None:
    sid = row.get("slurm_id")
    if not sid:
        return None
    return f"{sid}_{task}" if task is not None else str(sid)


async def _cancel_one(service: Any, p: dict[str, Any], row: dict[str, Any], graceful: bool, reason: str | None,
                      now_local: float) -> ControlOutcome:
    """The per-state cancel decision of section 4 ``job_control`` for one job/element."""
    store = service.store
    handle = row["handle"]
    ident = p["id"]
    cluster = row["cluster"]
    state = str(row["state"])
    att_state = row.get("attempt_state")
    clock = service.clock(cluster)
    now_ts = clock.remote_now()
    dependents = await store.read(lambda c: [h for h in _dependents(c, handle)
                                             if (store.get_job_base(c, h) or {}).get("state") in {str(s) for s in LIVE}])
    dep_note = ""
    if dependents:
        names = []
        for h in dependents:
            deps = await store.read(lambda c, h=h: loads_json(store.get_job_base(c, h) or {}, "depends_on_json", []))
            typ = next((d.get("type") for d in deps or [] if isinstance(d, dict) and d.get("handle") == handle), "afterok")
            names.append(f"{h} ({typ})")
        dep_note = f"; dependents: {', '.join(names)} will be re-evaluated"
    if state in {str(s) for s in TERMINAL}:
        return ControlOutcome(id=ident, accepted=False, outcome="already_terminal", message=f"job is {state}")
    sid = _slurm_target_id(row, p.get("task"))
    if state in PRE_SLURM_STATES or not sid:
        if att_state == str(AttemptState.UNCONFIRMED):
            await store.write(lambda c: store.update_job(c, handle, cancel_requested_ts=now_ts))
            return ControlOutcome(id=ident, accepted=True, outcome="cancel_pending_confirmation",
                                  message="submit unconfirmed; cancelled once the Monitor names its id" + dep_note)
        sub = service.components.get("submitter")
        if sub is not None:
            sub.cancel_task(handle)

        def fn(conn: Any) -> None:
            att = store.current_attempt(conn, handle)
            if att is not None:
                store.update_attempt(conn, int(att["id"]), state=AttemptState.FAILED, cause="user",
                                     reason=reason or "cancelled before submit", end_ts=now_ts, final_state=str(JobState.CANCELLED))
            store.update_job(conn, handle, state=JobState.CANCELLED, reason=reason or "cancelled by user", end_ts=now_ts,
                             terminal_ts=now_ts, cancel_requested_ts=now_ts)
            service.events.append(conn, "cancelled", handle, cluster, None, f"{handle} cancelled before it reached SLURM",
                                  {"by": "agent", "cause": "user", "exit_code": None, "restarts": int(row.get("restarts") or 0)},
                                  ts=now_ts, state=JobState.CANCELLED)
        await store.write(fn)
        return ControlOutcome(id=ident, accepted=True, outcome="cancelled", message="cancelled locally" + dep_note)
    client = service.client(cluster)
    if state == str(JobState.SUBMITTED) or not graceful:
        res = await service._guard(cluster, client.cancel([sid]))
        errors = res.get("errors") or []
        await store.write(lambda c: store.update_job(c, handle, cancel_requested_ts=now_ts, cancel_hard_ts=None))
        if p.get("task") is None and row.get("kind") == "alloc":
            pass
        if errors:
            return ControlOutcome(id=ident, accepted=False, outcome="scancel_error", message=errors[0].get("message"))
        return ControlOutcome(id=ident, accepted=True, outcome="cancelled",
                              message=f"scancel {sid}" + dep_note)
    # RUNNING / COMPLETING, graceful
    grace = int(_spec(row).get("grace_s") or 120)
    hard_ts = now_ts + grace
    try:
        await service._guard(cluster, client.write_file(f"{row['ctrl_dir']}/cancel.requested",
                                                        f"{now_ts} {reason or 'agent'}\n", "overwrite", mkdirs=True))
    except SlurmMcpError as e:  # the ctrl dir of an adopted job does not exist: the TERM still goes out
        if _spec(row).get("wrap", True):
            raise
    res = await service._guard(cluster, client.cancel([sid], signal="TERM", full=True))
    await store.write(lambda c: store.update_job(c, handle, cancel_requested_ts=now_ts, cancel_hard_ts=hard_ts))
    sub = service.components.get("submitter")
    if sub is not None:
        await sub._kick(cluster)
    errors = res.get("errors") or []
    if errors:
        return ControlOutcome(id=ident, accepted=False, outcome="scancel_error", message=errors[0].get("message"),
                              hard_kill_ts=hard_ts)
    return ControlOutcome(id=ident, accepted=True, outcome="terminating", hard_kill_ts=hard_ts,
                          message=f"TERM sent; hard kill at {hard_ts} (grace_s={grace})" + dep_note)


async def job_control(service: Any, ids: Sequence[str], action: str, signal: str | None = None, graceful: bool = True,
                      reason: str | None = None, confirm: bool = False) -> ControlResult:
    """Section 4 ``job_control``: per-id outcomes; partial failures are not errors."""
    if action not in ("cancel", "hold", "release", "requeue", "signal"):
        raise err("E_INVALID_SPEC", f"action must be cancel|hold|release|requeue|signal, got {action!r}")
    if not ids:
        raise err("E_INVALID_SPEC", "ids is empty")
    if len(ids) > CONFIRM_LIMIT and not confirm:
        raise err("E_CONFIRM_REQUIRED", f"{action} on {len(ids)} ids (limit {CONFIRM_LIMIT} without confirm)")
    if action == "signal" and not signal:
        raise err("E_INVALID_SPEC", "signal requires signal='USR1'-style name")
    parsed = [parse_id(i) for i in ids]
    store = service.store
    rows = await store.read(lambda c: {p["base"]: store.get_job(c, p["base"]) for p in parsed if p["kind"] == "job"})
    clusters = {r["cluster"] for r in rows.values() if r}
    for cluster in clusters:
        try:
            await service.tick_if_stale(cluster, 20)
        except Exception:
            pass
    if clusters:
        rows = await store.read(lambda c: {p["base"]: store.get_job(c, p["base"]) for p in parsed if p["kind"] == "job"})
    now_local = time.time()
    results: list[ControlOutcome] = []
    for p in parsed:
        try:
            results.append(await _control_one(service, p, rows.get(p.get("base")), action, signal, graceful, reason, now_local))
        except SlurmMcpError as e:
            results.append(ControlOutcome(id=p["id"], accepted=False, outcome="error", message=str(e)))
    ok = sum(1 for r in results if r.accepted)
    summary = f"{action}: {ok}/{len(results)} accepted" + "".join(
        f"; {r.id} {r.outcome}" + (f" ({r.message})" if r.message and not r.accepted else "") for r in results[:10])
    handles = [p["base"] for p in parsed if p["kind"] == "job"]
    nxt = f"wait_for_events(job_ids={handles!r}, timeout_s=300) to see the resulting state" if handles else None
    return ControlResult(summary=summary, unread_events=await service.unread(), action=action, results=results, next=nxt)


async def _control_one(service: Any, p: dict[str, Any], row: dict[str, Any] | None, action: str, signal: str | None,
                       graceful: bool, reason: str | None, now_local: float) -> ControlOutcome:
    store = service.store
    ident = p["id"]
    if p["kind"] == "adopt":
        raise err("E_INVALID_SPEC", f"adopt {ident} with job_status first, then control it by handle")
    if p["kind"] == "transfer":
        if action != "cancel":
            raise err("E_STATE", f"{action} is not valid for a transfer", state="transfer")
        comp = service.components.get("transfers") or service.components.get("transfer")
        cancel = getattr(comp, "cancel", None)
        if not callable(cancel):
            return ControlOutcome(id=ident, accepted=False, outcome="unavailable", message="transfers not available yet")
        res = cancel(p["transfer_id"])
        if hasattr(res, "__await__"):
            res = await res
        return ControlOutcome(id=ident, accepted=bool(res), outcome="cancelled" if res else "not_running")
    if row is None:
        raise err("E_UNKNOWN_ID", f"no job {p['base']!r}")
    cluster = row["cluster"]
    client = service.client(cluster)
    if p.get("cmd"):
        cmd = await store.read(lambda c: store.get_alloc_cmd(c, p["cmd"]))
        if cmd is None:
            raise err("E_UNKNOWN_ID", f"no command {p['cmd']!r}")
        if action != "cancel":
            raise err("E_STATE", f"{action} is not valid for a command", state=cmd["state"])
        if cmd["state"] not in ("queued", "running"):
            return ControlOutcome(id=ident, accepted=False, outcome="already_done", message=f"command is {cmd['state']}")
        await service._guard(cluster, client.write_file(cmd["kill_path"], f"{int(now_local)}\n", "overwrite", mkdirs=True))
        await store.write(lambda c: store.update_alloc_cmd(c, p["cmd"], kill_requested_local=now_local))
        return ControlOutcome(id=ident, accepted=True, outcome="kill_requested", message=f"wrote {cmd['kill_path']}")
    if action == "cancel":
        if row.get("kind") == "alloc" and p.get("task") is None and row.get("slurm_id") \
                and str(row["state"]) not in PRE_SLURM_STATES:
            try:
                await service._guard(cluster, client.write_file(f"{row['ctrl_dir']}/release", f"{int(now_local)}\n",
                                                                "overwrite", mkdirs=True))
            except SlurmMcpError:
                pass
            return await _cancel_one(service, p, row, False, reason, now_local)
        return await _cancel_one(service, p, row, graceful, reason, now_local)
    sid = _slurm_target_id(row, p.get("task"))
    state = str(row["state"])
    if not sid or state in PRE_SLURM_STATES:
        return ControlOutcome(id=ident, accepted=False, outcome="not_in_slurm", message=f"job is {state} without a SLURM id")
    if state in {str(s) for s in TERMINAL}:
        return ControlOutcome(id=ident, accepted=False, outcome="already_terminal", message=f"job is {state}")
    if action == "hold":
        res = await service._guard(cluster, client.hold([sid]))
        if res.get("ok"):
            base = await store.read(lambda c: store.get_job_base(c, row["handle"]))
            mode = (base or {}).get("placement_mode") or "auto"
            hold_reason = f"{mode}:{reason or 'user hold'}"

            def fn(conn: Any) -> None:
                store.update_job(conn, row["handle"], placement_mode="explicit", hold_reason=hold_reason, reason="JobHeldUser")
                service.events.append(conn, "held", row["handle"], cluster, str(row.get("slurm_id")),
                                      f"{row['handle']} held ({reason or 'user'})", {"reason": reason or "user"},
                                      state=row["state"])
            await store.write(fn)
    elif action == "release":
        res = await service._guard(cluster, client.release([sid]))
        if res.get("ok"):
            base = await store.read(lambda c: store.get_job_base(c, row["handle"]))
            prior = str((base or {}).get("hold_reason") or "")
            fields: dict[str, Any] = {"hold_reason": None, "reason": None}
            if prior.startswith("auto:") or prior.startswith("plan:"):
                fields["placement_mode"] = prior.split(":", 1)[0]
            await store.write(lambda c: store.update_job(c, row["handle"], **fields))
    elif action == "requeue":
        res = await service._guard(cluster, client.requeue([sid]))
    else:
        res = await service._guard(cluster, client.cancel([sid], signal=signal, full=True))
    if not res.get("ok"):
        msg = (res.get("stderr") or "").strip().splitlines()
        return ControlOutcome(id=ident, accepted=False, outcome=f"{action}_failed", message=msg[0][:200] if msg else f"rc {res.get('rc')}")
    return ControlOutcome(id=ident, accepted=True, outcome=f"{action}ed" if action != "signal" else "signalled",
                          message=f"{action} {sid}" + (f" ({signal})" if action == "signal" else ""))


# --- registration ------------------------------------------------------------------------------------------------------

def register(mcp: MCPServer, service: Any) -> None:
    @mcp.tool(name="submit_job", description=SUBMIT_DESC, annotations=_mcp.mutating())
    async def submit_job_tool(job: Optional[JobSpec] = None, plan_id: Optional[str] = None,
                              placement: str | list[str] = "auto", target: Optional[str] = None, hold: bool = False,
                              wait_s: int = 90, ctx: Optional[Context] = None) -> SubmitResult:
        return await run_tool(submit_job(service, job, plan_id, placement, target, hold, wait_s, ctx))

    @mcp.tool(name="list_jobs", description=LIST_DESC, annotations=_mcp.read_only())
    async def list_jobs_tool(cluster: Optional[str] = None,
                             state: Literal["active", "pending", "running", "terminal", "all"] = "active",
                             since_h: Optional[float] = None, name: Optional[str] = None,
                             kind: Literal["job", "alloc", "all"] = "all", include_untracked: bool = False,
                             limit: int = 50) -> JobListResult:
        return await run_tool(list_jobs(service, cluster, state, since_h, name, kind, include_untracked, limit))

    @mcp.tool(name="job_status", description=STATUS_DESC, annotations=_mcp.read_only())
    async def job_status_tool(ids: list[str], detail: Literal["normal", "full"] = "normal") -> JobStatusResult:
        return await run_tool(job_status(service, ids, detail))

    @mcp.tool(name="job_logs", description=LOGS_DESC, annotations=_mcp.read_only(), meta=dict(BIG_RESULT_META))
    async def job_logs_tool(id: str, stream: Literal["out", "err", "both"] = "out", tail_lines: int = 80,
                            grep: Optional[str] = None, offset: Optional[int] = None, max_chars: int = 12000) -> LogResult:
        return await run_tool(job_logs(service, id, stream, tail_lines, grep, offset, max_chars))

    @mcp.tool(name="job_control", description=CONTROL_DESC, annotations=_mcp.destructive())
    async def job_control_tool(ids: list[str], action: Literal["cancel", "hold", "release", "requeue", "signal"],
                               signal: Optional[str] = None, graceful: bool = True, reason: Optional[str] = None,
                               confirm: bool = False) -> ControlResult:
        return await run_tool(job_control(service, ids, action, signal, graceful, reason, confirm))


__all__ = ["register", "submit_job", "list_jobs", "job_status", "job_logs", "job_control", "parse_id", "CONFIRM_LIMIT",
           "SUBMIT_DESC", "LIST_DESC", "STATUS_DESC", "LOGS_DESC", "CONTROL_DESC"]
