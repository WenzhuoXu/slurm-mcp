"""Submit pipeline: ``Submitter`` (the component) and the per-handle ``SubmitTask`` (design section 5.1 steps 4-8,
section 4 ``submit_job``, section 5.3 injection, section 6.3 render/submit, section 7.2 ``submit.sh``, section 9.2
"Submit ambiguous"; changelog items 3, 7, 8, 9, 11).

Steps 1-3 of section 5.1 run inside :meth:`Submitter.submit` (validation, directive stripping, semantic dependency
resolution in the ledger, handle/token/paths committed in one ``BEGIN IMMEDIATE``); the tool returns the handle as
soon as they are committed. Steps 4-8 (helper deploy, inputs, placement, render, ``submit.sh``, bookkeeping) run in an
``asyncio`` task registered in :attr:`Submitter.tasks` (also reachable as ``service.submits``) that the tool merely
awaits for ``wait_s`` through :meth:`Submitter.await_result`. A client abort never cancels the task; a definite
submit failure is a ``submit_failed`` event plus a ``SlurmMcpError`` for a caller still waiting; an ambiguous reply
leaves the attempt ``UNCONFIRMED`` and the job ``SUBMITTING`` for the Monitor's recovery (section 5.2 step 9).

Inputs staging belongs to phase 3: a spec with ``inputs`` fails with ``E_UPLOAD`` unless a ``transfer`` component
offering ``stage_inputs`` is attached. Placement uses :func:`slurm_mcp.placer.rank` (the phase-2 minimal placer) and
takes the first feasible option; ``plan_id`` reuses the ``plans`` row when it exists and is younger than 15 min.
No cluster name appears in this module.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import posixpath
import secrets
import shlex
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from . import placer
from . import render
from .clock import format_duration
from .config import ClusterProfile, control_root as profile_control_root
from .errors import SlurmMcpError, err
from .helpers import bundle_sha8
from .models import JobSpec, PlacementPolicy, PlanOption, SubmitResult, Target, Uploads, parse_input
from .slurm.discovery import ensure_helpers
from .slurm.states import AttemptState, JobState, TERMINAL
from .store import LeaseLost, Store, loads_json
from .textio import normalize_text, read_local_text

MAX_TEST_ONLY = 4   # section 8: at most four --test-only passes per placement decision

log = logging.getLogger("slurm_mcp.submitter")

PROGRESS_INTERVAL_S = 5.0
INFEASIBLE_WINDOW_S = 30 * 60.0                # section 5.1 step 7: target_stats.infeasible_until_local
FALLBACK_CODES: tuple[str, ...] = ("E_QOS", "E_PARTITION", "E_NODE_CONFIG", "E_SUBMIT_LIMIT")
DEFAULT_DEPENDENCY_TYPE = "afterok"
PENDING_LEDGER_STATES: tuple[JobState, ...] = (JobState.SUBMITTING, JobState.SUBMITTED)
ProgressCb = Callable[[str], Any]


# --- small pure helpers -------------------------------------------------------------------------------------------

def array_size(expr: str | None) -> int | None:
    """Number of elements of an sbatch ``--array`` expression (``0-99`` -> 100, ``0-9:2`` -> 5, ``1,3,5-9`` -> 7)."""
    if not expr:
        return None
    total = 0
    for piece in expr.split(","):
        piece = piece.strip()
        if not piece:
            continue
        rng, _, step = piece.partition(":")
        st = int(step) if step.isdigit() else 1
        if "-" in rng:
            a, b = rng.split("-", 1)
            if not (a.isdigit() and b.isdigit()):
                return None
            lo, hi = int(a), int(b)
            if hi >= lo:
                total += (hi - lo) // max(1, st) + 1
        elif rng.isdigit():
            total += 1
        else:
            return None
    return total or None


def parse_dependency_entry(entry: str) -> tuple[str, str]:
    """``"j12"`` -> ``("afterok", "j12")``; ``"afternotok:j12"`` -> ``("afternotok", "j12")``; ``"singleton"`` ->
    ``("singleton", "singleton")`` (design section 3.2 ``depends_on`` grammar)."""
    text = entry.strip()
    if text == "singleton":
        return "singleton", "singleton"
    typ, sep, handle = text.partition(":")
    if not sep:
        return DEFAULT_DEPENDENCY_TYPE, typ
    return (typ or DEFAULT_DEPENDENCY_TYPE), handle


def target_from_json(value: Any) -> Target | None:
    """``attempts.target_json`` -> ``Target`` (None for the ``{}`` placeholder of an unplaced auto job)."""
    data = value
    if isinstance(value, str):
        try:
            data = json.loads(value) if value else None
        except ValueError:
            return None
    if not isinstance(data, Mapping) or not data.get("partitions"):
        return None
    try:
        return Target.model_validate(dict(data))
    except Exception:
        return None


def target_key(value: Any) -> str | None:
    tgt = target_from_json(value)
    return tgt.key if tgt is not None else None


class DependencyResolution:
    """Outcome of section 5.1 step 2 for one spec."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []      # [{handle, type, resolved_slurm_id}]
        self.slurm_parts: list[str] = []             # "afterok:615408"
        self.pinned_cluster: str | None = None
        self.waiting_for: list[str] = []             # handles without a SLURM id yet
        self.singleton = False
        self.resolved_text: list[str] = []           # SubmitResult.dependencies_resolved

    @property
    def dependency_arg(self) -> str | None:
        parts = list(self.slurm_parts)
        if self.singleton:
            parts.append("singleton")
        return ",".join(parts) if parts else None


def evaluate_terminal_dependency(dep_type: str, dep_handle: str, dep_state: str, array_states: Sequence[str] = (),
                                 ) -> bool:
    """Section 5.1 step 2 rules for a terminal dependency: True = satisfied (omit); raises ``E_DEPENDENCY`` when the
    type can never be satisfied."""
    if dep_type == "afterok":
        if dep_state == str(JobState.COMPLETED):
            return True
        raise err("E_DEPENDENCY", f"{dep_handle} ended {dep_state}; afterok can never be satisfied",
                  fix=f"resubmit {dep_handle} or use afterany/afternotok")
    if dep_type == "afternotok":
        if dep_state == str(JobState.COMPLETED):
            raise err("E_DEPENDENCY", f"{dep_handle} ended COMPLETED; afternotok can never be satisfied",
                      fix="use afterany or drop the dependency")
        return True
    if dep_type == "aftercorr":
        states = list(array_states) or [dep_state]
        if all(s == str(JobState.COMPLETED) for s in states):
            return True
        raise err("E_DEPENDENCY", f"{dep_handle} ended with elements not COMPLETED; aftercorr can never be satisfied",
                  fix=f"resubmit {dep_handle} or use afterany")
    return True     # afterany / after on anything that ended


def derive_paths(profile: ClusterProfile, caps: Mapping[str, Any] | None, spec: JobSpec, handle: str, attempt_no: int,
                 *, script_workdir: str | None = None) -> dict[str, str]:
    """Section 5.1 step 3 paths for one cluster: ``ctrl_root``, ``ctrl_dir``, ``workdir``, ``stdout_pattern``,
    ``stderr_pattern`` (absolute; ``$HOME`` expanded with the discovered home when known)."""
    home = (caps or {}).get("home") or None

    def expand(p: str) -> str:
        if home and ("$HOME" in p or "${HOME}" in p):
            return p.replace("${HOME}", home).replace("$HOME", home)
        return p

    croot = expand(profile_control_root(profile).rstrip("/"))
    ctrl_root = f"{croot}/jobs/{handle}"
    ctrl_dir = f"{ctrl_root}/a{attempt_no}"
    base = spec.workdir or script_workdir
    if base:
        workdir = expand(base.rstrip("/") or "/")
        if not workdir.startswith("/"):
            root = profile.remote_root or home or "$HOME"
            workdir = expand(f"{root.rstrip('/')}/{workdir}")
    else:
        root = profile.remote_root or home or "$HOME"
        workdir = expand(f"{root.rstrip('/')}/{spec.name}")
    out_p, err_p = render.resolve_output_patterns(spec, workdir, ctrl_root)
    return {"ctrl_root": ctrl_root, "ctrl_dir": ctrl_dir, "workdir": workdir, "stdout_pattern": out_p,
            "stderr_pattern": err_p}


def _dirs_to_create(paths: Mapping[str, str]) -> list[str]:
    out: list[str] = []
    for p in (paths["workdir"], paths["ctrl_dir"], posixpath.dirname(paths["stdout_pattern"]),
              posixpath.dirname(paths["stderr_pattern"])):
        if p and "%" not in p and p not in out:
            out.append(p)
    return out


def _uniq(items: Sequence[str]) -> list[str]:
    out: list[str] = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class _Prepared:
    """Everything step 1 produced for a spec (kept per handle for the task)."""

    def __init__(self, spec: JobSpec, script_text: str | None, stripped: list[str], warnings: list[str],
                 script_workdir: str | None) -> None:
        self.spec = spec
        self.script_text = script_text
        self.stripped_directives = stripped
        self.warnings = warnings
        self.script_workdir = script_workdir


class Submitter:
    """The ``submitter`` component (section 2, section 5.1): owns one ``SubmitTask`` per handle.

    ``tasks`` maps handle -> ``asyncio.Task`` (aliased as ``service.submits``); ``results`` keeps the final
    :class:`SubmitResult` (or the ``SlurmMcpError``) of finished tasks; ``phase`` the progress text of running ones.
    """

    def __init__(self, service: Any) -> None:
        self.service = service
        self.tasks: dict[str, asyncio.Task[Any]] = {}
        self.results: dict[str, SubmitResult | SlurmMcpError] = {}
        self.phase: dict[str, str] = {}
        self.notes: dict[str, list[str]] = {}
        self._prepared: dict[str, _Prepared] = {}
        self._options: dict[str, list[PlanOption]] = {}
        self._placement: dict[str, dict[str, Any]] = {}
        self._done: dict[str, asyncio.Event] = {}
        if getattr(service, "submits", None) is None:
            try:
                service.submits = self.tasks
            except Exception:  # pragma: no cover - a frozen service object
                pass

    # -- lifecycle -----------------------------------------------------------------------------------------------

    async def start(self) -> None:
        """Nothing to do at start: stale ``INTENT`` rows are the Monitor's sweep (section 5.2 step 10)."""

    async def stop(self) -> None:
        """Cancel running tasks; their attempts stay ``INTENT``/``UNCONFIRMED`` for recovery (section 5.8)."""
        for handle, task in list(self.tasks.items()):
            if not task.done():
                task.cancel()
        for task in list(self.tasks.values()):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self.tasks.clear()

    @property
    def store(self) -> Store:
        return self.service.store

    def running(self) -> list[str]:
        return [h for h, t in self.tasks.items() if not t.done()]

    def is_active(self, handle: str) -> bool:
        """True while this process still runs the handle's ``SubmitTask``.

        The Monitor's ``INTENT`` sweep (design section 5.2 step 10) only fails an attempt whose handle has no
        live task here, so it must be able to ask; without this the sweep would kill a running submit.
        """
        task = self.tasks.get(handle)
        return task is not None and not task.done()

    def active_handles(self) -> list[str]:
        """The handles with a live ``SubmitTask`` (the Monitor's fallback probe)."""
        return self.running()

    # -- writes ----------------------------------------------------------------------------------------------------

    async def _write(self, fn: Callable[[Any], Any]) -> Any:
        """Task writes are fenced with the process's monitor token when it holds one (section 5.8)."""
        token = getattr(self.service, "lease_token", None)
        if token is not None:
            return await self.store.write_fenced(token, fn)
        return await self.store.write(fn)

    # -- step 1: validation and script parsing -----------------------------------------------------------------------

    async def prepare_spec(self, spec_in: JobSpec | Mapping[str, Any], *, cluster_hint: str | None = None,
                           ) -> _Prepared:
        """Section 5.1 step 1: validate, fetch ``script_path`` text, strip the managed ``#SBATCH`` directives into
        spec fields (:func:`render.parse_sbatch` + :func:`render.merge_spec`). A raw mapping is merged before
        validation so ``resources.time`` may come from the script; a ``JobSpec`` keeps its explicit fields."""
        raw: Mapping[str, Any] | None = None if isinstance(spec_in, JobSpec) else dict(spec_in)
        if raw is not None:
            sources = [k for k in ("command", "script", "script_path") if raw.get(k) is not None]
            if len(sources) != 1:
                raise err("E_INVALID_SPEC", f"exactly one of command/script/script_path is required (got {sources or 'none'})",
                          fix="set command (bash body), script (full sbatch text) or script_path")
            script_text, warnings = self._script_source(raw.get("script"), raw.get("script_path"),
                                                        cluster=raw.get("cluster") or cluster_hint)
            if script_text is None and raw.get("script_path") is not None:
                script_text, w = await self._fetch_remote(raw["script_path"], raw.get("cluster") or cluster_hint)
                warnings += w
            if script_text is None:
                spec = JobSpec.parse(raw)
                return _Prepared(spec, None, [], list(spec.warnings), None)
            parsed = render.parse_sbatch(script_text, cluster=raw.get("cluster") or cluster_hint)
            merged, mw = render.merge_spec(raw, parsed)
            return _Prepared(merged, script_text, list(parsed.stripped_directives), _uniq(warnings + mw),
                             parsed.spec_fields.get("workdir"))
        spec = spec_in
        warnings = list(spec.warnings)
        if spec.source_kind == "command":
            return _Prepared(spec, None, [], warnings, None)
        script_text, w = self._script_source(spec.script, spec.script_path, cluster=spec.cluster or cluster_hint)
        warnings += w
        if script_text is None:
            script_text, w = await self._fetch_remote(spec.script_path or "", spec.cluster or cluster_hint)
            warnings += w
        parsed = render.parse_sbatch(script_text, cluster=spec.cluster or cluster_hint)
        merged, mw = render.merge_spec(spec, parsed)
        return _Prepared(merged, script_text, list(parsed.stripped_directives), _uniq(warnings + mw),
                         parsed.spec_fields.get("workdir"))

    @staticmethod
    def _script_source(script: str | None, script_path: str | None, *, cluster: str | None) -> tuple[str | None, list[str]]:
        """Inline script text or a ``local:`` file (utf-8-sig, CRLF -> LF); None for a remote path (fetched async)."""
        if script is not None:
            text, w = normalize_text(script)
            return text, w
        if script_path is not None and script_path.startswith("local:"):
            return read_local_text(script_path[len("local:"):])
        return None, []

    async def _fetch_remote(self, path: str, cluster: str | None) -> tuple[str, list[str]]:
        """``cat`` an absolute remote script (``idempotent=True``) and normalise it (section 5.1 step 1)."""
        if not path.startswith("/"):
            raise err("E_INVALID_SPEC", f"script_path {path!r} must be 'local:<path>' or an absolute remote path")
        cluster = cluster or (self.service.registry.names() or [None])[0]
        if cluster is None:
            raise err("E_INVALID_SPEC", "no cluster configured to read the remote script from")
        from .slurm.commands import path_quote
        client = self.service.client(cluster)
        res = await self.service._guard(cluster, client.run(f"cat {path_quote(path)}", idempotent=True))
        if not res.ok:
            raise err("E_INVALID_SPEC", f"cannot read remote script {path} on {cluster}: {res.stderr.strip()[:200]}",
                      fix="check script_path with remote_ls")
        return normalize_text(res.stdout)

    # -- step 2: dependencies --------------------------------------------------------------------------------------

    async def resolve_dependencies(self, spec: JobSpec, explicit_cluster: str | None = None) -> DependencyResolution:
        """Section 5.1 step 2: evaluate terminal dependencies now, pin the cluster of live ones and emit
        ``--dependency`` only on ids in controller memory."""
        res = DependencyResolution()
        if not spec.depends_on:
            return res
        entries = [parse_dependency_entry(e) for e in spec.depends_on]
        handles = [h for t, h in entries if t != "singleton"]
        rows = await self.store.read(lambda c: {h: self.store.get_job(c, h) for h in handles})
        # freshness: a live dependency observed longer than MinJobAge/2 ago gets one tick first
        for handle in handles:
            row = rows.get(handle)
            if row is None:
                raise err("E_UNKNOWN_ID", f"dependency {handle} is not a tracked job")
            if row["state"] in {str(s) for s in TERMINAL}:
                continue
            caps = self.service.caps_cached(row["cluster"]) or {}
            min_age = int(caps.get("min_job_age_s") or 300)
            clock = self.service.clock(row["cluster"])
            seen = row.get("last_seen_ts")
            if seen is None or clock.remote_now() - int(seen) > min_age / 2:
                if await self.service.tick_if_stale(row["cluster"], 0):
                    rows[handle] = await self.store.read(lambda c, h=handle: self.store.get_job(c, h))
        for dep_type, handle in entries:
            if dep_type == "singleton":
                res.singleton = True
                res.resolved_text.append("singleton")
                continue
            row = rows[handle]
            state = str(row["state"])
            if state in {str(s) for s in TERMINAL}:
                array_states = await self.store.read(
                    lambda c, h=handle: [r["state"] for r in self.store.array_tasks_for(c, h)]) \
                    if dep_type == "aftercorr" else []
                evaluate_terminal_dependency(dep_type, handle, state, array_states)
                res.resolved_text.append(f"{dep_type}:{handle} satisfied ({state})")
                continue
            if dep_type == "after" and (row.get("start_ts") or state in (str(JobState.RUNNING), str(JobState.COMPLETING))):
                res.resolved_text.append(f"{dep_type}:{handle} satisfied (started)")
                continue
            cluster = row["cluster"]
            if explicit_cluster is not None and explicit_cluster != cluster:
                raise err("E_DEP_CROSS_CLUSTER", f"{handle} runs on {cluster} but the target is on {explicit_cluster}",
                          cluster=cluster)
            if res.pinned_cluster is not None and res.pinned_cluster != cluster:
                raise err("E_DEP_CROSS_CLUSTER", f"dependencies live on different clusters ({res.pinned_cluster}, {cluster})",
                          cluster=res.pinned_cluster)
            res.pinned_cluster = cluster
            sid = row.get("slurm_id")
            if sid and row.get("attempt_state") == str(AttemptState.ACTIVE):
                res.slurm_parts.append(f"{dep_type}:{sid}")
                res.entries.append({"handle": handle, "type": dep_type, "resolved_slurm_id": str(sid)})
                res.resolved_text.append(f"{dep_type}:{handle}={sid}")
            else:
                res.waiting_for.append(handle)
                res.entries.append({"handle": handle, "type": dep_type, "resolved_slurm_id": None})
                res.resolved_text.append(f"{dep_type}:{handle} (waiting for its id)")
        return res

    # -- caps / policy helpers ------------------------------------------------------------------------------------

    async def _pending_counts(self, cluster: str, exclude: str | None = None) -> tuple[int, dict[str, int], dict[str, int]]:
        """``(cluster_total, per_partition, per_target_key)`` of my jobs pending in SLURM or being submitted."""
        def fn(conn: Any) -> tuple[int, dict[str, int], dict[str, int]]:
            rows = self.store.list_jobs(conn, cluster=cluster, states=list(PENDING_LEDGER_STATES))
            total = 0
            parts: dict[str, int] = {}
            keys: dict[str, int] = {}
            for r in rows:
                if r["handle"] == exclude:
                    continue
                total += 1
                tgt = target_from_json(r.get("target_json"))
                if tgt is not None:
                    keys[tgt.key] = keys.get(tgt.key, 0) + 1
                    for p in tgt.partitions:
                        parts[p] = parts.get(p, 0) + 1
            return total, parts, keys
        return await self.store.read(fn)

    async def cap_check(self, cluster: str, target: Target, policy: PlacementPolicy, caps: Mapping[str, Any] | None,
                        exclude: str | None = None) -> str | None:
        """Section 8 "hold locally": the ``why`` text when the target has no free pending slot, else None."""
        total, parts, keys = await self._pending_counts(cluster, exclude)
        if policy.max_pending_per_target is not None:
            n = keys.get(target.key, 0)
            if n >= policy.max_pending_per_target:
                return f"cap {n}/{policy.max_pending_per_target} pending on {target.key}"
            return None
        caps = caps or {}
        cap_part = caps.get("pending_cap_part")
        cap_user = caps.get("pending_cap")
        if isinstance(cap_part, int):
            for p in target.partitions:
                n = parts.get(p, 0)
                if n >= cap_part:
                    return f"cap {n}/{cap_part} pending on {cluster}:{p}"
        if isinstance(cap_user, int) and total >= cap_user:
            return f"cap {total}/{cap_user} pending on {cluster}"
        return None

    async def _my_counts(self, cluster: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """``(running_by_partition, pending_by_partition)`` as handle lists for the placer."""
        def fn(conn: Any) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
            run: dict[str, list[str]] = {}
            pend: dict[str, list[str]] = {}
            for r in self.store.list_jobs(conn, cluster=cluster, states=[JobState.RUNNING, JobState.COMPLETING,
                                                                          *PENDING_LEDGER_STATES]):
                tgt = target_from_json(r.get("target_json"))
                bucket = run if r["state"] in (str(JobState.RUNNING), str(JobState.COMPLETING)) else pend
                for p in (tgt.partitions if tgt else []):
                    bucket.setdefault(p, []).append(r["handle"])
            return run, pend
        return await self.store.read(fn)

    async def _target_stats(self, cluster: str) -> dict[str, dict[str, Any]]:
        rows = await self.store.read(lambda c: self.store.select(c, "target_stats", {"cluster": cluster}))
        return {r["target_key"]: r for r in rows}

    # -- step 3: intent commit ---------------------------------------------------------------------------------------

    async def submit(self, spec: JobSpec | Mapping[str, Any] | None = None, placement: str | Sequence[str] = "auto",
                     target: str | None = None, plan_id: str | None = None, hold: bool = False, *,
                     kind: str = "job", alloc_idle_release_s: int = 0) -> tuple[str, asyncio.Task[Any]]:
        """Section 5.1 steps 1-3; returns ``(handle, task)`` once the intent rows are committed."""
        if (spec is None) == (plan_id is None):
            raise err("E_INVALID_SPEC", "exactly one of job/plan_id is required")
        plan_row: dict[str, Any] | None = None
        if plan_id is not None:
            plan_row = await self.store.read(lambda c: self.store.get_plan(c, plan_id))
            if plan_row is None or float(plan_row.get("expires_local") or 0) < time.time():
                raise err("E_PLAN_EXPIRED", f"plan {plan_id} is unknown or older than 15 min")
            spec = json.loads(plan_row["spec_json"]) if isinstance(plan_row["spec_json"], str) else plan_row["spec_json"]
            if target is None and plan_row.get("recommended"):
                target = str(plan_row["recommended"])
        assert spec is not None
        explicit: Target | None = None
        if target:
            explicit = placer.resolve_explicit(target, self.service.registry.profiles)
        elif isinstance(placement, str) and placement.strip().lower() != "auto":
            explicit = placer.resolve_explicit(placement, self.service.registry.profiles)
        prepared = await self.prepare_spec(spec, cluster_hint=explicit.cluster if explicit else None)
        js = prepared.spec
        if explicit is not None and js.cluster and js.cluster != explicit.cluster:
            raise err("E_INVALID_SPEC", f"job.cluster={js.cluster} disagrees with target {explicit.key}")
        if explicit is None and js.partition and js.cluster:
            explicit = placer.resolve_explicit(Target(cluster=js.cluster, partitions=[js.partition], qos=js.qos,
                                                      account=js.account), self.service.registry.profiles)
        deps = await self.resolve_dependencies(js, explicit.cluster if explicit else js.cluster)
        names = self.service.registry.names()
        if explicit is not None:
            cluster = explicit.cluster
        elif deps.pinned_cluster:
            cluster = deps.pinned_cluster
        elif js.cluster:
            cluster = js.cluster
        elif names:
            cluster = names[0]
        else:
            raise err("E_NO_TARGET", "no cluster configured", fix="run 'slurm-mcp cluster add' first")
        self.service.profile(cluster)
        if explicit is not None and self.service.registry.profiles.get(cluster) is not None \
                and (self.service.caps_cached(cluster) or {}).get("partitions"):
            placer.resolve_explicit(explicit, self.service.registry.profiles,
                                    {cluster: self.service.caps_cached(cluster)})
        profile = self.service.profile(cluster)
        caps = self.service.caps_cached(cluster)
        policy = await self.service.placement_policy()
        mode = "plan" if plan_row is not None else ("explicit" if explicit is not None else "auto")
        state = JobState.UPLOADING if js.inputs else JobState.SUBMITTING
        queued_why: str | None = None
        if explicit is not None:
            queued_why = await self.cap_check(cluster, explicit, policy, caps)
            if queued_why:
                state = JobState.QUEUED
        token = "t-" + secrets.token_hex(6)
        spec_json = js.model_dump()

        def fn(conn: Any) -> str:
            handle = self.store.next_handle(conn, kind)
            paths = derive_paths(profile, caps, js, handle, 1, script_workdir=prepared.script_workdir)
            self.store.insert_job(conn, handle=handle, kind=kind, name=js.name, state=state, spec_json=spec_json,
                                  placement_mode=mode, attempt_no=1, array_size=array_size(js.array),
                                  depends_on_json=deps.entries if deps.entries else None,
                                  reason=queued_why)
            self.store.insert_attempt(conn, handle=handle, attempt_no=1, cluster=cluster, token=token,
                                      ctrl_root=paths["ctrl_root"], ctrl_dir=paths["ctrl_dir"], workdir=paths["workdir"],
                                      stdout_pattern=paths["stdout_pattern"], stderr_pattern=paths["stderr_pattern"],
                                      target_json=explicit.model_dump() if explicit is not None else {},
                                      state=AttemptState.INTENT, cause="initial")
            if queued_why:
                self.service.events.append(conn, "queued", handle, cluster, None,
                                           f"{handle} queued locally: {queued_why}",
                                           {"target": explicit.key if explicit else None, "why": queued_why},
                                           state=JobState.QUEUED)
            return handle
        handle = await self.store.write(fn)
        self._prepared[handle] = prepared
        self._placement[handle] = {"explicit": explicit, "placement": placement, "plan": plan_row, "hold": hold,
                                   "deps": deps, "kind": kind, "fallen_back": False,
                                   "idle_release_s": int(alloc_idle_release_s or 0)}
        self.notes[handle] = []
        self.phase[handle] = "queued" if queued_why else ("uploading" if js.inputs else "placing")
        self._done[handle] = asyncio.Event()
        if queued_why:
            self.results[handle] = await self.build_result(handle)
            self._done[handle].set()
            task = asyncio.create_task(asyncio.sleep(0), name=f"submit-{handle}-queued")
        else:
            task = asyncio.create_task(self._run(handle, start_step=4), name=f"submit-{handle}")
        self.tasks[handle] = task
        return handle, task

    # -- results / waiting -----------------------------------------------------------------------------------------

    async def build_result(self, handle: str, *, error: SlurmMcpError | None = None) -> SubmitResult:
        """The :class:`SubmitResult` for a handle from the ledger (section 4 ``submit_job`` fields)."""
        row = await self.store.read(lambda c: self.store.get_job(c, handle))
        if row is None:
            raise err("E_UNKNOWN_ID", f"no job {handle!r}")
        prepared = self._prepared.get(handle)
        state = JobState(row["state"])
        tgt = target_from_json(row.get("target_json"))
        placement = self._placement.get(handle) or {}
        deps: DependencyResolution | None = placement.get("deps")
        warnings = list(prepared.warnings) if prepared else []
        warnings += [n for n in self.notes.get(handle, []) if n not in warnings]
        injected = list((placement.get("rendered") or {}).get("injected") or [])
        kind = row.get("kind") or "job"
        result = SubmitResult(
            summary="", handle=handle, kind=kind, cluster=row.get("cluster"), slurm_id=row.get("slurm_id"),
            attempt_no=int(row.get("attempt_no") or 1), target=tgt.key if tgt else None, state=state,
            est_start_ts=row.get("est_start_ts"), cost_est_su=row.get("cost_est_su"), cost_worst_su=row.get("cost_worst_su"),
            submit_line=row.get("submit_line"), workdir=row.get("workdir"), ctrl_dir=row.get("ctrl_dir"),
            stdout_path=row.get("stdout_path"), stderr_path=row.get("stderr_path"), injected=injected,
            stripped_directives=list(prepared.stripped_directives) if prepared else [],
            dependencies_resolved=list(deps.resolved_text) if deps else [], uploads=Uploads(),
            array_size=row.get("array_size"), warnings=warnings)
        ids = f"['{handle}']"
        tail = f" ({'; '.join(self.notes.get(handle, []))})" if self.notes.get(handle) else ""
        if error is not None:
            result.summary = f"{handle} submit failed: {error}"
            result.next = "fix the spec and submit again"
        elif state == JobState.SUBMITTED:
            result.summary = (f"{handle} SUBMITTED as {row.get('slurm_id')} on {result.target} (attempt "
                              f"{result.attempt_no}){tail}")
            result.next = f"wait_for_events(job_ids={ids}, timeout_s=600) or job_status({ids})"
        elif state == JobState.QUEUED:
            result.summary = f"{handle} QUEUED locally ({row.get('reason') or 'no free pending slot'}); target {result.target}"
            result.next = f"wait_for_events(kinds=['submitted','submit_failed'], job_ids={ids})"
        elif state == JobState.SUBMITTING and row.get("attempt_state") == str(AttemptState.UNCONFIRMED):
            result.summary = (f"{handle} is being confirmed: submit.sh gave no definite answer; the Monitor recovers the "
                              f"id from the queue (never resubmits)")
            result.next = f"wait_for_events(kinds=['submitted','submit_failed','queued'], job_ids={ids})"
        elif state in (JobState.SUBMITTING, JobState.UPLOADING):
            result.summary = f"{handle} {state} ({self.phase.get(handle, 'in progress')}); the task continues server-side"
            result.next = f"wait_for_events(kinds=['submitted','submit_failed','queued'], job_ids={ids})"
        elif state in TERMINAL:
            result.summary = f"{handle} {state}: {row.get('reason') or ''}".rstrip(": ")
            result.next = f"job_status({ids})"
        else:
            result.summary = f"{handle} {state} on {result.target}"
            result.next = f"job_status({ids})"
        return result

    async def await_result(self, handle: str, wait_s: float = 90, progress_cb: ProgressCb | None = None,
                           ) -> SubmitResult:
        """Await the task for up to ``wait_s`` (progress every 5 s); a still-running task yields the honest
        non-terminal result; a definite failure re-raises its ``SlurmMcpError`` (section 4 ``submit_job``)."""
        deadline = time.monotonic() + max(0.0, float(wait_s))
        done = self._done.get(handle)
        while True:
            if done is None or done.is_set():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(done.wait(), timeout=min(PROGRESS_INTERVAL_S, remaining))
            except asyncio.TimeoutError:
                if progress_cb is not None and deadline - time.monotonic() > 0:
                    await _maybe_await(progress_cb(self.phase.get(handle, "submitting")))
        res = self.results.get(handle)
        if isinstance(res, SlurmMcpError):
            raise res
        if res is not None:
            return res
        return await self.build_result(handle)


    # -- planning hooks (design section 8 Estimates, section 4 plan_job) ------------------------------------

    async def estimate_target(self, spec: JobSpec, target: Target | str) -> dict[str, Any] | None:
        """One ``sbatch --test-only`` pass for a candidate target (section 6.3 "Estimate").

        Returns ``{"est_start_ts": int}`` on success, ``{"infeasible": True, "reason": ...}`` when SLURM refuses
        the job outright (a real answer: that target cannot run it), or ``None`` when the pass could not be made
        (timeout, dropped connection, no script) so the caller keeps its depth/history estimate instead. No job
        is created and nothing is charged; the script is written to a scratch path under the control root.
        """
        tgt = target if isinstance(target, Target) else placer.resolve_explicit(target, self.service.profiles())
        cluster = tgt.cluster
        try:
            caps = await self.service.caps(cluster)
            profile = self.service.profile(cluster)
            client = self.service.client(cluster)
        except Exception as e:
            log.info("estimate: %s unavailable (%s)", cluster, e)
            return None
        workdir = spec.workdir or profile.remote_root or caps.get("home") or "."
        ctrl_root = f"{profile_control_root(profile).rstrip('/')}/plan"
        script_path = f"{ctrl_root}/plan-{secrets.token_hex(4)}.sbatch"
        sha8 = caps.get("helper_sha8") or bundle_sha8()
        try:
            text = render.render_job_sbatch(spec, "plan", 1, "t-plan", ctrl_root, workdir,
                                            profile_control_root(profile), sha8)
            await client.mkdirs([ctrl_root])
            await client.write_file(script_path, text)
            args = render.strip_for_test_only(
                render.target_args(tgt, spec, profile, caps, 1, "plan", "t-plan", workdir=workdir,
                                   ctrl_root=ctrl_root))
            out = await client.test_only(workdir, args, script_path)
        except Exception as e:
            log.info("estimate for %s failed: %s", tgt.key, e)
            return None
        finally:
            try:
                await client.run(f"rm -f {shlex.quote(script_path)}", idempotent=True)
            except Exception:
                pass
        if out.get("ok"):
            return {"est_start_ts": out.get("est_start_ts"), "nodes": out.get("nodes"),
                    "partition": out.get("partition")}
        if out.get("timed_out"):
            return None
        return {"infeasible": True, "reason": out.get("reason") or "sbatch --test-only refused the job",
                "code": out.get("code")}

    async def preview(self, spec: JobSpec, target: Target | str) -> str:
        """The rendered ``job.sbatch`` for a target, for ``plan_job.rendered_preview`` (no SSH)."""
        tgt = target if isinstance(target, Target) else placer.resolve_explicit(target, self.service.profiles())
        profile = self.service.profile(tgt.cluster)
        caps = await self.service.caps(tgt.cluster)
        workdir = spec.workdir or profile.remote_root or caps.get("home") or "."
        ctrl_root = f"{profile_control_root(profile).rstrip('/')}/jobs/<handle>"
        sha8 = caps.get("helper_sha8") or bundle_sha8()
        return render.render_job_sbatch(spec, "<handle>", 1, "<token>", ctrl_root, workdir,
                                        profile_control_root(profile), sha8)

    def cancel_task(self, handle: str) -> bool:
        """Stop a running ``SubmitTask`` (job_control cancel on a pre-SLURM job); True when one was running."""
        task = self.tasks.get(handle)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    def resume_queued(self, handle: str) -> asyncio.Task[Any]:
        """Resume a ``QUEUED`` job at section 5.1 step 6 (Monitor, section 5.2 step 10) or after a lost task."""
        old = self.tasks.get(handle)
        if old is not None and not old.done():
            return old
        self.results.pop(handle, None)
        self._done[handle] = asyncio.Event()
        self.phase[handle] = "submitting"
        task = asyncio.create_task(self._run(handle, start_step=6), name=f"submit-{handle}-resume")
        self.tasks[handle] = task
        return task

    # -- the SubmitTask (steps 4-8) -----------------------------------------------------------------------------------

    async def _run(self, handle: str, *, start_step: int = 4) -> None:
        error: SlurmMcpError | None = None
        try:
            await self._pipeline(handle, start_step)
            self.results[handle] = await self.build_result(handle)
        except asyncio.CancelledError:
            self.phase[handle] = "cancelled"
            raise
        except LeaseLost as e:
            log.warning("%s: submit task stopped, monitor lease lost (%s)", handle, e)
            self.phase[handle] = "lease lost"
        except SlurmMcpError as e:
            error = e
            self.results[handle] = e
        except Exception as e:  # unexpected: the attempt stays INTENT/UNCONFIRMED for the sweep/recovery
            log.exception("%s: submit task died: %s", handle, e)
            self.results[handle] = err("E_SSH", f"submit task for {handle} died: {type(e).__name__}: {e}")
        finally:
            ev = self._done.get(handle)
            if ev is not None:
                ev.set()
        if error is not None:
            log.info("%s: submit failed: %s", handle, error)

    async def _load(self, handle: str) -> tuple[dict[str, Any], JobSpec, _Prepared]:
        row = await self.store.read(lambda c: self.store.get_job(c, handle))
        if row is None:
            raise err("E_UNKNOWN_ID", f"no job {handle!r}")
        prepared = self._prepared.get(handle)
        if prepared is None:
            spec = JobSpec.parse(loads_json(row, "spec_json", {}))
            prepared = await self.prepare_spec(spec, cluster_hint=row.get("cluster"))
            self._prepared[handle] = prepared
        return row, prepared.spec, prepared

    async def _pipeline(self, handle: str, start_step: int) -> None:
        row, spec, prepared = await self._load(handle)
        pl = self._placement.setdefault(handle, {"explicit": target_from_json(row.get("target_json")), "placement": "auto",
                                                 "plan": None, "hold": False, "deps": None, "kind": row.get("kind") or "job",
                                                 "fallen_back": False})
        cluster = row["cluster"]
        policy = await self.service.placement_policy()
        target: Target | None = pl.get("explicit") or target_from_json(row.get("target_json"))
        if start_step <= 5:
            # step 4: caps + helpers + inputs
            self.phase[handle] = "discovering"
            caps = await self.service.caps(cluster)
            if spec.wrap or pl.get("kind") == "alloc":
                self.phase[handle] = "deploying helpers"
                await self.service.helpers_ready(cluster)
            if spec.inputs:
                await self._stage_inputs(handle, spec, cluster, row["workdir"])
            # step 5: placement
            self.phase[handle] = "placing"
            if target is None:
                target = await self._auto_place(handle, spec, pl, policy)
                if target.cluster != cluster:
                    cluster = target.cluster
                    caps = await self.service.caps(cluster)
                    if spec.wrap:
                        await self.service.helpers_ready(cluster)
                    row = await self._rederive(handle, cluster, spec, prepared, attempt_no=int(row["attempt_no"]))
            elif "$HOME" in json.dumps([row["ctrl_root"], row["workdir"]]):
                row = await self._rederive(handle, cluster, spec, prepared, attempt_no=int(row["attempt_no"]))
            tgt_json = target.model_dump()

            def set_target(conn: Any) -> None:
                att = self.store.current_attempt(conn, handle)
                self.store.update_attempt(conn, int(att["id"]), target_json=tgt_json)
            await self._write(set_target)
            deps: DependencyResolution | None = pl.get("deps")
            if deps is None or deps.waiting_for:
                deps = await self.resolve_dependencies(spec, cluster)
                pl["deps"] = deps
            hold_why = await self.cap_check(cluster, target, policy, caps, exclude=handle)
            if deps.waiting_for and not hold_why:
                hold_why = f"waiting for {', '.join(deps.waiting_for)}'s id"
            if hold_why:
                await self._hold_locally(handle, cluster, target, hold_why)
                return
        assert target is not None
        row = await self.store.read(lambda c: self.store.get_job(c, handle))
        cluster = row["cluster"]
        caps = await self.service.caps(cluster)
        deps = pl.get("deps")
        if deps is None or deps.waiting_for:
            deps = await self.resolve_dependencies(spec, cluster)
            pl["deps"] = deps
            if deps.waiting_for:
                await self._hold_locally(handle, cluster, target, f"waiting for {', '.join(deps.waiting_for)}'s id")
                return
        # steps 6-7 with the one-time fallback of step 7
        while True:
            outcome = await self._render_and_submit(handle, row, spec, prepared, target, caps, policy, deps, pl)
            if outcome is None:
                return
            target, row = outcome

    async def _stage_inputs(self, handle: str, spec: JobSpec, cluster: str, workdir: str) -> None:
        """Section 5.1 step 4 inputs: delegated to the ``transfer`` component (phase 3); ``E_UPLOAD`` without it."""
        transfers = self.service.components.get("transfer") or self.service.components.get("transfers")
        stage = getattr(transfers, "stage_inputs", None)
        if not callable(stage):
            e = err("E_UPLOAD", "transfers not available yet: job.inputs cannot be staged in this build",
                    fix="upload the inputs with upload() (when available) or drop job.inputs")
            await self._fail(handle, cluster, e, reason=f"inputs: {e.message}")
            raise e
        self.phase[handle] = "uploading"
        await self._write(lambda c: self.store.update_job(c, handle, state=JobState.UPLOADING))

        def progress(text: str) -> None:
            self.phase[handle] = text
        try:
            await _maybe_await(stage(handle, spec, cluster, workdir, progress))
        except SlurmMcpError as e:
            await self._fail(handle, cluster, err("E_UPLOAD", f"inputs: {e.message}"), reason=f"inputs: {e.message}")
            raise
        await self._write(lambda c: self.store.update_job(c, handle, state=JobState.SUBMITTING))

    async def _auto_place(self, handle: str, spec: JobSpec, pl: dict[str, Any], policy: PlacementPolicy) -> Target:
        """Section 5.1 step 5 with the minimal placer: rank the candidates, pick the first feasible option."""
        plan = pl.get("plan")
        if plan is not None:
            options = [PlanOption.model_validate(o) for o in loads_json(plan, "options_json", [])]
            self._options[handle] = options
            rec = placer.recommended(options)
            if rec:
                return placer.resolve_explicit(rec, self.service.registry.profiles)
        clusters = [spec.cluster] if spec.cluster else self.service.registry.names()
        caps_by: dict[str, dict[str, Any] | None] = {}
        snaps: dict[str, dict[str, Any] | None] = {}
        running: dict[str, dict[str, list[str]]] = {}
        pending: dict[str, dict[str, list[str]]] = {}
        stats: dict[str, dict[str, dict[str, Any]]] = {}
        errors: list[str] = []
        for name in clusters:
            try:
                caps_by[name] = await self.service.caps(name)
            except SlurmMcpError as e:
                errors.append(f"{name}: {e.code}")
                caps_by[name] = None
                continue
            try:
                snaps[name] = await self.service.snapshot(name)
            except SlurmMcpError:
                snaps[name] = None
            running[name], pending[name] = await self._my_counts(name)
            stats[name] = await self._target_stats(name)
        self.phase[handle] = "ranking targets"
        options = placer.rank(spec, caps_by, snaps, self.service.registry.profiles, policy, my_running=running,
                              my_pending=pending, target_stats=stats, placement=pl.get("placement") or "auto")
        # Price and risk-weight the options, then spend the test-only budget on the best few, exactly as
        # plan_job does -- otherwise submit_job(placement="auto") and plan_job could recommend different
        # targets for the same spec (observed 2026-09-02: the plan said the free TRACE debug partition while
        # the submitter took an SU-charging Bridges-2 one whose --mem the site then refused).
        now_by = {}
        for name in caps_by:
            try:
                now_by[name] = int(self.service.client(name).clock.remote_now())
            except Exception:
                now_by[name] = None
        options = placer.enrich_options(options, spec, caps_by, snaps, self.service.registry.profiles, policy,
                                        target_stats=stats, now_by_cluster=now_by,
                                        inputs_cluster=spec.cluster)
        options = await self._estimate_options(spec, options)
        self._options[handle] = options
        rec = placer.recommended(options)
        if rec is None:
            whys = [f"{o.target}: {o.why}" for o in options if not o.feasible] + errors
            e = err("E_NO_TARGET", "no feasible target: " + ("; ".join(whys) if whys else "no candidates"))
            await self._fail(handle, (await self.store.read(lambda c: self.store.get_job(c, handle)))["cluster"], e,
                             reason=e.message[:200])
            raise e
        return placer.resolve_explicit(rec, self.service.registry.profiles)

    async def _estimate_options(self, spec: JobSpec, options: list[PlanOption]) -> list[PlanOption]:
        """Spend the section 8 test-only budget on the best few feasible options and re-sort.

        Shared with ``plan_job`` so both paths rank identically. A pass that cannot be made leaves the option
        on its depth/history estimate; a pass that SLURM refuses marks the option infeasible with the reason,
        which is what keeps the submitter from choosing a target the site would reject anyway.
        """
        feasible = [o for o in options if o.feasible][:MAX_TEST_ONLY]
        for opt in feasible:
            try:
                est = await self.estimate_target(spec, opt.target)
            except Exception as e:            # never let a planning probe fail the submit
                log.info("estimate for %s failed: %s", opt.target, e)
                continue
            if est is None:
                continue
            if est.get("infeasible"):
                opt.feasible = False
                opt.why = str(est.get("reason") or "sbatch --test-only refused the job")
                continue
            cluster = opt.target.split(":", 1)[0]
            try:
                now_ts = int(self.service.client(cluster).clock.remote_now())
            except Exception:
                now_ts = None
            placer.apply_estimate(opt, est_start_ts=est.get("est_start_ts"), now_ts=now_ts)
        ok = [o for o in options if o.feasible]
        ok.sort(key=lambda o: (o.score_h if o.score_h is not None else 1e9, o.est_wait_h or 0.0, o.target))
        return ok + [o for o in options if not o.feasible]

    async def _rederive(self, handle: str, cluster: str, spec: JobSpec, prepared: _Prepared, *, attempt_no: int,
                        ) -> dict[str, Any]:
        """Paths for a (new) cluster on the same attempt (section 5.1 step 5 "re-derive")."""
        profile = self.service.profile(cluster)
        caps = self.service.caps_cached(cluster)
        paths = derive_paths(profile, caps, spec, handle, attempt_no, script_workdir=prepared.script_workdir)

        def fn(conn: Any) -> None:
            att = self.store.current_attempt(conn, handle)
            self.store.update_attempt(conn, int(att["id"]), cluster=cluster, ctrl_root=paths["ctrl_root"],
                                      ctrl_dir=paths["ctrl_dir"], workdir=paths["workdir"],
                                      stdout_pattern=paths["stdout_pattern"], stderr_pattern=paths["stderr_pattern"])
        await self._write(fn)
        return await self.store.read(lambda c: self.store.get_job(c, handle))

    async def _hold_locally(self, handle: str, cluster: str, target: Target, why: str) -> None:
        """Section 5.1 step 5 / section 8: ``QUEUED{target, why}`` with the target fixed."""
        def fn(conn: Any) -> None:
            self.store.update_job(conn, handle, state=JobState.QUEUED, reason=why)
            att = self.store.current_attempt(conn, handle)
            self.store.update_attempt(conn, int(att["id"]), target_json=target.model_dump())
            self.service.events.append(conn, "queued", handle, cluster, None, f"{handle} queued locally: {why}",
                                       {"target": target.key, "why": why}, state=JobState.QUEUED)
        await self._write(fn)
        self.phase[handle] = "queued"

    async def _fail(self, handle: str, cluster: str, error: SlurmMcpError, *, reason: str, stderr: str = "") -> None:
        """Attempt ``FAILED``, job ``FAILED``, event ``submit_failed`` (section 5.1 steps 4/5/7)."""
        now_ts = self.service.clock(cluster).remote_now()

        def fn(conn: Any) -> None:
            att = self.store.current_attempt(conn, handle)
            if att is not None:
                self.store.update_attempt(conn, int(att["id"]), state=AttemptState.FAILED, reason=reason[:500],
                                          end_ts=now_ts, final_state=str(JobState.FAILED))
            self.store.update_job(conn, handle, state=JobState.FAILED, reason=reason[:500], end_ts=now_ts,
                                  terminal_ts=now_ts)
            self.service.events.append(conn, "submit_failed", handle, cluster, att.get("slurm_id") if att else None,
                                       f"{handle} submit failed: {error.code}: {error.message[:160]}",
                                       {"error_code": error.code, "stderr": stderr[:2000], "hint": error.fix},
                                       state=JobState.FAILED)
        await self._write(fn)
        self.phase[handle] = "failed"

    async def _render_and_submit(self, handle: str, row: dict[str, Any], spec: JobSpec, prepared: _Prepared, target: Target,
                                 caps: Mapping[str, Any], policy: PlacementPolicy, deps: DependencyResolution,
                                 pl: dict[str, Any]) -> tuple[Target, dict[str, Any]] | None:
        """Steps 6-7 for one attempt; returns ``(next_target, new_row)`` when step 7 fell back, else None."""
        cluster = row["cluster"]
        profile = self.service.profile(cluster)
        client = self.service.client(cluster)
        attempt_no = int(row["attempt_no"])
        token = row["token"]
        ctrl_dir, ctrl_root, workdir = row["ctrl_dir"], row["ctrl_root"], row["workdir"]
        sha8 = caps.get("helper_sha8") or client.helper_bin_dir().rsplit("/", 1)[-1]
        control_root = caps.get("control_root") or client.control_root()
        notify = self.service.notify_policy_cached()
        excluded = [n for n in (row.get("excluded_nodes") or "").split(",") if n]
        rendered = render.build_target_args(target, spec, profile, caps, attempt_no, handle, token,
                                            notify_email=notify.email, excluded_nodes=excluded, hold=bool(pl.get("hold")),
                                            dependency=deps.dependency_arg, stdout_pattern=row["stdout_pattern"],
                                            stderr_pattern=row["stderr_pattern"],
                                            mode="alloc" if pl.get("kind") == "alloc" else "job")
        pl["rendered"] = {"injected": rendered.injected, "warnings": rendered.warnings}
        for w in rendered.warnings:
            if w not in self.notes[handle]:
                self.notes[handle].append(w)
        # step 6: render + mkdirs + writes
        self.phase[handle] = "rendering"
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if pl.get("kind") == "alloc":
            script_name = "alloc.sbatch"
            job_script = render.render_alloc_sbatch(spec, handle, attempt_no, token, ctrl_dir, workdir, control_root, sha8,
                                                    idle_release_s=int(pl.get("idle_release_s") or 0), rendered_at=stamp)
        else:
            script_name = "job.sbatch"
            job_script = render.render_job_sbatch(spec, handle, attempt_no, token, ctrl_dir, workdir, control_root, sha8,
                                                  rendered_at=stamp)
        files: dict[str, str] = {script_name: job_script}
        if pl.get("kind") != "alloc":
            files["user_body.sh"] = render.render_user_body(spec, prepared.script_text)
        files["env.sh"] = render.render_env_sh(spec)
        files["spec.json"] = json.dumps({"handle": handle, "attempt": attempt_no, "token": token, "cluster": cluster,
                                         "target": target.model_dump(), "spec": spec.model_dump(),
                                         "submit_line": rendered.submit_line}, indent=1, sort_keys=True, default=str) + "\n"
        paths = {"workdir": workdir, "ctrl_dir": ctrl_dir, "stdout_pattern": row["stdout_pattern"],
                 "stderr_pattern": row["stderr_pattern"]}
        await self.service._guard(cluster, client.mkdirs(_dirs_to_create(paths)))
        for name, text in files.items():
            out = await self.service._guard(cluster, client.write_file(f"{ctrl_dir}/{name}", text, "overwrite",
                                                                       mkdirs=True, executable=name.endswith(".sh")))
            for w in out.get("warnings") or []:
                if w not in self.notes[handle]:
                    self.notes[handle].append(w)
        # step 7: UNCONFIRMED before submit.sh
        self.phase[handle] = "submitting"
        invoked = time.time()

        def mark(conn: Any) -> int:
            att = self.store.current_attempt(conn, handle)
            self.store.update_attempt(conn, int(att["id"]), state=AttemptState.UNCONFIRMED, invoked_local=invoked,
                                      submit_line=rendered.submit_line, target_json=target.model_dump())
            self.store.update_job(conn, handle, state=JobState.SUBMITTING, reason=None)
            return int(att["id"])
        attempt_id = await self._write(mark)
        out = await client.submit(workdir, ctrl_dir, token, rendered.args, f"{ctrl_dir}/{script_name}",
                                  bin_dir=client.helper_bin_dir(sha8))
        status = out.get("status")
        if status == "ok":
            await self._confirm(handle, attempt_id, cluster, row, spec, target, rendered, out, pl)
            return None
        if status == "err":
            code = out.get("code") or "E_SUBMIT_FAILED"
            stderr = (out.get("stderr") or "").strip()
            first = stderr.splitlines()[0][:200] if stderr else f"submit.sh ERR {out.get('rc')}"
            fallback = self._next_option(handle, target) if (pl.get("explicit") is None and not pl.get("fallen_back")
                                                            and code.startswith(FALLBACK_CODES)) else None
            if fallback is not None:
                pl["fallen_back"] = True
                self.notes[handle].append(f"fell back from {target.key} ({code}: {first}) to {fallback.key}")
                new_row = await self._fallback_attempt(handle, cluster, row, spec, prepared, target, fallback, code, first,
                                                       attempt_id)
                return fallback, new_row
            e = err(code, f"{target.key}: {first}", cluster=cluster, max_wall=format_duration(
                (caps.get("partitions", {}).get(target.partitions[0], {}).get("limits") or {}).get("max_wall_s")) or "the MaxWall")
            await self._fail(handle, cluster, e, reason=f"{code}: {first}", stderr=stderr)
            raise e
        # ambiguous: attempt stays UNCONFIRMED, job SUBMITTING (section 5.1 step 7, section 9.2)
        why = out.get("error") or "no JOBID/ERR line"
        self.notes[handle].append(f"submit reply ambiguous ({why}); awaiting confirmation")
        await self._write(lambda c: self.store.update_attempt(c, attempt_id, reason=f"ambiguous: {why}"[:500]))
        await self._kick(cluster)
        self.phase[handle] = "being confirmed"
        return None

    def _next_option(self, handle: str, failed: Target) -> Target | None:
        """The next feasible ranked option after ``failed`` (section 5.1 step 7 one-time fallback)."""
        options = self._options.get(handle) or []
        seen = False
        for o in options:
            if o.target == failed.key:
                seen = True
                continue
            if seen and o.feasible:
                try:
                    return placer.resolve_explicit(o.target, self.service.registry.profiles)
                except SlurmMcpError:
                    continue
        for o in options:      # the failed target was not in the list (plan recommendation): first other feasible
            if o.feasible and o.target != failed.key:
                try:
                    return placer.resolve_explicit(o.target, self.service.registry.profiles)
                except SlurmMcpError:
                    continue
        return None

    async def _fallback_attempt(self, handle: str, cluster: str, row: dict[str, Any], spec: JobSpec, prepared: _Prepared,
                                failed: Target, fallback: Target, code: str, first: str, attempt_id: int) -> dict[str, Any]:
        """Old attempt ``FAILED``, ``target_stats.infeasible_until_local``, a new ``INTENT`` attempt with new paths."""
        now_local = time.time()
        now_ts = self.service.clock(cluster).remote_now()
        new_no = int(row["attempt_no"]) + 1
        new_cluster = fallback.cluster
        profile = self.service.profile(new_cluster)
        caps = self.service.caps_cached(new_cluster)
        paths = derive_paths(profile, caps, spec, handle, new_no, script_workdir=prepared.script_workdir)
        token = "t-" + secrets.token_hex(6)

        def fn(conn: Any) -> None:
            self.store.update_attempt(conn, attempt_id, state=AttemptState.FAILED, reason=f"{code}: {first}"[:500],
                                      end_ts=now_ts, final_state=str(JobState.FAILED))
            self.store.upsert_target_stats(conn, cluster, failed.key, infeasible_until_local=now_local + INFEASIBLE_WINDOW_S,
                                           infeasible_reason=code, last_error=first[:200])
            self.store.insert_attempt(conn, handle=handle, attempt_no=new_no, cluster=new_cluster, token=token,
                                      ctrl_root=paths["ctrl_root"], ctrl_dir=paths["ctrl_dir"], workdir=paths["workdir"],
                                      stdout_pattern=paths["stdout_pattern"], stderr_pattern=paths["stderr_pattern"],
                                      target_json=fallback.model_dump(), state=AttemptState.INTENT, cause="initial")
            self.store.update_job(conn, handle, attempt_no=new_no, state=JobState.SUBMITTING)
        await self._write(fn)
        if new_cluster != cluster:
            await self.service.caps(new_cluster)
            if spec.wrap:
                await self.service.helpers_ready(new_cluster)
        return await self.store.read(lambda c: self.store.get_job(c, handle))

    async def _confirm(self, handle: str, attempt_id: int, cluster: str, row: dict[str, Any], spec: JobSpec, target: Target,
                       rendered: render.RenderedArgs, out: Mapping[str, Any], pl: dict[str, Any]) -> None:
        """``JOBID`` parsed: attempt ``ACTIVE``, job ``SUBMITTED``, output paths expanded, event ``submitted``."""
        sid = str(out["job_id"])
        caps = self.service.caps_cached(cluster) or {}
        user = caps.get("user") or self.service.profile(cluster).user
        clock = self.service.clock(cluster)
        now_ts = clock.remote_now()
        stdout_path = render.expand_pattern(row["stdout_pattern"], sid, spec.name, user) if not spec.array else None
        stderr_path = render.expand_pattern(row["stderr_pattern"], sid, spec.name, user) if not spec.array else None
        warn = (out.get("stderr") or "").strip()
        if warn:
            self.notes[handle].append(f"sbatch: {warn.splitlines()[0][:200]}")
        hold = bool(pl.get("hold"))
        est = None
        for o in self._options.get(handle, []):
            if o.target == target.key:
                est = o.est_start_ts
                break

        def fn(conn: Any) -> None:
            self.store.update_attempt(conn, attempt_id, slurm_id=sid, state=AttemptState.ACTIVE, confirmed_local=time.time(),
                                      submit_ts=now_ts, stdout_path=stdout_path, stderr_path=stderr_path,
                                      submit_line=rendered.submit_line, reason=None)
            fields: dict[str, Any] = {"state": JobState.SUBMITTED, "submit_ts": now_ts, "slurm_state": "PENDING",
                                      "est_start_ts": est, "last_seen_ts": None, "reason": "JobHeldUser" if hold else None}
            if hold:
                base = self.store.get_job_base(conn, handle) or {}
                fields["hold_reason"] = f"{base.get('placement_mode', 'auto')}:submitted with hold"
                fields["placement_mode"] = "explicit"
            if pl.get("deps") is not None and pl["deps"].entries:
                fields["depends_on_json"] = pl["deps"].entries
            self.store.update_job(conn, handle, **fields)
            payload = {"target": target.key, "attempt_no": int(row["attempt_no"]), "est_start_ts": est,
                       "stdout_path": stdout_path, "stderr_path": stderr_path, "workdir": row["workdir"],
                       "ctrl_dir": row["ctrl_dir"], "injected": rendered.injected, "warnings": list(self.notes[handle]),
                       "submit_line": rendered.submit_line, "hold": hold, "array_size": array_size(spec.array),
                       "dependencies_resolved": list(pl["deps"].resolved_text) if pl.get("deps") else []}
            self.service.events.append(conn, "submitted", handle, cluster, sid,
                                       f"{handle} submitted as {sid} on {target.key}" + (" (held)" if hold else ""),
                                       payload, ts=now_ts, state=JobState.SUBMITTED)
        await self._write(fn)
        self.phase[handle] = "submitted"
        await self._kick(cluster)

    async def _kick(self, cluster: str) -> None:
        mon = self.service.components.get("monitor")
        kick = getattr(mon, "kick", None)
        if callable(kick):
            try:
                await _maybe_await(kick(cluster))
            except Exception as e:  # pragma: no cover - a Monitor bug must not fail the submit
                log.warning("monitor kick failed: %s", e)


__all__ = ["Submitter", "DependencyResolution", "array_size", "parse_dependency_entry", "target_from_json",
           "target_key", "evaluate_terminal_dependency", "derive_paths", "PROGRESS_INTERVAL_S", "INFEASIBLE_WINDOW_S",
           "FALLBACK_CODES", "DEFAULT_DEPENDENCY_TYPE"]
