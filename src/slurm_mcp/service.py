"""Service facade: every business operation used by the MCP tools AND the CLI (design section 2 "service.py",
section 4 tool contracts, section 5.8 lease bookkeeping, rule 8 "human parity").

``ClusterRegistry`` creates per-cluster ``SSHTransport``/``ClusterClock``/``SlurmClient`` lazily (no SSH at
construction, section 5.8 "never SSH in the lifespan"). ``Service`` owns the store, the event bus, the registry,
the discovery cache and the optional components (``monitor``, ``notify``, ``submitter``, ``transfers``, ...)
attached by the lifespan; the phase-2 operations here are ``clusters``, ``cluster_status``, ``run_command``,
``remote_ls``/``remote_read``/``remote_write`` and ``configure``. Every result carries ``summary``,
``unread_events`` (unacknowledged events of this server's session client) and ``next``. ``SlurmMcpError`` is
raised as is; the tools convert it to ``ToolError``.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import asyncssh

from . import placer
from .clock import ClusterClock
from .config import ClusterProfile, has_transfer_host, target_override, transfer_endpoint
from .errors import SlurmMcpError, err
from .events import EventBus
from .models import (
    Charge, ClusterRow, ClusterStatusResult, ClustersResult, ConfigResult, GresCount, ListingEntry, ListingResult,
    MyJobs, NodeCounts, NotifyPolicy, PartitionInfo, PartitionLimits, PlacementPolicy, QueueRow, QuotaRow, ReadResult,
    ReservationRow, RunCommandResult, TargetRow, TrackedCounts, WriteResult, parse_input,
)
from .slurm import commands
from .slurm import parse as P
from .slurm.client import SlurmClient, TickFailed
from .slurm.discovery import (
    bootstrap, caps_age_s, caps_fresh, caps_key, ensure_helpers, partition_accessible, save_caps,
)
from .slurm.parse import IncompleteProbe
from .slurm.states import JobState
from .store import LeaseInfo, LeaseLost, Store
from .transport import AuthFailed, CommandTimeout, ConnectionDropped, SSHTransport, Unreachable

log = logging.getLogger("slurm_mcp.service")

RUN_COMMAND_MAX_CHARS = 4000
RUN_COMMAND_MAX_TIMEOUT_S = 600
SNAPSHOT_MAX_AGE_S = 60.0
TICK_STALE_S = 20.0
LEASE_LOOP_S = 60.0
MAX_PARTITIONS = 20
MAX_QUEUE_ROWS = 50
POLICY_PLACEMENT_KEY = "policy.placement"
POLICY_NOTIFY_KEY = "policy.notify"
SNAPSHOT_KEY_PREFIX = "snapshot."
QUEUED_STATES = (JobState.QUEUED,)
PENDING_STATES = (JobState.UPLOADING, JobState.SUBMITTING, JobState.SUBMITTED)
RUNNING_STATES = (JobState.RUNNING, JobState.COMPLETING)


class ClusterRegistry:
    """Per-cluster transports, clocks and clients, created on first use (section 2.2, section 5.8)."""

    def __init__(self, profiles: Mapping[str, ClusterProfile], store: Store | None = None) -> None:
        self.profiles: dict[str, ClusterProfile] = dict(profiles)
        self.store = store
        self.caps_cache: dict[str, dict[str, Any]] = {}
        self._transports: dict[tuple[str, str], SSHTransport] = {}
        self._clocks: dict[str, ClusterClock] = {}
        self._clients: dict[str, SlurmClient] = {}

    def names(self) -> list[str]:
        return list(self.profiles)

    def profile(self, name: str) -> ClusterProfile:
        p = self.profiles.get(name)
        if p is None:
            raise err("E_INVALID_SPEC", f"unknown cluster {name!r}",
                      fix=f"known clusters: {', '.join(sorted(self.profiles)) or '(none; run slurm-mcp cluster add)'}")
        return p

    def has_transport(self, name: str, role: str = "login") -> bool:
        return (name, role) in self._transports

    def clock(self, name: str) -> ClusterClock:
        self.profile(name)
        if name not in self._clocks:
            caps = self.caps_cache.get(name) or {}
            self._clocks[name] = ClusterClock(bool(caps.get("epoch_format", True)), int(caps.get("tz_offset_s") or 0))
        return self._clocks[name]

    def transport(self, name: str, role: str = "login") -> SSHTransport:
        profile = self.profile(name)
        key = (name, role)
        if key not in self._transports:
            caps_getter = lambda n=name: (self.caps_cache.get(n) or {}).get("cmd_timeout_s")  # noqa: E731
            if role == "transfer":
                port = ((self.caps_cache.get(name) or {}).get("transfer") or {}).get("port")
                host, port = transfer_endpoint(profile, port)
                self._transports[key] = SSHTransport(profile, host=host, port=port, role="transfer",
                                                     caps_cmd_timeout_s=caps_getter)
            else:
                self._transports[key] = SSHTransport(profile, role="login", caps_cmd_timeout_s=caps_getter)
        return self._transports[key]

    def client(self, name: str) -> SlurmClient:
        profile = self.profile(name)
        if name not in self._clients:
            transfer = self.transport(name, "transfer") if has_transfer_host(profile) else None
            self._clients[name] = SlurmClient(name, self.transport(name, "login"), transfer, self.clock(name),
                                              lambda n=name: self.caps_cache.get(n))
        return self._clients[name]

    async def close(self) -> None:
        for t in list(self._transports.values()):
            try:
                await t.close()
            except Exception:  # pragma: no cover - best effort at shutdown
                pass


class Service:
    """The facade (section 2). Components are attached by the lifespan and started by ``start()``."""

    def __init__(self, store: Store, events: EventBus, registry: ClusterRegistry, session_id: str, *,
                 lease: LeaseInfo | None = None) -> None:
        self.store = store
        self.events = events
        self.registry = registry
        self.session_id = session_id
        self.components: dict[str, Any] = {}
        self.lease: LeaseInfo | None = lease
        self.lease_token: int | None = lease.token if lease is not None and lease.acquired else None
        self.lease_lost_to: int | None = None
        self.started = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._caps_locks: dict[str, asyncio.Lock] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._snapshot_locks: dict[str, asyncio.Lock] = {}
        self._placement: PlacementPolicy = PlacementPolicy()
        self._notify: NotifyPolicy = NotifyPolicy()
        self._policies_loaded = False

    # -- components / lifecycle -------------------------------------------------------------------------------

    def attach(self, name: str, component: Any) -> Any:
        self.components[name] = component
        return component

    async def start(self) -> None:
        await self.load_policies()
        for name, comp in list(self.components.items()):
            start = getattr(comp, "start", None)
            if callable(start):
                try:
                    res = start()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as e:
                    log.exception("component %s failed to start: %s", name, e)
        self._tasks.append(asyncio.create_task(self._lease_loop(), name="slurm-mcp-lease"))
        self.started = True

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
        for name, comp in reversed(list(self.components.items())):
            stop = getattr(comp, "stop", None)
            if callable(stop):
                try:
                    res = stop()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as e:
                    log.warning("component %s failed to stop: %s", name, e)
        self.started = False

    # -- lease (section 5.8) ------------------------------------------------------------------------------------

    async def acquire_lease(self, force: bool = False) -> LeaseInfo:
        info = await self.store.write(lambda c: self.store.lease_acquire(c, force=force))
        self.lease = info
        self.lease_token = info.token if info.acquired else None
        if info.acquired:
            self.lease_lost_to = None
        return info

    async def release_lease(self) -> bool:
        if self.lease_token is None:
            return False
        token = self.lease_token
        self.lease_token = None
        return await self.store.write(lambda c: self.store.lease_release(c, token))

    async def renew_lease(self) -> bool:
        if self.lease_token is None:
            return False
        ok = await self.store.write(lambda c: self.store.lease_renew(c, self.lease_token))
        if not ok:
            await self.on_lease_lost()
        return ok

    async def on_lease_lost(self) -> None:
        """A fenced write or renew said 0 rows: stop the Monitor, serve tools from the ledger (section 5.2/5.8)."""
        row = await self.store.read(lambda c: self.store.lease_get(c))
        self.lease_lost_to = int(row["owner_pid"]) if row else None
        self.lease_token = None
        mon = self.components.get("monitor")
        stop = getattr(mon, "stop", None)
        if callable(stop):
            try:
                res = stop()
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                log.warning("monitor stop after lease loss failed: %s", e)
        try:
            await self.events.emit("needs_attention", summary=f"monitor lease lost to pid {self.lease_lost_to}",
                                   payload={"why": "lease_lost", "hint": "another slurm-mcp process runs the Monitor"})
        except Exception as e:  # pragma: no cover
            log.warning("could not record lease loss: %s", e)

    async def _lease_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(LEASE_LOOP_S)
                if self.lease_token is not None:
                    await self.renew_lease()
                else:
                    info = await self.acquire_lease()
                    if info.acquired:
                        log.info("monitor lease acquired (token %s, %s)", info.token, info.reason)
                        mon = self.components.get("monitor")
                        start = getattr(mon, "start", None)
                        if callable(start):
                            res = start()
                            if asyncio.iscoroutine(res):
                                await res
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("lease loop: %s", e)

    def monitor_status(self) -> str:
        """``self`` | ``held by pid N`` | ``lost to pid N`` | ``none`` (read from the ledger, section 4 clusters)."""
        if self.lease_lost_to is not None and self.lease_token is None:
            return f"lost to pid {self.lease_lost_to}"
        try:
            return self.store.read_sync(lambda c: self.store.monitor_status(c, self.lease_token))
        except Exception:
            return "none"

    # -- accessors -----------------------------------------------------------------------------------------------

    def profile(self, cluster: str) -> ClusterProfile:
        return self.registry.profile(cluster)

    def transport(self, cluster: str, role: str = "login") -> SSHTransport:
        return self.registry.transport(cluster, role)

    def client(self, cluster: str) -> SlurmClient:
        return self.registry.client(cluster)

    def clock(self, cluster: str) -> ClusterClock:
        return self.registry.clock(cluster)

    def caps_cached(self, cluster: str) -> dict[str, Any] | None:
        """The in-memory discovery cache (loaded from ``kv.caps.<cluster>`` on first use), no SSH."""
        if cluster in self.registry.caps_cache:
            return self.registry.caps_cache[cluster]
        try:
            caps = self.store.read_sync(lambda c: self.store.kv_get(c, caps_key(cluster)))
        except Exception:
            caps = None
        if caps:
            self.registry.caps_cache[cluster] = caps
        return caps

    async def caps(self, cluster: str, refresh: bool = False) -> dict[str, Any]:
        """Discovery cache (24 h) or a fresh bootstrap (section 6.1); serialised per cluster."""
        profile = self.profile(cluster)
        cached = self.caps_cached(cluster)
        if cached and not refresh and caps_fresh(cached):
            return cached
        lock = self._caps_locks.setdefault(cluster, asyncio.Lock())
        async with lock:
            cached = self.registry.caps_cache.get(cluster)
            if cached and not refresh and caps_fresh(cached):
                return cached
            caps = await self._guard(cluster, bootstrap(self.client(cluster), profile, self.store, refresh=refresh))
            self.registry.caps_cache[cluster] = caps
            clock = self.registry.clock(cluster)
            clock.epoch_format = bool(caps.get("epoch_format", True))
            clock.tz_offset_s = int(caps.get("tz_offset_s") or 0)
            return caps

    async def helpers_ready(self, cluster: str) -> str:
        """Deploy the helper bundle when stale (first W tool, section 6.1); returns the sha8."""
        caps = await self.caps(cluster)
        return await self._guard(cluster, ensure_helpers(self.client(cluster), self.profile(cluster), caps, self.store))

    async def snapshot(self, cluster: str, max_age_s: float = SNAPSHOT_MAX_AGE_S) -> dict[str, Any]:
        """The load snapshot of section 6.2, cached ``max_age_s`` (60 s) in memory and ``kv.snapshot.<cluster>``."""
        snap = self._snapshots.get(cluster)
        if snap and time.time() - float(snap.get("fetched_local") or 0) < max_age_s:
            return snap
        lock = self._snapshot_locks.setdefault(cluster, asyncio.Lock())
        async with lock:
            snap = self._snapshots.get(cluster)
            if snap and time.time() - float(snap.get("fetched_local") or 0) < max_age_s:
                return snap
            await self.caps(cluster)
            snap = await self._guard(cluster, self.client(cluster).snapshot())
            self._snapshots[cluster] = snap
            try:
                await self.store.write(lambda c: self.store.kv_set(c, SNAPSHOT_KEY_PREFIX + cluster, snap))
            except Exception as e:  # the cache is a convenience, never a failure
                log.debug("snapshot cache write failed: %s", e)
            return snap

    async def tick_if_stale(self, cluster: str, max_age_s: float = TICK_STALE_S) -> bool:
        """Delegate to the Monitor component when attached (section 4 job_status/job_control); else a no-op."""
        mon = self.components.get("monitor")
        if mon is None:
            return False
        fn = getattr(mon, "tick_if_stale", None)
        if callable(fn):
            res = fn(cluster, max_age_s)
            return bool(await res) if asyncio.iscoroutine(res) else bool(res)
        return False

    def last_tick_local(self, cluster: str) -> float | None:
        mon = self.components.get("monitor")
        if mon is None:
            return None
        fn = getattr(mon, "last_tick_local", None)
        try:
            if callable(fn):
                return fn(cluster)
            if isinstance(fn, Mapping):
                return fn.get(cluster)
        except Exception:
            return None
        return None

    async def unread(self) -> int:
        try:
            return await self.events.unread(self.session_id)
        except Exception as e:
            log.debug("unread count failed: %s", e)
            return 0

    # -- policies (section 3.3 kv policy.*) ---------------------------------------------------------------------

    async def load_policies(self) -> None:
        def fn(conn: Any) -> tuple[Any, Any]:
            return self.store.kv_get(conn, POLICY_PLACEMENT_KEY), self.store.kv_get(conn, POLICY_NOTIFY_KEY)
        placement, notify = await self.store.read(fn)
        self._placement = parse_input(PlacementPolicy, placement or {})
        self._notify = parse_input(NotifyPolicy, notify or {})
        self._policies_loaded = True

    def profiles(self) -> dict[str, ClusterProfile]:
        """Every configured profile by name (the placer needs them all to compare clusters, section 8)."""
        return {name: self.registry.profile(name) for name in self.registry.names()}

    async def my_counts(self, clusters: Sequence[str] | None = None,
                        ) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
        """``(running, pending)`` counts of *my* jobs per cluster and partition, from the ledger.

        Used by the placer for the etiquette caps and the "hold locally" pending caps (section 8). Reads the
        ledger, never SSH: the Monitor keeps it current and a planning call must not add login-node load.
        """
        names = list(clusters or self.registry.names())
        store = self.store
        rows = await store.read(lambda c: store.list_jobs(c, kind="job"))
        running: dict[str, dict[str, int]] = {n: {} for n in names}
        pending: dict[str, dict[str, int]] = {n: {} for n in names}
        for row in rows:
            cluster = row.get("cluster")
            if cluster not in running:
                continue
            state = str(row.get("state") or "")
            bucket = running if state in ("RUNNING", "COMPLETING") else (
                pending if state in ("QUEUED", "UPLOADING", "SUBMITTING", "SUBMITTED") else None)
            if bucket is None:
                continue
            target = row.get("target_json") or ""
            partition = ""
            if isinstance(target, str) and ":" in target:
                partition = target.split(":", 2)[1].split("@")[0]
            else:
                try:
                    import json as _json
                    data = _json.loads(target) if isinstance(target, str) else dict(target or {})
                    partition = ",".join(data.get("partitions") or [])
                except Exception:
                    partition = ""
            for p in (partition.split(",") if partition else [""]):
                if p:
                    bucket[cluster][p] = bucket[cluster].get(p, 0) + 1
            bucket[cluster]["_total"] = bucket[cluster].get("_total", 0) + 1
        return running, pending

    async def placement_policy(self) -> PlacementPolicy:
        if not self._policies_loaded:
            await self.load_policies()
        return self._placement

    async def notify_policy(self) -> NotifyPolicy:
        if not self._policies_loaded:
            await self.load_policies()
        return self._notify

    def notify_policy_cached(self) -> NotifyPolicy:
        """Sync getter for the Notifier (the cache is refreshed by ``configure`` and ``start``)."""
        return self._notify

    def placement_policy_cached(self) -> PlacementPolicy:
        return self._placement

    # -- error translation ---------------------------------------------------------------------------------------

    async def _guard(self, cluster: str, aw: Awaitable[Any]) -> Any:
        """Await a transport-backed operation translating non-catalogue failures to ``E_SSH`` (section 9.2)."""
        try:
            return await aw
        except SlurmMcpError:
            raise
        except CommandTimeout as e:
            raise err("E_SSH", f"{cluster}: command timed out after {e.timeout}s: {e.command[:120]}") from e
        except ConnectionDropped as e:
            raise err("E_SSH", f"{cluster}: connection dropped ({e.reason})") from e
        except (TickFailed, IncompleteProbe) as e:
            raise err("E_SSH", f"{cluster}: incomplete probe output ({e})") from e
        except asyncssh.SFTPError as e:
            raise err("E_PERMISSION", f"{cluster}: SFTP error: {e}", path="the remote path") from e
        except (asyncssh.Error, OSError) as e:
            raise err("E_SSH", f"{cluster}: {type(e).__name__}: {e}") from e

    # -- section 4: clusters ---------------------------------------------------------------------------------------

    async def _tracked_counts(self, cluster: str) -> TrackedCounts:
        def fn(conn: Any) -> TrackedCounts:
            rows = self.store.list_jobs(conn, cluster=cluster, states=[*QUEUED_STATES, *PENDING_STATES, *RUNNING_STATES])
            tc = TrackedCounts()
            for r in rows:
                st = r.get("state")
                if st in {s.value for s in QUEUED_STATES}:
                    tc.queued += 1
                elif st in {s.value for s in PENDING_STATES}:
                    tc.pending += 1
                else:
                    tc.running += 1
            return tc
        try:
            return await self.store.read(fn)
        except Exception:
            return TrackedCounts()

    @staticmethod
    def _quota_rows(caps: Mapping[str, Any] | None) -> list[QuotaRow]:
        out: list[QuotaRow] = []
        for row in (caps or {}).get("df") or []:
            kb_free = row.get("kb_free")
            out.append(QuotaRow(path=row.get("path") or row.get("mount") or "?", used_pct=row.get("used_pct"),
                                free_gb=round(kb_free / 1024 / 1024, 1) if isinstance(kb_free, (int, float)) else None,
                                role=row.get("role") if row.get("role") in ("home", "remote_root", "control_root", "project",
                                                                            "group", "upload_root") else None))
        return out

    async def clusters(self, refresh: bool = False) -> ClustersResult:
        """Section 4 ``clusters``: one row per profile; ``refresh=True`` re-runs discovery (read-only commands)."""
        rows: list[ClusterRow] = []
        monitor = self.monitor_status()
        names = self.registry.names()

        async def refresh_one(name: str) -> tuple[bool | None, str | None]:
            try:
                await self.caps(name, refresh=True)
                return True, None
            except SlurmMcpError as e:
                return False, str(e)

        results: dict[str, tuple[bool | None, str | None]] = {}
        if refresh and names:
            done = await asyncio.gather(*(refresh_one(n) for n in names))
            results = dict(zip(names, done))
        for name in names:
            profile = self.registry.profile(name)
            caps = self.caps_cached(name)
            warnings: list[str] = []
            transport = self.registry.transport(name) if self.registry.has_transport(name) else None
            connected = bool(transport and transport.connected)
            auth_failed = bool(transport and transport.auth_failed)
            reachable: bool | None = True if connected else None
            if name in results:
                reachable, error = results[name]
                if error:
                    warnings.append(error)
                    if error.startswith("E_AUTH"):
                        auth_failed = True
            if transport is not None:
                warnings += [f"new host key accepted: {n}" for n in transport.hostkey_notices[-3:]]
            if caps is None:
                warnings.append("not discovered yet: call clusters(refresh=True) or cluster_status")
            elif not caps_fresh(caps):
                warnings.append("discovery cache older than 24 h (refreshes on next use)")
            last_tick = self.last_tick_local(name)
            rows.append(ClusterRow(
                name=name, host=profile.host, transfer_host=profile.transfer_host or profile.data_host,
                connected=connected, auth_failed=auth_failed, reachable=reachable,
                last_tick_age_s=round(time.time() - last_tick, 1) if last_tick else None,
                tracked_jobs=await self._tracked_counts(name), su_balance=(caps or {}).get("su_balance"),
                quota=self._quota_rows(caps), monitor=monitor, warnings=warnings))
        parts = []
        for r in rows:
            state = "connected" if r.connected else ("unreachable" if r.reachable is False else "idle")
            parts.append(f"{r.name} {state} ({r.tracked_jobs.running} running, {r.tracked_jobs.pending} pending)")
        summary = f"{len(rows)} cluster(s): " + "; ".join(parts) if rows else \
            "no clusters configured; add one with 'slurm-mcp cluster add NAME --host H --user U'"
        summary += f"; monitor {monitor}"
        return ClustersResult(summary=summary, unread_events=await self.unread(), clusters=rows, session_id=self.session_id,
                              next="cluster_status('<name>') for partitions and queue depth" if rows else None)

    # -- section 4: cluster_status ---------------------------------------------------------------------------------

    async def _my_running_by_partition(self, cluster: str) -> dict[str, int]:
        def fn(conn: Any) -> dict[str, int]:
            out: dict[str, int] = {}
            for r in self.store.list_jobs(conn, cluster=cluster, states=list(RUNNING_STATES)):
                tj = r.get("target_json")
                try:
                    target = json.loads(tj) if isinstance(tj, str) else (tj or {})
                except ValueError:
                    target = {}
                for p in (target.get("partitions") or []):
                    out[p] = out.get(p, 0) + 1
            return out
        try:
            return await self.store.read(fn)
        except Exception:
            return {}

    async def _handles_by_slurm_id(self, cluster: str) -> dict[str, str]:
        def fn(conn: Any) -> dict[str, str]:
            rows = self.store.select(conn, "attempts", {"cluster": cluster})
            return {str(r["slurm_id"]): r["handle"] for r in rows if r.get("slurm_id")}
        try:
            return await self.store.read(fn)
        except Exception:
            return {}

    async def cluster_status(self, cluster: str, refresh: bool = False, detail: str = "partitions",
                             ) -> ClusterStatusResult:
        """Section 4 ``cluster_status`` from the discovery cache and the 60 s snapshot."""
        if detail not in ("summary", "partitions", "queue", "targets", "full"):
            raise err("E_INVALID_SPEC", f"detail must be summary|partitions|queue|targets|full, got {detail!r}")
        profile = self.profile(cluster)
        caps = await self.caps(cluster, refresh=refresh)
        snap = await self.snapshot(cluster, max_age_s=0 if refresh else SNAPSHOT_MAX_AGE_S)
        snap_parts: Mapping[str, Any] = snap.get("partitions") or {}
        partitions: Mapping[str, Mapping[str, Any]] = caps.get("partitions") or {}
        my_running = await self._my_running_by_partition(cluster)
        mine_pending: dict[str, dict[Any, int]] = {}
        for row in snap.get("mine") or []:
            d = P.classify_demand(row, partitions)
            for pname in d["partitions"] or [d["partition"]]:
                key = d["type"] if d["kind"] == "gpu" else "cpu"
                mine_pending.setdefault(pname, {})[key] = mine_pending.setdefault(pname, {}).get(key, 0) + 1
        names = sorted(partitions, key=lambda n: (not partitions[n].get("accessible", True), n))[:MAX_PARTITIONS]
        infos: list[PartitionInfo] = []
        for name in names if detail != "summary" else []:
            part = partitions[name]
            sp = snap_parts.get(name) or {}
            nodes = sp.get("nodes") or {}
            if not nodes:
                states = ((caps.get("sinfo") or {}).get(name) or {}).get("states") or {}
                nodes = {"idle": states.get("idle", 0), "mix": states.get("mix", 0), "alloc": states.get("alloc", 0),
                         "other": sum(v for k, v in states.items() if k not in ("idle", "mix", "alloc")),
                         "total": sum(states.values())}
            limits = part.get("limits") or {}
            charge = part.get("charge") or "free"

            def gres_rows(bucket: Mapping[Any, int], mine: Mapping[Any, int]) -> list[GresCount]:
                keys = set(bucket) | set(mine)
                out = []
                for k in sorted(keys, key=lambda x: (x is None, str(x))):
                    label = "untyped" if k is None else str(k)
                    out.append(GresCount(gres=label, count=int(bucket.get(k, 0)), mine=int(mine.get(k, 0))))
                return out

            pend_mine = mine_pending.get(name, {})
            infos.append(PartitionInfo(
                name=name, avail=part.get("state"), preempt_mode=",".join(part.get("preempt_mode") or []) or None,
                priority_tier=part.get("priority_tier"), grace_time_s=part.get("grace_time_s"),
                max_wall_s=limits.get("max_wall_s"), default_time_s=part.get("default_time_s"),
                nodes=NodeCounts(**{k: int(nodes.get(k, 0)) for k in ("idle", "mix", "alloc", "other", "total")}),
                gres_types=list(part.get("gres_type_list") or []),
                pending_by_gres=gres_rows(sp.get("pending") or {}, pend_mine),
                running_by_gres=gres_rows(sp.get("running") or {}, {}),
                my_jobs=MyJobs(pending=sum(pend_mine.values()), running=my_running.get(name, 0)),
                limits=PartitionLimits(max_wall_s=limits.get("max_wall_s"), max_jobs_pu=limits.get("max_jobs_pu"),
                                       max_submit_pu=limits.get("max_submit_pu"),
                                       max_tres_pj={k: float(v) for k, v in (limits.get("max_tres_pj") or {}).items()}),
                qos=(part.get("qos_candidates") or [None])[0],
                charge=Charge(**charge) if isinstance(charge, Mapping) else "free"))
        resv_rows = [ReservationRow(name=r["name"], start_ts=r.get("start_ts"), end_ts=r.get("end_ts"),
                                    partitions=P.parse_list(r.get("partition")), maint=bool(r.get("maintenance")))
                     for r in (snap.get("resv") or caps.get("reservations") or [])
                     if r.get("end_ts") is None or r["end_ts"] >= snap.get("ts", 0)]
        result = ClusterStatusResult(
            summary="", cluster=cluster, partitions=infos, su_balance=caps.get("su_balance"),
            quota=self._quota_rows(caps), reservations_upcoming=resv_rows, slurm_version=caps.get("slurm_version"),
            helper_version=caps.get("helper_sha8"), caps_age_s=caps_age_s(caps))
        if detail == "queue":
            handles = await self._handles_by_slurm_id(cluster)
            result.queue = [QueueRow(slurm_id=str(r.get("slurm_id")), partition=r.get("partition"), state="PENDING",
                                     reason=r.get("reason"), est_start_ts=r.get("start_ts"),
                                     handle=handles.get(str(r.get("slurm_id"))))
                            for r in (snap.get("mine") or [])[:MAX_QUEUE_ROWS]]
        if detail == "targets":
            policy = await self.placement_policy()
            result.targets = self._target_rows(cluster, profile, caps, policy)
        if detail == "full":
            result.config = dict((caps.get("config") or {}).get("raw") or {})
            result.config.update({"cmd_timeout_s": caps.get("cmd_timeout_s"), "pending_cap": caps.get("pending_cap"),
                                  "pending_cap_part": caps.get("pending_cap_part"), "squeue_O_zero": caps.get("squeue_O_zero"),
                                  "tools": caps.get("tools"), "default_account": caps.get("default_account"),
                                  "qos_candidates": caps.get("qos_candidates"), "control_root": caps.get("control_root")})
        idle_gpu = [f"{n}={sum((snap_parts.get(n) or {}).get('idle_gres', {}).values())}"
                    for n in names if partitions[n].get("has_gpu")][:4]
        pending_total = sum((snap_parts.get(n) or {}).get("pending_total", 0) for n in partitions)
        bal = f"{caps.get('su_balance'):.0f} SU" if isinstance(caps.get("su_balance"), (int, float)) else "n/a"
        result.summary = (f"{cluster}: {len(partitions)} partitions ({len(infos)} shown), idle GPU nodes "
                          f"{' '.join(idle_gpu) or 'none'}, {pending_total} pending jobs cluster-wide, SU balance {bal}, "
                          f"helpers {caps.get('helper_sha8') or 'not deployed'}, caps {int(result.caps_age_s or 0)} s old")
        result.unread_events = await self.unread()
        result.next = "plan_job(job) to rank targets; submit_job(job) to run"
        return result

    def _target_rows(self, cluster: str, profile: ClusterProfile, caps: Mapping[str, Any], policy: PlacementPolicy,
                     ) -> list[TargetRow]:
        rows: list[TargetRow] = []
        partitions: Mapping[str, Mapping[str, Any]] = caps.get("partitions") or {}
        keys: list[str] = []
        for name, part in partitions.items():
            if not part.get("accessible", partition_accessible(part, caps)):
                continue
            qos = (part.get("qos_candidates") or [None])[0]
            types = list(part.get("gres_type_list") or [])
            for gtype in types or [None]:
                key = f"{cluster}:{name}" + (f":{gtype}" if gtype else "") + (f"@{qos}" if qos else "")
                keys.append(key)
        for group in profile.partition_groups:
            if all(g in partitions for g in group):
                first = partitions[group[0]]
                qos = (first.get("qos_candidates") or [None])[0]
                for gtype in list(first.get("gres_type_list") or []) or [None]:
                    keys.append(f"{cluster}:{','.join(group)}" + (f":{gtype}" if gtype else "") + (f"@{qos}" if qos else ""))
        for key in keys:
            ov = target_override(profile, key)
            max_running = [v for g, v in policy.max_running_per_target.items() if fnmatch.fnmatchcase(key, g)]
            if ov.get("max_running") is not None:
                max_running.append(int(ov["max_running"]))
            max_pending = ov.get("max_pending", policy.max_pending_per_target)
            if max_pending is None:
                max_pending = caps.get("pending_cap_part") or caps.get("pending_cap")
            rows.append(TargetRow(target=key, enabled=bool(ov.get("enabled", True)), max_pending=max_pending,
                                  max_running=min(max_running) if max_running else None))
        return rows

    # -- section 4: run_command --------------------------------------------------------------------------------------

    async def run_command(self, cluster: str, command: str, timeout_s: int = 60, cwd: str | None = None,
                          max_chars: int = 8000) -> RunCommandResult:
        """Section 4 ``run_command``: the escape hatch under ``bash -lc``; refuses heredocs and > 4000 chars."""
        self.profile(cluster)
        text = command if isinstance(command, str) else str(command)
        if not text.strip():
            raise err("E_INVALID_SPEC", "command is empty")
        if len(text) > RUN_COMMAND_MAX_CHARS:
            raise err("E_CMD_TOO_LONG", f"command is {len(text)} chars (limit {RUN_COMMAND_MAX_CHARS})")
        if "<<" in text:
            raise err("E_CMD_TOO_LONG", "heredocs are refused in run_command")
        full = f"cd {commands.path_quote(cwd)} && {text}" if cwd else text
        timeout = max(1, min(int(timeout_s), RUN_COMMAND_MAX_TIMEOUT_S))
        client = self.client(cluster)
        t0 = time.monotonic()
        try:
            res = await client.run(full, timeout=timeout, idempotent=False)
        except CommandTimeout as e:
            out, trunc1 = _tail(e.stdout, max_chars)
            errt, trunc2 = _tail(e.stderr, max_chars)
            return RunCommandResult(summary=f"{cluster}: command timed out after {timeout} s (it may still be running)",
                                    unread_events=await self.unread(), rc=None, stdout_tail=out, stderr_tail=errt,
                                    truncated=trunc1 or trunc2, seconds=round(time.monotonic() - t0, 2),
                                    next="raise timeout_s, or run it as a job with submit_job")
        except SlurmMcpError:
            raise
        except (ConnectionDropped, asyncssh.Error, OSError) as e:
            raise err("E_SSH", f"{cluster}: {type(e).__name__}: {e}") from e
        out, trunc1 = _tail(res.stdout, max_chars)
        errt, trunc2 = _tail(res.stderr, max_chars)
        summary = f"{cluster}: rc={res.returncode} in {res.seconds:.1f} s"
        if res.returncode != 0 and errt.strip():
            summary += f": {errt.strip().splitlines()[-1][:160]}"
        return RunCommandResult(summary=summary, unread_events=await self.unread(), rc=res.returncode, stdout_tail=out,
                                stderr_tail=errt, truncated=trunc1 or trunc2, seconds=round(res.seconds, 2),
                                next=None if res.returncode == 0 else "read stderr_tail; use submit_job for real work")

    # -- section 4: files -------------------------------------------------------------------------------------------

    async def remote_ls(self, cluster: str, path: str, glob: str | None = None, max_entries: int = 200,
                        sort: str = "name") -> ListingResult:
        if sort not in ("name", "mtime", "size"):
            raise err("E_INVALID_SPEC", f"sort must be name|mtime|size, got {sort!r}")
        self.profile(cluster)
        await self.caps(cluster)
        out = await self._guard(cluster, self.client(cluster).ls(path, glob, max_entries, sort))
        entries = [ListingEntry(**e) for e in out["entries"]]
        n_dirs = sum(1 for e in entries if e.type == "dir")
        summary = f"{cluster}:{path}: {len(entries)} entries ({n_dirs} dirs)" + (" [truncated]" if out["truncated"] else "")
        return ListingResult(summary=summary, unread_events=await self.unread(), path=path, entries=entries,
                             truncated=out["truncated"],
                             next="raise max_entries or pass glob" if out["truncated"] else None)

    async def remote_read(self, cluster: str, path: str, tail_lines: int | None = 100, head_lines: int | None = None,
                          grep: str | None = None, offset: int | None = None, max_chars: int = 12000) -> ReadResult:
        self.profile(cluster)
        await self.caps(cluster)
        out = await self._guard(cluster, self.client(cluster).read_file(
            path, tail_lines=tail_lines, head_lines=head_lines, grep=grep, offset=offset, max_chars=max_chars))
        lines = out["text"].count("\n") + (1 if out["text"] and not out["text"].endswith("\n") else 0)
        summary = f"{cluster}:{path}: {lines} line(s) ({out['mode']}), size {out['size']} bytes"
        if out["truncated"]:
            summary += " [truncated]"
        return ReadResult(summary=summary, unread_events=await self.unread(), path=path, text=out["text"], size=out["size"],
                          next_offset=out["next_offset"], truncated=out["truncated"],
                          next=(f"remote_read(offset={out['next_offset']})" if out["next_offset"] is not None else None))

    async def remote_write(self, cluster: str, path: str, text: str, mode: str = "overwrite", mkdirs: bool = True,
                           executable: bool = False) -> WriteResult:
        self.profile(cluster)
        await self.caps(cluster)
        out = await self._guard(cluster, self.client(cluster).write_file(path, text, mode, mkdirs=mkdirs,
                                                                          executable=executable))
        summary = f"{cluster}:{path}: {'appended' if mode == 'append' else 'wrote'} {out['bytes']} bytes"
        if out["warnings"]:
            summary += f" ({', '.join(out['warnings'])})"
        return WriteResult(summary=summary, unread_events=await self.unread(), path=path, bytes=out["bytes"],
                           next=f"run_command('{cluster}', 'bash {path}')" if executable else None)

    # -- section 4: configure ---------------------------------------------------------------------------------------

    async def configure(self, placement: Mapping[str, Any] | None = None, notify: Mapping[str, Any] | None = None,
                        ) -> ConfigResult:
        """Merge patches into ``kv.policy.placement``/``kv.policy.notify`` (validated); no args = read."""
        await self.load_policies()
        changed: list[str] = []
        if placement is not None:
            if not isinstance(placement, Mapping):
                raise err("E_INVALID_SPEC", "placement must be an object of PlacementPolicy fields")
            merged = self._placement.model_dump()
            for k, v in placement.items():
                if k == "rebalance" and isinstance(v, Mapping):
                    merged["rebalance"] = {**merged.get("rebalance", {}), **v}
                else:
                    merged[k] = v
            self._placement = parse_input(PlacementPolicy, merged)
            changed += [f"placement.{k}" for k in placement]
        if notify is not None:
            if not isinstance(notify, Mapping):
                raise err("E_INVALID_SPEC", "notify must be an object of NotifyPolicy fields")
            merged = {**self._notify.model_dump(), **dict(notify)}
            self._notify = parse_input(NotifyPolicy, merged)
            changed += [f"notify.{k}" for k in notify]
        if changed:
            pl, nt = self._placement.model_dump(), self._notify.model_dump()

            def fn(conn: Any) -> None:
                self.store.kv_set(conn, POLICY_PLACEMENT_KEY, pl)
                self.store.kv_set(conn, POLICY_NOTIFY_KEY, nt)
            await self.store.write(fn)
        summary = (f"updated {', '.join(changed)}" if changed else "current policies") + \
            f": objective={self._placement.objective}, su_reserve={self._placement.su_reserve:g}, " \
            f"toast={'on' if self._notify.toast else 'off'}, email={self._notify.email or 'none'}"
        return ConfigResult(summary=summary, unread_events=await self.unread(), placement=self._placement,
                            notify=self._notify)


def _tail(text: str, max_chars: int) -> tuple[str, bool]:
    text = text or ""
    n = max(0, int(max_chars))
    if len(text) <= n:
        return text, False
    return text[-n:], True


__all__ = ["ClusterRegistry", "Service", "RUN_COMMAND_MAX_CHARS", "RUN_COMMAND_MAX_TIMEOUT_S", "SNAPSHOT_MAX_AGE_S",
           "TICK_STALE_S", "LEASE_LOOP_S", "POLICY_PLACEMENT_KEY", "POLICY_NOTIFY_KEY", "SNAPSHOT_KEY_PREFIX",
           "AuthFailed", "Unreachable", "LeaseLost", "save_caps"]
