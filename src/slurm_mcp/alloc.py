"""AllocManager: reusable interactive-style allocations (design sections 4 "Allocations", 5.7, 7.3).

An allocation is an ordinary batch job whose payload is ``alloc-agent.sh`` (section 7.3): a one-second event
loop on the allocated node that runs numbered command files, records each command's rc and output, honours kill
files, refreshes a heartbeat and exits on ``release``. That beats ``srun --jobid --overlap`` because a step
launched over an SSH channel dies with the channel, while a file queue survives reconnects, server restarts and
a sleeping laptop -- and every command keeps its exit code and output on the cluster.

Naming must match the agent exactly: command ``NNN.sh`` (foreground, one at a time) or ``NNN.bg.sh``
(detached, any number), with ``<base>.out``, ``<base>.pid``, ``<base>.started``, ``<base>.rc``, ``<base>.done``
and the kill file ``<base>.kill`` -- so ``002.kill`` for ``002.sh`` and ``003.bg.kill`` for ``003.bg.sh``.
"""
from __future__ import annotations

import asyncio
import logging
import posixpath
import shlex
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import SlurmMcpError, err
from .models import AllocRunResult, JobSpec, Resources, SubmitResult
from .slurm.states import CmdState, JobState

log = logging.getLogger("slurm_mcp.alloc")

POLL_S = 3.0                    # section 5.7: poll rc/out every 3 s while a caller waits
OUT_TAIL_BYTES = 4096
EXPIRING_S = 600                # alloc_expiring 10 min before the end


class AllocManager:
    """Attached as ``service.attach("alloc", AllocManager(service))``."""

    def __init__(self, service: Any) -> None:
        self.service = service

    @property
    def store(self) -> Any:
        return self.service.store

    def _client(self, cluster: str) -> Any:
        return self.service.client(cluster)

    # -- allocate ------------------------------------------------------------------------------------------

    async def allocate(self, cluster: str, resources: Resources | Mapping[str, Any], hours: float, *,
                       partition: str | None = None, qos: str | None = None, name: str = "alloc",
                       placement: str = "auto", workdir: str | None = None, setup: str | None = None,
                       modules: Sequence[str] | None = None, idle_release_min: int = 0,
                       wait_s: float = 90.0, progress: Any = None) -> SubmitResult:
        """Section 4 ``allocate``: submit the sleeper job through the normal submit pipeline, kind ``alloc``."""
        res = resources if isinstance(resources, Resources) else Resources.model_validate(resources)
        caps = await self.service.caps(cluster)
        hours = self._cap_hours(hours, caps, partition)
        res = res.model_copy(update={"time": _hms(int(hours * 3600))})
        spec = JobSpec(name=name, command=":", resources=res, cluster=cluster, partition=partition, qos=qos,
                       workdir=workdir, setup=setup, modules=list(modules or []), wrap=False, requeue=False)
        submitter = self.service.components.get("submitter")
        if submitter is None:
            raise err("E_STATE", "the submitter component is not attached",
                      fix="restart the server; allocations reuse the submit pipeline")
        handle, _task = await submitter.submit(spec, placement=placement if not partition else "auto",
                                               kind="alloc", alloc_idle_release_s=int(idle_release_min) * 60)
        # Two callback shapes meet here. The tool layer hands us the (fraction, message) callback that
        # run()'s polling loop below uses, but submitter.await_result calls its progress_cb with the
        # message alone -- and unlike _call it does not swallow the TypeError, so passing `progress`
        # straight through failed the whole allocation. Adapt instead of changing either contract.
        submit_cb = None
        if progress is not None:
            async def submit_cb(message: str) -> None:
                await _call(progress, 0.5, message)
        result = await submitter.await_result(handle, wait_s=wait_s, progress_cb=submit_cb)
        charge = _charge_note(caps, partition or (result.target or "").split(":", 1)[-1], hours)
        result.summary = (f"allocation {handle} on {cluster} for {hours:g} h" + charge
                          + f"; state {result.state}")
        result.next = f"wait_for_events(kinds=['alloc_ready'], job_ids=['{handle}']) then alloc_run('{handle}', ...)"
        return result

    @staticmethod
    def _cap_hours(hours: float, caps: Mapping[str, Any], partition: str | None) -> float:
        """Clamp to the partition's effective max wall (section 4: "hours is capped")."""
        part = ((caps.get("partitions") or {}).get(partition or "") or {})
        limit = ((part.get("limits") or {}).get("max_wall_s"))
        if limit and hours * 3600 > float(limit):
            return float(limit) / 3600.0
        return float(hours)

    # -- run / kill / release ------------------------------------------------------------------------------

    async def run(self, alloc_id: str, command: str, *, wait_s: float = 55.0, detach: bool = False,
                  cwd: str | None = None, progress: Any = None) -> AllocRunResult:
        """Section 4 ``alloc_run``: queue a command file for the agent and poll its rc/out while waiting."""
        job = await self._alloc_row(alloc_id)
        ctrl = job.get("ctrl_dir") or ""
        cluster = job["cluster"]
        client = self._client(cluster)
        # The store allocates n and the id atomically (max(n)+1 per handle) so two callers never collide.
        placeholder = f"{ctrl}/cmds"
        cmd_id, n = await self.store.write(lambda c: self.store.insert_alloc_cmd(
            c, handle=alloc_id, command=command, mode="bg" if detach else "fg", state=str(CmdState.queued),
            cwd=cwd, out_path=placeholder, kill_path=placeholder))
        base = f"{n:03d}" + (".bg" if detach else "")
        out_path = f"{ctrl}/cmds/{base}.out"
        kill_path = f"{ctrl}/cmds/{base}.kill"
        await self.store.write(lambda c: self.store.update_alloc_cmd(
            c, cmd_id, out_path=out_path, kill_path=kill_path))
        script = _cmd_script(cmd_id, cwd or job.get("workdir") or ".", command)
        # write the .sh LAST: the agent starts a command as soon as the file appears
        await client.write_file(f"{ctrl}/cmds/{base}.sh", script, executable=True)
        if detach:
            return AllocRunResult(
                summary=f"{cmd_id} started detached on {alloc_id}", cmd_id=cmd_id, alloc_id=alloc_id,
                state=CmdState.running, out_path=out_path, unread_events=await self.service.unread(),
                next=f"job_logs('{cmd_id}') or wait_for_events(kinds=['cmd_done'], job_ids=['{cmd_id}'])")
        deadline = time.time() + max(0.0, float(wait_s))
        started = time.time()
        while time.time() < deadline:
            await asyncio.sleep(POLL_S)
            rc, tail = await self._poll(client, ctrl, base)
            if progress:
                await _call(progress, 0.5, f"{cmd_id} running ({int(time.time() - started)}s)")
            if rc is not None:
                await self.store.write(lambda c: self.store.update_alloc_cmd(
                    c, cmd_id, state=str(CmdState.done), rc=int(rc), done_ts=int(time.time())))
                return AllocRunResult(
                    summary=f"{cmd_id} finished rc={rc} in {time.time() - started:.0f}s",
                    cmd_id=cmd_id, alloc_id=alloc_id, state=CmdState.done, rc=int(rc), out_tail=tail,
                    seconds=round(time.time() - started, 1), out_path=out_path,
                    unread_events=await self.service.unread(),
                    next=None if rc == 0 else f"job_logs('{cmd_id}') for the full output")
        rc, tail = await self._poll(client, ctrl, base)
        return AllocRunResult(
            summary=f"{cmd_id} still running after {wait_s:g}s", cmd_id=cmd_id, alloc_id=alloc_id,
            state=CmdState.running, out_tail=tail, out_path=out_path,
            seconds=round(time.time() - started, 1), unread_events=await self.service.unread(),
            next=f"job_logs('{cmd_id}') or wait_for_events(kinds=['cmd_done'], job_ids=['{cmd_id}'])")

    async def kill(self, cmd_id: str) -> bool:
        """Write ``<base>.kill``; the agent signals the command's process group within its 1 s loop (7.3)."""
        row = await self.store.read(lambda c: self.store.get_alloc_cmd(c, cmd_id))
        if row is None:
            raise err("E_UNKNOWN_ID", f"no allocation command {cmd_id!r}",
                      fix="job_status(['<alloc>']) lists the commands")
        job = await self._alloc_row(row["handle"])
        await self._client(job["cluster"]).write_file(row["kill_path"], "")
        await self.store.write(lambda c: self.store.update_alloc_cmd(
            c, cmd_id, kill_requested_local=time.time()))
        return True

    async def release(self, alloc_id: str) -> bool:
        """Write ``release`` (the agent exits cleanly), then ``scancel`` the job (sections 4, 5.7)."""
        job = await self._alloc_row(alloc_id)
        client = self._client(job["cluster"])
        try:
            await client.write_file(f"{job.get('ctrl_dir')}/release", "")
        except Exception as e:                       # the node may already be gone; scancel still applies
            log.info("release file for %s: %s", alloc_id, e)
        if job.get("slurm_id"):
            await client.cancel([job["slurm_id"]])
        return True

    # -- helpers -------------------------------------------------------------------------------------------

    async def _alloc_row(self, alloc_id: str) -> dict[str, Any]:
        job = await self.store.read(lambda c: self.store.get_job(c, alloc_id))
        if job is None or job.get("kind") != "alloc":
            raise err("E_UNKNOWN_ID", f"no allocation {alloc_id!r}",
                      fix="list_jobs(kind='alloc') to see them")
        state = str(job.get("state") or "")
        if state in ("QUEUED", "UPLOADING", "SUBMITTING", "SUBMITTED"):
            est = job.get("est_start_ts")
            raise err("E_ALLOC_NOT_READY",
                      f"{alloc_id} is still {state.lower()}" + (f", estimated start {est}" if est else ""),
                      fix=f"wait_for_events(kinds=['alloc_ready'], job_ids=['{alloc_id}'])")
        if state in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "PREEMPTED", "NODE_FAIL", "OOM", "LOST"):
            raise err("E_ALLOC_ENDED", f"{alloc_id} ended ({state})",
                      fix="allocate() a new one")
        return job

    async def _next_cmd(self, alloc_id: str) -> tuple[int, str]:
        """Monotonic per allocation (``alloc_cmds.n``), so file names never collide after a restart."""
        def fn(conn: Any) -> tuple[int, str]:
            rows = self.store.alloc_cmds_for(conn, alloc_id)
            n = max((int(r.get("n") or 0) for r in rows), default=0) + 1
            return n, f"{alloc_id}.c{n}"
        return await self.store.read(fn)

    async def _poll(self, client: Any, ctrl: str, base: str) -> tuple[int | None, str]:
        """``(rc, out_tail)`` in one exec: the agent writes ``<base>.rc`` only when the command finished."""
        rc_path = shlex.quote(f"{ctrl}/cmds/{base}.rc")
        out_path = shlex.quote(f"{ctrl}/cmds/{base}.out")
        res = await client.run(
            f"if [ -f {rc_path} ]; then echo \"::RC $(cat {rc_path})\"; fi; "
            f"tail -c {OUT_TAIL_BYTES} {out_path} 2>/dev/null", idempotent=True)
        rc: int | None = None
        lines = res.stdout.splitlines()
        if lines and lines[0].startswith("::RC "):
            try:
                rc = int(lines[0].split(None, 1)[1].strip())
            except (ValueError, IndexError):
                rc = None
            lines = lines[1:]
        return rc, "\n".join(lines)[-OUT_TAIL_BYTES:]


def _cmd_script(cmd_id: str, cwd: str, command: str) -> str:
    """The command file the agent runs (section 7.3 "Command files written by the server")."""
    return (f"# slurm-mcp cmd={cmd_id}\n"
            f"cd {shlex.quote(cwd)} || exit 97\n"
            f"set -o pipefail\n"
            f"{command}\n")


def _hms(seconds: int) -> str:
    h, rem = divmod(max(60, int(seconds)), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _charge_note(caps: Mapping[str, Any], partition: str, hours: float) -> str:
    part = ((caps.get("partitions") or {}).get(partition) or {})
    charge = part.get("charge")
    if not isinstance(charge, Mapping):
        return ""
    return f" (~{float(charge.get('su_per_unit_h') or 0) * hours:g} SU per {charge.get('unit')})"


async def _call(fn: Any, fraction: float, message: str) -> None:
    try:
        res = fn(fraction, message)
        if asyncio.iscoroutine(res):
            await res
    except Exception:
        pass


__all__ = ["AllocManager", "POLL_S", "EXPIRING_S"]
