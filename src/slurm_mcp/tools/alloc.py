"""Allocation tools: ``allocate`` and ``alloc_run`` (design section 4 "Allocations", section 5.7).

An allocation reserves a node for a stretch of time so many short commands run without re-queueing each one --
the interactive workflow, but driven from here and surviving disconnects.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .. import _mcp
from .._mcp import Context, MCPServer
from ..alloc import AllocManager
from ..models import AllocRunResult, Resources, SubmitResult
from . import run_tool

log = logging.getLogger("slurm_mcp.tools.alloc")

ALLOCATE_DESC = (
    "Reserve compute for a stretch of time and keep it: submits a placeholder job that idles on the allocated "
    "node running an agent, so you can then fire many short commands at it with alloc_run without queueing "
    "each one. resources takes the same shape as a job's (time is ignored; hours decides, capped at the "
    "partition's limit). placement='auto' ranks partitions the way submit_job does unless you name one. "
    "idle_release_min > 0 makes the allocation end itself after that many idle minutes so it stops charging. "
    "Returns handle a<N>; the alloc_ready event fires when the node is actually yours (that can take as long "
    "as any queue wait). Costs whatever the partition charges for the WHOLE reserved window, not just the "
    "time you use, so prefer short windows and idle_release_min on a charging cluster."
)
ALLOC_RUN_DESC = (
    "Run a command on an existing allocation's node. Foreground (default) queues it, waits up to wait_s "
    "(default 55 s) and returns rc plus the tail of its output; the allocation runs one foreground command at "
    "a time. detach=True returns immediately and the cmd_done event reports the exit code, so use it for "
    "anything long. Each command gets an id like a3.c2: job_logs('a3.c2') reads its full output and "
    "job_control(['a3.c2'], 'cancel') signals its process group. cwd defaults to the allocation's workdir. "
    "E_ALLOC_NOT_READY means the allocation is still queued (wait for alloc_ready); E_ALLOC_ENDED means it is "
    "over. Nothing runs on the login node: the command executes on the allocated compute node."
)


def _manager(service: Any) -> AllocManager:
    comp = service.components.get("alloc")
    if comp is None:
        comp = service.attach("alloc", AllocManager(service))
    return comp


async def allocate(service: Any, cluster: str, resources: Resources | dict, hours: float,
                   partition: str | None = None, qos: str | None = None, name: str = "alloc",
                   placement: str = "auto", workdir: str | None = None, setup: str | None = None,
                   modules: list[str] | None = None, idle_release_min: int = 0, wait_s: float = 90.0,
                   progress: Any = None) -> SubmitResult:
    """Section 4 ``allocate``."""
    service.profile(cluster)
    return await _manager(service).allocate(cluster, resources, hours, partition=partition, qos=qos, name=name,
                                            placement=placement, workdir=workdir, setup=setup, modules=modules,
                                            idle_release_min=idle_release_min, wait_s=wait_s, progress=progress)


async def alloc_run(service: Any, alloc_id: str, command: str, wait_s: float = 55.0, detach: bool = False,
                    cwd: str | None = None, progress: Any = None) -> AllocRunResult:
    """Section 4 ``alloc_run``."""
    return await _manager(service).run(alloc_id, command, wait_s=wait_s, detach=detach, cwd=cwd,
                                       progress=progress)


def register(mcp: MCPServer, service: Any) -> None:
    """Register the two allocation tools (section 4)."""

    @mcp.tool(name="allocate", description=ALLOCATE_DESC, annotations=_mcp.mutating())
    async def allocate_tool(cluster: str, resources: Resources, hours: float,
                            partition: Optional[str] = None, qos: Optional[str] = None, name: str = "alloc",
                            placement: str = "auto", workdir: Optional[str] = None,
                            setup: Optional[str] = None, modules: Optional[list[str]] = None,
                            idle_release_min: int = 0, wait_s: float = 90.0,
                            ctx: Context | None = None) -> SubmitResult:
        return await run_tool(allocate(service, cluster, resources, hours, partition, qos, name, placement,
                                       workdir, setup, modules, idle_release_min, wait_s, _progress(ctx)))

    @mcp.tool(name="alloc_run", description=ALLOC_RUN_DESC, annotations=_mcp.mutating())
    async def alloc_run_tool(alloc_id: str, command: str, wait_s: float = 55.0, detach: bool = False,
                             cwd: Optional[str] = None, ctx: Context | None = None) -> AllocRunResult:
        return await run_tool(alloc_run(service, alloc_id, command, wait_s, detach, cwd, _progress(ctx)))


def _progress(ctx: Context | None) -> Any:
    if ctx is None:
        return None

    async def cb(fraction: float, message: str) -> None:
        try:
            await ctx.report_progress(float(fraction), 1.0, message)
        except Exception:
            pass
    return cb


__all__ = ["allocate", "alloc_run", "register"]
