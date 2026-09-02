"""Cluster tools: ``clusters``, ``cluster_status``, ``run_command`` (design section 4 "Clusters")."""
from __future__ import annotations

from typing import Any, Literal, Optional

from .. import _mcp
from .._mcp import MCPServer
from ..models import ClusterStatusResult, ClustersResult, RunCommandResult
from . import BIG_RESULT_META, run_tool

CLUSTERS_DESC = (
    "List the configured SLURM clusters with connection state (connected, auth_failed, reachable), the age of the "
    "last monitor tick, tracked job counts (queued/pending/running), SU balance, disk quota rows (path, used %, "
    "free GB, role) and who runs the background Monitor. Also returns session_id, the default client_id of "
    "wait_for_events. refresh=True re-runs read-only capability discovery on every cluster (partitions, QOS, "
    "limits, tools, quota; cached 24 h otherwise) and reports unreachable clusters in warnings instead of failing. "
    "Cheap (~150 tokens): call it first when unsure which clusters exist or whether the VPN is up. Helper "
    "deployment happens on the first submit_job/allocate, not here."
)
STATUS_DESC = (
    "Describe one cluster from the discovery cache and a load snapshot cached 60 s: per partition the state, "
    "preempt mode, priority tier, effective max wall, node counts (idle/mix/alloc/other/total), GPU types, pending "
    "and running jobs per GPU type ('untyped' = jobs that asked for GPUs without a type, counted against every "
    "type), my pending/running jobs, QOS limits (max_jobs_pu, max_submit_pu, per-job TRES), the QOS the server "
    "will pass and the SU charge per unit-hour ('free' on uncharged clusters). Plus SU balance, quota rows, "
    "upcoming reservations (maint=true blocks placement), SLURM version, deployed helper version and cache age. "
    "detail: 'summary' (no partition rows), 'partitions' (default, <= 20), 'queue' (my pending rows with "
    "est_start_ts/reason/handle), 'targets' (candidate target keys with enabled/max_pending/max_running), "
    "'full' (config keys too). refresh=True forces discovery + snapshot. Use plan_job to rank targets for a job."
)
RUN_DESC = (
    "Run an arbitrary shell command on the cluster's login node under 'bash -lc' (modules/PATH as at login) and "
    "return rc, the last max_chars of stdout/stderr, truncated and seconds. Escape hatch for inspection "
    "(squeue/sacct/sinfo, ls, cat, git, module avail): DESTRUCTIVE, so never use it to submit work - "
    "submit_job/allocate keep lineage, recovery and events; use job_control/rebalance for scancel/hold/requeue. "
    "Refuses heredocs and commands longer than 4000 chars (E_CMD_TOO_LONG): remote_write the script, then run it. "
    "cwd sets the working directory; timeout_s is capped at 600 (a timed-out command returns rc=null and may still "
    "run). Login nodes are shared: keep commands short and never loop/poll (the Monitor already polls)."
)


def register(mcp: MCPServer, service: Any) -> None:
    @mcp.tool(name="clusters", description=CLUSTERS_DESC, annotations=_mcp.read_only())
    async def clusters(refresh: bool = False) -> ClustersResult:
        return await run_tool(service.clusters(refresh=refresh))

    @mcp.tool(name="cluster_status", description=STATUS_DESC, annotations=_mcp.read_only())
    async def cluster_status(cluster: str, refresh: bool = False,
                             detail: Literal["summary", "partitions", "queue", "targets", "full"] = "partitions",
                             ) -> ClusterStatusResult:
        return await run_tool(service.cluster_status(cluster, refresh=refresh, detail=detail))

    @mcp.tool(name="run_command", description=RUN_DESC, annotations=_mcp.destructive(), meta=dict(BIG_RESULT_META))
    async def run_command(cluster: str, command: str, timeout_s: int = 60, cwd: Optional[str] = None,
                          max_chars: int = 8000) -> RunCommandResult:
        return await run_tool(service.run_command(cluster, command, timeout_s=timeout_s, cwd=cwd, max_chars=max_chars))


__all__ = ["register", "CLUSTERS_DESC", "STATUS_DESC", "RUN_DESC"]
