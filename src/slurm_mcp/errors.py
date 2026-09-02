"""Error catalogue and the SlurmMcpError -> ToolError bridge (design section 9.1, section 4 "Errors").

Every failure surfaced to the MCP client is a ToolError whose text is
``E_CODE: what happened <EM DASH> fix: what to do`` (the separator is U+2014 EM DASH surrounded by spaces).
This module imports nothing from the package except ``_mcp`` (for the ToolError class).
"""
from __future__ import annotations

from typing import Any

from ._mcp import ToolError

EM_DASH = "\u2014"  # U+2014 EM DASH

# Fixed default ``fix`` templates for every code named in design section 9.1.
# Templates may reference keyword placeholders that ``err(code, message, **fmt)`` fills in;
# missing placeholders are left verbatim (never a KeyError at error-construction time).
CATALOGUE: dict[str, str] = {
    "E_AUTH": "run 'slurm-mcp auth set {cluster}' to store the password, then clusters(refresh=True)",
    "E_HOSTKEY": "verify the host key out of band, then 'slurm-mcp hostkeys forget {cluster}' and retry",
    "E_UNREACHABLE": "check the network/VPN ({hint}) and call clusters(refresh=True)",
    "E_SSH": "retry; if it persists check clusters() for the connection state",
    "E_CTLD_BUSY": "the SLURM controller is busy; wait 60 s and retry (the server already backs off)",
    "E_SUBMIT_AMBIGUOUS": "wait for wait_for_events(kinds=['submitted','submit_failed']); the server recovers the id from the queue, never resubmits",
    "E_SUBMIT_FAILED": "read the sbatch message, adjust the spec and submit again",
    "E_PARTITION": "pick a partition from cluster_status('{cluster}', detail='partitions')",
    "E_PARTITION_REQUIRED": "set job.partition or use placement='auto'",
    "E_ACCOUNT": "set job.account or default_account in the profile (see cluster_status detail='full')",
    "E_QOS": "set job.qos to a QOS allowed for this partition (cluster_status detail='partitions')",
    "E_QOS_MAXWALL": "lower resources.time to at most {max_wall} or choose a partition with a longer MaxWall",
    "E_QOS_SIZE": "reduce gpus/nodes/cpus to the QOS/partition per-job limits or choose another target",
    "E_QOS_POLICY": "you are at a QOS/association limit; wait for jobs to finish or choose another target",
    "E_SUBMIT_LIMIT": "wait for pending jobs to start or finish; the server holds new jobs locally (state QUEUED)",
    "E_NODE_CONFIG": "the requested node configuration does not exist; check cluster_status(detail='partitions') for gres types and node sizes",
    "E_GRES": "use a gres type present in the partition (cluster_status gres_types) and gpus <= per-node count",
    "E_MEM": "lower resources.mem or drop it on partitions listed in profile.no_mem_flag",
    "E_DEPENDENCY": "depends_on entries must be handles like 'j12' or 'afterok:j12'; the dependency must be tracked and live",
    "E_DEP_CROSS_CLUSTER": "dependencies must be on the same cluster; pin job.cluster to {cluster} or drop the dependency",
    "E_PERMISSION": "check ownership and permissions of the path on the cluster",
    "E_SCRIPT": "the script must start with '#!' and be a valid batch script; see stripped_directives in the plan",
    "E_QUOTA": "free space under {path} (used {used_pct}%) or choose another cluster",
    "E_NO_TARGET": "no feasible target; see plan_job(...).options[*].why and relax the spec or policy",
    "E_UNKNOWN_ID": "use a handle from list_jobs() (j17, a3, a3.c2, t4) or '<cluster>:<slurm_id>' to adopt a job",
    "E_NO_LOG_YET": "the job is {state}; call {next}",
    "E_ALLOC_NOT_READY": "wait_for_events(kinds=['alloc_ready'], job_ids=['{alloc_id}']) then retry",
    "E_ALLOC_ENDED": "the allocation ended; call allocate(...) again",
    "E_CMD_TOO_LONG": "remote_write the script, then run it",
    "E_TOO_MANY_FILES": "narrow remote_globs or raise max_files (largest paths are listed)",
    "E_TOO_MANY_BYTES": "narrow remote_globs or raise max_bytes (largest paths are listed)",
    "E_UPLOAD": "check the local path and remote quota, then retry the upload (incremental)",
    "E_CONFIRM_REQUIRED": "call again with confirm=True to act on more than 10 ids",
    "E_PLAN_EXPIRED": "plans are valid for 15 min; call plan_job again",
    "E_INVALID_SPEC": "correct the job spec field named in the message",
    "E_HELPER": "the remote helper bundle is missing or stale; submit_job/allocate redeploys it, or run 'slurm-mcp helpers deploy {cluster}'",
    "E_STATE": "the object is in state {state}; this action is not valid there",
}

CODES: frozenset[str] = frozenset(CATALOGUE)


class _SafeDict(dict):
    """dict for str.format_map that leaves unknown placeholders in place."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _fill(template: str, fmt: dict[str, Any]) -> str:
    try:
        return template.format_map(_SafeDict(fmt))
    except (ValueError, IndexError):
        return template


class SlurmMcpError(Exception):
    """A user-facing failure (design section 9.1).

    ``str(exc)`` is exactly ``f"{code}: {message} <EM DASH> fix: {fix}"``.
    """

    def __init__(self, code: str, message: str, fix: str | None = None) -> None:
        self.code = code
        self.message = message
        self.fix = fix if fix is not None else CATALOGUE.get(code, "see the message")
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"{self.code}: {self.message} {EM_DASH} fix: {self.fix}"

    def __repr__(self) -> str:
        return f"SlurmMcpError(code={self.code!r}, message={self.message!r}, fix={self.fix!r})"

    def to_tool_error(self) -> ToolError:
        return ToolError(str(self))


def err(code: str, message: str, **fmt: Any) -> SlurmMcpError:
    """Build a SlurmMcpError from the catalogue.

    ``fix`` may be passed as a keyword to override the template; every other keyword fills the
    template's placeholders (unknown placeholders are left verbatim). Unknown codes raise ValueError
    so a typo cannot ship a nameless error.
    """
    if code not in CATALOGUE:
        raise ValueError(f"unknown error code {code!r}")
    fix_override = fmt.pop("fix", None)
    fix = fix_override if fix_override is not None else _fill(CATALOGUE[code], fmt)
    return SlurmMcpError(code, _fill(message, fmt) if fmt else message, fix)


def to_tool_error(exc: BaseException) -> ToolError:
    """Convert any exception to the ToolError the MCP layer raises (design section 4 "Errors").

    SlurmMcpError keeps its text; a ToolError passes through; anything else is wrapped as E_SSH-style
    generic failure text without a catalogue code guess (``E_STATE`` is not implied).
    """
    if isinstance(exc, ToolError):
        return exc
    if isinstance(exc, SlurmMcpError):
        return exc.to_tool_error()
    text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return ToolError(f"E_SSH: {text} {EM_DASH} fix: {CATALOGUE['E_SSH']}")


__all__ = ["EM_DASH", "CATALOGUE", "CODES", "SlurmMcpError", "err", "to_tool_error", "ToolError"]
