"""Transfer tools: ``upload``, ``download``, ``collect_results`` (design section 4 "Files" and "Results").

The operations are module-level ``async`` functions taking the ``Service`` (CLI parity, section 1 rule 8);
``register(mcp, service)`` wraps them as MCP tools and attaches the ``transfer`` component on first use.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .. import _mcp
from .._mcp import Context, MCPServer
from ..errors import err
from ..models import CollectResult, CollectRow, TransferResult
from ..transfer import TransferManager
from . import BIG_RESULT_META, run_tool

log = logging.getLogger("slurm_mcp.tools.transfers")

UPLOAD_DESC = (
    "Upload a file or directory to a cluster, incrementally: only files whose size or mtime changed since the "
    "last upload to the same destination are sent. Ignores .git/, __pycache__/, *.pyc, .venv/, node_modules/, "
    ".slurm-mcp/, wandb/, *.ckpt plus any .slurm-mcpignore in the directory and the ignore list you pass. Picks "
    "tar (many or small files) or per-file SFTP automatically; mode= forces one. Refuses when the DESTINATION "
    "path lacks room (E_QUOTA names the free space and the five largest files) and caps a call at 2000 files / "
    "2 GB. dry_run=True lists what would be sent without sending it. Returns transfer_id t<N>; if the copy is "
    "still running after wait_s (default 600 s) the call returns state=running and the transfer_done event "
    "closes it. Files keep their bytes exactly; a warning names scripts containing CRLF."
)
DOWNLOAD_DESC = (
    "Download files matching remote globs (absolute, or relative to the cluster's remote_root; ** allowed) into "
    "a local directory, incrementally against the last download to the same directory. Files modified in the "
    "last 15 s are skipped as still-being-written and listed in skipped_in_progress. Remote names that are not "
    "legal on Windows (: ? * \" < > | control chars, trailing dots, CON/NUL/PRN/AUX/COM1-9/LPT1-9) are saved "
    "under a safe name and reported in renamed; paths longer than 260 chars use the extended-length prefix. "
    "Refuses rather than truncating when more than max_files (2000) or max_bytes (2 GB) match. Returns "
    "transfer_id t<N> and, when still running after wait_s, state=running plus the event to wait for."
)
COLLECT_DESC = (
    "Collect a finished (or running) job's results: the globs in its spec.outputs -- or the patterns you pass -- "
    "resolved against the job's workdir ON THE CLUSTER IT LAST RAN ON, plus stdout/stderr and progress.json when "
    "include_logs (default). Saved under <local_dir or ./results>/<name>-<handle>/, incrementally, so calling it "
    "again after more output appears fetches only the new bytes. Marks the job collected once it is terminal. "
    "One row per handle with state, exit_code, files, bytes and local_path, so a single call closes the loop "
    "after wait_for_events reports the job finished."
)


def _manager(service: Any) -> TransferManager:
    comp = service.components.get("transfer")
    if comp is None:
        comp = service.attach("transfer", TransferManager(service))
    return comp


async def upload(service: Any, cluster: str, local: str, remote: str, ignore: list[str] | None = None,
                 mode: str = "auto", dry_run: bool = False, wait_s: float = 600.0,
                 progress: Any = None) -> TransferResult:
    """Section 4 ``upload``."""
    service.profile(cluster)
    res = await _manager(service).upload(cluster, local, remote, ignore=ignore, mode=mode, dry_run=dry_run,
                                         wait_s=wait_s, progress=progress)
    res.unread_events = await service.unread()
    return res


async def download(service: Any, cluster: str, remote_globs: list[str], local_dir: str,
                   incremental: bool = True, max_files: int = 2000, max_bytes: int = 2_000_000_000,
                   wait_s: float = 600.0, progress: Any = None) -> TransferResult:
    """Section 4 ``download``."""
    service.profile(cluster)
    if not remote_globs:
        raise err("E_INVALID_SPEC", "remote_globs is empty", fix="pass at least one glob, e.g. ['logs/*.out']")
    res = await _manager(service).download(cluster, remote_globs, local_dir, incremental=incremental,
                                           max_files=max_files, max_bytes=max_bytes, wait_s=wait_s,
                                           progress=progress)
    res.unread_events = await service.unread()
    return res


async def collect_results(service: Any, ids: list[str], local_dir: str | None = None,
                          patterns: list[str] | None = None, include_logs: bool = True,
                          wait_s: float = 600.0, progress: Any = None) -> CollectResult:
    """Section 4 ``collect_results``."""
    if not ids:
        raise err("E_INVALID_SPEC", "ids is empty", fix="pass the job handles, e.g. ['j17']")
    rows = await _manager(service).collect(ids, local_dir=local_dir, patterns=patterns,
                                           include_logs=include_logs, wait_s=wait_s, progress=progress)
    files = sum(int(r.get("files") or 0) for r in rows)
    total = sum(int(r.get("bytes") or 0) for r in rows)
    states = ", ".join(f"{r['handle']} {r.get('state')}" for r in rows)
    return CollectResult(
        summary=f"collected {files} file(s) from {len(rows)} job(s): {states}",
        jobs=[CollectRow(**{k: v for k, v in r.items() if k in CollectRow.model_fields}) for r in rows],
        local_dir=local_dir, unread_events=await service.unread(),
        next=None if files else "no files matched; check spec.outputs or pass patterns=[...]")


def register(mcp: MCPServer, service: Any) -> None:
    """Register the three transfer tools (section 4)."""

    @mcp.tool(name="upload", description=UPLOAD_DESC, annotations=_mcp.mutating(), meta=BIG_RESULT_META)
    async def upload_tool(cluster: str, local: str, remote: str, ignore: Optional[list[str]] = None,
                          mode: str = "auto", dry_run: bool = False, wait_s: float = 600.0,
                          ctx: Context | None = None) -> TransferResult:
        return await run_tool(upload(service, cluster, local, remote, ignore, mode, dry_run, wait_s,
                                     _progress(ctx)))

    @mcp.tool(name="download", description=DOWNLOAD_DESC, annotations=_mcp.mutating(), meta=BIG_RESULT_META)
    async def download_tool(cluster: str, remote_globs: list[str], local_dir: str, incremental: bool = True,
                            max_files: int = 2000, max_bytes: int = 2_000_000_000, wait_s: float = 600.0,
                            ctx: Context | None = None) -> TransferResult:
        return await run_tool(download(service, cluster, remote_globs, local_dir, incremental, max_files,
                                       max_bytes, wait_s, _progress(ctx)))

    @mcp.tool(name="collect_results", description=COLLECT_DESC, annotations=_mcp.mutating(), meta=BIG_RESULT_META)
    async def collect_tool(ids: list[str], local_dir: Optional[str] = None,
                           patterns: Optional[list[str]] = None, include_logs: bool = True,
                           wait_s: float = 600.0, ctx: Context | None = None) -> CollectResult:
        return await run_tool(collect_results(service, ids, local_dir, patterns, include_logs, wait_s,
                                              _progress(ctx)))


def _progress(ctx: Context | None) -> Any:
    """A ``(fraction, message)`` callback forwarding to ``ctx.report_progress`` (keeps the idle timer alive)."""
    if ctx is None:
        return None

    async def cb(fraction: float, message: str) -> None:
        try:
            await ctx.report_progress(float(fraction), 1.0, message)
        except Exception:
            pass
    return cb


__all__ = ["upload", "download", "collect_results", "register"]
