"""File tools: ``remote_ls``, ``remote_read``, ``remote_write`` (design section 4 "Files").

``upload``/``download`` belong to the transfers group (``tools/transfers.py``, slice 5).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from .. import _mcp
from .._mcp import MCPServer
from ..models import ListingResult, ReadResult, WriteResult
from . import BIG_RESULT_META, run_tool

LS_DESC = (
    "List a remote directory (or stat one file) over SFTP: entries with name, type (file/dir/link/other), size and "
    "mtime_ts (cluster epoch seconds). path must be absolute on the cluster ($HOME is expanded); glob filters names "
    "(shell wildcard, e.g. '*.out'); sort by name (default), mtime (newest first) or size (largest first); "
    "max_entries caps the listing (truncated=true when more exist). Use it to find job outputs, check that an "
    "upload landed, or inspect a workdir before collect_results. Cheap; does not read file contents."
)
READ_DESC = (
    "Read part of a remote text file with one tail/head/grep command: default the last 100 lines (tail_lines), or "
    "the first head_lines, or 'grep -n -E' matches of a regex (max 200), or a byte window starting at offset. "
    "Output is capped at max_chars (default 12000, truncated=true beyond); size is the file size in bytes and "
    "next_offset tells where to continue for paging (pass it back as offset). Prefer job_logs for tracked jobs "
    "(it knows the paths and states); use remote_read for arbitrary files such as configs, progress JSON or "
    "adopted jobs' outputs. Missing files fail with an error naming the path."
)
WRITE_DESC = (
    "Write text to a remote file over SFTP: mode 'overwrite' writes atomically (temp file + rename), 'append' adds "
    "to the end; mkdirs creates parent directories; executable sets chmod 755. Text is normalised for the cluster "
    "(BOM removed, CRLF -> LF, NUL refused) so scripts written from Windows never break bash. Limit 1 MB - use "
    "upload for larger files or directories. Typical use: write a script or config, then submit_job with "
    "script_path=<remote path> or run_command('bash <path>'). Returns the path and the bytes written."
)


def register(mcp: MCPServer, service: Any) -> None:
    @mcp.tool(name="remote_ls", description=LS_DESC, annotations=_mcp.read_only())
    async def remote_ls(cluster: str, path: str, glob: Optional[str] = None, max_entries: int = 200,
                        sort: Literal["name", "mtime", "size"] = "name") -> ListingResult:
        return await run_tool(service.remote_ls(cluster, path, glob=glob, max_entries=max_entries, sort=sort))

    @mcp.tool(name="remote_read", description=READ_DESC, annotations=_mcp.read_only(), meta=dict(BIG_RESULT_META))
    async def remote_read(cluster: str, path: str, tail_lines: Optional[int] = 100, head_lines: Optional[int] = None,
                          grep: Optional[str] = None, offset: Optional[int] = None, max_chars: int = 12000,
                          ) -> ReadResult:
        return await run_tool(service.remote_read(cluster, path, tail_lines=tail_lines, head_lines=head_lines,
                                                  grep=grep, offset=offset, max_chars=max_chars))

    @mcp.tool(name="remote_write", description=WRITE_DESC,
              annotations=_mcp._ann(read_only_hint=False, destructive_hint=False, idempotent_hint=True))
    async def remote_write(cluster: str, path: str, text: str, mode: Literal["overwrite", "append"] = "overwrite",
                           mkdirs: bool = True, executable: bool = False) -> WriteResult:
        return await run_tool(service.remote_write(cluster, path, text, mode=mode, mkdirs=mkdirs, executable=executable))


__all__ = ["register", "LS_DESC", "READ_DESC", "WRITE_DESC"]
