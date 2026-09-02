"""TransferManager: uploads, downloads and result collection (design section 5.5, tools of section 4).

Every transfer is a row in ``transfers`` plus one ``transfer_files`` row per file, so an interrupted call (or a
server restart) resumes instead of starting over. The tool awaits the background task for ``wait_s`` and then
returns the ``t<N>`` handle with a non-terminal state (section 1 rule 7).

Host routing (section 2.3, measured 2026-09-02): file work goes to whichever host serves SFTP -- the login node
on TRACE, the DTN on Bridges-2 whose login node refuses the subsystem -- and tar archives are extracted wherever
an exec channel exists (never on a PSC DTN, which refuses every command). Uploads are therefore:

  * tar + exec on the transfer host   when that host runs commands (neither target cluster does today);
  * tar via SFTP + extract on login   when SFTP and exec live on different hosts (Bridges-2);
  * tar over exec stdin              when the login host has both (TRACE);
  * per-file SFTP                    for small deltas, with ``.part`` resume for large files.

Quota is checked against the *destination* path through :func:`slurm_mcp.slurm.parse.df_row_for_path`, never the
mount: TRACE reports 932 TB at 1 % for ``$HOME`` and 3 TB at 74 % for the group volume on the same mount.
"""
from __future__ import annotations

import asyncio
import fnmatch
import io
import logging
import os
import posixpath
import secrets
import shlex
import tarfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import SlurmMcpError, err
from .models import Renamed, TransferResult
from .slurm.parse import df_row_for_path
from .slurm.states import TransferState
from .store import transfer_handle
from .textio import local_safe_name, long_path, normcase_scope, posix_rel

log = logging.getLogger("slurm_mcp.transfer")

# section 5.5 / 11f
TAR_FILE_THRESHOLD = 16                      # >= this many changed files -> tar
TAR_MAX_BYTES = 256 * 1024 * 1024            # tar only below this delta
SMALL_FILE_BYTES = 1024 * 1024               # "small" file for the tar heuristic
SFTP_BLOCK = 65536
SFTP_MAX_REQUESTS = 64
RESUME_MIN_BYTES = 64 * 1024 * 1024          # files at/above this resume by offset
MAX_FILES = 2000                             # hard caps per call
MAX_BYTES = 2 * 1024 * 1024 * 1024
IN_PROGRESS_S = 15                           # skip files modified in the last 15 s (cluster time)
QUOTA_HEADROOM = 1.2                         # refuse when free < 1.2 x delta
DEFAULT_IGNORE = (".git/", "__pycache__/", "*.pyc", ".venv/", "node_modules/", ".slurm-mcp/", "wandb/", "*.ckpt")
IGNORE_FILE = ".slurm-mcpignore"


# --- ignore rules -------------------------------------------------------------------------------------------

def _load_patterns(local: Path, extra: Sequence[str] | None) -> list[str]:
    pats = list(DEFAULT_IGNORE)
    f = (local / IGNORE_FILE) if local.is_dir() else None
    if f is not None and f.exists():
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                pats.append(line)
    pats.extend(extra or ())
    return pats


def ignored(rel: str, patterns: Sequence[str]) -> bool:
    """gitignore-ish matching on a POSIX relative path: ``dir/`` prefixes and ``fnmatch`` globs."""
    parts = rel.split("/")
    for pat in patterns:
        p = pat.strip()
        if not p:
            continue
        if p.endswith("/"):
            if p.rstrip("/") in parts:
                return True
            continue
        if fnmatch.fnmatchcase(rel, p) or fnmatch.fnmatchcase(parts[-1], p):
            return True
        if "/" not in p and any(fnmatch.fnmatchcase(x, p) for x in parts):
            return True
    return False


# --- local scan ---------------------------------------------------------------------------------------------

def scan_local(local: Path, patterns: Sequence[str]) -> list[dict[str, Any]]:
    """``[{rel, abs, size, mtime_ns}]`` for a file or directory tree; ``rel`` is always POSIX (section 5.5)."""
    local = Path(local)
    if local.is_file():
        st = local.stat()
        return [{"rel": local.name, "abs": str(local), "size": st.st_size, "mtime_ns": st.st_mtime_ns}]
    out: list[dict[str, Any]] = []
    for root, dirs, files in os.walk(local):
        rroot = Path(root)
        rel_root = posix_rel(rroot.relative_to(local)) if rroot != local else ""
        dirs[:] = [d for d in dirs if not ignored(f"{rel_root}/{d}".strip("/") + "/", patterns)
                   and not ignored(f"{rel_root}/{d}".strip("/"), patterns)]
        for name in files:
            rel = f"{rel_root}/{name}".strip("/")
            if ignored(rel, patterns):
                continue
            try:
                st = (rroot / name).stat()
            except OSError:
                continue
            out.append({"rel": rel, "abs": str(rroot / name), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    out.sort(key=lambda e: e["rel"])
    return out


def choose_mode(delta: Sequence[Mapping[str, Any]]) -> str:
    """``"tar"`` or ``"sftp"`` per section 11f: tar for many files or a small-file heavy delta under 256 MB."""
    if not delta:
        return "sftp"
    total = sum(int(e["size"]) for e in delta)
    if total >= TAR_MAX_BYTES:
        return "sftp"
    if len(delta) >= TAR_FILE_THRESHOLD:
        return "tar"
    if sum(1 for e in delta if int(e["size"]) < SMALL_FILE_BYTES) >= max(2, len(delta) // 2) and len(delta) > 4:
        return "tar"
    return "sftp"


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


class TransferManager:
    """Attached as ``service.attach("transfer", TransferManager(service))``."""

    def __init__(self, service: Any) -> None:
        self.service = service
        self.tasks: dict[int, asyncio.Task] = {}

    # -- plumbing ------------------------------------------------------------------------------------------

    @property
    def store(self) -> Any:
        return self.service.store

    async def _caps(self, cluster: str) -> dict[str, Any]:
        return dict(await self.service.caps(cluster) or {})

    def _client(self, cluster: str) -> Any:
        return self.service.client(cluster)

    async def _quota_row(self, cluster: str, destination: str) -> dict[str, Any] | None:
        caps = await self._caps(cluster)
        return df_row_for_path(caps.get("df") or [], destination)

    async def _check_quota(self, cluster: str, destination: str, need_bytes: int,
                           biggest: Sequence[Mapping[str, Any]]) -> float | None:
        """Refuse when the destination's free space is below ``1.2 x`` the delta (section 5.5)."""
        row = await self._quota_row(cluster, destination)
        if not row or not row.get("kb_free"):
            return None
        free = int(row["kb_free"]) * 1024
        if free < need_bytes * QUOTA_HEADROOM:
            top = ", ".join(f"{e['rel']} ({_fmt_bytes(int(e['size']))})" for e in list(biggest)[:5])
            raise err("E_QUOTA",
                      f"{destination} on {cluster} has {_fmt_bytes(free)} free ({row.get('used_pct')}% used) but "
                      f"the upload needs {_fmt_bytes(need_bytes)}; largest: {top}",
                      fix="delete files, pick another destination, or narrow the upload with ignore patterns")
        after = (int(row["kb_used"]) * 1024 + need_bytes) / max(1.0, int(row["kb_total"]) * 1024) * 100.0
        return round(after, 1)

    # -- rows ----------------------------------------------------------------------------------------------

    async def _destination_exists(self, cluster: str, remote: str) -> bool:
        """Is the upload destination still on the cluster? One ``test -e``; failures answer True (do not
        discard a good manifest because a probe failed)."""
        try:
            res = await self._client(cluster).run(
                f"test -e {shlex.quote(remote)} && echo yes || echo no", idempotent=True)
            return "no" not in res.stdout.split()
        except Exception as e:
            log.info("%s: destination probe for %s failed (%s); keeping the manifest", cluster, remote, e)
            return True

    async def _new_transfer(self, *, kind: str, cluster: str, host_role: str, local: str, remote: str,
                            mode: str | None, files: Sequence[Mapping[str, Any]], handle: str | None) -> int:
        def fn(conn: Any) -> int:
            tid = self.store.insert_transfer(conn, kind=kind, cluster=cluster, host_role=host_role, local=local,
                                             remote=remote, state=str(TransferState.planned), mode=mode,
                                             files_total=len(files), bytes_total=sum(int(f["size"]) for f in files),
                                             handle=handle, started_local=time.time())
            for f in files:
                self.store.upsert_transfer_file(conn, tid, f["rel"], size=int(f["size"]),
                                                mtime_ns=int(f.get("mtime_ns") or 0), state="planned")
            return tid
        return await self.store.write(fn)

    async def _finish(self, tid: int, state: str, *, error: str | None = None, **fields: Any) -> None:
        await self.store.write(lambda c: self.store.update_transfer(
            c, tid, state=state, error=error, finished_local=time.time(), **fields))

    async def status(self, tid: int) -> dict[str, Any] | None:
        return await self.store.read(lambda c: self.store.get_transfer(c, tid))

    async def cancel(self, tid: int) -> bool:
        task = self.tasks.get(tid)
        if task and not task.done():
            task.cancel()
            return True
        return False

    # -- upload --------------------------------------------------------------------------------------------

    async def upload(self, cluster: str, local: str, remote: str, *, ignore: Sequence[str] | None = None,
                     mode: str = "auto", dry_run: bool = False, wait_s: float = 600.0,
                     progress: Any = None, handle: str | None = None) -> TransferResult:
        """Section 4 ``upload``: incremental against the ``up:<cluster>:<remote>`` manifest."""
        lpath = Path(local).expanduser()
        if not lpath.exists():
            raise err("E_INVALID_SPEC", f"local path does not exist: {local}", fix="check the path")
        if mode not in ("auto", "tar", "sftp"):
            raise err("E_INVALID_SPEC", f"mode must be auto|tar|sftp, got {mode!r}")
        patterns = _load_patterns(lpath, ignore)
        files = await asyncio.to_thread(scan_local, lpath, patterns)
        scope = f"up:{cluster}:{remote}"
        manifest = await self.store.read(lambda c: self.store.manifest(c, scope))
        if manifest and not await self._destination_exists(cluster, remote):
            # The manifest records what we sent, not what is still there. If the destination was deleted or
            # moved on the cluster, trusting it would silently skip every file and report "already current".
            log.info("%s: %s is gone; discarding the upload manifest and sending everything", cluster, remote)
            await self.store.write(lambda c: self.store.delete_manifest(c, scope))
            manifest = {}
        delta = [f for f in files
                 if (m := manifest.get(f["rel"])) is None
                 or int(m["size"]) != int(f["size"]) or int(m["mtime_ns"]) != int(f["mtime_ns"])]
        skipped = len(files) - len(delta)
        total_bytes = sum(int(f["size"]) for f in delta)
        if len(delta) > MAX_FILES:
            raise err("E_TOO_MANY_FILES", f"{len(delta)} files exceed the {MAX_FILES} per-call limit",
                      fix="upload subdirectories separately or add ignore patterns")
        if total_bytes > MAX_BYTES:
            raise err("E_TOO_MANY_BYTES", f"{_fmt_bytes(total_bytes)} exceeds the {_fmt_bytes(MAX_BYTES)} limit",
                      fix="upload in parts or stage the data with Globus")
        if dry_run:
            listing = [f["rel"] for f in sorted(delta, key=lambda e: -int(e["size"]))[:100]]
            return TransferResult(
                summary=(f"dry run: would send {len(delta)} file(s), {_fmt_bytes(total_bytes)} to "
                         f"{cluster}:{remote} ({skipped} unchanged)"),
                state=TransferState.planned, files_sent=0, files_skipped=skipped, bytes=total_bytes,
                mode=choose_mode(delta) if mode == "auto" else mode, remote=remote, would_send=listing,
                next="upload(...) without dry_run to send them")
        if not delta:
            return TransferResult(summary=f"nothing to send: {len(files)} file(s) already current on {cluster}",
                                  state=TransferState.done, files_skipped=skipped, remote=remote,
                                  local_dir=str(lpath))
        quota_after = await self._check_quota(cluster, remote, total_bytes,
                                              sorted(delta, key=lambda e: -int(e["size"])))
        chosen = choose_mode(delta) if mode == "auto" else mode
        caps = await self._caps(cluster)
        host_role = "transfer" if (caps.get("transfer") or {}).get("role") == "transfer" else "login"
        tid = await self._new_transfer(kind="inputs" if handle else "upload", cluster=cluster, host_role=host_role,
                                       local=str(lpath), remote=remote, mode=chosen, files=delta, handle=handle)
        task = asyncio.ensure_future(self._run_upload(tid, cluster, lpath, remote, delta, chosen, scope, progress))
        self.tasks[tid] = task
        return await self._await_task(task, tid, wait_s, kind="upload",
                                      base=dict(files_skipped=skipped, mode=chosen, remote=remote,
                                                local_dir=str(lpath), quota_after_pct=quota_after))

    async def _run_upload(self, tid: int, cluster: str, lpath: Path, remote: str,
                          delta: Sequence[Mapping[str, Any]], mode: str, scope: str, progress: Any) -> dict[str, Any]:
        t0 = time.time()
        await self.store.write(lambda c: self.store.update_transfer(c, tid, state=str(TransferState.running)))
        client = self._client(cluster)
        done_bytes = 0
        try:
            if mode == "tar":
                await self._upload_tar(client, cluster, lpath, remote, delta, progress)
            else:
                await self._upload_sftp(client, tid, lpath, remote, delta, progress)
            done_bytes = sum(int(f["size"]) for f in delta)

            def fn(conn: Any) -> None:
                for f in delta:
                    self.store.update_transfer_file(conn, tid, f["rel"], state="done", bytes_done=int(f["size"]))
                    self.store.upsert_manifest(conn, scope, f["rel"], size=int(f["size"]),
                                               mtime_ns=int(f["mtime_ns"]))
                self.store.update_transfer(conn, tid, files_done=len(delta), bytes_done=done_bytes,
                                           state=str(TransferState.done), finished_local=time.time(),
                                           seconds=time.time() - t0)
            await self.store.write(fn)
        except asyncio.CancelledError:
            await self._finish(tid, str(TransferState.cancelled), error="cancelled")
            raise
        except Exception as e:
            await self._finish(tid, str(TransferState.failed), error=str(e)[:500])
            await self._emit(tid, "transfer_failed", cluster, f"upload to {cluster}:{remote} failed: {e}",
                             {"transfer_id": transfer_handle(tid), "error": str(e)[:300]})
            raise
        await self._emit(tid, "transfer_done", cluster,
                         f"uploaded {len(delta)} file(s), {_fmt_bytes(done_bytes)} to {cluster}:{remote}",
                         {"transfer_id": transfer_handle(tid), "files": len(delta), "bytes": done_bytes,
                          "seconds": round(time.time() - t0, 1), "renamed": []})
        return {"files_sent": len(delta), "bytes": done_bytes, "seconds": time.time() - t0}

    async def _upload_sftp(self, client: Any, tid: int, lpath: Path, remote: str,
                           delta: Sequence[Mapping[str, Any]], progress: Any) -> None:
        sftp, role = await client.file_sftp()
        if sftp is None:                       # no SFTP anywhere: stream each file through an exec channel
            for i, f in enumerate(delta):
                target = posixpath.join(remote, f["rel"]) if not lpath.is_file() else remote
                data = await asyncio.to_thread(Path(f["abs"]).read_bytes)
                await client._exec_write(target, data)
                await self._file_done(tid, f, i, len(delta), progress)
            return
        dirs = {posixpath.dirname(posixpath.join(remote, f["rel"])) for f in delta}
        for d in sorted(x for x in dirs if x):
            await sftp.makedirs(client.expand_path(d), exist_ok=True)
        for i, f in enumerate(delta):
            target = client.expand_path(posixpath.join(remote, f["rel"]) if not lpath.is_file() else remote)
            size = int(f["size"])
            if size >= RESUME_MIN_BYTES:
                part = f"{target}.part-{tid}"
                start = 0
                try:
                    st = await sftp.stat(part)
                    start = int(getattr(st, "size", 0) or 0)
                except Exception:
                    start = 0
                if start > size:
                    start = 0
                async with sftp.open(part, "ab" if start else "wb") as fh:
                    with open(long_path(f["abs"]), "rb") as src:
                        src.seek(start)
                        while chunk := src.read(SFTP_BLOCK * 8):
                            await fh.write(chunk)
                await sftp.posix_rename(part, target)
            else:
                await sftp.put(long_path(f["abs"]), target, block_size=SFTP_BLOCK,
                               max_requests=SFTP_MAX_REQUESTS)
            await self._file_done(tid, f, i, len(delta), progress)

    async def _upload_tar(self, client: Any, cluster: str, lpath: Path, remote: str,
                          delta: Sequence[Mapping[str, Any]], progress: Any) -> None:
        """Build the archive locally, then extract it wherever an exec channel exists (section 2.3)."""
        buf = await asyncio.to_thread(self._build_tar, lpath, delta)
        caps = await self._caps(cluster)
        transfer_caps = dict(caps.get("transfer") or {})
        quoted_remote = shlex.quote(client.expand_path(remote))
        if progress:
            await _call(progress, 0.5, f"extracting {len(delta)} file(s) on {cluster}")
        import tempfile

        def _extract_host() -> Any:
            """The transport that both accepts an exec channel and can see the destination filesystem."""
            if transfer_caps.get("exec_ok") and client.transfer is not None:
                return client.transfer
            return client.login

        sftp, role = await client.file_sftp()
        if sftp is not None and role == "transfer":
            # SFTP and exec live on different hosts (Bridges-2): stage the tar with SFTP, extract over exec.
            staging = f"{client.expand_path(remote)}/.slurm-mcp-upload-{secrets.token_hex(4)}.tar"
            await sftp.makedirs(client.expand_path(remote), exist_ok=True)
            async with sftp.open(staging, "wb") as fh:
                await fh.write(buf)
            res = await _extract_host().run(
                f"tar -xf {shlex.quote(staging)} -C {quoted_remote} && rm -f {shlex.quote(staging)}",
                idempotent=True)
            if res.returncode != 0:
                raise err("E_UPLOAD", f"tar extract failed on {cluster}: {res.stderr.strip()[:300]}",
                          fix="check the destination path and quota")
            return
        # one host serves both channels (TRACE): stream the archive straight into tar
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as tmp:
            tmp.write(buf)
            tmp_path = tmp.name
        try:
            res = await client.login.run_with_stdin_file(
                f"mkdir -p {quoted_remote} && tar -xf - -C {quoted_remote}", tmp_path)
            if res.returncode != 0:
                raise err("E_UPLOAD", f"tar extract failed on {cluster}: {res.stderr.strip()[:300]}",
                          fix="check the destination path and quota")
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def _build_tar(lpath: Path, delta: Sequence[Mapping[str, Any]]) -> bytes:
        bio = io.BytesIO()
        with tarfile.open(fileobj=bio, mode="w") as tar:
            for f in delta:
                info = tar.gettarinfo(long_path(f["abs"]), arcname=f["rel"])
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                if f["rel"].endswith((".sh", ".py")):
                    info.mode |= 0o755
                with open(long_path(f["abs"]), "rb") as fh:
                    tar.addfile(info, fh)
        return bio.getvalue()

    async def _file_done(self, tid: int, f: Mapping[str, Any], i: int, total: int, progress: Any) -> None:
        await self.store.write(lambda c: self.store.update_transfer_file(
            c, tid, f["rel"], state="done", bytes_done=int(f["size"])))
        if progress and (i % 5 == 0 or i + 1 == total):
            await _call(progress, (i + 1) / max(1, total), f"{i + 1}/{total} files")

    # -- download / collect --------------------------------------------------------------------------------

    async def download(self, cluster: str, remote_globs: Sequence[str], local_dir: str, *,
                       incremental: bool = True, max_files: int = MAX_FILES, max_bytes: int = MAX_BYTES,
                       wait_s: float = 600.0, progress: Any = None, handle: str | None = None,
                       kind: str = "download", rel_base: str | None = None) -> TransferResult:
        """Section 4 ``download``: expand globs remotely, skip in-flight files, save under NTFS-safe names."""
        client = self._client(cluster)
        caps = await self._caps(cluster)
        root = caps.get("remote_root") or self.service.profile(cluster).remote_root or caps.get("home") or "."
        globs = [g if g.startswith("/") else posixpath.join(root, g) for g in remote_globs]
        listing = await self._remote_list(client, globs)
        # Cluster time, extrapolated from the last probe -- never the bootstrap snapshot (which can be hours
        # old, making every freshly written file look like it was modified "in the future" and be skipped).
        try:
            now = int(client.clock.remote_now())
        except Exception:
            now = int(caps.get("remote_now") or time.time())
        # 0 <= age < IN_PROGRESS_S is "still being written"; a negative age means our clock estimate lags the
        # cluster's, which must not hide the file.
        fresh = [e for e in listing if 0 <= (now - int(e.get("mtime") or 0)) < IN_PROGRESS_S]
        entries = [e for e in listing if e not in fresh]
        if len(entries) > max_files:
            raise err("E_TOO_MANY_FILES", f"{len(entries)} files match; the limit is {max_files}",
                      fix="narrow the globs")
        total = sum(int(e["size"]) for e in entries)
        if total > max_bytes:
            biggest = ", ".join(f"{e['path']} ({_fmt_bytes(int(e['size']))})"
                                for e in sorted(entries, key=lambda x: -int(x["size"]))[:5])
            raise err("E_TOO_MANY_BYTES", f"{_fmt_bytes(total)} exceeds the limit; largest: {biggest}",
                      fix="narrow the globs or raise max_bytes")
        ldir = Path(local_dir).expanduser()
        ldir.mkdir(parents=True, exist_ok=True)
        scope = f"down:{normcase_scope(str(ldir))}"
        manifest = await self.store.read(lambda c: self.store.manifest(c, scope)) if incremental else {}
        # ``rel_base`` lets collect_results control the local layout (logs flat, outputs under the workdir)
        # instead of mirroring the whole remote path from a common root that may be several levels up.
        base = rel_base if rel_base is not None else _common_root(globs)
        delta = []
        for e in entries:
            rel = posix_rel(posixpath.relpath(e["path"], base)) if base else posixpath.basename(e["path"])
            m = manifest.get(rel)
            if m and int(m["size"]) == int(e["size"]) and int(m["mtime_ns"]) == int(e["mtime"]) * 10**9:
                continue
            delta.append({"rel": rel, "path": e["path"], "size": int(e["size"]), "mtime_ns": int(e["mtime"]) * 10**9})
        if not delta:
            if fresh and not entries:
                # everything matched but was written within IN_PROGRESS_S: say so, do not claim "already local"
                return TransferResult(
                    summary=(f"nothing fetched: {len(fresh)} file(s) were modified in the last {IN_PROGRESS_S}s "
                             f"and may still be being written"),
                    state=TransferState.done, local_dir=str(ldir),
                    skipped_in_progress=[e["path"] for e in fresh],
                    next=f"call again in {IN_PROGRESS_S}s, or after the job reaches a terminal state")
            return TransferResult(summary=f"nothing to fetch: {len(entries)} file(s) already local",
                                  state=TransferState.done, files_skipped=len(entries), local_dir=str(ldir),
                                  skipped_in_progress=[e["path"] for e in fresh])
        tid = await self._new_transfer(kind=kind, cluster=cluster, host_role="transfer", local=str(ldir),
                                       remote=",".join(remote_globs), mode="sftp", files=delta, handle=handle)
        task = asyncio.ensure_future(self._run_download(tid, cluster, ldir, delta, scope, progress))
        self.tasks[tid] = task
        return await self._await_task(task, tid, wait_s, kind="download",
                                      base=dict(files_skipped=len(entries) - len(delta), local_dir=str(ldir),
                                                mode="sftp",
                                                skipped_in_progress=[e["path"] for e in fresh]))

    async def _remote_list(self, client: Any, globs: Sequence[str]) -> list[dict[str, Any]]:
        """``{path, size, mtime}`` for every file matching the globs (one exec, ``stat`` per match)."""
        parts = " ".join(shlex.quote(g) for g in globs)
        cmd = (f"for p in {parts}; do for f in $p; do [ -f \"$f\" ] && stat -c '%s|%Y|%n' \"$f\"; done; done "
               f"2>/dev/null | sort -u")
        res = await client.run(cmd, idempotent=True)
        out: list[dict[str, Any]] = []
        for line in res.stdout.splitlines():
            f = line.split("|", 2)
            if len(f) == 3 and f[0].isdigit():
                out.append({"size": int(f[0]), "mtime": int(f[1]), "path": f[2].strip()})
        return out

    async def _run_download(self, tid: int, cluster: str, ldir: Path, delta: Sequence[Mapping[str, Any]],
                            scope: str, progress: Any) -> dict[str, Any]:
        t0 = time.time()
        await self.store.write(lambda c: self.store.update_transfer(c, tid, state=str(TransferState.running)))
        client = self._client(cluster)
        renamed: list[dict[str, str]] = []
        got = 0
        try:
            sftp, role = await client.file_sftp()
            for i, f in enumerate(delta):
                safe = local_safe_name(f["rel"])
                if safe != f["rel"]:
                    renamed.append({"remote": f["rel"], "local": safe})
                dest = ldir / safe
                dest.parent.mkdir(parents=True, exist_ok=True)
                if sftp is not None:
                    await sftp.get(client.expand_path(f["path"]), long_path(str(dest)),
                                   block_size=SFTP_BLOCK, max_requests=SFTP_MAX_REQUESTS)
                else:
                    res = await client.run(f"cat {shlex.quote(f['path'])}", idempotent=True)
                    await asyncio.to_thread(Path(long_path(str(dest))).write_text, res.stdout, "utf-8")
                got += int(f["size"])
                await self.store.write(lambda c, r=f, s=safe: self.store.update_transfer_file(
                    c, tid, r["rel"], state="done", bytes_done=int(r["size"]),
                    local_name=None if s == r["rel"] else s))
                if progress and (i % 5 == 0 or i + 1 == len(delta)):
                    await _call(progress, (i + 1) / max(1, len(delta)), f"{i + 1}/{len(delta)} files")

            def fn(conn: Any) -> None:
                for r in delta:
                    self.store.upsert_manifest(conn, scope, r["rel"], size=int(r["size"]),
                                               mtime_ns=int(r["mtime_ns"]))
                self.store.update_transfer(conn, tid, files_done=len(delta), bytes_done=got,
                                           state=str(TransferState.done), finished_local=time.time(),
                                           seconds=time.time() - t0)
            await self.store.write(fn)
        except asyncio.CancelledError:
            await self._finish(tid, str(TransferState.cancelled), error="cancelled")
            raise
        except Exception as e:
            await self._finish(tid, str(TransferState.failed), error=str(e)[:500])
            await self._emit(tid, "transfer_failed", cluster, f"download from {cluster} failed: {e}",
                             {"transfer_id": transfer_handle(tid), "error": str(e)[:300]})
            raise
        await self._emit(tid, "transfer_done", cluster,
                         f"downloaded {len(delta)} file(s), {_fmt_bytes(got)} from {cluster}",
                         {"transfer_id": transfer_handle(tid), "files": len(delta), "bytes": got,
                          "seconds": round(time.time() - t0, 1), "renamed": renamed})
        return {"files_sent": len(delta), "bytes": got, "seconds": time.time() - t0, "renamed": renamed}

    async def collect(self, handles: Sequence[str], *, local_dir: str | None = None,
                      patterns: Sequence[str] | None = None, include_logs: bool = True,
                      wait_s: float = 600.0, progress: Any = None) -> list[dict[str, Any]]:
        """Section 4 ``collect_results``: ``spec.outputs`` (or ``patterns``) plus stdout/stderr of the **current
        attempt's** cluster and workdir, into ``<local_dir>/<name>-<handle>/``. Returns one row per handle."""
        rows: list[dict[str, Any]] = []
        base = Path(local_dir).expanduser() if local_dir else Path.cwd() / "results"
        for h in handles:
            job = await self.store.read(lambda c, hh=h: self.store.get_job(c, hh))
            if job is None:
                raise err("E_UNKNOWN_ID", f"no job {h!r}", fix="list_jobs() to see the handles")
            spec = _spec_of(job)
            globs = list(patterns or spec.get("outputs") or [])
            workdir = job.get("workdir") or ""
            abs_globs = [g if g.startswith("/") else posixpath.join(workdir, g) for g in globs]
            log_globs: list[str] = []
            if include_logs:
                for key in ("stdout_path", "stderr_path"):
                    if job.get(key):
                        log_globs.append(job[key])
                if job.get("ctrl_dir"):
                    log_globs.append(posixpath.join(job["ctrl_dir"], "progress*.json"))
            target = base / f"{job.get('name') or 'job'}-{h}"
            if not abs_globs and not log_globs:
                rows.append({"handle": h, "state": job.get("state"), "exit_code": job.get("exit_code"),
                             "transfer_id": None, "files": 0, "bytes": 0, "skipped": 0,
                             "local_path": str(target),
                             "note": "no outputs declared; pass patterns=[...] or set spec.outputs"})
                continue
            files = bytes_ = skipped = 0
            tid = None
            # outputs keep their layout relative to the job's workdir; logs land flat next to them, so a
            # collected job reads as "<name>-<handle>/slurm-<id>.out" and not a mirror of the remote tree.
            for globs_, rel in ((abs_globs, workdir), (log_globs, None)):
                if not globs_:
                    continue
                for group, gbase in _group_by_dir(globs_, rel):
                    res = await self.download(job["cluster"], group, str(target), wait_s=wait_s,
                                              progress=progress, handle=h, kind="collect", rel_base=gbase)
                    tid = tid or res.transfer_id
                    files += res.files_sent
                    bytes_ += res.bytes
                    skipped += res.files_skipped
            if job.get("state") in ("COMPLETED", "FAILED", "TIMEOUT", "OOM", "CANCELLED", "PREEMPTED",
                                    "NODE_FAIL"):
                await self.store.write(lambda c, hh=h: self.store.update_job(c, hh, collected_ts=int(time.time())))
            rows.append({"handle": h, "state": job.get("state"), "exit_code": job.get("exit_code"),
                         "transfer_id": tid, "files": files, "bytes": bytes_,
                         "skipped": skipped, "local_path": str(target)})
        return rows

    # -- shared --------------------------------------------------------------------------------------------

    async def _await_task(self, task: asyncio.Task, tid: int, wait_s: float, *, kind: str,
                          base: Mapping[str, Any]) -> TransferResult:
        handle = transfer_handle(tid)
        try:
            res = await asyncio.wait_for(asyncio.shield(task), timeout=max(0.0, float(wait_s)))
        except asyncio.TimeoutError:
            return TransferResult(summary=f"{kind} {handle} still running in the background",
                                  transfer_id=handle, state=TransferState.running, **dict(base),
                                  next=f"job_status(['{handle}']) or wait_for_events(kinds=['transfer_done'])")
        except SlurmMcpError:
            raise
        except Exception as e:
            row = await self.status(tid) or {}
            return TransferResult(summary=f"{kind} {handle} failed: {e}", transfer_id=handle,
                                  state=TransferState.failed, error=str(e)[:300], **dict(base),
                                  next="fix the cause and call again; finished files are not re-sent")
        renamed = [Renamed(**r) for r in (res.get("renamed") or [])]
        return TransferResult(
            summary=(f"{kind}: {res['files_sent']} file(s), {_fmt_bytes(res['bytes'])} in "
                     f"{res['seconds']:.1f}s" + (f"; {len(renamed)} renamed for Windows" if renamed else "")),
            transfer_id=handle, state=TransferState.done, files_sent=res["files_sent"], bytes=res["bytes"],
            seconds=round(res["seconds"], 1), renamed=renamed, **dict(base))

    async def _emit(self, tid: int, kind: str, cluster: str, summary: str, payload: Mapping[str, Any]) -> None:
        events = getattr(self.service, "events", None)
        if events is None:
            return
        row = await self.status(tid) or {}
        await events.emit(kind, handle=row.get("handle"), cluster=cluster, summary=summary, payload=dict(payload))


async def _call(fn: Any, fraction: float, message: str) -> None:
    try:
        res = fn(fraction, message)
        if asyncio.iscoroutine(res):
            await res
    except Exception:
        pass



def _group_by_dir(globs: Sequence[str], rel_base: str | None):
    """Group globs so each download gets a sensible ``rel_base``.

    With ``rel_base`` given (job outputs) every glob shares it. Without one (logs) each glob is based on its
    own directory, so the files land flat in the destination instead of mirroring the remote tree.
    """
    if rel_base is not None:
        return [(list(globs), rel_base)]
    out: dict[str, list[str]] = {}
    for g in globs:
        head = g
        for ch in ("*", "?", "["):
            idx = head.find(ch)
            if idx >= 0:
                head = head[:idx]
        out.setdefault(posixpath.dirname(head.rstrip("/")) or "/", []).append(g)
    return [(v, k) for k, v in out.items()]


def _spec_of(job: Mapping[str, Any]) -> dict[str, Any]:
    """The job's JobSpec dict from the ledger row (``spec_json``)."""
    import json
    raw = job.get("spec_json")
    if isinstance(raw, str):
        try:
            return dict(json.loads(raw))
        except ValueError:
            return {}
    return dict(raw or {})


def _common_root(globs: Sequence[str]) -> str:
    """The longest directory prefix shared by every glob, up to its first wildcard (download rel paths)."""
    fixed = []
    for g in globs:
        head = g
        for ch in ("*", "?", "["):
            idx = head.find(ch)
            if idx >= 0:
                head = head[:idx]
        fixed.append(posixpath.dirname(head.rstrip("/")) if not head.endswith("/") else head.rstrip("/"))
    if not fixed:
        return ""
    root = fixed[0]
    for f in fixed[1:]:
        while root and not (f == root or f.startswith(root + "/")):
            root = posixpath.dirname(root)
    return root


__all__ = ["TransferManager", "scan_local", "choose_mode", "ignored", "DEFAULT_IGNORE"]
