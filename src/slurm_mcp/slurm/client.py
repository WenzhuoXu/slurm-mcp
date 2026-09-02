"""SlurmClient: typed SLURM operations over one cluster's SSH transports (design sections 2 "slurm/client.py",
6.1 discovery, 6.2 tick/snapshot, 6.3 submit/test-only/control, 4 files).

Every remote SLURM command string comes from :mod:`slurm_mcp.slurm.commands`; every output goes through
:mod:`slurm_mcp.slurm.parse`; every timestamp is converted by the cluster's :class:`ClusterClock`. The only
command strings composed here are the file helpers (``remote_read``/``remote_ls`` of section 4), for which
``commands.py`` has no builder; they use its quoting helpers. No cluster name appears in this module.

Transport failures propagate as the transport's own exceptions (``AuthFailed``/``Unreachable`` are already
``SlurmMcpError``; ``CommandTimeout``/``ConnectionDropped`` are ambiguous and left to the caller) except in
``submit``/``test_only``, whose contracts say "ambiguous" / "no estimate" rather than an exception.
"""
from __future__ import annotations

import fnmatch
import logging
import posixpath
import re
import secrets
import shlex
import stat as statmod
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import asyncssh

from ..clock import ClusterClock
from ..config import ClusterProfile, control_root as profile_control_root
from ..errors import SlurmMcpError, err
from ..helpers import BUNDLE_FILES, bundle_files, bundle_sha8
from ..textio import normalize_text
from ..transport import CommandResult, CommandTimeout, ConnectionDropped, SSHTransport
from . import commands
from . import parse as P
from .parse import IncompleteProbe

log = logging.getLogger("slurm_mcp.client")

MAX_WRITE_BYTES = 1024 * 1024          # section 4 remote_write: <= 1 MB
DEFAULT_READ_CHARS = 12000             # section 4 remote_read default
GREP_MAX_MATCHES = 200                 # section 4: grep -n -E -m 200
CapsGetter = Callable[[], "Mapping[str, Any] | None"]


class TickFailed(IncompleteProbe):
    """A tick whose ``::SQUEUE``/``::SACCT`` section returned a non-zero rc (section 5.2: discarded whole)."""


def _sec(sections: Mapping[str, tuple[list[str], int | None]], name: str) -> list[str]:
    return list(sections.get(name, ([], None))[0])


def _rc(sections: Mapping[str, tuple[list[str], int | None]], name: str) -> int | None:
    return sections.get(name, ([], None))[1]


def _now_fallback(sections: Mapping[str, tuple[list[str], int | None]]) -> tuple[int, str] | None:
    """``::NOW <epoch> <host>`` via :func:`parse.parse_now`; tolerate a missing host (``hostname -s`` may fail on
    exotic login shells, which leaves the line in ``_pre``)."""
    now = P.parse_now(sections)
    if now is not None:
        return now
    for line in _sec(sections, "_pre"):
        m = re.match(r"^::NOW\s+(\d+)\s*(\S*)\s*$", line)
        if m:
            return int(m.group(1)), m.group(2)
    return None


def summarize_snapshot(snapshot: Mapping[str, Any], caps: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Per-partition view of a parsed snapshot (section 6.2 demand classification, section 4 cluster_status).

    Returns ``{partition: {"nodes": {idle, mix, alloc, other, total}, "idle_gres": {type: nodes},
    "pending": {type|None: jobs}, "running": {type|None: jobs}, "pending_total", "running_total"}}``. ``None``
    keys are untyped GPU demand (counted against every type by the placer); CPU demand lands under ``"cpu"``.
    Multi-partition pending rows count once per member partition.
    """
    partitions = dict((caps or {}).get("partitions") or {})
    out: dict[str, dict[str, Any]] = {}

    def entry(name: str) -> dict[str, Any]:
        return out.setdefault(name, {"nodes": {"idle": 0, "mix": 0, "alloc": 0, "other": 0, "total": 0},
                                     "idle_gres": {}, "pending": {}, "running": {}, "pending_total": 0,
                                     "running_total": 0})

    for row in snapshot.get("nodes") or []:
        e = entry(row["partition"])
        state = row.get("state") or "other"
        key = state if state in ("idle", "mix", "alloc") else "other"
        e["nodes"][key] += 1
        e["nodes"]["total"] += 1
        g = row.get("gres")
        if state == "idle" and g:
            e["idle_gres"][g["type"]] = e["idle_gres"].get(g["type"], 0) + 1
    for kind, key in (("pd", "pending"), ("r", "running")):
        for row in snapshot.get(kind) or []:
            d = P.classify_demand(row, partitions)
            gtype: Any = d["type"] if d["kind"] == "gpu" else "cpu"
            for pname in d["partitions"] or [d["partition"]]:
                e = entry(pname)
                e[key][gtype] = e[key].get(gtype, 0) + int(d["count"])
                e[key + "_total"] += int(d["count"])
    return out


class SlurmClient:
    """Typed operations for one cluster (design section 2 dependency direction: client -> transport).

    ``transport_login`` runs every SLURM command; ``transport_transfer`` (may be None) is used for probes only in
    this layer (transfers live in ``transfer.py``); ``clock`` is the cluster's :class:`ClusterClock`;
    ``caps_getter`` returns the discovery cache (or None before bootstrap).
    """

    def __init__(self, cluster_name: str, transport_login: SSHTransport, transport_transfer: SSHTransport | None,
                 clock: ClusterClock, caps_getter: CapsGetter | None = None) -> None:
        self.cluster = cluster_name
        self.login = transport_login
        self.transfer = transport_transfer
        self.clock = clock
        self._caps_getter = caps_getter
        self.profile: ClusterProfile = transport_login.profile
        self.last_probe_local: float | None = None

    # -- helpers ----------------------------------------------------------------------------------------

    @property
    def caps(self) -> dict[str, Any]:
        if self._caps_getter is None:
            return {}
        try:
            return dict(self._caps_getter() or {})
        except Exception:  # a broken cache must never stop a command
            return {}

    @property
    def cmd_timeout_s(self) -> float:
        return self.login.default_timeout()

    def control_root(self) -> str:
        return profile_control_root(self.profile)

    def home(self) -> str | None:
        return self.caps.get("home") or None

    def expand_path(self, path: str) -> str:
        """Expand ``$HOME``/``${HOME}`` with the discovered home (SFTP has no shell); other paths unchanged."""
        home = self.home()
        if home and ("$HOME" in path or "${HOME}" in path):
            return path.replace("${HOME}", home).replace("$HOME", home)
        return path

    def helper_bin_dir(self, sha8: str | None = None) -> str:
        """``<control_root>/bin/<sha8>`` (section 7); ``sha8`` defaults to the deployed one, else the packaged."""
        sha = sha8 or self.caps.get("helper_sha8") or bundle_sha8()
        return f"{self.control_root().rstrip('/')}/bin/{sha}"

    async def run(self, command: str, *, timeout: float | None = None, input: str | None = None,
                  idempotent: bool = True, login_shell: bool = True) -> CommandResult:
        """Raw exec on the login host (``run_command`` and the CLI use this)."""
        res = await self.login.run(command, timeout=timeout, input=input, idempotent=idempotent,
                                   login_shell=login_shell)
        self.last_probe_local = time.time()
        return res

    # -- file I/O routing (design section 2.3) ------------------------------------------------------------
    #
    # Not every host serves both channels. Measured 2026-09-02: the Bridges-2 LOGIN node refuses the SFTP
    # subsystem (ChannelOpenError "Session request failed") while exec works; its DTN refuses every exec
    # command ("Login denied: ... is not an allowed command") while SFTP works over the same /ocean and /jet
    # filesystems. TRACE serves both on its login node. So file operations pick, in order: the login SFTP,
    # the transfer host's SFTP (same filesystems), then exec-channel primitives.

    async def file_sftp(self) -> tuple[asyncssh.SFTPClient | None, str]:
        """``(sftp_client, role)`` for small-file work, or ``(None, "exec")`` when no host serves SFTP."""
        caps = self.caps
        if caps.get("login_sftp_ok", True):
            try:
                return await self.login.sftp(), "login"
            except (asyncssh.ChannelOpenError, asyncssh.SFTPError, OSError) as e:
                log.info("%s: login SFTP unavailable (%s); trying the transfer host", self.cluster, e)
        transfer_caps = dict(caps.get("transfer") or {})
        if self.transfer is not None and transfer_caps.get("sftp_ok", True) is not False:
            try:
                return await self.transfer.sftp(), "transfer"
            except (asyncssh.ChannelOpenError, asyncssh.SFTPError, OSError) as e:
                log.info("%s: transfer SFTP unavailable (%s); falling back to exec", self.cluster, e)
        return None, "exec"

    async def _exec_write(self, path: str, data: bytes, *, mode: str = "overwrite", mkdirs: bool = True,
                          executable: bool = False) -> None:
        """Atomic write over an exec channel: ``mkdir -p`` + ``cat > tmp`` + ``mv -f`` (append: ``cat >>``)."""
        target = shlex.quote(self.expand_path(path))
        text = data.decode("utf-8")
        parent = posixpath.dirname(self.expand_path(path).rstrip("/"))
        prefix = f"mkdir -p {shlex.quote(parent)} && " if (mkdirs and parent) else ""
        if mode == "append":
            cmd = f"{prefix}cat >> {target}"
            if executable:
                cmd += f" && chmod 755 {target}"
        else:
            tmp = shlex.quote(f"{self.expand_path(path)}.tmp-{secrets.token_hex(4)}")
            chmod = f"chmod 755 {tmp} && " if executable else ""
            cmd = f"{prefix}cat > {tmp} && {chmod}mv -f {tmp} {target}"
        res = await self.login.run(cmd, input=text, idempotent=True)
        if res.returncode != 0:
            raise err("E_PERMISSION", f"cannot write {path} on {self.cluster}: {res.stderr.strip()[:300]}",
                      fix="check the path and quota")

    async def _exec_mkdirs(self, paths: Iterable[str]) -> None:
        quoted = " ".join(shlex.quote(self.expand_path(p)) for p in paths if p)
        if not quoted:
            return
        res = await self.login.run(f"mkdir -p {quoted}", idempotent=True)
        if res.returncode != 0:
            raise err("E_PERMISSION", f"mkdir failed on {self.cluster}: {res.stderr.strip()[:300]}",
                      fix="check the path and quota")

    async def _exec_ls(self, path: str, glob: str | None, max_entries: int, sort: str) -> dict[str, Any]:
        """``find -maxdepth 1 -printf`` listing; the type letter, size, mtime and name are pipe separated."""
        target = self.expand_path(path)
        q = shlex.quote(target)
        cmd = (f"if [ -d {q} ]; then find {q} -maxdepth 1 -mindepth 1 -printf '%y|%s|%T@|%f\\n'; "
               f"elif [ -e {q} ]; then stat -c '%F|%s|%Y|%n' {q}; else echo '::MISSING'; fi")
        res = await self.login.run(cmd, idempotent=True)
        if "::MISSING" in res.stdout or res.returncode != 0:
            raise err("E_INVALID_SPEC", f"cannot stat {path} on {self.cluster}",
                      fix="check the path (absolute, on the cluster)")
        entries: list[dict[str, Any]] = []
        for line in res.stdout.splitlines():
            f = line.split("|")
            if len(f) < 4:
                continue
            kind, size, mtime, name = f[0], f[1], f[2], "|".join(f[3:])
            name = posixpath.basename(name.strip())
            if glob and not fnmatch.fnmatchcase(name, glob):
                continue
            typ = {"d": "dir", "f": "file", "l": "link"}.get(kind.strip(), "other")
            if kind.startswith("directory"):
                typ = "dir"
            elif kind.startswith("regular"):
                typ = "file"
            try:
                mt = int(float(mtime))
            except ValueError:
                mt = None
            entries.append({"name": name, "type": typ, "size": int(size) if size.isdigit() else None,
                            "mtime_ts": mt})
        if sort == "mtime":
            entries.sort(key=lambda e: (-(e["mtime_ts"] or 0), e["name"]))
        elif sort == "size":
            entries.sort(key=lambda e: (-(e["size"] or 0), e["name"]))
        else:
            entries.sort(key=lambda e: e["name"])
        return {"path": path, "entries": entries[: int(max_entries)], "truncated": len(entries) > int(max_entries)}

    def _to_epoch(self, value: Any) -> int | None:
        return self.clock.to_epoch(value)

    # -- 6.1 discovery ------------------------------------------------------------------------------------

    async def discover(self) -> dict[str, Any]:
        """Run the bootstrap probe (section 6.1) and parse it into the caps dict cached as ``kv.caps.<cluster>``.

        Keys (all present, None/empty when unknown): ``cluster, home, user, hostname, group, project, scratch,
        local, remote_now, tz_offset_s, slurm_version, cluster_name, epoch_format, cmd_timeout_s,
        message_timeout_s, min_job_age_s, kill_wait_s, job_requeue, preempt_mode, preempt_parameters,
        scheduler_parameters, comment_stored, enforce_part_limits, config`` (parse_config minus nothing),
        ``partitions`` (merged TRES + sinfo gres, plus ``limits``/``qos_candidates``/``charge``/``accessible`` from
        :func:`slurm_mcp.slurm.discovery.enrich_caps`), ``sinfo`` (per-partition aggregate), ``default_account,
        assoc, assocs, qos, qos_candidates, qos_for_partition`` (validated cache, kept across refreshes by the
        caller), ``sshare, su_balance, balance, charges, reservations, tools, squeue_O_zero, df, helper_sha8,
        pending_cap, pending_cap_part, rc`` and ``fetched_local`` (set by the caller).
        """
        profile = self.profile
        res = await self.run(commands.discovery(profile), timeout=max(self.cmd_timeout_s, 180))
        sections = P.parse_sections(res.stdout)
        caps = parse_discovery(sections, profile, cluster=self.cluster)
        caps["probe_rc"] = res.returncode
        # section 6.0 rule 5: the cluster's own clock
        self.clock.epoch_format = bool(caps.get("epoch_format", True))
        if caps.get("tz_offset_s") is not None:
            self.clock.tz_offset_s = int(caps["tz_offset_s"])
        if caps.get("remote_now") is not None:
            self.clock.update_from_remote(int(caps["remote_now"]))
        from .discovery import enrich_caps  # deferred: discovery.py imports this module
        return enrich_caps(caps, profile)

    async def helper_version(self, control_root: str | None = None) -> str | None:
        """The deployed helper sha8 (``<control_root>/bin/VERSION``) or None."""
        res = await self.run(commands.helper_deploy_check(control_root or self.control_root()))
        text = res.stdout.strip()
        return text.splitlines()[0].strip() if res.ok and text else None

    async def deploy_helpers(self, control_root: str | None = None) -> str:
        """SFTP ``wrap.sh``/``submit.sh``/``alloc-agent.sh`` to ``<control_root>/bin/<sha8>/``, chmod 755 and
        write ``VERSION`` (section 6.1 "Helper deploy", section 7). Content-addressed: never overwrites a running
        job's directory with different content. Returns the sha8."""
        root = self.expand_path((control_root or self.control_root()).rstrip("/"))
        sha8 = bundle_sha8()
        bin_dir = f"{root}/bin/{sha8}"
        sftp, role = await self.file_sftp()
        if sftp is None:                      # section 2.3: no SFTP anywhere -> exec channel
            await self._exec_mkdirs([bin_dir])
            for name, data in bundle_files().items():
                await self._exec_write(f"{bin_dir}/{name}", data, executable=True)
            await self._exec_write(f"{root}/bin/VERSION", (sha8 + "\n").encode("ascii"))
            return sha8
        await sftp.makedirs(bin_dir, exist_ok=True)
        for name, data in bundle_files().items():
            await self._sftp_write(sftp, f"{bin_dir}/{name}", data, executable=True)
        await self._sftp_write(sftp, f"{root}/bin/VERSION", (sha8 + "\n").encode("ascii"))
        return sha8

    async def backfill_history(self, user: str | None = None) -> list[dict[str, Any]]:
        """30-day wait-history rows (section 6.1 last bullet); ``[]`` when sacct fails. Timestamps as epoch."""
        res = await self.run(commands.backfill_history(user), timeout=max(self.cmd_timeout_s, 180))
        if not res.ok:
            log.warning("%s: backfill sacct rc=%s: %s", self.cluster, res.returncode, res.stderr.strip()[:200])
            return []
        rows = P.parse_backfill([l for l in res.stdout.splitlines() if l.strip()])
        for r in rows:
            r["submit_ts"] = self._to_epoch(r.get("submit"))
            r["start_ts"] = self._to_epoch(r.get("start"))
        return rows

    # -- 6.2 tick / snapshot ------------------------------------------------------------------------------

    async def tick(self, ids: Sequence[object], ctrl_dirs: Sequence[str] = (), rc_paths: Sequence[str] = (),
                   recover: bool = False, enrich_ids: Sequence[object] = (), stdout_paths: Sequence[str] = (),
                   ) -> dict[str, Any]:
        """One monitor tick (section 6.2): run the composite exec(s), parse every section, update the clock.

        Raises :class:`IncompleteProbe` (no ``::END``) or :class:`TickFailed` (non-zero ``::RC`` after
        ``::SQUEUE``/``::SACCT``); the Monitor treats both as a failed tick and changes nothing. Returns
        ``{now, host, squeue: [rows], restarts: {id: {...}}, sacct: {id: {rows, current, incarnations, steps}},
        files: {ctrl_dir: {file: content}}, cmds: {path: rc}, recover: [rows], enrich: {jobs, last_lines},
        rc: {section: rc}, healthy: bool}`` with ``start/end/submit`` fields converted to cluster epoch.
        """
        caps = self.caps
        merged: dict[str, tuple[list[str], int | None]] = {}
        for exec_text in commands.tick(list(ids), list(ctrl_dirs), list(rc_paths), recover, list(enrich_ids),
                                       list(stdout_paths), caps=caps):
            res = await self.run(exec_text, timeout=self.cmd_timeout_s)
            part = P.parse_sections(res.stdout)
            for name, (lines, rc) in part.items():
                if name in merged:
                    old_lines, old_rc = merged[name]
                    merged[name] = (old_lines + lines, old_rc if old_rc else rc)
                else:
                    merged[name] = (lines, rc)
        for name in ("SQUEUE", "SACCT", "RESTARTS"):
            rc = _rc(merged, name)
            if rc not in (None, 0):
                raise TickFailed(f"::{name} returned rc {rc}")
        now = _now_fallback(merged)
        if now is not None:
            self.clock.update_from_remote(now[0])
        squeue = P.parse_squeue_tick(_sec(merged, "SQUEUE"))
        for row in squeue:
            for k in ("start", "end", "submit"):
                row[k + "_ts"] = self._to_epoch(row.get(k))
        sacct = P.parse_sacct_tick(_sec(merged, "SACCT"))
        for group in sacct.values():
            for row in group["rows"] + group["steps"]:
                for k in ("start", "end", "submit"):
                    row[k + "_ts"] = self._to_epoch(row.get(k))
        recover_rows = P.parse_recover(_sec(merged, "RECOVER")) if "RECOVER" in merged else []
        for row in recover_rows:
            row["submit_ts"] = self._to_epoch(row.get("submit"))
        files = P.parse_files(_sec(merged, "FILES"))
        for entry in files.values():                       # ``tr`` turned the newline of jobid/heartbeat into a space
            for k, v in list(entry.items()):
                if isinstance(v, str):
                    entry[k] = v.strip()
        rcs = {name: merged[name][1] for name in merged if name not in ("_pre", "NOW")}
        return {
            "now": now[0] if now else None,
            "host": now[1] if now else None,
            "squeue": squeue,
            "restarts": P.parse_restarts(_sec(merged, "RESTARTS")),
            "sacct": sacct,
            "files": files,
            "cmds": P.parse_cmds(_sec(merged, "CMDS")),
            "recover": recover_rows,
            "recover_rc": _rc(merged, "RECOVER"),
            "enrich": P.parse_enrich(_sec(merged, "ENRICH")) if "ENRICH" in merged else {"jobs": {}, "last_lines": {}},
            "rc": rcs,
            "healthy": all(v in (None, 0) for k, v in rcs.items() if k in ("SQUEUE", "SACCT", "RECOVER")),
            "tick_local": time.time(),
        }

    async def snapshot(self) -> dict[str, Any]:
        """The cluster load snapshot of section 6.2: ``{nodes, pd, r, mine, resv, rc, ts, partitions}`` where
        ``partitions`` is :func:`summarize_snapshot` and ``ts`` the cluster epoch of the probe."""
        caps = self.caps
        res = await self.run(commands.snapshot(caps), timeout=self.cmd_timeout_s)
        sections = P.parse_sections(res.stdout)
        snap = P.parse_snapshot(sections)
        for row in snap["mine"]:
            row["start_ts"] = self._to_epoch(row.get("start"))
        for r in snap["resv"]:
            r["start_ts"] = self._to_epoch(r.get("start"))
            r["end_ts"] = self._to_epoch(r.get("end"))
        snap["ts"] = self.clock.remote_now()
        snap["fetched_local"] = time.time()
        snap["partitions"] = summarize_snapshot(snap, caps)
        return snap

    async def recheck_pending(self) -> dict[int, str] | None:
        """``squeue --me -h -o '%A|%T'`` -> ``{slurm_id: STATE}``; None when rc != 0 (unknown, section 5.4)."""
        try:
            res = await self.run(commands.recheck_pending(), timeout=self.cmd_timeout_s)
        except (CommandTimeout, ConnectionDropped):
            return None
        if not res.ok:
            return None
        out: dict[int, str] = {}
        for line in res.stdout.splitlines():
            sid, sep, state = line.strip().partition("|")
            if sep and sid.isdigit():
                out[int(sid)] = state.strip()
        return out

    # -- 6.3 submit / estimate ----------------------------------------------------------------------------

    async def submit(self, workdir: str, ctrl_dir: str, token: str, args: Sequence[object],
                     script: str | None = None, *, bin_dir: str | None = None) -> dict[str, Any]:
        """``submit.sh`` through the helper (sections 6.3, 7.2), ``idempotent=False``.

        Returns :func:`parse.parse_submit_output`: ``{"status": "ok", job_id, cluster, stderr}``,
        ``{"status": "err", rc, code, stderr}`` or ``{"status": "ambiguous", raw[, error]}``. A timeout or a
        dropped channel is ambiguous (the attempt stays ``UNCONFIRMED``; section 5.1 step 7), never a failure.
        """
        cmd = commands.submit(workdir, bin_dir or self.helper_bin_dir(), ctrl_dir, token, list(args), script)
        try:
            res = await self.run(cmd, timeout=self.cmd_timeout_s, idempotent=False)
        except CommandTimeout as e:
            return {"status": "ambiguous", "raw": e.stdout, "error": f"timeout after {e.timeout}s"}
        except ConnectionDropped as e:
            return {"status": "ambiguous", "raw": "", "error": f"connection dropped: {e.reason}"}
        out = P.parse_submit_output(res.stdout)
        out["rc_exec"] = res.returncode
        if res.returncode != 0 and out["status"] == "ambiguous":
            out["error"] = res.stderr.strip()[:500]
        return out

    async def test_only(self, workdir: str, args: Sequence[object], script: str, section: str = "T1",
                        ) -> dict[str, Any]:
        """``sbatch --test-only`` for one target (section 6.3 "Estimate"): the parsed estimate with
        ``est_start_ts`` (cluster epoch) or ``{"ok": False, reason, code, details}``; a timed-out pass yields
        ``{"ok": False, "timed_out": True}`` so the target is reported with ``est_wait_src="none"``."""
        cmd = commands.test_only(workdir, list(args), script, section)
        try:
            res = await self.run(cmd, timeout=self.cmd_timeout_s)
        except CommandTimeout as e:
            return {"ok": False, "reason": f"test-only timed out after {e.timeout}s", "code": None, "details": [],
                    "timed_out": True}
        except ConnectionDropped as e:
            return {"ok": False, "reason": f"connection dropped: {e.reason}", "code": None, "details": [],
                    "timed_out": True}
        try:
            sections = P.parse_sections(res.stdout)
        except IncompleteProbe:
            return {"ok": False, "reason": "incomplete test-only output", "code": None, "details": [],
                    "timed_out": True}
        lines, rc = sections.get(section, ([], None))
        out = P.parse_test_only("\n".join(lines) + ("\n" + res.stderr if res.stderr else ""), rc)
        if out.get("ok"):
            out["est_start_ts"] = self._to_epoch(out.get("est_start"))
        return out

    # -- 6.3 control ---------------------------------------------------------------------------------------

    async def _control(self, cmd: str, *, idempotent: bool = True) -> dict[str, Any]:
        res = await self.run(cmd, timeout=self.cmd_timeout_s, idempotent=idempotent)
        return {"rc": res.returncode, "stdout": res.stdout, "stderr": res.stderr, "ok": res.ok}

    async def cancel(self, ids: Sequence[object], signal: str | None = None, full: bool = False,
                     batch: bool = False) -> dict[str, Any]:
        """``scancel [--signal --full --batch] <ids>``; with a signal the call is non-idempotent (section 2.2).
        Adds ``errors`` (:func:`parse.scancel_errors`)."""
        out = await self._control(commands.cancel(list(ids), signal, full, batch), idempotent=signal is None)
        out["errors"] = P.scancel_errors(out["stderr"])
        return out

    async def hold(self, ids: Sequence[object]) -> dict[str, Any]:
        return await self._control(commands.hold(list(ids)))

    async def release(self, ids: Sequence[object]) -> dict[str, Any]:
        return await self._control(commands.release(list(ids)))

    async def requeue(self, ids: Sequence[object]) -> dict[str, Any]:
        return await self._control(commands.requeue(list(ids)))

    async def update_dependency(self, job_id: object, deps: str | Sequence[str]) -> dict[str, Any]:
        """``scontrol update JobId=<id> Dependency=<list>`` (section 5.2 step 11)."""
        return await self._control(commands.update_dependency(job_id, deps))

    async def show_job(self, job_id: object) -> dict[str, Any] | None:
        """``scontrol -o show job <id>`` parsed; None when ``Invalid job id specified`` (not in controller memory)."""
        res = await self.run(commands.show_job(job_id), timeout=self.cmd_timeout_s)
        if P.scontrol_job_missing(res.stderr) or P.scontrol_job_missing(res.stdout):
            return None
        if not res.ok:
            raise err("E_SSH", f"scontrol show job {job_id} failed (rc {res.returncode}): {res.stderr.strip()[:300]}")
        info = P.parse_scontrol_job(res.stdout)
        if not info:
            return None
        for k in ("submit_time", "start_time", "end_time"):
            info[k + "_ts"] = self._to_epoch(info.get(k))
        return info

    # -- section 4 files ----------------------------------------------------------------------------------

    def read_command(self, path: str, *, tail_lines: int | None = 100, head_lines: int | None = None,
                     grep: str | None = None, offset: int | None = None, max_chars: int = DEFAULT_READ_CHARS) -> str:
        """The single ``remote_read`` exec (section 4): existence check, size, then one ``tail``/``head``/``grep``/
        ``tail -c +N | head -c M`` pipeline framed as ``::SIZE n`` / ``::TEXT`` / ``::END``."""
        q = commands.path_quote(path)
        limit = max(1, int(max_chars)) + 1        # one extra char detects truncation
        if offset is not None:
            body = f"tail -c +{int(offset) + 1} -- {q} | head -c {limit}"
        elif grep:
            body = f"grep -n -E -m {GREP_MAX_MATCHES} -- {commands.shell_quote(grep)} {q} | head -c {limit}"
        elif head_lines is not None:
            body = f"head -n {int(head_lines)} -- {q} | head -c {limit}"
        else:
            body = f"tail -n {int(tail_lines if tail_lines is not None else 100)} -- {q} | head -c {limit}"
        return (f"{commands.PREAMBLE}; f={q}; if [ ! -e \"$f\" ]; then echo '::MISSING'; echo '::END'; exit 0; fi; "
                f"if [ -d \"$f\" ]; then echo '::ISDIR'; echo '::END'; exit 0; fi; "
                f"echo \"::SIZE $(stat -c %s -- \"$f\" 2>/dev/null || echo 0)\"; echo '::TEXT'; {body}; "
                f"printf '\\n::END\\n'")

    async def read_file(self, path: str, *, tail_lines: int | None = 100, head_lines: int | None = None,
                        grep: str | None = None, offset: int | None = None, max_chars: int = DEFAULT_READ_CHARS,
                        ) -> dict[str, Any]:
        """``remote_read`` (section 4): ``{path, text, size, next_offset, truncated, mode}``.

        ``E_NO_LOG_YET``-style misses are reported as ``E_PERMISSION``-free ``SlurmMcpError(E_INVALID_SPEC)``
        with the path (the tools decide the code); a directory is refused likewise.
        """
        res = await self.run(self.read_command(path, tail_lines=tail_lines, head_lines=head_lines, grep=grep,
                                               offset=offset, max_chars=max_chars), timeout=self.cmd_timeout_s)
        text = res.stdout
        if "::MISSING" in text.split("\n", 2)[0:2]:
            raise err("E_INVALID_SPEC", f"no such file on {self.cluster}: {path}",
                      fix="check the path with remote_ls")
        if text.startswith("::ISDIR"):
            raise err("E_INVALID_SPEC", f"{path} is a directory on {self.cluster}", fix="use remote_ls for directories")
        size: int | None = None
        body = ""
        m = re.match(r"^::SIZE\s+(\d+)\s*\n::TEXT\n?", text)
        if m:
            size = int(m.group(1))
            body = text[m.end():]
        else:
            body = text
        if body.endswith("\n::END\n"):
            body = body[: -len("\n::END\n")]
        elif body.endswith("::END\n"):
            body = body[: -len("::END\n")]
        elif body.endswith("::END"):
            body = body[: -len("::END")]
        truncated = len(body.encode("utf-8", "replace")) > int(max_chars)
        if truncated:
            body = body.encode("utf-8", "replace")[: int(max_chars)].decode("utf-8", "replace")
        next_offset: int | None = None
        if offset is not None:
            consumed = int(offset) + len(body.encode("utf-8", "replace"))
            if size is None or consumed < size or truncated:
                next_offset = consumed
        mode = "offset" if offset is not None else "grep" if grep else "head" if head_lines is not None else "tail"
        if mode != "offset" and truncated and size is not None:
            next_offset = max(0, size - int(max_chars)) if mode == "tail" else int(max_chars)
        return {"path": path, "text": body, "size": size, "next_offset": next_offset, "truncated": truncated,
                "mode": mode, "rc": res.returncode}

    async def _sftp_write(self, sftp: asyncssh.SFTPClient, path: str, data: bytes, *, executable: bool = False,
                          ) -> None:
        """``<path>.tmp-<8hex>`` + ``posix_rename`` (section 4 remote_write); chmod 755 for executables."""
        tmp = f"{path}.tmp-{secrets.token_hex(4)}"
        async with sftp.open(tmp, "wb") as fh:
            await fh.write(data)
        if executable:
            await sftp.chmod(tmp, 0o755)
        await sftp.posix_rename(tmp, path)

    async def write_file(self, path: str, text: str, mode: str = "overwrite", *, mkdirs: bool = True,
                         executable: bool = False) -> dict[str, Any]:
        """``remote_write`` (section 4): normalised text (CRLF/BOM/NUL rules of section 5.5) written atomically;
        ``append`` opens the file in ``a`` mode. Refuses more than 1 MB. Returns ``{path, bytes, warnings}``."""
        if mode not in ("overwrite", "append"):
            raise err("E_INVALID_SPEC", f"mode must be 'overwrite' or 'append', got {mode!r}")
        clean, warnings = normalize_text(text)
        data = clean.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise err("E_TOO_MANY_BYTES", f"remote_write limit is 1 MB; got {len(data)} bytes",
                      fix="upload() the file instead")
        target = self.expand_path(path)
        sftp, role = await self.file_sftp()
        if sftp is None:                      # section 2.3: no SFTP anywhere -> exec channel
            await self._exec_write(path, data, mode=mode, mkdirs=mkdirs, executable=executable)
            return {"path": path, "bytes": len(data), "warnings": warnings}
        parent = posixpath.dirname(target.rstrip("/"))
        if mkdirs and parent:
            await sftp.makedirs(parent, exist_ok=True)
        if mode == "append":
            async with sftp.open(target, "ab") as fh:
                await fh.write(data)
            if executable:
                await sftp.chmod(target, 0o755)
        else:
            await self._sftp_write(sftp, target, data, executable=executable)
        return {"path": path, "bytes": len(data), "warnings": warnings}

    async def ls(self, path: str, glob: str | None = None, max_entries: int = 200, sort: str = "name",
                 ) -> dict[str, Any]:
        """``remote_ls`` (section 4): ``{path, entries: [{name, type, size, mtime_ts}], truncated}``.

        SFTP where a host serves it, otherwise a ``find``/``stat`` listing over an exec channel (section 2.3).
        """
        target = self.expand_path(path)
        sftp, role = await self.file_sftp()
        if sftp is None:
            return await self._exec_ls(path, glob, max_entries, sort)
        try:
            st = await sftp.stat(target)
        except asyncssh.SFTPError as e:
            raise err("E_INVALID_SPEC", f"cannot stat {path} on {self.cluster}: {e}",
                      fix="check the path (absolute, on the cluster)") from e
        entries: list[dict[str, Any]] = []
        if not _is_dir(st):
            entries.append(_entry(posixpath.basename(target.rstrip("/")) or target, st))
        else:
            async for item in sftp.scandir(target):
                name = item.filename if isinstance(item.filename, str) else item.filename.decode("utf-8", "replace")
                if name in (".", ".."):
                    continue
                if glob and not fnmatch.fnmatchcase(name, glob):
                    continue
                entries.append(_entry(name, item.attrs))
        if sort == "mtime":
            entries.sort(key=lambda e: (-(e["mtime_ts"] or 0), e["name"]))
        elif sort == "size":
            entries.sort(key=lambda e: (-(e["size"] or 0), e["name"]))
        else:
            entries.sort(key=lambda e: e["name"])
        truncated = len(entries) > int(max_entries)
        return {"path": path, "entries": entries[: int(max_entries)], "truncated": truncated}

    async def mkdirs(self, paths: Iterable[str]) -> None:
        """``makedirs`` (exist_ok) for every path; SFTP where available, else ``mkdir -p`` (sections 5.1.6, 2.3)."""
        sftp, role = await self.file_sftp()
        if sftp is None:
            await self._exec_mkdirs(paths)
            return
        for p in paths:
            if p:
                await sftp.makedirs(self.expand_path(p), exist_ok=True)

    async def close(self) -> None:
        await self.login.close()
        if self.transfer is not None:
            await self.transfer.close()


def _is_dir(attrs: asyncssh.SFTPAttrs) -> bool:
    if getattr(attrs, "type", None) == asyncssh.FILEXFER_TYPE_DIRECTORY:
        return True
    perms = getattr(attrs, "permissions", None)
    return bool(perms is not None and statmod.S_ISDIR(perms))


def _entry(name: str, attrs: asyncssh.SFTPAttrs) -> dict[str, Any]:
    t = getattr(attrs, "type", None)
    perms = getattr(attrs, "permissions", None)
    if t == asyncssh.FILEXFER_TYPE_DIRECTORY or (perms is not None and statmod.S_ISDIR(perms)):
        kind = "dir"
    elif t == asyncssh.FILEXFER_TYPE_SYMLINK or (perms is not None and statmod.S_ISLNK(perms)):
        kind = "link"
    elif t == asyncssh.FILEXFER_TYPE_REGULAR or (perms is not None and statmod.S_ISREG(perms)):
        kind = "file"
    else:
        kind = "other"
    mtime = getattr(attrs, "mtime", None)
    return {"name": name, "type": kind, "size": getattr(attrs, "size", None),
            "mtime_ts": int(mtime) if mtime is not None else None}


# --- 6.1 pure parse of the discovery probe --------------------------------------------------------------

def parse_discovery(sections: Mapping[str, tuple[list[str], int | None]], profile: ClusterProfile | Any,
                    *, cluster: str | None = None) -> dict[str, Any]:
    """Pure section-6.1 parse of a framed discovery probe into the raw caps dict (see :meth:`SlurmClient.discover`).

    ``profile`` provides ``remote_root``/``control_root``/``quota_paths``/``balance_regex``/``cmd_timeout_s``/
    ``su_rates``; ``enrich_caps`` (discovery.py) adds the derived per-partition fields.
    """
    env = P.parse_env(_sec(sections, "ENV"))
    config = P.parse_config(_sec(sections, "CONFIG"))
    version = P.parse_version(_sec(sections, "VERSION")) or config.get("slurm_version") or env.get("slurm_version")
    partitions = P.parse_partitions(_sec(sections, "PARTITIONS"))
    sinfo_rows = P.parse_sinfo_nodes(_sec(sections, "SINFO"))
    sinfo_agg = P.aggregate_sinfo(sinfo_rows)
    partitions = P.merge_partition_gres(partitions, sinfo_agg)
    user = P.parse_user(_sec(sections, "USER"))
    assocs = P.parse_assoc(_sec(sections, "ASSOC"))
    qos = P.parse_qos(_sec(sections, "QOS"))
    sshare = P.parse_sshare(_sec(sections, "SSHARE"))
    balance = P.parse_balance(_sec(sections, "BALANCE"), getattr(profile, "balance_regex", None))
    reservations = P.parse_reservations(_sec(sections, "RESV"))
    tools = P.parse_tools(_sec(sections, "TOOLS"))
    cap_o = P.parse_cap_o(_sec(sections, "CAP_O"))
    helper_lines = [l.strip() for l in _sec(sections, "HELPER") if l.strip()]
    helper_sha8 = helper_lines[0] if helper_lines and re.fullmatch(r"[0-9a-f]{8}", helper_lines[0]) else None

    home = env.get("home") or None
    remote_root = getattr(profile, "remote_root", None)
    croot = profile_control_root(profile) if isinstance(profile, ClusterProfile) else commands.effective_control_root(profile)
    if home:
        croot = croot.replace("${HOME}", home).replace("$HOME", home)
    roles: dict[str, str] = {}
    for p in getattr(profile, "quota_paths", None) or []:
        roles[str(p)] = "upload_root"
    if env.get("project"):
        roles[str(env["project"])] = "project"
    if croot:
        roles[croot] = "control_root"
    if remote_root:
        roles[str(remote_root)] = "remote_root"
    if home:
        roles[home] = "home"
    df = P.parse_df(_sec(sections, "DF"), roles)
    preference = ("home", "remote_root", "project", "control_root", "upload_root")
    by_role = {r: p for p, r in roles.items()}
    for row in df:
        if row.get("role") is None:
            # captures without the appended path report the mount point: match the role paths by prefix
            mount = str(row.get("mount") or "").rstrip("/")
            for role in preference:
                p = by_role.get(role)
                if p and mount and (p == mount or p.startswith(mount + "/")):
                    row["role"] = role
                    break
            else:
                row["role"] = "group"

    default_account = user.get("default_account") or getattr(profile, "default_account", None)
    assoc = None
    for row in assocs:
        if row.get("account") == default_account and row.get("partition") is None:
            assoc = row
            break
    if assoc is None and assocs:
        assoc = next((r for r in assocs if r.get("partition") is None), assocs[0])

    su_balance = next((r["su_balance"] for r in sshare if r.get("su_balance") is not None), None)
    if su_balance is None and balance and balance.get("left") is not None:
        su_balance = float(balance["left"])

    sched = config.get("scheduler_parameters") or {}
    cap_user = sched.get("bf_max_job_user")
    cap_part = sched.get("bf_max_job_user_part")
    profile_timeout = getattr(profile, "cmd_timeout_s", None)
    cmd_timeout = int(profile_timeout) if profile_timeout else int(config.get("cmd_timeout_s") or 120)
    charges = any(p.get("tres_billing_weights") for p in partitions.values()) or bool(getattr(profile, "su_rates", None))

    return {
        "cluster": cluster or getattr(profile, "name", None),
        "home": home, "user": env.get("user") or None, "hostname": env.get("hostname"),
        "group": env.get("group"), "groups": env.get("groups"), "project": env.get("project"),
        "scratch": env.get("scratch"),
        "local": env.get("local"), "remote_now": env.get("remote_now"), "tz_offset_s": env.get("tz_offset_s"),
        "env": env,
        "slurm_version": version, "cluster_name": config.get("cluster_name"),
        "epoch_format": bool(config.get("epoch_format")) if _sec(sections, "CONFIG") else True,
        "cmd_timeout_s": cmd_timeout,
        "message_timeout_s": config.get("message_timeout_s"), "min_job_age_s": config.get("min_job_age_s"),
        "kill_wait_s": config.get("kill_wait_s"), "job_requeue": config.get("job_requeue"),
        "preempt_mode": config.get("preempt_mode") or [], "preempt_type": config.get("preempt_type"),
        "preempt_parameters": config.get("preempt_parameters") or [],
        "scheduler_parameters": sched, "comment_stored": bool(config.get("comment_stored")),
        "enforce_part_limits": config.get("enforce_part_limits"), "mail_prog": config.get("mail_prog"),
        "config": config,
        "partitions": partitions, "sinfo": sinfo_agg,
        "default_account": default_account, "assoc": assoc, "assocs": assocs,
        "qos": qos, "qos_candidates": {}, "qos_for_partition": {},
        "sshare": sshare, "su_balance": su_balance, "balance": balance, "charges": charges,
        "reservations": reservations, "tools": tools, "squeue_O_zero": cap_o, "df": df,
        "helper_sha8": helper_sha8, "control_root": croot,
        "pending_cap": (int(cap_user) - 2) if isinstance(cap_user, int) else None,
        "pending_cap_part": (int(cap_part) - 2) if isinstance(cap_part, int) else None,
        "rc": {name: sections[name][1] for name in sections if name not in ("_pre",)},
    }


__all__ = ["SlurmClient", "TickFailed", "IncompleteProbe", "parse_discovery", "summarize_snapshot",
           "MAX_WRITE_BYTES", "DEFAULT_READ_CHARS", "GREP_MAX_MATCHES", "BUNDLE_FILES"]
