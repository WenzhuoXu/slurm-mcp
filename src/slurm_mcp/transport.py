"""SSH transport built on asyncssh (design section 2.2, failure rows of section 9.2).

One persistent connection per (cluster, role), multiplexed for exec channels and SFTP. Handles:
  - password / key / agent auth; the password is read from the OS keyring off-thread by a single-use
    async callable at auth time (never held on the transport)
  - host keys: ``~/.slurm-mcp/known_hosts`` (OpenSSH format, several lines per alias allowed) plus a JSON
    side file ``known_hosts.json`` = ``{alias: {addr: [fingerprint, ...]}}`` implementing the per-pool
    acceptance rule (new key from an unseen address -> accept + append + log; same address presenting a
    different key of the same type -> ``HostKeyChanged`` / E_HOSTKEY)
  - keepalives, reconnect with backoff, a single retry for idempotent commands, an ``auth_failed`` latch
  - ``asyncio.Semaphore(profile.ssh_max_exec)`` around exec channel opens (SFTP holds its slot outside it)
  - ``tcp_probe`` / ``banner_probe`` helpers used by discovery and the VPN hint
"""
from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Callable, Sequence

import asyncssh

from . import credentials
from .config import KNOWN_HOSTS_PATH, ClusterProfile
from .errors import SlurmMcpError, err

log = logging.getLogger("slurm_mcp.transport")

DEFAULT_CMD_TIMEOUT_S = 120
DEFAULT_CONNECT_TIMEOUT_S = 45.0
DEFAULT_LOGIN_TIMEOUT_S = 90
ROLES = ("login", "transfer")
# Substrings of ChannelOpenError reasons that mean "the connection is gone" (not MaxSessions).
_DROP_REASONS = ("connection closed", "connection lost", "not open")


# --- results and exceptions ---------------------------------------------------------------------------

@dataclass
class CommandResult:
    """Outcome of one remote command (design section 2.2)."""

    command: str
    returncode: int
    stdout: str
    stderr: str
    seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def raise_for_status(self) -> "CommandResult":
        if not self.ok:
            raise RemoteCommandError(self)
        return self


class RemoteCommandError(RuntimeError):
    """Raised by ``CommandResult.raise_for_status`` / ``run(check=True)`` on a non-zero exit status."""

    def __init__(self, result: CommandResult):
        self.result = result
        msg = (f"remote command failed (rc={result.returncode}): {result.command}\n"
               f"{result.stderr.strip()[-2000:]}")
        super().__init__(msg)


class HostKeyChanged(SlurmMcpError, RuntimeError):
    """E_HOSTKEY: an address already seen for this alias presented a different key of the same type."""

    def __init__(self, alias: str, addr: str, old_fps: Sequence[str], new_fp: str, cluster: str = "") -> None:
        self.alias, self.addr, self.old_fps, self.new_fp = alias, addr, list(old_fps), new_fp
        e = err("E_HOSTKEY", f"host key for {alias} from {addr} changed: old {', '.join(old_fps) or '?'} new {new_fp}",
                cluster=cluster or alias)
        SlurmMcpError.__init__(self, e.code, e.message, e.fix)


class AuthFailed(SlurmMcpError, RuntimeError):
    """E_AUTH: missing password or ``PermissionDenied``; sets the transport's ``auth_failed`` latch."""

    def __init__(self, cluster: str, message: str) -> None:
        e = err("E_AUTH", message, cluster=cluster)
        SlurmMcpError.__init__(self, e.code, e.message, e.fix)


class Unreachable(SlurmMcpError, ConnectionError):
    """E_UNREACHABLE: TCP/handshake failures exhausted the retries (or the VPN probe failed)."""

    def __init__(self, cluster: str, message: str, hint: str | None = None) -> None:
        e = err("E_UNREACHABLE", message, hint=hint or "VPN/DNS")
        SlurmMcpError.__init__(self, e.code, e.message, e.fix)


class CommandTimeout(asyncio.TimeoutError):
    """The per-command timeout expired. ``ambiguous`` is True for non-idempotent commands: the command may
    still have run to completion on the cluster (design section 2.2, section 5.1 step 7)."""

    def __init__(self, command: str, timeout: float | None, *, ambiguous: bool,
                 stdout: str = "", stderr: str = "") -> None:
        self.command, self.timeout, self.ambiguous = command, timeout, ambiguous
        self.stdout, self.stderr = stdout, stderr
        kind = "ambiguous, non-idempotent" if ambiguous else "idempotent"
        super().__init__(f"command timed out after {timeout}s ({kind}): {command[:200]}")


class ConnectionDropped(RuntimeError):
    """The SSH connection died while a command ran. Raised for non-idempotent commands (the caller decides,
    the outcome is ambiguous) and for idempotent ones whose single retry also failed."""

    def __init__(self, command: str, reason: str, *, ambiguous: bool) -> None:
        self.command, self.reason, self.ambiguous = command, reason, ambiguous
        super().__init__(f"connection dropped during command ({reason}): {command[:200]}")


# --- host-key store ----------------------------------------------------------------------------------

def known_hosts_prefix(alias: str, port: int | None) -> str:
    """OpenSSH known_hosts host field: ``alias`` on port 22, else ``[alias]:port``."""
    return alias if port in (None, 22) else f"[{alias}]:{port}"


def parse_known_hosts_line(line: str) -> tuple[str, asyncssh.SSHKey] | None:
    """``(host_field, key)`` for one known_hosts line, None for blanks/comments/unparseable lines."""
    s = line.strip()
    if not s or s.startswith(("#", "@")):
        return None
    parts = s.split(None, 2)
    if len(parts) < 3:
        return None
    try:
        key = asyncssh.import_public_key(f"{parts[1]} {parts[2].split()[0]}")
    except (asyncssh.KeyImportError, ValueError, IndexError):
        return None
    return parts[0], key


def format_known_hosts_line(alias: str, port: int | None, key: asyncssh.SSHKey) -> str:
    return f"{known_hosts_prefix(alias, port)} {key.export_public_key('openssh').decode().strip()}\n"


@dataclass
class HostKeyDecision:
    """What the policy decided for one (alias, addr, key)."""

    verdict: str                      # "known" | "new" | "changed"
    fingerprint: str
    old_fingerprints: list[str] = field(default_factory=list)


class HostKeyStore:
    """The two host-key files of design section 2.2.

    ``path`` is the OpenSSH-format known_hosts (default ``~/.slurm-mcp/known_hosts``); the side file
    ``<path>.json`` maps ``{alias: {addr: [fingerprint, ...]}}`` so the pool rule can tell a new round-robin
    member from a changed key. Never touches ``~/.ssh/known_hosts``.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else KNOWN_HOSTS_PATH
        self.meta_path = self.path.with_name(self.path.name + ".json")

    # -- raw file access ---------------------------------------------------------------------------
    def _ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def lines(self) -> list[str]:
        if not self.path.exists():
            return []
        return self.path.read_text(encoding="utf-8").splitlines()

    def load_meta(self) -> dict[str, dict[str, list[str]]]:
        if not self.meta_path.exists():
            return {}
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8") or "{}")
        except (ValueError, OSError):
            log.warning("host-key side file %s is unreadable; treating as empty", self.meta_path)
            return {}
        return {str(a): {str(ip): [str(f) for f in fps] for ip, fps in (m or {}).items()}
                for a, m in (data or {}).items()}

    def save_meta(self, meta: dict[str, dict[str, list[str]]]) -> None:
        self._ensure()
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.meta_path)

    # -- queries ------------------------------------------------------------------------------------
    def keys_for(self, alias: str, port: int | None = None) -> list[asyncssh.SSHKey]:
        """Every stored key whose host field matches ``alias`` on ``port`` (port 22 also matches bare lines)."""
        wanted = {known_hosts_prefix(alias, port)}
        if port in (None, 22):
            wanted.add(alias)
        out: list[asyncssh.SSHKey] = []
        for line in self.lines():
            parsed = parse_known_hosts_line(line)
            if parsed and parsed[0] in wanted:
                out.append(parsed[1])
        return out

    def seen_addrs(self, alias: str) -> dict[str, list[str]]:
        return self.load_meta().get(alias, {})

    def decide(self, alias: str, port: int | None, addr: str, key: asyncssh.SSHKey) -> HostKeyDecision:
        """Apply the per-pool rule (design section 2.2 / section 9.2 rows 2-3) without writing anything."""
        fp = key.get_fingerprint()
        known = {k.get_fingerprint(): k for k in self.keys_for(alias, port)}
        if fp in known:
            return HostKeyDecision("known", fp)
        seen_fps = self.seen_addrs(alias).get(addr, [])
        same_type = [f for f in seen_fps if f in known and known[f].algorithm == key.algorithm]
        if same_type:
            return HostKeyDecision("changed", fp, same_type)
        return HostKeyDecision("new", fp)

    # -- mutations ----------------------------------------------------------------------------------
    def record(self, alias: str, port: int | None, addr: str, key: asyncssh.SSHKey) -> str:
        """Append the key line (once) and remember ``addr -> fingerprint``; returns the fingerprint."""
        self._ensure()
        fp = key.get_fingerprint()
        if fp not in {k.get_fingerprint() for k in self.keys_for(alias, port)}:
            with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(format_known_hosts_line(alias, port, key))
        meta = self.load_meta()
        fps = meta.setdefault(alias, {}).setdefault(addr, [])
        if fp not in fps:
            fps.append(fp)
        self.save_meta(meta)
        return fp

    def forget(self, alias: str) -> int:
        """Drop every line and side-file entry for ``alias`` (any port); returns the number of lines removed."""
        prefixes = {alias}
        kept, removed = [], 0
        for line in self.lines():
            parsed = parse_known_hosts_line(line)
            host_field = parsed[0] if parsed else line.split(None, 1)[0] if line.strip() else ""
            if host_field in prefixes or (host_field.startswith(f"[{alias}]:")):
                removed += 1
                continue
            kept.append(line)
        if self.path.exists():
            self.path.write_text("".join(l + "\n" for l in kept), encoding="utf-8", newline="\n")
        meta = self.load_meta()
        if alias in meta:
            del meta[alias]
            self.save_meta(meta)
        return removed

    # -- asyncssh integration ------------------------------------------------------------------------
    def known_hosts_callable(self, alias: str, port: int | None) -> Callable[[str, str, int | None], tuple]:
        """The ``known_hosts=`` callable: returns the stored keys for the alias (asyncssh consults
        ``validate_host_public_key`` only for keys not in this list)."""

        def _lookup(host: str, addr: str, p: int | None) -> tuple:
            return (self.keys_for(alias, port), [], [])

        return _lookup


def forget_host_keys(alias: str, store: HostKeyStore | None = None) -> int:
    """``slurm-mcp hostkeys forget <cluster>``: clear both stores for an alias (design section 2.2)."""
    return (store or HostKeyStore()).forget(alias)


class HpcClient(asyncssh.SSHClient):
    """``SSHClient`` whose ``validate_host_public_key`` implements the per-pool acceptance rule."""

    def __init__(self, store: HostKeyStore, alias: str, port: int | None, *, trust_new: bool = True,
                 cluster: str = "") -> None:
        self.store, self.alias, self.port, self.trust_new, self.cluster = store, alias, port, trust_new, cluster
        self.changed: HostKeyChanged | None = None
        self.accepted: list[str] = []          # "new host key for <alias> from <addr> <fp>" notices
        self.refused: str | None = None

    def validate_host_public_key(self, host: str, addr: str, port: int, key: asyncssh.SSHKey) -> bool:
        decision = self.store.decide(self.alias, self.port, addr, key)
        if decision.verdict == "known":
            return True
        if decision.verdict == "changed":
            self.changed = HostKeyChanged(self.alias, addr, decision.old_fingerprints, decision.fingerprint,
                                          self.cluster)
            log.error("%s", self.changed)
            return False
        if not self.trust_new:
            self.refused = f"unknown host key for {self.alias} from {addr} {decision.fingerprint}"
            return False
        fp = self.store.record(self.alias, self.port, addr, key)
        notice = f"new host key for {self.alias} from {addr} {fp}"
        self.accepted.append(notice)
        log.warning("%s", notice)
        return True


# --- probes ------------------------------------------------------------------------------------------

async def tcp_probe(host: str, port: int, timeout: float = 3) -> bool:
    """True when a TCP connection to ``host:port`` succeeds within ``timeout`` seconds (no SSH, no auth)."""
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except (OSError, asyncio.TimeoutError):
        pass
    return True


async def banner_probe(host: str, port: int, timeout: float = 5) -> str:
    """The server's identification line (``SSH-2.0-...``) without the line ending; '' when nothing arrives.

    Used once per transfer host to pick ``transfer_port`` (design section 2.2 / 5.5). Servers may send
    pre-banner text lines; the first line starting with ``SSH-`` wins.
    """
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    except (OSError, asyncio.TimeoutError):
        return ""
    banner = ""
    try:
        deadline = time.monotonic() + timeout
        for _ in range(20):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            raw = await asyncio.wait_for(reader.readline(), remaining)
            if not raw:
                break
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("SSH-"):
                banner = line
                break
    except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, asyncio.TimeoutError):
            pass
    return banner


# --- the transport -------------------------------------------------------------------------------------

def _is_drop(exc: BaseException) -> bool:
    """True for the exceptions that mean the connection is gone (design section 2.2)."""
    if isinstance(exc, (asyncssh.ConnectionLost, asyncssh.DisconnectError)):
        return True
    if isinstance(exc, asyncssh.ChannelOpenError):
        reason = str(exc).lower()
        return any(s in reason for s in _DROP_REASONS)
    return False


class SSHTransport:
    """One SSH connection to one host of one cluster (design section 2.2).

    ``role`` is ``"login"`` (default) or ``"transfer"``; ``host``/``port`` default to the profile's login
    host/port. ``caps_cmd_timeout_s`` is an optional callable returning the discovered ``caps.cmd_timeout_s``
    (or None); see ``default_timeout``.
    """

    def __init__(self, profile: ClusterProfile, *, host: str | None = None, port: int | None = None,
                 role: str = "login", caps_cmd_timeout_s: Callable[[], int | None] | None = None,
                 connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_S, login_timeout: float = DEFAULT_LOGIN_TIMEOUT_S,
                 max_sessions: int | None = None, trust_new_hosts: bool = True,
                 store: HostKeyStore | None = None):
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        self.profile = profile
        self.role = role
        self.host = host or profile.host
        self.port = int(port if port is not None else profile.port)
        self.caps_cmd_timeout_s = caps_cmd_timeout_s
        self.connect_timeout = connect_timeout
        self.login_timeout = login_timeout
        self.max_sessions = int(max_sessions if max_sessions is not None else profile.ssh_max_exec)
        self.trust_new_hosts = trust_new_hosts
        self.store = store or HostKeyStore()
        self.auth_failed = False
        self.auth_error: str | None = None
        self.hostkey_notices: list[str] = []
        self.retry_backoff_s = 2.0           # first reconnect delay; doubles up to retry_backoff_max_s
        self.retry_backoff_max_s = 30.0
        self.reconnects = 0
        self._conn: asyncssh.SSHClientConnection | None = None
        self._sftp: asyncssh.SFTPClient | None = None
        self._sftp_conn: asyncssh.SSHClientConnection | None = None
        self._client: HpcClient | None = None
        self._sem = asyncio.Semaphore(self.max_sessions)
        self._lock = asyncio.Lock()

    # ---- auth -------------------------------------------------------------------------------------
    def reset_auth(self) -> None:
        """Clear the ``auth_failed`` latch (after ``slurm-mcp auth set`` + ``clusters(refresh=True)``)."""
        self.auth_failed = False
        self.auth_error = None

    async def _password(self) -> str:
        """Single-use password provider awaited by asyncssh; reads the keyring off the event loop."""
        pw = await asyncio.to_thread(credentials.get_password, self.profile)
        if pw is None:
            p = self.profile
            raise AuthFailed(p.name, f"no password stored for cluster {p.name!r} ({p.credential_id})")
        return pw

    # ---- connection lifecycle -----------------------------------------------------------------------
    def _connect_options(self) -> dict[str, Any]:
        """The asyncssh.connect keyword arguments of design section 2.2."""
        p = self.profile
        self._client = HpcClient(self.store, self.host, self.port, trust_new=self.trust_new_hosts, cluster=p.name)
        client = self._client
        opts: dict[str, Any] = dict(
            host=self.host,
            port=self.port,
            username=p.user,
            known_hosts=self.store.known_hosts_callable(self.host, self.port),
            client_factory=lambda: client,
            config=None,
            connect_timeout=self.connect_timeout,
            login_timeout=self.login_timeout,
            keepalive_interval=30,
            keepalive_count_max=4,
            compression_algs=None,
            encoding="utf-8",
            errors="replace",
        )
        if p.auth == "password":
            opts.update(password=self._password, client_keys=None, agent_path=None, gss_host=None,
                        preferred_auth=["keyboard-interactive", "password"])
        elif p.auth == "key":
            opts.update(client_keys=[p.key_path], agent_path=None, preferred_auth="publickey")
        elif p.auth == "agent":
            opts.update(preferred_auth="publickey")
        return opts

    async def connect(self) -> asyncssh.SSHClientConnection:
        """Open (or reuse) the connection once; auth and host-key failures are translated, never retried."""
        async with self._lock:
            if self._conn is not None and not self._conn.is_closed():
                return self._conn
            if self.auth_failed:
                raise AuthFailed(self.profile.name, self.auth_error or "authentication previously failed")
            self._sftp = None
            self._conn = None
            opts = self._connect_options()
            try:
                conn = await asyncssh.connect(**opts)
            except AuthFailed as e:
                self.auth_failed, self.auth_error = True, e.message
                raise
            except asyncssh.PermissionDenied as e:
                self.auth_failed = True
                self.auth_error = f"{self.host} rejected the credentials for {self.profile.user}: {e.reason}"
                raise AuthFailed(self.profile.name, self.auth_error) from e
            except asyncssh.HostKeyNotVerifiable as e:
                client = self._client
                if client is not None and client.changed is not None:
                    raise client.changed from e
                raise err("E_HOSTKEY", (client.refused if client and client.refused else str(e)),
                          cluster=self.profile.name) from e
            if self._client is not None:
                self.hostkey_notices.extend(self._client.accepted)
            self._conn = conn
            log.info("connected to %s:%d as %s (%s)", self.host, self.port, self.profile.user, self.role)
            return conn

    async def ensure_connected(self, retries: int = 4) -> asyncssh.SSHClientConnection:
        """``connect`` with exponential backoff on transport errors; E_UNREACHABLE after ``retries`` failures.

        With ``profile.requires_vpn_hint`` set, a ``tcp_probe`` runs first and a closed port fails fast.
        """
        if self._conn is not None and not self._conn.is_closed():
            return self._conn
        hint = self.profile.requires_vpn_hint
        if hint and not await tcp_probe(self.host, self.port):
            raise Unreachable(self.profile.name, f"{self.host}:{self.port} does not answer on TCP", hint)
        delay = self.retry_backoff_s
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return await self.connect()
            except SlurmMcpError:
                raise  # auth / host-key / unreachable: never retried here
            except (OSError, asyncssh.Error, asyncio.TimeoutError) as e:
                last = e
                if attempt == retries:
                    break
                log.warning("connect to %s:%d failed (%s); retry in %.0fs", self.host, self.port, e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.retry_backoff_max_s)
        reachable = await tcp_probe(self.host, self.port)
        raise Unreachable(self.profile.name,
                          f"could not connect to {self.host}:{self.port} "
                          f"({'TCP open, SSH failed' if reachable else 'TCP closed'}): {last}", hint) from last

    async def close(self) -> None:
        if self._sftp is not None:
            try:
                self._sftp.exit()
            except Exception:
                pass
            self._sftp = None
        if self._conn is not None:
            conn, self._conn = self._conn, None
            conn.close()
            try:
                await asyncio.wait_for(conn.wait_closed(), 5)
            except Exception:
                pass

    async def _reconnect(self, reason: str) -> None:
        log.warning("connection to %s dropped (%s); reconnecting", self.host, reason)
        self.reconnects += 1
        await self.close()

    async def __aenter__(self) -> "SSHTransport":
        await self.ensure_connected()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def connected(self) -> bool:
        return self._conn is not None and not self._conn.is_closed()

    # ---- timeouts ----------------------------------------------------------------------------------
    def default_timeout(self) -> float:
        """``profile.cmd_timeout_s`` if set, else ``max(120, caps)`` from the caps callback, else 120 s."""
        if self.profile.cmd_timeout_s:
            return float(self.profile.cmd_timeout_s)
        if self.caps_cmd_timeout_s is not None:
            try:
                v = self.caps_cmd_timeout_s()
            except Exception:  # a broken caps cache must not stop commands
                v = None
            if v:
                return float(max(DEFAULT_CMD_TIMEOUT_S, v))
        return float(DEFAULT_CMD_TIMEOUT_S)

    # ---- commands ------------------------------------------------------------------------------------
    @staticmethod
    def wrap(command: str, login_shell: bool) -> str:
        """``bash -lc <quoted>`` so modules/PATH match an interactive login (design section 6.0)."""
        return f"bash -lc {shlex.quote(command)}" if login_shell else command

    async def _exec(self, command: str, wrapped: str, *, timeout: float | None, idempotent: bool,
                    run_kwargs: dict[str, Any]) -> asyncssh.SSHCompletedProcess:
        """Run one exec channel with the drop/retry policy of design section 2.2.

        Reconnect + one retry only for ``idempotent=True`` on ``ConnectionLost`` / ``ChannelOpenError(connection
        closed)`` / ``exit_status is None``. A timeout raises ``CommandTimeout`` (ambiguous when not idempotent).
        """
        attempts = 2 if idempotent else 1
        for attempt in range(1, attempts + 1):
            conn = await self.ensure_connected()
            try:
                async with self._sem:
                    proc = await conn.run(wrapped, timeout=timeout, check=False, **run_kwargs)
            except asyncssh.TimeoutError as e:
                raise CommandTimeout(command, timeout, ambiguous=not idempotent,
                                     stdout=_as_text(e.stdout), stderr=_as_text(e.stderr)) from e
            except (asyncssh.Error, OSError) as e:
                if not _is_drop(e):
                    raise
                await self._reconnect(str(e))
                if attempt == attempts:
                    raise ConnectionDropped(command, str(e), ambiguous=not idempotent) from e
                continue
            if proc.exit_status is None:
                await self._reconnect("exit status missing")
                if attempt == attempts:
                    raise ConnectionDropped(command, "channel closed without exit status", ambiguous=not idempotent)
                continue
            return proc
        raise AssertionError("unreachable")  # pragma: no cover

    async def run(self, command: str, *, timeout: float | None = None, input: str | None = None,
                  idempotent: bool = True, login_shell: bool = True, check: bool = False) -> CommandResult:
        """Run a shell command and capture text output (design section 2.2).

        ``timeout=None`` resolves via ``default_timeout``. ``idempotent=False`` for ``sbatch``/``submit.sh``/
        ``scancel`` with a signal / ``tar -x``: no retry, drops raise ``ConnectionDropped`` and timeouts
        ``CommandTimeout(ambiguous=True)``.
        """
        timeout = self.default_timeout() if timeout is None else timeout
        wrapped = self.wrap(command, login_shell)
        t0 = time.monotonic()
        proc = await self._exec(command, wrapped, timeout=timeout, idempotent=idempotent,
                                run_kwargs=dict(input=input, encoding="utf-8", errors="replace"))
        res = CommandResult(command, int(proc.exit_status), _as_text(proc.stdout), _as_text(proc.stderr),
                            time.monotonic() - t0)
        if check:
            res.raise_for_status()
        return res

    async def run_with_stdin_file(self, command: str, path: str | PurePath, timeout: float | None = None, *,
                                  idempotent: bool = False, login_shell: bool = True) -> CommandResult:
        """Run ``command`` with the local file ``path`` streamed to its stdin as bytes (tar upload, 5.5)."""
        timeout = self.default_timeout() if timeout is None else timeout
        wrapped = self.wrap(command, login_shell)
        t0 = time.monotonic()
        proc = await self._exec(command, wrapped, timeout=timeout, idempotent=idempotent,
                                run_kwargs=dict(stdin=str(path), encoding=None))
        return CommandResult(command, int(proc.exit_status), _as_text(proc.stdout), _as_text(proc.stderr),
                             time.monotonic() - t0)

    async def run_to_file(self, command: str, path: str | PurePath, timeout: float | None = None, *,
                          idempotent: bool = True, login_shell: bool = True) -> CommandResult:
        """Run ``command`` with its binary stdout written to the local file ``path`` (tar download, 5.5).

        ``stdout`` of the result is empty (it went to the file); ``stderr`` is decoded text.
        """
        timeout = self.default_timeout() if timeout is None else timeout
        wrapped = self.wrap(command, login_shell)
        t0 = time.monotonic()
        proc = await self._exec(command, wrapped, timeout=timeout, idempotent=idempotent,
                                run_kwargs=dict(stdout=str(path), encoding=None))
        return CommandResult(command, int(proc.exit_status), "", _as_text(proc.stderr), time.monotonic() - t0)

    # ---- sftp ----------------------------------------------------------------------------------------
    async def sftp(self) -> asyncssh.SFTPClient:
        """The cached SFTP client (one channel, outside the exec semaphore), recreated after any reconnect."""
        conn = await self.ensure_connected()
        if self._sftp is not None and self._sftp_conn is not conn:
            self._sftp = None
        if self._sftp is None:
            self._sftp = await conn.start_sftp_client()
            self._sftp_conn = conn
        return self._sftp

    # ---- probes --------------------------------------------------------------------------------------
    async def tcp_probe(self, host: str | None = None, port: int | None = None, timeout: float = 3) -> bool:
        return await tcp_probe(host or self.host, port or self.port, timeout)

    async def banner_probe(self, host: str | None = None, port: int | None = None, timeout: float = 5) -> str:
        return await banner_probe(host or self.host, port or self.port, timeout)


def _as_text(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return str(data)


__all__ = ["CommandResult", "RemoteCommandError", "HostKeyChanged", "AuthFailed", "Unreachable", "CommandTimeout",
           "ConnectionDropped", "HostKeyStore", "HostKeyDecision", "HpcClient", "forget_host_keys",
           "known_hosts_prefix", "parse_known_hosts_line", "format_known_hosts_line", "tcp_probe", "banner_probe",
           "SSHTransport", "DEFAULT_CMD_TIMEOUT_S"]
