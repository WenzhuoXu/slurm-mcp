"""In-memory stand-ins for ``SSHTransport`` and asyncssh's SFTP client, plus a builder that frames the captured
fixtures of ``tests/fixtures/<cluster>`` as a section-6.1 discovery probe (unit tests for client/discovery)."""
from __future__ import annotations

import posixpath
import stat as statmod
import time
from pathlib import Path
from typing import Any, Callable

import asyncssh

from slurm_mcp.config import ClusterProfile
from slurm_mcp.transport import CommandResult

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
USERS = {"trace": ("wxu2", "biosimmlab"), "bridges2": ("wxu7", "mch250030p")}

Handler = Callable[[str], "CommandResult | Exception | None"]


def ok(stdout: str, rc: int = 0, stderr: str = "") -> Callable[[str], CommandResult]:
    return lambda cmd: CommandResult(cmd, rc, stdout, stderr, 0.01)


def fixture_lines(cluster: str, name: str) -> list[str]:
    return [l for l in (FIXTURES / cluster / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def framed_discovery(cluster: str, *, helper: str | None = None, cap_o: bool = True, balance: list[str] | None = None,
                     env_line: str | None = None) -> str:
    """A ``::ENV .. ::END`` probe built from the captured fixtures (the design-form sections where captures differ)."""
    user, account = USERS[cluster]
    lines: list[str] = ["::ENV"]
    lines += [env_line] if env_line else fixture_lines(cluster, "env.out")
    lines += ["::VERSION", "slurm 22.05.11", "::CONFIG", *fixture_lines(cluster, "scontrol_config.out"),
              "::PARTITIONS", *fixture_lines(cluster, "scontrol_partitions.out"), "::RC 0",
              "::SINFO", *fixture_lines(cluster, "sinfo_nodes.out"), "::RC 0",
              "::USER", f"{user}|{account}",
              "::ASSOC", *fixture_lines(cluster, "sacctmgr_assoc.out"), "::RC 0",
              "::QOS", *fixture_lines(cluster, "sacctmgr_qos.out"), "::RC 0",
              "::SSHARE", *fixture_lines(cluster, "sshare_me.out"), "::RC 0",
              "::RESV", "No reservations in the system",
              "::TOOLS", *fixture_lines(cluster, "tools.out"),
              "::CAP_O", "rc=0" if cap_o else "rc=1",
              "::DF", *fixture_lines(cluster, "df_home.out")]
    if balance is not None:
        lines += ["::BALANCE", *balance]
    lines += ["::HELPER"] + ([helper] if helper else []) + ["::END"]
    return "\n".join(lines) + "\n"


class _FakeFile:
    def __init__(self, sftp: "FakeSFTP", path: str, mode: str) -> None:
        self.sftp, self.path, self.mode = sftp, path, mode
        self.buf = bytearray(sftp.files.get(path, b"") if "a" in mode else b"")

    async def __aenter__(self) -> "_FakeFile":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.sftp.files[self.path] = bytes(self.buf)
        self.sftp.mtimes[self.path] = time.time()
        self.sftp.dirs.add(posixpath.dirname(self.path))

    async def write(self, data: bytes) -> None:
        self.buf += data


class _Attrs:
    def __init__(self, *, size: int | None = None, is_dir: bool = False, mtime: float | None = None) -> None:
        self.type = asyncssh.FILEXFER_TYPE_DIRECTORY if is_dir else asyncssh.FILEXFER_TYPE_REGULAR
        self.permissions = (statmod.S_IFDIR | 0o755) if is_dir else (statmod.S_IFREG | 0o644)
        self.size = size
        self.mtime = mtime


class _Name:
    def __init__(self, filename: str, attrs: _Attrs) -> None:
        self.filename, self.attrs = filename, attrs


class FakeSFTP:
    """The subset of asyncssh.SFTPClient the client uses: open/makedirs/chmod/posix_rename/stat/scandir/realpath."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = {"/"}
        self.modes: dict[str, int] = {}
        self.mtimes: dict[str, float] = {}
        self.renames: list[tuple[str, str]] = []
        self.puts: list[str] = []

    def open(self, path: str, mode: str = "rb") -> _FakeFile:
        return _FakeFile(self, path, mode)

    async def makedirs(self, path: str, exist_ok: bool = False) -> None:
        p = path.rstrip("/")
        while p and p not in self.dirs:
            self.dirs.add(p)
            p = posixpath.dirname(p)

    async def chmod(self, path: str, mode: int) -> None:
        self.modes[path] = mode

    async def posix_rename(self, old: str, new: str) -> None:
        self.files[new] = self.files.pop(old)
        self.mtimes[new] = self.mtimes.pop(old, time.time())
        if old in self.modes:
            self.modes[new] = self.modes.pop(old)
        self.renames.append((old, new))
        self.puts.append(new)

    async def stat(self, path: str) -> _Attrs:
        p = path.rstrip("/") or "/"
        if p in self.files:
            return _Attrs(size=len(self.files[p]), mtime=self.mtimes.get(p))
        if p in self.dirs:
            return _Attrs(is_dir=True)
        raise asyncssh.SFTPNoSuchFile(f"no such file: {path}")

    async def realpath(self, path: str) -> str:
        return path

    async def scandir(self, path: str):
        p = path.rstrip("/") or "/"
        prefix = p + "/" if p != "/" else "/"
        seen: set[str] = set()
        for f in list(self.files):
            if f.startswith(prefix) and "/" not in f[len(prefix):]:
                name = f[len(prefix):]
                seen.add(name)
                yield _Name(name, _Attrs(size=len(self.files[f]), mtime=self.mtimes.get(f)))
        for d in list(self.dirs):
            if d.startswith(prefix) and "/" not in d[len(prefix):] and d[len(prefix):] not in seen:
                yield _Name(d[len(prefix):], _Attrs(is_dir=True))


class FakeTransport:
    """``SSHTransport`` look-alike: ``run`` dispatches on the command text through ordered ``handlers``."""

    def __init__(self, profile: ClusterProfile, handlers: list[tuple[str, Handler | CommandResult]] | None = None,
                 *, sftp: FakeSFTP | None = None, timeout_s: float = 120.0) -> None:
        self.profile = profile
        self.role = "login"
        self.host, self.port = profile.host, profile.port
        self.handlers: list[tuple[str, Any]] = list(handlers or [])
        self.calls: list[dict[str, Any]] = []
        self._sftp = sftp or FakeSFTP()
        self._timeout = timeout_s
        self.connected = True
        self.auth_failed = False
        self.hostkey_notices: list[str] = []

    def on(self, needle: str, response: Any) -> "FakeTransport":
        self.handlers.append((needle, response))
        return self

    def default_timeout(self) -> float:
        return self._timeout

    async def run(self, command: str, *, timeout: float | None = None, input: str | None = None,
                  idempotent: bool = True, login_shell: bool = True, check: bool = False) -> CommandResult:
        self.calls.append({"command": command, "timeout": timeout, "input": input, "idempotent": idempotent,
                           "login_shell": login_shell})
        for needle, response in self.handlers:
            if needle in command:
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, CommandResult):
                    return response
                out = response(command)
                if isinstance(out, Exception):
                    raise out
                if out is not None:
                    return out
        return CommandResult(command, 127, "", f"fake transport: no handler for: {command[:80]}", 0.0)

    async def sftp(self) -> FakeSFTP:
        return self._sftp

    async def close(self) -> None:
        self.connected = False

    def commands(self) -> list[str]:
        return [c["command"] for c in self.calls]


def profile_for(cluster: str, **kw: Any) -> ClusterProfile:
    base: dict[str, Any] = {"trace": dict(name="trace", host="trace.example", user="wxu2",
                                          remote_root="/trace/group/biosimmlab/wxu2",
                                          quota_paths=["/trace/group/biosimmlab"],
                                          target_overrides={"trace:biosimmlab*": {"enabled": False}}),
                            "bridges2": dict(name="bridges2", host="b2.example", user="wxu7",
                                             remote_root="/ocean/projects/mch250030p/wxu7",
                                             default_account="mch250030p", partition_groups=[["GPU-small", "GPU-shared"]],
                                             no_mem_flag=["RM-shared"], su_rates={"gpu:h100-80": 2, "gpu:*": 1, "cpu": 1},
                                             balance_command="projects",
                                             balance_regex=r"(?P<left>[\d,]+)\s*/\s*(?P<total>[\d,]+)\s*SU")}[cluster]
    base.update(kw)
    return ClusterProfile(**base)


__all__ = ["FIXTURES", "USERS", "ok", "fixture_lines", "framed_discovery", "FakeSFTP", "FakeTransport", "profile_for"]
