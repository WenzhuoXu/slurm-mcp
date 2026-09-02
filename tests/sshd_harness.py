"""In-process SSH server for tests (asyncssh).

* listens on 127.0.0.1 on a random port
* password auth for user "tester" / "pw"
* exec requests run the command through Git Bash (`bash.exe -c "<cmd>"`) with PATH prefixed by
  tests/fakeslurm/bin and FAKESLURM_STATE / FAKESLURM_PYTHON exported; stdin is redirected into the
  subprocess, stdout/stderr are relayed back, exit status propagated
* the shell starts in a temporary "remote home"; HOME and PWD carry its Git Bash spelling (/c/Users/...)
  so `pwd`, `$HOME` and the paths fakeslurm reports (WorkDir, StdOut, Command) all agree
* SFTP starts in that home; relative paths resolve against it and absolute POSIX paths (/c/Users/...,
  /tmp/...) are mapped to the Windows filesystem exactly like the shell does, so an StdOut path from
  scontrol can be stat'ed / downloaded as-is

Usage::

    async with fake_cluster("trace") as fc:
        fc.host, fc.port, fc.home, fc.state_path, fc.ctl("advance", "--seconds", "60")
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import asyncssh

TESTS_DIR = Path(__file__).resolve().parent
FAKESLURM_DIR = TESTS_DIR / "fakeslurm"
FAKESLURM_BIN = FAKESLURM_DIR / "bin"
FAKESLURM_PY = FAKESLURM_DIR / "fakeslurm.py"

if str(FAKESLURM_DIR) not in sys.path:
    sys.path.insert(0, str(FAKESLURM_DIR))
from fakeslurm import native_to_posix, posix_to_native  # noqa: E402

SSH_USER = "tester"
SSH_PASSWORD = "pw"


def find_bash() -> str:
    for cand in (os.environ.get("FAKESLURM_BASH"), r"C:\Program Files\Git\bin\bash.exe",
                 r"C:\Program Files\Git\usr\bin\bash.exe", shutil.which("bash")):
        if cand and os.path.exists(cand):
            return cand
    raise RuntimeError("Git Bash not found; set FAKESLURM_BASH")


def _to_posix(path: str) -> str:
    """C:\\x\\y -> /c/x/y (Git Bash spelling) for PATH entries, HOME and PWD."""
    return native_to_posix(path)


class _PosixSFTPServer(asyncssh.SFTPServer):
    """SFTP server whose namespace is the Git Bash one: relative paths live under the fake home,
    absolute paths are POSIX spellings (/c/Users/..., /tmp/...) mapped like the shell maps them."""

    def __init__(self, chan, home: str):
        super().__init__(chan)
        self._home = home.replace("\\", "/")

    def map_path(self, path: bytes) -> bytes:
        p = path.decode("utf-8", "surrogateescape")
        if p.startswith("/") and not (len(p) > 2 and p[2] == ":"):
            p = posix_to_native(p)
        elif p.startswith("/"):           # '/C:/x' as produced by asyncssh's own realpath
            p = p[1:]
        elif not (len(p) > 1 and p[1] == ":"):
            p = self._home + "/" + p if p not in ("", ".") else self._home
        return os.path.normpath(p).encode("utf-8", "surrogateescape")

    def reverse_map_path(self, path: bytes) -> bytes:
        p = path.decode("utf-8", "surrogateescape")
        if p.startswith("/") and len(p) > 2 and p[2] == ":":
            p = p[1:]
        return native_to_posix(p).encode("utf-8", "surrogateescape")


@dataclass
class FakeCluster:
    host: str
    port: int
    home: str
    state_path: str
    cluster: str
    python: str
    bash: str
    env: dict = field(default_factory=dict)
    server: object = None
    connections: list = field(default_factory=list)

    def ctl(self, *args: str) -> subprocess.CompletedProcess:
        """Run fakeslurm-ctl in-process-ish (subprocess of the test python) against this cluster's state."""
        env = dict(os.environ)
        env.update(self.env)
        return subprocess.run([self.python, str(FAKESLURM_PY), "fakeslurm-ctl", *args], env=env,
                              capture_output=True, text=True, check=True)

    def slurm(self, *argv: str, stdin: str = "") -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.update(self.env)
        return subprocess.run([self.python, str(FAKESLURM_PY), *argv], env=env, input=stdin,
                              capture_output=True, text=True, cwd=self.home)

    def state(self) -> dict:
        with open(self.state_path, encoding="utf-8") as fh:
            return json.load(fh)

    def drop_connections(self) -> None:
        """Forcibly close every server-side SSH connection (simulates a network drop)."""
        for conn in list(self.connections):
            try:
                conn.abort()
            except Exception:
                pass
        self.connections.clear()


class _Server(asyncssh.SSHServer):
    def __init__(self, fc: FakeCluster):
        self.fc = fc

    def connection_made(self, conn):
        self.fc.connections.append(conn)

    def connection_lost(self, exc):
        pass

    def begin_auth(self, username):
        return True

    def password_auth_supported(self):
        return True

    def validate_password(self, username, password):
        return username == SSH_USER and password == SSH_PASSWORD

    def kbdint_auth_supported(self):
        return True

    def get_kbdint_challenge(self, username, lang, submethods):
        return "", "", "", [("Password: ", False)]

    def validate_kbdint_response(self, username, responses):
        return username == SSH_USER and len(responses) == 1 and responses[0] == SSH_PASSWORD


def _make_process_handler(fc: FakeCluster):
    async def handle(process: asyncssh.SSHServerProcess):
        cmd = process.command
        if cmd is None:
            process.stderr.write("fake sshd: interactive shells are not supported\n")
            process.exit(1)
            return
        env = dict(os.environ)
        env.update(fc.env)
        try:
            proc = await asyncio.create_subprocess_exec(
                fc.bash, "-c", cmd, cwd=fc.home, env=env,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        except Exception as e:  # pragma: no cover
            process.stderr.write(f"fake sshd: cannot spawn bash: {e}\n")
            process.exit(127)
            return
        # client stdin -> subprocess stdin (EOF forwarded when the client sends it)
        await process.redirect(stdin=proc.stdin)
        out, err = await asyncio.gather(proc.stdout.read(), proc.stderr.read())
        rc = await proc.wait()
        if out:
            process.stdout.write(out)
        if err:
            process.stderr.write(err)
        process.exit(rc)

    return handle


@contextlib.asynccontextmanager
async def fake_cluster(cluster: str = "trace", *, now: str | None = None, start_jobid: int = 100000,
                       python: str | None = None):
    """Start the in-process SSH server backed by a freshly initialised fakeslurm state."""
    python = python or sys.executable
    bash = find_bash()
    tmp = tempfile.mkdtemp(prefix="fakecluster-")
    home = os.path.join(tmp, "home")
    os.makedirs(home)
    state_path = os.path.join(tmp, "state.json")
    fc = FakeCluster(host="127.0.0.1", port=0, home=home, state_path=state_path, cluster=cluster,
                     python=python, bash=bash)
    fc.env = {
        "FAKESLURM_STATE": state_path,
        "FAKESLURM_PYTHON": python.replace("\\", "/"),
        "FAKESLURM_BASH": bash,
        "PATH": _to_posix(str(FAKESLURM_BIN)) + os.pathsep + os.environ.get("PATH", ""),
        "HOME": _to_posix(home),
        # bash keeps an inherited $PWD that names the cwd, so `pwd` prints the /c/... spelling instead
        # of the /tmp/... mount alias; fakeslurm records WorkDir/StdOut from it (see fakeslurm.current_dir)
        "PWD": _to_posix(home),
        "MSYS2_PATH_TYPE": "inherit",
        "CHERE_INVOKING": "1",
    }
    init_args = ["init", "--cluster", cluster, "--start-jobid", str(start_jobid)]
    if now:
        init_args += ["--now", now]
    fc.ctl(*init_args)
    state = fc.state()
    fc.env["USER"] = state["user"]["name"]
    fc.env["USERNAME"] = state["user"]["name"]

    key = asyncssh.generate_private_key("ssh-ed25519")
    server = await asyncssh.listen(
        "127.0.0.1", 0, server_host_keys=[key], server_factory=lambda: _Server(fc),
        process_factory=_make_process_handler(fc), encoding=None,
        sftp_factory=lambda chan: _PosixSFTPServer(chan, home), allow_scp=True)
    fc.server = server
    fc.port = server.sockets[0].getsockname()[1]
    try:
        yield fc
    finally:
        fc.drop_connections()
        server.close()
        try:
            await asyncio.wait_for(server.wait_closed(), 5)
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)
