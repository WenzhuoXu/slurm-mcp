"""Unit tests for slurm_mcp.transport (design section 2.2, section 9.2 auth/host-key/drop rows).

No real SSH here except the host-key pool test, which uses an in-process asyncssh server on 127.0.0.1.
Everything else uses fabricated keys, local sockets and a fake connection object monkeypatched over
``asyncssh.connect``.
"""
from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

import asyncssh
import pytest

from slurm_mcp import transport as T
from slurm_mcp.config import ClusterProfile
from slurm_mcp.errors import SlurmMcpError
from slurm_mcp.transport import (AuthFailed, CommandResult, CommandTimeout, ConnectionDropped, HostKeyChanged,
                                 HostKeyStore, HpcClient, RemoteCommandError, SSHTransport, Unreachable,
                                 banner_probe, forget_host_keys, format_known_hosts_line, known_hosts_prefix,
                                 parse_known_hosts_line, tcp_probe)

def _profile(**kw) -> ClusterProfile:
    base = dict(name="unit", host="login.example.org", user="alice", port=22, auth="password")
    base.update(kw)
    return ClusterProfile(**base)


def _key(alg: str = "ssh-ed25519") -> asyncssh.SSHKey:
    return asyncssh.generate_private_key(alg).convert_to_public()


def _store(tmp_path: Path) -> HostKeyStore:
    return HostKeyStore(tmp_path / "known_hosts")


# --- CommandResult ------------------------------------------------------------------------------------

def test_command_result_ok_and_raise():
    good = CommandResult("true", 0, "", "", 0.1)
    assert good.ok and good.raise_for_status() is good
    bad = CommandResult("false", 1, "", "boom", 0.1)
    assert not bad.ok
    with pytest.raises(RemoteCommandError) as ei:
        bad.raise_for_status()
    assert ei.value.result is bad and "rc=1" in str(ei.value) and "boom" in str(ei.value)


# --- exceptions ---------------------------------------------------------------------------------------

def test_error_classes_carry_catalogue_codes():
    hk = HostKeyChanged("h", "10.0.0.1", ["SHA256:old"], "SHA256:new", "trace")
    assert isinstance(hk, SlurmMcpError) and isinstance(hk, RuntimeError)
    assert str(hk).startswith("E_HOSTKEY: host key for h from 10.0.0.1 changed: old SHA256:old new SHA256:new")
    assert "hostkeys forget trace" in str(hk)
    au = AuthFailed("bridges2", "rejected")
    assert au.code == "E_AUTH" and "auth set bridges2" in str(au)
    un = Unreachable("trace", "no route", "connect to the CMU VPN")
    assert un.code == "E_UNREACHABLE" and isinstance(un, ConnectionError) and "CMU VPN" in str(un)
    ct = CommandTimeout("sbatch x", 120, ambiguous=True, stdout="p")
    assert isinstance(ct, asyncio.TimeoutError) and ct.ambiguous and ct.stdout == "p" and "ambiguous" in str(ct)
    cd = ConnectionDropped("sbatch x", "lost", ambiguous=True)
    assert cd.ambiguous and "lost" in str(cd)


# --- known_hosts line helpers -------------------------------------------------------------------------

def test_known_hosts_prefix_and_line_roundtrip():
    key = _key()
    assert known_hosts_prefix("h", 22) == "h" and known_hosts_prefix("h", None) == "h"
    assert known_hosts_prefix("h", 2222) == "[h]:2222"
    line = format_known_hosts_line("h", 2222, key)
    assert line.startswith("[h]:2222 ssh-ed25519 ") and line.endswith("\n")
    host_field, parsed = parse_known_hosts_line(line)
    assert host_field == "[h]:2222" and parsed.get_fingerprint() == key.get_fingerprint()
    assert parse_known_hosts_line("") is None
    assert parse_known_hosts_line("# comment") is None
    assert parse_known_hosts_line("@cert-authority h ssh-rsa AAAA") is None
    assert parse_known_hosts_line("h ssh-ed25519 notbase64!!") is None
    assert parse_known_hosts_line("h onlytwo") is None


# --- HostKeyStore -------------------------------------------------------------------------------------

def test_store_record_appends_multiple_lines_and_json(tmp_path):
    st = _store(tmp_path)
    assert st.keys_for("h") == [] and st.seen_addrs("h") == {}
    k1, k2 = _key(), _key()
    fp1 = st.record("h", 22, "10.0.0.1", k1)
    fp2 = st.record("h", 22, "10.0.0.2", k2)
    assert fp1 == k1.get_fingerprint() and fp2 == k2.get_fingerprint()
    lines = [l for l in st.path.read_text().splitlines() if l.strip()]
    assert len(lines) == 2 and all(l.startswith("h ssh-ed25519 ") for l in lines)
    assert st.path.read_bytes().count(b"\r") == 0
    assert {k.get_fingerprint() for k in st.keys_for("h")} == {fp1, fp2}
    meta = json.loads(st.meta_path.read_text())
    assert meta == {"h": {"10.0.0.1": [fp1], "10.0.0.2": [fp2]}}
    # the same key from a second address adds an addr entry but not a duplicate line
    st.record("h", 22, "10.0.0.3", k1)
    assert len([l for l in st.path.read_text().splitlines() if l.strip()]) == 2
    assert st.seen_addrs("h")["10.0.0.3"] == [fp1]
    # recording twice for the same addr does not duplicate the fingerprint
    st.record("h", 22, "10.0.0.3", k1)
    assert st.seen_addrs("h")["10.0.0.3"] == [fp1]


def test_store_port_prefix_isolation(tmp_path):
    st = _store(tmp_path)
    k = _key()
    st.record("h", 2222, "10.0.0.1", k)
    assert st.path.read_text().startswith("[h]:2222 ")
    assert [x.get_fingerprint() for x in st.keys_for("h", 2222)] == [k.get_fingerprint()]
    assert st.keys_for("h", 22) == [] and st.keys_for("h") == []
    # a bare line also matches port-22 lookups
    st.record("h", None, "10.0.0.1", k)
    assert len(st.keys_for("h", 22)) == 1 and len(st.keys_for("h", None)) == 1


def test_store_decide_pool_rule(tmp_path):
    st = _store(tmp_path)
    ed1, ed2, rsa = _key(), _key(), _key("ssh-rsa")
    # nothing stored: everything is "new"
    assert st.decide("h", 22, "10.0.0.1", ed1).verdict == "new"
    st.record("h", 22, "10.0.0.1", ed1)
    # known key from any address: "known"
    assert st.decide("h", 22, "10.0.0.1", ed1).verdict == "known"
    assert st.decide("h", 22, "10.0.0.9", ed1).verdict == "known"
    # unseen address, unknown key: pool member -> "new"
    assert st.decide("h", 22, "10.0.0.2", ed2).verdict == "new"
    # seen address, different key of the SAME type -> "changed" with the old fingerprint
    d = st.decide("h", 22, "10.0.0.1", ed2)
    assert d.verdict == "changed" and d.old_fingerprints == [ed1.get_fingerprint()]
    assert d.fingerprint == ed2.get_fingerprint()
    # seen address, unknown key of a DIFFERENT type (server added an rsa key) -> "new"
    assert st.decide("h", 22, "10.0.0.1", rsa).verdict == "new"
    # another alias is a separate pool
    assert st.decide("other", 22, "10.0.0.1", ed2).verdict == "new"


def test_store_forget_clears_both(tmp_path):
    st = _store(tmp_path)
    st.record("h", 22, "10.0.0.1", _key())
    st.record("h", 2222, "10.0.0.1", _key())
    st.record("keep", 22, "10.0.0.5", _key())
    st.path.write_text(st.path.read_text() + "# a comment\n", encoding="utf-8")
    assert forget_host_keys("h", st) == 2
    text = st.path.read_text()
    assert "[h]:2222" not in text and not any(l.startswith("h ") for l in text.splitlines())
    assert "keep " in text and "# a comment" in text
    assert json.loads(st.meta_path.read_text()) == {"keep": {"10.0.0.5": [st.keys_for("keep")[0].get_fingerprint()]}}
    assert forget_host_keys("h", st) == 0
    assert forget_host_keys("missing", HostKeyStore(tmp_path / "nope" / "known_hosts")) == 0


def test_store_tolerates_garbage_side_file_and_lines(tmp_path):
    st = _store(tmp_path)
    st.path.parent.mkdir(exist_ok=True)
    st.path.write_text("garbage line\n\nh ssh-ed25519 !!!\n", encoding="utf-8")
    st.meta_path.write_text("{not json", encoding="utf-8")
    assert st.keys_for("h") == [] and st.load_meta() == {}
    k = _key()
    st.record("h", 22, "1.2.3.4", k)
    assert len(st.keys_for("h")) == 1 and st.seen_addrs("h") == {"1.2.3.4": [k.get_fingerprint()]}


def test_known_hosts_callable_returns_stored_keys(tmp_path):
    st = _store(tmp_path)
    k = _key()
    st.record("h", 2222, "10.0.0.1", k)
    fn = st.known_hosts_callable("h", 2222)
    keys, cas, revoked = fn("h", "10.0.0.1", 2222)
    assert [x.get_fingerprint() for x in keys] == [k.get_fingerprint()] and cas == [] and revoked == []
    assert st.known_hosts_callable("h", 22)("h", "10.0.0.1", None)[0] == []
    # asyncssh's own matcher accepts the callable's output shape
    trusted, *_ = asyncssh.match_known_hosts(fn, "h", "10.0.0.1", 2222)
    assert [x.get_fingerprint() for x in trusted] == [k.get_fingerprint()]


# --- HpcClient policy ---------------------------------------------------------------------------------

def test_hpc_client_validate_host_public_key(tmp_path, caplog):
    st = _store(tmp_path)
    ed1, ed2 = _key(), _key()
    c = HpcClient(st, "h", 22, cluster="trace")
    with caplog.at_level("WARNING", logger="slurm_mcp.transport"):
        assert c.validate_host_public_key("h", "10.0.0.1", 22, ed1) is True
    assert c.accepted == [f"new host key for h from 10.0.0.1 {ed1.get_fingerprint()}"]
    assert "new host key for h from 10.0.0.1" in caplog.text
    assert len(st.keys_for("h")) == 1
    # second pool member, different address: accepted + appended
    assert c.validate_host_public_key("h", "10.0.0.2", 22, ed2) is True
    assert len(st.keys_for("h")) == 2 and len(c.accepted) == 2
    # already known key: accepted without a new notice
    assert c.validate_host_public_key("h", "10.0.0.7", 22, ed1) is True and len(c.accepted) == 2
    # same address, a third ed25519 key -> refused, HostKeyChanged prepared, nothing written
    ed3 = _key()
    assert c.validate_host_public_key("h", "10.0.0.1", 22, ed3) is False
    assert isinstance(c.changed, HostKeyChanged) and c.changed.addr == "10.0.0.1"
    assert c.changed.old_fps == [ed1.get_fingerprint()] and c.changed.new_fp == ed3.get_fingerprint()
    assert "trace" in c.changed.fix and len(st.keys_for("h")) == 2


def test_hpc_client_trust_new_false_refuses_unknown(tmp_path):
    st = _store(tmp_path)
    c = HpcClient(st, "h", 22, trust_new=False)
    assert c.validate_host_public_key("h", "10.0.0.1", 22, _key()) is False
    assert c.refused and c.refused.startswith("unknown host key for h from 10.0.0.1") and c.changed is None
    assert st.keys_for("h") == []


# --- probes -------------------------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_tcp_probe_open_and_closed():
    async def handler(reader, writer):
        writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        assert await tcp_probe("127.0.0.1", port) is True
    finally:
        server.close()
        await server.wait_closed()
    assert await tcp_probe("127.0.0.1", _free_port(), timeout=2) is False
    assert await tcp_probe("no-such-host.invalid", 22, timeout=2) is False


@pytest.mark.asyncio
async def test_banner_probe_reads_ssh_line():
    async def handler(reader, writer):
        writer.write(b"SSH-2.0-test\r\n")
        await writer.drain()
        await asyncio.sleep(0.05)
        writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        assert await banner_probe("127.0.0.1", port) == "SSH-2.0-test"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_banner_probe_skips_prebanner_text_and_handles_silence():
    async def handler(reader, writer):
        writer.write(b"Welcome to the DTN\nSSH-2.0-OpenSSH_8.0\n")
        await writer.drain()
        writer.close()

    async def silent(reader, writer):
        await asyncio.sleep(2)
        writer.close()

    s1 = await asyncio.start_server(handler, "127.0.0.1", 0)
    s2 = await asyncio.start_server(silent, "127.0.0.1", 0)
    try:
        assert await banner_probe("127.0.0.1", s1.sockets[0].getsockname()[1]) == "SSH-2.0-OpenSSH_8.0"
        assert await banner_probe("127.0.0.1", s2.sockets[0].getsockname()[1], timeout=0.3) == ""
    finally:
        for s in (s1, s2):
            s.close()
            await s.wait_closed()
    assert await banner_probe("127.0.0.1", _free_port(), timeout=1) == ""


# --- fake connection for run() logic ----------------------------------------------------------------

class FakeProc:
    def __init__(self, exit_status=0, stdout="", stderr=""):
        self.exit_status, self.stdout, self.stderr = exit_status, stdout, stderr


class FakeSftp:
    def __init__(self, conn):
        self.conn = conn
        self.exited = False

    def exit(self):
        self.exited = True


class FakeConn:
    """Scripted stand-in for asyncssh.SSHClientConnection: each run() pops the next item of ``script``
    (a FakeProc, an exception to raise, or a coroutine function)."""

    def __init__(self, script=None):
        self.script = list(script or [])
        self.closed = False
        self.calls: list[tuple[str, dict]] = []
        self.active = 0
        self.max_active = 0

    def is_closed(self):
        return self.closed

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None

    async def run(self, cmd, **kw):
        self.calls.append((cmd, kw))
        item = self.script.pop(0) if self.script else FakeProc(0, "out", "")
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if isinstance(item, BaseException):
                raise item
            if callable(item):
                return await item()
            return item
        finally:
            self.active -= 1

    async def start_sftp_client(self):
        return FakeSftp(self)


class FakeConnect:
    """Monkeypatched ``asyncssh.connect``: hands out the next FakeConn (or raises the next exception)."""

    def __init__(self, conns):
        self.conns = list(conns)
        self.calls: list[dict] = []

    async def __call__(self, **opts):
        self.calls.append(opts)
        item = self.conns.pop(0)
        if isinstance(item, BaseException):
            raise item
        pw = opts.get("password")
        if callable(pw):                       # asyncssh awaits the single-use callable during auth
            res = pw()
            if asyncio.iscoroutine(res):
                await res
        return item


@pytest.fixture
def patch_connect(monkeypatch):
    def _install(*conns):
        fc = FakeConnect(conns)
        monkeypatch.setattr(T.asyncssh, "connect", fc)
        return fc
    return _install


@pytest.fixture
def transport(tmp_path, monkeypatch):
    monkeypatch.setattr(T.credentials, "get_password", lambda p: "secret")

    def _make(**kw):
        prof = kw.pop("profile", None) or _profile(**kw.pop("profile_kw", {}))
        t = SSHTransport(prof, store=_store(tmp_path), **kw)
        t.retry_backoff_s = 0.0
        return t
    return _make


def _lost():
    return asyncssh.ConnectionLost("Connection lost")


def _timeout(stdout="partial", stderr=""):
    return asyncssh.TimeoutError(None, "cmd", None, None, None, None, stdout, stderr)


# --- constructor / options ----------------------------------------------------------------------------

def test_constructor_defaults_and_roles(tmp_path):
    p = _profile(transfer_host="data.example.org", ssh_max_exec=3)
    t = SSHTransport(p, store=_store(tmp_path))
    assert (t.host, t.port, t.role, t.max_sessions) == ("login.example.org", 22, "login", 3)
    tt = SSHTransport(p, host="data.example.org", port=2222, role="transfer", store=_store(tmp_path))
    assert (tt.host, tt.port, tt.role) == ("data.example.org", 2222, "transfer")
    assert SSHTransport(p, max_sessions=8, store=_store(tmp_path)).max_sessions == 8
    with pytest.raises(ValueError):
        SSHTransport(p, role="bogus", store=_store(tmp_path))
    assert t.auth_failed is False and t.connected is False


def test_connect_options_password_profile(tmp_path):
    t = SSHTransport(_profile(), store=_store(tmp_path), connect_timeout=45, login_timeout=90)
    o = t._connect_options()
    assert o["host"] == "login.example.org" and o["port"] == 22 and o["username"] == "alice"
    assert o["client_keys"] is None and o["agent_path"] is None and o["gss_host"] is None
    assert o["preferred_auth"] == ["keyboard-interactive", "password"]
    assert callable(o["password"]) and callable(o["known_hosts"]) and o["config"] is None
    assert (o["connect_timeout"], o["login_timeout"]) == (45, 90)
    assert (o["keepalive_interval"], o["keepalive_count_max"]) == (30, 4)
    assert o["compression_algs"] is None and (o["encoding"], o["errors"]) == ("utf-8", "replace")
    assert isinstance(o["client_factory"](), HpcClient)


def test_connect_options_key_and_agent_profiles(tmp_path):
    ok = SSHTransport(_profile(auth="key", key_path="C:/k/id"), store=_store(tmp_path))._connect_options()
    assert ok["client_keys"] == ["C:/k/id"] and ok["agent_path"] is None and ok["preferred_auth"] == "publickey"
    assert "password" not in ok
    oa = SSHTransport(_profile(auth="agent"), store=_store(tmp_path))._connect_options()
    assert oa["preferred_auth"] == "publickey" and "client_keys" not in oa and "password" not in oa


@pytest.mark.asyncio
async def test_password_provider_reads_keyring_off_thread(tmp_path, monkeypatch):
    import threading
    seen = {}

    def fake_get(profile):
        seen["thread"] = threading.current_thread() is threading.main_thread()
        seen["profile"] = profile
        return "pw"

    monkeypatch.setattr(T.credentials, "get_password", fake_get)
    t = SSHTransport(_profile(), store=_store(tmp_path))
    assert await t._password() == "pw"
    assert seen["thread"] is False and seen["profile"] is t.profile
    monkeypatch.setattr(T.credentials, "get_password", lambda p: None)
    with pytest.raises(AuthFailed) as ei:
        await t._password()
    assert "no password stored" in ei.value.message


# --- default timeout ------------------------------------------------------------------------------------

def test_default_timeout_resolution(tmp_path):
    st = _store(tmp_path)
    assert SSHTransport(_profile(), store=st).default_timeout() == 120.0
    assert SSHTransport(_profile(cmd_timeout_s=45), store=st).default_timeout() == 45.0
    assert SSHTransport(_profile(), store=st, caps_cmd_timeout_s=lambda: 260).default_timeout() == 260.0
    assert SSHTransport(_profile(), store=st, caps_cmd_timeout_s=lambda: 90).default_timeout() == 120.0
    assert SSHTransport(_profile(), store=st, caps_cmd_timeout_s=lambda: None).default_timeout() == 120.0
    assert SSHTransport(_profile(cmd_timeout_s=45), store=st, caps_cmd_timeout_s=lambda: 260).default_timeout() == 45.0

    def broken():
        raise KeyError("caps")
    assert SSHTransport(_profile(), store=st, caps_cmd_timeout_s=broken).default_timeout() == 120.0


def test_wrap_login_shell():
    assert SSHTransport.wrap("echo 'a b'", True) == "bash -lc 'echo '\"'\"'a b'\"'\"''"
    assert SSHTransport.wrap("echo hi", False) == "echo hi"


# --- run(): happy path and drop/retry policy -------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_happy_path(transport, patch_connect):
    conn = FakeConn([FakeProc(0, "hello\n", "warn")])
    fc = patch_connect(conn)
    t = transport(profile_kw=dict(cmd_timeout_s=77))
    res = await t.run("echo hello")
    assert res.ok and res.stdout == "hello\n" and res.stderr == "warn" and res.command == "echo hello"
    assert res.seconds >= 0 and t.connected
    cmd, kw = conn.calls[0]
    assert cmd == "bash -lc 'echo hello'"
    assert kw == dict(timeout=77.0, check=False, input=None, encoding="utf-8", errors="replace")
    assert len(fc.calls) == 1 and fc.calls[0]["username"] == "alice"
    await t.run("raw", login_shell=False, timeout=5, input="x")
    assert conn.calls[1][0] == "raw" and conn.calls[1][1]["timeout"] == 5 and conn.calls[1][1]["input"] == "x"
    assert len(fc.calls) == 1  # connection reused
    await t.close()
    assert conn.closed and not t.connected


@pytest.mark.asyncio
async def test_run_check_raises(transport, patch_connect):
    patch_connect(FakeConn([FakeProc(2, "", "nope")]))
    t = transport()
    with pytest.raises(RemoteCommandError):
        await t.run("false", check=True)
    assert (await t.run("false")).returncode == 0  # default script: exit 0 after the scripted item is consumed


@pytest.mark.asyncio
async def test_run_idempotent_retries_once_after_connection_lost(transport, patch_connect):
    c1 = FakeConn([_lost()])
    c2 = FakeConn([FakeProc(0, "ok", "")])
    fc = patch_connect(c1, c2)
    t = transport()
    res = await t.run("squeue")
    assert res.stdout == "ok" and len(fc.calls) == 2 and c1.closed and t.reconnects == 1
    assert t._conn is c2


@pytest.mark.asyncio
async def test_run_idempotent_retries_once_on_missing_exit_status(transport, patch_connect):
    c1 = FakeConn([FakeProc(None, "", "")])
    c2 = FakeConn([FakeProc(0, "ok", "")])
    patch_connect(c1, c2)
    t = transport()
    assert (await t.run("squeue")).stdout == "ok" and t.reconnects == 1


@pytest.mark.asyncio
async def test_run_idempotent_retries_on_channel_open_error_connection_closed(transport, patch_connect):
    c1 = FakeConn([asyncssh.ChannelOpenError(2, "SSH connection closed")])
    c2 = FakeConn([FakeProc(0, "ok", "")])
    patch_connect(c1, c2)
    assert (await transport().run("squeue")).stdout == "ok"


@pytest.mark.asyncio
async def test_run_idempotent_gives_up_after_second_drop(transport, patch_connect):
    c1 = FakeConn([_lost()])
    c2 = FakeConn([_lost()])
    fc = patch_connect(c1, c2)
    t = transport()
    with pytest.raises(ConnectionDropped) as ei:
        await t.run("squeue")
    assert ei.value.ambiguous is False and len(fc.calls) == 2 and c2.closed


@pytest.mark.asyncio
async def test_run_non_idempotent_never_retries(transport, patch_connect):
    c1 = FakeConn([_lost()])
    c2 = FakeConn([FakeProc(0, "ok", "")])
    fc = patch_connect(c1, c2)
    t = transport()
    with pytest.raises(ConnectionDropped) as ei:
        await t.run("sbatch job.sbatch", idempotent=False)
    assert ei.value.ambiguous is True and ei.value.command == "sbatch job.sbatch"
    assert len(fc.calls) == 1 and c1.closed and c2.calls == []
    # missing exit status is equally ambiguous for a non-idempotent command
    c3 = FakeConn([FakeProc(None, "", "")])
    patch_connect(c3)
    t2 = transport()
    with pytest.raises(ConnectionDropped) as ei2:
        await t2.run("sbatch x", idempotent=False)
    assert ei2.value.ambiguous is True


@pytest.mark.asyncio
async def test_run_max_sessions_error_is_not_a_drop(transport, patch_connect):
    conn = FakeConn([asyncssh.ChannelOpenError(1, "open failed")])
    fc = patch_connect(conn, FakeConn())
    t = transport()
    with pytest.raises(asyncssh.ChannelOpenError):
        await t.run("squeue")
    assert len(fc.calls) == 1 and not conn.closed


@pytest.mark.asyncio
async def test_run_timeout_ambiguity(transport, patch_connect):
    patch_connect(FakeConn([_timeout("part", "e"), _timeout()]))
    t = transport()
    with pytest.raises(CommandTimeout) as ei:
        await t.run("sacct", timeout=3)
    assert ei.value.ambiguous is False and ei.value.timeout == 3 and ei.value.stdout == "part"
    with pytest.raises(CommandTimeout) as ei2:
        await t.run("sbatch", timeout=3, idempotent=False)
    assert ei2.value.ambiguous is True and ei2.value.command == "sbatch"
    assert t.connected  # a timeout does not tear the connection down


@pytest.mark.asyncio
async def test_run_serialises_channel_opens_with_semaphore(transport, patch_connect):
    async def slow():
        await asyncio.sleep(0.05)
        return FakeProc(0, "x", "")

    conn = FakeConn([slow] * 6)
    patch_connect(conn)
    t = transport(profile_kw=dict(ssh_max_exec=2))
    await asyncio.gather(*(t.run("cmd") for _ in range(6)))
    assert conn.max_active == 2 and len(conn.calls) == 6


# --- run_with_stdin_file / run_to_file ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_with_stdin_file_is_binary_and_non_idempotent(transport, patch_connect, tmp_path):
    c1 = FakeConn([_lost()])
    fc = patch_connect(c1, FakeConn())
    t = transport()
    tar = tmp_path / "x.tar"
    tar.write_bytes(b"data")
    with pytest.raises(ConnectionDropped) as ei:
        await t.run_with_stdin_file("tar -xf - -C r", tar, 600)
    assert ei.value.ambiguous is True and len(fc.calls) == 1
    conn = FakeConn([FakeProc(0, b"bytes out", b"bytes err")])
    patch_connect(conn)
    t = transport()
    res = await t.run_with_stdin_file("tar -xf - -C r", str(tar), 600)
    assert res.ok and res.stdout == "bytes out" and res.stderr == "bytes err"
    cmd, kw = conn.calls[0]
    assert cmd == "bash -lc 'tar -xf - -C r'" and kw["stdin"] == str(tar) and kw["encoding"] is None
    assert kw["timeout"] == 600 and "input" not in kw


@pytest.mark.asyncio
async def test_run_to_file_is_binary_and_idempotent(transport, patch_connect, tmp_path):
    c1 = FakeConn([_lost()])
    c2 = FakeConn([FakeProc(0, None, b"warn")])
    patch_connect(c1, c2)
    t = transport()
    out = tmp_path / "out.tar"
    res = await t.run_to_file("tar -cf - .", out)
    assert res.ok and res.stdout == "" and res.stderr == "warn" and t.reconnects == 1
    cmd, kw = c2.calls[0]
    assert kw["stdout"] == str(out) and kw["encoding"] is None and kw["timeout"] == 120.0
    assert cmd == "bash -lc 'tar -cf - .'"


# --- sftp cache ----------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sftp_cached_and_recreated_after_reconnect(transport, patch_connect):
    c1 = FakeConn([_lost()])
    c2 = FakeConn()
    patch_connect(c1, c2)
    t = transport()
    s1 = await t.sftp()
    assert s1 is await t.sftp() and s1.conn is c1
    await t.run("squeue")           # drops c1, reconnects to c2
    assert s1.exited
    s2 = await t.sftp()
    assert s2 is not s1 and s2.conn is c2
    await t.close()
    assert s2.exited and t._sftp is None


# --- auth latch -----------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_permission_denied_sets_latch_until_reset(transport, patch_connect):
    fc = patch_connect(asyncssh.PermissionDenied("Permission denied"), FakeConn([FakeProc(0, "ok", "")]))
    t = transport()
    with pytest.raises(AuthFailed) as ei:
        await t.run("squeue")
    assert t.auth_failed is True and ei.value.code == "E_AUTH" and "rejected the credentials" in ei.value.message
    assert "auth set unit" in str(ei.value)
    # no reconnect is attempted while latched: connect is not called again
    with pytest.raises(AuthFailed):
        await t.run("squeue")
    with pytest.raises(AuthFailed):
        await t.ensure_connected()
    assert len(fc.calls) == 1
    t.reset_auth()
    assert t.auth_failed is False and t.auth_error is None
    assert (await t.run("squeue")).stdout == "ok" and len(fc.calls) == 2


@pytest.mark.asyncio
async def test_missing_password_sets_latch(transport, patch_connect, monkeypatch):
    monkeypatch.setattr(T.credentials, "get_password", lambda p: None)
    fc = patch_connect(FakeConn(), FakeConn())
    t = transport()
    with pytest.raises(AuthFailed) as ei:
        await t.ensure_connected()
    assert "no password stored" in ei.value.message and t.auth_failed
    with pytest.raises(AuthFailed):
        await t.ensure_connected()
    assert len(fc.calls) == 1


# --- host key failures at connect ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_host_key_changed_surfaces_from_connect(transport, monkeypatch):
    class Refusing(FakeConnect):
        async def __call__(self, **opts):
            self.calls.append(opts)
            client = opts["client_factory"]()
            client.changed = HostKeyChanged("login.example.org", "1.2.3.4", ["SHA256:a"], "SHA256:b", "unit")
            raise asyncssh.HostKeyNotVerifiable("Host key is not trusted for host login.example.org")

    t = transport()
    fc = Refusing([])
    monkeypatch.setattr(T.asyncssh, "connect", fc)
    with pytest.raises(HostKeyChanged) as ei:
        await t.ensure_connected(retries=3)
    assert ei.value.code == "E_HOSTKEY" and len(fc.calls) == 1  # never retried
    assert not t.auth_failed


@pytest.mark.asyncio
async def test_unknown_host_key_refused_when_trust_new_hosts_false(transport, patch_connect):
    patch_connect(asyncssh.HostKeyNotVerifiable("Host key is not trusted for host x"))
    t = transport(trust_new_hosts=False)
    with pytest.raises(SlurmMcpError) as ei:
        await t.ensure_connected()
    assert ei.value.code == "E_HOSTKEY" and not isinstance(ei.value, HostKeyChanged)


# --- ensure_connected retries / unreachable ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_connected_backoff_then_unreachable(transport, patch_connect, monkeypatch):
    fc = patch_connect(OSError("refused"), asyncio.TimeoutError(), asyncssh.ConnectionLost("Login timeout expired"))
    sleeps: list[float] = []

    async def fake_sleep(d):
        sleeps.append(d)
    monkeypatch.setattr(T.asyncio, "sleep", fake_sleep)

    async def probe(host, port, timeout=3):
        return False
    monkeypatch.setattr(T, "tcp_probe", probe)
    t = transport()
    t.retry_backoff_s = 2.0
    with pytest.raises(Unreachable) as ei:
        await t.ensure_connected(retries=2)
    assert len(fc.calls) == 3 and sleeps == [2.0, 4.0]
    assert "TCP closed" in ei.value.message and "Login timeout" in ei.value.message
    assert ei.value.code == "E_UNREACHABLE" and "VPN/DNS" in ei.value.fix


@pytest.mark.asyncio
async def test_ensure_connected_recovers_after_transient_failure(transport, patch_connect):
    fc = patch_connect(OSError("refused"), FakeConn([FakeProc(0, "ok", "")]))
    t = transport()
    assert (await t.run("squeue")).stdout == "ok" and len(fc.calls) == 2


@pytest.mark.asyncio
async def test_vpn_hint_probe_fails_fast(transport, patch_connect, monkeypatch):
    fc = patch_connect(FakeConn())
    calls = []

    async def probe(host, port, timeout=3):
        calls.append((host, port))
        return False
    monkeypatch.setattr(T, "tcp_probe", probe)
    t = transport(profile_kw=dict(requires_vpn_hint="connect to Cisco AnyConnect"))
    with pytest.raises(Unreachable) as ei:
        await t.ensure_connected()
    assert calls == [("login.example.org", 22)] and fc.calls == []
    assert "Cisco AnyConnect" in str(ei.value) and "does not answer" in ei.value.message


@pytest.mark.asyncio
async def test_transport_probe_methods_delegate(transport, monkeypatch):
    async def probe(host, port, timeout=3):
        return (host, port, timeout)

    async def banner(host, port, timeout=5):
        return f"{host}:{port}:{timeout}"
    monkeypatch.setattr(T, "tcp_probe", probe)
    monkeypatch.setattr(T, "banner_probe", banner)
    t = transport(host="data.example.org", port=2222)
    assert await t.tcp_probe() == ("data.example.org", 2222, 3)
    assert await t.tcp_probe("x", 1, timeout=9) == ("x", 1, 9)
    assert await t.banner_probe() == "data.example.org:2222:5"


# --- real asyncssh server: host-key pool rule end to end ---------------------------------------------------

class _PwServer(asyncssh.SSHServer):
    def begin_auth(self, username):
        return True

    def password_auth_supported(self):
        return True

    def validate_password(self, username, password):
        return password == "secret"


async def _listen(key):
    async def handle(process):
        process.stdout.write("pong\n")
        process.exit(0)
    return await asyncssh.listen("127.0.0.1", 0, server_host_keys=[key], server_factory=_PwServer,
                                 process_factory=handle)


@pytest.mark.asyncio
async def test_live_host_key_pool_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(T.credentials, "get_password", lambda p: "secret")
    key_a, key_b = asyncssh.generate_private_key("ssh-ed25519"), asyncssh.generate_private_key("ssh-ed25519")
    store = _store(tmp_path)
    srv = await _listen(key_a)
    port = srv.sockets[0].getsockname()[1]
    prof = _profile(host="127.0.0.1", port=port)
    try:
        t = SSHTransport(prof, store=store, connect_timeout=10)
        res = await t.run("ping", login_shell=False)
        assert res.stdout == "pong\n"
        fp_a = key_a.convert_to_public().get_fingerprint()
        assert t.hostkey_notices == [f"new host key for 127.0.0.1 from 127.0.0.1 {fp_a}"]
        assert store.path.read_text().startswith(f"[127.0.0.1]:{port} ssh-ed25519 ")
        assert store.seen_addrs("127.0.0.1") == {"127.0.0.1": [fp_a]}
        await t.close()
        # second connect: the key is known via the known_hosts callable, no notice
        t2 = SSHTransport(prof, store=store, connect_timeout=10)
        await t2.ensure_connected()
        assert t2.hostkey_notices == []
        await t2.close()
    finally:
        srv.close()
        await srv.wait_closed()
    # same address, a different ed25519 key -> HostKeyChanged, nothing appended
    srv = await asyncssh.listen("127.0.0.1", port, server_host_keys=[key_b], server_factory=_PwServer,
                                process_factory=lambda p: p.exit(0))
    try:
        t3 = SSHTransport(prof, store=store, connect_timeout=10)
        with pytest.raises(HostKeyChanged) as ei:
            await t3.ensure_connected()
        assert ei.value.old_fps == [fp_a] and ei.value.new_fp == key_b.convert_to_public().get_fingerprint()
        assert len(store.keys_for("127.0.0.1", port)) == 1 and not t3.auth_failed
        # hostkeys forget clears both stores; the new key is then accepted as first use
        assert forget_host_keys("127.0.0.1", store) == 1
        assert store.load_meta() == {}
        t4 = SSHTransport(prof, store=store, connect_timeout=10)
        await t4.ensure_connected()
        assert len(t4.hostkey_notices) == 1 and len(store.keys_for("127.0.0.1", port)) == 1
        await t4.close()
    finally:
        srv.close()
        await srv.wait_closed()


@pytest.mark.asyncio
async def test_live_permission_denied_latches(tmp_path, monkeypatch):
    monkeypatch.setattr(T.credentials, "get_password", lambda p: "wrong")
    srv = await _listen(asyncssh.generate_private_key("ssh-ed25519"))
    port = srv.sockets[0].getsockname()[1]
    try:
        t = SSHTransport(_profile(host="127.0.0.1", port=port), store=_store(tmp_path), connect_timeout=10)
        with pytest.raises(AuthFailed):
            await t.ensure_connected()
        assert t.auth_failed
    finally:
        srv.close()
        await srv.wait_closed()
