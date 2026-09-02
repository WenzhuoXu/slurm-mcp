"""SSHTransport section 2.2 extensions against the in-process asyncssh server of tests/sshd_harness.py.

Covers what the fake-connection unit tests cannot: a real exec channel through ``bash -lc``, an SFTP round
trip, ``run_with_stdin_file`` / ``run_to_file`` as binary tar streams, the host-key store after a real
connect, reconnect after a server-side drop, and the ``auth_failed`` latch on a real ``PermissionDenied``.
"""
from __future__ import annotations

import asyncio
import io
import os
import tarfile
from pathlib import Path

import pytest

from conftest import make_profile
from slurm_mcp.transport import AuthFailed, HostKeyStore, SSHTransport


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.uid = ti.gid = 0
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


@pytest.mark.asyncio
async def test_live_run_and_sftp_round_trip(trace_cluster, trace_profile, tmp_path):
    fc = trace_cluster
    store = HostKeyStore(tmp_path / "known_hosts")
    async with SSHTransport(trace_profile, store=store, connect_timeout=15) as t:
        assert t.role == "login" and t.port == fc.port and t.max_sessions == trace_profile.ssh_max_exec
        res = await t.run("echo USER=$USER; sinfo -h -o %P")
        assert res.ok and "USER=wxu2" in res.stdout and "batch" in res.stdout.split(), res.stderr
        # non-idempotent commands run exactly the same way when nothing goes wrong
        res = await t.run("echo once", idempotent=False, timeout=30)
        assert res.ok and res.stdout.strip() == "once"
        # exit status and stderr are separated
        res = await t.run("echo err 1>&2; exit 3")
        assert res.returncode == 3 and res.stderr.strip() == "err" and res.stdout == ""

        # SFTP round trip through the chroot'ed home, cached client identity
        sftp = await t.sftp()
        assert sftp is await t.sftp()
        local = tmp_path / "in.txt"
        local.write_bytes(b"payload\r\nline2\n")
        await sftp.put(str(local), "in.txt")
        assert (Path(fc.home) / "in.txt").read_bytes() == b"payload\r\nline2\n"
        back = tmp_path / "back.txt"
        await sftp.get("in.txt", str(back))
        assert back.read_bytes() == b"payload\r\nline2\n"

        # run_with_stdin_file: binary tar streamed into `tar -xf -`
        tar = tmp_path / "up.tar"
        tar.write_bytes(_tar_bytes({"hello.txt": b"hi\n", "sub/run.sh": b"#!/bin/bash\necho x\n"}))
        res = await t.run_with_stdin_file("mkdir -p up && tar -xf - -C up", tar, 120)
        assert res.ok, res.stderr
        assert (Path(fc.home) / "up" / "hello.txt").read_bytes() == b"hi\n"
        assert (Path(fc.home) / "up" / "sub" / "run.sh").read_bytes().startswith(b"#!/bin/bash")

        # run_to_file: binary stdout to a local file, extracted with tarfile
        down = tmp_path / "down.tar"
        res = await t.run_to_file("tar -cf - -C up .", down, 120)
        assert res.ok and res.stdout == "", res.stderr
        with tarfile.open(down) as tf:
            names = {m.name.lstrip("./") for m in tf.getmembers() if m.isfile()}
            assert names == {"hello.txt", "sub/run.sh"}
            assert tf.extractfile("./hello.txt").read() == b"hi\n"

        # forced drop: the next idempotent run reconnects, SFTP client is recreated
        fc.drop_connections()
        await asyncio.sleep(0.5)
        res = await t.run("cat in.txt")
        assert res.ok and res.stdout == "payload\r\nline2\n"
        sftp2 = await t.sftp()
        assert sftp2 is not sftp and "in.txt" in await sftp2.listdir(".")
        assert t.reconnects >= 0 and t.connected

    # host-key store: one [host]:port line for the fake server plus the side file
    text = store.path.read_text(encoding="utf-8")
    assert text.count(f"[{fc.host}]:{fc.port} ssh-ed25519 ") == 1
    assert store.seen_addrs(fc.host) and list(store.seen_addrs(fc.host)) == ["127.0.0.1"]
    assert len(t.hostkey_notices) == 1 and t.hostkey_notices[0].startswith(f"new host key for {fc.host} from 127.0.0.1")


@pytest.mark.asyncio
async def test_live_wrong_password_latches(trace_cluster, tmp_path):
    prof = make_profile(trace_cluster, "fake-bad")
    os.environ["SLURM_MCP_PASSWORD_FAKE_BAD"] = "not-the-password"
    try:
        t = SSHTransport(prof, store=HostKeyStore(tmp_path / "known_hosts"), connect_timeout=15)
        with pytest.raises(AuthFailed) as ei:
            await t.run("echo hi")
        assert t.auth_failed and ei.value.code == "E_AUTH" and "auth set fake-bad" in str(ei.value)
        with pytest.raises(AuthFailed):
            await t.run("echo hi")
        t.reset_auth()
        os.environ["SLURM_MCP_PASSWORD_FAKE_BAD"] = "pw"
        res = await t.run("echo hi")
        assert res.ok and res.stdout.strip() == "hi"
        await t.close()
    finally:
        os.environ.pop("SLURM_MCP_PASSWORD_FAKE_BAD", None)
