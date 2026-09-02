"""End-to-end: the real SSHTransport against the in-process SSH server backed by fakeslurm."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from conftest import make_profile
from slurm_mcp import config
from slurm_mcp.transport import SSHTransport

SCRIPT = """#!/bin/bash
#SBATCH -p batch
#SBATCH --gpus=a40
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:30:00
#SBATCH -J e2e
#FAKESLURM duration=60 exit=0
echo hello from $SLURM_JOB_ID
"""


@pytest.mark.asyncio
async def test_transport_end_to_end(trace_cluster, trace_profile, mcp_home, tmp_path):
    fc = trace_cluster
    assert str(config.CONFIG_DIR) == mcp_home            # isolated home: trust-on-first-use lands here
    async with SSHTransport(trace_profile, connect_timeout=15) as t:
        res = await t.run("sinfo -h -o %P")
        assert res.ok and res.stdout.split() == ["batch", "cpuonly", "cpuonly-debug", "biosimmlab"], res.stderr

        res = await t.run("echo USER=$USER; pwd")
        assert res.ok and "USER=wxu2" in res.stdout

        res = await t.run("sbatch --parsable", input=SCRIPT)
        assert res.ok, res.stderr
        jid = int(res.stdout.strip())
        res = await t.run(f"scontrol show job {jid} -o")
        assert res.ok and f"JobId={jid} JobName=e2e" in res.stdout

        fc.ctl("advance", "--seconds", "5")
        res = await t.run(f"squeue -h -j {jid} -o '%T|%N'")
        assert res.stdout.strip().split("|") == ["RUNNING", "trace01"]

        res = await t.run("scontrol show job 1 -o")
        assert res.returncode == 1 and res.stderr.strip() == "slurm_load_jobs error: Invalid job id specified"

        # sftp upload / download through the chroot'ed home
        sftp = await t.sftp()
        local = tmp_path / "in.txt"
        local.write_bytes(b"payload\n")
        await sftp.put(str(local), "in.txt")
        assert (Path(fc.home) / "in.txt").read_bytes() == b"payload\n"
        res = await t.run("cat in.txt")
        assert res.stdout == "payload\n"
        back = tmp_path / "back.txt"
        await sftp.get("in.txt", str(back))
        assert back.read_bytes() == b"payload\n"
        listing = await sftp.listdir(".")
        assert "in.txt" in listing

        # forced connection drop: the server aborts the connection, the next run() reconnects
        fc.drop_connections()
        await asyncio.sleep(0.5)
        res = await t.run("squeue -h -o %i")
        assert res.ok and str(jid) in res.stdout.split()
        # a new SFTP client works after the reconnect too
        sftp2 = await t.sftp()
        assert "in.txt" in await sftp2.listdir(".")

        fc.ctl("advance", "--seconds", "100")
        res = await t.run(f"sacct -j {jid} -n -P -X --format=JobID,State,ExitCode")
        assert res.stdout.strip() == f"{jid}|COMPLETED|0:0"

    known_hosts = Path(mcp_home) / "known_hosts"
    assert known_hosts.exists() and f"[127.0.0.1]:{fc.port}" in known_hosts.read_text()
    assert os.environ.get("SLURM_MCP_PASSWORD_FAKE_TRACE") == "pw"


@pytest.mark.asyncio
async def test_transport_check_raises_and_timeout(trace_cluster, trace_profile):
    from slurm_mcp.transport import RemoteCommandError

    async with SSHTransport(trace_profile, connect_timeout=15) as t:
        with pytest.raises(RemoteCommandError):
            await t.run("sbatch -p nope --wrap hostname", check=True)
        res = await t.run("sbatch -p nope --wrap hostname")
        assert res.returncode == 1 and "Invalid partition name specified" in res.stderr
        res = await t.run("exit 3", login_shell=False)
        assert res.returncode == 3


@pytest.mark.asyncio
async def test_bridges2_profile_via_transport(bridges2_cluster, mcp_home):
    profile = make_profile(bridges2_cluster, "fake-b2")
    async with SSHTransport(profile, connect_timeout=15) as t:
        res = await t.run("sbatch --test-only -N 1 -n 1 -t 0:10:00 --wrap hostname")
        assert res.returncode == 1 and res.stderr == "allocation failure: Invalid qos specification\n"
        res = await t.run("scontrol show config | grep -E '^(ClusterName|JobRequeue|PreemptMode) '")
        assert res.ok and "ClusterName             = bridges2" in res.stdout and "JobRequeue              = 1" in res.stdout


LOGGED_SCRIPT = """#!/bin/bash
#SBATCH -p cpuonly
#SBATCH -n 1
#SBATCH -t 5
#SBATCH -J p
#SBATCH -o logs/%x_%j.out
#FAKESLURM duration=10 exit=0
echo out
"""


@pytest.mark.asyncio
async def test_job_paths_round_trip_through_ssh_and_sftp(trace_cluster, trace_profile, tmp_path):
    """The paths the fake reports (WorkDir/StdOut/StdErr/Command, squeue %Z/%o, sacct WorkDir) are the POSIX
    paths the remote shell and the SFTP server understand, so a 'fetch job output' flow works end to end."""
    fc = trace_cluster
    home = fc.env["HOME"]
    assert home.startswith("/c/")
    async with SSHTransport(trace_profile, connect_timeout=15) as t:
        res = await t.run("pwd; cd \"$HOME\" && pwd; echo $HOME")
        assert res.ok and res.stdout.split() == [home, home, home]          # one spelling everywhere
        res = await t.run("mkdir -p work/logs && cd work && sbatch --parsable", input=LOGGED_SCRIPT)
        assert res.ok, res.stderr
        jid = int(res.stdout.strip())
        res = await t.run(f"scontrol show job {jid} -o")
        kv = dict(tok.split("=", 1) for tok in res.stdout.split())
        assert kv["WorkDir"] == f"{home}/work" and kv["StdOut"] == kv["StdErr"] == f"{home}/work/logs/p_{jid}.out"
        assert kv["Command"] == "(null)"
        res = await t.run(f"squeue -h -j {jid} -o '%Z|%o'")
        assert res.stdout.strip() == f"{home}/work|(null)"
        res = await t.run(f"sacct -X -j {jid} -n -P --format=WorkDir")
        assert res.stdout.strip() == f"{home}/work"
        fc.ctl("advance", "--seconds", "30")
        # the sftp namespace is the shell's: absolute POSIX paths and relative-to-home both resolve
        sftp = await t.sftp()
        assert await sftp.stat(kv["StdOut"])
        assert (await sftp.stat(kv["WorkDir"])).type == 2   # directory
        assert await sftp.realpath(".") == home
        assert "logs" in await sftp.listdir("work") and f"p_{jid}.out" in await sftp.listdir(kv["WorkDir"] + "/logs")
        local = tmp_path / "fetched.out"
        await sftp.get(kv["StdOut"], str(local))
        assert local.exists()
        res = await t.run(f"cd {kv['WorkDir']!r} && pwd && cat {kv['StdOut']!r} && ls logs")
        assert res.ok and res.stdout.splitlines()[0] == f"{home}/work", res.stderr
        # a script submitted by path: Command is its absolute POSIX path
        await sftp.put(str(tmp_path / "fetched.out"), "dummy.txt")
        script = tmp_path / "job.sh"
        script.write_bytes(LOGGED_SCRIPT.encode())
        await sftp.put(str(script), "work/job.sh")
        res = await t.run("cd work && sbatch --parsable job.sh")
        assert res.ok, res.stderr
        jid2 = int(res.stdout.strip())
        res = await t.run(f"scontrol show job {jid2} -o")
        kv2 = dict(tok.split("=", 1) for tok in res.stdout.split())
        assert kv2["Command"] == f"{home}/work/job.sh" and kv2["WorkDir"] == f"{home}/work"
        assert (await sftp.stat(kv2["Command"])).size == len(LOGGED_SCRIPT)
        # missing output directory: the launch fails instead of being silently ignored
        res = await t.run("sbatch --parsable -o nodir/x_%j.out", input=LOGGED_SCRIPT)
        assert res.ok, res.stderr
        bad = int(res.stdout.strip())
        fc.ctl("advance", "--seconds", "5")
        res = await t.run(f"sacct -j {bad} -n -P -X --format=State,ExitCode")
        assert res.stdout.strip() == "FAILED|0:53"
        res = await t.run("ls nodir")
        assert res.returncode != 0
