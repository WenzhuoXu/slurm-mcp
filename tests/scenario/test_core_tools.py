"""Scenario tests for the phase-2 core on the in-process fake clusters: discovery, helper deploy, the Service
operations behind clusters/cluster_status/run_command/remote_*/configure and a submit.sh round trip."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import make_profile
from slurm_mcp.config import ClusterProfile
from slurm_mcp.errors import SlurmMcpError
from slurm_mcp.events import EventBus
from slurm_mcp.helpers import bundle_sha8
from slurm_mcp.models import ClusterStatusResult, ClustersResult, RunCommandResult
from slurm_mcp.service import ClusterRegistry, Service
from slurm_mcp.slurm.discovery import bootstrap, ensure_helpers
from slurm_mcp.store import Store
from sshd_harness import SSH_PASSWORD, SSH_USER

JOB_SCRIPT = """#!/bin/bash
#SBATCH -J t1
#SBATCH --open-mode=append
#FAKESLURM duration=60 exit=0
echo hello from $SLURM_JOB_ID
"""


def posix_profile(fc, name: str) -> ClusterProfile:
    """Like conftest.make_profile but with the POSIX spelling of the fake home (what a real cluster reports)."""
    os.environ["SLURM_MCP_PASSWORD_" + name.upper().replace("-", "_")] = SSH_PASSWORD
    return ClusterProfile(name=name, host=fc.host, user=SSH_USER, port=fc.port, auth="password",
                          remote_root=fc.env["HOME"] + "/work")


class World:
    def __init__(self, tmp_path: Path, profiles: dict[str, ClusterProfile]) -> None:
        self.store = Store(tmp_path / "state.db")
        self.events = EventBus(self.store, session_id="sess1")
        self.registry = ClusterRegistry(profiles, self.store)
        self.service = Service(self.store, self.events, self.registry, "sess1")

    async def __aenter__(self) -> "World":
        await self.service.acquire_lease()
        await self.service.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.service.stop()
        await self.registry.close()
        self.store.close()


@pytest.mark.asyncio
async def test_discover_on_both_fake_clusters(trace_cluster, bridges2_cluster, tmp_path):
    profiles = {"fake-trace": posix_profile(trace_cluster, "fake-trace"),
                "fake-b2": posix_profile(bridges2_cluster, "fake-b2")}
    async with World(tmp_path, profiles) as w:
        tc = await w.registry.client("fake-trace").discover()
        assert tc["cluster"] == "fake-trace" and tc["cluster_name"] == "trace" and tc["slurm_version"] == "22.05.11"
        assert tc["cmd_timeout_s"] == 120 and tc["min_job_age_s"] == 300 and tc["pending_cap"] == 8
        assert tc["home"] == trace_cluster.env["HOME"] and tc["user"] == "wxu2" and tc["default_account"] == "biosimmlab"
        assert tc["squeue_O_zero"] is True and tc["helper_sha8"] is None and tc["epoch_format"] is False
        assert tc["partitions"]["batch"]["gres_type_list"] == ["a40"] and tc["partitions"]["batch"]["qos_candidates"] == ["normal"]
        assert tc["partitions"]["cpuonly"]["qos_candidates"] == []          # AllowQos=ALL + assoc default
        assert tc["sinfo"]["batch"]["nodes"] == 4 and tc["tools"]["sacct"] is True
        assert {r["role"] for r in tc["df"]} >= {"home"} and tc["su_balance"] is None
        assert w.registry.clock("fake-trace").synced
        bc = await w.registry.client("fake-b2").discover()
        assert bc["cluster_name"] == "bridges2" and bc["job_requeue"] is True and bc["pending_cap_part"] == 18
        assert bc["partitions"]["GPU-shared"]["qos_candidates"][0] == "gpu"
        assert bc["partitions"]["RM-shared"]["qos_candidates"][0] == "low"
        assert "h100-80" in bc["partitions"]["GPU-shared"]["gres_types"] and bc["comment_stored"] is True


@pytest.mark.asyncio
async def test_deploy_helpers_is_idempotent(trace_cluster, tmp_path):
    fc = trace_cluster
    profile = posix_profile(fc, "fake-trace")
    async with World(tmp_path, {"fake-trace": profile}) as w:
        client = w.registry.client("fake-trace")
        caps = await bootstrap(client, profile, w.store)
        sha = await ensure_helpers(client, profile, caps, w.store)
        assert sha == bundle_sha8() and caps["helper_sha8"] == sha
        bin_dir = Path(fc.home) / "work" / ".slurm-mcp" / "bin"
        assert (bin_dir / "VERSION").read_text() == sha + "\n"
        wrap = bin_dir / sha / "wrap.sh"
        assert wrap.read_bytes().startswith(b"#!/bin/bash") and not list(bin_dir.glob("*.tmp-*"))
        stamp = wrap.stat().st_mtime_ns
        assert await client.helper_version() == sha
        # a fresh discovery sees the VERSION file; a caps entry without it re-checks and does not re-upload
        caps2 = await bootstrap(client, profile, w.store, refresh=True)
        assert caps2["helper_sha8"] == sha
        caps2["helper_sha8"] = None
        assert await ensure_helpers(client, profile, caps2) == sha and wrap.stat().st_mtime_ns == stamp
        res = await client.run(f"bash {fc.env['HOME']}/work/.slurm-mcp/bin/{sha}/submit.sh /nonexistent tok --",
                               login_shell=False)
        assert res.stdout.startswith("ERR 2")


@pytest.mark.asyncio
async def test_service_clusters_and_cluster_status(trace_cluster, tmp_path):
    profile = posix_profile(trace_cluster, "fake-trace")
    async with World(tmp_path, {"fake-trace": profile}) as w:
        svc = w.service
        before = await svc.clusters()
        assert isinstance(before, ClustersResult) and before.session_id == "sess1"
        row = before.clusters[0]
        assert row.name == "fake-trace" and row.connected is False and row.reachable is None and row.monitor == "self"
        assert any("not discovered" in x for x in row.warnings) and "1 cluster(s)" in before.summary
        after = await svc.clusters(refresh=True)
        row = after.clusters[0]
        assert row.connected and row.reachable and not row.auth_failed and row.warnings == [] or all(
            "host key" in x for x in row.warnings)
        assert row.tracked_jobs.running == 0 and row.su_balance is None and row.quota and row.quota[0].used_pct is not None
        assert after.unread_events == 0 and "connected" in after.summary
        st = await svc.cluster_status("fake-trace")
        assert isinstance(st, ClusterStatusResult) and st.cluster == "fake-trace" and st.slurm_version == "22.05.11"
        names = [p.name for p in st.partitions]
        assert names[:4] == ["batch", "biosimmlab", "cpuonly", "cpuonly-debug"] and st.caps_age_s is not None
        batch = st.partitions[0]
        assert batch.nodes.idle == 4 and batch.nodes.total == 4 and batch.gres_types == ["a40"]
        assert batch.qos == "normal" and batch.charge == "free" and batch.preempt_mode == "REQUEUE"
        assert batch.max_wall_s == 2 * 86400 and batch.limits.max_wall_s == 2 * 86400 and batch.priority_tier == 1
        assert "idle GPU nodes batch=4 biosimmlab=4" in st.summary and "0 pending jobs" in st.summary
        assert st.helper_version is None
        q = await svc.cluster_status("fake-trace", detail="queue")
        assert q.queue == [] and q.partitions
        t = await svc.cluster_status("fake-trace", detail="targets")
        keys = {r.target: r for r in t.targets or []}
        assert "fake-trace:batch:a40@normal" in keys and keys["fake-trace:batch:a40@normal"].max_pending == 8
        assert keys["fake-trace:batch:a40@normal"].enabled is True
        full = await svc.cluster_status("fake-trace", detail="full")
        assert full.config and full.config["ClusterName"] == "trace" and full.config["cmd_timeout_s"] == 120
        s = await svc.cluster_status("fake-trace", detail="summary")
        assert s.partitions == [] and s.summary
        with pytest.raises(SlurmMcpError) as e:
            await svc.cluster_status("fake-trace", detail="nope")
        assert e.value.code == "E_INVALID_SPEC"
        with pytest.raises(SlurmMcpError) as e:
            await svc.cluster_status("mars")
        assert "unknown cluster" in str(e.value)


@pytest.mark.asyncio
async def test_run_command_and_refusals(trace_cluster, tmp_path):
    profile = posix_profile(trace_cluster, "fake-trace")
    async with World(tmp_path, {"fake-trace": profile}) as w:
        svc = w.service
        r = await svc.run_command("fake-trace", "echo hi; echo err >&2")
        assert isinstance(r, RunCommandResult) and r.rc == 0 and r.stdout_tail == "hi\n" and r.stderr_tail == "err\n"
        assert "rc=0" in r.summary and r.truncated is False and r.next is None
        r = await svc.run_command("fake-trace", "sbatch -p nope --wrap hostname")
        assert r.rc == 1 and "Invalid partition" in r.stderr_tail and "Invalid partition" in r.summary
        home = trace_cluster.env["HOME"]
        r = await svc.run_command("fake-trace", "pwd", cwd=home)
        assert r.stdout_tail.strip() == home
        r = await svc.run_command("fake-trace", "seq 1 2000", max_chars=50)
        assert r.truncated and len(r.stdout_tail) == 50 and r.stdout_tail.endswith("2000\n")
        for bad in ("cat <<EOF\nx\nEOF", "x" * 4001):
            with pytest.raises(SlurmMcpError) as e:
                await svc.run_command("fake-trace", bad)
            assert e.value.code == "E_CMD_TOO_LONG"
        with pytest.raises(SlurmMcpError):
            await svc.run_command("fake-trace", "   ")
        r = await svc.run_command("fake-trace", "sleep 5; echo late", timeout_s=1)
        assert r.rc is None and "timed out" in r.summary


@pytest.mark.asyncio
async def test_remote_write_read_ls_round_trip(trace_cluster, tmp_path):
    fc = trace_cluster
    profile = posix_profile(fc, "fake-trace")
    home = fc.env["HOME"]
    async with World(tmp_path, {"fake-trace": profile}) as w:
        svc = w.service
        path = f"{home}/work/notes/a.txt"
        wr = await svc.remote_write("fake-trace", path, "line1\r\nline2\r\nerror: boom\r\n")
        assert wr.bytes == len("line1\nline2\nerror: boom\n") and "crlf_normalized" in wr.summary
        assert (Path(fc.home) / "work" / "notes" / "a.txt").read_bytes() == b"line1\nline2\nerror: boom\n"
        rd = await svc.remote_read("fake-trace", path)
        assert rd.text == "line1\nline2\nerror: boom\n" and rd.size == wr.bytes and rd.truncated is False
        rd = await svc.remote_read("fake-trace", path, tail_lines=1)
        assert rd.text == "error: boom\n" and "1 line(s)" in rd.summary
        rd = await svc.remote_read("fake-trace", path, head_lines=1)
        assert rd.text == "line1\n"
        rd = await svc.remote_read("fake-trace", path, grep="^err")
        assert rd.text == "3:error: boom\n"
        rd = await svc.remote_read("fake-trace", path, offset=0, max_chars=6)
        assert rd.text == "line1\n" and rd.next_offset == 6 and rd.truncated and rd.next == "remote_read(offset=6)"
        rd = await svc.remote_read("fake-trace", path, offset=6, max_chars=100)
        assert rd.text == "line2\nerror: boom\n" and rd.next_offset is None
        with pytest.raises(SlurmMcpError):
            await svc.remote_read("fake-trace", f"{home}/work/notes/missing.txt")
        ap = await svc.remote_write("fake-trace", path, "more\n", mode="append")
        assert ap.bytes == 5 and (await svc.remote_read("fake-trace", path, tail_lines=1)).text == "more\n"
        script = f"{home}/work/notes/run.sh"
        await svc.remote_write("fake-trace", script, "#!/bin/bash\necho ran $1\n", executable=True)
        r = await svc.run_command("fake-trace", f"bash {script} ok")
        assert r.rc == 0 and r.stdout_tail == "ran ok\n"
        ls = await svc.remote_ls("fake-trace", f"{home}/work/notes")
        assert [e.name for e in ls.entries] == ["a.txt", "run.sh"] and ls.entries[0].type == "file"
        assert ls.entries[0].size == wr.bytes + 5 and ls.entries[0].mtime_ts and "2 entries" in ls.summary
        ls = await svc.remote_ls("fake-trace", f"{home}/work/notes", glob="*.sh", sort="size")
        assert [e.name for e in ls.entries] == ["run.sh"]
        ls = await svc.remote_ls("fake-trace", f"{home}/work", max_entries=1)
        assert ls.entries and ls.entries[0].type == "dir"
        with pytest.raises(SlurmMcpError):
            await svc.remote_ls("fake-trace", f"{home}/nope")
        with pytest.raises(SlurmMcpError) as e:
            await svc.remote_ls("fake-trace", home, sort="weird")
        assert e.value.code == "E_INVALID_SPEC"


@pytest.mark.asyncio
async def test_configure_persists_and_validates(trace_cluster, tmp_path):
    profile = posix_profile(trace_cluster, "fake-trace")
    async with World(tmp_path, {"fake-trace": profile}) as w:
        svc = w.service
        cur = await svc.configure()
        assert cur.placement.objective == "balanced" and cur.notify.toast is True and "current policies" in cur.summary
        up = await svc.configure(placement={"objective": "fastest", "rebalance": {"min_gain_h": 2.5}},
                                 notify={"email": "me@x.org", "quiet_hours": [22, 7]})
        assert up.placement.objective == "fastest" and up.placement.rebalance.min_gain_h == 2.5
        assert up.placement.rebalance.interval_min == 10 and up.notify.email == "me@x.org"
        assert up.notify.quiet_hours == (22, 7) and "placement.objective" in up.summary
        assert svc.notify_policy_cached().email == "me@x.org"
        with pytest.raises(SlurmMcpError) as e:
            await svc.configure(placement={"objective": "fastest", "bogus": 1})
        assert e.value.code == "E_INVALID_SPEC"
        with pytest.raises(SlurmMcpError):
            await svc.configure(notify={"quiet_hours": [30, 1]})
    async with World(tmp_path, {"fake-trace": profile}) as w2:
        again = await w2.service.configure()
        assert again.placement.objective == "fastest" and again.notify.email == "me@x.org"


@pytest.mark.asyncio
async def test_client_submit_round_trip_through_helper(trace_cluster, tmp_path):
    fc = trace_cluster
    profile = posix_profile(fc, "fake-trace")
    home = fc.env["HOME"]
    async with World(tmp_path, {"fake-trace": profile}) as w:
        svc = w.service
        sha = await svc.helpers_ready("fake-trace")
        client = svc.client("fake-trace")
        assert client.helper_bin_dir() == f"{home}/work/.slurm-mcp/bin/{sha}"
        workdir = f"{home}/work/t1"
        ctrl = f"{home}/work/.slurm-mcp/jobs/j1/a1"
        await client.mkdirs([workdir, ctrl])
        await client.write_file(f"{ctrl}/job.sbatch", JOB_SCRIPT)
        args = ["-p", "batch", "--qos=normal", "-t", "00:10:00", "--gres=gpu:a40:1", "-o", f"{ctrl}/out.txt",
                "--comment=slurm-mcp:j1:1:t-abc", "--parsable"]
        est = await client.test_only(workdir, args, f"{ctrl}/job.sbatch")
        assert est["ok"] and est["partition"] == "batch" and isinstance(est["est_start_ts"], int)
        bad = await client.test_only(workdir, ["-p", "nope"], f"{ctrl}/job.sbatch")
        assert bad["ok"] is False and bad["code"] == "E_PARTITION"
        out = await client.submit(workdir, ctrl, "t-abc", args)
        assert out["status"] == "ok", out
        jid = out["job_id"]
        assert (Path(fc.home) / "work" / ".slurm-mcp" / "jobs" / "j1" / "a1" / "jobid").read_text().strip() == str(jid)
        again = await client.submit(workdir, ctrl, "t-abc", args)
        assert again["status"] == "ok" and again["job_id"] == jid          # submit.sh is idempotent per ctrl_dir
        info = await client.show_job(jid)
        assert info and info["job_name"] == "t1" and info["comment"] == "slurm-mcp:j1:1:t-abc"
        assert info["command"] == f"{ctrl}/job.sbatch" and info["work_dir"] == workdir
        assert await client.recheck_pending() == {jid: "PENDING"} or (await client.recheck_pending()) == {jid: "RUNNING"}
        obs = await client.tick([jid], [ctrl])
        assert obs["squeue"][0]["slurm_id"] == jid and obs["squeue"][0]["comment"] == "slurm-mcp:j1:1:t-abc"
        assert obs["files"][ctrl]["jobid"] == str(jid) and obs["healthy"]
        res = await client.cancel([jid])
        assert res["ok"] and res["errors"] == []
        fc.ctl("advance", "--seconds", "5")
        obs = await client.tick([jid])
        cur = obs["sacct"][jid]["current"]
        assert cur["job_state"].value == "CANCELLED"
        st = await svc.cluster_status("fake-trace", refresh=True)
        assert st.helper_version == sha
