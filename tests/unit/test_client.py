"""Unit tests for slurm_mcp.slurm.client (design sections 6.1-6.3, 4 files) with a fake transport."""
from __future__ import annotations

import pytest

from unit.fake_transport import FakeSFTP, FakeTransport, framed_discovery, ok, profile_for
from slurm_mcp.clock import ClusterClock
from slurm_mcp.errors import SlurmMcpError
from slurm_mcp.helpers import bundle_sha8
from slurm_mcp.slurm.client import IncompleteProbe, SlurmClient, TickFailed, summarize_snapshot
from slurm_mcp.transport import CommandTimeout, ConnectionDropped

TICK_OUT = """::NOW 1756760000 tracevm01
::SQUEUE
615421|615421|615421|N/A|PENDING|batch|normal|N/A|N/A|1756750000|1-00:00:00|0:00|1200|(null)|N/A|slurm-mcp:j1:1:t-abc|/w/.slurm-mcp/jobs/j1/a1/job.sbatch|/w|Priority
::RC 0
::RESTARTS
615421|0|1|
::RC 0
::SACCT
615400|615400|COMPLETED|0:0|0:0|batch|normal|trace01|1756740000|1756741000|1756745000|4000|1440|cpu=8,gres/gpu=1|cpu=8|None|/w
::RC 0
::FILES
/w/.slurm-mcp/jobs/j1/a1|jobid|615421
/w/.slurm-mcp/jobs/j1/a1|status.json|{"v":2,"phase":"running","rc":null}
::CMDS
/ctrl/a3/cmds/001.rc|0
::END
"""


def _client(cluster: str = "trace", handlers=None, caps=None, **kw) -> tuple[SlurmClient, FakeTransport]:
    profile = profile_for(cluster, **kw)
    t = FakeTransport(profile, handlers)
    store = {"caps": caps or {}}
    c = SlurmClient(profile.name, t, None, ClusterClock(), lambda: store["caps"])
    c._store = store  # type: ignore[attr-defined]
    return c, t


# --- discovery -------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discover_trace_fixture():
    c, t = _client("trace", [("echo '::ENV'", ok(framed_discovery("trace")))])
    caps = await c.discover()
    assert caps["cluster"] == "trace" and caps["cluster_name"] == "trace"
    assert caps["cmd_timeout_s"] == 120 and caps["message_timeout_s"] == 30
    assert caps["min_job_age_s"] == 300 and caps["job_requeue"] is False and caps["preempt_mode"] == ["GANG", "REQUEUE"]
    assert caps["pending_cap"] == 8 and caps["pending_cap_part"] is None
    assert caps["home"] == "/trace/home/wxu2" and caps["user"] == "wxu2" and caps["default_account"] == "biosimmlab"
    assert caps["squeue_O_zero"] is True and caps["helper_sha8"] is None and caps["charges"] is False
    batch = caps["partitions"]["batch"]
    assert batch["gres_types"] == {"a40": 29} and batch["gres_type_list"] == ["a40"] and batch["has_gpu"]
    assert batch["qos_candidates"] == ["normal"] and batch["accessible"] and batch["charge"] == "free"
    assert batch["limits"]["max_wall_s"] == 2 * 86400 and batch["limits"]["max_nodes"] is None
    assert caps["qos_candidates"]["batch"] == ["normal"] and caps["qos_for_partition"] == {}
    assert caps["assoc"]["qos_list"] == ["batchpartition", "cpuonly-debug-qos", "normal", "prioritypartition"]
    assert caps["tools"]["jq"] is True and caps["tools"]["rsync"] is True and "setsid" not in caps["tools"]
    assert [r["role"] for r in caps["df"]] == ["home"] and caps["df"][0]["used_pct"] == 1
    assert caps["su_balance"] is None and caps["epoch_format"] is False
    assert t.calls[0]["command"].startswith("export SLURM_TIME_FORMAT=%s LC_ALL=C")


@pytest.mark.asyncio
async def test_discover_bridges2_fixture_and_charges():
    c, t = _client("bridges2", [("echo '::ENV'", ok(framed_discovery("bridges2", helper="deadbeef")))])
    caps = await c.discover()
    assert caps["cmd_timeout_s"] == 260 and caps["pending_cap_part"] == 18 and caps["pending_cap"] is None
    assert caps["job_requeue"] is True and caps["comment_stored"] is True and caps["helper_sha8"] == "deadbeef"
    assert caps["scheduler_parameters"]["kill_invalid_depend"] is True
    gs = caps["partitions"]["GPU-shared"]
    assert "h100-80" in gs["gres_types"] and gs["qos_candidates"][0] == "gpu"
    assert caps["partitions"]["RM-shared"]["qos_candidates"][0] == "low"
    assert caps["partitions"]["GPU-small"]["limits"] == {
        "max_wall_s": 8 * 3600, "max_jobs_pu": 2, "max_submit_pu": 10, "max_tres_pj": {"gres/gpu": 16.0},
        "max_nodes": None, "max_cpus_node": 40, "max_mem_mb_node": 515000}
    assert caps["charges"] is True
    assert gs["charge"] == {"unit": "gpu:h100-80", "su_per_unit_h": 2.0}
    assert caps["partitions"]["RM-shared"]["charge"] == {"unit": "cpu", "su_per_unit_h": 1.0}
    assert {r["role"] for r in caps["df"]} == {"home", "remote_root"}


@pytest.mark.asyncio
async def test_discover_design_env_and_balance_updates_clock():
    env = "/jet/home/wxu7|wxu7|br012.ib.bridges2.psc.edu|/ocean/projects/mch250030p/wxu7|||1756760000|-0400|mch250030p"
    balance = ["Project: mch250030p", "  12,345 / 20,000 SU left"]
    c, t = _client("bridges2", [("echo '::ENV'", ok(framed_discovery("bridges2", env_line=env, balance=balance)))])
    caps = await c.discover()
    assert caps["tz_offset_s"] == -14400 and caps["remote_now"] == 1756760000 and caps["group"] == "mch250030p"
    assert caps["balance"] == {"left": 12345.0, "total": 20000.0} and caps["su_balance"] == 12345.0
    assert c.clock.synced and c.clock.tz_offset_s == -14400 and abs(c.clock.remote_now() - 1756760000) < 5
    roles = {r["path"]: r["role"] for r in caps["df"]}
    assert roles["/jet"] == "home" and roles["/ocean"] == "remote_root"


@pytest.mark.asyncio
async def test_discover_incomplete_probe_raises():
    c, t = _client("trace", [("echo '::ENV'", ok("::ENV\nx|y\n::VERSION\nslurm 22.05.11\n"))])
    with pytest.raises(IncompleteProbe):
        await c.discover()


# --- tick / snapshot -----------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tick_parses_every_section_and_updates_clock():
    c, t = _client("trace", [("::SQUEUE", ok(TICK_OUT))], caps={"squeue_O_zero": True})
    obs = await c.tick([615400, 615421], ["/w/.slurm-mcp/jobs/j1/a1"], ["/ctrl/a3/cmds/001.rc"])
    assert obs["now"] == 1756760000 and obs["host"] == "tracevm01" and c.clock.remote_now() >= 1756760000
    row = obs["squeue"][0]
    assert row["slurm_id"] == 615421 and row["job_state"].value == "SUBMITTED" and row["submit_ts"] == 1756750000
    assert row["comment"] == "slurm-mcp:j1:1:t-abc" and row["command"].endswith("/job.sbatch")
    assert obs["restarts"] == {615421: {"restarts": 0, "requeue": 1}}
    cur = obs["sacct"][615400]["current"]
    assert cur["job_state"].value == "COMPLETED" and cur["end_ts"] == 1756745000 and obs["sacct"][615400]["incarnations"] == 1
    assert obs["files"]["/w/.slurm-mcp/jobs/j1/a1"]["jobid"] == "615421"
    assert obs["files"]["/w/.slurm-mcp/jobs/j1/a1"]["status.json"]["phase"] == "running"
    assert obs["cmds"] == {"/ctrl/a3/cmds/001.rc": 0}
    assert obs["healthy"] is True and obs["rc"]["SQUEUE"] == 0
    cmd = t.calls[0]["command"]
    assert "::RESTARTS" in cmd and "sacct -n -P -X -D -j 615400,615421" in cmd and t.calls[0]["idempotent"] is True


@pytest.mark.asyncio
async def test_tick_failures_are_discarded():
    c, t = _client("trace", [("::SQUEUE", ok(TICK_OUT.replace("::END\n", "")))])
    with pytest.raises(IncompleteProbe):
        await c.tick([])
    c, t = _client("trace", [("::SQUEUE", ok(TICK_OUT.replace("::RC 0\n::RESTARTS", "::RC 1\n::RESTARTS", 1)))])
    with pytest.raises(TickFailed):
        await c.tick([])


@pytest.mark.asyncio
async def test_tick_tolerates_now_without_host():
    text = TICK_OUT.replace("::NOW 1756760000 tracevm01", "::NOW 1756760000 ")
    c, t = _client("trace", [("::SQUEUE", ok(text))])
    obs = await c.tick([])
    assert obs["now"] == 1756760000 and obs["host"] == ""


SNAP_OUT = """::NODES
batch|idle|gpu:a40:1|0/128/0/128
batch|alloc|gpu:a40:1|128/0/0/128
cpuonly|idle|(null)|0/96/0/96
::RC 0
::PD
    242 batch|N/A|N/A|
      3 batch|gres:gpu:a40:1|N/A|
     23 cpuonly|N/A|N/A|
::RC 0
::R
     30 batch|N/A|N/A|
::RC 0
::MINE
615421|batch|N/A|gres:gpu:a40|1200|N/A|Priority|
::RESV
ReservationName=maint_1 StartTime=1756800000 EndTime=1756900000 Duration=1-03:46:40 Nodes=trace[01-29] NodeCnt=29 CoreCnt=3712 Features=(null) PartitionName=batch Flags=MAINT,SPEC_NODES TRES=cpu=3712 Users=(null) Groups=(null) Accounts=root Licenses=(null) State=INACTIVE BurstBuffer=(null) Watts=n/a MaxStartDelay=(null)
::END
"""


@pytest.mark.asyncio
async def test_snapshot_summary_and_demand_classification():
    caps = {"squeue_O_zero": True, "partitions": {"batch": {"name": "batch", "gres_types": {"a40": 29}, "has_gpu": True},
                                                  "cpuonly": {"name": "cpuonly", "gres_types": {}, "has_gpu": False}}}
    c, t = _client("trace", [("::NODES", ok(SNAP_OUT))], caps=caps)
    snap = await c.snapshot()
    assert "squeue -h -t PD -O" in t.calls[0]["command"]
    p = snap["partitions"]
    assert p["batch"]["nodes"] == {"idle": 1, "mix": 0, "alloc": 1, "other": 0, "total": 2}
    assert p["batch"]["idle_gres"] == {"a40": 1}
    assert p["batch"]["pending"] == {None: 242, "a40": 3} and p["batch"]["pending_total"] == 245
    assert p["batch"]["running"] == {None: 30}
    assert p["cpuonly"]["pending"] == {"cpu": 23}
    assert snap["mine"][0]["slurm_id"] == 615421 and snap["mine"][0]["start_ts"] is None
    assert snap["resv"][0]["maintenance"] and snap["resv"][0]["start_ts"] == 1756800000
    assert isinstance(snap["ts"], int) and "fetched_local" in snap
    assert summarize_snapshot({"nodes": [], "pd": [], "r": []}, caps) == {}


# --- submit / test-only / control -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_ok_err_and_ambiguous():
    caps = {"helper_sha8": "0123abcd"}
    c, t = _client("trace", [("submit.sh", ok("JOBID 615500\nsbatch: warning: something\n"))], caps=caps)
    out = await c.submit("/w", "/w/.slurm-mcp/jobs/j1/a1", "t-abc", ["-p", "batch", "--parsable"])
    assert out["status"] == "ok" and out["job_id"] == 615500 and "warning" in out["stderr"]
    cmd = t.calls[0]["command"]
    assert "bash /trace/group/biosimmlab/wxu2/.slurm-mcp/bin/0123abcd/submit.sh /w/.slurm-mcp/jobs/j1/a1 t-abc -- -p batch --parsable" in cmd
    assert cmd.startswith("export SLURM_TIME_FORMAT=%s LC_ALL=C; cd /w &&") and t.calls[0]["idempotent"] is False
    assert cmd.endswith(" /w/.slurm-mcp/jobs/j1/a1/job.sbatch")

    c, t = _client("trace", [("submit.sh", ok("ERR 1\nsbatch: error: invalid partition specified: nope\n"))], caps=caps)
    out = await c.submit("/w", "/c", "t", [])
    assert out["status"] == "err" and out["code"] == "E_PARTITION" and out["rc"] == 1

    c, t = _client("trace", [("submit.sh", ok("ERR 3\nlock timeout\n"))], caps=caps)
    assert (await c.submit("/w", "/c", "t", []))["status"] == "ambiguous"

    c, t = _client("trace", [("submit.sh", CommandTimeout("x", 120, ambiguous=True, stdout="partial"))], caps=caps)
    out = await c.submit("/w", "/c", "t", [])
    assert out["status"] == "ambiguous" and "timeout" in out["error"] and out["raw"] == "partial"

    c, t = _client("trace", [("submit.sh", ConnectionDropped("x", "connection lost", ambiguous=True))], caps=caps)
    assert (await c.submit("/w", "/c", "t", []))["status"] == "ambiguous"


def test_helper_bin_dir_defaults_to_packaged_bundle():
    c, _ = _client("trace")
    assert c.helper_bin_dir() == f"/trace/group/biosimmlab/wxu2/.slurm-mcp/bin/{bundle_sha8()}"
    c2, _ = _client("trace", remote_root=None)
    assert c2.helper_bin_dir("abcdef01") == "$HOME/.slurm-mcp/bin/abcdef01"


@pytest.mark.asyncio
async def test_test_only_success_failure_and_timeout():
    est = "::T1\nsbatch: Job 615600 to start at 2026-09-01T23:40:00 using 64 processors on nodes trace03 in partition batch\n::RC 0\n::END\n"
    c, t = _client("trace", [("sbatch --test-only", ok(est))])
    c.clock.tz_offset_s = -14400
    out = await c.test_only("/w", ["-p", "batch", "--parsable", "--comment=x", "--hold"], "/c/job.sbatch")
    assert out["ok"] and out["partition"] == "batch" and out["est_start_ts"] == c.clock.to_epoch("2026-09-01T23:40:00")
    cmd = t.calls[0]["command"]
    assert "--parsable" not in cmd and "--comment" not in cmd and "--hold" not in cmd and "2>&1" in cmd

    fail = "::T2\nsbatch: error: QOSMaxWallDurationPerJobLimit\nallocation failure: Job violates accounting/QOS policy\n::RC 1\n::END\n"
    c, t = _client("trace", [("sbatch --test-only", ok(fail))])
    out = await c.test_only("/w", [], "/c/job.sbatch", section="T2")
    assert out["ok"] is False and out["code"] == "E_QOS_MAXWALL" and "QOS policy" in out["reason"]

    c, t = _client("trace", [("sbatch --test-only", CommandTimeout("x", 120, ambiguous=False))])
    out = await c.test_only("/w", [], "/c/job.sbatch")
    assert out["ok"] is False and out["timed_out"] is True


@pytest.mark.asyncio
async def test_control_commands_and_show_job():
    c, t = _client("trace", [
        ("scancel --signal=TERM --full 1 2", ok("")), ("scancel 3", ok("")), ("scontrol hold 4", ok("")),
        ("scontrol release 4", ok("")), ("scontrol requeue 5", ok("")),
        ("scontrol update JobId=6 Dependency=afterok:7", ok("")),
        ("scontrol -o show job 99", ok("", 1, "slurm_load_jobs error: Invalid job id specified\n")),
        ("scontrol -o show job 615411", ok("JobId=615411 JobName=wobl JobState=RUNNING Reason=None Dependency=(null) "
                                           "Requeue=1 Restarts=0 StartTime=1756750000 EndTime=1756836400 NodeList=trace03 "
                                           "BatchHost=trace03 StdOut=/w/logs/wobl_615411.out StdErr=/w/logs/wobl_615411.err "
                                           "WorkDir=/w Command=/w/train_wobl.job Comment=(null) TresPerJob=gres:gpu:a40\n")),
        ("squeue --me -h -o '%A|%T'", ok("1|PENDING\n2|RUNNING\n")),
    ])
    out = await c.cancel([1, 2], signal="TERM", full=True)
    assert out["ok"] and t.calls[-1]["idempotent"] is False and out["errors"] == []
    assert (await c.cancel([3]))["ok"] and t.calls[-1]["idempotent"] is True
    assert (await c.hold([4]))["ok"] and (await c.release([4]))["ok"] and (await c.requeue([5]))["ok"]
    assert (await c.update_dependency(6, ["afterok:7"]))["ok"]
    assert await c.show_job(99) is None
    info = await c.show_job(615411)
    assert info["job_state"] == "RUNNING" and info["std_out"] == "/w/logs/wobl_615411.out" and info["start_time_ts"] == 1756750000
    assert info["tres_per_job"] == {"type": "a40", "count": 1}
    assert await c.recheck_pending() == {1: "PENDING", 2: "RUNNING"}
    c, t = _client("trace", [("squeue --me -h -o '%A|%T'", ok("", 1, "slurm_load_jobs error: Socket timed out"))])
    assert await c.recheck_pending() is None


@pytest.mark.asyncio
async def test_backfill_history_rows():
    text = "615300|batch|normal|billing=64,cpu=64,gres/gpu:a40=1,mem=512G,node=1|1756000000|1756003600|COMPLETED\n" \
           "615301|batch|normal|cpu=8|1756010000|Unknown|CANCELLED by 1\n"
    c, t = _client("trace", [("sacct -nP -X -u", ok(text))])
    rows = await c.backfill_history()
    assert len(rows) == 2 and rows[0]["gres_type"] == "a40" and rows[0]["submit_ts"] == 1756000000
    assert rows[0]["start_ts"] == 1756003600 and rows[1]["start_ts"] is None
    c, t = _client("trace", [("sacct -nP -X -u", ok("", 1, "sacct: error"))])
    assert await c.backfill_history() == []


# --- files -----------------------------------------------------------------------------------------------------

def test_read_command_modes():
    c, _ = _client("trace")
    tail = c.read_command("/w/o ut.txt")
    assert "tail -n 100 -- '/w/o ut.txt' | head -c 12001" in tail and tail.startswith("export SLURM_TIME_FORMAT")
    assert "head -n 5 -- " in c.read_command("/w/x", head_lines=5, max_chars=100)
    assert "grep -n -E -m 200 -- 'err.*or' " in c.read_command("/w/x", grep="err.*or")
    assert "tail -c +101 -- /w/x | head -c 51" in c.read_command("/w/x", offset=100, max_chars=50)
    assert 'cat "$HOME/x"' not in c.read_command("$HOME/x") and '"$HOME/x"' in c.read_command("$HOME/x")


@pytest.mark.asyncio
async def test_read_file_parses_frames_truncation_and_errors():
    body = "::SIZE 2000\n::TEXT\n" + "x" * 101 + "\n::END\n"
    c, t = _client("trace", [("tail -n", ok(body))])
    out = await c.read_file("/w/x", max_chars=100)
    assert out["size"] == 2000 and out["truncated"] is True and len(out["text"]) == 100 and out["next_offset"] == 1900
    c, t = _client("trace", [("tail -c +11", ok("::SIZE 30\n::TEXT\nabcde\n::END\n"))])
    out = await c.read_file("/w/x", offset=10, max_chars=100)
    assert out["text"] == "abcde" and out["truncated"] is False and out["next_offset"] == 15
    c, t = _client("trace", [("tail -c +26", ok("::SIZE 29\n::TEXT\nlast\n::END\n"))])
    out = await c.read_file("/w/x", offset=25, max_chars=100)
    assert out["text"] == "last" and out["next_offset"] is None
    c, t = _client("trace", [("tail -n", ok("::MISSING\n::END\n"))])
    with pytest.raises(SlurmMcpError) as e:
        await c.read_file("/w/missing")
    assert e.value.code == "E_INVALID_SPEC" and "/w/missing" in str(e.value)
    c, t = _client("trace", [("tail -n", ok("::ISDIR\n::END\n"))])
    with pytest.raises(SlurmMcpError):
        await c.read_file("/w")


@pytest.mark.asyncio
async def test_write_file_ls_mkdirs_and_deploy_helpers():
    sftp = FakeSFTP()
    profile = profile_for("trace")
    t = FakeTransport(profile, sftp=sftp)
    caps = {"home": "/trace/home/wxu2"}
    c = SlurmClient("trace", t, None, ClusterClock(), lambda: caps)
    out = await c.write_file("/w/dir/run.sh", "#!/bin/bash\r\necho hi\r\n", executable=True)
    assert out == {"path": "/w/dir/run.sh", "bytes": len("#!/bin/bash\necho hi\n"), "warnings": ["crlf_normalized"]}
    assert sftp.files["/w/dir/run.sh"] == b"#!/bin/bash\necho hi\n" and sftp.modes["/w/dir/run.sh"] == 0o755
    assert "/w/dir" in sftp.dirs and sftp.renames[0][0].startswith("/w/dir/run.sh.tmp-")
    await c.write_file("/w/dir/run.sh", "echo more\n", mode="append")
    assert sftp.files["/w/dir/run.sh"].endswith(b"echo hi\necho more\n")
    with pytest.raises(SlurmMcpError) as e:
        await c.write_file("/w/big", "x" * (1024 * 1024 + 1))
    assert e.value.code == "E_TOO_MANY_BYTES"
    with pytest.raises(SlurmMcpError):
        await c.write_file("/w/x", "a\x00b")
    await c.write_file("$HOME/notes.txt", "n")
    assert "/trace/home/wxu2/notes.txt" in sftp.files
    await c.mkdirs(["/w/out", ""])
    assert "/w/out" in sftp.dirs
    listing = await c.ls("/w/dir")
    assert listing["entries"] == [{"name": "run.sh", "type": "file", "size": len(sftp.files["/w/dir/run.sh"]),
                                   "mtime_ts": int(sftp.mtimes["/w/dir/run.sh"])}] and listing["truncated"] is False
    await c.write_file("/w/dir/a.out", "1")
    await c.write_file("/w/dir/b.out", "22")
    assert [e["name"] for e in (await c.ls("/w/dir", glob="*.out", sort="size"))["entries"]] == ["b.out", "a.out"]
    assert (await c.ls("/w/dir", max_entries=1))["truncated"] is True
    assert (await c.ls("/w/dir/a.out"))["entries"][0]["name"] == "a.out"
    with pytest.raises(SlurmMcpError):
        await c.ls("/nope")
    sha = await c.deploy_helpers()
    assert sha == bundle_sha8()
    root = "/trace/group/biosimmlab/wxu2/.slurm-mcp/bin"
    assert sftp.files[f"{root}/VERSION"] == (sha + "\n").encode()
    for name in ("wrap.sh", "submit.sh", "alloc-agent.sh"):
        assert sftp.modes[f"{root}/{sha}/{name}"] == 0o755 and sftp.files[f"{root}/{sha}/{name}"].startswith(b"#!/bin/bash")
    t2 = FakeTransport(profile_for("trace", remote_root=None), sftp=sftp)
    c2 = SlurmClient("trace", t2, None, ClusterClock(), lambda: caps)
    await c2.deploy_helpers()
    assert f"/trace/home/wxu2/.slurm-mcp/bin/VERSION" in sftp.files
