"""Tests for the fakeslurm emulator: structural comparison against the captured fixtures plus behaviour."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES = TESTS_DIR / "fixtures"
sys.path.insert(0, str(TESTS_DIR / "fakeslurm"))
import fakeslurm  # noqa: E402

NOW = "2026-09-01T17:00:00"
SACCT_FMT = ("JobID,JobIDRaw,JobName,State,ExitCode,DerivedExitCode,Elapsed,ElapsedRaw,Start,End,Submit,Partition,"
             "Account,QOS,NodeList,AllocTRES,ReqTRES,MaxRSS,TotalCPU,Reason,WorkDir,Timelimit,TimelimitRaw,NCPUS,"
             "NNodes,Flags,SubmitLine")
SQUEUE_FMT = "%i|%j|%T|%P|%R|%M|%l|%D|%C|%b|%S|%V|%Q|%r|%N|%o|%Z|%u|%a|%q|%k|%e|%L"
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

TRACE_GPU_SCRIPT = """#!/bin/bash
#SBATCH -p batch
#SBATCH --gpus=a40
#SBATCH --ntasks-per-node=64
#SBATCH --mem=512G
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH -J wobl
#SBATCH -o logs/wobl_%j.out
#SBATCH -e logs/wobl_%j.err
#FAKESLURM duration=900 exit=0
echo hi
"""

B2_GPU_SCRIPT = """#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --account=mch250030p
#SBATCH --gres=gpu:h100-80:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --qos=gpu
#SBATCH -J t9c_mt
set -euo pipefail
python train.py
"""


def fixture(cluster: str, name: str) -> str:
    return (FIXTURES / cluster / name).read_text(encoding="utf-8")


class Fake:
    """Thin in-process driver around fakeslurm.run() with its own state file."""

    def __init__(self, tmp_path: Path, cluster: str, monkeypatch, start_jobid: int = 615442):
        self.state = tmp_path / cluster / "state.json"
        self.home = tmp_path / cluster / "home"
        self.home.mkdir(parents=True)
        (self.home / "logs").mkdir()
        monkeypatch.setenv("FAKESLURM_STATE", str(self.state))
        monkeypatch.delenv("FAKESLURM_NOW", raising=False)
        self.cluster = cluster
        rc, out, err = self("fakeslurm-ctl", "init", "--cluster", cluster, "--now", NOW, "--start-jobid", str(start_jobid))
        assert rc == 0, err
        self.user = self.dump()["user"]["name"]

    def __call__(self, *argv: str, stdin: str = ""):
        os.environ["FAKESLURM_STATE"] = str(self.state)   # tests may drive two clusters at once
        return fakeslurm.run(list(argv), stdin, cwd=str(self.home))

    def ok(self, *argv: str, stdin: str = "") -> str:
        rc, out, err = self(*argv, stdin=stdin)
        assert rc == 0, f"{argv} failed rc={rc}: {err}"
        return out

    def submit(self, *argv: str, script: str = TRACE_GPU_SCRIPT) -> int:
        out = self.ok("sbatch", "--parsable", *argv, stdin=script)
        return int(out.strip())

    def advance(self, seconds: int) -> None:
        self.ok("fakeslurm-ctl", "advance", "--seconds", str(seconds))

    def dump(self) -> dict:
        return json.loads(self.state.read_text(encoding="utf-8"))

    def job(self, jid: int) -> dict:
        return self.dump()["jobs"][str(jid)]

    def squeue_row(self, jid: int, fmt: str = SQUEUE_FMT) -> list[str] | None:
        for line in self.ok("squeue", "-h", "-t", "all", "-o", fmt).splitlines():
            f = line.split("|")
            if f[0] == str(jid):
                return f
        return None


@pytest.fixture
def trace(tmp_path, monkeypatch):
    return Fake(tmp_path, "trace", monkeypatch)


@pytest.fixture
def bridges2(tmp_path, monkeypatch):
    return Fake(tmp_path, "bridges2", monkeypatch, start_jobid=44809480)


@pytest.fixture(params=["trace", "bridges2"])
def any_cluster(request, tmp_path, monkeypatch):
    return Fake(tmp_path, request.param, monkeypatch)


# ------------------------------------------------------------------------------------------------
# structural comparisons against the fixtures
# ------------------------------------------------------------------------------------------------
def test_sinfo_partitions_shape(any_cluster):
    fake = any_cluster
    cmd = json.loads(fixture(fake.cluster, "index.json"))["sinfo_partitions"]["cmd"]
    fmt = re.search(r'-o "([^"]+)"', cmd).group(1)
    out = fake.ok("sinfo", "-h", "-o", fmt)
    fx_lines = fixture(fake.cluster, "sinfo_partitions.out").splitlines()
    lines = out.splitlines()
    assert lines
    for line in lines:
        f = line.split("|")
        assert len(f) == len(fx_lines[0].split("|")) == 10
        assert f[1] == "up"
        assert re.fullmatch(r"\d+/\d+/\d+/\d+", f[5]), f
        assert re.fullmatch(r"\d+/\d+/\d+/\d+", f[8]), f
        assert f[6] == "(null)" or f[6].startswith("gpu:")
    fx_parts = {ln.split("|")[0] for ln in fx_lines}
    assert {ln.split("|")[0] for ln in lines} <= fx_parts
    if fake.cluster == "bridges2":
        assert any(ln.startswith("RM*|") for ln in lines)   # default partition marked with '*'
        assert any(ln.split("|")[4] == "drain" for ln in lines)
    else:
        assert any(ln.split("|")[4] == "down*" for ln in lines)


def test_sinfo_summary_and_nodes(any_cluster):
    fake = any_cluster
    out = fake.ok("sinfo", "-s", "-h", "-o", "%P|%a|%l|%F|%N")
    fx = fixture(fake.cluster, "sinfo_summary.out").splitlines()
    for line in out.splitlines():
        f = line.split("|")
        assert len(f) == len(fx[0].split("|")) == 5
        assert re.fullmatch(r"\d+/\d+/\d+/\d+", f[3])
    out = fake.ok("sinfo", "-h", "-N", "-o", "%N|%P|%t|%c|%m|%G|%f|%C|%e|%O")
    fx = fixture(fake.cluster, "sinfo_nodes.out").splitlines()
    for line in out.splitlines():
        f = line.split("|")
        assert len(f) == len(fx[0].split("|")) == 10
        assert re.fullmatch(r"\d+", f[3]) and re.fullmatch(r"\d+", f[4])
    if fake.cluster == "trace":
        assert "trace01|batch|idle|128|2063700|gpu:a40:1|(null)|0/128/0/128|" in out
        assert "trace01|biosimmlab|idle|" in out


def test_scontrol_partition_keys_match_fixture(any_cluster):
    fake = any_cluster
    fx = {ln.split()[0].split("=")[1]: [t.split("=")[0] for t in ln.split()]
          for ln in fixture(fake.cluster, "scontrol_partitions.out").splitlines()}
    out = fake.ok("scontrol", "show", "partition", "-o")
    for line in out.splitlines():
        keys = [t.split("=")[0] for t in line.split()]
        name = line.split()[0].split("=")[1]
        assert name in fx, name
        assert keys == fx[name], f"{name}: {keys} != {fx[name]}"
    # single partition and multi-line form
    one = fake.ok("scontrol", "show", "partition", fake.dump()["partitions"][0]["name"])
    assert one.startswith("PartitionName=")
    assert "\n   " in one


def test_scontrol_config_keys_and_values(any_cluster):
    fake = any_cluster
    fx = {}
    for ln in fixture(fake.cluster, "scontrol_config.out").splitlines():
        m = re.match(r"^(\S+)\s+= (.*)$", ln)
        if m:
            fx[m.group(1)] = m.group(2)
    out = fake.ok("scontrol", "show", "config")
    got = {}
    for ln in out.splitlines():
        m = re.match(r"^(\S+)\s+= (.*)$", ln)
        if m:
            got[m.group(1)] = m.group(2)
    for key in ("ClusterName", "SLURM_VERSION", "JobRequeue", "PreemptMode", "PreemptType", "MinJobAge",
                "EnforcePartLimits", "KillWait", "MaxArraySize", "AccountingStorageEnforce", "SelectType",
                "SchedulerType", "PriorityType", "JobFileAppend", "GresTypes", "MaxJobId"):
        assert got[key] == fx[key], key
    assert out.startswith("Configuration data as of ")
    assert "Cgroup Support Configuration:" in out
    missing = set(fx) - set(got)
    assert len(missing) < 25, sorted(missing)


def test_squeue_format_shape_against_fixture(trace):
    fx_rows = [ln.split("|") for ln in fixture("trace", "squeue_me.out").splitlines()]
    n_fields = len(fx_rows[0])
    ids = [trace.submit() for _ in range(5)]          # 4 nodes with one GPU each -> 5th pends
    dep = trace.submit("--dependency=afterok:%d" % ids[0], "-J", "pred_pat")
    trace.advance(2)
    out = trace.ok("squeue", "-u", trace.user, "-h", "-o", SQUEUE_FMT)
    rows = {r[0]: r for r in (ln.split("|") for ln in out.splitlines())}
    assert all(len(r) == n_fields == 23 for r in rows.values())
    running = rows[str(ids[0])]
    pending = rows[str(ids[4])]
    depend = rows[str(dep)]
    fx_running = next(r for r in fx_rows if r[2] == "RUNNING")
    fx_pending = next(r for r in fx_rows if r[4] == "(Resources)")
    fx_dep = next(r for r in fx_rows if r[4] == "(Dependency)")
    assert running[2] == "RUNNING" and running[3] == "batch" and running[4].startswith("trace")
    assert running[13] == "None" and running[14] == running[4] and running[9] == "N/A" == fx_running[9]
    assert ISO.match(running[10]) and ISO.match(running[11]) and ISO.match(running[21])
    assert re.fullmatch(r"\d+:\d\d", running[5])
    assert running[6] == "1-00:00:00" == fx_running[6]
    assert running[20] == "(null)" == fx_running[20]
    assert pending[2] == "PENDING" and pending[4] == "(Resources)" == fx_pending[4]
    assert pending[13] == "Resources" and pending[14] == "" and pending[5] == "0:00" == fx_pending[5]
    assert ISO.match(pending[10]) and ISO.match(pending[21])   # backfill estimate present
    assert pending[22] == "1-00:00:00" == fx_pending[22]
    assert depend[4] == "(Dependency)" and depend[10] == "N/A" == fx_dep[10] and depend[21] == "N/A" == fx_dep[21]
    # default sort: pending first (by priority desc) then running
    order = [ln.split("|")[2] for ln in out.splitlines()]
    assert order.index("RUNNING") > order.index("PENDING")
    # --start fixture format
    fx_start = fixture("trace", "squeue_me_start.out").splitlines()[0].split("|")
    start = trace.ok("squeue", "-u", trace.user, "--start", "-h", "-o", "%i|%j|%P|%S|%R|%Q|%T")
    for ln in start.splitlines():
        f = ln.split("|")
        assert len(f) == len(fx_start) == 7 and f[6] == "PENDING"
    # default layout header resembles the real one
    hdr = trace.ok("squeue").splitlines()[0]
    assert hdr.split() == ["JOBID", "PARTITION", "NAME", "USER", "ST", "TIME", "NODES", "NODELIST(REASON)"]


def test_sacct_shape_running_matches_fixture(trace):
    fx = [ln.split("|") for ln in fixture("trace", "sacct_job_615411.out").splitlines()]
    jid = trace.submit()
    trace.advance(5)
    out = trace.ok("sacct", "-j", str(jid), "-n", "-P", "--format=" + SACCT_FMT)
    rows = [ln.split("|") for ln in out.splitlines()]
    assert len(rows) == len(fx) == 3
    for got, want in zip(rows, fx):
        assert len(got) == len(want) == 27
        # same emptiness pattern per field for alloc / batch / extern rows
        assert [bool(x) for x in got] == [bool(x) for x in want], (got, want)
    alloc, batch, extern = rows
    assert alloc[0] == alloc[1] == str(jid) and batch[0] == f"{jid}.batch" and extern[0] == f"{jid}.extern"
    assert alloc[3] == "RUNNING" and alloc[9] == "Unknown" == fx[0][9]
    assert alloc[15] == alloc[16] == "billing=64,cpu=64,gres/gpu=1,mem=512G,node=1"
    assert batch[15] == "cpu=64,gres/gpu=1,mem=512G,node=1" == fx[1][15]
    assert alloc[21] == "1-00:00:00" and alloc[22] == "1440" and alloc[23] == "64" and alloc[24] == "1"
    assert alloc[25] == "SchedMain,StartRecieved"
    assert alloc[26].startswith("sbatch ")
    # -X collapses to the allocation line; pending fixture shape
    assert len(trace.ok("sacct", "-j", str(jid), "-n", "-P", "-X", "--format=" + SACCT_FMT).splitlines()) == 1


def test_sacct_pending_row_matches_fixture(trace):
    fx = fixture("trace", "sacct_job_615427.out").strip().split("|")
    for _ in range(4):
        trace.submit()
    jid = trace.submit()
    trace.advance(2)
    out = trace.ok("sacct", "-j", str(jid), "-n", "-P", "--format=" + SACCT_FMT).strip().split("|")
    assert [bool(x) for x in out] == [bool(x) for x in fx]
    assert out[3] == "PENDING" and out[8] == out[9] == "Unknown" and out[14] == "None assigned"
    assert out[25] == "StartRecieved" and out[19] == "None"


def test_sacct_timeout_matches_fixture(bridges2):
    fx = [ln.split("|") for ln in fixture("bridges2", "sacct_job_44809480.out").splitlines()]
    jid = bridges2.submit(script=B2_GPU_SCRIPT)
    bridges2.advance(8 * 3600 + 60)
    out = bridges2.ok("sacct", "-j", str(jid), "-n", "-P", "--format=" + SACCT_FMT)
    rows = [ln.split("|") for ln in out.splitlines()]
    assert len(rows) == 3
    for got, want in zip(rows, fx):
        assert [bool(x) for x in got] == [bool(x) for x in want], (got, want)
    alloc, batch, extern = rows
    assert (alloc[3], alloc[4]) == ("TIMEOUT", "0:0")
    assert (batch[3], batch[4]) == ("CANCELLED", "0:15")
    assert (extern[3], extern[4]) == ("COMPLETED", "0:0")
    assert batch[17].endswith("K") and extern[17] == "0"
    assert alloc[21] == "08:00:00" and alloc[22] == "480" and alloc[13] == "gpu" and alloc[11] == "GPU-shared"
    assert alloc[15] == "billing=24,cpu=24,gres/gpu=2,mem=128G,node=1"
    # gone from the controller after MinJobAge (200 s on bridges2), still in accounting
    bridges2.advance(300)
    rc, _, err = bridges2("scontrol", "show", "job", str(jid), "-o")
    assert rc == 1 and err.strip() == fixture("bridges2", "scontrol_show_job_44809480.err").strip()
    assert bridges2.ok("sacct", "-j", str(jid), "-n", "-P", "-X", "--format=JobID,State").strip() == f"{jid}|TIMEOUT"


def test_scontrol_show_job_keys_match_fixture(trace):
    fx_running = [t.split("=")[0] for t in fixture("trace", "scontrol_show_job_615411.out").split()]
    fx_pending = [t.split("=")[0] for t in fixture("trace", "scontrol_show_job_615427.out").split()]
    ids = [trace.submit() for _ in range(5)]
    trace.advance(2)
    running = trace.ok("scontrol", "show", "job", str(ids[0]), "-o")
    pending = trace.ok("scontrol", "show", "job", str(ids[4]), "-o")
    assert [t.split("=")[0] for t in running.split()] == fx_running
    assert [t.split("=")[0] for t in pending.split()] == fx_pending
    assert running.endswith(" \n")  # real scontrol -o leaves a trailing blank
    kv = dict(t.split("=", 1) for t in running.split())
    assert kv["JobState"] == "RUNNING" and kv["Requeue"] == "1" and kv["Restarts"] == "0"
    assert kv["TRES"] == "cpu=64,mem=512G,node=1,billing=64,gres/gpu=1" and kv["TresPerJob"] == "gres:gpu:a40"
    assert kv["Partition"] == "batch" and kv["BatchHost"] == kv["NodeList"]
    assert kv["StdOut"].endswith(f"wobl_{ids[0]}.out")
    kvp = dict(t.split("=", 1) for t in pending.split())
    assert kvp["JobState"] == "PENDING" and kvp["Reason"] == "Resources" and kvp["NodeList"] == ""
    assert kvp["Scheduler"] == "Backfill:*" and kvp["SchedNodeList"].startswith("trace")
    # multi-line form groups keys and shows the same data
    multi = trace.ok("scontrol", "show", "job", str(ids[0]))
    assert multi.splitlines()[0].startswith(f"JobId={ids[0]} JobName=wobl")


def test_scontrol_missing_job_and_bad_updates(any_cluster):
    fake = any_cluster
    rc, out, err = fake("scontrol", "show", "job", "1", "-o")
    assert rc == 1 and out == "" and err.strip() == fixture(fake.cluster, "scontrol_show_job_missing.err").strip()
    rc, out, err = fake("scontrol", "update", "JobId=1", "TimeLimit=5")
    assert rc == 1 and "Invalid job id specified" in err
    rc, out, err = fake("scontrol", "update", "JobId=1", "Foo")
    assert rc == 1 and err.startswith("scontrol: error: ")


@pytest.mark.parametrize("name", ["sbatch_test_only_ok", "sbatch_test_only_bad_partition",
                                  "sbatch_test_only_bad_gres", "sbatch_test_only_bad_time"])
def test_sbatch_test_only_matches_fixture(any_cluster, name):
    fake = any_cluster
    index = json.loads(fixture(fake.cluster, "index.json"))[name]
    argv = index["cmd"].split()[1:]
    rc, out, err = fake("sbatch", *argv)
    assert rc == index["rc"] and out == ""
    want = fixture(fake.cluster, name + ".err")
    if rc == 0:
        m = re.fullmatch(r"sbatch: Job (\d+) to start at (\S+) using (\d+) processors on nodes (\S+) in partition (\S+)\n", err)
        assert m, err
        wm = re.fullmatch(r"sbatch: Job (\d+) to start at (\S+) using (\d+) processors on nodes (\S+) in partition (\S+)\n", want)
        assert m.group(3) == wm.group(3) and m.group(5) == wm.group(5)
        assert ISO.match(m.group(2))
        assert fake.dump()["jobs"] == {}          # --test-only creates no job
    else:
        assert err == want


def test_sbatch_errors_and_success_strings(trace, bridges2):
    rc, out, err = trace("sbatch", "-p", "nope", stdin=TRACE_GPU_SCRIPT)
    assert rc == 1 and err == ("sbatch: error: invalid partition specified: nope\n"
                               "sbatch: error: Batch job submission failed: Invalid partition name specified\n")
    rc, out, err = trace("sbatch", "-q", "nosuchqos", stdin=TRACE_GPU_SCRIPT)
    assert rc == 1 and err == "sbatch: error: Batch job submission failed: Invalid qos specification\n"
    rc, out, err = trace("sbatch", "-A", "otheracct", stdin=TRACE_GPU_SCRIPT)
    assert err == "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified\n"
    rc, out, err = trace("sbatch", "--gres=gpu:nosuchgpu:1", stdin=TRACE_GPU_SCRIPT)
    assert err == "sbatch: error: Batch job submission failed: Requested node configuration is not available\n"
    rc, out, err = trace("sbatch", "-t", "99-00:00:00", stdin=TRACE_GPU_SCRIPT)
    assert err.splitlines()[-1] == ("sbatch: error: Batch job submission failed: Job violates accounting/QOS policy "
                                    "(job submit limit, user's size and/or time limits)")
    rc, out, err = trace("sbatch", "-p", "cpuonly", "-t", "3-00:00:00", "--wrap", "hostname")
    assert err == "sbatch: error: Batch job submission failed: Requested time limit is invalid (missing or exceeds some limit)\n"
    rc, out, err = trace("sbatch", stdin="echo no shebang\n")
    assert rc == 1 and err.startswith("sbatch: error: This does not look like a batch script.")
    rc, out, err = trace("sbatch", "--dependency=afterok:999999", stdin=TRACE_GPU_SCRIPT)
    assert err == "sbatch: error: Batch job submission failed: Job dependency problem\n"
    rc, out, err = trace("sbatch", "--bogus-flag", stdin=TRACE_GPU_SCRIPT)
    assert rc == 1 and "unrecognized option '--bogus-flag'" in err
    rc, out, err = trace("sbatch", stdin=TRACE_GPU_SCRIPT)
    assert rc == 0 and out == "Submitted batch job 615442\n" and err == ""
    rc, out, err = trace("sbatch", "--parsable", stdin=TRACE_GPU_SCRIPT)
    assert rc == 0 and out == "615443\n"
    # bridges2: RM does not allow the default qos (fixture: every default submit fails with Invalid qos)
    rc, out, err = bridges2("sbatch", "-N", "1", "-n", "1", "-t", "0:10:00", "--wrap", "hostname")
    assert rc == 1 and err == "sbatch: error: Batch job submission failed: Invalid qos specification\n"
    rc, out, err = bridges2("sbatch", "-p", "RM-shared", "--qos=low", "--ntasks-per-node=4", "-t", "1:00:00", "--wrap", "hostname")
    assert rc == 0 and out.startswith("Submitted batch job ")
    rc, out, err = bridges2("sbatch", "-p", "RM-shared", "--qos=low", "-N", "2", "-t", "1:00:00", "--wrap", "hostname")
    assert rc == 1 and "Node count specification invalid" in err


def test_sbatch_directives_and_cli_override(trace):
    jid = trace.submit("-t", "30", "-J", "override", "--comment=mcp:abc", "--export=ALL,FOO=bar", "--mem-per-cpu=1000M")
    j = trace.job(jid)
    assert j["name"] == "override" and j["time_limit"] == 30 and j["comment"] == "mcp:abc"
    assert j["partitions"] == ["batch"] and j["gres"] == [{"name": "gpu", "type": "a40", "count": 1}]
    assert j["cpus_per_node"] == 64 and j["mem_mb"] == 64000 and j["requeue"] is True
    assert j["env"]["FOO"] == "bar" and j["script"] == TRACE_GPU_SCRIPT
    assert j["submit_line"].startswith("sbatch --parsable -t 30")
    assert j["stdout"] == "logs/wobl_%j.out" and j["workdir"] == str(trace.home)
    # JobRequeue=0 on trace: without --requeue the job is not requeueable
    jid2 = trace.submit(script=TRACE_GPU_SCRIPT.replace("#SBATCH --requeue\n", ""))
    assert trace.job(jid2)["requeue"] is False
    jid3 = trace.submit("--no-requeue")
    assert trace.job(jid3)["requeue"] is False
    # time formats
    for spec, minutes in (("90", 90), ("30:00", 30), ("2:00:00", 120), ("1-12", 36 * 60), ("1-00:30", 1470),
                          ("0-01:00:30", 61)):
        assert trace.job(trace.submit("-t", spec))["time_limit"] == minutes
    rc, out, err = trace("sbatch", "-t", "abc", stdin=TRACE_GPU_SCRIPT)
    assert rc == 1 and err == "sbatch: error: Invalid time limit specification\n"


def test_sacct_errors_and_time_specs(any_cluster):
    fake = any_cluster
    rc, out, err = fake("sacct", "-j", "1", "-n", "-P", "--format=JobID,State")
    assert (rc, out) == (0, "")
    rc, out, err = fake("sacct", "-n", "-P", "-S", "1756000000", "--format=JobID")
    assert rc == 1 and "Invalid time specification" in err
    rc, out, err = fake("sacct", "-n", "-P", "--format=JobID,Restarts")
    assert rc == 1 and err == 'Invalid field requested: "Restarts"\n'
    for spec in ("now-7days", "2026-09-01", "2026-09-01T00:00:00", "now", "today", "midnight", "now-2hours"):
        rc, out, err = fake("sacct", "-n", "-P", "-S", spec, "--format=JobID,State")
        assert rc == 0, (spec, err)
    rc, out, err = fake("sacct", "-n", "-p", "-S", "now-7days", "--format=JobID,State")
    assert rc == 0
    rc, out, err = fake("sacct", "-S", "now-1days", "--format=JobID%20,State")
    assert rc == 0 and out.splitlines()[0].strip().startswith("JobID") and set(out.splitlines()[1].strip()) == {"-", " "}


def test_sacctmgr_sshare_sprio_match_fixture(any_cluster):
    fake = any_cluster
    out = fake.ok("sacctmgr", "-n", "-P", "show", "assoc", f"user={fake.user}",
                  "format=cluster,account,partition,qos,maxjobs,maxsubmit,grptres")
    assert out == fixture(fake.cluster, "sacctmgr_assoc.out")
    out = fake.ok("sacctmgr", "-n", "-P", "show", "qos",
                  "format=name,priority,maxwall,maxtrespu,maxjobspu,maxsubmitpu,grptres,maxtres,flags")
    got = {ln.split("|")[0]: ln for ln in out.splitlines()}
    fx = {ln.split("|")[0]: ln for ln in fixture(fake.cluster, "sacctmgr_qos.out").splitlines()}
    assert set(got) <= set(fx)
    for name, line in got.items():
        assert line == fx[name], name
    rc, out, err = fake("sacctmgr", "-n", "-P", "show", "qos", "format=name,bogus")
    assert rc == 1 and "Unknown field 'bogus'" in err
    out = fake.ok("sacctmgr", "-n", "-P", "show", "user", fake.user, "withassoc",
                  "format=User,DefaultAccount,Account,Cluster,Partition,QOS,DefaultQOS")
    assert out.split("|")[0] == fake.user and out.split("|")[3] == fake.cluster
    assert fake.ok("sshare", "-U", "-P") == fixture(fake.cluster, "sshare_me.out")
    out = fake.ok("sprio", "-u", fake.user, "-o", "%i|%r|%Y|%A|%F|%J|%P|%Q|%T")
    assert out.splitlines()[0] == fixture(fake.cluster, "sprio_me.out").splitlines()[0]


def test_scancel_unknown_id_matches_fixture(any_cluster):
    fake = any_cluster
    rc, out, err = fake("scancel", "1")
    assert (rc, out, err) == (0, "", "")
    rc, out, err = fake("scancel")
    assert rc == 1 and err == "scancel: error: No job identification provided\n"
    fake.ok("fakeslurm-ctl", "set-config", "fake.scancel_strict", "1")
    rc, out, err = fake("scancel", "1")
    assert rc == 1 and err == "scancel: error: Kill job error on job id 1: Invalid job id specified\n"


# ------------------------------------------------------------------------------------------------
# behaviour
# ------------------------------------------------------------------------------------------------
def test_job_lifecycle_pending_running_completed(trace):
    jid = trace.submit()
    row = trace.squeue_row(jid)
    assert row[2] == "PENDING" and row[4] == "(None)"          # not yet considered by the scheduler
    trace.advance(1)
    row = trace.squeue_row(jid)
    assert row[2] == "RUNNING" and row[4].startswith("trace") and row[13] == "None"
    j = trace.job(jid)
    assert j["start"] is not None and j["priority"] > 0
    out_file = Path(j["workdir"]) / f"logs/wobl_{jid}.out"
    assert out_file.exists()                                    # slurmd creates the output file at start
    trace.advance(899)
    assert trace.squeue_row(jid)[2] == "RUNNING"
    trace.advance(1)
    row = trace.squeue_row(jid)
    assert row[2] == "COMPLETED"                                # still in controller memory
    assert trace.squeue_row(jid, "%i|%t")[1] == "CD"
    assert trace.ok("squeue", "-h", "-o", "%i").strip() == str(jid)   # squeue shows CD jobs until MinJobAge
    sacct = trace.ok("sacct", "-j", str(jid), "-n", "-P", "-X", "--format=JobID,State,ExitCode,ElapsedRaw,Start,End").strip().split("|")
    assert sacct[1:4] == ["COMPLETED", "0:0", "900"]
    assert int(fakeslurm.parse_iso(sacct[5])) - int(fakeslurm.parse_iso(sacct[4])) == 900
    # MinJobAge (300 s on trace): visible until then, then purged from squeue/scontrol but not sacct
    trace.advance(298)
    assert trace.squeue_row(jid) is not None
    assert trace.ok("scontrol", "show", "job", str(jid), "-o").startswith(f"JobId={jid}")
    trace.advance(5)
    assert trace.squeue_row(jid) is None
    rc, out, err = trace("scontrol", "show", "job", str(jid), "-o")
    assert rc == 1 and err == "slurm_load_jobs error: Invalid job id specified\n"
    assert trace.ok("sacct", "-j", str(jid), "-n", "-P", "-X", "--format=State").strip() == "COMPLETED"
    rc, out, err = trace("squeue", "-j", str(jid), "-h")
    assert rc == 1 and err == "slurm_load_jobs error: Invalid job id specified\n"


def test_failed_exit_code_and_ctl_finish(trace):
    jid = trace.submit(script=TRACE_GPU_SCRIPT.replace("exit=0", "exit=3"))
    trace.advance(1000)
    row = trace.ok("sacct", "-j", str(jid), "-n", "-P", "--format=JobID,State,ExitCode").splitlines()
    assert row[0] == f"{jid}|FAILED|3:0" and row[1] == f"{jid}.batch|FAILED|3:0" and row[2] == f"{jid}.extern|COMPLETED|0:0"
    jid2 = trace.submit(script=TRACE_GPU_SCRIPT.replace("#FAKESLURM duration=900 exit=0\n", ""))
    trace.advance(1)
    trace.ok("fakeslurm-ctl", "finish", str(jid2), "--exit", "1")
    assert trace.squeue_row(jid2)[2] == "FAILED"
    assert trace.ok("seff", str(jid2)).startswith(f"Job ID: {jid2}\nCluster: trace\n")


def test_timeout(trace):
    jid = trace.submit("-t", "2", script=TRACE_GPU_SCRIPT.replace("#FAKESLURM duration=900 exit=0\n", ""))
    trace.advance(1 + 119)
    assert trace.squeue_row(jid)[2] == "RUNNING"
    assert trace.squeue_row(jid)[22] == "0:01"
    trace.advance(1)
    assert trace.squeue_row(jid)[2] == "TIMEOUT"
    rows = trace.ok("sacct", "-j", str(jid), "-n", "-P", "--format=JobID,State,ExitCode,ElapsedRaw").splitlines()
    assert rows[0] == f"{jid}|TIMEOUT|0:0|120" and rows[1].startswith(f"{jid}.batch|CANCELLED|0:15|")
    kv = dict(t.split("=", 1) for t in trace.ok("scontrol", "show", "job", str(jid), "-o").split())
    assert kv["JobState"] == "TIMEOUT" and kv["Reason"] == "TimeLimit"


def test_preemption_requeue_and_restarts(trace):
    requeue = [trace.submit("-J", f"rq{i}") for i in range(3)]
    norq = trace.submit("-J", "norq", "--no-requeue")
    trace.advance(10)
    assert all(trace.squeue_row(j)[2] == "RUNNING" for j in requeue + [norq])
    # higher PriorityTier partition needs a GPU -> newest lower-tier job gets preempted (PreemptMode=REQUEUE)
    urgent = trace.submit("-p", "biosimmlab", "-J", "urgent")
    trace.advance(1)
    assert trace.squeue_row(urgent)[2] == "RUNNING"
    states = {j: trace.squeue_row(j) for j in requeue + [norq]}
    victims = [j for j, r in states.items() if r[2] != "RUNNING"]
    assert len(victims) == 1
    victim = victims[0]
    if victim == norq:
        assert states[victim][2] == "PREEMPTED"
    else:
        row = states[victim]
        assert row[2] == "PENDING" and row[13] == "BeginTime"
        kv = dict(t.split("=", 1) for t in trace.ok("scontrol", "show", "job", str(victim), "-o").split())
        assert kv["Restarts"] == "1" and kv["JobState"] == "PENDING"
        assert trace.ok("squeue", "-h", "-j", str(victim), "-O", "RestartCnt:0|,Requeue:0|").strip() == "1|1|"
        dup = trace.ok("sacct", "-D", "-X", "-j", str(victim), "-n", "-P", "--format=JobID,State").splitlines()
        assert dup == [f"{victim}|PREEMPTED", f"{victim}|PENDING"]
        assert trace.ok("sacct", "-X", "-j", str(victim), "-n", "-P", "--format=JobID,State").splitlines() == [f"{victim}|PENDING"]
    # explicit ctl preemption of a non-requeueable running job -> PREEMPTED terminal, batch CANCELLED 0:15
    if norq not in victims:
        trace.ok("fakeslurm-ctl", "preempt", str(norq))
        rows = trace.ok("sacct", "-j", str(norq), "-n", "-P", "--format=JobID,State,ExitCode").splitlines()
        assert rows[0] == f"{norq}|PREEMPTED|0:0" and rows[1] == f"{norq}.batch|CANCELLED|0:15"
        assert trace.job(norq)["signals"][-1]["signal"] == "TERM"
    if victim != norq:
        # after the requeue delay and once a node frees up the requeued job runs again
        trace.advance(1000)
        assert trace.squeue_row(victim)[2] == "RUNNING"


def test_scontrol_requeue_hold_release_update(trace):
    jid = trace.submit()
    trace.advance(5)
    trace.ok("scontrol", "requeue", str(jid))
    kv = dict(t.split("=", 1) for t in trace.ok("scontrol", "show", "job", str(jid), "-o").split())
    assert kv["JobState"] == "PENDING" and kv["Restarts"] == "1" and kv["Reason"] == "BeginTime"
    dup = trace.ok("sacct", "-D", "-X", "-j", str(jid), "-n", "-P", "--format=State").splitlines()
    assert dup == ["REQUEUED", "PENDING"]
    trace.ok("scontrol", "hold", str(jid))
    assert trace.squeue_row(jid)[13] == "JobHeldUser" and trace.squeue_row(jid)[12] == "0"
    trace.advance(200)
    assert trace.squeue_row(jid)[2] == "PENDING"
    trace.ok("scontrol", "release", str(jid))
    trace.advance(1)
    assert trace.squeue_row(jid)[2] == "RUNNING"
    # non-requeueable job cannot be requeued
    jid2 = trace.submit("--no-requeue")
    trace.advance(1)
    rc, out, err = trace("scontrol", "requeue", str(jid2))
    assert rc == 1
    # updates
    jid3 = trace.submit("--hold", "-t", "60")
    trace.ok("scontrol", "update", f"JobId={jid3}", "Partition=biosimmlab", "TimeLimit=30", "Comment=hello")
    j = trace.job(jid3)
    assert j["partitions"] == ["biosimmlab"] and j["time_limit"] == 30 and j["comment"] == "hello"
    rc, out, err = trace("scontrol", "update", f"JobId={jid3}", "TimeLimit=120")
    assert rc == 1 and "Access/permission denied" in err
    trace.ok("scontrol", "update", f"JobId={jid3}", "Nice=50")
    rc, out, err = trace("scontrol", "update", f"JobId={jid3}", "Priority=10")
    assert rc == 1
    out = trace.ok("scontrol", "write", "batch_script", str(jid3), "-")
    assert out == TRACE_GPU_SCRIPT
    target = trace.home / "script.sh"
    assert trace.ok("scontrol", "write", "batch_script", str(jid3), str(target)).strip() == f"batch script for job {jid3} written to {target}"
    assert target.read_text() == TRACE_GPU_SCRIPT


def test_dependencies_and_kill_on_invalid(trace):
    a = trace.submit("-J", "a", script=TRACE_GPU_SCRIPT.replace("duration=900", "duration=60"))
    b = trace.submit("-J", "b", f"--dependency=afterok:{a}")
    c = trace.submit("-J", "c", f"--dependency=afternotok:{a}", "--kill-on-invalid-dep=yes")
    d = trace.submit("-J", "d", f"--dependency=afterany:{a}")
    e = trace.submit("-J", "a", "--dependency=singleton")
    trace.advance(2)
    assert trace.squeue_row(a)[2] == "RUNNING"
    for j in (b, c, d, e):
        assert trace.squeue_row(j)[2] == "PENDING" and trace.squeue_row(j)[13] == "Dependency"
    assert trace.ok("squeue", "-h", "-j", str(b), "-O", "Dependency:0").strip() == f"afterok:{a}"
    trace.advance(120)
    assert trace.squeue_row(a)[2] == "COMPLETED"
    assert trace.squeue_row(b)[2] == "RUNNING" and trace.squeue_row(d)[2] == "RUNNING"
    assert trace.squeue_row(e)[2] == "RUNNING"
    assert trace.squeue_row(c)[2] == "CANCELLED"                 # afternotok can never be satisfied
    f = trace.submit("-J", "f", f"--dependency=afternotok:{a}")
    trace.advance(2)
    assert trace.squeue_row(f)[13] == "DependencyNeverSatisfied"


def test_array_jobs(trace):
    master = trace.submit("--array=0-3%2", "-J", "arr", script=TRACE_GPU_SCRIPT.replace("duration=900", "duration=100"))
    tasks = trace.ok("squeue", "-h", "-r", "-n", "arr", "-o", "%i|%A|%F|%K|%T").splitlines()
    assert [t.split("|")[0] for t in tasks] == [f"{master}_{i}" for i in range(4)]
    assert tasks[0].split("|")[1:4] == [str(master), str(master), "0"]
    collapsed = trace.ok("squeue", "-h", "-n", "arr", "-o", "%i").strip()
    assert collapsed == f"{master}_[0-3%2]"
    trace.advance(2)
    states = [t.split("|")[4] for t in trace.ok("squeue", "-h", "-r", "-n", "arr", "-o", "%i|%A|%F|%K|%T").splitlines()]
    assert states.count("RUNNING") == 2 and states.count("PENDING") == 2   # %2 throttle
    assert trace.ok("squeue", "-h", "-r", "-j", f"{master}_3", "-o", "%r").strip() == "JobArrayTaskLimit"
    trace.advance(150)
    states = [t.split("|")[4] for t in trace.ok("squeue", "-h", "-r", "-t", "all", "-n", "arr", "-o", "%i|%A|%F|%K|%T").splitlines()]
    assert states.count("COMPLETED") == 2 and states.count("RUNNING") == 2
    rows = trace.ok("sacct", "-X", "-n", "-P", "-j", str(master), "--format=JobID,JobIDRaw").splitlines()
    assert rows[0] == f"{master}_0|{master}" and rows[1] == f"{master}_1|{master + 1}"
    kv = dict(t.split("=", 1) for t in trace.ok("scontrol", "show", "job", f"{master}_1", "-o").split())
    assert kv["ArrayJobId"] == str(master) and kv["ArrayTaskId"] == "1" and kv["ArrayTaskThrottle"] == "2"
    assert kv["StdOut"].replace("\\", "/").endswith(f"logs/wobl_{master + 1}.out")   # %j = raw task id
    trace.ok("scancel", f"{master}_[2-3]")
    states = trace.ok("squeue", "-h", "-r", "-t", "all", "-j", f"{master}_2,{master}_3", "-o", "%i|%T").splitlines()
    assert states == [f"{master}_2|CANCELLED", f"{master}_3|CANCELLED"]


def test_hold_begin_and_qos_limits(trace, bridges2):
    h = trace.submit("--hold")
    b = trace.submit("--begin=now+1hour")
    trace.advance(5)
    assert trace.squeue_row(h)[2:5] == ["PENDING", "batch", "(JobHeldUser)"] and trace.squeue_row(h)[12] == "0"
    assert trace.squeue_row(b)[4] == "(BeginTime)"
    trace.advance(3600)
    assert trace.squeue_row(b)[2] == "RUNNING" and trace.squeue_row(h)[2] == "PENDING"
    # bridges2 GPU-small: MaxJobsPU=2 -> third job blocked with the QOS reason
    ids = [bridges2.submit("-p", "GPU-small", "--gres=gpu:v100-32:1", "-q", "gpu", "-t", "1:00:00",
                           "-J", f"small{i}", script="#!/bin/bash\nhostname\n") for i in range(3)]
    bridges2.advance(5)
    rows = [bridges2.squeue_row(i) for i in ids]
    assert [r[2] for r in rows] == ["RUNNING", "RUNNING", "PENDING"]
    assert rows[2][4] == "(QOSMaxJobsPerUserLimit)"
    # GPU-small MaxWall 8h enforced at submit (DenyOnLimit)
    rc, out, err = bridges2("sbatch", "-p", "GPU-small", "--gres=gpu:v100-32:1", "-q", "gpu", "-t", "9:00:00",
                            "--wrap", "hostname")
    assert rc == 1 and err.splitlines()[0] == "sbatch: error: QOSMaxWallDurationPerJobLimit"
    # gres type must exist in the partition
    rc, out, err = bridges2("sbatch", "-p", "GPU-shared", "--gres=gpu:a40:1", "-q", "gpu", "-t", "1:00:00", "--wrap", "hostname")
    assert "Requested node configuration is not available" in err
    # multi-partition request: the list is kept, the chosen partition placed first once running (sbatch(1))
    m = bridges2.submit("-p", "GPU-shared,GPU", "--gres=gpu:v100-32:1", "-q", "gpu", "-t", "1:00:00", script="#!/bin/bash\nhostname\n")
    assert bridges2.squeue_row(m)[3] == "GPU-shared,GPU"
    bridges2.advance(2)
    assert bridges2.squeue_row(m)[2] == "RUNNING" and bridges2.squeue_row(m)[3] == "GPU-shared,GPU"


def test_multi_partition_list_reordered_after_start(trace):
    """sbatch(1): "When the job is initiated, the name of the partition used will be placed first in the job
    record partition string" -- squeue %P / -O Partition / scontrol keep the whole list; sacct Partition and
    SLURM_JOB_PARTITION name the single partition the job runs in.  [verify on cluster: no fixture captures
    a running multi-partition job]"""
    script = TRACE_GPU_SCRIPT.replace("#SBATCH -p batch\n", "#SBATCH -p batch,biosimmlab\n")
    jid = trace.submit(script=script)
    assert trace.squeue_row(jid)[3] == "batch,biosimmlab"
    kv = dict(t.split("=", 1) for t in trace.ok("scontrol", "show", "job", str(jid), "-o").split())
    assert kv["Partition"] == "batch,biosimmlab"
    trace.advance(2)
    row = trace.squeue_row(jid)
    assert row[2] == "RUNNING"
    assert row[3] == "biosimmlab,batch"                       # biosimmlab (PriorityTier 20) was chosen
    assert trace.ok("squeue", "-h", "-j", str(jid), "-O", "Partition:0").strip() == "biosimmlab,batch"
    kv = dict(t.split("=", 1) for t in trace.ok("scontrol", "show", "job", str(jid), "-o").split())
    assert kv["Partition"] == "biosimmlab,batch" and kv["JobState"] == "RUNNING"
    assert trace.ok("sacct", "-j", str(jid), "-n", "-P", "-X", "--format=Partition").strip() == "biosimmlab"
    assert trace.job(jid)["partition"] == "biosimmlab" and trace.job(jid)["partitions"] == ["biosimmlab", "batch"]
    # the single run partition is still what -p filters and the sacct -r filter see
    assert trace.ok("squeue", "-h", "-p", "biosimmlab", "-o", "%i").strip() == str(jid)
    assert trace.ok("sacct", "-n", "-P", "-X", "-r", "biosimmlab", "--format=JobID").strip() == str(jid)
    trace.advance(1000)
    assert trace.squeue_row(jid)[2] == "COMPLETED" and trace.squeue_row(jid)[3] == "biosimmlab,batch"


def test_scancel_variants(trace):
    r = trace.submit("-J", "r")
    p = trace.submit("-J", "p", "--hold")
    trace.advance(2)
    trace.ok("scancel", "--signal=USR1", "--batch", str(r))
    assert trace.job(r)["signals"][-1] == {"time": trace.dump()["now"], "signal": "USR1", "source": "scancel",
                                           "batch": True, "full": False}
    assert trace.squeue_row(r)[2] == "RUNNING"
    trace.ok("scancel", str(r))
    rows = trace.ok("sacct", "-j", str(r), "-n", "-P", "--format=JobID,State,ExitCode,End").splitlines()
    uid = trace.dump()["user"]["uid"]
    assert rows[0].startswith(f"{r}|CANCELLED by {uid}|0:0|") and rows[1].startswith(f"{r}.batch|CANCELLED|0:15|")
    assert trace.squeue_row(r)[2] == "CANCELLED"
    trace.ok("scancel", "--state=PENDING", "--name=p", "-u", trace.user)
    row = trace.ok("sacct", "-j", str(p), "-n", "-P", "-X", "--format=State,Start,NodeList").strip()
    assert row == f"CANCELLED by {uid}|None|None assigned"
    rc, out, err = trace("scancel", "-Q", "999")
    assert rc == 0 and err == ""
    x = trace.submit("-J", "x")
    y = trace.submit("-J", "y")
    trace.ok("scancel", "-u", trace.user)
    assert trace.squeue_row(x)[2] == "CANCELLED" and trace.squeue_row(y)[2] == "CANCELLED"


def test_ctl_injections(bridges2):
    a = bridges2.submit(script=B2_GPU_SCRIPT)
    b = bridges2.submit(script=B2_GPU_SCRIPT)
    c = bridges2.submit("--no-requeue", script=B2_GPU_SCRIPT)
    bridges2.advance(5)
    bridges2.ok("fakeslurm-ctl", "oom", str(a))
    assert bridges2.ok("sacct", "-j", str(a), "-n", "-P", "--format=JobID,State,ExitCode").splitlines()[0] == f"{a}|OUT_OF_MEMORY|0:125"
    # JobRequeue=1 on bridges2: node failure requeues by default, cancels with --no-requeue
    node_b = bridges2.job(b)["nodes"][0]
    bridges2.ok("fakeslurm-ctl", "nodefail", str(b))
    kv = dict(t.split("=", 1) for t in bridges2.ok("scontrol", "show", "job", str(b), "-o").split())
    assert kv["JobState"] == "PENDING" and kv["Restarts"] == "1"
    assert bridges2.ok("sacct", "-D", "-X", "-j", str(b), "-n", "-P", "--format=State").splitlines() == ["NODE_FAIL", "PENDING"]
    assert any(ln.split("|")[:2] == [node_b, "down*"] for ln in bridges2.ok("sinfo", "-h", "-N", "-o", "%N|%t").splitlines())
    bridges2.ok("fakeslurm-ctl", "nodefail", str(c))
    assert bridges2.squeue_row(c)[2] == "NODE_FAIL"
    bridges2.ok("fakeslurm-ctl", "drain", "w002", "maintenance")
    assert any(ln.split("|")[:2] == ["w002", "drain"] for ln in bridges2.ok("sinfo", "-h", "-N", "-o", "%N|%t").splitlines())
    d = bridges2.submit(script=B2_GPU_SCRIPT)
    bridges2.advance(5)
    assert bridges2.squeue_row(d)[2] == "PENDING"    # both h100 nodes are down/drained
    assert bridges2.squeue_row(d)[13] == "ReqNodeNotAvail, UnavailableNodes:w[001-002]"
    bridges2.ok("fakeslurm-ctl", "undrain", "w002")
    bridges2.advance(1)
    assert bridges2.squeue_row(d)[2] == "RUNNING"
    bridges2.ok("fakeslurm-ctl", "cancel", str(d))
    assert bridges2.squeue_row(d)[2] == "CANCELLED"
    st = json.loads(bridges2.ok("fakeslurm-ctl", "dump"))
    assert st["cluster"] == "bridges2" and str(d) in st["jobs"]
    assert bridges2.ok("fakeslurm-ctl", "now").splitlines()[0] == fakeslurm.fmt_ts(st["now"])


def test_advance_to_iso_and_now_override(trace, monkeypatch):
    jid = trace.submit()
    trace.ok("fakeslurm-ctl", "advance", "--to", "2026-09-01T17:20:00")
    assert trace.ok("fakeslurm-ctl", "now").splitlines()[0] == "2026-09-01T17:20:00"
    assert trace.squeue_row(jid)[2] == "COMPLETED"
    monkeypatch.setenv("FAKESLURM_NOW", "2026-09-01T18:00:00")
    assert trace.squeue_row(jid) is None                     # past MinJobAge under the override


def test_squeue_long_format_and_filters(trace):
    a = trace.submit("-J", "alpha", "--comment=tag1")
    b = trace.submit("-J", "beta", "-p", "biosimmlab")
    trace.advance(2)
    out = trace.ok("squeue", "-h", "-O", "JobID:0|,Name:0|,State:0|,StateCompact:0|,Partition:0|,Reason:0|,TimeUsed:0|,"
                   "TimeLimit:0|,NumNodes:0|,NumCPUs:0|,tres-per-node:0|,StartTime:0|,SubmitTime:0|,Priority:0|,"
                   "NodeList:0|,Command:0|,WorkDir:0|,UserName:0|,Account:0|,QOS:0|,Comment:0|,EndTime:0|,TimeLeft:0|,"
                   "RestartCnt:0|,ArrayJobID:0|,ArrayTaskID:0|,Dependency:0|,tres-alloc:0|,BatchHost:0|")
    rows = {ln.split("|")[0]: ln.split("|") for ln in out.splitlines()}
    ra = rows[str(a)]
    assert ra[1:6] == ["alpha", "RUNNING", "R", "batch", "None"]
    assert ra[10] == "N/A" and ISO.match(ra[11]) and ra[20] == "tag1" and ra[23] == "0" and ra[26] == "(null)"
    assert ra[27] == "cpu=64,mem=512G,node=1,billing=64,gres/gpu=1" and ra[28] == ra[14]
    padded = trace.ok("squeue", "-h", "-O", "JobID:10,Name:6")
    assert padded.splitlines()[0][:10] == str(a).ljust(10) and padded.splitlines()[0][10:16] == "alpha "
    hdr = trace.ok("squeue", "-O", "JobID:0|,Name:0|").splitlines()[0]
    assert hdr == "JOBID|NAME|"
    assert trace.ok("squeue", "-h", "-p", "biosimmlab", "-o", "%i").strip() == str(b)
    assert trace.ok("squeue", "-h", "-n", "alpha", "-o", "%i").strip() == str(a)
    assert trace.ok("squeue", "-h", "-t", "PD,R", "-o", "%i").split() == [str(a), str(b)]
    assert trace.ok("squeue", "-h", "-t", "PENDING", "-o", "%i").strip() == ""
    assert trace.ok("squeue", "-h", "--me", "-o", "%u").split() == [trace.user, trace.user]
    assert trace.ok("squeue", "-h", "-j", f"{a},{b}", "-o", "%i", "--sort=-i").split() == [str(b), str(a)]
    rc, out, err = trace("squeue", "-j", "424242", "-h")
    assert rc == 1 and err == "slurm_load_jobs error: Invalid job id specified\n"


def test_sinfo_state_filters_and_long_format(trace):
    jid = trace.submit("-J", "x")
    trace.advance(2)
    node = trace.job(jid)["nodes"][0]
    out = trace.ok("sinfo", "-h", "-p", "batch", "-t", "mix", "-o", "%N|%t|%C")
    assert out.strip().split("|") == [node, "mix", "64/64/0/128"]
    out = trace.ok("sinfo", "-h", "-N", "-p", "batch", "-O", "NodeHost:0|,StateCompact:0|,Gres:0|,GresUsed:0|")
    lines = dict(ln.split("|", 1) for ln in out.splitlines())
    assert lines[node] == "mix|gpu:a40:1|gpu:a40:1(IDX:0)|"
    idle = next(n for n in lines if n != node)
    assert lines[idle] == "idle|gpu:a40:1|gpu:a40:0(IDX:N/A)|"
    out = trace.ok("sinfo", "-h", "-o", "%R|%M|%p|%h")
    assert "batch|REQUEUE|1|FORCE:1" in out and "biosimmlab|GANG,REQUEUE|20|FORCE:1" in out
    assert trace.ok("sinfo", "--version").strip() == "slurm 22.05.11"
    node_line = trace.ok("scontrol", "show", "node", node, "-o", "-d")
    kv = dict(re.findall(r"([A-Za-z_/:]+)=(\S*)", node_line))   # OS= value contains spaces, like real scontrol
    assert kv["State"] == "MIXED" and kv["CPUAlloc"] == "64" and kv["GresUsed"] == "gpu:a40:1(IDX:0)"
    assert kv["CfgTRES"] == "cpu=128,mem=2063700M,billing=128,gres/gpu=1" and kv["Partitions"] == "batch,biosimmlab"


def test_run_script_executes_with_slurm_env(trace):
    script = """#!/bin/bash
#SBATCH -p cpuonly
#SBATCH -J envtest
#SBATCH -o env_%j.out
#SBATCH --export=ALL,GREETING=hello
echo "id=$SLURM_JOB_ID name=$SLURM_JOB_NAME part=$SLURM_JOB_PARTITION nodes=$SLURM_JOB_NODELIST greet=$GREETING"
exit 7
"""
    jid = trace.submit(script=script)
    trace.advance(2)
    out = trace.ok("fakeslurm-ctl", "run-script", str(jid))
    assert out.strip() == f"job {jid} script exited with 7"
    text = (trace.home / f"env_{jid}.out").read_text()
    assert text.strip() == f"id={jid} name=envtest part=cpuonly nodes={trace.job(jid)['nodes'][0]} greet=hello"
    assert trace.squeue_row(jid)[2] == "FAILED"
    assert trace.ok("sacct", "-j", str(jid), "-n", "-P", "-X", "--format=ExitCode").strip() == "7:0"


def test_shims_work_from_git_bash(tmp_path, monkeypatch):
    from sshd_harness import FAKESLURM_BIN, _to_posix, find_bash
    state = tmp_path / "state.json"
    env = dict(os.environ)
    env.update({"FAKESLURM_STATE": str(state), "FAKESLURM_PYTHON": sys.executable.replace("\\", "/"),
                "PATH": _to_posix(str(FAKESLURM_BIN)) + os.pathsep + os.environ.get("PATH", ""),
                "MSYS2_PATH_TYPE": "inherit"})
    bash = find_bash()
    cmd = ("fakeslurm-ctl init --cluster trace --now 2026-09-01T17:00:00 >/dev/null && "
           "sinfo -h -o %P && printf '#!/bin/bash\\n#SBATCH -p cpuonly\\nhostname\\n' | sbatch -t 5 --parsable && "
           "which sbatch")
    r = subprocess.run([bash, "-lc", cmd], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[:4] == ["batch", "cpuonly", "cpuonly-debug", "biosimmlab"]
    assert lines[4].isdigit()
    assert lines[5].endswith("/fakeslurm/bin/sbatch")
    assert state.exists()


# ------------------------------------------------------------------------------------------------
# verifier findings: sacct Reason, --units/--noconvert, #FAKESLURM maxrss, path handling
# ------------------------------------------------------------------------------------------------
def test_sacct_reason_is_last_pending_reason_not_terminal_state(trace, bridges2):
    """sacct(1) Reason: "The last reason a job was blocked from running for something other than Priority or
    Resources" -- fixtures: TIMEOUT 44809480, FAILED 600478 and CANCELLED 600480 all print Reason=None.
    scontrol/squeue keep reporting the terminal reason (TimeLimit / NonZeroExitCode)."""
    # TIMEOUT (bridges2 fixture 44809480)
    fx = fixture("bridges2", "sacct_job_44809480.out").splitlines()[0].split("|")
    assert fx[19] == "None"
    jid = bridges2.submit(script=B2_GPU_SCRIPT)
    bridges2.advance(8 * 3600 + 60)
    out = bridges2.ok("sacct", "-j", str(jid), "-n", "-P", "--format=" + SACCT_FMT).splitlines()[0].split("|")
    assert out[3] == "TIMEOUT" and out[19] == "None" == fx[19]
    kv = dict(t.split("=", 1) for t in bridges2.ok("scontrol", "show", "job", str(jid), "-o").split())
    assert kv["JobState"] == "TIMEOUT" and kv["Reason"] == "TimeLimit"
    # FAILED / OUT_OF_MEMORY / CANCELLED-while-pending (trace fixture 600478 / 600480 style)
    failed = trace.submit(script=TRACE_GPU_SCRIPT.replace("exit=0", "exit=1"))
    oom = trace.submit("-p", "cpuonly", "-J", "oom", script="#!/bin/bash\nhostname\n")
    trace.advance(2)
    trace.ok("fakeslurm-ctl", "oom", str(oom))
    trace.advance(1000)
    cancelled = trace.submit("-J", "cancelme")             # cancelled before it was ever considered (600480)
    trace.ok("scancel", str(cancelled))
    for j, state in ((failed, "FAILED"), (oom, "OUT_OF_MEMORY"), (cancelled, "CANCELLED")):
        st, reason = trace.ok("sacct", "-j", str(j), "-n", "-P", "-X", "--format=State,Reason").strip().split("|")
        assert st.startswith(state) and reason == "None", (j, st, reason)
    kv = dict(t.split("=", 1) for t in trace.ok("scontrol", "show", "job", str(failed), "-o").split())
    assert kv["Reason"] == "NonZeroExitCode"
    assert trace.squeue_row(failed)[13] == "None"            # squeue %r is 'None' once no longer pending
    # a job that pended on a dependency and then ran keeps 'Dependency' in the accounting record
    base = trace.submit("-J", "base", script=TRACE_GPU_SCRIPT.replace("duration=900", "duration=60"))
    dep = trace.submit("-J", "dep", f"--dependency=afterany:{base}")
    trace.advance(2)
    assert trace.squeue_row(dep)[2] == "PENDING" and trace.squeue_row(dep)[13] == "Dependency"
    assert trace.ok("sacct", "-j", str(dep), "-n", "-P", "-X", "--format=Reason").strip() == "Dependency"
    trace.advance(120)
    assert trace.squeue_row(dep)[2] == "RUNNING"
    assert trace.ok("sacct", "-j", str(dep), "-n", "-P", "-X", "--format=Reason").strip() == "Dependency"
    trace.advance(1000)
    assert trace.ok("sacct", "-j", str(dep), "-n", "-P", "-X", "--format=State,Reason").strip() == "COMPLETED|Dependency"
    # held then released: JobHeldUser is the last non-Priority/Resources block reason
    held = trace.submit("--hold", "-J", "held")
    trace.advance(5)
    trace.ok("scontrol", "release", str(held))
    trace.advance(5)
    assert trace.squeue_row(held)[2] == "RUNNING"
    assert trace.ok("sacct", "-j", str(held), "-n", "-P", "-X", "--format=Reason").strip() == "JobHeldUser"
    # blocked only on Resources/Priority and then started -> None (like fixture 615411 / 600478)
    ids = [trace.submit("-J", f"q{i}") for i in range(5)]
    trace.advance(2)
    assert trace.squeue_row(ids[4])[13] in ("Resources", "Priority")
    trace.advance(1000)
    assert trace.squeue_row(ids[4])[2] == "RUNNING"
    assert trace.ok("sacct", "-j", str(ids[4]), "-n", "-P", "-X", "--format=Reason").strip() == "None"


def test_sacct_units_and_noconvert(trace):
    """--units=[KMGTP] rescales MaxRSS/AveRSS/MaxVMSize/ReqMem like Slurm's convert_num_unit(); --noconvert
    keeps the raw K/M value; the default is the EXACT auto-scaling that leaves 56459172K alone (fixture)."""
    a = trace.submit(script=TRACE_GPU_SCRIPT.replace("exit=0", "exit=0 maxrss=56459172"))
    b = trace.submit(script=TRACE_GPU_SCRIPT.replace("exit=0", "exit=0 maxrss=4194304"))
    trace.advance(1000)
    fmt = "--format=JobID,MaxRSS,AveRSS,MaxVMSize,ReqMem"

    def batch(jid, *extra):
        rows = trace.ok("sacct", "-j", str(jid), "-n", "-P", *extra, fmt).splitlines()
        alloc = rows[0].split("|")
        return alloc, next(r for r in rows if r.startswith(f"{jid}.batch|")).split("|")

    alloc, bt = batch(a)
    assert bt[1:4] == ["56459172K", "56459172K", "84688758K"] and alloc[4] == "512G"        # fixture value stays K
    assert alloc[1] == alloc[2] == alloc[3] == ""
    alloc, bt = batch(a, "--units=M")
    assert bt[1] == "55135.91M" and bt[2] == "55135.91M" and bt[3] == "82703.87M" and alloc[4] == "524288M"
    alloc, bt = batch(a, "--units=G")
    assert bt[1] == "53.84G" and alloc[4] == "512G"
    alloc, bt = batch(a, "--units=K")
    assert bt[1] == "56459172K" and alloc[4] == "536870912K"
    alloc, bt = batch(a, "--noconvert")
    assert bt[1] == "56459172K" and alloc[4] == "524288M"
    alloc, bt = batch(a, "--units=M", "--noconvert")                # --units takes precedence
    assert bt[1] == "55135.91M"
    # exact multiples auto-scale by default (EXACT flag), --noconvert stops that
    alloc, bt = batch(b)
    assert bt[1] == "4G" and bt[2] == "4G" and bt[3] == "6G"
    alloc, bt = batch(b, "--noconvert")
    assert bt[1] == "4194304K"
    alloc, bt = batch(b, "--units=M")
    assert bt[1] == "4096M"
    # research_6 section 2.2 query shape
    out = trace.ok("sacct", "-nP", "-j", str(a), "--units=M",
                   "-o", "JobID,JobName,State,ExitCode,MaxRSS,MaxRSSNode,MaxVMSize,AveRSS,TotalCPU,UserCPU,SystemCPU,Elapsed,NTasks,AllocTRES")
    rows = [ln.split("|") for ln in out.splitlines()]
    assert rows[0][4] == "" and rows[1][0] == f"{a}.batch" and rows[1][4] == "55135.91M" and rows[2][4] == "0"
    rc, out, err = trace("sacct", "-j", str(a), "-n", "-P", "--units=X", fmt)
    assert rc == 1 and err.startswith("sacct: error: Invalid --units")


def test_fakeslurm_maxrss_marker_is_deterministic(trace):
    """'#FAKESLURM maxrss=<K>[K|M|G]' fixes the batch step MaxRSS (it used to be reset at start and replaced
    by a synthetic value); bad values are an sbatch error, not a traceback."""
    plain = trace.submit(script=TRACE_GPU_SCRIPT.replace("exit=0", "exit=0 maxrss=52676680"))
    suffixed = trace.submit(script=TRACE_GPU_SCRIPT.replace("exit=0", "exit=0 maxrss=52676680K"))
    gig = trace.submit("-p", "cpuonly", "-J", "g", script="#!/bin/bash\n#FAKESLURM duration=10 maxrss=50G\nhostname\n")
    b2 = trace.submit(script=B2_GPU_SCRIPT.replace("#SBATCH --partition=GPU-shared\n", "#SBATCH -p batch\n")
                      .replace("#SBATCH --account=mch250030p\n", "").replace("#SBATCH --gres=gpu:h100-80:2\n", "#SBATCH --gpus=a40\n")
                      .replace("#SBATCH --qos=gpu\n", "#FAKESLURM duration=100 maxrss=56459172\n"))
    trace.advance(2000)
    for jid, want in ((plain, "52676680K"), (suffixed, "52676680K"), (gig, "52428800K"), (b2, "56459172K")):
        rows = trace.ok("sacct", "-j", str(jid), "-n", "-P", "--noconvert", "--format=JobID,State,MaxRSS").splitlines()
        assert rows[0].split("|")[1] == "COMPLETED", rows
        assert rows[1] == f"{jid}.batch|COMPLETED|{want}", rows
    assert trace.job(plain)["marker_max_rss_k"] == 52676680 and trace.job(plain)["max_rss_k"] == 52676680
    assert "Memory Utilized: 50.24 GB" in trace.ok("seff", str(plain))
    # a synthetic value is still substituted when there is no marker
    nomarker = trace.submit(script=TRACE_GPU_SCRIPT)
    trace.advance(1000)
    assert trace.ok("sacct", "-j", f"{nomarker}.batch", "-n", "-P", "--format=MaxRSS").strip() == "56459172K"
    # the marker survives a requeue (second incarnation reports the same value); ctl finish --maxrss overrides
    rq = trace.submit(script=TRACE_GPU_SCRIPT.replace("exit=0", "exit=0 maxrss=1048576"))
    trace.advance(5)
    trace.ok("scontrol", "requeue", str(rq))
    trace.advance(2000)
    dup = trace.ok("sacct", "-D", "-j", f"{rq}.batch", "-n", "-P", "--noconvert", "--format=JobID,State,MaxRSS").splitlines()
    assert dup == [f"{rq}.batch|CANCELLED|1048576K", f"{rq}.batch|COMPLETED|1048576K"]
    ov = trace.submit(script=TRACE_GPU_SCRIPT.replace("exit=0", "exit=0 maxrss=1048576").replace("duration=900", "duration=9000"))
    trace.advance(5)
    trace.ok("fakeslurm-ctl", "finish", str(ov), "--maxrss", "2097152")
    assert trace.ok("sacct", "-j", f"{ov}.batch", "-n", "-P", "--noconvert", "--format=MaxRSS").strip() == "2097152K"
    # bad marker values -> sbatch error rc 1 (no traceback), nothing submitted
    for marker in ("maxrss=52676680X", "maxrss=abc", "duration=soon", "exit=x"):
        rc, out, err = trace("sbatch", stdin=TRACE_GPU_SCRIPT.replace("duration=900 exit=0", marker))
        assert rc == 1 and out == "" and err.startswith(f"sbatch: error: invalid #FAKESLURM {marker.split('=')[0]}="), (marker, err)
    assert str(trace.dump()["next_jobid"]) not in trace.dump()["jobs"]


def test_missing_output_directory_fails_launch(trace):
    """Relative/absolute -o/-e whose directory does not exist: slurmd cannot open the file, the batch step fails
    at launch -> FAILED 0:53, no output file (used to be silently ignored).  [verify on cluster]"""
    bad = trace.submit(script=TRACE_GPU_SCRIPT.replace("logs/wobl_%j.out", "nodir/wobl_%j.out"))
    bad_err = trace.submit(script=TRACE_GPU_SCRIPT.replace("logs/wobl_%j.err", "nodir/wobl_%j.err"))
    good = trace.submit()
    trace.advance(2)
    for jid in (bad, bad_err):
        row = trace.squeue_row(jid)
        assert row[2] == "FAILED", row
        rows = trace.ok("sacct", "-j", str(jid), "-n", "-P", "--format=JobID,State,ExitCode,ElapsedRaw").splitlines()
        assert rows[0] == f"{jid}|FAILED|0:53|0" and rows[1] == f"{jid}.batch|FAILED|0:53|0" and rows[2] == f"{jid}.extern|COMPLETED|0:0|0"
        kv = dict(t.split("=", 1) for t in trace.ok("scontrol", "show", "job", str(jid), "-o").split())
        assert kv["JobState"] == "FAILED" and kv["ExitCode"] == "0:53" and kv["Reason"] == "NonZeroExitCode"
    assert not (trace.home / "nodir").exists()
    assert not (trace.home / "logs" / f"wobl_{bad_err}.out").exists()   # no output at all for a failed launch
    assert trace.squeue_row(good)[2] == "RUNNING" and (trace.home / "logs" / f"wobl_{good}.out").exists()
    assert [e["kind"] for e in trace.dump()["events"] if e.get("job") == bad][-2:] == ["launch_failed", "finish"]


def test_posix_paths_when_invoked_from_bash(trace, monkeypatch):
    """From the bash shims $PWD is the Git Bash spelling of the cwd (/c/Users/...); the fake records
    WorkDir/StdOut/StdErr/Command/SLURM_SUBMIT_DIR in that form and converts back only to touch files."""
    from fakeslurm import native_to_posix, posix_to_native, current_dir

    home_posix = native_to_posix(str(trace.home))
    assert home_posix.startswith("/c/") and posix_to_native(home_posix).lower() == str(trace.home).replace("\\", "/").lower()
    assert posix_to_native("/tmp/x/y").lower().endswith("/x/y") and posix_to_native("C:/a").startswith("C:")
    monkeypatch.setenv("PWD", home_posix)
    (trace.home / "work").mkdir()
    (trace.home / "job.sh").write_text(TRACE_GPU_SCRIPT, encoding="utf-8")
    jid = trace.submit()
    rel = trace.submit("job.sh")                                  # command path from a relative script arg
    sub = trace.submit("--chdir=work", "-o", "out_%j.txt", "-e", "out_%j.txt")   # relative --chdir: against the submit dir
    j = trace.job(jid)
    assert j["workdir"] == home_posix and j["stdout"] == "logs/wobl_%j.out"
    assert trace.job(rel)["command"] == home_posix + "/job.sh" and j["command"] == "(null)"
    assert trace.job(sub)["workdir"] == home_posix + "/work"
    trace.advance(2)
    kv = dict(t.split("=", 1) for t in trace.ok("scontrol", "show", "job", str(jid), "-o").split())
    assert kv["WorkDir"] == home_posix and kv["StdOut"] == f"{home_posix}/logs/wobl_{jid}.out" == kv["StdErr"].replace(".err", ".out")
    assert trace.squeue_row(jid)[16] == home_posix and trace.squeue_row(rel)[15] == home_posix + "/job.sh"
    assert trace.ok("squeue", "-h", "-j", str(sub), "-O", "WorkDir:0|,STDOUT:0").strip() == f"{home_posix}/work|{home_posix}/work/out_{sub}.txt"
    assert trace.ok("sacct", "-j", str(jid), "-n", "-P", "-X", "--format=WorkDir").strip() == home_posix
    assert "\\" not in kv["StdOut"] and "\\" not in kv["WorkDir"]
    assert (trace.home / "logs" / f"wobl_{jid}.out").exists() and (trace.home / "work" / f"out_{sub}.txt").exists()
    # the script really runs in the POSIX workdir with SLURM_SUBMIT_DIR in the same spelling
    envjob = trace.submit("--chdir=work", "-p", "cpuonly", "-J", "envp", "-o", "env_%j.out",
                          script="#!/bin/bash\necho dir=$SLURM_SUBMIT_DIR pwd=$PWD\n")
    trace.advance(2)
    trace.ok("fakeslurm-ctl", "run-script", str(envjob))
    assert (trace.home / "work" / f"env_{envjob}.out").read_text().strip() == f"dir={home_posix}/work pwd={home_posix}/work"
    # scontrol write batch_script accepts a POSIX target
    trace.ok("scontrol", "write", "batch_script", str(jid), home_posix + "/work/copy.sh")
    assert (trace.home / "work" / "copy.sh").read_text() == TRACE_GPU_SCRIPT
    # $PWD that does not name the cwd (e.g. pytest started from Git Bash) is ignored
    monkeypatch.setenv("PWD", "/c/definitely/not/here")
    assert current_dir() == os.getcwd()
    assert trace.job(trace.submit())["workdir"] == str(trace.home)
    # the MSYS runtime hands native programs a converted $PWD ('C:/Users/...'): still the POSIX spelling,
    # and drive-letter arguments produced by the same conversion (-o /c/x -> C:/x) are turned back
    monkeypatch.setenv("PWD", str(trace.home).replace("\\", "/"))
    conv = trace.submit("--chdir=" + str(trace.home).replace("\\", "/") + "/work", "-o", str(trace.home).replace("\\", "/") + "/logs/c_%j.out")
    assert trace.job(conv)["workdir"] == home_posix + "/work" and trace.job(conv)["stdout"] == home_posix + "/logs/c_%j.out"
