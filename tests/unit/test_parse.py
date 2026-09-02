"""Golden tests for slurm_mcp.slurm.parse (design sections 6.0-6.3).

Every file listed in tests/fixtures/{trace,bridges2}/index.json is parsed by at least one test with
concrete value assertions; the last test enforces that coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from slurm_mcp.slurm import parse as p
from slurm_mcp.slurm.states import JobState

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CLUSTERS = ("trace", "bridges2")
COVERED: dict[str, set[str]] = {cl: set() for cl in CLUSTERS}


def fx(cluster: str, name: str) -> str:
    """Fixture text; records the index key (file name without extension) as covered."""
    COVERED[cluster].add(name.rsplit(".", 1)[0])
    return (FIXTURES / cluster / name).read_text(encoding="utf-8")


def fxl(cluster: str, name: str) -> list[str]:
    return [l for l in fx(cluster, name).splitlines() if l.strip()]


# ---------------------------------------------------------------------------------------------------
# framing (6.0)
# ---------------------------------------------------------------------------------------------------

FRAMED = """::NOW 1756760000 tracevm01
::SQUEUE
615421|615421|615421|N/A|PENDING|batch
::RC 0
::SACCT
615411|615411|RUNNING
::RC 0
::SACCT
615427|615427|PENDING
::RC 1
::FILES
::CMDS
/ctrl/a1/c1.rc|0
::ENRICH
615400|615400|COMPLETED|0:0|56459172K|128G|10|cpu=8
::L /w/out.txt|last line of output
::RC 0
::END
trailing garbage after END is ignored
"""


def test_parse_sections_framing():
    s = p.parse_sections(FRAMED)
    assert s["NOW"] == (["1756760000 tracevm01"], None)
    assert p.parse_now(s) == (1756760000, "tracevm01")
    assert s["SQUEUE"] == (["615421|615421|615421|N/A|PENDING|batch"], 0)
    assert s["SACCT"] == (["615411|615411|RUNNING", "615427|615427|PENDING"], 1)   # chunks merged, first non-zero rc
    assert s["FILES"] == ([], None)
    assert s["CMDS"] == (["/ctrl/a1/c1.rc|0"], None)
    assert s["ENRICH"] == (["615400|615400|COMPLETED|0:0|56459172K|128G|10|cpu=8", "::L /w/out.txt|last line of output"], 0)
    assert "garbage" not in str(s)


def test_parse_sections_missing_end_raises():
    with pytest.raises(p.IncompleteProbe):
        p.parse_sections("::SQUEUE\n1|2\n::RC 0\n")
    with pytest.raises(p.IncompleteProbe):
        p.parse_sections("")


def test_parse_sections_crlf_and_pre_lines():
    s = p.parse_sections("banner line\r\n::T1\r\nsbatch: Job 1 to start at 5 using 1 processors on nodes n in partition x\r\n::RC 0\r\n::END\r\n")
    assert s["_pre"] == (["banner line"], None)
    assert s["T1"][1] == 0 and s["T1"][0][0].startswith("sbatch: Job 1")
    assert p.parse_now(s) is None


# ---------------------------------------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------------------------------------

def test_sentinels_and_scalars():
    for v in ("N/A", "Unknown", "None", "None assigned", "(null)", "UNLIMITED", "Partition_Limit", "INVALID", "", None):
        assert p.none_if_sentinel(v) is None and p.is_sentinel(v)
    assert p.none_if_sentinel(" trace13 ") == "trace13"
    assert p.parse_ts("1756760000") == 1756760000
    assert p.parse_ts("2026-09-01T20:58:03") == "2026-09-01T20:58:03"
    assert p.parse_ts("N/A") is None and p.parse_ts("Unknown") is None
    assert p.parse_int("251189") == 251189 and p.parse_int("N/A") is None and p.parse_int("x") is None
    assert p.parse_int("1.0") == 1
    assert p.parse_float("0.251142") == 0.251142 and p.parse_float("") is None
    assert p.parse_secs("300 sec") == 300 and p.parse_secs("0 min") == 0 and p.parse_secs("2000 usec") == 0
    assert p.parse_secs("00:00:00") == 0 and p.parse_secs("2-00:00:00") == 172800 and p.parse_secs("(null)") is None
    assert p.parse_exit_code("0:15") == (0, 15) and p.parse_exit_code("137:0") == (137, 0)
    assert p.parse_exit_code("") is None and p.parse_exit_code("1") == (1, 0)
    assert p.parse_tz_offset("-0400") == -14400 and p.parse_tz_offset("+0530") == 19800 and p.parse_tz_offset("x") is None
    assert p.strip_node_state("drain*") == "drain" and p.strip_node_state("idle~") == "idle" and p.strip_node_state("mix") == "mix"
    assert p.parse_aiot("124/4/0/128") == {"alloc": 124, "idle": 4, "other": 0, "total": 128}
    assert p.parse_aiot("N/A") is None and p.parse_aiot("1/2") is None
    assert p.parse_list("normal,batch") == ["normal", "batch"] and p.parse_list("(null)") == [] and p.parse_list("") == []
    assert p.split_fields("a|b|c|d", 3) == ["a", "b", "c|d"] and p.split_fields("a|b", 3) is None
    assert p.parse_version(["slurm 22.05.11"]) == "22.05.11"


def test_mem_helpers():
    assert p.mem_to_bytes("56459172K") == 56459172 * 1024
    assert p.mem_to_bytes("1.5G") == int(1.5 * 1024 ** 3)
    assert p.mem_to_bytes("512G") == 512 * 1024 ** 3
    assert p.mem_to_bytes("2048M") == 2048 * 1024 ** 2
    assert p.mem_to_bytes("0") == 0 and p.mem_to_bytes("") is None and p.mem_to_bytes("abc") is None
    assert p.req_mem_bytes("80Gn") == (80 * 1024 ** 3, "node")
    assert p.req_mem_bytes("8048Mc", 64) == (8048 * 1024 ** 2 * 64, "cpu")
    assert p.req_mem_bytes("8048Mc") == (8048 * 1024 ** 2, "cpu")
    assert p.req_mem_bytes("128G") == (128 * 1024 ** 3, None)
    assert p.req_mem_bytes("") == (None, None)


def test_parse_tres_and_gres_types():
    tres = p.parse_tres("billing=64,cpu=64,gres/gpu=1,gres/gpu:h100-80=2,mem=512G,node=1")
    assert tres == {"billing": 64, "cpu": 64, "gres/gpu": 1, "gres/gpu:h100-80": 2, "mem": "512G", "node": 1}
    assert p.gres_types_from_tres(tres) == {"h100-80": 2}
    assert p.parse_tres("") == {} and p.parse_tres("(null)") == {}
    assert p.parse_tres("defer,kill_invalid_depend") == {"defer": True, "kill_invalid_depend": True}


@pytest.mark.parametrize("text,expected", [
    ("gres:gpu:h100-80:1", {"type": "h100-80", "count": 1}),
    ("gres:gpu:1", {"type": None, "count": 1}),
    ("gres:gpu:4", {"type": None, "count": 4}),
    ("gres:gpu:h100-80", {"type": "h100-80", "count": 1}),
    ("gres:gpu:a40", {"type": "a40", "count": 1}),
    ("gres/gpu:h100-80=2", {"type": "h100-80", "count": 2}),
    ("gres/gpu=2", {"type": None, "count": 2}),
    ("gpu:v100-32:16(S:0-95)", {"type": "v100-32", "count": 16}),
    ("gpu:a40:1", {"type": "a40", "count": 1}),
    ("gpu:8", {"type": None, "count": 8}),
    ("gpu:mi210:4", {"type": "mi210", "count": 4}),
    ("gpu", {"type": None, "count": 1}),
    ("(null)", None), ("N/A", None), ("", None), (None, None), ("license:foo:2", None),
    ("nic:1,gpu:a100:2", {"type": "a100", "count": 2}),
])
def test_gres_spec_forms(text, expected):
    assert p.gres_spec(text) == expected


# ---------------------------------------------------------------------------------------------------
# 6.1 discovery sections (golden on fixtures)
# ---------------------------------------------------------------------------------------------------

def test_parse_env_design_form():
    """The tenth field is ``id -Gn`` (every group): AllowGroups is normally a supplementary group, so the
    primary group alone hid the partitions the user can really use (measured on TRACE 2026-09-02)."""
    env = p.parse_env(
        ["/trace/home/wxu2|wxu2|tracevm01.trace.cmu.edu|||/local|1756760000|-0400|users|users biosimmlab"])
    assert env == {"home": "/trace/home/wxu2", "user": "wxu2", "hostname": "tracevm01.trace.cmu.edu", "project": None,
                   "scratch": None, "local": "/local", "remote_now": 1756760000, "tz_offset_s": -14400,
                   "group": "users", "groups": ["users", "biosimmlab"]}
    # an older capture without the tenth field falls back to the primary group
    legacy = p.parse_env(["/h|u|host|||/local|1756760000|-0400|users"])
    assert legacy["group"] == "users" and legacy["groups"] == ["users"]
    assert p.parse_env([]) == {}


def test_parse_env_fixtures():
    trace = p.parse_env(fxl("trace", "env.out"))
    assert trace["home"] == "/trace/home/wxu2" and trace["user"] == "wxu2" and trace["project"] is None
    assert trace["shell"] == "/bin/bash" and trace["slurm_version"] == "22.05.11"
    b2 = p.parse_env(fxl("bridges2", "env.out"))
    assert b2["home"] == "/jet/home/wxu7" and b2["user"] == "wxu7"
    assert b2["project"] == "/ocean/projects/mch250030p/wxu7" and b2["slurm_version"] == "22.05.11"


def test_parse_config_trace():
    cfg = p.parse_config(fxl("trace", "scontrol_config.out"))
    assert cfg["cluster_name"] == "trace" and cfg["slurm_version"] == "22.05.11"
    assert cfg["job_requeue"] is False and cfg["raw"]["JobRequeue"] == "0"
    assert cfg["min_job_age_s"] == 300 and cfg["kill_wait_s"] == 300 and cfg["message_timeout_s"] == 30
    assert cfg["cmd_timeout_s"] == 120
    assert cfg["scheduler_parameters"] == {"bf_max_job_test": 2600, "bf_max_job_user": 10}
    assert cfg["preempt_mode"] == ["GANG", "REQUEUE"] and cfg["preempt_type"] == "preempt/partition_prio"
    assert cfg["preempt_exempt_time_s"] == 0
    assert cfg["epoch_format"] is False and cfg["boot_time"] == "2026-08-04T15:27:13"
    assert cfg["comment_stored"] is False
    assert cfg["accounting_storage_enforce"] == ["associations", "limits", "qos", "safe"]
    assert cfg["enforce_part_limits"] == "ALL" and cfg["def_mem_per_node"] is None
    assert cfg["max_array_size"] == 1001 and cfg["max_job_count"] == 100000 and cfg["mail_prog"] == "/bin/mail"
    assert cfg["priority_weights"] == {"age": 10000, "fairshare": 1000000, "qos": 5000000, "partition": 0, "jobsize": 0}


def test_parse_config_bridges2():
    cfg = p.parse_config(fxl("bridges2", "scontrol_config.out"))
    assert cfg["cluster_name"] == "bridges2" and cfg["job_requeue"] is True
    assert cfg["min_job_age_s"] == 200 and cfg["kill_wait_s"] == 400 and cfg["message_timeout_s"] == 200
    assert cfg["cmd_timeout_s"] == 260
    sched = cfg["scheduler_parameters"]
    assert sched["bf_max_job_user_part"] == 20 and sched["defer"] is True and sched["kill_invalid_depend"] is True
    assert sched["max_rpc_cnt"] == 100 and sched["bf_continue"] is True and "bf_max_job_user" not in sched
    assert cfg["preempt_mode"] == ["OFF"] and cfg["preempt_type"] == "preempt/none"
    assert cfg["comment_stored"] is True and cfg["epoch_format"] is False


def test_parse_config_grep_subset_with_epoch_boot_time():
    cfg = p.parse_config(["ClusterName             = fake", "BOOT_TIME               = 1756000000",
                          "MessageTimeout          = 10 sec", "JobRequeue              = 1",
                          "SchedulerParameters     = (null)"])
    assert cfg["epoch_format"] is True and cfg["boot_time"] == 1756000000
    assert cfg["cmd_timeout_s"] == 120 and cfg["job_requeue"] is True and cfg["scheduler_parameters"] == {}
    assert p.parse_config([])["job_requeue"] is None


def test_parse_partitions_trace():
    parts = p.parse_partitions(fxl("trace", "scontrol_partitions.out"))
    assert list(parts) == ["batch", "cpuonly", "cpuonly-debug", "biosimmlab"]
    b = parts["batch"]
    assert b["allow_qos"] == ["normal", "batch"] and b["qos"] == "batchpartition" and b["default"] is False
    assert b["max_time_s"] is None and b["default_time_s"] == 3600 and b["grace_time_s"] == 0
    assert b["max_nodes"] is None and b["priority_tier"] == 1 and b["over_subscribe"] == "FORCE:1"
    assert b["preempt_mode"] == ["REQUEUE"] and b["state"] == "UP"
    assert b["total_nodes"] == 29 and b["total_cpus"] == 3712 and b["def_mem_per_cpu"] == 8048
    assert b["max_mem_per_node"] is None and b["job_defaults"] == {}
    assert b["tres"]["gres/gpu"] == 29 and b["gpu_total"] == 29 and b["gres_types"] == {} and b["has_gpu"] is True
    assert b["nodes"] == "trace[01-29]" and b["tres_billing_weights"] == {}
    cpu = parts["cpuonly"]
    assert cpu["has_gpu"] is False and cpu["gpu_total"] is None and cpu["max_time_s"] == 172800
    assert cpu["preempt_mode"] == ["GANG", "REQUEUE"] and cpu["allow_qos"] == ["ALL"]
    assert cpu["allow_groups"][-1] == "biosimmlab" and cpu["qos"] is None
    assert parts["cpuonly-debug"]["max_time_s"] == 4 * 3600 and parts["biosimmlab"]["priority_tier"] == 20


def test_parse_partitions_bridges2():
    parts = p.parse_partitions(fxl("bridges2", "scontrol_partitions.out"))
    assert len(parts) == 13 and parts["RM"]["default"] is True
    gs = parts["GPU-shared"]
    assert gs["job_defaults"] == {"DefMemPerGPU": 63000} and gs["max_nodes"] == 1 and gs["over_subscribe"] == "YES:4"
    assert gs["gres_types"] == {"l40s-48": 24, "v100-16": 72, "v100-32": 200} and gs["gpu_total"] == 376
    assert gs["qos"] == "gpusharedpartition" and gs["allow_qos"] == ["gpu", "gpuinteract", "low", "push", "unlimited"]
    assert gs["preempt_mode"] == ["OFF"] and gs["def_mem_per_node"] is None
    assert parts["RM-shared"]["def_mem_per_cpu"] == 1900 and parts["RM"]["def_mem_per_node"] == 240000
    assert parts["EM"]["max_mem_per_cpu"] == 42555 and parts["applications"]["max_time_s"] == 8 * 3600
    assert parts["GPU-small"]["gres_types"] == {"v100-32": 8} and parts["HACC"]["gres_types"] == {"mi210": 16}
    assert parts["ROBO"]["default_time_s"] is None


def test_parse_sinfo_nodes_trace_and_aggregate():
    rows = p.parse_sinfo_nodes(fxl("trace", "sinfo_nodes.out"))
    assert len(rows) == 82
    r0 = rows[0]
    assert r0["node"] == "trace01" and r0["partition"] == "batch" and r0["state"] == "mix" and r0["responding"]
    assert r0["cpus"] == 128 and r0["mem_mb"] == 2063700 and r0["gres"] == {"type": "a40", "count": 1}
    assert r0["gres_raw"] == "gpu:a40:1" and r0["features"] == []
    assert r0["cpus_aiot"] == {"alloc": 124, "idle": 4, "other": 0, "total": 128}
    assert r0["free_mem_mb"] == 868612 and r0["cpu_load"] == 117.04
    agg = p.aggregate_sinfo(rows)
    assert agg["batch"]["nodes"] == 29 and agg["biosimmlab"]["nodes"] == 29
    assert agg["cpuonly"]["nodes"] == 23 and agg["cpuonly-debug"]["nodes"] == 1
    assert agg["batch"]["gres"] == {"a40": {"nodes": 29, "per_node": 1}} and agg["cpuonly"]["gres"] == {}
    assert agg["batch"]["max_cpus"] == 128 and agg["batch"]["max_mem_mb"] == 2063700
    assert agg["batch"]["states"] == {"mix": 6, "alloc": 23}
    assert agg["cpuonly"]["states"] == {"down": 1, "mix": 1, "alloc": 21}


def test_parse_sinfo_nodes_bridges2():
    rows = p.parse_sinfo_nodes(fxl("bridges2", "sinfo_nodes.out"))
    assert len(rows) == 1009
    dgx = [r for r in rows if r["gres_raw"] == "gpu:v100-32:16(S:0-95)"]
    assert dgx and dgx[0]["gres"] == {"type": "v100-32", "count": 16} and dgx[0]["node"] == "v034"
    assert dgx[0]["state"] == "drain" and dgx[0]["state_raw"] == "drain*" and dgx[0]["responding"] is False
    assert dgx[0]["features"] == ["v100", "v100-32", "dgx-2"]
    hacc = [r for r in rows if r["node"] == "hacc-gm001"][0]
    assert hacc["state"] == "down" and hacc["gres"] == {"type": "mi210", "count": 4} and hacc["free_mem_mb"] is None
    agg = p.aggregate_sinfo(rows)
    assert agg["GPU-shared"]["gres"]["v100-32"]["per_node"] == 16 and agg["GPU-shared"]["gres"]["h100-80"]["per_node"] == 8
    assert agg["GPU-shared"]["nodes"] == 46 and agg["RM"]["nodes"] == 484 and agg["GPU-dev"]["gres"] == {"a100": {"nodes": 1, "per_node": 1}}


def test_parse_sinfo_nodes_design_form():
    rows = p.parse_sinfo_nodes(["n1|GPU-shared|idle~|48|515000|gpu:8|feat1,feat2", "n2|RM|alloc|128|256000|(null)|(null)"])
    assert rows[0]["state"] == "idle" and rows[0]["gres"] == {"type": None, "count": 8} and rows[0]["features"] == ["feat1", "feat2"]
    assert rows[1]["gres"] is None and rows[1]["features"] == [] and "cpus_aiot" not in rows[1]
    assert p.aggregate_sinfo(rows)["GPU-shared"]["gres"] == {None: {"nodes": 1, "per_node": 8}}


def test_parse_sinfo_partitions_and_summary_fixtures():
    tp = p.parse_sinfo_partitions(fxl("trace", "sinfo_partitions.out"))
    assert len(tp) == 8 and tp[0]["partition"] == "batch" and tp[0]["max_time_s"] is None and tp[0]["node_count"] == 6
    assert tp[0]["state"] == "mix" and tp[0]["cpus"] == {"alloc": 666, "idle": 102, "other": 0, "total": 768}
    assert tp[0]["gres"] == {"type": "a40", "count": 1} and tp[0]["nodes_aiot"]["total"] == 6
    assert tp[2]["partition"] == "cpuonly" and tp[2]["state"] == "down" and tp[2]["max_time_s"] == 172800
    bp = p.parse_sinfo_partitions(fxl("bridges2", "sinfo_partitions.out"))
    assert bp[0]["partition"] == "RM" and bp[0]["default"] is True and bp[0]["state"] == "comp"
    ts = p.parse_sinfo_summary(fxl("trace", "sinfo_summary.out"))
    assert [r["partition"] for r in ts] == ["batch", "cpuonly", "cpuonly-debug", "biosimmlab"]
    assert ts[1]["nodes_aiot"] == {"alloc": 22, "idle": 0, "other": 1, "total": 23} and ts[2]["max_time_s"] == 14400
    bs = p.parse_sinfo_summary(fxl("bridges2", "sinfo_summary.out"))
    assert bs[0]["partition"] == "RM" and bs[0]["default"] and bs[-1]["partition"] == "applications"


def test_parse_user():
    assert p.parse_user(["wxu2|biosimmlab"]) == {"user": "wxu2", "default_account": "biosimmlab"}
    assert p.parse_user([]) == {"user": None, "default_account": None}


def test_parse_assoc_fixtures_and_design_form():
    t = p.parse_assoc(fxl("trace", "sacctmgr_assoc.out"))
    assert len(t) == 1 and t[0]["cluster"] == "trace" and t[0]["account"] == "biosimmlab" and t[0]["partition"] is None
    assert t[0]["qos_list"] == ["batchpartition", "cpuonly-debug-qos", "normal", "prioritypartition"]
    assert t[0]["max_jobs"] is None and t[0]["grp_tres"] == {}
    b = p.parse_assoc(fxl("bridges2", "sacctmgr_assoc.out"))
    assert b[0]["account"] == "mch250030p" and b[0]["qos_list"] == ["ft", "gpu", "gpuinteract", "low", "push", "unlimited"]
    d = p.parse_assoc(["trace|biosimmlab|batch|normal,batch|normal|cpu=24|billing=1000|5|10|gres/gpu=2|2-00:00:00"])
    assert d[0]["partition"] == "batch" and d[0]["default_qos"] == "normal" and d[0]["grp_tres"] == {"cpu": 24}
    assert d[0]["grp_tres_mins"] == {"billing": 1000} and d[0]["max_jobs"] == 5 and d[0]["max_submit"] == 10
    assert d[0]["max_tres"] == {"gres/gpu": 2} and d[0]["max_wall_s"] == 172800


def test_parse_qos_bridges2():
    q = p.parse_qos(fxl("bridges2", "sacctmgr_qos.out"))
    gs = q["gpusmallpartition"]
    assert gs["max_wall_s"] == 8 * 3600 and gs["max_jobs_pu"] == 2 and gs["max_submit_pu"] == 10
    assert gs["max_tres"] == {"gres/gpu": 16} and gs["flags"] == ["DenyOnLimit"] and gs["max_tres_pu"] == {}
    rs = q["rmsharedpartition"]
    assert rs["max_wall_s"] == 3 * 86400 and rs["max_tres_pu"] == {"cpu": 25600} and rs["max_tres"] == {"cpu": 64, "node": 1}
    assert q["gpupartition"]["max_wall_s"] == 2 * 86400 and q["gpupartition"]["max_tres"] == {"gres/gpu": 64, "node": 8}
    assert q["gpusharedpartition"]["max_tres"] == {} and q["gpusharedpartition"]["max_jobs_pu"] is None
    assert q["low"]["flags"] == ["NoReserve", "OverPartQOS"] and q["low"]["max_wall_s"] is None
    assert q["rminteract"]["priority"] == 50000000 and q["gpu"]["max_tres_pu"] == {"gres/gpu": 128}
    assert q["gpu"]["max_submit_pu"] == 5000 and q["normal"]["priority"] == 0


def test_parse_qos_trace_and_design_form():
    q = p.parse_qos(fxl("trace", "sacctmgr_qos.out"))
    assert q["batchpartition"]["max_wall_s"] == 2 * 86400 and q["priorityphase1node1"]["max_wall_s"] == 2 * 86400
    assert q["cpuonly-debug-qos"]["max_tres_pu"] == {"cpu": 24} and q["cpuonly-debug"]["max_jobs_pu"] == 1
    assert q["priorityoneandhalfnode"]["max_tres"] == {"cpu": 192, "gres/gpu": 2} and q["normal"]["max_wall_s"] is None
    d = p.parse_qos(["gpu|1000000|00:01:00|2-00:00:00|gres/gpu=8|gres/gpu=128|4|5000|cpu=10|low,normal|REQUEUE|DenyOnLimit|0.500000"])
    g = d["gpu"]
    assert g["priority"] == 1000000 and g["grace_time_s"] == 60 and g["max_wall_s"] == 172800
    assert g["max_tres"] == {"gres/gpu": 8} and g["max_tres_pu"] == {"gres/gpu": 128} and g["max_jobs_pu"] == 4
    assert g["max_submit_pu"] == 5000 and g["grp_tres"] == {"cpu": 10} and g["preempt"] == ["low", "normal"]
    assert g["preempt_mode"] == "REQUEUE" and g["flags"] == ["DenyOnLimit"] and g["usage_factor"] == 0.5


def test_parse_sshare_fixtures_and_design_form():
    t = p.parse_sshare(fxl("trace", "sshare_me.out"))
    assert len(t) == 1 and t[0]["account"] == "biosimmlab" and t[0]["user"] == "wxu2"
    assert t[0]["fair_share"] == 0.251142 and t[0]["raw_usage"] == 10301565 and t[0]["su_balance"] is None
    b = p.parse_sshare(fxl("bridges2", "sshare_me.out"))
    assert b[0]["account"] == "mch250030p" and b[0]["fair_share"] == 0.255344
    d = p.parse_sshare(["mch250030p|wxu7|0.25|billing=600000|billing=150000"])
    assert d[0]["su_balance"] == (600000 - 150000) / 60 and d[0]["grp_tres_mins"] == {"billing": 600000}
    assert p.parse_sshare([]) == []


def test_parse_balance():
    text = ["Project: mch250030p", "Total SUs: 12,500.00  Used: 3,000.5  Remaining: 9,499.50"]
    out = p.parse_balance(text, r"Total SUs: (?P<total>[\d,.]+).*Remaining: (?P<left>[\d,.]+)")
    assert out == {"total": 12500.0, "left": 9499.5}
    assert p.parse_balance(text, None) is None and p.parse_balance(text, r"nomatch (?P<left>\d+)") is None
    assert p.parse_balance(["left 5"], r"left (?P<left>\d+)") == {"left": 5.0}


def test_parse_reservations():
    lines = [
        "ReservationName=maint_2026 StartTime=1757000000 EndTime=1757086400 Duration=1-00:00:00 Nodes=trace[01-29] "
        "NodeCnt=29 CoreCnt=3712 Features=(null) PartitionName=batch Flags=MAINT,IGNORE_JOBS,SPEC_NODES TRES=cpu=3712 "
        "Users=(null) Groups=(null) Accounts=(null) Licenses=(null) State=INACTIVE BurstBuffer=(null) Watts=n/a",
        "ReservationName=lab_resv StartTime=2026-09-05T08:00:00 EndTime=2026-09-05T18:00:00 Duration=10:00:00 "
        "Nodes=v001 NodeCnt=1 PartitionName=(null) Flags=SPEC_NODES Users=wxu7,other Accounts=(null)",
        "ReservationName=Maintenance-window StartTime=1 EndTime=2 Nodes=ALL Flags=SPEC_NODES",
    ]
    rows = p.parse_reservations(lines)
    assert [r["name"] for r in rows] == ["maint_2026", "lab_resv", "Maintenance-window"]
    assert rows[0]["start"] == 1757000000 and rows[0]["end"] == 1757086400 and rows[0]["maintenance"] is True
    assert rows[0]["partition"] == "batch" and rows[0]["node_count"] == 29 and "MAINT" in rows[0]["flags"]
    assert rows[1]["start"] == "2026-09-05T08:00:00" and rows[1]["maintenance"] is False and rows[1]["partition"] is None
    assert rows[1]["users"] == ["wxu7", "other"] and rows[1]["accounts"] == []
    assert rows[2]["maintenance"] is True
    assert p.parse_reservations([]) == [] and p.parse_reservations(["No reservations in the system"]) == []


def test_parse_tools_fixtures_and_design_form():
    t = p.parse_tools(fxl("trace", "tools.out"))
    assert t["sbatch"] and t["rsync"] and t["jq"] and t["flock"] and t["seff"] and len(t) == 17
    b = p.parse_tools(fxl("bridges2", "tools.out"))
    assert b["rsync"] is False and b["sbatch"] and b["python3"]
    d = p.parse_tools(["tar=1", "setsid=0", "seff=1", "flock=1", "garbage"])
    assert d == {"tar": True, "setsid": False, "seff": True, "flock": True}


def test_parse_cap_o():
    assert p.parse_cap_o(["rc=0"]) is True
    assert p.parse_cap_o(["rc=1"]) is False and p.parse_cap_o([]) is False


def test_parse_df_fixtures():
    t = p.parse_df(fxl("trace", "df_home.out"))
    assert len(t) == 1 and t[0]["mount"] == "/trace" and t[0]["path"] == "/trace" and t[0]["paths"] == ["/trace", "/trace"]
    assert t[0]["kb_total"] == 932 * 1024 ** 3 and t[0]["used_pct"] == 1 and t[0]["filesystem"] == "172.19.21.14:/trace"
    b = p.parse_df(fxl("bridges2", "df_home.out"))
    assert [r["mount"] for r in b] == ["/jet", "/ocean"]
    assert b[0]["used_pct"] == 70 and b[0]["kb_total"] == 25 * 1024 ** 2 and b[1]["used_pct"] == 95
    assert b[1]["kb_free"] == 23 * 1024 ** 2


def test_parse_df_design_form_appends_path():
    lines = ["172.19.21.14:/trace 1000727379968 1717986918 998990389248 1% /trace /trace/home/wxu2",
             "172.19.21.14:/trace 1000727379968 1717986918 998990389248 1% /trace /trace/group/biosimmlab/wxu2",
             "vast:/group 3221225472 2147483648 1073741824 67% /trace/group /trace/group/biosimmlab"]
    rows = p.parse_df(lines, roles={"/trace/home/wxu2": "home", "/trace/group/biosimmlab": "quota"})
    assert len(rows) == 2
    assert rows[0] == {"path": "/trace/home/wxu2", "mount": "/trace", "filesystem": "172.19.21.14:/trace",
                       "kb_total": 1000727379968, "kb_used": 1717986918, "kb_free": 998990389248, "used_pct": 1,
                       "role": "home", "paths": ["/trace/home/wxu2", "/trace/group/biosimmlab/wxu2"]}
    assert rows[1]["path"] == "/trace/group/biosimmlab" and rows[1]["used_pct"] == 67 and rows[1]["role"] == "quota"
    assert p.parse_df([]) == [] and p.parse_df(["Filesystem 1024-blocks Used Available Capacity Mounted"]) == []


def test_quota_fixture_is_empty():
    for cl in CLUSTERS:
        assert fx(cl, "quota.out") == ""
        assert p.parse_df(fxl(cl, "quota.out")) == []


# ---------------------------------------------------------------------------------------------------
# 6.2 tick sections
# ---------------------------------------------------------------------------------------------------

SQUEUE_ME_FMT = "%i|%j|%T|%P|%R|%M|%l|%D|%C|%b|%S|%V|%Q|%r|%N|%o|%Z|%u|%a|%q|%k|%e|%L"


def test_squeue_me_fixture_trace():
    rows = p.parse_squeue_rows(fxl("trace", "squeue_me.out"), SQUEUE_ME_FMT)
    assert len(rows) == 12
    r = {x["display_id"]: x for x in rows}
    j = r["615421"]
    assert j["name"] == "mixed11_scalar" and j["state"] == "PENDING" and j["job_state"] is JobState.SUBMITTED
    assert j["partition"] == "batch" and j["partitions"] == ["batch"] and j["reason"] == "Resources"
    assert j["reason_or_nodes"] == "(Resources)" and j["elapsed"] == "0:00" and j["elapsed_s"] == 0
    assert j["time_limit"] == "1-00:00:00" and j["time_limit_s"] == 86400 and j["num_nodes"] == 1 and j["num_cpus"] == 64
    assert j["tres_per_node"] == "N/A" and j["gres"] is None
    assert j["start"] == "2026-09-01T20:58:03" and j["submit"] == "2026-09-01T16:44:35" and j["priority"] == 251189
    assert j["nodes"] is None and j["command"].endswith("/jobs/train_mixed11_scalar.job")
    assert j["workdir"] == "/trace/group/biosimmlab/wxu2/vascular_super_resolution"
    assert j["user"] == "wxu2" and j["account"] == "biosimmlab" and j["qos"] == "normal" and j["comment"] is None
    assert j["end"] == "2026-09-02T20:58:03" and j["time_left"] == "1-00:00:00"
    dep = r["615433"]
    assert dep["reason"] == "Dependency" and dep["start"] is None and dep["end"] is None and dep["time_limit_s"] == 6 * 3600
    run = r["615411"]
    assert run["state"] == "RUNNING" and run["job_state"] is JobState.RUNNING and run["nodes"] == "trace13"
    assert run["elapsed_s"] == 14 * 60 + 47 and run["reason"] == "None" and run["reason_or_nodes"] == "trace13"


def test_squeue_me_fixture_bridges2_empty():
    assert fx("bridges2", "squeue_me.out") == ""
    assert p.parse_squeue_rows(fxl("bridges2", "squeue_me.out"), SQUEUE_ME_FMT) == []
    assert fx("bridges2", "squeue_me_start.out") == ""
    assert p.parse_squeue_rows(fxl("bridges2", "squeue_me_start.out"), "%i|%j|%P|%S|%R|%Q|%T") == []


def test_squeue_me_start_fixture_trace():
    rows = p.parse_squeue_rows(fxl("trace", "squeue_me_start.out"), "%i|%j|%P|%S|%R|%Q|%T")
    assert len(rows) == 9 and rows[0]["display_id"] == "615421"
    assert rows[0]["start"] == "2026-09-01T20:58:03" and rows[0]["reason"] == "Resources" and rows[0]["priority"] == 251189
    assert rows[1]["display_id"] == "615427" and rows[1]["start"] == "2026-09-01T21:54:30" and rows[1]["reason"] == "Priority"
    assert rows[-1]["start"] is None and rows[-1]["reason"] == "Dependency" and rows[-1]["job_state"] is JobState.SUBMITTED


TICK_ROWS = [
    "615421|615421|615421|N/A|PENDING|batch|normal|1756760283|1756846683|1756745075|1-00:00:00|0:00|251189||N/A|(null)|"
    "/trace/group/biosimmlab/wxu2/vascular_super_resolution/jobs/train_mixed11_scalar.job|"
    "/trace/group/biosimmlab/wxu2/vascular_super_resolution|Resources",
    "615411|615411|615411|N/A|RUNNING|batch|normal|1756745390|1756831790|1756744713|1-00:00:00|14:47|251174|trace13|N/A|"
    "slurm-mcp:j17:1:deadbeef|/h/.slurm-mcp/jobs/j17/a1/job.sbatch|/h/work/j17|None",
    "700005|700001_4|700001|4|PENDING|GPU,GPU-shared|gpu|N/A|N/A|1756745390|UNLIMITED|0:00|1000||gres:gpu:h100-80:1|"
    "slurm-mcp:j18:1:cafe|/h/.slurm-mcp/jobs/j18/a1/job.sbatch|/h/work/my dir, with comma|Priority",
    "700009|700009|700009|N/A|COMPLETING|RM-shared|low|1756745390|1756745400|1756745000|02:00:00|2-01:02:03|5|r001|gres:gpu:2|"
    "(null)|/x/job.sbatch|/x|None",
    "800001|800001|800001|N/A|REQUEUED|batch|normal|N/A|N/A|1756745390|1-00:00:00|0:00|1|(null)|N/A|(null)|/x/job.sbatch|/x|"
    "JobHeldUser",
]


def test_parse_squeue_tick_all_fields_and_sentinels():
    rows = p.parse_squeue_tick(TICK_ROWS)
    assert len(rows) == 5
    a = rows[0]
    assert a["slurm_id"] == 615421 and a["display_id"] == "615421" and a["array_job_id"] == 615421 and a["array_index"] is None
    assert a["state"] == "PENDING" and a["job_state"] is JobState.SUBMITTED and a["partitions"] == ["batch"]
    assert a["qos"] == "normal" and a["start"] == 1756760283 and a["end"] == 1756846683 and a["submit"] == 1756745075
    assert a["time_limit_s"] == 86400 and a["elapsed_s"] == 0 and a["priority"] == 251189 and a["nodes"] is None
    assert a["tres_per_node"] == "N/A" and a["gres"] is None and a["comment"] is None
    assert a["command"].endswith("train_mixed11_scalar.job") and a["reason"] == "Resources"
    b = rows[1]
    assert b["job_state"] is JobState.RUNNING and b["nodes"] == "trace13" and b["comment"] == "slurm-mcp:j17:1:deadbeef"
    assert b["command"] == "/h/.slurm-mcp/jobs/j17/a1/job.sbatch" and b["elapsed_s"] == 887 and b["reason"] == "None"
    e = rows[2]
    assert e["slurm_id"] == 700005 and e["display_id"] == "700001_4" and e["array_job_id"] == 700001 and e["array_index"] == 4
    assert e["partitions"] == ["GPU", "GPU-shared"] and e["time_limit_s"] is None and e["start"] is None
    assert e["gres"] == {"type": "h100-80", "count": 1} and e["workdir"] == "/h/work/my dir, with comma"
    cg = rows[3]
    assert cg["job_state"] is JobState.COMPLETING and cg["elapsed_s"] == 2 * 86400 + 3723 and cg["gres"] == {"type": None, "count": 2}
    h = rows[4]
    assert h["job_state"] is JobState.SUBMITTED and h["reason"] == "JobHeldUser" and h["nodes"] is None


def test_parse_squeue_tick_skips_short_rows_and_unknown_state():
    rows = p.parse_squeue_tick(["garbage", "1|1|1|N/A|WEIRD|p|q|N/A|N/A|1|1:00|0:00|1||N/A|(null)|/c|/w|None"])
    assert len(rows) == 1 and rows[0]["state"] == "WEIRD" and rows[0]["job_state"] is None


def test_parse_restarts():
    assert p.parse_restarts(["615427|0|1|", "615411|2|0|", "bad", "x|1|"]) == {
        615427: {"restarts": 0, "requeue": 1}, 615411: {"restarts": 2, "requeue": 0}}
    assert p.parse_restarts(["5|3"]) == {5: {"restarts": 3, "requeue": None}}


def test_sacct_job_fixtures_trace():
    rows = p.parse_sacct_rows(fxl("trace", "sacct_job_615411.out"), p.FIXTURE_SACCT_FIELDS)
    assert [r["job_id"] for r in rows] == ["615411", "615411.batch", "615411.extern"]
    a = rows[0]
    assert a["slurm_id"] == 615411 and a["step"] is None and a["state"] == "RUNNING" and a["job_state"] is JobState.RUNNING
    assert a["exit_code"] == (0, 0) and a["derived_exit_code"] == (0, 0) and a["elapsed_s"] == 920
    assert a["start"] == "2026-09-01T16:49:50" and a["end"] is None and a["submit"] == "2026-09-01T16:38:33"
    assert a["partition"] == "batch" and a["account"] == "biosimmlab" and a["qos"] == "normal" and a["nodelist"] == "trace13"
    assert a["alloc_tres"] == {"billing": 64, "cpu": 64, "gres/gpu": 1, "mem": "512G", "node": 1}
    assert a["req_tres"]["gres/gpu"] == 1 and a["max_rss_bytes"] is None and a["reason"] == "Dependency"
    assert a["workdir"] == "/trace/group/biosimmlab/wxu2/vascular_super_resolution"
    assert a["timelimit"] == "1-00:00:00" and a["timelimit_min"] == 1440 and a["timelimit_s"] == 86400
    assert a["ncpus"] == 64 and a["nnodes"] == 1 and a["flags"] == ["SchedMain", "StartRecieved"]
    assert a["submit_line"] == "sbatch --dependency=afterok:615408 jobs/train_wobl_notaylor.job"
    assert rows[1]["step"] == "batch" and rows[1]["slurm_id"] == 615411 and rows[1]["partition"] is None
    g = p.group_incarnations(rows)[615411]
    assert g["incarnations"] == 1 and g["current"] is a and sorted(g["steps"], key=lambda r: r["step"])[0]["step"] == "batch"
    pend = p.parse_sacct_rows(fxl("trace", "sacct_job_615427.out"), p.FIXTURE_SACCT_FIELDS)
    assert len(pend) == 1 and pend[0]["state"] == "PENDING" and pend[0]["start"] is None and pend[0]["end"] is None
    assert pend[0]["nodelist"] is None and pend[0]["alloc_tres"] == {} and pend[0]["reason"] == "None"
    assert pend[0]["submit_line"] == "sbatch --parsable jobs/train_wobl.job" and pend[0]["elapsed_s"] == 0


def test_sacct_job_fixture_bridges2_timeout():
    rows = p.parse_sacct_rows(fxl("bridges2", "sacct_job_44809480.out"), p.FIXTURE_SACCT_FIELDS)
    a, batch, extern = rows
    assert a["state"] == "TIMEOUT" and a["job_state"] is JobState.TIMEOUT and a["exit_code"] == (0, 0)
    assert a["elapsed_s"] == 28805 and a["timelimit_min"] == 480 and a["timelimit_s"] == 28800
    assert a["partition"] == "GPU-shared" and a["qos"] == "gpu" and a["nodelist"] == "w008"
    assert a["alloc_tres"] == {"billing": 24, "cpu": 24, "gres/gpu": 2, "mem": "128G", "node": 1}
    assert a["start"] == "2026-08-29T22:04:52" and a["end"] == "2026-08-30T06:04:57"
    assert a["flags"] == ["SchedBackfill", "StartRecieved"] and "--parsable" in a["submit_line"]
    assert a["workdir"] == "/ocean/projects/mch250030p/wxu7/llm_finetune"
    assert batch["step"] == "batch" and batch["state"] == "CANCELLED" and batch["job_state"] is JobState.CANCELLED
    assert batch["exit_code"] == (0, 15) and batch["derived_exit_code"] is None and batch["cancelled_by"] is None
    assert batch["max_rss_bytes"] == 56459172 * 1024 and batch["elapsed_s"] == 28813
    assert extern["step"] == "extern" and extern["state"] == "COMPLETED" and extern["max_rss_bytes"] == 0
    g = p.group_incarnations(rows)[44809480]
    assert g["current"]["job_state"] is JobState.TIMEOUT and g["incarnations"] == 1
    assert sorted(r["step"] for r in g["steps"]) == ["batch", "extern"]


def test_sacct_bad_job_fixtures():
    t = p.parse_sacct_rows(fxl("trace", "sacct_bad_job.out"), ("JobID", "State"))
    assert len(t) == 1 and t[0]["slurm_id"] == 1 and t[0]["state"] == "CANCELLED" and t[0]["cancelled_by"] == "51559"
    assert t[0]["job_state"] is JobState.CANCELLED and t[0]["state_raw"] == "CANCELLED by 51559"
    assert fx("bridges2", "sacct_bad_job.out") == ""
    assert p.parse_sacct_rows(fxl("bridges2", "sacct_bad_job.out"), ("JobID", "State")) == []


@pytest.mark.parametrize("cluster,name,expect_steps", [
    ("trace", "sacct_me_alloc.out", False), ("trace", "sacct_me_steps.out", True),
    ("bridges2", "sacct_me_alloc.out", False), ("bridges2", "sacct_me_steps.out", True),
])
def test_sacct_history_fixtures_parse_completely(cluster, name, expect_steps):
    lines = fxl(cluster, name)
    rows = p.parse_sacct_rows(lines, p.FIXTURE_SACCT_FIELDS)
    assert len(rows) == len(lines)
    assert all(isinstance(r["slurm_id"], int) for r in rows)
    assert all(r["job_state"] is not None for r in rows), {r["state"] for r in rows if r["job_state"] is None}
    assert any(r["step"] for r in rows) is expect_steps
    groups = p.group_incarnations(rows)
    assert all(g["current"] is not None for g in groups.values())
    allocs = [r for r in rows if r["step"] is None]
    assert all(r["timelimit_min"] is not None and r["timelimit_s"] == r["timelimit_min"] * 60 for r in allocs)
    assert all(r["submit"] is not None for r in allocs)


def test_sacct_history_fixture_values():
    t = p.parse_sacct_rows(fxl("trace", "sacct_me_alloc.out"), p.FIXTURE_SACCT_FIELDS)
    assert t[0]["slurm_id"] == 532597 and t[0]["job_name"] == "geonorm_v2_mp15" and t[0]["state"] == "COMPLETED"
    assert t[0]["elapsed_s"] == 13953 and t[0]["nodelist"] == "trace11" and t[0]["alloc_tres"]["gres/gpu"] == 1
    b = p.parse_sacct_rows(fxl("bridges2", "sacct_me_alloc.out"), p.FIXTURE_SACCT_FIELDS)
    states = {r["state"] for r in b}
    assert {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"} <= states
    cancelled = [r for r in b if r["state"] == "CANCELLED"]
    assert cancelled and all(r["cancelled_by"] for r in cancelled)
    assert b[0]["partition"] == "RM-shared" and b[0]["qos"] == "low" and b[0]["req_tres"]["mem"] == "30400M"
    steps = p.parse_sacct_rows(fxl("bridges2", "sacct_me_steps.out"), p.FIXTURE_SACCT_FIELDS)
    rss = sorted((r["max_rss_bytes"] for r in steps if r["max_rss_bytes"]), reverse=True)
    assert rss[0] == 56459172 * 1024


TICK_SACCT = [
    "615427|615427|PREEMPTED|0:0|0:0|batch|normal|trace07|1756745391|1756750000|1756760000|9609|1440|"
    "billing=64,cpu=64,gres/gpu=1,mem=512G,node=1|billing=64,cpu=64,gres/gpu=1,mem=512G,node=1|None|/w",
    "615427|615427|PENDING|0:0|0:0|batch|normal|None assigned|1756760001|Unknown|Unknown|0|1440||"
    "billing=64,cpu=64,gres/gpu=1,mem=512G,node=1|BeginTime|/w",
    "700001|700001_4|CANCELLED by 2692968|0:0|0:0|GPU-shared|gpu|v001|1756745391|1756750000|1756751000|1000|UNLIMITED|"
    "cpu=5,gres/gpu=1,gres/gpu:v100-32=1,node=1|cpu=5,gres/gpu=1,node=1|DependencyNeverSatisfied|/w",
    "700002|700002|TIMEOUT|0:0|0:0|GPU-shared|gpu|w008|1|2|3|28805|480|cpu=24|cpu=24|None|/ocean/my dir, with comma",
    "700003|700003|FAILED|137:0|0:9|RM|rm|r001|1|2|3|10|Partition_Limit|||None|/w",
    "700004|700004|COMPLETED|0:0|0:0|RM|rm|r001|1|2|30|10|60|||None|/w",
    "700004|700004|COMPLETED|0:0|0:0|RM|rm|r002|1|2|40|10|60|||None|/w",
]


def test_parse_sacct_tick_incarnations_and_fields():
    out = p.parse_sacct_tick(TICK_SACCT)
    j = out[615427]
    assert j["incarnations"] == 2 and j["current"]["state"] == "PENDING" and j["current"]["job_state"] is JobState.SUBMITTED
    assert j["rows"][0]["job_state"] is JobState.PREEMPTED and j["rows"][0]["end"] == 1756760000 and j["rows"][0]["elapsed_s"] == 9609
    assert j["current"]["nodelist"] is None and j["current"]["start"] is None and j["current"]["end"] is None
    assert j["current"]["alloc_tres"] == {} and j["current"]["reason"] == "BeginTime" and j["current"]["workdir"] == "/w"
    a = out[700001]["current"]
    assert a["state"] == "CANCELLED" and a["cancelled_by"] == "2692968" and a["job_id"] == "700001_4"
    assert a["array_job_id"] == 700001 and a["array_index"] == 4 and a["slurm_id"] == 700001
    assert a["timelimit_min"] is None and a["timelimit_s"] is None and a["alloc_tres"]["gres/gpu:v100-32"] == 1
    assert a["reason"] == "DependencyNeverSatisfied"
    t = out[700002]["current"]
    assert t["job_state"] is JobState.TIMEOUT and t["elapsed_s"] == 28805 and t["timelimit_min"] == 480
    assert t["workdir"] == "/ocean/my dir, with comma"
    f = out[700003]["current"]
    assert f["exit_code"] == (137, 0) and f["derived_exit_code"] == (0, 9) and f["timelimit_min"] is None
    assert f["alloc_tres"] == {} and f["req_tres"] == {}
    c = out[700004]
    assert c["incarnations"] == 2 and c["current"]["end"] == 40 and c["current"]["nodelist"] == "r002"
    assert p.parse_sacct_tick([]) == {}


def test_parse_files_cmds_recover():
    files = p.parse_files([
        "/ctrl/j1/a1|jobid|615411",
        '/ctrl/j1/a1|status.json|{"phase": "exited", "rc": 0, "cause": "timeout", "restart": 1}',
        "/ctrl/j1/a1|heartbeat|1756760000",
        '/ctrl/j1/a1|progress.json|{"epoch": 3, "loss": 0.5}',
        "/ctrl/j2/a1|status.json|not json {",
        "short",
    ])
    assert files["/ctrl/j1/a1"]["jobid"] == "615411" and files["/ctrl/j1/a1"]["heartbeat"] == "1756760000"
    assert files["/ctrl/j1/a1"]["status.json"] == {"phase": "exited", "rc": 0, "cause": "timeout", "restart": 1}
    assert files["/ctrl/j1/a1"]["progress.json"] == {"epoch": 3, "loss": 0.5}
    assert files["/ctrl/j2/a1"]["status.json"] == "not json {"
    assert p.parse_cmds(["/ctrl/a1/cmds/c1.rc|0", "/ctrl/a1/cmds/c2.rc|137", "/x|weird", "bad"]) == {
        "/ctrl/a1/cmds/c1.rc": 0, "/ctrl/a1/cmds/c2.rc": 137, "/x": "weird"}
    rec = p.parse_recover([
        "615440|1756760000|PENDING|/w|sbatch --parsable --comment=slurm-mcp:j17:1:tok /h/.slurm-mcp/jobs/j17/a1/job.sbatch",
        "615441|1756760001|CANCELLED by 1|/w|sbatch -p batch /w/x.sh",
    ])
    assert rec[0]["slurm_id"] == 615440 and rec[0]["submit"] == 1756760000 and rec[0]["job_state"] is JobState.SUBMITTED
    assert rec[0]["workdir"] == "/w" and rec[0]["submit_line"].endswith("/h/.slurm-mcp/jobs/j17/a1/job.sbatch")
    assert rec[1]["cancelled_by"] == "1"


def test_parse_backfill():
    rows = p.parse_backfill([
        "1|GPU-shared|gpu|billing=24,cpu=24,gres/gpu=2,gres/gpu:h100-80=2,mem=128G,node=1|1756740000|1756745000|COMPLETED",
        "2|batch|normal|billing=64,cpu=64,gres/gpu=1,mem=512G,node=1|1756740000|Unknown|PENDING",
        "3|RM-shared|low|cpu=16,mem=30400M,node=1|1756740000|None|CANCELLED by 5",
    ])
    assert rows[0]["gres_type"] == "h100-80" and rows[0]["gpus"] == 2 and rows[0]["start"] == 1756745000
    assert rows[1]["gres_type"] is None and rows[1]["gpus"] == 1 and rows[1]["start"] is None
    assert rows[2]["gres_type"] is None and rows[2]["gpus"] is None and rows[2]["start"] is None


def test_parse_enrich():
    out = p.parse_enrich([
        "615411|615411|COMPLETED|0:0||128G|920|billing=64,cpu=64,gres/gpu=1,mem=512G,node=1",
        "615411.batch|615411.batch|COMPLETED|0:0|56459172K||920|cpu=64,gres/gpu=1,mem=512G,node=1",
        "615411.extern|615411.extern|COMPLETED|0:0|1.5G||920|cpu=64",
        "615412|615412|FAILED|137:0||8048Mc|10|billing=8,cpu=8,node=1",
        "615412.0|615412.0|FAILED|137:0|7000000K||10|cpu=8",
        "615413|615413|COMPLETED|0:0||80Gn|10|cpu=8",
        "::L /w/out.txt|epoch 3 loss 0.5",
        "::L /w/other.txt|",
    ])
    j = out["jobs"][615411]
    assert j["max_rss_bytes"] == 56459172 * 1024 and j["req_mem_bytes"] == 128 * 1024 ** 3
    assert j["steps"]["extern"]["max_rss_bytes"] == int(1.5 * 1024 ** 3)
    assert j["alloc"]["state"] == "COMPLETED" and set(j["steps"]) == {"batch", "extern"}
    assert j["steps"]["batch"]["max_rss_bytes"] == 56459172 * 1024
    k = out["jobs"][615412]
    assert k["req_mem_bytes"] == 8048 * 1024 ** 2 * 8 and k["max_rss_bytes"] == 7000000 * 1024
    assert k["alloc"]["req_mem_per"] == "cpu" and k["steps"]["0"]["exit_code"] == (137, 0)
    assert out["jobs"][615413]["req_mem_bytes"] == 80 * 1024 ** 3 and out["jobs"][615413]["max_rss_bytes"] is None
    assert out["last_lines"] == {"/w/out.txt": "epoch 3 loss 0.5", "/w/other.txt": ""}


# ---------------------------------------------------------------------------------------------------
# 6.2 snapshot
# ---------------------------------------------------------------------------------------------------

SNAPSHOT_TEXT = """::NODES
batch|mix|gpu:a40:1|124/4/0/128
batch|alloc|gpu:a40:1|128/0/0/128
GPU-shared|idle~|gpu:v100-32:16(S:0-95)|0/96/0/96
RM|drain*|(null)|0/0/128/128
::RC 0
::PD
   3433 GPU-shared|gres:gpu:h100-80:1|N/A|
    283 GPU-shared|N/A|N/A|
      5 GPU|N/A|gres:gpu:8|
    242 batch|N/A|N/A|
::RC 0
::R
     17 GPU-shared|gres:gpu:h100-80:1|N/A|
   1491 RM-shared|N/A|N/A|
::RC 0
::MINE
615421|batch|N/A|gres:gpu:a40|251189|1756760283|Resources|
615427|batch|N/A|gres:gpu:a40|251187|N/A|Priority|
::RESV
ReservationName=maint StartTime=1757000000 EndTime=1757086400 Nodes=trace[01-29] PartitionName=batch Flags=MAINT
::END
"""


def test_parse_snapshot_sections():
    snap = p.parse_snapshot(p.parse_sections(SNAPSHOT_TEXT))
    nodes = snap["nodes"]
    assert nodes[0] == {"partition": "batch", "state": "mix", "state_raw": "mix", "gres": {"type": "a40", "count": 1},
                        "gres_raw": "gpu:a40:1", "cpus": {"alloc": 124, "idle": 4, "other": 0, "total": 128}}
    assert nodes[2]["state"] == "idle" and nodes[2]["gres"] == {"type": "v100-32", "count": 16}
    assert nodes[3]["gres"] is None and nodes[3]["cpus"]["other"] == 128
    assert snap["pd"][0] == {"count": 3433, "partition": "GPU-shared", "tres_per_node": "gres:gpu:h100-80:1", "tres_per_job": "N/A"}
    assert snap["pd"][2]["tres_per_job"] == "gres:gpu:8" and snap["pd"][3]["count"] == 242
    assert snap["r"][1]["count"] == 1491 and snap["r"][1]["partition"] == "RM-shared"
    m = snap["mine"]
    assert m[0] == {"slurm_id": 615421, "partition": "batch", "partitions": ["batch"], "tres_per_node": "N/A",
                    "tres_per_job": "gres:gpu:a40", "priority": 251189, "start": 1756760283, "reason": "Resources"}
    assert m[1]["start"] is None and m[1]["reason"] == "Priority"
    assert snap["resv"][0]["maintenance"] is True and snap["resv"][0]["partition"] == "batch"
    assert snap["rc"] == {"NODES": 0, "PD": 0, "R": 0}


def test_parse_uniq_rows_and_mine_fallback_forms():
    rows = p.parse_uniq_rows(["    242 batch|N/A", "     30 batch|gres:gpu:a40:1"])
    assert rows[0] == {"count": 242, "partition": "batch", "tres_per_node": "N/A", "tres_per_job": None}
    assert rows[1]["tres_per_node"] == "gres:gpu:a40:1"
    assert p.parse_uniq_rows(["garbage", ""]) == []
    fixture_rows = p.parse_uniq_rows(["    242 batch|PENDING|N/A"], ("partition", "state", "tres_per_node"))
    assert fixture_rows[0] == {"count": 242, "partition": "batch", "state": "PENDING", "tres_per_node": "N/A", "tres_per_job": None}
    # -O form: 3 fields with the trailing '|' left by ':0|' -> partition|tres-per-node|tres-per-job
    o_rows = p.parse_uniq_rows(["   3433 GPU-shared|gres:gpu:h100-80:1|N/A|", "    242 batch|N/A|gres:gpu:a40|"])
    assert o_rows[0] == {"count": 3433, "partition": "GPU-shared", "tres_per_node": "gres:gpu:h100-80:1", "tres_per_job": "N/A"}
    assert o_rows[1]["tres_per_job"] == "gres:gpu:a40" and "state" not in o_rows[1]
    mine = p.parse_mine(["615421|batch|N/A|251189|N/A|Resources"])
    assert mine[0]["tres_per_job"] is None and mine[0]["priority"] == 251189 and mine[0]["start"] is None
    assert p.parse_mine(["short|row"]) == []


def merged_caps(cluster: str) -> dict[str, dict]:
    """parse_partitions + aggregate_sinfo merged per section 6.1 (what classify_demand must receive)."""
    parts = p.parse_partitions(fxl(cluster, "scontrol_partitions.out"))
    agg = p.aggregate_sinfo(p.parse_sinfo_nodes(fxl(cluster, "sinfo_nodes.out")))
    return p.merge_partition_gres(parts, agg)


def test_merge_partition_gres_bridges2_adds_sinfo_only_h100():
    parts = p.parse_partitions(fxl("bridges2", "scontrol_partitions.out"))
    agg = p.aggregate_sinfo(p.parse_sinfo_nodes(fxl("bridges2", "sinfo_nodes.out")))
    merged = p.merge_partition_gres(parts, agg)
    gs = merged["GPU-shared"]
    # TRES lists 3 types (24+72+200 = 296 of 376 GPUs); the missing 80 are the 10 x 8 h100-80 nodes in sinfo.
    assert gs["gres_types"] == {"l40s-48": 24, "v100-16": 72, "v100-32": 200, "h100-80": 80}
    assert gs["gres_nodes"]["h100-80"] == {"nodes": 10, "per_node": 8} and gs["gres_nodes"]["v100-32"]["per_node"] == 16
    assert gs["gpu_total"] == 376 and gs["has_gpu"] is True
    assert sum(gs["gres_types"].values()) == gs["gpu_total"]
    # TRES-derived counts win over the sinfo nodes x max-per-node product where both exist (v100-32: 200 vs 24 x 16).
    assert gs["gres_types"]["v100-32"] == 200
    assert merged["GPU-small"]["gres_types"] == {"v100-32": 8} and merged["RM"]["gres_types"] == {} and merged["RM"]["has_gpu"] is False
    assert set(merged) == set(parts)
    # pure: inputs untouched
    assert parts["GPU-shared"]["gres_types"] == {"l40s-48": 24, "v100-16": 72, "v100-32": 200}
    assert "gres_nodes" not in parts["GPU-shared"]


def test_merge_partition_gres_trace_types_untyped_tres():
    parts = p.parse_partitions(fxl("trace", "scontrol_partitions.out"))
    agg = p.aggregate_sinfo(p.parse_sinfo_nodes(fxl("trace", "sinfo_nodes.out")))
    merged = p.merge_partition_gres(parts, agg)
    for name in ("batch", "biosimmlab"):
        assert merged[name]["gres_types"] == {"a40": 29}, name
        assert merged[name]["gres_nodes"] == {"a40": {"nodes": 29, "per_node": 1}}
        assert merged[name]["gpu_total"] == 29 and merged[name]["has_gpu"] is True
    assert merged["cpuonly"]["gres_types"] == {} and merged["cpuonly"]["gres_nodes"] == {}
    assert merged["cpuonly"]["has_gpu"] is False and merged["cpuonly"]["gpu_total"] is None
    assert parts["batch"]["gres_types"] == {}


def test_merge_partition_gres_synthetic_edge_cases():
    parts = {"gpu": {"name": "gpu", "gres_types": {}, "gpu_total": None, "has_gpu": False},
             "cpu": {"name": "cpu", "gres_types": {}, "gpu_total": None, "has_gpu": False}}
    agg = {"gpu": {"nodes": 3, "states": {}, "gres": {None: {"nodes": 2, "per_node": 8}, "a100": {"nodes": 1, "per_node": 4}},
                   "max_cpus": 0, "max_mem_mb": 0},
           "ghost": {"nodes": 1, "states": {}, "gres": {"x": {"nodes": 1, "per_node": 1}}, "max_cpus": 0, "max_mem_mb": 0}}
    merged = p.merge_partition_gres(parts, agg)
    assert set(merged) == {"gpu", "cpu"}  # sinfo-only partitions are ignored
    g = merged["gpu"]
    assert g["gres_types"] == {"a100": 4}  # untyped gpu:8 nodes never become a type
    assert g["gres_nodes"] == agg["gpu"]["gres"] and g["gpu_total"] == 20 and g["has_gpu"] is True
    assert merged["cpu"] == {"name": "cpu", "gres_types": {}, "gpu_total": None, "has_gpu": False, "gres_nodes": {}}
    assert p.merge_partition_gres({}, agg) == {}
    assert p.merge_partition_gres(parts, {})["gpu"]["has_gpu"] is False


def test_squeue_all_counts_fixture_and_classify_demand_trace():
    caps = merged_caps("trace")
    rows = p.parse_uniq_rows(fxl("trace", "squeue_all_counts.out"), ("partition", "state", "tres_per_node"))
    assert [(r["count"], r["partition"], r["state"]) for r in rows] == [
        (242, "batch", "PENDING"), (30, "batch", "RUNNING"), (23, "cpuonly", "PENDING"), (20, "cpuonly", "RUNNING")]
    d = p.classify_demand(rows[0], caps)
    assert d["kind"] == "gpu" and d["untyped"] is True and d["type"] is None and d["gpus"] is None and d["count"] == 242
    assert d["against"] == ["a40"]
    # bare parse_partitions caps (TRES gres/gpu=29 only): still untyped GPU demand, but the inventory is unknown
    bare = p.classify_demand(rows[0], p.parse_partitions(fxl("trace", "scontrol_partitions.out")))
    assert bare["kind"] == "gpu" and bare["untyped"] is True and bare["against"] == []
    c = p.classify_demand(rows[2], caps)
    assert c == {"partition": "cpuonly", "partitions": ["cpuonly"], "count": 23, "kind": "cpu", "type": None, "gpus": 0, "untyped": False}


def test_squeue_all_counts_fixture_and_classify_demand_bridges2():
    caps = merged_caps("bridges2")
    rows = p.parse_uniq_rows(fxl("bridges2", "squeue_all_counts.out"), ("partition", "state", "tres_per_node"))
    by = {(r["partition"], r["state"], r["tres_per_node"]): r for r in rows}
    assert by[("GPU-shared", "PENDING", "N/A")]["count"] == 283
    assert by[("GPU-shared", "PENDING", "gres:gpu:h100-80:1")]["count"] == 3433
    assert by[("RM-shared", "PENDING", "N/A")]["count"] == 2982
    assert by[("GPU-shared", "RESV_DEL_HOLD", "gres:gpu:v100-32:1")]["count"] == 7
    untyped = p.classify_demand(by[("GPU-shared", "PENDING", "N/A")], caps)
    assert untyped["kind"] == "gpu" and untyped["untyped"]
    assert untyped["against"] == ["h100-80", "l40s-48", "v100-16", "v100-32"]
    bare = p.classify_demand(by[("GPU-shared", "PENDING", "N/A")], p.parse_partitions(fxl("bridges2", "scontrol_partitions.out")))
    assert bare["against"] == ["l40s-48", "v100-16", "v100-32"]  # TRES alone misses the sinfo-only h100-80
    typed = p.classify_demand(by[("GPU-shared", "PENDING", "gres:gpu:h100-80:1")], caps)
    assert typed == {"partition": "GPU-shared", "partitions": ["GPU-shared"], "count": 3433, "kind": "gpu",
                     "type": "h100-80", "gpus": 1, "untyped": False}
    assert p.classify_demand(by[("RM-shared", "PENDING", "N/A")], caps)["kind"] == "cpu"
    one = p.classify_demand(by[("GPU-shared", "RUNNING", "gres:gpu:h100-80")], caps)
    assert one["type"] == "h100-80" and one["gpus"] == 1
    plain = p.classify_demand(by[("GPU-shared", "PENDING", "gres:gpu:4")], caps)
    assert plain["type"] is None and plain["gpus"] == 4 and plain["untyped"] is True


def test_classify_demand_tres_per_job_and_multi_partition():
    caps = {"GPU": {"gres_types": {"v100-32": 200}, "has_gpu": True}, "RM": {"gres_types": {}, "has_gpu": False},
            "batch": {"gres_types": {}, "has_gpu": True}}
    assert p.classify_demand({"partition": "GPU", "tres_per_node": "N/A", "tres_per_job": "gres:gpu:a40", "count": 2}, caps) == {
        "partition": "GPU", "partitions": ["GPU"], "count": 2, "kind": "gpu", "type": "a40", "gpus": 1, "untyped": False}
    assert p.classify_demand({"partition": "GPU", "tres_per_node": "N/A", "tres_per_job": "gres/gpu:h100-80=2"}, caps)["gpus"] == 2
    assert p.classify_demand({"partition": "GPU", "tres_per_node": "gres:gpu:2", "tres_per_job": "gres:gpu:a40"}, caps)["gpus"] == 2
    multi = p.classify_demand({"partition": "RM,GPU", "tres_per_node": "N/A", "tres_per_job": "N/A"}, caps)
    assert multi["kind"] == "gpu" and multi["untyped"] and multi["against"] == ["v100-32"] and multi["partitions"] == ["RM", "GPU"]
    assert p.classify_demand({"partition": "RM", "tres_per_node": "N/A", "tres_per_job": None}, caps)["kind"] == "cpu"
    assert p.classify_demand({"partition": "unknown", "tres_per_node": "N/A", "tres_per_job": "N/A"}, caps)["kind"] == "cpu"
    untyped_part = p.classify_demand({"partition": "batch", "tres_per_node": "N/A", "tres_per_job": "N/A"}, caps)
    assert untyped_part["kind"] == "gpu" and untyped_part["untyped"] is True and untyped_part["against"] == []
    assert "*" not in untyped_part["against"]
    assert p.classify_demand({"partition": "GPU", "tres_per_node": "N/A"}, {"GPU": ["v100-32"]})["against"] == ["v100-32"]
    both = p.classify_demand({"partition": "batch,GPU", "tres_per_node": "N/A", "tres_per_job": "N/A"}, caps)
    assert both["kind"] == "gpu" and both["against"] == ["v100-32"]
    # plain iterable caps with a None entry (aggregate_sinfo untyped key) is tolerated
    assert p.classify_demand({"partition": "GPU", "tres_per_node": "N/A"}, {"GPU": [None, "a40"]})["against"] == ["a40"]
    only_untyped = p.classify_demand({"partition": "GPU", "tres_per_node": "N/A"}, {"GPU": [None]})
    assert only_untyped["kind"] == "gpu" and only_untyped["against"] == []
    assert p.classify_demand({"partition": "GPU", "tres_per_node": "N/A"}, {"GPU": []})["kind"] == "cpu"


# ---------------------------------------------------------------------------------------------------
# 6.3 estimate / submit / control
# ---------------------------------------------------------------------------------------------------

def test_parse_test_only_fixtures_trace():
    ok = p.parse_test_only(fx("trace", "sbatch_test_only_ok.err"), rc=0)
    assert ok == {"ok": True, "job_id": 615442, "est_start": "2026-09-01T17:04:53", "processors": 1, "nodes": "trace01",
                  "partition": "biosimmlab"}
    assert fx("trace", "sbatch_test_only_ok.out") == ""
    bp = p.parse_test_only(fx("trace", "sbatch_test_only_bad_partition.err"), rc=1)
    assert bp["ok"] is False and bp["code"] == "E_PARTITION" and bp["reason"] == "Invalid partition name specified"
    assert bp["details"] == ["invalid partition specified: no_such_partition"]
    bg = p.parse_test_only(fx("trace", "sbatch_test_only_bad_gres.err"), rc=1)
    assert bg["ok"] is False and bg["code"] == "E_NODE_CONFIG" and bg["reason"] == "Requested node configuration is not available"
    assert bg["details"] == []
    bt = p.parse_test_only(fx("trace", "sbatch_test_only_bad_time.err"), rc=1)
    assert bt["ok"] is False and bt["code"] == "E_QOS_MAXWALL" and bt["details"] == ["QOSMaxWallDurationPerJobLimit"]
    assert bt["reason"].startswith("Job violates accounting/QOS policy")
    for name in ("sbatch_test_only_bad_partition.out", "sbatch_test_only_bad_gres.out", "sbatch_test_only_bad_time.out"):
        assert fx("trace", name) == ""


def test_parse_test_only_fixtures_bridges2():
    for name in ("sbatch_test_only_ok", "sbatch_test_only_bad_gres", "sbatch_test_only_bad_time"):
        r = p.parse_test_only(fx("bridges2", name + ".err"), rc=1)
        assert r["ok"] is False and r["code"] == "E_QOS" and r["reason"] == "Invalid qos specification" and r["details"] == []
        assert fx("bridges2", name + ".out") == ""
    bp = p.parse_test_only(fx("bridges2", "sbatch_test_only_bad_partition.err"), rc=1)
    assert bp["code"] == "E_PARTITION" and fx("bridges2", "sbatch_test_only_bad_partition.out") == ""


def test_parse_test_only_framed_and_epoch():
    text = "::T1\nsbatch: Job 99 to start at 1756760283 using 64 processors on nodes trace[07-08] in partition batch\n::RC 0\n::END\n"
    r = p.parse_test_only(text, rc=0)
    assert r["ok"] and r["est_start"] == 1756760283 and r["nodes"] == "trace[07-08]" and r["partition"] == "batch"
    assert p.parse_test_only("", rc=1)["ok"] is False and p.parse_test_only("", rc=1)["code"] == "E_SUBMIT_FAILED"
    assert p.parse_test_only("", rc=0) == {"ok": False, "reason": "no estimate line", "code": None, "details": []}
    plain = p.parse_test_only("sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified", rc=1)
    assert plain["code"] == "E_ACCOUNT"
    submit_limit = p.parse_test_only("sbatch: error: QOSMaxSubmitJobPerUserLimit\nallocation failure: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)", rc=1)
    assert submit_limit["code"] == "E_SUBMIT_LIMIT"


@pytest.mark.parametrize("stderr,code", [
    ("sbatch: error: Batch job submission failed: Invalid partition name specified", "E_PARTITION"),
    ("sbatch: error: invalid partition specified: no_such_partition", "E_PARTITION"),
    ("sbatch: error: Batch job submission failed: No partition specified or system default partition", "E_PARTITION_REQUIRED"),
    ("sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified", "E_ACCOUNT"),
    ("allocation failure: Invalid qos specification", "E_QOS"),
    ("sbatch: error: QOSMaxWallDurationPerJobLimit", "E_QOS_MAXWALL"),
    ("sbatch: error: Batch job submission failed: Requested time limit is invalid (missing or exceeds some limit)", "E_QOS_MAXWALL"),
    ("sbatch: error: PartitionTimeLimit", "E_QOS_MAXWALL"),
    ("sbatch: error: QOSMaxCpuPerJobLimit", "E_QOS_SIZE"),
    ("sbatch: error: QOSMaxGRESPerJob", "E_QOS_SIZE"),
    ("sbatch: error: QOSMaxNodePerJobLimit", "E_QOS_SIZE"),
    ("sbatch: error: GPU-shared maximum is 4 GPUs", "E_QOS_SIZE"),
    ("sbatch: error: use GPU partition for multiple nodes", "E_QOS_SIZE"),
    ("sbatch: error: PartitionNodeLimit", "E_QOS_SIZE"),
    ("sbatch: error: QOSMaxSubmitJobPerUserLimit", "E_SUBMIT_LIMIT"),
    ("sbatch: error: AssocMaxSubmitJobLimit", "E_SUBMIT_LIMIT"),
    ("sbatch: error: QOSMaxJobsPerUserLimit", "E_SUBMIT_LIMIT"),
    ("sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)", "E_QOS_POLICY"),
    ("sbatch: error: Batch job submission failed: Requested node configuration is not available", "E_NODE_CONFIG"),
    ("sbatch: error: Batch job submission failed: Invalid generic resource (gres) specification", "E_GRES"),
    ("sbatch: error: Batch job submission failed: Memory required by task is not available", "E_MEM"),
    ("sbatch: error: Batch job submission failed: Job dependency problem", "E_DEPENDENCY"),
    ("sbatch: error: Batch job submission failed: Access/permission denied", "E_PERMISSION"),
    ("sbatch: error: Unable to open file job.sbatch: Disk quota exceeded", "E_QUOTA"),
    ("bash: /x: No space left on device", "E_QUOTA"),
    ("sbatch: error: Batch job submission failed: Socket timed out on send/recv operation", "E_CTLD_BUSY"),
    ("sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)", "E_CTLD_BUSY"),
    ("sbatch: error: Batch job submission failed: Zero Bytes were transmitted or received", "E_CTLD_BUSY"),
    ("sbatch: error: This does not look like a batch script.  The first line must start with #!", "E_SCRIPT"),
    ("sbatch: error: QOSMaxWallDurationPerJobLimit\nallocation failure: Job violates accounting/QOS policy", "E_QOS_MAXWALL"),
    ("something unknown", None), ("", None),
])
def test_map_sbatch_error_table(stderr, code):
    assert p.map_sbatch_error(stderr) == code


def test_parse_submit_output():
    assert p.parse_submit_output("JOBID 615442\n") == {"status": "ok", "job_id": 615442, "cluster": None, "stderr": ""}
    assert p.parse_submit_output("JOBID 45005113;bridges2\nsbatch: warning: x") == {
        "status": "ok", "job_id": 45005113, "cluster": "bridges2", "stderr": "sbatch: warning: x"}
    err = p.parse_submit_output("ERR 1\nsbatch: error: Batch job submission failed: Invalid qos specification\n")
    assert err == {"status": "err", "rc": 1, "code": "E_QOS",
                   "stderr": "sbatch: error: Batch job submission failed: Invalid qos specification"}
    assert p.parse_submit_output("ERR 1\nsomething odd")["code"] == "E_SUBMIT_FAILED"
    assert p.parse_submit_output("ERR 2\n")["code"] == "E_HELPER"
    assert p.parse_submit_output("ERR 3\n")["status"] == "ambiguous"
    assert p.parse_submit_output("")["status"] == "ambiguous"
    assert p.parse_submit_output("Connection closed by remote host")["status"] == "ambiguous"
    assert p.parse_submit_output("\n\nJOBID 7")["job_id"] == 7
    assert p.parse_submit_output("Submitted batch job 12345")["job_id"] == 12345
    assert p.parse_submit_output("12345;bridges2")["cluster"] == "bridges2"


def test_parse_scontrol_job_fixture_running():
    j = p.parse_scontrol_job(fx("trace", "scontrol_show_job_615411.out"))
    assert j["job_id"] == 615411 and j["job_name"] == "wobl_notaylor" and j["job_state"] == "RUNNING"
    assert j["state"] is JobState.RUNNING and j["reason"] is None and j["dependency"] is None
    assert j["requeue"] == 1 and j["restarts"] == 0 and j["exit_code"] == (0, 0) and j["priority"] == 251174
    assert j["submit_time"] == "2026-09-01T16:38:33" and j["start_time"] == "2026-09-01T16:49:50"
    assert j["end_time"] == "2026-09-02T16:49:50" and j["node_list"] == "trace13" and j["batch_host"] == "trace13"
    assert j["std_out"] == "/trace/group/biosimmlab/wxu2/vascular_super_resolution/logs/sweep/wobl_notaylor_615411.out"
    assert j["std_err"].endswith("/logs/sweep/wobl_notaylor_615411.err")
    assert j["work_dir"] == "/trace/group/biosimmlab/wxu2/vascular_super_resolution"
    assert j["command"] == "/trace/group/biosimmlab/wxu2/vascular_super_resolution/jobs/train_wobl_notaylor.job"
    assert j["comment"] is None and j["tres_per_job"] == {"type": "a40", "count": 1} and j["tres_per_node"] is None
    assert j["partition"] == "batch" and j["account"] == "biosimmlab" and j["qos"] == "normal"
    assert j["num_nodes"] == 1 and j["num_cpus"] == 64 and j["time_limit_s"] == 86400
    assert j["tres"] == {"cpu": 64, "mem": "512G", "node": 1, "billing": 64, "gres/gpu": 1}
    assert j["raw"]["AllocNode:Sid"] == "0.0.0.0:3131994" and j["raw"]["Power"] == "" and j["raw"]["UserId"] == "wxu2(2692968)"
    assert j["array_job_id"] is None and j["sched_node_list"] is None


def test_parse_scontrol_job_fixture_pending():
    j = p.parse_scontrol_job(fx("trace", "scontrol_show_job_615427.out"))
    assert j["job_state"] == "PENDING" and j["state"] is JobState.SUBMITTED and j["reason"] == "Priority"
    assert j["node_list"] is None and j["sched_node_list"] == "trace07" and j["start_time"] == "2026-09-01T21:54:30"
    assert j["raw"]["Scheduler"] == "Backfill:*" and j["restarts"] == 0
    assert j["std_out"].endswith("/logs/sweep/wobl_615427.out")


def test_parse_scontrol_job_missing_fixtures():
    for cluster, name in (("trace", "scontrol_show_job_missing.err"), ("bridges2", "scontrol_show_job_missing.err"),
                          ("bridges2", "scontrol_show_job_44809480.err")):
        text = fx(cluster, name)
        assert text.strip() == "slurm_load_jobs error: Invalid job id specified"
        assert p.scontrol_job_missing(text) is True and p.parse_scontrol_job(text) == {}
    for cluster, name in (("trace", "scontrol_show_job_missing.out"), ("bridges2", "scontrol_show_job_missing.out"),
                          ("bridges2", "scontrol_show_job_44809480.out")):
        assert fx(cluster, name) == "" and p.parse_scontrol_job(fx(cluster, name)) == {}
    assert p.scontrol_job_missing("") is False and p.parse_scontrol_job("nonsense") == {}


def test_parse_scontrol_job_values_with_spaces():
    text = ("JobId=1 JobName=x JobState=PENDING Reason=ReqNodeNotAvail, UnavailableNodes:trace[01-03] Dependency=afterok:5 "
            "Requeue=0 Restarts=2 StartTime=Unknown EndTime=Unknown NodeList=(null) BatchHost=(null) Command=/w/my script.sh --flag "
            "WorkDir=/w/dir with spaces StdOut=/w/out %j.txt StdErr=/w/err.txt Comment=slurm-mcp:j1:1:tok extra words "
            "TresPerNode=gres:gpu:h100-80:2 ArrayJobId=1 ArrayTaskId=4 TimeLimit=UNLIMITED")
    j = p.parse_scontrol_job(text)
    assert j["reason"] == "ReqNodeNotAvail, UnavailableNodes:trace[01-03]" and j["dependency"] == "afterok:5"
    assert j["requeue"] == 0 and j["restarts"] == 2 and j["start_time"] is None and j["node_list"] is None
    assert j["command"] == "/w/my script.sh --flag" and j["work_dir"] == "/w/dir with spaces"
    assert j["std_out"] == "/w/out %j.txt" and j["comment"] == "slurm-mcp:j1:1:tok extra words"
    assert j["tres_per_node"] == {"type": "h100-80", "count": 2} and j["array_task_id"] == "4" and j["time_limit_s"] is None


def test_scancel_fixture_and_errors():
    for cl in CLUSTERS:
        assert fx(cl, "scancel_bad_job.out") == ""
        assert p.scancel_errors(fx(cl, "scancel_bad_job.out")) == []
    errs = p.scancel_errors("scancel: error: Kill job error on job id 12345: Invalid job id specified\nother\n"
                            "scancel: error: Kill job error on job id 12_4: Job/step already completing or completed")
    assert errs == [{"job_id": "12345", "message": "Invalid job id specified"},
                    {"job_id": "12_4", "message": "Job/step already completing or completed"}]


# ---------------------------------------------------------------------------------------------------
# capture-only helpers and coverage guard
# ---------------------------------------------------------------------------------------------------

def test_sprio_fixtures_pipe_table():
    t = p.parse_pipe_table(fxl("trace", "sprio_me.out"))
    assert len(t) == 5 and t[0]["jobid"] == "615421" and t[0]["priority"] == "251189" and t[0]["fairshare"] == "251142"
    assert list(t[0])[:3] == ["jobid", "partition", "priority"]
    assert p.parse_pipe_table(fxl("bridges2", "sprio_me.out")) == []
    assert p.parse_pipe_table(["a|b|c"], ["x", "y"]) == [{"x": "a", "y": "b|c"}]
    assert p.parse_pipe_table([]) == []


def test_all_exports_exist():
    for name in p.__all__:
        assert hasattr(p, name), name


def test_every_indexed_fixture_is_parsed_by_some_test():
    # Runs last (file order): every key of index.json must have been loaded through fx() above.
    for cluster in CLUSTERS:
        index = json.loads((FIXTURES / cluster / "index.json").read_text(encoding="utf-8"))
        missing = sorted(set(index) - COVERED[cluster])
        assert not missing, f"{cluster}: fixtures not covered by a parse test: {missing}"


def test_parse_uniq_rows_autodetects_ptb_fixture_form_without_trailing_bar():
    """A 3-field row without the trailing '|' is %P|%T|%b (squeue_all_counts.out), not the -O form."""
    assert p.UNIQ_PTB_FIELDS == ("partition", "state", "tres_per_node")
    rows = p.parse_uniq_rows(["    242 batch|PENDING|N/A", "     30 batch|RUNNING|N/A"])
    assert rows[0] == {"count": 242, "partition": "batch", "state": "PENDING", "tres_per_node": "N/A", "tres_per_job": None}
    assert rows[1]["state"] == "RUNNING"
    # golden input of section 8/10: trace batch pending untyped demand is 242, not 242+30
    auto = p.parse_uniq_rows(fxl("trace", "squeue_all_counts.out"))
    explicit = p.parse_uniq_rows(fxl("trace", "squeue_all_counts.out"), p.UNIQ_PTB_FIELDS)
    assert auto == explicit
    pending_batch = sum(r["count"] for r in auto if r["partition"] == "batch" and r["state"] == "PENDING")
    assert pending_batch == 242
    b2 = p.parse_uniq_rows(fxl("bridges2", "squeue_all_counts.out"))
    by = {(r["partition"], r["state"], r["tres_per_node"]): r["count"] for r in b2}
    assert by[("GPU-shared", "PENDING", "N/A")] == 283 and by[("GPU-shared", "PENDING", "gres:gpu:h100-80:1")] == 3433
    # explicit fields always win over auto-detection
    forced = p.parse_uniq_rows(["    242 batch|PENDING|N/A"], p.DEMAND_FIELDS)
    assert forced[0]["tres_per_node"] == "PENDING" and "state" not in forced[0]


# --- per-directory quotas: rows sharing a mount must not collapse (measured on TRACE 2026-09-02) ---

def test_parse_df_keeps_per_directory_quota_rows():
    """VAST/NFS reports different totals for different paths of one mount; collapsing them hid the real quota."""
    lines = [
        "172.19.21.14:/trace 1031755399168 1720401920 1030034997248 1% /trace /trace/home/wxu2",
        "172.19.21.14:/trace 3173828608 2334050304 839778304 74% /trace /trace/group/biosimmlab",
        "172.19.21.14:/trace 3173828608 2334050304 839778304 74% /trace /trace/group/biosimmlab/wxu2",
    ]
    rows = p.parse_df(lines, {"/trace/home/wxu2": "home", "/trace/group/biosimmlab": "group"})
    assert len(rows) == 2, "the 932 TB home view and the 3 TB group quota are different rows"
    home = next(r for r in rows if r["used_pct"] == 1)
    group = next(r for r in rows if r["used_pct"] == 74)
    assert home["role"] == "home" and group["role"] == "group"
    # the two group paths share one row because their totals are identical
    assert group["paths"] == ["/trace/group/biosimmlab", "/trace/group/biosimmlab/wxu2"]


def test_parse_df_identical_rows_still_collapse():
    lines = [
        "10.8.8.18@o2ib20:/ocean 424673280 401520640 23152640 95% /ocean /ocean/projects/x/u",
        "10.8.8.18@o2ib20:/ocean 424673280 401520640 23152640 95% /ocean /ocean/projects/x/u/sub",
    ]
    rows = p.parse_df(lines)
    assert len(rows) == 1 and len(rows[0]["paths"]) == 2


def test_df_row_for_path_picks_the_longest_matching_prefix():
    lines = [
        "fs 1031755399168 1720401920 1030034997248 1% /trace /trace/home/wxu2",
        "fs 3173828608 2334050304 839778304 74% /trace /trace/group/biosimmlab",
    ]
    rows = p.parse_df(lines)
    assert p.df_row_for_path(rows, "/trace/group/biosimmlab/wxu2/proj")["used_pct"] == 74
    assert p.df_row_for_path(rows, "/trace/home/wxu2/x")["used_pct"] == 1
    assert p.df_row_for_path(rows, "/somewhere/else") is None
