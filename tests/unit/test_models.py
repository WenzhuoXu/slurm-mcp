"""Unit tests for slurm_mcp.models (design sections 3.2 and 4)."""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from slurm_mcp import models as m
from slurm_mcp.errors import SlurmMcpError
from slurm_mcp.slurm.states import CmdState, JobState, TransferState


def spec(**kw) -> m.JobSpec:
    base = dict(name="train", command="python train.py", resources={"time": "01:00:00"})
    base.update(kw)
    return m.JobSpec(**base)


# --- schemas --------------------------------------------------------------------------------------

ALL_MODELS = [
    cls for cls in vars(m).values()
    if isinstance(cls, type) and issubclass(cls, BaseModel) and cls.__module__ == m.__name__
]


def test_all_models_enumerated():
    names = {c.__name__ for c in ALL_MODELS}
    for required in ["Resources", "InputSpec", "JobSpec", "Target", "PlacementPolicy", "RebalancePolicy",
                     "NotifyPolicy", "Result", "ClustersResult", "ClusterStatusResult", "RunCommandResult",
                     "TransferResult", "ListingResult", "ReadResult", "WriteResult", "PlanResult", "PlanOption",
                     "SubmitResult", "JobListResult", "JobRow", "JobStatusResult", "JobDetail", "LogResult",
                     "LogStream", "ControlResult", "ControlOutcome", "RebalanceResult", "RebalanceProposal",
                     "AllocRunResult", "EventsResult", "EventRow", "CollectResult", "CollectRow", "ConfigResult"]:
        assert required in names, required


@pytest.mark.parametrize("cls", ALL_MODELS, ids=lambda c: c.__name__)
def test_every_model_has_json_schema(cls):
    schema = cls.model_json_schema()
    assert schema.get("type") == "object" or "$defs" in schema or "properties" in schema
    json.dumps(schema)


@pytest.mark.parametrize("cls", m.RESULT_MODELS, ids=lambda c: c.__name__)
def test_result_models_summary_first_and_base_fields(cls):
    assert issubclass(cls, m.Result)
    fields = list(cls.model_fields)
    assert fields[0] == "summary"
    assert "unread_events" in fields and "next" in fields
    schema = cls.model_json_schema()
    assert list(schema["properties"])[0] == "summary"
    assert schema["required"][0] == "summary"


def test_result_base_defaults():
    r = m.Result(summary="ok")
    assert r.unread_events == 0 and r.next is None
    assert list(m.Result.model_fields) == ["summary", "unread_events", "next"]


@pytest.mark.parametrize("cls,kw", [
    (m.ClustersResult, {}), (m.ClusterStatusResult, {"cluster": "c"}), (m.RunCommandResult, {}),
    (m.TransferResult, {}), (m.ListingResult, {"path": "/x"}), (m.ReadResult, {"path": "/x"}),
    (m.WriteResult, {"path": "/x"}), (m.PlanResult, {"plan_id": "p1"}), (m.SubmitResult, {"handle": "j1"}),
    (m.JobListResult, {}), (m.JobStatusResult, {}), (m.LogResult, {"id": "j1"}),
    (m.ControlResult, {"action": "cancel"}), (m.RebalanceResult, {}), (m.AllocRunResult, {"cmd_id": "a1.c1"}),
    (m.EventsResult, {}), (m.CollectResult, {}), (m.ConfigResult, {}),
])
def test_result_models_construct_and_dump_json(cls, kw):
    r = cls(summary="s", **kw)
    data = json.loads(r.model_dump_json())
    assert data["summary"] == "s"
    assert cls.model_validate(data) == r


def test_nested_result_models_roundtrip():
    r = m.JobStatusResult(summary="s", jobs=[m.JobDetail(handle="j17", state="RUNNING", exit={"rc": 1},
                                                          paths={"workdir": "/w"}, alloc={"ready": True},
                                                          transfer={"state": "running"}, cmd={"state": "done"})])
    d = r.model_dump(mode="json")
    assert d["jobs"][0]["state"] == "RUNNING" and d["jobs"][0]["alloc"]["ready"] is True
    assert d["jobs"][0]["transfer"]["state"] == "running" and d["jobs"][0]["cmd"]["state"] == "done"
    assert m.JobStatusResult.model_validate(d).jobs[0].state is JobState.RUNNING


def test_enum_fields_serialise_as_strings():
    r = m.SubmitResult(summary="s", handle="j1", state=JobState.SUBMITTED)
    assert r.model_dump(mode="json")["state"] == "SUBMITTED"
    assert m.TransferResult(summary="s", state="done").state is TransferState.done
    assert m.AllocRunResult(summary="s", cmd_id="a1.c1", state="killed").state is CmdState.killed


def test_events_result_fields():
    r = m.EventsResult(summary="s", events=[m.EventRow(seq=3, kind="completed", handle="j1", payload={"rc": 0})],
                       delivered_seqs=[3], next_seq=4, acked=2, unread_unmatched=1, timed_out=False,
                       snapshot={"running": 2})
    d = r.model_dump()
    assert d["snapshot"]["running"] == 2 and d["events"][0]["payload"] == {"rc": 0}


def test_partition_info_charge_union():
    p = m.PartitionInfo(name="GPU-shared", charge={"unit": "gpu", "su_per_unit_h": 2.0})
    assert isinstance(p.charge, m.Charge)
    assert m.PartitionInfo(name="batch").charge == "free"
    assert m.PlanOption(target="t", charge="free").charge == "free"


# --- Resources -----------------------------------------------------------------------------------

@pytest.mark.parametrize("time,seconds", [("30", 1800), ("05:00", 300), ("01:00:00", 3600), ("1-00:00:00", 86400)])
def test_resources_time_ok(time, seconds):
    assert m.Resources(time=time).time_s == seconds


@pytest.mark.parametrize("bad", ["UNLIMITED", "abc", "", "1:2:3:4"])
def test_resources_time_bad(bad):
    with pytest.raises(SlurmMcpError) as ei:
        m.Resources(time=bad)
    assert ei.value.code == "E_INVALID_SPEC"


def test_resources_defaults_and_checks():
    r = m.Resources(time="10")
    assert (r.gpus, r.gpu_types, r.cpus, r.tasks, r.mem, r.nodes, r.exclusive, r.constraint) == \
        (0, None, None, None, None, 1, False, None)
    with pytest.raises(SlurmMcpError):
        m.Resources(time="10", nodes=0)
    with pytest.raises(SlurmMcpError):
        m.Resources(time="10", gpu_types=["a40"])   # gpu_types without gpus
    with pytest.raises(SlurmMcpError):
        m.Resources(time="10", cpus=0)
    with pytest.raises(ValidationError):
        m.Resources(time="10", bogus=1)


# --- JobSpec -------------------------------------------------------------------------------------

def test_jobspec_minimal_defaults():
    s = spec()
    assert s.wrap is True and s.requeue is None and s.on_timeout == "fail" and s.grace_s == 120
    assert s.child_signal is None and s.max_restarts == 3 and s.checkpoint_interval_h is None
    assert s.inputs == [] and s.outputs == [] and s.modules == [] and s.env == {} and s.depends_on == []
    assert s.extra_sbatch == [] and s.tags == {} and s.warnings == [] and s.source_kind == "command"


@pytest.mark.parametrize("kw", [
    {}, {"command": None, "script": None, "script_path": None},
    {"script": "#!/bin/bash\n"}, {"script_path": "/remote/x.sh"},
    {"script": "#!/bin/bash\n", "script_path": "/remote/x.sh"},
])
def test_jobspec_exactly_one_source(kw):
    base = dict(name="n", command="x", resources={"time": "10"})
    base.update(kw)
    if kw == {}:
        m.JobSpec(**base)
        return
    if "command" not in kw:
        base["command"] = "x"   # two sources
    with pytest.raises(SlurmMcpError) as ei:
        m.JobSpec(**base)
    assert ei.value.code == "E_INVALID_SPEC" and "exactly one of command/script/script_path" in ei.value.message


def test_jobspec_script_and_script_path_sources():
    s = m.JobSpec(name="n", script="#!/bin/bash\r\necho\r\n", resources={"time": "10"})
    assert s.source_kind == "script" and s.script == "#!/bin/bash\necho\n" and s.warnings == ["crlf_normalized"]
    s2 = m.JobSpec(name="n", script_path="local:C:/x/y.sh", resources={"time": "10"})
    assert s2.source_kind == "script_path"
    with pytest.raises(SlurmMcpError):
        m.JobSpec(name="n", script_path="relative/x.sh", resources={"time": "10"})
    with pytest.raises(SlurmMcpError) as ei:
        m.JobSpec(name="n", script="echo no shebang", resources={"time": "10"})
    assert ei.value.code == "E_SCRIPT"


@pytest.mark.parametrize("name", ["a", "train_1", "a.b-c", "x" * 64, "A1_.-"])
def test_jobspec_name_ok(name):
    assert spec(name=name).name == name


@pytest.mark.parametrize("name", ["", "x" * 65, "a b", "a/b", "a:b", "sp\u00e9c"])
def test_jobspec_name_bad(name):
    with pytest.raises(SlurmMcpError) as ei:
        spec(name=name)
    assert "name" in ei.value.message


@pytest.mark.parametrize("array", ["0-99", "1", "0-9:2", "1,3,5-9", "0-99:3,200"])
def test_jobspec_array_ok(array):
    assert spec(array=array).array == array


@pytest.mark.parametrize("array", ["", "a-b", "0-", "-5", "0-99%4", "1,,2", "1;2"])
def test_jobspec_array_bad(array):
    with pytest.raises(SlurmMcpError) as ei:
        spec(array=array)
    assert "array" in ei.value.message


@pytest.mark.parametrize("dep", ["j12", "afterok:j12", "afterany:j1", "afternotok:j999", "after:j3",
                                 "aftercorr:j4", "singleton"])
def test_jobspec_depends_ok(dep):
    assert spec(depends_on=[dep]).depends_on == [dep]


@pytest.mark.parametrize("dep", ["12345", "afterok:12345", "afterok:", "j", "afterok:a3", "afterburst:j1", ""])
def test_jobspec_depends_bad(dep):
    with pytest.raises(SlurmMcpError) as ei:
        spec(depends_on=[dep])
    assert "depends_on" in ei.value.message


@pytest.mark.parametrize("sig,expected", [("USR1", "USR1"), ("term", "TERM"), ("SIGUSR2", "USR2"), ("HUP", "HUP")])
def test_jobspec_child_signal_ok(sig, expected):
    assert spec(child_signal=sig).child_signal == expected


@pytest.mark.parametrize("sig", ["KILL", "STOP", "SIGKILL", "FOO", "", "9"])
def test_jobspec_child_signal_bad(sig):
    with pytest.raises(SlurmMcpError) as ei:
        spec(child_signal=sig)
    assert "child_signal" in ei.value.message


def test_jobspec_on_timeout_requeue_requires_signal_and_checkpoint():
    with pytest.raises(SlurmMcpError) as ei:
        spec(on_timeout="requeue")
    assert str(ei.value) == (
        "E_INVALID_SPEC: on_timeout=requeue would rerun the job from scratch up to max_restarts times "
        "\u2014 fix: declare child_signal (the signal your program checkpoints on) and checkpoint_interval_h, "
        'or use on_timeout="fail" with a longer time')
    with pytest.raises(SlurmMcpError):
        spec(on_timeout="requeue", child_signal="USR1")
    with pytest.raises(SlurmMcpError):
        spec(on_timeout="requeue", checkpoint_interval_h=1.0)
    ok = spec(on_timeout="requeue", child_signal="USR1", checkpoint_interval_h=0.5)
    assert ok.on_timeout == "requeue"


def test_jobspec_on_timeout_literal():
    with pytest.raises(ValidationError):
        spec(on_timeout="explode")


def test_jobspec_command_normalised_and_warning():
    s = spec(command="\ufeffecho a\r\necho b\r\n")
    assert s.command == "echo a\necho b\n"
    assert s.warnings == ["crlf_normalized"]
    with pytest.raises(SlurmMcpError) as ei:
        spec(command="echo\x00")
    assert ei.value.code == "E_INVALID_SPEC"


def test_jobspec_warnings_property_is_copy():
    s = spec(command="a\r\n")
    w = s.warnings
    w.append("x")
    assert s.warnings == ["crlf_normalized"]


@pytest.mark.parametrize("kw", [{"grace_s": -1}, {"max_restarts": -1}, {"checkpoint_interval_h": 0},
                                {"array_parallel": 0}, {"env": {"1bad": "x"}}, {"env": {"a-b": "x"}}])
def test_jobspec_misc_bad(kw):
    with pytest.raises(SlurmMcpError):
        spec(**kw)


def test_jobspec_full_example_roundtrip():
    s = spec(cluster="trace", partition="batch", qos="normal", account="acct", workdir="/w",
             resources={"time": "2-00:00:00", "gpus": 1, "gpu_types": ["a40"], "cpus": 8, "mem": "64G"},
             inputs=[{"local": "C:/proj", "remote": "/w/proj", "ignore": ["*.log"]}], outputs=["out/*.pt"],
             modules=["cuda/12"], setup="source env.sh", env={"A": "1"}, array="0-3", array_parallel=2,
             depends_on=["afterok:j1"], stdout="logs/%x-%j.out", extra_sbatch=["--time-min=01:00:00"],
             tags={"exp": "1"})
    d = s.model_dump()
    assert m.JobSpec.model_validate(d) == s
    assert s.resources.gpu_types == ["a40"] and s.inputs[0].ignore == ["*.log"]


def test_jobspec_extra_forbidden_and_parse_helper():
    with pytest.raises(ValidationError):
        spec(bogus=1)
    with pytest.raises(SlurmMcpError) as ei:
        m.JobSpec.parse({"name": "n", "command": "x", "resources": {"time": "10"}, "bogus": 1})
    assert ei.value.code == "E_INVALID_SPEC" and "bogus" in ei.value.message
    with pytest.raises(SlurmMcpError) as ei:
        m.JobSpec.parse({"name": "n", "command": "x"})
    assert "resources" in ei.value.message
    s = spec()
    assert m.JobSpec.parse(s) is s
    assert m.parse_input(m.Target, {"cluster": "c", "partitions": ["p"]}).key == "c:p"


def test_signal_number():
    assert m.signal_number("USR1") == 10 and m.signal_number("TERM") == 15 and m.signal_number("SIGTERM") == 15
    assert m.signal_number("nope") is None


# --- Target --------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,cluster,parts,gres,qos", [
    ("trace:batch", "trace", ["batch"], None, None),
    ("trace:batch:a40", "trace", ["batch"], "a40", None),
    ("bridges2:GPU-small,GPU-shared:v100-32@gpu", "bridges2", ["GPU-small", "GPU-shared"], "v100-32", "gpu"),
    ("bridges2:RM-shared@rm", "bridges2", ["RM-shared"], None, "rm"),
    ("trace:biosimmlab,batch", "trace", ["biosimmlab", "batch"], None, None),
    (" trace : batch ", "trace", ["batch"], None, None),
])
def test_target_parse_and_key(text, cluster, parts, gres, qos):
    t = m.Target.parse(text)
    assert (t.cluster, t.partitions, t.gres_type, t.qos) == (cluster, parts, gres, qos)
    canonical = f"{cluster}:{','.join(parts)}" + (f":{gres}" if gres else "") + (f"@{qos}" if qos else "")
    assert t.key == canonical and str(t) == canonical
    assert m.Target.parse(t.key) == t


@pytest.mark.parametrize("bad", ["", "trace", "trace:", ":batch", "trace:batch:", "trace:batch@", "a:b:c:d",
                                 "trace:,"])
def test_target_parse_bad(bad):
    with pytest.raises(SlurmMcpError) as ei:
        m.Target.parse(bad)
    assert ei.value.code == "E_INVALID_SPEC"


def test_target_account_and_model_checks():
    t = m.Target.parse("trace:batch", account="acc")
    assert t.account == "acc" and "acc" not in t.key
    with pytest.raises(SlurmMcpError):
        m.Target(cluster="a:b", partitions=["p"])
    with pytest.raises(SlurmMcpError):
        m.Target(cluster="a", partitions=[])
    with pytest.raises(SlurmMcpError):
        m.Target(cluster="a", partitions=["p,q"])


# --- policies ------------------------------------------------------------------------------------

def test_placement_policy_defaults():
    p = m.PlacementPolicy()
    assert p.objective == "balanced" and p.su_to_hours is None and p.su_reserve == 50.0
    assert p.max_pending_per_target is None and p.max_running_per_target == {} and p.allow_self_preempt is False
    assert p.soft_caps == {} and p.etiquette_h == 2.0 and p.targets_allow == [] and p.targets_deny == []
    assert p.prefer_cluster is None and p.unknown_wait_h == 12.0
    assert p.rebalance == m.RebalancePolicy()
    assert p.effective_su_to_hours == 0.25
    assert m.PlacementPolicy(objective="fastest").effective_su_to_hours == 0.02
    assert m.PlacementPolicy(objective="cheapest").effective_su_to_hours == 2.0
    assert m.PlacementPolicy(su_to_hours=1.5).effective_su_to_hours == 1.5


def test_placement_policy_patch_merge_style():
    p = m.PlacementPolicy.model_validate({"objective": "cheapest", "rebalance": {"enabled": False}})
    assert p.rebalance.enabled is False and p.rebalance.interval_min == 10
    with pytest.raises(ValidationError):
        m.PlacementPolicy(objective="random")
    with pytest.raises(ValidationError):
        m.PlacementPolicy(nope=1)


def test_rebalance_policy_defaults():
    r = m.RebalancePolicy()
    assert (r.enabled, r.interval_min, r.min_gain_h, r.max_moves_per_job, r.max_extra_su, r.min_age_min,
            r.max_moves_per_hour, r.hysteresis_h) == (True, 10, 1.0, 3, 0.0, 5, 6, 0.5)


def test_notify_policy_defaults_and_quiet_hours():
    n = m.NotifyPolicy()
    assert n.toast is True and n.webhook_url is None and n.webhook_kinds == [] and n.email is None
    assert n.quiet_hours is None
    assert n.toast_kinds == ["completed", "failed", "timeout", "oom", "cancelled", "preempted", "node_fail", "lost",
                             "needs_attention", "alloc_ready", "alloc_expiring", "transfer_failed",
                             "cluster_unreachable"]
    assert n.effective_webhook_kinds == n.toast_kinds
    assert m.NotifyPolicy(webhook_kinds=["failed"]).effective_webhook_kinds == ["failed"]
    q = m.NotifyPolicy(quiet_hours=(22, 7))
    assert q.quiet_hours == (22, 7)
    assert m.NotifyPolicy.model_validate({"quiet_hours": [22, 7]}).quiet_hours == (22, 7)
    with pytest.raises(SlurmMcpError):
        m.NotifyPolicy(quiet_hours=(25, 1))
    with pytest.raises(ValidationError):
        m.NotifyPolicy(quiet_hours=(1, 2, 3))


def test_default_toast_kinds_not_shared_between_instances():
    a = m.NotifyPolicy()
    a.toast_kinds.append("x")
    assert "x" not in m.NotifyPolicy().toast_kinds


def test_config_result_holds_policies():
    r = m.ConfigResult(summary="s")
    assert isinstance(r.placement, m.PlacementPolicy) and isinstance(r.notify, m.NotifyPolicy)
    d = r.model_dump(mode="json")
    assert d["placement"]["rebalance"]["enabled"] is True
