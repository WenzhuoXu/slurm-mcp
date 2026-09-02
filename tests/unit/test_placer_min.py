"""Unit tests for the phase-2 placer (design section 8: candidates, hard feasibility, provisional ranking)."""
from __future__ import annotations

import pytest

from unit.fake_transport import framed_discovery, profile_for
from slurm_mcp import placer
from slurm_mcp.errors import SlurmMcpError
from slurm_mcp.models import JobSpec, PlacementPolicy, Target
from slurm_mcp.slurm.client import parse_discovery, summarize_snapshot
from slurm_mcp.slurm.discovery import enrich_caps
from slurm_mcp.slurm.parse import parse_sections


def caps_for(cluster, **kw):
    profile = profile_for(cluster, **kw)
    caps = enrich_caps(parse_discovery(parse_sections(framed_discovery(cluster)), profile, cluster=cluster), profile)
    caps["group"] = {"trace": "biosimmlab", "bridges2": "mch250030p"}[cluster]
    if cluster == "trace":
        caps["assoc"]["default_qos"] = "normal"      # as the live cluster reports (the capture lacks DefaultQOS)
    for part in caps["partitions"].values():
        part["accessible"] = placer.partition_accessible(part, caps)
    return caps, profile


@pytest.fixture(scope="module")
def world():
    tc, tp = caps_for("trace")
    bc, bp = caps_for("bridges2")
    return {"trace": tc, "bridges2": bc}, {"trace": tp, "bridges2": bp}


def spec(**kw) -> JobSpec:
    base = {"name": "t", "command": "echo hi", "resources": {"time": "04:00:00", "gpus": 1}}
    base.update(kw)
    return JobSpec.parse(base)


def snapshot(caps, pd_rows, nodes):
    snap = {"nodes": nodes, "pd": pd_rows, "r": [], "mine": [], "resv": [], "ts": 0}
    snap["partitions"] = summarize_snapshot(snap, caps)
    return snap


TRACE_PD = [{"count": 242, "partition": "batch", "tres_per_node": "N/A", "tres_per_job": "N/A"},
            {"count": 3, "partition": "batch", "tres_per_node": "gres:gpu:a40:1", "tres_per_job": "N/A"},
            {"count": 23, "partition": "cpuonly", "tres_per_node": "N/A", "tres_per_job": "N/A"}]
TRACE_NODES = [{"partition": "batch", "state": "idle", "gres": {"type": "a40", "count": 1}},
               {"partition": "biosimmlab", "state": "alloc", "gres": {"type": "a40", "count": 1}},
               {"partition": "cpuonly", "state": "idle", "gres": None}]


def test_resolve_explicit():
    t = placer.resolve_explicit("trace:batch:a40@normal")
    assert t.key == "trace:batch:a40@normal" and t.partitions == ["batch"]
    assert placer.resolve_explicit(["bridges2:GPU-small,GPU-shared:v100-32"]).partitions == ["GPU-small", "GPU-shared"]
    assert placer.resolve_explicit(Target(cluster="x", partitions=["p"])).key == "x:p"
    for bad in ("auto", ["a:b", "c:d"], "nocolon"):
        with pytest.raises(SlurmMcpError):
            placer.resolve_explicit(bad)
    caps, profile = caps_for("trace")
    with pytest.raises(SlurmMcpError) as e:
        placer.resolve_explicit("trace:nope", {"trace": profile}, {"trace": caps})
    assert e.value.code == "E_PARTITION"
    with pytest.raises(SlurmMcpError):
        placer.resolve_explicit("mars:batch", {"trace": profile})
    b2c, b2p = caps_for("bridges2")
    assert placer.resolve_explicit("bridges2:RM", {"bridges2": b2p}, {"bridges2": b2c}).account == "mch250030p"


def test_candidates_trace_and_bridges2(world):
    caps, profiles = world
    keys = [t.key for t in placer.candidates(spec(), caps, profiles, PlacementPolicy())]
    # one typed candidate per gres type; biosimmlab disabled by the profile override; CPU partitions skipped
    assert "trace:batch:a40@normal" in keys and not any(k.startswith("trace:biosimmlab") for k in keys)
    assert not any(k.startswith("trace:cpuonly") for k in keys)
    assert "bridges2:GPU-shared:h100-80@gpu" in keys and "bridges2:GPU-shared:v100-32@gpu" in keys
    assert "bridges2:GPU-small,GPU-shared:v100-32@gpu" in keys           # partition_groups joint candidate
    assert not any(k.startswith("bridges2:GPU-small,GPU-shared:h100") for k in keys)   # GPU-small has no h100
    assert not any("untyped" in k or k.endswith("GPU-shared@gpu") for k in keys)
    # CPU job: one untyped candidate per accessible CPU partition, no gres part in the key. GPU partitions
    # are NOT offered to a job that asked for no GPUs -- an accelerator would idle and, on a charging cluster,
    # be billed at the GPU rate (observed 2026-09-02: a 1-core job was auto-placed on a Bridges-2 GPU
    # partition). Naming the partition explicitly still allows it.
    cpu = [t.key for t in placer.candidates(spec(resources={"time": "01:00:00"}), caps, profiles)]
    assert "trace:cpuonly" in cpu and "bridges2:RM-shared@low" in cpu
    assert "trace:batch@normal" not in cpu, "batch carries A40s; a CPU-only job must not land there"
    assert not any(k.startswith("bridges2:GPU") for k in cpu)
    pinned = [t.key for t in placer.candidates(spec(resources={"time": "01:00:00"}, partition="batch"),
                                               caps, profiles)]
    assert pinned == ["trace:batch@normal"], "an explicit partition is always honoured"
    # gpu_types restricts, spec.cluster pins, explicit placement lists restrict, allow/deny globs
    v100 = [t.key for t in placer.candidates(spec(resources={"time": "1:00:00", "gpus": 1, "gpu_types": ["v100-32"]}),
                                             caps, profiles)]
    assert v100 and all(":v100-32" in k for k in v100)
    assert all(k.startswith("trace:") for k in (t.key for t in placer.candidates(spec(cluster="trace"), caps, profiles)))
    restricted = [t.key for t in placer.candidates(spec(), caps, profiles, placement=["bridges2:GPU-shared"])]
    assert restricted and all(k.startswith("bridges2:GPU-shared:") for k in restricted)
    only = [t.key for t in placer.candidates(spec(), caps, profiles, placement="trace:batch:a40")]
    assert only == ["trace:batch:a40@normal"]
    pol = PlacementPolicy(targets_allow=["trace:*"], targets_deny=["*:batch*"])
    assert placer.candidates(spec(), caps, profiles, pol) == []
    pol = PlacementPolicy(prefer_cluster="bridges2")
    assert placer.candidates(spec(), caps, profiles, pol)[0].cluster == "bridges2"
    assert placer.candidates(spec(), {"trace": None}, profiles) == []


def test_feasibility_hard_rules(world):
    caps, profiles = world
    tc, bc = caps["trace"], caps["bridges2"]
    ok, why = placer.feasibility(Target.parse("trace:batch:a40@normal"), spec(), tc)
    assert ok and why == ""
    ok, why = placer.feasibility(Target.parse("trace:batch:a40"), spec(resources={"time": "3-00:00:00", "gpus": 1}), tc)
    assert not ok and "exceeds max wall 2-00:00:00" in why
    ok, why = placer.feasibility(Target.parse("trace:batch:h100"), spec(), tc)
    assert not ok and "no h100 GPUs in batch" in why
    ok, why = placer.feasibility(Target.parse("trace:batch:a40"), spec(resources={"time": "1:00:00", "gpus": 2}), tc)
    assert not ok and "per node" in why
    ok, why = placer.feasibility(Target.parse("bridges2:GPU-small:v100-32@gpu"),
                                 spec(resources={"time": "1:00:00", "gpus": 8, "nodes": 3}), bc)
    assert not ok and "gpus x nodes 24 > per-job limit 16" in why
    ok, why = placer.feasibility(Target.parse("bridges2:RM-shared@low"),
                                 spec(resources={"time": "1:00:00", "nodes": 2}), bc)
    assert not ok and "MaxNodes 1" in why
    ok, why = placer.feasibility(Target.parse("bridges2:RM-shared@low"), spec(resources={"time": "1:00:00", "cpus": 200}), bc)
    assert not ok and "cpus" in why
    ok, why = placer.feasibility(Target.parse("bridges2:RM-shared@low"), spec(resources={"time": "1:00:00", "mem": "9T"}), bc)
    assert not ok and "node memory" in why
    ok, why = placer.feasibility(Target.parse("trace:nope"), spec(), tc)
    assert not ok and "unknown partition" in why
    ok, why = placer.feasibility(Target.parse("trace:batch"), spec(), None)
    assert not ok and "not discovered" in why
    down = {**tc, "partitions": {**tc["partitions"], "batch": {**tc["partitions"]["batch"], "state": "DOWN"}}}
    assert placer.feasibility(Target.parse("trace:batch:a40"), spec(), down)[1] == "partition batch is DOWN"
    ok, why = placer.feasibility(Target.parse("trace:batch:a40"), spec(), tc, my_running={"batch": ["j1"]},
                                 policy=PlacementPolicy(max_running_per_target={"trace:batch*": 1}))
    assert not ok and "etiquette cap" in why
    tp = profiles["trace"]
    ok, why = placer.feasibility(Target.parse("trace:batch:a40"), spec(), tc, my_running={"batch": 1},
                                 profile=profile_for("trace", target_overrides={"trace:batch*": {"max_running": 1}}))
    assert not ok
    ok, why = placer.feasibility(Target.parse("bridges2:GPU-small:v100-32@gpu"), spec(), bc, my_running={"GPU-small": 2})
    assert not ok and "max_jobs_pu 2" in why
    ok, why = placer.feasibility(Target.parse("bridges2:GPU-small:v100-32@gpu"), spec(), bc, my_running={"GPU-small": 1},
                                 my_pending={"GPU-small": 9})
    assert not ok and "max_submit_pu 10" in why
    stats = {"breaker_open_until_local": 2000.0, "last_error": "NODE_FAIL x2"}
    ok, why = placer.feasibility(Target.parse("trace:batch:a40"), spec(), tc, target_stats=stats, now_local=1000.0)
    assert not ok and "circuit breaker" in why and "NODE_FAIL" in why
    stats = {"infeasible_until_local": 1500.0, "infeasible_reason": "E_QOS_SIZE"}
    ok, why = placer.feasibility(Target.parse("trace:batch:a40"), spec(), tc, target_stats=stats, now_local=1000.0)
    assert not ok and "recently refused" in why
    assert placer.feasibility(Target.parse("trace:batch:a40"), spec(), tc, target_stats=stats, now_local=2000.0)[0]
    assert tp is profiles["trace"]


def test_self_preemption_rule(world):
    caps, profiles = world
    tc = caps["trace"]
    tgt = Target.parse("trace:biosimmlab:a40@normal")
    snap = snapshot(tc, TRACE_PD, TRACE_NODES)       # biosimmlab has no idle a40 node, batch has one
    ok, why = placer.feasibility(tgt, spec(), tc, snap, my_running={"batch": ["j12", "j14"]})
    assert not ok and "would preempt my own batch job(s) (j12, j14)" in why
    assert placer.feasibility(tgt, spec(), tc, snap, my_running={"batch": 0})[0]
    assert placer.feasibility(tgt, spec(), tc, snap, my_running={"batch": 1}, policy=PlacementPolicy(allow_self_preempt=True))[0]
    prof = profile_for("trace", target_overrides={"trace:biosimmlab*": {"allow_self_preempt": True}})
    assert placer.feasibility(tgt, spec(), tc, snap, my_running={"batch": 1}, profile=prof)[0]
    idle = snapshot(tc, TRACE_PD, TRACE_NODES + [{"partition": "biosimmlab", "state": "idle", "gres": {"type": "a40", "count": 1}}])
    assert placer.feasibility(tgt, spec(), tc, idle, my_running={"batch": 1})[0]
    # the lower-tier partition itself is never "self-preempting"
    assert placer.feasibility(Target.parse("trace:batch:a40"), spec(), tc, snap, my_running={"biosimmlab": 1})[0]
    assert placer.expand_hostlist("trace[01-03],x[7,9-10],solo") == {"trace01", "trace02", "trace03", "x7", "x9", "x10", "solo"}


def test_queue_depth_and_rank_order(world):
    caps, profiles = world
    tc = caps["trace"]
    snap = snapshot(tc, TRACE_PD, TRACE_NODES)
    assert placer.queue_depth(snap, tc, Target.parse("trace:batch:a40"), spec()) == (3, 242)
    assert placer.queue_depth(snap, tc, Target.parse("trace:cpuonly"), spec(resources={"time": "1:00:00"})) == (23, 0)
    assert placer.queue_depth(None, tc, Target.parse("trace:batch:a40"), spec()) == (None, None)
    b2_snap = snapshot(caps["bridges2"], [{"count": 5, "partition": "GPU-shared", "tres_per_node": "gres:gpu:h100-80:1",
                                            "tres_per_job": "N/A"}], [])
    options = placer.rank(spec(cluster=None), caps, {"trace": snap, "bridges2": b2_snap}, profiles, PlacementPolicy(),
                          placement="auto", max_options=20)
    assert options and all(o.feasible for o in options[:1])
    by = {o.target: o for o in options}
    trace_opt = by["trace:batch:a40@normal"]
    assert trace_opt.queue_ahead == 245 and trace_opt.queue_ahead_untyped == 242 and trace_opt.est_wait_src == "depth"
    assert trace_opt.est_wait_h == 12.0 and trace_opt.score_h == pytest.approx(16.0) and trace_opt.charge == "free"
    h100 = by["bridges2:GPU-shared:h100-80@gpu"]
    assert h100.est_wait_h == pytest.approx(1.25) and h100.charge.unit == "gpu:h100-80" and h100.requeueable is False
    assert options[0].target == "bridges2:GPU-shared:l40s-48@gpu" or options[0].est_wait_h == 0.0
    scores = [o.score_h for o in options if o.feasible]
    assert scores == sorted(scores)
    assert placer.recommended(options) == options[0].target
    # infeasible rows last, at most 3, with why
    big = spec(resources={"time": "1:00:00", "gpus": 99})
    opts = placer.rank(big, caps, {}, profiles, PlacementPolicy())
    assert opts and all(not o.feasible and o.why for o in opts) and len(opts) <= placer.MAX_INFEASIBLE_ROWS
    assert placer.recommended(opts) is None
    # etiquette soft caps and penalties raise the score; staging penalty when inputs live elsewhere
    pol = PlacementPolicy(soft_caps={"trace:batch*": 1}, etiquette_h=2.0)
    o = placer.rank(spec(cluster="trace"), caps, {"trace": snap}, profiles, pol, my_running={"trace": {"batch": 1}})[0]
    assert o.etiquette_h == 2.0 and o.score_h == pytest.approx(18.0) and "etiquette" in o.why
    prof = {**profiles, "trace": profile_for("trace", target_overrides={"trace:batch*": {"penalty_h": 1.5},
                                                                        "trace:biosimmlab*": {"enabled": False}})}
    o = placer.rank(spec(cluster="trace"), caps, {"trace": snap}, prof, PlacementPolicy(), inputs_cluster="bridges2")[0]
    assert o.score_h == pytest.approx(16.0 + 1.5 + 0.5)
    o = placer.rank(spec(cluster="trace"), caps, {}, profiles, PlacementPolicy())[0]
    assert o.est_wait_src == "none" and o.est_wait_h == 12.0
