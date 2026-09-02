"""Unit tests for slurm_mcp.slurm.discovery (design section 6.1) on fixture-built caps and a temp Store."""
from __future__ import annotations

import time

import pytest

from unit.fake_transport import FakeSFTP, FakeTransport, framed_discovery, ok, profile_for
from slurm_mcp.clock import ClusterClock
from slurm_mcp.helpers import bundle_sha8
from slurm_mcp.slurm import discovery as D
from slurm_mcp.slurm.client import SlurmClient, parse_discovery
from slurm_mcp.slurm.parse import parse_sections
from slurm_mcp.store import Store


def caps_for(cluster: str, **kw):
    profile = profile_for(cluster, **kw)
    caps = parse_discovery(parse_sections(framed_discovery(cluster)), profile, cluster=cluster)
    return D.enrich_caps(caps, profile), profile


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "state.db", pid=1, host="lap", pid_exists=lambda pid: True)
    yield s
    s.close()


def test_cmd_timeout_rule_from_fixtures():
    trace, _ = caps_for("trace")
    b2, _ = caps_for("bridges2")
    assert trace["cmd_timeout_s"] == 120 and b2["cmd_timeout_s"] == 260
    assert caps_for("bridges2", cmd_timeout_s=400)[0]["cmd_timeout_s"] == 400


def test_qos_candidates_section_6_1_examples():
    trace, tp = caps_for("trace")
    b2, bp = caps_for("bridges2")
    assert D.qos_for_partition(trace, tp, "batch") == ["normal"]
    assert D.qos_for_partition(b2, bp, "GPU-shared")[0] == "gpu"
    assert D.qos_for_partition(b2, bp, "RM-shared")[0] == "low"
    assert D.qos_for_partition(b2, bp, "GPU-shared", spec_qos="push") == ["push"]
    # the validated cache wins and qos_map overrides discovery
    b2["qos_for_partition"] = {"GPU-shared": "low"}
    assert D.qos_for_partition(b2, bp, "GPU-shared")[:2] == ["low", "gpu"]
    assert D.qos_for_partition(b2, profile_for("bridges2", qos_map={"GPU-shared": "ft"}), "GPU-shared") == ["ft"]
    # AllowQos=ALL with an assoc default -> no --qos
    trace["assoc"]["default_qos"] = "normal"
    assert D.qos_for_partition(trace, tp, "cpuonly") == []


def test_effective_limits_and_partition_access():
    b2, bp = caps_for("bridges2")
    small = D.effective_limits(b2, "GPU-small", "gpu")
    assert small["max_wall_s"] == 8 * 3600 and small["max_jobs_pu"] == 2 and small["max_submit_pu"] == 10
    assert small["max_tres_pj"] == {"gres/gpu": 16.0}
    shared = D.effective_limits(b2, "RM-shared", "low")
    assert shared["max_wall_s"] == 3 * 86400 and shared["max_nodes"] == 1 and shared["max_tres_pj"]["cpu"] == 64.0
    gpu = D.effective_limits(b2, "GPU", "gpu")
    assert gpu["max_nodes"] == 8 and gpu["max_tres_pj"]["gres/gpu"] == 64.0
    assert D.effective_limits(b2, "nope")["max_wall_s"] is None
    trace, tp = caps_for("trace")
    assert D.effective_limits(trace, "batch")["max_wall_s"] == 2 * 86400
    assert D.effective_limits(trace, "cpuonly-debug")["max_wall_s"] == 4 * 3600
    # access: group unknown in the fixture env -> lenient; a known group is enforced
    assert trace["partitions"]["cpuonly"]["accessible"] is True
    trace["group"] = "users"
    assert D.partition_accessible(trace["partitions"]["cpuonly"], trace) is False
    assert D.partition_accessible(trace["partitions"]["batch"], trace) is True
    trace["group"] = "biosimmlab"
    assert D.partition_accessible(trace["partitions"]["biosimmlab"], trace) is True
    trace["assoc"]["partition"] = "batch"
    assert D.partition_accessible(trace["partitions"]["cpuonly"], trace) is False


def test_charge_for_rates_and_weights():
    b2, bp = caps_for("bridges2")
    assert D.charge_for(b2, bp, "GPU-shared", "h100-80") == {"unit": "gpu:h100-80", "su_per_unit_h": 2.0}
    assert D.charge_for(b2, bp, "GPU-shared", "v100-32") == {"unit": "gpu:v100-32", "su_per_unit_h": 1.0}
    assert D.charge_for(b2, bp, "RM", None) == {"unit": "cpu", "su_per_unit_h": 1.0}
    trace, tp = caps_for("trace")
    assert D.charge_for(trace, tp, "batch", "a40") == "free"
    trace["partitions"]["batch"]["tres_billing_weights"] = {"cpu": "0.5", "gres/gpu:a40": "3.0"}
    assert D.charge_for(trace, tp, "batch", "a40") == {"unit": "gpu:a40", "su_per_unit_h": 3.0}
    assert D.charge_for(trace, tp, "batch", "other") == {"unit": "cpu", "su_per_unit_h": 0.5}


def _client(cluster, handlers, sftp=None):
    profile = profile_for(cluster)
    t = FakeTransport(profile, handlers, sftp=sftp)
    holder = {}
    c = SlurmClient(cluster, t, None, ClusterClock(), lambda: holder.get("caps"))
    return c, t, holder


@pytest.mark.asyncio
async def test_bootstrap_caches_24h_and_refreshes(store):
    c, t, holder = _client("trace", [("echo '::ENV'", ok(framed_discovery("trace")))])
    caps = await D.bootstrap(c, c.profile, store)
    assert caps["fetched_local"] > 0 and len(t.calls) == 1
    again = await D.bootstrap(c, c.profile, store)
    assert again["fetched_local"] == caps["fetched_local"] and len(t.calls) == 1
    caps["qos_for_partition"] = {"batch": "normal"}
    caps["transfer"] = {"host": "data.example", "port": 22}
    await D.save_caps(store, "trace", caps)
    fresh = await D.bootstrap(c, c.profile, store, refresh=True)
    assert len(t.calls) == 2 and fresh["qos_for_partition"] == {"batch": "normal"} and fresh["transfer"]["port"] == 22
    stale = dict(fresh, fetched_local=time.time() - D.CAPS_TTL_S - 1)
    await D.save_caps(store, "trace", stale)
    assert not D.caps_fresh(stale) and D.caps_age_s(stale) > D.CAPS_TTL_S
    await D.bootstrap(c, c.profile, store)
    assert len(t.calls) == 3
    assert store.read_sync(lambda conn: store.kv_get(conn, D.caps_key("trace")))["cluster"] == "trace"


@pytest.mark.asyncio
async def test_ensure_helpers_deploys_only_when_version_differs(store):
    sha = bundle_sha8()
    sftp = FakeSFTP()
    c, t, holder = _client("trace", [("cat '/trace/group/biosimmlab/wxu2/.slurm-mcp/bin/VERSION'", ok(""))], sftp=sftp)
    caps = {"helper_sha8": None}
    assert await D.ensure_helpers(c, c.profile, caps, store) == sha
    assert caps["helper_sha8"] == sha and len(sftp.renames) == 4
    assert store.read_sync(lambda conn: store.kv_get(conn, D.caps_key("trace")))["helper_sha8"] == sha
    assert await D.ensure_helpers(c, c.profile, caps) == sha and len(sftp.renames) == 4
    caps2 = {"helper_sha8": "stale000"}
    c2, t2, _ = _client("trace", [("bin/VERSION", ok(sha + "\n"))], sftp=sftp)
    assert await D.ensure_helpers(c2, c2.profile, caps2) == sha and len(sftp.renames) == 4 and caps2["helper_sha8"] == sha
    c3, t3, _ = _client("trace", [("bin/VERSION", ok("old00000\n"))], sftp=sftp)
    assert await D.ensure_helpers(c3, c3.profile, {"helper_sha8": None}) == sha and len(sftp.renames) == 8


@pytest.mark.asyncio
async def test_backfill_wait_history_once_per_cluster(store):
    text = ("615300|batch|normal|billing=64,cpu=64,gres/gpu=1,gres/gpu:a40=1,mem=512G,node=1|1756000000|1756003600|COMPLETED\n"
            "615301|cpuonly|normal|cpu=8|1756010000|Unknown|CANCELLED by 1\n"
            "615302|cpuonly|normal|cpu=8|1756020000|1756020010|COMPLETED\n")
    c, t, holder = _client("trace", [("sacct -nP -X -u", ok(text))])
    assert await D.backfill_wait_history(c, store) == 2
    rows = store.read_sync(lambda conn: store.wait_history(conn, "trace", "trace:batch:a40"))
    assert len(rows) == 1 and rows[0]["wait_s"] == 3600 and rows[0]["source"] == "backfill" and rows[0]["gpus"] == 1
    assert store.read_sync(lambda conn: store.wait_history(conn, "trace", "trace:cpuonly"))[0]["wait_s"] == 10
    assert await D.backfill_wait_history(c, store) == 0 and len(t.calls) == 1
    assert D.backfill_target_key("x", {"partition": None}) is None


@pytest.mark.asyncio
async def test_transfer_capabilities_probe_and_login_fallback():
    class T:
        async def run(self, cmd, **kw):
            return ok("ok\n")(cmd)

        async def sftp(self):
            class S:
                async def realpath(self, p):
                    return "/home"
            return S()

    banners = {22: "SSH-2.0-OpenSSH_8.0", 2222: "SSH-2.0-OpenSSH_8.0-hpn14v15"}

    async def banner(host, port, timeout=5):
        return banners.get(port, "")

    p = profile_for("bridges2", transfer_host="data.example")
    caps = await D.transfer_capabilities(p, T(), banner_probe=banner)
    assert caps == {"host": "data.example", "port": 2222, "banner": banners[2222], "exec_ok": True, "sftp_ok": True,
                    "mb_per_s": None, "role": "transfer"}
    banners[2222] = ""
    caps = await D.transfer_capabilities(p, None, banner_probe=banner)
    assert caps["port"] == 22 and caps["exec_ok"] is None
    caps = await D.transfer_capabilities(profile_for("bridges2", transfer_host="data.example", transfer_port=2200), None,
                                         banner_probe=banner)
    assert caps["port"] == 2200
    login = await D.transfer_capabilities(profile_for("trace"), None, banner_probe=banner)
    assert login["role"] == "login" and login["host"] == "trace.example" and login["exec_ok"] is True


def test_target_enabled_and_globs():
    p = profile_for("trace")
    assert D.target_enabled(p, "trace:biosimmlab:a40@normal") is False and D.target_enabled(p, "trace:batch:a40") is True
    assert D.matches_any("trace:batch:a40", ["trace:*"]) and not D.matches_any("x", [])


# --- supplementary groups decide partition access (measured on TRACE 2026-09-02) ---------------------------

def test_partition_accessible_matches_any_of_the_users_groups():
    """AllowGroups is normally a SUPPLEMENTARY group.

    On TRACE the primary group is ``users`` while ``cpuonly``, ``cpuonly-debug`` and ``biosimmlab`` allow the
    supplementary ``biosimmlab``; matching only the primary group hid every partition the user can actually
    submit to, so ``plan_job`` reported "no feasible target" for a job ``submit_job`` accepted.
    """
    from slurm_mcp.slurm.discovery import partition_accessible

    part = {"name": "cpuonly-debug",
            "allow_groups": ["pscstaff", "dabo", "cmustaff", "acmegroup", "toconnor", "biosimmlab"]}
    caps_primary_only = {"group": "users", "groups": ["users"]}
    caps_with_supp = {"group": "users", "groups": ["users", "biosimmlab"]}
    assert partition_accessible(part, caps_primary_only) is False
    assert partition_accessible(part, caps_with_supp) is True
    # ALL still wins, and an unknown/numeric group is never held against a partition
    assert partition_accessible({"name": "batch", "allow_groups": ["ALL"]}, caps_primary_only) is True
    assert partition_accessible(part, {"group": "1234", "groups": ["1234"]}) is True
    assert partition_accessible(part, {}) is True
