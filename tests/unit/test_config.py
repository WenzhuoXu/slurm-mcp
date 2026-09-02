"""Unit tests for slurm_mcp.config (design section 2.1)."""
from __future__ import annotations

import json
import logging
from dataclasses import fields

import pytest

from slurm_mcp import config
from slurm_mcp.config import (DEFAULT_POLL, ClusterProfile, control_root, has_transfer_host, load_profiles,
                              poll_settings, profile_from_dict, save_profiles, target_override, transfer_endpoint,
                              transfer_host)

SECTION_2_1_FIELDS = {
    "transfer_host": None, "transfer_port": None, "control_root": None, "partition_groups": [], "qos_map": {},
    "no_mem_flag": [], "su_rates": {}, "balance_command": None, "balance_regex": None, "quota_command": None,
    "quota_paths": [], "poll": {"base_s": 60, "min_s": 30, "max_s": 120}, "requires_vpn_hint": None,
    "target_overrides": {}, "ssh_max_exec": 6, "cmd_timeout_s": None,
}
LEGACY_FIELDS = ["name", "host", "user", "port", "auth", "key_path", "data_host", "remote_root", "default_account",
                 "default_partition", "extra"]


def minimal(**kw) -> ClusterProfile:
    base = dict(name="c", host="login.example.org", user="me")
    base.update(kw)
    return ClusterProfile(**base)


def test_all_fields_present_with_defaults():
    names = [f.name for f in fields(ClusterProfile)]
    assert names[:11] == LEGACY_FIELDS
    assert set(SECTION_2_1_FIELDS) <= set(names)
    p = minimal()
    for k, v in SECTION_2_1_FIELDS.items():
        assert getattr(p, k) == v, k
    assert p.port == 22 and p.auth == "password" and p.extra == {}


def test_defaults_are_not_shared():
    a, b = minimal(), minimal()
    a.partition_groups.append(["x", "y"])
    a.poll["base_s"] = 5
    a.su_rates["cpu"] = 1
    assert b.partition_groups == [] and b.poll["base_s"] == 60 and b.su_rates == {}


def test_poll_partial_is_merged_with_defaults():
    p = minimal(poll={"base_s": 90})
    assert p.poll == {"base_s": 90, "min_s": 30, "max_s": 120}
    assert poll_settings(p) == p.poll
    assert poll_settings(minimal()) == DEFAULT_POLL


def test_validate_legacy_rules():
    minimal().validate()
    with pytest.raises(ValueError):
        minimal(auth="magic").validate()
    with pytest.raises(ValueError):
        minimal(auth="key").validate()
    minimal(auth="key", key_path="~/.ssh/id").validate()


@pytest.mark.parametrize("kw", [
    {"ssh_max_exec": 0}, {"cmd_timeout_s": 0}, {"transfer_port": 70000}, {"poll": {"min_s": 200}},
    {"poll": {"min_s": 1, "base_s": 1, "max_s": 1}}, {"balance_regex": "("},
    {"target_overrides": {"trace:*": {"bogus": 1}}}, {"target_overrides": {"trace:*": 5}},
    {"partition_groups": [["only-one"]]},
])
def test_validate_new_rules(kw):
    with pytest.raises(ValueError):
        minimal(**kw).validate()


def test_validate_new_rules_ok():
    minimal(ssh_max_exec=2, cmd_timeout_s=260, transfer_port=2222, poll={"min_s": 10, "base_s": 20, "max_s": 30},
            balance_regex=r"(?P<left>[\d,]+)\s*/\s*(?P<total>[\d,]+)\s*SU",
            target_overrides={"trace:biosimmlab*": {"enabled": False, "max_running": 1, "allow_self_preempt": True}},
            partition_groups=[["GPU-small", "GPU-shared"]]).validate()


def test_credential_id_unchanged():
    assert minimal(port=2222).credential_id == "me@login.example.org:2222"


# --- helpers ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw,expected", [
    ({}, "$HOME/.slurm-mcp"),
    ({"remote_root": "/ocean/projects/p/u"}, "/ocean/projects/p/u/.slurm-mcp"),
    ({"remote_root": "/ocean/projects/p/u/"}, "/ocean/projects/p/u/.slurm-mcp"),
    ({"remote_root": "/r", "control_root": "/ctrl/"}, "/ctrl"),
    ({"control_root": "/ctrl"}, "/ctrl"),
])
def test_control_root(kw, expected):
    assert control_root(minimal(**kw)) == expected


def test_transfer_host_resolution():
    assert transfer_host(minimal()) is None
    assert transfer_host(minimal(data_host="data.old")) == "data.old"
    assert transfer_host(minimal(transfer_host="data.new")) == "data.new"
    assert transfer_host(minimal(data_host="data.old", transfer_host="data.new")) == "data.new"
    assert transfer_host(minimal(transfer_host="")) is None
    assert not has_transfer_host(minimal())
    assert not has_transfer_host(minimal(transfer_host="login.example.org"))
    assert has_transfer_host(minimal(transfer_host="data.x"))


def test_transfer_endpoint():
    assert transfer_endpoint(minimal(port=2200)) == ("login.example.org", 2200)
    assert transfer_endpoint(minimal(transfer_host="d")) == ("d", 22)
    assert transfer_endpoint(minimal(transfer_host="d"), discovered_port=2222) == ("d", 2222)
    assert transfer_endpoint(minimal(transfer_host="d", transfer_port=2222), discovered_port=22) == ("d", 2222)
    assert transfer_endpoint(minimal(data_host="legacy")) == ("legacy", 22)


def test_target_override_merges_matching_globs():
    p = minimal(target_overrides={"trace:*": {"penalty_h": 1.0}, "trace:biosimmlab*": {"enabled": False},
                                  "bridges2:*": {"max_pending": 3}})
    assert target_override(p, "trace:biosimmlab:a40") == {"penalty_h": 1.0, "enabled": False}
    assert target_override(p, "trace:batch:a40") == {"penalty_h": 1.0}
    assert target_override(p, "bridges2:GPU-shared:v100-32@gpu") == {"max_pending": 3}
    assert target_override(p, "other:x") == {}


# --- load / save -----------------------------------------------------------------------------------

def test_load_legacy_config_defaults_new_fields(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"clusters": {"trace": {"host": "h", "user": "u", "auth": "password",
                                                       "data_host": "data.h", "remote_root": "/r"}}}))
    profiles = load_profiles(cfg)
    p = profiles["trace"]
    assert p.name == "trace" and p.data_host == "data.h" and p.port == 22
    for k, v in SECTION_2_1_FIELDS.items():
        assert getattr(p, k) == v, k
    assert transfer_host(p) == "data.h"


def test_load_unknown_keys_warn_and_are_kept_in_extra(tmp_path, caplog):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"clusters": {"c": {"host": "h", "user": "u", "future_key": 1, "extra": {"a": 1}}}}))
    with caplog.at_level(logging.WARNING, logger="slurm_mcp.config"):
        p = load_profiles(cfg)["c"]
    assert "future_key" in caplog.text
    assert p.extra["a"] == 1 and p.extra["_unknown"] == {"future_key": 1}
    assert not hasattr(p, "future_key")


def test_profile_from_dict_name_default():
    p = profile_from_dict("x", {"host": "h", "user": "u"})
    assert p.name == "x"
    assert profile_from_dict("x", {"name": "y", "host": "h", "user": "u"}).name == "y"


def test_save_then_load_roundtrip_all_fields(tmp_path):
    cfg = tmp_path / "config.json"
    p = minimal(name="b2", transfer_host="data.b2", transfer_port=2222, control_root="/c", remote_root="/r",
                partition_groups=[["GPU-small", "GPU-shared"]], qos_map={"GPU-shared": "gpu"},
                no_mem_flag=["RM-shared"], su_rates={"gpu:h100-80": 2, "gpu:*": 1, "cpu": 1},
                balance_command="projects", balance_regex=r"(?P<left>\d+)", quota_command="my_quotas",
                quota_paths=["/ocean/projects/x"], poll={"base_s": 45}, requires_vpn_hint="vpn",
                target_overrides={"b2:GPU-small*": {"max_running": 2}}, ssh_max_exec=4, cmd_timeout_s=260,
                default_account="acct", extra={"k": "v"})
    save_profiles({"b2": p}, cfg)
    assert cfg.exists() and not cfg.with_suffix(".json.tmp").exists()
    raw = json.loads(cfg.read_text(encoding="utf-8"))
    assert raw["clusters"]["b2"]["su_rates"] == {"gpu:h100-80": 2.0, "gpu:*": 1.0, "cpu": 1.0}
    loaded = load_profiles(cfg)["b2"]
    assert loaded == p
    assert loaded.poll == {"base_s": 45, "min_s": 30, "max_s": 120}


def test_load_missing_file_returns_empty(tmp_path):
    assert load_profiles(tmp_path / "none.json") == {}


def test_default_paths_use_mcp_home(mcp_home):
    assert str(config.CONFIG_DIR).startswith(mcp_home)
    assert config.CONFIG_PATH.name == "config.json" and config.KNOWN_HOSTS_PATH.name == "known_hosts"


def test_get_profile_default_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(config, "KNOWN_HOSTS_PATH", tmp_path / "known_hosts")
    save_profiles({"t": minimal(name="t")})
    assert config.get_profile("t").host == "login.example.org"
    assert (tmp_path / "known_hosts").exists()
    with pytest.raises(KeyError):
        config.get_profile("missing")
