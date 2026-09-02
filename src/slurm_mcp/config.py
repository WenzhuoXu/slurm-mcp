"""Cluster profiles (design section 2.1). Stored as JSON in ~/.slurm-mcp/config.json (override with
SLURM_MCP_HOME).

Profiles never contain secrets. Passwords live in the OS keyring (see credentials.py).
Backward compatibility: a config.json written by an older version loads unchanged (missing keys are
defaulted, unknown keys are ignored with a warning and preserved under ``extra``).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG_DIR = Path(os.environ.get("SLURM_MCP_HOME", str(Path.home() / ".slurm-mcp")))
CONFIG_PATH = CONFIG_DIR / "config.json"
KNOWN_HOSTS_PATH = CONFIG_DIR / "known_hosts"

AUTH_METHODS = ("password", "key", "agent")
DEFAULT_POLL: dict[str, int] = {"base_s": 60, "min_s": 30, "max_s": 120}
DEFAULT_SSH_MAX_EXEC = 6
# Keys accepted inside profile.target_overrides[<glob>] (design section 2.1).
TARGET_OVERRIDE_KEYS = ("enabled", "max_pending", "max_running", "soft_cap", "preference", "penalty_h",
                        "allow_self_preempt")


def _default_poll() -> dict:
    return dict(DEFAULT_POLL)


@dataclass
class ClusterProfile:
    """One cluster (design section 2.1). Existing fields kept; new fields all default."""

    name: str
    host: str
    user: str
    port: int = 22
    auth: str = "password"            # password | key | agent
    key_path: str | None = None       # for auth == key
    data_host: str | None = None      # dedicated transfer node (kept for backward compat; transfer_host wins)
    remote_root: str | None = None    # default remote directory for uploaded work
    default_account: str | None = None
    default_partition: str | None = None
    extra: dict = field(default_factory=dict)
    # --- section 2.1 additions ---
    transfer_host: str | None = None
    transfer_port: int | None = None
    control_root: str | None = None
    partition_groups: list[list[str]] = field(default_factory=list)
    qos_map: dict[str, str] = field(default_factory=dict)
    no_mem_flag: list[str] = field(default_factory=list)
    su_rates: dict[str, float] = field(default_factory=dict)
    balance_command: str | None = None
    balance_regex: str | None = None
    quota_command: str | None = None
    quota_paths: list[str] = field(default_factory=list)
    poll: dict = field(default_factory=_default_poll)
    requires_vpn_hint: str | None = None
    target_overrides: dict[str, dict] = field(default_factory=dict)
    ssh_max_exec: int = DEFAULT_SSH_MAX_EXEC
    cmd_timeout_s: int | None = None

    def __post_init__(self) -> None:
        # Fill missing poll keys so callers can index without checks.
        merged = dict(DEFAULT_POLL)
        merged.update({k: int(v) for k, v in (self.poll or {}).items() if v is not None})
        self.poll = merged
        self.partition_groups = [list(g) for g in (self.partition_groups or [])]
        self.su_rates = {str(k): float(v) for k, v in (self.su_rates or {}).items()}

    def validate(self) -> None:
        if self.auth not in AUTH_METHODS:
            raise ValueError(f"auth must be one of {AUTH_METHODS}, got {self.auth!r}")
        if self.auth == "key" and not self.key_path:
            raise ValueError("auth=key requires key_path")
        if self.ssh_max_exec < 1:
            raise ValueError("ssh_max_exec must be >= 1")
        if self.cmd_timeout_s is not None and self.cmd_timeout_s <= 0:
            raise ValueError("cmd_timeout_s must be > 0")
        if self.transfer_port is not None and not (0 < self.transfer_port < 65536):
            raise ValueError("transfer_port must be in 1..65535")
        if not (self.poll["min_s"] <= self.poll["base_s"] <= self.poll["max_s"]) or self.poll["min_s"] < 5:
            raise ValueError(f"poll must satisfy 5 <= min_s <= base_s <= max_s, got {self.poll}")
        if self.balance_regex:
            import re
            try:
                re.compile(self.balance_regex)
            except re.error as e:
                raise ValueError(f"balance_regex is not a valid regex: {e}") from None
        for glob, ov in self.target_overrides.items():
            if not isinstance(ov, dict):
                raise ValueError(f"target_overrides[{glob!r}] must be an object")
            unknown = sorted(set(ov) - set(TARGET_OVERRIDE_KEYS))
            if unknown:
                raise ValueError(f"target_overrides[{glob!r}] has unknown keys {unknown}; allowed: {list(TARGET_OVERRIDE_KEYS)}")
        for group in self.partition_groups:
            if len(group) < 2:
                raise ValueError(f"partition_groups entries need >= 2 partitions, got {group}")

    @property
    def credential_id(self) -> str:
        return f"{self.user}@{self.host}:{self.port}"


# --- resolution helpers (section 2.1 defaults) -----------------------------------------------------

def control_root(profile: ClusterProfile) -> str:
    """``profile.control_root`` or ``<remote_root>/.slurm-mcp`` or ``$HOME/.slurm-mcp``."""
    if profile.control_root:
        return profile.control_root.rstrip("/") or "/"
    if profile.remote_root:
        return profile.remote_root.rstrip("/") + "/.slurm-mcp"
    return "$HOME/.slurm-mcp"


def transfer_host(profile: ClusterProfile) -> str | None:
    """The dedicated transfer node: ``transfer_host`` wins over the legacy ``data_host``; None if neither."""
    return profile.transfer_host or profile.data_host or None


def has_transfer_host(profile: ClusterProfile) -> bool:
    """True when a transfer host exists and differs from the login host."""
    th = transfer_host(profile)
    return bool(th) and th != profile.host


def transfer_endpoint(profile: ClusterProfile, discovered_port: int | None = None) -> tuple[str, int]:
    """``(host, port)`` for the transfer role.

    Host: the transfer host, else the login host. Port: ``profile.transfer_port``, else the port discovered
    by the banner probe (``caps``), else the login port when no transfer host is set, else 22.
    """
    th = transfer_host(profile)
    if not th:
        return profile.host, profile.port
    if profile.transfer_port is not None:
        return th, profile.transfer_port
    if discovered_port is not None:
        return th, discovered_port
    return th, 22


def poll_settings(profile: ClusterProfile) -> dict[str, int]:
    """``{"base_s", "min_s", "max_s"}`` with defaults applied."""
    merged = dict(DEFAULT_POLL)
    merged.update(profile.poll or {})
    return merged


def target_override(profile: ClusterProfile, target_key: str) -> dict:
    """Merged override for a target key: every ``target_overrides`` glob that matches, in declaration order."""
    import fnmatch
    out: dict = {}
    for glob, ov in profile.target_overrides.items():
        if fnmatch.fnmatchcase(target_key, glob):
            out.update(ov)
    return out


# --- load / save -----------------------------------------------------------------------------------

_KNOWN_FIELDS: frozenset[str] = frozenset(f.name for f in fields(ClusterProfile))


def profile_from_dict(name: str, d: dict) -> ClusterProfile:
    """Build a profile from a JSON object; unknown keys are logged and kept in ``extra['_unknown']``."""
    d = dict(d)
    d.setdefault("name", name)
    unknown = {k: d.pop(k) for k in list(d) if k not in _KNOWN_FIELDS}
    if unknown:
        log.warning("config: cluster %r has unknown keys %s (ignored)", name, sorted(unknown))
        extra = dict(d.get("extra") or {})
        extra.setdefault("_unknown", {}).update(unknown)
        d["extra"] = extra
    return ClusterProfile(**d)


def _ensure_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not KNOWN_HOSTS_PATH.exists():
        KNOWN_HOSTS_PATH.touch()


def load_profiles(path: Path | None = None) -> dict[str, ClusterProfile]:
    _ensure_dir()
    cfg = path or CONFIG_PATH
    if not cfg.exists():
        return {}
    raw = json.loads(cfg.read_text(encoding="utf-8") or "{}")
    out: dict[str, ClusterProfile] = {}
    for name, d in raw.get("clusters", {}).items():
        out[name] = profile_from_dict(name, d)
    return out


def save_profiles(profiles: dict[str, ClusterProfile], path: Path | None = None) -> None:
    _ensure_dir()
    cfg = path or CONFIG_PATH
    data = {"clusters": {n: asdict(p) for n, p in profiles.items()}}
    tmp = cfg.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(cfg)


def get_profile(name: str) -> ClusterProfile:
    profiles = load_profiles()
    if name not in profiles:
        known = ", ".join(sorted(profiles)) or "(none configured)"
        raise KeyError(f"unknown cluster {name!r}; known clusters: {known}")
    return profiles[name]


__all__ = ["CONFIG_DIR", "CONFIG_PATH", "KNOWN_HOSTS_PATH", "AUTH_METHODS", "DEFAULT_POLL",
           "DEFAULT_SSH_MAX_EXEC", "TARGET_OVERRIDE_KEYS", "ClusterProfile", "control_root", "transfer_host",
           "has_transfer_host", "transfer_endpoint", "poll_settings", "target_override", "profile_from_dict",
           "load_profiles", "save_profiles", "get_profile"]
