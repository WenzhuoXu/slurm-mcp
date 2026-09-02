"""Parsers for every SLURM output the server reads (design sections 6.0-6.3; golden-tested on the
captured fixtures under ``tests/fixtures``).

Conventions: every parser is a pure function over the *lines* of one ``::SECTION`` (or the raw text
of a single-command exec); timestamps are returned by :func:`parse_ts` as an ``int`` epoch when the
cluster honours ``SLURM_TIME_FORMAT=%s``, as the ISO string otherwise (``ClusterClock.to_epoch``
converts it) and ``None`` for the sentinels of section 6.0. Durations are seconds
(``clock.parse_duration``); ``TimelimitRaw`` is minutes and reported as both.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from slurm_mcp.clock import SENTINELS, parse_duration
from slurm_mcp.slurm.states import TERMINAL, JobState, map_slurm_state

# ---------------------------------------------------------------------------------------------------
# framing (section 6.0)
# ---------------------------------------------------------------------------------------------------


class IncompleteProbe(RuntimeError):
    """Raised when a framed probe lacks its ``::END`` marker (truncated or timed-out output)."""


_SECTION_RE = re.compile(r"^::([A-Z][A-Z0-9_]*)$")
_RC_RE = re.compile(r"^::RC\s+(-?\d+)\s*$")
_NOW_RE = re.compile(r"^::NOW\s+(\d+)\s+(\S+)\s*$")

Section = tuple[list[str], "int | None"]


def parse_sections(text: str) -> dict[str, tuple[list[str], int | None]]:
    """Split a framed probe into ``{section: (lines, rc)}``.

    Honours ``::SECTION`` headers, ``::RC n`` (rc of the section; ``None`` when absent), ``::NOW <epoch>
    <host>`` (stored as section ``NOW`` with the single line ``"<epoch> <host>"``) and ``::END``.
    ``::L ...`` lines are data (section 6.2 ``::ENRICH``). Lines before the first header land in ``_pre``.
    Repeated sections (``::SACCT`` chunks) are concatenated; their rc is the first non-zero one.
    Empty lines are dropped; a missing ``::END`` raises :class:`IncompleteProbe`.
    """
    out: dict[str, tuple[list[str], int | None]] = {}
    current = "_pre"
    lines: list[str] = []
    rc: int | None = None
    ended = False

    def flush() -> None:
        if current == "_pre" and not lines:
            return
        if current in out:
            old_lines, old_rc = out[current]
            merged_rc = old_rc if old_rc else rc
            out[current] = (old_lines + lines, merged_rc)
        else:
            out[current] = (lines, rc)

    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if line == "::END":
            ended = True
            break
        if line.startswith("::"):
            m = _RC_RE.match(line)
            if m:
                rc = int(m.group(1))
                continue
            m = _NOW_RE.match(line)
            if m:
                flush()
                current, lines, rc = "NOW", [f"{m.group(1)} {m.group(2)}"], None
                flush()
                current, lines, rc = "_pre", [], None
                continue
            m = _SECTION_RE.match(line)
            if m:
                flush()
                current, lines, rc = m.group(1), [], None
                continue
        if line.strip():
            lines.append(line)
    if not ended:
        raise IncompleteProbe("probe output has no ::END marker")
    flush()
    return out


def parse_now(sections: Mapping[str, tuple[list[str], int | None]]) -> tuple[int, str] | None:
    """``(epoch, host)`` from the ``::NOW`` line, or None."""
    now = sections.get("NOW")
    if not now or not now[0]:
        return None
    epoch, _, host = now[0][0].partition(" ")
    return int(epoch), host


# ---------------------------------------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------------------------------------

_MEM_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMGTPkmgtp]?)(?:i?B)?$")
_UNITS = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4, "P": 1024 ** 5}


def is_sentinel(value: object) -> bool:
    return value is None or str(value).strip() in SENTINELS


def none_if_sentinel(value: object) -> str | None:
    """The stripped string, or None for every section 6.0 sentinel / empty value."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text in SENTINELS else text


def parse_ts(value: object) -> int | str | None:
    """SLURM timestamp field: ``int`` epoch when all digits, ISO string as is, None for sentinels."""
    text = none_if_sentinel(value)
    if text is None:
        return None
    if text.isdigit():
        return int(text)
    return text


def parse_int(value: object) -> int | None:
    """Int or None (sentinels, empty, non-numeric)."""
    text = none_if_sentinel(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return None


def parse_float(value: object) -> float | None:
    text = none_if_sentinel(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_secs(value: object) -> int | None:
    """``N sec`` / ``N min`` / ``N usec`` config values and plain SLURM durations -> seconds (None if unknown)."""
    text = none_if_sentinel(value)
    if text is None:
        return None
    m = re.match(r"^(\d+)\s*(sec|min|usec|s)?$", text)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "min":
            return n * 60
        if unit == "usec":
            return n // 1_000_000
        return n
    return parse_duration(text)


def parse_exit_code(value: object) -> tuple[int, int] | None:
    """``rc:sig`` -> ``(rc, sig)``; None when empty/sentinel."""
    text = none_if_sentinel(value)
    if text is None:
        return None
    rc, sep, sig = text.partition(":")
    if not rc.isdigit():
        return None
    return int(rc), int(sig) if sig.isdigit() else 0


def mem_to_bytes(value: object) -> int | None:
    """``56459172K`` / ``1.5G`` / ``512G`` / ``2048M`` -> bytes; bare digits are bytes; None for sentinels."""
    text = none_if_sentinel(value)
    if text is None:
        return None
    m = _MEM_RE.match(text)
    if not m:
        return None
    number, unit = m.groups()
    return int(float(number) * _UNITS[unit.upper()])


def req_mem_bytes(value: object, cpus: int | None = None) -> tuple[int | None, str | None]:
    """sacct ``ReqMem`` -> ``(bytes, per)``: ``80Gn`` per node, ``8048Mc`` per core (times ``cpus`` when known),
    plain ``128G`` per node (22.05 allocation rows)."""
    text = none_if_sentinel(value)
    if text is None:
        return None, None
    per: str | None = None
    if text[-1] in "nc":
        per = "node" if text[-1] == "n" else "cpu"
        text = text[:-1]
    b = mem_to_bytes(text)
    if b is None:
        return None, per
    if per == "cpu" and cpus:
        return b * cpus, per
    return b, per


def parse_tres(value: object) -> dict[str, Any]:
    """``billing=64,cpu=64,gres/gpu:h100-80=2,mem=512G,node=1`` -> dict (ints where numeric, else the string)."""
    text = none_if_sentinel(value)
    if not text:
        return {}
    out: dict[str, Any] = {}
    for item in text.split(","):
        key, sep, val = item.partition("=")
        key = key.strip()
        if not key:
            continue
        if not sep:
            out[key] = True
            continue
        val = val.strip()
        out[key] = int(val) if val.isdigit() else val
    return out


def gres_types_from_tres(tres: Mapping[str, Any]) -> dict[str, int]:
    """``{type: count}`` from the ``gres/gpu:<type>=N`` keys of a TRES dict."""
    return {k.split(":", 1)[1]: int(v) for k, v in tres.items()
            if k.startswith("gres/gpu:") and isinstance(v, int)}


def parse_list(value: object) -> list[str]:
    """Comma list -> list of stripped names ([] for sentinels/empty)."""
    text = none_if_sentinel(value)
    if not text:
        return []
    return [p.strip() for p in text.split(",") if p.strip()]


_GRES_PAREN_RE = re.compile(r"\([^)]*\)")


def gres_spec(value: object) -> dict[str, Any] | None:
    """GPU request/inventory string -> ``{"type": str | None, "count": int}``; None when no GPU is named.

    Accepted (section 6.1/6.2): ``gres:gpu:h100-80:1``, ``gres:gpu:1``, ``gres:gpu:h100-80`` (count 1),
    ``gres/gpu:h100-80=2``, ``gres/gpu=2``, ``gpu:v100-32:16(S:0-95)``, ``gpu:a40:1``, ``gpu:8``, ``gpu``;
    ``(null)``, ``N/A``, empty and non-gpu gres -> None. Comma lists take the first gpu entry.
    """
    text = none_if_sentinel(value)
    if not text:
        return None
    for item in text.split(","):
        item = _GRES_PAREN_RE.sub("", item.strip())
        if not item:
            continue
        if item.startswith("gres:") or item.startswith("gres/"):
            item = item[5:]
        parts = re.split(r"[:=]", item)
        if not parts or parts[0] != "gpu":
            continue
        rest = [p for p in parts[1:] if p != ""]
        count = 1
        if rest and rest[-1].isdigit():
            count = int(rest.pop())
        gtype = rest[0] if rest else None
        return {"type": gtype, "count": count}
    return None


def strip_node_state(value: object) -> str:
    """``sinfo %t`` compact state without the ``*~#!%$@^-`` suffixes."""
    return str(value or "").strip().rstrip("*~#!%$@^-")


def parse_aiot(value: object) -> dict[str, int] | None:
    """``A/I/O/T`` CPU or node counts -> ``{"alloc","idle","other","total"}``."""
    text = none_if_sentinel(value)
    if text is None:
        return None
    parts = text.split("/")
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        return None
    a, i, o, t = (int(p) for p in parts)
    return {"alloc": a, "idle": i, "other": o, "total": t}


def split_fields(line: str, n: int) -> list[str] | None:
    """Split a ``|`` row into exactly ``n`` fields (the last one keeps embedded ``|``); None when short."""
    parts = line.rstrip("\r").split("|", n - 1)
    if len(parts) < n:
        return None
    return parts


# ---------------------------------------------------------------------------------------------------
# 6.1 discovery sections
# ---------------------------------------------------------------------------------------------------

def parse_env(lines: Sequence[str]) -> dict[str, Any]:
    """``::ENV`` -> ``home, user, hostname, project, scratch, local, remote_now, tz_offset_s, group``.

    Also accepts the ``KEY=VALUE`` lines of the older ``env.out`` capture (``HOME=``, ``USER=``,
    ``PROJECT=``, ``SHELL=``, ``SLURM=slurm 22.05.11``) -> ``home, user, project, shell, slurm_version``.
    """
    rows = [l for l in lines if l.strip()]
    if not rows:
        return {}
    if "|" in rows[0]:
        f = rows[0].split("|")
        f += [""] * (10 - len(f))
        primary = f[8] or None
        # field 9 is ``id -Gn`` (every group); older captures lack it, so fall back to the primary group
        groups = [g for g in (f[9] or "").split() if g] or ([primary] if primary else [])
        return {
            "home": f[0], "user": f[1], "hostname": f[2], "project": f[3] or None, "scratch": f[4] or None,
            "local": f[5] or None, "remote_now": parse_int(f[6]), "tz_offset_s": parse_tz_offset(f[7]),
            "group": primary, "groups": groups,
        }
    kv: dict[str, str] = {}
    for row in rows:
        k, sep, v = row.partition("=")
        if sep:
            kv[k.strip()] = v.strip()
    out: dict[str, Any] = {"home": kv.get("HOME"), "user": kv.get("USER"), "project": kv.get("PROJECT") or None,
                           "shell": kv.get("SHELL")}
    if "SLURM" in kv:
        out["slurm_version"] = parse_version([kv["SLURM"]])
    return out


def parse_tz_offset(value: object) -> int | None:
    """``-0400`` -> -14400 seconds; ``+0530`` -> 19800."""
    text = none_if_sentinel(value)
    if text is None:
        return None
    m = re.match(r"^([+-])(\d{2})(\d{2})$", text)
    if not m:
        return None
    sign = -1 if m.group(1) == "-" else 1
    return sign * (int(m.group(2)) * 3600 + int(m.group(3)) * 60)


def parse_version(lines: Sequence[str]) -> str | None:
    """``sinfo --version`` (``slurm 22.05.11``) -> ``22.05.11``."""
    for line in lines:
        m = re.search(r"(\d+\.\d+(?:\.\d+)?)", line)
        if m:
            return m.group(1)
    return None


_CONFIG_RE = re.compile(r"^(\w+(?:\[\d+\])?)\s*=\s*(.*)$")
_SEC_KEYS = ("MinJobAge", "KillWait", "MessageTimeout", "GraceTime")


def parse_config(lines: Sequence[str]) -> dict[str, Any]:
    """``scontrol show config`` (full or grep-filtered) -> derived caps plus ``raw`` (section 6.1).

    Derived keys: ``cluster_name, slurm_version, epoch_format`` (``BOOT_TIME`` all digits),
    ``min_job_age_s, kill_wait_s, message_timeout_s, grace_time_s`` (``N sec``), ``cmd_timeout_s``
    (``max(120, MessageTimeout + 60)``; the profile override is applied by the caller), ``job_requeue``,
    ``preempt_mode`` (list), ``preempt_type, preempt_exempt_time_s, scheduler_parameters`` (dict; bare
    flags map to True), ``comment_stored`` (``AccountingStoreFlags`` has ``job_comment``),
    ``accounting_storage_enforce`` (list), ``enforce_part_limits, def_mem_per_cpu, def_mem_per_node,
    max_mem_per_cpu`` (MB or None), ``max_array_size, max_job_count, mail_prog, priority_weights``.
    """
    raw: dict[str, str] = {}
    for line in lines:
        m = _CONFIG_RE.match(line.strip())
        if m:
            raw[m.group(1)] = m.group(2).strip()
    boot = raw.get("BOOT_TIME", "")
    message_timeout = parse_secs(raw.get("MessageTimeout"))
    sched: dict[str, Any] = {}
    for item in parse_list(raw.get("SchedulerParameters")):
        k, sep, v = item.partition("=")
        sched[k] = (int(v) if v.isdigit() else v) if sep else True
    weights = {name.lower(): parse_int(raw.get(f"PriorityWeight{name}"))
               for name in ("Age", "FairShare", "QOS", "Partition", "JobSize")}
    return {
        "raw": raw,
        "cluster_name": raw.get("ClusterName"),
        "slurm_version": raw.get("SLURM_VERSION"),
        "epoch_format": bool(boot) and boot.isdigit(),
        "boot_time": parse_ts(boot),
        "min_job_age_s": parse_secs(raw.get("MinJobAge")),
        "kill_wait_s": parse_secs(raw.get("KillWait")),
        "message_timeout_s": message_timeout,
        "grace_time_s": parse_secs(raw.get("GraceTime")),
        "cmd_timeout_s": max(120, (message_timeout or 0) + 60),
        "job_requeue": (raw["JobRequeue"] == "1") if raw.get("JobRequeue") in ("0", "1") else None,
        "preempt_mode": parse_list(raw.get("PreemptMode")),
        "preempt_type": raw.get("PreemptType"),
        "preempt_exempt_time_s": parse_secs(raw.get("PreemptExemptTime")),
        "preempt_parameters": parse_list(raw.get("PreemptParameters")),
        "scheduler_parameters": sched,
        "comment_stored": "job_comment" in parse_list(raw.get("AccountingStoreFlags")),
        "accounting_storage_enforce": parse_list(raw.get("AccountingStorageEnforce")),
        "enforce_part_limits": raw.get("EnforcePartLimits"),
        "def_mem_per_cpu": parse_int(raw.get("DefMemPerCPU")),
        "def_mem_per_node": parse_int(raw.get("DefMemPerNode")),
        "max_mem_per_cpu": parse_int(raw.get("MaxMemPerCPU")),
        "max_array_size": parse_int(raw.get("MaxArraySize")),
        "max_job_count": parse_int(raw.get("MaxJobCount")),
        "mail_prog": none_if_sentinel(raw.get("MailProg")),
        "priority_weights": weights,
    }


def _kv_tokens(line: str) -> dict[str, str]:
    """``Key=Value`` tokens split on single spaces (partition/reservation lines; values have no spaces)."""
    out: dict[str, str] = {}
    for tok in line.strip().split(" "):
        if not tok:
            continue
        k, sep, v = tok.partition("=")
        if sep:
            out[k] = v
    return out


def parse_partitions(lines: Sequence[str]) -> dict[str, dict[str, Any]]:
    """``scontrol show partition -o`` -> ``{name: partition}`` (section 6.1 ``::PARTITIONS``).

    Each partition carries ``raw`` (all tokens) and typed keys: ``allow_groups, allow_accounts, allow_qos``
    (lists; ``["ALL"]``), ``default`` (bool), ``qos`` (None for N/A), ``default_time_s, grace_time_s,
    max_time_s`` (None = unlimited), ``max_nodes`` (None = unlimited), ``priority_tier, over_subscribe,
    preempt_mode`` (list), ``state, total_nodes, total_cpus, def_mem_per_cpu, def_mem_per_node,
    max_mem_per_node, max_mem_per_cpu`` (MB or None), ``job_defaults`` (dict, e.g. DefMemPerGPU),
    ``tres`` (dict), ``gres_types`` (``{type: count}``), ``gpu_total`` (``gres/gpu``), ``has_gpu``,
    ``tres_billing_weights`` (dict), ``nodes``.
    """
    out: dict[str, dict[str, Any]] = {}
    for line in lines:
        raw = _kv_tokens(line)
        name = raw.get("PartitionName")
        if not name:
            continue
        tres = parse_tres(raw.get("TRES"))
        gres_types = gres_types_from_tres(tres)
        gpu_total = tres.get("gres/gpu") if isinstance(tres.get("gres/gpu"), int) else None
        job_defaults = parse_tres(raw.get("JobDefaults"))
        out[name] = {
            "name": name,
            "raw": raw,
            "allow_groups": parse_list(raw.get("AllowGroups")),
            "allow_accounts": parse_list(raw.get("AllowAccounts")),
            "allow_qos": parse_list(raw.get("AllowQos")),
            "default": raw.get("Default", "NO").upper() == "YES",
            "qos": none_if_sentinel(raw.get("QoS")),
            "default_time_s": parse_duration(none_if_sentinel(raw.get("DefaultTime"))),
            "grace_time_s": parse_secs(raw.get("GraceTime")),
            "max_time_s": parse_duration(none_if_sentinel(raw.get("MaxTime"))),
            "max_nodes": parse_int(raw.get("MaxNodes")),
            "min_nodes": parse_int(raw.get("MinNodes")),
            "priority_tier": parse_int(raw.get("PriorityTier")),
            "priority_job_factor": parse_int(raw.get("PriorityJobFactor")),
            "over_subscribe": none_if_sentinel(raw.get("OverSubscribe")),
            "preempt_mode": parse_list(raw.get("PreemptMode")),
            "state": raw.get("State"),
            "total_nodes": parse_int(raw.get("TotalNodes")),
            "total_cpus": parse_int(raw.get("TotalCPUs")),
            "def_mem_per_cpu": parse_int(raw.get("DefMemPerCPU")),
            "def_mem_per_node": parse_int(raw.get("DefMemPerNode")),
            "max_mem_per_node": parse_int(raw.get("MaxMemPerNode")),
            "max_mem_per_cpu": parse_int(raw.get("MaxMemPerCPU")),
            "job_defaults": job_defaults,
            "tres": tres,
            "gres_types": gres_types,
            "gpu_total": gpu_total,
            "has_gpu": gpu_total is not None and gpu_total > 0,
            "tres_billing_weights": parse_tres(raw.get("TRESBillingWeights")),
            "nodes": none_if_sentinel(raw.get("Nodes")),
        }
    return out


def parse_sinfo_nodes(lines: Sequence[str]) -> list[dict[str, Any]]:
    """``sinfo -h -e -N -o '%N|%R|%t|%c|%m|%G|%f'`` rows (section 6.1 ``::SINFO``); one row per node per partition.

    The 10-field capture ``%N|%P|%t|%c|%m|%G|%f|%C|%e|%O`` is accepted too (``%P`` ``*`` stripped; the
    extra columns land in ``cpus_aiot``, ``free_mem_mb``, ``cpu_load``).
    Row: ``node, partition, state`` (suffix stripped), ``state_raw, responding, cpus, mem_mb, gres``
    (:func:`gres_spec` or None), ``gres_raw, features`` (list).
    """
    out: list[dict[str, Any]] = []
    for line in lines:
        f = line.rstrip("\r").split("|")
        if len(f) < 7:
            continue
        state_raw = f[2].strip()
        row: dict[str, Any] = {
            "node": f[0].strip(), "partition": f[1].strip().rstrip("*"), "state": strip_node_state(state_raw),
            "state_raw": state_raw, "responding": "*" not in state_raw, "cpus": parse_int(f[3]),
            "mem_mb": parse_int(f[4]), "gres": gres_spec(f[5]), "gres_raw": none_if_sentinel(f[5]),
            "features": parse_list(f[6]),
        }
        if len(f) >= 10:
            row["cpus_aiot"] = parse_aiot(f[7])
            row["free_mem_mb"] = parse_int(f[8])
            row["cpu_load"] = parse_float(f[9])
        out.append(row)
    return out


def aggregate_sinfo(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-partition aggregate of :func:`parse_sinfo_nodes` rows (section 6.1): ``nodes`` (count),
    ``states`` (``{state: n}``), ``gres`` (``{type: {"nodes": n, "per_node": max count}}``; untyped GPUs
    under key ``None``), ``max_cpus``, ``max_mem_mb``."""
    out: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    for r in rows:
        key = (r["node"], r["partition"])
        if key in seen:
            continue
        seen.add(key)
        agg = out.setdefault(r["partition"], {"nodes": 0, "states": {}, "gres": {}, "max_cpus": 0, "max_mem_mb": 0})
        agg["nodes"] += 1
        agg["states"][r["state"]] = agg["states"].get(r["state"], 0) + 1
        agg["max_cpus"] = max(agg["max_cpus"], r.get("cpus") or 0)
        agg["max_mem_mb"] = max(agg["max_mem_mb"], r.get("mem_mb") or 0)
        g = r.get("gres")
        if g:
            entry = agg["gres"].setdefault(g["type"], {"nodes": 0, "per_node": 0})
            entry["nodes"] += 1
            entry["per_node"] = max(entry["per_node"], g["count"])
    return out


def merge_partition_gres(partitions: Mapping[str, Mapping[str, Any]],
                         sinfo_agg: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Union the ``::SINFO`` gres inventory into the ``::PARTITIONS`` table (section 6.1, 6.2).

    ``partitions`` is :func:`parse_partitions` output, ``sinfo_agg`` is :func:`aggregate_sinfo` output. The
    partition ``TRES`` string is not a complete type inventory: Bridges-2 ``GPU-shared`` lists
    ``l40s-48/v100-16/v100-32`` but not the 10 ``h100-80`` nodes, and TRACE ``batch`` only carries the
    untyped ``gres/gpu=29``. Returns a **new** dict (inputs untouched) where every partition gets
    ``gres_types`` = TRES types + sinfo types (count = TRES count, else ``nodes x per_node`` from sinfo),
    ``gres_nodes`` = ``{type: {"nodes", "per_node"}}`` from sinfo (untyped ``gpu:N`` nodes under key None),
    ``gpu_total`` = TRES ``gres/gpu`` or the sinfo sum, ``has_gpu`` = any of those. Partitions seen only
    in sinfo are ignored (they cannot be submitted to without their ``scontrol`` record). This merged
    table is what :func:`classify_demand` and the placer (section 8) must be handed.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, part in partitions.items():
        merged = dict(part)
        gres_types = dict(part.get("gres_types") or {})
        sinfo_gres = dict((sinfo_agg.get(name) or {}).get("gres") or {})
        sinfo_total = 0
        for gtype, entry in sinfo_gres.items():
            per_partition = int(entry.get("nodes", 0)) * int(entry.get("per_node", 0))
            sinfo_total += per_partition
            if gtype is not None and gtype not in gres_types:
                gres_types[gtype] = per_partition
        gpu_total = part.get("gpu_total")
        if gpu_total is None and sinfo_total:
            gpu_total = sinfo_total
        merged["gres_types"] = gres_types
        merged["gres_nodes"] = sinfo_gres
        merged["gpu_total"] = gpu_total
        merged["has_gpu"] = bool(gres_types) or bool(sinfo_gres) or bool(gpu_total)
        out[name] = merged
    return out


def parse_user(lines: Sequence[str]) -> dict[str, Any]:
    """``sacctmgr -nP show user ... format=User,DefaultAccount`` -> ``{user, default_account}``."""
    for line in lines:
        f = line.split("|")
        if len(f) >= 2:
            return {"user": f[0].strip(), "default_account": none_if_sentinel(f[1])}
    return {"user": None, "default_account": None}


def parse_assoc(lines: Sequence[str]) -> list[dict[str, Any]]:
    """``::ASSOC`` rows (11 fields) -> ``account, partition`` (None = any), ``qos_list, default_qos, grp_tres,
    grp_tres_mins, max_jobs, max_submit, max_tres, max_wall_s``. The 7-field capture
    (``cluster,account,partition,qos,maxjobs,maxsubmit,grptres``) is accepted too.
    """
    out: list[dict[str, Any]] = []
    for line in lines:
        f = line.rstrip("\r").split("|")
        if len(f) >= 11:
            cluster, account, partition, qos, dqos, gtres, gmins, mj, ms, mtres, mwall = f[:11]
        elif len(f) >= 7:
            cluster, account, partition, qos, mj, ms, gtres = f[:7]
            dqos = gmins = mtres = mwall = ""
        else:
            continue
        out.append({
            "cluster": cluster, "account": account, "partition": partition or None, "qos_list": parse_list(qos),
            "default_qos": none_if_sentinel(dqos), "grp_tres": parse_tres(gtres), "grp_tres_mins": parse_tres(gmins),
            "max_jobs": parse_int(mj), "max_submit": parse_int(ms), "max_tres": parse_tres(mtres),
            "max_wall_s": parse_duration(none_if_sentinel(mwall)),
        })
    return out


def parse_qos(lines: Sequence[str]) -> dict[str, dict[str, Any]]:
    """``::QOS`` rows (13 fields ``Name,Priority,GraceTime,MaxWall,MaxTRES,MaxTRESPU,MaxJobsPU,MaxSubmitPU,
    GrpTRES,Preempt,PreemptMode,Flags,UsageFactor``) -> ``{name: qos}``. The 9-field capture
    (``name,priority,maxwall,maxtrespu,maxjobspu,maxsubmitpu,grptres,maxtres,flags``) is accepted too.
    Empty field = no limit (None / {} / [])."""
    out: dict[str, dict[str, Any]] = {}
    for line in lines:
        f = line.rstrip("\r").split("|")
        if len(f) >= 13:
            (name, prio, grace, maxwall, maxtres, maxtrespu, maxjobspu, maxsubmitpu, grptres, preempt, pmode, flags,
             usage) = f[:13]
        elif len(f) >= 9:
            name, prio, maxwall, maxtrespu, maxjobspu, maxsubmitpu, grptres, maxtres, flags = f[:9]
            grace = preempt = pmode = usage = ""
        else:
            continue
        if not name:
            continue
        out[name] = {
            "name": name, "priority": parse_int(prio) or 0, "grace_time_s": parse_secs(grace),
            "max_wall_s": parse_duration(none_if_sentinel(maxwall)), "max_tres": parse_tres(maxtres),
            "max_tres_pu": parse_tres(maxtrespu), "max_jobs_pu": parse_int(maxjobspu),
            "max_submit_pu": parse_int(maxsubmitpu), "grp_tres": parse_tres(grptres), "preempt": parse_list(preempt),
            "preempt_mode": none_if_sentinel(pmode), "flags": parse_list(flags), "usage_factor": parse_float(usage),
        }
    return out


def parse_sshare(lines: Sequence[str]) -> list[dict[str, Any]]:
    """``sshare -nP -U -u ... -o Account,User,FairShare,GrpTRESMins,GrpTRESRaw`` -> rows with ``account, user,
    fair_share, grp_tres_mins, grp_tres_raw, su_balance`` (``(billing limit - billing used) / 60`` or None).
    A header line (``Account|User|...``, the ``sshare -U -P`` capture) is honoured for column names."""
    rows = [l.rstrip("\r") for l in lines if l.strip()]
    if not rows:
        return []
    names = ["account", "user", "fairshare", "grptresmins", "grptresraw"]
    if rows[0].lower().startswith("account|"):
        names = [c.strip().lower() for c in rows[0].split("|")]
        rows = rows[1:]
    out: list[dict[str, Any]] = []
    for row in rows:
        f = row.split("|")
        rec = {names[i]: f[i] for i in range(min(len(names), len(f)))}
        mins = parse_tres(rec.get("grptresmins"))
        used = parse_tres(rec.get("grptresraw"))
        balance = None
        if isinstance(mins.get("billing"), int) and isinstance(used.get("billing"), int):
            balance = (mins["billing"] - used["billing"]) / 60
        out.append({
            "account": rec.get("account"), "user": rec.get("user"), "fair_share": parse_float(rec.get("fairshare")),
            "raw_usage": parse_int(rec.get("rawusage")), "norm_shares": parse_float(rec.get("normshares")),
            "grp_tres_mins": mins, "grp_tres_raw": used, "su_balance": balance,
        })
    return out


def parse_balance(lines: Sequence[str], regex: str | None) -> dict[str, float] | None:
    """``::BALANCE`` + ``profile.balance_regex`` (``left``/``total`` groups, commas stripped) -> ``{left, total}``."""
    if not regex:
        return None
    text = "\n".join(lines)
    m = re.search(regex, text)
    if not m:
        return None
    out: dict[str, float] = {}
    for name in ("left", "total"):
        try:
            val = m.group(name)
        except IndexError:
            val = None
        if val is not None:
            num = parse_float(val.replace(",", ""))
            if num is not None:
                out[name] = num
    return out or None


_RESV_KEYS = ("ReservationName", "StartTime", "EndTime", "Duration", "Nodes", "NodeCnt", "CoreCnt", "Features",
              "PartitionName", "Flags", "TRES", "Users", "Groups", "Accounts", "Licenses", "State", "BurstBuffer",
              "Watts", "MaxStartDelay", "Comment")


def parse_reservations(lines: Sequence[str]) -> list[dict[str, Any]]:
    """``scontrol -o show reservation`` -> rows with ``name, start, end, nodes, partition, flags`` (list),
    ``maintenance`` (``MAINT`` flag or name matching ``/maint/i``), ``users, accounts, raw``."""
    out: list[dict[str, Any]] = []
    for line in lines:
        raw = _kv_regex(line, _RESV_KEYS)
        name = raw.get("ReservationName")
        if not name:
            continue
        flags = parse_list(raw.get("Flags"))
        out.append({
            "name": name, "start": parse_ts(raw.get("StartTime")), "end": parse_ts(raw.get("EndTime")),
            "nodes": none_if_sentinel(raw.get("Nodes")), "node_count": parse_int(raw.get("NodeCnt")),
            "partition": none_if_sentinel(raw.get("PartitionName")), "flags": flags,
            "maintenance": "MAINT" in flags or re.search(r"maint", name, re.I) is not None,
            "users": parse_list(raw.get("Users")), "accounts": parse_list(raw.get("Accounts")), "raw": raw,
        })
    return out


def parse_tools(lines: Sequence[str]) -> dict[str, bool]:
    """``tool=1|0`` (design ``::TOOLS``) or ``tool=/path|-`` (capture) -> ``{tool: present}``."""
    out: dict[str, bool] = {}
    for line in lines:
        k, sep, v = line.strip().partition("=")
        if not sep or not k:
            continue
        v = v.strip()
        out[k] = v not in ("", "0", "-")
    return out


def parse_cap_o(lines: Sequence[str]) -> bool:
    """``::CAP_O`` ``rc=0`` -> True (the ``-O ...:0|`` form and ``tres-per-job`` work)."""
    return any(l.strip() == "rc=0" for l in lines)


_DF_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)([KMGTPkmgtp]?)i?$")


def _df_kb(value: str) -> int | None:
    m = _DF_SIZE_RE.match(value.strip())
    if not m:
        return None
    number, unit = m.groups()
    if not unit:
        return int(float(number))
    return int(float(number) * _UNITS[unit.upper()] / 1024)


def parse_df(lines: Sequence[str], roles: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """``::DF`` rows (``df -Pk`` line + the queried path appended) -> ``[{path, mount, kb_total, kb_used, kb_free,
    used_pct, role, paths}]``. ``df -hP`` captures without the appended path are accepted (``path`` = mount).

    De-duplication is by ``(mount, kb_total, kb_used)``, **not** by mount alone: a filesystem with per-directory
    quotas reports different totals for different paths of the same mount. Measured on TRACE (VAST over NFS,
    2026-09-02): ``df -Pk $HOME`` -> ``172.19.21.14:/trace 1031755399168 ... 1% /trace`` while
    ``df -Pk /trace/group/biosimmlab`` -> ``172.19.21.14:/trace 3173828608 ... 74% /trace``. Collapsing those two
    into one row hid the 3 TB group quota (the one every upload must be checked against) behind a 932 TB
    filesystem view, so the quota guard of section 5.5 would always pass.
    """
    out: list[dict[str, Any]] = []
    seen: dict[tuple[str, int | None, int | None], dict[str, Any]] = {}
    for line in lines:
        f = line.split()
        if len(f) < 6 or not f[4].endswith("%"):
            continue
        mount = f[5]
        path = f[6] if len(f) >= 7 else mount
        kb_total, kb_used = _df_kb(f[1]), _df_kb(f[2])
        key = (mount, kb_total, kb_used)
        if key in seen:
            seen[key]["paths"].append(path)
            if seen[key]["role"] is None:
                seen[key]["role"] = (roles or {}).get(path)
            continue
        row = {
            "path": path, "mount": mount, "filesystem": f[0], "kb_total": kb_total, "kb_used": kb_used,
            "kb_free": _df_kb(f[3]), "used_pct": parse_int(f[4].rstrip("%")),
            "role": (roles or {}).get(path), "paths": [path],
        }
        seen[key] = row
        out.append(row)
    return out


def df_row_for_path(rows: Sequence[Mapping[str, Any]], path: str) -> dict[str, Any] | None:
    """The ``::DF`` row governing ``path``: the one whose queried paths contain the longest prefix of ``path``.

    Used by the quota guard (section 5.5) and ``quota_warning``: with per-directory quotas several rows share a
    mount, so the *destination* path decides which row applies, never the mount.
    """
    best: dict[str, Any] | None = None
    best_len = -1
    for row in rows:
        for candidate in row.get("paths") or [row.get("path")]:
            if not candidate:
                continue
            c = candidate.rstrip("/")
            if (path == c or path.startswith(c + "/")) and len(c) > best_len:
                best, best_len = dict(row), len(c)
    return best


# ---------------------------------------------------------------------------------------------------
# 6.2 tick sections
# ---------------------------------------------------------------------------------------------------

TICK_SQUEUE_CODES = "%A|%i|%F|%K|%T|%P|%q|%S|%e|%V|%l|%M|%Q|%N|%b|%k|%o|%Z|%r"

# squeue % code -> (key, converter)
_SQUEUE_CODES: dict[str, tuple[str, Any]] = {
    "%A": ("slurm_id", parse_int), "%i": ("display_id", str.strip), "%F": ("array_job_id", parse_int),
    "%K": ("array_index", parse_int), "%T": ("state", str.strip), "%t": ("state_compact", str.strip),
    "%P": ("partition", str.strip), "%q": ("qos", none_if_sentinel), "%S": ("start", parse_ts),
    "%e": ("end", parse_ts), "%V": ("submit", parse_ts), "%l": ("time_limit", str.strip),
    "%M": ("elapsed", str.strip), "%L": ("time_left", str.strip), "%Q": ("priority", parse_int),
    "%N": ("nodes", none_if_sentinel), "%b": ("tres_per_node", str.strip), "%k": ("comment", none_if_sentinel),
    "%o": ("command", str.strip), "%Z": ("workdir", str.strip), "%r": ("reason", str.strip),
    "%R": ("reason_or_nodes", str.strip), "%j": ("name", str.strip), "%D": ("num_nodes", parse_int),
    "%C": ("num_cpus", parse_int), "%u": ("user", str.strip), "%a": ("account", none_if_sentinel),
    "%Y": ("sched_nodes", none_if_sentinel), "%E": ("dependency", none_if_sentinel), "%B": ("batch_host", none_if_sentinel),
}


def parse_squeue_rows(lines: Sequence[str], fmt: str = TICK_SQUEUE_CODES) -> list[dict[str, Any]]:
    """Generic ``squeue -h -o '<fmt>'`` parser: ``fmt`` is the ``|``-joined % code list.

    Derived keys: ``job_state`` (mapped from ``%T``), ``partitions`` (``%P`` comma list), ``time_limit_s``,
    ``elapsed_s``, ``gres`` (:func:`gres_spec` of ``%b``), ``reason`` (parentheses of ``%R`` stripped when
    ``%r`` is absent), ``array_index`` None for ``N/A``.
    """
    codes = fmt.split("|")
    n = len(codes)
    out: list[dict[str, Any]] = []
    for line in lines:
        f = split_fields(line, n)
        if f is None:
            continue
        row: dict[str, Any] = {}
        for code, value in zip(codes, f):
            spec = _SQUEUE_CODES.get(code)
            if spec is None:
                row[code] = value
                continue
            key, conv = spec
            row[key] = conv(value)
        if "state" in row:
            row["job_state"] = map_slurm_state(row["state"])
        if "partition" in row:
            row["partitions"] = parse_list(row["partition"])
        if "time_limit" in row:
            row["time_limit_s"] = parse_duration(row["time_limit"])
        if "elapsed" in row:
            row["elapsed_s"] = parse_duration(row["elapsed"])
        if "tres_per_node" in row:
            row["gres"] = gres_spec(row["tres_per_node"])
        if "reason" not in row and "reason_or_nodes" in row:
            ron = row["reason_or_nodes"]
            if ron.startswith("(") and ron.endswith(")"):
                row["reason"] = ron[1:-1]
            else:
                row["reason"] = None
                row.setdefault("nodes", none_if_sentinel(ron))
        out.append(row)
    return out


def parse_squeue_tick(lines: Sequence[str]) -> list[dict[str, Any]]:
    """The 19-field tick squeue (section 6.2): ``slurm_id, display_id, array_job_id, array_index, state,
    job_state, partition(s), qos, start, end, submit, time_limit(_s), elapsed(_s), priority, nodes,
    tres_per_node, gres, comment, command, workdir, reason``."""
    return parse_squeue_rows(lines, TICK_SQUEUE_CODES)


def parse_restarts(lines: Sequence[str]) -> dict[int, dict[str, int | None]]:
    """``::RESTARTS`` rows ``615427|0|1|`` -> ``{slurm_id: {"restarts", "requeue"}}``."""
    out: dict[int, dict[str, int | None]] = {}
    for line in lines:
        f = line.strip().rstrip("|").split("|")
        if len(f) < 2:
            continue
        sid = parse_int(f[0])
        if sid is None:
            continue
        out[sid] = {"restarts": parse_int(f[1]), "requeue": parse_int(f[2]) if len(f) > 2 else None}
    return out


TICK_SACCT_FIELDS: tuple[str, ...] = (
    "JobIDRaw", "JobID", "State", "ExitCode", "DerivedExitCode", "Partition", "QOS", "NodeList", "Submit",
    "Start", "End", "ElapsedRaw", "TimelimitRaw", "AllocTRES", "ReqTRES", "Reason", "WorkDir",
)
ENRICH_FIELDS: tuple[str, ...] = ("JobIDRaw", "JobID", "State", "ExitCode", "MaxRSS", "ReqMem", "ElapsedRaw",
                                  "AllocTRES")
RECOVER_FIELDS: tuple[str, ...] = ("JobIDRaw", "Submit", "State", "WorkDir", "SubmitLine")
BACKFILL_FIELDS: tuple[str, ...] = ("JobIDRaw", "Partition", "QOS", "ReqTRES", "Submit", "Start", "State")
# The 27-field capture format of tests/fixtures/*/sacct_*.out
FIXTURE_SACCT_FIELDS: tuple[str, ...] = (
    "JobID", "JobIDRaw", "JobName", "State", "ExitCode", "DerivedExitCode", "Elapsed", "ElapsedRaw", "Start", "End",
    "Submit", "Partition", "Account", "QOS", "NodeList", "AllocTRES", "ReqTRES", "MaxRSS", "TotalCPU", "Reason",
    "WorkDir", "Timelimit", "TimelimitRaw", "NCPUS", "NNodes", "Flags", "SubmitLine",
)

_SACCT_CONV: dict[str, tuple[str, Any]] = {
    "JobIDRaw": ("job_id_raw", str.strip), "JobID": ("job_id", str.strip), "JobName": ("job_name", str.strip),
    "State": ("state_raw", str.strip), "ExitCode": ("exit_code", parse_exit_code),
    "DerivedExitCode": ("derived_exit_code", parse_exit_code), "Elapsed": ("elapsed", str.strip),
    "ElapsedRaw": ("elapsed_s", parse_int), "Start": ("start", parse_ts), "End": ("end", parse_ts),
    "Submit": ("submit", parse_ts), "Partition": ("partition", none_if_sentinel), "Account": ("account", none_if_sentinel),
    "QOS": ("qos", none_if_sentinel), "NodeList": ("nodelist", none_if_sentinel), "AllocTRES": ("alloc_tres", parse_tres),
    "ReqTRES": ("req_tres", parse_tres), "MaxRSS": ("max_rss_bytes", mem_to_bytes), "TotalCPU": ("total_cpu", str.strip),
    "Reason": ("reason", str.strip), "WorkDir": ("workdir", none_if_sentinel), "Timelimit": ("timelimit", str.strip),
    "TimelimitRaw": ("timelimit_min", parse_int), "NCPUS": ("ncpus", parse_int), "NNodes": ("nnodes", parse_int),
    "Flags": ("flags", parse_list), "SubmitLine": ("submit_line", str.strip), "ReqMem": ("req_mem", none_if_sentinel),
}


def parse_sacct_rows(lines: Sequence[str], fields: Sequence[str] = TICK_SACCT_FIELDS) -> list[dict[str, Any]]:
    """Generic ``sacct -n -P -o <fields>`` parser (section 6.2 rules).

    Derived: ``slurm_id`` (numeric base of ``JobIDRaw``/``JobID``), ``step`` (``batch``/``extern``/``0``/None),
    ``state`` (first token), ``job_state`` (mapped), ``cancelled_by`` (uid after ``CANCELLED by``),
    ``array_job_id/array_index`` (``JobID`` ``123_4``), ``timelimit_s`` (``TimelimitRaw`` minutes x 60),
    ``req_mem_bytes`` (``ReqMem`` with per-core x allocated cpus). Short rows are skipped.
    """
    n = len(fields)
    out: list[dict[str, Any]] = []
    for line in lines:
        f = split_fields(line, n)
        if f is None:
            continue
        row: dict[str, Any] = {}
        for name, value in zip(fields, f):
            spec = _SACCT_CONV.get(name)
            if spec is None:
                row[name] = value
            else:
                key, conv = spec
                row[key] = conv(value)
        ident = row.get("job_id_raw") or row.get("job_id") or ""
        base, _, step = ident.partition(".")
        base = base.split("_", 1)[0].split("+", 1)[0]
        row["slurm_id"] = int(base) if base.isdigit() else None
        row["step"] = step or None
        if "state_raw" in row:
            state_raw = row["state_raw"]
            row["state"] = state_raw.split()[0] if state_raw else ""
            row["job_state"] = map_slurm_state(state_raw)
            m = re.search(r"\bby\s+(\S+)", state_raw)
            row["cancelled_by"] = m.group(1) if m else None
        if "job_id" in row:
            jid = row["job_id"].split(".", 1)[0]
            if "_" in jid:
                a, _, b = jid.partition("_")
                row["array_job_id"] = parse_int(a)
                row["array_index"] = parse_int(b) if b.isdigit() else None
            else:
                row["array_job_id"] = None
                row["array_index"] = None
        if "timelimit_min" in row:
            row["timelimit_s"] = row["timelimit_min"] * 60 if row["timelimit_min"] is not None else None
        if "req_mem" in row:
            cpus = row.get("alloc_tres", {}).get("cpu") if isinstance(row.get("alloc_tres"), dict) else None
            row["req_mem_bytes"], row["req_mem_per"] = req_mem_bytes(row["req_mem"], cpus if isinstance(cpus, int) else None)
        out.append(row)
    return out


def _end_key(row: Mapping[str, Any]) -> tuple[int, Any]:
    end = row.get("end")
    if isinstance(end, int):
        return (2, end)
    if isinstance(end, str):
        return (1, end)
    return (0, "")


def group_incarnations(rows: Iterable[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Group sacct rows by ``slurm_id`` applying the ``-D`` rule (section 6.2): ``current`` = the non-terminal
    allocation row if any, else the row with the latest ``End``; ``incarnations`` = number of allocation
    rows; step rows go to ``steps``."""
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        sid = row.get("slurm_id")
        if sid is None:
            continue
        g = out.setdefault(sid, {"rows": [], "steps": [], "current": None, "incarnations": 0})
        (g["steps"] if row.get("step") else g["rows"]).append(row)
    for g in out.values():
        rows_ = g["rows"]
        g["incarnations"] = len(rows_)
        if not rows_:
            continue
        live = [r for r in rows_ if r.get("job_state") is not None and r["job_state"] not in TERMINAL]
        if live:
            g["current"] = live[-1]
        else:
            best = rows_[0]
            for r in rows_[1:]:
                if _end_key(r) >= _end_key(best):
                    best = r
            g["current"] = best
    return out


def parse_sacct_tick(lines: Sequence[str]) -> dict[int, dict[str, Any]]:
    """``::SACCT`` (17 fields, ``-X -D``) -> ``{slurm_id: {rows, current, incarnations, steps}}``."""
    return group_incarnations(parse_sacct_rows(lines, TICK_SACCT_FIELDS))


def parse_files(lines: Sequence[str]) -> dict[str, dict[str, Any]]:
    """``::FILES`` lines ``<ctrl_dir>|<file>|<content>`` -> ``{ctrl_dir: {file: content}}``; ``status.json`` and
    ``progress.json`` are ``json.loads``-ed when valid (else the raw text is kept)."""
    out: dict[str, dict[str, Any]] = {}
    for line in lines:
        f = split_fields(line, 3)
        if f is None:
            continue
        d, name, content = f
        value: Any = content
        if name.endswith(".json"):
            try:
                value = json.loads(content)
            except ValueError:
                value = content
        out.setdefault(d, {})[name] = value
    return out


def parse_cmds(lines: Sequence[str]) -> dict[str, int | str]:
    """``::CMDS`` lines ``<rc path>|<rc>`` -> ``{path: rc}`` (int when numeric)."""
    out: dict[str, int | str] = {}
    for line in lines:
        f = split_fields(line, 2)
        if f is None:
            continue
        path, rc = f
        rc = rc.strip()
        out[path] = int(rc) if re.fullmatch(r"-?\d+", rc) else rc
    return out


def parse_recover(lines: Sequence[str]) -> list[dict[str, Any]]:
    """``::RECOVER`` rows (``JobIDRaw,Submit,State,WorkDir,SubmitLine``) -> ``slurm_id, submit, state, job_state,
    workdir, submit_line``."""
    return parse_sacct_rows(lines, RECOVER_FIELDS)


def parse_backfill(lines: Sequence[str]) -> list[dict[str, Any]]:
    """Back-fill rows (``JobIDRaw,Partition,QOS,ReqTRES,Submit,Start,State``); ``gres`` from ``ReqTRES``."""
    rows = parse_sacct_rows(lines, BACKFILL_FIELDS)
    for r in rows:
        types = gres_types_from_tres(r.get("req_tres", {}))
        gpu = r.get("req_tres", {}).get("gres/gpu")
        r["gres_type"] = next(iter(types), None)
        r["gpus"] = gpu if isinstance(gpu, int) else None
    return rows


def parse_enrich(lines: Sequence[str]) -> dict[str, Any]:
    """``::ENRICH`` -> ``{"jobs": {slurm_id: {max_rss_bytes, req_mem_bytes, alloc, steps}}, "last_lines": {path: text}}``.

    Step rows carry ``MaxRSS``; the job's ``max_rss_bytes`` is the maximum over its steps; ``req_mem_bytes``
    comes from the allocation row (``ReqMem``, per-core values multiplied by the allocated cpus).
    """
    data_lines = [l for l in lines if not l.startswith("::L ")]
    last: dict[str, str] = {}
    for l in lines:
        if l.startswith("::L "):
            path, _, text = l[4:].partition("|")
            last[path] = text
    jobs: dict[int, dict[str, Any]] = {}
    for row in parse_sacct_rows(data_lines, ENRICH_FIELDS):
        sid = row["slurm_id"]
        if sid is None:
            continue
        j = jobs.setdefault(sid, {"max_rss_bytes": None, "req_mem_bytes": None, "alloc": None, "steps": {}})
        if row["step"]:
            j["steps"][row["step"]] = row
        else:
            j["alloc"] = row
            if row.get("req_mem_bytes") is not None:
                j["req_mem_bytes"] = row["req_mem_bytes"]
        rss = row.get("max_rss_bytes")
        if rss is not None and (j["max_rss_bytes"] is None or rss > j["max_rss_bytes"]):
            j["max_rss_bytes"] = rss
        if j["req_mem_bytes"] is None and row.get("req_mem_bytes") is not None:
            j["req_mem_bytes"] = row["req_mem_bytes"]
    return {"jobs": jobs, "last_lines": last}


# ---------------------------------------------------------------------------------------------------
# 6.2 snapshot
# ---------------------------------------------------------------------------------------------------

DEMAND_FIELDS: tuple[str, ...] = ("partition", "tres_per_node", "tres_per_job")


UNIQ_PTB_FIELDS: tuple[str, ...] = ("partition", "state", "tres_per_node")


def _uniq_row_fields(rest: str, nfields: int) -> list[str]:
    """Auto-detect the column names of one ``uniq -c`` row (section 6.2).

    The ``-O '...:0|'`` form always leaves a trailing ``|`` (``partition|tres-per-node|tres-per-job|``); a 3-field
    row *without* it is the ``%P|%T|%b`` shape of the captured ``squeue_all_counts.out`` fixtures
    (``partition|state|tres_per_node``). Two fields are the ``%P|%b`` fallback.
    """
    if nfields == 3 and not rest.endswith("|"):
        return list(UNIQ_PTB_FIELDS)
    return list(DEMAND_FIELDS[:nfields])


def parse_uniq_rows(lines: Sequence[str], fields: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """``sort | uniq -c`` rows ``   3433 GPU-shared|gres:gpu:h100-80:1|N/A|`` -> ``{count, partition, tres_per_node,
    tres_per_job}``. Without ``fields`` the ``-O`` form (3 fields, trailing ``|``), the ``%P|%T|%b`` fixture form
    (3 fields, no trailing ``|`` -> ``partition, state, tres_per_node``) and the ``%P|%b`` fallback (2 fields;
    ``tres_per_job`` None = unknown) are auto-detected."""
    out: list[dict[str, Any]] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        count_s, _, rest = text.partition(" ")
        if not count_s.isdigit():
            continue
        rest = rest.strip()
        raw = rest
        if rest.endswith("|"):
            rest = rest[:-1]
        f = rest.split("|")
        names = list(fields) if fields else _uniq_row_fields(raw, len(f))
        row: dict[str, Any] = {"count": int(count_s)}
        for name, value in zip(names, f):
            row[name] = value.strip()
        for name in DEMAND_FIELDS:
            row.setdefault(name, None)
        out.append(row)
    return out


def parse_mine(lines: Sequence[str]) -> list[dict[str, Any]]:
    """``::MINE`` rows (``-O`` 7 fields or ``%A|%P|%b|%Q|%S|%r`` 6 fields) -> ``slurm_id, partition, tres_per_node,
    tres_per_job, priority, start, reason``."""
    out: list[dict[str, Any]] = []
    for line in lines:
        text = line.strip()
        if text.endswith("|"):
            text = text[:-1]
        f = text.split("|")
        if len(f) >= 7:
            sid, part, tpn, tpj, prio, start, reason = f[:7]
        elif len(f) >= 6:
            sid, part, tpn, prio, start, reason = f[:6]
            tpj = None
        else:
            continue
        out.append({
            "slurm_id": parse_int(sid), "partition": part, "partitions": parse_list(part), "tres_per_node": tpn,
            "tres_per_job": tpj, "priority": parse_int(prio), "start": parse_ts(start), "reason": reason.strip(),
        })
    return out


def parse_snapshot_nodes(lines: Sequence[str]) -> list[dict[str, Any]]:
    """``::NODES`` rows ``%R|%t|%G|%C`` -> ``partition, state, gres, gres_raw, cpus`` (A/I/O/T dict)."""
    out: list[dict[str, Any]] = []
    for line in lines:
        f = line.rstrip("\r").split("|")
        if len(f) < 4:
            continue
        out.append({
            "partition": f[0].strip().rstrip("*"), "state": strip_node_state(f[1]), "state_raw": f[1].strip(),
            "gres": gres_spec(f[2]), "gres_raw": none_if_sentinel(f[2]), "cpus": parse_aiot(f[3]),
        })
    return out


def parse_snapshot(sections: Mapping[str, tuple[list[str], int | None]]) -> dict[str, Any]:
    """Compose the snapshot sections (section 6.2) -> ``{nodes, pd, r, mine, resv, rc}``."""
    def sec(name: str) -> list[str]:
        return list(sections.get(name, ([], None))[0])
    return {
        "nodes": parse_snapshot_nodes(sec("NODES")),
        "pd": parse_uniq_rows(sec("PD")),
        "r": parse_uniq_rows(sec("R")),
        "mine": parse_mine(sec("MINE")),
        "resv": parse_reservations(sec("RESV")),
        "rc": {name: sections[name][1] for name in ("NODES", "PD", "R") if name in sections},
    }


def _partition_gres_types(partition_caps: Mapping[str, Any], name: str) -> tuple[bool, list[str]]:
    """``(has_gpu, types)`` of a partition from ``partition_caps`` (a :func:`merge_partition_gres` /
    :func:`parse_partitions` dict, a ``{"gres_types": ..., "has_gpu": ...}`` dict, or a plain iterable of
    type names); ``(False, [])`` when the partition is unknown. ``has_gpu`` is true when the partition
    advertises GPUs even if its type inventory is empty (TRACE ``batch`` before the sinfo merge)."""
    caps = partition_caps.get(name)
    if caps is None:
        return False, []
    if isinstance(caps, Mapping):
        gt = caps.get("gres_types") or {}
        types = [t for t in (gt.keys() if isinstance(gt, Mapping) else gt) if t is not None]
        return bool(types) or bool(caps.get("has_gpu")), types
    entries = list(caps)
    return bool(entries), [t for t in entries if t is not None]  # a bare None entry = untyped GPU nodes


def classify_demand(row: Mapping[str, Any], partition_caps: Mapping[str, Any]) -> dict[str, Any]:
    """Section 6.2 demand classification of one ``uniq -c`` / ``::MINE`` row.

    GPU request = ``tres_per_node`` if not ``N/A`` else ``tres_per_job`` if not ``N/A``/unknown. A row with
    both unset in a partition that has GPUs is **untyped GPU demand** (``type=None``, ``gpus=None``,
    ``untyped=True``) counted against *every* gres type of the partition: ``against`` is the sorted union
    of the ``gres_types`` of the row's partition(s). ``partition_caps`` MUST be the
    :func:`merge_partition_gres` table (TRES + ``::SINFO`` types); with bare :func:`parse_partitions` output
    ``against`` misses sinfo-only types (Bridges-2 ``h100-80``) and is ``[]`` for an untyped-TRES partition
    (TRACE ``batch``) - an empty ``against`` on a ``kind="gpu"`` row means "every type, inventory unknown".
    In a partition without GPUs the row is CPU demand. Returns ``{kind: "gpu"|"cpu", type, gpus, untyped,
    partition, partitions, count[, against]}``; a comma-list partition (multi-partition pending job) is
    GPU-capable when any member is.
    """
    partition = str(row.get("partition") or "")
    partitions = parse_list(partition)
    count = row.get("count", 1)
    req = gres_spec(row.get("tres_per_node")) or gres_spec(row.get("tres_per_job"))
    base = {"partition": partition, "partitions": partitions, "count": count}
    if req is not None:
        return {**base, "kind": "gpu", "type": req["type"], "gpus": req["count"], "untyped": req["type"] is None}
    any_gpu = False
    advertised: set[str] = set()
    for p in partitions:
        has_gpu, types = _partition_gres_types(partition_caps, p)
        any_gpu = any_gpu or has_gpu
        advertised.update(types)
    if any_gpu:
        return {**base, "kind": "gpu", "type": None, "gpus": None, "untyped": True, "against": sorted(advertised)}
    return {**base, "kind": "cpu", "type": None, "gpus": 0, "untyped": False}


# ---------------------------------------------------------------------------------------------------
# 6.3 submit / estimate / control
# ---------------------------------------------------------------------------------------------------

# Ordered substring -> code map of section 6.3 (E_QOS_POLICY last: only when nothing more specific matched).
SBATCH_ERROR_MAP: tuple[tuple[tuple[str, ...], str], ...] = (
    (("Invalid partition name specified", "invalid partition specified"), "E_PARTITION"),
    (("No partition specified or system default partition",), "E_PARTITION_REQUIRED"),
    (("Invalid account or account/partition combination specified",), "E_ACCOUNT"),
    (("Invalid qos specification",), "E_QOS"),
    (("QOSMaxWallDurationPerJobLimit", "Requested time limit is invalid", "PartitionTimeLimit"), "E_QOS_MAXWALL"),
    (("QOSMaxCpuPerJobLimit", "QOSMaxGRESPerJob", "QOSMaxNodePerJobLimit", "GPU-shared maximum is",
      "use GPU partition for multiple nodes", "PartitionNodeLimit"), "E_QOS_SIZE"),
    (("QOSMaxSubmitJobPerUserLimit", "AssocMaxSubmitJobLimit", "QOSMaxJobsPerUserLimit"), "E_SUBMIT_LIMIT"),
    (("Requested node configuration is not available",), "E_NODE_CONFIG"),
    (("Invalid generic resource (gres) specification",), "E_GRES"),
    # "mem-per-core higher than maximum of NNNNM/core" is Bridges-2 RM-shared/RM-small refusing --mem
    # (MaxMemPerCPU); it is a size limit, not a permission problem (observed 2026-09-02).
    (("Memory required by task is not available", "mem-per-core higher than maximum",
      "Requested node configuration is not available (memory)"), "E_MEM"),
    (("Job dependency problem",), "E_DEPENDENCY"),
    (("Access/permission denied",), "E_PERMISSION"),
    (("Disk quota exceeded", "No space left on device"), "E_QUOTA"),
    (("Socket timed out", "Unable to contact slurm controller", "Zero Bytes were transmitted"), "E_CTLD_BUSY"),
    (("This does not look like a batch script",), "E_SCRIPT"),
    (("Job violates accounting/QOS policy",), "E_QOS_POLICY"),
)


def map_sbatch_error(stderr: str) -> str | None:
    """sbatch/``--test-only`` stderr -> ``E_*`` code per the section 6.3 table (case-insensitive substring
    match over the whole text; the ``sbatch: error: X`` detail wins over the generic policy line)."""
    text = (stderr or "").lower()
    if not text.strip():
        return None
    for needles, code in SBATCH_ERROR_MAP:
        if any(n.lower() in text for n in needles):
            return code
    return None


_ESTIMATE_RE = re.compile(r"^sbatch: Job (\d+) to start at (\S+) using (\d+) processors on nodes (\S+) in partition (\S+)$")
_ALLOC_FAIL_RE = re.compile(r"^(?:sbatch: )?(?:allocation failure|error: Batch job submission failed): (.+)$")
_SBATCH_ERR_RE = re.compile(r"^sbatch: error: (.+)$")


def parse_test_only(text: str, rc: int | None = None) -> dict[str, Any]:
    """``sbatch --test-only`` output (stdout+stderr merged, framing lines ignored) -> estimate or infeasibility.

    Success: ``{"ok": True, "job_id", "est_start"`` (epoch int or ISO string), ``"processors", "nodes",
    "partition"}``. Failure: ``{"ok": False, "reason"`` (the ``allocation failure:`` text), ``"code"``
    (:func:`map_sbatch_error`, ``E_SUBMIT_FAILED`` when unmapped), ``"details"`` (``sbatch: error:`` lines)``}``.
    """
    lines = [l.rstrip("\r") for l in text.splitlines() if l.strip() and not l.startswith("::")]
    for line in lines:
        m = _ESTIMATE_RE.match(line.strip())
        if m:
            return {"ok": True, "job_id": int(m.group(1)), "est_start": parse_ts(m.group(2)),
                    "processors": int(m.group(3)), "nodes": m.group(4), "partition": m.group(5)}
    details = [m.group(1) for m in (_SBATCH_ERR_RE.match(l.strip()) for l in lines) if m]
    reason = None
    for line in lines:
        m = _ALLOC_FAIL_RE.match(line.strip())
        if m:
            reason = m.group(1)
            break
    code = map_sbatch_error("\n".join(details)) or map_sbatch_error(text)
    if reason is None and not details:
        if rc == 0:
            return {"ok": False, "reason": "no estimate line", "code": None, "details": []}
        reason = lines[-1] if lines else "no output"
    return {"ok": False, "reason": reason or (details[0] if details else "unknown"),
            "code": code or "E_SUBMIT_FAILED", "details": details}


def parse_submit_output(text: str) -> dict[str, Any]:
    """First line of ``submit.sh`` output (section 6.3 / 7.2) -> ``{"status": "ok", "job_id", "cluster", "stderr"}``,
    ``{"status": "err", "rc", "code", "stderr"}`` or ``{"status": "ambiguous", "raw"}``.

    ``ERR 1`` maps through :func:`map_sbatch_error` (``E_SUBMIT_FAILED`` when unmapped), ``ERR 2`` ->
    ``E_HELPER``; ``ERR 3`` (lock timeout) and any other first line are ambiguous (the caller marks the
    attempt ``UNCONFIRMED``). Raw sbatch output (``12345``, ``12345;cluster``, ``Submitted batch job 12345``)
    is accepted for completeness.
    """
    lines = [l.rstrip("\r") for l in (text or "").splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return {"status": "ambiguous", "raw": text or ""}
    first = lines[0].strip()
    rest = "\n".join(lines[1:])
    m = re.match(r"^JOBID\s+(\d+)(?:;(\S+))?\s*$", first)
    if m:
        return {"status": "ok", "job_id": int(m.group(1)), "cluster": m.group(2), "stderr": rest}
    m = re.match(r"^ERR\s+(\d+)\s*$", first)
    if m:
        rc = int(m.group(1))
        if rc == 3:
            return {"status": "ambiguous", "raw": text, "rc": rc}
        if rc == 2:
            return {"status": "err", "rc": rc, "code": "E_HELPER", "stderr": rest}
        return {"status": "err", "rc": rc, "code": map_sbatch_error(rest) or "E_SUBMIT_FAILED", "stderr": rest}
    m = re.match(r"^(?:Submitted batch job )?(\d+)(?:;(\S+))?(?: on cluster \S+)?$", first)
    if m:
        return {"status": "ok", "job_id": int(m.group(1)), "cluster": m.group(2), "stderr": rest}
    return {"status": "ambiguous", "raw": text}


SCONTROL_JOB_KEYS: tuple[str, ...] = (
    "JobId", "JobName", "UserId", "GroupId", "MCS_label", "Priority", "Nice", "Account", "QOS", "JobState", "Reason",
    "Dependency", "Requeue", "Restarts", "BatchFlag", "Reboot", "ExitCode", "DerivedExitCode", "RunTime", "TimeLimit",
    "TimeMin", "SubmitTime", "EligibleTime", "AccrueTime", "StartTime", "EndTime", "Deadline", "PreemptEligibleTime",
    "PreemptTime", "SuspendTime", "SecsPreSuspend", "LastSchedEval", "Scheduler", "Partition", "AllocNode:Sid",
    "ReqNodeList", "ExcNodeList", "NodeList", "SchedNodeList", "BatchHost", "NumNodes", "NumCPUs", "NumTasks",
    "CPUs/Task", "ReqB:S:C:T", "TRES", "Socks/Node", "NtasksPerN:B:S:C", "CoreSpec", "MinCPUsNode", "MinMemoryNode",
    "MinMemoryCPU", "MinTmpDiskNode", "Features", "DelayBoot", "OverSubscribe", "Contiguous", "Licenses", "Network",
    "Command", "WorkDir", "StdErr", "StdIn", "StdOut", "Power", "TresPerJob", "TresPerNode", "TresPerTask",
    "TresPerSocket", "MailUser", "MailType", "Comment", "ArrayJobId", "ArrayTaskId", "ArrayTaskThrottle",
    "GresEnforceBind", "CpusPerTres", "MemPerTres", "NtasksPerTRES", "AdminComment", "SystemComment",
    "Reservation", "WCKey", "Container", "CronJob", "Extra", "HetJobId", "HetJobOffset", "HetJobIdSet",
)


def _kv_regex(line: str, keys: Sequence[str]) -> dict[str, str]:
    """Parse ``Key=Value`` tokens where values may contain spaces: a value runs to the next *known* key."""
    pattern = re.compile(r"(?:^|\s)(" + "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True)) + r")=")
    matches = list(pattern.finditer(line))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        out[m.group(1)] = line[m.end():end].strip()
    return out


def scontrol_job_missing(stderr: str) -> bool:
    """True for ``slurm_load_jobs error: Invalid job id specified`` (not in controller memory; section 6.3)."""
    return "Invalid job id specified" in (stderr or "")


def parse_scontrol_job(text: str) -> dict[str, Any]:
    """``scontrol -o show job <id>`` -> typed dict (section 6.3; regex over the known key set because
    ``Command``, ``WorkDir``, ``Comment`` and ``Reason`` may contain spaces). ``{}`` when the job is unknown.

    Keys: ``raw`` (every token), ``job_id, job_name, job_state, state`` (mapped), ``reason, dependency`` (None
    for ``(null)``), ``requeue, restarts, exit_code, derived_exit_code, submit_time, start_time, end_time,
    node_list, sched_node_list, batch_host, std_out, std_err, work_dir, command, comment, tres_per_job,
    tres_per_node`` (:func:`gres_spec`), ``partition, account, qos, array_job_id, array_task_id, num_nodes,
    num_cpus, time_limit_s, tres, priority``.
    """
    if not text or scontrol_job_missing(text):
        return {}
    line = " ".join(l.strip() for l in text.splitlines() if l.strip() and not l.startswith("::"))
    raw = _kv_regex(line, SCONTROL_JOB_KEYS)
    if "JobId" not in raw:
        return {}
    state = raw.get("JobState")
    return {
        "raw": raw,
        "job_id": parse_int(raw.get("JobId")),
        "job_name": raw.get("JobName"),
        "job_state": state,
        "state": map_slurm_state(state),
        "reason": none_if_sentinel(raw.get("Reason")),
        "dependency": none_if_sentinel(raw.get("Dependency")),
        "requeue": parse_int(raw.get("Requeue")),
        "restarts": parse_int(raw.get("Restarts")),
        "exit_code": parse_exit_code(raw.get("ExitCode")),
        "derived_exit_code": parse_exit_code(raw.get("DerivedExitCode")),
        "submit_time": parse_ts(raw.get("SubmitTime")),
        "start_time": parse_ts(raw.get("StartTime")),
        "end_time": parse_ts(raw.get("EndTime")),
        "node_list": none_if_sentinel(raw.get("NodeList")),
        "sched_node_list": none_if_sentinel(raw.get("SchedNodeList")),
        "batch_host": none_if_sentinel(raw.get("BatchHost")),
        "std_out": none_if_sentinel(raw.get("StdOut")),
        "std_err": none_if_sentinel(raw.get("StdErr")),
        "work_dir": none_if_sentinel(raw.get("WorkDir")),
        "command": none_if_sentinel(raw.get("Command")),
        "comment": none_if_sentinel(raw.get("Comment")),
        "tres_per_job": gres_spec(raw.get("TresPerJob")),
        "tres_per_node": gres_spec(raw.get("TresPerNode")),
        "partition": none_if_sentinel(raw.get("Partition")),
        "account": none_if_sentinel(raw.get("Account")),
        "qos": none_if_sentinel(raw.get("QOS")),
        "array_job_id": parse_int(raw.get("ArrayJobId")),
        "array_task_id": none_if_sentinel(raw.get("ArrayTaskId")),
        "num_nodes": parse_int(raw.get("NumNodes")),
        "num_cpus": parse_int(raw.get("NumCPUs")),
        "time_limit_s": parse_duration(none_if_sentinel(raw.get("TimeLimit"))),
        "tres": parse_tres(raw.get("TRES")),
        "priority": parse_int(raw.get("Priority")),
    }


def scancel_errors(stderr: str) -> list[dict[str, Any]]:
    """``scancel`` stderr -> ``[{job_id, message}]`` for ``Kill job error on job id N: ...`` lines (a finished job
    exits 0 silently on 22.05, fixture ``scancel_bad_job``)."""
    out: list[dict[str, Any]] = []
    for line in (stderr or "").splitlines():
        m = re.search(r"Kill job error on job id (\S+?): (.+)$", line)
        if m:
            out.append({"job_id": m.group(1), "message": m.group(2).strip()})
    return out


# ---------------------------------------------------------------------------------------------------
# generic / capture-only helpers (used by golden tests and cluster_status details)
# ---------------------------------------------------------------------------------------------------

def parse_pipe_table(lines: Sequence[str], names: Sequence[str] | None = None) -> list[dict[str, str]]:
    """``|``-separated rows -> list of dicts; without ``names`` the first line is the header (lower-cased)."""
    rows = [l.rstrip("\r") for l in lines if l.strip()]
    if not rows:
        return []
    cols = list(names) if names else [c.strip().lower() for c in rows[0].split("|")]
    if not names:
        rows = rows[1:]
    out: list[dict[str, str]] = []
    for row in rows:
        f = row.split("|", len(cols) - 1)
        out.append({cols[i]: f[i] for i in range(min(len(cols), len(f)))})
    return out


def parse_sinfo_partitions(lines: Sequence[str]) -> list[dict[str, Any]]:
    """``sinfo -h -o '%P|%a|%l|%D|%t|%C|%G|%m|%F|%N'`` rows -> ``partition, default, avail, max_time_s, node_count,
    state, cpus`` (A/I/O/T), ``gres, mem_mb, nodes_aiot, nodelist``."""
    out: list[dict[str, Any]] = []
    for line in lines:
        f = line.rstrip("\r").split("|")
        if len(f) < 10:
            continue
        out.append({
            "partition": f[0].rstrip("*"), "default": f[0].endswith("*"), "avail": f[1],
            "max_time_s": parse_duration(f[2]), "node_count": parse_int(f[3]), "state": strip_node_state(f[4]),
            "cpus": parse_aiot(f[5]), "gres": gres_spec(f[6]), "mem_mb": parse_int(f[7]), "nodes_aiot": parse_aiot(f[8]),
            "nodelist": f[9],
        })
    return out


def parse_sinfo_summary(lines: Sequence[str]) -> list[dict[str, Any]]:
    """``sinfo -s -h -o '%P|%a|%l|%F|%N'`` rows -> ``partition, default, avail, max_time_s, nodes_aiot, nodelist``."""
    out: list[dict[str, Any]] = []
    for line in lines:
        f = line.rstrip("\r").split("|")
        if len(f) < 5:
            continue
        out.append({"partition": f[0].rstrip("*"), "default": f[0].endswith("*"), "avail": f[1],
                    "max_time_s": parse_duration(f[2]), "nodes_aiot": parse_aiot(f[3]), "nodelist": f[4]})
    return out


__all__ = [
    "IncompleteProbe", "parse_sections", "parse_now",
    "is_sentinel", "none_if_sentinel", "parse_ts", "parse_int", "parse_float", "parse_secs", "parse_exit_code",
    "mem_to_bytes", "req_mem_bytes", "parse_tres", "gres_types_from_tres", "parse_list", "gres_spec",
    "strip_node_state", "parse_aiot", "split_fields",
    "parse_env", "parse_tz_offset", "parse_version", "parse_config", "parse_partitions", "parse_sinfo_nodes",
    "aggregate_sinfo", "parse_user", "parse_assoc", "parse_qos", "parse_sshare", "parse_balance",
    "parse_reservations", "parse_tools", "parse_cap_o", "parse_df", "df_row_for_path",
    "TICK_SQUEUE_CODES", "parse_squeue_rows", "parse_squeue_tick", "parse_restarts", "TICK_SACCT_FIELDS",
    "ENRICH_FIELDS", "RECOVER_FIELDS", "BACKFILL_FIELDS", "FIXTURE_SACCT_FIELDS", "parse_sacct_rows",
    "group_incarnations", "parse_sacct_tick", "parse_files", "parse_cmds", "parse_recover", "parse_backfill",
    "parse_enrich",
    "DEMAND_FIELDS", "UNIQ_PTB_FIELDS", "parse_uniq_rows", "parse_mine", "parse_snapshot_nodes", "parse_snapshot", "classify_demand",
    "SBATCH_ERROR_MAP", "map_sbatch_error", "parse_test_only", "parse_submit_output", "SCONTROL_JOB_KEYS",
    "scontrol_job_missing", "parse_scontrol_job", "scancel_errors",
    "parse_pipe_table", "parse_sinfo_partitions", "parse_sinfo_summary",
]
