"""Bootstrap: capability discovery cache, helper deploy/verify, QOS candidates, effective limits, transfer-host
probes and the 30-day wait-history back-fill (design section 6.1; section 5.8 "Discovery cache older than 24 h
refreshes lazily"; section 5.5 transfer-host capabilities).

Pure functions (``enrich_caps``, ``effective_limits``, ``qos_for_partition``, ``charge_for``,
``partition_accessible``) work on the caps dict produced by :func:`slurm_mcp.slurm.client.parse_discovery`;
the ``async`` functions take a :class:`SlurmClient` and the :class:`Store`. No cluster name appears here.
"""
from __future__ import annotations

import fnmatch
import logging
import time
from types import SimpleNamespace
from typing import Any, Mapping

from ..config import ClusterProfile, control_root as profile_control_root, has_transfer_host, target_override, \
    transfer_endpoint, transfer_host as profile_transfer_host
from ..helpers import bundle_sha8
from ..render import choose_qos
from ..store import Store
from .client import SlurmClient

log = logging.getLogger("slurm_mcp.discovery")

CAPS_TTL_S = 24 * 3600.0
BACKFILL_KEY_PREFIX = "backfill."
TRANSFER_PORTS = (22, 2222)


def caps_key(cluster: str) -> str:
    return f"caps.{cluster}"


def caps_age_s(caps: Mapping[str, Any] | None, now: float | None = None) -> float | None:
    if not caps or caps.get("fetched_local") is None:
        return None
    return max(0.0, (time.time() if now is None else now) - float(caps["fetched_local"]))


def caps_fresh(caps: Mapping[str, Any] | None, ttl_s: float = CAPS_TTL_S) -> bool:
    age = caps_age_s(caps)
    return age is not None and age < ttl_s


# --- pure derivations (section 6.1 bullets "Effective limits", "QOS selection", "Charge") ----------------

def partition_accessible(part: Mapping[str, Any], caps: Mapping[str, Any]) -> bool:
    """AllowGroups contains the user's group or ALL; AllowAccounts contains the default account or ALL; the assoc
    partition filter (when set) names it (section 8 candidates). An unknown group (``id -gn`` unavailable, or a
    bare gid) cannot be checked and is not held against the partition."""
    groups = [g for g in part.get("allow_groups") or [] if g]
    # Every group the user belongs to (``id -Gn``), not just the primary one: AllowGroups is normally a
    # supplementary group, so matching the primary alone hid the partitions the user can actually use.
    mine = [g for g in (caps.get("groups") or []) if g] or ([caps.get("group")] if caps.get("group") else [])
    known = [g for g in mine if g and not str(g).isdigit()]
    if groups and not any(g.upper() == "ALL" for g in groups):
        if known and not (set(known) & set(groups)):
            return False
    accounts = [a for a in part.get("allow_accounts") or [] if a]
    account = caps.get("default_account")
    if accounts and not any(a.upper() == "ALL" for a in accounts) and account and account not in accounts:
        return False
    assoc = caps.get("assoc") or {}
    if assoc.get("partition") and assoc["partition"] != part.get("name"):
        return False
    return True


def effective_limits(caps: Mapping[str, Any], partition: str, qos: str | None = None) -> dict[str, Any]:
    """Section 6.1 "Effective limits per partition": partition ``MaxTime`` AND the partition QOS ``MaxWall`` AND the
    chosen job QOS ``MaxWall`` AND the assoc ``MaxWall`` (None ignored); ``max_jobs_pu``/``max_submit_pu``/
    ``max_tres_pj`` from the partition QOS (the job QOS narrows them when it has values); ``max_nodes`` from the
    partition and the QOS ``max_tres.node``."""
    part = (caps.get("partitions") or {}).get(partition) or {}
    qos_table = caps.get("qos") or {}
    part_qos = qos_table.get(part.get("qos") or "") or {}
    job_qos = qos_table.get(qos or "") or {}
    assoc = caps.get("assoc") or {}
    walls = [w for w in (part.get("max_time_s"), part_qos.get("max_wall_s"), job_qos.get("max_wall_s"),
                         assoc.get("max_wall_s")) if w is not None]
    max_tres: dict[str, float] = {}
    for src in (part_qos.get("max_tres") or {}, job_qos.get("max_tres") or {}, assoc.get("max_tres") or {}):
        for k, v in src.items():
            if isinstance(v, (int, float)):
                max_tres[k] = min(max_tres[k], float(v)) if k in max_tres else float(v)
    nodes = [n for n in (part.get("max_nodes"), max_tres.get("node")) if n is not None]
    jobs = [n for n in (part_qos.get("max_jobs_pu"), job_qos.get("max_jobs_pu"), assoc.get("max_jobs")) if n is not None]
    subs = [n for n in (part_qos.get("max_submit_pu"), job_qos.get("max_submit_pu"), assoc.get("max_submit"))
            if n is not None]
    return {
        "max_wall_s": min(walls) if walls else None,
        "max_jobs_pu": int(min(jobs)) if jobs else None,
        "max_submit_pu": int(min(subs)) if subs else None,
        "max_tres_pj": max_tres,
        "max_nodes": int(min(nodes)) if nodes else None,
        "max_cpus_node": (caps.get("sinfo") or {}).get(partition, {}).get("max_cpus") or part.get("total_cpus"),
        "max_mem_mb_node": (caps.get("sinfo") or {}).get(partition, {}).get("max_mem_mb") or part.get("max_mem_per_node"),
    }


def qos_for_partition(caps: Mapping[str, Any], profile: ClusterProfile | Any, partition: str,
                      spec_qos: str | None = None) -> list[str]:
    """Ordered QOS candidates for a partition (section 6.1 "QOS selection"): the validated cache
    ``caps.qos_for_partition[partition]`` first, then :func:`render.choose_qos`'s order. An empty list means
    "no ``--qos``" (``AllowQos=ALL`` with an assoc default). Validation by ``--test-only`` is the submitter's job."""
    if spec_qos:
        return [spec_qos]
    mapped = (getattr(profile, "qos_map", None) or {}).get(partition)
    if mapped:
        return [mapped]
    part = (caps.get("partitions") or {}).get(partition) or {"name": partition}
    cands = list(choose_qos(SimpleNamespace(qos=None), profile, part, caps.get("assoc")))
    cached = (caps.get("qos_for_partition") or {}).get(partition)
    if cached and cached in cands:
        cands.remove(cached)
        cands.insert(0, cached)
    elif cached:
        cands.insert(0, cached)
    return cands


def charge_for(caps: Mapping[str, Any], profile: ClusterProfile | Any, partition: str,
               gres_type: str | None = None) -> dict[str, Any] | str:
    """``{"unit", "su_per_unit_h"}`` or ``"free"`` (section 6.1 Charge, section 8 Cost): ``TRESBillingWeights``
    when present (per gres type, then ``gres/gpu``, then ``cpu``), else ``profile.su_rates`` (``gpu:<type>`` ->
    ``gpu:*`` -> ``cpu``), else free."""
    part = (caps.get("partitions") or {}).get(partition) or {}
    weights = part.get("tres_billing_weights") or {}
    gpu_part = bool(part.get("has_gpu"))

    def _num(v: Any) -> float | None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    if weights:
        if gres_type and _num(weights.get(f"gres/gpu:{gres_type}")) is not None:
            return {"unit": f"gpu:{gres_type}", "su_per_unit_h": _num(weights[f"gres/gpu:{gres_type}"])}
        if gpu_part and _num(weights.get("gres/gpu")) is not None:
            return {"unit": f"gpu:{gres_type}" if gres_type else "gpu", "su_per_unit_h": _num(weights["gres/gpu"])}
        if _num(weights.get("cpu")) is not None:
            return {"unit": "cpu", "su_per_unit_h": _num(weights["cpu"])}
    rates = dict(getattr(profile, "su_rates", None) or {})
    if rates:
        if gpu_part:
            if gres_type and f"gpu:{gres_type}" in rates:
                return {"unit": f"gpu:{gres_type}", "su_per_unit_h": float(rates[f"gpu:{gres_type}"])}
            if "gpu:*" in rates:
                return {"unit": f"gpu:{gres_type}" if gres_type else "gpu", "su_per_unit_h": float(rates["gpu:*"])}
        if "cpu" in rates:
            return {"unit": "cpu", "su_per_unit_h": float(rates["cpu"])}
    return "free"


def enrich_caps(caps: dict[str, Any], profile: ClusterProfile | Any) -> dict[str, Any]:
    """Add the derived per-partition fields to a raw caps dict (in place, returned): ``accessible``, ``limits``
    (:func:`effective_limits` with the first QOS candidate), ``qos_candidates``, ``charge`` (for the first gres
    type or cpu), ``gres_type_list`` (typed gres names) and the top-level ``qos_candidates`` map."""
    partitions: dict[str, dict[str, Any]] = caps.get("partitions") or {}
    cands_map: dict[str, list[str]] = {}
    for name, part in partitions.items():
        part.setdefault("name", name)
        cands = qos_for_partition(caps, profile, name)
        cands_map[name] = cands
        part["qos_candidates"] = cands
        part["accessible"] = partition_accessible(part, caps)
        part["limits"] = effective_limits(caps, name, cands[0] if cands else None)
        types = sorted(t for t in (part.get("gres_types") or {}) if t)
        part["gres_type_list"] = types
        part["charge"] = charge_for(caps, profile, name, types[0] if types else None)
    caps["qos_candidates"] = cands_map
    caps.setdefault("qos_for_partition", {})
    caps["charges"] = bool(caps.get("charges")) or any(p.get("charge") != "free" for p in partitions.values())
    return caps


# --- async bootstrap ----------------------------------------------------------------------------------------

async def bootstrap(client: SlurmClient, profile: ClusterProfile, store: Store, *, refresh: bool = False,
                    ttl_s: float = CAPS_TTL_S) -> dict[str, Any]:
    """Return the cluster's caps: the ``kv.caps.<cluster>`` cache when younger than ``ttl_s`` (24 h) and
    ``refresh`` is False, else a fresh :meth:`SlurmClient.discover` stored with ``fetched_local``.

    The validated QOS cache (``qos_for_partition``) and the transfer-host probe (``transfer``) survive a refresh.
    """
    key = caps_key(profile.name)
    cached = await store.read(lambda c: store.kv_get(c, key))
    if cached and not refresh and caps_fresh(cached, ttl_s):
        return cached
    caps = await client.discover()
    caps["fetched_local"] = time.time()
    caps["login_sftp_ok"] = await probe_login_sftp(client)
    if cached:
        caps["qos_for_partition"] = dict(cached.get("qos_for_partition") or {})
        if cached.get("transfer"):
            caps["transfer"] = cached["transfer"]
        # a missing ::HELPER line means the bundle is gone: helper_sha8 stays None so ensure_helpers redeploys
    await store.write(lambda c: store.kv_set(c, key, caps))
    return caps


async def probe_login_sftp(client: SlurmClient) -> bool:
    """Does the login host serve the SFTP subsystem (design section 2.3)?

    Measured 2026-09-02: the Bridges-2 login node answers exec channels but refuses SFTP with
    ``ChannelOpenError: Session request failed``, while its DTN does the opposite. One probe at bootstrap
    decides whether small-file work uses SFTP, the transfer host's SFTP, or exec-channel primitives.
    """
    try:
        sftp = await client.login.sftp()
        await sftp.realpath(".")
        return True
    except Exception as e:
        log.info("%s: login host does not serve SFTP (%s); file I/O falls back (section 2.3)", client.cluster, e)
        return False


async def save_caps(store: Store, cluster: str, caps: Mapping[str, Any]) -> None:
    await store.write(lambda c: store.kv_set(c, caps_key(cluster), dict(caps)))


async def ensure_helpers(client: SlurmClient, profile: ClusterProfile, caps: dict[str, Any],
                         store: Store | None = None) -> str:
    """Deploy the helper bundle only when ``<control_root>/bin/VERSION`` differs from the packaged sha8
    (section 6.1 "Helper deploy"). Updates ``caps["helper_sha8"]`` (and the kv cache when ``store`` is given)."""
    packaged = bundle_sha8()
    if caps.get("helper_sha8") == packaged:
        return packaged
    root = profile_control_root(profile)
    deployed = await client.helper_version(root)
    if deployed != packaged:
        deployed = await client.deploy_helpers(root)
        log.info("%s: deployed helper bundle %s to %s/bin", profile.name, deployed, root)
    caps["helper_sha8"] = deployed
    if store is not None:
        await save_caps(store, profile.name, caps)
    return deployed


async def transfer_capabilities(profile: ClusterProfile, transport_transfer: Any | None = None,
                                *, tcp_probe: Any = None, banner_probe: Any = None) -> dict[str, Any]:
    """Transfer-host capabilities (section 5.5 / 6.1): ``{host, port, banner, exec_ok, sftp_ok, mb_per_s}``.

    Port: ``profile.transfer_port`` if set, else the banner probe (22, then 2222 when the 22 banner lacks ``hpn``
    and 2222 answers). ``exec_ok`` = ``echo ok`` over an exec channel; ``sftp_ok`` = an SFTP ``realpath``.
    Without a transfer host the login host is reported with ``exec_ok``/``sftp_ok`` True (it is the transport).
    """
    from ..transport import banner_probe as _banner, tcp_probe as _tcp
    banner_fn = banner_probe or _banner
    host = profile_transfer_host(profile)
    if not has_transfer_host(profile):
        h, p = transfer_endpoint(profile)
        return {"host": h, "port": p, "banner": None, "exec_ok": True, "sftp_ok": True, "mb_per_s": None,
                "role": "login"}
    port = profile.transfer_port
    banner = ""
    if port is None:
        banner = await banner_fn(host, TRANSFER_PORTS[0])
        port = TRANSFER_PORTS[0]
        if "hpn" not in banner.lower():
            alt = await banner_fn(host, TRANSFER_PORTS[1])
            if alt.startswith("SSH-"):
                banner, port = alt, TRANSFER_PORTS[1]
    else:
        banner = await banner_fn(host, port)
    out: dict[str, Any] = {"host": host, "port": port, "banner": banner or None, "exec_ok": None, "sftp_ok": None,
                           "mb_per_s": None, "role": "transfer"}
    if transport_transfer is not None:
        try:
            res = await transport_transfer.run("echo ok", timeout=30, login_shell=False)
            out["exec_ok"] = bool(res.ok and res.stdout.strip() == "ok")
        except Exception as e:  # restricted shells raise or return garbage: not exec-capable
            log.info("%s transfer host %s: exec probe failed: %s", profile.name, host, e)
            out["exec_ok"] = False
        try:
            sftp = await transport_transfer.sftp()
            await sftp.realpath(".")
            out["sftp_ok"] = True
        except Exception as e:
            log.info("%s transfer host %s: sftp probe failed: %s", profile.name, host, e)
            out["sftp_ok"] = False
    return out


def backfill_target_key(cluster: str, row: Mapping[str, Any]) -> str | None:
    partition = row.get("partition")
    if not partition:
        return None
    key = f"{cluster}:{partition}"
    if row.get("gres_type"):
        key += f":{row['gres_type']}"
    return key


async def backfill_wait_history(client: SlurmClient, store: Store, caps: Mapping[str, Any] | None = None,
                                *, force: bool = False) -> int:
    """Once per cluster: ``sacct -S now-30days`` rows with a real ``Start`` -> ``wait_history(source="backfill")``
    (section 6.1 last bullet). Returns the number of rows inserted (0 when already done)."""
    cluster = client.cluster
    key = BACKFILL_KEY_PREFIX + cluster
    done = await store.read(lambda c: store.kv_get(c, key))
    if done and not force:
        return 0
    rows = await client.backfill_history()
    inserted = 0

    def fn(conn: Any) -> int:
        n = 0
        for r in rows:
            submit, start = r.get("submit_ts"), r.get("start_ts")
            tk = backfill_target_key(cluster, r)
            if submit is None or start is None or tk is None or start < submit:
                continue
            store.insert_wait_history(conn, cluster=cluster, target_key=tk, submit_ts=int(submit), start_ts=int(start),
                                      source="backfill", gpus=r.get("gpus"))
            n += 1
        store.kv_set(conn, key, {"done_local": time.time(), "rows": n})
        return n

    inserted = await store.write(fn)
    return inserted


def target_enabled(profile: ClusterProfile, target_key: str) -> bool:
    """``profile.target_overrides[glob].enabled`` (default True), for ``cluster_status(detail="targets")``."""
    ov = target_override(profile, target_key)
    return bool(ov.get("enabled", True))


def matches_any(key: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(key, g) for g in globs)


__all__ = ["CAPS_TTL_S", "caps_key", "caps_age_s", "caps_fresh", "partition_accessible", "effective_limits",
           "qos_for_partition", "charge_for", "enrich_caps", "bootstrap", "save_caps", "ensure_helpers",
           "transfer_capabilities", "backfill_target_key", "backfill_wait_history", "target_enabled", "matches_any"]
