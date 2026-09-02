"""Configuration tool: ``configure`` (design section 4 "Configuration")."""
from __future__ import annotations

from typing import Any, Optional

from .. import _mcp
from .._mcp import MCPServer
from ..models import ConfigResult
from . import run_tool

CONFIGURE_DESC = (
    "Read or patch the server policies; no arguments = read. placement (PlacementPolicy): objective "
    "balanced|fastest|cheapest, su_to_hours (SU-to-hours weight, default by objective 0.25/0.02/2.0), su_reserve "
    "(SU kept untouched, 50), max_pending_per_target (null = discovered bf_max_job_user cap; jobs above it are held "
    "locally as QUEUED), max_running_per_target {target-glob: n} (etiquette caps), allow_self_preempt, soft_caps, "
    "etiquette_h, targets_allow/targets_deny (globs on '<cluster>:<partition>[:<gres>][@qos]'), prefer_cluster, "
    "unknown_wait_h, rebalance {enabled, interval_min, min_gain_h, max_moves_per_job, max_extra_su, min_age_min, "
    "max_moves_per_hour, hysteresis_h}. notify (NotifyPolicy): toast (Windows toasts on/off), toast_kinds, "
    "webhook_url + webhook_kinds (JSON POST per event), email (adds --mail-type=END,FAIL,REQUEUE,TIME_LIMIT_90 "
    "--mail-user to every submission), quiet_hours [start, end) local hours. Patches merge into the stored "
    "policy, are validated (E_INVALID_SPEC on unknown keys) and take effect for the next placement/tick. "
    "Returns the effective policies."
)


def register(mcp: MCPServer, service: Any) -> None:
    @mcp.tool(name="configure", description=CONFIGURE_DESC,
              annotations=_mcp._ann(read_only_hint=False, destructive_hint=False, idempotent_hint=True))
    async def configure(placement: Optional[dict[str, Any]] = None, notify: Optional[dict[str, Any]] = None,
                        ) -> ConfigResult:
        return await run_tool(service.configure(placement=placement, notify=notify))


__all__ = ["register", "CONFIGURE_DESC"]
