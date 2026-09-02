"""The MCP server: ``MCPServer("slurm")``, INSTRUCTIONS, the lifespan of design section 5.8 and the stdio entry
point (design section 2 "server.py", section 4 "Server", section 11g/h).

``build_server(service=None)`` returns the configured ``MCPServer``. With a ``Service`` (tests, the CLI mirror)
the lifespan only yields it; without one ``app_lifespan`` opens the ledger at ``~/.slurm-mcp/state.db``
(``SLURM_MCP_HOME`` honoured), creates the ``EventBus``/``ClusterRegistry``/``Service``, acquires the monitor
lease (never SSH: ``initialize`` must answer within 30 s), attaches the components that exist (``notify`` always;
``monitor``/``submitter``/``transfers``/``alloc`` when their modules are present) and starts them. Tools are
registered against a ``ServiceProxy`` that forwards to the live ``Service`` once the lifespan bound it.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import logging
import secrets
import sys
from collections.abc import AsyncIterator
from typing import Any

from . import _mcp
from ._mcp import MCPServer, ToolError
from .config import CONFIG_DIR, load_profiles
from .errors import EM_DASH
from .events import EventBus
from .notify import Notifier, default_toast_sender, default_webhook_sender, threaded
from .service import ClusterRegistry, Service
from .store import Store
from .tools import register_all

log = logging.getLogger("slurm_mcp.server")

SERVER_NAME = "slurm"
STATE_DB_NAME = "state.db"
OPTIONAL_COMPONENTS: tuple[tuple[str, str, str], ...] = (
    # (component name, module, class) -- attached when the module exists (slices 3, 5, 7)
    ("monitor", "slurm_mcp.monitor", "Monitor"),
    ("submitter", "slurm_mcp.submitter", "Submitter"),
    ("transfers", "slurm_mcp.transfer", "TransferManager"),
    ("alloc", "slurm_mcp.alloc", "AllocManager"),
)

INSTRUCTIONS = (
    "Tools for running SLURM jobs on SSH clusters configured with `slurm-mcp cluster add`. Job handles look like "
    "`j17`, allocations `a3` (commands `a3.c2`), transfers `t4`, plans `p9`. Typical flow: `plan_job` (optional) -> "
    "`submit_job` -> `wait_for_events(timeout_s=300)` or `job_status` -> `job_logs` -> `collect_results`. Every "
    "response includes `unread_events`; when > 0 call `wait_for_events(timeout_s=0)`. `wait_for_events` with a long "
    "timeout is the right way to wait for a job: Claude Code moves the call to the background after 2 minutes and "
    "returns the result as a task notification. Events are delivered until you acknowledge them: pass the previous "
    "result's `next_seq` as `ack_seq` on your next call (if you never saw a result, call again without it and the "
    "same events are replayed). Prefer `placement=\"auto\"` so the server balances partitions/clusters by wait time, "
    "SU cost and policy. `submit_job` returns a handle immediately and finishes in the background; the "
    "`submitted`/`submit_failed` event closes it. Use `job_control`/`rebalance` rather than `run_command`. Never "
    "poll faster than every 30 s; the server already does."
)


class ServiceProxy:
    """Forwards attribute access to the ``Service`` bound by the lifespan (tools are registered before it exists)."""

    def __init__(self, service: Service | None = None) -> None:
        self._service = service

    def bind(self, service: Service | None) -> None:
        self._service = service

    @property
    def bound(self) -> bool:
        return self._service is not None

    def __getattr__(self, name: str) -> Any:
        svc = self.__dict__.get("_service")
        if svc is None:
            raise ToolError(f"E_STATE: the server is still starting {EM_DASH} fix: retry in a few seconds")
        return getattr(svc, name)


def state_db_path() -> Any:
    return CONFIG_DIR / STATE_DB_NAME


def attach_optional_components(service: Service) -> list[str]:
    """Attach the components of later slices when importable; a missing module is normal in phase 2."""
    attached: list[str] = []
    for name, module_name, class_name in OPTIONAL_COMPONENTS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        cls = getattr(module, class_name, None)
        if cls is None:
            log.warning("%s has no %s; component %s not attached", module_name, class_name, name)
            continue
        try:
            service.attach(name, cls(service))
            attached.append(name)
        except Exception as e:
            log.warning("component %s could not be constructed: %s", name, e)
    return attached


def make_notifier(service: Service) -> Notifier:
    return Notifier(service.store, service.events, service.notify_policy_cached,
                    toast_sender=threaded(default_toast_sender), webhook_sender=threaded(default_webhook_sender))


@contextlib.asynccontextmanager
async def app_lifespan(server: MCPServer, proxy: ServiceProxy | None = None, *, fake: bool = False,
                       ) -> AsyncIterator[Service]:
    """Section 5.8 lifespan start: open SQLite, migrate, load policies, lazy transports, session_id, lease."""
    if fake:
        log.warning("--fake requested: the in-process fake cluster mode arrives in phase 3; serving real profiles")
    store = Store(state_db_path())
    if store.recovered:
        log.error("state.db was corrupt and moved to %s; starting fresh", store.corrupt_backup)
    session_id = secrets.token_hex(4)
    events = EventBus(store, session_id=session_id)
    profiles = load_profiles()
    registry = ClusterRegistry(profiles, store)
    service = Service(store, events, registry, session_id)
    info = await service.acquire_lease()
    log.info("session %s: monitor lease %s (%s)", session_id, "acquired" if info.acquired else "not acquired", info.reason)
    if store.recovered:
        try:
            await events.emit("needs_attention", summary="state.db was corrupt and recreated",
                              payload={"why": "db_corrupt", "hint": f"backup at {store.corrupt_backup}"})
        except Exception:  # pragma: no cover
            pass
    service.attach("notify", make_notifier(service))
    attach_optional_components(service)
    await service.start()
    if proxy is not None:
        proxy.bind(service)
    try:
        yield service
    finally:
        if proxy is not None:
            proxy.bind(None)
        try:
            await service.stop()
        finally:
            try:
                await service.release_lease()
            except Exception as e:  # pragma: no cover
                log.warning("lease release failed: %s", e)
            await registry.close()
            store.close()


def build_server(service: Service | None = None, *, fake: bool = False) -> MCPServer:
    """The configured ``MCPServer``. With ``service`` the lifespan yields it unchanged (tests/CLI)."""
    proxy = ServiceProxy(service)

    if service is not None:
        @contextlib.asynccontextmanager
        async def lifespan(_server: MCPServer) -> AsyncIterator[Service]:
            yield service
    else:
        @contextlib.asynccontextmanager
        async def lifespan(server_: MCPServer) -> AsyncIterator[Service]:
            async with app_lifespan(server_, proxy, fake=fake) as svc:
                yield svc

    mcp = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS, lifespan=lifespan, log_level="INFO")
    groups = register_all(mcp, proxy)
    log.info("registered tool groups: %s", ", ".join(groups))
    return mcp


def configure_logging(level: int = logging.INFO) -> None:
    """stderr only: stdout belongs to the protocol (research note section 2.3)."""
    logging.basicConfig(level=level, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main(argv: list[str] | None = None, *, fake: bool = False) -> None:
    """Run the server on stdio (``slurm-mcp serve``)."""
    ap = argparse.ArgumentParser(prog="slurm-mcp serve")
    ap.add_argument("--fake", action="store_true", help="phase 3: serve the in-process fake cluster (placeholder)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv or [])
    configure_logging(logging.DEBUG if args.debug else logging.INFO)
    build_server(fake=fake or args.fake).run(transport="stdio")


if __name__ == "__main__":  # python -m slurm_mcp.server
    main(sys.argv[1:])


__all__ = ["SERVER_NAME", "INSTRUCTIONS", "ServiceProxy", "app_lifespan", "build_server", "attach_optional_components",
           "make_notifier", "state_db_path", "configure_logging", "main", "_mcp"]
