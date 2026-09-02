"""Human notifications: Windows toasts, webhook POSTs, SLURM mail options, quiet hours, coalescing
(design section 5.6 "notify.py runs after each tick", section 2 "notify.py", slice 4).

``Notifier.after_tick()`` reads the events with ``notified=0`` (younger than 24 h) whose kind is in
``NotifyPolicy.toast_kinds`` / ``webhook_kinds``, delivers them and marks them ``notified=1`` on success so a
restart never re-toasts. Senders are injectable (tests pass fakes); the defaults run ``win11toast.notify`` and
``urllib`` in worker threads so the event loop never blocks. Quiet hours suppress toasts (the events are still
marked, never re-delivered later) and never touch ledger events.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
import urllib.request
from typing import Any, Awaitable, Callable

from .events import EventBus
from .models import EventRow, NotifyPolicy
from .render import MAIL_TYPES
from .store import Store

log = logging.getLogger("slurm_mcp.notify")

COALESCE_S = 10.0
WEBHOOK_TIMEOUT_S = 5.0
WEBHOOK_RETRIES = 3
MISSED_MAX_AGE_H = 24.0
MAX_DELIVERY_ATTEMPTS = 3

ToastSender = Callable[[str, str], Any]
WebhookSender = Callable[[str, dict[str, Any]], Any]
PolicyGetter = Callable[[], "NotifyPolicy | Awaitable[NotifyPolicy]"]


def mail_options(policy: NotifyPolicy | None) -> list[str]:
    """``--mail-type=END,FAIL,REQUEUE,TIME_LIMIT_90 --mail-user=<email>`` when ``notify.email`` is set (section 5.6),
    else ``[]``; the render layer appends these to the sbatch command line."""
    if policy is None or not policy.email:
        return []
    return [f"--mail-type={MAIL_TYPES}", f"--mail-user={policy.email}"]


def in_quiet_hours(policy: NotifyPolicy | None, hour: int | None = None) -> bool:
    """True when the local hour falls in ``policy.quiet_hours = [start, end)`` (wraps past midnight)."""
    if policy is None or not policy.quiet_hours:
        return False
    start, end = policy.quiet_hours
    h = time.localtime().tm_hour if hour is None else int(hour)
    if start == end:
        return False
    if start < end:
        return start <= h < end
    return h >= start or h < end


def toast_title(event: EventRow) -> str:
    """``slurm j17 COMPLETED (bridges2)`` (section 5.6)."""
    state = event.payload.get("state") if isinstance(event.payload, dict) else None
    label = str(state) if state else event.kind
    who = event.handle or event.slurm_id or ""
    parts = ["slurm", who, label]
    title = " ".join(p for p in parts if p)
    return f"{title} ({event.cluster})" if event.cluster else title


def default_toast_sender(title: str, body: str) -> None:
    """``win11toast.notify`` (blocking; run it in a thread). Silently no-op where the module is missing."""
    try:
        import win11toast  # type: ignore
    except ImportError:  # pragma: no cover - non-Windows hosts
        log.info("toast (no win11toast): %s - %s", title, body)
        return
    win11toast.notify(title, body)


def default_webhook_sender(url: str, payload: dict[str, Any]) -> None:
    """One JSON POST with a 5 s timeout (blocking; run it in a thread); raises on any failure."""
    data = json.dumps(payload, default=str).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT_S) as resp:  # noqa: S310 - user-configured URL
        resp.read()


def event_payload(event: EventRow) -> dict[str, Any]:
    return {"seq": event.seq, "ts": event.ts, "kind": event.kind, "handle": event.handle, "cluster": event.cluster,
            "slurm_id": event.slurm_id, "summary": event.summary, "payload": dict(event.payload)}


class Notifier:
    """Deliver ledger events to the human (section 5.6). Component protocol: ``start()``/``stop()``/``after_tick()``."""

    def __init__(self, store: Store, events: EventBus, policy_getter: PolicyGetter, *,
                 toast_sender: ToastSender | None = None, webhook_sender: WebhookSender | None = None,
                 now: Callable[[], float] = time.time, coalesce_s: float = COALESCE_S,
                 local_hour: Callable[[], int] | None = None) -> None:
        self.store = store
        self.events = events
        self.policy_getter = policy_getter
        self.toast_sender = toast_sender or default_toast_sender
        self.webhook_sender = webhook_sender or default_webhook_sender
        self._now = now
        self.coalesce_s = float(coalesce_s)
        self._local_hour = local_hour
        self.last_toast_local: float | None = None
        self.attempts: dict[int, int] = {}
        self.sent_toasts: int = 0
        self.sent_webhooks: int = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Deliver the events missed while no server ran (< 24 h, ``notified=0``) as one coalesced toast."""
        try:
            await self.after_tick(coalesce_all=True)
        except Exception as e:  # never stop the lifespan for a notification problem
            log.warning("notifier startup delivery failed: %s", e)

    async def stop(self) -> None:
        return None

    async def policy(self) -> NotifyPolicy:
        pol = self.policy_getter()
        if inspect.isawaitable(pol):
            pol = await pol
        return pol if isinstance(pol, NotifyPolicy) else NotifyPolicy.model_validate(pol or {})

    async def after_tick(self, *, coalesce_all: bool = False) -> dict[str, int]:
        """Toast + webhook every unnotified event of the configured kinds; returns delivery counts."""
        async with self._lock:
            policy = await self.policy()
            toast_kinds = set(policy.toast_kinds) if policy.toast else set()
            webhook_kinds = set(policy.effective_webhook_kinds) if policy.webhook_url else set()
            kinds = toast_kinds | webhook_kinds
            stats = {"toasts": 0, "webhooks": 0, "suppressed": 0, "marked": 0, "deferred": 0}
            if not kinds:
                return stats
            pending = await self.events.unnotified(sorted(kinds), MISSED_MAX_AGE_H)
            if not pending:
                return stats
            done: set[int] = set()
            failed: set[int] = set()
            # --- toasts (coalesced to one per window) -------------------------------------------------
            toast_events = [e for e in pending if e.kind in toast_kinds]
            if toast_events:
                now = self._now()
                quiet = in_quiet_hours(policy, self._local_hour() if self._local_hour else None)
                if quiet:
                    stats["suppressed"] += len(toast_events)
                    done.update(e.seq for e in toast_events)
                elif (self.last_toast_local is not None and now - self.last_toast_local < self.coalesce_s
                      and not coalesce_all):
                    stats["deferred"] += len(toast_events)      # next tick sends them as one toast
                else:
                    ok = await self._send_toast(toast_events)
                    self.last_toast_local = now
                    if ok:
                        stats["toasts"] += 1
                        done.update(e.seq for e in toast_events)
                    else:
                        failed.update(e.seq for e in toast_events)
            # --- webhooks (one POST per event) ------------------------------------------------------
            for e in pending:
                if e.kind not in webhook_kinds:
                    continue
                if await self._send_webhook(policy.webhook_url or "", e):
                    stats["webhooks"] += 1
                    if e.kind not in toast_kinds or e.seq in done:
                        done.add(e.seq)
                else:
                    failed.add(e.seq)
                    done.discard(e.seq)
            # --- bookkeeping ------------------------------------------------------------------------
            for seq in failed:
                self.attempts[seq] = self.attempts.get(seq, 0) + 1
                if self.attempts[seq] >= MAX_DELIVERY_ATTEMPTS:
                    log.warning("giving up notifying event %d after %d attempts", seq, self.attempts[seq])
                    done.add(seq)
            if done:
                stats["marked"] = await self.events.mark_notified(sorted(done))
                for seq in done:
                    self.attempts.pop(seq, None)
            return stats

    async def _send_toast(self, events: list[EventRow]) -> bool:
        if len(events) == 1:
            title, body = toast_title(events[0]), events[0].summary or events[0].kind
        else:
            title = f"slurm: {len(events)} events"
            body = "; ".join((e.summary or f"{e.handle or ''} {e.kind}").strip() for e in events[:5])
            if len(events) > 5:
                body += f"; +{len(events) - 5} more"
        try:
            result = self.toast_sender(title, body[:250])
            if inspect.isawaitable(result):
                await result
            elif self.toast_sender is default_toast_sender:
                pass
            self.sent_toasts += 1
            return True
        except Exception as e:
            log.warning("toast failed: %s", e)
            return False

    async def _send_webhook(self, url: str, event: EventRow) -> bool:
        payload = event_payload(event)
        for attempt in range(1, WEBHOOK_RETRIES + 1):
            try:
                result = self.webhook_sender(url, payload)
                if inspect.isawaitable(result):
                    await result
                self.sent_webhooks += 1
                return True
            except Exception as e:
                log.warning("webhook attempt %d/%d failed for event %d: %s", attempt, WEBHOOK_RETRIES, event.seq, e)
                if attempt < WEBHOOK_RETRIES:
                    await asyncio.sleep(0)
        return False


def threaded(sender: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """Wrap a blocking sender so the Notifier awaits it on a worker thread (the production wiring)."""
    async def run(*args: Any) -> Any:
        return await asyncio.to_thread(sender, *args)
    return run


__all__ = ["COALESCE_S", "WEBHOOK_TIMEOUT_S", "WEBHOOK_RETRIES", "MISSED_MAX_AGE_H", "MAX_DELIVERY_ATTEMPTS",
           "mail_options", "in_quiet_hours", "toast_title", "default_toast_sender", "default_webhook_sender",
           "event_payload", "Notifier", "threaded"]
