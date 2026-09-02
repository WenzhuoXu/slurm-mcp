"""Unit tests for slurm_mcp.notify (design section 5.6) with injected senders and a temp Store."""
from __future__ import annotations

import pytest

from slurm_mcp.events import EventBus
from slurm_mcp.models import NotifyPolicy
from slurm_mcp.notify import Notifier, in_quiet_hours, mail_options, toast_title
from slurm_mcp.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "state.db", pid=1, host="lap", pid_exists=lambda pid: True)
    yield s
    s.close()


@pytest.fixture
def bus(store):
    return EventBus(store, session_id="sess")


def emit(bus: EventBus, kind: str, handle: str = "j1", cluster: str = "trace", state: str | None = "COMPLETED",
         summary: str | None = None, **payload) -> int:
    return bus.store.write_sync(lambda c: bus.append(c, kind, handle, cluster, "42", summary or f"{handle} {kind}",
                                                     payload, ts=1700000000, state=state))


class Recorder:
    def __init__(self, fail_times: int = 0) -> None:
        self.toasts: list[tuple[str, str]] = []
        self.webhooks: list[tuple[str, dict]] = []
        self.fail_times = fail_times

    def toast(self, title: str, body: str) -> None:
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("toast broken")
        self.toasts.append((title, body))

    async def webhook(self, url: str, payload: dict) -> None:
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("webhook down")
        self.webhooks.append((url, payload))


def notifier(store, bus, policy: NotifyPolicy, rec: Recorder, clock: list[float], hour: int = 12) -> Notifier:
    return Notifier(store, bus, lambda: policy, toast_sender=rec.toast, webhook_sender=rec.webhook,
                    now=lambda: clock[0], local_hour=lambda: hour)


def test_mail_options_and_quiet_hours():
    assert mail_options(NotifyPolicy()) == []
    assert mail_options(NotifyPolicy(email="me@x.org")) == ["--mail-type=END,FAIL,REQUEUE,TIME_LIMIT_90",
                                                             "--mail-user=me@x.org"]
    assert mail_options(None) == []
    p = NotifyPolicy(quiet_hours=(22, 7))
    assert in_quiet_hours(p, 23) and in_quiet_hours(p, 3) and not in_quiet_hours(p, 12)
    assert in_quiet_hours(NotifyPolicy(quiet_hours=(9, 17)), 10) and not in_quiet_hours(NotifyPolicy(), 10)
    assert not in_quiet_hours(NotifyPolicy(quiet_hours=(5, 5)), 5)


@pytest.mark.asyncio
async def test_single_toast_marks_event_notified(store, bus):
    rec = Recorder()
    clock = [1000.0]
    n = notifier(store, bus, NotifyPolicy(), rec, clock)
    seq = emit(bus, "completed", summary="j1 finished rc=0")
    emit(bus, "started", state="RUNNING")          # not a toast kind by default
    stats = await n.after_tick()
    assert stats["toasts"] == 1 and stats["marked"] == 1
    assert rec.toasts == [("slurm j1 COMPLETED (trace)", "j1 finished rc=0")]
    left = await bus.unnotified()
    assert [e.kind for e in left] == ["started"]
    assert (await n.after_tick())["toasts"] == 0 and len(rec.toasts) == 1
    assert toast_title(left[0]) == "slurm j1 RUNNING (trace)"


@pytest.mark.asyncio
async def test_coalescing_within_10s_and_startup_batch(store, bus):
    rec = Recorder()
    clock = [1000.0]
    n = notifier(store, bus, NotifyPolicy(), rec, clock)
    emit(bus, "completed", handle="j1")
    assert (await n.after_tick())["toasts"] == 1
    emit(bus, "failed", handle="j2", state="FAILED")
    clock[0] = 1005.0
    stats = await n.after_tick()
    assert stats["deferred"] == 1 and stats["toasts"] == 0 and len(rec.toasts) == 1
    emit(bus, "timeout", handle="j3", state="TIMEOUT")
    clock[0] = 1011.0
    stats = await n.after_tick()
    assert stats["toasts"] == 1 and stats["marked"] == 2
    assert rec.toasts[-1][0] == "slurm: 2 events" and "j2 failed" in rec.toasts[-1][1] and "j3 timeout" in rec.toasts[-1][1]
    # startup: everything missed comes as one toast regardless of the window
    for h in ("j4", "j5", "j6"):
        emit(bus, "cancelled", handle=h, state="CANCELLED")
    clock[0] = 1012.0
    await n.start()
    assert rec.toasts[-1][0] == "slurm: 3 events" and await bus.unnotified() == []


@pytest.mark.asyncio
async def test_quiet_hours_suppress_and_disabled_toast(store, bus):
    rec = Recorder()
    clock = [1000.0]
    n = notifier(store, bus, NotifyPolicy(quiet_hours=(22, 7)), rec, clock, hour=23)
    emit(bus, "completed")
    stats = await n.after_tick()
    assert stats["suppressed"] == 1 and stats["marked"] == 1 and rec.toasts == []
    n2 = notifier(store, bus, NotifyPolicy(toast=False), rec, clock)
    emit(bus, "completed", handle="j9")
    assert (await n2.after_tick()) == {"toasts": 0, "webhooks": 0, "suppressed": 0, "marked": 0, "deferred": 0}
    assert len(await bus.unnotified()) == 1


@pytest.mark.asyncio
async def test_webhook_retries_and_kinds(store, bus):
    rec = Recorder(fail_times=2)
    clock = [1000.0]
    pol = NotifyPolicy(toast=False, webhook_url="http://hook.example/x", webhook_kinds=["completed"])
    n = notifier(store, bus, pol, rec, clock)
    emit(bus, "completed", summary="done")
    emit(bus, "failed", state="FAILED")
    stats = await n.after_tick()
    assert stats["webhooks"] == 1 and stats["marked"] == 1
    url, payload = rec.webhooks[0]
    assert url == pol.webhook_url and payload["kind"] == "completed" and payload["payload"]["state"] == "COMPLETED"
    assert [e.kind for e in await bus.unnotified()] == ["failed"]
    # a webhook that keeps failing is retried on later ticks, then given up after 3 attempts
    rec2 = Recorder(fail_times=100)
    n2 = notifier(store, bus, NotifyPolicy(toast=False, webhook_url="http://hook.example/x", webhook_kinds=["failed"]),
                  rec2, clock)
    for _ in range(3):
        stats = await n2.after_tick()
    assert stats["marked"] == 1 and await bus.unnotified() == []


@pytest.mark.asyncio
async def test_toast_failure_keeps_event_for_retry(store, bus):
    rec = Recorder(fail_times=1)
    clock = [1000.0]
    n = notifier(store, bus, NotifyPolicy(), rec, clock)
    emit(bus, "completed")
    assert (await n.after_tick())["toasts"] == 0 and len(await bus.unnotified()) == 1
    clock[0] = 1020.0
    assert (await n.after_tick())["toasts"] == 1 and await bus.unnotified() == []


@pytest.mark.asyncio
async def test_async_policy_getter_and_dict(store, bus):
    rec = Recorder()

    async def getter():
        return {"toast_kinds": ["oom"]}

    n = Notifier(store, bus, getter, toast_sender=rec.toast, webhook_sender=rec.webhook)
    emit(bus, "oom", state="OOM")
    assert (await n.after_tick())["toasts"] == 1
