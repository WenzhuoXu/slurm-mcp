"""Unit tests for slurm_mcp.events (design sections 3.4, 5.6, 11g; section 4 wait_for_events)."""
from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from slurm_mcp.events import LAST_SESSION_KEY, Delivery, EventBus, WaitResult, matches
from slurm_mcp.models import EventRow
from slurm_mcp.store import LeaseLost, Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "state.db", pid=1, host="lap", pid_exists=lambda pid: True)
    yield s
    s.close()


@pytest.fixture
def bus(store):
    return EventBus(store, session_id="sess1")


def emit(bus: EventBus, kind: str, handle: str | None = "j1", **payload) -> int:
    """Synchronous append through a write transaction (no loop needed)."""
    return bus.store.write_sync(lambda c: bus.append(c, kind, handle, "trace", "42", f"{handle} {kind}", payload, ts=1700000000))


# --- append ----------------------------------------------------------------------------------------

def test_append_returns_monotonic_seq_and_fills_payload(bus, store):
    s1 = emit(bus, "submitted", state="SUBMITTED", target="trace:gpu")
    s2 = emit(bus, "started")
    assert (s1, s2) == (1, 2)
    rows = store.read_sync(lambda c: bus.events_for_sync(c))
    assert [r.seq for r in rows] == [1, 2]
    assert rows[0].payload == {"handle": "j1", "cluster": "trace", "slurm_id": "42", "state": "SUBMITTED", "target": "trace:gpu"}
    assert rows[1].payload == {"handle": "j1", "cluster": "trace", "slurm_id": "42", "state": None}
    assert rows[0].ts == 1700000000 and rows[0].summary == "j1 submitted" and isinstance(rows[0], EventRow)
    raw = store.read_sync(lambda c: c.execute("SELECT ts_local, notified FROM events WHERE seq=1").fetchone())
    assert raw[0] > 0 and raw[1] == 0


def test_append_state_argument_and_version(bus, store):
    v0 = bus.version
    store.write_sync(lambda c: bus.append(c, "completed", "j1", state="COMPLETED"))
    assert bus.version == v0 + 1
    assert store.read_sync(lambda c: bus.events_for_sync(c))[0].payload["state"] == "COMPLETED"


def test_append_outside_transaction_is_refused(bus, store):
    conn = store._conn
    with pytest.raises(RuntimeError):
        bus.append(conn, "x")


# --- cursor initialisation (section 5.6 "Clients") ----------------------------------------------------

def test_new_client_sees_only_events_after_first_use(bus, store):
    emit(bus, "submitted")
    emit(bus, "started")
    d = store.write_sync(lambda c: bus.read_sync(c, "cli"))
    assert d == Delivery([], [], None, 0, 0)
    assert store.read_sync(lambda c: store.kv_get(c, "cursor.cli")) == 2
    emit(bus, "completed")
    d = store.write_sync(lambda c: bus.read_sync(c, "cli"))
    assert d.delivered_seqs == [3] and d.next_seq == 4 and d.unread_events == 1


def test_session_client_starts_from_last_session(bus, store):
    emit(bus, "submitted")
    store.write_sync(lambda c: store.kv_set(c, LAST_SESSION_KEY, 1))
    emit(bus, "completed", observed_late=True)          # emitted "while no server ran"
    d = store.write_sync(lambda c: bus.read_sync(c, "sess1"))
    assert d.delivered_seqs == [2]
    store.write_sync(lambda c: bus.ack_sync(c, "sess1", d.next_seq))
    assert store.read_sync(lambda c: store.kv_get(c, LAST_SESSION_KEY)) == 2
    assert store.read_sync(lambda c: store.kv_get(c, "cursor.sess1")) == 2
    # default client_id is the session id
    assert store.write_sync(lambda c: bus.read_sync(c)).unread_events == 0


def test_client_id_required_without_session(store):
    b = EventBus(store)
    with pytest.raises(ValueError):
        store.write_sync(lambda c: b.read_sync(c))


# --- deliver-then-ack (section 5.6) --------------------------------------------------------------------

def test_returning_never_consumes_and_ack_needs_matching_seq(bus, store):
    store.write_sync(lambda c: bus.read_sync(c, "cli"))
    emit(bus, "submitted")
    emit(bus, "started")
    d1 = store.write_sync(lambda c: bus.read_sync(c, "cli"))
    d2 = store.write_sync(lambda c: bus.read_sync(c, "cli"))
    assert d1.delivered_seqs == d2.delivered_seqs == [1, 2] and d1.next_seq == 3
    assert [e.seq for e in d2.events] == [1, 2]
    # a stale / wrong ack_seq is ignored with a warning and acknowledges nothing
    acked, warnings = store.write_sync(lambda c: bus.ack_sync(c, "cli", 99))
    assert acked == 0 and len(warnings) == 1 and "99" in warnings[0]
    assert store.write_sync(lambda c: bus.read_sync(c, "cli")).delivered_seqs == [1, 2]
    # the matching ack acknowledges exactly the delivery, raises the floor and prunes event_acks
    acked, warnings = store.write_sync(lambda c: bus.ack_sync(c, "cli", 3))
    assert (acked, warnings) == (2, [])
    assert store.read_sync(lambda c: store.kv_get(c, "cursor.cli")) == 2
    assert store.read_sync(lambda c: store.count(c, "event_acks")) == 0
    assert store.write_sync(lambda c: bus.read_sync(c, "cli")) == Delivery([], [], None, 0, 0)
    # idempotent: acking again counts 0 and warns nothing
    assert store.write_sync(lambda c: bus.ack_sync(c, "cli", 3)) == (0, [])
    assert store.write_sync(lambda c: bus.ack_sync(c, "cli", None)) == (0, [])


def test_same_since_seq_redelivers_and_does_not_ack(bus, store):
    store.write_sync(lambda c: bus.read_sync(c, "cli"))
    for k in ("submitted", "started", "completed"):
        emit(bus, k)
    a = store.write_sync(lambda c: bus.read_sync(c, "cli", since_seq=2))
    b = store.write_sync(lambda c: bus.read_sync(c, "cli", since_seq=2))
    assert a.delivered_seqs == b.delivered_seqs == [2, 3] and a.next_seq == 4
    assert a.unread_events == 3
    # since_seq without ack acknowledges nothing: a floor-based read still returns everything
    assert store.write_sync(lambda c: bus.read_sync(c, "cli")).delivered_seqs == [1, 2, 3]


def test_filters_keep_unread_accurate_and_never_hide_permanently(bus, store):
    store.write_sync(lambda c: bus.read_sync(c, "cli"))
    emit(bus, "submitted", "j17")           # 1
    emit(bus, "alloc_ready", "a3")          # 2
    emit(bus, "started", "j17")             # 3
    emit(bus, "cmd_done", "a3.c2")          # 4 (handle base a3)
    emit(bus, "completed", "j18[7]")        # 5 (handle base j18)
    d = store.write_sync(lambda c: bus.read_sync(c, "cli", job_ids=["j17"]))
    assert d.delivered_seqs == [1, 3] and d.unread_events == 5 and d.unread_unmatched == 3
    d = store.write_sync(lambda c: bus.read_sync(c, "cli", job_ids=["a3"]))
    assert d.delivered_seqs == [2, 4]
    d = store.write_sync(lambda c: bus.read_sync(c, "cli", kinds=["completed"], job_ids=["j18"]))
    assert d.delivered_seqs == [5] and d.unread_unmatched == 4
    # ack the filtered j17 delivery: floor stays below the unacked alloc_ready(a3)
    d = store.write_sync(lambda c: bus.read_sync(c, "cli", job_ids=["j17"]))
    acked, _ = store.write_sync(lambda c: bus.ack_sync(c, "cli", d.next_seq))
    assert acked == 2
    assert store.read_sync(lambda c: store.kv_get(c, "cursor.cli")) == 1        # 1 acked -> floor 1; 2 unacked
    assert store.read_sync(lambda c: store.count(c, "event_acks", client_id="cli")) == 1   # seq 3 kept above the floor
    d = store.write_sync(lambda c: bus.read_sync(c, "cli"))
    assert d.delivered_seqs == [2, 4, 5] and d.unread_events == 3 and d.unread_unmatched == 0
    store.write_sync(lambda c: bus.ack_sync(c, "cli", d.next_seq))
    assert store.read_sync(lambda c: store.kv_get(c, "cursor.cli")) == 5
    assert store.read_sync(lambda c: store.count(c, "event_acks")) == 0
    assert store.read_sync(lambda c: bus.unread_sync(c, "cli")) == 0


def test_max_events_truncation_then_rest_after_ack(bus, store):
    store.write_sync(lambda c: bus.read_sync(c, "cli"))
    for i in range(7):
        emit(bus, "started", f"j{i}")
    d = store.write_sync(lambda c: bus.read_sync(c, "cli", max_events=3))
    assert d.delivered_seqs == [1, 2, 3] and d.next_seq == 4 and d.unread_events == 7 and d.unread_unmatched == 0
    store.write_sync(lambda c: bus.ack_sync(c, "cli", 4))
    d = store.write_sync(lambda c: bus.read_sync(c, "cli", max_events=3))
    assert d.delivered_seqs == [4, 5, 6] and d.unread_events == 4
    assert store.write_sync(lambda c: bus.read_sync(c, "cli", max_events=0)) == Delivery([], [], None, 4, 0)


def test_overwritten_delivery_makes_old_ack_unknown(bus, store):
    store.write_sync(lambda c: bus.read_sync(c, "cli"))
    emit(bus, "started")
    d1 = store.write_sync(lambda c: bus.read_sync(c, "cli"))
    emit(bus, "completed")
    d2 = store.write_sync(lambda c: bus.read_sync(c, "cli"))
    assert d1.next_seq == 2 and d2.next_seq == 3
    assert store.write_sync(lambda c: bus.ack_sync(c, "cli", d1.next_seq))[0] == 0     # superseded delivery
    assert store.write_sync(lambda c: bus.ack_sync(c, "cli", d2.next_seq))[0] == 2


def test_include_acked_lists_history(bus, store):
    store.write_sync(lambda c: bus.read_sync(c, "cli"))
    emit(bus, "started")
    d = store.write_sync(lambda c: bus.read_sync(c, "cli"))
    store.write_sync(lambda c: bus.ack_sync(c, "cli", d.next_seq))
    assert store.write_sync(lambda c: bus.read_sync(c, "cli", since_seq=1)).events == []
    hist = store.write_sync(lambda c: bus.read_sync(c, "cli", since_seq=1, include_acked=True))
    assert hist.delivered_seqs == [1] and hist.unread_events == 0


def test_unread_does_not_persist_cursor(bus, store):
    emit(bus, "started")
    assert store.read_sync(lambda c: bus.unread_sync(c, "fresh")) == 0
    assert store.read_sync(lambda c: store.kv_get(c, "cursor.fresh")) is None
    store.write_sync(lambda c: store.kv_set(c, LAST_SESSION_KEY, 0))
    assert store.read_sync(lambda c: bus.unread_sync(c, "sess1")) == 1


def test_matches_helper():
    e = EventRow(seq=1, kind="cmd_done", handle="a3.c2")
    assert matches(e, None, None) and matches(e, ["cmd_done"], ["a3"]) and matches(e, None, ["a3.c2"])
    assert not matches(e, ["started"], None) and not matches(e, None, ["j1"])
    assert not matches(EventRow(seq=2, kind="cluster_unreachable"), None, ["j1"])


# --- async API, long-poll ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_read_ack_async(bus, store):
    await bus.read("cli")
    seq = await bus.emit("submitted", "j1", "trace", "7", "s", {"target": "t"}, ts=5, state="SUBMITTED")
    assert seq == 1
    d = await bus.read("cli")
    assert d.delivered_seqs == [1] and d.events[0].payload["state"] == "SUBMITTED"
    assert await bus.ack("cli", d.next_seq) == (1, [])
    assert await bus.unread("cli") == 0
    assert [e.seq for e in await bus.events_for("j1")] == [1]


@pytest.mark.asyncio
async def test_emit_fenced_raises_lease_lost(bus, store):
    info = await store.write(lambda c: store.lease_acquire(c))
    await bus.emit("started", "j1", token=info.token)
    with pytest.raises(LeaseLost):
        await bus.emit("completed", "j1", token=info.token + 1)
    assert [e.kind for e in await bus.events_for("j1")] == ["started"]


@pytest.mark.asyncio
async def test_wait_wakes_within_100ms_of_append(bus, store):
    await bus.read("cli")
    task = asyncio.create_task(bus.wait("cli", timeout_s=5, poll_s=10))
    await asyncio.sleep(0.05)
    assert not task.done()
    t0 = time.monotonic()
    await bus.emit("completed", "j1", "trace", "42", "done")
    result = await asyncio.wait_for(task, timeout=2)
    assert time.monotonic() - t0 < 0.1
    assert isinstance(result, WaitResult)
    assert result.delivered_seqs == [1] and result.next_seq == 2 and not result.timed_out and result.acked == 0


@pytest.mark.asyncio
async def test_wait_times_out_and_reports_progress(bus):
    await bus.read("cli")
    progress: list[tuple[int, float]] = []

    async def cb(i: int, elapsed: float) -> None:
        progress.append((i, elapsed))

    t0 = time.monotonic()
    result = await bus.wait("cli", timeout_s=0.3, poll_s=0.05, progress_cb=cb)
    assert 0.25 <= time.monotonic() - t0 < 1.5
    assert result.timed_out and result.events == [] and result.next_seq is None
    assert len(progress) >= 3 and progress[0][0] == 1 and progress[-1][0] == len(progress)


@pytest.mark.asyncio
async def test_wait_zero_timeout_lists_without_blocking(bus):
    await bus.read("cli")
    result = await bus.wait("cli", timeout_s=0)
    assert not result.timed_out and result.events == []
    await bus.emit("started", "j1")
    result = await bus.wait("cli", timeout_s=0, progress_cb=lambda i, e: None)
    assert result.delivered_seqs == [1]


@pytest.mark.asyncio
async def test_wait_acks_previous_delivery_first(bus):
    await bus.read("cli")
    await bus.emit("started", "j1")
    d = await bus.read("cli")
    await bus.emit("completed", "j1")
    result = await bus.wait("cli", timeout_s=1, ack_seq=d.next_seq)
    assert result.acked == 1 and result.warnings == [] and result.delivered_seqs == [2]
    result = await bus.wait("cli", timeout_s=0, ack_seq=999)
    assert result.acked == 0 and len(result.warnings) == 1 and result.delivered_seqs == [2]


@pytest.mark.asyncio
async def test_filtered_wait_ignores_unmatched_but_counts_them(bus):
    await bus.read("cli")
    task = asyncio.create_task(bus.wait("cli", timeout_s=0.5, poll_s=0.05, kinds=["completed"]))
    await asyncio.sleep(0.05)
    await bus.emit("started", "j1")
    result = await task
    assert result.timed_out and result.events == [] and result.unread_events == 1 and result.unread_unmatched == 1


@pytest.mark.asyncio
async def test_wait_picks_up_append_from_another_process_on_poll(bus, store, tmp_path):
    other = Store(tmp_path / "state.db", pid=2)
    other_bus = EventBus(other)
    try:
        await bus.read("cli")
        task = asyncio.create_task(bus.wait("cli", timeout_s=3, poll_s=0.05))
        await asyncio.sleep(0.05)
        other.write_sync(lambda c: other_bus.append(c, "started", "j1"))      # no Condition wake in this process
        result = await asyncio.wait_for(task, timeout=2)
        assert result.delivered_seqs == [1]
    finally:
        other.close()


# --- notifications (notify.py) -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unnotified_and_mark_notified(bus, store):
    old = await store.write(lambda c: bus.append(c, "completed", "j1", ts_local=time.time() - 30 * 3600))
    s2 = await bus.emit("failed", "j2")
    s3 = await bus.emit("started", "j3")
    assert [e.seq for e in await bus.unnotified()] == [s2, s3]
    assert [e.seq for e in await bus.unnotified(kinds=["completed", "failed"], max_age_h=48)] == [old, s2]
    assert await bus.mark_notified([s2]) == 1
    assert await bus.mark_notified([]) == 0
    assert [e.seq for e in await bus.unnotified()] == [s3]
    assert [e.seq for e in await bus.events_for(kinds=["failed"])] == [s2]
    assert [e.seq for e in await bus.events_for(limit=2)] == [s2, s3]
    assert [e.seq for e in await bus.events_for(since_seq=s3)] == [s3]


# --- property: acking every delivery in order acknowledges everything exactly once -----------------------

@settings(max_examples=25, deadline=None)
@given(st.lists(st.tuples(st.sampled_from(["started", "completed", "failed"]), st.sampled_from(["j1", "j2", "a3"])),
                min_size=1, max_size=12),
       st.integers(min_value=1, max_value=4))
def test_ack_in_order_drains_the_log(kinds, max_events):
    tmp = tempfile.mkdtemp(prefix="slurm-mcp-events-")
    with Store(Path(tmp) / "state.db") as store:
        bus = EventBus(store)
        store.write_sync(lambda c: bus.read_sync(c, "cli"))
        for kind, handle in kinds:
            store.write_sync(lambda c, k=kind, h=handle: bus.append(c, k, h))
        seen: list[int] = []
        ack_seq = None
        while True:
            d = store.write_sync(lambda c: bus.read_sync(c, "cli", max_events=max_events))
            if ack_seq is not None:
                assert d.delivered_seqs[:1] != [ack_seq - 1]  # acked events are gone
            if not d.events:
                break
            seen.extend(d.delivered_seqs)
            acked, warnings = store.write_sync(lambda c: bus.ack_sync(c, "cli", d.next_seq))
            ack_seq = d.next_seq
            assert acked == len(d.delivered_seqs) and warnings == []
        assert seen == list(range(1, len(kinds) + 1))
        assert store.read_sync(lambda c: store.kv_get(c, "cursor.cli")) == len(kinds)
        assert store.read_sync(lambda c: store.count(c, "event_acks")) == 0
        assert store.read_sync(lambda c: bus.unread_sync(c, "cli")) == 0
