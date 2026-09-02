"""EventBus: durable event log with per-client deliver-then-ack cursors and long-poll wake-ups
(design sections 3.4, 5.6, 11g; the ``wait_for_events`` algorithm of section 4 "Events").

Per client: ``kv.cursor.<client_id>`` = the ack floor ``F`` (every ``seq <= F`` is acknowledged),
``event_acks(client_id, seq)`` for acknowledged seqs above ``F``, and ``deliveries(client_id)`` = the last
delivery ``{next_seq, seqs}``. Returning an event never consumes it; only ``ack(client_id, ack_seq)`` with the
``next_seq`` of a recorded delivery acknowledges exactly the seqs of that delivery.
Imports ``store`` and ``models`` only.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
import time
from typing import Any, Awaitable, Callable, Iterable, NamedTuple

from .models import EventRow
from .store import Store, _dumps

CURSOR_PREFIX = "cursor."
LAST_SESSION_KEY = "cursor.last_session"
DEFAULT_MAX_EVENTS = 50
DEFAULT_POLL_S = 30.0

ProgressCb = Callable[[int, float], Any]


class Delivery(NamedTuple):
    """Result of ``EventBus.read`` (design section 5.6 step 2)."""

    events: list[EventRow]
    delivered_seqs: list[int]
    next_seq: int | None
    unread_events: int
    unread_unmatched: int


class WaitResult(NamedTuple):
    """Result of ``EventBus.wait``: a ``Delivery`` plus the ack outcome and the timeout flag."""

    events: list[EventRow]
    delivered_seqs: list[int]
    next_seq: int | None
    unread_events: int
    unread_unmatched: int
    acked: int
    warnings: list[str]
    timed_out: bool


def _row_to_event(row: sqlite3.Row | dict[str, Any]) -> EventRow:
    payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
    return EventRow(seq=int(row["seq"]), ts=row["ts"], kind=row["kind"], handle=row["handle"], cluster=row["cluster"],
                    slurm_id=row["slurm_id"], summary=row["summary"] or "", payload=payload if isinstance(payload, dict) else {})


def _handle_base(handle: str | None) -> str | None:
    """``j18[7]`` -> ``j18``, ``a3.c2`` -> ``a3``."""
    if handle is None:
        return None
    return handle.split("[", 1)[0].split(".", 1)[0]


def matches(event: EventRow, kinds: Iterable[str] | None, job_ids: Iterable[str] | None) -> bool:
    """Apply the ``kinds``/``job_ids`` filters (``job_ids`` match the handle or its base handle)."""
    if kinds is not None:
        if event.kind not in set(kinds):
            return False
    if job_ids is not None:
        ids = set(job_ids)
        if event.handle not in ids and _handle_base(event.handle) not in ids:
            return False
    return True


class EventBus:
    """Appends events inside ledger transactions and long-polls them per client (design section 5.6)."""

    def __init__(self, store: Store, *, session_id: str | None = None, now: Callable[[], float] = time.time) -> None:
        self.store = store
        self.session_id = session_id
        self._now = now
        self._cond: asyncio.Condition | None = None
        self._version = 0

    # -- wake-ups ------------------------------------------------------------------------------------

    @property
    def cond(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    @property
    def version(self) -> int:
        """Bumped on every committed append in this process (waiters compare it to what they saw)."""
        return self._version

    async def notify_all(self) -> None:
        """Wake every ``wait`` in this process (called after the appending transaction committed)."""
        async with self.cond:
            self.cond.notify_all()

    def _on_commit(self) -> Awaitable[None] | None:
        self._version += 1
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return None
        return self.notify_all()

    # -- append (design section 3.4) -----------------------------------------------------------------

    def append(self, conn: sqlite3.Connection, kind: str, handle: str | None = None, cluster: str | None = None,
               slurm_id: str | None = None, summary: str = "", payload: dict[str, Any] | None = None,
               ts: int | None = None, *, state: Any = None, ts_local: float | None = None) -> int:
        """Insert an event inside the caller's (fenced) transaction; returns ``seq``.

        The payload always carries ``handle, cluster, slurm_id, state`` (filled from the arguments when the
        caller did not set them). ``notify_all`` runs after the transaction commits.
        """
        body = dict(payload or {})
        body.setdefault("handle", handle)
        body.setdefault("cluster", cluster)
        body.setdefault("slurm_id", slurm_id)
        if "state" not in body:
            body["state"] = None if state is None else str(state)
        cur = conn.execute(
            "INSERT INTO events(ts_local, ts, kind, handle, cluster, slurm_id, summary, payload_json) VALUES(?,?,?,?,?,?,?,?)",
            (self._now() if ts_local is None else ts_local, ts, kind, handle, cluster, slurm_id, summary, _dumps(body)))
        self.store.after_commit(self._on_commit)
        return int(cur.lastrowid)

    async def emit(self, kind: str, handle: str | None = None, cluster: str | None = None, slurm_id: str | None = None,
                   summary: str = "", payload: dict[str, Any] | None = None, ts: int | None = None, *,
                   token: int | None = None, state: Any = None) -> int:
        """One-event convenience: ``write`` (or ``write_fenced`` when ``token`` is given) + notify."""
        def fn(conn: sqlite3.Connection) -> int:
            return self.append(conn, kind, handle, cluster, slurm_id, summary, payload, ts, state=state)
        if token is not None:
            return await self.store.write_fenced(token, fn)
        return await self.store.write(fn)

    # -- cursors -------------------------------------------------------------------------------------

    def _client(self, client_id: str | None) -> str:
        cid = client_id or self.session_id
        if not cid:
            raise ValueError("client_id is required (no session_id configured)")
        return cid

    @staticmethod
    def max_seq(conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COALESCE(MAX(seq), 0) FROM events").fetchone()
        return int(row[0])

    def floor(self, conn: sqlite3.Connection, client_id: str, *, persist: bool = True) -> int:
        """The ack floor ``F`` of a client, initialising it on first use (design section 5.6 "Clients").

        A new client starts at ``max(seq)`` (it sees only later events); the session client starts from
        ``kv.cursor.last_session`` so events emitted while no server ran reach the next session.
        """
        key = CURSOR_PREFIX + client_id
        value = self.store.kv_get(conn, key)
        if value is not None:
            return int(value)
        if self.session_id is not None and client_id == self.session_id:
            floor = int(self.store.kv_get(conn, LAST_SESSION_KEY, 0))
        else:
            floor = self.max_seq(conn)
        if persist:
            self._set_floor(conn, client_id, floor)
        return floor

    def _set_floor(self, conn: sqlite3.Connection, client_id: str, floor: int) -> None:
        self.store.kv_set(conn, CURSOR_PREFIX + client_id, int(floor))
        if self.session_id is not None and client_id == self.session_id:
            self.store.kv_set(conn, LAST_SESSION_KEY, int(floor))

    def _acked_above(self, conn: sqlite3.Connection, client_id: str, floor: int) -> list[int]:
        return [int(r[0]) for r in conn.execute(
            "SELECT seq FROM event_acks WHERE client_id=? AND seq>? ORDER BY seq", (client_id, floor))]

    def _raise_floor(self, conn: sqlite3.Connection, client_id: str, floor: int) -> int:
        """Advance ``F`` through the contiguous acknowledged prefix and prune ``event_acks`` below it."""
        new_floor = floor
        for seq in self._acked_above(conn, client_id, floor):
            if seq == new_floor + 1:
                new_floor = seq
            else:
                break
        if new_floor != floor:
            self._set_floor(conn, client_id, new_floor)
            conn.execute("DELETE FROM event_acks WHERE client_id=? AND seq<=?", (client_id, new_floor))
        return new_floor

    # -- ack (design section 5.6 step 1) -------------------------------------------------------------

    def ack_sync(self, conn: sqlite3.Connection, client_id: str | None, ack_seq: int | None) -> tuple[int, list[str]]:
        """Acknowledge the delivery whose ``next_seq == ack_seq``; returns ``(acked, warnings)``.

        Never acknowledges by range: an ``ack_seq`` that matches no recorded delivery is ignored with a
        warning. Idempotent (a repeated ack counts 0).
        """
        if ack_seq is None:
            return 0, []
        cid = self._client(client_id)
        floor = self.floor(conn, cid)
        row = conn.execute("SELECT next_seq, seqs_json FROM deliveries WHERE client_id=?", (cid,)).fetchone()
        if row is None or int(row["next_seq"]) != int(ack_seq):
            known = None if row is None else int(row["next_seq"])
            return 0, [f"ack_seq={ack_seq} does not match the last delivery to client {cid!r}"
                       f" (next_seq={known}); nothing acknowledged"]
        seqs = [int(s) for s in json.loads(row["seqs_json"])]
        acked = 0
        ts = self._now()
        for seq in seqs:
            if seq <= floor:
                continue
            cur = conn.execute("INSERT OR IGNORE INTO event_acks(client_id, seq, acked_local) VALUES(?,?,?)", (cid, seq, ts))
            acked += cur.rowcount
        self._raise_floor(conn, cid, floor)
        return acked, []

    async def ack(self, client_id: str | None, ack_seq: int | None) -> tuple[int, list[str]]:
        return await self.store.write(lambda conn: self.ack_sync(conn, client_id, ack_seq))

    # -- read (design section 5.6 step 2) ------------------------------------------------------------

    def _unacked(self, conn: sqlite3.Connection, client_id: str, floor: int, *, start: int | None = None,
                 include_acked: bool = False) -> list[EventRow]:
        if include_acked:
            sql, params = "SELECT * FROM events WHERE seq>=? ORDER BY seq", [int(start or 0)]
        else:
            sql = ("SELECT e.* FROM events e LEFT JOIN event_acks a ON a.client_id=? AND a.seq=e.seq "
                   "WHERE e.seq>? AND a.seq IS NULL ORDER BY e.seq")
            params = [client_id, max(floor, (start or 0) - 1)]
        return [_row_to_event(r) for r in conn.execute(sql, params)]

    def read_sync(self, conn: sqlite3.Connection, client_id: str | None = None, *, since_seq: int | None = None,
                  kinds: Iterable[str] | None = None, job_ids: Iterable[str] | None = None,
                  max_events: int = DEFAULT_MAX_EVENTS, include_acked: bool = False) -> Delivery:
        """Deliver unacknowledged events ``>= since_seq or F+1`` matching the filters (never consumes).

        When events are returned the delivery ``{next_seq: max(seq)+1, seqs}`` is recorded so a later
        ``ack(next_seq)`` acknowledges exactly these. ``unread_events`` counts every unacknowledged event of
        the client regardless of filters/since_seq; ``unread_unmatched`` those hidden by the filters.
        """
        cid = self._client(client_id)
        floor = self.floor(conn, cid)
        all_unacked = self._unacked(conn, cid, floor)
        kinds_l = list(kinds) if kinds is not None else None
        ids_l = list(job_ids) if job_ids is not None else None
        unread_events = len(all_unacked)
        unread_unmatched = sum(1 for e in all_unacked if not matches(e, kinds_l, ids_l))
        start = floor + 1 if since_seq is None else int(since_seq)
        if include_acked:
            candidates = self._unacked(conn, cid, floor, start=start, include_acked=True)
        else:
            candidates = [e for e in all_unacked if e.seq >= start]
        selected = [e for e in candidates if matches(e, kinds_l, ids_l)][:max(0, int(max_events))]
        if not selected:
            return Delivery([], [], None, unread_events, unread_unmatched)
        seqs = [e.seq for e in selected]
        next_seq = max(seqs) + 1
        conn.execute("INSERT INTO deliveries(client_id, next_seq, seqs_json, delivered_local) VALUES(?,?,?,?) "
                     "ON CONFLICT(client_id) DO UPDATE SET next_seq=excluded.next_seq, seqs_json=excluded.seqs_json, "
                     "delivered_local=excluded.delivered_local", (cid, next_seq, _dumps(seqs), self._now()))
        return Delivery(selected, seqs, next_seq, unread_events, unread_unmatched)

    async def read(self, client_id: str | None = None, *, since_seq: int | None = None, kinds: Iterable[str] | None = None,
                   job_ids: Iterable[str] | None = None, max_events: int = DEFAULT_MAX_EVENTS,
                   include_acked: bool = False) -> Delivery:
        return await self.store.write(lambda conn: self.read_sync(
            conn, client_id, since_seq=since_seq, kinds=kinds, job_ids=job_ids, max_events=max_events,
            include_acked=include_acked))

    def unread_sync(self, conn: sqlite3.Connection, client_id: str | None = None) -> int:
        """``unread_events`` for every tool response (design rule 1): unacknowledged events of the client.

        Does not persist a cursor for an unknown client (a pure read)."""
        cid = self._client(client_id)
        floor = self.floor(conn, cid, persist=False)
        row = conn.execute(
            "SELECT COUNT(*) FROM events e LEFT JOIN event_acks a ON a.client_id=? AND a.seq=e.seq WHERE e.seq>? AND a.seq IS NULL",
            (cid, floor)).fetchone()
        return int(row[0])

    async def unread(self, client_id: str | None = None) -> int:
        return await self.store.read(lambda conn: self.unread_sync(conn, client_id))

    # -- wait (long-poll, design section 4 "wait_for_events") ----------------------------------------

    async def wait(self, client_id: str | None = None, *, timeout_s: float = 300, kinds: Iterable[str] | None = None,
                   job_ids: Iterable[str] | None = None, since_seq: int | None = None, ack_seq: int | None = None,
                   max_events: int = DEFAULT_MAX_EVENTS, progress_cb: ProgressCb | None = None,
                   poll_s: float = DEFAULT_POLL_S) -> WaitResult:
        """Ack (step 1), then deliver matching unacknowledged events or block until one is appended or
        ``timeout_s`` elapses; ``progress_cb(i, elapsed_s)`` is called every ``poll_s`` while waiting.

        Cross-process appends (the CLI) are not signalled through the Condition, so the log is re-read at
        every ``poll_s`` tick as well.
        """
        cid = self._client(client_id)
        acked, warnings = await self.ack(cid, ack_seq) if ack_seq is not None else (0, [])
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        started = time.monotonic()
        i = 0
        while True:
            seen = self._version
            delivery = await self.read(cid, since_seq=since_seq, kinds=kinds, job_ids=job_ids, max_events=max_events)
            if delivery.events:
                return WaitResult(*delivery, acked, warnings, False)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return WaitResult(*delivery, acked, warnings, timeout_s > 0)
            woke = await self._wait_for_change(seen, min(remaining, poll_s))
            if not woke:
                i += 1
                if progress_cb is not None:
                    result = progress_cb(i, time.monotonic() - started)
                    if inspect.isawaitable(result):
                        await result

    async def _wait_for_change(self, seen: int, timeout: float) -> bool:
        """True when an append happened since ``seen``; False on timeout."""
        async with self.cond:
            try:
                await asyncio.wait_for(self.cond.wait_for(lambda: self._version != seen), timeout=max(0.0, timeout))
                return True
            except asyncio.TimeoutError:
                return False

    # -- listing / notifications (notify.py, job_status) ---------------------------------------------

    def events_for_sync(self, conn: sqlite3.Connection, handle: str | None = None, *, kinds: Iterable[str] | None = None,
                        since_seq: int | None = None, limit: int | None = None) -> list[EventRow]:
        """Plain log listing (no cursor involvement), newest last."""
        conds: list[str] = []
        params: list[Any] = []
        if handle is not None:
            conds.append("handle=?")
            params.append(handle)
        if kinds is not None:
            ks = list(kinds)
            conds.append(f"kind IN ({', '.join('?' for _ in ks)})" if ks else "0")
            params.extend(ks)
        if since_seq is not None:
            conds.append("seq>=?")
            params.append(int(since_seq))
        sql = "SELECT * FROM events" + (" WHERE " + " AND ".join(conds) if conds else "") + " ORDER BY seq"
        if limit is not None:
            sql = f"SELECT * FROM ({sql.replace('ORDER BY seq', 'ORDER BY seq DESC')} LIMIT {int(limit)}) ORDER BY seq"
        return [_row_to_event(r) for r in conn.execute(sql, params)]

    async def events_for(self, handle: str | None = None, *, kinds: Iterable[str] | None = None,
                         since_seq: int | None = None, limit: int | None = None) -> list[EventRow]:
        return await self.store.read(lambda conn: self.events_for_sync(conn, handle, kinds=kinds, since_seq=since_seq, limit=limit))

    def unnotified_sync(self, conn: sqlite3.Connection, kinds: Iterable[str] | None = None, max_age_h: float = 24,
                        *, now: float | None = None) -> list[EventRow]:
        """Events with ``notified=0`` younger than ``max_age_h`` (design section 5.6 "Missed events at startup")."""
        ts = self._now() if now is None else now
        conds = ["notified=0", "ts_local>=?"]
        params: list[Any] = [ts - float(max_age_h) * 3600.0]
        if kinds is not None:
            ks = list(kinds)
            conds.append(f"kind IN ({', '.join('?' for _ in ks)})" if ks else "0")
            params.extend(ks)
        sql = "SELECT * FROM events WHERE " + " AND ".join(conds) + " ORDER BY seq"
        return [_row_to_event(r) for r in conn.execute(sql, params)]

    async def unnotified(self, kinds: Iterable[str] | None = None, max_age_h: float = 24) -> list[EventRow]:
        return await self.store.read(lambda conn: self.unnotified_sync(conn, kinds, max_age_h))

    @staticmethod
    def mark_notified_sync(conn: sqlite3.Connection, seqs: Iterable[int]) -> int:
        """Set ``events.notified=1`` for the given seqs (so restarts never re-toast); returns rows updated."""
        ids = [int(s) for s in seqs]
        if not ids:
            return 0
        return conn.execute(f"UPDATE events SET notified=1 WHERE seq IN ({', '.join('?' for _ in ids)})", ids).rowcount

    async def mark_notified(self, seqs: Iterable[int]) -> int:
        ids = list(seqs)
        return await self.store.write(lambda conn: self.mark_notified_sync(conn, ids))


__all__ = ["CURSOR_PREFIX", "LAST_SESSION_KEY", "DEFAULT_MAX_EVENTS", "DEFAULT_POLL_S", "Delivery", "WaitResult",
           "matches", "EventBus"]
