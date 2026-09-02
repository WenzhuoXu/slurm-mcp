"""SQLite ledger: schema, migrations, fenced transactions, monitor lease and typed DAO helpers
(design sections 3.3, 5.8, 11c, 11i, 9.2 "SQLite locked / corrupt").

Every write is ``BEGIN IMMEDIATE; fn(conn); COMMIT`` executed on a worker thread (``asyncio.to_thread``)
under an in-process ``threading.Lock`` that only serialises this process; cross-process safety comes from
SQLite's writer lock (``busy_timeout=5000``) and the fencing token of the ``lease`` row. Handles and plan
ids are allocated from ``kv`` counters inside the same transaction as the row they name.

DAO helpers take the transaction's ``conn`` as their first argument so they compose inside ``write``,
``write_fenced`` and ``read``; the ``*_sync`` variants exist for the CLI (no event loop).
Imports nothing from the package.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import socket
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TypeVar

import psutil

T = TypeVar("T")

LEASE_NAME = "monitor"
LEASE_STALE_S = 300.0                 # section 5.8: renewed_local older than 5 min
PLAN_TTL_S = 900.0                    # section 4: plans are valid for 15 min
SCHEMA_VERSION = 1

# --- section 3.3 schema (column names verbatim) -------------------------------------------------------

SCHEMA_V1 = """
CREATE TABLE jobs(
  handle TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  state TEXT NOT NULL, slurm_state TEXT, reason TEXT,
  spec_json TEXT NOT NULL, placement_mode TEXT NOT NULL,
  attempt_no INTEGER NOT NULL DEFAULT 1,
  submit_ts INTEGER, start_ts INTEGER, end_ts INTEGER, est_start_ts INTEGER,
  exit_code INTEGER, exit_signal INTEGER, restarts INTEGER NOT NULL DEFAULT 0, moves INTEGER NOT NULL DEFAULT 0,
  cost_est_su REAL, cost_worst_su REAL, cost_actual_su REAL,
  last_seen_ts INTEGER, stale_ticks INTEGER NOT NULL DEFAULT 0, terminal_ts INTEGER, enriched INTEGER NOT NULL DEFAULT 0,
  collected_ts INTEGER, cancel_requested_ts INTEGER, cancel_hard_ts INTEGER, hold_reason TEXT,
  alloc_ready INTEGER NOT NULL DEFAULT 0, alloc_end_ts INTEGER, array_size INTEGER, depends_on_json TEXT,
  heartbeat_ts INTEGER, progress_json TEXT, last_line TEXT,
  created_local REAL NOT NULL, updated_local REAL NOT NULL);
CREATE TABLE attempts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, handle TEXT NOT NULL REFERENCES jobs(handle), attempt_no INTEGER NOT NULL,
  cluster TEXT NOT NULL, token TEXT NOT NULL UNIQUE,
  slurm_id TEXT,
  ctrl_root TEXT NOT NULL, ctrl_dir TEXT NOT NULL,
  workdir TEXT NOT NULL,
  stdout_pattern TEXT, stderr_pattern TEXT,
  stdout_path TEXT, stderr_path TEXT,
  node TEXT,
  target_json TEXT NOT NULL, submit_line TEXT, state TEXT NOT NULL,
  cause TEXT NOT NULL,
  intent_local REAL NOT NULL, invoked_local REAL, confirmed_local REAL, submit_ts INTEGER, end_ts INTEGER,
  final_state TEXT, exit_code INTEGER, reason TEXT, excluded_nodes TEXT,
  UNIQUE(handle, attempt_no));
CREATE VIEW jobs_current AS
  SELECT j.*, a.cluster, a.slurm_id, a.ctrl_root, a.ctrl_dir, a.workdir, a.stdout_path, a.stderr_path, a.stdout_pattern, a.stderr_pattern,
         a.node, a.token, a.target_json, a.submit_line, a.state AS attempt_state, a.excluded_nodes
  FROM jobs j JOIN attempts a ON a.handle = j.handle AND a.attempt_no = j.attempt_no;
CREATE TABLE lease(name TEXT PRIMARY KEY,
  owner_pid INTEGER NOT NULL, owner_host TEXT NOT NULL, token INTEGER NOT NULL,
  acquired_local REAL NOT NULL, renewed_local REAL NOT NULL);
CREATE TABLE event_acks(client_id TEXT NOT NULL, seq INTEGER NOT NULL, acked_local REAL NOT NULL, PRIMARY KEY(client_id, seq));
CREATE TABLE deliveries(client_id TEXT PRIMARY KEY, next_seq INTEGER NOT NULL, seqs_json TEXT NOT NULL, delivered_local REAL NOT NULL);
CREATE TABLE array_tasks(handle TEXT NOT NULL, task_id INTEGER NOT NULL, slurm_id TEXT, state TEXT NOT NULL, exit_code INTEGER,
  start_ts INTEGER, end_ts INTEGER, node TEXT, PRIMARY KEY(handle, task_id));
CREATE TABLE events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, ts_local REAL NOT NULL, ts INTEGER,
  kind TEXT NOT NULL, handle TEXT, cluster TEXT, slurm_id TEXT, summary TEXT NOT NULL, payload_json TEXT NOT NULL,
  notified INTEGER NOT NULL DEFAULT 0);
CREATE INDEX events_handle ON events(handle, seq);
CREATE TABLE transfers(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  cluster TEXT NOT NULL, host_role TEXT NOT NULL,
  local TEXT NOT NULL, remote TEXT NOT NULL, state TEXT NOT NULL, mode TEXT,
  files_total INTEGER, files_done INTEGER NOT NULL DEFAULT 0, bytes_total INTEGER, bytes_done INTEGER NOT NULL DEFAULT 0,
  error TEXT, handle TEXT, started_local REAL NOT NULL, finished_local REAL, seconds REAL);
CREATE TABLE transfer_files(transfer_id INTEGER NOT NULL REFERENCES transfers(id), rel_path TEXT NOT NULL,
  size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha1 TEXT, state TEXT NOT NULL, bytes_done INTEGER NOT NULL DEFAULT 0,
  local_name TEXT,
  PRIMARY KEY(transfer_id, rel_path));
CREATE TABLE manifests(scope TEXT NOT NULL, rel_path TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha1 TEXT,
  updated_local REAL NOT NULL, PRIMARY KEY(scope, rel_path));
CREATE TABLE alloc_cmds(id TEXT PRIMARY KEY,
  handle TEXT NOT NULL REFERENCES jobs(handle), n INTEGER NOT NULL, command TEXT NOT NULL, cwd TEXT, mode TEXT NOT NULL,
  state TEXT NOT NULL, submitted_local REAL NOT NULL, started_ts INTEGER, done_ts INTEGER, rc INTEGER, out_path TEXT NOT NULL,
  kill_path TEXT NOT NULL,
  kill_requested_local REAL);
CREATE TABLE plans(plan_id TEXT PRIMARY KEY, created_local REAL NOT NULL, expires_local REAL NOT NULL, spec_json TEXT NOT NULL,
  options_json TEXT NOT NULL, recommended TEXT);
CREATE TABLE wait_history(id INTEGER PRIMARY KEY AUTOINCREMENT, cluster TEXT NOT NULL, target_key TEXT NOT NULL,
  submit_ts INTEGER NOT NULL, start_ts INTEGER NOT NULL, wait_s INTEGER NOT NULL, gpus INTEGER, hours REAL, source TEXT NOT NULL);
CREATE INDEX wait_history_key ON wait_history(cluster, target_key, start_ts);
CREATE TABLE target_stats(cluster TEXT NOT NULL, target_key TEXT NOT NULL, consecutive_failures INTEGER NOT NULL DEFAULT 0,
  breaker_open_until_local REAL, last_error TEXT, infeasible_until_local REAL, infeasible_reason TEXT,
  last_node_fail_node TEXT, last_node_fail_local REAL, PRIMARY KEY(cluster, target_key));
CREATE TABLE kv(key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
"""

# PRAGMA user_version ladder: (version, script). Applied in order for every version > the file's.
MIGRATIONS: tuple[tuple[int, str], ...] = ((1, SCHEMA_V1),)

PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "jobs": ("handle",), "attempts": ("id",), "lease": ("name",), "event_acks": ("client_id", "seq"),
    "deliveries": ("client_id",), "array_tasks": ("handle", "task_id"), "events": ("seq",), "transfers": ("id",),
    "transfer_files": ("transfer_id", "rel_path"), "manifests": ("scope", "rel_path"), "alloc_cmds": ("id",),
    "plans": ("plan_id",), "wait_history": ("id",), "target_stats": ("cluster", "target_key"), "kv": ("key",),
}
TABLES: tuple[str, ...] = tuple(PRIMARY_KEYS)
VIEWS: tuple[str, ...] = ("jobs_current",)
INDEXES: tuple[str, ...] = ("events_handle", "wait_history_key")


class LeaseLost(RuntimeError):
    """Raised by ``write_fenced`` when the monitor lease token differs from ours (design sections 3.3, 5.8)."""

    def __init__(self, token: int | None, holder: dict[str, Any] | None) -> None:
        self.token = token
        self.holder = holder
        pid = holder.get("owner_pid") if holder else None
        super().__init__(f"monitor lease lost (our token {token}, held by pid {pid})" if holder
                         else f"monitor lease lost (our token {token}, no lease row)")


class StoreClosed(RuntimeError):
    pass


@dataclass(frozen=True)
class LeaseInfo:
    """Outcome of ``lease_acquire`` (design section 5.8)."""

    acquired: bool
    token: int | None            # our token when acquired, else the holder's
    owner_pid: int | None
    owner_host: str | None
    renewed_local: float | None
    reason: str                  # new | mine | stale | forced | held-live | held-fresh | held-unknown

    @property
    def monitor(self) -> str:
        """The ``clusters().monitor`` string for a process that just tried to acquire."""
        if self.acquired:
            return "self"
        return f"held by pid {self.owner_pid}" if self.owner_pid is not None else "none"


def _now() -> float:
    return time.time()


def _dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def _encode(column: str, value: Any) -> Any:
    """JSON-encode dict/list values for ``*_json`` columns; everything else passes through."""
    if column.endswith("_json") and not isinstance(value, (str, bytes, type(None))):
        return _dumps(value)
    if isinstance(value, bool):
        return int(value)
    return value


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def loads_json(row: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    """Decode a ``*_json`` column of a row dict (None/missing -> default)."""
    if row is None:
        return default
    raw = row.get(key)
    if raw is None or raw == "":
        return default
    return json.loads(raw)


def transfer_handle(transfer_id: int) -> str:
    """``transfers.id`` -> ``"t<id>"`` (design section 3.3)."""
    return f"t{int(transfer_id)}"


def parse_transfer_handle(handle: str) -> int | None:
    if isinstance(handle, str) and len(handle) > 1 and handle[0] == "t" and handle[1:].isdigit():
        return int(handle[1:])
    return None


class Store:
    """The SQLite ledger (design section 3.3). One connection, one in-process lock, WAL, fenced writes."""

    def __init__(self, path: str | os.PathLike[str], *, pid: int | None = None, host: str | None = None,
                 pid_exists: Callable[[int], bool] | None = None, now: Callable[[], float] = _now) -> None:
        self.path = Path(path)
        self.pid = int(os.getpid() if pid is None else pid)
        self.host = host or socket.gethostname()
        self._pid_exists = pid_exists or psutil.pid_exists
        self._now = now
        self._lock = threading.Lock()
        self._hooks: list[Callable[[], Any]] = []
        self._in_txn = False
        self._closed = False
        self.recovered: bool = False              # True when a corrupt file was moved aside and recreated
        self.corrupt_backup: Path | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._open_or_recover()
        self.columns: dict[str, tuple[str, ...]] = {t: self._table_columns(t) for t in TABLES}
        self.columns["jobs_current"] = self._table_columns("jobs_current")

    # -- open / migrate / corruption -----------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.DatabaseError:
            conn.close()          # release the handle so a corrupt file can be renamed (Windows)
            raise
        return conn

    def _open_or_recover(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            check = conn.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]).lower() != "ok":
                raise sqlite3.DatabaseError(f"quick_check: {check[0] if check else 'no result'}")
            self._migrate(conn)
            return conn
        except sqlite3.DatabaseError as exc:
            if _is_lock_error(exc):
                raise
            if conn is not None:
                conn.close()          # Windows cannot rename an open file
            self._move_corrupt_aside()
            conn = self._connect()
            self._migrate(conn)
            self.recovered = True
            return conn

    def _move_corrupt_aside(self) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._now()))
        backup = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
        n = 0
        while backup.exists():
            n += 1
            backup = self.path.with_name(f"{self.path.name}.corrupt-{stamp}-{n}")
        if self.path.exists():
            os.replace(self.path, backup)
        for suffix in ("-wal", "-shm", "-journal"):
            side = self.path.with_name(self.path.name + suffix)
            if side.exists():
                try:
                    os.replace(side, backup.with_name(backup.name + suffix))
                except OSError:
                    side.unlink(missing_ok=True)
        self.corrupt_backup = backup

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """``PRAGMA user_version`` ladder (design section 11c)."""
        current = int(conn.execute("PRAGMA user_version").fetchone()[0])
        for version, script in MIGRATIONS:
            if version <= current:
                continue
            conn.execute("BEGIN IMMEDIATE")
            try:
                _exec_many(conn, script)
                conn.execute(f"PRAGMA user_version={version}")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            current = version

    def _table_columns(self, table: str) -> tuple[str, ...]:
        return tuple(r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})"))

    @property
    def user_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            with self._lock:
                self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transactions (design section 3.3 "Transactions") ---------------------------------------------

    def after_commit(self, hook: Callable[[], Any]) -> None:
        """Register a callable run after the current transaction commits (dropped on rollback).

        Async ``write`` awaits awaitable results on the event-loop thread; ``write_sync`` runs hooks in
        the calling thread. ``EventBus.append`` uses this for ``notify_all``.
        """
        if not self._in_txn:
            raise RuntimeError("after_commit() must be called inside a transaction")
        self._hooks.append(hook)

    def _run_txn(self, fn: Callable[[sqlite3.Connection], T], *, immediate: bool, fenced: bool = False,
                 token: int | None = None) -> tuple[T, list[Callable[[], Any]]]:
        if self._closed:
            raise StoreClosed(str(self.path))
        with self._lock:
            conn = self._conn
            self._hooks = []
            self._in_txn = True
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                if fenced:
                    self._check_fence(conn, token)
                result = fn(conn)
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                self._hooks = []
                raise
            finally:
                self._in_txn = False
            hooks, self._hooks = self._hooks, []
        return result, hooks

    def _check_fence(self, conn: sqlite3.Connection, token: int | None) -> None:
        row = row_to_dict(conn.execute("SELECT * FROM lease WHERE name=?", (LEASE_NAME,)).fetchone())
        if row is None or token is None or int(row["token"]) != int(token):
            raise LeaseLost(token, row)

    @staticmethod
    def _run_hooks_sync(hooks: Iterable[Callable[[], Any]]) -> None:
        for hook in hooks:
            result = hook()
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None:
                    loop.create_task(_await(result))
                else:
                    _close_awaitable(result)

    @staticmethod
    async def _run_hooks(hooks: Iterable[Callable[[], Any]]) -> None:
        for hook in hooks:
            result = hook()
            if inspect.isawaitable(result):
                await result

    def write_sync(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """``BEGIN IMMEDIATE; fn(conn); COMMIT`` in the calling thread (CLI)."""
        result, hooks = self._run_txn(fn, immediate=True)
        self._run_hooks_sync(hooks)
        return result

    def write_fenced_sync(self, token: int | None, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Like ``write_sync`` but first checks ``lease.token == token``; raises ``LeaseLost`` otherwise."""
        result, hooks = self._run_txn(fn, immediate=True, fenced=True, token=token)
        self._run_hooks_sync(hooks)
        return result

    def read_sync(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """A deferred read transaction (consistent snapshot) in the calling thread."""
        result, _hooks = self._run_txn(fn, immediate=False)
        return result

    async def write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Async ``write_sync`` via ``asyncio.to_thread``; post-commit hooks run on the loop thread."""
        result, hooks = await asyncio.to_thread(self._run_txn, fn, immediate=True)
        await self._run_hooks(hooks)
        return result

    async def write_fenced(self, token: int | None, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Async fenced write (design section 3.3): raises ``LeaseLost`` when the lease token differs."""
        result, hooks = await asyncio.to_thread(self._run_txn, fn, immediate=True, fenced=True, token=token)
        await self._run_hooks(hooks)
        return result

    async def read(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        result, _hooks = await asyncio.to_thread(self._run_txn, fn, immediate=False)
        return result

    # -- counters (design section 3.3 "Handle allocation") ---------------------------------------------

    def _bump_counter(self, conn: sqlite3.Connection, key: str) -> int:
        n = int(self.kv_get(conn, key, 0)) + 1
        self.kv_set(conn, key, n)
        return n

    def next_handle(self, conn: sqlite3.Connection, kind: str) -> str:
        """``kv.counter.handle += 1`` -> ``j<n>`` (kind ``job``) or ``a<n>`` (kind ``alloc``). One shared counter."""
        prefix = {"job": "j", "alloc": "a"}.get(kind)
        if prefix is None:
            raise ValueError(f"unknown handle kind {kind!r} (job | alloc)")
        return f"{prefix}{self._bump_counter(conn, 'counter.handle')}"

    def next_plan_id(self, conn: sqlite3.Connection) -> str:
        """``"p" + kv.counter.plan`` (design section 3.3)."""
        return f"p{self._bump_counter(conn, 'counter.plan')}"

    # -- kv ------------------------------------------------------------------------------------------

    @staticmethod
    def kv_get(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
        row = conn.execute("SELECT value_json FROM kv WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    @staticmethod
    def kv_set(conn: sqlite3.Connection, key: str, value: Any) -> None:
        conn.execute("INSERT INTO kv(key, value_json) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
                     (key, _dumps(value)))

    @staticmethod
    def kv_delete(conn: sqlite3.Connection, key: str) -> bool:
        return conn.execute("DELETE FROM kv WHERE key=?", (key,)).rowcount > 0

    @staticmethod
    def kv_keys(conn: sqlite3.Connection, prefix: str = "") -> list[str]:
        return [r[0] for r in conn.execute("SELECT key FROM kv WHERE substr(key, 1, ?)=? ORDER BY key",
                                           (len(prefix), prefix))]

    # -- generic CRUD ----------------------------------------------------------------------------------

    def _check_columns(self, table: str, fields: Iterable[str]) -> None:
        known = self.columns.get(table)
        if known is None:
            raise ValueError(f"unknown table {table!r}")
        bad = [f for f in fields if f not in known]
        if bad:
            raise ValueError(f"unknown column(s) {bad} for table {table}")

    def insert(self, conn: sqlite3.Connection, table: str, **fields: Any) -> int:
        """INSERT one row; returns ``lastrowid`` (the autoincrement id where the table has one)."""
        self._check_columns(table, fields)
        cols = list(fields)
        sql = f"INSERT INTO {table}({', '.join(cols)}) VALUES({', '.join('?' for _ in cols)})"
        cur = conn.execute(sql, [_encode(c, fields[c]) for c in cols])
        return int(cur.lastrowid or 0)

    def upsert(self, conn: sqlite3.Connection, table: str, **fields: Any) -> None:
        """INSERT ... ON CONFLICT(primary key) DO UPDATE of every non-key field."""
        self._check_columns(table, fields)
        pk = PRIMARY_KEYS[table]
        missing = [k for k in pk if k not in fields]
        if missing:
            raise ValueError(f"upsert into {table} needs primary key column(s) {missing}")
        cols = list(fields)
        updates = [c for c in cols if c not in pk]
        sql = f"INSERT INTO {table}({', '.join(cols)}) VALUES({', '.join('?' for _ in cols)}) ON CONFLICT({', '.join(pk)}) "
        sql += "DO UPDATE SET " + ", ".join(f"{c}=excluded.{c}" for c in updates) if updates else "DO NOTHING"
        conn.execute(sql, [_encode(c, fields[c]) for c in cols])

    def update(self, conn: sqlite3.Connection, table: str, where: Mapping[str, Any], **fields: Any) -> int:
        """UPDATE rows matching ``where`` (equality AND); returns the row count. No-op without fields."""
        if not fields:
            return 0
        self._check_columns(table, list(fields) + list(where))
        sets = ", ".join(f"{c}=?" for c in fields)
        cond, params = _where(where)
        cur = conn.execute(f"UPDATE {table} SET {sets}{cond}", [_encode(c, fields[c]) for c in fields] + params)
        return cur.rowcount

    def select(self, conn: sqlite3.Connection, table: str, where: Mapping[str, Any] | None = None, *,
               order_by: str | None = None, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
        """SELECT * rows as dicts. ``where`` values may be lists/tuples (IN) or None (IS NULL)."""
        self._check_columns(table, where or {})
        cond, params = _where(where or {})
        sql = f"SELECT * FROM {table}{cond}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        return [dict(r) for r in conn.execute(sql, params)]

    def select_one(self, conn: sqlite3.Connection, table: str, **where: Any) -> dict[str, Any] | None:
        rows = self.select(conn, table, where, limit=1)
        return rows[0] if rows else None

    def delete(self, conn: sqlite3.Connection, table: str, **where: Any) -> int:
        self._check_columns(table, where)
        cond, params = _where(where)
        return conn.execute(f"DELETE FROM {table}{cond}", params).rowcount

    def count(self, conn: sqlite3.Connection, table: str, **where: Any) -> int:
        self._check_columns(table, where)
        cond, params = _where(where)
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}{cond}", params).fetchone()[0])

    # -- jobs / attempts (design section 3.3 jobs, attempts, jobs_current) --------------------------------

    def insert_job(self, conn: sqlite3.Connection, *, handle: str, kind: str, name: str, state: Any, spec_json: Any,
                   placement_mode: str, attempt_no: int = 1, now: float | None = None, **fields: Any) -> str:
        """Insert a ``jobs`` row; ``created_local``/``updated_local`` default to now. Returns the handle."""
        ts = self._now() if now is None else now
        self.insert(conn, "jobs", handle=handle, kind=kind, name=name, state=str(state), spec_json=spec_json,
                    placement_mode=placement_mode, attempt_no=attempt_no,
                    created_local=fields.pop("created_local", ts), updated_local=fields.pop("updated_local", ts), **fields)
        return handle

    def insert_attempt(self, conn: sqlite3.Connection, *, handle: str, attempt_no: int, cluster: str, token: str,
                       ctrl_root: str, ctrl_dir: str, workdir: str, target_json: Any, state: Any, cause: str,
                       now: float | None = None, **fields: Any) -> int:
        """Insert an ``attempts`` row (``intent_local`` defaults to now). Returns ``attempts.id``."""
        ts = self._now() if now is None else now
        return self.insert(conn, "attempts", handle=handle, attempt_no=attempt_no, cluster=cluster, token=token,
                           ctrl_root=ctrl_root, ctrl_dir=ctrl_dir, workdir=workdir, target_json=target_json,
                           state=str(state), cause=cause, intent_local=fields.pop("intent_local", ts), **fields)

    def get_job(self, conn: sqlite3.Connection, handle: str) -> dict[str, Any] | None:
        """The ``jobs_current`` row (job + current attempt's cluster-relative fields) as a dict, or None."""
        return row_to_dict(conn.execute("SELECT * FROM jobs_current WHERE handle=?", (handle,)).fetchone())

    def get_job_base(self, conn: sqlite3.Connection, handle: str) -> dict[str, Any] | None:
        """The bare ``jobs`` row (exists even before the first attempt is inserted)."""
        return self.select_one(conn, "jobs", handle=handle)

    def list_jobs(self, conn: sqlite3.Connection, *, states: Iterable[Any] | None = None, kind: str | None = None,
                  cluster: str | None = None, handles: Iterable[str] | None = None, name: str | None = None,
                  limit: int | None = None, offset: int = 0, order_by: str = "created_local DESC, handle DESC",
                  ) -> list[dict[str, Any]]:
        """``jobs_current`` rows filtered by state/kind/cluster/handles/name (all optional, AND-ed)."""
        where: dict[str, Any] = {}
        if states is not None:
            where["state"] = [str(s) for s in states]
        if kind is not None:
            where["kind"] = kind
        if cluster is not None:
            where["cluster"] = cluster
        if handles is not None:
            where["handle"] = list(handles)
        if name is not None:
            where["name"] = name
        return self.select(conn, "jobs_current", where, order_by=order_by, limit=limit, offset=offset)

    def update_job(self, conn: sqlite3.Connection, handle: str, **fields: Any) -> int:
        """UPDATE ``jobs`` by handle; ``updated_local`` is bumped automatically. Returns rows updated."""
        if "state" in fields and fields["state"] is not None:
            fields["state"] = str(fields["state"])
        fields.setdefault("updated_local", self._now())
        return self.update(conn, "jobs", {"handle": handle}, **fields)

    def update_attempt(self, conn: sqlite3.Connection, attempt_id: int, **fields: Any) -> int:
        if "state" in fields and fields["state"] is not None:
            fields["state"] = str(fields["state"])
        return self.update(conn, "attempts", {"id": attempt_id}, **fields)

    def current_attempt(self, conn: sqlite3.Connection, handle: str) -> dict[str, Any] | None:
        return row_to_dict(conn.execute(
            "SELECT a.* FROM attempts a JOIN jobs j ON j.handle=a.handle AND j.attempt_no=a.attempt_no WHERE j.handle=?",
            (handle,)).fetchone())

    def attempts_for(self, conn: sqlite3.Connection, handle: str) -> list[dict[str, Any]]:
        return self.select(conn, "attempts", {"handle": handle}, order_by="attempt_no")

    def attempt_by_token(self, conn: sqlite3.Connection, token: str) -> dict[str, Any] | None:
        return self.select_one(conn, "attempts", token=token)

    def delete_job(self, conn: sqlite3.Connection, handle: str) -> int:
        """Remove a job with its attempts, array tasks and alloc cmds (tests / ``db recover``)."""
        self.delete(conn, "alloc_cmds", handle=handle)
        self.delete(conn, "array_tasks", handle=handle)
        self.delete(conn, "attempts", handle=handle)
        return self.delete(conn, "jobs", handle=handle)

    # -- lease (design section 5.8) -------------------------------------------------------------------

    def lease_get(self, conn: sqlite3.Connection) -> dict[str, Any] | None:
        return self.select_one(conn, "lease", name=LEASE_NAME)

    def _holder_alive(self, row: Mapping[str, Any]) -> bool | None:
        """True/False when the holder is on this host (``psutil.pid_exists``); None when it cannot be checked."""
        if row["owner_host"] != self.host:
            return None
        return bool(self._pid_exists(int(row["owner_pid"])))

    def lease_acquire(self, conn: sqlite3.Connection, *, force: bool = False, now: float | None = None) -> LeaseInfo:
        """Try to take the monitor lease inside the caller's ``BEGIN IMMEDIATE`` transaction.

        Rules (section 5.8): acquire (token += 1) when there is no row, when the row is ours (same pid and
        host: keep the token, renew), when the owner pid is **provably dead on this host** (a dead local pid
        cannot wake up and write, so waiting out the staleness window would only leave jobs unmonitored --
        measured: every Claude Code restart left the Monitor dead for 5 minutes), or when ``renewed_local`` is
        older than 5 min and liveness cannot be verified (a different host). A lease held by a live pid is
        never taken over unless ``force`` (a human ran ``slurm-mcp monitor takeover --force``). Otherwise the
        current holder is reported.
        """
        ts = self._now() if now is None else now
        row = self.lease_get(conn)
        if row is None:
            self.insert(conn, "lease", name=LEASE_NAME, owner_pid=self.pid, owner_host=self.host, token=1,
                        acquired_local=ts, renewed_local=ts)
            return LeaseInfo(True, 1, self.pid, self.host, ts, "new")
        if int(row["owner_pid"]) == self.pid and row["owner_host"] == self.host and not force:
            self.update(conn, "lease", {"name": LEASE_NAME}, renewed_local=ts)
            return LeaseInfo(True, int(row["token"]), self.pid, self.host, ts, "mine")
        alive = self._holder_alive(row)
        stale = (ts - float(row["renewed_local"])) > LEASE_STALE_S
        if force:
            reason = "forced"
        elif alive is False:
            # Provably dead on this host: take over at once, regardless of how recently it renewed.
            reason = "dead-owner"
        elif stale and alive is not True:
            reason = "stale"
        else:
            reason = "held-live" if alive else ("held-fresh" if alive is False else "held-unknown")
            return LeaseInfo(False, int(row["token"]), int(row["owner_pid"]), row["owner_host"],
                             float(row["renewed_local"]), reason)
        token = int(row["token"]) + 1
        self.update(conn, "lease", {"name": LEASE_NAME}, owner_pid=self.pid, owner_host=self.host, token=token,
                    acquired_local=ts, renewed_local=ts)
        return LeaseInfo(True, token, self.pid, self.host, ts, reason)

    def lease_renew(self, conn: sqlite3.Connection, token: int | None, *, now: float | None = None) -> bool:
        """``UPDATE lease SET renewed_local=now WHERE name='monitor' AND token=?``; False (0 rows) = lost."""
        if token is None:
            return False
        ts = self._now() if now is None else now
        cur = conn.execute("UPDATE lease SET renewed_local=? WHERE name=? AND token=?", (ts, LEASE_NAME, int(token)))
        return cur.rowcount == 1

    def lease_release(self, conn: sqlite3.Connection, token: int | None) -> bool:
        """``DELETE FROM lease WHERE name='monitor' AND token=?`` (the CLI's one-tick lease, section 5.8)."""
        if token is None:
            return False
        return conn.execute("DELETE FROM lease WHERE name=? AND token=?", (LEASE_NAME, int(token))).rowcount == 1

    def monitor_status(self, conn: sqlite3.Connection, my_token: int | None) -> str:
        """``clusters().monitor``: ``self`` | ``held by pid N`` | ``lost to pid N`` | ``none``."""
        row = self.lease_get(conn)
        if row is None:
            return "none"
        if my_token is not None and int(row["token"]) == int(my_token):
            return "self"
        if my_token is not None:
            return f"lost to pid {row['owner_pid']}"
        return f"held by pid {row['owner_pid']}"

    # -- transfers -----------------------------------------------------------------------------------

    def insert_transfer(self, conn: sqlite3.Connection, *, kind: str, cluster: str, host_role: str, local: str,
                        remote: str, state: Any, now: float | None = None, **fields: Any) -> int:
        ts = self._now() if now is None else now
        return self.insert(conn, "transfers", kind=kind, cluster=cluster, host_role=host_role, local=local, remote=remote,
                           state=str(state), started_local=fields.pop("started_local", ts), **fields)

    def update_transfer(self, conn: sqlite3.Connection, transfer_id: int, **fields: Any) -> int:
        if "state" in fields and fields["state"] is not None:
            fields["state"] = str(fields["state"])
        return self.update(conn, "transfers", {"id": int(transfer_id)}, **fields)

    def get_transfer(self, conn: sqlite3.Connection, transfer_id: int) -> dict[str, Any] | None:
        return self.select_one(conn, "transfers", id=int(transfer_id))

    def list_transfers(self, conn: sqlite3.Connection, *, states: Iterable[Any] | None = None, handle: str | None = None,
                       cluster: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        where: dict[str, Any] = {}
        if states is not None:
            where["state"] = [str(s) for s in states]
        if handle is not None:
            where["handle"] = handle
        if cluster is not None:
            where["cluster"] = cluster
        return self.select(conn, "transfers", where, order_by="id DESC", limit=limit)

    def upsert_transfer_file(self, conn: sqlite3.Connection, transfer_id: int, rel_path: str, *, size: int, mtime_ns: int,
                             state: Any, **fields: Any) -> None:
        self.upsert(conn, "transfer_files", transfer_id=int(transfer_id), rel_path=rel_path, size=size, mtime_ns=mtime_ns,
                    state=str(state), **fields)

    def update_transfer_file(self, conn: sqlite3.Connection, transfer_id: int, rel_path: str, **fields: Any) -> int:
        if "state" in fields and fields["state"] is not None:
            fields["state"] = str(fields["state"])
        return self.update(conn, "transfer_files", {"transfer_id": int(transfer_id), "rel_path": rel_path}, **fields)

    def transfer_files_for(self, conn: sqlite3.Connection, transfer_id: int, *, states: Iterable[Any] | None = None,
                           ) -> list[dict[str, Any]]:
        where: dict[str, Any] = {"transfer_id": int(transfer_id)}
        if states is not None:
            where["state"] = [str(s) for s in states]
        return self.select(conn, "transfer_files", where, order_by="rel_path")

    # -- manifests -----------------------------------------------------------------------------------

    def manifest(self, conn: sqlite3.Connection, scope: str) -> dict[str, dict[str, Any]]:
        """``rel_path -> row`` for a manifest scope (``up:<cluster>:<remote_root>`` | ``down:<local_dir>``)."""
        return {r["rel_path"]: r for r in self.select(conn, "manifests", {"scope": scope})}

    def upsert_manifest(self, conn: sqlite3.Connection, scope: str, rel_path: str, *, size: int, mtime_ns: int,
                        sha1: str | None = None, now: float | None = None) -> None:
        self.upsert(conn, "manifests", scope=scope, rel_path=rel_path, size=size, mtime_ns=mtime_ns, sha1=sha1,
                    updated_local=self._now() if now is None else now)

    def delete_manifest(self, conn: sqlite3.Connection, scope: str, rel_path: str | None = None) -> int:
        where: dict[str, Any] = {"scope": scope}
        if rel_path is not None:
            where["rel_path"] = rel_path
        return self.delete(conn, "manifests", **where)

    # -- alloc_cmds ----------------------------------------------------------------------------------

    def insert_alloc_cmd(self, conn: sqlite3.Connection, *, handle: str, command: str, mode: str, state: Any,
                         out_path: str, kill_path: str, cwd: str | None = None, now: float | None = None,
                         **fields: Any) -> tuple[str, int]:
        """Insert the next command of an allocation: ``n`` = max(n)+1 per handle, id ``<handle>.c<n>``."""
        row = conn.execute("SELECT COALESCE(MAX(n), 0) FROM alloc_cmds WHERE handle=?", (handle,)).fetchone()
        n = int(row[0]) + 1
        cmd_id = f"{handle}.c{n}"
        self.insert(conn, "alloc_cmds", id=cmd_id, handle=handle, n=n, command=command, cwd=cwd, mode=mode,
                    state=str(state), submitted_local=self._now() if now is None else now, out_path=out_path,
                    kill_path=kill_path, **fields)
        return cmd_id, n

    def update_alloc_cmd(self, conn: sqlite3.Connection, cmd_id: str, **fields: Any) -> int:
        if "state" in fields and fields["state"] is not None:
            fields["state"] = str(fields["state"])
        return self.update(conn, "alloc_cmds", {"id": cmd_id}, **fields)

    def get_alloc_cmd(self, conn: sqlite3.Connection, cmd_id: str) -> dict[str, Any] | None:
        return self.select_one(conn, "alloc_cmds", id=cmd_id)

    def alloc_cmds_for(self, conn: sqlite3.Connection, handle: str, *, states: Iterable[Any] | None = None,
                       ) -> list[dict[str, Any]]:
        where: dict[str, Any] = {"handle": handle}
        if states is not None:
            where["state"] = [str(s) for s in states]
        return self.select(conn, "alloc_cmds", where, order_by="n")

    # -- plans ---------------------------------------------------------------------------------------

    def insert_plan(self, conn: sqlite3.Connection, *, spec_json: Any, options_json: Any, recommended: str | None = None,
                    ttl_s: float = PLAN_TTL_S, now: float | None = None) -> str:
        """Allocate ``p<counter.plan>`` and insert the plan (valid ``ttl_s``, 15 min by default)."""
        ts = self._now() if now is None else now
        plan_id = self.next_plan_id(conn)
        self.insert(conn, "plans", plan_id=plan_id, created_local=ts, expires_local=ts + ttl_s, spec_json=spec_json,
                    options_json=options_json, recommended=recommended)
        return plan_id

    def get_plan(self, conn: sqlite3.Connection, plan_id: str) -> dict[str, Any] | None:
        return self.select_one(conn, "plans", plan_id=plan_id)

    def purge_expired_plans(self, conn: sqlite3.Connection, *, now: float | None = None) -> int:
        ts = self._now() if now is None else now
        return conn.execute("DELETE FROM plans WHERE expires_local < ?", (ts,)).rowcount

    # -- wait_history / target_stats -----------------------------------------------------------------

    def insert_wait_history(self, conn: sqlite3.Connection, *, cluster: str, target_key: str, submit_ts: int, start_ts: int,
                            source: str, gpus: int | None = None, hours: float | None = None, wait_s: int | None = None) -> int:
        return self.insert(conn, "wait_history", cluster=cluster, target_key=target_key, submit_ts=submit_ts, start_ts=start_ts,
                           wait_s=max(0, int(start_ts - submit_ts)) if wait_s is None else int(wait_s), gpus=gpus, hours=hours,
                           source=source)

    def wait_history(self, conn: sqlite3.Connection, cluster: str, target_key: str, *, since_ts: int | None = None,
                     limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM wait_history WHERE cluster=? AND target_key=?"
        params: list[Any] = [cluster, target_key]
        if since_ts is not None:
            sql += " AND start_ts >= ?"
            params.append(int(since_ts))
        sql += " ORDER BY start_ts DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [dict(r) for r in conn.execute(sql, params)]

    def get_target_stats(self, conn: sqlite3.Connection, cluster: str, target_key: str) -> dict[str, Any] | None:
        return self.select_one(conn, "target_stats", cluster=cluster, target_key=target_key)

    def upsert_target_stats(self, conn: sqlite3.Connection, cluster: str, target_key: str, **fields: Any) -> None:
        self.upsert(conn, "target_stats", cluster=cluster, target_key=target_key, **fields)

    # -- array_tasks ---------------------------------------------------------------------------------

    def upsert_array_task(self, conn: sqlite3.Connection, handle: str, task_id: int, *, state: Any, **fields: Any) -> None:
        self.upsert(conn, "array_tasks", handle=handle, task_id=int(task_id), state=str(state), **fields)

    def array_tasks_for(self, conn: sqlite3.Connection, handle: str) -> list[dict[str, Any]]:
        return self.select(conn, "array_tasks", {"handle": handle}, order_by="task_id")


# --- helpers -----------------------------------------------------------------------------------------

def _where(where: Mapping[str, Any]) -> tuple[str, list[Any]]:
    """Equality/IN/IS NULL conditions AND-ed; returns ("", []) for an empty mapping."""
    if not where:
        return "", []
    parts: list[str] = []
    params: list[Any] = []
    for col, val in where.items():
        if val is None:
            parts.append(f"{col} IS NULL")
        elif isinstance(val, (list, tuple, set, frozenset)):
            vals = list(val)
            if not vals:
                parts.append("0")
            else:
                parts.append(f"{col} IN ({', '.join('?' for _ in vals)})")
                params.extend(_encode(col, v) for v in vals)
        else:
            parts.append(f"{col}=?")
            params.append(_encode(col, val))
    return " WHERE " + " AND ".join(parts), params


def _exec_many(conn: sqlite3.Connection, script: str) -> None:
    """Execute a ``;``-separated DDL script statement by statement inside the caller's transaction
    (``executescript`` would issue its own COMMIT first)."""
    for stmt in script.split(";"):
        text = stmt.strip()
        if text:
            conn.execute(text)


def _is_lock_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text


async def _await(awaitable: Any) -> None:
    await awaitable


def _close_awaitable(awaitable: Any) -> None:
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()


__all__ = [
    "LEASE_NAME", "LEASE_STALE_S", "PLAN_TTL_S", "SCHEMA_VERSION", "SCHEMA_V1", "MIGRATIONS", "PRIMARY_KEYS", "TABLES",
    "VIEWS", "INDEXES", "LeaseLost", "StoreClosed", "LeaseInfo", "Store", "row_to_dict", "loads_json", "transfer_handle",
    "parse_transfer_handle",
]
