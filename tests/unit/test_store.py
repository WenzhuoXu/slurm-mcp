"""Unit tests for slurm_mcp.store (design sections 3.3, 5.8, 11c, 11i)."""
from __future__ import annotations

import asyncio
import sqlite3
import time

import pytest

from slurm_mcp.store import (
    INDEXES, LEASE_STALE_S, TABLES, VIEWS, LeaseLost, Store, StoreClosed, loads_json, parse_transfer_handle,
    transfer_handle,
)

# Design section 3.3, column names verbatim.
EXPECTED_COLUMNS: dict[str, list[str]] = {
    "jobs": ["handle", "kind", "name", "state", "slurm_state", "reason", "spec_json", "placement_mode", "attempt_no",
             "submit_ts", "start_ts", "end_ts", "est_start_ts", "exit_code", "exit_signal", "restarts", "moves",
             "cost_est_su", "cost_worst_su", "cost_actual_su", "last_seen_ts", "stale_ticks", "terminal_ts", "enriched",
             "collected_ts", "cancel_requested_ts", "cancel_hard_ts", "hold_reason", "alloc_ready", "alloc_end_ts",
             "array_size", "depends_on_json", "heartbeat_ts", "progress_json", "last_line", "created_local", "updated_local"],
    "attempts": ["id", "handle", "attempt_no", "cluster", "token", "slurm_id", "ctrl_root", "ctrl_dir", "workdir",
                 "stdout_pattern", "stderr_pattern", "stdout_path", "stderr_path", "node", "target_json", "submit_line",
                 "state", "cause", "intent_local", "invoked_local", "confirmed_local", "submit_ts", "end_ts", "final_state",
                 "exit_code", "reason", "excluded_nodes"],
    "lease": ["name", "owner_pid", "owner_host", "token", "acquired_local", "renewed_local"],
    "event_acks": ["client_id", "seq", "acked_local"],
    "deliveries": ["client_id", "next_seq", "seqs_json", "delivered_local"],
    "array_tasks": ["handle", "task_id", "slurm_id", "state", "exit_code", "start_ts", "end_ts", "node"],
    "events": ["seq", "ts_local", "ts", "kind", "handle", "cluster", "slurm_id", "summary", "payload_json", "notified"],
    "transfers": ["id", "kind", "cluster", "host_role", "local", "remote", "state", "mode", "files_total", "files_done",
                  "bytes_total", "bytes_done", "error", "handle", "started_local", "finished_local", "seconds"],
    "transfer_files": ["transfer_id", "rel_path", "size", "mtime_ns", "sha1", "state", "bytes_done", "local_name"],
    "manifests": ["scope", "rel_path", "size", "mtime_ns", "sha1", "updated_local"],
    "alloc_cmds": ["id", "handle", "n", "command", "cwd", "mode", "state", "submitted_local", "started_ts", "done_ts", "rc",
                   "out_path", "kill_path", "kill_requested_local"],
    "plans": ["plan_id", "created_local", "expires_local", "spec_json", "options_json", "recommended"],
    "wait_history": ["id", "cluster", "target_key", "submit_ts", "start_ts", "wait_s", "gpus", "hours", "source"],
    "target_stats": ["cluster", "target_key", "consecutive_failures", "breaker_open_until_local", "last_error",
                     "infeasible_until_local", "infeasible_reason", "last_node_fail_node", "last_node_fail_local"],
    "kv": ["key", "value_json"],
}
JOBS_CURRENT_EXTRA = ["cluster", "slurm_id", "ctrl_root", "ctrl_dir", "workdir", "stdout_path", "stderr_path",
                      "stdout_pattern", "stderr_pattern", "node", "token", "target_json", "submit_line", "attempt_state",
                      "excluded_nodes"]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "state.db", pid=1000, host="lap", pid_exists=lambda pid: True)
    yield s
    s.close()


def add_job(store: Store, conn, handle: str = "j1", *, cluster: str = "trace", state: str = "SUBMITTED",
            token: str | None = None, name: str = "train", kind: str = "job") -> str:
    store.insert_job(conn, handle=handle, kind=kind, name=name, state=state, spec_json={"name": name}, placement_mode="explicit")
    store.insert_attempt(conn, handle=handle, attempt_no=1, cluster=cluster, token=token or f"t-{handle}", ctrl_root="/c",
                         ctrl_dir="/c/a1", workdir="/w", target_json={"cluster": cluster}, state="INTENT", cause="initial")
    return handle


# --- schema (design section 3.3) ------------------------------------------------------------------

def test_schema_tables_and_columns_exactly_match_design(store):
    assert set(TABLES) == set(EXPECTED_COLUMNS)
    for table, cols in EXPECTED_COLUMNS.items():
        assert list(store.columns[table]) == cols, table
    names = {r[0]: r[1] for r in store.read_sync(lambda c: c.execute("SELECT name, type FROM sqlite_master").fetchall())}
    for table in EXPECTED_COLUMNS:
        assert names.get(table) == "table"
    for view in VIEWS:
        assert names.get(view) == "view"
    for index in INDEXES:
        assert names.get(index) == "index"
    assert store.user_version == 1


def test_jobs_current_view_columns(store):
    cols = list(store.columns["jobs_current"])
    assert cols == EXPECTED_COLUMNS["jobs"] + JOBS_CURRENT_EXTRA


def test_pragmas(store):
    def fn(c):
        return {p: c.execute(f"PRAGMA {p}").fetchone()[0] for p in ("journal_mode", "synchronous", "foreign_keys", "busy_timeout")}
    got = store.read_sync(fn)
    assert got["journal_mode"] == "wal"
    assert got["synchronous"] == 1          # NORMAL
    assert got["foreign_keys"] == 1
    assert got["busy_timeout"] == 5000


def test_reopen_is_idempotent_and_keeps_data(tmp_path):
    path = tmp_path / "state.db"
    with Store(path) as s:
        s.write_sync(lambda c: s.kv_set(c, "x", {"a": 1}))
    with Store(path) as s2:
        assert s2.user_version == 1
        assert not s2.recovered
        assert s2.read_sync(lambda c: s2.kv_get(c, "x")) == {"a": 1}


def test_foreign_keys_enforced(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.write_sync(lambda c: store.insert_attempt(
            c, handle="j99", attempt_no=1, cluster="x", token="t-1", ctrl_root="/", ctrl_dir="/", workdir="/",
            target_json={}, state="INTENT", cause="initial"))


# --- corruption (section 9.2) ---------------------------------------------------------------------

def test_corrupt_file_is_moved_aside_and_recreated(tmp_path):
    path = tmp_path / "state.db"
    path.write_bytes(b"this is not a sqlite database at all" * 100)
    s = Store(path)
    try:
        assert s.recovered is True
        assert s.corrupt_backup is not None and s.corrupt_backup.exists()
        assert s.corrupt_backup.name.startswith("state.db.corrupt-")
        assert s.user_version == 1
        assert s.write_sync(lambda c: s.next_handle(c, "job")) == "j1"
    finally:
        s.close()


def test_healthy_file_is_not_flagged(store):
    assert store.recovered is False and store.corrupt_backup is None


# --- transactions ----------------------------------------------------------------------------------

def test_write_rolls_back_on_exception(store):
    def fn(c):
        store.kv_set(c, "k", 1)
        raise RuntimeError("boom")
    with pytest.raises(RuntimeError):
        store.write_sync(fn)
    assert store.read_sync(lambda c: store.kv_get(c, "k")) is None


def test_after_commit_hooks_run_only_on_commit(store):
    calls: list[str] = []

    def ok(c):
        store.after_commit(lambda: calls.append("ok"))

    def bad(c):
        store.after_commit(lambda: calls.append("bad"))
        raise ValueError

    store.write_sync(ok)
    with pytest.raises(ValueError):
        store.write_sync(bad)
    assert calls == ["ok"]
    with pytest.raises(RuntimeError):
        store.after_commit(lambda: None)      # outside a transaction


@pytest.mark.asyncio
async def test_async_write_awaits_awaitable_hooks(store):
    done = asyncio.Event()

    async def hook():
        done.set()

    await store.write(lambda c: store.after_commit(hook))
    assert done.is_set()


@pytest.mark.asyncio
async def test_async_read_and_write_roundtrip(store):
    await store.write(lambda c: store.kv_set(c, "policy.notify", {"toast": True}))
    assert await store.read(lambda c: store.kv_get(c, "policy.notify")) == {"toast": True}


def test_closed_store_refuses_transactions(tmp_path):
    s = Store(tmp_path / "state.db")
    s.close()
    with pytest.raises(StoreClosed):
        s.read_sync(lambda c: None)


# --- counters (handles, plans) -------------------------------------------------------------------

def test_next_handle_uses_one_shared_counter(store):
    assert store.write_sync(lambda c: store.next_handle(c, "job")) == "j1"
    assert store.write_sync(lambda c: store.next_handle(c, "alloc")) == "a2"
    assert store.write_sync(lambda c: store.next_handle(c, "job")) == "j3"
    assert store.read_sync(lambda c: store.kv_get(c, "counter.handle")) == 3
    with pytest.raises(ValueError):
        store.write_sync(lambda c: store.next_handle(c, "plan"))


@pytest.mark.asyncio
async def test_handle_counter_monotonic_under_concurrent_writers(tmp_path):
    path = tmp_path / "state.db"
    a = Store(path, pid=1)
    b = Store(path, pid=2)          # a second "process" on the same file
    try:
        async def alloc(s: Store, kind: str) -> str:
            return await s.write(lambda c: s.next_handle(c, kind))

        tasks = [alloc(a if i % 2 else b, "job" if i % 3 else "alloc") for i in range(60)]
        handles = await asyncio.gather(*tasks)
        numbers = sorted(int(h[1:]) for h in handles)
        assert numbers == list(range(1, 61))
        assert len(set(handles)) == 60
    finally:
        a.close()
        b.close()


def test_plan_ids_and_expiry(store):
    pid = store.write_sync(lambda c: store.insert_plan(c, spec_json={"n": 1}, options_json=[{"target": "t"}], recommended="t", now=100.0))
    assert pid == "p1"
    assert store.write_sync(lambda c: store.next_plan_id(c)) == "p2"
    row = store.read_sync(lambda c: store.get_plan(c, "p1"))
    assert row["expires_local"] == 100.0 + 900.0
    assert loads_json(row, "options_json") == [{"target": "t"}]
    assert store.write_sync(lambda c: store.purge_expired_plans(c, now=99.0)) == 0
    assert store.write_sync(lambda c: store.purge_expired_plans(c, now=10_000.0)) == 1
    assert store.read_sync(lambda c: store.get_plan(c, "p1")) is None


# --- jobs / attempts DAO ----------------------------------------------------------------------------

def test_insert_get_update_job_and_attempts(store):
    def setup(c):
        add_job(store, c, "j1")
        return store.get_job(c, "j1")
    row = store.write_sync(setup)
    assert row["handle"] == "j1" and row["cluster"] == "trace" and row["attempt_state"] == "INTENT"
    assert row["token"] == "t-j1" and loads_json(row, "spec_json") == {"name": "train"}
    assert row["created_local"] == row["updated_local"]
    assert row["restarts"] == 0 and row["moves"] == 0 and row["enriched"] == 0

    time.sleep(0.01)
    n = store.write_sync(lambda c: store.update_job(c, "j1", state="RUNNING", start_ts=1700000000, progress_json={"pct": 5}))
    assert n == 1
    row2 = store.read_sync(lambda c: store.get_job(c, "j1"))
    assert row2["state"] == "RUNNING" and row2["start_ts"] == 1700000000
    assert loads_json(row2, "progress_json") == {"pct": 5}
    assert row2["updated_local"] > row["updated_local"]
    assert store.write_sync(lambda c: store.update_job(c, "nope", state="X")) == 0

    att = store.read_sync(lambda c: store.current_attempt(c, "j1"))
    assert att["attempt_no"] == 1 and att["intent_local"] > 0
    store.write_sync(lambda c: store.update_attempt(c, att["id"], state="ACTIVE", slurm_id="4242", node="n01"))
    assert store.read_sync(lambda c: store.get_job(c, "j1"))["slurm_id"] == "4242"
    assert store.read_sync(lambda c: store.attempt_by_token(c, "t-j1"))["node"] == "n01"

    # a second attempt on another cluster flips every cluster-relative field of jobs_current
    def move(c):
        store.insert_attempt(c, handle="j1", attempt_no=2, cluster="bridges2", token="t-j1b", ctrl_root="/o/c",
                             ctrl_dir="/o/c/a2", workdir="/o/w", target_json={"cluster": "bridges2"}, state="ACTIVE",
                             cause="rebalanced", slurm_id="77")
        store.update_job(c, "j1", attempt_no=2, moves=1)
    store.write_sync(move)
    cur = store.read_sync(lambda c: store.get_job(c, "j1"))
    assert cur["cluster"] == "bridges2" and cur["slurm_id"] == "77" and cur["workdir"] == "/o/w" and cur["moves"] == 1
    assert [a["attempt_no"] for a in store.read_sync(lambda c: store.attempts_for(c, "j1"))] == [1, 2]
    assert store.read_sync(lambda c: store.current_attempt(c, "j1"))["attempt_no"] == 2
    assert store.read_sync(lambda c: store.get_job_base(c, "j1"))["attempt_no"] == 2


def test_get_job_is_none_without_attempt(store):
    store.write_sync(lambda c: store.insert_job(c, handle="j5", kind="job", name="n", state="QUEUED", spec_json={},
                                                placement_mode="auto"))
    assert store.read_sync(lambda c: store.get_job(c, "j5")) is None
    assert store.read_sync(lambda c: store.get_job_base(c, "j5"))["state"] == "QUEUED"


def test_list_jobs_filters(store):
    def setup(c):
        add_job(store, c, "j1", cluster="trace", state="RUNNING")
        add_job(store, c, "j2", cluster="bridges2", state="COMPLETED")
        add_job(store, c, "a3", cluster="trace", state="RUNNING", kind="alloc", name="alloc")
    store.write_sync(setup)
    all_rows = store.read_sync(lambda c: store.list_jobs(c))
    assert {r["handle"] for r in all_rows} == {"j1", "j2", "a3"}
    assert [r["handle"] for r in store.read_sync(lambda c: store.list_jobs(c, states=["RUNNING"], cluster="trace"))] in (["j1", "a3"], ["a3", "j1"])
    assert [r["handle"] for r in store.read_sync(lambda c: store.list_jobs(c, kind="alloc"))] == ["a3"]
    assert [r["handle"] for r in store.read_sync(lambda c: store.list_jobs(c, handles=["j2"]))] == ["j2"]
    assert [r["handle"] for r in store.read_sync(lambda c: store.list_jobs(c, name="train", order_by="handle"))] == ["j1", "j2"]
    assert len(store.read_sync(lambda c: store.list_jobs(c, limit=2))) == 2
    assert store.read_sync(lambda c: store.list_jobs(c, states=[])) == []
    assert store.write_sync(lambda c: store.delete_job(c, "j1")) == 1
    assert store.read_sync(lambda c: store.get_job(c, "j1")) is None


def test_unknown_column_is_refused(store):
    with pytest.raises(ValueError):
        store.write_sync(lambda c: store.update_job(c, "j1", cluster="x"))     # cluster lives on attempts
    with pytest.raises(ValueError):
        store.write_sync(lambda c: store.insert(c, "nosuch", a=1))


# --- kv ----------------------------------------------------------------------------------------------

def test_kv_roundtrip_json(store):
    store.write_sync(lambda c: store.kv_set(c, "caps.trace", {"epoch_format": True, "n": [1, 2]}))
    store.write_sync(lambda c: store.kv_set(c, "caps.bridges2", 5))
    assert store.read_sync(lambda c: store.kv_get(c, "caps.trace")) == {"epoch_format": True, "n": [1, 2]}
    assert store.read_sync(lambda c: store.kv_get(c, "missing", "dflt")) == "dflt"
    assert store.read_sync(lambda c: store.kv_keys(c, "caps.")) == ["caps.bridges2", "caps.trace"]
    store.write_sync(lambda c: store.kv_set(c, "caps.trace", None))
    assert store.read_sync(lambda c: store.kv_get(c, "caps.trace", "dflt")) is None
    assert store.write_sync(lambda c: store.kv_delete(c, "caps.trace")) is True
    assert store.write_sync(lambda c: store.kv_delete(c, "caps.trace")) is False


# --- generic CRUD ----------------------------------------------------------------------------------

def test_generic_crud_where_semantics(store):
    def setup(c):
        store.insert(c, "wait_history", cluster="trace", target_key="k", submit_ts=1, start_ts=2, wait_s=1, gpus=None, source="observed")
        store.insert(c, "wait_history", cluster="trace", target_key="k", submit_ts=3, start_ts=9, wait_s=6, gpus=2, source="backfill")
        store.insert(c, "wait_history", cluster="b2", target_key="k", submit_ts=3, start_ts=9, wait_s=6, gpus=4, source="observed")
    store.write_sync(setup)
    assert store.read_sync(lambda c: store.count(c, "wait_history")) == 3
    assert store.read_sync(lambda c: store.count(c, "wait_history", gpus=None)) == 1
    assert store.read_sync(lambda c: store.count(c, "wait_history", gpus=[2, 4])) == 2
    assert store.read_sync(lambda c: store.count(c, "wait_history", gpus=[])) == 0
    rows = store.read_sync(lambda c: store.select(c, "wait_history", {"cluster": "trace"}, order_by="start_ts DESC", limit=1))
    assert rows[0]["source"] == "backfill"
    assert store.write_sync(lambda c: store.update(c, "wait_history", {"cluster": "trace"}, hours=1.5)) == 2
    assert store.write_sync(lambda c: store.update(c, "wait_history", {"cluster": "trace"})) == 0
    assert store.write_sync(lambda c: store.delete(c, "wait_history", cluster="b2")) == 1
    store.write_sync(lambda c: store.upsert(c, "target_stats", cluster="trace", target_key="k", consecutive_failures=2))
    store.write_sync(lambda c: store.upsert(c, "target_stats", cluster="trace", target_key="k", consecutive_failures=3, last_error="x"))
    assert store.read_sync(lambda c: store.count(c, "target_stats")) == 1
    assert store.read_sync(lambda c: store.select_one(c, "target_stats", cluster="trace"))["consecutive_failures"] == 3
    with pytest.raises(ValueError):
        store.write_sync(lambda c: store.upsert(c, "target_stats", cluster="trace", consecutive_failures=1))


# --- lease (design section 5.8) -------------------------------------------------------------------

def test_lease_acquire_renew_release_and_status(store):
    info = store.write_sync(lambda c: store.lease_acquire(c, now=1000.0))
    assert info.acquired and info.token == 1 and info.reason == "new" and info.monitor == "self"
    again = store.write_sync(lambda c: store.lease_acquire(c, now=1001.0))
    assert again.acquired and again.token == 1 and again.reason == "mine"
    assert store.write_sync(lambda c: store.lease_renew(c, 1, now=1002.0)) is True
    assert store.write_sync(lambda c: store.lease_renew(c, 2)) is False
    assert store.write_sync(lambda c: store.lease_renew(c, None)) is False
    assert store.read_sync(lambda c: store.lease_get(c))["renewed_local"] == 1002.0
    assert store.read_sync(lambda c: store.monitor_status(c, 1)) == "self"
    assert store.read_sync(lambda c: store.monitor_status(c, None)) == "held by pid 1000"
    assert store.read_sync(lambda c: store.monitor_status(c, 7)) == "lost to pid 1000"
    assert store.write_sync(lambda c: store.lease_release(c, 2)) is False
    assert store.write_sync(lambda c: store.lease_release(c, 1)) is True
    assert store.read_sync(lambda c: store.monitor_status(c, None)) == "none"


def test_two_process_lease_rules(tmp_path):
    path = tmp_path / "state.db"
    alive = {1: True}
    holder = Store(path, pid=1, host="lap", pid_exists=lambda pid: alive.get(pid, False))
    other = Store(path, pid=2, host="lap", pid_exists=lambda pid: alive.get(pid, False))
    try:
        first = holder.write_sync(lambda c: holder.lease_acquire(c, now=0.0))
        assert first.token == 1
        # fresh lease, live pid -> never taken
        r = other.write_sync(lambda c: other.lease_acquire(c, now=10.0))
        assert not r.acquired and r.reason == "held-live" and r.owner_pid == 1 and r.monitor == "held by pid 1"
        # stale lease but live pid (suspended laptop) -> still never taken automatically
        r = other.write_sync(lambda c: other.lease_acquire(c, now=LEASE_STALE_S + 1.0))
        assert not r.acquired and r.reason == "held-live"
        # fresh lease but the owner pid is provably dead on this host -> taken over at once (token + 1).
        # A dead local pid cannot resume and write, so waiting out LEASE_STALE_S would only leave jobs
        # unmonitored; measured 2026-09-02, every Claude Code restart cost 5 minutes of monitoring.
        alive[1] = False
        r = other.write_sync(lambda c: other.lease_acquire(c, now=100.0))
        assert r.acquired and r.token == 2 and r.reason == "dead-owner"
        # the old holder's renew returns 0 rows and its fenced writes raise LeaseLost
        assert holder.write_sync(lambda c: holder.lease_renew(c, 1)) is False
        with pytest.raises(LeaseLost) as ei:
            holder.write_fenced_sync(1, lambda c: holder.kv_set(c, "never", 1))
        assert ei.value.token == 1 and ei.value.holder["owner_pid"] == 2
        assert holder.read_sync(lambda c: holder.kv_get(c, "never")) is None
        assert holder.read_sync(lambda c: holder.monitor_status(c, 1)) == "lost to pid 2"
        # the new holder's fenced writes succeed
        other.write_fenced_sync(2, lambda c: other.kv_set(c, "ok", 1))
        assert other.read_sync(lambda c: other.kv_get(c, "ok")) == 1
        # force takes a live lease (human takeover)
        alive[2] = True
        r = holder.write_sync(lambda c: holder.lease_acquire(c, force=True, now=1.0))
        assert r.acquired and r.token == 3 and r.reason == "forced"
        with pytest.raises(LeaseLost):
            other.write_fenced_sync(2, lambda c: None)
    finally:
        holder.close()
        other.close()


def test_lease_on_another_host_cannot_be_checked(tmp_path):
    path = tmp_path / "state.db"
    a = Store(path, pid=1, host="hostA", pid_exists=lambda pid: True)
    b = Store(path, pid=1, host="hostB", pid_exists=lambda pid: True)
    try:
        a.write_sync(lambda c: a.lease_acquire(c, now=0.0))
        r = b.write_sync(lambda c: b.lease_acquire(c, now=1.0))
        assert not r.acquired and r.reason == "held-unknown"
        r = b.write_sync(lambda c: b.lease_acquire(c, now=LEASE_STALE_S + 1.0))
        assert r.acquired and r.reason == "stale" and r.token == 2
    finally:
        a.close()
        b.close()


@pytest.mark.asyncio
async def test_async_write_fenced(store):
    info = await store.write(lambda c: store.lease_acquire(c))
    await store.write_fenced(info.token, lambda c: store.kv_set(c, "fenced", True))
    assert await store.read(lambda c: store.kv_get(c, "fenced")) is True
    with pytest.raises(LeaseLost):
        await store.write_fenced(info.token + 1, lambda c: None)
    with pytest.raises(LeaseLost):
        await store.write_fenced(None, lambda c: None)


# --- transfers / manifests / alloc_cmds / wait_history / target_stats / array_tasks -------------------

def test_transfers_and_transfer_files(store):
    def setup(c):
        tid = store.insert_transfer(c, kind="upload", cluster="trace", host_role="transfer", local="C:/p", remote="/r",
                                    state="running", mode="tar", files_total=2, bytes_total=30, now=5.0)
        store.upsert_transfer_file(c, tid, "a/b.txt", size=10, mtime_ns=1, state="planned")
        store.upsert_transfer_file(c, tid, "c.bin", size=20, mtime_ns=2, state="planned", sha1="abc")
        return tid
    tid = store.write_sync(setup)
    assert tid == 1 and transfer_handle(tid) == "t1" and parse_transfer_handle("t1") == 1
    assert parse_transfer_handle("j1") is None and parse_transfer_handle("t") is None
    assert store.read_sync(lambda c: store.get_transfer(c, tid))["started_local"] == 5.0
    store.write_sync(lambda c: store.update_transfer_file(c, tid, "a/b.txt", state="done", bytes_done=10, local_name="a/b_.txt"))
    store.write_sync(lambda c: store.upsert_transfer_file(c, tid, "c.bin", size=20, mtime_ns=2, state="done", bytes_done=20))
    files = store.read_sync(lambda c: store.transfer_files_for(c, tid))
    assert [f["rel_path"] for f in files] == ["a/b.txt", "c.bin"]
    assert files[0]["local_name"] == "a/b_.txt" and files[1]["sha1"] == "abc"
    assert store.read_sync(lambda c: store.transfer_files_for(c, tid, states=["planned"])) == []
    store.write_sync(lambda c: store.update_transfer(c, tid, state="done", files_done=2, bytes_done=30, finished_local=9.0, seconds=4.0))
    assert store.read_sync(lambda c: store.list_transfers(c, states=["done"]))[0]["seconds"] == 4.0
    assert store.read_sync(lambda c: store.list_transfers(c, handle="j1")) == []
    with pytest.raises(sqlite3.IntegrityError):
        store.write_sync(lambda c: store.upsert_transfer_file(c, 999, "x", size=1, mtime_ns=1, state="planned"))


def test_manifests(store):
    store.write_sync(lambda c: store.upsert_manifest(c, "up:trace:/r", "a.txt", size=1, mtime_ns=10, now=1.0))
    store.write_sync(lambda c: store.upsert_manifest(c, "up:trace:/r", "b.txt", size=2, mtime_ns=20, sha1="s", now=2.0))
    store.write_sync(lambda c: store.upsert_manifest(c, "up:trace:/r", "a.txt", size=3, mtime_ns=30, now=3.0))
    m = store.read_sync(lambda c: store.manifest(c, "up:trace:/r"))
    assert set(m) == {"a.txt", "b.txt"} and m["a.txt"]["size"] == 3 and m["a.txt"]["updated_local"] == 3.0
    assert store.read_sync(lambda c: store.manifest(c, "down:C:/x")) == {}
    assert store.write_sync(lambda c: store.delete_manifest(c, "up:trace:/r", "a.txt")) == 1
    assert store.write_sync(lambda c: store.delete_manifest(c, "up:trace:/r")) == 1


def test_alloc_cmds_numbering(store):
    def setup(c):
        add_job(store, c, "a3", kind="alloc", name="alloc")
        ids = []
        for mode in ("fg", "bg", "fg"):
            base = f"00{len(ids) + 1}" + (".bg" if mode == "bg" else "")
            cmd_id, n = store.insert_alloc_cmd(c, handle="a3", command="nvidia-smi", mode=mode, state="queued",
                                               out_path=f"/c/cmds/{base}.out", kill_path=f"/c/cmds/{base}.kill", now=1.0)
            ids.append((cmd_id, n))
        return ids
    ids = store.write_sync(setup)
    assert ids == [("a3.c1", 1), ("a3.c2", 2), ("a3.c3", 3)]
    assert store.read_sync(lambda c: store.get_alloc_cmd(c, "a3.c2"))["kill_path"] == "/c/cmds/002.bg.kill"
    store.write_sync(lambda c: store.update_alloc_cmd(c, "a3.c1", state="done", rc=0, done_ts=5))
    assert [r["id"] for r in store.read_sync(lambda c: store.alloc_cmds_for(c, "a3", states=["queued"]))] == ["a3.c2", "a3.c3"]
    assert [r["n"] for r in store.read_sync(lambda c: store.alloc_cmds_for(c, "a3"))] == [1, 2, 3]


def test_wait_history_and_target_stats(store):
    def setup(c):
        store.insert_wait_history(c, cluster="trace", target_key="trace:gpu", submit_ts=100, start_ts=160, source="observed", gpus=1)
        store.insert_wait_history(c, cluster="trace", target_key="trace:gpu", submit_ts=10, start_ts=20, source="backfill", wait_s=5)
        store.upsert_target_stats(c, "trace", "trace:gpu", consecutive_failures=1, last_error="e")
        store.upsert_target_stats(c, "trace", "trace:gpu", consecutive_failures=0, breaker_open_until_local=None)
    store.write_sync(setup)
    rows = store.read_sync(lambda c: store.wait_history(c, "trace", "trace:gpu"))
    assert [r["wait_s"] for r in rows] == [60, 5]
    assert len(store.read_sync(lambda c: store.wait_history(c, "trace", "trace:gpu", since_ts=100, limit=5))) == 1
    stats = store.read_sync(lambda c: store.get_target_stats(c, "trace", "trace:gpu"))
    assert stats["consecutive_failures"] == 0 and stats["last_error"] == "e"
    assert store.read_sync(lambda c: store.get_target_stats(c, "b2", "x")) is None


def test_array_tasks_upsert(store):
    def setup(c):
        add_job(store, c, "j18")
        store.upsert_array_task(c, "j18", 7, state="RUNNING", slurm_id="100_7")
        store.upsert_array_task(c, "j18", 3, state="PENDING")
        store.upsert_array_task(c, "j18", 7, state="COMPLETED", exit_code=0, end_ts=5)
    store.write_sync(setup)
    rows = store.read_sync(lambda c: store.array_tasks_for(c, "j18"))
    assert [(r["task_id"], r["state"]) for r in rows] == [(3, "PENDING"), (7, "COMPLETED")]
    assert rows[1]["slurm_id"] == "100_7" and rows[1]["exit_code"] == 0


# --- lease takeover from a dead local owner (measured 2026-09-02: Claude Code restarts left the Monitor
# --- dead for LEASE_STALE_S because the previous server's pid was gone but its lease was still "fresh") ---

def test_lease_taken_over_immediately_when_local_owner_is_dead(tmp_path, monkeypatch):
    from slurm_mcp import store as store_mod

    s = store_mod.Store(tmp_path / "state.db")
    try:
        # A previous server on this host took the lease and renewed it one second ago, then died.
        info = s.write_sync(lambda c: s.lease_acquire(c, now=1000.0))
        assert info.acquired
        s.write_sync(lambda c: s.update(c, "lease", {"name": store_mod.LEASE_NAME},
                                        owner_pid=999999, owner_host=s.host, renewed_local=1000.0))
        monkeypatch.setattr(s, "_pid_exists", lambda pid: False)
        got = s.write_sync(lambda c: s.lease_acquire(c, now=1001.0))   # one second later, not "stale"
        assert got.acquired and got.reason == "dead-owner"
        assert got.token == info.token + 1, "a takeover must bump the fencing token"
    finally:
        s.close()


def test_lease_not_taken_over_while_the_local_owner_lives(tmp_path, monkeypatch):
    from slurm_mcp import store as store_mod

    s = store_mod.Store(tmp_path / "state.db")
    try:
        s.write_sync(lambda c: s.lease_acquire(c, now=1000.0))
        s.write_sync(lambda c: s.update(c, "lease", {"name": store_mod.LEASE_NAME},
                                        owner_pid=999999, owner_host=s.host, renewed_local=1000.0))
        monkeypatch.setattr(s, "_pid_exists", lambda pid: True)
        # even long past the staleness window a live pid keeps the lease
        got = s.write_sync(lambda c: s.lease_acquire(c, now=1000.0 + store_mod.LEASE_STALE_S + 60))
        assert not got.acquired and got.reason == "held-live"
    finally:
        s.close()
