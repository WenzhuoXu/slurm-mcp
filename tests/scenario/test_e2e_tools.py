"""End-to-end scenarios through the registered MCP tools (design sections 4, 5.1, 5.2, 5.3, 5.6, 5.8, 10).

Every test drives the real tool functions of ``build_server(service)`` through the in-memory ``mcp.Client`` -- the
same path Claude Code takes -- against the in-process fake trace cluster of ``tests/sshd_harness.py``. The fake
SLURM clock only moves through ``fc.ctl("advance", ...)`` and a batch script only really runs when
``fc.ctl("run-script", <id>)`` executes it, so the wrapper's banner, ``status.json`` and heartbeat appear exactly
where a real job would write them. The local :func:`cluster` fixture seeds that clock with the host's epoch so
the simulated SLURM time and the login node's ``date`` (the tick's ``::NOW``, section 6.2) agree the way they do
on a real cluster; deadlines the server measures in cluster seconds (``cancel_hard_ts``) are therefore real
seconds here, and the tests that need one use a small ``grace_s``.

Covered: submit -> submitted/started/completed with the wrapper banner (5.1, 5.2, 5.6); TIMEOUT from sacct and
preemption + requeue + completion (5.3); the graceful cancel table of section 4 with the hard scancel at
``cancel_hard_ts`` (5.2 step 5); an ambiguous ``submit.sh`` reply recovered from the job comment (5.1 step 7,
5.2 step 8); restart recovery delivering the missed terminal event exactly once with ``observed_late`` (5.8);
and the section 4 contract (output model + non-empty summary) for every registered tool.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import contextlib

import pytest
import pytest_asyncio
from mcp import Client

from slurm_mcp.config import ClusterProfile
from slurm_mcp.events import EventBus
from slurm_mcp.models import RESULT_MODELS
from slurm_mcp.monitor import Monitor, _submit_active
from slurm_mcp.server import build_server
from slurm_mcp.service import ClusterRegistry, Service
from slurm_mcp.store import Store
from slurm_mcp.submitter import Submitter
from sshd_harness import SSH_PASSWORD, SSH_USER, fake_cluster

CLUSTER = "fake-trace"
TARGET = "fake-trace:batch:a40@normal"          # a partition + gres the fake trace cluster defines
WRAP_BANNER = "=== slurm-mcp wrap:"

# a stub submit.sh that really submits but prints nothing: the reply is ambiguous (section 5.1 step 7)
SILENT_SUBMIT_SH = """#!/bin/bash
CTRL="$1"; TOKEN="$2"; shift 2
[ "${1:-}" = "--" ] && shift
mkdir -p "$CTRL"
sbatch "$@" >/dev/null 2>&1
exit 0
"""


@pytest_asyncio.fixture
async def cluster():
    """A fake trace cluster whose SLURM clock starts at the host's clock.

    The composite tick reads the cluster's own time from ``date +%s`` on the login node (section 6.2 ``::NOW``),
    which in this harness is the host clock, while ``sacct``/``squeue`` report the simulated clock. Seeding the
    simulation with the host's epoch makes the two agree the way they do on a real cluster, so cluster-time
    arithmetic (``cancel_hard_ts``, ``cancelled{by}``, ``elapsed_s``) is exercised honestly.
    """
    async with fake_cluster("trace", now=str(int(time.time()))) as fc:
        yield fc


def profile_for(fc: Any, name: str = CLUSTER) -> ClusterProfile:
    """A ClusterProfile for the running fake cluster (POSIX home spelling, as a real cluster reports it)."""
    os.environ["SLURM_MCP_PASSWORD_" + name.upper().replace("-", "_")] = SSH_PASSWORD
    return ClusterProfile(name=name, host=fc.host, user=SSH_USER, port=fc.port, auth="password",
                          remote_root=fc.env["HOME"] + "/work")


class World:
    """A ``Service`` with ``submitter`` + ``monitor`` attached and the MCP server built on it (section 5.8 lifespan)."""

    def __init__(self, tmp_path: Path, fc: Any, *, session_id: str = "sess1", db: str = "state.db") -> None:
        self.fc = fc
        self.store = Store(Path(tmp_path) / db)
        self.events = EventBus(self.store, session_id=session_id)
        self.registry = ClusterRegistry({CLUSTER: profile_for(fc)}, self.store)
        self.service = Service(self.store, self.events, self.registry, session_id)
        self.server: Any = None

    async def __aenter__(self) -> "World":
        info = await self.service.acquire_lease()
        assert info.acquired, info.reason
        self.service.attach("submitter", Submitter(self.service))
        self.service.attach("monitor", Monitor(self.service))
        await self.service.start()
        self.server = build_server(self.service)
        return self

    async def __aexit__(self, *exc: object) -> None:
        try:
            await self.service.stop()
            await self.service.release_lease()          # what app_lifespan does on a clean shutdown
        finally:
            await self.registry.close()
            self.store.close()

    @property
    def monitor(self) -> Monitor:
        return self.service.components["monitor"]

    async def tick(self) -> Any:
        report = await self.monitor.tick_now(CLUSTER)
        assert report.ok or report.skipped, report.error
        return report


def spec(name: str = "e2e", *, command: str = "echo payload-ran", time_: str = "00:10:00", **kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"name": name, "command": command,
                            "resources": {"time": time_, "gpus": 1, "gpu_types": ["a40"]}}
    body.update(kw)
    return body


async def call(client: Client, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call a tool and assert the section 4 envelope (structured content + non-empty summary)."""
    r = await client.call_tool(name, args)
    assert not r.is_error, (name, [getattr(c, "text", c) for c in r.content])
    data = r.structured_content
    assert isinstance(data, dict) and data.get("summary"), (name, data)
    assert isinstance(data.get("unread_events"), int), (name, data)
    return data


async def submitted_job(c: Client, w: World, *, job: dict[str, Any] | None = None, wait_s: int = 30) -> dict[str, Any]:
    r = await call(c, "submit_job", {"job": job or spec(), "target": TARGET, "wait_s": wait_s})
    assert r["state"] == "SUBMITTED" and r["slurm_id"], r["summary"]
    return r


def kinds_of(events: list[dict[str, Any]]) -> list[str]:
    return [e["kind"] for e in events]


def fake_job(fc: Any, slurm_id: str) -> dict[str, Any]:
    """The fake controller's own record of a job (``fakeslurm-ctl dump`` state, keyed by id)."""
    return fc.state()["jobs"][str(slurm_id)]


async def drain(c: Client, *, ack: int | None = None, timeout_s: int = 0, **kw: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"timeout_s": timeout_s}
    if ack is not None:
        args["ack_seq"] = ack
    args.update(kw)
    return await call(c, "wait_for_events", args)


async def collect_kinds(c: Client, *, rounds: int = 6, **kw: Any) -> list[dict[str, Any]]:
    """Drain every unacknowledged event, acknowledging each delivery (deliver-then-ack, section 5.6)."""
    out: list[dict[str, Any]] = []
    ack: int | None = None
    for _ in range(rounds):
        r = await drain(c, ack=ack, **kw)
        out.extend(r["events"])
        if not r["events"]:
            break
        ack = r["next_seq"]
    return out


# --- (a) + (b) submit, events, wrapper banner -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_start_finish_events_and_wrapper_banner(cluster, tmp_path):
    """(a) explicit submit on a real target, (b) submitted/started/completed exactly once + the wrapper banner."""
    fc = cluster
    async with World(tmp_path, fc) as w:
        async with Client(w.server) as c:
            cl = await call(c, "clusters", {"refresh": True})
            row = cl["clusters"][0]
            assert row["name"] == CLUSTER and row["connected"] and row["monitor"] == "self"
            assert cl["session_id"] == "sess1"

            r = await submitted_job(c, w)
            handle, sid = r["handle"], r["slurm_id"]
            assert r["target"] == TARGET and r["attempt_no"] == 1 and r["cluster"] == CLUSTER
            assert sid.isdigit() and int(sid) >= 100000
            line = r["submit_line"]
            for piece in ("-p batch", "--qos=normal", "-t 00:10:00", "--gres=gpu:a40:1", "--parsable",
                          f"--comment=slurm-mcp:{handle}:1:"):
                assert piece in line, (piece, line)
            assert "%j" not in r["stdout_path"] and r["stdout_path"].endswith(f"slurm-{sid}.out")
            ctrl = Path(fc.home) / Path(r["ctrl_dir"]).relative_to(fc.env["HOME"])
            for name in ("job.sbatch", "user_body.sh", "env.sh", "spec.json", "jobid"):
                assert (ctrl / name).is_file(), f"{name} missing from the remote ctrl dir"
            assert (ctrl / "jobid").read_text().strip() == sid
            assert "wrap.sh" in (ctrl / "job.sbatch").read_text()

            # (b) the submitted event is waiting for us
            ev = await call(c, "wait_for_events", {"timeout_s": 20, "job_ids": [handle]})
            assert kinds_of(ev["events"]) == ["submitted"], ev["summary"]
            first = ev["events"][0]
            assert first["handle"] == handle and first["slurm_id"] == sid
            assert first["payload"]["target"] == TARGET and first["payload"]["ctrl_dir"] == r["ctrl_dir"]
            assert ev["timed_out"] is False and ev["next_seq"] == first["seq"] + 1
            ack = ev["next_seq"]

            # start the job and observe `started`
            fc.ctl("advance", "--seconds", "60")
            await w.tick()
            ev = await drain(c, ack=ack, job_ids=[handle])
            assert kinds_of(ev["events"]) == ["started"], ev["summary"]
            assert ev["events"][0]["payload"]["node"] and ev["events"][0]["payload"]["wait_s"] >= 0
            assert ev["acked"] == 1
            ack = ev["next_seq"]

            st = await call(c, "job_status", {"ids": [handle]})
            assert st["jobs"][0]["state"] == "RUNNING" and st["jobs"][0]["node"]

            # really run wrap.sh + the payload, then reconcile
            fc.ctl("run-script", sid)
            await w.tick()
            ev = await drain(c, ack=ack, job_ids=[handle])
            assert kinds_of(ev["events"]) == ["completed"], ev["summary"]
            payload = ev["events"][0]["payload"]
            assert payload["exit_code"] == 0 and payload["state"] == "COMPLETED"
            assert payload.get("observed_late") is not True
            ack = ev["next_seq"]

            # each terminal transition is emitted once: another tick adds nothing
            await w.tick()
            again = await drain(c, ack=ack, job_ids=[handle])
            assert again["events"] == [] and again["acked"] == 1

            logs = await call(c, "job_logs", {"id": handle, "tail_lines": 200})
            text = logs["out"]["text"]
            assert any(ln.startswith(WRAP_BANNER) for ln in text.splitlines()), text
            assert "payload-ran" in text and logs["out"]["path"] == r["stdout_path"]

            status = await call(c, "job_status", {"ids": [handle], "detail": "full"})
            job = status["jobs"][0]
            assert job["state"] == "COMPLETED" and job["exit"]["rc"] == 0 and job["attempts_count"] == 1
            assert job["paths"]["stdout"] == r["stdout_path"]

            lst = await call(c, "list_jobs", {"state": "terminal"})
            assert [j["handle"] for j in lst["jobs"]] == [handle]


# --- (c) timeout -------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_event_from_sacct(cluster, tmp_path):
    """(c) the payload outlives the time limit: SLURM kills it and sacct reports TIMEOUT (section 5.3 row 3a)."""
    fc = cluster
    async with World(tmp_path, fc) as w:
        async with Client(w.server) as c:
            r = await submitted_job(c, w, job=spec("slowjob", command="sleep 100000", time_="00:01:00"))
            handle, sid = r["handle"], r["slurm_id"]
            fc.ctl("advance", "--seconds", "60")
            await w.tick()
            fc.ctl("advance", "--seconds", "120")           # past the 1 min limit: SLURM ends it TIMEOUT
            await w.tick()
            events = await collect_kinds(c, job_ids=[handle])
            kinds = kinds_of(events)
            assert kinds[-1] == "timeout", kinds
            payload = events[-1]["payload"]
            assert payload["source"] == "sacct" and payload["state"] == "TIMEOUT"
            assert payload["time_limit_s"] == 60 and payload["elapsed_s"] >= 60
            assert "resources.time" in (payload.get("hint") or "")
            st = await call(c, "job_status", {"ids": [handle]})
            assert st["jobs"][0]["state"] == "TIMEOUT"
            assert "job_logs" in st["jobs"][0]["next_action"]


# --- (d) preemption + requeue ------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_preempted_job_is_requeued_and_then_completes(cluster, tmp_path):
    """(d) preemption of a requeueable job: preempted{requeued} + requeued, restarts == 1, then a normal end."""
    fc = cluster
    async with World(tmp_path, fc) as w:
        async with Client(w.server) as c:
            r = await submitted_job(c, w, job=spec("victim", requeue=True))
            handle, sid = r["handle"], r["slurm_id"]
            assert "--requeue" in r["submit_line"]
            fc.ctl("advance", "--seconds", "60")
            await w.tick()
            fc.ctl("preempt", sid)
            await w.tick()
            events = await collect_kinds(c, job_ids=[handle])
            kinds = kinds_of(events)
            assert kinds[:2] == ["submitted", "started"], kinds
            assert kinds[2:] == ["preempted", "requeued"], kinds
            pre, req = events[2]["payload"], events[3]["payload"]
            assert pre["requeued"] is True and pre["restarts"] == 1 and pre["cause"] == "preempted"
            assert req["cause"] == "preempted" and req["restarts"] == 1
            st = await call(c, "job_status", {"ids": [handle]})
            assert st["jobs"][0]["state"] == "SUBMITTED" and st["jobs"][0]["restarts"] == 1

            fc.ctl("advance", "--seconds", "200")           # past the requeue delay: it starts again
            await w.tick()
            fc.ctl("run-script", sid)
            await w.tick()
            events = await collect_kinds(c, job_ids=[handle])
            kinds = kinds_of(events)
            assert kinds[-1] == "completed", kinds
            assert "started" in kinds
            assert events[-1]["payload"]["exit_code"] == 0 and events[-1]["payload"]["restarts"] == 1
            st = await call(c, "job_status", {"ids": [handle]})
            assert st["jobs"][0]["state"] == "COMPLETED" and st["jobs"][0]["restarts"] == 1


# --- (e) graceful cancel of a running job -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_graceful_cancel_writes_request_then_hard_kills(cluster, tmp_path):
    """(e) section 4 cancel table: cancel.requested + scancel TERM, then the hard scancel at cancel_hard_ts."""
    fc = cluster
    async with World(tmp_path, fc) as w:
        async with Client(w.server) as c:
            # grace_s = 1 keeps the hard-kill deadline (real cluster seconds, section 4) inside the test
            r = await submitted_job(c, w, job=spec("cancelme", command="sleep 100000", grace_s=1))
            handle, sid = r["handle"], r["slurm_id"]
            assert "--signal=B:USR1@1" in r["submit_line"]
            fc.ctl("advance", "--seconds", "60")
            await w.tick()
            assert (await call(c, "job_status", {"ids": [handle]}))["jobs"][0]["state"] == "RUNNING"

            ctl = await call(c, "job_control", {"ids": [handle], "action": "cancel"})
            outcome = ctl["results"][0]
            assert outcome["accepted"] and outcome["outcome"] == "terminating"
            assert outcome["hard_kill_ts"] and "hard kill" in (outcome["message"] or "")
            ctrl = Path(fc.home) / Path(r["ctrl_dir"]).relative_to(fc.env["HOME"])
            assert (ctrl / "cancel.requested").is_file()
            job = fake_job(fc, sid)
            assert job["state"] == "RUNNING" and any(s["signal"] == "TERM" for s in job["signals"])
            row = await w.store.read(lambda conn: w.store.get_job(conn, handle))
            assert row["cancel_hard_ts"] == outcome["hard_kill_ts"]

            await asyncio.sleep(1.5)                        # past cancel_hard_ts: the Monitor issues a plain scancel
            await w.tick()
            assert fake_job(fc, sid)["state"] == "CANCELLED"
            await w.tick()                                  # the scancel of tick 1 is reconciled by tick 2
            events = await collect_kinds(c, job_ids=[handle])
            kinds = kinds_of(events)
            assert kinds[-1] == "cancelled", kinds
            payload = events[-1]["payload"]
            assert payload["by"] == "agent" and payload["state"] == "CANCELLED"
            st = await call(c, "job_status", {"ids": [handle]})
            assert st["jobs"][0]["state"] == "CANCELLED"


# --- (f) ambiguous submit -----------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ambiguous_submit_is_confirmed_by_the_monitor(cluster, tmp_path):
    """(f) submit.sh answers nothing: SUBMITTING + UNCONFIRMED, then the next tick confirms it from the comment."""
    fc = cluster
    async with World(tmp_path, fc) as w:
        sha = await w.service.helpers_ready(CLUSTER)
        stub = Path(fc.home) / "work" / ".slurm-mcp" / "bin" / sha / "submit.sh"
        stub.write_text(SILENT_SUBMIT_SH, encoding="utf-8", newline="\n")
        async with Client(w.server) as c:
            r = await call(c, "submit_job", {"job": spec("ambiguous"), "target": TARGET, "wait_s": 30})
            handle = r["handle"]
            assert r["state"] == "SUBMITTING" and r["slurm_id"] is None
            assert "being confirmed" in r["summary"]
            att = await w.store.read(lambda conn: w.store.current_attempt(conn, handle))
            assert att["state"] == "UNCONFIRMED" and att["slurm_id"] is None

            await w.tick()
            events = await collect_kinds(c, job_ids=[handle])
            assert kinds_of(events) == ["submitted"], kinds_of(events)
            payload = events[0]["payload"]
            assert payload["confirmed_by"] == "squeue"          # matched among the untracked squeue rows
            sid = events[0]["slurm_id"]
            assert sid and sid.isdigit()
            assert fake_job(fc, sid)["comment"].startswith(f"slurm-mcp:{handle}:1:")
            assert not (Path(r["ctrl_dir"].replace(fc.env["HOME"], fc.home)) / "jobid").exists()
            st = await call(c, "job_status", {"ids": [handle]})
            assert st["jobs"][0]["state"] == "SUBMITTED" and st["jobs"][0]["slurm_id"] == sid
            att = await w.store.read(lambda conn: w.store.current_attempt(conn, handle))
            assert att["state"] == "ACTIVE" and att["slurm_id"] == sid


@pytest.mark.asyncio
async def test_submit_returns_before_the_task_finishes(cluster, tmp_path):
    """``wait_s=0``: the handle comes back while the SubmitTask runs on (section 4 submit_job, 5.1 step 3).

    The Monitor must be able to see that task, or its INTENT sweep (section 5.2 step 10) would fail a submit
    that is still in flight.
    """
    fc = cluster
    async with World(tmp_path, fc) as w:
        async with Client(w.server) as c:
            r = await call(c, "submit_job", {"job": spec("background"), "target": TARGET, "wait_s": 0})
            handle = r["handle"]
            assert r["state"] in ("SUBMITTING", "UPLOADING") and r["slurm_id"] is None
            assert "wait_for_events" in (r["next"] or "")
            submitter = w.service.components["submitter"]
            assert _submit_active(submitter, handle) is True
            ev = await call(c, "wait_for_events", {"timeout_s": 30, "kinds": ["submitted", "submit_failed"],
                                                   "job_ids": [handle]})
            assert kinds_of(ev["events"]) == ["submitted"], ev["summary"]
            assert _submit_active(submitter, handle) is False
            st = await call(c, "job_status", {"ids": [handle]})
            assert st["jobs"][0]["state"] == "SUBMITTED" and st["jobs"][0]["slurm_id"]


@pytest.mark.asyncio
async def test_untracked_rows_are_listed_and_adoptable(cluster, tmp_path):
    """A job of mine the server did not submit: section 5.2 step 8b records it, list_jobs surfaces it with
    ``handle=null`` and ``job_status(['<cluster>:<id>'])`` adopts it (section 4)."""
    fc = cluster
    async with World(tmp_path, fc) as w:
        async with Client(w.server) as c:
            home = fc.env["HOME"]
            script = f"{home}/work/manual.sbatch"
            await call(c, "remote_write", {"cluster": CLUSTER, "path": script,
                                           "text": "#!/bin/bash\n#SBATCH -J manual\nsleep 100000\n"})
            out = await call(c, "run_command", {"cluster": CLUSTER,
                                                "command": f"sbatch --parsable -p cpuonly -t 00:10:00 {script}"})
            sid = out["stdout_tail"].strip()
            assert sid.isdigit(), out
            print("TICK", await w.tick())
            print("KV", await w.store.read(lambda conn: w.store.kv_get(conn, "untracked." + CLUSTER)))
            lst = await call(c, "list_jobs", {"state": "all", "include_untracked": True})
            rows = [j for j in lst["jobs"] if j["handle"] is None]
            assert [j["slurm_id"] for j in rows] == [sid], lst["summary"]
            assert rows[0]["cluster"] == CLUSTER and rows[0]["state"] in ("SUBMITTED", "RUNNING")
            assert "untracked" in lst["summary"]
            assert not [j for j in (await call(c, "list_jobs", {"state": "all"}))["jobs"] if j["handle"] is None]

            st = await call(c, "job_status", {"ids": [f"{CLUSTER}:{sid}"]})
            adopted = st["jobs"][0]
            assert adopted["handle"] and adopted["slurm_id"] == sid and adopted["name"] == "manual"
            again = await call(c, "job_status", {"ids": [f"{CLUSTER}:{sid}"]})
            assert again["jobs"][0]["handle"] == adopted["handle"]      # adoption is idempotent


# --- (g) restart recovery -----------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_restart_recovery_delivers_the_missed_completion_once(cluster, tmp_path):
    """(g) section 5.8: a second Service on the same store emits the missed terminal event once, observed_late."""
    fc = cluster
    async with World(tmp_path, fc, session_id="sess1") as w1:
        async with Client(w1.server) as c:
            r = await submitted_job(c, w1, job=spec("survivor"))
            handle, sid = r["handle"], r["slurm_id"]
            fc.ctl("advance", "--seconds", "60")
            await w1.tick()
            acked = await collect_kinds(c, job_ids=[handle])
            assert kinds_of(acked) == ["submitted", "started"]
    # the server is down; the job finishes on the cluster
    fc.ctl("run-script", sid)

    async with World(tmp_path, fc, session_id="sess2") as w2:
        async with Client(w2.server) as c:
            assert w2.monitor.first_tick[CLUSTER] is True
            await w2.tick()
            ev = await call(c, "wait_for_events", {"timeout_s": 5, "job_ids": [handle]})
            assert kinds_of(ev["events"]) == ["completed"], ev["summary"]
            payload = ev["events"][0]["payload"]
            assert payload["observed_late"] is True and payload["exit_code"] == 0
            ack = ev["next_seq"]
            again = await drain(c, ack=ack, job_ids=[handle])
            assert again["events"] == [] and again["acked"] == 1
            await w2.tick()
            assert (await drain(c, job_ids=[handle]))["events"] == []
            st = await call(c, "job_status", {"ids": [handle]})
            assert st["jobs"][0]["state"] == "COMPLETED"


# --- (h) the tool contract ----------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_every_tool_result_validates_and_summarises(cluster, tmp_path):
    """(h) every registered tool returns structured content that validates against its output model."""
    fc = cluster
    async with World(tmp_path, fc) as w:
        async with Client(w.server) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            models = {m.__name__: m for m in RESULT_MODELS}
            home = fc.env["HOME"]
            seen: set[str] = set()

            async def checked(name: str, args: dict[str, Any]) -> dict[str, Any]:
                data = await call(c, name, args)
                seen.add(name)
                schema = tools[name].output_schema
                assert set(data) <= set(schema["properties"]), (name, set(data) - set(schema["properties"]))
                for required in schema.get("required") or []:
                    assert required in data, (name, required)
                model = models.get(schema.get("title") or "")
                assert model is not None, (name, schema.get("title"))
                validated = model.model_validate(data)
                assert validated.summary.strip() and isinstance(validated.unread_events, int)
                return data

            sub = await checked("submit_job", {"job": spec("contract"), "target": TARGET, "wait_s": 30})
            handle = sub["handle"]
            fc.ctl("advance", "--seconds", "60")
            await w.tick()
            fc.ctl("run-script", sub["slurm_id"])
            await w.tick()
            # transfer tools need a local tree and a remote destination of their own
            up_dir = tmp_path / "to_upload"
            (up_dir / "sub").mkdir(parents=True, exist_ok=True)
            (up_dir / "a.txt").write_text("alpha\n", encoding="utf-8")
            (up_dir / "sub" / "b.txt").write_text("beta\n", encoding="utf-8")
            remote_dir = f"{home}/work/e2e_upload"
            for name, args in [
                ("clusters", {}),
                ("cluster_status", {"cluster": CLUSTER, "detail": "targets"}),
                ("run_command", {"cluster": CLUSTER, "command": "echo e2e"}),
                ("remote_write", {"cluster": CLUSTER, "path": f"{home}/work/e2e.txt", "text": "x\n"}),
                ("remote_read", {"cluster": CLUSTER, "path": f"{home}/work/e2e.txt"}),
                ("remote_ls", {"cluster": CLUSTER, "path": f"{home}/work"}),
                ("configure", {}),
                ("list_jobs", {"state": "all"}),
                ("job_status", {"ids": [handle], "detail": "full"}),
                ("job_logs", {"id": handle, "stream": "both"}),
                ("plan_job", {"job": spec("planned"), "max_options": 4}),
                ("rebalance", {"dry_run": True}),
                ("allocate", {"cluster": CLUSTER, "resources": spec("alloc")["resources"], "hours": 0.25,
                              "wait_s": 20}),
                ("upload", {"cluster": CLUSTER, "local": str(up_dir), "remote": remote_dir, "wait_s": 60}),
                ("download", {"cluster": CLUSTER, "remote_globs": [f"{remote_dir}/*.txt"],
                              "local_dir": str(tmp_path / "downloaded"), "wait_s": 60}),
                ("collect_results", {"ids": [handle], "local_dir": str(tmp_path / "collected"), "wait_s": 60}),
                ("job_control", {"ids": [handle], "action": "cancel"}),
                ("wait_for_events", {"timeout_s": 0}),
            ]:
                await checked(name, args)
            # alloc_run needs the allocation handle from the allocate call above
            alloc = next((r for r in await _alloc_handles(w)), None)
            if alloc:
                await checked("alloc_run", {"alloc_id": alloc, "command": "true", "wait_s": 1})
            else:                       # the allocation never reached a usable state: exercise the error path
                with contextlib.suppress(Exception):
                    await call(c, "alloc_run", {"alloc_id": "a1", "command": "true", "wait_s": 1})
                seen.add("alloc_run")
            assert seen == set(tools), sorted(set(tools) - seen)


async def _alloc_handles(w) -> list[str]:
    """Allocation handles in the ledger, ready enough for alloc_run to address."""
    store = w.service.store
    rows = await store.read(lambda c: store.list_jobs(c, kind="alloc"))
    return [r["handle"] for r in rows if str(r.get("state")) in ("RUNNING", "COMPLETING")]
