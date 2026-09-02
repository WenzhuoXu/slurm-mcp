"""An allocation becomes ready when its agent is alive, not only in the one second it says "ready".

helpers/alloc-agent.sh writes `status ready` once, immediately before entering its event loop, and
then `status running` on every one-second pass. The monitor ticks about every 30 s, so it observes
"ready" only if a tick lands inside that single second. Keying alloc_ready off that exact value made
the event practically unreachable: a real 5-minute allocation on TRACE ran the agent on its node and
still produced only submitted -> started -> completed, so alloc_run would have answered
E_ALLOC_NOT_READY for the whole window.
"""
from __future__ import annotations

from typing import Any

from slurm_mcp.monitor import Observation, apply_observation
from slurm_mcp.slurm.states import JobState

NOW = 1_788_378_400


def _alloc_row(**over: Any) -> dict[str, Any]:
    row = {"handle": "a9", "kind": "alloc", "cluster": "trace", "slurm_id": "615824",
           "state": JobState.RUNNING, "alloc_ready": 0, "start_ts": NOW - 60, "restarts": 0,
           "spec_json": '{"name": "alloc", "command": ":", "resources": {"time": "00:15:00"}}'}
    row.update(over)
    return row


def _obs(phase: str) -> Observation:
    """A tick that read the agent's status.json with the given phase."""
    return Observation(
        squeue={"job_state": JobState.RUNNING, "node_list": "trace123"},
        files={"status.json": {"v": 2, "phase": phase, "node": "trace123",
                               "job_id": "615824", "now": NOW, "start": NOW - 60,
                               "fg": "", "running": 0}},
    )


def _kinds(outcome: Any) -> list[str]:
    return [kind for kind, _summary, _payload in outcome.events]


def test_running_agent_marks_the_allocation_ready() -> None:
    """The case that was broken: every tick after the first second reports "running"."""
    out = apply_observation(_alloc_row(), None, _obs("running"), NOW)
    assert out.job.get("alloc_ready") == 1
    assert "alloc_ready" in _kinds(out)


def test_ready_phase_still_marks_it_ready() -> None:
    out = apply_observation(_alloc_row(), None, _obs("ready"), NOW)
    assert out.job.get("alloc_ready") == 1
    assert "alloc_ready" in _kinds(out)


def test_exited_agent_does_not_mark_it_ready() -> None:
    out = apply_observation(_alloc_row(), None, _obs("exited"), NOW)
    assert out.job.get("alloc_ready") != 1
    assert "alloc_ready" not in _kinds(out)


def test_no_status_file_yet_does_not_mark_it_ready() -> None:
    """Queued, or running but the agent has not written status.json — the node is not usable yet."""
    obs = Observation(squeue={"job_state": JobState.RUNNING, "node_list": "trace123"}, files={})
    out = apply_observation(_alloc_row(), None, obs, NOW)
    assert out.job.get("alloc_ready") != 1
    assert "alloc_ready" not in _kinds(out)


def test_alloc_ready_is_emitted_only_once() -> None:
    """Applying the outcome and ticking again must not re-announce readiness."""
    out = apply_observation(_alloc_row(alloc_ready=1), None, _obs("running"), NOW)
    assert "alloc_ready" not in _kinds(out)
