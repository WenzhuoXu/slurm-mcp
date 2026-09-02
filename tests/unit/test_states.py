"""Unit tests for slurm_mcp.slurm.states (design section 3.1, 5.2, 5.3)."""
from __future__ import annotations

import itertools

import pytest

from slurm_mcp.slurm.states import (LIVE, PRE_SLURM, SLURM_STATE_MAP, TERMINAL, AttemptState, CmdState, JobState,
                                    TransferState, classify_reason, is_live, is_terminal, map_slurm_state,
                                    transition)

J = JobState


def test_enum_values_and_sets():
    assert [s.value for s in J] == ["QUEUED", "UPLOADING", "SUBMITTING", "SUBMITTED", "RUNNING", "COMPLETING",
                                    "COMPLETED", "FAILED", "TIMEOUT", "OOM", "CANCELLED", "PREEMPTED",
                                    "NODE_FAIL", "LOST"]
    assert TERMINAL == {J.COMPLETED, J.FAILED, J.TIMEOUT, J.OOM, J.CANCELLED, J.PREEMPTED, J.NODE_FAIL, J.LOST}
    assert LIVE == {J.QUEUED, J.UPLOADING, J.SUBMITTING, J.SUBMITTED, J.RUNNING, J.COMPLETING}
    assert TERMINAL | LIVE == set(J) and not (TERMINAL & LIVE)
    assert PRE_SLURM < LIVE
    assert str(J.RUNNING) == "RUNNING" and J.RUNNING == "RUNNING"


def test_other_enums():
    assert [s.value for s in AttemptState] == ["INTENT", "UNCONFIRMED", "ACTIVE", "SUPERSEDED", "FAILED", "DONE"]
    assert [s.value for s in TransferState] == ["planned", "running", "done", "failed", "cancelled"]
    assert [s.value for s in CmdState] == ["queued", "running", "done", "killed", "aborted"]
    assert TransferState("done") is TransferState.done and str(CmdState.killed) == "killed"


DESIGN_MAP = {
    **{k: J.SUBMITTED for k in ["PENDING", "REQUEUED", "REQUEUE_HOLD", "SPECIAL_EXIT", "RESV_DEL_HOLD", "REQUEUE_FED"]},
    **{k: J.RUNNING for k in ["RUNNING", "SUSPENDED", "RESIZING", "CONFIGURING", "SIGNALING", "STAGE_OUT", "STOPPED"]},
    "COMPLETING": J.COMPLETING, "COMPLETED": J.COMPLETED,
    **{k: J.FAILED for k in ["FAILED", "BOOT_FAIL", "DEADLINE", "LAUNCH_FAILED", "REVOKED"]},
    "TIMEOUT": J.TIMEOUT, "OUT_OF_MEMORY": J.OOM, "CANCELLED": J.CANCELLED, "PREEMPTED": J.PREEMPTED,
    "NODE_FAIL": J.NODE_FAIL,
}


def test_slurm_state_map_is_exactly_section_3_1():
    assert SLURM_STATE_MAP == DESIGN_MAP


@pytest.mark.parametrize("token,expected", list(DESIGN_MAP.items()))
def test_map_slurm_state_each(token, expected):
    assert map_slurm_state(token) is expected


@pytest.mark.parametrize("token,expected", [
    ("CANCELLED by 123", J.CANCELLED), ("CANCELLED by 39874", J.CANCELLED), ("  PENDING  ", J.SUBMITTED),
    ("pending", J.SUBMITTED), ("COMPLETED+", J.COMPLETED), ("", None), (None, None), ("PD", None),
    ("WEIRD_STATE", None), ("R", None),
])
def test_map_slurm_state_edge(token, expected):
    assert map_slurm_state(token) is expected


@pytest.mark.parametrize("state", list(J))
def test_is_terminal_is_live(state):
    assert is_terminal(state) == (state in TERMINAL)
    assert is_terminal(state.value) == (state in TERMINAL)
    assert is_live(state) == (state in LIVE)


def test_is_terminal_garbage():
    assert not is_terminal(None) and not is_terminal("nope") and not is_live("nope")


@pytest.mark.parametrize("reason,cls", [
    ("None", "normal"), ("Priority", "normal"), ("Resources", "normal"), ("BeginTime", "normal"),
    ("Cleaning", "normal"), ("WaitingForScheduling", "normal"), ("SchedDefer", "normal"), ("", "normal"),
    (None, "normal"), ("(Priority)", "normal"), ("(Resources)", "normal"),
    ("QOSMaxJobsPerUserLimit", "limit"), ("QOSGrpGRES", "limit"), ("AssocGrpCPUMinutesLimit", "limit"),
    ("AssociationJobLimit", "limit"), ("AssocMaxJobsLimit", "limit"), ("QOSResourceLimit", "limit"),
    ("QOSTimeLimit", "limit"), ("QOSUsageThreshold", "limit"), ("PartitionNodeLimit", "limit"),
    ("PartitionTimeLimit", "limit"), ("JobArrayTaskLimit", "limit"), ("(QOSMaxGRESPerUser)", "limit"),
    ("JobHeldUser", "held"), ("JobHeldAdmin", "held"), ("JobHoldMaxRequeue", "held"),
    ("Dependency", "dependency"), ("DependencyNeverSatisfied", "dependency"),
    ("ReqNodeNotAvail, UnavailableNodes:trace[01-03]", "reservation"), ("Reservation", "reservation"),
    ("ReservationDeleted", "reservation"), ("Reserved for maintenance", "reservation"),
    ("NodeDown", "unknown"), ("BadConstraints", "unknown"), ("InvalidQOS", "unknown"), ("Licenses", "unknown"),
])
def test_classify_reason(reason, cls):
    assert classify_reason(reason) == cls


# --- transition ------------------------------------------------------------------------------------

@pytest.mark.parametrize("state", list(J))
def test_transition_duplicate_rejected_and_from_none_ok(state):
    assert transition(state, state) is False
    assert transition(None, state) is True
    assert transition(state.value, state.value) is False


@pytest.mark.parametrize("old,new", [
    (J.QUEUED, J.UPLOADING), (J.QUEUED, J.SUBMITTING), (J.UPLOADING, J.SUBMITTING), (J.SUBMITTING, J.SUBMITTED),
    (J.SUBMITTED, J.RUNNING), (J.RUNNING, J.COMPLETING), (J.COMPLETING, J.COMPLETED), (J.RUNNING, J.COMPLETED),
    (J.RUNNING, J.FAILED), (J.RUNNING, J.TIMEOUT), (J.RUNNING, J.OOM), (J.SUBMITTED, J.CANCELLED),
    (J.QUEUED, J.CANCELLED), (J.SUBMITTING, J.FAILED), (J.SUBMITTED, J.LOST), (J.RUNNING, J.PREEMPTED),
    (J.RUNNING, J.NODE_FAIL),
    (J.RUNNING, J.SUBMITTED), (J.COMPLETING, J.SUBMITTED),            # requeue observed live
    (J.PREEMPTED, J.SUBMITTED), (J.NODE_FAIL, J.SUBMITTED), (J.TIMEOUT, J.SUBMITTED),   # requeue path
    (J.UPLOADING, J.QUEUED), (J.SUBMITTING, J.QUEUED),                # held locally by a cap
    (J.FAILED, J.TIMEOUT), (J.FAILED, J.OOM), (J.CANCELLED, J.TIMEOUT),   # section 5.3 upgrades
    (J.LOST, J.COMPLETED), (J.LOST, J.FAILED),                          # sacct caught up
])
def test_transition_legal(old, new):
    assert transition(old, new) is True
    assert transition(old.value, new.value) is True


@pytest.mark.parametrize("old,new", [
    (J.COMPLETED, J.RUNNING), (J.COMPLETED, J.SUBMITTED), (J.FAILED, J.RUNNING), (J.CANCELLED, J.SUBMITTED),
    (J.OOM, J.SUBMITTED), (J.LOST, J.RUNNING), (J.COMPLETED, J.QUEUED), (J.TIMEOUT, J.RUNNING),
    (J.COMPLETED, J.FAILED), (J.TIMEOUT, J.FAILED), (J.OOM, J.FAILED), (J.COMPLETED, J.CANCELLED),
    (J.PREEMPTED, J.RUNNING), (J.NODE_FAIL, J.QUEUED),
    (J.RUNNING, J.QUEUED), (J.SUBMITTED, J.UPLOADING), (J.SUBMITTED, J.SUBMITTING), (J.RUNNING, J.UPLOADING),
    (J.COMPLETING, J.RUNNING), (J.SUBMITTED, J.QUEUED),
])
def test_transition_illegal(old, new):
    assert transition(old, new) is False


def test_transition_garbage():
    assert transition("NOPE", J.RUNNING) is False
    assert transition(J.RUNNING, "NOPE") is False
    assert transition(J.RUNNING, None) is False


def test_terminal_to_live_only_requeue_path():
    for old, new in itertools.product(TERMINAL, LIVE):
        ok = transition(old, new)
        assert ok == (new is J.SUBMITTED and old in {J.PREEMPTED, J.NODE_FAIL, J.TIMEOUT}), (old, new)
