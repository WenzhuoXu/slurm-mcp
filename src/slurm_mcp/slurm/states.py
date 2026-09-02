"""Job/attempt/transfer/command states, SLURM state mapping, transitions and reason classes
(design section 3.1; transition rules from sections 5.2 and 5.3).

Imports nothing from the package.
"""
from __future__ import annotations

import re
from enum import Enum


class JobState(str, Enum):
    """Our job states: a superset of SLURM base states (design section 3.1)."""

    QUEUED = "QUEUED"
    UPLOADING = "UPLOADING"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    OOM = "OOM"
    CANCELLED = "CANCELLED"
    PREEMPTED = "PREEMPTED"
    NODE_FAIL = "NODE_FAIL"
    LOST = "LOST"

    def __str__(self) -> str:  # so f"{state}" renders the bare value
        return self.value


TERMINAL: frozenset[JobState] = frozenset({
    JobState.COMPLETED, JobState.FAILED, JobState.TIMEOUT, JobState.OOM, JobState.CANCELLED,
    JobState.PREEMPTED, JobState.NODE_FAIL, JobState.LOST,
})
LIVE: frozenset[JobState] = frozenset({
    JobState.QUEUED, JobState.UPLOADING, JobState.SUBMITTING, JobState.SUBMITTED, JobState.RUNNING,
    JobState.COMPLETING,
})
# States before the job exists in SLURM (only submitter.py writes these; section 2).
PRE_SLURM: frozenset[JobState] = frozenset({JobState.QUEUED, JobState.UPLOADING, JobState.SUBMITTING})

# SLURM long state name (squeue %T / sacct State first token) -> ours. Exactly design section 3.1.
SLURM_STATE_MAP: dict[str, JobState] = {
    "PENDING": JobState.SUBMITTED,
    "REQUEUED": JobState.SUBMITTED,
    "REQUEUE_HOLD": JobState.SUBMITTED,
    "SPECIAL_EXIT": JobState.SUBMITTED,
    "RESV_DEL_HOLD": JobState.SUBMITTED,
    "REQUEUE_FED": JobState.SUBMITTED,
    "RUNNING": JobState.RUNNING,
    "SUSPENDED": JobState.RUNNING,
    "RESIZING": JobState.RUNNING,
    "CONFIGURING": JobState.RUNNING,
    "SIGNALING": JobState.RUNNING,
    "STAGE_OUT": JobState.RUNNING,
    "STOPPED": JobState.RUNNING,
    "COMPLETING": JobState.COMPLETING,
    "COMPLETED": JobState.COMPLETED,
    "FAILED": JobState.FAILED,
    "BOOT_FAIL": JobState.FAILED,
    "DEADLINE": JobState.FAILED,
    "LAUNCH_FAILED": JobState.FAILED,
    "REVOKED": JobState.FAILED,
    "TIMEOUT": JobState.TIMEOUT,
    "OUT_OF_MEMORY": JobState.OOM,
    "CANCELLED": JobState.CANCELLED,
    "PREEMPTED": JobState.PREEMPTED,
    "NODE_FAIL": JobState.NODE_FAIL,
}


class AttemptState(str, Enum):
    """Attempt lifecycle (design section 3.1)."""

    INTENT = "INTENT"
    UNCONFIRMED = "UNCONFIRMED"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    FAILED = "FAILED"
    DONE = "DONE"

    def __str__(self) -> str:
        return self.value


class TransferState(str, Enum):
    planned = "planned"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"

    def __str__(self) -> str:
        return self.value


class CmdState(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    killed = "killed"
    aborted = "aborted"

    def __str__(self) -> str:
        return self.value


def map_slurm_state(token: object) -> JobState | None:
    """Map a SLURM state string to ours; unknown/empty -> None (caller keeps the previous state).

    Only the first whitespace-separated token counts, so ``"CANCELLED by 12345"`` -> CANCELLED and a
    ``"PENDING"`` squeue value maps as is. Compact codes are not accepted (we always ask for long names).
    """
    if token is None:
        return None
    text = str(token).strip()
    if not text:
        return None
    first = text.split()[0].upper()
    # sacct may print flags after a '+' (e.g. "RUNNING+" is not seen, but be tolerant of "COMPLETED+").
    first = first.rstrip("+")
    return SLURM_STATE_MAP.get(first)


def is_terminal(state: JobState | str | None) -> bool:
    """True for the terminal set (design section 3.1)."""
    if state is None:
        return False
    try:
        return JobState(str(state)) in TERMINAL
    except ValueError:
        return False


def is_live(state: JobState | str | None) -> bool:
    if state is None:
        return False
    try:
        return JobState(str(state)) in LIVE
    except ValueError:
        return False


_LIMIT_RE = re.compile(r"^(QOS|Assoc|Association)(Grp|Max|Job|Resource|Time|Usage)")
_LIMIT2_RE = re.compile(r"PartitionNodeLimit|PartitionTimeLimit|JobArrayTaskLimit")
_HELD_RE = re.compile(r"JobHeldUser|JobHeldAdmin|JobHoldMaxRequeue")
_DEP_RE = re.compile(r"Dependency|DependencyNeverSatisfied")
_RESV_RE = re.compile(r"ReqNodeNotAvail|Reservation|Reserved for maintenance")
_NORMAL_RE = re.compile(r"^(None|Priority|Resources|BeginTime|Cleaning|WaitingForScheduling|SchedDefer)$")


def classify_reason(reason: object) -> str:
    """Classify a squeue/sacct Reason into ``normal | limit | held | dependency | reservation | unknown``
    (design section 3.1 regexes). Surrounding parentheses (squeue ``%R``) and whitespace are ignored.
    """
    if reason is None:
        return "normal"
    text = str(reason).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if not text:
        return "normal"
    if _LIMIT_RE.search(text) or _LIMIT2_RE.search(text):
        return "limit"
    if _HELD_RE.search(text):
        return "held"
    if _DEP_RE.search(text):
        return "dependency"
    if _RESV_RE.search(text):
        return "reservation"
    if _NORMAL_RE.match(text):
        return "normal"
    return "unknown"


def transition(old: JobState | str | None, new: JobState | str | None) -> bool:
    """Validate a job state move (design section 5.2 "Events are emitted once per transition").

    Returns False for duplicates (``old == new``) and illegal moves; True for a legal move.
    Rules:
    - ``None`` (no state yet) -> anything is legal.
    - terminal -> live is illegal except the requeue path back to SUBMITTED from PREEMPTED/NODE_FAIL
      (SLURM requeued the job, section 5.3) and from TIMEOUT (``on_timeout="requeue"``).
    - terminal -> terminal is legal only as an upgrade (FAILED -> TIMEOUT/OOM per section 5.3;
      LOST -> any terminal when sacct catches up); otherwise illegal.
    - live -> live must not move backwards in the pipeline except the requeue path
      (RUNNING/COMPLETING -> SUBMITTED) and cap-holding (UPLOADING/SUBMITTING -> QUEUED).
    """
    if new is None:
        return False
    try:
        n = JobState(str(new))
        o = JobState(str(old)) if old is not None else None
    except ValueError:
        return False
    if o is None:
        return True
    if o == n:
        return False
    if o in TERMINAL:
        if n in LIVE:
            return n == JobState.SUBMITTED and o in _REQUEUE_FROM
        # terminal -> terminal upgrades
        return (o, n) in _TERMINAL_UPGRADES or (o == JobState.LOST)
    # o live
    if n in TERMINAL:
        return True
    if n == JobState.SUBMITTED and o in (JobState.RUNNING, JobState.COMPLETING):
        return True   # requeue observed as RUNNING -> PENDING
    if n == JobState.QUEUED and o in PRE_SLURM:
        return True   # held locally by a cap after upload/placement (section 5.1 step 5, 5.2 step 10)
    return _LIVE_ORDER[n] > _LIVE_ORDER[o]


_REQUEUE_FROM: frozenset[JobState] = frozenset({JobState.PREEMPTED, JobState.NODE_FAIL, JobState.TIMEOUT})
_TERMINAL_UPGRADES: frozenset[tuple[JobState, JobState]] = frozenset({
    (JobState.FAILED, JobState.TIMEOUT),
    (JobState.FAILED, JobState.OOM),
    (JobState.CANCELLED, JobState.TIMEOUT),
    (JobState.CANCELLED, JobState.OOM),
})
_LIVE_ORDER: dict[JobState, int] = {
    JobState.QUEUED: 0, JobState.UPLOADING: 1, JobState.SUBMITTING: 2, JobState.SUBMITTED: 3,
    JobState.RUNNING: 4, JobState.COMPLETING: 5,
}


__all__ = ["JobState", "TERMINAL", "LIVE", "PRE_SLURM", "SLURM_STATE_MAP", "AttemptState", "TransferState",
           "CmdState", "map_slurm_state", "is_terminal", "is_live", "classify_reason", "transition"]
