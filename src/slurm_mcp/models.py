"""Pydantic models: job specs, targets, policies (design section 3.2) and every tool result (section 4).

Rule 1 of the design: every tool returns a model with ``summary: str`` first, ``unread_events`` and
``next``. Input models forbid unknown keys so a typo is an ``E_INVALID_SPEC`` instead of a silent no-op.
Imports only ``errors``, ``clock`` and ``textio`` from the package.
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from .clock import parse_duration
from .errors import SlurmMcpError
from .slurm.states import AttemptState, CmdState, JobState, TransferState
from .textio import normalize_text

# --- validation grammars (design section 3.2) ----------------------------------------------------

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
ARRAY_RE = re.compile(r"^\d+(-\d+)?(:\d+)?(,\d+(-\d+)?)*$")
DEPENDS_RE = re.compile(r"^(after|afterok|afterany|afternotok|aftercorr)?:?j\d+$|^singleton$")
HANDLE_RE = re.compile(r"^(j\d+(\[\d+\])?|a\d+(\.c\d+)?|t\d+|p\d+)$")

# POSIX signal names (Linux x86-64 numbering) -- the client runs on Windows, whose signal module lacks
# most of these, so the table is static. ``KILL``/``STOP`` cannot be caught and are refused as child_signal.
SIGNAL_NUMBERS: dict[str, int] = {
    "HUP": 1, "INT": 2, "QUIT": 3, "ILL": 4, "TRAP": 5, "ABRT": 6, "BUS": 7, "FPE": 8, "KILL": 9,
    "USR1": 10, "SEGV": 11, "USR2": 12, "PIPE": 13, "ALRM": 14, "TERM": 15, "STKFLT": 16, "CHLD": 17,
    "CONT": 18, "STOP": 19, "TSTP": 20, "TTIN": 21, "TTOU": 22, "URG": 23, "XCPU": 24, "XFSZ": 25,
    "VTALRM": 26, "PROF": 27, "WINCH": 28, "IO": 29, "PWR": 30, "SYS": 31,
}
UNCATCHABLE_SIGNALS: frozenset[str] = frozenset({"KILL", "STOP"})

ON_TIMEOUT_REQUEUE_MSG = (
    "on_timeout=requeue would rerun the job from scratch up to max_restarts times"
)
ON_TIMEOUT_REQUEUE_FIX = (
    "declare child_signal (the signal your program checkpoints on) and checkpoint_interval_h, "
    'or use on_timeout="fail" with a longer time'
)


def signal_number(name: str) -> int | None:
    """Linux signal number for a name without the ``SIG`` prefix (``"USR1"`` -> 10); None if unknown."""
    key = name.upper()
    if key.startswith("SIG"):
        key = key[3:]
    return SIGNAL_NUMBERS.get(key)


def _invalid(message: str, fix: str | None = None) -> SlurmMcpError:
    return SlurmMcpError("E_INVALID_SPEC", message, fix)


class _Input(BaseModel):
    """Base for tool-input models: unknown keys are refused."""

    model_config = ConfigDict(extra="forbid")


# --- section 3.2: specs ----------------------------------------------------------------------------

class Resources(_Input):
    time: str
    gpus: int = 0
    gpu_types: Optional[list[str]] = None
    cpus: Optional[int] = None
    tasks: Optional[int] = None
    mem: Optional[str] = None
    nodes: int = 1
    exclusive: bool = False
    constraint: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "Resources":
        if parse_duration(self.time) is None:
            raise _invalid(f"resources.time {self.time!r} is not a SLURM time (MM, MM:SS, HH:MM:SS, D-HH:MM:SS)",
                           'set resources.time like "02:00:00"')
        if self.gpus < 0 or self.nodes < 1:
            raise _invalid("resources.gpus must be >= 0 and resources.nodes >= 1")
        if self.cpus is not None and self.cpus < 1:
            raise _invalid("resources.cpus must be >= 1")
        if self.tasks is not None and self.tasks < 1:
            raise _invalid("resources.tasks must be >= 1")
        if self.gpu_types is not None and self.gpus == 0:
            raise _invalid("resources.gpu_types given but resources.gpus is 0", "set resources.gpus >= 1")
        return self

    @property
    def time_s(self) -> int:
        return parse_duration(self.time) or 0


class InputSpec(_Input):
    local: str
    remote: Optional[str] = None
    ignore: list[str] = Field(default_factory=list)


class JobSpec(_Input):
    name: str
    command: Optional[str] = None
    script: Optional[str] = None
    script_path: Optional[str] = None
    workdir: Optional[str] = None
    cluster: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    account: Optional[str] = None
    resources: Resources
    inputs: list[InputSpec] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    modules: list[str] = Field(default_factory=list)
    setup: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)
    array: Optional[str] = None
    array_parallel: Optional[int] = None
    depends_on: list[str] = Field(default_factory=list)
    wrap: bool = True
    requeue: Optional[bool] = None
    on_timeout: Literal["fail", "requeue"] = "fail"
    grace_s: int = 120
    child_signal: Optional[str] = None
    max_restarts: int = 3
    checkpoint_interval_h: Optional[float] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    extra_sbatch: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)

    _warnings: list[str] = PrivateAttr(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        """Normalisation warnings produced at validation (``crlf_normalized``)."""
        return list(self._warnings)

    @property
    def source_kind(self) -> Literal["command", "script", "script_path"]:
        if self.command is not None:
            return "command"
        if self.script is not None:
            return "script"
        return "script_path"

    @model_validator(mode="after")
    def _check(self) -> "JobSpec":
        sources = [k for k in ("command", "script", "script_path") if getattr(self, k) is not None]
        if len(sources) != 1:
            raise _invalid(f"exactly one of command/script/script_path is required (got {sources or 'none'})",
                           "set command (bash body), script (full sbatch text) or script_path")
        if not NAME_RE.match(self.name):
            raise _invalid(f"name {self.name!r} must match [A-Za-z0-9_.-]{{1,64}}")
        if self.array is not None and not ARRAY_RE.match(self.array):
            raise _invalid(f"array {self.array!r} must look like 0-99, 0-99:2 or 1,3,5-9")
        if self.array_parallel is not None and self.array_parallel < 1:
            raise _invalid("array_parallel must be >= 1")
        for dep in self.depends_on:
            if not DEPENDS_RE.match(dep):
                raise _invalid(f"depends_on entry {dep!r} must be 'j12', 'afterok:j12', 'afternotok:j12', "
                               "'afterany:j12', 'after:j12', 'aftercorr:j12' or 'singleton'",
                               "use job handles, never raw SLURM ids")
        if self.grace_s < 0:
            raise _invalid("grace_s must be >= 0")
        if self.max_restarts < 0:
            raise _invalid("max_restarts must be >= 0")
        if self.checkpoint_interval_h is not None and self.checkpoint_interval_h <= 0:
            raise _invalid("checkpoint_interval_h must be > 0")
        if self.child_signal is not None:
            sig = self.child_signal.strip().upper()
            if sig.startswith("SIG"):
                sig = sig[3:]
            if sig not in SIGNAL_NUMBERS:
                raise _invalid(f"child_signal {self.child_signal!r} is not a signal name (without SIG)",
                               'use e.g. "USR1" or "TERM"')
            if sig in UNCATCHABLE_SIGNALS:
                raise _invalid(f"child_signal {sig} cannot be caught by the payload", 'use e.g. "USR1" or "TERM"')
            self.child_signal = sig
        if self.on_timeout == "requeue" and (self.child_signal is None or self.checkpoint_interval_h is None):
            raise _invalid(ON_TIMEOUT_REQUEUE_MSG, ON_TIMEOUT_REQUEUE_FIX)
        warnings: list[str] = []
        if self.command is not None:
            self.command, w = normalize_text(self.command)
            warnings += w
        if self.script is not None:
            self.script, w = normalize_text(self.script)
            warnings += w
            if not self.script.lstrip().startswith("#!"):
                raise SlurmMcpError("E_SCRIPT", "script must start with '#!' (e.g. #!/bin/bash)")
        if self.script_path is not None and not (self.script_path.startswith("local:")
                                                 or self.script_path.startswith("/")):
            raise _invalid(f"script_path {self.script_path!r} must be 'local:<path>' or an absolute remote path")
        for key in self.env:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                raise _invalid(f"env key {key!r} is not a valid shell variable name")
        self._warnings = sorted(set(warnings))
        return self

    @classmethod
    def parse(cls, obj: Any) -> "JobSpec":
        """Validate a dict/JobSpec, converting pydantic errors into ``E_INVALID_SPEC``."""
        return parse_input(cls, obj)


def parse_input(model: type[BaseModel], obj: Any) -> Any:
    """``model.model_validate(obj)`` with pydantic ``ValidationError`` rendered as ``E_INVALID_SPEC``."""
    if isinstance(obj, model):
        return obj
    try:
        return model.model_validate(obj)
    except ValidationError as e:
        parts = []
        for item in e.errors():
            loc = ".".join(str(x) for x in item.get("loc", ())) or model.__name__
            parts.append(f"{loc}: {item.get('msg')}")
        raise _invalid("; ".join(parts)[:1000]) from None


class Target(_Input):
    """Placement target, string form ``<cluster>:<partition[,partition]>[:<gres-type>][@<qos>]`` (section 8)."""

    cluster: str
    partitions: list[str]
    gres_type: Optional[str] = None
    qos: Optional[str] = None
    account: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "Target":
        if not self.cluster or ":" in self.cluster or "@" in self.cluster:
            raise _invalid(f"target cluster {self.cluster!r} is invalid")
        if not self.partitions or any((not p) or ":" in p or "@" in p or "," in p for p in self.partitions):
            raise _invalid("target needs at least one partition name")
        return self

    @property
    def key(self) -> str:
        s = f"{self.cluster}:{','.join(self.partitions)}"
        if self.gres_type:
            s += f":{self.gres_type}"
        if self.qos:
            s += f"@{self.qos}"
        return s

    def __str__(self) -> str:
        return self.key

    @classmethod
    def parse(cls, text: str, account: str | None = None) -> "Target":
        """Parse the grammar; raises ``E_INVALID_SPEC`` on malformed input."""
        s = text.strip()
        qos: str | None = None
        if "@" in s:
            s, qos = s.rsplit("@", 1)
            qos = qos.strip() or None
            if qos is None:
                raise _invalid(f"target {text!r}: empty qos after '@'")
        parts = s.split(":")
        if len(parts) < 2 or len(parts) > 3:
            raise _invalid(f"target {text!r} must look like '<cluster>:<partition[,partition]>[:<gres-type>][@<qos>]'")
        cluster = parts[0].strip()
        partitions = [p.strip() for p in parts[1].split(",") if p.strip()]
        gres = parts[2].strip() if len(parts) == 3 else None
        if len(parts) == 3 and not gres:
            raise _invalid(f"target {text!r}: empty gres type after ':'")
        if not cluster or not partitions:
            raise _invalid(f"target {text!r} needs a cluster and at least one partition")
        return cls(cluster=cluster, partitions=partitions, gres_type=gres, qos=qos, account=account)


# --- section 3.2: policies -------------------------------------------------------------------------

class RebalancePolicy(_Input):
    enabled: bool = True
    interval_min: int = 10
    min_gain_h: float = 1.0
    max_moves_per_job: int = 3
    max_extra_su: float = 0.0
    min_age_min: int = 5
    max_moves_per_hour: int = 6
    hysteresis_h: float = 0.5


OBJECTIVE_SU_TO_HOURS: dict[str, float] = {"balanced": 0.25, "fastest": 0.02, "cheapest": 2.0}


class PlacementPolicy(_Input):
    objective: Literal["balanced", "fastest", "cheapest"] = "balanced"
    su_to_hours: Optional[float] = None
    su_reserve: float = 50.0
    max_pending_per_target: Optional[int] = None
    max_running_per_target: dict[str, int] = Field(default_factory=dict)
    allow_self_preempt: bool = False
    soft_caps: dict[str, int] = Field(default_factory=dict)
    etiquette_h: float = 2.0
    targets_allow: list[str] = Field(default_factory=list)
    targets_deny: list[str] = Field(default_factory=list)
    prefer_cluster: Optional[str] = None
    unknown_wait_h: float = 12.0
    rebalance: RebalancePolicy = Field(default_factory=RebalancePolicy)

    @property
    def effective_su_to_hours(self) -> float:
        """``su_to_hours`` or the objective preset (0.25 balanced / 0.02 fastest / 2.0 cheapest, section 8)."""
        if self.su_to_hours is not None:
            return self.su_to_hours
        return OBJECTIVE_SU_TO_HOURS[self.objective]


DEFAULT_TOAST_KINDS: list[str] = [
    "completed", "failed", "timeout", "oom", "cancelled", "preempted", "node_fail", "lost", "needs_attention",
    "alloc_ready", "alloc_expiring", "transfer_failed", "cluster_unreachable",
]


class NotifyPolicy(_Input):
    toast: bool = True
    toast_kinds: list[str] = Field(default_factory=lambda: list(DEFAULT_TOAST_KINDS))
    webhook_url: Optional[str] = None
    webhook_kinds: list[str] = Field(default_factory=list)
    email: Optional[str] = None
    quiet_hours: Optional[tuple[int, int]] = None

    @model_validator(mode="after")
    def _check(self) -> "NotifyPolicy":
        if self.quiet_hours is not None:
            a, b = self.quiet_hours
            if not (0 <= a <= 24 and 0 <= b <= 24):
                raise _invalid("quiet_hours must be two local hours in 0..24")
        return self

    @property
    def effective_webhook_kinds(self) -> list[str]:
        return list(self.webhook_kinds) if self.webhook_kinds else list(self.toast_kinds)


# --- section 4: result models ----------------------------------------------------------------------

class Result(BaseModel):
    """Common base of every tool result (design section 4): ``summary`` first, always."""

    summary: str
    unread_events: int = 0
    next: Optional[str] = None


# clusters -----------------------------------------------------------------------------------------

class TrackedCounts(BaseModel):
    queued: int = 0
    pending: int = 0
    running: int = 0


QuotaRole = Literal["home", "remote_root", "control_root", "project", "group", "upload_root"]


class QuotaRow(BaseModel):
    path: str
    used_pct: Optional[float] = None
    free_gb: Optional[float] = None
    role: Optional[QuotaRole] = None


class ClusterRow(BaseModel):
    name: str
    host: str
    transfer_host: Optional[str] = None
    connected: bool = False
    auth_failed: bool = False
    reachable: Optional[bool] = None
    last_tick_age_s: Optional[float] = None
    tracked_jobs: TrackedCounts = Field(default_factory=TrackedCounts)
    su_balance: Optional[float] = None
    quota: list[QuotaRow] = Field(default_factory=list)
    monitor: str = "none"          # "self" | "held by pid N" | "lost to pid N" | "none"
    warnings: list[str] = Field(default_factory=list)


class ClustersResult(Result):
    clusters: list[ClusterRow] = Field(default_factory=list)
    session_id: Optional[str] = None


class NodeCounts(BaseModel):
    idle: int = 0
    mix: int = 0
    alloc: int = 0
    other: int = 0
    total: int = 0


class GresCount(BaseModel):
    gres: str
    count: int = 0
    mine: int = 0


class MyJobs(BaseModel):
    pending: int = 0
    running: int = 0


class PartitionLimits(BaseModel):
    max_wall_s: Optional[int] = None
    max_jobs_pu: Optional[int] = None
    max_submit_pu: Optional[int] = None
    max_tres_pj: dict[str, float] = Field(default_factory=dict)


class Charge(BaseModel):
    unit: str
    su_per_unit_h: float


class PartitionInfo(BaseModel):
    name: str
    avail: Optional[str] = None
    preempt_mode: Optional[str] = None
    priority_tier: Optional[int] = None
    grace_time_s: Optional[int] = None
    max_wall_s: Optional[int] = None
    default_time_s: Optional[int] = None
    nodes: NodeCounts = Field(default_factory=NodeCounts)
    gres_types: list[str] = Field(default_factory=list)
    pending_by_gres: list[GresCount] = Field(default_factory=list)
    running_by_gres: list[GresCount] = Field(default_factory=list)
    my_jobs: MyJobs = Field(default_factory=MyJobs)
    limits: PartitionLimits = Field(default_factory=PartitionLimits)
    qos: Optional[str] = None
    charge: Charge | Literal["free"] = "free"


class ReservationRow(BaseModel):
    name: str
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None
    partitions: list[str] = Field(default_factory=list)
    maint: bool = False


class QueueRow(BaseModel):
    slurm_id: str
    name: Optional[str] = None
    partition: Optional[str] = None
    state: Optional[str] = None
    reason: Optional[str] = None
    est_start_ts: Optional[int] = None
    handle: Optional[str] = None


class TargetRow(BaseModel):
    target: str
    enabled: bool = True
    max_pending: Optional[int] = None
    max_running: Optional[int] = None


class ClusterStatusResult(Result):
    cluster: str
    partitions: list[PartitionInfo] = Field(default_factory=list)
    su_balance: Optional[float] = None
    quota: list[QuotaRow] = Field(default_factory=list)
    reservations_upcoming: list[ReservationRow] = Field(default_factory=list)
    slurm_version: Optional[str] = None
    helper_version: Optional[str] = None
    caps_age_s: Optional[float] = None
    queue: Optional[list[QueueRow]] = None
    targets: Optional[list[TargetRow]] = None
    config: Optional[dict[str, Any]] = None


class RunCommandResult(Result):
    """Result of ``run_command`` (named so it does not clash with transport.CommandResult)."""

    rc: Optional[int] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    truncated: bool = False
    seconds: float = 0.0


# files --------------------------------------------------------------------------------------------

class Renamed(BaseModel):
    remote: str
    local: str


class TransferResult(Result):
    transfer_id: Optional[str] = None
    state: TransferState = TransferState.planned
    files_sent: int = 0
    files_skipped: int = 0
    bytes: int = 0
    seconds: float = 0.0
    mode: Optional[str] = None          # "tar" | "sftp"
    host_role: Optional[str] = None     # "login" | "transfer"
    quota_after_pct: Optional[float] = None
    remote: Optional[str] = None
    local_dir: Optional[str] = None
    renamed: list[Renamed] = Field(default_factory=list)
    skipped_in_progress: list[str] = Field(default_factory=list)
    would_send: list[str] = Field(default_factory=list)   # dry_run listing (<= 100)
    error: Optional[str] = None


class ListingEntry(BaseModel):
    name: str
    type: str                             # "file" | "dir" | "link" | "other"
    size: Optional[int] = None
    mtime_ts: Optional[int] = None


class ListingResult(Result):
    path: str
    entries: list[ListingEntry] = Field(default_factory=list)
    truncated: bool = False


class ReadResult(Result):
    path: str
    text: str = ""
    size: Optional[int] = None
    next_offset: Optional[int] = None
    truncated: bool = False


class WriteResult(Result):
    path: str
    bytes: int = 0


# jobs ---------------------------------------------------------------------------------------------

EstWaitSrc = Literal["test_only", "history", "depth", "none"]


class PlanOption(BaseModel):
    target: str
    feasible: bool = True
    est_wait_h: Optional[float] = None
    est_wait_src: EstWaitSrc = "none"
    est_start_ts: Optional[int] = None
    queue_ahead: Optional[int] = None
    queue_ahead_untyped: Optional[int] = None
    cost_su: Optional[float] = None
    cost_worst_su: Optional[float] = None
    requeueable: Optional[bool] = None
    charge: Charge | Literal["free"] = "free"
    risk_pct: Optional[float] = None
    etiquette_h: float = 0.0
    score_h: Optional[float] = None
    why: str = ""


class PlanResult(Result):
    plan_id: str
    options: list[PlanOption] = Field(default_factory=list)
    recommended: Optional[str] = None
    rendered_preview: str = ""
    warnings: list[str] = Field(default_factory=list)
    stripped_directives: list[str] = Field(default_factory=list)
    expires_ts: Optional[int] = None


class Uploads(BaseModel):
    transfer_ids: list[str] = Field(default_factory=list)
    files_sent: int = 0
    bytes: int = 0


class SubmitResult(Result):
    handle: str
    kind: Literal["job", "alloc"] = "job"
    cluster: Optional[str] = None
    slurm_id: Optional[str] = None
    attempt_no: int = 1
    target: Optional[str] = None
    state: JobState = JobState.QUEUED
    est_start_ts: Optional[int] = None
    cost_est_su: Optional[float] = None
    cost_worst_su: Optional[float] = None
    submit_line: Optional[str] = None
    workdir: Optional[str] = None
    ctrl_dir: Optional[str] = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    injected: list[str] = Field(default_factory=list)
    stripped_directives: list[str] = Field(default_factory=list)
    dependencies_resolved: list[str] = Field(default_factory=list)
    uploads: Uploads = Field(default_factory=Uploads)
    array_size: Optional[int] = None
    warnings: list[str] = Field(default_factory=list)


class JobRow(BaseModel):
    handle: Optional[str] = None          # null for untracked squeue rows
    kind: Literal["job", "alloc"] = "job"
    name: Optional[str] = None
    cluster: Optional[str] = None
    slurm_id: Optional[str] = None
    state: Optional[JobState] = None
    reason: Optional[str] = None
    target: Optional[str] = None
    elapsed_s: Optional[int] = None
    time_limit_s: Optional[int] = None
    restarts: int = 0
    moves: int = 0
    est_start_ts: Optional[int] = None


class JobListResult(Result):
    jobs: list[JobRow] = Field(default_factory=list)
    counts_by_state: dict[str, int] = Field(default_factory=dict)
    truncated: bool = False


class ExitInfo(BaseModel):
    rc: Optional[int] = None
    signal: Optional[int] = None


class JobPaths(BaseModel):
    cluster: Optional[str] = None
    workdir: Optional[str] = None
    ctrl_dir: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class DependencyRow(BaseModel):
    handle: str
    type: str
    status: Optional[str] = None


class AllocInfo(BaseModel):
    ready: bool = False
    end_ts: Optional[int] = None
    cmds_outstanding: int = 0


class TransferInfo(BaseModel):
    state: TransferState = TransferState.planned
    files_done: int = 0
    files_total: Optional[int] = None
    bytes_done: int = 0
    bytes_total: Optional[int] = None
    error: Optional[str] = None


class CmdInfo(BaseModel):
    state: CmdState = CmdState.queued
    rc: Optional[int] = None
    started_ts: Optional[int] = None
    done_ts: Optional[int] = None


class AttemptRow(BaseModel):
    attempt_no: int
    state: AttemptState
    cluster: Optional[str] = None
    slurm_id: Optional[str] = None
    target: Optional[str] = None
    cause: Optional[str] = None
    submit_ts: Optional[int] = None
    end_ts: Optional[int] = None
    workdir: Optional[str] = None


class JobDetail(JobRow):
    submit_ts: Optional[int] = None
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None
    exit: ExitInfo = Field(default_factory=ExitInfo)
    node: Optional[str] = None
    progress: Optional[Any] = None
    heartbeat_age_s: Optional[float] = None
    last_log_line: Optional[str] = None
    cost_su: Optional[float] = None
    cost_worst_su: Optional[float] = None
    attempts_count: int = 0
    paths: JobPaths = Field(default_factory=JobPaths)
    dependencies: list[DependencyRow] = Field(default_factory=list)
    dependents: list[str] = Field(default_factory=list)
    alloc: Optional[AllocInfo] = None
    transfer: Optional[TransferInfo] = None
    cmd: Optional[CmdInfo] = None
    next_action: Optional[str] = None
    attempts: Optional[list[AttemptRow]] = None       # detail="full"
    raw: Optional[dict[str, Any]] = None              # detail="full": raw squeue/sacct fields
    efficiency: Optional[dict[str, Any]] = None       # detail="full": seff-style


class JobStatusResult(Result):
    jobs: list[JobDetail] = Field(default_factory=list)


class LogStream(BaseModel):
    text: str = ""
    size: Optional[int] = None
    next_offset: Optional[int] = None
    path: Optional[str] = None
    truncated: bool = False


class LogResult(Result):
    id: str
    state: Optional[str] = None
    out: Optional[LogStream] = None
    err: Optional[LogStream] = None


class ControlOutcome(BaseModel):
    id: str
    accepted: bool = False
    outcome: Optional[str] = None
    message: Optional[str] = None
    hard_kill_ts: Optional[int] = None


class ControlResult(Result):
    action: str
    results: list[ControlOutcome] = Field(default_factory=list)


class RebalanceProposal(BaseModel):
    handle: str
    from_target: Optional[str] = None
    to_target: Optional[str] = None
    est_wait_now_h: Optional[float] = None
    est_wait_new_h: Optional[float] = None
    gain_h: Optional[float] = None
    cost_delta_su: Optional[float] = None
    will_move: bool = False
    why: str = ""


class SkippedRow(BaseModel):
    handle: str
    why: str = ""


class RebalanceResult(Result):
    dry_run: bool = True
    proposals: list[RebalanceProposal] = Field(default_factory=list)
    skipped: list[SkippedRow] = Field(default_factory=list)
    moved: list[str] = Field(default_factory=list)
    moving: list[str] = Field(default_factory=list)


class AllocRunResult(Result):
    cmd_id: str
    alloc_id: Optional[str] = None
    state: CmdState = CmdState.queued
    rc: Optional[int] = None
    out_tail: str = ""
    started_ts: Optional[int] = None
    seconds: Optional[float] = None
    out_path: Optional[str] = None


# events -------------------------------------------------------------------------------------------

class EventRow(BaseModel):
    seq: int
    ts: Optional[int] = None
    kind: str
    handle: Optional[str] = None
    cluster: Optional[str] = None
    slurm_id: Optional[str] = None
    summary: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class EventSnapshot(BaseModel):
    queued: int = 0
    pending: int = 0
    running: int = 0
    alloc_ready: int = 0
    transfers_running: int = 0
    submits_running: int = 0


class EventsResult(Result):
    events: list[EventRow] = Field(default_factory=list)
    delivered_seqs: list[int] = Field(default_factory=list)
    next_seq: Optional[int] = None
    acked: int = 0
    unread_unmatched: int = 0
    timed_out: bool = False
    snapshot: EventSnapshot = Field(default_factory=EventSnapshot)
    client_id: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


# results ------------------------------------------------------------------------------------------

class CollectRow(BaseModel):
    handle: str
    state: Optional[JobState] = None
    exit_code: Optional[int] = None
    transfer_id: Optional[str] = None
    files: int = 0
    bytes: int = 0
    skipped: int = 0
    local_path: Optional[str] = None
    error: Optional[str] = None


class CollectResult(Result):
    jobs: list[CollectRow] = Field(default_factory=list)
    local_dir: Optional[str] = None


# configuration ------------------------------------------------------------------------------------

class ConfigResult(Result):
    placement: PlacementPolicy = Field(default_factory=PlacementPolicy)
    notify: NotifyPolicy = Field(default_factory=NotifyPolicy)


RESULT_MODELS: tuple[type[Result], ...] = (
    ClustersResult, ClusterStatusResult, RunCommandResult, TransferResult, ListingResult, ReadResult,
    WriteResult, PlanResult, SubmitResult, JobListResult, JobStatusResult, LogResult, ControlResult,
    RebalanceResult, AllocRunResult, EventsResult, CollectResult, ConfigResult,
)
INPUT_MODELS: tuple[type[BaseModel], ...] = (
    Resources, InputSpec, JobSpec, Target, PlacementPolicy, RebalancePolicy, NotifyPolicy,
)

__all__ = [
    "NAME_RE", "ARRAY_RE", "DEPENDS_RE", "HANDLE_RE", "SIGNAL_NUMBERS", "UNCATCHABLE_SIGNALS",
    "ON_TIMEOUT_REQUEUE_MSG", "ON_TIMEOUT_REQUEUE_FIX", "OBJECTIVE_SU_TO_HOURS", "DEFAULT_TOAST_KINDS",
    "signal_number", "parse_input",
    "Resources", "InputSpec", "JobSpec", "Target", "PlacementPolicy", "RebalancePolicy", "NotifyPolicy",
    "Result", "TrackedCounts", "QuotaRow", "ClusterRow", "ClustersResult", "NodeCounts", "GresCount", "MyJobs",
    "PartitionLimits", "Charge", "PartitionInfo", "ReservationRow", "QueueRow", "TargetRow",
    "ClusterStatusResult", "RunCommandResult", "Renamed", "TransferResult", "ListingEntry", "ListingResult",
    "ReadResult", "WriteResult", "PlanOption", "PlanResult", "Uploads", "SubmitResult", "JobRow",
    "JobListResult", "ExitInfo", "JobPaths", "DependencyRow", "AllocInfo", "TransferInfo", "CmdInfo",
    "AttemptRow", "JobDetail", "JobStatusResult", "LogStream", "LogResult", "ControlOutcome", "ControlResult",
    "RebalanceProposal", "SkippedRow", "RebalanceResult", "AllocRunResult", "EventRow", "EventSnapshot",
    "EventsResult", "CollectRow", "CollectResult", "ConfigResult", "RESULT_MODELS", "INPUT_MODELS",
]
