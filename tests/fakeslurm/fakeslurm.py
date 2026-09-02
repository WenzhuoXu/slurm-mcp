#!/usr/bin/env python3
"""fakeslurm - a single-file, dependency-free emulator of the SLURM 22.05 command line.

Dispatched by argv[1] (sbatch|squeue|sacct|sinfo|scontrol|scancel|sprio|sshare|sacctmgr|seff|fakeslurm-ctl).
All state lives in a JSON file named by $FAKESLURM_STATE.  A simulated clock lives in the state and only
moves through `fakeslurm-ctl advance` (or the FAKESLURM_NOW override); every command invocation first runs a
scheduler tick so the queue evolves deterministically.

Output formats were derived from real captures in tests/fixtures/{trace,bridges2} (SLURM 22.05.11).
"""
from __future__ import annotations

import calendar
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, fields

SLURM_VERSION = "22.05.11"
HERE = os.path.dirname(os.path.abspath(__file__))
CLUSTERS_DIR = os.path.join(HERE, "clusters")


# --------------------------------------------------------------------------------------------------
# path handling: the fake is driven from Git Bash on Windows, so job paths (WorkDir, StdOut, Command,
# SLURM_SUBMIT_DIR) are kept as POSIX strings exactly as the shell sees them (/c/Users/..., /tmp/...)
# and are converted to native Windows paths only when a file is actually opened.
# --------------------------------------------------------------------------------------------------
def _git_root() -> str | None:
    bash = os.environ.get("FAKESLURM_BASH")
    if not bash:
        return None
    d = os.path.dirname(os.path.abspath(bash))          # .../Git/bin or .../Git/usr/bin
    for tail in (os.sep + "usr" + os.sep + "bin", os.sep + "bin"):
        if d.lower().endswith(tail):
            return d[: -len(tail)]
    return None


def posix_to_native(path: str) -> str:
    """'/c/Users/x/y' -> 'C:/Users/x/y', '/tmp/x' -> '<TEMP>/x' (Git Bash mounts), '/usr/bin' -> '<Git>/usr/bin'.
    Windows paths and POSIX hosts pass through unchanged."""
    if os.name != "nt" or not path.startswith("/"):
        return path
    m = re.match(r"^/([A-Za-z])(?:/(.*))?$", path)
    if m:
        return m.group(1).upper() + ":/" + (m.group(2) or "")
    if path == "/tmp" or path.startswith("/tmp/"):
        import tempfile
        return tempfile.gettempdir().replace("\\", "/") + path[4:]
    root = _git_root()
    if root:
        return root.replace("\\", "/") + path
    return path


def native_to_posix(path: str) -> str:
    """'C:\\Users\\x' -> '/c/Users/x' (Git Bash spelling). Already-POSIX paths pass through."""
    if os.name != "nt" or not path:
        return path
    p = path.replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = "/" + p[0].lower() + p[2:]
    return p


def is_posix_path(path: str) -> bool:
    return path.startswith("/")


def path_join(base: str, rel: str) -> str:
    """Join keeping the flavour of `base` (POSIX bases stay POSIX)."""
    if is_posix_path(rel) or (len(rel) > 1 and rel[1] == ":"):
        return rel
    if is_posix_path(base):
        import posixpath
        return posixpath.normpath(posixpath.join(base, rel))
    return os.path.normpath(os.path.join(base, rel))


def current_dir() -> str:
    """The directory sbatch was invoked from. When run from the bash shims $PWD carries the shell's
    spelling of the cwd: /c/Users/... or /tmp/..., or C:/Users/... once the MSYS runtime has converted
    the variable for the native python. Use it (in POSIX spelling) if it really names os.getcwd()."""
    cwd = os.getcwd()
    pwd = os.environ.get("PWD", "")
    if pwd:
        try:
            if os.path.samefile(posix_to_native(pwd), cwd):
                return pwd if pwd.startswith("/") else native_to_posix(pwd)
        except (OSError, ValueError):
            pass
    return cwd


def shell_path(path: str, submit_dir: str) -> str:
    """A path argument as the invoking shell would spell it: when sbatch runs from bash (POSIX submit
    dir), drive-letter paths produced by MSYS argument conversion ('C:/x') become '/c/x' again."""
    return native_to_posix(path) if path and is_posix_path(submit_dir) else path

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED",
                   "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY"}
STATE_SHORT = {"PENDING": "PD", "RUNNING": "R", "SUSPENDED": "S", "COMPLETED": "CD", "FAILED": "F",
               "CANCELLED": "CA", "TIMEOUT": "TO", "NODE_FAIL": "NF", "PREEMPTED": "PR",
               "BOOT_FAIL": "BF", "DEADLINE": "DL", "OUT_OF_MEMORY": "OOM", "COMPLETING": "CG",
               "CONFIGURING": "CF", "REQUEUED": "RQ", "REQUEUE_HOLD": "RH", "SPECIAL_EXIT": "SE",
               "STOPPED": "ST", "RESV_DEL_HOLD": "RD", "SIGNALING": "SI", "STAGE_OUT": "SO"}
STATE_LONG = {v: k for k, v in STATE_SHORT.items()}
STATE_ORDER = {"PENDING": 0, "RUNNING": 1, "SUSPENDED": 2, "COMPLETED": 3, "CANCELLED": 4, "FAILED": 5,
               "TIMEOUT": 6, "NODE_FAIL": 7, "PREEMPTED": 8, "BOOT_FAIL": 9, "DEADLINE": 10,
               "OUT_OF_MEMORY": 11}


class CommandError(Exception):
    """Raised to terminate a command with a message on stderr and a return code."""

    def __init__(self, msg: str, rc: int = 1):
        super().__init__(msg)
        self.msg = msg
        self.rc = rc


class Ctx:
    """Per-invocation output buffers."""

    def __init__(self, prog: str, stdin_text: str = ""):
        self.prog = prog
        self.out = io.StringIO()
        self.err = io.StringIO()
        self.stdin_text = stdin_text
        self.rc = 0

    def p(self, s: str = "") -> None:
        self.out.write(s + "\n")

    def e(self, s: str) -> None:
        self.err.write(s + "\n")

    def error(self, s: str) -> None:
        self.err.write(f"{self.prog}: error: {s}\n")


# --------------------------------------------------------------------------------------------------
# hostlist helpers
# --------------------------------------------------------------------------------------------------
_HOST_RE = re.compile(r"^(.*?)(\d+)$")


def hostlist_expand(expr: str) -> list[str]:
    """Expand 'trace[01-03,07],r001' into individual host names."""
    out: list[str] = []
    if not expr:
        return out
    i = 0
    tokens: list[str] = []
    depth = 0
    cur = ""
    while i < len(expr):
        ch = expr[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            tokens.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    if cur:
        tokens.append(cur)
    for tok in tokens:
        m = re.match(r"^([^\[]*)\[([^\]]*)\](.*)$", tok)
        if not m:
            out.append(tok)
            continue
        prefix, ranges, suffix = m.groups()
        for r in ranges.split(","):
            if "-" in r:
                a, b = r.split("-", 1)
                width = len(a)
                for n in range(int(a), int(b) + 1):
                    out.append(f"{prefix}{n:0{width}d}{suffix}")
            else:
                out.append(f"{prefix}{r}{suffix}")
    return out


def hostlist_compress(names: list[str]) -> str:
    """Compress ['trace01','trace02','trace07'] into 'trace[01-02,07]' like SLURM does."""
    if not names:
        return ""
    groups: dict[tuple[str, int], list[int]] = {}
    plain: list[str] = []
    order: list[tuple[str, int]] = []
    for n in names:
        m = _HOST_RE.match(n)
        if not m:
            plain.append(n)
            continue
        key = (m.group(1), len(m.group(2)))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(int(m.group(2)))
    parts: list[str] = []
    for key in sorted(order):
        prefix, width = key
        nums = sorted(set(groups[key]))
        if len(nums) == 1:
            parts.append(f"{prefix}{nums[0]:0{width}d}")
            continue
        ranges: list[str] = []
        start = prev = nums[0]
        for n in nums[1:] + [None]:
            if n is not None and n == prev + 1:
                prev = n
                continue
            if start == prev:
                ranges.append(f"{start:0{width}d}")
            else:
                ranges.append(f"{start:0{width}d}-{prev:0{width}d}")
            if n is not None:
                start = prev = n
        parts.append(f"{prefix}[{','.join(ranges)}]")
    return ",".join(parts + sorted(plain))


# --------------------------------------------------------------------------------------------------
# time helpers
# --------------------------------------------------------------------------------------------------
def fmt_ts(epoch: int | None) -> str:
    if epoch is None:
        return "Unknown"
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(int(epoch)))


def parse_iso(s: str) -> int:
    """Parse YYYY-MM-DD[THH:MM[:SS]] as local time."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return int(time.mktime(time.strptime(s, fmt)))
        except ValueError:
            continue
    raise ValueError(s)


def parse_slurm_time(s: str, now: int) -> int:
    """Subset of slurm's parse_time(): ISO, HH:MM[:SS], now[+-N(units)], today, midnight, noon.
    A bare epoch number is rejected like real sacct does."""
    s = s.strip()
    if re.fullmatch(r"\d+", s):
        raise ValueError(s)
    if s.startswith("now"):
        rest = s[3:]
        if not rest:
            return now
        m = re.fullmatch(r"([+-])(\d+)(seconds?|minutes?|hours?|days?|weeks?)?", rest)
        if not m:
            raise ValueError(s)
        sign, n, unit = m.groups()
        mult = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800,
                None: 60}[(unit or "").rstrip("s") or None]
        delta = int(n) * mult
        return now + delta if sign == "+" else now - delta
    lt = time.localtime(now)
    if s in ("today", "midnight"):
        return int(time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1)))
    if s == "noon":
        return int(time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 12, 0, 0, 0, 0, -1)))
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        return int(time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, int(m.group(1)), int(m.group(2)),
                                int(m.group(3) or 0), 0, 0, -1)))
    return parse_iso(s)


def parse_time_limit(s: str) -> int | None:
    """sbatch -t: minutes | mm:ss | hh:mm:ss | d-hh | d-hh:mm | d-hh:mm:ss -> minutes (None=UNLIMITED)."""
    s = s.strip()
    if s.upper() in ("UNLIMITED", "INFINITE", "-1"):
        return None
    if re.fullmatch(r"\d+", s):
        return int(s)
    m = re.fullmatch(r"(\d+)-(\d+)(?::(\d+))?(?::(\d+))?", s)
    if m:
        d, h, mi, se = (int(x) if x is not None else 0 for x in m.groups())
        secs = d * 86400 + h * 3600 + mi * 60 + se
        return -(-secs // 60)
    parts = s.split(":")
    if not all(re.fullmatch(r"\d+", p) for p in parts):
        raise ValueError(s)
    if len(parts) == 2:
        secs = int(parts[0]) * 60 + int(parts[1])
        return -(-secs // 60)
    if len(parts) == 3:
        secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return -(-secs // 60)
    raise ValueError(s)


def fmt_compact(secs: int) -> str:
    """squeue/sinfo style: 0:00, 14:47, 6:00:00, 1-00:00:00."""
    secs = max(0, int(secs))
    d, rem = divmod(secs, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def fmt_hms(secs: int) -> str:
    """sacct/scontrol style: 00:15:20, 1-00:00:00."""
    secs = max(0, int(secs))
    d, rem = divmod(secs, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_limit_hms(minutes: int | None) -> str:
    return "UNLIMITED" if minutes is None else fmt_hms(minutes * 60)


def fmt_limit_compact(minutes: int | None, unlimited: str = "UNLIMITED") -> str:
    return unlimited if minutes is None else fmt_compact(minutes * 60)


def fmt_cpu_time(secs: float) -> str:
    """sacct TotalCPU: 00:00:00 when zero, MM:SS.mmm under an hour, else [D-]HH:MM:SS."""
    if secs <= 0:
        return "00:00:00"
    if secs < 3600:
        m, s = divmod(secs, 60)
        return f"{int(m):02d}:{int(s):02d}.{int(round((s - int(s)) * 1000)):03d}"
    return fmt_hms(int(secs))


# --------------------------------------------------------------------------------------------------
# memory / gres helpers
# --------------------------------------------------------------------------------------------------
def parse_mem(s: str) -> int:
    m = re.fullmatch(r"(\d+)([KkMmGgTt]?)[Bb]?", s.strip())
    if not m:
        raise ValueError(s)
    n = int(m.group(1))
    unit = m.group(2).upper()
    return {"": n, "K": -(-n // 1024), "M": n, "G": n * 1024, "T": n * 1024 * 1024}[unit]


def fmt_mem(mb: int) -> str:
    if mb and mb % (1024 * 1024) == 0:
        return f"{mb // (1024 * 1024)}T"
    if mb and mb % 1024 == 0:
        return f"{mb // 1024}G"
    return f"{mb}M"


def parse_gres(spec: str) -> list[dict]:
    """'gpu:a40:1,gpu:2' -> [{'name':'gpu','type':'a40','count':1}, ...]."""
    out: list[dict] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        name = parts[0]
        typ = ""
        count = 1
        if len(parts) == 2:
            if re.fullmatch(r"\d+", parts[1]):
                count = int(parts[1])
            else:
                typ = parts[1]
        elif len(parts) >= 3:
            typ = parts[1]
            if not re.fullmatch(r"\d+", parts[2]):
                raise ValueError(spec)
            count = int(parts[2])
        out.append({"name": name, "type": typ, "count": count})
    return out


def gres_string(gres: list[dict], with_count_if_one: bool = True) -> str:
    items = []
    for g in gres:
        s = "gres:" + g["name"]
        if g["type"]:
            s += ":" + g["type"]
        if with_count_if_one or g["count"] != 1:
            s += f":{g['count']}"
        items.append(s)
    return ",".join(items)


def node_gres_map(gres_str: str) -> dict[tuple[str, str], int]:
    """'gpu:a40:1' -> {('gpu','a40'): 1}."""
    out: dict[tuple[str, str], int] = {}
    if not gres_str:
        return out
    for g in parse_gres(gres_str):
        out[(g["name"], g["type"])] = out.get((g["name"], g["type"]), 0) + g["count"]
    return out


def parse_tres(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in (s or "").split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = parse_mem(v)
                except ValueError:
                    pass
    return out


# --------------------------------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------------------------------
@dataclass
class Node:
    name: str
    cpus: int
    real_memory: int
    sockets: int = 2
    cores_per_socket: int = 1
    threads_per_core: int = 1
    gres: str = ""
    features: str = ""
    state: str = "idle"        # idle | down | drain (base state for admin-set states)
    reason: str = ""
    weight: int = 1


@dataclass
class Partition:
    name: str
    nodes: str
    default: bool = False
    allow_groups: str = "ALL"
    allow_accounts: str = "ALL"
    allow_qos: str = "ALL"
    qos: str = "N/A"
    default_time: str = "01:00:00"
    max_time: str = "UNLIMITED"
    max_nodes: object = "UNLIMITED"
    min_nodes: int = 0
    priority_job_factor: int = 1
    priority_tier: int = 1
    oversubscribe: str = "NO"
    preempt_mode: str = "OFF"
    state: str = "UP"
    def_mem_per_cpu: object = None
    def_mem_per_node: object = None
    def_mem_per_gpu: object = None
    max_mem_per_node: object = "UNLIMITED"

    def node_names(self) -> list[str]:
        return hostlist_expand(self.nodes)

    def max_time_minutes(self) -> int | None:
        return parse_time_limit(str(self.max_time))

    def default_time_minutes(self) -> int | None:
        return parse_time_limit(str(self.default_time))

    def max_nodes_int(self) -> int | None:
        return None if str(self.max_nodes).upper() == "UNLIMITED" else int(self.max_nodes)


@dataclass
class Job:
    id: int
    name: str
    user: str
    uid: int
    account: str
    qos: str
    partitions: list = field(default_factory=list)
    partition: str = ""
    state: str = "PENDING"
    reason: str = "None"
    submit: int = 0
    eligible: int = 0
    start: int | None = None
    end: int | None = None
    time_limit: int | None = 60          # minutes; None = UNLIMITED
    time_min: int | None = None
    num_nodes: int = 1
    ntasks: int = 1
    ntasks_per_node: int = 0
    cpus_per_task: int = 1
    cpus_per_node: int = 1
    mem_mb: int = 0                       # per node
    mem_per_cpu: int | None = None
    mem_per_gpu: int | None = None
    gres: list = field(default_factory=list)   # per node
    gres_per_job: bool = False
    exclusive: bool = False
    constraint: str = ""
    req_nodes: list = field(default_factory=list)
    exc_nodes: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    sched_nodes: list = field(default_factory=list)
    priority: int = 0
    nice: int = 0
    held: bool = False
    hold_reason: str = ""
    begin: int | None = None
    dependency: str = ""
    kill_on_invalid_dep: bool = False
    requeue: bool = False
    restarts: int = 0
    comment: str = ""
    script: str = ""
    command: str = ""
    workdir: str = ""
    stdout: str = ""
    stderr: str = ""
    open_mode: str = "truncate"
    signal: str = ""
    mail_type: str = ""
    mail_user: str = ""
    export: str = "ALL"
    env: dict = field(default_factory=dict)
    submit_line: str = ""
    array_job_id: int | None = None
    array_task_id: int | None = None
    array_task_throttle: int = 0
    array_spec: str = ""
    duration: int | None = None          # planned run time in seconds
    planned_exit: int = 0
    exit_code: int = 0
    exit_signal: int = 0
    batch_state: str = ""
    batch_exit: int = 0
    batch_signal: int = 0
    max_rss_k: int | None = None
    marker_max_rss_k: int | None = None   # from '#FAKESLURM maxrss=' (kept across requeues)
    last_pending_reason: str = "None"     # what sacct's Reason column reports (see _note_pending_reason)
    total_cpu: float = 0.0
    scheduler: str = "Main"
    flags: list = field(default_factory=lambda: ["StartRecieved"])
    preempt_time: int | None = None
    last_sched_eval: int | None = None
    est_start: int | None = None
    signals: list = field(default_factory=list)
    incarnations: list = field(default_factory=list)
    purged: bool = False
    cancelled_by: int | None = None
    alloc_sid: int = 0

    # ---- derived ---------------------------------------------------------------------------
    def is_pending(self) -> bool:
        return self.state == "PENDING"

    def is_running(self) -> bool:
        return self.state == "RUNNING"

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def limit_secs(self) -> int | None:
        return None if self.time_limit is None else self.time_limit * 60

    def elapsed(self, now: int) -> int:
        if self.start is None:
            return 0
        if self.end is not None:
            return max(0, self.end - self.start)
        return max(0, now - self.start)

    def total_cpus(self) -> int:
        return self.cpus_per_node * self.num_nodes

    def gpu_count(self) -> int:
        return sum(g["count"] for g in self.gres if g["name"] == "gpu") * (1 if self.gres_per_job else self.num_nodes)

    def exit_str(self) -> str:
        return f"{self.exit_code}:{self.exit_signal}"

    def id_str(self) -> str:
        if self.array_job_id is not None and self.array_task_id is not None:
            return f"{self.array_job_id}_{self.array_task_id}"
        return str(self.id)

    def state_str_sacct(self) -> str:
        if self.state == "CANCELLED" and self.cancelled_by is not None:
            return f"CANCELLED by {self.cancelled_by}"
        return self.state


def job_from_dict(d: dict) -> Job:
    names = {f.name for f in fields(Job)}
    return Job(**{k: v for k, v in d.items() if k in names})


# --------------------------------------------------------------------------------------------------
# state persistence
# --------------------------------------------------------------------------------------------------
class StateLock:
    def __init__(self, path: str):
        self.lock_path = path + ".lock"
        self.fd = None

    def __enter__(self):
        deadline = time.time() + 30
        while True:
            try:
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() > deadline:
                    try:
                        os.unlink(self.lock_path)  # stale lock
                    except OSError:
                        pass
                time.sleep(0.02)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        try:
            os.unlink(self.lock_path)
        except OSError:
            pass


def state_path() -> str:
    p = os.environ.get("FAKESLURM_STATE")
    if not p:
        raise CommandError("fakeslurm: FAKESLURM_STATE environment variable is not set", 2)
    return p


def load_cluster_def(name: str) -> dict:
    path = name if name.endswith(".json") else os.path.join(CLUSTERS_DIR, name + ".json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def new_state(cluster: str, now: int, start_jobid: int) -> dict:
    cdef = load_cluster_def(cluster)
    state = {
        "cluster": cdef["cluster"],
        "config": dict(cdef["config"]),
        "fake": dict(cdef.get("fake", {})),
        "user": cdef["user"],
        "qos": cdef["qos"],
        "partitions": [asdict(Partition(**p)) for p in cdef["partitions"]],
        "nodes": [asdict(Node(**n)) for n in cdef["nodes"]],
        "now": int(now),
        "init_now": int(now),
        "wall_anchor": int(time.time()),
        "clock_mode": "manual",
        "next_jobid": int(start_jobid),
        "jobs": {},
        "reservations": [],
        "seq": 0,
        "events": [],
    }
    return state


class Sim:
    """The cluster simulation over a loaded state dict."""

    def __init__(self, state: dict):
        self.state = state
        self.jobs: dict[int, Job] = {int(k): job_from_dict(v) for k, v in state["jobs"].items()}
        self.nodes: dict[str, Node] = {n["name"]: Node(**n) for n in state["nodes"]}
        self.partitions: list[Partition] = [Partition(**p) for p in state["partitions"]]
        self.config = state["config"]
        self.fake = state.get("fake", {})
        self.user = state["user"]
        self.qos_table = state["qos"]
        env_now = os.environ.get("FAKESLURM_NOW")
        if env_now:
            state["now"] = parse_iso(env_now) if not env_now.isdigit() else int(env_now)
        elif state.get("clock_mode") == "wall":
            state["now"] = state["init_now"] + int(time.time()) - state["wall_anchor"]
        self.now: int = int(state["now"])

    # ---- persistence ----------------------------------------------------------------------
    def to_state(self) -> dict:
        self.state["jobs"] = {str(k): asdict(v) for k, v in sorted(self.jobs.items())}
        self.state["nodes"] = [asdict(n) for n in self.nodes.values()]
        self.state["partitions"] = [asdict(p) for p in self.partitions]
        self.state["now"] = self.now
        return self.state

    # ---- config accessors -----------------------------------------------------------------
    def cfg_int(self, key: str, default: int = 0) -> int:
        v = self.config.get(key, default)
        if isinstance(v, str):
            m = re.match(r"-?\d+", v)
            return int(m.group(0)) if m else default
        return int(v)

    def min_job_age(self) -> int:
        return self.cfg_int("MinJobAge", 300)

    def job_requeue_default(self) -> bool:
        return bool(self.cfg_int("JobRequeue", 1))

    def partition(self, name: str) -> Partition | None:
        for p in self.partitions:
            if p.name == name:
                return p
        return None

    def default_partition(self) -> Partition | None:
        forced = self.fake.get("submit_default_partition")
        if forced:
            return self.partition(forced)
        for p in self.partitions:
            if p.default:
                return p
        return None

    def qos(self, name: str) -> dict | None:
        return self.qos_table.get(name)

    def user_account(self, account: str) -> dict | None:
        return self.user["accounts"].get(account)

    def current_user(self) -> str:
        return os.environ.get("FAKESLURM_USER") or self.user["name"]

    def next_seq(self) -> int:
        self.state["seq"] = self.state.get("seq", 0) + 1
        return self.state["seq"]

    def event(self, kind: str, job: Job | None, **kw) -> None:
        ev = {"seq": self.next_seq(), "time": self.now, "kind": kind}
        if job is not None:
            ev["job"] = job.id
        ev.update(kw)
        self.state.setdefault("events", []).append(ev)

    # ---- resources ------------------------------------------------------------------------
    def node_usage(self, name: str) -> tuple[int, int, dict[tuple[str, str], int], list[Job]]:
        cpus = mem = 0
        gres: dict[tuple[str, str], int] = {}
        jobs = []
        node = self.nodes[name]
        for j in self.jobs.values():
            if j.state in ("RUNNING", "COMPLETING") and name in j.nodes:
                jobs.append(j)
                if j.exclusive:
                    cpus = node.cpus
                    mem = node.real_memory
                    for k, v in node_gres_map(node.gres).items():
                        gres[k] = v
                    continue
                cpus += j.cpus_per_node
                mem += j.mem_mb
                for g in j.gres:
                    key = (g["name"], g["type"])
                    cnt = g["count"]
                    if not g["type"]:
                        # untyped request: charge against any type present on the node
                        for k in node_gres_map(node.gres):
                            if k[0] == g["name"]:
                                key = k
                                break
                    gres[key] = gres.get(key, 0) + cnt
        return min(cpus, node.cpus), min(mem, node.real_memory), gres, jobs

    def node_state(self, name: str) -> str:
        node = self.nodes[name]
        if node.state in ("down", "down*"):
            return "down*" if node.reason == "Not responding" else "down"
        cpus, _, _, jobs = self.node_usage(name)
        if node.state == "drain":
            return "drng" if jobs else "drain"
        if not jobs:
            return "idle"
        if cpus >= node.cpus:
            return "alloc"
        return "mix"

    def node_available(self, name: str) -> bool:
        return self.nodes[name].state not in ("down", "drain", "down*")

    def job_fits_node(self, job: Job, node_name: str, part: Partition, ignore_jobs: set[int] | None = None) -> bool:
        node = self.nodes[node_name]
        if not self.node_available(node_name):
            return False
        cpus, mem, gres, jobs = self.node_usage(node_name)
        if ignore_jobs:
            # recompute without the ignored jobs (preemption what-if)
            cpus = mem = 0
            gres = {}
            for j in jobs:
                if j.id in ignore_jobs:
                    continue
                if j.exclusive:
                    return False
                cpus += j.cpus_per_node
                mem += j.mem_mb
                for g in j.gres:
                    key = (g["name"], g["type"])
                    gres[key] = gres.get(key, 0) + g["count"]
            jobs = [j for j in jobs if j.id not in ignore_jobs]
        if any(j.exclusive for j in jobs):
            return False
        if job.exclusive or part.oversubscribe.upper().startswith("EXCLUSIVE"):
            return not jobs
        if job.cpus_per_node > node.cpus - cpus:
            return False
        if job.mem_mb > node.real_memory - mem:
            return False
        if job.constraint:
            feats = set(node.features.split(",")) if node.features else set()
            for c in re.split(r"[&,]", job.constraint):
                if c and c not in feats:
                    return False
        avail = node_gres_map(node.gres)
        for g in job.gres:
            need = g["count"]
            if g["type"]:
                have = avail.get((g["name"], g["type"]), 0) - gres.get((g["name"], g["type"]), 0)
            else:
                have = sum(v for k, v in avail.items() if k[0] == g["name"]) - \
                    sum(v for k, v in gres.items() if k[0] == g["name"])
            if have < need:
                return False
        return True

    def job_could_ever_fit(self, job: Job, part: Partition, available_only: bool = False) -> bool:
        """Does an empty node of the partition satisfy the per-node request?"""
        for name in part.node_names():
            if name not in self.nodes:
                continue
            if available_only and not self.node_available(name):
                continue
            node = self.nodes[name]
            if job.cpus_per_node > node.cpus or job.mem_mb > node.real_memory:
                continue
            avail = node_gres_map(node.gres)
            ok = True
            for g in job.gres:
                if g["type"]:
                    have = avail.get((g["name"], g["type"]), 0)
                else:
                    have = sum(v for k, v in avail.items() if k[0] == g["name"])
                if have < g["count"]:
                    ok = False
            if job.constraint:
                feats = set(node.features.split(",")) if node.features else set()
                if any(c and c not in feats for c in re.split(r"[&,]", job.constraint)):
                    ok = False
            if ok:
                return True
        return False

    def find_nodes(self, job: Job, part: Partition, ignore_jobs: set[int] | None = None) -> list[str] | None:
        cands = [n for n in part.node_names() if n in self.nodes]
        if job.req_nodes:
            cands = [n for n in cands if n in job.req_nodes]
        if job.exc_nodes:
            cands = [n for n in cands if n not in job.exc_nodes]
        fit = [n for n in cands if self.job_fits_node(job, n, part, ignore_jobs)]
        # prefer already-busy nodes (best fit / consolidation) then name order
        fit.sort(key=lambda n: (self.node_usage(n)[0] == 0, n))
        if len(fit) >= job.num_nodes:
            return fit[: job.num_nodes]
        return None

    # ---- priority ---------------------------------------------------------------------------
    def compute_priority(self, job: Job) -> int:
        if job.held:
            return 0
        acct = self.user_account(job.account) or {}
        fs = float(acct.get("fairshare", 0.5))
        w_fs = self.cfg_int("PriorityWeightFairShare", 1000000)
        w_age = self.cfg_int("PriorityWeightAge", 10000)
        w_qos = self.cfg_int("PriorityWeightQOS", 5000000)
        max_age = (parse_time_limit(str(self.config.get("PriorityMaxAge", "7-00:00:00"))) or 10080) * 60
        age = max(0, self.now - (job.eligible or job.submit))
        age_part = int(w_age * min(1.0, age / max_age))
        q = self.qos(job.qos) or {}
        max_q = max([int(v.get("priority", 0)) for v in self.qos_table.values()] + [1])
        qos_part = int(w_qos * (int(q.get("priority", 0)) / max_q)) if max_q else 0
        return max(0, int(w_fs * fs) + age_part + qos_part - job.nice)

    # ---- job life cycle ------------------------------------------------------------------
    def _note_pending_reason(self, job: Job) -> None:
        """sacct Reason = "the last reason a job was blocked from running for something other than
        Priority or Resources" (sacct(1)); it is saved even if the job then runs to completion."""
        if job.reason and job.reason not in ("None", "Resources", "Priority"):
            job.last_pending_reason = job.reason

    def start_job(self, job: Job, part: Partition, nodes: list[str], scheduler: str = "Main") -> None:
        self._note_pending_reason(job)
        job.state = "RUNNING"
        job.reason = "None"
        job.partition = part.name
        if part.name in job.partitions:
            # sbatch(1): "When the job is initiated, the name of the partition used will be placed first
            # in the job record partition string"  [verify on cluster: no fixture has a running multi-partition job]
            job.partitions = [part.name] + [p for p in job.partitions if p != part.name]
        job.nodes = list(nodes)
        job.sched_nodes = []
        job.start = self.now
        job.end = None
        job.est_start = None
        job.scheduler = scheduler
        job.last_sched_eval = self.now
        job.flags = ["SchedMain" if scheduler == "Main" else "SchedBackfill", "StartRecieved"]
        job.batch_state = "RUNNING"
        job.exit_code = job.exit_signal = 0
        job.batch_exit = job.batch_signal = 0
        job.max_rss_k = job.marker_max_rss_k
        job.priority = self.compute_priority(job)
        launched = self._touch_output_files(job)
        self.event("start", job, nodes=nodes, partition=part.name)
        if not launched:
            # slurmstepd cannot open StdOut/StdErr (directory missing): "IO setup failed", the batch step
            # dies at launch and the job ends FAILED with ExitCode 0:53, no output file written.
            # [verify on cluster]
            self.event("launch_failed", job)
            self.finish_job(job, "FAILED", exit_code=0, exit_signal=53, batch_state="FAILED",
                            batch_exit=0, batch_signal=53)

    def _touch_output_files(self, job: Job) -> bool:
        """slurmd creates the output files at launch; return False (creating nothing) when a directory is missing."""
        paths = [posix_to_native(p) for p in {self.resolve_pattern(job, job.stdout), self.resolve_pattern(job, job.stderr)} if p]
        if any(not os.path.isdir(os.path.dirname(p) or ".") for p in paths):
            return False
        try:
            mode = "a" if job.open_mode == "append" else "w"
            for p in paths:
                with open(p, mode, encoding="utf-8"):
                    pass
        except OSError:
            return False
        return True

    def resolve_pattern(self, job: Job, pattern: str) -> str:
        if not pattern:
            return ""
        first_node = job.nodes[0] if job.nodes else ""

        def repl(m: re.Match) -> str:
            width = m.group(1)
            code = m.group(2)
            val = {"j": str(job.id), "J": f"{job.id}.batch", "A": str(job.array_job_id or job.id),
                   "a": str(job.array_task_id if job.array_task_id is not None else 4294967294),
                   "u": job.user, "x": job.name, "N": first_node, "n": "0", "s": "batch", "t": "0",
                   "%": "%"}.get(code, m.group(0))
            if width and val.isdigit():
                val = val.zfill(int(width))
            return val

        path = re.sub(r"%(\d*)([jJAauxNnst%])", repl, pattern)
        if job.workdir:
            path = path_join(job.workdir, path)
        return path

    def finish_job(self, job: Job, state: str, end: int | None = None, exit_code: int = 0,
                   exit_signal: int = 0, batch_state: str | None = None, batch_exit: int | None = None,
                   batch_signal: int | None = None, cancelled_by: int | None = None) -> None:
        end = self.now if end is None else end
        job.state = state
        job.end = end
        job.exit_code = exit_code
        job.exit_signal = exit_signal
        job.batch_state = batch_state or state
        job.batch_exit = exit_code if batch_exit is None else batch_exit
        job.batch_signal = exit_signal if batch_signal is None else batch_signal
        job.cancelled_by = cancelled_by
        if job.start is not None:
            elapsed = max(0, end - job.start)
            if job.max_rss_k is None:
                job.max_rss_k = job.marker_max_rss_k
            if job.max_rss_k is None:
                job.max_rss_k = int(min(job.mem_mb * 1024 * 0.15, 56459172)) if job.mem_mb else 4456848
            job.total_cpu = round(elapsed * job.cpus_per_node * 0.6, 3)
        else:
            job.nodes = []
        # squeue/scontrol Reason of a finished job (sacct prints last_pending_reason instead, see _sacct_row)
        if state in ("COMPLETED",):
            job.reason = "None"
        elif state == "FAILED":
            job.reason = "NonZeroExitCode"
        elif state == "TIMEOUT":
            job.reason = "TimeLimit"
        elif state == "OUT_OF_MEMORY":
            job.reason = "OutOfMemory"
        self.event("finish", job, state=state, exit=f"{exit_code}:{exit_signal}")

    def snapshot_incarnation(self, job: Job, state: str) -> None:
        snap = {"submit": job.submit, "eligible": job.eligible, "start": job.start, "end": self.now,
                "state": state, "nodes": list(job.nodes), "partition": job.partition,
                "exit_code": 0, "exit_signal": 0, "restarts": job.restarts,
                "flags": list(job.flags), "max_rss_k": job.max_rss_k, "reason": job.last_pending_reason,
                "total_cpu": round(job.elapsed(self.now) * job.cpus_per_node * 0.6, 3),
                "cpus_per_node": job.cpus_per_node, "num_nodes": job.num_nodes}
        job.incarnations.append(snap)

    def requeue_job(self, job: Job, prior_state: str, reason: str = "BeginTime") -> None:
        """Put a running job back to PENDING (preemption, scontrol requeue, node failure)."""
        self.snapshot_incarnation(job, prior_state)
        job.restarts += 1
        job.state = "PENDING"
        job.reason = reason
        job.nodes = []
        job.start = None
        job.end = None
        job.submit = self.now
        job.eligible = self.now + int(self.fake.get("requeue_delay", 120))
        job.begin = job.eligible
        job.flags = ["StartRecieved"]
        job.batch_state = ""
        job.preempt_time = None
        job.exit_code = job.exit_signal = 0
        job.scheduler = "Main"
        job.est_start = None
        self.event("requeue", job, prior_state=prior_state)

    def preempt_job(self, job: Job, by: Job | None = None) -> None:
        job.preempt_time = self.now
        job.signals.append({"time": self.now, "signal": "TERM", "source": "preempt"})
        if job.requeue:
            self.requeue_job(job, "PREEMPTED")
        else:
            self.finish_job(job, "PREEMPTED", exit_code=0, exit_signal=0,
                            batch_state="CANCELLED", batch_exit=0, batch_signal=15)
        self.event("preempt", job, by=by.id if by else None, requeued=job.requeue)

    # ---- dependencies ---------------------------------------------------------------------
    def dependency_status(self, job: Job) -> str:
        """'ok' | 'wait' | 'never'."""
        if not job.dependency:
            return "ok"
        spec = job.dependency
        any_of = "?" in spec
        results = []
        for clause in re.split(r"[,?]", spec):
            if not clause:
                continue
            parts = clause.split(":")
            typ = parts[0]
            if typ == "singleton":
                others = [j for j in self.jobs.values() if j.id != job.id and j.name == job.name and
                          j.user == job.user and j.state in ("RUNNING", "COMPLETING") and not j.purged]
                results.append("wait" if others else "ok")
                continue
            for tok in parts[1:]:
                tok = tok.split("+")[0]
                try:
                    dep_id = int(tok.split("_")[0])
                except ValueError:
                    results.append("never")
                    continue
                dep = self.jobs.get(dep_id)
                if dep is None:
                    results.append("never" if typ != "after" else "ok")
                    continue
                if typ == "after":
                    results.append("ok" if dep.start is not None or dep.is_terminal() else "wait")
                elif typ == "afterany":
                    results.append("ok" if dep.is_terminal() else "wait")
                elif typ == "afterok":
                    if dep.state == "COMPLETED":
                        results.append("ok")
                    elif dep.is_terminal():
                        results.append("never")
                    else:
                        results.append("wait")
                elif typ == "afternotok":
                    if dep.state == "COMPLETED":
                        results.append("never")
                    elif dep.is_terminal():
                        results.append("ok")
                    else:
                        results.append("wait")
                elif typ in ("aftercorr", "afterburstbuffer"):
                    results.append("ok" if dep.is_terminal() else "wait")
                else:
                    results.append("never")
        if not results:
            return "ok"
        if any_of:
            if "ok" in results:
                return "ok"
            return "never" if all(r == "never" for r in results) else "wait"
        if "never" in results:
            return "never"
        return "ok" if all(r == "ok" for r in results) else "wait"

    def remaining_dependency(self, job: Job) -> str:
        if not job.dependency:
            return "(null)"
        out = []
        for clause in re.split(r"([,?])", job.dependency):
            if clause in (",", "?", ""):
                if clause:
                    out.append(clause)
                continue
            parts = clause.split(":")
            keep = []
            for tok in parts[1:]:
                dep = self.jobs.get(int(tok.split("_")[0].split("+")[0])) if tok.split("_")[0].split("+")[0].isdigit() else None
                if dep is None or not dep.is_terminal():
                    keep.append(tok)
            if parts[0] == "singleton":
                out.append("singleton")
            elif keep:
                out.append(":".join([parts[0]] + keep))
        s = "".join(out).strip(",?")
        return s or "(null)"

    # ---- the tick ---------------------------------------------------------------------------
    def tick(self) -> None:
        now = self.now
        # 1. finish running jobs whose planned duration or time limit elapsed (in time order)
        changed = True
        while changed:
            changed = False
            for job in sorted(self.jobs.values(), key=lambda j: j.id):
                if job.state != "RUNNING" or job.start is None:
                    continue
                ends = []
                if job.duration is not None:
                    ends.append((job.start + job.duration, "script"))
                lim = job.limit_secs()
                if lim is not None:
                    ends.append((job.start + lim, "timeout"))
                ends.sort()
                for when, how in ends:
                    if when <= now:
                        if how == "script":
                            st = "COMPLETED" if job.planned_exit == 0 else "FAILED"
                            self.finish_job(job, st, end=when, exit_code=job.planned_exit)
                        else:
                            self.finish_job(job, "TIMEOUT", end=when, exit_code=0, exit_signal=0,
                                            batch_state="CANCELLED", batch_exit=0, batch_signal=15)
                        changed = True
                        break
        # 2. purge finished jobs older than MinJobAge from controller memory
        for job in self.jobs.values():
            if job.is_terminal() and not job.purged and job.end is not None and \
                    job.end + self.min_job_age() <= now:
                job.purged = True
        # 3. schedule pending jobs
        self.schedule()

    def _eligible(self, job: Job) -> tuple[bool, str]:
        """Return (can_be_considered, reason_if_not)."""
        if job.held:
            return False, job.hold_reason or "JobHeldUser"
        if job.begin is not None and job.begin > self.now:
            return False, "BeginTime"
        if job.submit + int(self.fake.get("sched_interval", 1)) > self.now:
            return False, "None"
        dep = self.dependency_status(job)
        if dep == "wait":
            return False, "Dependency"
        if dep == "never":
            if job.kill_on_invalid_dep:
                self.finish_job(job, "CANCELLED", cancelled_by=job.uid)
                return False, "DependencyNeverSatisfied"
            return False, "DependencyNeverSatisfied"
        # array throttle
        if job.array_job_id is not None and job.array_task_throttle:
            running = sum(1 for j in self.jobs.values() if j.array_job_id == job.array_job_id and j.is_running())
            if running >= job.array_task_throttle:
                return False, "JobArrayTaskLimit"
        # QOS MaxJobsPU (job qos and partition qos)
        for qname in {job.qos} | {self.partition(p).qos for p in job.partitions if self.partition(p)}:
            q = self.qos(qname) if qname else None
            if q and q.get("maxjobspu"):
                running = sum(1 for j in self.jobs.values() if j.user == job.user and j.is_running() and
                              (j.qos == qname or any((self.partition(p) and self.partition(p).qos == qname)
                                                     for p in [j.partition])))
                if running >= int(q["maxjobspu"]):
                    return False, "QOSMaxJobsPerUserLimit"
        return True, ""

    def schedule(self) -> None:
        pending = [j for j in self.jobs.values() if j.is_pending()]
        for j in pending:
            j.priority = self.compute_priority(j)
            self._note_pending_reason(j)
        blocked_in_part: dict[str, int] = {}
        # highest partition tier first, then priority, then id
        def tier(j: Job) -> int:
            return max([self.partition(p).priority_tier for p in j.partitions if self.partition(p)] + [0])
        pending.sort(key=lambda j: (-tier(j), -j.priority, j.id))
        for job in pending:
            if job.state != "PENDING":
                continue
            ok, reason = self._eligible(job)
            if not ok:
                job.reason = reason
                job.last_sched_eval = self.now
                continue
            parts = [self.partition(p) for p in job.partitions if self.partition(p)]
            parts.sort(key=lambda p: -p.priority_tier)
            started = False
            for part in parts:
                nodes = self.find_nodes(job, part)
                if nodes:
                    self.start_job(job, part, nodes, "Main" if blocked_in_part.get(part.name, 0) == 0 else "Backfill")
                    started = True
                    break
            if started:
                continue
            # preemption (partition_prio + REQUEUE)
            if "partition_prio" in str(self.config.get("PreemptType", "")):
                for part in parts:
                    victims = self._preemption_candidates(job, part)
                    if victims is not None:
                        for v in victims:
                            self.preempt_job(v, by=job)
                        nodes = self.find_nodes(job, part)
                        if nodes:
                            self.start_job(job, part, nodes, "Main")
                            started = True
                            break
            if started:
                continue
            # blocked: figure out the reason
            part = parts[0] if parts else None
            if part is None:
                job.reason = "PartitionConfig"
            elif not any(self.job_could_ever_fit(job, p) for p in parts):
                job.reason = "BadConstraints"
            elif not any(self.job_could_ever_fit(job, p, available_only=True) for p in parts):
                unavailable = sorted({n for p in parts for n in p.node_names()
                                      if n in self.nodes and not self.node_available(n)})
                job.reason = "ReqNodeNotAvail, UnavailableNodes:" + hostlist_compress(unavailable)
            else:
                n = blocked_in_part.get(part.name, 0)
                job.reason = "Resources" if n == 0 else "Priority"
                blocked_in_part[part.name] = n + 1
                job.est_start = self._estimate_start(job, part, n)
                job.sched_nodes = self._sched_nodes(job, part)
            job.last_sched_eval = self.now

    def _preemption_candidates(self, job: Job, part: Partition) -> list[Job] | None:
        """Lower-tier running jobs whose removal lets `job` start on `part`."""
        if "REQUEUE" not in part.preempt_mode.upper() and "CANCEL" not in part.preempt_mode.upper():
            # preemption is driven by the *victim's* partition mode; check victims below
            pass
        victims: list[Job] = []
        for name in part.node_names():
            if name not in self.nodes or not self.node_available(name):
                continue
            _, _, _, running = self.node_usage(name)
            lower = [j for j in running if (self.partition(j.partition) and
                                            self.partition(j.partition).priority_tier < part.priority_tier and
                                            ("REQUEUE" in self.partition(j.partition).preempt_mode.upper() or
                                             "CANCEL" in self.partition(j.partition).preempt_mode.upper()))]
            if not lower:
                continue
            ids = {j.id for j in lower}
            if self.job_fits_node(job, name, part, ignore_jobs=ids):
                # take the minimum set: youngest first until it fits
                lower.sort(key=lambda j: -(j.start or 0))
                chosen: list[Job] = []
                for v in lower:
                    chosen.append(v)
                    if self.job_fits_node(job, name, part, ignore_jobs={c.id for c in chosen}):
                        break
                victims.extend(chosen)
                if len({n for v in victims for n in v.nodes}) >= job.num_nodes:
                    return victims
        return None

    def _node_free_times(self, part: Partition) -> list[tuple[int, str]]:
        times = []
        for name in part.node_names():
            if name not in self.nodes or not self.node_available(name):
                continue
            _, _, _, running = self.node_usage(name)
            ends = [((j.start or self.now) + (j.limit_secs() or 365 * 86400)) for j in running]
            times.append((max(ends) if ends else self.now, name))
        times.sort()
        return times

    def _estimate_start(self, job: Job, part: Partition, rank: int) -> int | None:
        times = self._node_free_times(part)
        if not times:
            return None
        idx = min(rank + job.num_nodes - 1, len(times) - 1)
        return max(self.now, times[idx][0])

    def _sched_nodes(self, job: Job, part: Partition) -> list[str]:
        times = self._node_free_times(part)
        return [n for _, n in times[: job.num_nodes]]


def load_sim() -> tuple[Sim, str]:
    path = state_path()
    if not os.path.exists(path):
        raise CommandError("fakeslurm: state file not found; run `fakeslurm-ctl init --cluster <name>` first", 2)
    with open(path, encoding="utf-8") as fh:
        state = json.load(fh)
    return Sim(state), path


def save_sim(sim: Sim, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(sim.to_state(), fh, indent=1, sort_keys=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------------
# generic getopt_long-like parser
# --------------------------------------------------------------------------------------------------
def parse_opts(argv: list[str], spec: dict[str, tuple[str | None, object]], prog: str,
               allow_positional: bool = True) -> tuple[list[tuple[str, str | None]], list[str]]:
    """spec: long_name -> (short_letter|None, takes_value: True|False|'optional').
    Returns ([(long_name, value), ...], positionals). Supports --x=v, --x v, -xv, -x v, unique prefixes."""
    short_map = {v[0]: k for k, v in spec.items() if v[0]}
    opts: list[tuple[str, str | None]] = []
    pos: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--":
            pos.extend(argv[i + 1:])
            break
        if a.startswith("--") and len(a) > 2:
            name, eq, val = a[2:].partition("=")
            if name not in spec:
                cands = [k for k in spec if k.startswith(name)]
                if len(cands) == 1:
                    name = cands[0]
                elif len(cands) > 1:
                    raise CommandError(f"{prog}: option '--{name}' is ambiguous\nTry \"{prog} --help\" for more information")
                else:
                    raise CommandError(f"{prog}: unrecognized option '--{name}'\nTry \"{prog} --help\" for more information")
            takes = spec[name][1]
            if takes is True:
                if not eq:
                    if i + 1 >= len(argv):
                        raise CommandError(f"{prog}: option '--{name}' requires an argument\nTry \"{prog} --help\" for more information")
                    i += 1
                    val = argv[i]
                opts.append((name, val))
            elif takes == "optional":
                opts.append((name, val if eq else None))
            else:
                opts.append((name, None))
        elif a.startswith("-") and len(a) > 1 and not a[1:].isdigit():
            j = 1
            while j < len(a):
                ch = a[j]
                if ch not in short_map:
                    raise CommandError(f"{prog}: invalid option -- '{ch}'\nTry \"{prog} --help\" for more information")
                name = short_map[ch]
                takes = spec[name][1]
                if takes is True:
                    if j + 1 < len(a):
                        opts.append((name, a[j + 1:]))
                    else:
                        if i + 1 >= len(argv):
                            raise CommandError(f"{prog}: option requires an argument -- '{ch}'\nTry \"{prog} --help\" for more information")
                        i += 1
                        opts.append((name, argv[i]))
                    break
                elif takes == "optional":
                    opts.append((name, a[j + 1:] if j + 1 < len(a) else None))
                    break
                else:
                    opts.append((name, None))
                j += 1
        else:
            if not allow_positional:
                raise CommandError(f"{prog}: unexpected argument {a!r}")
            pos.append(a)
        i += 1
    return opts, pos


# --------------------------------------------------------------------------------------------------
# sbatch
# --------------------------------------------------------------------------------------------------
SBATCH_SPEC: dict[str, tuple[str | None, object]] = {
    "partition": ("p", True), "account": ("A", True), "qos": ("q", True), "time": ("t", True),
    "job-name": ("J", True), "output": ("o", True), "error": ("e", True), "input": ("i", True),
    "nodes": ("N", True), "ntasks": ("n", True), "cpus-per-task": ("c", True),
    "ntasks-per-node": (None, True), "mem": (None, True), "mem-per-cpu": (None, True),
    "mem-per-gpu": (None, True), "gres": (None, True), "gpus": ("G", True), "gpus-per-node": (None, True),
    "gpus-per-task": (None, True), "cpus-per-gpu": (None, True), "requeue": (None, False),
    "no-requeue": (None, False), "signal": (None, True), "dependency": ("d", True), "array": ("a", True),
    "comment": (None, True), "export": (None, True), "export-file": (None, True), "chdir": ("D", True),
    "workdir": (None, True), "open-mode": (None, True), "parsable": (None, False), "test-only": (None, False),
    "wrap": (None, True), "hold": ("H", False), "begin": ("b", True), "time-min": (None, True),
    "kill-on-invalid-dep": (None, True), "mail-type": (None, True), "mail-user": (None, True),
    "nice": (None, "optional"), "exclusive": (None, "optional"), "constraint": ("C", True),
    "nodelist": ("w", True), "exclude": ("x", True), "verbose": ("v", False), "quiet": ("Q", False),
    "deadline": (None, True), "priority": (None, True), "propagate": (None, "optional"),
    "wait": ("W", False), "version": ("V", False), "help": ("h", False), "usage": (None, False),
    "distribution": ("m", True), "licenses": ("L", True), "reservation": (None, True),
    "oversubscribe": ("s", False), "wckey": (None, True), "tmp": (None, True), "get-user-env": (None, "optional"),
    "no-kill": ("k", "optional"), "clusters": ("M", True), "uid": (None, True), "gid": (None, True),
    "profile": (None, True), "acctg-freq": (None, True), "core-spec": ("S", True), "threads-per-core": (None, True),
    "sockets-per-node": (None, True), "cores-per-socket": (None, True), "ntasks-per-socket": (None, True),
    "ntasks-per-core": (None, True), "ntasks-per-gpu": (None, True), "hint": (None, True), "mincpus": (None, True),
    "switches": (None, True), "spread-job": (None, False), "use-min-nodes": (None, False),
    "prefer": (None, True), "container": (None, True), "network": (None, True), "batch": (None, True),
    "delay-boot": (None, True), "gpu-bind": (None, True), "gpu-freq": (None, True), "mem-bind": (None, True),
    "cpu-freq": (None, True), "extra-node-info": ("B", True), "overcommit": ("O", False),
    "ignore-pbs": (None, False), "kill-on-bad-exit": ("K", "optional"), "contiguous": (None, False),
    "cluster-constraint": (None, True), "mcs-label": (None, True), "power": (None, True),
    "thread-spec": (None, True), "bb": (None, True), "bbf": (None, True), "immediate": ("I", "optional"),
    "no-requeue-dummy": (None, False),
}


def _extract_directives(script: str) -> list[str]:
    """Collect '#SBATCH ...' option tokens that precede the first command line."""
    out: list[str] = []
    lines = script.splitlines()
    for idx, line in enumerate(lines):
        if idx == 0 and line.startswith("#!"):
            continue
        s = line.strip()
        if not s:
            continue
        if s.startswith("#SBATCH"):
            body = s[len("#SBATCH"):].strip()
            body = body.split(" #")[0].strip() if " #" in body else body
            if body:
                try:
                    out.extend(shlex.split(body))
                except ValueError:
                    out.extend(body.split())
            continue
        if s.startswith("#"):
            continue
        break
    return out


def _fake_markers(script: str) -> dict[str, str]:
    """'#FAKESLURM duration=30 exit=1 maxrss=123456[K|M|G]' -> dict of raw strings."""
    out: dict[str, str] = {}
    for line in script.splitlines():
        s = line.strip()
        if s.startswith("#FAKESLURM"):
            for tok in s[len("#FAKESLURM"):].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    out[k] = v
    return out


def parse_marker_kbytes(value: str) -> int:
    """'52676680' / '52676680K' / '50G' -> kilobytes."""
    m = re.fullmatch(r"(\d+)([KkMmGgTt]?)[Bb]?", value.strip())
    if not m:
        raise ValueError(value)
    return int(m.group(1)) * {"": 1, "K": 1, "M": 1024, "G": 1024 ** 2, "T": 1024 ** 3}[m.group(2).upper()]


def _submit_error(ctx: Ctx, test_only: bool, reason: str, pre: list[str] | None = None) -> int:
    for line in pre or []:
        ctx.e(line)
    if test_only:
        ctx.e(f"allocation failure: {reason}")
    else:
        ctx.error(f"Batch job submission failed: {reason}")
    return 1


def cmd_sbatch(ctx: Ctx, sim: Sim, argv: list[str]) -> int:
    opts, pos = parse_opts(argv, SBATCH_SPEC, "sbatch")
    cli = dict(opts)
    if "help" in cli or "usage" in cli:
        ctx.p("Usage: sbatch [OPTIONS(0)...] [ : [OPTIONS(N)...]] script(0) [args(0)...]")
        return 0
    if "version" in cli:
        ctx.p(f"slurm {SLURM_VERSION}")
        return 0
    test_only = "test-only" in cli
    parsable = "parsable" in cli
    wrap = cli.get("wrap")
    script_path = None
    script_args: list[str] = []
    if wrap is not None:
        if pos:
            raise CommandError("sbatch: error: Batch script contains DOS line breaks (\\r\\n)" if False else
                               "sbatch: fatal: Cannot use --wrap with a batch script", 1)
        script = "#!/bin/sh\n# This script was created by sbatch --wrap.\n\n" + wrap + "\n"
        default_name = "wrap"
    elif pos:
        script_path = pos[0]
        script_args = pos[1:]
        try:
            with open(posix_to_native(script_path), encoding="utf-8", errors="replace") as fh:
                script = fh.read()
        except OSError:
            ctx.error(f"Unable to open file {script_path}")
            return 1
        default_name = os.path.basename(script_path)
    else:
        script = ctx.stdin_text
        default_name = "sbatch"
    if not script.startswith("#!"):
        ctx.error("This does not look like a batch script.  The first line must start with #! followed by the path to an interpreter.")
        ctx.e("For instance: #!/bin/sh")
        return 1

    # directives first, then command line overrides
    dopts, _ = parse_opts(_extract_directives(script), SBATCH_SPEC, "sbatch")
    merged: dict[str, str | None] = {}
    for k, v in dopts:
        merged[k] = v
    mem_group = ("mem", "mem-per-cpu", "mem-per-gpu")
    for k, v in opts:
        if k in mem_group:
            for other in mem_group:
                merged.pop(other, None)
        merged[k] = v
    if "workdir" in merged and "chdir" not in merged:
        merged["chdir"] = merged.pop("workdir")
    markers = _fake_markers(script)

    user = sim.user
    now = sim.now
    submit_line = "sbatch " + " ".join(argv)

    # --- account / qos ---------------------------------------------------------------------
    account = merged.get("account") or user["default_account"]
    acct = sim.user_account(account)
    # --- partition ---------------------------------------------------------------------------
    pre_lines: list[str] = []
    if merged.get("partition"):
        part_names = [p for p in merged["partition"].split(",") if p]
    else:
        dp = sim.default_partition()
        part_names = [dp.name] if dp else []
    parts: list[Partition] = []
    for pn in part_names:
        part = sim.partition(pn)
        if part is None:
            pre_lines.append(f"sbatch: error: invalid partition specified: {pn}")
            return _submit_error(ctx, test_only, "Invalid partition name specified", pre_lines)
        parts.append(part)
    if not parts:
        return _submit_error(ctx, test_only, "Invalid partition name specified")
    if acct is None:
        return _submit_error(ctx, test_only, "Invalid account or account/partition combination specified")
    qos_name = merged.get("qos") or acct.get("default_qos", "normal")
    for part in parts:
        allowed_by_part = part.allow_qos.upper() == "ALL" or qos_name in part.allow_qos.split(",")
        if qos_name not in acct.get("qos", []) or not allowed_by_part or sim.qos(qos_name) is None:
            return _submit_error(ctx, test_only, "Invalid qos specification")
        if part.allow_groups.upper() != "ALL" and not (set(part.allow_groups.split(",")) & set(user.get("groups", []))):
            return _submit_error(ctx, test_only, "User's group not permitted to use this partition")
        if part.allow_accounts.upper() != "ALL" and account not in part.allow_accounts.split(","):
            return _submit_error(ctx, test_only, "Invalid account or account/partition combination specified")

    # --- time ----------------------------------------------------------------------------------
    try:
        time_limit = parse_time_limit(merged["time"]) if merged.get("time") else parts[0].default_time_minutes()
        time_min = parse_time_limit(merged["time-min"]) if merged.get("time-min") else None
    except ValueError:
        ctx.error("Invalid time limit specification")
        return 1
    # --- nodes / tasks / cpus --------------------------------------------------------------
    try:
        num_nodes = 1
        if merged.get("nodes"):
            num_nodes = int(str(merged["nodes"]).split("-")[0])
        ntasks = int(merged["ntasks"]) if merged.get("ntasks") else None
        ntpn = int(merged["ntasks-per-node"]) if merged.get("ntasks-per-node") else 0
        cpt = int(merged["cpus-per-task"]) if merged.get("cpus-per-task") else 1
    except ValueError:
        ctx.error("Invalid numeric value in node/task specification")
        return 1
    if ntasks is None:
        ntasks = ntpn * num_nodes if ntpn else num_nodes
    if ntpn:
        tasks_per_node = ntpn
    else:
        tasks_per_node = -(-ntasks // num_nodes)
    cpus_per_node = max(1, tasks_per_node * cpt)
    # --- gres ---------------------------------------------------------------------------------
    gres: list[dict] = []
    gres_per_job = False
    try:
        if merged.get("gres"):
            gres = parse_gres(merged["gres"])
        elif merged.get("gpus-per-node"):
            gres = parse_gres("gpu:" + merged["gpus-per-node"])
        elif merged.get("gpus"):
            gres = parse_gres("gpu:" + merged["gpus"])
            gres_per_job = True
        elif merged.get("gpus-per-task"):
            g = parse_gres("gpu:" + merged["gpus-per-task"])
            for x in g:
                x["count"] *= tasks_per_node
            gres = g
    except ValueError:
        ctx.error("Invalid generic resource (gres) specification")
        return 1
    for g in gres:
        if g["name"] != "gpu":
            return _submit_error(ctx, test_only, "Invalid generic resource (gres) specification")
    gpus_per_node = sum(g["count"] for g in gres)
    # --- memory --------------------------------------------------------------------------------
    try:
        mem_per_cpu = parse_mem(merged["mem-per-cpu"]) if merged.get("mem-per-cpu") else None
        mem_per_gpu = parse_mem(merged["mem-per-gpu"]) if merged.get("mem-per-gpu") else None
        mem_mb = parse_mem(merged["mem"]) if merged.get("mem") else None
    except ValueError:
        ctx.error("Invalid --mem specification")
        return 1
    part0 = parts[0]
    if mem_mb is None:
        if mem_per_cpu is not None:
            mem_mb = mem_per_cpu * cpus_per_node
        elif mem_per_gpu is not None and gpus_per_node:
            mem_mb = mem_per_gpu * gpus_per_node
        elif part0.def_mem_per_gpu and gpus_per_node:
            mem_mb = int(part0.def_mem_per_gpu) * gpus_per_node
            mem_per_gpu = int(part0.def_mem_per_gpu)
        elif part0.def_mem_per_cpu:
            mem_per_cpu = int(part0.def_mem_per_cpu)
            mem_mb = mem_per_cpu * cpus_per_node
        elif part0.def_mem_per_node and str(part0.def_mem_per_node).upper() != "UNLIMITED":
            mem_mb = int(part0.def_mem_per_node)
        else:
            mem_mb = 0
    exclusive = "exclusive" in merged or any(p.oversubscribe.upper().startswith("EXCLUSIVE") for p in parts)
    if exclusive:
        # whole node: take all cpus / memory of the smallest node in the partition
        nodes0 = [sim.nodes[n] for n in part0.node_names() if n in sim.nodes]
        if nodes0:
            cpus_per_node = max(cpus_per_node, min(n.cpus for n in nodes0))
            if mem_mb == 0:
                mem_mb = min(n.real_memory for n in nodes0)
    if mem_mb == 0:
        nodes0 = [sim.nodes[n] for n in part0.node_names() if n in sim.nodes]
        mem_mb = min(n.real_memory for n in nodes0) if nodes0 else 0

    # --- partition limits (EnforcePartLimits=ALL) -----------------------------------------------
    for part in parts:
        pmax = part.max_time_minutes()
        if pmax is not None and (time_limit is None or time_limit > pmax):
            return _submit_error(ctx, test_only, "Requested time limit is invalid (missing or exceeds some limit)")
        mn = part.max_nodes_int()
        if mn is not None and num_nodes > mn:
            return _submit_error(ctx, test_only, "Node count specification invalid")
    # --- QOS limits (DenyOnLimit) -----------------------------------------------------------
    qos_names = [qos_name] + [p.qos for p in parts if p.qos and p.qos != "N/A"]
    for qn in qos_names:
        q = sim.qos(qn)
        if not q:
            continue
        deny = "DenyOnLimit" in str(q.get("flags", ""))
        mw = parse_time_limit(q["maxwall"]) if q.get("maxwall") else None
        if mw is not None and (time_limit is None or time_limit > mw):
            if deny or qn == qos_name:
                return _submit_error(ctx, test_only,
                                     "Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)",
                                     ["sbatch: error: QOSMaxWallDurationPerJobLimit"])
        if q.get("maxsubmitpu"):
            count = sum(1 for j in sim.jobs.values() if j.user == user["name"] and not j.is_terminal() and
                        (j.qos == qn or any(sim.partition(p) and sim.partition(p).qos == qn for p in j.partitions)))
            if count >= int(q["maxsubmitpu"]):
                return _submit_error(ctx, test_only,
                                     "Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)",
                                     ["sbatch: error: QOSMaxSubmitJobPerUserLimit"])
        maxtres = parse_tres(q.get("maxtres", ""))
        if "cpu" in maxtres and cpus_per_node * num_nodes > maxtres["cpu"]:
            return _submit_error(ctx, test_only,
                                 "Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)",
                                 ["sbatch: error: QOSMaxCpuPerJobLimit"])
        if "node" in maxtres and num_nodes > maxtres["node"]:
            return _submit_error(ctx, test_only,
                                 "Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)",
                                 ["sbatch: error: QOSMaxNodePerJobLimit"])
        if "gres/gpu" in maxtres and gpus_per_node * num_nodes > maxtres["gres/gpu"]:
            return _submit_error(ctx, test_only,
                                 "Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)",
                                 ["sbatch: error: QOSMaxGRESPerJob"])
    # --- dependency ----------------------------------------------------------------------------
    dependency = merged.get("dependency") or ""
    if dependency:
        for clause in re.split(r"[,?]", dependency):
            parts_c = clause.split(":")
            if parts_c[0] not in ("after", "afterany", "afterok", "afternotok", "aftercorr",
                                  "afterburstbuffer", "singleton"):
                return _submit_error(ctx, test_only, "Job dependency problem")
            for tok in parts_c[1:]:
                base = tok.split("_")[0].split("+")[0]
                if not base.isdigit():
                    return _submit_error(ctx, test_only, "Job dependency problem")
                dep = sim.jobs.get(int(base))
                if dep is None or dep.purged:
                    return _submit_error(ctx, test_only, "Job dependency problem")
    # --- array ---------------------------------------------------------------------------------
    array_spec = merged.get("array")
    task_ids: list[int] = []
    throttle = 0
    if array_spec:
        spec = array_spec
        if "%" in spec:
            spec, thr = spec.split("%", 1)
            throttle = int(thr)
        try:
            for piece in spec.split(","):
                step = 1
                if ":" in piece:
                    piece, st = piece.split(":", 1)
                    step = int(st)
                if "-" in piece:
                    a, b = piece.split("-", 1)
                    task_ids.extend(range(int(a), int(b) + 1, step))
                else:
                    task_ids.append(int(piece))
        except ValueError:
            return _submit_error(ctx, test_only, "Invalid job array specification")
        if not task_ids or max(task_ids) >= sim.cfg_int("MaxArraySize", 1001):
            return _submit_error(ctx, test_only, "Invalid job array specification")

    # --- build the job -----------------------------------------------------------------------
    submit_dir = current_dir()
    workdir = path_join(submit_dir, shell_path(merged["chdir"], submit_dir)) if merged.get("chdir") else submit_dir
    name = merged.get("job-name") or default_name
    requeue = sim.job_requeue_default()
    if "requeue" in merged:
        requeue = True
    if "no-requeue" in merged:
        requeue = False
    try:
        begin = parse_slurm_time(merged["begin"], now) if merged.get("begin") else None
    except ValueError:
        ctx.error("Invalid --begin specification")
        return 1
    export = merged.get("export") or "ALL"
    env: dict[str, str] = {}
    for tok in export.split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            env[k] = v
    env["FAKESLURM_EXPORT"] = export
    nice = 0
    if "nice" in merged:
        nice = int(merged["nice"]) if merged["nice"] else 100
    out_pat = shell_path(merged.get("output") or ("slurm-%A_%a.out" if array_spec else "slurm-%j.out"), submit_dir)
    err_pat = shell_path(merged.get("error") or "", submit_dir) or out_pat
    marker_max_rss = None
    duration = None
    planned_exit = 0
    for key in ("duration", "exit", "maxrss"):
        if key not in markers:
            continue
        try:
            if key == "duration":
                duration = int(markers[key])
            elif key == "exit":
                planned_exit = int(markers[key])
            else:
                marker_max_rss = parse_marker_kbytes(markers[key])
        except ValueError:
            expect = "kilobytes with an optional K/M/G/T suffix" if key == "maxrss" else "an integer"
            ctx.error(f"invalid #FAKESLURM {key}={markers[key]!r} (expected {expect})")
            return 1

    def make_job(jid: int, task_id: int | None, array_id: int | None) -> Job:
        job = Job(id=jid, name=name, user=user["name"], uid=int(user["uid"]), account=account, qos=qos_name,
                  partitions=[p.name for p in parts], partition=parts[0].name, submit=now, eligible=now,
                  time_limit=time_limit, time_min=time_min, num_nodes=num_nodes, ntasks=ntasks,
                  ntasks_per_node=ntpn, cpus_per_task=cpt, cpus_per_node=cpus_per_node, mem_mb=mem_mb,
                  mem_per_cpu=mem_per_cpu, mem_per_gpu=mem_per_gpu, gres=[dict(g) for g in gres],
                  gres_per_job=gres_per_job, exclusive=bool("exclusive" in merged),
                  constraint=merged.get("constraint") or "",
                  req_nodes=hostlist_expand(merged.get("nodelist") or ""),
                  exc_nodes=hostlist_expand(merged.get("exclude") or ""),
                  nice=nice, held="hold" in merged, hold_reason="JobHeldUser" if "hold" in merged else "",
                  begin=begin, dependency=dependency,
                  kill_on_invalid_dep=str(merged.get("kill-on-invalid-dep", "no")).lower() == "yes",
                  requeue=requeue, comment=merged.get("comment") or "", script=script,
                  command=path_join(submit_dir, shell_path(script_path, submit_dir)) if script_path else "(null)",
                  workdir=workdir, stdout=out_pat, stderr=err_pat, open_mode=merged.get("open-mode") or "truncate",
                  signal=merged.get("signal") or "", mail_type=merged.get("mail-type") or "",
                  mail_user=merged.get("mail-user") or "", export=export, env=env, submit_line=submit_line,
                  array_job_id=array_id, array_task_id=task_id, array_task_throttle=throttle,
                  array_spec=array_spec or "", duration=duration, planned_exit=planned_exit,
                  marker_max_rss_k=marker_max_rss, alloc_sid=3131000 + sim.next_seq())
        if job.held:
            job.reason = "JobHeldUser"
        elif begin and begin > now:
            job.reason = "BeginTime"
        elif dependency:
            job.reason = "Dependency"
        job.priority = sim.compute_priority(job)
        return job

    probe = make_job(sim.state["next_jobid"], None, None)
    # --- node configuration ---------------------------------------------------------------
    if not any(sim.job_could_ever_fit(probe, p) for p in parts):
        return _submit_error(ctx, test_only, "Requested node configuration is not available")

    if test_only:
        jid = sim.state["next_jobid"]
        parts_sorted = sorted(parts, key=lambda p: -p.priority_tier)
        for part in parts_sorted:
            nodes = sim.find_nodes(probe, part)
            if nodes:
                ctx.e(f"sbatch: Job {jid} to start at {fmt_ts(now)} using {probe.total_cpus()} processors "
                      f"on nodes {hostlist_compress(nodes)} in partition {part.name}")
                return 0
        part = parts_sorted[0]
        rank = sum(1 for j in sim.jobs.values() if j.is_pending() and j.reason in ("Resources", "Priority")
                   and part.name in j.partitions)
        est = sim._estimate_start(probe, part, rank) or now
        nodes = sim._sched_nodes(probe, part) or [part.node_names()[0]]
        ctx.e(f"sbatch: Job {jid} to start at {fmt_ts(est)} using {probe.total_cpus()} processors "
              f"on nodes {hostlist_compress(nodes)} in partition {part.name}")
        return 0

    # --- create job(s) ----------------------------------------------------------------------
    first_id = sim.state["next_jobid"]
    if task_ids:
        for idx, tid in enumerate(task_ids):
            jid = sim.state["next_jobid"]
            sim.state["next_jobid"] += 1
            sim.jobs[jid] = make_job(jid, tid, first_id)
    else:
        sim.state["next_jobid"] += 1
        sim.jobs[first_id] = probe
    sim.event("submit", sim.jobs[first_id], name=name, partition=parts[0].name)
    if parsable:
        ctx.p(str(first_id))
    else:
        ctx.p(f"Submitted batch job {first_id}")
    return 0


# --------------------------------------------------------------------------------------------------
# squeue
# --------------------------------------------------------------------------------------------------
SQUEUE_SPEC: dict[str, tuple[str | None, object]] = {
    "user": ("u", True), "me": (None, False), "noheader": ("h", False), "states": ("t", True),
    "partition": ("p", True), "jobs": ("j", True), "name": ("n", True), "format": ("o", True),
    "Format": ("O", True), "start": (None, False), "array": ("r", False), "sort": ("S", True),
    "nodelist": ("w", True), "account": ("A", True), "qos": ("q", True), "long": ("l", False),
    "steps": ("s", "optional"), "iterate": ("i", True), "verbose": ("v", False), "version": ("V", False),
    "help": (None, False), "noconvert": (None, False), "clusters": ("M", True), "all": ("a", False),
    "hide": (None, False), "reservation": ("R", True), "priority": ("P", False), "array-unique": (None, False),
    "json": (None, "optional"), "yaml": (None, "optional"), "local": (None, False), "sibling": (None, False),
    "federation": (None, False), "licenses": ("L", True), "only-job-state": (None, False),
}

SQUEUE_HEADERS = {
    "i": "JOBID", "j": "NAME", "T": "STATE", "t": "ST", "P": "PARTITION", "R": "NODELIST(REASON)",
    "M": "TIME", "l": "TIME_LIMIT", "D": "NODES", "C": "CPUS", "b": "TRES_PER_NODE", "S": "START_TIME",
    "V": "SUBMIT_TIME", "Q": "PRIORITY", "r": "REASON", "N": "NODELIST", "o": "COMMAND", "Z": "WORK_DIR",
    "u": "USER", "a": "ACCOUNT", "q": "QOS", "k": "COMMENT", "e": "END_TIME", "L": "TIME_LEFT",
    "A": "JOBID", "F": "ARRAY_JOB_ID", "K": "ARRAY_TASK_ID", "Y": "SCHEDNODES", "E": "DEPENDENCY",
    "B": "EXEC_HOST", "y": "NICE", "U": "UID", "p": "PRIORITY", "m": "MIN_MEMORY", "c": "MIN_CPUS",
    "W": "LICENSES", "v": "RESERVATION", "g": "GROUP", "G": "GROUP_ID", "H": "SOCKETS_PER_NODE",
    "I": "CORES_PER_SOCKET", "J": "THREADS_PER_CORE", "d": "MIN_TMP_DISK", "f": "FEATURES",
    "n": "REQ_NODES", "O": "CONTIGUOUS", "s": "OVERSUBSCRIBE", "w": "WCKEY", "x": "EXC_NODES",
    "X": "CORE_SPEC", "z": "S:C:T", "h": "OVER_SUBSCRIBE", "z2": "",
}


def job_display_state(job: Job) -> str:
    return job.state


def job_time_left(job: Job, now: int) -> str:
    if job.time_limit is None:
        return "UNLIMITED"
    lim = job.time_limit * 60
    if job.start is None:
        return fmt_compact(lim)
    return fmt_compact(max(0, lim - job.elapsed(now)))


def job_end_estimate(job: Job) -> int | None:
    if job.end is not None:
        return job.end
    if job.time_limit is None:
        return None
    base = job.start if job.start is not None else job.est_start
    if base is None:
        return None
    return base + job.time_limit * 60


def squeue_value(sim: Sim, job: Job, code: str, collapsed: str | None = None) -> str:
    now = sim.now
    pending_reason = job.reason if job.is_pending() else "None"
    if code == "i":
        return collapsed or job.id_str()
    if code == "A":
        return str(job.id)
    if code == "F":
        return str(job.array_job_id) if job.array_job_id is not None else str(job.id)
    if code == "K":
        return str(job.array_task_id) if job.array_task_id is not None else "N/A"
    if code == "j":
        return job.name
    if code == "T":
        return job_display_state(job)
    if code == "t":
        return STATE_SHORT.get(job.state, job.state[:2])
    if code == "P":
        # comma-list request: the full list is kept, the chosen partition first once started (start_job)
        return ",".join(job.partitions) if job.partitions else job.partition
    if code == "R":
        if job.nodes:
            return hostlist_compress(job.nodes)
        return f"({job.reason})" if job.reason and job.reason != "None" else "(None)" if job.is_pending() else "(null)"
    if code == "r":
        return job.reason if job.is_pending() else "None"
    if code == "M":
        return fmt_compact(job.elapsed(now))
    if code == "l":
        return fmt_limit_compact(job.time_limit)
    if code == "L":
        return job_time_left(job, now)
    if code == "D":
        return str(len(job.nodes) if job.nodes else job.num_nodes)
    if code == "C":
        return str(job.total_cpus())
    if code == "c":
        return str(job.cpus_per_node)
    if code == "m":
        return f"{job.mem_mb}M" if job.mem_per_cpu is None else f"{job.mem_per_cpu}Mc"
    if code == "b":
        return gres_string(job.gres) if job.gres and not job.gres_per_job else "N/A"
    if code == "S":
        t = job.start if job.start is not None else job.est_start
        return fmt_ts(t) if t is not None else "N/A"
    if code == "V":
        return fmt_ts(job.submit)
    if code == "e":
        t = job_end_estimate(job)
        return fmt_ts(t) if t is not None else "N/A"
    if code == "Q":
        return str(job.priority)
    if code == "p":
        return f"{min(1.0, job.priority / 4294967295.0):.8f}"
    if code == "y":
        return str(job.nice)
    if code == "N":
        return hostlist_compress(job.nodes)
    if code == "Y":
        return hostlist_compress(job.sched_nodes) if job.is_pending() and job.sched_nodes else "(null)"
    if code == "B":
        return job.nodes[0] if job.nodes else "n/a"
    if code == "o":
        return job.command
    if code == "Z":
        return job.workdir
    if code == "u":
        return job.user
    if code == "U":
        return str(job.uid)
    if code == "g":
        return sim.user.get("group", "users")
    if code == "G":
        return str(sim.user.get("gid", 100))
    if code == "a":
        return job.account
    if code == "q":
        return job.qos
    if code == "k":
        return job.comment or "(null)"
    if code == "E":
        return sim.remaining_dependency(job)
    if code == "W" or code == "v" or code == "w" or code == "f":
        return "(null)"
    if code == "n":
        return hostlist_compress(job.req_nodes) if job.req_nodes else ""
    if code == "x":
        return hostlist_compress(job.exc_nodes) if job.exc_nodes else ""
    if code == "d":
        return "0"
    if code in ("H", "I", "J"):
        return "*"
    if code == "O":
        return "0"
    if code in ("s", "h"):
        return "OK"
    if code == "X":
        return "N/A"
    if code == "z":
        return "*:*:*"
    if code == "%":
        return "%"
    return ""


SQUEUE_LONG_FIELDS = {
    "jobid": "i", "jobarrayid": "F", "name": "j", "state": "T", "statecompact": "t", "partition": "P",
    "reason": "r", "reasonlist": "R", "timeused": "M", "timelimit": "l", "timeleft": "L", "numnodes": "D",
    "numcpus": "C", "tres-per-node": "b", "starttime": "S", "submittime": "V", "endtime": "e",
    "priority": "p", "prioritylong": "Q", "nodelist": "N", "schednodes": "Y", "batchhost": "B",
    "command": "o", "workdir": "Z", "username": "u", "userid": "U", "account": "a", "qos": "q",
    "comment": "k", "dependency": "E", "arrayjobid": "F", "arraytaskid": "K", "nice": "y",
    "minmemory": "m", "mincpus": "c", "restartcnt": "RESTARTS", "requeue": "REQUEUE", "stdout": "STDOUT",
    "stderr": "STDERR", "stdin": "STDIN", "tres-alloc": "TRESALLOC", "tres-per-job": "TRESPERJOB",
    "exit_code": "EXITCODE", "derivedec": "DERIVEDEC", "pendingtime": "PENDINGTIME", "preempttime": "PREEMPTTIME",
    "eligibletime": "ELIGIBLE", "accruetime": "ELIGIBLE", "groupname": "g", "groupid": "G", "cluster": "CLUSTER",
    "numtasks": "NUMTASKS", "cpus-per-task": "CPT", "nodes": "D", "arraytaskthrottle": "THROTTLE",
    "origin": "CLUSTER", "reservation": "v", "wckey": "w", "licenses": "W", "feature": "f",
    "reqnodes": "n", "sct": "z", "corespec": "X", "oversubscribe": "s",
}

SQUEUE_LONG_HEADERS = {
    "RESTARTS": "RESTARTS", "REQUEUE": "REQUEUE", "STDOUT": "STDOUT", "STDERR": "STDERR", "STDIN": "STDIN",
    "TRESALLOC": "TRES_ALLOC", "TRESPERJOB": "TRES_PER_JOB", "EXITCODE": "EXIT_CODE", "DERIVEDEC": "DERIVED_EC",
    "PENDINGTIME": "PENDING_TIME", "PREEMPTTIME": "PREEMPT_TIME", "ELIGIBLE": "ELIGIBLE_TIME",
    "CLUSTER": "CLUSTER", "NUMTASKS": "TASKS", "CPT": "CPUS_PER_TASK", "THROTTLE": "ARRAY_TASK_THROTTLE",
}


def tres_alloc_string(sim: Sim, job: Job, scontrol_order: bool = False, include_billing: bool = True) -> str:
    cpus = job.total_cpus()
    items = {"cpu": str(cpus), "mem": fmt_mem(job.mem_mb * (len(job.nodes) or job.num_nodes)),
             "node": str(len(job.nodes) or job.num_nodes)}
    if include_billing:
        items["billing"] = str(cpus)
    gpus = job.gpu_count()
    if gpus:
        items["gres/gpu"] = str(gpus)
        for g in job.gres:
            if g["name"] == "gpu" and g["type"] and g["type"] in sim.fake.get("accounting_gpu_types", []):
                items[f"gres/gpu:{g['type']}"] = str(g["count"] * (1 if job.gres_per_job else (len(job.nodes) or job.num_nodes)))
    if scontrol_order:
        order = ["cpu", "mem", "node", "billing"] + sorted(k for k in items if k.startswith("gres"))
        return ",".join(f"{k}={items[k]}" for k in order if k in items)
    return ",".join(f"{k}={items[k]}" for k in sorted(items))


def squeue_long_value(sim: Sim, job: Job, key: str) -> str:
    if len(key) == 1:
        return squeue_value(sim, job, key)
    now = sim.now
    if key == "RESTARTS":
        return str(job.restarts)
    if key == "REQUEUE":
        return "1" if job.requeue else "0"
    if key == "STDOUT":
        return sim.resolve_pattern(job, job.stdout)
    if key == "STDERR":
        return sim.resolve_pattern(job, job.stderr)
    if key == "STDIN":
        return "/dev/null"
    if key == "TRESALLOC":
        return tres_alloc_string(sim, job, scontrol_order=True)
    if key == "TRESPERJOB":
        return gres_string(job.gres, with_count_if_one=False) if job.gres_per_job else "N/A"
    if key == "EXITCODE":
        return job.exit_str()
    if key == "DERIVEDEC":
        return "0:0"
    if key == "PENDINGTIME":
        return str((job.start if job.start is not None else now) - job.submit)
    if key == "PREEMPTTIME":
        return fmt_ts(job.preempt_time) if job.preempt_time else "None"
    if key == "ELIGIBLE":
        return fmt_ts(job.eligible)
    if key == "CLUSTER":
        return sim.config.get("ClusterName", "fake")
    if key == "NUMTASKS":
        return str(job.ntasks)
    if key == "CPT":
        return str(job.cpus_per_task)
    if key == "THROTTLE":
        return str(job.array_task_throttle)
    return ""


def render_format_line(fmt: str, getter, header: bool = False) -> str:
    """squeue/sinfo/sprio '%[.][size]code' renderer. getter(code) -> value; header uses HEADERS via getter."""
    out = []
    i = 0
    while i < len(fmt):
        ch = fmt[i]
        if ch != "%":
            out.append(ch)
            i += 1
            continue
        i += 1
        right = False
        size = ""
        if i < len(fmt) and fmt[i] == ".":
            right = True
            i += 1
        while i < len(fmt) and fmt[i].isdigit():
            size += fmt[i]
            i += 1
        if i >= len(fmt):
            out.append("%")
            break
        code = fmt[i]
        i += 1
        if code == "%":
            out.append("%")
            continue
        val = getter(code)
        if size:
            n = int(size)
            if len(val) > n:
                val = val[:n]
            val = val.rjust(n) if right else val.ljust(n)
        out.append(val)
    return "".join(out)


def parse_long_format(spec: str) -> list[tuple[str, int, bool, str]]:
    """'JobID:0|,Name:.10' -> [(name, size, right, suffix)] with size default 20."""
    items = []
    for tok in spec.split(","):
        if not tok:
            continue
        name, _, rest = tok.partition(":")
        right = False
        size = 20
        suffix = ""
        if rest:
            if rest.startswith("."):
                right = True
                rest = rest[1:]
            m = re.match(r"(\d*)(.*)$", rest)
            if m.group(1):
                size = int(m.group(1))
            suffix = m.group(2)
        items.append((name, size, right, suffix))
    return items


def render_long(items, getter) -> str:
    out = []
    for name, size, right, suffix in items:
        val = getter(name)
        if size:
            val = val[:size]
            val = val.rjust(size) if right else val.ljust(size)
        out.append(val + suffix)
    return "".join(out)


def _parse_states(s: str) -> set[str] | None:
    if s.lower() == "all":
        return None
    out = set()
    for tok in s.split(","):
        t = tok.strip().upper()
        if not t:
            continue
        out.add(STATE_LONG.get(t, t))
    return out


def _sort_key_for(sim: Sim, spec: str):
    keys = []
    for tok in spec.split(","):
        if not tok:
            continue
        desc = tok.startswith("-")
        code = tok.lstrip("+-")
        keys.append((desc, code))

    def key(job: Job):
        vals = []
        for desc, code in keys:
            if code == "P":
                v = job.partition
            elif code == "t":
                v = STATE_ORDER.get(job.state, 99)
            elif code in ("p", "Q"):
                v = job.priority
            elif code == "S":
                t = job.start if job.start is not None else job.est_start
                v = t if t is not None else 1 << 62
            elif code == "i":
                v = job.id
            elif code in ("j", "u", "a", "q"):
                v = squeue_value(sim, job, code)
            elif code == "V":
                v = job.submit
            else:
                v = 0
            if desc:
                if isinstance(v, (int, float)):
                    v = -v
                else:
                    v = tuple(-ord(c) for c in v)
            vals.append(v)
        return tuple(vals)
    return key


def cmd_squeue(ctx: Ctx, sim: Sim, argv: list[str]) -> int:
    opts, pos = parse_opts(argv, SQUEUE_SPEC, "squeue")
    o = dict(opts)
    if "help" in o:
        ctx.p("Usage: squeue [OPTIONS]")
        return 0
    if "version" in o:
        ctx.p(f"slurm {SLURM_VERSION}")
        return 0
    noheader = "noheader" in o
    users = None
    if "me" in o:
        users = {sim.current_user()}
    if o.get("user"):
        users = set(o["user"].split(","))
    states = None
    if o.get("states"):
        states = _parse_states(o["states"])
    if "start" in o:
        states = {"PENDING"}
    parts = set(o["partition"].split(",")) if o.get("partition") else None
    names = set(o["name"].split(",")) if o.get("name") else None
    accounts = set(o["account"].split(",")) if o.get("account") else None
    qoss = set(o["qos"].split(",")) if o.get("qos") else None
    node_filter = set(hostlist_expand(o["nodelist"])) if o.get("nodelist") else None
    job_filter = None
    if o.get("jobs"):
        job_filter = set()
        for tok in o["jobs"].split(","):
            if tok:
                job_filter.add(tok)
    jobs = [j for j in sim.jobs.values() if not j.purged]
    if job_filter is not None:
        sel = []
        for j in jobs:
            if str(j.id) in job_filter or j.id_str() in job_filter or \
                    (j.array_job_id is not None and str(j.array_job_id) in job_filter):
                sel.append(j)
        if not sel and job_filter:
            ctx.e("slurm_load_jobs error: Invalid job id specified")
            return 1
        jobs = sel
    if users is not None:
        jobs = [j for j in jobs if j.user in users]
    if states is not None:
        jobs = [j for j in jobs if j.state in states]
    if parts is not None:
        jobs = [j for j in jobs if set(j.partitions) & parts or j.partition in parts]
    if names is not None:
        jobs = [j for j in jobs if j.name in names]
    if accounts is not None:
        jobs = [j for j in jobs if j.account in accounts]
    if qoss is not None:
        jobs = [j for j in jobs if j.qos in qoss]
    if node_filter is not None:
        jobs = [j for j in jobs if set(j.nodes) & node_filter]

    # array collapsing: pending tasks of one array share a line unless -r
    rows: list[tuple[Job, str | None]] = []
    if "array" in o:
        rows = [(j, None) for j in jobs]
    else:
        seen: set[int] = set()
        for j in jobs:
            if j.array_job_id is not None and j.is_pending():
                if j.array_job_id in seen:
                    continue
                seen.add(j.array_job_id)
                tasks = sorted(t.array_task_id for t in jobs if t.array_job_id == j.array_job_id and t.is_pending())
                label = f"{j.array_job_id}_[" + hostlist_compress([str(t) for t in tasks])[1:-1] if len(tasks) > 1 else None
                if label is not None:
                    if j.array_task_throttle:
                        label += f"%{j.array_task_throttle}"
                    label += "]"
                    # hostlist_compress produced '[..]' from bare numbers only when prefix is empty
                    label = f"{j.array_job_id}_[" + ",".join(_ranges(tasks)) + (f"%{j.array_task_throttle}" if j.array_task_throttle else "") + "]"
                rows.append((j, label))
            else:
                rows.append((j, None))

    sort_spec = o.get("sort") or ("S" if "start" in o else "P,t,-p")
    rows.sort(key=lambda r: _sort_key_for(sim, sort_spec)(r[0]))

    if o.get("Format"):
        items = parse_long_format(o["Format"])

        def getter_for(job: Job, collapsed):
            def g(name: str) -> str:
                key = SQUEUE_LONG_FIELDS.get(name.lower())
                if key is None:
                    raise CommandError(f"squeue: error: Invalid job format specification: {name}")
                if key == "i":
                    return collapsed or job.id_str()
                return squeue_long_value(sim, job, key)
            return g

        if not noheader:
            def hg(name: str) -> str:
                key = SQUEUE_LONG_FIELDS.get(name.lower())
                if key is None:
                    raise CommandError(f"squeue: error: Invalid job format specification: {name}")
                return SQUEUE_HEADERS.get(key, SQUEUE_LONG_HEADERS.get(key, name.upper())) if len(key) == 1 \
                    else SQUEUE_LONG_HEADERS.get(key, name.upper())
            ctx.p(render_long(items, hg))
        for job, collapsed in rows:
            ctx.p(render_long(items, getter_for(job, collapsed)))
        return 0

    if o.get("format"):
        fmt = o["format"]
    elif "start" in o:
        fmt = "%.18i %.9P %.8j %.8u %.2t %.19S %.6D %20Y %R"
    elif "long" in o:
        fmt = "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R"
    else:
        fmt = "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
    if not noheader:
        if "long" in o:
            ctx.p(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime(sim.now)))
        ctx.p(render_format_line(fmt, lambda c: SQUEUE_HEADERS.get(c, "")))
    for job, collapsed in rows:
        ctx.p(render_format_line(fmt, lambda c, j=job, cl=collapsed: squeue_value(sim, j, c, cl)))
    return 0


def _ranges(nums: list[int]) -> list[str]:
    out = []
    if not nums:
        return out
    start = prev = nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return out


# --------------------------------------------------------------------------------------------------
# sinfo
# --------------------------------------------------------------------------------------------------
SINFO_SPEC: dict[str, tuple[str | None, object]] = {
    "noheader": ("h", False), "summarize": ("s", False), "Node": ("N", False), "partition": ("p", True),
    "states": ("t", True), "format": ("o", True), "Format": ("O", True), "nodes": ("n", True),
    "responding": ("r", False), "list-reasons": ("R", False), "exact": ("e", False), "long": ("l", False),
    "all": ("a", False), "reservation": ("T", False), "noconvert": (None, False), "dead": ("d", False),
    "iterate": ("i", True), "sort": ("S", True), "verbose": ("v", False), "version": ("V", False),
    "help": (None, False), "clusters": ("M", True), "future": ("F", False), "json": (None, "optional"),
    "yaml": (None, "optional"), "hide": (None, False), "local": (None, False), "federation": (None, False),
}

SINFO_HEADERS = {
    "P": "PARTITION", "R": "PARTITION", "a": "AVAIL", "l": "TIMELIMIT", "L": "DEFAULTTIME", "s": "JOB_SIZE",
    "D": "NODES", "t": "STATE", "T": "STATE", "C": "CPUS(A/I/O/T)", "F": "NODES(A/I/O/T)", "A": "NODES(A/I)",
    "G": "GRES", "m": "MEMORY", "e": "FREE_MEM", "O": "CPU_LOAD", "c": "CPUS", "f": "AVAIL_FEATURES",
    "b": "ACTIVE_FEATURES", "N": "NODELIST", "n": "HOSTNAMES", "E": "REASON", "H": "TIMESTAMP", "u": "USER",
    "U": "USER", "w": "WEIGHT", "X": "SOCKETS", "Y": "CORES", "Z": "THREADS", "z": "S:C:T", "d": "TMP_DISK",
    "g": "GROUPS", "h": "OVERSUBSCRIBE", "I": "PRIO_JOB_FACTOR", "p": "PRIO_TIER", "M": "PREEMPT_MODE",
    "v": "VERSION", "V": "CLUSTER", "o": "NODE_ADDR", "r": "ROOT", "S": "ALLOCNODES", "B": "MAX_CPUS_PER_NODE",
    "i": "ALLOC_MEM",
}

STATE_SORT = ["comp", "drain*", "down*", "drng", "drain", "mix", "alloc", "down", "idle", "resv", "maint", "unk"]


def node_view(sim: Sim, name: str) -> dict:
    node = sim.nodes[name]
    cpus, mem, gres, jobs = sim.node_usage(name)
    state = sim.node_state(name)
    if state in ("down", "down*", "drain", "drain*"):
        a, i, o = 0, 0, node.cpus
    elif state == "drng":
        a, i, o = cpus, 0, node.cpus - cpus
    else:
        a, i, o = cpus, node.cpus - cpus, 0
    seed = sum(ord(c) for c in name)
    free_mem = node.real_memory - mem - (seed * 3517 % 200000)
    load = round(cpus * (0.9 + (seed % 20) / 100.0), 2) if state not in ("down", "down*") else None
    return {"node": node, "state": state, "alloc_cpus": a, "idle_cpus": i, "other_cpus": o, "alloc_mem": mem,
            "free_mem": max(0, free_mem), "load": load, "jobs": jobs, "gres_used": gres}


def sinfo_value(sim: Sim, part: Partition, group_nodes: list[str], code: str, per_node: bool) -> str:
    views = [node_view(sim, n) for n in group_nodes]
    v0 = views[0] if views else None
    n0 = v0["node"] if v0 else None
    default_mark = "*" if part.default else ""
    if code == "P":
        return part.name + default_mark
    if code == "R":
        return part.name
    if code == "a":
        return part.state.lower()
    if code == "l":
        return fmt_limit_compact(part.max_time_minutes(), "infinite")
    if code == "L":
        return fmt_limit_compact(part.default_time_minutes(), "n/a")
    if code == "s":
        mn = part.max_nodes_int()
        return f"{part.min_nodes or 1}-{'infinite' if mn is None else mn}"
    if code == "D":
        return str(len(group_nodes))
    if code == "t":
        return v0["state"] if v0 else "n/a"
    if code == "T":
        long = {"idle": "idle", "mix": "mixed", "alloc": "allocated", "down": "down", "down*": "down*",
                "drain": "drained", "drng": "draining", "comp": "completing", "resv": "reserved"}
        return long.get(v0["state"], v0["state"]) if v0 else "n/a"
    if code == "C":
        return "/".join(str(sum(v[k] for v in views)) for k in ("alloc_cpus", "idle_cpus", "other_cpus")) + \
            "/" + str(sum(v["node"].cpus for v in views))
    if code == "F" or code == "A":
        a = sum(1 for v in views if v["state"] in ("alloc", "mix", "comp", "drng"))
        o = sum(1 for v in views if v["state"] in ("down", "down*", "drain", "drain*", "maint", "unk"))
        i = len(views) - a - o
        return f"{a}/{i}/{o}/{len(views)}" if code == "F" else f"{a}/{i}"
    if code == "G":
        return n0.gres if n0 and n0.gres else "(null)"
    if code == "m":
        return str(n0.real_memory) if n0 else "0"
    if code == "i":
        return str(v0["alloc_mem"]) if v0 else "0"
    if code == "e":
        return str(v0["free_mem"]) if v0 and v0["load"] is not None else "N/A"
    if code == "O":
        return f"{v0['load']:.2f}" if v0 and v0["load"] is not None else "N/A"
    if code == "c":
        return str(n0.cpus) if n0 else "0"
    if code == "f" or code == "b":
        return n0.features if n0 and n0.features else "(null)"
    if code == "N" or code == "n":
        return hostlist_compress(group_nodes)
    if code == "o":
        return group_nodes[0] if group_nodes else ""
    if code == "E":
        return n0.reason if n0 and n0.reason else "none"
    if code == "H":
        return fmt_ts(sim.state.get("init_now", sim.now)) if n0 and n0.reason else "Unknown"
    if code in ("u", "U"):
        return "root(0)" if n0 and n0.reason else "Unknown"
    if code == "w":
        return str(n0.weight) if n0 else "1"
    if code == "X":
        return str(n0.sockets) if n0 else "0"
    if code == "Y":
        return str(n0.cores_per_socket) if n0 else "0"
    if code == "Z":
        return str(n0.threads_per_core) if n0 else "0"
    if code == "z":
        return f"{n0.sockets}:{n0.cores_per_socket}:{n0.threads_per_core}" if n0 else "0:0:0"
    if code == "d":
        return "0"
    if code == "g":
        return part.allow_groups.lower() if part.allow_groups.upper() != "ALL" else "all"
    if code == "h":
        return part.oversubscribe
    if code == "I":
        return str(part.priority_job_factor)
    if code == "p":
        return str(part.priority_tier)
    if code == "M":
        return part.preempt_mode
    if code == "v":
        return SLURM_VERSION
    if code == "V":
        return sim.config.get("ClusterName", "fake")
    if code == "r":
        return "no"
    if code == "S":
        return "all"
    if code == "B":
        return "UNLIMITED"
    return ""


SINFO_LONG_FIELDS = {
    "partition": "P", "partitionname": "R", "available": "a", "time": "l", "defaulttime": "L", "size": "s",
    "nodes": "D", "statecompact": "t", "statelong": "T", "statecomplete": "T", "cpusstate": "C", "nodeaiot": "F",
    "nodeai": "A", "gres": "G", "gresused": "GRESUSED", "memory": "m", "freemem": "e", "allocmem": "i",
    "cpusload": "O", "cpus": "c", "features": "f", "features_act": "b", "nodelist": "N", "nodehost": "n",
    "reason": "E", "timestamp": "H", "user": "u", "userlong": "U", "weight": "w", "sockets": "X", "cores": "Y",
    "threads": "Z", "socketcorethread": "z", "disk": "d", "groups": "g", "oversubscribe": "h",
    "priorityjobfactor": "I", "prioritytier": "p", "preemptmode": "M", "version": "v", "cluster": "V",
    "nodeaddr": "o", "root": "r", "allocnodes": "S", "maxcpuspernode": "B", "all": "ALL", "comment": "COMMENT",
    "extra": "EXTRA", "port": "PORT",
}


def cmd_sinfo(ctx: Ctx, sim: Sim, argv: list[str]) -> int:
    opts, pos = parse_opts(argv, SINFO_SPEC, "sinfo")
    o = dict(opts)
    if "help" in o:
        ctx.p("Usage: sinfo [OPTIONS]")
        return 0
    if "version" in o:
        ctx.p(f"slurm {SLURM_VERSION}")
        return 0
    noheader = "noheader" in o
    per_node = "Node" in o
    summarize = "summarize" in o
    parts = list(sim.partitions)
    if o.get("partition"):
        want = o["partition"].split(",")
        parts = [p for p in parts if p.name in want]
        if not parts:
            ctx.e("sinfo: error: Invalid partition name specified")  # matches real message text
            return 1
    state_filter = None
    if o.get("states"):
        state_filter = {s.strip().lower() for s in o["states"].split(",")}
        alias = {"allocated": "alloc", "mixed": "mix", "drained": "drain", "draining": "drng",
                 "completing": "comp", "reserved": "resv", "idle": "idle", "down": "down"}
        state_filter = {alias.get(s, s) for s in state_filter}
    node_filter = set(hostlist_expand(o["nodes"])) if o.get("nodes") else None

    if o.get("Format"):
        items = parse_long_format(o["Format"])
        fmt = None
    else:
        items = None
        if summarize:
            fmt = o.get("format") or "%9P %.5a %.10l %.16F  %N"
        elif per_node:
            fmt = o.get("format") or ("%N %.6D %.9P %.11T %.4c %.8z %.6m %.8d %.6w %.8f %20E" if "long" in o
                                      else "%N %.6D %.9P %6t")
        else:
            fmt = o.get("format") or ("%9P %.5a %.10l %.10s %.4r %.8h %.10g %.6D %.11T %N" if "long" in o
                                      else "%9P %.5a %.10l %.6D %.6t %N")

    def getter(part, nodes):
        def g(code):
            if items is not None:
                key = SINFO_LONG_FIELDS.get(code.lower())
                if key is None:
                    raise CommandError(f"sinfo: error: Invalid node format specification: {code}")
                if key == "GRESUSED":
                    if not nodes:
                        return ""
                    n = sim.nodes[nodes[0]]
                    used = sim.node_usage(nodes[0])[2]
                    if not n.gres:
                        return "gpu:0"
                    outs = []
                    for (gname, gtype), cnt in node_gres_map(n.gres).items():
                        u = used.get((gname, gtype), 0)
                        idx = f"(IDX:0-{u-1})" if u > 1 else "(IDX:0)" if u == 1 else "(IDX:N/A)"
                        outs.append(f"{gname}:{gtype}:{u}{idx}" if gtype else f"{gname}:{u}{idx}")
                    return ",".join(outs)
                if key in ("ALL", "COMMENT", "EXTRA", "PORT"):
                    return "(null)" if key != "PORT" else "6818"
                return sinfo_value(sim, part, nodes, key, per_node)
            return sinfo_value(sim, part, nodes, code, per_node)
        return g

    def header_getter(code):
        if items is not None:
            key = SINFO_LONG_FIELDS.get(code.lower(), code)
            return SINFO_HEADERS.get(key, code.upper())
        return SINFO_HEADERS.get(code, "")

    lines: list[str] = []
    if not noheader:
        lines.append(render_long(items, header_getter) if items is not None else render_format_line(fmt, header_getter))
    codes_used = set(re.findall(r"%\.?\d*([A-Za-z])", fmt)) if fmt else {SINFO_LONG_FIELDS.get(n.lower(), "") for n, *_ in items}
    for part in parts:
        names = [n for n in part.node_names() if n in sim.nodes]
        if node_filter is not None:
            names = [n for n in names if n in node_filter]
        if summarize:
            if state_filter is not None:
                names = [n for n in names if sim.node_state(n) in state_filter]
            g = getter(part, names)
            lines.append(render_long(items, g) if items is not None else render_format_line(fmt, g))
            continue
        if per_node:
            groups = [[n] for n in sorted(names)]
        else:
            # group nodes with identical values in the *displayed* per-node fields (like real sinfo)
            key_codes = [c for c in codes_used if c in ("t", "T", "G", "m", "c", "f", "b", "e", "O", "i", "w",
                                                        "X", "Y", "Z", "z", "E", "H", "u", "U", "d")]
            buckets: dict[tuple, list[str]] = {}
            order: list[tuple] = []
            for n in names:
                k = tuple(sinfo_value(sim, part, [n], c, False) for c in sorted(key_codes))
                if k not in buckets:
                    buckets[k] = []
                    order.append(k)
                buckets[k].append(n)

            def _state_rank(k: tuple) -> int:
                st = sim.node_state(buckets[k][0])
                return STATE_SORT.index(st) if st in STATE_SORT else 99
            order.sort(key=lambda k: (_state_rank(k), k))
            groups = [buckets[k] for k in order]
        for grp in groups:
            if state_filter is not None and sim.node_state(grp[0]) not in state_filter:
                continue
            g = getter(part, grp)
            lines.append(render_long(items, g) if items is not None else render_format_line(fmt, g))
    for ln in lines:
        ctx.p(ln)
    return 0


# --------------------------------------------------------------------------------------------------
# sacct
# --------------------------------------------------------------------------------------------------
SACCT_SPEC: dict[str, tuple[str | None, object]] = {
    "user": ("u", True), "jobs": ("j", True), "allocations": ("X", False), "noheader": ("n", False),
    "parsable2": ("P", False), "parsable": ("p", False), "starttime": ("S", True), "endtime": ("E", True),
    "state": ("s", True), "duplicates": ("D", False), "format": ("o", True), "brief": ("b", False),
    "long": ("l", False), "allusers": ("a", False), "accounts": ("A", True), "partition": ("r", True),
    "qos": ("q", True), "name": (None, True), "truncate": ("T", False), "units": (None, True),
    "noconvert": (None, False), "delimiter": (None, True), "helpformat": ("e", False), "help": ("h", False),
    "version": ("V", False), "verbose": ("v", False), "clusters": ("M", True), "allclusters": ("L", False),
    "nodelist": ("N", True), "completion": ("c", False), "gid": ("g", True), "uid": (None, True),
    "wckeys": ("W", True), "array": (None, False), "whole-hetjob": (None, "optional"), "batch-script": ("B", False),
    "env-vars": (None, False), "json": (None, "optional"), "yaml": (None, "optional"), "constraints": ("C", True),
    "reason": (None, True), "flags": ("f", True), "timelimit-min": ("k", True), "timelimit-max": ("K", True),
    "nnodes": ("i", True), "ncpus": ("I", True), "associations": (None, True), "federation": (None, False),
    "local": (None, False), "use-local-uid": (None, False), "expand-patterns": (None, False),
}

SACCT_FIELDS = {  # canonical name -> default width
    "JobID": 12, "JobIDRaw": 12, "JobName": 10, "State": 10, "ExitCode": 8, "DerivedExitCode": 15,
    "Elapsed": 10, "ElapsedRaw": 10, "Start": 19, "End": 19, "Submit": 19, "Eligible": 19, "Partition": 10,
    "Account": 10, "QOS": 10, "NodeList": 15, "AllocTRES": 30, "ReqTRES": 30, "MaxRSS": 10, "TotalCPU": 10,
    "CPUTimeRAW": 10, "CPUTime": 10, "Reason": 10, "WorkDir": 30, "Comment": 20, "Timelimit": 10,
    "TimelimitRaw": 12, "NCPUS": 10, "NNodes": 8, "Flags": 20, "SubmitLine": 30, "User": 9, "UID": 5,
    "Cluster": 10, "Priority": 10, "AllocCPUS": 10, "AllocNodes": 10, "ReqCPUS": 8, "ReqNodes": 8,
    "ReqMem": 10, "MaxVMSize": 10, "AveRSS": 10, "UserCPU": 10, "SystemCPU": 10, "NTasks": 8,
    "Layout": 10, "Group": 9, "GID": 5, "Suspended": 10, "Planned": 10, "Reserved": 10, "MaxRSSNode": 10,
    "MaxRSSTask": 10, "AveCPU": 10, "DBIndex": 10, "JobIDRaw2": 12, "AllocGRES": 10, "ReqGRES": 10,
    "TRESUsageInTot": 20, "TRESUsageInMax": 20, "Constraints": 10, "Container": 10, "AdminComment": 20,
    "SystemComment": 20, "WCKey": 10, "Reservation": 10, "ReservationId": 13, "SubmitTime": 19,
}
SACCT_FIELD_LOOKUP = {k.lower(): k for k in SACCT_FIELDS}


UNIT_ORDER = "\0KMGTP"


def convert_num_unit(num: float, orig: str = "K", units: str | None = None, noconvert: bool = False) -> str:
    """Port of Slurm's convert_num_unit2(): `units` (K/M/G/T/P) forces the unit, `noconvert` keeps the
    original unit, otherwise the value is scaled up only while it stays exact to a half unit
    (CONVERT_NUM_UNIT_EXACT, the sacct default: 56459172K stays K, 4194304K -> 4G, 2560K -> 2.50M)."""
    if int(num) == 0:
        return "0"
    idx = UNIT_ORDER.index(orig)
    if units:
        target = UNIT_ORDER.index(units.upper())
        while target < idx:
            num *= 1024
            idx -= 1
        while target > idx:
            num /= 1024
            idx += 1
    elif noconvert:
        pass
    else:
        while num >= 1024 and int(num) % 512 == 0:
            num /= 1024
            idx += 1
    unit = UNIT_ORDER[idx] if idx > 0 else ""
    return (f"{int(num)}{unit}" if float(int(num)) == num else f"{num:.2f}{unit}")


def _sacct_row(sim: Sim, job: Job, step: str | None, snap: dict | None = None, units: str | None = None,
               noconvert: bool = False) -> dict[str, str]:
    """Build the value map for an allocation row (step None) or '.batch'/'.extern' step row."""
    now = sim.now

    def mem(kbytes: int | None, orig: str = "K") -> str:
        return convert_num_unit(float(kbytes or 0), orig, units, noconvert)

    if snap is None:
        state = job.state_str_sacct()
        start, end = job.start, job.end
        nodes = job.nodes
        restarts = job.restarts
        flags = job.flags
        exit_s = job.exit_str()
        max_rss = job.max_rss_k
        reason = job.last_pending_reason or "None"
        total_cpu = job.total_cpu if job.end is not None else (job.elapsed(now) * job.cpus_per_node * 0.6 if job.start else 0)
        submit = job.submit
        batch_state = job.batch_state or state
        batch_exit = f"{job.batch_exit}:{job.batch_signal}"
    else:
        state = snap["state"]
        start, end = snap["start"], snap["end"]
        nodes = snap["nodes"]
        flags = snap.get("flags", [])
        exit_s = "0:0"
        max_rss = snap.get("max_rss_k")
        reason = snap.get("reason") or "None"
        total_cpu = snap.get("total_cpu", 0)
        submit = snap["submit"]
        batch_state = "CANCELLED" if state in ("PREEMPTED", "REQUEUED") else state
        batch_exit = "0:15" if batch_state == "CANCELLED" else "0:0"
    running = state == "RUNNING"
    elapsed = 0 if start is None else (max(0, (end if end is not None else now) - start))
    never_started = start is None
    nnodes = len(nodes) if nodes else job.num_nodes
    ncpus = job.cpus_per_node * nnodes
    jid = job.id_str()
    raw = str(job.id)
    is_alloc = step is None
    tres_alloc = "" if never_started else tres_alloc_string(sim, job)
    step_tres = "" if never_started else tres_alloc_string(sim, job, include_billing=(step == "extern"))
    if start is None:
        start_s = "None" if state.startswith("CANCELLED") else "Unknown"
    else:
        start_s = fmt_ts(start)
    end_s = fmt_ts(end) if end is not None else "Unknown"
    if is_alloc:
        st = state
        ex = exit_s
        step_end = end_s
        step_elapsed = elapsed
    else:
        if step == "batch":
            st = batch_state if not running else "RUNNING"
            ex = batch_exit if not running else "0:0"
            step_elapsed = elapsed + (3 if st == "CANCELLED" and end is not None else 0)
            step_end = fmt_ts(end + 3) if (end is not None and st == "CANCELLED") else end_s
        else:
            st = "COMPLETED" if not running else "RUNNING"
            ex = "0:0"
            step_elapsed = elapsed
            step_end = end_s
    row = {
        "JobID": jid if is_alloc else f"{jid}.{step}",
        "JobIDRaw": raw if is_alloc else f"{raw}.{step}",
        "JobName": job.name if is_alloc else step,
        "State": st,
        "ExitCode": ex,
        "DerivedExitCode": "0:0" if is_alloc else "",
        "Elapsed": fmt_hms(step_elapsed),
        "ElapsedRaw": str(step_elapsed),
        "Start": start_s,
        "End": step_end,
        "Submit": fmt_ts(submit) if is_alloc else (fmt_ts(start) if start is not None else "Unknown"),
        "SubmitTime": fmt_ts(submit),
        "Eligible": fmt_ts(job.eligible) if is_alloc else (fmt_ts(start) if start is not None else "Unknown"),
        "Partition": job.partition if is_alloc else "",
        "Account": job.account,
        "QOS": job.qos if is_alloc else "",
        "NodeList": (hostlist_compress(nodes) if nodes else "None assigned"),
        "AllocTRES": tres_alloc if is_alloc else step_tres,
        "ReqTRES": tres_alloc_string(sim, job) if is_alloc else "",
        "MaxRSS": "" if is_alloc else ("" if running else (mem(max_rss) if step == "batch" and max_rss else "0")),
        "TotalCPU": fmt_cpu_time(total_cpu if step != "extern" else 0),
        "CPUTimeRAW": str(step_elapsed * ncpus),
        "CPUTime": fmt_hms(step_elapsed * ncpus),
        # sacct(1): "The last reason a job was blocked from running for something other than Priority or
        # Resources" -- NOT the terminal reason: fixtures show Reason=None for TIMEOUT/FAILED/CANCELLED jobs
        "Reason": reason if is_alloc else "",
        "WorkDir": job.workdir if is_alloc else "",
        "Comment": (job.comment if "job_comment" in str(sim.config.get("AccountingStoreFlags", "")) else "") if is_alloc else "",
        "Timelimit": fmt_limit_hms(job.time_limit) if is_alloc else "",
        "TimelimitRaw": ("" if job.time_limit is None else str(job.time_limit)) if is_alloc else "",
        "NCPUS": str(ncpus),
        "NNodes": str(nnodes),
        "Flags": ",".join(flags) if is_alloc else "",
        "SubmitLine": job.submit_line if is_alloc else "",
        "User": job.user if is_alloc else "",
        "UID": str(job.uid) if is_alloc else "",
        "Cluster": sim.config.get("ClusterName", "fake"),
        "Priority": str(job.priority) if is_alloc else "",
        "AllocCPUS": str(ncpus),
        "AllocNodes": str(nnodes),
        "ReqCPUS": str(job.total_cpus()),
        "ReqNodes": str(job.num_nodes),
        "ReqMem": mem(job.mem_mb, "M") + ("c" if job.mem_per_cpu else "") if is_alloc else "",
        "MaxVMSize": "" if is_alloc else (mem(int((max_rss or 0) * 1.5)) if step == "batch" and max_rss else "0"),
        "AveRSS": "" if is_alloc else (mem(max_rss) if step == "batch" and max_rss else "0"),
        "UserCPU": fmt_cpu_time(total_cpu * 0.9 if step != "extern" else 0),
        "SystemCPU": fmt_cpu_time(total_cpu * 0.1 if step != "extern" else 0),
        "NTasks": "" if is_alloc else "1",
        "Layout": "" if is_alloc else "Unknown",
        "Group": sim.user.get("group", "users") if is_alloc else "",
        "GID": str(sim.user.get("gid", 100)) if is_alloc else "",
        "Suspended": "00:00:00",
        "Planned": fmt_hms(max(0, (start or submit) - job.eligible)) if is_alloc else "",
        "Reserved": fmt_hms(max(0, (start or submit) - job.eligible)) if is_alloc else "",
        "MaxRSSNode": "" if is_alloc else (nodes[0] if nodes else ""),
        "MaxRSSTask": "" if is_alloc else "0",
        "AveCPU": "" if is_alloc else fmt_cpu_time(total_cpu),
        "DBIndex": str(job.id * 7 + (0 if is_alloc else 1)),
        "AllocGRES": "", "ReqGRES": "",
        "TRESUsageInTot": "" if is_alloc else f"cpu={fmt_hms(int(total_cpu))},mem={max_rss or 0}K",
        "TRESUsageInMax": "" if is_alloc else f"cpu={fmt_hms(int(total_cpu))},mem={max_rss or 0}K",
        "Constraints": job.constraint if is_alloc else "",
        "Container": "", "AdminComment": "", "SystemComment": "", "WCKey": "", "Reservation": "",
        "ReservationId": "",
    }
    return row


def cmd_sacct(ctx: Ctx, sim: Sim, argv: list[str]) -> int:
    opts, pos = parse_opts(argv, SACCT_SPEC, "sacct")
    o = dict(opts)
    if "help" in o:
        ctx.p("Usage: sacct [options]")
        return 0
    if "version" in o:
        ctx.p(f"slurm {SLURM_VERSION}")
        return 0
    if "helpformat" in o:
        names = sorted(SACCT_FIELDS)
        for i in range(0, len(names), 5):
            ctx.p("".join(f"{n:<20}" for n in names[i:i + 5]).rstrip())
        return 0
    now = sim.now
    fmt_spec = o.get("format")
    if "brief" in o:
        fmt_spec = "JobID,State,ExitCode"
    if not fmt_spec:
        fmt_spec = "JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode"
    fields: list[tuple[str, int, bool]] = []
    for tok in fmt_spec.split(","):
        if not tok:
            continue
        name, _, w = tok.partition("%")
        canon = SACCT_FIELD_LOOKUP.get(name.strip().lower())
        if canon is None:
            ctx.e(f'Invalid field requested: "{name}"')
            return 1
        width = SACCT_FIELDS[canon]
        left = False
        if w:
            if w.startswith("-"):
                left = True
                w = w[1:]
            if w.isdigit():
                width = int(w)
        fields.append((canon, width, left))
    parsable2 = "parsable2" in o
    parsable = "parsable" in o
    delim = o.get("delimiter") or "|"
    noheader = "noheader" in o
    units = None
    if o.get("units") is not None:
        units = str(o["units"]).upper()
        if units not in ("K", "M", "G", "T", "P"):
            ctx.error(f"Invalid --units specification: {o['units']} (use one of K, M, G, T, P)")
            return 1
    noconvert = "noconvert" in o

    job_ids = None
    if o.get("jobs"):
        job_ids = [t for t in o["jobs"].split(",") if t]
    try:
        start_t = parse_slurm_time(o["starttime"], now) if o.get("starttime") else None
        end_t = parse_slurm_time(o["endtime"], now) if o.get("endtime") else None
    except ValueError as e:
        ctx.error(f"Invalid time specification ({e.args[0]})")
        return 1
    states = _parse_states(o["state"]) if o.get("state") else None
    if job_ids is None and start_t is None:
        lt = time.localtime(now)
        start_t = int(time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))) if states is None else now
    if end_t is None:
        end_t = now
    users = None if "allusers" in o else {sim.current_user()}
    if o.get("user"):
        users = set(o["user"].split(","))
    accounts = set(o["accounts"].split(",")) if o.get("accounts") else None
    partitions = set(o["partition"].split(",")) if o.get("partition") else None
    qoss = set(o["qos"].split(",")) if o.get("qos") else None
    names = set(o["name"].split(",")) if o.get("name") else None

    jobs = sorted(sim.jobs.values(), key=lambda j: j.id)
    sel: list[tuple[Job, str | None]] = []   # (job, step filter)
    if job_ids is not None:
        for j in jobs:
            for tok in job_ids:
                base, _, step = tok.partition(".")
                if base == str(j.id) or base == j.id_str() or (j.array_job_id is not None and base == str(j.array_job_id)):
                    sel.append((j, step or None))
                    break
    else:
        for j in jobs:
            s = j.submit
            e = j.end if j.end is not None else now
            if start_t is not None and e < start_t:
                continue
            if end_t is not None and s > end_t:
                continue
            sel.append((j, None))
    if users is not None:
        sel = [(j, s) for j, s in sel if j.user in users]
    if accounts is not None:
        sel = [(j, s) for j, s in sel if j.account in accounts]
    if partitions is not None:
        sel = [(j, s) for j, s in sel if j.partition in partitions]
    if qoss is not None:
        sel = [(j, s) for j, s in sel if j.qos in qoss]
    if names is not None:
        sel = [(j, s) for j, s in sel if j.name in names]
    if states is not None:
        sel = [(j, s) for j, s in sel if j.state in states]

    rows: list[dict[str, str]] = []
    for job, stepf in sel:
        incarnations = job.incarnations if "duplicates" in o else []
        for snap in incarnations:
            if stepf is None:
                rows.append(_sacct_row(sim, job, None, snap, units, noconvert))
            if "allocations" not in o and snap["start"] is not None:
                for step in ("batch", "extern"):
                    if stepf is None or stepf == step:
                        rows.append(_sacct_row(sim, job, step, snap, units, noconvert))
        if stepf is None:
            rows.append(_sacct_row(sim, job, None, None, units, noconvert))
        if "allocations" not in o and job.start is not None:
            for step in ("batch", "extern"):
                if stepf is None or stepf == step:
                    rows.append(_sacct_row(sim, job, step, None, units, noconvert))

    def out_line(vals: list[str]) -> str:
        if parsable2:
            return delim.join(vals)
        if parsable:
            return delim.join(vals) + delim
        cells = []
        for (name, width, left), v in zip(fields, vals):
            if len(v) > width:
                v = v[: width - 1] + "+"
            cells.append(v.ljust(width) if left else v.rjust(width))
        return " ".join(cells)

    if not noheader:
        if parsable or parsable2:
            ctx.p(out_line([f for f, _, _ in fields]))
        else:
            hdr = []
            dash = []
            for name, width, left in fields:
                hdr.append(name[:width].ljust(width) if left else name[:width].rjust(width))
                dash.append("-" * width)
            ctx.p(" ".join(hdr))
            ctx.p(" ".join(dash))
    for row in rows:
        ctx.p(out_line([row.get(f, "") for f, _, _ in fields]))
    return 0


# --------------------------------------------------------------------------------------------------
# scontrol
# --------------------------------------------------------------------------------------------------
def scontrol_job_line(sim: Sim, job: Job) -> str:
    now = sim.now
    part = sim.partition(job.partition)
    kv: list[tuple[str, str]] = [("JobId", str(job.id))]
    if job.array_job_id is not None:
        kv += [("ArrayJobId", str(job.array_job_id)), ("ArrayTaskId", str(job.array_task_id))]
        if job.array_task_throttle:
            kv.append(("ArrayTaskThrottle", str(job.array_task_throttle)))
    kv += [("JobName", job.name), ("UserId", f"{job.user}({job.uid})"),
           ("GroupId", f"{sim.user.get('group', 'users')}({sim.user.get('gid', 100)})"), ("MCS_label", "N/A"),
           ("Priority", str(job.priority)), ("Nice", str(job.nice)), ("Account", job.account), ("QOS", job.qos),
           ("JobState", job.state), ("Reason", job.reason if job.reason else "None"),
           ("Dependency", sim.remaining_dependency(job)), ("Requeue", "1" if job.requeue else "0"),
           ("Restarts", str(job.restarts)), ("BatchFlag", "1"), ("Reboot", "0"), ("ExitCode", job.exit_str()),
           ("RunTime", fmt_hms(job.elapsed(now))),
           ("TimeLimit", fmt_limit_hms(job.time_limit)),
           ("TimeMin", "N/A" if job.time_min is None else fmt_limit_hms(job.time_min)),
           ("SubmitTime", fmt_ts(job.submit)), ("EligibleTime", fmt_ts(job.eligible)),
           ("AccrueTime", fmt_ts(job.eligible))]
    if job.start is not None:
        kv.append(("StartTime", fmt_ts(job.start)))
        end = job.end if job.end is not None else (job.start + job.time_limit * 60 if job.time_limit else None)
        kv.append(("EndTime", fmt_ts(end) if end is not None else "Unknown"))
    else:
        est = job.est_start
        kv.append(("StartTime", fmt_ts(est) if est is not None else "Unknown"))
        kv.append(("EndTime", fmt_ts(est + job.time_limit * 60) if (est is not None and job.time_limit) else "Unknown"))
    kv.append(("Deadline", "N/A"))
    if job.start is not None and job.state != "PENDING":
        kv.append(("PreemptEligibleTime", fmt_ts(job.start)))
        kv.append(("PreemptTime", fmt_ts(job.preempt_time) if job.preempt_time else "None"))
    kv += [("SuspendTime", "None"), ("SecsPreSuspend", "0"),
           ("LastSchedEval", fmt_ts(job.last_sched_eval if job.last_sched_eval else job.submit)),
           ("Scheduler", ("Backfill:*" if job.est_start else "Main") if job.is_pending() else job.scheduler),
           ("Partition", ",".join(job.partitions) if job.partitions else job.partition),
           ("AllocNode:Sid", f"0.0.0.0:{job.alloc_sid}"),
           ("ReqNodeList", hostlist_compress(job.req_nodes) or "(null)"),
           ("ExcNodeList", hostlist_compress(job.exc_nodes) or "(null)"),
           ("NodeList", hostlist_compress(job.nodes))]
    if job.is_pending() and job.sched_nodes:
        kv.append(("SchedNodeList", hostlist_compress(job.sched_nodes)))
    if job.nodes:
        kv.append(("BatchHost", job.nodes[0]))
    kv += [("NumNodes", str(len(job.nodes) or job.num_nodes)), ("NumCPUs", str(job.total_cpus())),
           ("NumTasks", str(job.ntasks)), ("CPUs/Task", str(job.cpus_per_task)), ("ReqB:S:C:T", "0:0:*:*"),
           ("TRES", tres_alloc_string(sim, job, scontrol_order=True)), ("Socks/Node", "*"),
           ("NtasksPerN:B:S:C", f"{job.ntasks_per_node}:0:*:*"), ("CoreSpec", "*"),
           ("MinCPUsNode", str(job.cpus_per_node))]
    if job.mem_per_cpu is not None and not job.exclusive:
        kv.append(("MinMemoryCPU", f"{job.mem_per_cpu}M"))
    else:
        kv.append(("MinMemoryNode", fmt_mem(job.mem_mb)))
    kv += [("MinTmpDiskNode", "0"), ("Features", job.constraint or "(null)"), ("DelayBoot", "00:00:00"),
           ("OverSubscribe", "OK" if not job.exclusive else "NO"), ("Contiguous", "0"), ("Licenses", "(null)"),
           ("Network", "(null)"), ("Command", job.command), ("WorkDir", job.workdir)]
    if job.comment:
        kv.append(("Comment", job.comment))
    kv += [("StdErr", sim.resolve_pattern(job, job.stderr)), ("StdIn", "/dev/null"),
           ("StdOut", sim.resolve_pattern(job, job.stdout)), ("Power", "")]
    if job.gres:
        if job.gres_per_job:
            kv.append(("TresPerJob", gres_string(job.gres, with_count_if_one=False)))
        else:
            kv.append(("TresPerNode", gres_string(job.gres)))
    if job.mem_per_gpu is not None:
        kv.append(("MemPerTres", f"gres:gpu:{job.mem_per_gpu}"))
    if job.mail_type:
        kv += [("MailUser", job.mail_user or job.user), ("MailType", job.mail_type.upper())]
    return " ".join(f"{k}={v}" for k, v in kv) + " "


SCONTROL_JOB_GROUPS = [
    ("JobId", "JobName"), ("UserId", "GroupId", "MCS_label"), ("Priority", "Nice", "Account", "QOS"),
    ("JobState", "Reason", "Dependency"), ("Requeue", "Restarts", "BatchFlag", "Reboot", "ExitCode"),
    ("DerivedExitCode",), ("RunTime", "TimeLimit", "TimeMin"), ("SubmitTime", "EligibleTime"), ("AccrueTime",),
    ("StartTime", "EndTime", "Deadline"), ("PreemptEligibleTime", "PreemptTime"),
    ("SuspendTime", "SecsPreSuspend", "LastSchedEval", "Scheduler"),
    ("Partition", "AllocNode:Sid"), ("ReqNodeList", "ExcNodeList"), ("NodeList", "SchedNodeList"), ("BatchHost",),
    ("NumNodes", "NumCPUs", "NumTasks", "CPUs/Task", "ReqB:S:C:T"), ("TRES",),
    ("Socks/Node", "NtasksPerN:B:S:C", "CoreSpec"), ("MinCPUsNode", "MinMemoryNode", "MinMemoryCPU", "MinTmpDiskNode"),
    ("Features", "DelayBoot"), ("OverSubscribe", "Contiguous", "Licenses", "Network"), ("Command",), ("WorkDir",),
    ("Comment",), ("StdErr",), ("StdIn",), ("StdOut",), ("Power",), ("TresPerJob",), ("TresPerNode",),
    ("MemPerTres",), ("MailUser", "MailType"),
]


def _split_kv_line(line: str, known_keys: list[str]) -> list[tuple[str, str]]:
    """Split a 'K=V K=V' one-liner using the known key set (values may contain spaces)."""
    pattern = re.compile(r"(?:^| )(" + "|".join(re.escape(k) for k in known_keys) + r")=")
    positions = [(m.start(1), m.end(), m.group(1)) for m in pattern.finditer(line)]
    out = []
    for idx, (s, e, k) in enumerate(positions):
        nxt = positions[idx + 1][0] if idx + 1 < len(positions) else len(line)
        out.append((k, line[e:nxt].strip()))
    return out


def scontrol_multiline(line: str, groups, known_keys: list[str], indent: str = "   ") -> list[str]:
    kv = dict(_split_kv_line(line, known_keys))
    order = [k for k, _ in _split_kv_line(line, known_keys)]
    lines = []
    used = set()
    for grp in groups:
        present = [k for k in grp if k in kv]
        if not present:
            continue
        used.update(present)
        prefix = "" if not lines else indent
        lines.append(prefix + " ".join(f"{k}={kv[k]}" for k in present))
    for k in order:
        if k not in used:
            lines.append(indent + f"{k}={kv[k]}")
    return lines


SCONTROL_JOB_KEYS = [k for grp in SCONTROL_JOB_GROUPS for k in grp] + ["ArrayJobId", "ArrayTaskId", "ArrayTaskThrottle"]


def scontrol_partition_line(sim: Sim, part: Partition) -> str:
    names = [n for n in part.node_names() if n in sim.nodes]
    cpus = sum(sim.nodes[n].cpus for n in names)
    mem = sum(sim.nodes[n].real_memory for n in names)
    gpus = sum(sum(node_gres_map(sim.nodes[n].gres).values()) for n in names)
    tres = f"cpu={cpus},mem={mem}M,node={len(names)},billing={cpus}"
    if gpus:
        tres += f",gres/gpu={gpus}"
        typed: dict[str, int] = {}
        for n in names:
            for (g, t), c in node_gres_map(sim.nodes[n].gres).items():
                if t and t in sim.fake.get("accounting_gpu_types", []):
                    typed[t] = typed.get(t, 0) + c
        for t in sorted(typed):
            tres += f",gres/gpu:{t}={typed[t]}"
    kv = [("PartitionName", part.name), ("AllowGroups", part.allow_groups), ("AllowAccounts", part.allow_accounts),
          ("AllowQos", part.allow_qos), ("AllocNodes", "ALL"), ("Default", "YES" if part.default else "NO"),
          ("QoS", part.qos), ("DefaultTime", part.default_time if str(part.default_time).upper() != "NONE" else "NONE"),
          ("DisableRootJobs", "NO"), ("ExclusiveUser", "NO"), ("GraceTime", "0"), ("Hidden", "NO"),
          ("MaxNodes", str(part.max_nodes)), ("MaxTime", fmt_limit_hms(part.max_time_minutes())),
          ("MinNodes", str(part.min_nodes)), ("LLN", "NO"), ("MaxCPUsPerNode", "UNLIMITED"), ("Nodes", part.nodes),
          ("PriorityJobFactor", str(part.priority_job_factor)), ("PriorityTier", str(part.priority_tier)),
          ("RootOnly", "NO"), ("ReqResv", "NO"), ("OverSubscribe", part.oversubscribe), ("OverTimeLimit", "NONE"),
          ("PreemptMode", part.preempt_mode), ("State", part.state), ("TotalCPUs", str(cpus)),
          ("TotalNodes", str(len(names))), ("SelectTypeParameters", "NONE"),
          ("JobDefaults", f"DefMemPerGPU={part.def_mem_per_gpu}" if part.def_mem_per_gpu else "(null)")]
    if part.def_mem_per_cpu:
        kv.append(("DefMemPerCPU", str(part.def_mem_per_cpu)))
    else:
        kv.append(("DefMemPerNode", str(part.def_mem_per_node) if part.def_mem_per_node else "UNLIMITED"))
    kv.append(("MaxMemPerNode", str(part.max_mem_per_node)))
    kv.append(("TRES", tres))
    return " ".join(f"{k}={v}" for k, v in kv)


SCONTROL_PART_GROUPS = [
    ("PartitionName",), ("AllowGroups", "AllowAccounts", "AllowQos"), ("AllocNodes", "Default", "QoS"),
    ("DefaultTime", "DisableRootJobs", "ExclusiveUser", "GraceTime", "Hidden"),
    ("MaxNodes", "MaxTime", "MinNodes", "LLN", "MaxCPUsPerNode"), ("Nodes",),
    ("PriorityJobFactor", "PriorityTier", "RootOnly", "ReqResv", "OverSubscribe"),
    ("OverTimeLimit", "PreemptMode"), ("State", "TotalCPUs", "TotalNodes", "SelectTypeParameters"),
    ("JobDefaults",), ("DefMemPerCPU", "DefMemPerNode", "MaxMemPerNode"), ("TRES",),
]
SCONTROL_PART_KEYS = [k for grp in SCONTROL_PART_GROUPS for k in grp]


def scontrol_node_line(sim: Sim, name: str, details: bool) -> str:
    node = sim.nodes[name]
    v = node_view(sim, name)
    st = v["state"]
    long_state = {"idle": "IDLE", "mix": "MIXED", "alloc": "ALLOCATED", "down": "DOWN", "down*": "DOWN*",
                  "drain": "IDLE+DRAIN", "drng": "MIXED+DRAIN", "comp": "COMPLETING"}.get(st, st.upper())
    parts = [p.name for p in sim.partitions if name in p.node_names()]
    cfg = f"cpu={node.cpus},mem={node.real_memory}M,billing={node.cpus}"
    gm = node_gres_map(node.gres)
    if gm:
        cfg += f",gres/gpu={sum(gm.values())}"
        for (g, t), c in gm.items():
            if t and t in sim.fake.get("accounting_gpu_types", []):
                cfg += f",gres/gpu:{t}={c}"
    alloc = ""
    if v["alloc_cpus"]:
        alloc = f"cpu={v['alloc_cpus']},mem={fmt_mem(v['alloc_mem'])}"
        used = sum(v["gres_used"].values())
        if used:
            alloc += f",gres/gpu={used}"
    kv = [("NodeName", name), ("Arch", "x86_64"), ("CoresPerSocket", str(node.cores_per_socket)),
          ("CPUAlloc", str(v["alloc_cpus"])), ("CPUEfctv", str(node.cpus)), ("CPUTot", str(node.cpus)),
          ("CPULoad", f"{v['load']:.2f}" if v["load"] is not None else "N/A"),
          ("AvailableFeatures", node.features or "(null)"), ("ActiveFeatures", node.features or "(null)"),
          ("Gres", node.gres or "(null)")]
    if details:
        kv.append(("GresDrain", "N/A"))
        if node.gres:
            outs = []
            for (g, t), c in gm.items():
                u = v["gres_used"].get((g, t), 0)
                idx = f"(IDX:0-{u - 1})" if u > 1 else "(IDX:0)" if u == 1 else "(IDX:N/A)"
                outs.append(f"{g}:{t}:{u}{idx}" if t else f"{g}:{u}{idx}")
            kv.append(("GresUsed", ",".join(outs)))
        else:
            kv.append(("GresUsed", "gpu:0"))
    kv += [("NodeAddr", name), ("NodeHostName", name), ("Version", SLURM_VERSION), ("OS", "Linux 4.18.0-513.el8.x86_64"),
           ("RealMemory", str(node.real_memory)), ("AllocMem", str(v["alloc_mem"])),
           ("FreeMem", str(v["free_mem"]) if v["load"] is not None else "N/A"), ("Sockets", str(node.sockets)),
           ("Boards", "1"), ("State", long_state), ("ThreadsPerCore", str(node.threads_per_core)), ("TmpDisk", "0"),
           ("Weight", str(node.weight)), ("Owner", "N/A"), ("MCS_label", "N/A"), ("Partitions", ",".join(parts)),
           ("BootTime", fmt_ts(sim.state.get("init_now", sim.now) - 86400 * 7)),
           ("SlurmdStartTime", fmt_ts(sim.state.get("init_now", sim.now) - 86400 * 7 + 120)),
           ("LastBusyTime", fmt_ts(sim.now)), ("CfgTRES", cfg), ("AllocTRES", alloc), ("CapWatts", "n/a"),
           ("CurrentWatts", "0"), ("AveWatts", "0"), ("ExtSensorsJoules", "n/s"), ("ExtSensorsWatts", "0"),
           ("ExtSensorsTemp", "n/s")]
    if node.reason:
        kv.append(("Reason", f"{node.reason} [root@{fmt_ts(sim.state.get('init_now', sim.now))}]"))
    return " ".join(f"{k}={v}" for k, v in kv)


SCONTROL_NODE_GROUPS = [
    ("NodeName", "Arch", "CoresPerSocket"), ("CPUAlloc", "CPUEfctv", "CPUTot", "CPULoad"),
    ("AvailableFeatures",), ("ActiveFeatures",), ("Gres",), ("GresDrain",), ("GresUsed",),
    ("NodeAddr", "NodeHostName", "Version"), ("OS",), ("RealMemory", "AllocMem", "FreeMem", "Sockets", "Boards"),
    ("State", "ThreadsPerCore", "TmpDisk", "Weight", "Owner", "MCS_label"), ("Partitions",),
    ("BootTime", "SlurmdStartTime"), ("LastBusyTime",), ("CfgTRES",), ("AllocTRES",),
    ("CapWatts",), ("CurrentWatts", "AveWatts"), ("ExtSensorsJoules", "ExtSensorsWatts", "ExtSensorsTemp"),
    ("Reason",),
]
SCONTROL_NODE_KEYS = [k for grp in SCONTROL_NODE_GROUPS for k in grp]

SCONTROL_CONFIG_ORDER = [
    "AccountingStorageBackupHost", "AccountingStorageEnforce", "AccountingStorageHost", "AccountingStorageExternalHost",
    "AccountingStorageParameters", "AccountingStoragePort", "AccountingStorageTRES", "AccountingStorageType",
    "AccountingStorageUser", "AccountingStoreFlags", "AcctGatherEnergyType", "AcctGatherFilesystemType",
    "AcctGatherInterconnectType", "AcctGatherNodeFreq", "AcctGatherProfileType", "AllowSpecResourcesUsage",
    "AuthAltTypes", "AuthAltParameters", "AuthInfo", "AuthType", "BatchStartTimeout", "BcastExclude", "BcastParameters",
    "BOOT_TIME", "BurstBufferType", "CliFilterPlugins", "ClusterName", "CommunicationParameters", "CompleteWait",
    "CoreSpecPlugin", "CpuFreqDef", "CpuFreqGovernors", "CredType", "DebugFlags", "DefMemPerNode",
    "DependencyParameters", "DisableRootJobs", "EioTimeout", "EnforcePartLimits", "Epilog", "EpilogMsgTime",
    "EpilogSlurmctld", "ExtSensorsType", "ExtSensorsFreq", "FairShareDampeningFactor", "FederationParameters",
    "FirstJobId", "GetEnvTimeout", "GresTypes", "GpuFreqDef", "GroupUpdateForce", "GroupUpdateTime", "HASH_VAL",
    "HealthCheckInterval", "HealthCheckNodeState", "HealthCheckProgram", "InactiveLimit", "InteractiveStepOptions",
    "JobAcctGatherFrequency", "JobAcctGatherType", "JobAcctGatherParams", "JobCompHost", "JobCompLoc", "JobCompPort",
    "JobCompType", "JobCompUser", "JobContainerType", "JobCredentialPrivateKey", "JobCredentialPublicCertificate",
    "JobDefaults", "JobFileAppend", "JobRequeue", "JobSubmitPlugins", "KillOnBadExit", "KillWait", "LaunchParameters",
    "LaunchType", "Licenses", "LogTimeFormat", "MailDomain", "MailProg", "MaxArraySize", "MaxDBDMsgs", "MaxJobCount",
    "MaxJobId", "MaxMemPerNode", "MaxNodeCount", "MaxStepCount", "MaxTasksPerNode", "MCSPlugin", "MCSParameters",
    "MessageTimeout", "MinJobAge", "MpiDefault", "MpiParams", "NEXT_JOB_ID", "NodeFeaturesPlugins", "OverTimeLimit",
    "PluginDir", "PlugStackConfig", "PowerParameters", "PowerPlugin", "PreemptMode", "PreemptType",
    "PreemptExemptTime", "PrEpParameters", "PrEpPlugins", "PriorityParameters", "PrioritySiteFactorParameters",
    "PrioritySiteFactorPlugin", "PriorityDecayHalfLife", "PriorityCalcPeriod", "PriorityFavorSmall", "PriorityFlags",
    "PriorityMaxAge", "PriorityUsageResetPeriod", "PriorityType", "PriorityWeightAge", "PriorityWeightAssoc",
    "PriorityWeightFairShare", "PriorityWeightJobSize", "PriorityWeightPartition", "PriorityWeightQOS",
    "PriorityWeightTRES", "PrivateData", "ProctrackType", "Prolog", "PrologEpilogTimeout", "PrologSlurmctld",
    "PrologFlags", "PropagatePrioProcess", "PropagateResourceLimits", "PropagateResourceLimitsExcept",
    "RebootProgram", "ReconfigFlags", "RequeueExit", "RequeueExitHold", "ResumeFailProgram", "ResumeProgram",
    "ResumeRate", "ResumeTimeout", "ResvEpilog", "ResvOverRun", "ResvProlog", "ReturnToService", "RoutePlugin",
    "SchedulerParameters", "SchedulerTimeSlice", "SchedulerType", "ScronParameters", "SelectType",
    "SelectTypeParameters", "SlurmUser", "SlurmctldAddr", "SlurmctldDebug", "SlurmctldHost[0]", "SlurmctldLogFile",
    "SlurmctldPort", "SlurmctldSyslogDebug", "SlurmctldPrimaryOffProg", "SlurmctldPrimaryOnProg", "SlurmctldTimeout",
    "SlurmctldParameters", "SlurmdDebug", "SlurmdLogFile", "SlurmdParameters", "SlurmdPidFile", "SlurmdPort",
    "SlurmdSpoolDir", "SlurmdSyslogDebug", "SlurmdTimeout", "SlurmdUser", "SlurmSchedLogFile", "SlurmSchedLogLevel",
    "SlurmctldPidFile", "SlurmctldPlugstack", "SLURM_CONF", "SLURM_VERSION", "SrunEpilog", "SrunPortRange",
    "SrunProlog", "StateSaveLocation", "SuspendExcNodes", "SuspendExcParts", "SuspendProgram", "SuspendRate",
    "SuspendTime", "SuspendTimeout", "SwitchParameters", "SwitchType", "TaskEpilog", "TaskPlugin", "TaskPluginParam",
    "TaskProlog", "TCPTimeout", "TmpFS", "TopologyParam", "TopologyPlugin", "TrackWCKey", "TreeWidth", "UsePam",
    "UnkillableStepProgram", "UnkillableStepTimeout", "VSizeFactor", "WaitTime", "X11Parameters",
]

SCONTROL_CONFIG_DEFAULTS = {
    "AccountingStorageBackupHost": "(null)", "AccountingStorageExternalHost": "(null)",
    "AccountingStorageParameters": "(null)", "AccountingStoragePort": "6819", "AccountingStorageUser": "N/A",
    "AcctGatherEnergyType": "acct_gather_energy/none", "AcctGatherFilesystemType": "acct_gather_filesystem/none",
    "AcctGatherInterconnectType": "acct_gather_interconnect/none", "AcctGatherNodeFreq": "0 sec",
    "AcctGatherProfileType": "acct_gather_profile/none", "AllowSpecResourcesUsage": "No", "AuthAltTypes": "(null)",
    "AuthAltParameters": "(null)", "AuthInfo": "(null)", "AuthType": "auth/munge", "BatchStartTimeout": "10 sec",
    "BcastExclude": "/lib,/usr/lib,/lib64,/usr/lib64", "BcastParameters": "(null)", "BurstBufferType": "(null)",
    "CliFilterPlugins": "(null)", "CommunicationParameters": "(null)", "CompleteWait": "0 sec",
    "CoreSpecPlugin": "core_spec/none", "CpuFreqDef": "Unknown", "CpuFreqGovernors": "OnDemand,Performance,UserSpace",
    "CredType": "cred/munge", "DebugFlags": "(null)", "DependencyParameters": "(null)", "DisableRootJobs": "No",
    "EioTimeout": "60", "Epilog": "(null)", "EpilogMsgTime": "2000 usec", "EpilogSlurmctld": "(null)",
    "ExtSensorsType": "ext_sensors/none", "ExtSensorsFreq": "0 sec", "FairShareDampeningFactor": "1",
    "FederationParameters": "(null)", "GetEnvTimeout": "2 sec", "GpuFreqDef": "high,memory=high",
    "GroupUpdateForce": "1", "GroupUpdateTime": "600 sec", "HASH_VAL": "Match", "HealthCheckInterval": "900 sec",
    "HealthCheckNodeState": "CYCLE,ANY", "HealthCheckProgram": "/usr/sbin/nhc", "InactiveLimit": "120 sec",
    "InteractiveStepOptions": "--interactive --preserve-env --pty $SHELL", "JobAcctGatherFrequency": "30",
    "JobAcctGatherType": "jobacct_gather/cgroup", "JobAcctGatherParams": "(null)", "JobCompHost": "localhost",
    "JobCompLoc": "/var/log/slurm_jobcomp.log", "JobCompPort": "0", "JobCompType": "jobcomp/none",
    "JobCompUser": "root", "JobContainerType": "job_container/none", "JobCredentialPrivateKey": "(null)",
    "JobCredentialPublicCertificate": "(null)", "JobDefaults": "(null)", "KillOnBadExit": "0",
    "LaunchType": "launch/slurm", "Licenses": "(null)", "LogTimeFormat": "iso8601_ms", "MailDomain": "(null)",
    "MailProg": "/bin/mail", "MaxDBDMsgs": "200212", "MaxNodeCount": "53", "MaxStepCount": "40000",
    "MCSPlugin": "mcs/none", "MCSParameters": "(null)", "MessageTimeout": "30 sec", "MpiDefault": "none",
    "MpiParams": "(null)", "NodeFeaturesPlugins": "(null)", "PluginDir": "/usr/lib64/slurm",
    "PlugStackConfig": "(null)", "PowerParameters": "(null)", "PowerPlugin": "", "PrEpParameters": "(null)",
    "PrEpPlugins": "prep/script", "PriorityParameters": "(null)", "PrioritySiteFactorParameters": "(null)",
    "PrioritySiteFactorPlugin": "(null)", "PriorityCalcPeriod": "00:05:00", "PriorityFavorSmall": "No",
    "PriorityFlags": "", "PriorityUsageResetPeriod": "NONE", "PriorityWeightTRES": "(null)",
    "ProctrackType": "proctrack/cgroup", "Prolog": "(null)", "PrologEpilogTimeout": "65534",
    "PrologSlurmctld": "(null)", "PropagatePrioProcess": "0", "PropagateResourceLimits": "CORE",
    "PropagateResourceLimitsExcept": "(null)", "RebootProgram": "(null)", "ReconfigFlags": "(null)",
    "RequeueExit": "(null)", "RequeueExitHold": "(null)", "ResumeFailProgram": "(null)", "ResumeProgram": "(null)",
    "ResumeRate": "300 nodes/min", "ResumeTimeout": "60 sec", "ResvEpilog": "(null)", "ResvOverRun": "0 min",
    "ResvProlog": "(null)", "ReturnToService": "0", "RoutePlugin": "route/default", "SchedulerTimeSlice": "30 sec",
    "ScronParameters": "(null)", "SlurmUser": "slurm(1010)", "SlurmctldAddr": "(null)", "SlurmctldDebug": "info",
    "SlurmctldLogFile": "/var/log/slurmctld.log", "SlurmctldPort": "6817", "SlurmctldSyslogDebug": "(null)",
    "SlurmctldPrimaryOffProg": "(null)", "SlurmctldPrimaryOnProg": "(null)",
    "SlurmctldParameters": "enable_configless,user_resv_delete", "SlurmdDebug": "info",
    "SlurmdLogFile": "/var/log/slurmd.log", "SlurmdParameters": "(null)", "SlurmdPidFile": "/var/run/slurmd.pid",
    "SlurmdPort": "6818", "SlurmdSpoolDir": "/var/spool/slurm/d", "SlurmdSyslogDebug": "(null)",
    "SlurmdUser": "root(0)", "SlurmSchedLogFile": "(null)", "SlurmSchedLogLevel": "0",
    "SlurmctldPidFile": "/var/run/slurmctld.pid", "SlurmctldPlugstack": "(null)", "SLURM_CONF": "/etc/slurm/slurm.conf",
    "SrunEpilog": "(null)", "SrunPortRange": "0-0", "SrunProlog": "(null)", "SuspendExcNodes": "(null)",
    "SuspendExcParts": "(null)", "SuspendProgram": "(null)", "SuspendRate": "60 nodes/min", "SuspendTime": "INFINITE",
    "SuspendTimeout": "30 sec", "SwitchParameters": "(null)", "SwitchType": "switch/none", "TaskEpilog": "(null)",
    "TaskPlugin": "task/affinity,task/cgroup", "TaskPluginParam": "(null type)", "TaskProlog": "(null)",
    "TCPTimeout": "2 sec", "TopologyParam": "(null)", "TopologyPlugin": "topology/none", "TrackWCKey": "No",
    "TreeWidth": "50", "UsePam": "No", "UnkillableStepProgram": "(null)", "UnkillableStepTimeout": "180 sec",
    "VSizeFactor": "0 percent", "WaitTime": "0 sec", "X11Parameters": "(null)",
}


def _find_job_arg(sim: Sim, arg: str) -> list[Job]:
    """Resolve 'N', 'N_M' or a job name to in-memory job records (empty when unknown)."""
    out = []
    if re.fullmatch(r"\d+", arg):
        j = sim.jobs.get(int(arg))
        if j and not j.purged:
            out.append(j)
        out.extend(t for t in sim.jobs.values() if t.array_job_id == int(arg) and not t.purged and t.id != int(arg))
    elif re.fullmatch(r"\d+_\d+", arg):
        a, t = arg.split("_")
        out.extend(j for j in sim.jobs.values() if j.array_job_id == int(a) and j.array_task_id == int(t) and not j.purged)
    else:
        out.extend(j for j in sim.jobs.values() if j.name == arg and not j.purged)
    return out


def cmd_scontrol(ctx: Ctx, sim: Sim, argv: list[str]) -> int:
    oneliner = False
    details = False
    args: list[str] = []
    for a in argv:
        if a in ("-o", "--oneliner"):
            oneliner = True
        elif a in ("-d", "--details", "-dd"):
            details = True
        elif a in ("-a", "--all", "-Q", "--quiet", "-v", "--verbose", "-h", "--hide", "-F", "--future"):
            pass
        elif a in ("-V", "--version"):
            ctx.p(f"slurm {SLURM_VERSION}")
            return 0
        elif a.startswith("-") and not args:
            ctx.error(f"invalid option -- '{a.lstrip('-')}'")
            return 1
        else:
            args.append(a)
    if not args:
        ctx.p("scontrol: interactive mode not supported by fakeslurm")
        return 1
    cmd = args[0].lower()
    rest = args[1:]
    if cmd in ("show", "list"):
        if not rest:
            ctx.error("Invalid entity: (null)")
            return 1
        entity = rest[0].lower()
        ident = rest[1] if len(rest) > 1 else None
        if entity in ("job", "jobs"):
            if ident is not None:
                jobs = _find_job_arg(sim, ident)
                if not jobs:
                    ctx.e("slurm_load_jobs error: Invalid job id specified")
                    return 1
            else:
                jobs = [j for j in sorted(sim.jobs.values(), key=lambda j: j.id) if not j.purged]
                if not jobs:
                    ctx.p("No jobs in the system")
                    return 0
            for idx, job in enumerate(jobs):
                line = scontrol_job_line(sim, job)
                if oneliner:
                    ctx.p(line)
                else:
                    for ln in scontrol_multiline(line, SCONTROL_JOB_GROUPS, SCONTROL_JOB_KEYS):
                        ctx.p(ln)
                    ctx.p("")
            return 0
        if entity in ("partition", "partitions"):
            parts = sim.partitions if ident is None else [p for p in sim.partitions if p.name == ident]
            if not parts:
                ctx.e("Partition " + str(ident) + " not found")
                return 1
            for part in parts:
                line = scontrol_partition_line(sim, part)
                if oneliner:
                    ctx.p(line)
                else:
                    for ln in scontrol_multiline(line, SCONTROL_PART_GROUPS, SCONTROL_PART_KEYS):
                        ctx.p(ln)
                    ctx.p("")
            return 0
        if entity in ("node", "nodes"):
            names = sorted(sim.nodes) if ident is None else [n for n in hostlist_expand(ident) if n in sim.nodes]
            if ident is not None and not names:
                ctx.e(f"Node {ident} not found")
                return 1
            for name in names:
                line = scontrol_node_line(sim, name, details)
                if oneliner:
                    ctx.p(line)
                else:
                    for ln in scontrol_multiline(line, SCONTROL_NODE_GROUPS, SCONTROL_NODE_KEYS):
                        ctx.p(ln)
                    ctx.p("")
            return 0
        if entity == "config":
            ctx.p(f"Configuration data as of {fmt_ts(sim.now)}")
            cfg = dict(SCONTROL_CONFIG_DEFAULTS)
            cfg.update({k: str(v) for k, v in sim.config.items()})
            cfg["NEXT_JOB_ID"] = str(sim.state["next_jobid"])
            cfg["BOOT_TIME"] = fmt_ts(sim.state.get("init_now", sim.now) - 86400 * 7)
            for key in SCONTROL_CONFIG_ORDER:
                if key in cfg:
                    ctx.p(f"{key:<23} = {cfg[key]}")
            for key in sorted(k for k in cfg if k not in SCONTROL_CONFIG_ORDER):
                ctx.p(f"{key:<23} = {cfg[key]}")
            ctx.p("")
            ctx.p("Cgroup Support Configuration:")
            for k, v in [("AllowedKmemSpace", "(null)"), ("AllowedRAMSpace", "100.0%"), ("AllowedSwapSpace", "0.0%"),
                         ("CgroupAutomount", "yes"), ("CgroupMountpoint", "/sys/fs/cgroup"), ("CgroupPlugin", "(null)"),
                         ("ConstrainCores", "yes"), ("ConstrainDevices", "yes"), ("ConstrainKmemSpace", "no"),
                         ("ConstrainRAMSpace", "yes"), ("ConstrainSwapSpace", "yes"), ("IgnoreSystemd", "no"),
                         ("IgnoreSystemdOnFailure", "no"), ("MaxKmemPercent", "100.0%"), ("MaxRAMPercent", "98.0%"),
                         ("MaxSwapPercent", "100.0%"), ("MemorySwappiness", "(null)"), ("MinKmemSpace", "30 MB"),
                         ("MinRAMSpace", "30 MB")]:
                ctx.p(f"{k:<23} = {v}")
            ctx.p("")
            ctx.p(f"Slurmctld(primary) at {sim.config.get('SlurmctldHost[0]', 'ctld')} is UP")
            return 0
        if entity in ("reservation", "reservations", "res"):
            resv = sim.state.get("reservations", [])
            if not resv:
                ctx.p("No reservations in the system")
                return 0
            for r in resv:
                ctx.p(" ".join(f"{k}={v}" for k, v in r.items()))
            return 0
        if entity == "hostnames":
            for n in hostlist_expand(ident or os.environ.get("SLURM_JOB_NODELIST", "")):
                ctx.p(n)
            return 0
        if entity == "hostlist":
            ctx.p(hostlist_compress((ident or "").split(",")))
            return 0
        if entity == "hostlistsorted":
            ctx.p(hostlist_compress(sorted((ident or "").split(","))))
            return 0
        ctx.error(f"Invalid entity: {rest[0]}")
        return 1
    if cmd == "ping":
        ctx.p(f"Slurmctld(primary) at {sim.config.get('SlurmctldHost[0]', 'ctld')} is UP")
        return 0
    if cmd == "version":
        ctx.p(f"slurm {SLURM_VERSION}")
        return 0
    if cmd in ("hold", "uhold", "release"):
        rc = 0
        for ident in rest:
            jobs = _find_job_arg(sim, ident)
            if not jobs:
                ctx.e(f"slurm_{'update' if cmd != 'release' else 'update'} error: Invalid job id specified")
                rc = 1
                continue
            for job in jobs:
                if cmd == "release":
                    if job.held:
                        job.held = False
                        job.hold_reason = ""
                        job.reason = "None"
                        job.priority = sim.compute_priority(job)
                        sim.event("release", job)
                else:
                    if not job.is_pending():
                        ctx.e("slurm_update error: Job is no longer pending execution")
                        rc = 1
                        continue
                    job.held = True
                    job.hold_reason = "JobHeldUser"
                    job.reason = "JobHeldUser"
                    job.priority = 0
                    sim.event("hold", job)
        return rc
    if cmd in ("requeue", "requeuehold"):
        rc = 0
        idents = [r for r in rest if not r.lower().startswith("state=") and r.lower() != "incomplete"]
        for ident in idents:
            jobs = _find_job_arg(sim, ident)
            if not jobs:
                ctx.e("slurm_requeue error: Invalid job id specified")
                rc = 1
                continue
            for job in jobs:
                if not job.requeue and job.state != "PENDING":
                    ctx.e(f"slurm_requeue error: Requested operation is presently disabled")
                    rc = 1
                    continue
                if job.state == "RUNNING":
                    job.signals.append({"time": sim.now, "signal": "TERM", "source": "requeue"})
                    sim.requeue_job(job, "REQUEUED")
                elif job.is_terminal() and not job.purged:
                    job.incarnations.append({"submit": job.submit, "eligible": job.eligible, "start": job.start,
                                             "end": job.end, "state": job.state, "nodes": list(job.nodes),
                                             "partition": job.partition, "flags": list(job.flags),
                                             "max_rss_k": job.max_rss_k, "total_cpu": job.total_cpu,
                                             "reason": job.last_pending_reason})
                    job.restarts += 1
                    job.state = "PENDING"
                    job.reason = "BeginTime"
                    job.nodes = []
                    job.start = job.end = None
                    job.submit = sim.now
                    job.eligible = job.begin = sim.now + int(sim.fake.get("requeue_delay", 120))
                    job.flags = ["StartRecieved"]
                    job.exit_code = job.exit_signal = 0
                    job.batch_state = ""
                    job.cancelled_by = None
                if cmd == "requeuehold":
                    job.held = True
                    job.hold_reason = "JobHeldUser"
                    job.reason = "JobHeldUser"
                    if any(r.lower() == "state=specialexit" for r in rest):
                        job.state = "SPECIAL_EXIT"
        return rc
    if cmd == "update":
        kv = {}
        for tok in rest:
            if "=" not in tok:
                ctx.error(f"Invalid input: {tok}")
                ctx.error("Request aborted")
                return 1
            k, v = tok.split("=", 1)
            kv[k.lower()] = v
        if "jobid" in kv or "jobname" in kv or "name" in kv and "jobid" not in kv:
            ident = kv.pop("jobid", None) or kv.pop("jobname", None) or kv.pop("name")
            jobs = _find_job_arg(sim, ident)
            if not jobs:
                ctx.e("slurm_update error: Invalid job id specified")
                return 1
            for job in jobs:
                for k, v in kv.items():
                    if k == "partition":
                        if not job.is_pending():
                            ctx.e("slurm_update error: Job is no longer pending execution")
                            return 1
                        names = v.split(",")
                        for n in names:
                            if sim.partition(n) is None:
                                ctx.e("slurm_update error: Invalid partition name specified")
                                return 1
                        job.partitions = names
                        job.partition = names[0]
                    elif k == "timelimit":
                        try:
                            if v.startswith("+") or v.startswith("-"):
                                delta = parse_time_limit(v[1:]) or 0
                                new = (job.time_limit or 0) + (delta if v[0] == "+" else -delta)
                            else:
                                new = parse_time_limit(v)
                        except ValueError:
                            ctx.e("slurm_update error: Invalid time limit specification")
                            return 1
                        if new is None or (job.time_limit is not None and new > job.time_limit):
                            ctx.e("slurm_update error: Access/permission denied")
                            return 1
                        job.time_limit = new
                    elif k == "comment":
                        job.comment = v
                    elif k == "priority":
                        if v == "0":
                            job.held = True
                            job.hold_reason = "JobHeldUser"
                            job.reason = "JobHeldUser"
                            job.priority = 0
                        else:
                            ctx.e("slurm_update error: Access/permission denied")
                            return 1
                    elif k == "nice":
                        n = int(v) if v else 100
                        if n < job.nice:
                            ctx.e("slurm_update error: Access/permission denied")
                            return 1
                        job.nice = n
                        job.priority = sim.compute_priority(job)
                    elif k in ("jobname", "name"):
                        job.name = v
                    elif k == "dependency":
                        job.dependency = "" if v.lower() in ("", "none") else v
                    elif k == "requeue":
                        job.requeue = v == "1"
                    elif k == "qos":
                        if sim.qos(v) is None:
                            ctx.e("slurm_update error: Invalid qos specification")
                            return 1
                        job.qos = v
                    elif k == "account":
                        if sim.user_account(v) is None:
                            ctx.e("slurm_update error: Invalid account or account/partition combination specified")
                            return 1
                        job.account = v
                    elif k in ("stdout", "stderr"):
                        setattr(job, k, v)
                    elif k in ("numnodes", "nodes"):
                        job.num_nodes = int(v.split("-")[0])
                    elif k == "numcpus":
                        job.cpus_per_node = max(1, int(v) // job.num_nodes)
                    elif k == "minmemorynode":
                        job.mem_mb = parse_mem(v)
                    elif k == "timemin":
                        job.time_min = parse_time_limit(v)
                    elif k in ("starttime", "eligibletime", "begintime"):
                        job.begin = parse_slurm_time(v, sim.now)
                        job.reason = "BeginTime"
                    elif k in ("mailtype",):
                        job.mail_type = v
                    elif k in ("mailuser",):
                        job.mail_user = v
                    elif k == "arraytaskthrottle":
                        for t in sim.jobs.values():
                            if t.array_job_id == (job.array_job_id or job.id):
                                t.array_task_throttle = int(v)
                    elif k in ("reqnodelist", "excnodelist", "features", "gres", "trespernode", "licenses",
                               "reservationname", "wckey", "deadline", "oversubscribe", "endtime", "userid",
                               "corespec", "contiguous", "taskspernode", "mincpusnode"):
                        if k in ("endtime", "oversubscribe", "corespec", "contiguous", "taskspernode"):
                            ctx.e("slurm_update error: Access/permission denied")
                            return 1
                    else:
                        ctx.error(f"Invalid input: {k}={v}")
                        ctx.error("Request aborted")
                        return 1
                sim.event("update", job, changes=kv)
            return 0
        if "nodename" in kv:
            names = hostlist_expand(kv.pop("nodename"))
            for n in names:
                if n not in sim.nodes:
                    ctx.e("slurm_update error: Invalid node name specified")
                    return 1
                st = kv.get("state", "").lower()
                if st in ("drain", "down", "resume", "idle", "undrain"):
                    sim.nodes[n].state = "idle" if st in ("resume", "idle", "undrain") else st
                    sim.nodes[n].reason = kv.get("reason", "") if st in ("drain", "down") else ""
            return 0
        if "partitionname" in kv:
            ctx.e("slurm_update error: Access/permission denied")
            return 1
        ctx.error("Invalid input: " + " ".join(rest))
        ctx.error("Request aborted")
        return 1
    if cmd == "write":
        if len(rest) >= 2 and rest[0].lower() == "batch_script":
            jobs = _find_job_arg(sim, rest[1])
            if not jobs:
                ctx.e("slurm_load_jobs error: Invalid job id specified")
                return 1
            job = jobs[0]
            target = rest[2] if len(rest) > 2 else f"slurm-{job.id}.sh"
            if target == "-":
                ctx.out.write(job.script)
                return 0
            try:
                with open(posix_to_native(target), "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(job.script)
            except OSError as e:
                ctx.error(f"unable to write to {target}: {e.strerror}")
                return 1
            ctx.p(f"batch script for job {job.id} written to {target}")
            return 0
        ctx.error("Invalid entity: " + " ".join(rest))
        return 1
    if cmd == "notify":
        if rest and _find_job_arg(sim, rest[0]):
            return 0
        ctx.e("slurm_notify error: Invalid job id specified")
        return 1
    if cmd == "top":
        return 0
    if cmd in ("suspend", "resume"):
        ctx.e("slurm_suspend error: Access/permission denied")
        return 1
    ctx.error(f"invalid keyword: {cmd}")
    return 1


# --------------------------------------------------------------------------------------------------
# scancel
# --------------------------------------------------------------------------------------------------
SCANCEL_SPEC: dict[str, tuple[str | None, object]] = {
    "user": ("u", True), "me": (None, False), "name": ("n", True), "jobname": (None, True), "state": ("t", True),
    "partition": ("p", True), "account": ("A", True), "qos": ("q", True), "reservation": ("R", True),
    "nodelist": ("w", True), "wckey": (None, True), "signal": ("s", True), "batch": ("b", False),
    "full": ("f", False), "quiet": ("Q", False), "verbose": ("v", False), "interactive": ("i", False),
    "hurry": ("H", False), "ctld": (None, False), "clusters": ("M", True), "sibling": (None, False),
    "help": (None, False), "version": ("V", False), "usage": (None, False), "cron": (None, False),
}

SIGNAL_NAMES = {"HUP": 1, "INT": 2, "QUIT": 3, "KILL": 9, "USR1": 10, "USR2": 12, "TERM": 15, "CONT": 18,
                "STOP": 19, "ALRM": 14}


def _cancel_job(sim: Sim, job: Job, uid: int) -> None:
    if job.is_running():
        job.signals.append({"time": sim.now, "signal": "TERM", "source": "scancel"})
        sim.finish_job(job, "CANCELLED", exit_code=0, exit_signal=0, batch_state="CANCELLED",
                       batch_exit=0, batch_signal=15, cancelled_by=uid)
    elif job.is_pending():
        sim.finish_job(job, "CANCELLED", cancelled_by=uid)
    sim.event("cancel", job, by=uid)


def cmd_scancel(ctx: Ctx, sim: Sim, argv: list[str]) -> int:
    opts, pos = parse_opts(argv, SCANCEL_SPEC, "scancel")
    o = dict(opts)
    if "help" in o or "usage" in o:
        ctx.p("Usage: scancel [OPTIONS] [job_id[_array_id][.step_id]]")
        return 0
    if "version" in o:
        ctx.p(f"slurm {SLURM_VERSION}")
        return 0
    quiet = "quiet" in o
    strict = int(sim.fake.get("scancel_strict", 0)) or bool(os.environ.get("FAKESLURM_SCANCEL_STRICT"))
    me = sim.current_user()
    uid = int(sim.user["uid"]) if me == sim.user["name"] else 0
    filters_used = any(k in o for k in ("user", "me", "name", "jobname", "state", "partition", "account",
                                         "qos", "reservation", "nodelist", "wckey"))
    if not pos and not filters_used:
        ctx.error("No job identification provided")
        return 1
    sig = None
    if o.get("signal"):
        s = o["signal"].upper().replace("SIG", "")
        sig = s if s in SIGNAL_NAMES else (s if s.isdigit() else None)
        if sig is None:
            ctx.error(f"Unknown job signal: {o['signal']}")
            return 1
    users = {me} if "me" in o else (set(o["user"].split(",")) if o.get("user") else None)
    names = set((o.get("name") or o.get("jobname") or "").split(",")) if (o.get("name") or o.get("jobname")) else None
    states = _parse_states(o["state"]) if o.get("state") else None
    parts = set(o["partition"].split(",")) if o.get("partition") else None
    accounts = set(o["account"].split(",")) if o.get("account") else None
    qoss = set(o["qos"].split(",")) if o.get("qos") else None
    nodes = set(hostlist_expand(o["nodelist"])) if o.get("nodelist") else None

    def matches(j: Job) -> bool:
        if users is not None and j.user not in users:
            return False
        if names is not None and j.name not in names:
            return False
        if states is not None and j.state not in states:
            return False
        if parts is not None and j.partition not in parts and not (set(j.partitions) & parts):
            return False
        if accounts is not None and j.account not in accounts:
            return False
        if qoss is not None and j.qos not in qoss:
            return False
        if nodes is not None and not (set(j.nodes) & nodes):
            return False
        return True

    rc = 0
    targets: list[Job] = []
    if pos:
        for tok in pos:
            base = tok.split(".")[0]
            found: list[Job] = []
            m = re.fullmatch(r"(\d+)_\[([^\]]+)\]", base)
            if m:
                ids = set()
                for r in m.group(2).split(","):
                    if "-" in r:
                        a, b = r.split("-")
                        ids.update(range(int(a), int(b) + 1))
                    else:
                        ids.add(int(r))
                found = [j for j in sim.jobs.values() if j.array_job_id == int(m.group(1)) and j.array_task_id in ids and not j.purged]
            elif re.fullmatch(r"\d+(_\d+)?", base):
                found = _find_job_arg(sim, base)
            else:
                ctx.error(f"Invalid job id {tok}")
                rc = 1
                continue
            found = [j for j in found if not j.is_terminal()]
            if not found:
                if strict and not quiet:
                    ctx.error(f"Kill job error on job id {base}: Invalid job id specified")
                    rc = 1
                continue
            targets.extend(j for j in found if matches(j))
    else:
        targets = [j for j in sim.jobs.values() if not j.purged and not j.is_terminal() and matches(j)]
        if users is None:
            targets = [j for j in targets if j.user == me]
    for job in targets:
        if job.user != me:
            if not quiet:
                ctx.error(f"Kill job error on job id {job.id}: Access/permission denied")
            rc = 1
            continue
        if sig is not None and sig not in ("KILL", "9"):
            if job.is_running():
                job.signals.append({"time": sim.now, "signal": sig, "source": "scancel",
                                    "batch": "batch" in o, "full": "full" in o})
                sim.event("signal", job, signal=sig)
            elif not quiet:
                ctx.error(f"Kill job error on job id {job.id}: Job is pending execution")
                rc = 1
            continue
        _cancel_job(sim, job, uid)
    return rc


# --------------------------------------------------------------------------------------------------
# sprio
# --------------------------------------------------------------------------------------------------
SPRIO_SPEC: dict[str, tuple[str | None, object]] = {
    "user": ("u", True), "jobs": ("j", True), "format": ("o", True), "Format": ("O", True), "noheader": ("h", False),
    "norm": ("n", False), "long": ("l", False), "weights": ("w", False), "partition": ("p", True),
    "sort": ("S", True), "verbose": ("v", False), "version": ("V", False), "help": (None, False),
    "clusters": ("M", True), "local": (None, False), "federation": (None, False), "sibling": (None, False),
}
SPRIO_HEADERS = {"i": "JOBID", "r": "PARTITION", "u": "USER", "Y": "PRIORITY", "S": "SITE", "A": "AGE", "a": "AGE",
                 "B": "ASSOC", "b": "ASSOC", "F": "FAIRSHARE", "f": "FAIRSHARE", "J": "JOBSIZE", "j": "JOBSIZE",
                 "P": "PARTITION", "p": "PARTITION", "Q": "QOS", "q": "QOS", "T": "TRES", "t": "TRES", "N": "NICE",
                 "n": "QOSNAME", "o": "ACCOUNT", "y": "PRIORITY"}


def cmd_sprio(ctx: Ctx, sim: Sim, argv: list[str]) -> int:
    opts, pos = parse_opts(argv, SPRIO_SPEC, "sprio")
    o = dict(opts)
    if "help" in o:
        ctx.p("Usage: sprio [OPTIONS]")
        return 0
    if "version" in o:
        ctx.p(f"slurm {SLURM_VERSION}")
        return 0
    w_age = sim.cfg_int("PriorityWeightAge", 10000)
    w_fs = sim.cfg_int("PriorityWeightFairShare", 1000000)
    w_qos = sim.cfg_int("PriorityWeightQOS", 5000000)
    if "weights" in o:
        ctx.p("          JOBID PARTITION   PRIORITY       SITE        AGE  FAIRSHARE    JOBSIZE  PARTITION        QOS")
        ctx.p(f"        Weights                          1 {w_age:>10} {w_fs:>10} {sim.cfg_int('PriorityWeightJobSize'):>10} "
              f"{sim.cfg_int('PriorityWeightPartition'):>10} {w_qos:>10}")
        return 0
    users = set(o["user"].split(",")) if o.get("user") else None
    ids = set(o["jobs"].split(",")) if o.get("jobs") else None
    parts = set(o["partition"].split(",")) if o.get("partition") else None
    jobs = [j for j in sim.jobs.values() if j.is_pending() and not j.purged]
    if users is not None:
        jobs = [j for j in jobs if j.user in users]
    if ids is not None:
        jobs = [j for j in jobs if str(j.id) in ids or j.id_str() in ids]
    if parts is not None:
        jobs = [j for j in jobs if j.partition in parts]
    jobs.sort(key=lambda j: (-j.priority, j.id))
    max_age = (parse_time_limit(str(sim.config.get("PriorityMaxAge", "7-00:00:00"))) or 10080) * 60
    max_q = max([int(v.get("priority", 0)) for v in sim.qos_table.values()] + [1])

    def val(job: Job, code: str) -> str:
        acct = sim.user_account(job.account) or {}
        fs = float(acct.get("fairshare", 0.5))
        age_n = min(1.0, max(0, sim.now - job.eligible) / max_age)
        q = sim.qos(job.qos) or {}
        qos_n = int(q.get("priority", 0)) / max_q if max_q else 0
        m = {"i": str(job.id), "r": job.partition, "u": job.user, "Y": str(job.priority), "y": str(job.priority),
             "S": "0", "A": str(int(w_age * age_n)), "a": f"{age_n:.7f}", "B": "0", "b": "0.0000000",
             "F": str(int(w_fs * fs)), "f": f"{fs:.7f}", "J": "0", "j": "0.0000000", "P": "0", "p": "0.0000000",
             "Q": str(int(w_qos * qos_n)), "q": f"{qos_n:.7f}", "T": "", "t": "", "N": str(job.nice),
             "n": job.qos, "o": job.account}
        return m.get(code, "")

    fmt = o.get("format") or "%.15i %.9r %.10Y %.10S %.10A %.10F %.10J %.10P %.10Q %20T"
    if "long" in o:
        fmt = "%.15i %.9r %.8u %.10Y %.10S %.10A %.10B %.10F %.10J %.10P %.10Q %.6N %20T"
    if "noheader" not in o:
        ctx.p(render_format_line(fmt, lambda c: SPRIO_HEADERS.get(c, "")))
    for job in jobs:
        ctx.p(render_format_line(fmt, lambda c, j=job: val(j, c)))
    return 0


# --------------------------------------------------------------------------------------------------
# sshare
# --------------------------------------------------------------------------------------------------
SSHARE_SPEC: dict[str, tuple[str | None, object]] = {
    "Users": ("U", False), "parsable2": ("P", False), "parsable": ("p", False), "noheader": ("n", False),
    "users": ("u", True), "accounts": ("A", True), "all": ("a", False), "long": ("l", False),
    "format": ("o", True), "partition": ("m", False), "verbose": ("v", False), "version": ("V", False),
    "help": ("h", False), "clusters": ("M", True), "json": (None, "optional"), "yaml": (None, "optional"),
}
SSHARE_FIELDS = ["Account", "User", "RawShares", "NormShares", "RawUsage", "NormUsage", "EffectvUsage",
                 "FairShare", "LevelFS", "GrpTRESMins", "GrpTRESRaw", "TRESRunMins", "Cluster", "ID", "Partition"]


def cmd_sshare(ctx: Ctx, sim: Sim, argv: list[str]) -> int:
    opts, pos = parse_opts(argv, SSHARE_SPEC, "sshare")
    o = dict(opts)
    if "help" in o:
        ctx.p("Usage: sshare [OPTIONS]")
        return 0
    if "version" in o:
        ctx.p(f"slurm {SLURM_VERSION}")
        return 0
    if o.get("format"):
        fields = []
        for f in o["format"].split(","):
            name = f.split("%")[0]
            canon = next((x for x in SSHARE_FIELDS if x.lower() == name.lower()), None)
            if canon is None:
                ctx.error(f"Invalid field requested: \"{name}\"")
                return 1
            fields.append(canon)
    elif "long" in o:
        fields = ["Account", "User", "RawShares", "NormShares", "RawUsage", "NormUsage", "EffectvUsage",
                  "FairShare", "LevelFS", "GrpTRESMins", "TRESRunMins"]
    else:
        fields = ["Account", "User", "RawShares", "NormShares", "RawUsage", "EffectvUsage", "FairShare"]
    parsable2 = "parsable2" in o
    parsable = "parsable" in o
    rows = []
    user = sim.user
    accounts = list(user["accounts"].items())
    if o.get("accounts"):
        accounts = [(a, d) for a, d in accounts if a in o["accounts"].split(",")]
    for acct_name, acct in accounts:
        if "Users" not in o and "users" not in o:
            rows.append({"Account": acct_name, "User": "", "RawShares": str(acct.get("raw_shares", 1)),
                         "NormShares": f"{acct.get('norm_shares', 0.0):.6f}", "RawUsage": str(acct.get("raw_usage", 0)),
                         "NormUsage": f"{acct.get('effective_usage', 0.0):.6f}",
                         "EffectvUsage": f"{acct.get('effective_usage', 0.0):.6f}",
                         "FairShare": "", "LevelFS": "", "GrpTRESMins": "", "GrpTRESRaw": "", "TRESRunMins": "",
                         "Cluster": sim.config.get("ClusterName", ""), "ID": "1", "Partition": ""})
        rows.append({"Account": acct_name, "User": user["name"], "RawShares": str(acct.get("raw_shares", 1)),
                     "NormShares": f"{acct.get('norm_shares', 0.0):.6f}", "RawUsage": str(acct.get("raw_usage", 0)),
                     "NormUsage": f"{acct.get('effective_usage', 0.0):.6f}",
                     "EffectvUsage": f"{acct.get('effective_usage', 0.0):.6f}",
                     "FairShare": f"{acct.get('fairshare', 0.5):.6f}", "LevelFS": f"{acct.get('fairshare', 0.5) * 4:.6f}",
                     "GrpTRESMins": "", "GrpTRESRaw": "cpu=0,mem=0,energy=0,node=0,billing=0,fs/disk=0,vmem=0,pages=0,gres/gpu=0",
                     "TRESRunMins": "cpu=0,mem=0,energy=0,node=0,billing=0,fs/disk=0,vmem=0,pages=0,gres/gpu=0",
                     "Cluster": sim.config.get("ClusterName", ""), "ID": "2", "Partition": ""})
    widths = {"Account": 20, "User": 10, "RawShares": 10, "NormShares": 11, "RawUsage": 11, "NormUsage": 11,
              "EffectvUsage": 12, "FairShare": 10, "LevelFS": 10, "GrpTRESMins": 30, "GrpTRESRaw": 30,
              "TRESRunMins": 30, "Cluster": 10, "ID": 6, "Partition": 10}

    def line(vals):
        if parsable2:
            return "|".join(vals)
        if parsable:
            return "|".join(vals) + "|"
        return " ".join(v.rjust(widths[f]) if f != "Account" else v.ljust(widths[f]) for f, v in zip(fields, vals))

    if "noheader" not in o:
        ctx.p(line(fields))
        if not parsable and not parsable2:
            ctx.p(" ".join("-" * widths[f] for f in fields))
    for r in rows:
        ctx.p(line([r[f] for f in fields]))
    return 0


# --------------------------------------------------------------------------------------------------
# sacctmgr
# --------------------------------------------------------------------------------------------------
ASSOC_FIELDS = {"cluster": "Cluster", "account": "Account", "user": "User", "partition": "Partition", "share": "Share",
                "fairshare": "Share", "priority": "Priority", "qos": "QOS", "defaultqos": "Def QOS",
                "maxjobs": "MaxJobs", "maxsubmit": "MaxSubmit", "maxsubmitjobs": "MaxSubmit", "maxwall": "MaxWall",
                "maxtresperjob": "MaxTRES", "maxtres": "MaxTRES", "maxtresminsperjob": "MaxTRESMins",
                "maxtrespernode": "MaxTRESPerNode", "grptres": "GrpTRES", "grptresmins": "GrpTRESMins",
                "grptresrunmins": "GrpTRESRunMins", "grpjobs": "GrpJobs", "grpsubmitjobs": "GrpSubmit",
                "grpsubmit": "GrpSubmit", "grpwall": "GrpWall", "id": "ID", "parentid": "Par ID",
                "parentname": "Par Name", "lft": "LFT", "rgt": "RGT", "maxjobsaccrue": "MaxJobsAccrue",
                "grpjobsaccrue": "GrpJobsAccrue", "minpriothresh": "MinPrioThres", "comment": "Comment"}
QOS_FIELDS = {"name": "Name", "priority": "Priority", "gracetime": "GraceTime", "preempt": "Preempt",
              "preemptmode": "PreemptMode", "preemptexempttime": "PreemptExemptTime", "flags": "Flags",
              "usagethres": "UsageThres", "usagefactor": "UsageFactor", "grptres": "GrpTRES",
              "grptresmins": "GrpTRESMins", "grptresrunmins": "GrpTRESRunMins", "grpjobs": "GrpJobs",
              "grpsubmit": "GrpSubmit", "grpsubmitjobs": "GrpSubmit", "grpwall": "GrpWall", "maxtres": "MaxTRES",
              "maxtresperjob": "MaxTRES", "maxtrespernode": "MaxTRESPerNode", "maxtresmins": "MaxTRESMins",
              "maxtresminsperjob": "MaxTRESMins", "maxwall": "MaxWall", "maxtrespu": "MaxTRESPU",
              "maxtresperuser": "MaxTRESPU", "maxjobspu": "MaxJobsPU", "maxjobsperuser": "MaxJobsPU",
              "maxsubmitpu": "MaxSubmitPU", "maxsubmitjobspu": "MaxSubmitPU", "maxsubmitjobsperuser": "MaxSubmitPU",
              "maxtrespa": "MaxTRESPA", "maxjobspa": "MaxJobsPA", "maxsubmitpa": "MaxSubmitPA", "mintres": "MinTRES",
              "limitfactor": "LimitFactor", "minpriothreshold": "MinPrioThres", "id": "ID", "description": "Descr",
              "maxjobsaccruepu": "MaxJobsAccruePU", "maxjobsaccruepa": "MaxJobsAccruePA", "grpjobsaccrue": "GrpJobsAccrue"}
USER_FIELDS = {"user": "User", "defaultaccount": "Def Acct", "defaultwckey": "Def WCKey", "adminlevel": "Admin",
               "account": "Account", "cluster": "Cluster", "partition": "Partition", "qos": "QOS",
               "defaultqos": "Def QOS", "share": "Share", "fairshare": "Share", "maxjobs": "MaxJobs",
               "maxsubmit": "MaxSubmit", "maxwall": "MaxWall", "coordinators": "Coord Accounts"}


def cmd_sacctmgr(ctx: Ctx, sim: Sim, argv: list[str]) -> int:
    noheader = parsable = parsable2 = False
    args: list[str] = []
    for a in argv:
        if a in ("-n", "--noheader"):
            noheader = True
        elif a in ("-p", "--parsable"):
            parsable = True
        elif a in ("-P", "--parsable2"):
            parsable2 = True
        elif a in ("-i", "--immediate", "-r", "--readonly", "-s", "--associations", "-v", "--verbose", "-Q", "--quiet"):
            pass
        elif a in ("-V", "--version"):
            ctx.p(f"slurm {SLURM_VERSION}")
            return 0
        elif a.startswith("-") and a[1:2].isalpha() and len(a) <= 3 and not args:
            # combined short flags like -nP
            for ch in a[1:]:
                if ch == "n":
                    noheader = True
                elif ch == "P":
                    parsable2 = True
                elif ch == "p":
                    parsable = True
        else:
            args.append(a)
    if not args:
        ctx.p("sacctmgr: interactive mode not supported by fakeslurm")
        return 1
    verb = args[0].lower()
    if verb not in ("show", "list"):
        if verb in ("add", "create", "modify", "delete", "remove", "load", "dump", "archive", "clear", "reconfigure", "shutdown"):
            ctx.e(" Access denied: you must be an admin to perform this operation")
            return 1
        ctx.e(f" Unknown command: {args[0]}")
        return 1
    if len(args) < 2:
        ctx.e(" Unknown option: show")
        return 1
    entity = args[1].lower().rstrip("s")
    kv: dict[str, str] = {}
    flags: set[str] = set()
    for tok in args[2:]:
        if tok.lower() == "where":
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.lower()] = v
        else:
            flags.add(tok.lower())
    user = sim.user
    cluster = sim.config.get("ClusterName", "fake")

    def emit(fields: list[str], header_map: dict[str, str], rows: list[dict[str, str]]) -> None:
        widths = {}
        for f in fields:
            widths[f] = max([len(header_map[f])] + [len(r.get(f, "")) for r in rows] + [4])
        if not noheader:
            if parsable or parsable2:
                ctx.p("|".join(header_map[f] for f in fields) + ("|" if parsable else ""))
            else:
                ctx.p(" ".join(header_map[f].ljust(widths[f]) for f in fields).rstrip())
                ctx.p(" ".join("-" * widths[f] for f in fields))
        for r in rows:
            if parsable or parsable2:
                ctx.p("|".join(r.get(f, "") for f in fields) + ("|" if parsable else ""))
            else:
                ctx.p(" ".join(r.get(f, "").ljust(widths[f]) for f in fields).rstrip())

    def parse_format(spec: str | None, table: dict[str, str], default: list[str]) -> list[str] | None:
        if not spec:
            return default
        out = []
        for f in spec.split(","):
            name = f.split("%")[0].strip().lower()
            if name not in table:
                ctx.e(f" Unknown field '{f}'")
                return None
            out.append(name)
        return out

    if entity in ("assoc", "association"):
        fields = parse_format(kv.get("format"), ASSOC_FIELDS,
                              ["cluster", "account", "user", "partition", "share", "priority", "grpjobs", "grptres",
                               "grpsubmit", "grpwall", "maxjobs", "maxtres", "maxtrespernode", "maxsubmit", "maxwall",
                               "qos", "defaultqos"])
        if fields is None:
            return 1
        rows = []
        want_user = kv.get("user")
        want_acct = kv.get("account")
        for acct_name, acct in user["accounts"].items():
            if want_acct and acct_name not in want_acct.split(","):
                continue
            base = {"cluster": cluster, "account": acct_name, "user": "", "partition": "",
                    "share": str(acct.get("raw_shares", 1)), "priority": "", "qos": ",".join(sorted(acct.get("qos", []))),
                    "defaultqos": acct.get("default_qos", ""), "maxjobs": "", "maxsubmit": "", "maxwall": "",
                    "maxtres": "", "maxtresmins": "", "maxtrespernode": "", "grptres": "", "grptresmins": "",
                    "grptresrunmins": "", "grpjobs": "", "grpsubmit": "", "grpwall": "", "id": "1", "parentid": "0",
                    "parentname": "root", "lft": "1", "rgt": "4", "maxjobsaccrue": "", "grpjobsaccrue": "",
                    "minpriothresh": "", "comment": ""}
            if not want_user:
                rows.append(dict(base))
            if not want_user or want_user == user["name"]:
                r = dict(base)
                r["user"] = user["name"]
                r["id"] = "2"
                r["parentid"] = "1"
                r["parentname"] = acct_name
                rows.append(r)
        emit(fields, ASSOC_FIELDS, rows)
        return 0
    if entity == "qo":
        fields = parse_format(kv.get("format"), QOS_FIELDS,
                              ["name", "priority", "gracetime", "preempt", "preemptexempttime", "preemptmode", "flags",
                               "usagethres", "usagefactor", "grptres", "grptresmins", "grptresrunmins", "grpjobs",
                               "grpsubmit", "grpwall", "maxtres", "maxtrespernode", "maxtresmins", "maxwall",
                               "maxtrespu", "maxjobspu", "maxsubmitpu", "maxtrespa", "maxjobspa", "maxsubmitpa",
                               "mintres"])
        if fields is None:
            return 1
        rows = []
        want = kv.get("name") or kv.get("names")
        for qname, q in sim.qos_table.items():
            if want and qname not in want.split(","):
                continue
            rows.append({"name": qname, "priority": str(q.get("priority", 0)), "gracetime": "00:00:00",
                         "preempt": "", "preemptmode": "cluster", "preemptexempttime": "", "flags": q.get("flags", ""),
                         "usagethres": "", "usagefactor": "1.000000", "grptres": q.get("grptres", ""), "grptresmins": "",
                         "grptresrunmins": "", "grpjobs": "", "grpsubmit": "", "grpwall": "", "maxtres": q.get("maxtres", ""),
                         "maxtrespernode": "", "maxtresmins": "", "maxwall": q.get("maxwall") or "",
                         "maxtrespu": q.get("maxtrespu", ""), "maxjobspu": str(q["maxjobspu"]) if q.get("maxjobspu") else "",
                         "maxsubmitpu": str(q["maxsubmitpu"]) if q.get("maxsubmitpu") else "", "maxtrespa": "",
                         "maxjobspa": "", "maxsubmitpa": "", "mintres": "", "limitfactor": "", "minpriothreshold": "",
                         "id": str(1 + list(sim.qos_table).index(qname)), "description": qname,
                         "maxjobsaccruepu": "", "maxjobsaccruepa": "", "grpjobsaccrue": ""})
        emit(fields, QOS_FIELDS, rows)
        return 0
    if entity == "user":
        withassoc = "withassoc" in flags
        default = ["user", "defaultaccount", "adminlevel"]
        if withassoc:
            default = ["user", "defaultaccount", "adminlevel", "cluster", "account", "partition", "share", "maxjobs",
                       "maxsubmit", "maxwall", "qos", "defaultqos"]
        fields = parse_format(kv.get("format"), USER_FIELDS, default)
        if fields is None:
            return 1
        want = kv.get("user") or kv.get("name") or (flags - {"withassoc", "withcoord", "withdeleted"})
        if isinstance(want, set):
            want = next(iter(want), None)
        if want and want != user["name"]:
            emit(fields, USER_FIELDS, [])
            return 0
        rows = []
        if withassoc:
            for acct_name, acct in user["accounts"].items():
                rows.append({"user": user["name"], "defaultaccount": user["default_account"], "defaultwckey": "",
                             "adminlevel": "None", "cluster": cluster, "account": acct_name, "partition": "",
                             "share": str(acct.get("raw_shares", 1)), "maxjobs": "", "maxsubmit": "", "maxwall": "",
                             "qos": ",".join(sorted(acct.get("qos", []))), "defaultqos": acct.get("default_qos", ""),
                             "coordinators": ""})
        else:
            rows.append({"user": user["name"], "defaultaccount": user["default_account"], "defaultwckey": "",
                         "adminlevel": "None", "coordinators": ""})
        emit(fields, USER_FIELDS, rows)
        return 0
    if entity == "account":
        fields = parse_format(kv.get("format"), {"account": "Account", "description": "Descr", "organization": "Org",
                                                 "cluster": "Cluster", "user": "User", "qos": "QOS"},
                              ["account", "description", "organization"])
        if fields is None:
            return 1
        rows = [{"account": a, "description": a, "organization": a, "cluster": cluster, "user": "",
                 "qos": ",".join(sorted(d.get("qos", [])))} for a, d in user["accounts"].items()]
        emit(fields, {"account": "Account", "description": "Descr", "organization": "Org", "cluster": "Cluster",
                      "user": "User", "qos": "QOS"}, rows)
        return 0
    if entity == "cluster":
        emit(["cluster", "controlhost", "controlport", "rpc", "share", "grptres", "maxtres", "maxwall", "qos", "defqos"],
             {"cluster": "Cluster", "controlhost": "ControlHost", "controlport": "ControlPort", "rpc": "RPC",
              "share": "Share", "grptres": "GrpTRES", "maxtres": "MaxTRES", "maxwall": "MaxWall", "qos": "QOS",
              "defqos": "Def QOS"},
             [{"cluster": cluster, "controlhost": "10.0.0.1", "controlport": "6817", "rpc": "9728", "share": "1",
               "grptres": "", "maxtres": "", "maxwall": "", "qos": "normal", "defqos": ""}])
        return 0
    if entity in ("tre", "tres"):
        emit(["type", "name", "id"], {"type": "Type", "name": "Name", "id": "ID"},
             [{"type": "cpu", "name": "", "id": "1"}, {"type": "mem", "name": "", "id": "2"},
              {"type": "energy", "name": "", "id": "3"}, {"type": "node", "name": "", "id": "4"},
              {"type": "billing", "name": "", "id": "5"}, {"type": "gres", "name": "gpu", "id": "1001"}])
        return 0
    ctx.e(f" Unknown entity: {args[1]}")
    return 1


# --------------------------------------------------------------------------------------------------
# seff
# --------------------------------------------------------------------------------------------------
def _fmt_bytes(nbytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(nbytes)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    return f"{v:.2f} {units[i]}"


def cmd_seff(ctx: Ctx, sim: Sim, argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("-")]
    if "-h" in argv or "--help" in argv or not args:
        ctx.p("Usage: seff [Options] <Jobid>")
        return 0 if args or "-h" in argv or "--help" in argv else 1
    ident = args[0].split(".")[0]
    jobs = _find_job_arg(sim, ident) or [j for j in sim.jobs.values() if str(j.id) == ident or j.id_str() == ident]
    if not jobs:
        ctx.e(f"Job not found.")
        return 1
    job = jobs[0]
    if job.start is None:
        ctx.p(f"Job ID: {job.id}")
        ctx.p(f"Cluster: {sim.config.get('ClusterName', 'fake')}")
        ctx.p(f"User/Group: {job.user}/{sim.user.get('group', 'users')}")
        ctx.p(f"State: {job.state}")
        ctx.p("Cores: " + str(job.total_cpus()))
        ctx.p("Efficiency not available for jobs in the PENDING state.")
        return 0
    elapsed = job.elapsed(sim.now)
    cpus = job.total_cpus()
    cpu_used = job.total_cpu if job.end is not None else elapsed * job.cpus_per_node * 0.6
    core_wall = elapsed * cpus
    mem_used = (job.max_rss_k or 0) * 1024
    mem_total = job.mem_mb * 1024 * 1024 * (len(job.nodes) or job.num_nodes)
    if job.is_running():
        ctx.p("WARNING: Efficiency statistics may be misleading for RUNNING jobs.")
    ctx.p(f"Job ID: {job.id}")
    ctx.p(f"Cluster: {sim.config.get('ClusterName', 'fake')}")
    ctx.p(f"User/Group: {job.user}/{sim.user.get('group', 'users')}")
    ctx.p(f"State: {job.state} (exit code {job.exit_code})")
    ctx.p(f"Nodes: {len(job.nodes) or job.num_nodes}")
    ctx.p(f"Cores per node: {job.cpus_per_node}")
    ctx.p(f"CPU Utilized: {fmt_hms(int(cpu_used))}")
    eff = (cpu_used / core_wall * 100) if core_wall else 0.0
    ctx.p(f"CPU Efficiency: {eff:.2f}% of {fmt_hms(int(core_wall))} core-walltime")
    ctx.p(f"Job Wall-clock time: {fmt_hms(elapsed)}")
    ctx.p(f"Memory Utilized: {_fmt_bytes(mem_used)}")
    meff = (mem_used / mem_total * 100) if mem_total else 0.0
    ctx.p(f"Memory Efficiency: {meff:.2f}% of {_fmt_bytes(mem_total)}")
    return 0


# --------------------------------------------------------------------------------------------------
# fakeslurm-ctl
# --------------------------------------------------------------------------------------------------
def _ctl_job(sim: Sim, ident: str) -> Job:
    jobs = _find_job_arg(sim, ident) or [j for j in sim.jobs.values() if str(j.id) == ident]
    if not jobs:
        raise CommandError(f"fakeslurm-ctl: no such job {ident}", 1)
    return jobs[0]


def cmd_ctl(ctx: Ctx, argv: list[str], sim: Sim | None, path: str) -> tuple[int, Sim | None]:
    if not argv:
        ctx.p("usage: fakeslurm-ctl init|advance|finish|preempt|oom|nodefail|cancel|drain|undrain|dump|set-config|run-script|now|tick ...")
        return 1, sim
    sub = argv[0]
    rest = argv[1:]
    if sub == "init":
        cluster = "trace"
        now = int(time.time())
        start_jobid = 100000
        i = 0
        while i < len(rest):
            a = rest[i]
            if a in ("--cluster", "-c"):
                cluster = rest[i + 1]
                i += 1
            elif a.startswith("--cluster="):
                cluster = a.split("=", 1)[1]
            elif a == "--now":
                now = parse_iso(rest[i + 1]) if not rest[i + 1].isdigit() else int(rest[i + 1])
                i += 1
            elif a.startswith("--now="):
                v = a.split("=", 1)[1]
                now = parse_iso(v) if not v.isdigit() else int(v)
            elif a == "--start-jobid":
                start_jobid = int(rest[i + 1])
                i += 1
            elif a.startswith("--start-jobid="):
                start_jobid = int(a.split("=", 1)[1])
            elif a in ("--wall-clock",):
                pass
            else:
                raise CommandError(f"fakeslurm-ctl init: unknown argument {a}", 2)
            i += 1
        state = new_state(cluster, now, start_jobid)
        if "--wall-clock" in rest:
            state["clock_mode"] = "wall"
        sim = Sim(state)
        ctx.p(f"initialised cluster {sim.config.get('ClusterName')} at {fmt_ts(sim.now)} (next job id {start_jobid})")
        return 0, sim
    if sim is None:
        raise CommandError("fakeslurm-ctl: state not initialised", 2)
    if sub == "now":
        ctx.p(fmt_ts(sim.now))
        ctx.p(str(sim.now))
        return 0, sim
    if sub == "tick":
        sim.tick()
        return 0, sim
    if sub == "advance":
        target = None
        i = 0
        while i < len(rest):
            a = rest[i]
            if a == "--seconds":
                target = sim.now + int(rest[i + 1])
                i += 1
            elif a.startswith("--seconds="):
                target = sim.now + int(a.split("=", 1)[1])
            elif a == "--to":
                v = rest[i + 1]
                target = parse_iso(v) if not v.isdigit() else int(v)
                i += 1
            elif a.startswith("--to="):
                v = a.split("=", 1)[1]
                target = parse_iso(v) if not v.isdigit() else int(v)
            elif re.fullmatch(r"\d+", a):
                target = sim.now + int(a)
            else:
                raise CommandError(f"fakeslurm-ctl advance: unknown argument {a}", 2)
            i += 1
        if target is None:
            raise CommandError("fakeslurm-ctl advance: need --seconds N or --to ISO", 2)
        # advance in small steps so start/finish events land at the right times
        while sim.now < target:
            nxt = target
            for j in sim.jobs.values():
                if j.is_running() and j.start is not None:
                    for cand in ([j.start + j.duration] if j.duration is not None else []) + \
                                ([j.start + j.limit_secs()] if j.limit_secs() is not None else []):
                        if sim.now < cand < nxt:
                            nxt = cand
                elif j.is_pending():
                    for cand in [j.begin, j.submit + int(sim.fake.get("sched_interval", 1))]:
                        if cand is not None and sim.now < cand < nxt:
                            nxt = cand
            sim.now = nxt
            sim.tick()
        ctx.p(fmt_ts(sim.now))
        return 0, sim
    if sub == "finish":
        job = _ctl_job(sim, rest[0])
        exit_code = 0
        i = 1
        while i < len(rest):
            if rest[i] == "--exit":
                exit_code = int(rest[i + 1])
                i += 1
            elif rest[i].startswith("--exit="):
                exit_code = int(rest[i].split("=", 1)[1])
            elif rest[i] == "--maxrss":
                job.max_rss_k = int(rest[i + 1])
                i += 1
            i += 1
        if not job.is_running():
            raise CommandError(f"fakeslurm-ctl finish: job {job.id} is not running ({job.state})", 1)
        sim.finish_job(job, "COMPLETED" if exit_code == 0 else "FAILED", exit_code=exit_code)
        return 0, sim
    if sub == "preempt":
        job = _ctl_job(sim, rest[0])
        if not job.is_running():
            raise CommandError(f"fakeslurm-ctl preempt: job {job.id} is not running ({job.state})", 1)
        sim.preempt_job(job)
        return 0, sim
    if sub == "oom":
        job = _ctl_job(sim, rest[0])
        if not job.is_running():
            raise CommandError(f"fakeslurm-ctl oom: job {job.id} is not running ({job.state})", 1)
        sim.finish_job(job, "OUT_OF_MEMORY", exit_code=0, exit_signal=125, batch_state="OUT_OF_MEMORY",
                       batch_exit=0, batch_signal=125)
        return 0, sim
    if sub == "nodefail":
        job = _ctl_job(sim, rest[0])
        if not job.is_running():
            raise CommandError(f"fakeslurm-ctl nodefail: job {job.id} is not running ({job.state})", 1)
        for n in job.nodes:
            sim.nodes[n].state = "down"
            sim.nodes[n].reason = "Not responding"
        if job.requeue:
            sim.requeue_job(job, "NODE_FAIL")
        else:
            sim.finish_job(job, "NODE_FAIL", exit_code=0, exit_signal=0, batch_state="CANCELLED",
                           batch_exit=0, batch_signal=15)
        return 0, sim
    if sub == "fail-node":
        for n in hostlist_expand(rest[0]):
            if n not in sim.nodes:
                raise CommandError(f"fakeslurm-ctl fail-node: unknown node {n}", 1)
            sim.nodes[n].state = "down"
            sim.nodes[n].reason = "Not responding"
            for job in list(sim.jobs.values()):
                if job.is_running() and n in job.nodes:
                    if job.requeue:
                        sim.requeue_job(job, "NODE_FAIL")
                    else:
                        sim.finish_job(job, "NODE_FAIL", batch_state="CANCELLED", batch_signal=15)
        return 0, sim
    if sub == "cancel":
        job = _ctl_job(sim, rest[0])
        _cancel_job(sim, job, int(sim.user["uid"]))
        return 0, sim
    if sub in ("drain", "undrain", "resume-node"):
        for n in hostlist_expand(rest[0]):
            if n not in sim.nodes:
                raise CommandError(f"fakeslurm-ctl {sub}: unknown node {n}", 1)
            sim.nodes[n].state = "drain" if sub == "drain" else "idle"
            sim.nodes[n].reason = (rest[1] if len(rest) > 1 else "maintenance") if sub == "drain" else ""
        return 0, sim
    if sub == "dump":
        ctx.p(json.dumps(sim.to_state(), indent=1))
        return 0, sim
    if sub == "set-config":
        key, value = rest[0], rest[1]
        if key.startswith("fake."):
            k = key[5:]
            sim.fake[k] = int(value) if re.fullmatch(r"-?\d+", value) else value
        else:
            sim.config[key] = int(value) if re.fullmatch(r"-?\d+", value) else value
        return 0, sim
    if sub == "events":
        since = int(rest[0]) if rest else 0
        for ev in sim.state.get("events", []):
            if ev["seq"] > since:
                ctx.p(json.dumps(ev))
        return 0, sim
    if sub == "run-script":
        job = _ctl_job(sim, rest[0])
        no_finish = "--no-finish" in rest
        if not job.is_running():
            raise CommandError(f"fakeslurm-ctl run-script: job {job.id} is not running ({job.state})", 1)
        rc = run_job_script(sim, job)
        ctx.p(f"job {job.id} script exited with {rc}")
        if not no_finish:
            sim.finish_job(job, "COMPLETED" if rc == 0 else "FAILED", exit_code=rc)
        return 0, sim
    raise CommandError(f"fakeslurm-ctl: unknown subcommand {sub}", 2)


def run_job_script(sim: Sim, job: Job) -> int:
    """Actually execute the stored batch script with bash, SLURM_* exported, output to StdOut/StdErr."""
    bash = os.environ.get("FAKESLURM_BASH") or "bash"
    env = dict(os.environ)
    if job.export.upper().startswith("NONE") or job.export.upper() == "NIL":
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "USER", "TMP", "TEMP", "SYSTEMROOT",
                                                            "FAKESLURM_STATE", "FAKESLURM_PYTHON")}
    env.update({k: v for k, v in job.env.items() if k != "FAKESLURM_EXPORT"})
    env.update({
        "SLURM_JOB_ID": str(job.id), "SLURM_JOBID": str(job.id), "SLURM_JOB_NAME": job.name,
        "SLURM_JOB_PARTITION": job.partition, "SLURM_JOB_ACCOUNT": job.account, "SLURM_JOB_QOS": job.qos,
        "SLURM_JOB_NODELIST": hostlist_compress(job.nodes), "SLURM_NODELIST": hostlist_compress(job.nodes),
        "SLURM_JOB_NUM_NODES": str(len(job.nodes) or job.num_nodes), "SLURM_NNODES": str(len(job.nodes) or job.num_nodes),
        "SLURM_SUBMIT_DIR": job.workdir, "SLURM_SUBMIT_HOST": "login01", "SLURM_CLUSTER_NAME": sim.config.get("ClusterName", "fake"),
        "SLURM_CPUS_ON_NODE": str(job.cpus_per_node), "SLURM_JOB_CPUS_PER_NODE": str(job.cpus_per_node),
        "SLURM_NTASKS": str(job.ntasks), "SLURM_NPROCS": str(job.ntasks), "SLURMD_NODENAME": job.nodes[0] if job.nodes else "",
        "SLURM_TASK_PID": str(os.getpid()), "SLURM_JOB_UID": str(job.uid), "SLURM_JOB_USER": job.user,
        "SLURM_MEM_PER_NODE": str(job.mem_mb),
    })
    if job.cpus_per_task > 1:
        env["SLURM_CPUS_PER_TASK"] = str(job.cpus_per_task)
    if job.restarts:
        env["SLURM_RESTART_COUNT"] = str(job.restarts)
    if job.array_job_id is not None:
        env["SLURM_ARRAY_JOB_ID"] = str(job.array_job_id)
        env["SLURM_ARRAY_TASK_ID"] = str(job.array_task_id)
    if job.gpu_count():
        env["SLURM_GPUS_ON_NODE"] = str(sum(g["count"] for g in job.gres))
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(sum(g["count"] for g in job.gres)))
    if job.comment:
        env["SLURM_JOB_COMMENT"] = job.comment
    workdir_native = posix_to_native(job.workdir) if job.workdir else os.getcwd()
    script_path = os.path.join(workdir_native, f".fakeslurm_job{job.id}.sh")
    with open(script_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(job.script)
    out_path = posix_to_native(sim.resolve_pattern(job, job.stdout))
    err_path = posix_to_native(sim.resolve_pattern(job, job.stderr))
    mode = "a" if job.open_mode == "append" else "w"
    if job.workdir:
        env["PWD"] = job.workdir
    try:
        with open(out_path, mode, encoding="utf-8") as out_fh:
            err_fh = out_fh if err_path == out_path else open(err_path, mode, encoding="utf-8")
            try:
                proc = subprocess.run([bash, script_path], cwd=workdir_native, env=env, stdout=out_fh,
                                      stderr=err_fh, stdin=subprocess.DEVNULL)
                rc = proc.returncode
            finally:
                if err_fh is not out_fh:
                    err_fh.close()
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
    return rc


# --------------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------------
COMMANDS = {
    "sbatch": cmd_sbatch, "squeue": cmd_squeue, "sacct": cmd_sacct, "sinfo": cmd_sinfo, "scontrol": cmd_scontrol,
    "scancel": cmd_scancel, "sprio": cmd_sprio, "sshare": cmd_sshare, "sacctmgr": cmd_sacctmgr, "seff": cmd_seff,
}
READ_ONLY = {"squeue", "sacct", "sinfo", "sprio", "sshare", "sacctmgr", "seff"}


def run(argv: list[str], stdin_text: str = "", cwd: str | None = None) -> tuple[int, str, str]:
    """In-process entry point: run(['squeue','-h']) -> (rc, stdout, stderr)."""
    if not argv:
        return 2, "", "fakeslurm: usage: fakeslurm <command> [args]\n"
    prog = os.path.basename(argv[0])
    if prog.endswith(".py"):
        prog = argv[1] if len(argv) > 1 else ""
        argv = argv[1:]
    args = argv[1:]
    ctx = Ctx(prog, stdin_text)
    old_cwd = None
    if cwd:
        old_cwd = os.getcwd()
        os.chdir(cwd)
    try:
        if prog == "fakeslurm-ctl":
            path = state_path()
            with StateLock(path):
                sim = None
                if os.path.exists(path) and (not args or args[0] != "init"):
                    with open(path, encoding="utf-8") as fh:
                        sim = Sim(json.load(fh))
                    if args and args[0] not in ("dump", "now", "events"):
                        sim.tick()
                rc, sim = cmd_ctl(ctx, args, sim, path)
                if sim is not None:
                    save_sim(sim, path)
            return rc, ctx.out.getvalue(), ctx.err.getvalue()
        if prog not in COMMANDS:
            return 2, "", f"fakeslurm: unknown command {prog!r}\n"
        if args and args[0] in ("--version", "-V"):
            return 0, f"slurm {SLURM_VERSION}\n", ""
        path = state_path()
        with StateLock(path):
            sim, path = load_sim()
            sim.tick()
            rc = COMMANDS[prog](ctx, sim, args)
            save_sim(sim, path)
        return rc, ctx.out.getvalue(), ctx.err.getvalue()
    except CommandError as e:
        return e.rc, ctx.out.getvalue(), ctx.err.getvalue() + e.msg + ("\n" if not e.msg.endswith("\n") else "")
    finally:
        if old_cwd:
            os.chdir(old_cwd)


def main() -> int:
    argv = sys.argv[1:]
    stdin_text = ""
    if argv and argv[0] == "sbatch":
        # sbatch reads the script from stdin only when no file argument / --wrap is given
        probe_opts, probe_pos = [], []
        try:
            probe_opts, probe_pos = parse_opts(argv[1:], SBATCH_SPEC, "sbatch")
        except CommandError:
            pass
        if not probe_pos and not any(k == "wrap" for k, _ in probe_opts) and \
                not any(k in ("help", "usage", "version") for k, _ in probe_opts):
            stdin_text = sys.stdin.read()
    rc, out, err = run(argv, stdin_text)
    # write bytes so Windows does not translate LF into CRLF
    sys.stdout.buffer.write(out.encode("utf-8", "replace"))
    sys.stdout.flush()
    sys.stderr.buffer.write(err.encode("utf-8", "replace"))
    sys.stderr.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
