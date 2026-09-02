"""Exact remote command strings for every SLURM interaction (design section 6.0-6.3).

Pure functions: no I/O, no cluster names. Every command begins with the section 6.0 preamble
(``export SLURM_TIME_FORMAT=%s LC_ALL=C``); composite probes are framed with ``::SECTION`` /
``::RC n`` / ``::END`` lines that ``slurm.parse.parse_sections`` understands. Remote paths are
quoted with :func:`path_quote`, which keeps ``$HOME``-style expansions intact (the default
``control_root`` is ``$HOME/.slurm-mcp``, design section 2.1) and shell-quotes everything else.
"""
from __future__ import annotations

import shlex
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

PREAMBLE = "export SLURM_TIME_FORMAT=%s LC_ALL=C"
MAX_CMD_BYTES = 8 * 1024          # section 6.2: a tick longer than 8 KB is split into two execs
SACCT_CHUNK = 100                 # section 6.2: chunks of 100 ids produce additional ::SACCT sections

# Field lists / formats shared with slurm.parse (section 6.1, 6.2).
TICK_SQUEUE_FORMAT = "%A|%i|%F|%K|%T|%P|%q|%S|%e|%V|%l|%M|%Q|%N|%b|%k|%o|%Z|%r"
TICK_RESTARTS_FORMAT = "JobID:0|,RestartCnt:0|,Requeue:0|"
TICK_SACCT_FIELDS: tuple[str, ...] = (
    "JobIDRaw", "JobID", "State", "ExitCode", "DerivedExitCode", "Partition", "QOS", "NodeList", "Submit",
    "Start", "End", "ElapsedRaw", "TimelimitRaw", "AllocTRES", "ReqTRES", "Reason", "WorkDir",
)
RECOVER_FIELDS: tuple[str, ...] = ("JobIDRaw", "Submit", "State", "WorkDir", "SubmitLine")
ENRICH_FIELDS: tuple[str, ...] = ("JobIDRaw", "JobID", "State", "ExitCode", "MaxRSS", "ReqMem", "ElapsedRaw",
                                  "AllocTRES")
BACKFILL_FIELDS: tuple[str, ...] = ("JobIDRaw", "Partition", "QOS", "ReqTRES", "Submit", "Start", "State")
SINFO_FORMAT = "%N|%R|%t|%c|%m|%G|%f"
SNAPSHOT_NODES_FORMAT = "%R|%t|%G|%C"
SNAPSHOT_DEMAND_O = "Partition:0|,tres-per-node:0|,tres-per-job:0|"
SNAPSHOT_MINE_O = "JobID:0|,Partition:0|,tres-per-node:0|,tres-per-job:0|,PriorityLong:0|,StartTime:0|,Reason:0|"
SNAPSHOT_DEMAND_FALLBACK = "%P|%b"
SNAPSHOT_MINE_FALLBACK = "%A|%P|%b|%Q|%S|%r"
ASSOC_FIELDS: tuple[str, ...] = ("Cluster", "Account", "Partition", "QOS", "DefaultQOS", "GrpTRES", "GrpTRESMins",
                                 "MaxJobs", "MaxSubmit", "MaxTRES", "MaxWall")
QOS_FIELDS: tuple[str, ...] = ("Name", "Priority", "GraceTime", "MaxWall", "MaxTRES", "MaxTRESPU", "MaxJobsPU",
                               "MaxSubmitPU", "GrpTRES", "Preempt", "PreemptMode", "Flags", "UsageFactor")
SSHARE_FIELDS: tuple[str, ...] = ("Account", "User", "FairShare", "GrpTRESMins", "GrpTRESRaw")
CONFIG_GREP = (
    "^(ClusterName|SLURM_VERSION|MinJobAge|MessageTimeout|PreemptMode|PreemptType|PreemptExemptTime|"
    "PreemptParameters|GraceTime|JobRequeue|KillWait|MaxArraySize|MaxJobCount|SchedulerParameters|DefMemPerCPU|"
    "DefMemPerNode|MaxMemPerCPU|MailProg|AccountingStorageEnforce|AccountingStoreFlags|EnforcePartLimits|"
    "PriorityWeight(Age|FairShare|QOS|Partition|JobSize)|BOOT_TIME) "
)
TOOLS: tuple[str, ...] = ("tar", "sacct", "squeue", "sbatch", "scontrol", "sinfo", "sacctmgr", "scancel", "srun",
                          "sshare", "sha256sum", "stat", "timeout", "setsid", "rsync", "jq", "seff", "flock")
TEST_ONLY_STRIP: tuple[str, ...] = ("--parsable", "--hold", "--comment")


# -- quoting helpers ------------------------------------------------------------------------------

def shell_quote(value: object) -> str:
    """POSIX shell quoting (``shlex.quote``) of one argument; ints are rendered bare."""
    text = str(value)
    return shlex.quote(text) if text else "''"


def path_quote(path: str) -> str:
    """Quote a remote path: ``$VAR`` references stay expandable (double quotes), else ``shlex.quote``.

    A path containing ``$`` is wrapped in double quotes with ``\\``, ``"`` and backticks escaped so
    ``$HOME/.slurm-mcp`` still expands on the cluster; any other path is shell-quoted verbatim.
    """
    if "$" in path:
        escaped = path.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
        return f'"{escaped}"'
    return shell_quote(path)


def join_args(args: Iterable[object]) -> str:
    """Space-join arguments, each shell-quoted."""
    return " ".join(shell_quote(a) for a in args)


def join_ids(ids: Iterable[object], sep: str = ",") -> str:
    """Join SLURM ids (ints, ``123_4`` strings) with ``sep``; ids are validated to be id-shaped."""
    out: list[str] = []
    for i in ids:
        text = str(i).strip()
        if not text or not all(c.isalnum() or c in "_.[]-," for c in text):
            raise ValueError(f"not a SLURM job id: {i!r}")
        out.append(text)
    return sep.join(out)


def chunks(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield consecutive slices of at most ``size`` items."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def command_bytes(command: str) -> int:
    """Length of a command in bytes as sent over the channel (UTF-8)."""
    return len(command.encode("utf-8"))


def needs_split(command: str, limit: int = MAX_CMD_BYTES) -> bool:
    """Section 6.2 split rule: True when the command exceeds ``limit`` bytes."""
    return command_bytes(command) > limit


def rc_echo() -> str:
    """``echo "::RC $?"`` framing line (section 6.0)."""
    return 'echo "::RC $?"'


def section_echo(name: str) -> str:
    """``echo '::<NAME>'`` framing line (section 6.0)."""
    return f"echo '::{name}'"


def _get(profile: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a profile-like dict or dataclass."""
    if isinstance(profile, Mapping):
        return profile.get(name, default)
    return getattr(profile, name, default)


def effective_control_root(profile: Any) -> str:
    """``profile.control_root`` or the section 2.1 default (``<remote_root>/.slurm-mcp`` / ``$HOME/.slurm-mcp``)."""
    control_root = _get(profile, "control_root")
    if control_root:
        return str(control_root)
    remote_root = _get(profile, "remote_root")
    return f"{remote_root}/.slurm-mcp" if remote_root else "$HOME/.slurm-mcp"


# -- 6.1 discovery --------------------------------------------------------------------------------

def discovery(profile: Any) -> str:
    """The bootstrap probe of design section 6.1, verbatim modulo profile placeholders.

    ``profile`` is a ``ClusterProfile`` or a dict with ``remote_root``, ``control_root``,
    ``quota_paths`` and ``balance_command``; ``::BALANCE`` is emitted only when a balance command
    is configured.
    """
    remote_root = _get(profile, "remote_root")
    control_root = effective_control_root(profile)
    quota_paths = list(_get(profile, "quota_paths", None) or [])
    balance_command = _get(profile, "balance_command")
    df_paths = ['"$HOME"']
    if remote_root:
        df_paths.append(path_quote(str(remote_root)))
    df_paths.append(path_quote(control_root))
    df_paths += ['"${PROJECT:-}"', '"${GROUP:-}"']
    df_paths += [path_quote(str(p)) for p in quota_paths]
    lines = [
        PREAMBLE,
        # id -Gn lists EVERY group; AllowGroups on a partition is usually a supplementary group (measured on
        # TRACE: primary "users", but cpuonly/cpuonly-debug/biosimmlab allow the supplementary "biosimmlab").
        "echo '::ENV'; echo \"$HOME|$USER|$(hostname -f)|${PROJECT:-}|${SCRATCH:-}|${LOCAL:-}|$(date +%s)|"
        "$(date +%z)|$(id -gn)|$(id -Gn)\"",
        "echo '::VERSION'; sinfo --version",
        f"echo '::CONFIG'; scontrol show config | grep -E '{CONFIG_GREP}'",
        f"echo '::PARTITIONS'; scontrol show partition -o; {rc_echo()}",
        f"echo '::SINFO'; sinfo -h -e -N -o '{SINFO_FORMAT}'; {rc_echo()}",
        "echo '::USER'; sacctmgr -nP show user \"$USER\" format=User,DefaultAccount",
        f"echo '::ASSOC'; sacctmgr -nP show assoc where user=\"$USER\" format={','.join(ASSOC_FIELDS)}; {rc_echo()}",
        f"echo '::QOS'; sacctmgr -nP show qos format={','.join(QOS_FIELDS)}; {rc_echo()}",
        f"echo '::SSHARE'; sshare -nP -U -u \"$USER\" -o {','.join(SSHARE_FIELDS)}; {rc_echo()}",
        "echo '::RESV'; scontrol -o show reservation 2>/dev/null",
        f"echo '::TOOLS'; for t in {' '.join(TOOLS)}; do printf '%s=' \"$t\"; "
        "command -v \"$t\" >/dev/null 2>&1 && echo 1 || echo 0; done",
        "echo '::CAP_O'; squeue --me -h -t all -O 'JobID:0|,RestartCnt:0|,tres-per-node:0|,tres-per-job:0|' "
        ">/dev/null 2>&1; echo \"rc=$?\"",
        f"echo '::DF'; for p in {' '.join(df_paths)}; do [ -n \"$p\" ] && [ -d \"$p\" ] && "
        "df -Pk \"$p\" 2>/dev/null | tail -n +2 | sed \"s|\\$| $p|\"; done",
    ]
    if balance_command:
        lines.append(f"echo '::BALANCE'; {balance_command} 2>/dev/null | head -40")
    lines.append(f"echo '::HELPER'; cat {path_quote(control_root + '/bin/VERSION')} 2>/dev/null")
    lines.append("echo '::END'")
    return "\n".join(lines)


def helper_deploy_check(control_root: str) -> str:
    """Read ``<control_root>/bin/VERSION`` (the deployed helper sha8; section 6.1 "Helper deploy")."""
    return f"{PREAMBLE}; cat {path_quote(control_root.rstrip('/') + '/bin/VERSION')} 2>/dev/null"


def backfill_history(user: str | None = None) -> str:
    """30-day wait-history back-fill query (section 6.1, last bullet)."""
    who = '"$USER"' if user is None else shell_quote(user)
    return (f"{PREAMBLE}; sacct -nP -X -u {who} -S now-30days -o {','.join(BACKFILL_FIELDS)}")


# -- 6.2 tick ---------------------------------------------------------------------------------------

def _files_lines(ctrl_dirs: Sequence[str]) -> str:
    if not ctrl_dirs:
        return section_echo("FILES")
    dirs = " ".join(path_quote(d) for d in ctrl_dirs)
    return (
        f"echo '::FILES'; for d in {dirs}; do for f in jobid status.json heartbeat; do "
        "[ -f \"$d/$f\" ] && printf '%s|%s|%s\\n' \"$d\" \"$f\" \"$(head -c 1000 \"$d/$f\" | tr '\\n\\r|' '   ')\"; "
        "done; [ -f \"$d/progress.json\" ] && printf '%s|progress.json|%s\\n' \"$d\" "
        "\"$(tail -c 1024 \"$d/progress.json\" | tail -n 1 | tr '|' ' ')\"; done"
    )


def _cmds_lines(rc_paths: Sequence[str]) -> str:
    if not rc_paths:
        return section_echo("CMDS")
    paths = " ".join(path_quote(p) for p in rc_paths)
    return f"echo '::CMDS'; for f in {paths}; do [ -f \"$f\" ] && printf '%s|%s\\n' \"$f\" \"$(cat \"$f\")\"; done"


def _sacct_lines(ids: Sequence[object]) -> list[str]:
    if not ids:
        return [section_echo("SACCT")]
    return [
        f"echo '::SACCT'; sacct -n -P -X -D -j {join_ids(chunk)} -o {','.join(TICK_SACCT_FIELDS)}; {rc_echo()}"
        for chunk in chunks(list(ids), SACCT_CHUNK)
    ]


def _enrich_lines(enrich_ids: Sequence[object], stdout_paths: Sequence[str]) -> str:
    line = f"echo '::ENRICH'; sacct -n -P -j {join_ids(enrich_ids)} -o {','.join(ENRICH_FIELDS)}; {rc_echo()}"
    if stdout_paths:
        paths = " ".join(path_quote(p) for p in stdout_paths)
        line += (f"; for f in {paths}; do printf '::L %s|%s\\n' \"$f\" "
                 "\"$(tail -n 1 \"$f\" 2>/dev/null | head -c 300 | tr '|' ' ')\"; done")
    return line


def tick(ids: Sequence[object], ctrl_dirs: Sequence[str] = (), rc_paths: Sequence[str] = (),
         recover: bool = False, enrich_ids: Sequence[object] = (), stdout_paths: Sequence[str] = (),
         caps: Mapping[str, Any] | None = None, limit: int = MAX_CMD_BYTES) -> list[str]:
    """The monitor tick of design section 6.2 as one or more exec strings.

    Sections always present: ``::NOW``, ``::SQUEUE``, ``::SACCT`` (header only when ``ids`` is empty),
    ``::FILES``, ``::CMDS``. Conditional: ``::RESTARTS`` (``caps["squeue_O_zero"]``), ``::RECOVER``
    (``recover=True``), ``::ENRICH`` (``enrich_ids`` non-empty; ``::L`` lines for ``stdout_paths``).
    ``ids`` beyond 100 produce additional ``::SACCT`` sections. When the whole command exceeds
    ``limit`` bytes the ``::FILES``/``::CMDS`` parts move to further execs (each within the limit
    where possible); the first exec always carries ``::NOW`` and the queue sections.
    """
    caps = caps or {}
    head = [PREAMBLE, 'echo "::NOW $(date +%s) $(hostname -s)"',
            f"echo '::SQUEUE'; squeue --me -h -r -t all -o '{TICK_SQUEUE_FORMAT}'; {rc_echo()}"]
    if caps.get("squeue_O_zero"):
        head.append(f"echo '::RESTARTS'; squeue --me -h -r -t all -O '{TICK_RESTARTS_FORMAT}'; {rc_echo()}")
    head += _sacct_lines(list(ids))
    tail: list[str] = []
    if recover:
        tail.append(f"echo '::RECOVER'; sacct -n -P -X -u \"$USER\" -S now-2hours -o {','.join(RECOVER_FIELDS)}; "
                    f"{rc_echo()}")
    if enrich_ids:
        tail.append(_enrich_lines(list(enrich_ids), list(stdout_paths)))
    end = "echo '::END'"
    full = "\n".join(head + [_files_lines(list(ctrl_dirs)), _cmds_lines(list(rc_paths))] + tail + [end])
    if not needs_split(full, limit):
        return [full]
    first = "\n".join(head + tail + [end])
    return [first] + _pack_file_sections(list(ctrl_dirs), list(rc_paths), limit)


def _pack_file_sections(ctrl_dirs: list[str], rc_paths: list[str], limit: int) -> list[str]:
    """Greedily pack ``::FILES``/``::CMDS`` items into execs of at most ``limit`` bytes (one item per exec
    at minimum)."""
    units: list[tuple[str, str]] = [("dir", d) for d in ctrl_dirs] + [("rc", p) for p in rc_paths]

    def render(batch: list[tuple[str, str]]) -> str:
        dirs = [v for k, v in batch if k == "dir"]
        paths = [v for k, v in batch if k == "rc"]
        return "\n".join([PREAMBLE, _files_lines(dirs), _cmds_lines(paths), "echo '::END'"])

    out: list[str] = []
    batch: list[tuple[str, str]] = []
    for unit in units:
        if batch and needs_split(render(batch + [unit]), limit):
            out.append(render(batch))
            batch = []
        batch.append(unit)
    if batch or not out:
        out.append(render(batch))
    return out


def recheck_pending() -> str:
    """Cheap per-id state recheck (``squeue --me -h -o '%A|%T'``; section 5.2 pending checks)."""
    return f"{PREAMBLE}; squeue --me -h -o '%A|%T'"


# -- 6.2 snapshot -----------------------------------------------------------------------------------

def snapshot(caps: Mapping[str, Any] | None = None) -> str:
    """The cluster snapshot of design section 6.2 (``-O`` demand views with ``caps.squeue_O_zero``,
    else the ``%P|%b`` fallback)."""
    caps = caps or {}
    if caps.get("squeue_O_zero"):
        pd = f"squeue -h -t PD -O '{SNAPSHOT_DEMAND_O}'"
        r = f"squeue -h -t R  -O '{SNAPSHOT_DEMAND_O}'"
        mine = f"squeue --me -h -t PD -O '{SNAPSHOT_MINE_O}'"
    else:
        pd = f"squeue -h -t PD -o '{SNAPSHOT_DEMAND_FALLBACK}'"
        r = f"squeue -h -t R  -o '{SNAPSHOT_DEMAND_FALLBACK}'"
        mine = f"squeue --me -h -t PD -o '{SNAPSHOT_MINE_FALLBACK}'"
    return "\n".join([
        PREAMBLE,
        f"echo '::NODES'; sinfo -h -e -N -o '{SNAPSHOT_NODES_FORMAT}'; {rc_echo()}",
        f"echo '::PD'; {pd} | sort | uniq -c; echo \"::RC ${{PIPESTATUS[0]}}\"",
        f"echo '::R';  {r} | sort | uniq -c; echo \"::RC ${{PIPESTATUS[0]}}\"",
        f"echo '::MINE'; {mine}",
        "echo '::RESV'; scontrol -o show reservation 2>/dev/null",
        "echo '::END'",
    ])


# -- 6.3 submit / estimate / control ---------------------------------------------------------------

def submit(workdir: str, bin_dir: str, ctrl_dir: str, token: str, args: Sequence[object],
           script_path: str | None = None) -> str:
    """``cd <workdir> && bash <bin>/submit.sh <ctrl_dir> <token> -- <args> <ctrl_dir>/job.sbatch`` (section 6.3)."""
    script = script_path or f"{ctrl_dir.rstrip('/')}/job.sbatch"
    parts = [f"bash {path_quote(bin_dir.rstrip('/') + '/submit.sh')}", path_quote(ctrl_dir), shell_quote(token), "--"]
    if args:
        parts.append(join_args(args))
    parts.append(path_quote(script))
    return f"{PREAMBLE}; cd {path_quote(workdir)} && {' '.join(parts)}"


def strip_for_test_only(args: Sequence[object]) -> list[str]:
    """Drop ``--parsable``, ``--hold`` and ``--comment[=...]`` (also the two-token form) from target args."""
    out: list[str] = []
    skip_next = False
    for a in args:
        text = str(a)
        if skip_next:
            skip_next = False
            continue
        if text in TEST_ONLY_STRIP:
            skip_next = text == "--comment"
            continue
        if text.startswith("--comment="):
            continue
        out.append(text)
    return out


def test_only(workdir: str, args: Sequence[object], script_path: str, section: str = "T1") -> str:
    """One ``sbatch --test-only`` estimate exec per target (section 6.3 "Estimate"); stderr merged (``2>&1``).

    ``--parsable``, ``--hold`` and ``--comment`` are stripped from ``args``; ``section`` names the frame
    (``T1``..``T4``, one per candidate target).
    """
    clean = strip_for_test_only(args)
    argtext = (join_args(clean) + " ") if clean else ""
    return (f"{PREAMBLE}; cd {path_quote(workdir)}\n"
            f"echo '::{section}'; sbatch --test-only {argtext}{path_quote(script_path)} 2>&1; {rc_echo()}; "
            "echo '::END'")


def cancel(ids: Sequence[object], signal: str | None = None, full: bool = False, batch: bool = False) -> str:
    """``scancel [--signal=<sig>] [--full] [--batch] <ids...>`` (section 6.3 "Control"); ids space-separated."""
    if not ids:
        raise ValueError("cancel() needs at least one id")
    opts = []
    if signal:
        opts.append(f"--signal={signal}")
    if full:
        opts.append("--full")
    if batch:
        opts.append("--batch")
    return f"{PREAMBLE}; scancel {' '.join(opts + [join_ids(ids, ' ')])}"


def _scontrol_ids(verb: str, ids: Sequence[object]) -> str:
    if not ids:
        raise ValueError(f"{verb}() needs at least one id")
    return f"{PREAMBLE}; scontrol {verb} {join_ids(ids)}"


def hold(ids: Sequence[object]) -> str:
    """``scontrol hold <ids>``."""
    return _scontrol_ids("hold", ids)


def release(ids: Sequence[object]) -> str:
    """``scontrol release <ids>``."""
    return _scontrol_ids("release", ids)


def requeue(ids: Sequence[object]) -> str:
    """``scontrol requeue <ids>``."""
    return _scontrol_ids("requeue", ids)


def update_dependency(job_id: object, deps: str | Sequence[str]) -> str:
    """``scontrol update JobId=<id> Dependency=<list>`` (section 5.2 step 11); an empty list clears it."""
    dep_text = deps if isinstance(deps, str) else ",".join(str(d) for d in deps)
    return f"{PREAMBLE}; scontrol update JobId={join_ids([job_id])} Dependency={shell_quote(dep_text) if dep_text else ''}"


def show_job(job_id: object) -> str:
    """``scontrol -o show job <id>``."""
    return f"{PREAMBLE}; scontrol -o show job {join_ids([job_id])}"


__all__ = [
    "PREAMBLE", "MAX_CMD_BYTES", "SACCT_CHUNK", "TICK_SQUEUE_FORMAT", "TICK_RESTARTS_FORMAT", "TICK_SACCT_FIELDS",
    "RECOVER_FIELDS", "ENRICH_FIELDS", "BACKFILL_FIELDS", "SINFO_FORMAT", "SNAPSHOT_NODES_FORMAT",
    "ASSOC_FIELDS", "QOS_FIELDS", "SSHARE_FIELDS", "CONFIG_GREP", "TOOLS",
    "shell_quote", "path_quote", "join_args", "join_ids", "chunks", "command_bytes", "needs_split",
    "rc_echo", "section_echo", "effective_control_root", "discovery", "helper_deploy_check", "backfill_history", "tick", "recheck_pending",
    "snapshot", "submit", "strip_for_test_only", "test_only", "cancel", "hold", "release", "requeue",
    "update_dependency", "show_job",
]
