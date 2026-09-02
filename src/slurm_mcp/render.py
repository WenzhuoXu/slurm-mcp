"""Rendering: user script -> spec fields, spec -> job.sbatch / user_body.sh / env.sh / alloc.sbatch, target -> sbatch
CLI args, QOS selection, requeue flag, output-pattern resolution and expansion.

Design sections implemented: 5.1 step 1 (directive stripping), 5.3 (requeue flag, injected options), 6.1 (QOS
selection), 6.3 (rendered files, target_args, output patterns, alloc.sbatch), 11d (auto-injection), changelog 13.3.
Imports only ``models``, ``errors``, ``textio`` and ``clock`` from the package (no cluster names anywhere).
"""
from __future__ import annotations

import posixpath
import re
import shlex
from datetime import datetime
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

from .clock import parse_duration
from .errors import SlurmMcpError
from .models import DEPENDS_RE, JobSpec, Target
from .textio import normalize_text

# --- constants ------------------------------------------------------------------------------------

DEFAULT_STDOUT_PATTERN = "slurm-%j.out"
DEFAULT_ARRAY_STDOUT_PATTERN = "slurm-%A_%a.out"
MAIL_TYPES = "END,FAIL,REQUEUE,TIME_LIMIT_90"
COMMENT_PREFIX = "slurm-mcp"
ALLOC_CLI_ARGS: tuple[str, ...] = ("--signal=B:TERM@60", "--no-requeue")
# sbatch(1) pattern letters that only the controller can expand (first node, node index, task id, step id).
LATE_PATTERN_LETTERS = frozenset("Nnts")
NO_VAL_ARRAY_INDEX = "4294967294"     # what sbatch substitutes for %a outside an array job
DEPENDENCY_TYPES = frozenset({"after", "afterok", "afterany", "afternotok", "aftercorr"})

# Short option -> long name (sbatch getopt table, only the managed subset; design section 5.1 step 1).
_SHORT: dict[str, str] = {
    "p": "partition", "q": "qos", "A": "account", "t": "time", "N": "nodes", "n": "ntasks", "c": "cpus-per-task",
    "C": "constraint", "J": "job-name", "o": "output", "e": "error", "a": "array", "d": "dependency", "D": "chdir",
    "x": "exclude", "H": "hold",
}
# Managed directives that take an argument (``--name=value`` or ``--name value`` / ``-X value`` / ``-Xvalue``).
MANAGED_WITH_ARG: frozenset[str] = frozenset({
    "partition", "qos", "account", "time", "nodes", "ntasks", "ntasks-per-node", "cpus-per-task", "gres", "gpus",
    "gpus-per-node", "mem", "mem-per-cpu", "mem-per-gpu", "constraint", "job-name", "output", "error", "open-mode",
    "signal", "array", "dependency", "kill-on-invalid-dep", "chdir", "mail-type", "mail-user", "comment", "exclude",
    "export",
})
# Managed directives without an argument (``--exclusive`` may carry ``=user|mcs``; ``-H/--hold`` never).
MANAGED_FLAGS: frozenset[str] = frozenset({"exclusive", "requeue", "no-requeue", "hold"})
MANAGED_DIRECTIVES: frozenset[str] = MANAGED_WITH_ARG | MANAGED_FLAGS

_MEM_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kKmMgGtT]?)[bB]?$")
_PATTERN_RE = re.compile(r"%(\d*)([%jJxuAaNnts])")
_SBATCH_RE = re.compile(r"^\s*#SBATCH(?:\s+(.*))?$")


def _invalid(message: str, fix: str | None = None) -> SlurmMcpError:
    return SlurmMcpError("E_INVALID_SPEC", message, fix)


# --- generic accessors (caps/assoc may be dicts or objects; keys tried in several spellings) --------

def _get(obj: Any, *keys: str, default: Any = None) -> Any:
    """First present value among ``keys`` on a mapping or an object (attribute access)."""
    if obj is None:
        return default
    for k in keys:
        if isinstance(obj, Mapping):
            if k in obj and obj[k] is not None:
                return obj[k]
        elif hasattr(obj, k):
            v = getattr(obj, k)
            if v is not None and not callable(v):
                return v
    return default


def _as_list(value: Any) -> list[str]:
    """``"a,b"`` / ``["a","b"]`` / None -> list of stripped non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    return [str(p).strip() for p in value if str(p).strip()]


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


# =================================================================================================
# 5.1 step 1: #SBATCH parsing and stripping
# =================================================================================================

class ParsedScript(NamedTuple):
    """Result of :func:`parse_sbatch` (unpacks as ``spec_fields, extra_sbatch, body, stripped_directives, warnings``)."""

    spec_fields: dict[str, Any]
    extra_sbatch: list[str]
    body: str
    stripped_directives: list[str]
    warnings: list[str]


def _strip_trailing_comment(text: str) -> str:
    """Drop an unquoted ``# ...`` tail from a directive argument string (sbatch stops at it)."""
    out: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
            out.append(ch)
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "#" and (i == 0 or text[i - 1].isspace()):
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip()


class _Opt(NamedTuple):
    name: str            # long name without leading dashes (unknown options keep their spelling)
    value: str | None
    managed: bool
    original: str        # the tokens as written, joined by a space


def _tokenize_directive(arg_text: str) -> list[_Opt]:
    """Split one ``#SBATCH`` argument string into options (managed or not)."""
    try:
        tokens = shlex.split(arg_text, posix=True)
    except ValueError:
        tokens = arg_text.split()
    opts: list[_Opt] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if tok.startswith("--") and len(tok) > 2:
            name, eq, val = tok[2:].partition("=")
            if name in MANAGED_WITH_ARG:
                if eq:
                    opts.append(_Opt(name, val, True, tok))
                elif nxt is not None:
                    opts.append(_Opt(name, nxt, True, f"{tok} {nxt}"))
                    i += 1
                else:
                    opts.append(_Opt(name, "", True, tok))
            elif name in MANAGED_FLAGS:
                opts.append(_Opt(name, val if eq else None, True, tok))
            else:
                if not eq and nxt is not None and not nxt.startswith("-"):
                    opts.append(_Opt(name, nxt, False, f"{tok} {nxt}"))
                    i += 1
                else:
                    opts.append(_Opt(name, val if eq else None, False, tok))
        elif tok.startswith("-") and len(tok) > 1 and not tok.startswith("--"):
            letter, rest = tok[1], tok[2:]
            name = _SHORT.get(letter)
            if name in MANAGED_FLAGS:
                opts.append(_Opt(name, None, True, tok))
            elif name is not None:
                if rest:
                    opts.append(_Opt(name, rest, True, tok))
                elif nxt is not None:
                    opts.append(_Opt(name, nxt, True, f"{tok} {nxt}"))
                    i += 1
                else:
                    opts.append(_Opt(name, "", True, tok))
            else:
                if not rest and nxt is not None and not nxt.startswith("-"):
                    opts.append(_Opt(tok, nxt, False, f"{tok} {nxt}"))
                    i += 1
                else:
                    opts.append(_Opt(tok, rest or None, False, tok))
        else:
            opts.append(_Opt(tok, None, False, tok))
        i += 1
    return opts


def _parse_gpu_spec(value: str, what: str) -> tuple[str | None, int]:
    """``[type:]N`` (``--gpus``/``--gpus-per-node``) -> (type, count); ``a40`` -> (a40, 1); ``2`` -> (None, 2)."""
    parts = [p for p in value.strip().split(":") if p != ""]
    if not parts:
        return None, 1
    if len(parts) == 1:
        return (None, int(parts[0])) if parts[0].isdigit() else (parts[0], 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    raise _invalid(f"script {what}={value!r} is not [type:]N")


def _parse_gres(value: str) -> tuple[str | None, int | None, list[str]]:
    """``--gres=gpu[:type][:N][,other]`` -> (gpu type, gpu count, non-gpu entries kept verbatim)."""
    gpu_type: str | None = None
    gpu_count: int | None = None
    others: list[str] = []
    for entry in (e.strip() for e in value.split(",") if e.strip()):
        core = re.sub(r"\(.*\)$", "", entry)
        fields = core.split(":")
        if fields[0] != "gpu":
            others.append(entry)
            continue
        rest = fields[1:]
        if not rest:
            gpu_type, gpu_count = None, 1
        elif len(rest) == 1:
            if rest[0].isdigit():
                gpu_type, gpu_count = None, int(rest[0])
            else:
                gpu_type, gpu_count = rest[0], 1
        elif len(rest) == 2 and rest[1].isdigit():
            gpu_type, gpu_count = rest[0], int(rest[1])
        else:
            raise _invalid(f"script --gres={value!r}: gpu entry must be gpu[:type][:N]")
    return gpu_type, gpu_count, others


def _mem_to_mb(value: str) -> float | None:
    m = _MEM_RE.match(value.strip())
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).upper() or "M"
    return num * {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]


def _mb_to_str(mb: float) -> str:
    mb_int = int(round(mb))
    if mb_int >= 1024 and mb_int % 1024 == 0:
        return f"{mb_int // 1024}G"
    return f"{mb_int}M"


def _parse_signal(value: str) -> tuple[str | None, int | None, bool]:
    """``[R:][B:]<sig>[@<sec>]`` -> (signal name without SIG or None if numeric-unknown, seconds, batch_only)."""
    sig_part, _, sec_part = value.partition("@")
    fields = sig_part.split(":")
    flags = {f.upper() for f in fields[:-1]}
    sig = fields[-1].strip()
    if sig.upper().startswith("SIG"):
        sig = sig[3:]
    seconds = int(sec_part) if sec_part.strip().isdigit() else None
    return (sig.upper() or None), seconds, ("B" in flags)


def _parse_dependency(value: str, cluster: str | None) -> list[str]:
    """SLURM ``-d`` syntax -> ``depends_on`` entries; raw numeric ids are refused (design section 5.1 step 1)."""
    out: list[str] = []
    for piece in re.split(r"[,?]", value.strip()):
        piece = piece.strip()
        if not piece:
            continue
        if piece == "singleton":
            out.append("singleton")
            continue
        typ, _, ids = piece.partition(":")
        typ = typ.strip()
        if typ not in DEPENDENCY_TYPES:
            raise SlurmMcpError("E_DEPENDENCY", f"unsupported dependency type {typ!r} in script -d {value!r}")
        for raw in (x for x in ids.split(":") if x):
            ident = raw.split("+", 1)[0]
            if re.fullmatch(r"j\d+", ident):
                out.append(f"{typ}:{ident}")
            else:
                where = f"{cluster or '<cluster>'}:{ident}"
                raise SlurmMcpError("E_DEPENDENCY", f"dependency on raw SLURM id {ident}",
                                    f"use the tracked handle or adopt it with job_status(['{where}'])")
    return out


def parse_sbatch(script_text: str, *, cluster: str | None = None) -> ParsedScript:
    """Convert every server-managed ``#SBATCH`` directive of a user script into spec fields and strip it
    (design section 5.1 step 1, section 6.3, changelog item 3).

    Only the *leading* directive block is parsed (sbatch ignores directives after the first command);
    the shebang is dropped; the remainder becomes the ``user_body.sh`` text. Unknown directives are kept
    verbatim in ``extra_sbatch``; everything stripped is echoed in ``stripped_directives``. Text is normalised
    first (CRLF -> LF with the ``crlf_normalized`` warning, BOM stripped, NUL refused).
    """
    text, warnings = normalize_text(script_text)
    lines = text.split("\n")
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    header_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            header_end = i + 1
            continue
        break
    header, body_lines = lines[:header_end], lines[header_end:]
    body = "\n".join(body_lines).strip("\n")
    if not body.strip():
        raise SlurmMcpError("E_SCRIPT", "script has no commands after its #SBATCH block")
    body += "\n"

    raw: dict[str, str | None] = {}
    extra: list[str] = []
    stripped_directives: list[str] = []
    for line in header:
        m = _SBATCH_RE.match(line)
        if not m:
            continue
        arg_text = _strip_trailing_comment(m.group(1) or "")
        if not arg_text:
            continue
        opts = _tokenize_directive(arg_text)
        if not any(o.managed for o in opts):
            extra.append(arg_text)
            continue
        for o in opts:
            if not o.managed:
                extra.append(o.original)
                continue
            stripped_directives.append(o.original)
            if o.name in raw and o.name not in ("gres",):
                warnings.append(f"script repeats --{o.name}; the last value ({o.value!r}) wins")
            raw[o.name] = o.value

    fields: dict[str, Any] = {}
    res: dict[str, Any] = {}

    def _int(name: str, value: str) -> int:
        first = re.match(r"\d+", value.strip())
        if not first:
            raise _invalid(f"script --{name}={value!r} is not an integer")
        return int(first.group(0))

    nodes = _int("nodes", raw["nodes"]) if raw.get("nodes") else None
    if nodes is not None:
        res["nodes"] = max(nodes, 1)
    nodes_eff = res.get("nodes", 1)

    for key in ("partition", "qos", "account"):
        if raw.get(key):
            fields[key] = raw[key].strip()
    if raw.get("time"):
        if parse_duration(raw["time"].strip()) is None:
            raise _invalid(f"script -t {raw['time']!r} is not a SLURM time")
        res["time"] = raw["time"].strip()
    if raw.get("cpus-per-task"):
        res["cpus"] = _int("cpus-per-task", raw["cpus-per-task"])
    if raw.get("ntasks-per-node"):
        res["tasks"] = _int("ntasks-per-node", raw["ntasks-per-node"])
    elif raw.get("ntasks"):
        n = _int("ntasks", raw["ntasks"])
        if n % nodes_eff:
            raise _invalid(f"script --ntasks={n} is not divisible by --nodes={nodes_eff}",
                           "use --ntasks-per-node in the script or set resources.tasks explicitly")
        res["tasks"] = n // nodes_eff
    if raw.get("constraint"):
        res["constraint"] = raw["constraint"].strip()
    if "exclusive" in raw:
        res["exclusive"] = True

    gpu_type: str | None = None
    gpu_count: int | None = None
    if raw.get("gres") is not None:
        t, c, others = _parse_gres(raw["gres"] or "")
        if others:
            extra.append("--gres=" + ",".join(others))
            warnings.append("script --gres non-gpu entries kept in extra_sbatch: " + ",".join(others)
                            + " (overridden by the typed --gres the server sends when gpus > 0)")
        if c is not None:
            gpu_type, gpu_count = t, c
    if raw.get("gpus-per-node") is not None:
        gpu_type, gpu_count = _parse_gpu_spec(raw["gpus-per-node"] or "", "--gpus-per-node")
    if raw.get("gpus") is not None:
        t, per_job = _parse_gpu_spec(raw["gpus"] or "", "--gpus")
        if per_job % nodes_eff:
            raise _invalid(f"script --gpus={per_job} is per job and not divisible by --nodes={nodes_eff}",
                           "use --gpus-per-node or set resources.gpus (per node) explicitly")
        gpu_type, gpu_count = t, per_job // nodes_eff
    if gpu_count is not None:
        res["gpus"] = gpu_count
        if gpu_type and gpu_count > 0:
            res["gpu_types"] = [gpu_type]

    if raw.get("mem"):
        res["mem"] = raw["mem"].strip()
    elif raw.get("mem-per-cpu"):
        mb = _mem_to_mb(raw["mem-per-cpu"])
        cpus_per_node = (res.get("cpus") or 1) * (res.get("tasks") or 1)
        if mb is None:
            res["mem"] = raw["mem-per-cpu"].strip()
            warnings.append(f"script --mem-per-cpu={raw['mem-per-cpu']} taken as resources.mem verbatim")
        else:
            res["mem"] = _mb_to_str(mb * cpus_per_node)
            warnings.append(f"script --mem-per-cpu={raw['mem-per-cpu']} x {cpus_per_node} cpus/node -> "
                            f"resources.mem={res['mem']}")
    elif raw.get("mem-per-gpu"):
        mb = _mem_to_mb(raw["mem-per-gpu"])
        if mb is None or not res.get("gpus"):
            res["mem"] = raw["mem-per-gpu"].strip()
            warnings.append(f"script --mem-per-gpu={raw['mem-per-gpu']} taken as resources.mem verbatim")
        else:
            res["mem"] = _mb_to_str(mb * res["gpus"])
            warnings.append(f"script --mem-per-gpu={raw['mem-per-gpu']} x {res['gpus']} gpus/node -> "
                            f"resources.mem={res['mem']}")

    if raw.get("job-name"):
        fields["name"] = raw["job-name"].strip()
    if raw.get("output"):
        fields["stdout"] = raw["output"].strip()
    if raw.get("error"):
        fields["stderr"] = raw["error"].strip()
    if raw.get("chdir"):
        fields["workdir"] = raw["chdir"].strip()
    if "requeue" in raw:
        fields["requeue"] = True
    if "no-requeue" in raw:
        fields["requeue"] = False
    if raw.get("open-mode") and raw["open-mode"].strip().lower() != "append":
        warnings.append(f"script --open-mode={raw['open-mode']} ignored; the server always renders --open-mode=append")
    if raw.get("signal"):
        sig, seconds, batch_only = _parse_signal(raw["signal"])
        if seconds is not None:
            fields["grace_s"] = seconds
        if sig and not batch_only and not sig.isdigit():
            fields["child_signal"] = sig
            warnings.append(f"script --signal={raw['signal']} -> child_signal={sig} (the payload declared it handles it)")
        else:
            warnings.append(f"script --signal={raw['signal']} stripped; the server sends --signal=B:USR1@grace_s")
    if raw.get("array"):
        arr, _, par = raw["array"].strip().partition("%")
        fields["array"] = arr
        if par.strip().isdigit():
            fields["array_parallel"] = int(par)
    if raw.get("dependency"):
        fields["depends_on"] = _parse_dependency(raw["dependency"], cluster)
    if raw.get("mail-type") or raw.get("mail-user"):
        warnings.append("script --mail-type/--mail-user stripped; configure(notify={'email': ...}) sends SLURM mail")
    if raw.get("comment") is not None:
        warnings.append("script --comment stripped; the server sets --comment=slurm-mcp:<handle>:<attempt>:<token>")
    if raw.get("exclude"):
        warnings.append(f"script --exclude={raw['exclude']} stripped; node exclusion is per attempt (after NODE_FAIL)")
    if "hold" in raw:
        warnings.append("script --hold stripped; pass hold=True to submit_job or use job_control(..., 'hold')")
    if raw.get("export") is not None:
        warnings.append(f"script --export={raw['export']} stripped; the server never sets --export (default ALL)")
    for entry in ("dependency", "kill-on-invalid-dep"):
        raw.pop(entry, None)

    if res:
        fields["resources"] = res
    return ParsedScript(fields, extra, body, stripped_directives, warnings)


# =================================================================================================
# merge parsed fields with an explicit spec
# =================================================================================================

def _explicit_keys(spec: JobSpec | Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Top-level and ``resources`` keys the caller set explicitly (dict keys or pydantic ``model_fields_set``)."""
    if isinstance(spec, JobSpec):
        top = set(spec.model_fields_set)
        res = set(spec.resources.model_fields_set)
        return top, res
    top = {k for k, v in spec.items() if v is not None}
    r = spec.get("resources") or {}
    res = {k for k, v in r.items() if v is not None} if isinstance(r, Mapping) else set(getattr(r, "model_fields_set", ()))
    return top, res


def merge_spec(spec: JobSpec | Mapping[str, Any], parsed: ParsedScript) -> tuple[JobSpec, list[str]]:
    """Merge script-derived fields into a spec: explicit spec fields win, each override is a warning
    (design section 5.1 step 1). ``extra_sbatch`` lists are concatenated. Returns ``(JobSpec, warnings)``.

    ``spec`` may be the raw tool input dict (recommended: then ``resources.time`` may come from the script)
    or a validated ``JobSpec`` (explicitness from ``model_fields_set``).
    """
    top_set, res_set = _explicit_keys(spec)
    base: dict[str, Any] = spec.model_dump() if isinstance(spec, JobSpec) else dict(spec)
    base_res: dict[str, Any] = dict(base.get("resources") or {}) if isinstance(base.get("resources"), Mapping) \
        else (base["resources"].model_dump() if base.get("resources") is not None else {})
    warnings = list(parsed.warnings)
    for key, value in parsed.spec_fields.items():
        if key == "resources":
            for rk, rv in value.items():
                if rk in res_set:
                    if base_res.get(rk) != rv:
                        warnings.append(f"spec.resources.{rk}={base_res.get(rk)!r} overrides script value {rv!r}")
                else:
                    base_res[rk] = rv
            continue
        if key in top_set:
            if base.get(key) != value:
                warnings.append(f"spec.{key}={base.get(key)!r} overrides script value {value!r}")
        else:
            base[key] = value
    if base_res:
        base["resources"] = base_res
    existing = [x for x in (base.get("extra_sbatch") or [])]
    for line in parsed.extra_sbatch:
        if line not in existing:
            existing.append(line)
    base["extra_sbatch"] = existing
    merged = JobSpec.parse(base)
    for w in merged.warnings:
        if w not in warnings:
            warnings.append(w)
    return merged, warnings


# =================================================================================================
# 6.3 rendered files
# =================================================================================================

def _q(path: str) -> str:
    return shlex.quote(path)


def _header_lines(spec: JobSpec, job_name: str) -> list[str]:
    r = spec.resources
    lines = ["#!/bin/bash", f"#SBATCH -J {job_name}", f"#SBATCH -N {r.nodes}"]
    if r.tasks:
        lines.append(f"#SBATCH --ntasks-per-node={r.tasks}")
    if r.cpus:
        lines.append(f"#SBATCH --cpus-per-task={r.cpus}")
    if r.exclusive:
        lines.append("#SBATCH --exclusive")
    if r.constraint:
        lines.append(f"#SBATCH -C {r.constraint}")
    return lines


def _tracking_line(handle: str, attempt_no: int, token: str, rendered_at: datetime | str | None) -> str:
    if rendered_at is None:
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    elif isinstance(rendered_at, datetime):
        stamp = rendered_at.isoformat(timespec="seconds")
    else:
        stamp = str(rendered_at)
    return f"# slurm-mcp handle={handle} attempt={attempt_no} token={token} rendered={stamp}"


def _extra_lines(spec: JobSpec) -> list[str]:
    out: list[str] = []
    for line in spec.extra_sbatch:
        text = line.strip()
        if text.startswith("#SBATCH"):
            text = text[len("#SBATCH"):].strip()
        if text:
            out.append(f"#SBATCH {text}")
    return out


def render_job_sbatch(spec: JobSpec, handle: str, attempt_no: int, token: str, ctrl_dir: str, workdir: str,
                      control_root: str, sha8: str, *, rendered_at: datetime | str | None = None) -> str:
    """``job.sbatch`` exactly as design section 6.3: nothing target-specific, no ``-t``/``--mem``/``-o``/``-e``.

    ``wrap=False`` specs exec ``bash user_body.sh`` directly (sacct-only tracking, section 11a).
    """
    lines = _header_lines(spec, spec.name)
    lines.append("#SBATCH --open-mode=append")
    lines += _extra_lines(spec)
    lines.append(_tracking_line(handle, attempt_no, token, rendered_at))
    lines.append(f"source {_q(ctrl_dir + '/env.sh')}")
    lines.append(f"cd {_q(workdir)}")
    body = f"bash {_q(ctrl_dir + '/user_body.sh')}"
    if spec.wrap:
        wrap = _q(f"{control_root.rstrip('/')}/bin/{sha8}/wrap.sh")
        lines.append(f"exec {wrap} {_q(ctrl_dir)} -- {body}")
    else:
        lines.append(f"exec {body}")
    text, _ = normalize_text("\n".join(lines) + "\n")
    return text


def render_alloc_sbatch(spec: JobSpec, handle: str, attempt_no: int, token: str, ctrl_dir: str, workdir: str,
                        control_root: str, sha8: str, *, idle_release_s: int = 0,
                        rendered_at: datetime | str | None = None) -> str:
    """``alloc.sbatch`` (design section 6.3 last paragraph): the job header with ``-J alloc-<handle>`` and
    ``-t <time>``; body ``source env.sh; cd workdir; exec <bin>/alloc-agent.sh <ctrl_dir> <idle_release_s>``.
    The CLI additionally carries :data:`ALLOC_CLI_ARGS` (``--signal=B:TERM@60 --no-requeue``).
    """
    lines = _header_lines(spec, f"alloc-{handle}")
    lines.append(f"#SBATCH -t {spec.resources.time}")
    lines.append("#SBATCH --open-mode=append")
    lines += _extra_lines(spec)
    lines.append(_tracking_line(handle, attempt_no, token, rendered_at))
    lines.append(f"source {_q(ctrl_dir + '/env.sh')}")
    lines.append(f"cd {_q(workdir)}")
    agent = _q(f"{control_root.rstrip('/')}/bin/{sha8}/alloc-agent.sh")
    lines.append(f"exec {agent} {_q(ctrl_dir)} {int(idle_release_s)}")
    text, _ = normalize_text("\n".join(lines) + "\n")
    return text


def render_env_sh(spec: JobSpec, *, grace_s: int | None = None, on_timeout: str | None = None,
                  max_restarts: int | None = None, child_signal: str | None = None, timelimit_s: int | None = None,
                  modules: Sequence[str] | None = None, setup: str | None = None,
                  env: Mapping[str, str] | None = None) -> str:
    """``env.sh`` (design section 6.3): the ``SLURM_MCP_*`` exports (``CHILD_SIGNAL`` only when set), the user env,
    then ``module load ...`` lines and the setup block. Every keyword defaults to the spec's value;
    ``timelimit_s`` defaults to the parsed ``resources.time``.
    """
    grace = spec.grace_s if grace_s is None else int(grace_s)
    ot = spec.on_timeout if on_timeout is None else on_timeout
    mr = spec.max_restarts if max_restarts is None else int(max_restarts)
    cs = spec.child_signal if child_signal is None else child_signal
    tl = spec.resources.time_s if timelimit_s is None else int(timelimit_s)
    mods = list(spec.modules if modules is None else modules)
    st = spec.setup if setup is None else setup
    ev = dict(spec.env if env is None else env)
    lines = ["# slurm-mcp env.sh (sourced by job.sbatch before wrap.sh)",
             f"export SLURM_MCP_GRACE={grace}",
             f"export SLURM_MCP_ON_TIMEOUT={shlex.quote(str(ot))}",
             f"export SLURM_MCP_MAX_RESTARTS={mr}"]
    if cs:
        lines.append(f"export SLURM_MCP_CHILD_SIGNAL={shlex.quote(str(cs))}")
    lines.append(f"export SLURM_MCP_TIMELIMIT_S={tl}")
    for k, v in ev.items():
        lines.append(f"export {k}={shlex.quote(str(v))}")
    for m in mods:
        if str(m).strip():
            lines.append(f"module load {shlex.quote(str(m).strip())}")
    if st and st.strip():
        lines.append(st.rstrip("\n"))
    text, _ = normalize_text("\n".join(lines) + "\n")
    return text


def render_user_body(spec: JobSpec, script_text: str | None = None) -> str:
    """``user_body.sh`` = ``spec.command`` or the user script body (design section 6.3).

    For ``script_path`` specs pass the fetched text as ``script_text`` (a remote path is cat'ed by the submitter).
    """
    if spec.command is not None:
        text, _ = normalize_text(spec.command)
        return text.rstrip("\n") + "\n"
    source = script_text if script_text is not None else spec.script
    if source is None:
        raise _invalid("script_path specs need the fetched script text to render user_body.sh")
    return parse_sbatch(source).body


# =================================================================================================
# 5.3 requeue flag, 6.1 QOS selection
# =================================================================================================

def _partition_name(partition_caps: Any) -> str | None:
    return _get(partition_caps, "name", "PartitionName", "partition")


def _preempt_mode(partition_caps: Any) -> str:
    """Partition ``PreemptMode`` as an upper-case comma string (parse.py gives a list, scontrol a ``GANG,REQUEUE`` string)."""
    return ",".join(_as_list(_get(partition_caps, "preempt_mode", "PreemptMode", default=None))).upper()


def cluster_charges(cluster_caps: Any) -> bool:
    """True when the cluster charges SUs: ``caps.charges`` if present, else any ``su_rates`` configured."""
    charges = _get(cluster_caps, "charges", "charging")
    if charges is not None:
        return _truthy(charges)
    return bool(_get(cluster_caps, "su_rates", default=None))


def requeue_flag(spec: JobSpec, partition_caps: Any, cluster_caps: Any) -> list[str]:
    """The requeue CLI flags per design section 5.3.

    ``--requeue --open-mode=append`` when ``spec.requeue`` is True, or None and (partition ``PreemptMode`` contains
    ``REQUEUE`` or ``on_timeout == "requeue"``); ``--no-requeue`` when ``spec.requeue`` is False, or None on a
    cluster that charges SUs and has ``JobRequeue=1``; otherwise nothing (site default).
    """
    if spec.requeue is True:
        return ["--requeue", "--open-mode=append"]
    if spec.requeue is False:
        return ["--no-requeue"]
    if "REQUEUE" in _preempt_mode(partition_caps).split(",") or spec.on_timeout == "requeue":
        return ["--requeue", "--open-mode=append"]
    if cluster_charges(cluster_caps) and _truthy(_get(cluster_caps, "job_requeue", "JobRequeue", default=False)):
        return ["--no-requeue"]
    return []


def requeueable(spec: JobSpec, partition_caps: Any, cluster_caps: Any) -> bool:
    """``--requeue`` rendered, or nothing rendered and ``JobRequeue=1`` (section 5.3; drives worst-case cost, section 8)."""
    flags = requeue_flag(spec, partition_caps, cluster_caps)
    if "--requeue" in flags:
        return True
    if flags:
        return False
    return _truthy(_get(cluster_caps, "job_requeue", "JobRequeue", default=False))


def partition_family(partition: str) -> str:
    """Lowercase alphabetic prefix of a partition name: ``GPU-shared`` -> ``gpu``, ``RM-shared`` -> ``rm``, ``EM`` -> ``em``."""
    m = re.match(r"[A-Za-z]+", partition or "")
    return m.group(0).lower() if m else (partition or "").lower()


def choose_qos(spec: JobSpec, profile: Any, partition_caps: Any, assoc: Any) -> list[str]:
    """Ordered QOS candidates for one partition (design section 6.1 "QOS selection"); caching is the caller's job.

    ``spec.qos`` -> ``profile.qos_map[partition]`` -> (``AllowQos == ALL`` and an assoc default QOS: ``[]``, no
    ``--qos``) -> ``AllowQos`` intersect assoc ``qos_list`` ordered: assoc default first, then names whose lowercase
    prefix matches the partition family, then ``low``, then the rest; names containing ``interact`` are excluded.
    """
    if spec.qos:
        return [spec.qos]
    partition = _partition_name(partition_caps) or ""
    qos_map = _get(profile, "qos_map", default={}) or {}
    if partition in qos_map and qos_map[partition]:
        return [qos_map[partition]]
    allow_raw = _get(partition_caps, "allow_qos", "AllowQos", default=None)
    allow = _as_list(allow_raw)
    allow_all = (not allow) or any(a.upper() == "ALL" for a in allow)
    default_qos = _get(assoc, "default_qos", "DefaultQOS", default=None) or None
    assoc_list = _as_list(_get(assoc, "qos_list", "qos", "QOS", default=None))
    if allow_all and default_qos:
        return []
    if allow_all:
        pool = list(assoc_list)
    elif assoc_list:
        pool = [q for q in allow if q in set(assoc_list)]
    else:
        pool = list(allow)
    pool = [q for q in pool if "interact" not in q.lower()]
    family = partition_family(partition)
    ordered: list[str] = []
    if default_qos and default_qos in pool:
        ordered.append(default_qos)
    ordered += [q for q in pool if q not in ordered and family and q.lower().startswith(family)]
    ordered += [q for q in pool if q not in ordered and q.lower() == "low"]
    ordered += [q for q in pool if q not in ordered]
    return ordered


# =================================================================================================
# 6.3 output patterns
# =================================================================================================

def _abs_pattern(pattern: str | None, workdir: str) -> str | None:
    if pattern is None or not pattern.strip():
        return None
    p = pattern.strip()
    if p.startswith("/"):
        return posixpath.normpath(p)
    return posixpath.normpath(posixpath.join(workdir.rstrip("/") or "/", p))


def resolve_output_patterns(spec: JobSpec, workdir: str, ctrl_root: str, is_array: bool | None = None) -> tuple[str, str]:
    """``(stdout_pattern, stderr_pattern)``: the spec's patterns resolved against ``workdir`` (always absolute),
    default ``<ctrl_root>/out/slurm-%j.out`` (arrays ``%A_%a``); stderr defaults to the stdout pattern, as sbatch
    merges both streams when ``-e`` is absent (design section 6.3, 5.1 step 3).
    """
    arr = bool(spec.array) if is_array is None else bool(is_array)
    default = f"{ctrl_root.rstrip('/')}/out/{DEFAULT_ARRAY_STDOUT_PATTERN if arr else DEFAULT_STDOUT_PATTERN}"
    out = _abs_pattern(spec.stdout, workdir) or default
    err = _abs_pattern(spec.stderr, workdir) or out
    return out, err


def pattern_needs_controller(pattern: str) -> bool:
    """True when the pattern contains ``%N``, ``%n``, ``%t`` or ``%s`` (expandable only by the controller)."""
    return any(m.group(2) in LATE_PATTERN_LETTERS for m in _PATTERN_RE.finditer(pattern))


def expand_pattern(pattern: str, slurm_id: str | int, name: str, user: str,
                   array_index: int | str | None = None) -> str | None:
    """Expand an sbatch filename pattern per sbatch(1) (design section 6.3 "Output path expansion").

    ``%j``/``%J`` -> job id, ``%x`` -> name, ``%u`` -> user, ``%A`` -> array base id (``slurm_id``), ``%a`` -> array
    index (sbatch's NO_VAL outside arrays), ``%%`` -> ``%``, ``%<n>j`` zero-pads the numeric fields. Returns None
    when the pattern needs the controller (``%N %n %t %s``).
    """
    if pattern_needs_controller(pattern):
        return None
    sid = str(slurm_id)
    idx = NO_VAL_ARRAY_INDEX if array_index is None else str(array_index)

    def repl(m: re.Match[str]) -> str:
        width, letter = m.group(1), m.group(2)
        if letter == "%":
            return "%"
        if letter in "jJA":
            val = sid
        elif letter == "a":
            val = idx
        elif letter == "x":
            return name
        else:  # u
            return user
        return val.zfill(int(width)) if width else val

    return _PATTERN_RE.sub(repl, pattern)


# =================================================================================================
# 6.3 target -> sbatch command line
# =================================================================================================

class RenderedArgs(NamedTuple):
    """Everything :func:`build_target_args` decided: the CLI args plus what to report."""

    args: list[str]
    injected: list[str]
    warnings: list[str]
    qos: str | None
    account: str | None
    requeueable: bool
    stdout_pattern: str
    stderr_pattern: str

    @property
    def submit_line(self) -> str:
        return format_submit_line(self.args)


def format_submit_line(args: Iterable[str]) -> str:
    """The exact shell text of an argument list (``shlex.join``)."""
    return shlex.join(list(args))


def strip_for_test_only(args: Sequence[str]) -> list[str]:
    """Target args without ``--parsable``, ``--comment=...`` and ``--hold`` (the estimate exec, section 6.3)."""
    return [a for a in args if a != "--parsable" and a != "--hold" and not a.startswith("--comment=")]


def _partition_caps_for(caps: Any, partition: str) -> Any:
    parts = _get(caps, "partitions", default=None)
    if isinstance(parts, Mapping):
        return parts.get(partition)
    if parts:
        for p in parts:
            if _partition_name(p) == partition:
                return p
    return None


def build_target_args(target: Target | str, spec: JobSpec, profile: Any, caps: Any, attempt: int, handle: str,
                      token: str, *, notify_email: str | None = None, excluded_nodes: Sequence[str] | None = None,
                      hold: bool = False, dependency: str | None = None, workdir: str | None = None,
                      ctrl_root: str | None = None, stdout_pattern: str | None = None,
                      stderr_pattern: str | None = None, qos: str | None = None,
                      mode: str = "job") -> RenderedArgs:
    """Build the sbatch command line for one target in the documented order (design section 6.3):

    ``-p`` ``--qos=`` ``-A`` ``-t`` ``--mem=`` ``--gres=`` requeue flags ``--signal=`` ``-o`` ``-e`` ``--array=``
    ``--dependency= --kill-on-invalid-dep=yes`` ``--mail-type= --mail-user=`` ``--exclude=`` ``--hold``
    ``--comment=slurm-mcp:<handle>:<attempt>:<token>`` ``--parsable``.

    ``caps`` is the cluster's discovery cache (``partitions``, ``assoc``, ``job_requeue``, ``charges``,
    ``default_account``, ``qos_for_partition``); ``dependency`` is the already resolved SLURM list
    (``afterok:615408``); output patterns come from ``stdout_pattern``/``stderr_pattern`` or are resolved from
    ``workdir`` + ``ctrl_root``. ``mode="alloc"`` swaps the wrapper signal/requeue flags for
    :data:`ALLOC_CLI_ARGS`.
    """
    tgt = Target.parse(target) if isinstance(target, str) else target
    warnings: list[str] = []
    injected: list[str] = []
    args: list[str] = []
    r = spec.resources
    p0 = tgt.partitions[0]
    pcaps = _partition_caps_for(caps, p0)
    assoc = _get(caps, "assoc", default=None)

    args += ["-p", ",".join(tgt.partitions)]

    chosen_qos = tgt.qos or qos or spec.qos
    if chosen_qos is None:
        cached = _get(caps, "qos_for_partition", default={}) or {}
        chosen_qos = cached.get(p0) if isinstance(cached, Mapping) else None
    if chosen_qos is None:
        cands = choose_qos(spec, profile, pcaps if pcaps is not None else {"name": p0}, assoc)
        chosen_qos = cands[0] if cands else None
    if chosen_qos:
        args.append(f"--qos={chosen_qos}")
        if chosen_qos != spec.qos:
            injected.append(f"--qos={chosen_qos}")

    account = spec.account or tgt.account or _get(profile, "default_account", default=None) \
        or _get(caps, "default_account", "DefaultAccount", default=None)
    if account:
        args += ["-A", account]
        if account != spec.account:
            injected.append(f"-A {account}")

    args += ["-t", str(r.time)]

    if r.mem:
        no_mem = set(_get(profile, "no_mem_flag", default=[]) or [])
        blocked = [p for p in tgt.partitions if p in no_mem]
        if blocked:
            warnings.append(f"--mem={r.mem} dropped: partition {','.join(blocked)} is in profile.no_mem_flag")
        else:
            args.append(f"--mem={r.mem}")

    if r.gpus > 0:
        gtype = tgt.gres_type or (r.gpu_types[0] if r.gpu_types else None)
        if gtype:
            args.append(f"--gres=gpu:{gtype}:{r.gpus}")
        else:
            args.append(f"--gres=gpu:{r.gpus}")
            warnings.append("untyped --gres=gpu:N sent: the target has no gres type and the spec no gpu_types")

    if mode == "alloc":
        args += list(ALLOC_CLI_ARGS)
        injected += list(ALLOC_CLI_ARGS)
        is_requeueable = False
    else:
        rq = requeue_flag(spec, pcaps, caps)
        args += rq
        if spec.requeue is None and rq:
            injected += rq
        is_requeueable = requeueable(spec, pcaps, caps)
        if spec.wrap:
            sig = f"--signal=B:USR1@{spec.grace_s}"
            args.append(sig)
            injected.append(sig)

    if stdout_pattern is None or stderr_pattern is None:
        if workdir is None or ctrl_root is None:
            raise ValueError("build_target_args needs stdout_pattern/stderr_pattern or workdir + ctrl_root")
        out_p, err_p = resolve_output_patterns(spec, workdir, ctrl_root)
        stdout_pattern = stdout_pattern or out_p
        stderr_pattern = stderr_pattern or err_p
    args += ["-o", stdout_pattern, "-e", stderr_pattern]
    if not spec.stdout:
        injected.append(f"-o {stdout_pattern}")
    if not spec.stderr:
        injected.append(f"-e {stderr_pattern}")

    if spec.array:
        arr = spec.array + (f"%{spec.array_parallel}" if spec.array_parallel else "")
        args.append(f"--array={arr}")

    dep = dependency
    if dep is None and "singleton" in spec.depends_on:
        dep = "singleton"
    if dep:
        args += [f"--dependency={dep}", "--kill-on-invalid-dep=yes"]
        injected.append("--kill-on-invalid-dep=yes")

    if notify_email:
        args += [f"--mail-type={MAIL_TYPES}", f"--mail-user={notify_email}"]
        injected += [f"--mail-type={MAIL_TYPES}", f"--mail-user={notify_email}"]

    nodes = [n for n in (excluded_nodes or []) if n]
    if nodes:
        args.append(f"--exclude={','.join(nodes)}")
        injected.append(f"--exclude={','.join(nodes)}")

    if hold:
        args.append("--hold")

    comment = f"--comment={COMMENT_PREFIX}:{handle}:{attempt}:{token}"
    args += [comment, "--parsable"]
    injected += [comment, "--parsable"]

    return RenderedArgs(args, injected, warnings, chosen_qos, account, is_requeueable, stdout_pattern, stderr_pattern)


def target_args(target: Target | str, spec: JobSpec, profile: Any, caps: Any, attempt: int, handle: str, token: str,
                notify_email: str | None = None, excluded_nodes: Sequence[str] | None = None, hold: bool = False,
                **kw: Any) -> list[str]:
    """The sbatch CLI args for a target (design section 6.3); see :func:`build_target_args` for the details."""
    return build_target_args(target, spec, profile, caps, attempt, handle, token, notify_email=notify_email,
                             excluded_nodes=excluded_nodes, hold=hold, **kw).args


__all__ = [
    "DEFAULT_STDOUT_PATTERN", "DEFAULT_ARRAY_STDOUT_PATTERN", "MAIL_TYPES", "COMMENT_PREFIX", "ALLOC_CLI_ARGS",
    "LATE_PATTERN_LETTERS", "NO_VAL_ARRAY_INDEX", "DEPENDENCY_TYPES", "MANAGED_WITH_ARG", "MANAGED_FLAGS",
    "MANAGED_DIRECTIVES", "ParsedScript", "parse_sbatch", "merge_spec", "render_job_sbatch", "render_alloc_sbatch",
    "render_env_sh", "render_user_body", "requeue_flag", "requeueable", "cluster_charges", "partition_family",
    "choose_qos", "resolve_output_patterns", "pattern_needs_controller", "expand_pattern", "RenderedArgs",
    "format_submit_line", "strip_for_test_only", "build_target_args", "target_args",
]
