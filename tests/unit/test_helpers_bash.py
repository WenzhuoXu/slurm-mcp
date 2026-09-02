"""pytest wrapper around tests/bash/test_helpers.sh (design section 7 helpers, section 10 "real bash").

On Windows the Git Bash smoke subset runs (no setsid); on Linux the full suite. One pytest test per bash test
name; ``skip -`` lines become pytest skips. See tests/bash/README.md.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "bash" / "test_helpers.sh"
HELPERS = ROOT / "src" / "slurm_mcp" / "helpers"


def _candidate_bashes() -> list[str]:
    out: list[str] = []
    env = os.environ.get("SLURM_MCP_BASH")
    if env:
        out.append(env)
    if sys.platform == "win32":
        git = shutil.which("git")
        if git:
            gdir = Path(git).resolve().parent.parent
            out += [str(gdir / "bin" / "bash.exe"), str(gdir / "usr" / "bin" / "bash.exe")]
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                     os.environ.get("LOCALAPPDATA", "")):
            if base:
                out += [str(Path(base) / "Git" / "bin" / "bash.exe"), str(Path(base) / "Git" / "usr" / "bin" / "bash.exe")]
    which = shutil.which("bash")
    if which and "system32" not in which.lower():    # the WSL launcher is not a usable bash here
        out.append(which)
    seen: list[str] = []
    for c in out:
        if c and c not in seen and Path(c).exists():
            seen.append(c)
    return seen


def _find_bash() -> str | None:
    for cand in _candidate_bashes():
        try:
            r = subprocess.run([cand, "-c", "echo ok"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and r.stdout.strip() == "ok":
            return cand
    return None


BASH = None if os.environ.get("SLURM_MCP_SKIP_BASH_TESTS") else _find_bash()


def _posix(p: Path) -> str:
    return str(p).replace("\\", "/")


def _list_names() -> list[str]:
    if BASH is None:
        return []
    r = subprocess.run([BASH, _posix(SCRIPT), "--list"], capture_output=True, text=True, timeout=30)
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


NAMES = _list_names()

pytestmark = pytest.mark.skipif(BASH is None, reason="no usable bash found (set SLURM_MCP_BASH)")


def test_bash_lists_tests():
    assert NAMES, "test_helpers.sh --list returned nothing"
    assert "wrap_rc0" in NAMES and "submit_ok" in NAMES and "alloc_fg" in NAMES


@pytest.mark.parametrize("name", NAMES or ["<none>"])
def test_helper(name: str, tmp_path: Path):
    if name == "<none>":
        pytest.skip("no bash tests listed")
    env = dict(os.environ)
    env["TMPDIR"] = _posix(tmp_path)
    env["SLURM_MCP_HELPERS"] = _posix(HELPERS)
    r = subprocess.run([BASH, _posix(SCRIPT), name], capture_output=True, text=True, timeout=180, env=env)
    lines = r.stdout.splitlines()
    result = next((ln for ln in lines if ln.endswith(f"- {name}") or f"- {name}:" in ln), None)
    assert result is not None, f"no result line for {name}:\n{r.stdout}\n{r.stderr}"
    if result.startswith("skip -"):
        pytest.skip(result.split(":", 1)[1].strip() if ":" in result else result)
    assert result.startswith("ok -"), f"{result}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
    assert r.returncode == 0
