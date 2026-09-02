"""Unit tests for slurm_mcp.helpers (design section 7: bundle files, sha8, verbatim scripts)."""
from __future__ import annotations

import difflib
import hashlib
import re
from pathlib import Path

import pytest

from slurm_mcp import helpers
from slurm_mcp.helpers import (BUNDLE_FILES, bundle_bytes, bundle_files, bundle_sha8, helper_path, read_helper,
                               read_helper_bytes)

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "docs" / "design.md"

# The one documented deviation from design.md section 7.1 (see the F4 report): a trapped signal makes bash's
# ``wait`` return 128+signum *before* the trap runs, and the design's ``while kill -0 "$CHILD"`` re-wait is skipped
# once the child has been reaped, so the interrupted-wait status (138 on Linux, 158 on MSYS where USR1=30) leaked
# out as the payload's rc. wrap.sh re-waits while a trap flag is set instead. Everything else is verbatim.
WRAP_SH_PATCH: tuple[tuple[str, str], ...] = (
    ("on_usr1() {  # time-limit warning: record it; forward ONLY when the payload declared it handles $CSIG\n"
     "  SIGNALED=USR1;",
     "on_usr1() {  # time-limit warning: record it; forward ONLY when the payload declared it handles $CSIG\n"
     "  INTR=1; SIGNALED=USR1;"),
    ("on_term() {  # scancel / time limit / preemption: forward TERM as TERM, never $CSIG\n  SIGNALED=TERM;",
     "on_term() {  # scancel / time limit / preemption: forward TERM as TERM, never $CSIG\n  INTR=1; SIGNALED=TERM;"),
    ('CHILD=$!\nwait "$CHILD"; RC=$?\n'
     'while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD"; RC=$?; done   # re-wait when a trap interrupted wait\n',
     'CHILD=$!; INTR=""\nwait "$CHILD"; RC=$?\n'
     "# A trapped signal makes `wait` return 128+signum before the trap runs; re-wait for the real status "
     "(a 127 means it was already collected)\n"
     'while [ -n "$INTR" ] && [ "$RC" -gt 128 ]; do INTR=""; PREV=$RC; wait "$CHILD"; RC=$?; '
     '[ "$RC" -eq 127 ] && RC=$PREV; done\n'),
)


def _design_blocks() -> list[str]:
    text = DESIGN.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(#!/bin/bash\n.*?)```", text, re.S)
    assert len(blocks) >= 3
    return blocks[:3]


def test_bundle_files_order():
    assert BUNDLE_FILES == ("wrap.sh", "submit.sh", "alloc-agent.sh")


@pytest.mark.parametrize("name", BUNDLE_FILES)
def test_helper_is_lf_bash_script(name: str):
    raw = read_helper_bytes(name)
    assert raw.startswith(b"#!/bin/bash\n")
    assert b"\r" not in raw
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.endswith(b"\n")
    assert read_helper(name) == raw.decode("utf-8")
    assert helper_path(name).name == name and helper_path(name).is_file()


def test_unknown_helper_refused():
    with pytest.raises(KeyError):
        helper_path("probe.sh")


def test_bundle_sha8_is_sha256_prefix_over_concatenation():
    expected = hashlib.sha256(b"".join(read_helper_bytes(n) for n in BUNDLE_FILES)).hexdigest()[:8]
    assert bundle_sha8() == expected
    assert re.fullmatch(r"[0-9a-f]{8}", bundle_sha8())
    assert bundle_bytes() == b"".join(read_helper_bytes(n) for n in BUNDLE_FILES)
    assert bundle_sha8() == hashlib.sha256(bundle_bytes()).hexdigest()[:8]


def test_bundle_files_mapping():
    files = bundle_files()
    assert list(files) == list(BUNDLE_FILES)
    assert all(isinstance(v, bytes) and v for v in files.values())


def test_sha8_changes_with_content(tmp_path, monkeypatch):
    original = bundle_sha8()
    changed = {n: read_helper_bytes(n) for n in BUNDLE_FILES}
    changed["wrap.sh"] += b"# x\n"
    monkeypatch.setattr(helpers, "read_helper_bytes", lambda n: changed[n])
    assert helpers.bundle_sha8() != original


@pytest.mark.skipif(not DESIGN.exists(), reason="docs/design.md not present")
def test_helpers_verbatim_from_design_section_7():
    """submit.sh and alloc-agent.sh are copied exactly from the ```bash blocks of design.md section 7;
    wrap.sh is the design text with exactly the documented re-wait patch applied."""
    blocks = dict(zip(BUNDLE_FILES, _design_blocks()))
    assert read_helper("submit.sh") == blocks["submit.sh"]
    assert read_helper("alloc-agent.sh") == blocks["alloc-agent.sh"]
    patched = blocks["wrap.sh"]
    for old, new in WRAP_SH_PATCH:
        assert patched.count(old) == 1, f"design.md wrap.sh no longer contains the patched line: {old!r}"
        patched = patched.replace(old, new)
    assert read_helper("wrap.sh") == patched, "wrap.sh differs from design.md beyond the documented re-wait patch"


@pytest.mark.skipif(not DESIGN.exists(), reason="docs/design.md not present")
def test_wrap_sh_patch_is_minimal():
    """Only the three patched hunks differ between design.md and the shipped wrap.sh."""
    design = _design_blocks()[0].splitlines()
    shipped = read_helper("wrap.sh").splitlines()
    changed = [ln for ln in difflib.unified_diff(design, shipped, lineterm="", n=0)
               if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---"))]
    removed = [ln for ln in changed if ln.startswith("-")]
    added = [ln for ln in changed if ln.startswith("+")]
    assert 3 <= len(removed) <= 5
    assert all("INTR" in ln or "wait" in ln for ln in added)


def test_wrap_sh_contract_markers():
    wrap = read_helper("wrap.sh")
    for needle in ('"v":2', "SLURM_MCP_CHILD_SIGNAL", "trap on_usr1 USR1", "trap on_term TERM", "exit 75",
                   "scontrol requeue", "cancel.requested", "requeue.requested", "--signal=B:USR1@GRACE"):
        assert needle in wrap


def test_submit_sh_contract_markers():
    sub = read_helper("submit.sh")
    for needle in ("JOBID", "ERR 2", "ERR 3", "sbatch --parsable", "mkdir \"$CTRL/.submit.lock\"",
                   "socket timed out|unable to contact|zero bytes"):
        assert needle in sub


def test_alloc_agent_contract_markers():
    agent = read_helper("alloc-agent.sh")
    for needle in ("*.bg.sh", '"$b.kill"', '"$CTRL/release"', "idle release", "status ready", '"fg":"%s"'):
        assert needle in agent
