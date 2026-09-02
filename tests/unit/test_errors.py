"""Unit tests for slurm_mcp.errors (design section 9.1)."""
from __future__ import annotations

import pytest

from slurm_mcp import _mcp
from slurm_mcp.errors import CATALOGUE, CODES, EM_DASH, SlurmMcpError, err, to_tool_error

DESIGN_CODES = [
    "E_AUTH", "E_HOSTKEY", "E_UNREACHABLE", "E_SSH", "E_CTLD_BUSY", "E_SUBMIT_AMBIGUOUS", "E_SUBMIT_FAILED",
    "E_PARTITION", "E_PARTITION_REQUIRED", "E_ACCOUNT", "E_QOS", "E_QOS_MAXWALL", "E_QOS_SIZE", "E_QOS_POLICY",
    "E_SUBMIT_LIMIT", "E_NODE_CONFIG", "E_GRES", "E_MEM", "E_DEPENDENCY", "E_DEP_CROSS_CLUSTER", "E_PERMISSION",
    "E_SCRIPT", "E_QUOTA", "E_NO_TARGET", "E_UNKNOWN_ID", "E_NO_LOG_YET", "E_ALLOC_NOT_READY", "E_ALLOC_ENDED",
    "E_CMD_TOO_LONG", "E_TOO_MANY_FILES", "E_TOO_MANY_BYTES", "E_UPLOAD", "E_CONFIRM_REQUIRED", "E_PLAN_EXPIRED",
    "E_INVALID_SPEC", "E_HELPER", "E_STATE",
]


def test_em_dash_is_u2014():
    assert EM_DASH == "\u2014"
    assert len(EM_DASH) == 1


def test_catalogue_covers_every_design_code_exactly():
    assert set(CATALOGUE) == set(DESIGN_CODES)
    assert CODES == frozenset(DESIGN_CODES)


@pytest.mark.parametrize("code", DESIGN_CODES)
def test_every_code_has_nonempty_fix(code):
    assert CATALOGUE[code].strip()
    e = err(code, "boom")
    assert e.code == code
    assert e.fix


def test_str_format_exact():
    e = SlurmMcpError("E_QUOTA", "disk full", "free space")
    assert str(e) == "E_QUOTA: disk full \u2014 fix: free space"
    assert e.args[0] == str(e)


def test_default_fix_from_catalogue():
    e = SlurmMcpError("E_PLAN_EXPIRED", "plan p9 is gone")
    assert e.fix == CATALOGUE["E_PLAN_EXPIRED"]
    assert str(e).startswith("E_PLAN_EXPIRED: plan p9 is gone \u2014 fix: ")


def test_unknown_code_in_constructor_gets_generic_fix():
    e = SlurmMcpError("E_NOPE", "x")
    assert e.fix == "see the message"


def test_err_fills_template_placeholders():
    e = err("E_QUOTA", "quota exceeded", path="/ocean", used_pct=97)
    assert "/ocean" in e.fix and "97%" in e.fix
    assert "{path}" not in e.fix


def test_err_leaves_unknown_placeholders_and_fills_message():
    e = err("E_NO_LOG_YET", "job {handle} has no log", handle="j17", state="SUBMITTED")
    assert e.message == "job j17 has no log"
    assert "SUBMITTED" in e.fix
    assert "{next}" in e.fix  # not supplied -> left verbatim, never KeyError


def test_err_fix_override():
    e = err("E_STATE", "bad", fix="custom")
    assert e.fix == "custom"


def test_err_unknown_code_raises():
    with pytest.raises(ValueError):
        err("E_DOES_NOT_EXIST", "x")


def test_to_tool_error_roundtrip():
    e = err("E_CMD_TOO_LONG", "4200 chars")
    te = to_tool_error(e)
    assert isinstance(te, _mcp.ToolError)
    assert str(te) == str(e)
    assert e.to_tool_error().args[0] == str(e)


def test_to_tool_error_passthrough_and_generic():
    te = _mcp.ToolError("already")
    assert to_tool_error(te) is te
    generic = to_tool_error(RuntimeError("kaboom"))
    assert isinstance(generic, _mcp.ToolError)
    assert str(generic).startswith("E_SSH: RuntimeError: kaboom \u2014 fix: ")
    empty = to_tool_error(ValueError())
    assert str(empty).startswith("E_SSH: ValueError \u2014 fix: ")


def test_is_exception_and_repr():
    e = err("E_AUTH", "denied", cluster="trace")
    with pytest.raises(SlurmMcpError):
        raise e
    assert "E_AUTH" in repr(e) and "trace" in e.fix


def test_mcp_reexports_toolerror():
    assert "ToolError" in _mcp.__all__
    assert issubclass(_mcp.ToolError, Exception)
