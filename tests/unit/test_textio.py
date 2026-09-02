"""Unit tests for slurm_mcp.textio (design sections 3.2, 4 download, 5.5, changelog 10)."""
from __future__ import annotations

import os
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from slurm_mcp.errors import SlurmMcpError
from slurm_mcp.textio import (LONG_PATH_THRESHOLD, local_safe_name, long_path, normalize_text, normcase_scope,
                              posix_rel, read_local_text)


# --- normalize_text --------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected,warn", [
    ("a\nb\n", "a\nb\n", []),
    ("a\r\nb\r\n", "a\nb\n", ["crlf_normalized"]),
    ("a\rb", "a\nb", ["crlf_normalized"]),
    ("\ufeff#!/bin/bash\n", "#!/bin/bash\n", []),
    ("\ufeff#!/bin/bash\r\necho\r\n", "#!/bin/bash\necho\n", ["crlf_normalized"]),
    ("", "", []),
    ("mixed\r\nand\rand\n", "mixed\nand\nand\n", ["crlf_normalized"]),
])
def test_normalize_text(text, expected, warn):
    assert normalize_text(text) == (expected, warn)


def test_normalize_text_refuses_nul():
    with pytest.raises(SlurmMcpError) as ei:
        normalize_text("abc\x00def")
    assert ei.value.code == "E_INVALID_SPEC"
    assert "NUL" in str(ei.value)


# --- read_local_text -------------------------------------------------------------------------------

def test_read_local_text_utf8_sig_and_crlf(tmp_path):
    p = tmp_path / "s.sh"
    p.write_bytes(b"\xef\xbb\xbf#!/bin/bash\r\necho hi\r\n")
    text, warnings = read_local_text(p)
    assert text == "#!/bin/bash\necho hi\n"
    assert warnings == ["crlf_normalized"]
    text2, w2 = read_local_text(str(p))
    assert (text2, w2) == (text, warnings)


def test_read_local_text_missing_and_binary(tmp_path):
    with pytest.raises(SlurmMcpError) as ei:
        read_local_text(tmp_path / "nope.sh")
    assert ei.value.code == "E_INVALID_SPEC"
    bad = tmp_path / "bin.sh"
    bad.write_bytes(b"\xff\xfe\x00\x00garbage\xff")
    with pytest.raises(SlurmMcpError) as ei:
        read_local_text(bad)
    assert ei.value.code == "E_INVALID_SPEC"


# --- posix_rel -------------------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("a/b/c", "a/b/c"), ("a\\b\\c", "a/b/c"), ("./a/b", "a/b"), ("a//b/", "a/b"), ("a\\.\\b", "a/b"),
    (PurePosixPath("x/y"), "x/y"), (PureWindowsPath("x\\y\\z.txt"), "x/y/z.txt"), ("", ""), (".", ""),
    ("..\\a", "../a"), ("a b/c d.txt", "a b/c d.txt"),
])
def test_posix_rel(value, expected):
    out = posix_rel(value)
    assert out == expected
    assert "\\" not in out


# --- normcase_scope --------------------------------------------------------------------------------

def test_normcase_scope_is_stable_across_case_and_separators(tmp_path):
    d = tmp_path / "Proj"
    d.mkdir()
    a = normcase_scope(d)
    b = normcase_scope(str(d).replace("\\", "/"))
    assert a == b
    if os.name == "nt":
        assert normcase_scope(str(d).upper()) == a
        assert a == a.lower()
    assert os.path.isabs(a)


def test_normcase_scope_relative_resolves_to_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert normcase_scope("sub") == os.path.normcase(os.path.realpath(os.path.join(str(tmp_path), "sub")))


# --- local_safe_name -------------------------------------------------------------------------------

@pytest.mark.parametrize("remote,expected", [
    ("run_2026-09-01T12:00:00", "run_2026-09-01T12_00_00"),
    ("a/b:c/d?e*f\"g<h>i|j", "a/b_c/d_e_f_g_h_i_j"),
    ("dir./file.txt.", "dir/file.txt"),
    ("dir /name ", "dir/name"),
    ("CON", "_CON"), ("con.txt", "_con.txt"), ("NUL", "_NUL"), ("PRN.log", "_PRN.log"), ("AUX", "_AUX"),
    ("COM1", "_COM1"), ("COM9.dat", "_COM9.dat"), ("LPT1", "_LPT1"), ("LPT9", "_LPT9"),
    ("COM0", "COM0"), ("COM10", "COM10"), ("CONSOLE", "CONSOLE"), ("nulled.txt", "nulled.txt"),
    ("a/CON/b", "a/_CON/b"),
    ("ctrl\x01\x1f\x7fchar", "ctrl___char"),
    ("...", "_"), ("plain/ok_name-1.txt", "plain/ok_name-1.txt"), ("a\\b:c", "a/b_c"),
])
def test_local_safe_name(remote, expected):
    assert local_safe_name(remote) == expected


def test_local_safe_name_idempotent():
    for s in ["run_2026-09-01T12:00:00", "CON", "dir./x.", "a/b|c"]:
        once = local_safe_name(s)
        assert local_safe_name(once) == once


# --- long_path -------------------------------------------------------------------------------------

def test_long_path_short_is_noop():
    assert long_path("C:\\short\\path.txt") == "C:\\short\\path.txt"
    assert long_path("relative/path") == "relative/path"


def test_long_path_prefix_for_long_absolute_paths():
    tail = "x" * (LONG_PATH_THRESHOLD + 20)
    p = "C:\\base\\" + tail
    out = long_path(p)
    if os.name == "nt":
        assert out.startswith("\\\\?\\C:\\base\\")
        assert out.endswith(tail)
        assert long_path(out) == out  # idempotent
        unc = "\\\\server\\share\\" + tail
        assert long_path(unc).startswith("\\\\?\\UNC\\server\\share\\")
        assert long_path("C:/base/" + tail).startswith("\\\\?\\C:\\base\\")
        assert long_path("relative/" + tail) == "relative/" + tail
    else:
        assert out == p


def test_long_path_accepts_pathlike(tmp_path):
    assert long_path(tmp_path) == str(tmp_path) or long_path(tmp_path).endswith(str(tmp_path)[2:])
