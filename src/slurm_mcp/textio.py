"""Text and path helpers for a Windows client talking to POSIX clusters (design sections 2 "textio.py",
3.2 validation, 4 "download", 5.5, changelog item 10).

Imports only ``errors`` from the package.
"""
from __future__ import annotations

import os
import re
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

from .errors import SlurmMcpError

BOM = "\ufeff"
_INVALID_CHARS_RE = re.compile(r'[:?*"<>|\x00-\x1f\x7f]')
_RESERVED_RE = re.compile(r"^(CON|NUL|PRN|AUX|COM[1-9]|LPT[1-9])(\.[^.]*)?$", re.IGNORECASE)
LONG_PATH_THRESHOLD = 240
_LONG_PREFIX = "\\\\?\\"


def normalize_text(text: str) -> tuple[str, list[str]]:
    """Normalise user-supplied script text: BOM stripped, CRLF/CR -> LF, NUL refused.

    Returns ``(text, warnings)``; ``warnings`` contains ``"crlf_normalized"`` when any CR was removed.
    A NUL byte raises ``SlurmMcpError(E_INVALID_SPEC)`` (a binary file was passed as a script).
    """
    if "\x00" in text:
        raise SlurmMcpError("E_INVALID_SPEC", "text contains a NUL byte (binary file?)",
                            "pass a UTF-8 text script; upload binaries with upload()")
    warnings: list[str] = []
    if text.startswith(BOM):
        text = text[len(BOM):]
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        warnings.append("crlf_normalized")
    return text, warnings


def read_local_text(path: str | os.PathLike[str]) -> tuple[str, list[str]]:
    """Read a local text file as utf-8-sig and normalise it (section 3.2 ``script_path='local:...'``)."""
    p = Path(path)
    try:
        raw = p.read_bytes().decode("utf-8-sig")   # bytes: no universal-newline translation before normalising
    except FileNotFoundError:
        raise SlurmMcpError("E_INVALID_SPEC", f"local file not found: {p}",
                            "check script_path / inputs[].local") from None
    except UnicodeDecodeError:
        raise SlurmMcpError("E_INVALID_SPEC", f"local file is not UTF-8 text: {p}",
                            "pass a UTF-8 text script; upload binaries with upload()") from None
    return normalize_text(raw)


def posix_rel(path_like: str | os.PathLike[str] | PurePath) -> str:
    """Render a relative path as ``a/b/c`` regardless of the host OS separator (never ``os.sep``).

    Backslashes are treated as separators (Windows input); ``.`` components are dropped; a leading
    ``./`` is removed. The result is a *relative* POSIX string suitable for manifests and tar members.
    """
    s = str(path_like)
    s = s.replace("\\", "/")
    parts = [p for p in s.split("/") if p not in ("", ".")]
    return "/".join(parts)


def normcase_scope(local_path: str | os.PathLike[str]) -> str:
    """Normalised key for a local path (manifest scopes): ``os.path.normcase(os.path.realpath(path))``."""
    return os.path.normcase(os.path.realpath(os.fspath(local_path)))


def _safe_component(name: str) -> str:
    out = _INVALID_CHARS_RE.sub("_", name)
    out = out.rstrip(". ")
    if not out:
        out = "_"
    if _RESERVED_RE.match(out):
        out = "_" + out
    return out


def local_safe_name(rel_posix_path: str) -> str:
    """Rewrite a remote relative path so every component is a valid NTFS name (section 4 ``download``).

    ``: ? * " < > |`` and control characters become ``_``; trailing dots/spaces are stripped per
    component; reserved device names (CON NUL PRN AUX COM1-9 LPT1-9, with or without extension) get a
    ``_`` prefix. The result is still a POSIX-style relative path (``/`` separators).
    """
    parts = [p for p in rel_posix_path.replace("\\", "/").split("/") if p not in ("", ".")]
    return "/".join(_safe_component(p) for p in parts)


def long_path(path: str | os.PathLike[str]) -> str:
    """Prefix an absolute Windows path longer than 240 chars with ``\\\\?\\`` (no-op elsewhere).

    UNC paths become ``\\\\?\\UNC\\server\\share\\...``. Relative paths and non-Windows hosts are returned
    unchanged (as strings).
    """
    s = os.fspath(path)
    if os.name != "nt":
        return s
    if s.startswith(_LONG_PREFIX):
        return s
    if len(s) <= LONG_PATH_THRESHOLD:
        return s
    win = PureWindowsPath(s)
    if not win.is_absolute():
        return s
    normalized = str(win).replace("/", "\\")
    if normalized.startswith("\\\\"):
        return _LONG_PREFIX + "UNC\\" + normalized[2:]
    return _LONG_PREFIX + normalized


__all__ = ["BOM", "LONG_PATH_THRESHOLD", "normalize_text", "read_local_text", "posix_rel",
           "normcase_scope", "local_safe_name", "long_path", "PurePosixPath"]
