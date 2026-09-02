"""Remote helper bundle (design section 7): ``wrap.sh``, ``submit.sh``, ``alloc-agent.sh`` as package data.

``bundle_sha8()`` is the first 8 hex chars of ``sha256(wrap.sh + submit.sh + alloc-agent.sh)`` (bytes, in
that order); the bundle is deployed to ``<control_root>/bin/<sha8>/`` and ``<control_root>/bin/VERSION``
holds the sha8 (section 6.1 "Helper deploy"). Imports nothing from the package.
"""
from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path

# Order matters: it is the concatenation order of the sha256 (design section 7).
BUNDLE_FILES: tuple[str, ...] = ("wrap.sh", "submit.sh", "alloc-agent.sh")


def helper_path(name: str) -> Path:
    """Filesystem path of one helper (``name`` must be in ``BUNDLE_FILES``)."""
    if name not in BUNDLE_FILES:
        raise KeyError(f"unknown helper {name!r}; expected one of {BUNDLE_FILES}")
    return Path(str(resources.files(__package__).joinpath(name)))


def read_helper_bytes(name: str) -> bytes:
    """Raw bytes of one helper, exactly as deployed (LF line endings, no BOM)."""
    return helper_path(name).read_bytes()


def read_helper(name: str) -> str:
    """Text of one helper (UTF-8, LF line endings)."""
    return read_helper_bytes(name).decode("utf-8")


def bundle_bytes() -> bytes:
    """``wrap.sh + submit.sh + alloc-agent.sh`` concatenated in ``BUNDLE_FILES`` order."""
    return b"".join(read_helper_bytes(n) for n in BUNDLE_FILES)


def bundle_sha8() -> str:
    """First 8 hex chars of the sha256 over the concatenated bundle (design section 7)."""
    return hashlib.sha256(bundle_bytes()).hexdigest()[:8]


def bundle_files() -> dict[str, bytes]:
    """``{name: bytes}`` for every helper, for the deployer (SFTP put + chmod 755)."""
    return {n: read_helper_bytes(n) for n in BUNDLE_FILES}


__all__ = ["BUNDLE_FILES", "helper_path", "read_helper", "read_helper_bytes", "bundle_bytes", "bundle_sha8",
           "bundle_files"]
