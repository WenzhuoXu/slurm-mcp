"""Shared fixtures: an isolated SLURM_MCP_HOME, the in-process fake cluster and a ClusterProfile for it."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

# Point slurm_mcp.config at a scratch home *before* it is imported anywhere (CONFIG_DIR is computed at import).
_MCP_HOME = tempfile.mkdtemp(prefix="slurm-mcp-home-")
os.environ["SLURM_MCP_HOME"] = _MCP_HOME

from sshd_harness import SSH_PASSWORD, SSH_USER, fake_cluster  # noqa: E402


@pytest.fixture(scope="session")
def mcp_home() -> str:
    return _MCP_HOME


@pytest_asyncio.fixture
async def trace_cluster():
    async with fake_cluster("trace", now="2026-09-01T17:00:00") as fc:
        yield fc


@pytest_asyncio.fixture
async def bridges2_cluster():
    async with fake_cluster("bridges2", now="2026-09-01T17:00:00") as fc:
        yield fc


def make_profile(fc, name: str = "fake"):
    """Build a ClusterProfile for a running fake cluster and set the password env override."""
    from slurm_mcp.config import ClusterProfile

    os.environ["SLURM_MCP_PASSWORD_" + name.upper().replace("-", "_")] = SSH_PASSWORD
    return ClusterProfile(name=name, host=fc.host, user=SSH_USER, port=fc.port, auth="password",
                          remote_root=fc.home)


@pytest.fixture
def trace_profile(trace_cluster, mcp_home):
    return make_profile(trace_cluster, "fake-trace")
