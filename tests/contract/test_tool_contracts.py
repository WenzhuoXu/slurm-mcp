"""Contract tests through the in-memory MCP client (design section 10 layer 7, rule 1 of section 1): every
registered tool has annotations, a description under 2 KB and a valid input schema; the phase-2 tools return
structured content that validates against their output schema with a non-empty summary; errors start with E_."""
from __future__ import annotations

import os

import pytest
from mcp import Client

from slurm_mcp.config import ClusterProfile
from slurm_mcp.events import EventBus
from slurm_mcp.models import (
    RESULT_MODELS, ClusterStatusResult, ClustersResult, ConfigResult, ListingResult, ReadResult, RunCommandResult,
    WriteResult,
)
from slurm_mcp.server import INSTRUCTIONS, ServiceProxy, build_server
from slurm_mcp.service import ClusterRegistry, Service
from slurm_mcp.store import Store
from slurm_mcp.tools import MAX_DESCRIPTION_CHARS
from sshd_harness import SSH_PASSWORD, SSH_USER

PHASE2_TOOLS: dict[str, type] = {
    "clusters": ClustersResult, "cluster_status": ClusterStatusResult, "run_command": RunCommandResult,
    "remote_ls": ListingResult, "remote_read": ReadResult, "remote_write": WriteResult, "configure": ConfigResult,
}


@pytest.fixture
async def world(trace_cluster, tmp_path):
    name = "fake-trace"
    os.environ["SLURM_MCP_PASSWORD_FAKE_TRACE"] = SSH_PASSWORD
    profile = ClusterProfile(name=name, host=trace_cluster.host, user=SSH_USER, port=trace_cluster.port,
                             auth="password", remote_root=trace_cluster.env["HOME"] + "/work")
    store = Store(tmp_path / "state.db")
    events = EventBus(store, session_id="sess1")
    registry = ClusterRegistry({name: profile}, store)
    service = Service(store, events, registry, "sess1")
    await service.acquire_lease()
    await service.start()
    server = build_server(service)
    try:
        # the Client is opened inside each test: anyio cancel scopes must be entered/exited in the same task
        yield {"server": server, "service": service, "home": trace_cluster.env["HOME"], "events": events}
    finally:
        await service.stop()
        await registry.close()
        store.close()


def _validate_against_schema(schema: dict, data: dict) -> None:
    props = schema.get("properties") or {}
    assert schema.get("type") == "object" and props, schema
    unknown = set(data) - set(props)
    assert not unknown, f"structured content has keys outside the output schema: {unknown}"
    for req in schema.get("required") or []:
        assert req in data, f"required {req} missing"


@pytest.mark.asyncio
async def test_every_tool_declares_contract(world):
  async with Client(world["server"]) as client:
    tools = (await client.list_tools()).tools
    names = {t.name for t in tools}
    assert set(PHASE2_TOOLS) <= names
    for t in tools:
        assert t.annotations is not None and isinstance(t.annotations.read_only_hint, bool), t.name
        assert isinstance(t.annotations.destructive_hint, bool), t.name
        assert t.description and len(t.description.encode("utf-8")) < MAX_DESCRIPTION_CHARS, t.name
        assert t.input_schema.get("type") == "object" and isinstance(t.input_schema.get("properties"), dict), t.name
        assert t.output_schema and "summary" in (t.output_schema.get("properties") or {}), t.name
        assert "summary" in (t.output_schema.get("required") or []), t.name
        assert "unread_events" in t.output_schema["properties"] and "next" in t.output_schema["properties"], t.name
    by = {t.name: t for t in tools}
    assert by["run_command"].annotations.destructive_hint is True and by["run_command"].meta == {"anthropic/maxResultSizeChars": 60000}
    assert by["remote_read"].meta == {"anthropic/maxResultSizeChars": 60000}
    assert by["clusters"].annotations.read_only_hint is True and by["remote_write"].annotations.destructive_hint is False
    assert by["cluster_status"].input_schema["properties"]["detail"]["default"] == "partitions"
    assert by["run_command"].input_schema["properties"]["timeout_s"]["default"] == 60
    assert by["remote_read"].input_schema["properties"]["max_chars"]["default"] == 12000
    assert len(INSTRUCTIONS.encode("utf-8")) <= 2048 and client.instructions == INSTRUCTIONS
    assert all("summary" in m.model_fields for m in RESULT_MODELS)


@pytest.mark.asyncio
async def test_phase2_tools_return_structured_content(world):
  async with Client(world["server"]) as c:
    home = world["home"]
    schemas = {t.name: t.output_schema for t in (await c.list_tools()).tools}
    calls = [
        ("configure", {}),
        ("clusters", {"refresh": True}),
        ("cluster_status", {"cluster": "fake-trace", "detail": "targets"}),
        ("run_command", {"cluster": "fake-trace", "command": "echo contract"}),
        ("remote_write", {"cluster": "fake-trace", "path": f"{home}/work/c.txt", "text": "a\nb\n"}),
        ("remote_read", {"cluster": "fake-trace", "path": f"{home}/work/c.txt", "tail_lines": 1}),
        ("remote_ls", {"cluster": "fake-trace", "path": f"{home}/work"}),
        ("clusters", {}),
    ]
    for name, args in calls:
        r = await c.call_tool(name, args)
        assert not r.is_error, (name, r.content)
        data = r.structured_content
        assert isinstance(data, dict) and data.get("summary"), (name, data)
        _validate_against_schema(schemas[name], data)
        model = PHASE2_TOOLS[name].model_validate(data)
        assert model.summary and isinstance(model.unread_events, int)
        if name == "run_command":
            assert data["rc"] == 0 and data["stdout_tail"] == "contract\n"
        if name == "remote_read":
            assert data["text"] == "b\n"
        if name == "cluster_status":
            assert data["targets"] and data["partitions"][0]["name"] == "batch"
    # unread_events reflects unacknowledged events of this session's client
    await world["events"].emit("completed", handle="j1", cluster="fake-trace", summary="j1 done", state="COMPLETED")
    r = await c.call_tool("configure", {})
    assert r.structured_content["unread_events"] == 1


@pytest.mark.asyncio
async def test_errors_are_tool_errors_with_codes(world):
  async with Client(world["server"]) as c:
    r = await c.call_tool("cluster_status", {"cluster": "mars"})
    assert r.is_error and "E_INVALID_SPEC:" in r.content[0].text and "fix:" in r.content[0].text
    r = await c.call_tool("run_command", {"cluster": "fake-trace", "command": "cat <<EOF\nx\nEOF"})
    assert r.is_error and "E_CMD_TOO_LONG:" in r.content[0].text and "remote_write" in r.content[0].text
    r = await c.call_tool("configure", {"placement": {"objective": "sideways"}})
    assert r.is_error and "E_INVALID_SPEC:" in r.content[0].text
    r = await c.call_tool("remote_read", {"cluster": "fake-trace", "path": f"{world['home']}/nope.txt"})
    assert r.is_error and r.content[0].text.startswith("Error executing tool remote_read: E_")
    r = await c.call_tool("clusters", {"refresh": "maybe"})
    assert r.is_error


@pytest.mark.asyncio
async def test_unbound_proxy_reports_starting_state():
    proxy = ServiceProxy()
    server = build_server(None)      # real app lifespan is not entered by Client? it is: the store opens under SLURM_MCP_HOME
    assert proxy.bound is False
    from slurm_mcp._mcp import ToolError
    with pytest.raises(ToolError) as e:
        proxy.clusters
    assert str(e.value).startswith("E_STATE")
    async with Client(server) as c:
        r = await c.call_tool("configure", {})
        assert not r.is_error and r.structured_content["placement"]["objective"] in ("balanced", "fastest", "cheapest")
