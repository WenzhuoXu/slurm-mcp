"""AllocManager.allocate must bridge the two progress-callback shapes it sits between.

`tools/alloc.py` builds one callback, `cb(fraction, message)`, and uses it for both allocate and
alloc_run. alloc_run's polling loop calls it that way, but `Submitter.await_result` calls its
`progress_cb` with the message alone -- and it does not swallow the resulting TypeError the way
alloc's `_call` does, so handing the callback straight through failed the whole allocation with
"E_SSH: TypeError: cb() missing 1 required positional argument: 'message'".
"""
from __future__ import annotations

from typing import Any

import pytest

from slurm_mcp.alloc import AllocManager
from slurm_mcp.models import Resources, SubmitResult
from slurm_mcp.slurm.states import JobState


class _Submitter:
    """Stands in for Submitter, calling progress_cb the way await_result really does: message only."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def submit(self, spec: Any, **kw: Any) -> tuple[str, None]:
        return "a1", None

    async def await_result(self, handle: str, wait_s: float = 90, progress_cb: Any = None,
                           **kw: Any) -> SubmitResult:
        if progress_cb is not None:
            await progress_cb("submitting")          # one argument, exactly like the real submitter
        return SubmitResult(summary="", handle=handle, kind="alloc", cluster="trace",
                            state=JobState.RUNNING, target="trace:cpuonly-debug")


class _Service:
    def __init__(self, submitter: _Submitter) -> None:
        self.components = {"submitter": submitter}

    async def caps(self, cluster: str) -> dict[str, Any]:
        return {"partitions": {}}

    def profile(self, cluster: str) -> None:
        return None


@pytest.mark.asyncio
async def test_allocate_adapts_the_tool_layers_two_arg_progress_callback() -> None:
    submitter = _Submitter()
    mgr = AllocManager(_Service(submitter))
    seen: list[tuple[float, str]] = []

    async def tool_layer_cb(fraction: float, message: str) -> None:   # the shape tools/alloc.py builds
        seen.append((fraction, message))

    result = await mgr.allocate("trace", Resources(time="00:10:00", cpus=2), hours=0.25,
                                progress=tool_layer_cb)

    assert result.handle == "a1"
    assert seen == [(0.5, "submitting")], "the two-arg tool callback should still be driven"


@pytest.mark.asyncio
async def test_allocate_without_progress_passes_none_through() -> None:
    submitter = _Submitter()
    mgr = AllocManager(_Service(submitter))
    result = await mgr.allocate("trace", Resources(time="00:10:00", cpus=2), hours=0.25)
    assert result.handle == "a1"
