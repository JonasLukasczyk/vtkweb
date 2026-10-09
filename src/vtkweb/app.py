from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from trame.app import get_server

from vtkweb.app_controller import initialize_app_controller
from vtkweb.catalog import AlgorithmCatalog
from vtkweb.distributed import context as distributed
from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering import RenderManager
from vtkweb.stream_hub import StreamHub, StreamSink
from vtkweb.views import ViewManager
from vtkweb.workspace import WorkspaceManager


server = get_server(client_type="vue3")

catalog = AlgorithmCatalog()

if distributed.is_root:
    print(
        f"Discovered {len(catalog.algorithms)} algorithms "
        f"(MPI ranks: {distributed.size})",
        flush=True,
    )

pipeline = PipelineGraph(server.state)
stream_hub = StreamHub(server) if distributed.is_root else None
stream_sink = StreamSink(stream_hub)
rendering = RenderManager(server.state, pipeline, stream_sink=stream_sink)
views = ViewManager(server.state, rendering)
workspace = WorkspaceManager(server.state)

# Bootstrap IDs must be identical on every rank because subsequent replicated
# mutations refer to logical view IDs, not rank-local runtime objects.
root_container = workspace.create_workspace(container_id="root")
default_view = views.create_view("mitsuba", name="View 1", view_id="default-view")
# The bootstrap view is the cluster-rendered primary view. Newly-created views
# default to rank-0-only rendering until Distributed Rendering is enabled.
rendering.set_view_property(default_view, "distributed", distributed.enabled)
workspace.assign_view(root_container, default_view)
rendering.set_active_view(default_view)

if distributed.is_root:
    from vtkweb.ui import build_ui

    build_ui(
        server,
        pipeline,
        rendering,
        views,
        workspace,
        catalog,
    )
else:
    # Workers need the exact same mutation handlers, but no browser-facing UI
    # or HTTP/WebSocket server.
    initialize_app_controller(
        server,
        pipeline,
        rendering,
        views,
        workspace,
        catalog,
    )


async def _run_worker() -> None:
    rendering.frames.ensure_all()
    await distributed.worker_loop()


def _startup_state_path(argv: list[str]) -> Path | None:
    """Consume a final .py state argument without altering Trame flags."""
    if len(argv) < 2 or not argv[-1].lower().endswith(".py"):
        return None
    candidate = Path(argv[-1]).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"vtkweb startup state file not found: {candidate}")
    argv.pop()
    return candidate


if __name__ == "__main__":
    state_path = _startup_state_path(sys.argv)
    if state_path is not None:
        # The UI/controller has already been registered above. State files
        # execute trusted Python and must never be loaded from untrusted input.
        server.controller.open_python_state_file(str(state_path))
        if distributed.is_root:
            print(f"Loaded vtkweb state: {state_path}", flush=True)
    if distributed.is_root:
        server.start()
    else:
        asyncio.run(_run_worker())
