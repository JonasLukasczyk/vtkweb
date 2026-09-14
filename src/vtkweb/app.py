from __future__ import annotations

import asyncio

from trame.app import get_server

from vtkweb.app_controller import initialize_app_controller
from vtkweb.catalog import AlgorithmCatalog
from vtkweb.distributed import context as distributed
from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering import RenderManager
from vtkweb.rendering.distributed_transport import DistributedFrameTransport
from vtkweb.rendering.frame_transport import WebSocketFrameTransport
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
websocket_transport = WebSocketFrameTransport(server) if distributed.is_root else None
frame_transport = DistributedFrameTransport(websocket_transport)
rendering = RenderManager(server.state, pipeline, frame_transport=frame_transport)
views = ViewManager(server.state, rendering)
workspace = WorkspaceManager(server.state)

# Bootstrap IDs must be identical on every rank because subsequent replicated
# mutations refer to logical view IDs, not rank-local runtime objects.
root_container = workspace.create_workspace(container_id="root")
default_view = views.create_view("vtk", name="View 1", view_id="default-view")
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


if __name__ == "__main__":
    if distributed.is_root:
        server.start()
    else:
        asyncio.run(_run_worker())
