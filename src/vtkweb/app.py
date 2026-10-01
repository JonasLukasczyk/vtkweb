from __future__ import annotations

import asyncio

from trame.app import get_server

from vtkweb.activity import DistributedActivityReporter
from vtkweb.app_controller import initialize_app_controller
from vtkweb.catalog import AlgorithmCatalog
from vtkweb.distributed import context as distributed
from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering import RenderManager
from vtkweb.rendering.distributed_transport import DistributedFrameTransport
from vtkweb.rendering.frame_transport import H264WebSocketFrameTransport
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
video_transport = H264WebSocketFrameTransport(server) if distributed.is_root else None
activity_reporter = DistributedActivityReporter(
    server if distributed.is_root else None,
    publisher=video_transport.publish_message if video_transport is not None else None,
)
frame_transport = DistributedFrameTransport(video_transport, activity_reporter=activity_reporter)
rendering = RenderManager(
    server.state,
    pipeline,
    frame_transport=frame_transport,
    activity_reporter=activity_reporter,
)

if distributed.is_root:
    # Start the existing MPI receiver as soon as the server event loop is bound,
    # so bake progress is visible even before the first rendered tile arrives.
    server.controller.on_server_bind.add(lambda _http_server: frame_transport.ensure_receiver())
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


if __name__ == "__main__":
    if distributed.is_root:
        server.start()
    else:
        asyncio.run(_run_worker())
