from trame.app import get_server

from vtkweb.catalog import AlgorithmCatalog
from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering import RenderManager
from vtkweb.ui import build_ui
from vtkweb.views import ViewManager
from vtkweb.workspace import WorkspaceManager


server = get_server(client_type="vue3")

catalog = AlgorithmCatalog()

print(f"Discovered {len(catalog.algorithms)} algorithms")

pipeline = PipelineGraph(server.state)
rendering = RenderManager(server.state, pipeline)
views = ViewManager(server.state, rendering)
workspace = WorkspaceManager(server.state)

root_container = workspace.create_workspace(container_id="root")
default_view = views.create_view("vtk", name="View 1")
workspace.assign_view(root_container, default_view)
rendering.set_active_view(default_view)
rendering.reset_camera(default_view)

build_ui(
    server,
    pipeline,
    rendering,
    views,
    workspace,
    catalog,
)


if __name__ == "__main__":
    server.start()
