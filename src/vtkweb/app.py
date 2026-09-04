from trame.app import get_server

from vtkweb.catalog import AlgorithmCatalog
from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering import RenderManager
from vtkweb.ui import build_ui


server = get_server(client_type="vue3")

catalog = AlgorithmCatalog()

print(f"Discovered {len(catalog.algorithms)} algorithms")

pipeline = PipelineGraph(server.state)

rendering = RenderManager(
    server.state,
    pipeline,
)

build_ui(
    server,
    pipeline,
    rendering,
    catalog,
)


if __name__ == "__main__":
    server.start()
