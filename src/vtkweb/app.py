import vtk

from trame.app import get_server

from vtkweb.catalog import AlgorithmCatalog
from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering import RenderManager
from vtkweb.ui import build_ui


server = get_server(
    client_type="vue3"
)

catalog = AlgorithmCatalog()

pipeline = PipelineGraph()

sphere = pipeline.add_node(
    vtk.vtkSphereSource(),
    name="Sphere",
    visible=True,
)

elevation = pipeline.add_node(
    vtk.vtkElevationFilter(),
    name="Elevation",
)

pipeline.connect(
    sphere.id,
    elevation.id,
)

sphere.algorithm.SetThetaResolution(48)
sphere.algorithm.SetPhiResolution(48)

rendering = RenderManager(pipeline)

build_ui(
    server,
    pipeline,
    rendering,
    catalog,
)


if __name__ == "__main__":
    server.start()
