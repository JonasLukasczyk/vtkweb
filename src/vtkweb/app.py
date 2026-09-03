import vtk

from trame.app import get_server

from vtkweb.catalog import AlgorithmCatalog
from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering import RenderManager
from vtkweb.ui import build_ui


server = get_server(client_type="vue3")

catalog = AlgorithmCatalog()

print(f"Discovered {len(catalog.algorithms)} algorithms")

pipeline = PipelineGraph(server.state)

rt = pipeline.add_node(
    vtk.vtkRTAnalyticSource(),
    name="RT Analytic Source",
)

contour = pipeline.add_node(
    vtk.vtkContourFilter(),
    name="Contour",
)

pipeline.connect(
    rt.id,
    contour.id,
)

# # -------------------------------------------------------------------------
# # Hard-coded contour setup for debugging
# # -------------------------------------------------------------------------
#
# rt.algorithm.Update()
#
# rt_output = rt.algorithm.GetOutput()
# rt_data = rt_output.GetPointData().GetArray("RTData")
#
# print(
#     "RT source points:",
#     rt_output.GetNumberOfPoints(),
# )
#
# print(
#     "RTData range:",
#     rt_data.GetRange(),
# )
#
# minimum, maximum = rt_data.GetRange()
# iso_value = 0.5 * (minimum + maximum)
#
# contour.algorithm.SetInputArrayToProcess(
#     0,
#     0,
#     0,
#     vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
#     "RTData",
# )
#
# contour.algorithm.SetNumberOfContours(1)
# contour.algorithm.SetValue(
#     0,
#     iso_value,
# )
#
# print(
#     "Contour value:",
#     contour.algorithm.GetValue(0),
# )
#
# contour.algorithm.Update()
#
# contour_output = contour.algorithm.GetOutput()
#
# print(
#     "Contour output points:",
#     contour_output.GetNumberOfPoints(),
# )
#
# print(
#     "Contour output cells:",
#     contour_output.GetNumberOfCells(),
# )

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
