import pytest
import vtk

from vtkweb.pipeline import PipelineGraph


def test_create_pipeline():
    graph = PipelineGraph()

    sphere = graph.add_node(vtk.vtkSphereSource())
    elevation = graph.add_node(vtk.vtkElevationFilter())

    edge = graph.connect(
        sphere.id,
        elevation.id,
    )

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1

    assert edge.source_node_id == sphere.id
    assert edge.target_node_id == elevation.id

    assert elevation.algorithm.GetNumberOfInputConnections(0) == 1


def test_pipeline_executes():
    graph = PipelineGraph()

    sphere = graph.add_node(vtk.vtkSphereSource())
    elevation = graph.add_node(vtk.vtkElevationFilter())

    graph.connect(sphere.id, elevation.id)

    elevation.algorithm.Update()

    output = elevation.algorithm.GetOutput()

    assert output is not None
    assert output.GetNumberOfPoints() > 0


def test_disconnect():
    graph = PipelineGraph()

    sphere = graph.add_node(vtk.vtkSphereSource())
    elevation = graph.add_node(vtk.vtkElevationFilter())

    edge = graph.connect(sphere.id, elevation.id)

    assert elevation.algorithm.GetNumberOfInputConnections(0) == 1

    graph.disconnect(edge)

    assert elevation.algorithm.GetNumberOfInputConnections(0) == 0


def test_invalid_port_is_rejected():
    graph = PipelineGraph()

    sphere = graph.add_node(vtk.vtkSphereSource())
    elevation = graph.add_node(vtk.vtkElevationFilter())

    with pytest.raises(ValueError):
        graph.connect(
            sphere.id,
            elevation.id,
            source_port=123,
        )


def test_cycle_is_rejected():
    graph = PipelineGraph()

    source = graph.add_node(vtk.vtkSphereSource())
    first = graph.add_node(vtk.vtkElevationFilter())
    second = graph.add_node(vtk.vtkElevationFilter())

    graph.connect(source.id, first.id)
    graph.connect(first.id, second.id)

    with pytest.raises(ValueError, match="cycle"):
        graph.connect(second.id, first.id)
