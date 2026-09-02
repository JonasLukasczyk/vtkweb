from __future__ import annotations

from dataclasses import dataclass

import vtk

from vtkweb.pipeline import PipelineGraph


@dataclass
class Representation:
    mapper: vtk.vtkMapper
    actor: vtk.vtkActor

    representation_mode: str = "surface"

    array_name: str | None = None
    association: str = "point"

    scalar_range: (
        tuple[float, float] | None
    ) = None


class RenderManager:
    def __init__(
        self,
        pipeline: PipelineGraph,
    ) -> None:
        self.pipeline = pipeline

        self.renderer = vtk.vtkRenderer()

        self.render_window = (
            vtk.vtkRenderWindow()
        )

        self.render_window.AddRenderer(
            self.renderer
        )

        self.render_window.SetOffScreenRendering(
            1
        )

        self.representations: dict[
            str,
            Representation,
        ] = {}

        for node in pipeline.nodes.values():
            self.add_representation(
                node.id
            )

        self.renderer.ResetCamera()

    def add_representation(
        self,
        node_id: str,
    ) -> Representation:
        node = self.pipeline.nodes[
            node_id
        ]

        mapper = vtk.vtkDataSetMapper()

        mapper.SetInputConnection(
            node.algorithm.GetOutputPort()
        )

        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        actor.SetVisibility(
            node.visible
        )

        self.renderer.AddActor(actor)

        representation = Representation(
            mapper=mapper,
            actor=actor,
        )

        self.representations[
            node_id
        ] = representation

        return representation

    def remove_representation(
        self,
        node_id: str,
    ) -> None:
        representation = (
            self.representations.pop(
                node_id
            )
        )

        self.renderer.RemoveActor(
            representation.actor
        )

    def set_visibility(
        self,
        node_id: str,
        visible: bool,
    ) -> None:
        self.pipeline.nodes[
            node_id
        ].visible = visible

        self.representations[
            node_id
        ].actor.SetVisibility(
            visible
        )

    def toggle_visibility(
        self,
        node_id: str,
    ) -> None:
        node = self.pipeline.nodes[
            node_id
        ]

        self.set_visibility(
            node_id,
            not node.visible,
        )

    def set_representation_mode(
        self,
        node_id: str,
        mode: str,
    ) -> None:
        representation = (
            self.representations[node_id]
        )

        representation.representation_mode = (
            mode
        )

        prop = (
            representation.actor.GetProperty()
        )

        if mode == "wireframe":
            prop.SetRepresentationToWireframe()
        else:
            prop.SetRepresentationToSurface()

    def get_arrays(
        self,
        node_id: str,
    ) -> dict[str, list[str]]:
        algorithm = self.pipeline.nodes[
            node_id
        ].algorithm

        algorithm.Update()

        data = algorithm.GetOutputDataObject(
            0
        )

        result = {
            "point": [],
            "cell": [],
        }

        if data is None:
            return result

        point_data = data.GetPointData()
        cell_data = data.GetCellData()

        for i in range(
            point_data.GetNumberOfArrays()
        ):
            name = (
                point_data.GetArrayName(i)
            )

            if name:
                result["point"].append(
                    name
                )

        for i in range(
            cell_data.GetNumberOfArrays()
        ):
            name = (
                cell_data.GetArrayName(i)
            )

            if name:
                result["cell"].append(
                    name
                )

        return result

    def set_array(
        self,
        node_id: str,
        array_name: str | None,
        association: str = "point",
    ) -> None:
        representation = (
            self.representations[node_id]
        )

        mapper = representation.mapper

        representation.array_name = (
            array_name
        )
        representation.association = (
            association
        )

        if array_name is None:
            mapper.ScalarVisibilityOff()
            return

        mapper.ScalarVisibilityOn()

        if association == "point":
            mapper.SetScalarModeToUsePointFieldData()
        else:
            mapper.SetScalarModeToUseCellFieldData()

        mapper.SelectColorArray(
            array_name
        )

        scalar_range = (
            self.get_array_range(
                node_id,
                array_name,
                association,
            )
        )

        if scalar_range is not None:
            self.set_scalar_range(
                node_id,
                *scalar_range,
            )

    def get_array_range(
        self,
        node_id: str,
        array_name: str,
        association: str = "point",
    ) -> tuple[float, float] | None:
        algorithm = self.pipeline.nodes[
            node_id
        ].algorithm

        algorithm.Update()

        data = algorithm.GetOutputDataObject(
            0
        )

        if data is None:
            return None

        if association == "point":
            attributes = (
                data.GetPointData()
            )
        else:
            attributes = (
                data.GetCellData()
            )

        array = attributes.GetArray(
            array_name
        )

        if array is None:
            return None

        minimum, maximum = (
            array.GetRange()
        )

        return (
            float(minimum),
            float(maximum),
        )

    def set_scalar_range(
        self,
        node_id: str,
        minimum: float,
        maximum: float,
    ) -> None:
        representation = (
            self.representations[node_id]
        )

        representation.scalar_range = (
            float(minimum),
            float(maximum),
        )

        representation.mapper.SetScalarRange(
            minimum,
            maximum,
        )
