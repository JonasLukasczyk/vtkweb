from __future__ import annotations

from dataclasses import dataclass

import vtk

from vtkweb.rendering.base import (
    RenderingBackend,
    Representation,
    ViewSettings,
)


@dataclass
class VTKRepresentationHandle:
    mapper: vtk.vtkMapper
    actor: vtk.vtkActor
    kind: str
    pipeline_filter: vtk.vtkAlgorithm | None = None


class VTKRenderingBackend(
    RenderingBackend
):
    name = "vtk"

    def __init__(self) -> None:
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

        self._handles: dict[
            str,
            VTKRepresentationHandle,
        ] = {}

    def add_representation(
        self,
        representation: Representation,
        source: vtk.vtkAlgorithm,
    ) -> None:
        handle = self._create_handle(
            representation,
            source,
        )

        self._handles[
            representation.id
        ] = handle

        self.renderer.AddActor(
            handle.actor
        )

        self._apply_representation(
            representation,
            handle,
        )

    def update_representation(
        self,
        representation: Representation,
        source: vtk.vtkAlgorithm,
    ) -> None:
        handle = self._handles.get(
            representation.id
        )

        if handle is None:
            self.add_representation(
                representation,
                source,
            )
            return

        # Outline requires a different VTK pipeline,
        # so recreate the backend object when the type changes.
        if handle.kind != representation.kind:
            self.remove_representation(
                representation.id
            )

            self.add_representation(
                representation,
                source,
            )

            return

        self._apply_representation(
            representation,
            handle,
        )

    def remove_representation(
        self,
        representation_id: str,
    ) -> None:
        handle = self._handles.pop(
            representation_id,
            None,
        )

        if handle is None:
            return

        self.renderer.RemoveActor(
            handle.actor
        )

    def set_view_settings(
        self,
        settings: ViewSettings,
    ) -> None:
        self.renderer.SetBackground(
            *settings.background_color
        )

    def reset_camera(self) -> None:
        self.renderer.ResetCamera()

    def _create_handle(
        self,
        representation: Representation,
        source: vtk.vtkAlgorithm,
    ) -> VTKRepresentationHandle:
        mapper = vtk.vtkDataSetMapper()

        pipeline_filter = None

        if representation.kind == "outline":
            pipeline_filter = (
                vtk.vtkOutlineFilter()
            )

            pipeline_filter.SetInputConnection(
                source.GetOutputPort(0)
            )

            mapper.SetInputConnection(
                pipeline_filter.GetOutputPort(0)
            )

        else:
            mapper.SetInputConnection(
                source.GetOutputPort(0)
            )

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return VTKRepresentationHandle(
            mapper=mapper,
            actor=actor,
            kind=representation.kind,
            pipeline_filter=pipeline_filter,
        )

    def _apply_representation(
        self,
        representation: Representation,
        handle: VTKRepresentationHandle,
    ) -> None:
        actor = handle.actor
        mapper = handle.mapper

        actor.SetVisibility(
            1 if representation.visible else 0
        )

        prop = actor.GetProperty()

        if representation.kind == "wireframe":
            prop.SetRepresentationToWireframe()
        else:
            prop.SetRepresentationToSurface()

        if (
            representation.kind == "outline"
            or representation.array_name is None
        ):
            mapper.ScalarVisibilityOff()
            mapper.Modified()
            actor.Modified()
            return

        mapper.ScalarVisibilityOn()

        if representation.association == "point":
            mapper.SetScalarModeToUsePointFieldData()
        else:
            mapper.SetScalarModeToUseCellFieldData()

        mapper.SelectColorArray(
            representation.array_name
        )

        mapper.UseLookupTableScalarRangeOff()

        if representation.scalar_range is not None:
            mapper.SetScalarRange(
                *representation.scalar_range
            )

        mapper.Modified()
        actor.Modified()
