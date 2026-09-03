from __future__ import annotations

from dataclasses import dataclass

import vtk

from vtkweb.rendering.base import (
    RenderView,
    RenderingBackend,
    Representation,
)


@dataclass
class VTKViewHandle:
    renderer: vtk.vtkRenderer
    render_window: vtk.vtkRenderWindow


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
        self._views: dict[
            str,
            VTKViewHandle,
        ] = {}

        self._representations: dict[
            tuple[str, str],
            VTKRepresentationHandle,
        ] = {}

    # -------------------------------------------------------------------------
    # Views
    # -------------------------------------------------------------------------

    def add_view(
        self,
        view: RenderView,
    ) -> None:
        renderer = vtk.vtkRenderer()

        render_window = (
            vtk.vtkRenderWindow()
        )

        render_window.AddRenderer(
            renderer
        )

        render_window.SetOffScreenRendering(
            1
        )

        self._views[
            view.id
        ] = VTKViewHandle(
            renderer=renderer,
            render_window=render_window,
        )

        self.set_view_settings(
            view
        )

    def remove_view(
        self,
        view_id: str,
    ) -> None:
        keys = [
            key
            for key in self._representations
            if key[1] == view_id
        ]

        for representation_id, _ in keys:
            self.remove_representation(
                representation_id,
                view_id,
            )

        self._views.pop(
            view_id,
            None,
        )

    def get_render_window(
        self,
        view_id: str,
    ) -> vtk.vtkRenderWindow:
        return self._views[
            view_id
        ].render_window

    def set_view_settings(
        self,
        view: RenderView,
    ) -> None:
        handle = self._views[
            view.id
        ]

        handle.renderer.SetBackground(
            *view.settings.background_color
        )

    def reset_camera(
        self,
        view_id: str,
    ) -> None:
        self._views[
            view_id
        ].renderer.ResetCamera()

    # -------------------------------------------------------------------------
    # Representations
    # -------------------------------------------------------------------------

    def add_representation(
        self,
        representation: Representation,
        view: RenderView,
        source: vtk.vtkAlgorithm,
    ) -> None:
        key = (
            representation.id,
            view.id,
        )

        if key in self._representations:
            return

        handle = self._create_handle(
            representation,
            source,
        )

        self._representations[
            key
        ] = handle

        self._views[
            view.id
        ].renderer.AddActor(
            handle.actor
        )

        self._apply_representation(
            representation,
            handle,
        )

    def update_representation(
        self,
        representation: Representation,
        view: RenderView,
        source: vtk.vtkAlgorithm,
    ) -> None:
        key = (
            representation.id,
            view.id,
        )

        handle = self._representations.get(
            key
        )

        if handle is None:
            self.add_representation(
                representation,
                view,
                source,
            )
            return

        if handle.kind != representation.kind:
            self.remove_representation(
                representation.id,
                view.id,
            )

            self.add_representation(
                representation,
                view,
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
        view_id: str,
    ) -> None:
        key = (
            representation_id,
            view_id,
        )

        handle = self._representations.pop(
            key,
            None,
        )

        if handle is None:
            return

        view = self._views.get(
            view_id
        )

        if view is not None:
            view.renderer.RemoveActor(
                handle.actor
            )

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _create_handle(
        self,
        representation: Representation,
        source: vtk.vtkAlgorithm,
    ) -> VTKRepresentationHandle:
        mapper = vtk.vtkDataSetMapper()

        pipeline_filter = None

        source_port = source.GetOutputPort(
            representation.output_port
        )

        if representation.kind == "outline":
            pipeline_filter = (
                vtk.vtkOutlineFilter()
            )

            pipeline_filter.SetInputConnection(
                source_port
            )

            mapper.SetInputConnection(
                pipeline_filter.GetOutputPort()
            )

        else:
            mapper.SetInputConnection(
                source_port
            )

        actor = vtk.vtkActor()
        actor.SetMapper(
            mapper
        )

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
        mapper = handle.mapper
        actor = handle.actor
        prop = actor.GetProperty()

        actor.SetVisibility(
            1
        )

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
