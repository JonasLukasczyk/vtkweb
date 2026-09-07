from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

    keepalive_source: vtk.vtkSphereSource
    keepalive_mapper: vtk.vtkPolyDataMapper
    keepalive_actor: vtk.vtkActor


@dataclass
class VTKRepresentationHandle:
    mapper: Any
    actor: Any
    kind: str

    pipeline_filter: vtk.vtkAlgorithm | None = None
    color_function: vtk.vtkColorTransferFunction | None = None
    opacity_function: vtk.vtkPiecewiseFunction | None = None


class VTKRenderingBackend(RenderingBackend):
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

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)

        render_window.SetOffScreenRendering(1)

        # ---------------------------------------------------------------------
        # VtkLocalView keepalive workaround
        # ---------------------------------------------------------------------
        #
        # VtkLocalView / vtk.js can end up without a usable current renderer
        # when the scene becomes completely empty.
        #
        # Keep one tiny backend-private actor in every render view so the
        # renderer never becomes empty.
        # ---------------------------------------------------------------------

        keepalive_source = vtk.vtkSphereSource()

        keepalive_source.SetCenter(
            0.0,
            0.0,
            0.0,
        )

        keepalive_source.SetRadius(0.1)

        keepalive_source.SetThetaResolution(8)

        keepalive_source.SetPhiResolution(8)

        keepalive_source.Update()
        keepalive_mapper = vtk.vtkPolyDataMapper()
        keepalive_mapper.SetInputDataObject(keepalive_source.GetOutputDataObject(0))

        keepalive_actor = vtk.vtkActor()

        keepalive_actor.SetMapper(keepalive_mapper)

        # Keep this visible for now so we can verify that the workaround
        # actually fixes the empty-scene issue.
        keepalive_actor.GetProperty().SetOpacity(1.0)

        keepalive_actor.SetPickable(False)

        renderer.AddActor(keepalive_actor)

        self._views[view.id] = VTKViewHandle(
            renderer=renderer,
            render_window=render_window,
            keepalive_source=keepalive_source,
            keepalive_mapper=keepalive_mapper,
            keepalive_actor=keepalive_actor,
        )

        self.set_view_settings(view)

    def remove_view(
        self,
        view_id: str,
    ) -> None:
        keys = [key for key in self._representations if key[1] == view_id]

        for representation_id, _ in keys:
            self.remove_representation(
                representation_id,
                view_id,
            )

        self._views.pop(
            view_id,
            None,
        )

    def rename_view(
        self,
        view_id: str,
        new_view_id: str,
    ) -> None:
        if new_view_id != view_id and new_view_id in self._views:
            raise ValueError(f"View ID already exists: {new_view_id}")

        handle = self._views.pop(view_id)
        self._views[new_view_id] = handle

        renamed = {}
        for (
            representation_id,
            current_view_id,
        ), representation in self._representations.items():
            key = (
                representation_id,
                new_view_id if current_view_id == view_id else current_view_id,
            )
            renamed[key] = representation

        self._representations = renamed

    def get_render_window(
        self,
        view_id: str,
    ) -> vtk.vtkRenderWindow:
        return self._views[view_id].render_window

    def set_view_settings(
        self,
        view: RenderView,
    ) -> None:
        handle = self._views[view.id]

        handle.renderer.SetBackground(*view.settings.background_color)

        handle.renderer.Modified()
        handle.render_window.Modified()

    def reset_camera(
        self,
        view_id: str,
    ) -> None:
        handle = self._views[view_id]

        handle.renderer.ResetCamera()
        handle.renderer.ResetCameraClippingRange()

        handle.renderer.Modified()
        handle.render_window.Modified()

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

        self._representations[key] = handle

        view_handle = self._views[view.id]

        self._apply_representation(
            representation,
            handle,
        )

        if self._source_has_geometry(
            source,
            representation.output_port,
        ):
            if representation.kind == "volume":
                view_handle.renderer.AddVolume(handle.actor)
            else:
                view_handle.renderer.AddActor(handle.actor)

        view_handle.renderer.Modified()
        view_handle.render_window.Modified()

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

        handle = self._representations.get(key)

        if handle is not None:
            source_data = source.GetOutputDataObject(representation.output_port)
            if handle.pipeline_filter is not None:
                if source_data is not None:
                    handle.pipeline_filter.SetInputDataObject(source_data)
                    handle.pipeline_filter.Update()
                    handle.mapper.SetInputDataObject(
                        handle.pipeline_filter.GetOutputDataObject(0)
                    )
            elif source_data is not None:
                handle.mapper.SetInputDataObject(source_data)

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

        view_handle = self._views[view.id]

        has_geometry = self._source_has_geometry(
            source,
            representation.output_port,
        )

        has_actor = bool(view_handle.renderer.HasViewProp(handle.actor))

        if has_geometry and not has_actor:
            if representation.kind == "volume":
                view_handle.renderer.AddVolume(handle.actor)
            else:
                view_handle.renderer.AddActor(handle.actor)

        elif not has_geometry and has_actor:
            view_handle.renderer.RemoveViewProp(handle.actor)

        view_handle.renderer.Modified()
        view_handle.render_window.Modified()

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

        view = self._views.get(view_id)

        if view is None:
            return

        if view.renderer.HasViewProp(handle.actor):
            view.renderer.RemoveViewProp(handle.actor)

        # Keepalive actor remains, so this renderer never becomes empty.
        view.renderer.Modified()
        view.render_window.Modified()

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _source_has_geometry(
        self,
        source: vtk.vtkAlgorithm,
        output_port: int,
    ) -> bool:
        # Do not execute an unconfigured file-backed source merely because a
        # representation was created for it. Many VTK readers expose
        # GetFileName() and emit pipeline errors when Update() is called before
        # a filename is assigned. Once FileName changes, RenderManager.refresh_node
        # calls back into this method and the source is executed normally.
        get_file_name = getattr(source, "GetFileName", None)
        if source.GetNumberOfInputPorts() == 0 and callable(get_file_name):
            try:
                if not get_file_name():
                    return False
            except Exception:
                pass

        output = source.GetOutputDataObject(output_port)

        if output is None:
            return False

        if isinstance(
            output,
            vtk.vtkDataSet,
        ):
            return output.GetNumberOfPoints() > 0 and output.GetNumberOfCells() > 0

        # For non-vtkDataSet outputs, fall back to assuming that an existing
        # data object is renderable.
        return True

    def _create_handle(
        self,
        representation: Representation,
        source: vtk.vtkAlgorithm,
    ) -> VTKRepresentationHandle:
        source_data = source.GetOutputDataObject(representation.output_port)

        if representation.kind == "volume":
            mapper = vtk.vtkSmartVolumeMapper()
            if source_data is not None:
                mapper.SetInputDataObject(source_data)

            color_function = vtk.vtkColorTransferFunction()
            opacity_function = vtk.vtkPiecewiseFunction()
            volume_property = vtk.vtkVolumeProperty()
            volume_property.SetColor(color_function)
            volume_property.SetScalarOpacity(opacity_function)

            actor = vtk.vtkVolume()
            actor.SetMapper(mapper)
            actor.SetProperty(volume_property)

            return VTKRepresentationHandle(
                mapper=mapper,
                actor=actor,
                kind=representation.kind,
                color_function=color_function,
                opacity_function=opacity_function,
            )

        mapper = vtk.vtkDataSetMapper()
        pipeline_filter = None

        if representation.kind == "outline":
            pipeline_filter = vtk.vtkOutlineFilter()
            if source_data is not None:
                pipeline_filter.SetInputDataObject(source_data)
                pipeline_filter.Update()
                mapper.SetInputDataObject(pipeline_filter.GetOutputDataObject(0))
        elif source_data is not None:
            mapper.SetInputDataObject(source_data)

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
        properties = representation.properties
        mapper = handle.mapper
        actor = handle.actor
        prop = actor.GetProperty()

        # If a concrete backend representation exists for this view,
        # it is visible by definition.
        actor.SetVisibility(1)

        if representation.kind == "volume":
            volume_property = actor.GetProperty()

            if properties.get("interpolation", "linear") == "nearest":
                volume_property.SetInterpolationTypeToNearest()
            else:
                volume_property.SetInterpolationTypeToLinear()

            if properties.get("shade", True):
                volume_property.ShadeOn()
            else:
                volume_property.ShadeOff()
            volume_property.SetAmbient(float(properties.get("ambient", 0.1)))
            volume_property.SetDiffuse(float(properties.get("diffuse", 0.9)))
            volume_property.SetSpecular(float(properties.get("specular", 0.2)))
            volume_property.SetSpecularPower(
                float(properties.get("specular_power", 10.0))
            )

            if properties.get("blend_mode", "composite") == "maximum":
                mapper.SetBlendModeToMaximumIntensity()
            elif properties.get("blend_mode", "composite") == "minimum":
                mapper.SetBlendModeToMinimumIntensity()
            else:
                mapper.SetBlendModeToComposite()

            mapper.SetAutoAdjustSampleDistances(
                1 if properties.get("auto_adjust_sample_distances", True) else 0
            )
            mapper.SetSampleDistance(
                max(1e-12, float(properties.get("sample_distance", 1.0)))
            )

            if hasattr(mapper, "SetGlobalIlluminationReach"):
                mapper.SetGlobalIlluminationReach(
                    max(
                        0.0,
                        min(
                            1.0, float(properties.get("global_illumination_reach", 0.0))
                        ),
                    )
                )
            if hasattr(mapper, "SetVolumetricScatteringBlending"):
                mapper.SetVolumetricScatteringBlending(
                    max(
                        0.0,
                        min(
                            1.0,
                            float(
                                properties.get("volumetric_scattering_blending", 0.0)
                            ),
                        ),
                    )
                )

            if properties.get("scalar_array") is not None:
                if properties.get("scalar_association", "point") == "cell":
                    mapper.SetScalarModeToUseCellFieldData()
                else:
                    mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectScalarArray(properties.get("scalar_array"))
                if hasattr(mapper, "SetVectorMode"):
                    mapper.SetVectorMode(0)
                    mapper.SetVectorComponent(
                        max(0, int(properties.get("scalar_component", 0)))
                    )

            scalar_range = properties.get("scalar_range") or (0.0, 1.0)
            minimum, maximum = scalar_range
            if maximum <= minimum:
                maximum = minimum + 1.0

            color_function = handle.color_function
            opacity_function = handle.opacity_function
            if color_function is not None:
                color_function.RemoveAllPoints()
                color_function.AddRGBPoint(minimum, 0.0, 0.0, 0.0)
                color_function.AddRGBPoint(maximum, 1.0, 1.0, 1.0)
                color_function.Modified()
            if opacity_function is not None:
                opacity_function.RemoveAllPoints()
                opacity_function.AddPoint(minimum, 0.0)
                opacity_function.AddPoint(maximum, 1.0)
                opacity_function.Modified()

            mapper.Modified()
            volume_property.Modified()
            actor.Modified()
            return

        if representation.kind == "wireframe":
            prop.SetRepresentationToWireframe()
        else:
            prop.SetRepresentationToSurface()

        if representation.kind == "outline" or properties.get("scalar_array") is None:
            mapper.ScalarVisibilityOff()

            if representation.kind != "outline":
                color = properties.get("color", "#ffffff").lstrip("#")
                prop.SetColor(
                    int(color[0:2], 16) / 255.0,
                    int(color[2:4], 16) / 255.0,
                    int(color[4:6], 16) / 255.0,
                )

            mapper.Modified()
            actor.Modified()

            return

        mapper.ScalarVisibilityOn()

        if properties.get("scalar_association", "point") == "point":
            mapper.SetScalarModeToUsePointFieldData()
        else:
            mapper.SetScalarModeToUseCellFieldData()

        mapper.SelectColorArray(properties.get("scalar_array"))

        mapper.UseLookupTableScalarRangeOff()

        if properties.get("scalar_range") is not None:
            mapper.SetScalarRange(*properties.get("scalar_range"))

        mapper.Modified()
        actor.Modified()
