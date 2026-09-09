from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

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
    lookup_table: vtk.vtkLookupTable | None = None
    color_data: Any | None = None


class VTKRenderingBackend(RenderingBackend):
    name = "vtk"

    def __init__(self, state=None) -> None:
        self.state = state
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
        lookup_table = vtk.vtkLookupTable()
        lookup_table.SetNumberOfTableValues(256)
        lookup_table.Build()
        mapper.SetLookupTable(lookup_table)

        return VTKRepresentationHandle(
            mapper=mapper,
            actor=actor,
            kind=representation.kind,
            pipeline_filter=pipeline_filter,
            lookup_table=lookup_table,
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

            color_by = properties.get("color_by")
            tf = self._transfer_function(color_by)
            if color_by is not None:
                array_name, association = color_by
                source_data = mapper.GetInputDataObject(0, 0)
                color_data, selected_name = self._data_for_coloring(
                    source_data, array_name, association
                )
                if color_data is not None:
                    handle.color_data = color_data
                    mapper.SetInputDataObject(color_data)
                if association == "cell":
                    mapper.SetScalarModeToUseCellFieldData()
                else:
                    mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectScalarArray(selected_name)

            minimum, maximum = tf["range"]
            if maximum <= minimum:
                maximum = minimum + 1.0
            color_function = handle.color_function
            opacity_function = handle.opacity_function
            if color_function is not None:
                color_function.RemoveAllPoints()
            if opacity_function is not None:
                opacity_function.RemoveAllPoints()
            for t, r, g, b, _opacity in tf["control_points"]:
                x = minimum + float(t) * (maximum - minimum)
                if color_function is not None:
                    color_function.AddRGBPoint(x, r, g, b)
                if opacity_function is not None:
                    opacity_function.AddPoint(x, 1.0)
            if color_function is not None:
                color_function.Modified()
            if opacity_function is not None:
                opacity_function.Modified()

            mapper.Modified()
            volume_property.Modified()
            actor.Modified()
            return

        if representation.kind == "wireframe":
            prop.SetRepresentationToWireframe()
        else:
            prop.SetRepresentationToSurface()

        color_by = properties.get("color_by")
        if representation.kind == "outline" or color_by is None:
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

        array_name, association = color_by
        source_data = mapper.GetInputDataObject(0, 0)
        color_data, selected_name = self._data_for_coloring(
            source_data, array_name, association
        )
        if color_data is not None:
            handle.color_data = color_data
            mapper.SetInputDataObject(color_data)

        mapper.ScalarVisibilityOn()
        if association == "point":
            mapper.SetScalarModeToUsePointFieldData()
        else:
            mapper.SetScalarModeToUseCellFieldData()
        mapper.SelectColorArray(selected_name)

        tf = self._transfer_function(color_by)
        minimum, maximum = tf["range"]
        if maximum <= minimum:
            maximum = minimum + 1.0
        lut = handle.lookup_table
        if lut is not None:
            lut.SetRange(minimum, maximum)
            lut.SetNumberOfTableValues(256)
            for i in range(256):
                t = i / 255.0
                r, g, b = self._sample_tf(tf["control_points"], t)
                lut.SetTableValue(i, r, g, b, 1.0)
            lut.Build()
            mapper.SetLookupTable(lut)
        mapper.UseLookupTableScalarRangeOn()
        mapper.Modified()
        actor.Modified()

    def _transfer_function(self, color_by) -> dict:
        default = {
            "control_points": [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            "range": [0.0, 1.0],
        }
        if color_by is None or self.state is None:
            return default
        return self.state.transfer_functions.get(color_by[0], default)

    @staticmethod
    def _sample_tf(points, t: float) -> tuple[float, float, float]:
        points = sorted(points, key=lambda p: p[0])
        if t <= points[0][0]:
            return tuple(float(v) for v in points[0][1:4])
        if t >= points[-1][0]:
            return tuple(float(v) for v in points[-1][1:4])
        for left, right in zip(points, points[1:]):
            if left[0] <= t <= right[0]:
                span = right[0] - left[0]
                u = 0.0 if span <= 0 else (t - left[0]) / span
                return tuple(
                    float(left[i] + u * (right[i] - left[i])) for i in range(1, 4)
                )
        return tuple(float(v) for v in points[-1][1:4])

    @staticmethod
    def _data_for_coloring(data, array_name: str, association: str):
        """Return data plus scalar name, materializing vector magnitude if needed."""
        if data is None or not isinstance(data, vtk.vtkDataSet):
            return data, array_name
        attributes = (
            data.GetPointData() if association == "point" else data.GetCellData()
        )
        array = attributes.GetArray(array_name)
        if array is None or array.GetNumberOfComponents() <= 1:
            return data, array_name

        copied = data.NewInstance()
        copied.ShallowCopy(data)
        target = (
            copied.GetPointData() if association == "point" else copied.GetCellData()
        )
        magnitude_name = f"__vtkweb_magnitude_{array_name}"
        magnitude = vtk.vtkDoubleArray()
        magnitude.SetName(magnitude_name)
        magnitude.SetNumberOfComponents(1)
        magnitude.SetNumberOfTuples(array.GetNumberOfTuples())
        components = array.GetNumberOfComponents()
        for i in range(array.GetNumberOfTuples()):
            value = math.sqrt(
                sum(float(array.GetComponent(i, c)) ** 2 for c in range(components))
            )
            magnitude.SetValue(i, value)
        target.AddArray(magnitude)
        return copied, magnitude_name
