from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

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
    coloring_data: vtk.vtkDataObject | None = None


class VTKRenderingBackend(RenderingBackend):
    name = "vtk"

    def __init__(
        self,
        transfer_function_provider: Callable[[str], dict[str, Any] | None]
        | None = None,
    ) -> None:
        self._transfer_function_provider = transfer_function_provider or (
            lambda _name: None
        )
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
            source.GetOutputDataObject(representation.output_port),
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

        source_data = source.GetOutputDataObject(representation.output_port)
        if handle is not None and handle.pipeline_filter is not None:
            if source_data is not None:
                handle.pipeline_filter.SetInputDataObject(source_data)
                handle.pipeline_filter.Update()
                handle.mapper.SetInputDataObject(
                    handle.pipeline_filter.GetOutputDataObject(0)
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
            source_data,
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
        source_data: vtk.vtkDataObject | None,
    ) -> None:
        properties = representation.properties
        mapper = handle.mapper
        actor = handle.actor
        prop = actor.GetProperty()

        actor.SetVisibility(1)

        color_by = properties.get("color_by")
        array_name = None
        association = "point"
        tf = None
        selected_array_name = None

        if color_by is not None:
            array_name = str(color_by[0])
            association = str(color_by[1])
            tf = self._transfer_function_provider(array_name)

        if representation.kind != "outline" and source_data is not None:
            coloring_data, selected_array_name = _data_for_coloring(
                source_data,
                array_name,
                association,
            )
            handle.coloring_data = (
                coloring_data if coloring_data is not source_data else None
            )
            mapper.SetInputDataObject(coloring_data)

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

            if selected_array_name is not None and tf is not None:
                if association == "cell":
                    mapper.SetScalarModeToUseCellFieldData()
                else:
                    mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectScalarArray(selected_array_name)
                _apply_volume_transfer_function(
                    handle.color_function,
                    handle.opacity_function,
                    tf,
                )
            elif color_by is None:
                _apply_fixed_volume_color(
                    handle.color_function,
                    handle.opacity_function,
                    properties.get("color", "#ffffff"),
                )

            mapper.Modified()
            volume_property.Modified()
            actor.Modified()
            return

        if representation.kind == "wireframe":
            prop.SetRepresentationToWireframe()
        else:
            prop.SetRepresentationToSurface()

        if (
            representation.kind == "outline"
            or selected_array_name is None
            or tf is None
        ):
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
        if association == "point":
            mapper.SetScalarModeToUsePointFieldData()
        else:
            mapper.SetScalarModeToUseCellFieldData()
        mapper.SelectColorArray(selected_array_name)

        lookup_table = _build_lookup_table(tf)
        handle.lookup_table = lookup_table
        mapper.SetLookupTable(lookup_table)
        mapper.UseLookupTableScalarRangeOn()

        mapper.Modified()
        actor.Modified()


def _data_for_coloring(
    source_data: vtk.vtkDataObject,
    array_name: str | None,
    association: str,
) -> tuple[vtk.vtkDataObject, str | None]:
    if array_name is None or not isinstance(source_data, vtk.vtkDataSet):
        return source_data, None

    attributes = (
        source_data.GetCellData()
        if association == "cell"
        else source_data.GetPointData()
    )
    array = attributes.GetArray(array_name)
    if array is None:
        return source_data, None
    if array.GetNumberOfComponents() <= 1:
        return source_data, array_name

    values = vtk_to_numpy(array)
    if values.ndim == 1:
        magnitude = np.abs(values)
    else:
        magnitude = np.linalg.norm(values, axis=1)

    copied = source_data.NewInstance()
    copied.ShallowCopy(source_data)
    magnitude_array = numpy_to_vtk(np.asarray(magnitude), deep=True)
    magnitude_name = f"__vtkweb_magnitude__{association}__{array_name}"
    magnitude_array.SetName(magnitude_name)

    copied_attributes = (
        copied.GetCellData() if association == "cell" else copied.GetPointData()
    )
    copied_attributes.AddArray(magnitude_array)
    return copied, magnitude_name


def _tf_value(data_range: list[float] | tuple[float, float], t: float) -> float:
    minimum = float(data_range[0])
    maximum = float(data_range[1])
    return minimum + float(t) * (maximum - minimum)


def _build_lookup_table(tf: dict[str, Any], samples: int = 256) -> vtk.vtkLookupTable:
    data_range = tf["range"]
    control_points = sorted(tf["control_points"], key=lambda point: point[0])
    positions = np.asarray([point[0] for point in control_points], dtype=float)
    channels = [
        np.asarray([point[index] for point in control_points], dtype=float)
        for index in range(1, 5)
    ]
    sample_positions = np.linspace(0.0, 1.0, max(2, int(samples)))

    lookup_table = vtk.vtkLookupTable()
    lookup_table.SetNumberOfTableValues(len(sample_positions))
    lookup_table.SetRange(float(data_range[0]), float(data_range[1]))
    lookup_table.Build()

    sampled = [np.interp(sample_positions, positions, channel) for channel in channels]
    for index in range(len(sample_positions)):
        lookup_table.SetTableValue(
            index,
            float(sampled[0][index]),
            float(sampled[1][index]),
            float(sampled[2][index]),
            float(sampled[3][index]),
        )
    lookup_table.Modified()
    return lookup_table


def _apply_volume_transfer_function(
    color_function: vtk.vtkColorTransferFunction | None,
    opacity_function: vtk.vtkPiecewiseFunction | None,
    tf: dict[str, Any],
) -> None:
    data_range = tf["range"]
    control_points = sorted(tf["control_points"], key=lambda point: point[0])

    if color_function is not None:
        color_function.RemoveAllPoints()
        for t, r, g, b, _opacity in control_points:
            color_function.AddRGBPoint(
                _tf_value(data_range, t), float(r), float(g), float(b)
            )
        color_function.Modified()

    if opacity_function is not None:
        opacity_function.RemoveAllPoints()
        for t, _r, _g, _b, opacity in control_points:
            opacity_function.AddPoint(_tf_value(data_range, t), float(opacity))
        opacity_function.Modified()


def _apply_fixed_volume_color(
    color_function: vtk.vtkColorTransferFunction | None,
    opacity_function: vtk.vtkPiecewiseFunction | None,
    color: str,
) -> None:
    value = str(color).lstrip("#")
    rgb = (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )
    if color_function is not None:
        color_function.RemoveAllPoints()
        color_function.AddRGBPoint(0.0, *rgb)
        color_function.AddRGBPoint(1.0, *rgb)
        color_function.Modified()
    if opacity_function is not None:
        opacity_function.RemoveAllPoints()
        opacity_function.AddPoint(0.0, 1.0)
        opacity_function.AddPoint(1.0, 1.0)
        opacity_function.Modified()
