from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from vtkweb.rendering.base import RenderView, RenderingBackend, Representation


@dataclass
class VTKViewHandle:
    renderer: vtk.vtkRenderer
    render_window: vtk.vtkRenderWindow
    capture: vtk.vtkWindowToImageFilter
    writer: vtk.vtkJPEGWriter
    width: int = 0
    height: int = 0


@dataclass
class VTKRepresentationHandle:
    mapper: Any
    actor: Any
    kind: str
    outline_filter: vtk.vtkOutlineFilter | None = None
    color_function: vtk.vtkColorTransferFunction | None = None
    opacity_function: vtk.vtkPiecewiseFunction | None = None
    coloring_data: vtk.vtkDataObject | None = None


class VTKRenderingBackend(RenderingBackend):
    """Server-side VTK renderer producing encoded image frames."""

    name = "vtk"
    # Rendering continuously at an unbounded rate only steals resources from
    # other server renderers. The scheduler still runs continuously, but
    # paces VTK to a display-oriented rate.
    target_fps = 30.0

    def __init__(
        self,
        transfer_function_provider: Callable[[str], dict[str, Any] | None]
        | None = None,
    ) -> None:
        self._transfer_function_provider = transfer_function_provider or (
            lambda _name: None
        )
        # VTK render windows/OpenGL contexts are used only from one worker at a
        # time. The same lock also protects scene mutations against an in-flight
        # render without adding per-view synchronization state.
        self._lock = threading.RLock()
        self._views: dict[str, VTKViewHandle] = {}
        self._representations: dict[tuple[str, str], VTKRepresentationHandle] = {}

    # ------------------------------------------------------------------
    # Views / frame rendering
    # ------------------------------------------------------------------

    def add_view(self, view: RenderView) -> None:
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.AddRenderer(renderer)

        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(render_window)
        capture.ReadFrontBufferOff()
        capture.SetInputBufferTypeToRGB()

        writer = vtk.vtkJPEGWriter()
        writer.SetInputConnection(capture.GetOutputPort())
        writer.SetQuality(90)
        writer.WriteToMemoryOn()

        self._views[view.id] = VTKViewHandle(renderer, render_window, capture, writer)

    def remove_view(self, view_id: str) -> None:
        with self._lock:
            for representation_id, current_view_id in tuple(self._representations):
                if current_view_id == view_id:
                    self.remove_representation(representation_id, view_id)
            self._views.pop(view_id, None)

    def get_view_property(self, view_id: str, name: str) -> Any:
        handle = self._views[view_id]
        if name == "background_color":
            return _rgb_to_hex(tuple(float(v) for v in handle.renderer.GetBackground()))
        if name == "camera":
            camera = handle.renderer.GetActiveCamera()
            return {
                "position": list(camera.GetPosition()),
                "target": list(camera.GetFocalPoint()),
                "up": list(camera.GetViewUp()),
                "fov": float(camera.GetViewAngle()),
                "parallel_projection": bool(camera.GetParallelProjection()),
                "parallel_scale": float(camera.GetParallelScale()),
            }
        return None

    def set_view_property(self, view_id: str, name: str, value: Any) -> None:
        with self._lock:
            handle = self._views[view_id]
            if name == "background_color":
                if isinstance(value, str):
                    value = _hex_to_rgb(value)
                handle.renderer.SetBackground(*value)
            elif name == "camera":
                camera = handle.renderer.GetActiveCamera()
                if value.get("position") is not None:
                    camera.SetPosition(*value["position"])
                if value.get("target") is not None:
                    camera.SetFocalPoint(*value["target"])
                if value.get("up") is not None:
                    camera.SetViewUp(*value["up"])
                if value.get("fov") is not None:
                    camera.SetViewAngle(float(value["fov"]))
                if value.get("parallel_projection") is not None:
                    camera.SetParallelProjection(bool(value["parallel_projection"]))
                if value.get("parallel_scale") is not None:
                    camera.SetParallelScale(float(value["parallel_scale"]))
                handle.renderer.ResetCameraClippingRange()
            else:
                # World ambient properties are meaningful to path-tracing
                # backends but VTK currently has no equivalent view property.
                return
            handle.renderer.Modified()
            handle.render_window.Modified()

    def set_render_size(self, view_id: str, width: int, height: int) -> bool:
        width = max(1, int(width))
        height = max(1, int(height))
        with self._lock:
            handle = self._views[view_id]
            if handle.width == width and handle.height == height:
                return False
            handle.width = width
            handle.height = height
            handle.render_window.SetSize(width, height)
        return True

    def has_renderable_scene(self, view_id: str) -> bool:
        handle = self._views[view_id]
        return handle.width > 0 and handle.height > 0

    def render_frame(self, view_id: str) -> bytes | None:
        # The dedicated worker always renders the latest state available when it
        # acquires the backend lock. Camera/property changes that arrive during a
        # frame are therefore picked up naturally by the next loop iteration.
        with self._lock:
            handle = self._views.get(view_id)
            if handle is None:
                return None

            handle.renderer.ResetCameraClippingRange()
            handle.render_window.Render()
            handle.capture.Modified()
            handle.capture.Update()

            handle.writer.Write()
            result = handle.writer.GetResult()
            if result is None or result.GetNumberOfValues() == 0:
                return None
            return bytes(memoryview(result))

    def release_render_resources(self, view_id: str) -> None:
        """Finalize the VTK graphics context on its dedicated render thread."""
        handle = self._views.get(view_id)
        if handle is not None:
            handle.render_window.Finalize()

    def reset_camera(self, view_id: str) -> None:
        with self._lock:
            handle = self._views[view_id]
            handle.renderer.ResetCamera()
            handle.renderer.ResetCameraClippingRange()
            handle.renderer.Modified()
            handle.render_window.Modified()

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------

    def add_representation(
        self,
        representation: Representation,
        view: RenderView,
        source: vtk.vtkAlgorithm,
    ) -> None:
        key = (representation.id, view.id)
        if key in self._representations:
            return
        with self._lock:
            handle = self._create_handle(representation, source)
            self._representations[key] = handle
            self._views[view.id].renderer.AddViewProp(handle.actor)
            self._apply_representation(representation, handle, source)

    def update_representation(
        self,
        representation: Representation,
        view: RenderView,
        source: vtk.vtkAlgorithm,
    ) -> None:
        key = (representation.id, view.id)
        with self._lock:
            handle = self._representations.get(key)
            if handle is None:
                self.add_representation(representation, view, source)
                return
            if handle.kind != representation.kind:
                self._views[view.id].renderer.RemoveViewProp(handle.actor)
                handle = self._create_handle(representation, source)
                self._representations[key] = handle
                self._views[view.id].renderer.AddViewProp(handle.actor)
            self._apply_representation(representation, handle, source)

    def remove_representation(self, representation_id: str, view_id: str) -> None:
        with self._lock:
            handle = self._representations.pop((representation_id, view_id), None)
            view = self._views.get(view_id)
            if handle is None or view is None:
                return
            view.renderer.RemoveViewProp(handle.actor)

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
                kind="volume",
                color_function=color_function,
                opacity_function=opacity_function,
            )

        if representation.kind == "outline":
            outline_filter = vtk.vtkOutlineFilter()
            if source_data is not None:
                outline_filter.SetInputDataObject(source_data)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(outline_filter.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            return VTKRepresentationHandle(
                mapper=mapper,
                actor=actor,
                kind="outline",
                outline_filter=outline_filter,
            )

        mapper = vtk.vtkDataSetMapper()
        if source_data is not None:
            mapper.SetInputDataObject(source_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return VTKRepresentationHandle(
            mapper=mapper, actor=actor, kind=representation.kind
        )

    def _apply_representation(
        self,
        representation: Representation,
        handle: VTKRepresentationHandle,
        source: vtk.vtkAlgorithm,
    ) -> None:
        source_data = source.GetOutputDataObject(representation.output_port)
        properties = representation.properties
        mapper = handle.mapper
        actor = handle.actor

        if handle.kind == "outline":
            if handle.outline_filter is not None:
                handle.outline_filter.SetInputDataObject(source_data)
            mapper.ScalarVisibilityOff()
            prop = actor.GetProperty()
            prop.SetColor(*_hex_to_rgb(properties.get("color", "#ffffff")))
            prop.SetLineWidth(float(properties.get("line_width", 1.0)))
            actor.Modified()
            return

        if source_data is None:
            actor.SetVisibility(False)
            return
        actor.SetVisibility(True)

        color_by = properties.get("color_by")
        array_name = str(color_by[0]) if color_by is not None else None
        association = str(color_by[1]) if color_by is not None else "point"
        tf = (
            self._transfer_function_provider(array_name)
            if array_name is not None
            else None
        )
        coloring_data, selected_array_name = _data_for_coloring(
            source_data, array_name, association
        )
        handle.coloring_data = None if coloring_data is source_data else coloring_data
        mapper.SetInputDataObject(coloring_data)

        if handle.kind == "volume":
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

            blend_mode = properties.get("blend_mode", "composite")
            if blend_mode == "maximum":
                mapper.SetBlendModeToMaximumIntensity()
            elif blend_mode == "minimum":
                mapper.SetBlendModeToMinimumIntensity()
            else:
                mapper.SetBlendModeToComposite()
            mapper.SetAutoAdjustSampleDistances(
                1 if properties.get("auto_adjust_sample_distances", True) else 0
            )
            mapper.SetSampleDistance(
                max(1e-12, float(properties.get("sample_distance", 1.0)))
            )

            if selected_array_name is not None and tf is not None:
                if association == "cell":
                    mapper.SetScalarModeToUseCellFieldData()
                else:
                    mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectScalarArray(selected_array_name)
                _apply_volume_transfer_function(
                    handle.color_function, handle.opacity_function, tf
                )
            else:
                _apply_fixed_volume_color(
                    handle.color_function,
                    handle.opacity_function,
                    properties.get("color", "#ffffff"),
                )
            actor.Modified()
            return

        prop = actor.GetProperty()
        if handle.kind == "wireframe":
            prop.SetRepresentationToWireframe()
            prop.SetLineWidth(float(properties.get("line_width", 1.0)))
        else:
            prop.SetRepresentationToSurface()

        if selected_array_name is None or tf is None:
            mapper.ScalarVisibilityOff()
            prop.SetColor(*_hex_to_rgb(properties.get("color", "#ffffff")))
        else:
            mapper.ScalarVisibilityOn()
            if association == "point":
                mapper.SetScalarModeToUsePointFieldData()
            else:
                mapper.SetScalarModeToUseCellFieldData()
            mapper.SelectColorArray(selected_array_name)
            mapper.SetLookupTable(_build_surface_color_map(tf))
            mapper.SetScalarRange(
                float(tf["control_points"][0][0]),
                float(tf["control_points"][-1][0]),
            )
            mapper.SetColorModeToMapScalars()
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
    magnitude = np.abs(values) if values.ndim == 1 else np.linalg.norm(values, axis=1)
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


def _build_surface_color_map(tf: dict[str, Any]) -> vtk.vtkColorTransferFunction:
    color_map = vtk.vtkColorTransferFunction()
    color_map.SetColorSpaceToRGB()
    color_map.SetClamping(True)
    for value, r, g, b, _opacity in tf["control_points"]:
        color_map.AddRGBPoint(float(value), float(r), float(g), float(b))
    return color_map


def _apply_volume_transfer_function(
    color_function: vtk.vtkColorTransferFunction | None,
    opacity_function: vtk.vtkPiecewiseFunction | None,
    tf: dict[str, Any],
) -> None:
    if color_function is not None:
        color_function.RemoveAllPoints()
    if opacity_function is not None:
        opacity_function.RemoveAllPoints()
    for value, r, g, b, opacity in tf["control_points"]:
        if color_function is not None:
            color_function.AddRGBPoint(float(value), float(r), float(g), float(b))
        if opacity_function is not None:
            opacity_function.AddPoint(float(value), float(opacity))


def _apply_fixed_volume_color(
    color_function: vtk.vtkColorTransferFunction | None,
    opacity_function: vtk.vtkPiecewiseFunction | None,
    color: str,
) -> None:
    rgb = _hex_to_rgb(color)
    if color_function is not None:
        color_function.RemoveAllPoints()
        color_function.AddRGBPoint(0.0, *rgb)
        color_function.AddRGBPoint(1.0, *rgb)
    if opacity_function is not None:
        opacity_function.RemoveAllPoints()
        opacity_function.AddPoint(0.0, 0.0)
        opacity_function.AddPoint(1.0, 1.0)


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    value = str(color).lstrip("#")
    if len(value) != 6:
        return (1.0, 1.0, 1.0)
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


def _rgb_to_hex(color: tuple[float, float, float]) -> str:
    values = [round(max(0.0, min(1.0, component)) * 255) for component in color]
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"
