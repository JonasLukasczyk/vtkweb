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
    interactor: vtk.vtkRenderWindowInteractor

    keepalive_source: vtk.vtkSphereSource
    keepalive_mapper: vtk.vtkPolyDataMapper
    keepalive_actor: vtk.vtkActor


@dataclass
class VTKRepresentationHandle:
    mapper: Any
    actor: Any
    kind: str

    # Non-volume representations keep both render branches alive for the
    # lifetime of the representation. trame-vtklocal mirrors VTK objects by
    # id, so toggling visibility is safer than replacing/removing branches.
    outline_filter: vtk.vtkOutlineFilter | None = None
    outline_mapper: vtk.vtkPolyDataMapper | None = None
    outline_actor: vtk.vtkActor | None = None
    color_function: vtk.vtkColorTransferFunction | None = None
    opacity_function: vtk.vtkPiecewiseFunction | None = None
    lookup_table: Any = None
    coloring_data: vtk.vtkDataObject | None = None
    surface_filter: vtk.vtkDataSetSurfaceFilter | None = None
    render_data: vtk.vtkDataObject | None = None


class VTKRenderingBackend(RenderingBackend):
    name = "vtk"

    def __init__(
        self,
        transfer_function_provider: Callable[[str], dict[str, Any] | None] | None = None,
    ) -> None:
        self._transfer_function_provider = transfer_function_provider or (lambda _name: None)
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

        # trame-vtklocal mirrors the actual VTK object graph into VTK/WASM.
        # Unlike trame_vtk.VtkLocalView, it therefore needs a real VTK
        # interactor attached to every render window so the client runtime can
        # start its local interaction event loop. Keep a Python reference to
        # the interactor in VTKViewHandle as well.
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        interactor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        render_window.SetOffScreenRendering(1)

        # ---------------------------------------------------------------------
        # Local VTK view keepalive
        # ---------------------------------------------------------------------
        #
        # A local mirrored VTK view can end up without a usable current renderer
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
        keepalive_mapper.SetInputDataObject(keepalive_source.GetOutput())

        keepalive_actor = vtk.vtkActor()

        keepalive_actor.SetMapper(keepalive_mapper)

        # Keep the renderer structurally non-empty without contributing visible
        # geometry to the scene.
        keepalive_actor.SetVisibility(False)
        keepalive_actor.SetPickable(False)

        renderer.AddActor(keepalive_actor)

        self._views[view.id] = VTKViewHandle(
            renderer=renderer,
            render_window=render_window,
            interactor=interactor,
            keepalive_source=keepalive_source,
            keepalive_mapper=keepalive_mapper,
            keepalive_actor=keepalive_actor,
        )

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


    def get_render_window(
        self,
        view_id: str,
    ) -> vtk.vtkRenderWindow:
        return self._views[view_id].render_window

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
        handle = self._views[view_id]
        if name == "background_color":
            if isinstance(value, str):
                value = _hex_to_rgb(value)
            handle.renderer.SetBackground(*value)
            handle.renderer.Modified()
            handle.render_window.Modified()
            return

        if name != "camera":
            return

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
        source_data = source.GetOutputDataObject(representation.output_port)
        render_data = self._set_representation_input(handle, source_data)

        self._apply_representation(
            representation,
            handle,
            render_data,
        )

        if self._source_has_geometry(
            source,
            representation.output_port,
        ):
            if representation.kind == "volume":
                view_handle.renderer.AddVolume(handle.actor)
            else:
                # Keep both non-volume branches in the renderer from the first
                # synchronization onward. Representation kind changes then only
                # toggle visibility and never churn vtklocal object ids.
                view_handle.renderer.AddActor(handle.actor)
                if handle.outline_actor is not None:
                    view_handle.renderer.AddActor(handle.outline_actor)

        view_handle.renderer.Modified()
        view_handle.render_window.Modified()

    def update_representation(
        self,
        representation: Representation,
        view: RenderView,
        source: vtk.vtkAlgorithm,
    ) -> None:
        key = (representation.id, view.id)
        handle = self._representations.get(key)

        if handle is None:
            self.add_representation(representation, view, source)
            return

        # vtkActor and vtkVolume are fundamentally different prop types. Keep
        # the non-volume graph stable, but volume transitions still require a
        # replacement representation.
        old_is_volume = handle.kind == "volume"
        new_is_volume = representation.kind == "volume"
        if old_is_volume != new_is_volume:
            self.remove_representation(representation.id, view.id)
            self.add_representation(representation, view, source)
            return

        source_data = source.GetOutputDataObject(representation.output_port)
        handle.kind = representation.kind
        render_data = self._set_representation_input(handle, source_data)
        self._apply_representation(representation, handle, render_data)

        view_handle = self._views[view.id]
        has_geometry = self._source_has_geometry(source, representation.output_port)

        if representation.kind == "volume":
            has_actor = bool(view_handle.renderer.HasViewProp(handle.actor))
            if has_geometry and not has_actor:
                view_handle.renderer.AddVolume(handle.actor)
            elif not has_geometry and has_actor:
                view_handle.renderer.RemoveViewProp(handle.actor)
        else:
            for actor in (handle.actor, handle.outline_actor):
                if actor is None:
                    continue
                has_actor = bool(view_handle.renderer.HasViewProp(actor))
                if has_geometry and not has_actor:
                    view_handle.renderer.AddActor(actor)
                elif not has_geometry and has_actor:
                    view_handle.renderer.RemoveViewProp(actor)

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
        if handle.outline_actor is not None and view.renderer.HasViewProp(handle.outline_actor):
            view.renderer.RemoveViewProp(handle.outline_actor)

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


    def _set_representation_input(
        self,
        handle: VTKRepresentationHandle,
        source_data: vtk.vtkDataObject | None,
    ) -> vtk.vtkDataObject | None:
        """Prepare the data object mirrored by trame-vtklocal.

        The application pipeline stays server-side. For non-volume rendering we
        additionally normalize vtkDataSet inputs to vtkPolyData. vtkObjectManager
        currently leaves dangling dependency ids when vtkImageData is attached
        directly to a mapper (for example vtkRTAnalyticSource output). A
        server-side vtkDataSetSurfaceFilter keeps structured datasets out of the
        mirrored object graph while preserving the visible surface and arrays.
        """
        if source_data is None:
            handle.render_data = None
            return None

        if handle.kind == "volume":
            render_data = source_data
        elif isinstance(source_data, vtk.vtkPolyData):
            render_data = source_data
            handle.surface_filter = None
        elif isinstance(source_data, vtk.vtkDataSet):
            if handle.surface_filter is None:
                handle.surface_filter = vtk.vtkDataSetSurfaceFilter()
            handle.surface_filter.SetInputDataObject(source_data)
            handle.surface_filter.Update()
            render_data = handle.surface_filter.GetOutputDataObject(0)
        else:
            render_data = source_data

        handle.render_data = render_data
        handle.mapper.SetInputDataObject(render_data)

        if handle.outline_filter is not None and handle.outline_mapper is not None:
            handle.outline_filter.SetInputDataObject(render_data)
            handle.outline_filter.Update()
            outline_data = handle.outline_filter.GetOutputDataObject(0)
            handle.outline_mapper.SetInputDataObject(outline_data)
            handle.outline_mapper.Modified()

        handle.mapper.Modified()
        return render_data

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

        # Stable non-volume graph: surface/wireframe and outline branches are
        # both created once and remain part of the renderer.
        mapper = vtk.vtkDataSetMapper()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        outline_filter = vtk.vtkOutlineFilter()
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)

        return VTKRepresentationHandle(
            mapper=mapper,
            actor=actor,
            kind=representation.kind,
            outline_filter=outline_filter,
            outline_mapper=outline_mapper,
            outline_actor=outline_actor,
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

        if representation.kind == "volume":
            actor.SetVisibility(1)
        else:
            actor.SetVisibility(0 if representation.kind == "outline" else 1)
            if handle.outline_actor is not None:
                handle.outline_actor.SetVisibility(1 if representation.kind == "outline" else 0)
                handle.outline_actor.SetPickable(representation.kind == "outline")

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
            # The render graph is data-only. Restore the source data when no
            # derived coloring data is needed; otherwise mirror the derived data.
            if coloring_data is source_data:
                handle.coloring_data = None
                mapper.SetInputDataObject(source_data)
            else:
                handle.coloring_data = coloring_data
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

        if representation.kind == "outline":
            mapper.ScalarVisibilityOff()
            if handle.outline_mapper is not None:
                handle.outline_mapper.ScalarVisibilityOff()
                handle.outline_mapper.Modified()
            if handle.outline_actor is not None:
                outline_prop = handle.outline_actor.GetProperty()
                color = properties.get("color", "#ffffff").lstrip("#")
                outline_prop.SetColor(
                    int(color[0:2], 16) / 255.0,
                    int(color[2:4], 16) / 255.0,
                    int(color[4:6], 16) / 255.0,
                )
                outline_prop.SetLineWidth(float(properties.get("line_width", 1.0)))
                handle.outline_actor.Modified()
            actor.Modified()
            return

        if representation.kind == "wireframe":
            prop.SetRepresentationToWireframe()
        else:
            prop.SetRepresentationToSurface()

        if selected_array_name is None or tf is None:
            mapper.ScalarVisibilityOff()
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

        color_map = _build_surface_color_map(tf)
        handle.lookup_table = color_map
        mapper.SetLookupTable(color_map)
        mapper.SetScalarRange(float(tf["control_points"][0][0]), float(tf["control_points"][-1][0]))
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
        source_data.GetCellData() if association == "cell" else source_data.GetPointData()
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


def _build_surface_color_map(tf: dict[str, Any]) -> vtk.vtkColorTransferFunction:
    """Create the VTK/vtk.js surface color map directly from global TF state."""
    color_map = vtk.vtkColorTransferFunction()
    color_map.SetColorSpaceToRGB()
    color_map.SetClamping(True)
    for value, r, g, b, _opacity in tf["control_points"]:
        color_map.AddRGBPoint(float(value), float(r), float(g), float(b))
    color_map.Modified()
    return color_map


def _apply_volume_transfer_function(
    color_function: vtk.vtkColorTransferFunction | None,
    opacity_function: vtk.vtkPiecewiseFunction | None,
    tf: dict[str, Any],
) -> None:
    control_points = tf["control_points"]

    if color_function is not None:
        color_function.RemoveAllPoints()
        for value, r, g, b, _opacity in control_points:
            color_function.AddRGBPoint(float(value), float(r), float(g), float(b))
        color_function.Modified()

    if opacity_function is not None:
        opacity_function.RemoveAllPoints()
        for value, _r, _g, _b, opacity in control_points:
            opacity_function.AddPoint(float(value), float(opacity))
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


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


def _rgb_to_hex(color: tuple[float, float, float]) -> str:
    values = [round(max(0.0, min(1.0, component)) * 255) for component in color]
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"
