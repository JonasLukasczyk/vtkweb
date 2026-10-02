from __future__ import annotations

import importlib
import math
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from vtkweb.rendering.base import RenderedFrame, RenderView, RenderingBackend, Representation


@dataclass
class MitsubaViewHandle:
    background_color: tuple[float, float, float] = (0.1, 0.1, 0.1)
    world_ambient_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    world_ambient_intensity: float = 1.0
    width: int = 0
    height: int = 0
    camera_origin: tuple[float, float, float] = (0.0, 0.0, 5.0)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    camera_fov: float = 30.0
    camera_focal_length_mm: float = 44.78460969082653
    camera_focus_distance: float = 0.0
    camera_aperture_size: float = 0.0
    center_of_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accumulation: np.ndarray | None = None
    accumulated_spp: int = 0
    next_seed: int = 1
    render_revision: int = 0
    accumulation_key: tuple | None = None
    cached_scene: Any | None = None
    cached_scene_key: tuple | None = None
    cached_scene_params: Any | None = None
    cached_sensor_key: tuple | None = None
    integrator: Any | None = None
    integrator_key: tuple | None = None


@dataclass
class MitsubaRepresentationHandle:
    kind: str
    activity_scope: str = ""
    activity_view_id: str = ""
    scene_object: Any | None = None
    bounds: tuple[float, float, float, float, float, float] | None = None
    scalar_volume: Any | None = None
    scalar_volume_key: tuple[Any, ...] | None = None
    color_mapping: dict[str, Any] | None = None
    opacity_mapping: dict[str, Any] | None = None
    sample_distance: float = 1.0
    opacity_reference_distance: float = 1.0
    gradient_step: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gradient_volume: Any | None = None
    gradient_volume_key: tuple[Any, ...] | None = None
    shade: bool = False
    ambient: float = 0.1
    diffuse: float = 0.9
    specular: float = 0.2
    specular_power: float = 10.0
    global_illumination_reach: float = 0.0
    volumetric_scattering_blending: float = 0.0
    scattering_anisotropy: float = 0.0
    environment_scattering_strength: float = 1.0
    environment_scattering_samples: int = 0
    environment_scattering_step_factor: float = 4.0
    environment_lighting_volumes: tuple[Any, ...] = ()
    environment_lighting_key: tuple[Any, ...] | None = None
    scalar_range: tuple[float, float] = (0.0, 1.0)
    scalar_values: np.ndarray | None = None
    shadow_volume: Any | None = None
    shadow_volume_key: tuple[Any, ...] | None = None
    scalar_volume_precision: str = "f32"
    gradient_volume_precision: str = "f32"
    shadow_volume_precision: str = "f32"
    environment_volume_precision: str = "f32"


class MitsubaRenderingBackend(RenderingBackend):
    @staticmethod
    def _srgb_to_linear(
        color: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Convert normalized sRGB values to linear-light RGB."""

        def convert(channel: float) -> float:
            value = max(0.0, min(1.0, float(channel)))
            if value <= 0.04045:
                return value / 12.92
            return ((value + 0.055) / 1.055) ** 2.4

        return tuple(convert(channel) for channel in color)

    def __init__(self, transfer_function_provider=None, activity_reporter=None) -> None:
        import drjit as dr
        import mitsuba as mi

        self._transfer_function_provider = transfer_function_provider or (
            lambda _name: None
        )
        self._activity = activity_reporter
        self.mi = mi
        self.dr = dr

        preferred_variants = ("cuda_ad_rgb", "metal_ad_rgb", "llvm_ad_rgb")
        available_variants = set(mi.variants())
        selected_variant = None
        variant_errors = []

        # A variant being listed (and even mi.set_variant() succeeding) does not
        # guarantee that the corresponding device/runtime can execute on this
        # process. This matters for MPI workers, where device visibility can
        # differ per rank. Probe one tiny JIT operation before accepting a
        # candidate so each rank falls back independently when necessary.
        for variant in preferred_variants:
            if variant not in available_variants:
                variant_errors.append(f"{variant}: unavailable")
                continue
            try:
                mi.set_variant(variant)
                probe = mi.Float([1.0, 2.0]) * 2.0
                dr.eval(probe)
                dr.sync_thread()
            except Exception as exc:
                variant_errors.append(f"{variant}: {type(exc).__name__}: {exc}")
                continue
            selected_variant = variant
            break

        if selected_variant is None:
            details = "; ".join(variant_errors) or "no preferred variants reported by Mitsuba"
            raise RuntimeError(
                "Mitsuba backend initialization failed: none of the preferred variants "
                f"could execute ({', '.join(preferred_variants)}). {details}"
            )

        try:
            from vtkweb.distributed import context as distributed_context
            mpi_rank = int(distributed_context.rank)
        except Exception:
            mpi_rank = 0
        print(f"[Mitsuba] rank={mpi_rank} variant: {selected_variant}", flush=True)
        self._direct_volume_types: dict[tuple[bool, bool, bool], type] = {}
        self._dvr_integrator_type = _make_dvr_integrator_type(mi, dr)

        # Protect only the small canonical render-state snapshot shared
        # between the Trame/server thread and the dedicated render worker.
        # Rendering and accumulation never hold this lock.
        self._state_lock = threading.RLock()

        # Mitsuba/Dr.Jit rendering is serialized within one backend instance
        # (and therefore within one MPI rank). Multiple views on the same rank
        # must not execute Dr.Jit render work concurrently. Separate MPI ranks
        # own separate backend instances and remain fully parallel.
        #
        # Keep this lock through Bitmap -> NumPy materialization as Dr.Jit may
        # defer work until the rendered tensor is consumed.
        self._render_lock = threading.RLock()
        self._views: dict[str, MitsubaViewHandle] = {}
        self._representations: dict[tuple[str, str], MitsubaRepresentationHandle] = {}

    @staticmethod
    def _precision(value: Any, *, allow_off: bool) -> str:
        value = str(value or "f32").lower()
        allowed = {"f32", "f16"} | ({"off"} if allow_off else set())
        return value if value in allowed else "f32"

    def _texture_storage_types(self, precision: str):
        """Return the Dr.Jit tensor/array/texture types for one storage precision."""
        precision = self._precision(precision, allow_off=False)
        tensor_type = self.mi.TensorXf16 if precision == "f16" else self.mi.TensorXf
        backend_module = importlib.import_module(tensor_type.__module__)
        texture_name = "Texture3f16" if precision == "f16" else "Texture3f"
        array_name = "Float16" if precision == "f16" else "Float"
        texture_type = getattr(backend_module, texture_name, None)
        array_type = getattr(backend_module, array_name, None)
        if texture_type is None or array_type is None:
            raise RuntimeError(
                f"Dr.Jit backend {tensor_type.__module__!r} does not provide "
                f"{texture_name}/{array_name}"
            )
        return tensor_type, array_type, texture_type

    def _texture_from_tensor(self, tensor, precision: str, *, interpolation: str = "linear"):
        _tensor_type, _array_type, texture_type = self._texture_storage_types(precision)
        filter_mode = (
            self.dr.FilterMode.Nearest
            if interpolation == "nearest"
            else self.dr.FilterMode.Linear
        )
        return texture_type(
            tensor,
            filter_mode=filter_mode,
            wrap_mode=self.dr.WrapMode.Clamp,
        )

    def _make_texture3d_device(
        self,
        flat_array,
        shape: tuple[int, int, int, int],
        precision: str,
        *,
        interpolation: str = "linear",
    ):
        """Create a 3D texture from an already device-resident flat Dr.Jit array."""
        tensor_type, _array_type, _texture_type = self._texture_storage_types(precision)
        try:
            tensor = tensor_type(flat_array, shape=shape)
        except TypeError:
            tensor = tensor_type(flat_array, shape)
        return self._texture_from_tensor(tensor, precision, interpolation=interpolation)

    def _make_texture3d(
        self,
        array: np.ndarray,
        precision: str,
        *,
        interpolation: str = "linear",
    ):
        """Upload a CPU array without first materializing a full FP32/FP16 host copy.

        Dr.Jit performs the conversion to the requested device storage type during
        tensor construction.  This is important for large VTK int16 volumes: the
        VTK-owned host buffer can be uploaded directly instead of first creating
        another multi-gigabyte NumPy float array.
        """
        precision = self._precision(precision, allow_off=False)
        source = np.asarray(array)
        if source.ndim == 3:
            source = source[..., None]
        # Preserve the source dtype. ascontiguousarray is a no-op for the usual
        # VTK scalar view and only copies genuinely strided input.
        source = np.ascontiguousarray(source)
        tensor_type, _array_type, _texture_type = self._texture_storage_types(precision)
        tensor = tensor_type(source)
        return self._texture_from_tensor(tensor, precision, interpolation=interpolation)

    @staticmethod
    def _gpu_texture_position_from_indices(mi, x, y, z, nx: int, ny: int, nz: int):
        """Return texture coordinates for integer voxel-center indices."""
        return mi.Point3f(
            (mi.Float(x) + 0.5) / float(nx),
            (mi.Float(y) + 0.5) / float(ny),
            (mi.Float(z) + 0.5) / float(nz),
        )

    def _compile_gpu_opacity_mapping(self, mapping):
        minimum, maximum = map(float, mapping["range"])
        width = maximum - minimum
        if abs(width) < 1.0e-20:
            width = 1.0
        points = np.asarray(mapping["control_points"], dtype=np.float32)
        x = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
        lut = np.interp(
            x,
            points[:, 0],
            np.clip(points[:, 1], 0.0, 1.0),
        ).astype(np.float32)
        return (
            self.dr.opaque(self.mi.Float, minimum),
            self.dr.opaque(self.mi.Float, 1.0 / width),
            self.mi.Float(np.ascontiguousarray(lut)),
            len(lut),
        )

    def _gpu_opacity_from_scalar(self, scalar, compiled):
        minimum, inv_width, lut, size = compiled
        x = self.dr.clip((scalar - minimum) * inv_width, 0.0, 1.0)
        scaled = x * float(size - 1)
        i0 = self.mi.UInt32(self.dr.floor(scaled))
        i1 = self.dr.minimum(i0 + 1, self.mi.UInt32(size - 1))
        t = scaled - self.mi.Float(i0)
        v0 = self.dr.gather(self.mi.Float, lut, i0)
        v1 = self.dr.gather(self.mi.Float, lut, i1)
        return self.dr.clip(self.dr.lerp(v0, v1, t), 0.0, 1.0 - 1.0e-6)

    def _direct_volume_type(
        self, *, gradient: bool, shadow: bool, environment: bool
    ):
        key = (bool(gradient), bool(shadow), bool(environment))
        volume_type = self._direct_volume_types.get(key)
        if volume_type is None:
            volume_type = _make_direct_volume_type(
                self.mi,
                self.dr,
                has_gradient_texture=key[0],
                has_shadow_texture=key[1],
                has_environment_texture=key[2],
            )
            self._direct_volume_types[key] = volume_type
        return volume_type

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def _activity_start(
        self,
        key: str,
        label: str,
        *,
        view_id: str,
        details: dict[str, Any] | None = None,
        determinate: bool = True,
    ) -> float:
        if self._activity is None:
            return time.monotonic()
        return self._activity.start(
            key,
            label,
            view_id=view_id,
            details=details,
            determinate=determinate,
        )

    def _activity_progress(self, key: str, progress: float) -> None:
        if self._activity is not None:
            self._activity.progress(key, progress)

    def _activity_done(
        self, key: str, started_at: float, *, details: dict[str, Any] | None = None
    ) -> None:
        if self._activity is not None:
            self._activity.done(key, started_at, details=details)

    def add_view(self, view: RenderView) -> None:
        with self._state_lock:
            self._views[view.id] = MitsubaViewHandle()

    def remove_view(self, view_id: str) -> None:
        with self._state_lock:
            for key in [key for key in self._representations if key[1] == view_id]:
                self._representations.pop(key, None)
            self._views.pop(view_id, None)

    def get_view_property(self, view_id: str, name: str) -> Any:
        with self._state_lock:
            handle = self._views[view_id]
            if name in {"background_color", "world_ambient_color"}:
                return _rgb_to_hex(getattr(handle, name))
            if name == "world_ambient_intensity":
                return float(handle.world_ambient_intensity)
            if name == "camera_focal_length_mm":
                return float(handle.camera_focal_length_mm)
            if name == "camera_focus_distance":
                return float(handle.camera_focus_distance)
            if name == "camera_aperture_size":
                return float(handle.camera_aperture_size)
            if name == "camera":
                return {
                    "position": list(handle.camera_origin),
                    "target": list(handle.camera_target),
                    "up": list(handle.camera_up),
                    "center_of_rotation": list(handle.center_of_rotation),
                    "fov": float(handle.camera_fov),
                }
        return None

    def set_view_property(self, view_id: str, name: str, value: Any) -> None:
        with self._state_lock:
            handle = self._views[view_id]
            if name in {"background_color", "world_ambient_color"}:
                if isinstance(value, str):
                    value = _hex_to_rgb(value)
                setattr(handle, name, tuple(float(v) for v in value))
                handle.render_revision += 1
                return
            if name == "world_ambient_intensity":
                handle.world_ambient_intensity = float(value)
                handle.render_revision += 1
                return
            if name == "camera_focal_length_mm":
                focal_length = max(1.0, float(value))
                handle.camera_focal_length_mm = focal_length
                # 35 mm-equivalent focal length using a 24 mm film height.
                handle.camera_fov = math.degrees(2.0 * math.atan(12.0 / focal_length))
                handle.render_revision += 1
                return
            if name == "camera_focus_distance":
                handle.camera_focus_distance = max(0.0, float(value))
                handle.render_revision += 1
                return
            if name == "camera_aperture_size":
                handle.camera_aperture_size = max(0.0, float(value))
                handle.render_revision += 1
                return
            if name != "camera":
                return
            if "position" in value:
                handle.camera_origin = tuple(float(v) for v in value["position"])
            if "target" in value:
                handle.camera_target = tuple(float(v) for v in value["target"])
            if "up" in value:
                handle.camera_up = tuple(float(v) for v in value["up"])
            if value.get("fov") is not None:
                handle.camera_fov = float(value["fov"])
                half = 0.5 * math.radians(max(1.0e-6, handle.camera_fov))
                handle.camera_focal_length_mm = 12.0 / max(math.tan(half), 1.0e-12)
            if "center_of_rotation" in value:
                handle.center_of_rotation = tuple(
                    float(v) for v in value["center_of_rotation"]
                )
            else:
                handle.center_of_rotation = tuple(handle.camera_target)
            handle.render_revision += 1

    def set_render_size(self, view_id: str, width: int, height: int) -> bool:
        """Update transient film dimensions and advance the render revision."""
        width = max(1, int(width))
        height = max(1, int(height))
        with self._state_lock:
            handle = self._views[view_id]
            if handle.width == width and handle.height == height:
                return False
            handle.width = width
            handle.height = height
            handle.render_revision += 1
        return True

    def invalidate_scene(self, view_id: str) -> int:
        """Advance render state without touching worker-owned accumulation."""
        with self._state_lock:
            handle = self._views[view_id]
            handle.render_revision += 1
            return handle.render_revision

    def reset_camera(self, view_id: str) -> None:
        with self._state_lock:
            bounds = self._visible_bounds_unlocked(view_id)
            handle = self._views[view_id]

            if bounds is None:
                handle.camera_origin = (0.0, 0.0, 5.0)
                handle.camera_target = (0.0, 0.0, 0.0)
                handle.camera_up = (0.0, 1.0, 0.0)
                handle.center_of_rotation = (0.0, 0.0, 0.0)
                handle.render_revision += 1
                return

            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            center = np.array(
                [
                    0.5 * (xmin + xmax),
                    0.5 * (ymin + ymax),
                    0.5 * (zmin + zmax),
                ],
                dtype=np.float64,
            )
            diagonal = np.array(
                [xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.float64
            )
            radius = max(0.5 * float(np.linalg.norm(diagonal)), 1.0e-6)

            half_fov = math.radians(handle.camera_fov) * 0.5
            distance = 1.15 * radius / math.sin(half_fov)

            origin = center + np.array([0.0, 0.0, distance])
            handle.camera_origin = tuple(float(v) for v in origin)
            handle.camera_target = tuple(float(v) for v in center)
            handle.camera_up = (0.0, 1.0, 0.0)
            handle.center_of_rotation = tuple(float(v) for v in center)
            handle.render_revision += 1

    def has_renderable_scene(self, view_id: str) -> bool:
        with self._state_lock:
            handle = self._views[view_id]
            return (
                handle.width > 0
                and handle.height > 0
                and any(
                    current_view_id == view_id
                    and (rep.scene_object is not None or rep.scalar_volume is not None)
                    for (_, current_view_id), rep in self._representations.items()
                )
            )

    def _clear_accumulation_worker(
        self,
        handle: MitsubaViewHandle,
        render_key: tuple,
    ) -> None:
        """Reset progressive state. Called only by the dedicated render worker."""
        handle.accumulation = None
        handle.accumulated_spp = 0
        handle.next_seed = 1
        handle.accumulation_key = render_key

    def _snapshot_render_state(self, view_id: str) -> tuple[dict[str, Any], int]:
        """Atomically copy the state consumed by one render pass."""
        with self._state_lock:
            handle = self._views[view_id]
            snapshot = {
                "camera": {
                    "position": tuple(handle.camera_origin),
                    "target": tuple(handle.camera_target),
                    "up": tuple(handle.camera_up),
                    "fov": float(handle.camera_fov),
                    "focal_length_mm": float(handle.camera_focal_length_mm),
                    "focus_distance": float(handle.camera_focus_distance),
                    "aperture_size": float(handle.camera_aperture_size),
                },
                "width": int(handle.width),
                "height": int(handle.height),
                "background_color": tuple(handle.background_color),
                "world_ambient_color": tuple(handle.world_ambient_color),
                "world_ambient_intensity": float(handle.world_ambient_intensity),
                # Keep strong references to exactly the scene objects represented
                # by this revision even if the server thread replaces them later.
                "objects": tuple(
                    (rep.kind, representation_id, rep.scene_object)
                    for (
                        representation_id,
                        current_view_id,
                    ), rep in self._representations.items()
                    if current_view_id == view_id and rep.scene_object is not None
                    and rep.kind != "volume"
                ),
                "volumes": tuple(
                    rep
                    for (_representation_id, current_view_id), rep
                    in self._representations.items()
                    if current_view_id == view_id
                    and rep.kind == "volume"
                    and rep.scalar_volume is not None
                    and rep.color_mapping is not None
                    and rep.opacity_mapping is not None
                ),
                "bounds": tuple(
                    rep.bounds
                    for (_representation_id, current_view_id), rep
                    in self._representations.items()
                    if current_view_id == view_id and rep.bounds is not None
                ),
            }
            return snapshot, int(handle.render_revision)

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------

    def add_representation(
        self,
        representation: Representation,
        view: RenderView,
        source: Any,
    ) -> None:
        key = (representation.id, view.id)
        with self._state_lock:
            if key in self._representations:
                return
        rep_handle = self._create_handle(
            representation, source, activity_view_id=view.id
        )
        with self._state_lock:
            self._representations[key] = rep_handle
            self._views[view.id].render_revision += 1

    def update_representation(
        self,
        representation: Representation,
        view: RenderView,
        source: Any,
    ) -> None:
        with self._state_lock:
            previous = self._representations.get((representation.id, view.id))
        rep_handle = self._create_handle(
            representation,
            source,
            previous=previous,
            activity_view_id=view.id,
        )
        with self._state_lock:
            self._representations[(representation.id, view.id)] = rep_handle
            self._views[view.id].render_revision += 1

    def remove_representation(self, representation_id: str, view_id: str) -> None:
        with self._state_lock:
            self._representations.pop((representation_id, view_id), None)
            if view_id in self._views:
                self._views[view_id].render_revision += 1

    def _create_handle(
        self,
        representation: Representation,
        source: Any,
        *,
        previous: MitsubaRepresentationHandle | None = None,
        activity_view_id: str = "",
    ) -> MitsubaRepresentationHandle:
        if representation.kind == "surface":
            return self._create_surface_handle(representation, source)
        if representation.kind == "wireframe":
            return self._create_wireframe_handle(representation, source)
        if representation.kind == "outline":
            return self._create_outline_handle(representation, source)
        if representation.kind == "volume":
            return self._create_volume_handle(
                representation,
                source,
                previous=previous,
                activity_view_id=activity_view_id,
            )

        print(
            "Mitsuba backend: representation kind "
            f"'{representation.kind}' is not implemented"
        )
        return MitsubaRepresentationHandle(kind=representation.kind)

    def _create_surface_handle(
        self,
        representation: Representation,
        source: Any,
    ) -> MitsubaRepresentationHandle:
        data = source.GetOutputDataObject(representation.output_port)
        if data is None:
            return MitsubaRepresentationHandle(kind="surface")

        surface = vtk.vtkDataSetSurfaceFilter()
        surface.SetInputDataObject(data)
        surface.Update()

        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputConnection(surface.GetOutputPort())
        triangles.PassLinesOff()
        triangles.PassVertsOff()
        triangles.Update()

        return self._create_polydata_mesh_handle(
            "surface", representation, triangles.GetOutput()
        )

    def _create_wireframe_handle(
        self,
        representation: Representation,
        source: Any,
    ) -> MitsubaRepresentationHandle:
        data = source.GetOutputDataObject(representation.output_port)
        if data is None:
            return MitsubaRepresentationHandle(kind="wireframe")

        # Extract edges before triangulation so polygon diagonals are not added.
        surface = vtk.vtkDataSetSurfaceFilter()
        surface.SetInputDataObject(data)

        edges = vtk.vtkExtractEdges()
        edges.SetInputConnection(surface.GetOutputPort())
        edges.Update()

        return self._create_tube_handle(
            "wireframe", representation, edges.GetOutput(), data.GetBounds()
        )

    def _create_outline_handle(
        self,
        representation: Representation,
        source: Any,
    ) -> MitsubaRepresentationHandle:
        data = source.GetOutputDataObject(representation.output_port)
        if data is None:
            return MitsubaRepresentationHandle(kind="outline")

        outline = vtk.vtkOutlineFilter()
        outline.SetInputDataObject(data)
        outline.Update()

        return self._create_tube_handle(
            "outline", representation, outline.GetOutput(), data.GetBounds()
        )

    def _create_tube_handle(
        self,
        kind: str,
        representation: Representation,
        lines: vtk.vtkPolyData,
        source_bounds,
    ) -> MitsubaRepresentationHandle:
        # Surface normals propagated onto lines are not valid tube-frame normals.
        # Preserve scalar arrays, but let vtkTubeFilter build its own line frame.
        line_data = vtk.vtkPolyData()
        line_data.ShallowCopy(lines)
        line_data.GetPointData().SetNormals(None)

        radius = max(0.0, float(representation.properties.get("line_width", 0.01)))
        if radius <= 0.0 or line_data.GetNumberOfLines() == 0:
            return MitsubaRepresentationHandle(
                kind=kind,
                bounds=tuple(float(v) for v in source_bounds),
            )

        sides = max(3, int(representation.properties.get("tube_sides", 3)))
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(line_data)
        tube.SetRadius(radius)
        tube.SetNumberOfSides(sides)
        tube.CappingOn()

        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputConnection(tube.GetOutputPort())
        triangles.PassLinesOff()
        triangles.PassVertsOff()
        triangles.Update()

        return self._create_polydata_mesh_handle(
            kind, representation, triangles.GetOutput()
        )

    def _create_polydata_mesh_handle(
        self,
        kind: str,
        representation: Representation,
        polydata: vtk.vtkPolyData,
    ) -> MitsubaRepresentationHandle:
        if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfPolys() == 0:
            return MitsubaRepresentationHandle(kind=kind)

        vertices = np.asarray(
            vtk_to_numpy(polydata.GetPoints().GetData()), dtype=np.float32
        )
        connectivity = vtk_to_numpy(polydata.GetPolys().GetConnectivityArray())
        faces = np.asarray(connectivity, dtype=np.uint32).reshape(-1, 3)

        color_by = representation.properties.get("color_by")
        vertex_colors = None
        if color_by is not None:
            array_name = str(color_by[0])
            association = str(color_by[1])
            tf = self._transfer_function_provider(array_name)
            if tf is not None:
                colors = _polydata_transfer_colors(
                    polydata,
                    array_name=array_name,
                    association=association,
                    transfer_function=tf,
                )
                if colors is not None:
                    if association == "cell":
                        vertices = vertices[faces].reshape(-1, 3)
                        faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
                        vertex_colors = np.repeat(colors, 3, axis=0)
                    else:
                        vertex_colors = colors

        # Mesh construction/traversal and BSDF creation touch Mitsuba/Dr.Jit
        # runtime state. Serialize those operations with rendering on this rank.
        with self._render_lock:
            mesh = self.mi.Mesh(
                f"vtkweb_{representation.id}_{kind}",
                vertex_count=len(vertices),
                face_count=len(faces),
                has_vertex_normals=False,
                has_vertex_texcoords=False,
            )
            params = self.mi.traverse(mesh)
            params["vertex_positions"] = vertices.reshape(-1)
            params["faces"] = faces.reshape(-1)
            params.update()

            try:
                if vertex_colors is not None:
                    mesh.add_attribute(
                        "vertex_color",
                        3,
                        np.asarray(vertex_colors, dtype=np.float32).reshape(-1),
                    )
                    reflectance = {
                        "type": "mesh_attribute",
                        "name": "vertex_color",
                    }
                else:
                    color = _hex_to_rgb(
                        representation.properties.get("color", "#d9d9d9")
                    )
                    reflectance = {"type": "rgb", "value": list(color)}

                mesh.set_bsdf(
                    self.mi.load_dict(
                        {
                            "type": "diffuse",
                            "reflectance": reflectance,
                        }
                    )
                )
            except Exception as exc:
                print(f"Mitsuba backend: could not set {kind} color: {exc}")

        bounds = polydata.GetBounds()
        return MitsubaRepresentationHandle(
            kind=kind,
            scene_object=mesh,
            bounds=tuple(float(v) for v in bounds),
        )

    def _create_volume_handle(
        self,
        representation: Representation,
        source: Any,
        *,
        previous: MitsubaRepresentationHandle | None = None,
        activity_view_id: str = "",
    ) -> MitsubaRepresentationHandle:
        """Create the persistent scalar grid used by the direct-volume integrator.

        Classification is deliberately *not* baked into 3D RGB/opacity grids.
        The current transfer function is stored as lightweight mapping metadata
        and evaluated by Dr.Jit at each ray-march sample.
        """
        data = source.GetOutputDataObject(representation.output_port)
        if not isinstance(data, vtk.vtkImageData):
            print("Mitsuba backend: volume rendering currently requires vtkImageData")
            return MitsubaRepresentationHandle(kind="volume")

        bounds = tuple(float(v) for v in data.GetBounds())
        color_by = representation.properties.get("color_by")
        if not color_by:
            print("Mitsuba backend: volume rendering requires a selected scalar array")
            return MitsubaRepresentationHandle(kind="volume", bounds=bounds)

        array_name = str(color_by[0])
        association = str(color_by[1])
        transfer_function = self._transfer_function_provider(array_name)
        if transfer_function is None:
            return MitsubaRepresentationHandle(kind="volume", bounds=bounds)

        interpolation = str(representation.properties.get("interpolation", "linear"))
        scalar_precision = self._precision(
            representation.properties.get("scalar_volume", "f32"), allow_off=False
        )
        gradient_precision = self._precision(
            representation.properties.get("gradient_volume", "f32"), allow_off=True
        )
        shadow_precision = self._precision(
            representation.properties.get("shadow_volume", "f32"), allow_off=True
        )
        environment_precision = self._precision(
            representation.properties.get("environment_volume", "f32"), allow_off=True
        )
        scalar_key = (
            data.GetAddressAsString("vtkweb"),
            int(data.GetMTime()),
            array_name,
            association,
            interpolation,
            scalar_precision,
        )
        scalar_volume = None
        scalar_values = None
        scalar_range = None
        if previous is not None and previous.scalar_volume_key == scalar_key:
            scalar_volume = previous.scalar_volume
            scalar_values = previous.scalar_values
            scalar_range = previous.scalar_range

        if scalar_volume is None:
            scalar_values = _image_scalar_values(data, array_name, association)
            if scalar_values is None:
                print(
                    "Mitsuba backend: selected volume array is unavailable or has "
                    "an incompatible tuple count"
                )
                return MitsubaRepresentationHandle(kind="volume", bounds=bounds)

            scalar_range = (float(np.min(scalar_values)), float(np.max(scalar_values)))
            activity_key = f"scalar:{activity_view_id}:{representation.id}"
            activity_started = self._activity_start(
                activity_key,
                "Uploading scalar volume",
                view_id=str(activity_view_id),
                details={
                    "shape": tuple(int(v) for v in scalar_values.shape),
                    "dtype": scalar_precision,
                },
                determinate=False,
            )
            with self._render_lock:
                scalar_volume = self._make_texture3d(
                    scalar_values,
                    scalar_precision,
                    interpolation=interpolation,
                )
            self._activity_done(activity_key, activity_started)

        handle = MitsubaRepresentationHandle(
            kind="volume",
            activity_scope=f"{activity_view_id}:{representation.id}",
            activity_view_id=str(activity_view_id),
            bounds=bounds,
            scalar_volume=scalar_volume,
            scalar_volume_key=scalar_key,
            color_mapping=transfer_function["color"],
            opacity_mapping=transfer_function["opacity"],
            sample_distance=_volume_sample_distance(data, representation.properties),
            opacity_reference_distance=_volume_opacity_reference_distance(data),
            gradient_step=_volume_gradient_step(data),
            shade=bool(representation.properties.get("shade", True)),
            ambient=float(representation.properties.get("ambient", 0.1)),
            diffuse=float(representation.properties.get("diffuse", 0.9)),
            specular=float(representation.properties.get("specular", 0.2)),
            specular_power=float(representation.properties.get("specular_power", 10.0)),
            global_illumination_reach=float(
                representation.properties.get("global_illumination_reach", 0.0)
            ),
            volumetric_scattering_blending=float(
                representation.properties.get("volumetric_scattering_blending", 0.0)
            ),
            scattering_anisotropy=float(
                representation.properties.get("scattering_anisotropy", 0.0)
            ),
            environment_scattering_strength=max(
                0.0, float(representation.properties.get("environment_scattering_strength", 1.0))
            ),
            environment_scattering_samples=max(
                0, int(representation.properties.get("environment_scattering_samples", 0))
            ),
            environment_scattering_step_factor=max(
                1.0, float(representation.properties.get("environment_scattering_step_factor", 4.0))
            ),
            scalar_range=scalar_range or (0.0, 1.0),
            scalar_values=scalar_values,
            scalar_volume_precision=scalar_precision,
            gradient_volume_precision=gradient_precision,
            shadow_volume_precision=shadow_precision,
            environment_volume_precision=environment_precision,
        )

        # Reuse the baked shadow field across representation refreshes whenever
        # its true inputs are unchanged. In particular, camera updates never
        # recreate representation handles, so they cannot invalidate this cache.
        if previous is not None:
            handle.gradient_volume = previous.gradient_volume
            handle.gradient_volume_key = previous.gradient_volume_key
            handle.shadow_volume = previous.shadow_volume
            handle.shadow_volume_key = previous.shadow_volume_key
            handle.environment_lighting_volumes = previous.environment_lighting_volumes
            handle.environment_lighting_key = previous.environment_lighting_key

        # Auxiliary textures are built lazily in render_pass() only when the
        # selected representation settings and shading path actually need them.
        return handle

    def _build_gradient_field(self, volume: MitsubaRepresentationHandle) -> None:
        """Bake the scalar gradient directly on the active Dr.Jit device.

        The only full-resolution persistent allocation is the requested gradient
        texture itself.  Work is emitted in small Z slabs so the six neighboring
        scalar samples never create full-volume temporary arrays on either the CPU
        or GPU.
        """
        scalars = volume.scalar_values
        bounds = volume.bounds
        texture = volume.scalar_volume
        if scalars is None or bounds is None or texture is None or scalars.ndim != 3:
            return
        precision = volume.gradient_volume_precision
        if precision == "off":
            volume.gradient_volume = None
            volume.gradient_volume_key = None
            return
        key = (
            volume.scalar_volume_key,
            tuple(float(v) for v in volume.gradient_step),
            tuple(float(v) for v in bounds),
            precision,
            "gpu-central-difference-v1",
        )
        if volume.gradient_volume is not None and volume.gradient_volume_key == key:
            return

        nz, ny, nx = (int(v) for v in scalars.shape)
        voxel_count = int(nz * ny * nx)
        tensor_type, storage_type, _texture_type = self._texture_storage_types(precision)
        del tensor_type
        start = time.perf_counter()
        activity_key = f"gradient:{volume.activity_scope}"
        activity_started = self._activity_start(
            activity_key,
            "Baking gradient volume",
            view_id=volume.activity_view_id,
            details={
                "shape": (nz, ny, nx),
                "dtype": precision,
                "device": self.mi.variant(),
            },
        )
        print(
            f"Mitsuba bake gradient (device): start shape={(nz, ny, nx)} dtype={precision}",
            flush=True,
        )

        hx, hy, hz = (max(abs(float(v)), 1.0e-12) for v in volume.gradient_step)
        with self._render_lock:
            output = self.dr.zeros(storage_type, voxel_count * 3)
            plane = int(ny * nx)
            # Keep each launch to roughly <= 8 million voxels. This is large
            # enough for good GPU occupancy but bounds temporary JIT arrays.
            slab_depth = max(1, min(nz, int(max(1, 8_000_000 // max(1, plane)))))
            report_stride = max(1, int(math.ceil(nz / 10.0)))

            for z0 in range(0, nz, slab_depth):
                z1 = min(nz, z0 + slab_depth)
                count = int((z1 - z0) * plane)
                local = self.dr.arange(self.mi.UInt32, count)
                x = local % self.mi.UInt32(nx)
                yz = local // self.mi.UInt32(nx)
                y = yz % self.mi.UInt32(ny)
                z = yz // self.mi.UInt32(ny) + self.mi.UInt32(z0)
                p = self._gpu_texture_position_from_indices(self.mi, x, y, z, nx, ny, nz)
                dx = self.mi.Vector3f(1.0 / float(nx), 0.0, 0.0)
                dy = self.mi.Vector3f(0.0, 1.0 / float(ny), 0.0)
                dz = self.mi.Vector3f(0.0, 0.0, 1.0 / float(nz))
                gx = (texture.eval(p + dx)[0] - texture.eval(p - dx)[0]) / (2.0 * hx)
                gy = (texture.eval(p + dy)[0] - texture.eval(p - dy)[0]) / (2.0 * hy)
                gz = (texture.eval(p + dz)[0] - texture.eval(p - dz)[0]) / (2.0 * hz)
                global_index = local + self.mi.UInt32(z0 * plane)
                base = global_index * self.mi.UInt32(3)
                self.dr.scatter(output, storage_type(gx), base)
                self.dr.scatter(output, storage_type(gy), base + 1)
                self.dr.scatter(output, storage_type(gz), base + 2)
                # Force each slab now so deferred JIT graphs cannot accumulate
                # into a second full-volume temporary representation.
                self.dr.eval(output)
                if z1 == nz or z1 % report_stride < slab_depth:
                    self._activity_progress(activity_key, 0.95 * z1 / max(1, nz))

            volume.gradient_volume = self._make_texture3d_device(
                output,
                (nz, ny, nx, 3),
                precision,
                interpolation="linear",
            )
            self.dr.eval(output)
        volume.gradient_volume_key = key
        self._activity_progress(activity_key, 0.98)
        elapsed = time.perf_counter() - start
        self._activity_done(activity_key, activity_started)
        print(f"Mitsuba bake gradient (device): end {elapsed:.3f}s", flush=True)

    def _build_directional_shadow_field(
        self, volume: MitsubaRepresentationHandle
    ) -> None:
        """Bake the +Z optical-depth texture directly on the Dr.Jit device."""
        scalars = volume.scalar_values
        mapping = volume.opacity_mapping
        bounds = volume.bounds
        texture = volume.scalar_volume
        if scalars is None or mapping is None or bounds is None or texture is None:
            return
        if scalars.ndim != 3 or scalars.shape[0] < 2:
            return

        sample_count = 2
        opacity_signature = (
            tuple(float(v) for v in mapping["range"]),
            tuple(
                tuple(float(component) for component in point)
                for point in mapping["control_points"]
            ),
        )
        precision = volume.shadow_volume_precision
        if precision == "off":
            volume.shadow_volume = None
            volume.shadow_volume_key = None
            return
        shadow_key = (
            volume.scalar_volume_key,
            opacity_signature,
            float(volume.opacity_reference_distance),
            (0.0, 0.0, 1.0),
            tuple(float(v) for v in bounds),
            precision,
            "gpu-z-optical-depth-v1",
        )
        if volume.shadow_volume is not None and volume.shadow_volume_key == shadow_key:
            return

        nz, ny, nx = (int(v) for v in scalars.shape)
        plane = int(ny * nx)
        voxel_count = int(nz * plane)
        _tensor_type, storage_type, _texture_type = self._texture_storage_types(precision)
        opacity_lut = self._compile_gpu_opacity_mapping(mapping)
        opacity_reference_distance = max(float(volume.opacity_reference_distance), 1.0e-12)
        zmin, zmax = float(bounds[4]), float(bounds[5])
        dz_world = (zmax - zmin) / float(max(nz - 1, 1))
        nodes, weights = np.polynomial.legendre.leggauss(sample_count)
        u_values = tuple(float(v) for v in 0.5 * (nodes + 1.0))
        weights = tuple(float(v) for v in 0.5 * weights)

        start = time.perf_counter()
        activity_key = f"shadow:{volume.activity_scope}"
        activity_started = self._activity_start(
            activity_key,
            "Baking shadow volume",
            view_id=volume.activity_view_id,
            details={
                "shape": (nz, ny, nx),
                "q": sample_count,
                "dtype": precision,
                "device": self.mi.variant(),
            },
        )
        print(
            f"Mitsuba bake directional shadow (device): start shape={(nz, ny, nx)} "
            f"quadrature_samples={sample_count} dtype={precision}",
            flush=True,
        )

        with self._render_lock:
            output = self.dr.zeros(storage_type, voxel_count)
            running_tau = self.dr.zeros(self.mi.Float, plane)
            xy = self.dr.arange(self.mi.UInt32, plane)
            x = xy % self.mi.UInt32(nx)
            y = xy // self.mi.UInt32(nx)
            report_stride = max(1, int(math.ceil(max(1, nz - 1) / 10.0)))

            # output[nz-1] is already zero at the light-facing boundary.
            for completed, k in enumerate(range(nz - 2, -1, -1), start=1):
                mean_sigma = self.dr.zeros(self.mi.Float, plane)
                for u, weight in zip(u_values, weights):
                    z = self.mi.Float(k + u)
                    p = self._gpu_texture_position_from_indices(
                        self.mi, x, y, z, nx, ny, nz
                    )
                    scalar = texture.eval(p)[0]
                    alpha = self._gpu_opacity_from_scalar(scalar, opacity_lut)
                    sigma = -self.dr.log(1.0 - alpha) / opacity_reference_distance
                    mean_sigma += float(weight) * sigma
                running_tau = running_tau + mean_sigma * dz_world
                indices = xy + self.mi.UInt32(k * plane)
                self.dr.scatter(output, storage_type(running_tau), indices)
                self.dr.eval(output, running_tau)
                if completed % report_stride == 0 or k == 0:
                    self._activity_progress(
                        activity_key,
                        0.95 * completed / max(1, nz - 1),
                    )

            volume.shadow_volume = self._make_texture3d_device(
                output,
                (nz, ny, nx, 1),
                precision,
                interpolation="linear",
            )
            self.dr.eval(output)
        volume.shadow_volume_key = shadow_key
        self._activity_progress(activity_key, 0.98)
        elapsed = time.perf_counter() - start
        self._activity_done(activity_key, activity_started)
        print(
            f"Mitsuba bake directional shadow (device): end {elapsed:.3f}s",
            flush=True,
        )

    def _gpu_gather_zyx(self, data, shape, z, y, x):
        """Gather one flat ZYX device volume, returning zero outside its bounds."""
        nz, ny, nx = (int(v) for v in shape)
        iz = self.mi.Int32(z)
        iy = self.mi.Int32(y)
        ix = self.mi.Int32(x)
        valid = (
            (iz >= 0) & (iz < nz) &
            (iy >= 0) & (iy < ny) &
            (ix >= 0) & (ix < nx)
        )
        izc = self.dr.minimum(self.dr.maximum(iz, 0), nz - 1)
        iyc = self.dr.minimum(self.dr.maximum(iy, 0), ny - 1)
        ixc = self.dr.minimum(self.dr.maximum(ix, 0), nx - 1)
        index = (
            self.mi.UInt32(izc) * self.mi.UInt32(ny * nx)
            + self.mi.UInt32(iyc) * self.mi.UInt32(nx)
            + self.mi.UInt32(ixc)
        )
        value = self.dr.gather(self.mi.Float, data, index)
        return self.dr.select(valid, value, 0.0)

    def _gpu_trilinear_flat(self, data, shape, z, y, x):
        """Trilinearly sample a flat ZYX float array, using zero outside."""
        z0 = self.mi.Int32(self.dr.floor(z))
        y0 = self.mi.Int32(self.dr.floor(y))
        x0 = self.mi.Int32(self.dr.floor(x))
        fz = z - self.mi.Float(z0)
        fy = y - self.mi.Float(y0)
        fx = x - self.mi.Float(x0)
        result = self.mi.Float(0.0)
        for dz in (0, 1):
            wz = fz if dz else 1.0 - fz
            for dy in (0, 1):
                wy = fy if dy else 1.0 - fy
                for dx in (0, 1):
                    wx = fx if dx else 1.0 - fx
                    result += (
                        self._gpu_gather_zyx(data, shape, z0 + dz, y0 + dy, x0 + dx)
                        * wz * wy * wx
                    )
        return result

    def _gpu_bilinear_tau_slice(
        self,
        tau,
        shape,
        dominant: int,
        fixed_index: int,
        other0,
        other1,
    ):
        """Bilinearly sample one fixed-axis slice of a flat ZYX tau volume."""
        axes = [0, 1, 2]
        axes.remove(dominant)
        o0, o1 = axes
        a0 = self.mi.Int32(self.dr.floor(other0))
        b0 = self.mi.Int32(self.dr.floor(other1))
        fa = other0 - self.mi.Float(a0)
        fb = other1 - self.mi.Float(b0)
        result = self.mi.Float(0.0)
        for da in (0, 1):
            wa = fa if da else 1.0 - fa
            for db in (0, 1):
                wb = fb if db else 1.0 - fb
                coords = [None, None, None]
                coords[dominant] = self.mi.Int32(fixed_index)
                coords[o0] = a0 + da
                coords[o1] = b0 + db
                result += self._gpu_gather_zyx(
                    tau, shape, coords[0], coords[1], coords[2]
                ) * wa * wb
        return result

    @staticmethod
    def _real_sh_basis(direction: tuple[float, float, float]) -> tuple[float, ...]:
        """Real, orthonormal spherical harmonics through degree 2."""
        x, y, z = map(float, direction)
        return (
            0.28209479177387814,
            0.4886025119029199 * y,
            0.4886025119029199 * z,
            0.4886025119029199 * x,
            1.0925484305920792 * x * y,
            1.0925484305920792 * y * z,
            0.31539156525252005 * (3.0 * z * z - 1.0),
            1.0925484305920792 * x * z,
            0.5462742152960396 * (x * x - y * y),
        )

    @staticmethod
    def _shadow_extent_for_volume(volume: MitsubaRepresentationHandle) -> float:
        xmin, xmax, ymin, ymax, zmin, zmax = volume.bounds
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        max_extent = math.sqrt(dx * dx + dy * dy + dz * dz)
        min_extent = min(float(volume.opacity_reference_distance), max_extent)
        reach = max(0.0, min(1.0, float(volume.global_illumination_reach)))
        return (min_extent - max_extent) * ((1.0 - reach) ** 0.33) + max_extent

    def _build_environment_lighting_field(
        self, volume: MitsubaRepresentationHandle
    ) -> None:
        """Bake L2 environment visibility entirely on the active Dr.Jit device.

        The reduced extinction grid, per-direction optical-depth field, SH
        accumulation, and final FP16/FP32 textures remain device-resident.  CPU
        memory use is limited to tiny direction/LUT metadata and the original
        VTK scalar buffer.
        """
        if volume.environment_scattering_samples <= 0 or not volume.shade:
            volume.environment_lighting_volumes = ()
            volume.environment_lighting_key = None
            return
        mapping = volume.opacity_mapping
        bounds = volume.bounds
        scalars = volume.scalar_values
        scalar_texture = volume.scalar_volume
        if mapping is None or bounds is None or scalars is None or scalar_texture is None:
            return

        opacity_signature = (
            tuple(float(v) for v in mapping["range"]),
            tuple(tuple(float(c) for c in point) for point in mapping["control_points"]),
        )
        precision = volume.environment_volume_precision
        if precision == "off":
            volume.environment_lighting_volumes = ()
            volume.environment_lighting_key = None
            return
        direction_count = max(0, int(volume.environment_scattering_samples))
        resolution_factor = max(1, int(round(volume.environment_scattering_step_factor)))
        key = (
            volume.scalar_volume_key,
            opacity_signature,
            float(volume.opacity_reference_distance),
            float(volume.global_illumination_reach),
            direction_count,
            resolution_factor,
            tuple(float(v) for v in bounds),
            2,
            precision,
            "gpu-environment-sh-v1",
        )
        if volume.environment_lighting_key == key and volume.environment_lighting_volumes:
            return

        nz, ny, nx = (int(v) for v in scalars.shape)
        rz = int(math.ceil(nz / float(resolution_factor)))
        ry = int(math.ceil(ny / float(resolution_factor)))
        rx = int(math.ceil(nx / float(resolution_factor)))
        reduced_shape = (rz, ry, rx)
        reduced_count = int(rz * ry * rx)
        opacity_lut = self._compile_gpu_opacity_mapping(mapping)
        opacity_reference_distance = max(float(volume.opacity_reference_distance), 1.0e-12)
        directions = _environment_directions(direction_count)

        start = time.perf_counter()
        activity_key = f"environment:{volume.activity_scope}"
        activity_started = self._activity_start(
            activity_key,
            "Baking environment lighting",
            view_id=volume.activity_view_id,
            details={
                "shape": (nz, ny, nx),
                "directions": direction_count,
                "resolution_factor": resolution_factor,
                "dtype": precision,
                "device": self.mi.variant(),
            },
        )
        print(
            f"Mitsuba bake environment lighting (device): start shape={(nz, ny, nx)} "
            f"directions={direction_count} resolution_factor={resolution_factor} "
            f"dtype={precision}",
            flush=True,
        )

        xmin, xmax, ymin, ymax, zmin, zmax = map(float, bounds)
        sx = (xmax - xmin) / max(rx - 1, 1)
        sy = (ymax - ymin) / max(ry - 1, 1)
        sz = (zmax - zmin) / max(rz - 1, 1)
        axis_spacing = (sz, sy, sx)  # z, y, x
        reach = self._shadow_extent_for_volume(volume)
        weight = 4.0 * math.pi / float(max(1, direction_count))

        with self._render_lock:
            # Build a reduced extinction grid directly from the scalar texture.
            # Average extinction (not scalar values) over each source block so
            # nonlinear/sharp opacity mappings do not create light leaks.
            reduced_index = self.dr.arange(self.mi.UInt32, reduced_count)
            ox = reduced_index % self.mi.UInt32(rx)
            oyz = reduced_index // self.mi.UInt32(rx)
            oy = oyz % self.mi.UInt32(ry)
            oz = oyz // self.mi.UInt32(ry)
            sigma = self.dr.zeros(self.mi.Float, reduced_count)
            sample_total = int(resolution_factor ** 3)
            sample_number = 0
            for dz in range(resolution_factor):
                iz = self.dr.minimum(
                    oz * self.mi.UInt32(resolution_factor) + self.mi.UInt32(dz),
                    self.mi.UInt32(nz - 1),
                )
                for dy in range(resolution_factor):
                    iy = self.dr.minimum(
                        oy * self.mi.UInt32(resolution_factor) + self.mi.UInt32(dy),
                        self.mi.UInt32(ny - 1),
                    )
                    for dx in range(resolution_factor):
                        ix = self.dr.minimum(
                            ox * self.mi.UInt32(resolution_factor) + self.mi.UInt32(dx),
                            self.mi.UInt32(nx - 1),
                        )
                        p = self._gpu_texture_position_from_indices(
                            self.mi, ix, iy, iz, nx, ny, nz
                        )
                        scalar = scalar_texture.eval(p)[0]
                        alpha = self._gpu_opacity_from_scalar(scalar, opacity_lut)
                        sigma += -self.dr.log(1.0 - alpha) / opacity_reference_distance
                        sample_number += 1
                        if sample_number % 8 == 0 or sample_number == sample_total:
                            self.dr.eval(sigma)
            sigma /= float(max(1, sample_total))
            self.dr.eval(sigma)
            self._activity_progress(activity_key, 0.08)

            coeffs = [self.dr.zeros(self.mi.Float, reduced_count) for _ in range(9)]
            report_stride = max(1, int(math.ceil(max(1, len(directions)) / 10.0)))

            # Base coordinates of every reduced-grid voxel, reused for the
            # reach-offset lookup after each directional dynamic-programming pass.
            x_all = reduced_index % self.mi.UInt32(rx)
            yz_all = reduced_index // self.mi.UInt32(rx)
            y_all = yz_all % self.mi.UInt32(ry)
            z_all = yz_all // self.mi.UInt32(ry)

            for direction_index, direction in enumerate(directions, start=1):
                d_xyz = np.asarray(direction, dtype=np.float64)
                norm = float(np.linalg.norm(d_xyz))
                if norm <= 1.0e-12:
                    continue
                d_xyz /= norm
                d_axis = np.asarray((d_xyz[2], d_xyz[1], d_xyz[0]), dtype=np.float64)
                dominant = int(np.argmax(np.abs(d_axis)))
                dom = float(d_axis[dominant])
                if abs(dom) <= 1.0e-8:
                    continue

                sizes = reduced_shape
                axes = [0, 1, 2]
                axes.remove(dominant)
                o0, o1 = axes
                step_length = float(axis_spacing[dominant] / abs(dom))
                offset0 = float(d_axis[o0] * step_length / axis_spacing[o0])
                offset1 = float(d_axis[o1] * step_length / axis_spacing[o1])
                tau = self.dr.zeros(self.mi.Float, reduced_count)

                n0, n1 = sizes[o0], sizes[o1]
                slice_count = int(n0 * n1)
                uv = self.dr.arange(self.mi.UInt32, slice_count)
                a = uv // self.mi.UInt32(n1)
                b = uv % self.mi.UInt32(n1)
                if dom > 0.0:
                    slice_indices = range(sizes[dominant] - 1, -1, -1)
                    next_delta = 1
                else:
                    slice_indices = range(0, sizes[dominant])
                    next_delta = -1

                for fixed in slice_indices:
                    coords = [None, None, None]
                    coords[dominant] = self.mi.UInt32(fixed)
                    coords[o0] = a
                    coords[o1] = b
                    flat_index = (
                        coords[0] * self.mi.UInt32(ry * rx)
                        + coords[1] * self.mi.UInt32(rx)
                        + coords[2]
                    )
                    sigma_slice = self.dr.gather(self.mi.Float, sigma, flat_index)
                    nxt = fixed + next_delta
                    if 0 <= nxt < sizes[dominant]:
                        continuation = self._gpu_bilinear_tau_slice(
                            tau,
                            reduced_shape,
                            dominant,
                            nxt,
                            self.mi.Float(a) + offset0,
                            self.mi.Float(b) + offset1,
                        )
                    else:
                        continuation = self.mi.Float(0.0)
                    current = sigma_slice * step_length + continuation
                    self.dr.scatter(tau, current, flat_index)
                    self.dr.eval(tau)

                offsets = (
                    float(direction[2]) * reach / max(sz, 1.0e-12),
                    float(direction[1]) * reach / max(sy, 1.0e-12),
                    float(direction[0]) * reach / max(sx, 1.0e-12),
                )
                tau_end = self._gpu_trilinear_flat(
                    tau,
                    reduced_shape,
                    self.mi.Float(z_all) + offsets[0],
                    self.mi.Float(y_all) + offsets[1],
                    self.mi.Float(x_all) + offsets[2],
                )
                visibility = self.dr.exp(-self.dr.maximum(tau - tau_end, 0.0))
                basis = self._real_sh_basis(direction)
                for index, value in enumerate(basis):
                    coeffs[index] += float(weight * value) * visibility
                self.dr.eval(*coeffs)

                if direction_index % report_stride == 0 or direction_index == len(directions):
                    self._activity_progress(
                        activity_key,
                        0.08 + 0.82 * direction_index / max(1, len(directions)),
                    )

            _tensor_type, storage_type, _texture_type = self._texture_storage_types(precision)
            indices = self.dr.arange(self.mi.UInt32, reduced_count)
            packed6 = self.dr.zeros(storage_type, reduced_count * 6)
            packed3 = self.dr.zeros(storage_type, reduced_count * 3)
            for channel in range(6):
                self.dr.scatter(
                    packed6,
                    storage_type(coeffs[channel]),
                    indices * self.mi.UInt32(6) + self.mi.UInt32(channel),
                )
            for channel in range(3):
                self.dr.scatter(
                    packed3,
                    storage_type(coeffs[channel + 6]),
                    indices * self.mi.UInt32(3) + self.mi.UInt32(channel),
                )
            self.dr.eval(packed6, packed3)
            volume.environment_lighting_volumes = (
                self._make_texture3d_device(
                    packed6,
                    (rz, ry, rx, 6),
                    precision,
                    interpolation="linear",
                ),
                self._make_texture3d_device(
                    packed3,
                    (rz, ry, rx, 3),
                    precision,
                    interpolation="linear",
                ),
            )

        volume.environment_lighting_key = key
        self._activity_progress(activity_key, 0.98)
        elapsed = time.perf_counter() - start
        self._activity_done(activity_key, activity_started)
        print(
            f"Mitsuba bake environment lighting (device): end {elapsed:.3f}s "
            f"field0:channels=6,resolution={(rz, ry, rx)} "
            f"field1:channels=3,resolution={(rz, ry, rx)}",
            flush=True,
        )

    def _ensure_configured_volume_resources(
        self, volumes: tuple[MitsubaRepresentationHandle, ...]
    ) -> None:
        """Build only enabled auxiliary textures that the current shading path uses."""
        for volume in volumes:
            # `off` is an explicit memory-control request, so release old
            # resources even when the current shading path would not use them.
            if volume.gradient_volume_precision == "off":
                volume.gradient_volume = None
                volume.gradient_volume_key = None
            if volume.shadow_volume_precision == "off":
                volume.shadow_volume = None
                volume.shadow_volume_key = None
            if volume.environment_volume_precision == "off":
                volume.environment_lighting_volumes = ()
                volume.environment_lighting_key = None

            if not volume.shade:
                continue

            if volume.gradient_volume_precision != "off":
                self._build_gradient_field(volume)

            if volume.volumetric_scattering_blending <= 0.0:
                continue

            if volume.shadow_volume_precision != "off":
                self._build_directional_shadow_field(volume)

            if (
                volume.environment_scattering_samples > 0
                and volume.environment_volume_precision != "off"
            ):
                self._build_environment_lighting_field(volume)
            elif volume.environment_scattering_samples <= 0:
                volume.environment_lighting_volumes = ()
                volume.environment_lighting_key = None

    # ------------------------------------------------------------------
    # Rendering / frame transport
    # ------------------------------------------------------------------

    def render_pass(
        self,
        view_id: str,
        snapshot: dict[str, Any],
        render_key: tuple,
        *,
        region: tuple[int, int, int, int],
        full_size: tuple[int, int],
        spp: int = 1,
        render_seed: int = 0,
    ) -> np.ndarray:
        """Render one progressive sample for exactly one image-space tile."""
        handle = self._views[view_id]
        spp = max(1, int(spp))
        camera = snapshot["camera"]
        x, y, width, height = map(int, region)
        full_width, full_height = map(int, full_size)

        aperture_size = max(0.0, float(camera["aperture_size"]))
        dof_enabled = aperture_size > 0.0
        sensor_dict: dict[str, Any] = {
            "type": "thinlens" if dof_enabled else "perspective",
            "fov": float(camera["fov"]),
            "fov_axis": "y",
            "to_world": self.mi.ScalarTransform4f().look_at(
                origin=camera["position"],
                target=camera["target"],
                up=camera["up"],
            ),
            "film": {
                    "type": "hdrfilm",
                    # Keep the full logical film size so camera rays match the
                    # browser viewport, but ask Mitsuba to trace only this
                    # rank's crop window. The rendered tensor is tile-sized.
                    "width": full_width,
                    "height": full_height,
                    "crop_offset_x": x,
                    "crop_offset_y": y,
                    "crop_width": width,
                    "crop_height": height,
                    "pixel_format": "rgb",
                },
            "sampler": {"type": "independent", "sample_count": spp},
        }
        if dof_enabled:
            sensor_dict["focus_distance"] = (
                float(camera["focus_distance"])
                if float(camera["focus_distance"]) > 0.0
                else max(
                    float(
                        np.linalg.norm(
                            np.asarray(camera["target"], dtype=np.float64)
                            - np.asarray(camera["position"], dtype=np.float64)
                        )
                    ),
                    1.0e-6,
                )
            )
            sensor_dict["aperture_radius"] = aperture_size

        scene_dict: dict[str, Any] = {
            "type": "scene",
            "integrator": {
                "type": "path",
                "max_depth": 4,
                "hide_emitters": True,
            },
            "sensor": sensor_dict,
            "environment": {
                "type": "constant",
                "radiance": {
                    "type": "rgb",
                    "value": (
                        np.asarray(
                            self._srgb_to_linear(snapshot["world_ambient_color"]),
                            dtype=np.float32,
                        )
                        * snapshot["world_ambient_intensity"]
                    ).tolist(),
                },
            },
        }

        for shape_index, (_kind, representation_id, scene_object) in enumerate(
            snapshot["objects"]
        ):
            scene_dict[f"shape_{shape_index}_{representation_id}"] = scene_object

        # Keep camera-only changes out of the expensive scene/integrator cache
        # keys. Film/crop/FOV and geometry remain structural scene inputs, while
        # pose/focus/aperture are exposed Mitsuba sensor parameters and can be
        # updated in place on the cached scene.
        scene_key = (
            tuple((kind, representation_id, id(scene_object)) for kind, representation_id, scene_object in snapshot["objects"]),
            full_size,
            region,
            float(camera["fov"]),
            bool(dof_enabled),
            int(spp),
            tuple(float(v) for v in snapshot["world_ambient_color"]),
            float(snapshot["world_ambient_intensity"]),
        )

        self._ensure_configured_volume_resources(snapshot["volumes"])

        # Scene creation and render/materialization all touch Dr.Jit runtime
        # state. Keep them serialized across Mitsuba views within this rank.
        with self._render_lock:
            focus_distance = None
            if dof_enabled:
                focus_distance = (
                    float(camera["focus_distance"])
                    if float(camera["focus_distance"]) > 0.0
                    else max(
                        float(
                            np.linalg.norm(
                                np.asarray(camera["target"], dtype=np.float64)
                                - np.asarray(camera["position"], dtype=np.float64)
                            )
                        ),
                        1.0e-6,
                    )
                )
                sensor_key = (
                    tuple(float(v) for v in camera["position"]),
                    tuple(float(v) for v in camera["target"]),
                    tuple(float(v) for v in camera["up"]),
                    float(focus_distance),
                    aperture_size,
                )
            else:
                # With DOF disabled, zoom/orbit only change the sensor transform.
                # Focus distance is deliberately absent from the cache key.
                sensor_key = (
                    tuple(float(v) for v in camera["position"]),
                    tuple(float(v) for v in camera["target"]),
                    tuple(float(v) for v in camera["up"]),
                )

            if handle.cached_scene is None or handle.cached_scene_key != scene_key:
                reason = "initial" if handle.cached_scene is None else "structural-key-change"
                print(
                    f"[Mitsuba] scene rebuild: view={view_id} reason={reason} "
                    f"objects={len(snapshot['objects'])} volumes={len(snapshot['volumes'])} "
                    f"region={region} full_size={full_size}",
                    flush=True,
                )
                handle.cached_scene = self.mi.load_dict(scene_dict)
                handle.cached_scene_key = scene_key
                handle.cached_scene_params = self.mi.traverse(handle.cached_scene)
                handle.cached_sensor_key = sensor_key
            elif handle.cached_sensor_key != sensor_key:
                # Camera pose is traversable for both sensor types. Thin-lens
                # focus/aperture are updated only while DOF is actually enabled.
                params = handle.cached_scene_params
                if params is None:
                    params = self.mi.traverse(handle.cached_scene)
                    handle.cached_scene_params = params
                params["sensor.to_world"] = self.mi.ScalarTransform4f().look_at(
                    origin=camera["position"],
                    target=camera["target"],
                    up=camera["up"],
                )
                if dof_enabled:
                    params["sensor.focus_distance"] = float(focus_distance)
                    params["sensor.aperture_radius"] = aperture_size
                params.update()
                handle.cached_sensor_key = sensor_key
            scene = handle.cached_scene

            # Auxiliary texture presence is a Python-time specialization: each
            # volume gets a DirectVolume type matching exactly the resources it
            # owns. Precision changes recreate textures but do not add runtime
            # branches inside the ray-march kernel.
            structural_volume_key = tuple(
                (
                    id(volume.scalar_volume),
                    tuple(float(v) for v in volume.bounds),
                    id(volume.gradient_volume),
                    id(volume.shadow_volume),
                    tuple(id(field) for field in volume.environment_lighting_volumes),
                    bool(volume.shade),
                    int(volume.environment_scattering_samples),
                    bool(float(volume.volumetric_scattering_blending) > 0.0),
                    bool(float(volume.volumetric_scattering_blending) < 1.0),
                    bool(abs(float(volume.scattering_anisotropy)) < 0.01),
                )
                for volume in snapshot["volumes"]
            )
            integrator_key = (structural_volume_key, bool(snapshot["objects"]))
            ambient_light = tuple(
                component * snapshot["world_ambient_intensity"]
                for component in self._srgb_to_linear(snapshot["world_ambient_color"])
            )
            if handle.integrator is None or handle.integrator_key != integrator_key:
                kernel_activity_key = f"kernel:{view_id}"
                kernel_activity_started = self._activity_start(
                    kernel_activity_key,
                    "Building rendering kernel",
                    view_id=view_id,
                    details={
                        "volumes": len(snapshot["volumes"]),
                        "surfaces": bool(snapshot["objects"]),
                        "gradient": [v.gradient_volume_precision for v in snapshot["volumes"]],
                        "shadow": [v.shadow_volume_precision for v in snapshot["volumes"]],
                        "environment": [v.environment_volume_precision for v in snapshot["volumes"]],
                    },
                    determinate=False,
                )
                print(
                    f"[Mitsuba] integrator rebuild: view={view_id} "
                    f"volumes={len(snapshot['volumes'])} surfaces={bool(snapshot['objects'])} "
                    f"resources={[ (v.gradient_volume_precision, v.shadow_volume_precision, v.environment_volume_precision) for v in snapshot['volumes'] ]}",
                    flush=True,
                )
                direct_volumes = tuple(
                    self._direct_volume_type(
                        gradient=volume.gradient_volume is not None,
                        shadow=volume.shadow_volume is not None,
                        environment=bool(volume.environment_lighting_volumes),
                    )(
                        volume.scalar_volume,
                        volume.bounds,
                        volume.color_mapping,
                        volume.opacity_mapping,
                        volume.sample_distance,
                        volume.opacity_reference_distance,
                        volume.gradient_step,
                        volume.gradient_volume,
                        volume.shade,
                        volume.ambient,
                        volume.diffuse,
                        volume.specular,
                        volume.specular_power,
                        volume.global_illumination_reach,
                        volume.volumetric_scattering_blending,
                        volume.scattering_anisotropy,
                        volume.scalar_range,
                        ambient_light,
                        volume.shadow_volume,
                        volume.environment_scattering_strength,
                        volume.environment_scattering_samples,
                        volume.environment_scattering_step_factor,
                        volume.environment_lighting_volumes,
                    )
                    for volume in snapshot["volumes"]
                )
                handle.integrator = self._dvr_integrator_type(
                    direct_volumes,
                    self._srgb_to_linear(snapshot["background_color"]),
                    bool(snapshot["objects"]),
                )
                handle.integrator_key = integrator_key
                self._activity_done(kernel_activity_key, kernel_activity_started)
            else:
                handle.integrator.update(
                    snapshot["volumes"],
                    background_color=self._srgb_to_linear(snapshot["background_color"]),
                    ambient_light=ambient_light,
                )
            integrator = handle.integrator

            seed = max(1, int(render_seed) + 1)
            try:
                integrator.render(
                    scene=scene,
                    sensor=scene.sensors()[0],
                    seed=seed,
                    spp=spp,
                    develop=False,
                    evaluate=False,
                )
                rendered = np.array(
                    self.mi.Bitmap(scene.sensors()[0].film().develop()),
                    dtype=np.float32,
                    copy=True,
                )
            except Exception as exc:
                print(
                    f"[Mitsuba] render failure: view={view_id} seed={seed} spp={spp} "
                    f"volumes={len(integrator.volumes)} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
                raise

        return rendered[..., :3]

    def _accumulate_pass_worker(
        self,
        handle: MitsubaViewHandle,
        sample: np.ndarray,
        *,
        spp: int,
        render_key: tuple,
    ) -> None:
        """Accumulate one RGB tile sample."""
        if handle.accumulation_key != render_key:
            self._clear_accumulation_worker(handle, render_key)
        if handle.accumulation is None or handle.accumulation.shape != sample.shape:
            handle.accumulation = np.zeros_like(sample, dtype=np.float32)
            handle.accumulated_spp = 0
            handle.accumulation_key = render_key
        handle.accumulation += sample * float(spp)
        handle.accumulated_spp += int(spp)

    def rgb_frame(self, image: np.ndarray) -> tuple[bytes, int, int]:
        """Convert Mitsuba linear RGB output to top-to-bottom RGB24."""
        bitmap = self.mi.Bitmap(image).convert(
            self.mi.Bitmap.PixelFormat.RGB,
            self.mi.Struct.Type.UInt8,
            True,
        )
        rgb = np.asarray(bitmap, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise RuntimeError(f"Unexpected Mitsuba bitmap shape: {rgb.shape}")
        rgb = np.ascontiguousarray(rgb[..., :3])
        height, width, _ = rgb.shape
        return rgb.tobytes(), width, height

    def _resolved_accumulated_frame_worker(
        self, handle: MitsubaViewHandle
    ) -> RenderedFrame | None:
        if handle.accumulation is None or handle.accumulated_spp <= 0:
            return None
        averaged = handle.accumulation / float(handle.accumulated_spp)
        rgb, width, height = self.rgb_frame(averaged)
        return RenderedFrame(rgb=rgb, width=width, height=height)

    def render_frame(
        self, view_id: str, *, region=None, full_size=None
    ) -> RenderedFrame | None:
        """Render and progressively accumulate one full frame or MPI tile."""
        handle = self._views.get(view_id)
        if handle is None:
            return None

        snapshot, revision = self._snapshot_render_state(view_id)
        if full_size is None:
            full_size = (snapshot["width"], snapshot["height"])
        full_width, full_height = map(int, full_size)
        if region is None:
            region = (0, 0, full_width, full_height)
        x, y, width, height = map(int, region)
        region = (x, y, max(1, width), max(1, height))
        full_size = (max(1, full_width), max(1, full_height))
        render_key = (revision, full_size, region)
        if handle.accumulation_key != render_key:
            self._clear_accumulation_worker(handle, render_key)

        if snapshot["volumes"]:
            # Cached mode samples baked TF-dependent lighting fields. Live mode
            # evaluates the same current opacity state through secondary volume
            # marches, while both paths share scalar and gradient volumes.
            dvr_seed = handle.next_seed
            handle.next_seed += 1
            sample = self.render_pass(
                view_id,
                snapshot,
                render_key,
                region=region,
                full_size=full_size,
                spp=1,
                render_seed=dvr_seed,
            )
            self._accumulate_pass_worker(
                handle,
                sample,
                spp=1,
                render_key=render_key,
            )
            return self._resolved_accumulated_frame_worker(handle)

        # Surface-only scenes also use the custom integrator. Give
        # every pass a fresh seed so path tracing / sub-pixel sampling continues
        # to converge instead of replaying the identical sample forever.
        surface_seed = handle.next_seed
        handle.next_seed += 1
        sample = self.render_pass(
            view_id,
            snapshot,
            render_key,
            region=region,
            full_size=full_size,
            spp=1,
            render_seed=surface_seed,
        )
        self._accumulate_pass_worker(
            handle,
            sample,
            spp=1,
            render_key=render_key,
        )
        return self._resolved_accumulated_frame_worker(handle)

    def _visible_bounds_unlocked(
        self, view_id: str
    ) -> tuple[float, float, float, float, float, float] | None:
        bounds = [
            handle.bounds
            for (_, current_view_id), handle in self._representations.items()
            if current_view_id == view_id and handle.bounds is not None
        ]
        if not bounds:
            return None

        return (
            min(item[0] for item in bounds),
            max(item[1] for item in bounds),
            min(item[2] for item in bounds),
            max(item[3] for item in bounds),
            min(item[4] for item in bounds),
            max(item[5] for item in bounds),
        )


def _polydata_transfer_colors(
    polydata: vtk.vtkPolyData,
    *,
    array_name: str,
    association: str,
    transfer_function: dict[str, Any],
) -> np.ndarray | None:
    """Evaluate a renderer-neutral transfer function on one polydata array.

    Point arrays return one RGB triplet per polydata point. Cell arrays return
    one RGB triplet per triangle. Multi-component arrays are reduced to vector
    magnitude before the one-dimensional transfer function is evaluated.
    """

    if association == "cell":
        attributes = polydata.GetCellData()
        expected_count = polydata.GetNumberOfCells()
    elif association == "point":
        attributes = polydata.GetPointData()
        expected_count = polydata.GetNumberOfPoints()
    else:
        return None

    array = attributes.GetArray(array_name)
    if array is None or array.GetNumberOfTuples() != expected_count:
        return None

    values = np.asarray(vtk_to_numpy(array))
    component_count = int(array.GetNumberOfComponents())
    if component_count <= 1:
        scalars = np.asarray(values, dtype=np.float64).reshape(-1)
    else:
        tuples = np.asarray(values, dtype=np.float64).reshape(-1, component_count)
        scalars = np.linalg.norm(tuples, axis=1)

    return _evaluate_transfer_function(transfer_function, scalars)


def _evaluate_transfer_function(
    transfer_function: dict[str, Any],
    values: np.ndarray,
) -> np.ndarray:
    """Map scalar values to RGB using the normalized color mapping."""

    mapping = transfer_function["color"]
    control_points = mapping["control_points"]
    points = np.asarray(control_points, dtype=np.float64)

    minimum, maximum = map(float, mapping["range"])
    scalars = np.asarray(values, dtype=np.float64)
    normalized = (scalars - minimum) / (maximum - minimum)
    positions = points[:, 0]
    rgb = np.column_stack(
        [
            np.interp(
                normalized, positions, np.clip(points[:, channel], 0.0, 1.0)
            )
            for channel in (1, 2, 3)
        ]
    )
    return np.asarray(rgb, dtype=np.float32)



def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = str(value).lstrip("#")
    if len(value) != 6:
        return (0.85, 0.85, 0.85)
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


def _rgb_to_hex(color: tuple[float, float, float]) -> str:
    values = [round(max(0.0, min(1.0, component)) * 255) for component in color]
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def _environment_directions(count: int) -> tuple[tuple[float, float, float], ...]:
    """Return the deterministic sphere directions shared by baked/live lighting."""
    count = max(1, int(count))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    directions = []
    for index in range(count):
        z = 1.0 - 2.0 * ((index + 0.5) / count)
        radial = math.sqrt(max(0.0, 1.0 - z * z))
        phi = golden_angle * index
        directions.append((radial * math.cos(phi), radial * math.sin(phi), z))
    return tuple(directions)


def _make_direct_volume_type(
    mi,
    dr,
    *,
    has_gradient_texture: bool,
    has_shadow_texture: bool,
    has_environment_texture: bool,
):
    """Create one DVR volume type specialized by available GPU textures."""

    class DirectVolume:
        def __init__(
            self,
            scalar_volume,
            bounds,
            color_mapping: dict[str, Any],
            opacity_mapping: dict[str, Any],
            sample_distance: float,
            opacity_reference_distance: float,
            gradient_step: tuple[float, float, float],
            gradient_volume,
            shade: bool,
            ambient: float,
            diffuse: float,
            specular: float,
            specular_power: float,
            global_illumination_reach: float,
            volumetric_scattering_blending: float,
            scattering_anisotropy: float,
            scalar_range: tuple[float, float],
            ambient_light: tuple[float, float, float],
            shadow_volume,
            environment_scattering_strength: float,
            environment_scattering_samples: int,
            environment_scattering_step_factor: float,
            environment_lighting_volumes,
        ) -> None:
            self.scalar_volume = scalar_volume
            self.bounds = tuple(map(float, bounds))
            self.sample_distance = max(float(sample_distance), 1.0e-12)
            self.opacity_reference_distance = max(float(opacity_reference_distance), 1.0e-12)
            self.gradient_step = tuple(max(abs(float(v)), 1.0e-12) for v in gradient_step)
            self.gradient_volume = gradient_volume
            self.shade = bool(shade)
            self.ambient = dr.opaque(mi.Float, max(0.0, float(ambient)))
            self.diffuse = dr.opaque(mi.Float, max(0.0, float(diffuse)))
            self.specular = dr.opaque(mi.Float, max(0.0, float(specular)))
            self.specular_power = dr.opaque(mi.Float, max(1.0, float(specular_power)))
            self.global_illumination_reach = max(
                0.0, min(1.0, float(global_illumination_reach))
            )
            self.volumetric_scattering_blending = max(
                0.0, min(2.0, float(volumetric_scattering_blending))
            )
            self.scattering_anisotropy = max(
                -1.0, min(1.0, float(scattering_anisotropy))
            )
            self.scalar_range = tuple(map(float, scalar_range))
            self.ambient_light = mi.Color3f(*map(float, ambient_light))
            self.shadow_volume = shadow_volume
            self.environment_scattering_strength = dr.opaque(
                mi.Float, max(0.0, float(environment_scattering_strength))
            )
            self.environment_scattering_samples = max(
                0, int(environment_scattering_samples)
            )
            self.environment_scattering_step_factor = max(
                1.0, float(environment_scattering_step_factor)
            )
            self.environment_lighting_volumes = tuple(environment_lighting_volumes or ())
            self._color_mapping_signature = self._mapping_signature(color_mapping)
            self._opacity_mapping_signature = self._mapping_signature(opacity_mapping)
            self.color = self._compile_mapping(color_mapping)
            self.opacity = self._compile_mapping(opacity_mapping)

        @staticmethod
        def _mapping_signature(mapping):
            return (
                tuple(float(v) for v in mapping["range"]),
                tuple(
                    tuple(float(component) for component in point)
                    for point in mapping["control_points"]
                ),
            )

        def update_runtime(self, volume, *, ambient_light) -> None:
            """Refresh non-structural inputs without replacing the integrator.

            Structural inputs are deliberately handled by the compact cache key
            in ``render_pass``. Everything here is cheap state replacement.
            """
            color_signature = self._mapping_signature(volume.color_mapping)
            if color_signature != self._color_mapping_signature:
                self.color = self._compile_mapping(volume.color_mapping)
                self._color_mapping_signature = color_signature

            opacity_signature = self._mapping_signature(volume.opacity_mapping)
            if opacity_signature != self._opacity_mapping_signature:
                self.opacity = self._compile_mapping(volume.opacity_mapping)
                self._opacity_mapping_signature = opacity_signature

            self.sample_distance = max(float(volume.sample_distance), 1.0e-12)
            self.opacity_reference_distance = max(
                float(volume.opacity_reference_distance), 1.0e-12
            )
            self.gradient_step = tuple(
                max(abs(float(v)), 1.0e-12) for v in volume.gradient_step
            )
            self.ambient = dr.opaque(mi.Float, max(0.0, float(volume.ambient)))
            self.diffuse = dr.opaque(mi.Float, max(0.0, float(volume.diffuse)))
            self.specular = dr.opaque(mi.Float, max(0.0, float(volume.specular)))
            self.specular_power = dr.opaque(
                mi.Float, max(1.0, float(volume.specular_power))
            )
            self.global_illumination_reach = max(
                0.0, min(1.0, float(volume.global_illumination_reach))
            )
            self.volumetric_scattering_blending = max(
                0.0, min(2.0, float(volume.volumetric_scattering_blending))
            )
            self.scattering_anisotropy = max(
                -1.0, min(1.0, float(volume.scattering_anisotropy))
            )
            self.scalar_range = tuple(map(float, volume.scalar_range))
            self.ambient_light = mi.Color3f(*map(float, ambient_light))
            self.environment_scattering_strength = dr.opaque(
                mi.Float, max(0.0, float(volume.environment_scattering_strength))
            )
            self.environment_scattering_step_factor = max(
                1.0, float(volume.environment_scattering_step_factor)
            )
            self.shadow_volume = volume.shadow_volume
            self.environment_lighting_volumes = tuple(
                volume.environment_lighting_volumes or ()
            )

        @staticmethod
        def _compile_mapping(mapping):
            minimum, maximum = map(float, mapping["range"])
            width = maximum - minimum
            if abs(width) < 1.0e-20:
                width = 1.0
            points = np.asarray(mapping["control_points"], dtype=np.float32)
            positions = points[:, 0]
            size = 1024
            x = np.linspace(0.0, 1.0, size, dtype=np.float32)
            channels = points.shape[1] - 1
            lut = np.empty((size, channels), dtype=np.float32)
            for channel in range(channels):
                lut[:, channel] = np.interp(
                    x, positions, np.clip(points[:, channel + 1], 0.0, 1.0)
                ).astype(np.float32)
            return (
                dr.opaque(mi.Float, minimum),
                dr.opaque(mi.Float, 1.0 / width),
                tuple(mi.Float(np.ascontiguousarray(lut[:, c])) for c in range(channels)),
                size,
            )

        @staticmethod
        def _normalized_scalar(scalar, compiled):
            minimum, inv_width, _lut, _size = compiled
            return dr.clip((scalar - minimum) * inv_width, 0.0, 1.0)

        @staticmethod
        def _lut_sample(x, compiled, active=True):
            _minimum, _inv_width, channels, size = compiled
            scaled = dr.clip(x, 0.0, 1.0) * float(size - 1)
            i0 = mi.UInt32(dr.floor(scaled))
            i1 = dr.minimum(i0 + 1, mi.UInt32(size - 1))
            t = scaled - mi.Float(i0)
            values = []
            for channel in channels:
                v0 = dr.gather(mi.Float, channel, i0, active)
                v1 = dr.gather(mi.Float, channel, i1, active)
                values.append(dr.lerp(v0, v1, t))
            return values

        @classmethod
        def _opacity(cls, x, compiled, active=True):
            return dr.clip(cls._lut_sample(x, compiled, active)[0], 0.0, 1.0)

        @classmethod
        def _color(cls, x, compiled, active=True):
            values = cls._lut_sample(x, compiled, active)
            return dr.clip(mi.Color3f(values[0], values[1], values[2]), 0.0, 1.0)

        def _texture_position(self, position, texture):
            """Map world coordinates to Dr.Jit texture coordinates.

            VTK samples lie on the world-space bounds, while Dr.Jit places
            texels at half-cell centers. Account for that half-cell convention
            so texture interpolation reproduces the original endpoint grid.
            """
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            shape = tuple(int(v) for v in texture.shape)
            if len(shape) >= 4:
                nz, ny, nx = shape[-4:-1]
            else:
                nz, ny, nx = shape[:3]

            def axis(value, lower, upper, count):
                t = (value - lower) / (upper - lower)
                if count <= 1:
                    return mi.Float(0.5)
                return (t * float(count - 1) + 0.5) / float(count)

            return mi.Point3f(
                axis(position.x, xmin, xmax, nx),
                axis(position.y, ymin, ymax, ny),
                axis(position.z, zmin, zmax, nz),
            )

        def _sample_scalar(self, position, active=True):
            return self.scalar_volume.eval(
                self._texture_position(position, self.scalar_volume), active
            )[0]

        def _gradient(self, position, active=True):
            if has_gradient_texture:
                value = self.gradient_volume.eval(
                    self._texture_position(position, self.gradient_volume), active
                )
                return mi.Vector3f(value[0], value[1], value[2])
            # Live central differences are used when the gradient texture is off.
            hx, hy, hz = self.gradient_step
            dx = mi.Vector3f(hx, 0.0, 0.0)
            dy = mi.Vector3f(0.0, hy, 0.0)
            dz = mi.Vector3f(0.0, 0.0, hz)
            gx = (self._sample_scalar(position + dx, active) - self._sample_scalar(position - dx, active)) / (2.0 * hx)
            gy = (self._sample_scalar(position + dy, active) - self._sample_scalar(position - dy, active)) / (2.0 * hy)
            gz = (self._sample_scalar(position + dz, active) - self._sample_scalar(position - dz, active)) / (2.0 * hz)
            return mi.Vector3f(gx, gy, gz)

        def _phase_function(self, cos_angle):
            g = self.scattering_anisotropy
            if abs(g) < 0.01:
                return mi.Float(1.0)
            g2 = g * g
            d = dr.maximum(1.0 + g2 - 2.0 * g * cos_angle, 1.0e-6)
            # Match VTK's 4*pi-normalized Henyey-Greenstein convention.
            return (1.0 - g2) / (d * dr.sqrt(d))

        def _shadow_extent(self):
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            dx = xmax - xmin
            dy = ymax - ymin
            dz = zmax - zmin
            max_extent = math.sqrt(dx * dx + dy * dy + dz * dz)
            min_extent = min(self.opacity_reference_distance, max_extent)
            reach = self.global_illumination_reach
            # VTK maps [0, 1] non-linearly from roughly one primary sample
            # to the full volume diagonal. Preserve that behavior in world space.
            return (min_extent - max_extent) * ((1.0 - reach) ** 0.33) + max_extent

        def _opacity_at(self, position, step_distance, active=True):
            scalar = self._sample_scalar(position, active)
            opacity_x = self._normalized_scalar(scalar, self.opacity)
            alpha = self._opacity(opacity_x, self.opacity, active)
            ratio = step_distance / self.opacity_reference_distance
            transmission = dr.maximum(1.0 - alpha, 1.0e-6)
            return dr.clip(1.0 - dr.power(transmission, ratio), 0.0, 1.0)

        def _sample_shadow_tau(self, position, active=True):
            if not has_shadow_texture:
                return mi.Float(0.0)
            return self.shadow_volume.eval(
                self._texture_position(position, self.shadow_volume), active
            )[0]

        def _distance_to_volume_exit(self, position, direction):
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            eps = 1.0e-12
            inf = mi.Float(float("inf"))

            def axis_distance(p, d, lower, upper):
                positive = d > eps
                negative = d < -eps
                return dr.select(
                    positive,
                    (upper - p) / d,
                    dr.select(negative, (lower - p) / d, inf),
                )

            tx = axis_distance(position.x, direction.x, xmin, xmax)
            ty = axis_distance(position.y, direction.y, ymin, ymax)
            tz = axis_distance(position.z, direction.z, zmin, zmax)
            return dr.maximum(dr.minimum(tx, dr.minimum(ty, tz)), 0.0)

        def _live_transmittance(
            self,
            position,
            direction,
            *,
            step_distance,
            active=True,
        ):
            """March the current opacity TF directly through the scalar volume."""
            direction = dr.normalize(direction)
            step = mi.Float(max(float(step_distance), self.sample_distance))
            start = position + direction * (0.5 * step)
            max_distance = dr.minimum(
                mi.Float(self._shadow_extent()),
                self._distance_to_volume_exit(start, direction),
            )
            t = mi.Float(0.0)
            transmission = mi.Float(1.0)
            march_active = active & (max_distance > 0.0)

            def loop_cond(t, transmission, march_active):
                del t, transmission
                return march_active

            def loop_body(t, transmission, march_active):
                remaining = dr.maximum(max_distance - t, 0.0)
                current_step = dr.minimum(step, remaining)
                probe = start + direction * t
                alpha = self._opacity_at(probe, current_step, march_active)
                transmission = transmission * (1.0 - alpha)
                t = t + current_step
                march_active = (
                    active
                    & (t < max_distance)
                    & (transmission > 1.0e-4)
                    & (current_step > 0.0)
                )
                return t, transmission, march_active

            t, transmission, march_active = dr.while_loop(
                state=(t, transmission, march_active),
                cond=loop_cond,
                body=loop_body,
                mode="symbolic",
                label="vtkweb live secondary volume march",
            )
            return dr.clip(transmission, 0.0, 1.0)

        def _volume_shadow(self, position, light_direction, sample_index, active=True):
            del sample_index
            direction = dr.normalize(light_direction)
            if has_shadow_texture:
                base_step = mi.Float(self.sample_distance)
                start = position + direction * base_step
                reach = mi.Float(self._shadow_extent())
                xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
                start_z = dr.clip(start.z, zmin, zmax)
                end_z = dr.minimum(start_z + reach, zmax)
                start_p = mi.Point3f(start.x, start.y, start_z)
                end_p = mi.Point3f(start.x, start.y, end_z)
                tau_start = self._sample_shadow_tau(start_p, active)
                tau_end = self._sample_shadow_tau(end_p, active)
                tau = dr.maximum(tau_start - tau_end, 0.0)
                return dr.exp(-tau)

            return self._live_transmittance(
                position,
                direction,
                step_distance=self.sample_distance,
                active=active,
            )

        @staticmethod
        def _sh_basis(direction):
            x = direction.x
            y = direction.y
            z = direction.z
            return (
                mi.Float(0.28209479177387814),
                0.4886025119029199 * y,
                0.4886025119029199 * z,
                0.4886025119029199 * x,
                1.0925484305920792 * x * y,
                1.0925484305920792 * y * z,
                0.31539156525252005 * (3.0 * z * z - 1.0),
                1.0925484305920792 * x * z,
                0.5462742152960396 * (x * x - y * y),
            )

        def _environment_scatter(
            self, color, position, view, sample_index, active=True
        ):
            del sample_index
            if self.environment_scattering_samples <= 0:
                return mi.Color3f(0.0)

            if has_environment_texture:
                # Incoming directions were projected into real SH coefficients in
                # volume space. Camera-dependent Henyey-Greenstein weighting is
                # applied here without rebaking the visibility field.
                query_direction = -dr.normalize(view)
                basis = self._sh_basis(query_direction)
                g = mi.Float(self.scattering_anisotropy)
                if len(self.environment_lighting_volumes) != 2:
                    return mi.Color3f(0.0)
                c0_texture = self.environment_lighting_volumes[0]
                c1_texture = self.environment_lighting_volumes[1]
                c0 = c0_texture.eval(
                    self._texture_position(position, c0_texture), active
                )
                c1 = c1_texture.eval(
                    self._texture_position(position, c1_texture), active
                )
                g2 = g * g
                total = (
                    c0[0] * basis[0]
                    + g * (c0[1] * basis[1] + c0[2] * basis[2] + c0[3] * basis[3])
                    + g2 * (
                        c0[4] * basis[4]
                        + c0[5] * basis[5]
                        + c1[0] * basis[6]
                        + c1[1] * basis[7]
                        + c1[2] * basis[8]
                    )
                )
                total = dr.maximum(total, 0.0)
            else:
                # Live mode evaluates visibility from the current opacity TF for
                # the same deterministic sphere directions used by the baker.
                directions = _environment_directions(self.environment_scattering_samples)
                query_view = dr.normalize(view)
                total = mi.Float(0.0)
                step = self.sample_distance * max(1.0, float(self.environment_scattering_step_factor))
                for direction_tuple in directions:
                    direction = mi.Vector3f(*direction_tuple)
                    visibility = self._live_transmittance(
                        position,
                        direction,
                        step_distance=step,
                        active=active,
                    )
                    phase = self._phase_function(dr.dot(-direction, query_view))
                    total = total + visibility * phase
                total = total / float(max(1, len(directions)))

            return (
                color
                * self.diffuse
                * self.environment_scattering_strength
                * total
                * self.ambient_light
            )

        def _shade_color(
            self, color, alpha, position, view_direction, sample_index, active=True
        ):
            gradient = self._gradient(position, active)
            gradient_length = dr.norm(gradient)
            valid_normal = active & (gradient_length > 1.0e-12)
            normal = gradient / dr.maximum(gradient_length, 1.0e-12)
            view = dr.normalize(view_direction)

            # Fixed world-space directional light. `light_direction` points
            # from the sample toward the light, so light travels along -Z.
            light_direction = mi.Vector3f(0.0, 0.0, 1.0)

            # Match VTK's default two-sided volume lighting: gradients facing
            # away from the light contribute with the opposite orientation.
            ndotl_signed = dr.dot(normal, light_direction)
            ndotl = dr.abs(ndotl_signed)

            # Blinn-Phong specular term using the fixed light and camera view.
            half_vector = dr.normalize(light_direction + view)
            ndoth = dr.clip(dr.abs(dr.dot(normal, half_vector)), 0.0, 1.0)

            ambient_term = self.ambient * self.ambient_light
            diffuse_term = self.diffuse * ndotl
            specular_term = self.specular * dr.power(ndoth, self.specular_power)
            local = color * (ambient_term + diffuse_term) + mi.Color3f(specular_term)
            local = dr.select(valid_normal, local, color * ambient_term)

            if self.volumetric_scattering_blending <= 0.0:
                return local

            advanced_active = active

            # Stochastic transmittance estimate toward the fixed
            # directional light. Every shaded primary sample receives a cheap
            # estimate on every progressive frame.
            shadow = self._volume_shadow(
                position, light_direction, sample_index, advanced_active
            )
            phase_cosine = dr.dot(-light_direction, view)
            phase = self._phase_function(phase_cosine)
            secondary = (
                shadow * phase * color * self.diffuse
                + self.ambient * self.ambient_light
            )
            secondary = secondary + self._environment_scatter(
                color, position, view, sample_index, advanced_active
            )

            # VTK's shader uses the normalized scalar-gradient magnitude in
            # its surface/volumetric blend. Approximate the same dimensionless
            # quantity in world space by scaling the derivative with one voxel
            # and normalizing by 25% of the scalar data range.
            scalar_width = max(abs(self.scalar_range[1] - self.scalar_range[0]), 1.0e-12)
            voxel_scale = min(self.gradient_step)
            gradient_factor = dr.clip(
                gradient_length * voxel_scale / (0.25 * scalar_width),
                0.0,
                1.0,
            )
            gradient_factor = dr.select(valid_normal, gradient_factor, 0.0)

            # VTK stores half of the public [0, 2] scattering value in the
            # shader and uses this piecewise blend between local/surface and
            # volumetric shading.
            public_blend = self.volumetric_scattering_blending
            b = 0.5 * public_blend
            exponential = dr.exp(-2.0 * b * gradient_factor * alpha)
            if public_blend < 1.0:
                volumetric_weight = 2.0 * b * exponential
            else:
                volumetric_weight = (
                    2.0 * (1.0 - b) * exponential + 2.0 * b - 1.0
                )
            volumetric_weight = dr.clip(volumetric_weight, 0.0, 1.0)
            advanced = dr.lerp(local, secondary, volumetric_weight)
            return dr.select(advanced_active, advanced, local)

        def evaluate_alpha(self, position, step_distance, active=True):
            scalar = self._sample_scalar(position, active)
            opacity_x = self._normalized_scalar(scalar, self.opacity)
            alpha = self._opacity(opacity_x, self.opacity, active)
            ratio = step_distance / self.opacity_reference_distance
            transmission = dr.maximum(1.0 - alpha, 1.0e-6)
            return dr.clip(1.0 - dr.power(transmission, ratio), 0.0, 1.0)

        def evaluate(
            self,
            position,
            step_distance,
            view_direction,
            active=True,
            sample_index=None,
        ):
            scalar = self._sample_scalar(position, active)

            opacity_x = self._normalized_scalar(scalar, self.opacity)
            color_x = self._normalized_scalar(scalar, self.color)
            alpha = self._opacity(opacity_x, self.opacity, active)
            color = self._color(color_x, self.color, active)

            # TF opacity is defined for a fixed physical reference distance
            # derived from the voxel spacing. The marcher step is independent.
            ratio = step_distance / self.opacity_reference_distance
            transmission = dr.maximum(1.0 - alpha, 1.0e-6)
            alpha = 1.0 - dr.power(transmission, ratio)
            alpha = dr.clip(alpha, 0.0, 1.0)

            if self.shade:
                shade_active = active & (alpha > 1.0e-4)
                color = dr.select(
                    shade_active,
                    self._shade_color(
                        color,
                        alpha,
                        position,
                        -view_direction,
                        sample_index,
                        shade_active,
                    ),
                    color,
                )
            return color, alpha

        def ray_segment(self, ray, active=True):
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            lower = mi.Point3f(xmin, ymin, zmin)
            upper = mi.Point3f(xmax, ymax, zmax)
            inv_d = dr.rcp(ray.d)
            t0 = (lower - ray.o) * inv_d
            t1 = (upper - ray.o) * inv_d
            t_near = dr.maximum(
                dr.maximum(dr.minimum(t0.x, t1.x), dr.minimum(t0.y, t1.y)),
                dr.minimum(t0.z, t1.z),
            )
            t_far = dr.minimum(
                dr.minimum(dr.maximum(t0.x, t1.x), dr.maximum(t0.y, t1.y)),
                dr.maximum(t0.z, t1.z),
            )
            entry = dr.maximum(t_near, mi.Float(0.0))
            valid = active & (t_far >= entry)
            return entry, t_far, valid

    return DirectVolume


def _make_dvr_integrator_type(mi, dr):
    """Create the vtkweb direct-volume integrator with opaque surface composition.

    Surface geometry is shaded by Mitsuba's normal path integrator. The nearest
    surface intersection also clips the DVR march, so outlines and opaque
    surfaces correctly hide voxels behind them without participating in the
    baked volume-lighting fields.
    """

    class DirectVolumeIntegrator(mi.SamplingIntegrator):
        def __init__(self, volumes, background_color, has_surfaces=False) -> None:
            super().__init__(mi.Properties())
            self.volumes = tuple(volumes)
            self.background_color = mi.Color3f(*map(float, background_color))
            self.has_surfaces = bool(has_surfaces)
            self.surface_integrator = (
                mi.load_dict(
                    {
                        "type": "path",
                        "max_depth": 4,
                        "hide_emitters": True,
                    }
                )
                if self.has_surfaces
                else None
            )

        def update(self, volumes, *, background_color, ambient_light):
            self.background_color = mi.Color3f(*map(float, background_color))
            for direct_volume, volume in zip(self.volumes, volumes):
                direct_volume.update_runtime(
                    volume,
                    ambient_light=ambient_light,
                )

        def sample(self, scene, sampler, ray, medium=None, active=True):
            ray = mi.Ray3f(ray)
            active = mi.Bool(active)

            surface_result = mi.Color3f(0.0)
            surface_valid = mi.Bool(False)
            surface_limit = mi.Float(float("inf"))
            if self.surface_integrator is not None:
                si = scene.ray_intersect(ray, active=active)
                surface_valid = active & si.is_valid()
                surface_limit = dr.select(surface_valid, si.t, surface_limit)
                surface_result, _path_valid, _ = self.surface_integrator.sample(
                    scene, sampler, ray, medium, active
                )

            result = mi.Color3f(0.0)
            accumulated_alpha = mi.Float(0.0)
            volume_hit = mi.Bool(False)

            # The Python loop is only over the small/static set of vtkweb volume
            # representations. The per-ray march itself is an explicit symbolic
            # Dr.Jit loop, so it executes in generated device code without
            # relying on @dr.syntax's transformed name-resolution environment.
            for volume_index, volume in enumerate(self.volumes):
                entry, exit, volume_active = volume.ray_segment(ray, active)
                exit = dr.minimum(exit, surface_limit)
                volume_active = volume_active & (exit >= entry)
                volume_hit = volume_hit | volume_active

                step = mi.Float(volume.sample_distance)
                t = entry + 0.5 * step
                sample_index = mi.UInt32(0)
                march_active = (
                    volume_active
                    & (t <= exit)
                    & (accumulated_alpha < 0.995)
                )

                def loop_cond(t, sample_index, result, accumulated_alpha, march_active):
                    del t, sample_index, result, accumulated_alpha
                    return march_active

                def loop_body(t, sample_index, result, accumulated_alpha, march_active):
                    position = ray.o + ray.d * t
                    color, alpha = volume.evaluate(
                        position,
                        step,
                        ray.d,
                        march_active,
                        sample_index,
                    )
                    alpha_before = accumulated_alpha
                    weight = (1.0 - alpha_before) * alpha
                    result = result + weight * color
                    alpha_after = alpha_before + weight


                    accumulated_alpha = alpha_after
                    t = t + step
                    sample_index = sample_index + 1
                    march_active = (
                        volume_active
                        & (t <= exit)
                        & (accumulated_alpha < 0.995)
                    )
                    return (t, sample_index, result, accumulated_alpha, march_active)

                try:
                    (
                        t,
                        sample_index,
                        result,
                        accumulated_alpha,
                        march_active,
                    ) = dr.while_loop(
                        state=(t, sample_index, result, accumulated_alpha, march_active),
                        cond=loop_cond,
                        body=loop_body,
                        mode="symbolic",
                        label="vtkweb direct volume march",
                    )
                except Exception as exc:
                    env_fields = volume.environment_lighting_volumes
                    env_meta = []
                    for field_index, field in enumerate(env_fields):
                        try:
                            env_meta.append(
                                f"{field_index}:channels={int(field.channel_count())},"
                                f"resolution={tuple(int(v) for v in field.resolution())}"
                            )
                        except Exception as meta_exc:
                            env_meta.append(f"{field_index}:metadata-error={meta_exc}")
                    print(
                        f"[Mitsuba] DVR while_loop failure: volume_index={volume_index} "
                        f"env_samples={volume.environment_scattering_samples} "
                        f"env_fields=[{' ; '.join(env_meta)}] "
                        f"shade={volume.shade} scattering={volume.volumetric_scattering_blending} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                    raise

            remaining = 1.0 - accumulated_alpha
            behind = dr.select(surface_valid, surface_result, self.background_color)
            result = result + remaining * behind
            return mi.Spectrum(result), (volume_hit | surface_valid), []

        def to_string(self):
            return (
                f"DirectVolumeIntegrator[volumes={len(self.volumes)}, "
                f"surfaces={self.has_surfaces}]"
            )

    return DirectVolumeIntegrator


def _image_scalar_values(
    image: vtk.vtkImageData,
    array_name: str,
    association: str,
) -> np.ndarray | None:
    dimensions = tuple(int(v) for v in image.GetDimensions())
    if association == "point":
        attributes = image.GetPointData()
        grid_dimensions = dimensions
    elif association == "cell":
        attributes = image.GetCellData()
        grid_dimensions = tuple(max(0, value - 1) for value in dimensions)
    else:
        return None

    if any(value <= 0 for value in grid_dimensions):
        return None

    array = attributes.GetArray(array_name)
    if array is None:
        return None

    expected_count = int(np.prod(grid_dimensions))
    if int(array.GetNumberOfTuples()) != expected_count:
        return None

    # Keep the VTK-owned scalar buffer in its native dtype.  For ordinary
    # single-component CT data this is a zero-copy NumPy view (typically int16).
    # The Dr.Jit upload path converts directly into the requested GPU texture
    # precision, avoiding a second full-resolution host allocation.
    values = np.asarray(vtk_to_numpy(array))
    components = int(array.GetNumberOfComponents())
    if components <= 1:
        scalars = values.reshape(-1)
    else:
        # Vector magnitude necessarily creates a derived scalar array. Keep this
        # exceptional path in float32; scalar medical volumes do not use it.
        scalars = np.linalg.norm(
            values.reshape(-1, components).astype(np.float32, copy=False), axis=1
        ).astype(np.float32, copy=False)

    nx, ny, nz = grid_dimensions
    shaped = scalars.reshape(nz, ny, nx)
    return shaped if shaped.flags.c_contiguous else np.ascontiguousarray(shaped)



def _volume_gradient_step(image: vtk.vtkImageData) -> tuple[float, float, float]:
    spacing = tuple(abs(float(value)) for value in image.GetSpacing())
    fallback = _volume_sample_distance(image, {"auto_adjust_sample_distances": True})
    return tuple(value if value > 0.0 else fallback for value in spacing)

def _volume_opacity_reference_distance(image: vtk.vtkImageData) -> float:
    """Physical distance for which transfer-function opacity is defined.

    Keep this independent of the user-controlled ray-march step so changing
    sample_distance changes integration accuracy, not material extinction.
    The minimum non-zero voxel spacing preserves the previous appearance at
    the default auto-adjusted sample distance.
    """
    spacing = [abs(float(value)) for value in image.GetSpacing() if abs(float(value)) > 0]
    if spacing:
        return min(spacing)
    return 1.0


def _volume_sample_distance(
    image: vtk.vtkImageData,
    properties: dict[str, Any],
) -> float:
    if properties.get("auto_adjust_sample_distances", True):
        spacing = [abs(float(value)) for value in image.GetSpacing() if abs(float(value)) > 0]
        if spacing:
            return min(spacing)
    return max(float(properties.get("sample_distance", 1.0)), 1.0e-12)
