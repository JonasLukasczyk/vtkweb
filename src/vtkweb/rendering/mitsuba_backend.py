from __future__ import annotations

import importlib
import os
import math
import threading
import time
import traceback
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from vtkweb.rendering.base import (
    DIRECTIONAL_LIGHT_DIRECTIONS,
    DIRECTIONAL_LIGHT_PROPERTY_NAMES,
    RenderedFrame,
    RenderView,
    RenderingBackend,
    Representation,
)
# Experimental 1-byte directional shadow cache: "u8" in shadow_volume.
# Values are averaged over active lights before quantization, and restored
# to summed illumination when sampled. Other volume formats are unchanged.

VPT_DEBUG = os.environ.get("VTKWEB_VPT_DEBUG", "0").lower() in ("1", "true", "yes")

DEBUG_SHADOW_STORAGE = True  # Print actual GPU array types and logical storage size after baking.


def _debug_shadow_array(label, data, voxel_count, expected_bytes):
    """Metadata-only checks: no CPU download of the volume."""
    dtype = getattr(data, "dtype", None)
    dtype_name = str(dtype)
    cls = type(data)
    try:
        element_bytes = np.dtype(dtype).itemsize
    except (TypeError, ValueError):
        element_bytes = None
    actual_bytes = voxel_count * element_bytes if element_bytes is not None else None
    print(
        f"[shadow-storage] {label}: class={cls.__module__}.{cls.__name__}, "
        f"dtype={dtype_name}, elements={voxel_count:,}, "
        f"bytes/element={element_bytes}, logical_bytes={actual_bytes}, "
        f"expected_bytes={expected_bytes:,}", flush=True,
    )
    return element_bytes


class _ShadowUNorm8:
    """GPU-resident 1-byte/voxel field with manual trilinear interpolation."""

    def __init__(self, data, shape, mi, dr, light_count):
        self.data = data
        self.shape = tuple(shape)
        self.mi = mi
        self.dr = dr
        self.light_count = int(light_count)

    def eval(self, uv, active=True):
        mi, dr = self.mi, self.dr
        nz, ny, nx = self.shape[:3]
        # Match the existing half-texel mapping and clamp texture boundaries.
        fx = dr.clip(uv.x * nx - 0.5, 0.0, float(nx - 1))
        fy = dr.clip(uv.y * ny - 0.5, 0.0, float(ny - 1))
        fz = dr.clip(uv.z * nz - 0.5, 0.0, float(nz - 1))
        x0 = mi.UInt32(dr.floor(fx)); y0 = mi.UInt32(dr.floor(fy)); z0 = mi.UInt32(dr.floor(fz))
        x1 = dr.minimum(x0 + 1, nx - 1)
        y1 = dr.minimum(y0 + 1, ny - 1)
        z1 = dr.minimum(z0 + 1, nz - 1)
        tx = fx - mi.Float(x0); ty = fy - mi.Float(y0); tz = fz - mi.Float(z0)

        def at(x, y, z):
            index = (z * ny + y) * nx + x
            return mi.Float(dr.gather(type(self.data), self.data, index, active))

        a00 = dr.lerp(at(x0,y0,z0), at(x1,y0,z0), tx)
        a10 = dr.lerp(at(x0,y1,z0), at(x1,y1,z0), tx)
        a01 = dr.lerp(at(x0,y0,z1), at(x1,y0,z1), tx)
        a11 = dr.lerp(at(x0,y1,z1), at(x1,y1,z1), tx)
        value = dr.lerp(dr.lerp(a00,a10,ty), dr.lerp(a01,a11,ty), tz)
        return (value * (self.light_count / 255.0),)


PREINTEGRATION_SIZE = 512

# DEBUG: live shadow distance before consulting the 1x baked illumination.
# Measured in scalar-voxel spacings along the active axis-aligned light.
# 0.0 restores the original cached-only path.
DEBUG_HYBRID_SHADOW_VOXELS = 4.0
DEBUG_HYBRID_ENV_VOXELS = 2.0  # 0 = six cached lookups; 2 = two voxels of live correction


PREINTEGRATION_SAMPLES = 48


def _diagnose_drjit_exception(stage, exc):
    """Report nested exceptions on stderr without swallowing the original."""
    print(f"[Mitsuba DIAGNOSTIC] {stage}: {type(exc).__name__}: {exc!r}", file=sys.stderr, flush=True)
    traceback.print_exception(type(exc), exc, exc.__traceback__, chain=True, file=sys.stderr)
    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        print(f"[Mitsuba DIAGNOSTIC] underlying: {type(cause).__name__}: {cause!r}", file=sys.stderr, flush=True)



def _build_gpu_preintegration_tables(
    mi, dr, mapping, step_ratio, *, size=PREINTEGRATION_SIZE,
    integration_samples=PREINTEGRATION_SAMPLES, with_centroid=True,
):
    """Generate preintegrated extinction and centroid directly on the GPU.

    Each GPU lane owns one (start, end) interval. The integration loop is a
    symbolic Dr.Jit loop, with no intermediate CPU readbacks or per-step evals.
    The resulting arrays stay on the GPU for use by DVR and the shadow baker.
    """
    points = np.asarray(mapping["control_points"], dtype=np.float32)
    if len(points) < 1:
        raise ValueError("Opacity mapping must contain at least one control point")
    count = int(size) * int(size)
    lane = dr.arange(mi.UInt32, count)
    start = mi.Float(lane // mi.UInt32(size)) / float(size - 1)
    end = mi.Float(lane % mi.UInt32(size)) / float(size - 1)
    positions = mi.Float(np.ascontiguousarray(points[:, 0]))
    opacities = mi.Float(np.ascontiguousarray(np.clip(points[:, 1], 0.0, 1.0)))
    point_count = len(points)

    def opacity_at(x):
        # Binary search the piecewise-linear TF, with NumPy interp's endpoint
        # clamping. Loop bounds depend on control point count, not table size.
        lo = dr.zeros(mi.UInt32, count)
        hi = dr.full(mi.UInt32, point_count - 1, count)
        # Fixed binary-search depth avoids a global reduction/synchronization
        # inside each integration sample. All lanes execute the same steps.
        for _ in range((point_count - 1).bit_length()):
            mid = (lo + hi) // 2
            unresolved = hi - lo > 1
            before = x < dr.gather(mi.Float, positions, mid)
            hi = dr.select(unresolved & before, mid, hi)
            lo = dr.select(unresolved & ~before, mid, lo)
        x0 = dr.gather(mi.Float, positions, lo)
        x1 = dr.gather(mi.Float, positions, hi)
        a0 = dr.gather(mi.Float, opacities, lo)
        a1 = dr.gather(mi.Float, opacities, hi)
        frac = dr.clip((x - x0) / dr.maximum(x1 - x0, 1.0e-20), 0.0, 1.0)
        return dr.clip(dr.lerp(a0, a1, frac), 0.0, 1.0)

    # dr.opaque prevents specializing a new integration kernel for each
    # sampling ratio. It also keeps runtime parameters GPU-native.
    ratio = dr.opaque(mi.Float, float(step_ratio))
    k = mi.UInt32(0)
    total = dr.zeros(mi.Float, count)
    optical_depth = dr.zeros(mi.Float, count)
    weighted_t = dr.zeros(mi.Float, count)
    weight_sum = dr.zeros(mi.Float, count)

    def cond(k, total, optical_depth, weighted_t, weight_sum):
        return k < integration_samples

    def body(k, total, optical_depth, weighted_t, weight_sum):
        t = (mi.Float(k) + 0.5) / float(integration_samples)
        scalar = dr.lerp(start, end, t)
        alpha = opacity_at(scalar)
        extinction = -dr.log(dr.maximum(1.0 - alpha, 1.0e-6))
        total = total + extinction / float(integration_samples)
        if with_centroid:
            delta = extinction * (ratio / float(integration_samples))
            weight = dr.exp(-optical_depth) * (1.0 - dr.exp(-delta))
            weighted_t = weighted_t + weight * t
            weight_sum = weight_sum + weight
            optical_depth = optical_depth + delta
        return k + 1, total, optical_depth, weighted_t, weight_sum

    try:
        _k, tau, _optical_depth, weighted_t, weight_sum = dr.while_loop(
            state=(k, total, optical_depth, weighted_t, weight_sum),
            cond=cond, body=body, label="gpu_preintegration",
        )
    except Exception as exc:
        _diagnose_drjit_exception(
            f"GPU preintegration loop size={size} samples={integration_samples} "
            f"points={point_count} centroid={with_centroid}", exc
        )
        raise
    centroid = (
        dr.select(weight_sum > 1.0e-8, weighted_t / dr.maximum(weight_sum, 1.0e-8), 0.5)
        if with_centroid else None
    )
    try:
        if centroid is None:
            dr.eval(tau)
        else:
            dr.eval(tau, centroid)
    except Exception as exc:
        _diagnose_drjit_exception(f"GPU preintegration evaluation size={size}", exc)
        raise
    return tau, centroid


@dataclass
class MitsubaViewHandle:
    background_color: tuple[float, float, float] = (0.1, 0.1, 0.1)
    world_ambient_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    world_ambient_intensity: float = 1.0
    hdri: str = ""
    directional_lights: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
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
class VolumeResource:
    mode: str = "off"
    texture: Any = None
    key: tuple[Any, ...] | None = None


@dataclass
class VolumeResources:
    scalar: VolumeResource = field(default_factory=lambda: VolumeResource("f32"))
    shadow: VolumeResource = field(default_factory=lambda: VolumeResource("f32", ()))
    environment: VolumeResource = field(default_factory=lambda: VolumeResource("f32", ()))


@dataclass
class MitsubaRepresentationHandle:
    kind: str
    activity_scope: str = ""
    activity_view_id: str = ""
    scene_object: Any | None = None
    bounds: tuple[float, float, float, float, float, float] | None = None
    resources: VolumeResources = field(default_factory=VolumeResources)
    color_mapping: dict[str, Any] | None = None
    opacity_mapping: dict[str, Any] | None = None
    sample_distance: float = 1.0
    preintegration_size: int = 512
    vpt_transport: str = "delta"
    scattering_albedo: float = 0.8
    vpt_max_depth: int = 1
    vpt_anisotropy: float = 0.0
    opacity_reference_distance: float = 1.0
    shade: bool = False
    ambient: float = 0.1
    diffuse: float = 0.9
    global_illumination_reach: float = 0.0
    volumetric_scattering_blending: float = 2.0
    scattering_anisotropy: float = 0.0
    environment_scattering_strength: float = 1.0
    environment_scattering_samples: int = 0
    environment_scattering_step_factor: float = 4.0
    scalar_range: tuple[float, float] = (0.0, 1.0)
    scalar_values: np.ndarray | None = None


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

    def __init__(self, transfer_function_provider=None, stream_sink=None) -> None:
        import drjit as dr
        import mitsuba as mi

        self._transfer_function_provider = transfer_function_provider or (
            lambda _name: None
        )
        self._activity = stream_sink
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
        allowed = {"f32", "f16", "u8"} | ({"off"} if allow_off else set())
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

    def _compile_gpu_preintegration_tau(self, mapping):
        minimum, maximum = map(float, mapping["range"])
        width = maximum - minimum
        if abs(width) < 1.0e-20:
            width = 1.0
        tau, _centroid = _build_gpu_preintegration_tables(
            self.mi, self.dr, mapping, 1.0, with_centroid=False,
        )
        return (
            self.dr.opaque(self.mi.Float, minimum),
            self.dr.opaque(self.mi.Float, 1.0 / width),
            tau,
            PREINTEGRATION_SIZE,
        )

    def _gpu_preintegration_tau(self, scalar0, scalar1, compiled):
        minimum, inv_width, table, size = compiled
        u = self.dr.clip((scalar0 - minimum) * inv_width, 0.0, 1.0) * float(size - 1)
        v = self.dr.clip((scalar1 - minimum) * inv_width, 0.0, 1.0) * float(size - 1)
        i0 = self.mi.UInt32(self.dr.floor(u))
        j0 = self.mi.UInt32(self.dr.floor(v))
        i1 = self.dr.minimum(i0 + 1, self.mi.UInt32(size - 1))
        j1 = self.dr.minimum(j0 + 1, self.mi.UInt32(size - 1))
        fu = u - self.mi.Float(i0)
        fv = v - self.mi.Float(j0)

        def sample(i, j):
            return self.dr.gather(
                self.mi.Float,
                table,
                i * self.mi.UInt32(size) + j,
            )

        a = self.dr.lerp(sample(i0, j0), sample(i1, j0), fu)
        b = self.dr.lerp(sample(i0, j1), sample(i1, j1), fu)
        return self.dr.lerp(a, b, fv)

    def _direct_volume_type(
        self, *, shadow: bool, environment: bool, preintegration: bool
    ):
        key = (bool(shadow), bool(environment), bool(preintegration))
        volume_type = self._direct_volume_types.get(key)
        if volume_type is None:
            volume_type = _make_direct_volume_type(
                self.mi,
                self.dr,
                has_shadow_texture=key[0],
                has_environment_texture=key[1],
                has_preintegration=key[2],
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
            if name == "hdri":
                return handle.hdri
            if name == "world_ambient_intensity":
                return float(handle.world_ambient_intensity)
            if name in DIRECTIONAL_LIGHT_PROPERTY_NAMES:
                return float(handle.directional_lights[DIRECTIONAL_LIGHT_PROPERTY_NAMES.index(name)])
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
            if name == "hdri":
                handle.hdri = str(value or "").strip()
                handle.render_revision += 1
                return
            if name == "world_ambient_intensity":
                handle.world_ambient_intensity = float(value)
                handle.render_revision += 1
                return
            if name in DIRECTIONAL_LIGHT_PROPERTY_NAMES:
                lights = list(handle.directional_lights)
                lights[DIRECTIONAL_LIGHT_PROPERTY_NAMES.index(name)] = max(0.0, float(value))
                handle.directional_lights = tuple(lights)
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
                    and (rep.kind == "vpt" or rep.scene_object is not None or rep.resources.scalar.texture is not None)
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
                "hdri": handle.hdri,
                "world_ambient_color": tuple(handle.world_ambient_color),
                "world_ambient_intensity": float(handle.world_ambient_intensity),
                "directional_lights": tuple(float(v) for v in handle.directional_lights),
                # Keep strong references to exactly the scene objects represented
                # by this revision even if the server thread replaces them later.
                "objects": tuple(
                    (rep.kind, representation_id, rep.scene_object)
                    for (
                        representation_id,
                        current_view_id,
                    ), rep in self._representations.items()
                    if current_view_id == view_id and rep.scene_object is not None
                    and rep.kind not in {"dvr", "vpt"}
                ),
                "volumes": tuple(
                    rep
                    for (_representation_id, current_view_id), rep
                    in self._representations.items()
                    if current_view_id == view_id
                    and rep.kind == "dvr"
                    and rep.resources.scalar.texture is not None
                    and rep.color_mapping is not None
                    and rep.opacity_mapping is not None
                ),
                "vpt_volumes": tuple(
                    rep for (_rid, current_view_id), rep in self._representations.items()
                    if current_view_id == view_id and rep.kind == "vpt"
                    and rep.bounds is not None
                    and rep.resources.scalar.texture is not None
                    and rep.opacity_mapping is not None
                ),
                "vpt_bounds": tuple(
                    rep.bounds for (_rid, current_view_id), rep in self._representations.items()
                    if current_view_id == view_id and rep.kind == "vpt"
                    and rep.bounds is not None
                ),
                "vpt_active": any(
                    current_view_id == view_id and rep.kind == "vpt"
                    for (_representation_id, current_view_id), rep
                    in self._representations.items()
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
        if representation.kind == "vpt":
            # Share scalar upload and transfer-function metadata with DVR, not
            # its integration or lighting caches.
            volume = self._create_volume_handle(
                representation, source, previous=previous,
                activity_view_id=activity_view_id,
            )
            volume.kind = "vpt"
            return volume
        if representation.kind == "dvr":
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
            return MitsubaRepresentationHandle(kind="dvr")

        bounds = tuple(float(v) for v in data.GetBounds())
        color_by = representation.properties.get("color_by")
        if not color_by:
            print("Mitsuba backend: volume rendering requires a selected scalar array")
            return MitsubaRepresentationHandle(kind="dvr", bounds=bounds)

        array_name = str(color_by[0])
        association = str(color_by[1])
        transfer_function = self._transfer_function_provider(array_name)
        if transfer_function is None:
            return MitsubaRepresentationHandle(kind="dvr", bounds=bounds)

        interpolation = str(representation.properties.get("interpolation", "linear"))
        scalar_precision = self._precision(
            representation.properties.get("scalar_volume", "f32"), allow_off=False
        )
        if scalar_precision == "u8":
            raise ValueError("u8 is currently supported only for shadow_volume")
        shadow_precision = self._precision(
            representation.properties.get("shadow_volume", "f32"), allow_off=True
        )
        environment_precision = self._precision(
            representation.properties.get("environment_volume", "f32"), allow_off=True
        )
        if environment_precision == "u8":
            raise ValueError("u8 is currently supported only for shadow_volume")
        scalar_key = (
            data.GetAddressAsString("vtkweb"),
            int(data.GetMTime()),
            array_name,
            association,
            interpolation,
            scalar_precision,
            "original-voxel-scalars-v1",
        )
        scalar_volume = None
        scalar_values = None
        scalar_range = None
        if previous is not None and previous.resources.scalar.key == scalar_key:
            scalar_volume = previous.resources.scalar.texture
            scalar_values = previous.scalar_values
            scalar_range = previous.scalar_range

        if scalar_volume is None:
            scalar_values = _image_scalar_values(data, array_name, association)
            if scalar_values is None:
                print(
                    "Mitsuba backend: selected volume array is unavailable or has "
                    "an incompatible tuple count"
                )
                return MitsubaRepresentationHandle(kind="dvr", bounds=bounds)

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
            kind="dvr",
            activity_scope=f"{activity_view_id}:{representation.id}",
            activity_view_id=str(activity_view_id),
            bounds=bounds,
            resources=VolumeResources(
                scalar=VolumeResource(scalar_precision, scalar_volume, scalar_key),
                shadow=VolumeResource(shadow_precision),
                environment=VolumeResource(environment_precision, ()),
            ),
            color_mapping=transfer_function["color"],
            opacity_mapping=transfer_function["opacity"],
            sample_distance=_volume_sample_distance(data, representation.properties),
            preintegration_size=int(representation.properties.get("preintegration", "512")),
            vpt_transport=str(representation.properties.get("vpt_transport", "delta")),
            scattering_albedo=float(representation.properties.get("scattering_albedo", 0.8)),
            vpt_max_depth=int(representation.properties.get("vpt_max_depth", 1)),
            vpt_anisotropy=float(representation.properties.get("vpt_anisotropy", 0.0)),
            opacity_reference_distance=_volume_opacity_reference_distance(data),
            shade=bool(representation.properties.get("shade", True)),
            ambient=float(representation.properties.get("ambient", 0.1)),
            diffuse=float(representation.properties.get("diffuse", 0.9)),
            global_illumination_reach=float(
                representation.properties.get("global_illumination_reach", 0.0)
            ),
            volumetric_scattering_blending=float(
                representation.properties.get("volumetric_scattering_blending", 2.0)
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
        )

        # Reuse the baked shadow field across representation refreshes whenever
        # its true inputs are unchanged. In particular, camera updates never
        # recreate representation handles, so they cannot invalidate this cache.
        if previous is not None:
            handle.resources.shadow.texture = previous.resources.shadow.texture
            handle.resources.shadow.key = previous.resources.shadow.key
            handle.resources.environment.texture = previous.resources.environment.texture
            handle.resources.environment.key = previous.resources.environment.key

        # Auxiliary textures are built lazily in render_pass() only when the
        # selected representation settings and shading path actually need them.
        return handle

    def _build_directional_shadow_field(
        self,
        volume: MitsubaRepresentationHandle,
        directional_lights: tuple[float, ...],
    ) -> None:
        """Bake full-reach, intensity-weighted isotropic illumination in one texture."""
        scalars = volume.scalar_values
        mapping = volume.opacity_mapping
        bounds = volume.bounds
        texture = volume.resources.scalar.texture
        if scalars is None or mapping is None or bounds is None or texture is None:
            return
        if scalars.ndim != 3 or min(scalars.shape) < 2:
            return

        precision = volume.resources.shadow.mode
        if precision == "off":
            volume.resources.shadow.texture = ()
            volume.resources.shadow.key = None
            return

        active_indices = tuple(
            index for index, intensity in enumerate(directional_lights)
            if float(intensity) > 0.0
        )
        if not active_indices:
            volume.resources.shadow.texture = ()
            volume.resources.shadow.key = None
            return

        opacity_signature = (
            tuple(float(v) for v in mapping["range"]),
            tuple(
                tuple(float(component) for component in point)
                for point in mapping["control_points"]
            ),
        )
        shadow_key = (
            volume.resources.scalar.key,
            opacity_signature,
            float(volume.opacity_reference_distance),
            tuple(float(v) for v in directional_lights),
            tuple(float(v) for v in bounds),
            precision,
            PREINTEGRATION_SIZE,
            "gpu-axis-torus-combined-illumination-u8-v1",
        )
        if volume.resources.shadow.texture and volume.resources.shadow.key == shadow_key:
            return

        nz, ny, nx = (int(v) for v in scalars.shape)
        dims_xyz = (nx, ny, nz)
        extents_xyz = (
            float(bounds[1]) - float(bounds[0]),
            float(bounds[3]) - float(bounds[2]),
            float(bounds[5]) - float(bounds[4]),
        )
        if precision == "u8":
            backend_module = importlib.import_module(self.mi.Float.__module__)
            uint8_type = getattr(backend_module, "UInt8", None)
            if uint8_type is None:
                raise RuntimeError("This Dr.Jit backend does not expose a GPU UInt8 array")
        else:
            _tensor_type, storage_type, _texture_type = self._texture_storage_types(precision)
        preintegration = self._compile_gpu_preintegration_tau(mapping)
        opacity_reference_distance = max(float(volume.opacity_reference_distance), 1.0e-12)

        start = time.perf_counter()
        activity_key = f"shadow:{volume.activity_scope}"
        activity_started = self._activity_start(
            activity_key,
            "Baking shadow volume",
            view_id=volume.activity_view_id,
            determinate=False,
            details={
                "shape": (nz, ny, nx),
                "preintegration": f"{PREINTEGRATION_SIZE}x{PREINTEGRATION_SIZE}",
                "directions": [DIRECTIONAL_LIGHT_PROPERTY_NAMES[i] for i in active_indices],
                "dtype": precision,
                "device": self.mi.variant(),
            },
        )
        print(
            f"Mitsuba bake directional shadow (device): start shape={(nz, ny, nx)} "
            f"directions={[DIRECTIONAL_LIGHT_PROPERTY_NAMES[i] for i in active_indices]} "
            f"preintegration={PREINTEGRATION_SIZE}x{PREINTEGRATION_SIZE} dtype={precision}",
            flush=True,
        )

        illumination = self.dr.zeros(self.mi.Float, nz * ny * nx)
        with self._render_lock:
            for light_index in active_indices:
                direction = DIRECTIONAL_LIGHT_DIRECTIONS[light_index]
                if direction[0]:
                    axis, sign, count = 0, 1 if direction[0] > 0 else -1, nx
                    plane = nz * ny
                elif direction[1]:
                    axis, sign, count = 1, 1 if direction[1] > 0 else -1, ny
                    plane = nz * nx
                else:
                    axis, sign, count = 2, 1 if direction[2] > 0 else -1, nz
                    plane = ny * nx

                axis_step = extents_xyz[axis] / float(max(count - 1, 1))
                running_tau = self.dr.zeros(self.mi.Float, plane)
                local = self.dr.arange(self.mi.UInt32, plane)

                if axis == 0:  # X, local indexes ZY
                    z_fixed = local // self.mi.UInt32(ny)
                    y_fixed = local % self.mi.UInt32(ny)
                elif axis == 1:  # Y, local indexes ZX
                    z_fixed = local // self.mi.UInt32(nx)
                    x_fixed = local % self.mi.UInt32(nx)
                else:  # Z, local indexes YX
                    y_fixed = local // self.mi.UInt32(nx)
                    x_fixed = local % self.mi.UInt32(nx)

                # Boundary voxel centers receive unattenuated light. The live
                # marcher also begins at the sample point (no exterior half-cell).
                boundary = count - 1 if sign > 0 else 0
                if axis == 0:
                    boundary_indices = z_fixed * self.mi.UInt32(ny * nx) + y_fixed * self.mi.UInt32(nx) + self.mi.UInt32(boundary)
                elif axis == 1:
                    boundary_indices = z_fixed * self.mi.UInt32(ny * nx) + self.mi.UInt32(boundary * nx) + x_fixed
                else:
                    boundary_indices = local + self.mi.UInt32(boundary * ny * nx)
                previous = self.dr.gather(self.mi.Float, illumination, boundary_indices)
                self.dr.scatter(illumination, previous + float(directional_lights[light_index]), boundary_indices)
                # One symbolic GPU loop per light: no per-plane dr.eval() or
                # progress callbacks. Rows remain parallel; planes are sequential.
                self.dr.eval(illumination)
                initial_k = count - 2 if sign > 0 else 1
                delta_k = -1 if sign > 0 else 1
                k = self.mi.Int32(initial_k)

                def loop_cond(k, running_tau, illumination):
                    return k >= 0 if sign > 0 else k < count

                def loop_body(k, running_tau, illumination):
                    segment_start = k if sign > 0 else k - 1
                    segment_end = segment_start + 1
                    if axis == 0:
                        p0 = self._gpu_texture_position_from_indices(
                            self.mi, self.mi.Float(segment_start), y_fixed, z_fixed, nx, ny, nz
                        )
                        p1 = self._gpu_texture_position_from_indices(
                            self.mi, self.mi.Float(segment_end), y_fixed, z_fixed, nx, ny, nz
                        )
                    elif axis == 1:
                        p0 = self._gpu_texture_position_from_indices(
                            self.mi, x_fixed, self.mi.Float(segment_start), z_fixed, nx, ny, nz
                        )
                        p1 = self._gpu_texture_position_from_indices(
                            self.mi, x_fixed, self.mi.Float(segment_end), z_fixed, nx, ny, nz
                        )
                    else:
                        p0 = self._gpu_texture_position_from_indices(
                            self.mi, x_fixed, y_fixed, self.mi.Float(segment_start), nx, ny, nz
                        )
                        p1 = self._gpu_texture_position_from_indices(
                            self.mi, x_fixed, y_fixed, self.mi.Float(segment_end), nx, ny, nz
                        )
                    # Both endpoints use the same voxelized scalar texture as DVR.
                    scalar0 = texture.eval(p0)[0]
                    scalar1 = texture.eval(p1)[0]
                    tau_reference = self._gpu_preintegration_tau(
                        scalar0, scalar1, preintegration
                    )
                    segment_tau = tau_reference * (axis_step / opacity_reference_distance)
                    running_tau = running_tau + segment_tau
                    if axis == 0:
                        indices = (
                            z_fixed * self.mi.UInt32(ny * nx)
                            + y_fixed * self.mi.UInt32(nx)
                            + self.mi.UInt32(k)
                        )
                    elif axis == 1:
                        indices = (
                            z_fixed * self.mi.UInt32(ny * nx)
                            + self.mi.UInt32(k * nx)
                            + x_fixed
                        )
                    else:
                        indices = local + self.mi.UInt32(k * ny * nx)
                    previous = self.dr.gather(self.mi.Float, illumination, indices)
                    contribution = float(directional_lights[light_index]) * self.dr.exp(-running_tau)
                    self.dr.scatter(illumination, previous + contribution, indices)
                    return k + delta_k, running_tau, illumination

                try:
                    k, running_tau, illumination = self.dr.while_loop(
                        state=(k, running_tau, illumination),
                        cond=loop_cond,
                        body=loop_body,
                        mode="symbolic",
                        label="vtkweb directional illumination prefix",
                    )
                    self.dr.eval(illumination)
                except Exception as exc:
                    _diagnose_drjit_exception(
                        f"directional illumination loop light_index={light_index} axis={axis}", exc
                    )
                    raise

            # Accumulate in FP32; quantize only the finished texture.
            if precision == "u8":
                # A single byte stores the mean light contribution, not the sum.
                # At render time, _ShadowUNorm8 restores the sum by multiplying
                # by the number of active lights (e.g. (0.5+0.6)/2 -> 0.55 -> 1.1).
                normalized = self.dr.clip(illumination / float(len(active_indices)), 0.0, 1.0)
                quantized = uint8_type(self.dr.floor(normalized * 255.0 + 0.5))
                volume.resources.shadow.texture = _ShadowUNorm8(
                    quantized, (nz, ny, nx, 1), self.mi, self.dr, len(active_indices)
                )
                if DEBUG_SHADOW_STORAGE:
                    self.dr.eval(quantized)
                    nvox = nx * ny * nz
                    itemsize = _debug_shadow_array("quantized backing array", quantized, nvox, nvox)
                    print(
                        f"[shadow-storage] storage wrapper={type(volume.resources.shadow.texture).__name__}; "
                        f"backing array is {'8-bit (confirmed)' if itemsize == 1 else 'NOT confirmed 8-bit'}; "
                        f"trilinear interpolation promotes fetched bytes to Float32 at runtime; "
                        f"lights={len(active_indices)}; reconstructed_scale={len(active_indices)}/255",
                        flush=True,
                    )
            else:
                stored = storage_type(illumination)
                volume.resources.shadow.texture = self._make_texture3d_device(
                    stored, (nz, ny, nx, 1), precision,
                    interpolation="linear",
                )
                if DEBUG_SHADOW_STORAGE:
                    nvox = nx * ny * nz
                    _debug_shadow_array("floating texture source", stored, nvox, nvox * (2 if precision == "f16" else 4))
                    print(f"[shadow-storage] texture wrapper={type(volume.resources.shadow.texture).__name__}; "
                          "GPU texture allocation may have backend-specific padding/representation", flush=True)
            self.dr.eval(illumination)

        volume.resources.shadow.key = shadow_key
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
        """Bake six-direction environment visibility on the active Dr.Jit device.

        The per-direction preintegrated optical-depth fields and final
        FP16/FP32 textures remain device-resident. CPU
        memory use is limited to tiny direction/LUT metadata and the original
        VTK scalar buffer.
        """
        if (volume.environment_scattering_samples != 6 or not volume.shade
                or abs(volume.scattering_anisotropy) >= 0.01):
            volume.resources.environment.texture = ()
            volume.resources.environment.key = None
            return
        mapping = volume.opacity_mapping
        bounds = volume.bounds
        scalars = volume.scalar_values
        scalar_texture = volume.resources.scalar.texture
        if mapping is None or bounds is None or scalars is None or scalar_texture is None:
            return

        opacity_signature = (
            tuple(float(v) for v in mapping["range"]),
            tuple(tuple(float(c) for c in point) for point in mapping["control_points"]),
        )
        precision = volume.resources.environment.mode
        if precision == "off":
            volume.resources.environment.texture = ()
            volume.resources.environment.key = None
            return
        direction_count = max(0, int(volume.environment_scattering_samples))
        resolution_factor = 1  # Match scalar grid for this six-axis experiment
        key = (
            volume.resources.scalar.key,
            opacity_signature,
            float(volume.opacity_reference_distance),
            float(volume.global_illumination_reach),
            direction_count,
            resolution_factor,
            tuple(float(v) for v in bounds),
            2,
            precision,
            "gpu-environment-six-direction-hybrid-preintegrated-v2",
        )
        if volume.resources.environment.key == key and volume.resources.environment.texture:
            return

        nz, ny, nx = (int(v) for v in scalars.shape)
        rz = int(math.ceil(nz / float(resolution_factor)))
        ry = int(math.ceil(ny / float(resolution_factor)))
        rx = int(math.ceil(nx / float(resolution_factor)))
        reduced_shape = (rz, ry, rx)
        reduced_count = int(rz * ry * rx)
        preintegration = self._compile_gpu_preintegration_tau(mapping)
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
            f"dtype={precision} axis_prefix={direction_count == 6}",
            flush=True,
        )

        xmin, xmax, ymin, ymax, zmin, zmax = map(float, bounds)
        sx = (xmax - xmin) / max(rx - 1, 1)
        sy = (ymax - ymin) / max(ry - 1, 1)
        sz = (zmax - zmin) / max(rz - 1, 1)
        axis_spacing = (sz, sy, sx)  # z, y, x
        reach = self._shadow_extent_for_volume(volume)
        weight = 1.0 / float(max(1, direction_count))

        with self._render_lock:
            # Use the original scalar texture and the same GPU preintegration LUT
            # as the directional-shadow prefix baker. No intermediate extinction
            # grid: a sharp transfer function must be integrated along each segment.
            reduced_index = self.dr.arange(self.mi.UInt32, reduced_count)

            directional_textures = []

            # Base coordinates of every reduced-grid voxel, reused for the
            # reach-offset lookup after each directional dynamic-programming pass.
            x_all = reduced_index % self.mi.UInt32(rx)
            yz_all = reduced_index // self.mi.UInt32(rx)
            y_all = yz_all % self.mi.UInt32(ry)
            z_all = yz_all // self.mi.UInt32(ry)

            if direction_count == 6:
                # Axis-aligned prefix integration: parallel rows, one symbolic
                # GPU loop over planes, no Python per-plane launch/sync.
                for direction_index, direction in enumerate(directions, start=1):
                    d_axis = (direction[2], direction[1], direction[0])
                    dominant = next(i for i, component in enumerate(d_axis) if component)
                    sign = int(d_axis[dominant])
                    sizes = reduced_shape
                    others = [i for i in range(3) if i != dominant]
                    o0, o1 = others
                    n0, n1 = sizes[o0], sizes[o1]
                    uv = self.dr.arange(self.mi.UInt32, n0 * n1)
                    a = uv // self.mi.UInt32(n1)
                    b = uv % self.mi.UInt32(n1)
                    tau = self.dr.zeros(self.mi.Float, reduced_count)
                    running = self.dr.zeros(self.mi.Float, n0 * n1)
                    boundary = sizes[dominant] - 1 if sign > 0 else 0
                    first = boundary - sign
                    k = self.mi.Int32(first)
                    step_length = float(axis_spacing[dominant])

                    def plane_indices(fixed):
                        coords = [None, None, None]
                        coords[dominant] = self.mi.UInt32(fixed)
                        coords[o0] = a
                        coords[o1] = b
                        return (coords[0] * self.mi.UInt32(ry * rx)
                                + coords[1] * self.mi.UInt32(rx) + coords[2])

                    def cond(k, running, tau):
                        return k >= 0 if sign > 0 else k < sizes[dominant]

                    def body(k, running, tau):
                        # Identical optical-depth calculation to directional
                        # shadow prefix: sample the two scalar endpoints and
                        # integrate opacity through the shared preintegration LUT.
                        here = plane_indices(k)
                        coords0 = [None, None, None]
                        coords1 = [None, None, None]
                        coords0[dominant] = self.mi.Float(k)
                        coords1[dominant] = self.mi.Float(k + sign)
                        coords0[o0] = coords1[o0] = a
                        coords0[o1] = coords1[o1] = b
                        p0 = self._gpu_texture_position_from_indices(
                            self.mi, coords0[2], coords0[1], coords0[0], nx, ny, nz
                        )
                        p1 = self._gpu_texture_position_from_indices(
                            self.mi, coords1[2], coords1[1], coords1[0], nx, ny, nz
                        )
                        scalar0 = scalar_texture.eval(p0)[0]
                        scalar1 = scalar_texture.eval(p1)[0]
                        tau_reference = self._gpu_preintegration_tau(
                            scalar0, scalar1, preintegration
                        )
                        running = running + tau_reference * (step_length / opacity_reference_distance)
                        self.dr.scatter(tau, running, here)
                        return k - sign, running, tau

                    try:
                        k, running, tau = self.dr.while_loop(
                            state=(k, running, tau), cond=cond, body=body,
                            mode="symbolic", label="vtkweb environment axis prefix",
                        )
                        self.dr.eval(tau)
                    except Exception as exc:
                        _diagnose_drjit_exception(
                            f"environment axis prefix direction={direction}", exc
                        )
                        raise
                    offsets = (
                        float(direction[2]) * reach / max(sz, 1.0e-12),
                        float(direction[1]) * reach / max(sy, 1.0e-12),
                        float(direction[0]) * reach / max(sx, 1.0e-12),
                    )
                    tau_end = self._gpu_trilinear_flat(
                        tau, reduced_shape,
                        self.mi.Float(z_all) + offsets[0],
                        self.mi.Float(y_all) + offsets[1],
                        self.mi.Float(x_all) + offsets[2],
                    )
                    visibility = self.dr.exp(-self.dr.maximum(tau - tau_end, 0.0))
                    # Store the unweighted directional transmittance. Live and
                    # hybrid paths apply the same 1/6 and phase weighting.
                    _tensor_type, storage_type, _texture_type = self._texture_storage_types(precision)
                    packed = storage_type(visibility)
                    self.dr.eval(packed)
                    directional_textures.append(self._make_texture3d_device(
                        packed, (rz, ry, rx, 1), precision, interpolation="linear"
                    ))
                    print(
                        f"Mitsuba bake environment lighting (device): "
                        f"direction {direction_index}/{len(directions)} finished",
                        flush=True,
                    )
            else:
                raise ValueError("Scalar environment cache requires exactly six axis directions")

            volume.resources.environment.texture = tuple(directional_textures)

        volume.resources.environment.key = key
        elapsed = time.perf_counter() - start
        self._activity_done(activity_key, activity_started)
        print(
            f"Mitsuba bake environment lighting (device): end {elapsed:.3f}s "
            f"fields=6,channels=1,resolution={(rz, ry, rx)}",
            flush=True,
        )

    def _ensure_configured_volume_resources(
        self,
        volumes: tuple[MitsubaRepresentationHandle, ...],
        directional_lights: tuple[float, ...],
    ) -> None:
        """Build only enabled auxiliary textures that the current shading path uses."""
        for volume in volumes:
            # `off` is an explicit memory-control request, so release old
            # resources even when the current shading path would not use them.
            if volume.resources.shadow.mode == "off":
                volume.resources.shadow.texture = ()
                volume.resources.shadow.key = None
            if volume.resources.environment.mode == "off":
                volume.resources.environment.texture = ()
                volume.resources.environment.key = None

            if not volume.shade:
                continue

            if volume.volumetric_scattering_blending <= 0.0:
                continue

            if volume.resources.shadow.mode != "off" and abs(volume.scattering_anisotropy) < 0.01:
                self._build_directional_shadow_field(volume, directional_lights)
            else:
                volume.resources.shadow.texture = ()
                volume.resources.shadow.key = None

            if (
                volume.environment_scattering_samples > 0
                and volume.resources.environment.mode != "off"
                and volume.environment_scattering_samples == 6
                and abs(volume.scattering_anisotropy) < 0.01
            ):
                self._build_environment_lighting_field(volume)
            else:
                volume.resources.environment.texture = ()
                volume.resources.environment.key = None

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

        # VPT hello world: an empty scene with a visible environment.
        # Mitsuba's path integrator evaluates the emitter for camera misses.
        if snapshot["vpt_active"]:
            scene_dict["integrator"]["hide_emitters"] = False

        # Mitsuba envmap uses a lat-long HDR texture and builds its own
        # importance distribution. Loading the scene uploads the texture to
        # the active backend; the structural scene key controls reloads.
        if snapshot["hdri"]:
            from pathlib import Path
            hdri_file = Path(snapshot["hdri"]).expanduser().resolve()
            if not hdri_file.is_file():
                raise FileNotFoundError(f"HDRI file does not exist: {hdri_file}")
            scene_dict["environment"] = {
                "type": "envmap", "filename": str(hdri_file),
            }

        # Directional-light values are intentionally unitless vtkweb intensities.
        # For Mitsuba surface rendering they are used as relative irradiance.
        if snapshot["objects"]:
            for light_index, intensity in enumerate(snapshot["directional_lights"]):
                if float(intensity) <= 0.0:
                    continue
                toward_light = DIRECTIONAL_LIGHT_DIRECTIONS[light_index]
                scene_dict[f"directional_light_{light_index}"] = {
                    "type": "directional",
                    # Mitsuba's `direction` is the direction in which light
                    # propagates; vtkweb stores the direction from sample to light.
                    "direction": [-float(v) for v in toward_light],
                    "irradiance": {
                        "type": "rgb",
                        "value": [float(intensity)] * 3,
                    },
                }

        for shape_index, (_kind, representation_id, scene_object) in enumerate(
            snapshot["objects"]
        ):
            scene_dict[f"shape_{shape_index}_{representation_id}"] = scene_object

        # VPT milestone 2: a diagnostic boundary shell. These six colored
        # faces are temporary scene geometry, NOT participating medium or
        # physical scattering. Miss rays still see the HDRI. Each face uses
        # the actual vtkImageData world-space bounds (including spacing).
        # VPT bounds are evaluated in a dedicated Dr.Jit integrator below.
        # Do not add diagnostic rectangles to the scene: they would occlude
        # surfaces and would not exercise the actual ray/AABB intersection.

        # Keep camera-only changes out of the expensive scene/integrator cache
        # keys. Film/crop/FOV and geometry remain structural scene inputs, while
        # pose/focus/aperture are exposed Mitsuba sensor parameters and can be
        # updated in place on the cached scene.
        scene_key = (
            tuple((kind, representation_id, id(scene_object)) for kind, representation_id, scene_object in snapshot["objects"]),
            bool(snapshot["vpt_active"]),
            snapshot["vpt_bounds"],
            full_size,
            region,
            float(camera["fov"]),
            bool(dof_enabled),
            int(spp),
            tuple(float(v) for v in snapshot["world_ambient_color"]),
            float(snapshot["world_ambient_intensity"]),
            (str(__import__("pathlib").Path(snapshot["hdri"]).expanduser().resolve()),
             __import__("os").stat(__import__("pathlib").Path(snapshot["hdri"]).expanduser()).st_mtime_ns)
            if snapshot["hdri"] else None,
            tuple(float(v) for v in snapshot["directional_lights"]) if snapshot["objects"] else (),
        )

        # VPT currently contributes only the environment. Keep other scene
        # objects (including outlines) visible rather than deleting them.
        # VPT medium geometry and stochastic transport remain future work.

        self._ensure_configured_volume_resources(snapshot["volumes"], snapshot["directional_lights"])

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

            if snapshot["vpt_active"]:
                seed = max(1, int(render_seed) + 1)
                vpt_key = tuple((id(v.resources.scalar.texture), v.bounds,
                    repr(v.opacity_mapping), v.sample_distance,
                    v.opacity_reference_distance, v.vpt_transport, v.scattering_albedo, v.vpt_max_depth, v.vpt_anisotropy,
                    repr(v.color_mapping)) for v in snapshot["vpt_volumes"])
                if getattr(handle, "vpt_diagnostic_key", None) != vpt_key:
                    use_delta = all(v.vpt_transport == "delta" for v in snapshot["vpt_volumes"])
                    if VPT_DEBUG:
                        print(f"[VPT DEBUG] build integrator: transport={'delta' if use_delta else 'deterministic'} volumes={len(snapshot['vpt_volumes'])} spp={spp} seed={seed}", flush=True)
                        for vi, v in enumerate(snapshot["vpt_volumes"]):
                            print(f"[VPT DEBUG] volume[{vi}]: bounds={v.bounds} reference_distance={v.opacity_reference_distance} texture={type(v.resources.scalar.texture).__name__}", flush=True)
                    handle.vpt_diagnostic_integrator = (
                        _make_vpt_delta_integrator(self.mi, self.dr, snapshot["vpt_volumes"])
                        if use_delta
                        else _make_vpt_extinction_integrator(self.mi, self.dr, snapshot["vpt_volumes"])
                    )
                    handle.vpt_diagnostic_key = vpt_key
                try:
                    image = self.mi.render(
                        scene, sensor=scene.sensors()[0], spp=spp, seed=seed,
                        integrator=handle.vpt_diagnostic_integrator,
                    )
                    return np.asarray(image, dtype=np.float32)[..., :3].copy()
                except Exception as exc:
                    import traceback
                    if VPT_DEBUG:
                        print(f"[VPT DEBUG] mi.render failed: {type(exc).__name__}: {exc}", flush=True)
                        traceback.print_exception(type(exc), exc, exc.__traceback__)
                    raise

            # Auxiliary texture presence is a Python-time specialization: each
            # volume gets a DirectVolume type matching exactly the resources it
            # owns. Precision changes recreate textures but do not add runtime
            # branches inside the ray-march kernel.
            structural_volume_key = tuple(
                (
                    id(volume.resources.scalar.texture),
                    tuple(float(v) for v in volume.bounds),
                    (volume.resources.shadow.mode, type(volume.resources.shadow.texture).__name__)
                    if volume.resources.shadow.texture else None,
                    tuple(type(field).__name__ for field in volume.resources.environment.texture),
                    bool(volume.shade),
                    int(volume.environment_scattering_samples),
                    bool(float(volume.volumetric_scattering_blending) > 0.0),
                    bool(float(volume.volumetric_scattering_blending) < 1.0),
                    bool(abs(float(volume.scattering_anisotropy)) < 0.01),
                    bool(volume.preintegration_size),
                )
                for volume in snapshot["volumes"]
            )
            active_light_indices = tuple(
                index for index, intensity in enumerate(snapshot["directional_lights"])
                if float(intensity) > 0.0
            )
            integrator_key = (structural_volume_key, bool(snapshot["objects"]), active_light_indices)
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
                        "shadow": [v.resources.shadow.mode for v in snapshot["volumes"]],
                        "environment": [v.resources.environment.mode for v in snapshot["volumes"]],
                    },
                    determinate=False,
                )
                print(
                    f"[Mitsuba] integrator rebuild: view={view_id} "
                    f"volumes={len(snapshot['volumes'])} surfaces={bool(snapshot['objects'])} "
                    f"resources={[ (v.resources.shadow.mode, v.resources.environment.mode) for v in snapshot['volumes'] ]}",
                    flush=True,
                )
                direct_volumes = tuple(
                    self._direct_volume_type(
                        shadow=bool(volume.resources.shadow.texture),
                        environment=bool(volume.resources.environment.texture),
                        preintegration=bool(volume.preintegration_size),
                    )(
                        volume.resources.scalar.texture,
                        volume.bounds,
                        volume.color_mapping,
                        volume.opacity_mapping,
                        volume.sample_distance,
                        volume.preintegration_size,
                        volume.opacity_reference_distance,
                        volume.shade,
                        volume.ambient,
                        volume.diffuse,
                        volume.global_illumination_reach,
                        volume.volumetric_scattering_blending,
                        volume.scattering_anisotropy,
                        volume.scalar_range,
                        ambient_light,
                        snapshot["directional_lights"],
                        volume.resources.shadow.texture,
                        volume.environment_scattering_strength,
                        volume.environment_scattering_samples,
                        volume.environment_scattering_step_factor,
                        volume.resources.environment.texture,
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
                    directional_lights=snapshot["directional_lights"],
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
                _diagnose_drjit_exception("render pass", exc)
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
            # marches while sharing the same scalar texture and TF semantics.
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
    if count == 6:
        return ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
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
    has_shadow_texture: bool,
    has_environment_texture: bool,
    has_preintegration: bool,
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
            preintegration_size: int,
            opacity_reference_distance: float,
            shade: bool,
            ambient: float,
            diffuse: float,
            global_illumination_reach: float,
            volumetric_scattering_blending: float,
            scattering_anisotropy: float,
            scalar_range: tuple[float, float],
            ambient_light: tuple[float, float, float],
            directional_lights: tuple[float, ...],
            shadow_volume,
            environment_scattering_strength: float,
            environment_scattering_samples: int,
            environment_scattering_step_factor: float,
            environment_lighting_volumes,
        ) -> None:
            self.scalar_volume = scalar_volume
            self.bounds = tuple(map(float, bounds))
            self.sample_distance = dr.opaque(mi.Float, max(float(sample_distance), 1.0e-12))
            self.preintegration_size = int(preintegration_size)
            self.opacity_reference_distance = max(float(opacity_reference_distance), 1.0e-12)
            self.shade = bool(shade)
            self.ambient = dr.opaque(mi.Float, max(0.0, float(ambient)))
            self.diffuse = dr.opaque(mi.Float, max(0.0, float(diffuse)))
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
            self.light_indices = tuple(
                index for index, intensity in enumerate(directional_lights)
                if float(intensity) > 0.0
            )
            self.light_directions = tuple(
                mi.Vector3f(*DIRECTIONAL_LIGHT_DIRECTIONS[index])
                for index in self.light_indices
            )
            self.light_intensities = tuple(
                dr.opaque(mi.Float, float(directional_lights[index]))
                for index in self.light_indices
            )
            self.total_light_intensity = dr.opaque(
                mi.Float, sum(float(directional_lights[index]) for index in self.light_indices)
            )
            self.shadow_volumes = shadow_volume
            self.hybrid_shadow_voxels = dr.opaque(mi.Float, max(0.0, DEBUG_HYBRID_SHADOW_VOXELS))
            self.hybrid_env_voxels = dr.opaque(mi.Float, max(0.0, DEBUG_HYBRID_ENV_VOXELS))
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
            self._preintegration_ratio = max(float(sample_distance), 1.0e-12) / self.opacity_reference_distance
            self.preintegration = (
                self._compile_preintegration(
                    opacity_mapping, self._preintegration_ratio, self.preintegration_size
                )
                if has_preintegration
                else None
            )


        @staticmethod
        def _mapping_signature(mapping):
            return (
                tuple(float(v) for v in mapping["range"]),
                tuple(
                    tuple(float(component) for component in point)
                    for point in mapping["control_points"]
                ),
            )

        def update_runtime(
            self, volume, *, ambient_light, directional_lights
        ) -> None:
            """Refresh non-structural inputs without replacing the integrator.

            Structural inputs are deliberately handled by the compact cache key
            in ``render_pass``. Everything here is cheap state replacement.
            """
            color_signature = self._mapping_signature(volume.color_mapping)
            if color_signature != self._color_mapping_signature:
                self.color = self._compile_mapping(volume.color_mapping)
                self._color_mapping_signature = color_signature

            opacity_signature = self._mapping_signature(volume.opacity_mapping)
            opacity_changed = opacity_signature != self._opacity_mapping_signature
            if opacity_changed:
                self.opacity = self._compile_mapping(volume.opacity_mapping)
                self._opacity_mapping_signature = opacity_signature

            self.sample_distance = dr.opaque(mi.Float, max(float(volume.sample_distance), 1.0e-12))
            self.preintegration_size = int(volume.preintegration_size)
            self.opacity_reference_distance = max(
                float(volume.opacity_reference_distance), 1.0e-12
            )
            preintegration_ratio = max(float(volume.sample_distance), 1.0e-12) / self.opacity_reference_distance
            ratio_changed = abs(preintegration_ratio - self._preintegration_ratio) > 1.0e-6
            if has_preintegration and (
                opacity_changed
                or ratio_changed
                or self.preintegration is None
                or self.preintegration[1] != self.preintegration_size
            ):
                self.preintegration = self._compile_preintegration(
                    volume.opacity_mapping, preintegration_ratio, self.preintegration_size
                )
            self._preintegration_ratio = preintegration_ratio
            self.ambient = dr.opaque(mi.Float, max(0.0, float(volume.ambient)))
            self.diffuse = dr.opaque(mi.Float, max(0.0, float(volume.diffuse)))
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
            self.light_intensities = tuple(
                dr.opaque(mi.Float, float(directional_lights[index]))
                for index in self.light_indices
            )
            self.total_light_intensity = dr.opaque(
                mi.Float, sum(float(directional_lights[index]) for index in self.light_indices)
            )
            self.environment_scattering_strength = dr.opaque(
                mi.Float, max(0.0, float(volume.environment_scattering_strength))
            )
            self.environment_scattering_step_factor = max(
                1.0, float(volume.environment_scattering_step_factor)
            )
            self.shadow_volumes = volume.resources.shadow.texture
            self.hybrid_shadow_voxels = dr.opaque(mi.Float, max(0.0, DEBUG_HYBRID_SHADOW_VOXELS))
            self.hybrid_env_voxels = dr.opaque(mi.Float, max(0.0, DEBUG_HYBRID_ENV_VOXELS))
            self.environment_lighting_volumes = tuple(
                volume.resources.environment.texture or ()
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
        def _compile_preintegration(
            mapping,
            step_ratio,
            size=PREINTEGRATION_SIZE,
            integration_samples=PREINTEGRATION_SAMPLES,
        ):
            """Compile interval extinction and contribution-centroid tables."""
            tau, centroid = _build_gpu_preintegration_tables(
                mi, dr, mapping, step_ratio,
                size=size,
                integration_samples=integration_samples,
            )
            assert centroid is not None
            # Interleave the two GPU arrays without a CPU readback. The texture
            # performs both bilinear interpolations in a single lookup.
            texels = dr.ravel(mi.Vector2f(tau, centroid))
            tensor = mi.TensorXf(texels, shape=(int(size), int(size), 2))
            texture = mi.Texture2f(tensor, use_accel=False)
            return (texture, int(size))

        @staticmethod
        def _preintegration_sample(x0, x1, compiled, active=True):
            texture, _size = compiled
            # The first tensor dimension is the starting scalar and the
            # second is the ending scalar. Texture coordinates are (end,start).
            # Half-texel mapping reproduces the original endpoint-grid LUT.
            u = (dr.clip(x1, 0.0, 1.0) * float(_size - 1) + 0.5) / float(_size)
            v = (dr.clip(x0, 0.0, 1.0) * float(_size - 1) + 0.5) / float(_size)
            values = texture.eval(mi.Point2f(u, v), active)
            return values[0], dr.clip(values[1], 0.0, 1.0)

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
            """Sample the uploaded voxelized scalar field (from the input dataset)."""
            return self.scalar_volume.eval(
                self._texture_position(position, self.scalar_volume), active
            )[0]

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

        def _segment_preintegration(
            self, start_position, direction, step_distance, active=True,
            scalar_start=None,
        ):
            """Pre-integrate extinction, optionally reusing the start scalar."""
            end_position = start_position + direction * step_distance
            scalar0 = (self._sample_scalar(start_position, active)
                       if scalar_start is None else scalar_start)
            scalar1 = self._sample_scalar(end_position, active)
            x0 = self._normalized_scalar(scalar0, self.opacity)
            x1 = self._normalized_scalar(scalar1, self.opacity)
            tau_reference, centroid = self._preintegration_sample(
                x0, x1, self.preintegration, active
            )
            tau = tau_reference * (step_distance / self.opacity_reference_distance)
            alpha = dr.clip(1.0 - dr.exp(-tau), 0.0, 1.0)
            scalar = dr.lerp(scalar0, scalar1, centroid)
            position = start_position + direction * (centroid * step_distance)
            return alpha, scalar, position, scalar1

        def _segment_point_sample(
            self, start_position, direction, step_distance, active=True
        ):
            """Classify one segment at its midpoint without pre-integration."""
            position = start_position + direction * (0.5 * step_distance)
            scalar = self._sample_scalar(position, active)
            x = self._normalized_scalar(scalar, self.opacity)
            alpha_reference = self._opacity(x, self.opacity, active)
            tau_reference = -dr.log(dr.maximum(1.0 - alpha_reference, 1.0e-6))
            tau = tau_reference * (step_distance / self.opacity_reference_distance)
            alpha = dr.clip(1.0 - dr.exp(-tau), 0.0, 1.0)
            return alpha, scalar, position, scalar

        def _segment_eval(
            self, start_position, direction, step_distance, active=True,
            scalar_start=None,
        ):
            if has_preintegration:
                return self._segment_preintegration(
                    start_position, direction, step_distance, active,
                    scalar_start=scalar_start,
                )
            return self._segment_point_sample(
                start_position, direction, step_distance, active
            )

        def _segment_transmission(
            self, start_position, direction, step_distance, active=True
        ):
            alpha, _scalar, _position, _end_scalar = self._segment_eval(
                start_position, direction, step_distance, active
            )
            return 1.0 - alpha

        def _live_transmittance(
            self,
            position,
            direction,
            *,
            step_distance,
            march_phase,
            active=True,
            distance_limit=None,
            return_endpoint=False,
        ):
            """March the current opacity TF directly through the scalar volume."""
            direction = dr.normalize(direction)
            step = dr.maximum(mi.Float(step_distance), mi.Float(1.0e-12))
            max_distance = dr.minimum(
                mi.Float(self._shadow_extent()),
                self._distance_to_volume_exit(position, direction),
            )
            if distance_limit is not None:
                max_distance = dr.minimum(max_distance, dr.maximum(mi.Float(distance_limit), 0.0))
            t = mi.Float(0.0)
            transmission = mi.Float(1.0)
            first_segment = mi.Bool(True)
            march_active = active & (max_distance > 0.0)
            scalar_start = self._sample_scalar(position, march_active) if has_preintegration else mi.Float(0.0)

            def loop_cond(t, transmission, first_segment, march_active, scalar_start):
                del t, transmission, first_segment, scalar_start
                return march_active

            def loop_body(t, transmission, first_segment, march_active, scalar_start):
                remaining = dr.maximum(max_distance - t, 0.0)
                regular_step = dr.minimum(step, remaining)
                phase_step = dr.minimum(step * march_phase, remaining)
                use_phase_step = first_segment & (phase_step > step * 1.0e-6)
                current_step = dr.select(use_phase_step, phase_step, regular_step)
                segment_start = position + direction * t
                alpha, _scalar, _position, scalar_end = self._segment_eval(
                    segment_start, direction, current_step, march_active,
                    scalar_start=scalar_start if has_preintegration else None,
                )
                segment_transmission = 1.0 - alpha
                scalar_start = dr.select(march_active, scalar_end, scalar_start)
                transmission = transmission * segment_transmission
                t = t + current_step
                first_segment = mi.Bool(False)
                march_active = (
                    active
                    & (t < max_distance)
                    & (transmission > 1.0e-4)
                    & (current_step > 0.0)
                )
                return t, transmission, first_segment, march_active, scalar_start

            try:
                t, transmission, first_segment, march_active, scalar_start = dr.while_loop(
                    state=(t, transmission, first_segment, march_active, scalar_start),
                    cond=loop_cond,
                    body=loop_body,
                    mode="symbolic",
                    label="vtkweb live secondary volume march",
                )
            except Exception as exc:
                _diagnose_drjit_exception("live secondary volume march", exc)
                raise
            transmission = dr.clip(transmission, 0.0, 1.0)
            if return_endpoint:
                return transmission, position + direction * t, t
            return transmission

        def _volume_shadow(self, position, light_direction, light_index,
                           sample_index, march_phase, active=True):
            """Live shadow fallback (used when no baked illumination exists)."""
            return self._live_transmittance(
                position, dr.normalize(light_direction),
                step_distance=self.sample_distance,
                march_phase=march_phase, active=active,
            )

        def _hybrid_cached_shadow(self, position, light_direction, march_phase, active=True):
            """Live first, then read the baseline combined illumination cache.

            Valid only with exactly one active directional light, where the
            combined texture equals intensity * that light's transmittance.
            """
            direction = dr.normalize(light_direction)
            xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
            shape = tuple(int(v) for v in self.scalar_volume.shape)
            nz, ny, nx = shape[-4:-1] if len(shape) >= 4 else shape[:3]
            # Light directions in this renderer are axis aligned. Convert the
            # debug voxel count to world-space distance along the light ray.
            spacings = ((xmax - xmin) / max(nx - 1, 1),
                        (ymax - ymin) / max(ny - 1, 1),
                        (zmax - zmin) / max(nz - 1, 1))
            light_axis = max(range(3), key=lambda i: abs(float(DIRECTIONAL_LIGHT_DIRECTIONS[self.light_indices[0]][i])))
            target = self.hybrid_shadow_voxels * spacings[light_axis]
            local, endpoint, travelled = self._live_transmittance(
                position, direction,
                step_distance=self.sample_distance,
                march_phase=march_phase,
                active=active,
                distance_limit=target,
                return_endpoint=True,
            )
            exit_distance = self._distance_to_volume_exit(position, direction)
            # No cache needed when the local ray reaches the lightward boundary
            # or when local attenuation has already extinguished the ray.
            lookup_active = active & (travelled < exit_distance - 1.0e-6) & (local > 1.0e-4)
            cached = self.shadow_volumes.eval(
                self._texture_position(endpoint, self.shadow_volumes), lookup_active
            )[0]
            # The cache already contains light intensity for a single light.
            return local * dr.select(lookup_active, cached, self.total_light_intensity)

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
            self, color, position, view, sample_index, march_phase, active=True
        ):
            del sample_index
            if self.environment_scattering_samples <= 0:
                return mi.Color3f(0.0)

            if has_environment_texture:
                # Six independent directional transmittance fields. Each short
                # live segment ends at its own direction-specific cache lookup.
                if len(self.environment_lighting_volumes) != 6:
                    return mi.Color3f(0.0)
                directions = _environment_directions(6)
                xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
                shape = tuple(int(v) for v in self.scalar_volume.shape)
                nz, ny, nx = shape[-4:-1] if len(shape) >= 4 else shape[:3]
                spacings = ((xmax - xmin) / max(nx - 1, 1),
                            (ymax - ymin) / max(ny - 1, 1),
                            (zmax - zmin) / max(nz - 1, 1))
                total = mi.Float(0.0)
                for i, direction_tuple in enumerate(directions):
                    direction = mi.Vector3f(*direction_tuple)
                    texture = self.environment_lighting_volumes[i]
                    if DEBUG_HYBRID_ENV_VOXELS > 0.0:
                        axis = next(j for j, component in enumerate(direction_tuple) if component)
                        distance = self.hybrid_env_voxels * spacings[axis]
                        local, endpoint, travelled = self._live_transmittance(
                            position, direction, step_distance=self.sample_distance,
                            march_phase=march_phase, active=active,
                            distance_limit=distance, return_endpoint=True,
                        )
                        exit_distance = self._distance_to_volume_exit(position, direction)
                        # If the local segment reaches the exit or the lighting
                        # reach, there is no remaining cached segment to consume.
                        lookup_active = (active & (travelled < exit_distance - 1.0e-6)
                                         & (travelled < mi.Float(self._shadow_extent()) - 1.0e-6)
                                         & (local > 1.0e-4))
                        cached = texture.eval(
                            self._texture_position(endpoint, texture), lookup_active
                        )[0]
                        visibility = local * dr.select(lookup_active, cached, mi.Float(1.0))
                    else:
                        visibility = texture.eval(
                            self._texture_position(position, texture), active
                        )[0]
                    total += visibility
                total *= 1.0 / 6.0  # Match the live isotropic phase convention (phase = 1).
            else:
                # Live mode evaluates visibility from the current opacity TF for
                # the same deterministic sphere directions used by the baker.
                directions = _environment_directions(self.environment_scattering_samples)
                query_view = dr.normalize(view)
                total = mi.Float(0.0)
                # Match primary and directional-shadow segment lengths exactly.
                step = self.sample_distance
                for direction_tuple in directions:
                    direction = mi.Vector3f(*direction_tuple)
                    visibility = self._live_transmittance(
                        position,
                        direction,
                        step_distance=step,
                        march_phase=march_phase,
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

        def _scatter_color(
            self,
            color,
            position,
            segment_start,
            segment_end,
            view_direction,
            sample_index,
            march_phase,
            active=True,
        ):
            """Pure volumetric lighting with no gradient/Phong surface term.

            Lighting is evaluated at the segment contribution centroid. In
            directional shadow rays start at the contribution centroid.
            """
            del sample_index
            view = dr.normalize(view_direction)

            # Preserve an unlit path at scattering=0 while making 2.0 the full
            # scattering model. Intermediate values simply interpolate between
            # the transfer-function color and the physically motivated lighting.
            secondary = mi.Color3f(0.0)
            if has_shadow_texture and self.shadow_volumes:
                # Isotropic phase (g=0): all directional lights can share one
                # intensity-weighted full-reach illumination field.
                if len(self.light_indices) == 1 and DEBUG_HYBRID_SHADOW_VOXELS > 0.0:
                    baked = self._hybrid_cached_shadow(
                        position, self.light_directions[0], march_phase, active
                    )
                else:
                    # Exact baseline for distance=0 and multi-light scenes.
                    baked = self.shadow_volumes.eval(
                        self._texture_position(position, self.shadow_volumes), active
                    )[0]
                lighting = self.ambient * self.total_light_intensity + (1.0 - self.ambient) * baked
                secondary = color * self.diffuse * lighting
            else:
                for light_index, light_direction, intensity in zip(
                    self.light_indices, self.light_directions, self.light_intensities
                ):
                    shadow = self._volume_shadow(
                        position, light_direction, light_index, 0, march_phase, active
                    )
                    phase = self._phase_function(dr.dot(-light_direction, view))
                    visibility = self.ambient + (1.0 - self.ambient) * shadow
                    secondary += intensity * visibility * phase * color * self.diffuse

            secondary = secondary + self._environment_scatter(
                color, position, view, 0, march_phase, active
            )
            weight = dr.opaque(
                mi.Float,
                max(0.0, min(1.0, 0.5 * self.volumetric_scattering_blending)),
            )
            return dr.lerp(color, secondary, weight)

        def evaluate_segment(
            self,
            start_position,
            direction,
            step_distance,
            view_direction,
            march_phase,
            active=True,
            sample_index=None,
            scalar_start=None,
        ):
            """Pre-integrate one primary segment from its endpoint scalars.

            Classification/extinction is integrated over the scalar interval
            instead of point-sampling the transfer function. Lighting is
            evaluated once at the extinction-weighted contribution centroid.
            """
            alpha, scalar, position, scalar_end = self._segment_eval(
                start_position, direction, step_distance, active,
                scalar_start=scalar_start,
            )
            color_x = self._normalized_scalar(scalar, self.color)
            color = self._color(color_x, self.color, active)
            if self.shade and self.volumetric_scattering_blending > 0.0:
                shade_active = active & (alpha > 1.0e-4)
                color = dr.select(
                    shade_active,
                    self._scatter_color(
                        color,
                        position,
                        start_position,
                        start_position + direction * step_distance,
                        -view_direction,
                        sample_index,
                        march_phase,
                        shade_active,
                    ),
                    color,
                )
            return color, alpha, scalar_end

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

        def update(
            self,
            volumes,
            *,
            background_color,
            ambient_light,
            directional_lights,
        ):
            self.background_color = mi.Color3f(*map(float, background_color))
            for direct_volume, volume in zip(self.volumes, volumes):
                direct_volume.update_runtime(
                    volume,
                    ambient_light=ambient_light,
                    directional_lights=directional_lights,
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

            # Runtime-only per-pixel march phase. The sampler is reseeded for
            # every progressive pass, so the pattern changes between frames
            # without embedding frame-dependent Python constants in JIT kernels.
            pixel_phase = sampler.next_1d(active)

            # The Python loop is only over the small/static set of vtkweb volume
            # representations. The per-ray march itself is an explicit symbolic
            # Dr.Jit loop, so it executes in generated device code without
            # relying on @dr.syntax's transformed name-resolution environment.
            for volume_index, volume in enumerate(self.volumes):
                entry, exit, volume_active = volume.ray_segment(ray, active)
                exit = dr.minimum(exit, surface_limit)
                volume_active = volume_active & (exit >= entry)
                volume_hit = volume_hit | volume_active

                step = volume.sample_distance
                march_phase = pixel_phase
                t = entry
                sample_index = mi.UInt32(0)
                march_active = (
                    volume_active
                    & (t < exit)
                    & (accumulated_alpha < 0.995)
                )

                scalar_start = (volume._sample_scalar(ray.o + ray.d * entry, march_active)
                                if volume.preintegration is not None else mi.Float(0.0))

                def loop_cond(t, sample_index, result, accumulated_alpha, march_active, scalar_start):
                    del t, sample_index, result, accumulated_alpha, scalar_start
                    return march_active

                def loop_body(t, sample_index, result, accumulated_alpha, march_active, scalar_start):
                    remaining = dr.maximum(exit - t, 0.0)
                    regular_step = dr.minimum(step, remaining)
                    phase_step = dr.minimum(step * march_phase, remaining)
                    use_phase_step = (
                        (sample_index == mi.UInt32(0))
                        & (phase_step > step * 1.0e-6)
                    )
                    current_step = dr.select(use_phase_step, phase_step, regular_step)
                    segment_start = ray.o + ray.d * t
                    color, alpha, scalar_end = volume.evaluate_segment(
                        segment_start,
                        ray.d,
                        current_step,
                        ray.d,
                        march_phase,
                        march_active,
                        sample_index,
                        scalar_start=scalar_start if volume.preintegration is not None else None,
                    )
                    scalar_start = dr.select(march_active, scalar_end, scalar_start)
                    alpha_before = accumulated_alpha
                    weight = (1.0 - alpha_before) * alpha
                    result = result + weight * color
                    alpha_after = alpha_before + weight

                    accumulated_alpha = alpha_after
                    t = t + current_step
                    sample_index = sample_index + 1
                    march_active = (
                        volume_active
                        & (t < exit)
                        & (current_step > 0.0)
                        & (accumulated_alpha < 0.995)
                    )
                    return (t, sample_index, result, accumulated_alpha, march_active, scalar_start)

                try:
                    (
                        t,
                        sample_index,
                        result,
                        accumulated_alpha,
                        march_active,
                        scalar_start,
                    ) = dr.while_loop(
                        state=(t, sample_index, result, accumulated_alpha, march_active, scalar_start),
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
                    _diagnose_drjit_exception("primary DVR loop", exc)
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



def _volume_opacity_reference_distance(image: vtk.vtkImageData) -> float:
    """Physical distance for which transfer-function opacity is defined.

    Keep this independent of the user-controlled ray-march step so changing
    sample_distance changes integration accuracy, not material extinction.
    """
    spacing = [abs(float(value)) for value in image.GetSpacing() if abs(float(value)) > 0]
    if spacing:
        return min(spacing)
    return 1.0


def _volume_sample_distance(
    image: vtk.vtkImageData,
    properties: dict[str, Any],
) -> float:
    del image
    return max(float(properties.get("sample_distance", 1.0)), 1.0e-12)


def _make_vpt_bounds_integrator(mi, dr, bounds_list):
    """Diagnostic VPT integrator: world-space slab intersection, no voxel shading.

    Returns the closest visible box face, or the regular path-traced scene
    (including the HDRI) on misses. Cameras inside a box see the exit face.
    The next milestone replaces face coloring with scalar-field transport.
    """
    colors = (
        mi.Color3f(1.0, .25, .25), mi.Color3f(.25, 1.0, 1.0),
        mi.Color3f(.25, 1.0, .25), mi.Color3f(1.0, .25, 1.0),
        mi.Color3f(.25, .45, 1.0), mi.Color3f(1.0, .85, .25),
    )

    class VPTBoundsIntegrator(mi.SamplingIntegrator):
        def __init__(self):
            super().__init__(mi.Properties())
            self.surface = mi.load_dict({"type": "path", "max_depth": 4,
                                         "hide_emitters": False})

        def sample(self, scene, sampler, ray, medium=None, active=True):
            ray = mi.Ray3f(ray)
            active = mi.Bool(active)
            background, valid, aovs = self.surface.sample(
                scene, sampler, ray, medium, active
            )
            si = scene.ray_intersect(ray, active=active)
            closest = dr.select(si.is_valid(), si.t, mi.Float(float("inf")))
            face_color = mi.Color3f(0.0)
            hit = mi.Bool(False)
            for bounds in bounds_list:
                xmin, xmax, ymin, ymax, zmin, zmax = bounds
                if not (xmin < xmax and ymin < ymax and zmin < zmax):
                    continue
                lower = (xmin, ymin, zmin)
                upper = (xmax, ymax, zmax)
                near = mi.Float(-float("inf"))
                far = mi.Float(float("inf"))
                near_face = mi.UInt32(0)
                far_face = mi.UInt32(0)
                for axis, (origin, direction) in enumerate(
                    ((ray.o.x, ray.d.x), (ray.o.y, ray.d.y),
                     (ray.o.z, ray.d.z))
                ):
                    parallel = dr.abs(direction) < 1e-12
                    safe_dir = dr.select(parallel, mi.Float(1.0), direction)
                    ta = (lower[axis] - origin) / safe_dir
                    tb = (upper[axis] - origin) / safe_dir
                    lo = dr.minimum(ta, tb)
                    hi = dr.maximum(ta, tb)
                    lo = dr.select(parallel, mi.Float(-float("inf")), lo)
                    hi = dr.select(parallel, mi.Float(float("inf")), hi)
                    inside_slab = (origin >= lower[axis]) & (origin <= upper[axis])
                    hi = dr.select(parallel & ~inside_slab,
                                   mi.Float(-float("inf")), hi)
                    neg = direction > 0
                    entering_face = mi.UInt32(2 * axis) + dr.select(neg, 0, 1)
                    exiting_face = mi.UInt32(2 * axis) + dr.select(neg, 1, 0)
                    near_face = dr.select(lo > near, entering_face, near_face)
                    far_face = dr.select(hi < far, exiting_face, far_face)
                    near = dr.maximum(near, lo)
                    far = dr.minimum(far, hi)
                inside = near < 0
                distance = dr.select(inside, far, near)
                face = dr.select(inside, far_face, near_face)
                visible = active & (far >= dr.maximum(near, 0)) & (distance >= 0) & (distance < closest)
                selected = mi.Color3f(0.0)
                for i, color in enumerate(colors):
                    selected = dr.select(face == i, color, selected)
                face_color = dr.select(visible, selected, face_color)
                closest = dr.select(visible, distance, closest)
                hit |= visible
            return dr.select(hit, face_color, background), valid | hit, aovs

    return VPTBoundsIntegrator()


def _make_vpt_extinction_integrator(mi, dr, volumes):
    """Absorption-only VPT checkpoint: attenuate scene/HDRI radiance by voxel extinction.

    No scattering or emission. Optical depth uses DVR reference-distance units.
    """
    if len(volumes) != 1 and any(v.scattering_albedo > 0 for v in volumes):
        raise NotImplementedError(
            "VPT single scattering currently supports exactly one volume; "
            "cross-volume shadow transmittance is not yet implemented")
    compiled = []
    for volume in volumes:
        mapping = volume.opacity_mapping
        size = max(2, int(volume.preintegration_size))
        tau_lut, _ = _build_gpu_preintegration_tables(
            mi, dr, mapping, 1.0, size=size,
            integration_samples=PREINTEGRATION_SAMPLES, with_centroid=False,
        )
        tensor = mi.TensorXf(tau_lut, shape=(size, size, 1))
        preintegration = mi.Texture2f(tensor, use_accel=False)
        compiled.append((volume.bounds, volume.resources.scalar.texture,
                         float(mapping["range"][0]),
                         max(float(mapping["range"][1]) - float(mapping["range"][0]), 1e-20),
                         preintegration, size,
                         max(float(volume.opacity_reference_distance), 1e-12),
                         max(float(volume.sample_distance), 1e-8)))

    class VPTExtinctionIntegrator(mi.SamplingIntegrator):
        def __init__(self):
            super().__init__(mi.Properties())
            self.surface = mi.load_dict({"type": "path", "max_depth": 4,
                                         "hide_emitters": False})

        def sample(self, scene, sampler, ray, medium=None, active=True):
            ray = mi.Ray3f(ray)
            active = mi.Bool(active)
            background, valid, aovs = self.surface.sample(scene, sampler, ray, medium, active)
            si = scene.ray_intersect(ray, active=active)
            surface_t = dr.select(si.is_valid(), si.t, mi.Float(float("inf")))
            result = background
            total_tau = mi.Float(0)
            hit_any = mi.Bool(False)
            scattered_light = mi.Color3f(0.0)
            for bounds, texture, minimum, width, preintegration, table_size, reference, step in compiled:
                xmin, xmax, ymin, ymax, zmin, zmax = bounds
                if not (xmin < xmax and ymin < ymax and zmin < zmax):
                    continue
                near = mi.Float(-float("inf"))
                far = mi.Float(float("inf"))
                for origin, direction, low, high in (
                    (ray.o.x, ray.d.x, xmin, xmax),
                    (ray.o.y, ray.d.y, ymin, ymax),
                    (ray.o.z, ray.d.z, zmin, zmax),
                ):
                    parallel = dr.abs(direction) < 1e-12
                    safe = dr.select(parallel, mi.Float(1), direction)
                    ta, tb = (low-origin)/safe, (high-origin)/safe
                    lo, hi = dr.minimum(ta, tb), dr.maximum(ta, tb)
                    inside = (origin >= low) & (origin <= high)
                    near = dr.maximum(near, dr.select(parallel, -float("inf"), lo))
                    far = dr.minimum(far, dr.select(parallel,
                        dr.select(inside, float("inf"), -float("inf")), hi))
                start = dr.maximum(near, mi.Float(0))
                end = dr.minimum(far, surface_t)
                active_volume = active & (end > start)
                # Match DVR's physical sample distance and endpoint-based
                # preintegration. Do not stretch a fixed sample budget across
                # the whole volume: that changes the integration accuracy.
                segment = dr.maximum(end - start, 0)
                count = mi.UInt32(dr.ceil(segment / step))
                tau = mi.Float(0)
                index = mi.UInt32(0)
                def sample_scalar(t, mask):
                    pos = ray.o + ray.d * t
                    uvw = mi.Point3f((pos.x-xmin)/(xmax-xmin),
                                     (pos.y-ymin)/(ymax-ymin),
                                     (pos.z-zmin)/(zmax-zmin))
                    return dr.clip((texture.eval(uvw, mask)[0]-minimum)/width, 0, 1)

                scalar0 = sample_scalar(start, active_volume)
                def cond(index, tau, scalar0):
                    return (index < count) & active_volume & (tau < 16.0)

                def body(index, tau, scalar0):
                    marching = active_volume & (index < count) & (tau < 16.0)
                    t0 = start + mi.Float(index) * step
                    ds = dr.minimum(step, dr.maximum(end - t0, 0))
                    scalar1 = sample_scalar(t0 + ds, marching)
                    u = (scalar1 * float(table_size - 1) + 0.5) / float(table_size)
                    v = (scalar0 * float(table_size - 1) + 0.5) / float(table_size)
                    tau_reference = preintegration.eval(mi.Point2f(u, v), marching)[0]
                    tau = tau + dr.select(marching, tau_reference * ds / reference, 0)
                    return index + 1, tau, scalar1

                index, tau, scalar0 = dr.while_loop(
                    state=(index, tau, scalar0), cond=cond, body=body,
                    label="vpt_preintegrated_absorption",
                )
                total_tau += dr.select(active_volume, tau, 0)
                hit_any |= active_volume
            # No in-scattering or emission: attenuate the radiance behind the
            # volume, including environment emitters and visible surfaces.
            result = background * dr.exp(-total_tau)
            return result, valid | hit_any, aovs

    return VPTExtinctionIntegrator()


def _vpt_brick_majorants(volume, brick_size=8):
    """Conservative extinction bounds for trilinearly interpolated scalar data.

    The one-voxel halo covers texture interpolation at brick boundaries.
    A piecewise-linear opacity TF reaches its maximum on a scalar interval
    at an interval endpoint or an interior control point.
    """
    values = np.asarray(volume.scalar_values)
    if values.ndim != 3:
        raise ValueError("VPT requires a 3D scalar array")
    nz, ny, nx = values.shape
    bx, by, bz = [(n + brick_size - 1) // brick_size for n in (nx, ny, nz)]
    points = np.asarray(volume.opacity_mapping["control_points"], dtype=np.float64)
    reference = max(float(volume.opacity_reference_distance), 1e-12)
    lo, hi = map(float, volume.opacity_mapping["range"])
    width = max(hi - lo, 1e-20)
    result = np.zeros((bz, by, bx), dtype=np.float32)
    for z in range(bz):
        for y in range(by):
            for x in range(bx):
                # Texture coordinates map across the whole image extent. Include
                # a halo so no trilinear footprint crosses outside this range.
                patch = values[max(0,z*brick_size-1):min(nz,(z+1)*brick_size+2),
                               max(0,y*brick_size-1):min(ny,(y+1)*brick_size+2),
                               max(0,x*brick_size-1):min(nx,(x+1)*brick_size+2)]
                vmin = np.clip((float(np.min(patch))-lo)/width, 0, 1)
                vmax = np.clip((float(np.max(patch))-lo)/width, 0, 1)
                candidates = [vmin, vmax]
                candidates.extend(float(q) for q in points[:,0] if vmin <= q <= vmax)
                alpha = max(float(np.interp(q, points[:,0], points[:,1])) for q in candidates)
                sigma = -np.log1p(-min(max(alpha, 0.0), 1-1e-6)) / reference
                result[z,y,x] = np.float32(sigma * (1 + 1e-5) + 1e-7) if sigma > 0 else np.float32(0)
    return np.ascontiguousarray(result), (bx, by, bz)


def _make_vpt_delta_integrator(mi, dr, volumes):
    """Spatial-majorant delta tracking with optional HDRI single scattering."""
    compiled = []
    for volume in volumes:
        mapping = volume.opacity_mapping
        points = np.asarray(mapping["control_points"], dtype=np.float32)
        x = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
        lut = np.interp(x, points[:,0], np.clip(points[:,1], 0, 1-1e-6))
        # TF color tints scattered radiance, not scalar extinction. The color
        # mapping may have a different scalar range from the opacity mapping.
        color_mapping = volume.color_mapping
        color_points = np.asarray(color_mapping["control_points"], dtype=np.float32)
        color_x = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
        color_luts = tuple(
            mi.Float(np.ascontiguousarray(np.interp(
                color_x, color_points[:, 0], color_points[:, channel])))
            for channel in (1, 2, 3)
        )
        color_min = float(color_mapping["range"][0])
        color_width = max(float(color_mapping["range"][1]) - color_min, 1e-20)
        grid, shape = _vpt_brick_majorants(volume)
        if VPT_DEBUG:
            print(f"[VPT DEBUG] majorant grid: shape={shape} min={float(np.min(grid)):.6g} max={float(np.max(grid)):.6g} nonzero={int(np.count_nonzero(grid))}/{grid.size}", flush=True)
        compiled.append((volume.bounds, volume.resources.scalar.texture,
                         float(mapping["range"][0]),
                         max(float(mapping["range"][1])-float(mapping["range"][0]), 1e-20),
                         mi.Float(np.ascontiguousarray(lut)), len(lut),
                         mi.Float(grid.ravel()), shape,
                         max(float(volume.opacity_reference_distance), 1e-12),
                         float(np.clip(volume.scattering_albedo, 0, 1)),
                         color_min, color_width, color_luts,
                         max(1, min(4, int(volume.vpt_max_depth))),
                         float(np.clip(volume.vpt_anisotropy, -0.9, 0.9))))

    def next_pcg32(state):
        """Advance a per-lane 32-bit PCG hash state explicitly."""
        state = state * mi.UInt32(747796405) + mi.UInt32(2891336453)
        shift = (state >> 28) + mi.UInt32(4)
        word = ((state >> shift) ^ state) * mi.UInt32(277803737)
        word = (word >> 22) ^ word
        # Convert the high 24 bits to a representable uniform float in [0, 1).
        u = mi.Float(word >> 8) * (1.0 / 16777216.0)
        return state, u

    def shadow_ratio_tracking(ray, active, rng_state, bounds, texture,
                              minimum, width, lut, lut_size, grid, shape, reference):
        """Unbiased null-collision transmittance estimate, using brick DDA.

        This shares the spatial-majorant representation with camera tracking.
        Each null collision multiplies throughput by (1 - sigma / majorant).
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        nx, ny, nz = shape
        near = mi.Float(-float("inf"))
        far = mi.Float(float("inf"))
        for origin, direction, low, high in (
            (ray.o.x, ray.d.x, xmin, xmax),
            (ray.o.y, ray.d.y, ymin, ymax),
            (ray.o.z, ray.d.z, zmin, zmax),
        ):
            parallel = dr.abs(direction) < 1e-12
            safe = dr.select(parallel, mi.Float(1), direction)
            ta, tb = (low - origin) / safe, (high - origin) / safe
            inside = (origin >= low) & (origin <= high)
            near = dr.maximum(near, dr.select(parallel, -float("inf"), dr.minimum(ta, tb)))
            far = dr.minimum(far, dr.select(parallel,
                dr.select(inside, float("inf"), -float("inf")), dr.maximum(ta, tb)))
        start = dr.maximum(near, mi.Float(0))
        end = far
        running = active & (end > start)
        p0 = ray.o + ray.d * start

        def initial_index(coord, direction, low, high, count):
            coord = (coord - low) * (count / (high - low))
            index = dr.select(direction < 0, dr.ceil(coord) - 1, dr.floor(coord))
            return mi.Int32(dr.clip(index, 0, count - 1))

        ix = initial_index(p0.x, ray.d.x, xmin, xmax, nx)
        iy = initial_index(p0.y, ray.d.y, ymin, ymax, ny)
        iz = initial_index(p0.z, ray.d.z, zmin, zmax, nz)
        t = start
        weight = mi.Float(1.0)
        iterations = mi.UInt32(0)
        limit = 100000

        def cond(t, ix, iy, iz, running, rng_state, weight, iterations):
            return running & (iterations < limit)

        def body(t, ix, iy, iz, running, rng_state, weight, iterations):
            index = mi.UInt32(ix + nx * (iy + ny * iz))
            sigma_max = dr.gather(mi.Float, grid, index, running)

            def plane_time(index, direction, origin, low, high, count):
                positive, negative = direction > 0, direction < 0
                plane = low + dr.select(positive, mi.Float(index + 1),
                                        mi.Float(index)) * ((high - low) / count)
                safe = dr.select(positive | negative, direction, 1.0)
                return dr.select(positive | negative,
                                 (plane - origin) / safe, float("inf"))

            tx = plane_time(ix, ray.d.x, ray.o.x, xmin, xmax, nx)
            ty = plane_time(iy, ray.d.y, ray.o.y, ymin, ymax, ny)
            tz = plane_time(iz, ray.d.z, ray.o.z, zmin, zmax, nz)
            next_plane = dr.minimum(tx, dr.minimum(ty, tz))
            boundary = dr.minimum(end, next_plane)
            rng_next, u = next_pcg32(rng_state)
            rng_state = dr.select(running, rng_next, rng_state)
            flight = -dr.log(1 - dr.clip(u, 1e-7, 1 - 1e-7)) / dr.maximum(sigma_max, 1e-12)
            candidate_t = t + flight
            candidate = running & (sigma_max > 0) & (candidate_t > t) & (candidate_t < boundary)
            pos = ray.o + ray.d * candidate_t
            uvw = mi.Point3f((pos.x - xmin) / (xmax - xmin),
                             (pos.y - ymin) / (ymax - ymin),
                             (pos.z - zmin) / (zmax - zmin))
            scalar = texture.eval(uvw, candidate)[0]
            scaled = dr.clip((scalar - minimum) / width, 0, 1) * (lut_size - 1)
            i0 = mi.UInt32(dr.floor(scaled))
            i1 = dr.minimum(i0 + 1, mi.UInt32(lut_size - 1))
            opacity = dr.lerp(dr.gather(mi.Float, lut, i0, candidate),
                              dr.gather(mi.Float, lut, i1, candidate),
                              scaled - mi.Float(i0))
            sigma = -dr.log(dr.maximum(1 - opacity, 1e-6)) / reference
            weight *= dr.select(candidate,
                1 - dr.clip(sigma / dr.maximum(sigma_max, 1e-12), 0, 1), 1)
            crossing = running & ~candidate & (next_plane <= end)
            ix += dr.select(crossing & (tx <= next_plane),
                            dr.select(ray.d.x > 0, 1, -1), 0)
            iy += dr.select(crossing & (ty <= next_plane),
                            dr.select(ray.d.y > 0, 1, -1), 0)
            iz += dr.select(crossing & (tz <= next_plane),
                            dr.select(ray.d.z > 0, 1, -1), 0)
            t = dr.maximum(dr.select(candidate, candidate_t, boundary), start)
            in_grid = ((ix >= 0) & (ix < nx) & (iy >= 0) &
                       (iy < ny) & (iz >= 0) & (iz < nz))
            running = running & (t < end) & in_grid & (weight > 0)
            return t, ix, iy, iz, running, rng_state, weight, iterations + 1

        t, ix, iy, iz, running, rng_state, weight, iterations = dr.while_loop(
            state=(t, ix, iy, iz, running, rng_state, weight, iterations),
            cond=cond, body=body, label="vpt_shadow_ratio_tracking")
        if dr.any(running):
            raise RuntimeError("VPT shadow ratio tracking exceeded 100000 iterations")
        return weight

    def hg_phase(cos_theta, g):
        """Henyey-Greenstein phase density per steradian."""
        denom = dr.maximum(1.0 + g*g - 2.0*g*cos_theta, 1e-8)
        return (1.0 - g*g) / (4.0 * np.pi * denom * dr.sqrt(denom))

    def sample_hg_direction(incoming, u1, u2, g):
        """Sample HG about the incident propagation direction (not -incoming)."""
        if abs(g) < 1e-7:
            cos_theta = 1.0 - 2.0*u1
        else:
            ratio = (1.0 - g*g) / (1.0 - g + 2.0*g*u1)
            cos_theta = dr.clip((1.0 + g*g - ratio*ratio) / (2.0*g), -1.0, 1.0)
        sin_theta = dr.sqrt(dr.maximum(0.0, 1.0 - cos_theta*cos_theta))
        phi = 2.0*np.pi*u2
        # Construct a stable orthonormal frame around incoming.
        helper = dr.select(dr.abs(incoming.z) < 0.999,
                           mi.Vector3f(0.0, 0.0, 1.0),
                           mi.Vector3f(0.0, 1.0, 0.0))
        tangent = dr.normalize(dr.cross(helper, incoming))
        bitangent = dr.cross(incoming, tangent)
        return mi.Vector3f(incoming*cos_theta +
                           tangent*(sin_theta*dr.cos(phi)) +
                           bitangent*(sin_theta*dr.sin(phi)))

    class VPTDeltaAbsorptionIntegrator(mi.SamplingIntegrator):
        def __init__(self):
            super().__init__(mi.Properties())
            self.surface = mi.load_dict({"type":"path", "max_depth":4,
                                         "hide_emitters":False})

        def sample(self, scene, sampler, ray, medium=None, active=True):
            ray = mi.Ray3f(ray)
            active = mi.Bool(active)
            background, valid, aovs = self.surface.sample(scene, sampler, ray, medium, active)
            si = scene.ray_intersect(ray, active=active)
            primary_surface_t = dr.select(si.is_valid(), si.t, mi.Float(float("inf")))
            if len(compiled) != 1:
                raise ValueError("VPT multiple scattering currently supports exactly one volume")
            (bounds, texture, minimum, width, lut, lut_size, grid, shape,
             reference, albedo, color_min, color_width, color_luts, max_depth, g) = compiled[0]
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            env = scene.environment()
            seed_u = sampler.next_1d(active)
            rng_state = (mi.UInt32(seed_u * 16777216.0) ^
                         (mi.UInt32(dr.arange(mi.UInt32, dr.width(ray.o.x))) * mi.UInt32(2246822519)) ^
                         mi.UInt32(0x9e3779b9))
            throughput = mi.Color3f(1.0)
            result = mi.Color3f(0.0)
            path_active = active
            hit_any = mi.Bool(False)
            for depth in range(max_depth):
                # Recompute the entry/exit interval for the current path ray.
                near = mi.Float(-float("inf"))
                far = mi.Float(float("inf"))
                for origin, direction, low, high in (
                    (ray.o.x, ray.d.x, xmin, xmax),
                    (ray.o.y, ray.d.y, ymin, ymax),
                    (ray.o.z, ray.d.z, zmin, zmax),
                ):
                    parallel = dr.abs(direction) < 1e-12
                    safe = dr.select(parallel, mi.Float(1), direction)
                    ta, tb = (low-origin)/safe, (high-origin)/safe
                    inside = (origin >= low) & (origin <= high)
                    near = dr.maximum(near, dr.select(parallel, -float("inf"), dr.minimum(ta,tb)))
                    far = dr.minimum(far, dr.select(parallel,
                        dr.select(inside, float("inf"), -float("inf")), dr.maximum(ta,tb)))
                start = dr.maximum(near, mi.Float(0))
                # Only the camera ray uses the surface/background computed by
                # Mitsuba. Secondary paths terminate at the first surface.
                if depth == 0:
                    end = dr.minimum(far, primary_surface_t)
                else:
                    secondary_si = scene.ray_intersect(ray, active=path_active)
                    secondary_t = dr.select(secondary_si.is_valid(), secondary_si.t,
                                            mi.Float(float("inf")))
                    end = dr.minimum(far, secondary_t)
                running = path_active & (end > start)
                if depth == 0:
                    hit_any |= running
                # Rays with no real event retain the background only at depth 0.
                scatter_pos = mi.Point3f(0.0)
                scatter_scalar = mi.Float(0.0)
                scatter_mask = mi.Bool(False)
                real_event = mi.Bool(False)
                # Integer-cell DDA: indices are advanced at exact grid planes.
                # No world-space epsilon is used to select the next brick.
                nx, ny, nz = shape
                p0 = ray.o + ray.d * start
                def initial_index(coord, direction, low, high, count):
                    grid_coord = (coord - low) * (count / (high - low))
                    # At an exact boundary a negative ray enters the lower cell.
                    index = dr.select(direction < 0, dr.ceil(grid_coord) - 1,
                                      dr.floor(grid_coord))
                    return mi.Int32(dr.clip(index, 0, count - 1))
                ix = initial_index(p0.x, ray.d.x, xmin, xmax, nx)
                iy = initial_index(p0.y, ray.d.y, ymin, ymax, ny)
                iz = initial_index(p0.z, ray.d.z, zmin, zmax, nz)
                t = start
                # Seed once from Mitsuba's per-pixel sampler and decorrelate
                # across lanes. All subsequent draws have explicit loop state.
                scatter_pos = mi.Point3f(0.0)
                scatter_scalar = mi.Float(0.0)
                scatter_mask = mi.Bool(False)
                iterations = mi.UInt32(0)
                limit = 100000  # Fail loudly instead of returning a biased frame.

                def cond(t, ix, iy, iz, running, rng_state, iterations, scatter_pos, scatter_scalar, scatter_mask, real_event):
                    return running & (iterations < limit)

                def body(t, ix, iy, iz, running, rng_state, iterations, scatter_pos, scatter_scalar, scatter_mask, real_event):
                    index = mi.UInt32(ix + nx * (iy + ny * iz))
                    sigma_max = dr.gather(mi.Float, grid, index, running)

                    def plane_time(index, direction, origin, low, high, count):
                        positive = direction > 0
                        negative = direction < 0
                        plane = low + dr.select(positive, mi.Float(index + 1),
                                                mi.Float(index)) * ((high - low) / count)
                        safe_direction = dr.select(positive | negative, direction, 1.0)
                        return dr.select(positive | negative,
                                         (plane - origin) / safe_direction,
                                         float("inf"))

                    tx = plane_time(ix, ray.d.x, ray.o.x, xmin, xmax, nx)
                    ty = plane_time(iy, ray.d.y, ray.o.y, ymin, ymax, ny)
                    tz = plane_time(iz, ray.d.z, ray.o.z, zmin, zmax, nz)
                    next_plane = dr.minimum(tx, dr.minimum(ty, tz))
                    boundary = dr.minimum(end, next_plane)
                    next_state, random_u = next_pcg32(rng_state)
                    rng_state = dr.select(running, next_state, rng_state)
                    u = dr.clip(random_u, 1e-7, 1 - 1e-7)
                    flight = -dr.log(1 - u) / dr.maximum(sigma_max, 1e-12)
                    candidate_t = t + flight
                    # A float32 free-flight increment can round back to t.
                    # Treat such an event as a boundary crossing, not a null collision.
                    progressed = candidate_t > t
                    candidate = running & (sigma_max > 0) & progressed & (candidate_t < boundary)
                    sample_pos = ray.o + ray.d * candidate_t
                    uvw = mi.Point3f((sample_pos.x-xmin)/(xmax-xmin),
                                     (sample_pos.y-ymin)/(ymax-ymin),
                                     (sample_pos.z-zmin)/(zmax-zmin))
                    scalar = texture.eval(uvw, candidate)[0]
                    scaled = dr.clip((scalar-minimum)/width, 0, 1) * (lut_size-1)
                    i0 = mi.UInt32(dr.floor(scaled))
                    i1 = dr.minimum(i0+1, mi.UInt32(lut_size-1))
                    opacity = dr.lerp(dr.gather(mi.Float,lut,i0,candidate),
                                      dr.gather(mi.Float,lut,i1,candidate),
                                      scaled-mi.Float(i0))
                    sigma = -dr.log(dr.maximum(1-opacity,1e-6))/reference
                    next_state, accept_u = next_pcg32(rng_state)
                    rng_state = dr.select(candidate, next_state, rng_state)
                    accept = candidate & (accept_u <
                                          dr.clip(sigma/dr.maximum(sigma_max,1e-12),0,1))
                    # A real event is either absorption or scattering. Null
                    # collisions do not consume path depth.
                    next_state, scatter_u = next_pcg32(rng_state)
                    rng_state = dr.select(accept, next_state, rng_state)
                    real_event |= accept
                    scattered = accept & (scatter_u < albedo)
                    scatter_pos = dr.select(scattered, sample_pos, scatter_pos)
                    scatter_scalar = dr.select(scattered, scalar, scatter_scalar)
                    scatter_mask |= scattered
                    crossing = running & ~candidate & (next_plane <= end)
                    # Update every tied axis, including edges and corners.
                    ix += dr.select(crossing & (tx <= next_plane),
                                    dr.select(ray.d.x > 0, 1, -1), 0)
                    iy += dr.select(crossing & (ty <= next_plane),
                                    dr.select(ray.d.y > 0, 1, -1), 0)
                    iz += dr.select(crossing & (tz <= next_plane),
                                    dr.select(ray.d.z > 0, 1, -1), 0)
                    t = dr.select(candidate, candidate_t, boundary)
                    # If a computed plane is behind t because of rounding,
                    # the integer DDA still advances the corresponding cell.
                    t = dr.maximum(t, start)
                    in_grid = ((ix >= 0) & (ix < nx) & (iy >= 0) &
                               (iy < ny) & (iz >= 0) & (iz < nz))
                    running = running & ~accept & (t < end) & in_grid
                    return (t, ix, iy, iz, running, rng_state, iterations + 1, scatter_pos, scatter_scalar, scatter_mask, real_event)

                (t, ix, iy, iz, running, rng_state, iterations, scatter_pos, scatter_scalar, scatter_mask, real_event) = dr.while_loop(
                    state=(t, ix, iy, iz, running, rng_state, iterations, scatter_pos, scatter_scalar, scatter_mask, real_event),
                    cond=cond, body=body, label="vpt_spatial_delta_dda")
                if dr.any(running):
                    raise RuntimeError(
                        "VPT spatial delta tracking exceeded 100000 iterations; "
                        "enable VTKWEB_VPT_DEBUG=1 for majorant-grid diagnostics")
                # A ray with no real collision reaches the surface/HDRI.
                # For the primary ray, this reproduces absorption-only VPT.
                if depth == 0:
                    result += dr.select(path_active & ~real_event, background, mi.Color3f(0.0))
                    # Absorbed rays must not contribute the background.
                    # The tracking loop records all real collisions below.
                if albedo > 0 and env is not None:
                    rng_state, u1 = next_pcg32(rng_state)
                    rng_state, u2 = next_pcg32(rng_state)
                    ref = mi.Interaction3f()
                    ref.p = scatter_pos
                    ds, emitter_weight = env.sample_direction(
                        ref, mi.Point2f(u1, u2), scatter_mask)
                    shadow_ray = mi.Ray3f(scatter_pos, ds.d)
                    shadow_visible = ~scene.ray_test(shadow_ray, active=scatter_mask)
                    shadow_mask = scatter_mask & shadow_visible
                    transmittance = shadow_ratio_tracking(
                        shadow_ray, shadow_mask, rng_state, bounds, texture,
                        minimum, width, lut, lut_size, grid, shape, reference)
                    color_scaled = dr.clip(
                        (scatter_scalar - color_min) / color_width, 0, 1) * 1023
                    ci0 = mi.UInt32(dr.floor(color_scaled))
                    ci1 = dr.minimum(ci0 + 1, mi.UInt32(1023))
                    cf = color_scaled - mi.Float(ci0)
                    rgb = [dr.lerp(dr.gather(mi.Float, channel, ci0, scatter_mask),
                                   dr.gather(mi.Float, channel, ci1, scatter_mask), cf)
                           for channel in color_luts]
                    tint = mi.Color3f(*rgb)
                    throughput = dr.select(scatter_mask, throughput * tint, throughput)
                    result += dr.select(shadow_mask,
                        throughput * emitter_weight * hg_phase(dr.dot(ray.d, ds.d), g) * transmittance,
                        mi.Color3f(0.0))
                if depth + 1 < max_depth:
                    # Sample the HG phase density around the incoming
                    # propagation direction. The phase/pdf ratio is 1.
                    rng_state, u1 = next_pcg32(rng_state)
                    rng_state, u2 = next_pcg32(rng_state)
                    direction = sample_hg_direction(ray.d, u1, u2, g)
                    ray = mi.Ray3f(scatter_pos, direction)
                    path_active = scatter_mask
            return result, valid | hit_any, aovs
    return VPTDeltaAbsorptionIntegrator()
