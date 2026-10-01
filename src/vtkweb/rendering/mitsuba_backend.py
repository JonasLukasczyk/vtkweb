from __future__ import annotations

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
    cached_integrator: Any | None = None
    cached_integrator_key: tuple | None = None
    live_integrator: Any | None = None
    live_integrator_key: tuple | None = None
    caching: bool = True


@dataclass
class MitsubaRepresentationHandle:
    kind: str
    scene_object: Any | None = None
    bounds: tuple[float, float, float, float, float, float] | None = None
    scalar_volume: Any | None = None
    scalar_volume_key: tuple[Any, ...] | None = None
    color_mapping: dict[str, Any] | None = None
    opacity_mapping: dict[str, Any] | None = None
    sample_distance: float = 1.0
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

    def __init__(self, transfer_function_provider=None) -> None:
        import drjit as dr
        import mitsuba as mi

        self._transfer_function_provider = transfer_function_provider or (
            lambda _name: None
        )
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
        self._cached_direct_volume_type = _make_direct_volume_type(mi, dr, cached_lighting=True)
        self._live_direct_volume_type = _make_direct_volume_type(mi, dr, cached_lighting=False)
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

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

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
            if name == "caching":
                return bool(handle.caching)
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
            if name == "caching":
                handle.caching = bool(value)
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
                "caching": bool(handle.caching),
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
        rep_handle = self._create_handle(representation, source)
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
        rep_handle = self._create_handle(representation, source, previous=previous)
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
    ) -> MitsubaRepresentationHandle:
        if representation.kind == "surface":
            return self._create_surface_handle(representation, source)
        if representation.kind == "wireframe":
            return self._create_wireframe_handle(representation, source)
        if representation.kind == "outline":
            return self._create_outline_handle(representation, source)
        if representation.kind == "volume":
            return self._create_volume_handle(representation, source, previous=previous)

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
        scalar_key = (
            data.GetAddressAsString("vtkweb"),
            int(data.GetMTime()),
            array_name,
            association,
            interpolation,
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
            filter_type = "nearest" if interpolation == "nearest" else "trilinear"
            to_world = _image_volume_transform(self.mi, bounds)
            with self._render_lock:
                scalar_volume = self.mi.load_dict(
                    {
                        "type": "gridvolume",
                        "data": self.mi.TensorXf(scalar_values),
                        "raw": True,
                        "filter_type": filter_type,
                        "wrap_mode": "clamp",
                        "to_world": to_world,
                    }
                )

        handle = MitsubaRepresentationHandle(
            kind="volume",
            bounds=bounds,
            scalar_volume=scalar_volume,
            scalar_volume_key=scalar_key,
            color_mapping=transfer_function["color"],
            opacity_mapping=transfer_function["opacity"],
            sample_distance=_volume_sample_distance(data, representation.properties),
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

        # Dataset-static preprocessing is shared by both render paths.
        # TF/light-dependent lighting caches are built lazily only when the
        # cached path is actually selected for rendering.
        self._build_gradient_field(handle)
        return handle

    def _build_gradient_field(self, volume: MitsubaRepresentationHandle) -> None:
        """Bake a camera/TF-independent float32 scalar-gradient field."""
        scalars = volume.scalar_values
        bounds = volume.bounds
        if scalars is None or bounds is None or scalars.ndim != 3:
            return
        key = (
            volume.scalar_volume_key,
            tuple(float(v) for v in volume.gradient_step),
            tuple(float(v) for v in bounds),
            "float32",
        )
        if volume.gradient_volume is not None and volume.gradient_volume_key == key:
            return

        start = time.perf_counter()
        print(
            f"Mitsuba bake gradient: start shape={tuple(int(v) for v in scalars.shape)} dtype=float32"
        )
        hx, hy, hz = (max(abs(float(v)), 1.0e-12) for v in volume.gradient_step)
        # numpy arrays are laid out as (z, y, x). edge_order=1 keeps the bake
        # valid for dimensions with only two samples while matching central
        # differences in the interior.
        gz, gy, gx = np.gradient(
            np.asarray(scalars, dtype=np.float32),
            np.float32(hz),
            np.float32(hy),
            np.float32(hx),
            edge_order=1,
        )
        gradient = np.empty((*scalars.shape, 3), dtype=np.float32)
        gradient[..., 0] = np.asarray(gx, dtype=np.float32)
        gradient[..., 1] = np.asarray(gy, dtype=np.float32)
        gradient[..., 2] = np.asarray(gz, dtype=np.float32)
        gradient = np.ascontiguousarray(gradient, dtype=np.float32)

        with self._render_lock:
            volume.gradient_volume = self.mi.load_dict(
                {
                    "type": "gridvolume",
                    "data": self.mi.TensorXf(gradient),
                    "raw": True,
                    "filter_type": "trilinear",
                    "wrap_mode": "clamp",
                    "to_world": _image_volume_transform(self.mi, bounds),
                }
            )
        volume.gradient_volume_key = key
        elapsed = time.perf_counter() - start
        print(f"Mitsuba bake gradient: end {elapsed:.3f}s")

    def _build_directional_shadow_field(
        self, volume: MitsubaRepresentationHandle
    ) -> None:
        """Bake one high-quality +Z optical-depth field and cache it.

        The field depends on the scalar data, opacity mapping, sample distance,
        and bake quadrature quality, but not on the camera. Rendering therefore
        only samples the cached field until one of those inputs changes.
        """
        scalars = volume.scalar_values
        mapping = volume.opacity_mapping
        bounds = volume.bounds
        if scalars is None or mapping is None or bounds is None:
            return
        if scalars.ndim != 3 or scalars.shape[0] < 2:
            return

        # Two-point Gauss-Legendre quadrature per voxel interval is an
        # internal quality choice. One sample is already close for linearly
        # interpolated scalar data, while two robustly handles intervals that
        # cross transfer-function control-point changes without exposing a
        # misleading end-user quality parameter.
        sample_count = 2
        opacity_signature = (
            tuple(float(v) for v in mapping["range"]),
            tuple(
                tuple(float(component) for component in point)
                for point in mapping["control_points"]
            ),
        )
        shadow_key = (
            volume.scalar_volume_key,
            opacity_signature,
            float(volume.sample_distance),
            (0.0, 0.0, 1.0),  # fixed directional-light direction
            tuple(float(v) for v in bounds),
        )
        if volume.shadow_volume is not None and volume.shadow_volume_key == shadow_key:
            return

        start = time.perf_counter()
        print(
            f"Mitsuba bake directional shadow: start shape={tuple(int(v) for v in scalars.shape)} "
            f"quadrature_samples={sample_count} dtype=float32"
        )
        nz, ny, nx = scalars.shape
        lower = np.asarray(scalars[:-1], dtype=np.float32)
        upper = np.asarray(scalars[1:], dtype=np.float32)

        # Gauss-Legendre quadrature gives a high-quality one-time integral
        # within every voxel interval without requiring progressive passes.
        nodes, weights = np.polynomial.legendre.leggauss(sample_count)
        u_values = np.asarray(0.5 * (nodes + 1.0), dtype=np.float32)
        weights = np.asarray(0.5 * weights, dtype=np.float32)
        mean_sigma = np.zeros((nz - 1, ny, nx), dtype=np.float32)
        sample_distance = np.float32(max(float(volume.sample_distance), 1.0e-12))
        for u, weight in zip(u_values, weights):
            sample_scalars = lower + u * (upper - lower)
            alpha = _evaluate_opacity_mapping(mapping, sample_scalars)
            alpha = np.clip(alpha, np.float32(0.0), np.float32(1.0 - 1.0e-6))
            sigma = np.asarray(-np.log1p(-alpha) / sample_distance, dtype=np.float32)
            mean_sigma += weight * sigma

        zmin, zmax = float(bounds[4]), float(bounds[5])
        dz = np.float32((zmax - zmin) / float(max(nz - 1, 1)))
        segment_tau = np.asarray(mean_sigma * dz, dtype=np.float32)

        # tau[k] is the optical depth from grid plane k toward +Z. The final
        # plane lies on the light-facing boundary and therefore has tau=0.
        tau = np.zeros((nz, ny, nx), dtype=np.float32)
        tau[:-1] = np.cumsum(segment_tau[::-1], axis=0, dtype=np.float32)[::-1]

        with self._render_lock:
            volume.shadow_volume = self.mi.load_dict(
                {
                    "type": "gridvolume",
                    "data": self.mi.TensorXf(np.ascontiguousarray(tau)),
                    "raw": True,
                    "filter_type": "trilinear",
                    "wrap_mode": "clamp",
                    "to_world": _image_volume_transform(self.mi, bounds),
                }
            )
        volume.shadow_volume_key = shadow_key
        elapsed = time.perf_counter() - start
        print(f"Mitsuba bake directional shadow: end {elapsed:.3f}s")

    @staticmethod
    def _bilinear_shift(slice2d: np.ndarray, offset0: float, offset1: float) -> np.ndarray:
        """Sample a 2D slice at coordinates shifted by fixed fractional offsets."""
        n0, n1 = slice2d.shape
        c0 = np.arange(n0, dtype=np.float64)[:, None] + float(offset0)
        c1 = np.arange(n1, dtype=np.float64)[None, :] + float(offset1)
        i0 = np.floor(c0).astype(np.int64)
        i1 = np.floor(c1).astype(np.int64)
        f0 = c0 - i0
        f1 = c1 - i1
        j0 = i0 + 1
        j1 = i1 + 1

        valid00 = (i0 >= 0) & (i0 < n0) & (i1 >= 0) & (i1 < n1)
        valid10 = (j0 >= 0) & (j0 < n0) & (i1 >= 0) & (i1 < n1)
        valid01 = (i0 >= 0) & (i0 < n0) & (j1 >= 0) & (j1 < n1)
        valid11 = (j0 >= 0) & (j0 < n0) & (j1 >= 0) & (j1 < n1)

        ci0 = np.clip(i0, 0, n0 - 1)
        cj0 = np.clip(j0, 0, n0 - 1)
        ci1 = np.clip(i1, 0, n1 - 1)
        cj1 = np.clip(j1, 0, n1 - 1)

        out = np.zeros((n0, n1), dtype=np.float32)
        out += np.where(valid00, slice2d[ci0, ci1], 0.0) * ((1.0 - f0) * (1.0 - f1))
        out += np.where(valid10, slice2d[cj0, ci1], 0.0) * (f0 * (1.0 - f1))
        out += np.where(valid01, slice2d[ci0, cj1], 0.0) * ((1.0 - f0) * f1)
        out += np.where(valid11, slice2d[cj0, cj1], 0.0) * (f0 * f1)
        return out

    @staticmethod
    def _shift_axis_linear(data: np.ndarray, offset: float, axis: int) -> np.ndarray:
        """Linearly sample ``data`` at a constant fractional index offset."""
        n = data.shape[axis]
        base = math.floor(float(offset))
        frac = float(offset) - base
        coords0 = np.arange(n, dtype=np.int64) + base
        coords1 = coords0 + 1
        valid0 = (coords0 >= 0) & (coords0 < n)
        valid1 = (coords1 >= 0) & (coords1 < n)
        take0 = np.take(data, np.clip(coords0, 0, n - 1), axis=axis)
        take1 = np.take(data, np.clip(coords1, 0, n - 1), axis=axis)
        shape = [1] * data.ndim
        shape[axis] = n
        take0 = take0 * valid0.reshape(shape)
        take1 = take1 * valid1.reshape(shape)
        return np.asarray((1.0 - frac) * take0 + frac * take1, dtype=np.float32)

    @classmethod
    def _shift_volume_linear(
        cls, data: np.ndarray, offsets: tuple[float, float, float]
    ) -> np.ndarray:
        result = np.asarray(data, dtype=np.float32)
        for axis, offset in enumerate(offsets):
            result = cls._shift_axis_linear(result, offset, axis)
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
        min_extent = min(float(volume.sample_distance), max_extent)
        reach = max(0.0, min(1.0, float(volume.global_illumination_reach)))
        return (min_extent - max_extent) * ((1.0 - reach) ** 0.33) + max_extent

    def _build_environment_extinction_array(
        self, volume: MitsubaRepresentationHandle
    ) -> np.ndarray | None:
        """Build one float32 extinction grid reused by all environment directions."""
        scalars = volume.scalar_values
        mapping = volume.opacity_mapping
        if scalars is None or mapping is None or scalars.ndim != 3:
            return None

        # Evaluate opacity before spatial downsampling. This avoids missing thin
        # opaque structures when the TF is nonlinear or sharply peaked.
        alpha_full = _evaluate_opacity_mapping(mapping, np.asarray(scalars, dtype=np.float32))
        alpha_full = np.clip(alpha_full, np.float32(0.0), np.float32(1.0 - 1.0e-6))
        sigma_full = np.asarray(
            -np.log1p(-alpha_full) / np.float32(max(float(volume.sample_distance), 1.0e-12)),
            dtype=np.float32,
        )

        factor = max(1, int(round(float(volume.environment_scattering_step_factor))))
        if factor <= 1 or min(sigma_full.shape) < 2:
            return np.ascontiguousarray(sigma_full, dtype=np.float32)

        # Average extinction over factor-sized blocks instead of striding scalar
        # samples. Averaging the linear extinction quantity is both more stable
        # and less prone to light leaks through thin features.
        nz, ny, nx = sigma_full.shape
        pz = (-nz) % factor
        py = (-ny) % factor
        px = (-nx) % factor
        padded = np.pad(
            sigma_full,
            ((0, pz), (0, py), (0, px)),
            mode="edge",
        )
        oz, oy, ox = (v // factor for v in padded.shape)
        reduced = padded.reshape(oz, factor, oy, factor, ox, factor).mean(
            axis=(1, 3, 5), dtype=np.float32
        )
        return np.ascontiguousarray(reduced, dtype=np.float32)

    def _build_environment_shadow_array(
        self,
        volume: MitsubaRepresentationHandle,
        direction: tuple[float, float, float],
        sigma: np.ndarray,
    ) -> np.ndarray | None:
        """Bake cumulative optical depth toward one arbitrary direction."""
        bounds = volume.bounds
        if bounds is None or sigma is None or sigma.ndim != 3:
            return None

        nz, ny, nx = sigma.shape
        sx = (float(bounds[1]) - float(bounds[0])) / max(nx - 1, 1)
        sy = (float(bounds[3]) - float(bounds[2])) / max(ny - 1, 1)
        sz = (float(bounds[5]) - float(bounds[4])) / max(nz - 1, 1)
        axis_spacing = np.asarray((sz, sy, sx), dtype=np.float64)
        d_xyz = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(d_xyz))
        if norm <= 1.0e-12:
            return np.zeros_like(sigma, dtype=np.float32)
        d_xyz /= norm
        d_axis = np.asarray((d_xyz[2], d_xyz[1], d_xyz[0]), dtype=np.float64)
        dominant = int(np.argmax(np.abs(d_axis)))
        dom = float(d_axis[dominant])
        if abs(dom) <= 1.0e-8:
            return np.zeros_like(sigma, dtype=np.float32)

        perm = (dominant,) + tuple(axis for axis in range(3) if axis != dominant)
        inv_perm = tuple(np.argsort(perm))
        sigma_p = np.transpose(sigma, perm)
        spacing_p = axis_spacing[list(perm)]
        direction_p = d_axis[list(perm)]

        step_length = float(spacing_p[0] / abs(direction_p[0]))
        offset0 = float(direction_p[1] * step_length / spacing_p[1])
        offset1 = float(direction_p[2] * step_length / spacing_p[2])
        tau_p = np.zeros_like(sigma_p, dtype=np.float32)

        if direction_p[0] > 0.0:
            indices = range(sigma_p.shape[0] - 1, -1, -1)
            def next_index(i):
                return i + 1
        else:
            indices = range(0, sigma_p.shape[0])
            def next_index(i):
                return i - 1

        for i in indices:
            j = next_index(i)
            if 0 <= j < sigma_p.shape[0]:
                continuation = self._bilinear_shift(tau_p[j], offset0, offset1)
            else:
                continuation = 0.0
            tau_p[i] = sigma_p[i] * step_length + continuation

        return np.ascontiguousarray(np.transpose(tau_p, inv_perm), dtype=np.float32)

    def _build_environment_lighting_field(
        self, volume: MitsubaRepresentationHandle
    ) -> None:
        """Bake view-independent environment visibility into L2 SH volumes.

        The expensive angular visibility integration is independent of the camera.
        Camera-dependent Henyey-Greenstein weighting is applied later in the DVR
        kernel by convolving the stored SH coefficients with the phase function.
        """
        if volume.environment_scattering_samples <= 0 or not volume.shade:
            volume.environment_lighting_volumes = ()
            volume.environment_lighting_key = None
            return
        mapping = volume.opacity_mapping
        if mapping is None or volume.bounds is None or volume.scalar_values is None:
            return

        opacity_signature = (
            tuple(float(v) for v in mapping["range"]),
            tuple(tuple(float(c) for c in point) for point in mapping["control_points"]),
        )
        direction_count = max(0, int(volume.environment_scattering_samples))
        resolution_factor = max(1, int(round(volume.environment_scattering_step_factor)))
        key = (
            volume.scalar_volume_key,
            opacity_signature,
            float(volume.sample_distance),
            float(volume.global_illumination_reach),
            direction_count,
            resolution_factor,
            tuple(float(v) for v in volume.bounds),
            2,
        )
        if volume.environment_lighting_key == key and volume.environment_lighting_volumes:
            return

        start = time.perf_counter()
        print(
            f"Mitsuba bake environment lighting: start shape={tuple(int(v) for v in volume.scalar_values.shape)} "
            f"directions={direction_count} resolution_factor={resolution_factor} dtype=float32"
        )
        directions = _environment_directions(direction_count)
        sigma = self._build_environment_extinction_array(volume)
        if sigma is None:
            volume.environment_lighting_volumes = ()
            volume.environment_lighting_key = None
            elapsed = time.perf_counter() - start
            print(f"Mitsuba bake environment lighting: end {elapsed:.3f}s no-extinction")
            return
        coeffs = None
        weight = 4.0 * math.pi / float(direction_count)
        reach = self._shadow_extent_for_volume(volume)

        for direction in directions:
            tau = self._build_environment_shadow_array(volume, direction, sigma)
            if tau is None:
                continue
            nz, ny, nx = tau.shape
            sx = (float(volume.bounds[1]) - float(volume.bounds[0])) / max(nx - 1, 1)
            sy = (float(volume.bounds[3]) - float(volume.bounds[2])) / max(ny - 1, 1)
            sz = (float(volume.bounds[5]) - float(volume.bounds[4])) / max(nz - 1, 1)
            offsets = (
                float(direction[2]) * reach / max(sz, 1.0e-12),
                float(direction[1]) * reach / max(sy, 1.0e-12),
                float(direction[0]) * reach / max(sx, 1.0e-12),
            )
            tau_end = self._shift_volume_linear(tau, offsets)
            visibility = np.exp(-np.maximum(tau - tau_end, 0.0)).astype(np.float32)
            basis = self._real_sh_basis(direction)
            if coeffs is None:
                coeffs = [np.zeros_like(visibility, dtype=np.float32) for _ in basis]
            for index, value in enumerate(basis):
                coeffs[index] += np.float32(weight * value) * visibility

        if not coeffs:
            volume.environment_lighting_volumes = ()
            volume.environment_lighting_key = None
            elapsed = time.perf_counter() - start
            print(f"Mitsuba bake environment lighting: end {elapsed:.3f}s no-data")
            return

        # Pack the nine L2 SH coefficients into one 6-channel and one
        # 3-channel volume. Mitsuba can then fetch all coefficients with just
        # eval_6() + eval_3() instead of nine scalar texture lookups.
        packed6 = np.ascontiguousarray(np.stack(coeffs[:6], axis=-1), dtype=np.float32)
        packed3 = np.ascontiguousarray(np.stack(coeffs[6:9], axis=-1), dtype=np.float32)
        built = []
        with self._render_lock:
            transform = _image_volume_transform(self.mi, volume.bounds)
            for packed in (packed6, packed3):
                built.append(
                    self.mi.load_dict(
                        {
                            "type": "gridvolume",
                            "data": self.mi.TensorXf(packed),
                            "raw": True,
                            "filter_type": "trilinear",
                            "wrap_mode": "clamp",
                            "to_world": transform,
                        }
                    )
                )
        volume.environment_lighting_volumes = tuple(built)
        volume.environment_lighting_key = key
        elapsed = time.perf_counter() - start
        metadata = []
        for index, field in enumerate(built):
            try:
                metadata.append(
                    f"field{index}:channels={int(field.channel_count())},resolution={tuple(int(v) for v in field.resolution())}"
                )
            except Exception as exc:
                metadata.append(f"field{index}:metadata-error={exc}")
        print(
            f"Mitsuba bake environment lighting: end {elapsed:.3f}s " + " ".join(metadata),
            flush=True,
        )

    def _ensure_cached_lighting(
        self, volumes: tuple[MitsubaRepresentationHandle, ...]
    ) -> None:
        """Build only the TF-dependent lighting fields required by cached mode."""
        for volume in volumes:
            if not volume.shade or volume.volumetric_scattering_blending <= 0.0:
                continue
            self._build_directional_shadow_field(volume)
            if volume.environment_scattering_samples > 0:
                self._build_environment_lighting_field(volume)
            else:
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

        if snapshot["caching"]:
            self._ensure_cached_lighting(snapshot["volumes"])

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

            # Cached and live volume paths are separate specialized DirectVolume
            # types. The integrators themselves are rebuilt only for structural
            # changes. Transfer functions and ordinary shading parameters are
            # refreshed in-place on the existing volume objects.
            caching = bool(snapshot["caching"])
            direct_volume_type = (
                self._cached_direct_volume_type if caching else self._live_direct_volume_type
            )
            structural_volume_key = tuple(
                (
                    id(volume.scalar_volume),
                    tuple(float(v) for v in volume.bounds),
                    id(volume.gradient_volume),
                    bool(volume.shade),
                    int(volume.environment_scattering_samples),
                    # These values select Python-time branches in DirectVolume.
                    bool(float(volume.volumetric_scattering_blending) > 0.0),
                    bool(float(volume.volumetric_scattering_blending) < 1.0),
                    bool(abs(float(volume.scattering_anisotropy)) < 0.01),
                )
                for volume in snapshot["volumes"]
            )
            integrator_key = (
                structural_volume_key,
                bool(snapshot["objects"]),
            )
            integrator = handle.cached_integrator if caching else handle.live_integrator
            previous_key = (
                handle.cached_integrator_key if caching else handle.live_integrator_key
            )
            ambient_light = tuple(
                component * snapshot["world_ambient_intensity"]
                for component in self._srgb_to_linear(snapshot["world_ambient_color"])
            )
            if integrator is None or previous_key != integrator_key:
                mode = "cached" if caching else "live"
                print(
                    f"[Mitsuba] {mode} integrator rebuild: view={view_id} "
                    f"volumes={len(snapshot['volumes'])} surfaces={bool(snapshot['objects'])} "
                    f"env_samples={[int(v.environment_scattering_samples) for v in snapshot['volumes']]}",
                    flush=True,
                )
                direct_volumes = tuple(
                    direct_volume_type(
                        volume.scalar_volume,
                        volume.bounds,
                        volume.color_mapping,
                        volume.opacity_mapping,
                        volume.sample_distance,
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
                        volume.shadow_volume if caching else None,
                        volume.environment_scattering_strength,
                        volume.environment_scattering_samples,
                        volume.environment_scattering_step_factor,
                        volume.environment_lighting_volumes if caching else (),
                    )
                    for volume in snapshot["volumes"]
                )
                integrator = self._dvr_integrator_type(
                    direct_volumes,
                    self._srgb_to_linear(snapshot["background_color"]),
                    bool(snapshot["objects"]),
                )
                if caching:
                    handle.cached_integrator = integrator
                    handle.cached_integrator_key = integrator_key
                else:
                    handle.live_integrator = integrator
                    handle.live_integrator_key = integrator_key
            else:
                integrator.update(
                    snapshot["volumes"],
                    background_color=self._srgb_to_linear(snapshot["background_color"]),
                    ambient_light=ambient_light,
                    cached_lighting=caching,
                )

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



def _evaluate_opacity_mapping(
    mapping: dict[str, Any], values: np.ndarray
) -> np.ndarray:
    """Evaluate normalized piecewise-linear opacity as float32."""
    points = np.asarray(mapping["control_points"], dtype=np.float32)
    minimum, maximum = (np.float32(v) for v in mapping["range"])
    width = np.float32(max(float(maximum - minimum), 1.0e-20))
    normalized = np.clip(
        (np.asarray(values, dtype=np.float32) - minimum) / width,
        np.float32(0.0),
        np.float32(1.0),
    )
    result = np.interp(
        normalized,
        points[:, 0],
        np.clip(points[:, 1], np.float32(0.0), np.float32(1.0)),
    )
    return np.asarray(result, dtype=np.float32)

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


def _make_direct_volume_type(mi, dr, *, cached_lighting: bool):
    """Create one specialized DVR volume type.

    ``cached_lighting`` is a Python-time specialization constant, so cached and
    live lighting paths produce separate Dr.Jit kernels without a per-sample
    runtime branch. Both paths share scalar, gradient, TF, and local shading
    code.
    """

    class DirectVolume:
        def __init__(
            self,
            scalar_volume,
            bounds,
            color_mapping: dict[str, Any],
            opacity_mapping: dict[str, Any],
            sample_distance: float,
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

        def update_runtime(self, volume, *, ambient_light, cached_lighting: bool) -> None:
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
            if cached_lighting:
                self.shadow_volume = volume.shadow_volume
                self.environment_lighting_volumes = tuple(
                    volume.environment_lighting_volumes or ()
                )
            else:
                self.shadow_volume = None
                self.environment_lighting_volumes = ()

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

        def _sample_scalar(self, position, active=True):
            it = dr.zeros(mi.Interaction3f)
            it.p = position
            return self.scalar_volume.eval_1(it, active)

        def _gradient(self, position, active=True):
            if self.gradient_volume is not None:
                it = dr.zeros(mi.Interaction3f)
                it.p = position
                return self.gradient_volume.eval_3(it, active)
            # Defensive fallback for malformed/unsupported inputs. Normal volume
            # rendering should use the cached float32 gradient field above.
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
            min_extent = min(self.sample_distance, max_extent)
            reach = self.global_illumination_reach
            # VTK maps [0, 1] non-linearly from roughly one primary sample
            # to the full volume diagonal. Preserve that behavior in world space.
            return (min_extent - max_extent) * ((1.0 - reach) ** 0.33) + max_extent

        def _opacity_at(self, position, step_distance, active=True):
            scalar = self._sample_scalar(position, active)
            opacity_x = self._normalized_scalar(scalar, self.opacity)
            alpha = self._opacity(opacity_x, self.opacity, active)
            ratio = step_distance / self.sample_distance
            transmission = dr.maximum(1.0 - alpha, 1.0e-6)
            return dr.clip(1.0 - dr.power(transmission, ratio), 0.0, 1.0)

        def _sample_shadow_tau(self, position, active=True):
            if self.shadow_volume is None:
                return mi.Float(0.0)
            it = dr.zeros(mi.Interaction3f)
            it.p = position
            return self.shadow_volume.eval_1(it, active)

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
            if cached_lighting:
                if self.shadow_volume is None:
                    return mi.Float(1.0)

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

            if cached_lighting:
                if not self.environment_lighting_volumes:
                    return mi.Color3f(0.0)

                # Incoming directions were projected into real SH coefficients in
                # volume space. Camera-dependent Henyey-Greenstein weighting is
                # applied here without rebaking the visibility field.
                query_direction = -dr.normalize(view)
                basis = self._sh_basis(query_direction)
                g = mi.Float(self.scattering_anisotropy)
                if len(self.environment_lighting_volumes) != 2:
                    return mi.Color3f(0.0)
                it = dr.zeros(mi.Interaction3f)
                it.p = position
                c0 = self.environment_lighting_volumes[0].eval_6(it, active)
                c1 = self.environment_lighting_volumes[1].eval_3(it, active)
                g2 = g * g
                total = (
                    c0[0] * basis[0]
                    + g * (c0[1] * basis[1] + c0[2] * basis[2] + c0[3] * basis[3])
                    + g2 * (
                        c0[4] * basis[4]
                        + c0[5] * basis[5]
                        + c1.x * basis[6]
                        + c1.y * basis[7]
                        + c1.z * basis[8]
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
            ratio = step_distance / self.sample_distance
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

            # TF opacity is defined for the representation's reference sample
            # distance. Correct it if the marcher later uses a different step.
            ratio = step_distance / self.sample_distance
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

        def update(self, volumes, *, background_color, ambient_light, cached_lighting):
            self.background_color = mi.Color3f(*map(float, background_color))
            for direct_volume, volume in zip(self.volumes, volumes):
                direct_volume.update_runtime(
                    volume,
                    ambient_light=ambient_light,
                    cached_lighting=cached_lighting,
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

    values = np.asarray(vtk_to_numpy(array), dtype=np.float32)
    components = int(array.GetNumberOfComponents())
    if components <= 1:
        scalars = values.reshape(-1)
    else:
        scalars = np.linalg.norm(values.reshape(-1, components), axis=1)

    nx, ny, nz = grid_dimensions
    return np.ascontiguousarray(scalars.reshape(nz, ny, nx), dtype=np.float32)


def _image_volume_transform(mi, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = map(float, bounds)
    extent = [xmax - xmin, ymax - ymin, zmax - zmin]
    if any(value <= 0.0 for value in extent):
        raise ValueError("Mitsuba volume bounds must have positive extent on all axes")
    return mi.ScalarTransform4f().translate([xmin, ymin, zmin]).scale(extent)




def _volume_gradient_step(image: vtk.vtkImageData) -> tuple[float, float, float]:
    spacing = tuple(abs(float(value)) for value in image.GetSpacing())
    fallback = _volume_sample_distance(image, {"auto_adjust_sample_distances": True})
    return tuple(value if value > 0.0 else fallback for value in spacing)

def _volume_sample_distance(
    image: vtk.vtkImageData,
    properties: dict[str, Any],
) -> float:
    if properties.get("auto_adjust_sample_distances", True):
        spacing = [abs(float(value)) for value in image.GetSpacing() if abs(float(value)) > 0]
        if spacing:
            return min(spacing)
    return max(float(properties.get("sample_distance", 1.0)), 1.0e-12)
