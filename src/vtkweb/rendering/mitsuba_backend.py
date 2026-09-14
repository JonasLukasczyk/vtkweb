from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from vtkweb.rendering.base import RenderView, RenderingBackend, Representation


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
    center_of_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accumulation: np.ndarray | None = None
    accumulated_spp: int = 0
    next_seed: int = 1
    render_revision: int = 0
    accumulation_key: tuple | None = None
    cached_scene: Any | None = None
    cached_scene_key: tuple | None = None


@dataclass
class MitsubaRepresentationHandle:
    kind: str
    mesh: Any | None = None
    bounds: tuple[float, float, float, float, float, float] | None = None


class MitsubaRenderingBackend(RenderingBackend):
    @staticmethod
    def _srgb_to_linear(
        color: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Convert normalized display/sRGB values to linear-light RGB."""

        def convert(channel: float) -> float:
            value = max(0.0, min(1.0, float(channel)))
            if value <= 0.04045:
                return value / 12.92
            return ((value + 0.055) / 1.055) ** 2.4

        return tuple(convert(channel) for channel in color)

    """Minimal server-side Mitsuba backend.

    Surface representations are converted to Mitsuba triangle meshes.
    Wireframe and outline representations are generated with VTK as world-space
    tubes, then converted to the same Mitsuba triangle-mesh representation.
    No coordinate normalization, centering, scaling, or axis conversion is
    performed.
    """

    name = "mitsuba"

    def __init__(self, transfer_function_provider=None) -> None:
        import mitsuba as mi

        self._transfer_function_provider = transfer_function_provider or (
            lambda _name: None
        )
        self.mi = mi
        if mi.variant() != "cuda_ad_rgb":
            mi.set_variant("cuda_ad_rgb")

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
                    current_view_id == view_id and rep.mesh is not None
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
        handle.cached_scene = None
        handle.cached_scene_key = None

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
                },
                "width": int(handle.width),
                "height": int(handle.height),
                "background_color": tuple(handle.background_color),
                "world_ambient_color": tuple(handle.world_ambient_color),
                "world_ambient_intensity": float(handle.world_ambient_intensity),
                # Keep strong references to exactly the meshes represented by
                # this revision even if the server thread replaces them later.
                "shapes": tuple(
                    (representation_id, rep.mesh)
                    for (
                        representation_id,
                        current_view_id,
                    ), rep in self._representations.items()
                    if current_view_id == view_id and rep.mesh is not None
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
        rep_handle = self._create_handle(representation, source)
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
    ) -> MitsubaRepresentationHandle:
        if representation.kind == "surface":
            return self._create_surface_handle(representation, source)
        if representation.kind == "wireframe":
            return self._create_wireframe_handle(representation, source)
        if representation.kind == "outline":
            return self._create_outline_handle(representation, source)

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
            mesh=mesh,
            bounds=tuple(float(v) for v in bounds),
        )

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
    ) -> np.ndarray:
        """Render one progressive sample for exactly one image-space tile."""
        handle = self._views[view_id]
        spp = max(1, int(spp))
        camera = snapshot["camera"]
        x, y, width, height = map(int, region)
        full_width, full_height = map(int, full_size)

        scene_dict: dict[str, Any] = {
            "type": "scene",
            "integrator": {
                "type": "path",
                "max_depth": 4,
                "hide_emitters": True,
            },
            "sensor": {
                "type": "perspective",
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
                    "pixel_format": "rgba",
                },
                "sampler": {"type": "independent", "sample_count": spp},
            },
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

        for shape_index, (representation_id, mesh) in enumerate(snapshot["shapes"]):
            scene_dict[f"shape_{shape_index}_{representation_id}"] = mesh

        # Scene creation and render/materialization all touch Dr.Jit runtime
        # state. Keep them serialized across Mitsuba views within this rank.
        with self._render_lock:
            if handle.cached_scene is None or handle.cached_scene_key != render_key:
                handle.cached_scene = self.mi.load_dict(scene_dict)
                handle.cached_scene_key = render_key
            scene = handle.cached_scene

            seed = handle.next_seed
            handle.next_seed += 1
            image = self.mi.render(scene, spp=spp, seed=seed)
            rendered = np.array(
                self.mi.Bitmap(image),
                dtype=np.float32,
                copy=True,
            )

        if rendered.shape[-1] >= 4:
            rgb = rendered[..., :3]
            alpha = np.clip(rendered[..., 3:4], 0.0, 1.0)
            background = np.asarray(
                self._srgb_to_linear(snapshot["background_color"]),
                dtype=np.float32,
            ).reshape((1, 1, 3))
            return rgb * alpha + background * (1.0 - alpha)

        return rendered[..., :3]

    def _accumulate_pass_worker(
        self,
        handle: MitsubaViewHandle,
        sample: np.ndarray,
        *,
        spp: int,
        render_key: tuple,
    ) -> None:
        """Accumulate one tile sample. Called only by the render worker."""
        if handle.accumulation_key != render_key:
            self._clear_accumulation_worker(handle, render_key)
        if handle.accumulation is None or handle.accumulation.shape != sample.shape:
            handle.accumulation = np.zeros_like(sample, dtype=np.float32)
            handle.accumulated_spp = 0
            handle.accumulation_key = render_key
        handle.accumulation += sample * float(spp)
        handle.accumulated_spp += int(spp)

    def encoded_frame(self, image: np.ndarray) -> bytes:
        """Encode a linear RGB image to JPEG entirely in memory.

        Mitsuba performs the linear->sRGB/UInt8 conversion. VTK's JPEG writer
        is then used in WriteToMemory mode, avoiding all temporary files and
        introducing no additional image-encoding dependency.
        """
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

        # VTK image coordinates have +Y upward whereas NumPy/Mitsuba image rows
        # are top-to-bottom. Flip while importing so the encoded JPEG preserves
        # the browser/image-space orientation.
        vtk_pixels = np.ascontiguousarray(np.flipud(rgb).reshape(-1, 3))
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(width, height, 1)
        vtk_array = numpy_to_vtk(
            vtk_pixels,
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
        vtk_array.SetNumberOfComponents(3)
        vtk_image.GetPointData().SetScalars(vtk_array)

        writer = vtk.vtkJPEGWriter()
        writer.SetInputData(vtk_image)
        writer.SetQuality(90)
        writer.WriteToMemoryOn()
        writer.Write()
        result = writer.GetResult()
        if result is None or result.GetNumberOfValues() == 0:
            raise RuntimeError("VTK JPEG writer produced an empty in-memory result")
        return bytes(memoryview(result))

    def _encoded_accumulated_frame_worker(
        self,
        handle: MitsubaViewHandle,
    ) -> bytes | None:
        if handle.accumulation is None or handle.accumulated_spp <= 0:
            return None
        averaged = handle.accumulation / float(handle.accumulated_spp)
        return self.encoded_frame(averaged)

    def render_frame(
        self, view_id: str, *, region=None, full_size=None
    ) -> bytes | None:
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

        # Progressive state is specific not only to scene/camera revision but
        # also to the crop window. This matters when toggling distributed mode
        # without changing the logical view itself.
        render_key = (revision, full_size, region)
        if handle.accumulation_key != render_key:
            self._clear_accumulation_worker(handle, render_key)

        sample = self.render_pass(
            view_id,
            snapshot,
            render_key,
            region=region,
            full_size=full_size,
            spp=1,
        )
        self._accumulate_pass_worker(
            handle,
            sample,
            spp=1,
            render_key=render_key,
        )
        return self._encoded_accumulated_frame_worker(handle)

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
    """Map scalar values to RGB using scalar-space global TF control points."""

    control_points = transfer_function.get("control_points") or []
    if not control_points:
        return np.zeros((len(values), 3), dtype=np.float32)

    points = np.asarray(control_points, dtype=np.float64)
    order = np.argsort(points[:, 0], kind="stable")
    points = points[order]

    # Control-point positions are stored directly in scalar data space.
    # np.interp clamps values outside the first/last point to the endpoints.
    scalars = np.asarray(values, dtype=np.float64)
    positions = points[:, 0]
    rgb = np.column_stack(
        [
            np.interp(scalars, positions, np.clip(points[:, channel], 0.0, 1.0))
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
