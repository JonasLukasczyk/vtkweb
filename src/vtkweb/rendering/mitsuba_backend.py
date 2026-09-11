from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

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
    scene_generation: int = 0
    accumulation_generation: int = -1


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

        self._views: dict[str, MitsubaViewHandle] = {}
        self._representations: dict[tuple[str, str], MitsubaRepresentationHandle] = {}

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def add_view(self, view: RenderView) -> None:
        self._views[view.id] = MitsubaViewHandle()

    def remove_view(self, view_id: str) -> None:
        for key in [key for key in self._representations if key[1] == view_id]:
            self._representations.pop(key, None)
        self._views.pop(view_id, None)

    def rename_view(self, view_id: str, new_view_id: str) -> None:
        if new_view_id != view_id and new_view_id in self._views:
            raise ValueError(f"View ID already exists: {new_view_id}")

        self._views[new_view_id] = self._views.pop(view_id)
        renamed = {}
        for (
            representation_id,
            current_view_id,
        ), handle in self._representations.items():
            renamed[
                (
                    representation_id,
                    new_view_id if current_view_id == view_id else current_view_id,
                )
            ] = handle
        self._representations = renamed

    def set_view_property(self, view_id: str, name: str, value: Any) -> None:
        if name not in {
            "background_color",
            "world_ambient_color",
            "world_ambient_intensity",
        }:
            return
        setattr(self._views[view_id], name, value)
        self.invalidate_scene(view_id)

    def set_render_size(self, view_id: str, width: int, height: int) -> bool:
        """Update transient film dimensions and invalidate the scene."""
        handle = self._views[view_id]
        width = max(1, int(width))
        height = max(1, int(height))
        if handle.width == width and handle.height == height:
            return False

        handle.width = width
        handle.height = height
        self.invalidate_scene(view_id)
        return True

    def invalidate_scene(self, view_id: str) -> int:
        """Advance the render generation and clear progressive accumulation."""
        handle = self._views[view_id]
        handle.scene_generation += 1
        self.clear_accumulation(view_id, handle.scene_generation)
        return handle.scene_generation

    def reset_camera(self, view_id: str) -> None:
        print(f"[camera] Mitsuba reset_camera(view_id={view_id})", flush=True)
        bounds = self._visible_bounds(view_id)
        handle = self._views[view_id]

        if bounds is None:
            handle.camera_origin = (0.0, 0.0, 5.0)
            handle.camera_target = (0.0, 0.0, 0.0)
            handle.camera_up = (0.0, 1.0, 0.0)
            handle.center_of_rotation = (0.0, 0.0, 0.0)
            self.invalidate_scene(view_id)
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
        diagonal = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.float64)
        radius = max(0.5 * float(np.linalg.norm(diagonal)), 1.0e-6)

        half_fov = math.radians(handle.camera_fov) * 0.5
        distance = 1.15 * radius / math.sin(half_fov)

        origin = center + np.array([0.0, 0.0, distance])
        handle.camera_origin = tuple(float(v) for v in origin)
        handle.camera_target = tuple(float(v) for v in center)
        handle.camera_up = (0.0, 1.0, 0.0)
        handle.center_of_rotation = tuple(float(v) for v in center)
        self.invalidate_scene(view_id)

    def get_camera_state(self, view_id: str) -> dict[str, Any]:
        handle = self._views[view_id]
        return {
            "position": list(handle.camera_origin),
            "target": list(handle.camera_target),
            "up": list(handle.camera_up),
            "center_of_rotation": list(handle.center_of_rotation),
            "fov": float(handle.camera_fov),
        }

    def set_camera_state(self, view_id: str, value: dict[str, Any]) -> None:
        handle = self._views[view_id]
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
        self.invalidate_scene(view_id)

    def interact_camera(
        self,
        view_id: str,
        mode: str,
        dx: float,
        dy: float,
        viewport_height: float,
    ) -> int:
        """Apply an orbit, pan, or dolly delta to the runtime camera."""
        handle = self._views[view_id]
        dx = float(dx)
        dy = float(dy)

        position = np.asarray(handle.camera_origin, dtype=np.float64)
        target = np.asarray(handle.camera_target, dtype=np.float64)
        up = _normalized(np.asarray(handle.camera_up, dtype=np.float64))
        center = np.asarray(handle.center_of_rotation, dtype=np.float64)

        if mode == "orbit":
            offset = position - center
            radians_per_pixel = math.radians(0.35)
            offset = _rotate_vector(offset, up, -dx * radians_per_pixel)
            forward = _normalized(-offset)
            right = np.cross(forward, up)
            if np.linalg.norm(right) > 1.0e-12:
                right = _normalized(right)
                pitch = -dy * radians_per_pixel
                offset = _rotate_vector(offset, right, pitch)
                up = _normalized(_rotate_vector(up, right, pitch))
            position = center + offset
            target = center.copy()

        elif mode == "pan":
            forward_vector = target - position
            distance = max(float(np.linalg.norm(forward_vector)), 1.0e-9)
            forward = _normalized(forward_vector)
            right = np.cross(forward, up)
            if np.linalg.norm(right) > 1.0e-12:
                right = _normalized(right)
                screen_up = _normalized(np.cross(right, forward))
                height = max(float(viewport_height), 1.0)
                world_per_pixel = (
                    2.0
                    * distance
                    * math.tan(0.5 * math.radians(handle.camera_fov))
                    / height
                )
                shift = (-dx * right + dy * screen_up) * world_per_pixel
                position += shift
                target += shift
                center += shift
                up = screen_up

        elif mode == "zoom":
            offset = position - target
            distance = float(np.linalg.norm(offset))
            if distance > 1.0e-12:
                factor = math.exp(max(-4.0, min(4.0, dy * 0.01)))
                next_distance = max(1.0e-9, min(1.0e12, distance * factor))
                position = target + offset * (next_distance / distance)
        else:
            raise ValueError(f"Unknown camera interaction mode: {mode}")

        handle.camera_origin = tuple(float(v) for v in position)
        handle.camera_target = tuple(float(v) for v in target)
        handle.camera_up = tuple(float(v) for v in up)
        handle.center_of_rotation = tuple(float(v) for v in center)
        return self.invalidate_scene(view_id)

    def render_snapshot(self, view_id: str) -> tuple[int, dict[str, Any]]:
        handle = self._views[view_id]
        camera = self.get_camera_state(view_id)
        return handle.scene_generation, camera

    def has_renderable_scene(self, view_id: str) -> bool:
        handle = self._views[view_id]
        return (
            handle.width > 0
            and handle.height > 0
            and any(
                current_view_id == view_id and rep.mesh is not None
                for (_, current_view_id), rep in self._representations.items()
            )
        )

    def clear_accumulation(self, view_id: str, generation: int | None = None) -> None:
        handle = self._views[view_id]
        handle.accumulation = None
        handle.accumulated_spp = 0
        handle.next_seed = 1
        handle.accumulation_generation = (
            handle.scene_generation if generation is None else int(generation)
        )

    def accumulation_generation(self, view_id: str) -> int:
        return self._views[view_id].accumulation_generation

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
        if key in self._representations:
            return
        self._representations[key] = self._create_handle(representation, source)
        self.invalidate_scene(view.id)

    def update_representation(
        self,
        representation: Representation,
        view: RenderView,
        source: Any,
    ) -> None:
        self._representations[(representation.id, view.id)] = self._create_handle(
            representation, source
        )
        self.invalidate_scene(view.id)

    def remove_representation(self, representation_id: str, view_id: str) -> None:
        self._representations.pop((representation_id, view_id), None)
        if view_id in self._views:
            self.invalidate_scene(view_id)

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
                color = _hex_to_rgb(representation.properties.get("color", "#d9d9d9"))
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
        camera: dict[str, Any],
        *,
        spp: int = 1,
    ) -> np.ndarray:
        handle = self._views[view_id]
        spp = max(1, int(spp))

        scene_dict: dict[str, Any] = {
            "type": "scene",
            # The environment is used only for illumination.  The requested
            # view background is composited after rendering so changing a UI
            # background color does not also change scene exposure.
            "integrator": {
                "type": "path",
                "max_depth": 4,
                "hide_emitters": True,
            },
            "sensor": {
                "type": "perspective",
                "fov": float(camera.get("fov", handle.camera_fov)),
                "fov_axis": "y",
                "to_world": self.mi.ScalarTransform4f().look_at(
                    origin=camera["position"],
                    target=camera["target"],
                    up=camera["up"],
                ),
                "film": {
                    "type": "hdrfilm",
                    "width": handle.width,
                    "height": handle.height,
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
                            self._srgb_to_linear(handle.world_ambient_color),
                            dtype=np.float32,
                        )
                        * float(handle.world_ambient_intensity)
                    ).tolist(),
                },
            },
        }

        shape_index = 0
        for (representation_id, current_view_id), rep in self._representations.items():
            if current_view_id != view_id:
                continue
            if rep.mesh is not None:
                scene_dict[f"shape_{shape_index}_{representation_id}"] = rep.mesh
                shape_index += 1

        scene = self.mi.load_dict(scene_dict)
        seed = handle.next_seed
        handle.next_seed += 1
        image = self.mi.render(scene, spp=spp, seed=seed)
        rendered = np.array(self.mi.Bitmap(image), dtype=np.float32, copy=True)

        if rendered.shape[-1] >= 4:
            rgb = rendered[..., :3]
            alpha = np.clip(rendered[..., 3:4], 0.0, 1.0)
            background = np.asarray(
                self._srgb_to_linear(handle.background_color), dtype=np.float32
            ).reshape((1, 1, 3))
            return rgb * alpha + background * (1.0 - alpha)

        # Fallback for Mitsuba configurations that do not expose alpha even
        # when an RGBA film is requested.
        return rendered[..., :3]

    def accumulate_pass(
        self,
        view_id: str,
        sample: np.ndarray,
        *,
        spp: int,
        generation: int,
    ) -> None:
        handle = self._views[view_id]
        if handle.accumulation_generation != generation:
            self.clear_accumulation(view_id, generation)
        if handle.accumulation is None or handle.accumulation.shape != sample.shape:
            handle.accumulation = np.zeros_like(sample, dtype=np.float32)
            handle.accumulated_spp = 0
            handle.accumulation_generation = generation
        handle.accumulation += sample * float(spp)
        handle.accumulated_spp += int(spp)

    def encoded_frame(self, image: np.ndarray) -> bytes:
        """Encode one linear RGB image as JPEG bytes for binary transport."""
        bitmap = self.mi.Bitmap(image).convert(
            self.mi.Bitmap.PixelFormat.RGB,
            self.mi.Struct.Type.UInt8,
            True,
        )
        filename = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as stream:
                filename = stream.name
            bitmap.write(filename)
            with open(filename, "rb") as stream:
                payload = stream.read()
        finally:
            if filename is not None:
                try:
                    os.unlink(filename)
                except FileNotFoundError:
                    pass
        return payload

    def encoded_accumulated_frame(self, view_id: str) -> bytes | None:
        handle = self._views[view_id]
        if handle.accumulation is None or handle.accumulated_spp <= 0:
            return None
        averaged = handle.accumulation / float(handle.accumulated_spp)
        return self.encoded_frame(averaged)

    def render_frame(self, view_id: str, *, spp: int = 1) -> bytes:
        generation, camera = self.render_snapshot(view_id)
        if self.accumulation_generation(view_id) != generation:
            self.clear_accumulation(view_id, generation)
        sample = self.render_pass(view_id, camera, spp=spp)
        self.accumulate_pass(view_id, sample, spp=spp, generation=generation)
        return self.encoded_accumulated_frame(view_id) or b""

    def get_accumulated_spp(self, view_id: str) -> int:
        return self._views[view_id].accumulated_spp

    def _visible_bounds(
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


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        return vector.copy()
    return vector / norm


def _rotate_vector(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalized(axis)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
    