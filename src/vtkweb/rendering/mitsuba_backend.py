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
    background_color: tuple[float, float, float]
    world_ambient_color: tuple[float, float, float]
    world_ambient_intensity: float
    width: int = 1024
    height: int = 768
    camera_origin: tuple[float, float, float] = (0.0, 0.0, 5.0)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    center_of_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accumulation: np.ndarray | None = None
    accumulated_spp: int = 0
    next_seed: int = 1
    camera_revision: int = 0
    accumulation_revision: int = -1


@dataclass
class MitsubaRepresentationHandle:
    kind: str
    mesh: Any | None = None
    bounds: tuple[float, float, float, float, float, float] | None = None


class MitsubaRenderingBackend(RenderingBackend):
    """Minimal server-side Mitsuba backend.

    This first implementation deliberately supports only surface
    representations. VTK world-space coordinates are copied directly into the
    Mitsuba mesh without normalization, centering, scaling, or axis changes.
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
        self._views[view.id] = MitsubaViewHandle(
            background_color=view.settings.background_color,
            world_ambient_color=view.settings.world_ambient_color,
            world_ambient_intensity=view.settings.world_ambient_intensity,
        )

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

    def set_view_settings(self, view: RenderView) -> None:
        handle = self._views[view.id]
        handle.background_color = view.settings.background_color
        handle.world_ambient_color = view.settings.world_ambient_color
        handle.world_ambient_intensity = view.settings.world_ambient_intensity
        self.invalidate_accumulation(view.id)

    def set_render_size(self, view_id: str, width: int, height: int) -> bool:
        """Set the film size for a Mitsuba view without touching app state.

        Returns True when the size changed.  Size changes advance the same
        render revision used by camera interaction so an in-flight pass from
        the old resolution can never be accumulated into the new framebuffer.
        """
        handle = self._views[view_id]
        width = max(1, int(width))
        height = max(1, int(height))
        if handle.width == width and handle.height == height:
            return False

        handle.width = width
        handle.height = height
        handle.camera_revision += 1
        self.reset_accumulation(view_id)
        return True

    def reset_accumulation(self, view_id: str) -> None:
        handle = self._views[view_id]
        handle.accumulation = None
        handle.accumulated_spp = 0
        handle.next_seed = 1
        handle.accumulation_revision = handle.camera_revision

    def invalidate_accumulation(self, view_id: str) -> None:
        """Invalidate progressive rendering after a scene-level change.

        ``camera_revision`` is the backend's render-generation token.  It must
        advance for any change that can alter a rendered sample, not only for
        camera motion.  Otherwise an in-flight pass from the previous scene can
        be accepted into a freshly cleared accumulation buffer.
        """
        handle = self._views[view_id]
        handle.camera_revision += 1
        self.reset_accumulation(view_id)

    def reset_camera(self, view_id: str) -> None:
        bounds = self._visible_bounds(view_id)
        handle = self._views[view_id]

        if bounds is None:
            handle.camera_origin = (0.0, 0.0, 5.0)
            handle.camera_target = (0.0, 0.0, 0.0)
            handle.camera_up = (0.0, 1.0, 0.0)
            handle.center_of_rotation = (0.0, 0.0, 0.0)
            handle.camera_revision += 1
            self.reset_accumulation(view_id)
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

        # The perspective sensor uses a 45 degree field of view below. Fit a
        # conservative bounding sphere so the entire VTK dataset is visible.
        half_fov = math.radians(45.0) * 0.5
        distance = 1.15 * radius / math.sin(half_fov)

        origin = center + np.array([0.0, 0.0, distance])
        handle.camera_origin = tuple(float(v) for v in origin)
        handle.camera_target = tuple(float(v) for v in center)
        handle.camera_up = (0.0, 1.0, 0.0)
        handle.center_of_rotation = tuple(float(v) for v in center)
        handle.camera_revision += 1
        self.reset_accumulation(view_id)

    def get_camera_state(self, view_id: str) -> dict[str, Any]:
        handle = self._views[view_id]
        return {
            "position": list(handle.camera_origin),
            "target": list(handle.camera_target),
            "up": list(handle.camera_up),
            "center_of_rotation": list(handle.center_of_rotation),
            "fov": 45.0,
        }

    def set_camera_state(self, view_id: str, camera: dict[str, Any]) -> int:
        handle = self._views[view_id]
        handle.camera_origin = tuple(float(v) for v in camera["position"])
        handle.camera_target = tuple(float(v) for v in camera["target"])
        handle.camera_up = tuple(float(v) for v in camera["up"])
        center = camera.get("center_of_rotation", camera["target"])
        handle.center_of_rotation = tuple(float(v) for v in center)
        handle.camera_revision += 1
        return handle.camera_revision

    def camera_snapshot(self, view_id: str) -> tuple[int, dict[str, Any]]:
        handle = self._views[view_id]
        return handle.camera_revision, self.get_camera_state(view_id)

    def has_renderable_scene(self, view_id: str) -> bool:
        return any(
            current_view_id == view_id and rep.mesh is not None
            for (_, current_view_id), rep in self._representations.items()
        )

    def clear_accumulation(self, view_id: str, revision: int | None = None) -> None:
        handle = self._views[view_id]
        handle.accumulation = None
        handle.accumulated_spp = 0
        handle.accumulation_revision = (
            handle.camera_revision if revision is None else int(revision)
        )

    def accumulation_revision(self, view_id: str) -> int:
        return self._views[view_id].accumulation_revision

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

        # Make first display useful without requiring an extra explicit reset.
        if representation.kind == "surface":
            self.reset_camera(view.id)

    def update_representation(
        self,
        representation: Representation,
        view: RenderView,
        source: Any,
    ) -> None:
        self._representations[(representation.id, view.id)] = self._create_handle(
            representation, source
        )
        self.invalidate_accumulation(view.id)

    def remove_representation(self, representation_id: str, view_id: str) -> None:
        self._representations.pop((representation_id, view_id), None)
        if view_id in self._views:
            self.invalidate_accumulation(view_id)

    def _create_handle(
        self,
        representation: Representation,
        source: Any,
    ) -> MitsubaRepresentationHandle:
        if representation.kind != "surface":
            print(
                "Mitsuba backend: representation kind "
                f"'{representation.kind}' is not implemented"
            )
            return MitsubaRepresentationHandle(kind=representation.kind)

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

        polydata = triangles.GetOutput()
        if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfPolys() == 0:
            return MitsubaRepresentationHandle(kind="surface")

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
                        # Mitsuba interpolates vertex attributes. Duplicate the
                        # triangle vertices for cell coloring so all three
                        # vertices of each face carry the same color and the
                        # result remains flat, matching VTK cell semantics.
                        vertices = vertices[faces].reshape(-1, 3)
                        faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
                        vertex_colors = np.repeat(colors, 3, axis=0)
                    else:
                        vertex_colors = colors

        # No coordinate conversion here: VTK (1, 2, 3) is Mitsuba (1, 2, 3).
        mesh = self.mi.Mesh(
            f"vtkweb_{representation.id}",
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
            # A bare Mitsuba mesh is still renderable; don't make the prototype
            # depend on material assignment details of a particular version.
            print(f"Mitsuba backend: could not set surface color: {exc}")

        bounds = polydata.GetBounds()
        return MitsubaRepresentationHandle(
            kind="surface",
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
                "fov": float(camera.get("fov", 45.0)),
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
                            _srgb_to_linear(handle.world_ambient_color),
                            dtype=np.float32,
                        )
                        * float(handle.world_ambient_intensity)
                    ).tolist(),
                },
            },
        }

        surface_index = 0
        for (representation_id, current_view_id), rep in self._representations.items():
            if current_view_id != view_id or rep.mesh is None:
                continue
            scene_dict[f"surface_{surface_index}_{representation_id}"] = rep.mesh
            surface_index += 1

        scene = self.mi.load_dict(scene_dict)
        seed = handle.next_seed
        handle.next_seed += 1
        image = self.mi.render(scene, spp=spp, seed=seed)
        rendered = np.array(self.mi.Bitmap(image), dtype=np.float32, copy=True)

        if rendered.shape[-1] >= 4:
            rgb = rendered[..., :3]
            alpha = np.clip(rendered[..., 3:4], 0.0, 1.0)
            background = np.asarray(
                _srgb_to_linear(handle.background_color), dtype=np.float32
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
        revision: int,
    ) -> None:
        handle = self._views[view_id]
        if handle.accumulation_revision != revision:
            self.clear_accumulation(view_id, revision)
        if handle.accumulation is None or handle.accumulation.shape != sample.shape:
            handle.accumulation = np.zeros_like(sample, dtype=np.float32)
            handle.accumulated_spp = 0
            handle.accumulation_revision = revision
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
        revision, camera = self.camera_snapshot(view_id)
        if self.accumulation_revision(view_id) != revision:
            self.clear_accumulation(view_id, revision)
        sample = self.render_pass(view_id, camera, spp=spp)
        self.accumulate_pass(view_id, sample, spp=spp, revision=revision)
        return self.encoded_accumulated_frame(view_id) or ""

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
    """Map scalar values to RGB using normalized global TF control points."""

    control_points = transfer_function.get("control_points") or []
    if not control_points:
        return np.zeros((len(values), 3), dtype=np.float32)

    points = np.asarray(control_points, dtype=np.float64)
    order = np.argsort(points[:, 0], kind="stable")
    points = points[order]

    tf_range = transfer_function.get("range") or [0.0, 1.0]
    range_min = float(tf_range[0])
    range_max = float(tf_range[1])
    width = range_max - range_min
    if abs(width) < 1.0e-20:
        normalized = np.zeros(len(values), dtype=np.float64)
    else:
        normalized = (np.asarray(values, dtype=np.float64) - range_min) / width
    normalized = np.clip(normalized, 0.0, 1.0)

    # np.interp also gives the desired endpoint clamping outside the first and
    # last control point. Opacity intentionally remains renderer-neutral state
    # for now; Mitsuba surface transparency needs separate BSDF semantics.
    t = np.clip(points[:, 0], 0.0, 1.0)
    rgb = np.column_stack(
        [
            np.interp(normalized, t, np.clip(points[:, channel], 0.0, 1.0))
            for channel in (1, 2, 3)
        ]
    )
    return np.asarray(rgb, dtype=np.float32)


def _srgb_to_linear(
    color: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert a UI/display sRGB color to linear RGB for Mitsuba radiance."""

    def convert(component: float) -> float:
        component = max(0.0, min(1.0, float(component)))
        if component <= 0.04045:
            return component / 12.92
        return ((component + 0.055) / 1.055) ** 2.4

    return tuple(convert(component) for component in color)


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
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * float(np.dot(axis, vector)) * (1.0 - cosine)
    )
