from __future__ import annotations

import base64
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
    width: int = 1024
    height: int = 768
    camera_origin: tuple[float, float, float] = (0.0, 0.0, 5.0)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    center_of_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accumulation: np.ndarray | None = None
    accumulated_spp: int = 0
    next_seed: int = 1
    frame_data_url: str | None = None
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

    def __init__(self) -> None:
        import mitsuba as mi

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
        self._views[view.id].background_color = view.settings.background_color
        self.reset_accumulation(view.id)

    def reset_accumulation(self, view_id: str) -> None:
        handle = self._views[view_id]
        handle.accumulation = None
        handle.accumulated_spp = 0
        handle.next_seed = 1
        handle.accumulation_revision = handle.camera_revision

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
        self.reset_accumulation(view.id)

    def remove_representation(self, representation_id: str, view_id: str) -> None:
        self._representations.pop((representation_id, view_id), None)
        if view_id in self._views:
            self.reset_accumulation(view_id)

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

        # Keep the first material intentionally simple. Scalar coloring can be
        # layered onto this backend later without changing geometry transport.
        color = _hex_to_rgb(representation.properties.get("color", "#d9d9d9"))
        try:
            mesh.set_bsdf(
                self.mi.load_dict(
                    {
                        "type": "diffuse",
                        "reflectance": {"type": "rgb", "value": list(color)},
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
            "integrator": {"type": "path", "max_depth": 4},
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
                },
                "sampler": {"type": "independent", "sample_count": spp},
            },
            "environment": {
                "type": "constant",
                "radiance": {"type": "rgb", "value": [0.7, 0.7, 0.7]},
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
        return np.array(self.mi.Bitmap(image), dtype=np.float32, copy=True)[..., :3]

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

    def encoded_frame(self, image: np.ndarray) -> str:
        """Encode one linear RGB image as the base64 JPEG sent to the client."""
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
                payload = base64.b64encode(stream.read()).decode("ascii")
        finally:
            if filename is not None:
                try:
                    os.unlink(filename)
                except FileNotFoundError:
                    pass
        return f"data:image/jpeg;base64,{payload}"

    def encoded_accumulated_frame(self, view_id: str) -> str | None:
        handle = self._views[view_id]
        if handle.accumulation is None or handle.accumulated_spp <= 0:
            return None
        averaged = handle.accumulation / float(handle.accumulated_spp)
        handle.frame_data_url = self.encoded_frame(averaged)
        return handle.frame_data_url

    def render_frame(self, view_id: str, *, spp: int = 1) -> str:
        revision, camera = self.camera_snapshot(view_id)
        if self.accumulation_revision(view_id) != revision:
            self.clear_accumulation(view_id, revision)
        sample = self.render_pass(view_id, camera, spp=spp)
        self.accumulate_pass(view_id, sample, spp=spp, revision=revision)
        return self.encoded_accumulated_frame(view_id) or ""

    def get_frame_data_url(self, view_id: str) -> str | None:
        return self._views[view_id].frame_data_url

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
