from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SceneGuideSettings:
    enabled: bool = True
    grid_enabled: bool = True
    axes_enabled: bool = True
    grid_spacing: float = 1.0
    grid_extent: float = 10.0


@dataclass(frozen=True)
class SceneCamera:
    position: tuple[float, float, float] = (0.0, 0.0, 5.0)
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = 30.0


def camera_from_mapping(value) -> SceneCamera:
    value = dict(value or {})
    return SceneCamera(
        position=tuple(float(v) for v in value.get("position", (0.0, 0.0, 5.0))),
        target=tuple(float(v) for v in value.get("target", (0.0, 0.0, 0.0))),
        up=tuple(float(v) for v in value.get("up", (0.0, 1.0, 0.0))),
        fov=max(1.0e-6, float(value.get("fov", 30.0))),
    )


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if not np.isfinite(length) or length < 1.0e-12:
        return fallback.copy()
    return v / length


def view_projection(camera: SceneCamera, width: int, height: int) -> np.ndarray:
    """Return Vulkan-clip-space projection * view, column-vector convention."""
    eye = np.asarray(camera.position, dtype=np.float64)
    target = np.asarray(camera.target, dtype=np.float64)
    up = np.asarray(camera.up, dtype=np.float64)

    forward = _normalize(target - eye, np.array([0.0, 0.0, -1.0]))
    right = _normalize(np.cross(forward, up), np.array([1.0, 0.0, 0.0]))
    true_up = _normalize(np.cross(right, forward), np.array([0.0, 1.0, 0.0]))

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -float(np.dot(right, eye))
    view[1, 3] = -float(np.dot(true_up, eye))
    view[2, 3] = float(np.dot(forward, eye))

    aspect = max(float(width) / max(float(height), 1.0), 1.0e-6)
    half = 0.5 * math.radians(camera.fov)
    f = 1.0 / max(math.tan(half), 1.0e-12)
    near, far = 0.01, 10000.0
    # Vulkan NDC: x/y in [-1, 1], z in [0, 1]. Negative Y compensates for the
    # framebuffer's top-left image convention so camera interaction matches VTK/Mitsuba.
    projection = np.zeros((4, 4), dtype=np.float64)
    projection[0, 0] = f / aspect
    projection[1, 1] = -f
    projection[2, 2] = far / (near - far)
    projection[2, 3] = (far * near) / (near - far)
    projection[3, 2] = -1.0
    return (projection @ view).astype(np.float32)


def build_guide_vertices(settings: SceneGuideSettings) -> np.ndarray:
    """Return interleaved xyz rgba vertices for line-list rendering."""
    vertices: list[tuple[float, float, float, float, float, float, float]] = []
    extent = max(float(settings.grid_extent), 1.0e-6)
    spacing = max(float(settings.grid_spacing), 1.0e-6)

    def line(a, b, color):
        vertices.append((*a, *color))
        vertices.append((*b, *color))

    # Blender-style world guide: XY grid at Z=0 plus explicit X/Y/Z axes.
    if settings.grid_enabled:
        count = min(200, int(math.floor(extent / spacing)))
        minor = (0.34, 0.34, 0.34, 0.52)
        major = (0.52, 0.52, 0.52, 0.68)
        for i in range(-count, count + 1):
            coordinate = i * spacing
            color = major if i % 5 == 0 else minor
            line((-extent, coordinate, 0.0), (extent, coordinate, 0.0), color)
            line((coordinate, -extent, 0.0), (coordinate, extent, 0.0), color)

    if settings.axes_enabled:
        axis_extent = max(extent, spacing * 4.0)
        line((0.0, 0.0, 0.0), (axis_extent, 0.0, 0.0), (1.0, 0.18, 0.18, 1.0))
        line((0.0, 0.0, 0.0), (0.0, axis_extent, 0.0), (0.25, 1.0, 0.25, 1.0))
        line((0.0, 0.0, 0.0), (0.0, 0.0, axis_extent), (0.25, 0.45, 1.0, 1.0))

    if not vertices:
        return np.empty((0, 7), dtype=np.float32)
    return np.asarray(vertices, dtype=np.float32)


class CpuSceneGuideRenderer:
    """Portable fallback/reference rasterizer for the viewport guide layer.

    The Vulkan renderer is preferred. This intentionally modest CPU line rasterizer
    keeps non-Vulkan/macOS deployments usable and makes camera/geometry semantics
    deterministic in tests.
    """

    name = "cpu"

    def __init__(self) -> None:
        self.width = 0
        self.height = 0

    def resize(self, width: int, height: int) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))

    @staticmethod
    def _blend_pixel(image: np.ndarray, x: int, y: int, rgba: np.ndarray) -> None:
        if x < 0 or y < 0 or y >= image.shape[0] or x >= image.shape[1]:
            return
        alpha = float(np.clip(rgba[3], 0.0, 1.0))
        image[y, x] = np.clip(
            image[y, x].astype(np.float32) * (1.0 - alpha)
            + np.asarray(rgba[:3], dtype=np.float32) * 255.0 * alpha,
            0.0,
            255.0,
        ).astype(np.uint8)

    def render(
        self,
        rgb: np.ndarray,
        camera: SceneCamera,
        settings: SceneGuideSettings,
    ) -> np.ndarray:
        if not settings.enabled:
            return rgb
        height, width = rgb.shape[:2]
        if (width, height) != (self.width, self.height):
            self.resize(width, height)
        vertices = build_guide_vertices(settings)
        if vertices.size == 0:
            return rgb

        mvp = view_projection(camera, width, height)
        result = rgb.copy()
        positions = np.concatenate(
            [vertices[:, :3], np.ones((len(vertices), 1), dtype=np.float32)], axis=1
        )
        clip = (mvp @ positions.T).T
        for index in range(0, len(vertices), 2):
            c0, c1 = clip[index], clip[index + 1]
            if c0[3] <= 1.0e-6 or c1[3] <= 1.0e-6:
                continue
            n0, n1 = c0[:3] / c0[3], c1[:3] / c1[3]
            # Simple reject; Vulkan performs proper clipping in the primary path.
            if ((n0[0] < -1 and n1[0] < -1) or (n0[0] > 1 and n1[0] > 1)
                    or (n0[1] < -1 and n1[1] < -1) or (n0[1] > 1 and n1[1] > 1)):
                continue
            p0 = np.array([(n0[0] * 0.5 + 0.5) * (width - 1), (n0[1] * 0.5 + 0.5) * (height - 1)])
            p1 = np.array([(n1[0] * 0.5 + 0.5) * (width - 1), (n1[1] * 0.5 + 0.5) * (height - 1)])
            delta = p1 - p0
            steps = max(1, min(4096, int(math.ceil(float(np.max(np.abs(delta)))))))
            color = vertices[index, 3:7]
            for step in range(steps + 1):
                t = step / steps
                p = p0 + delta * t
                self._blend_pixel(result, int(round(p[0])), int(round(p[1])), color)
        return result

    def close(self) -> None:
        pass


def create_scene_guide_renderer():
    """Prefer the persistent Vulkan raster layer, retaining a CPU fallback."""
    try:
        from vtkweb.rendering.vulkan_scene_guides import VulkanSceneGuideRenderer
        return VulkanSceneGuideRenderer()
    except Exception as exc:
        print(f"[scene-guides] Vulkan unavailable, using CPU renderer: {exc}", flush=True)
        return CpuSceneGuideRenderer()
