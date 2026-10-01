from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PostprocessSettings:
    debug: bool = False
    ssao_slices: int = 0
    ssao_steps: int = 6
    ssao_radius: float = 10.0
    ssao_strength: float = 1.0
    ssao_thickness: float = 0.4
    dof_mode: str = "physical"
    focus_distance: float = 1.0
    aperture_size: float = 0.0
    camera_fov: float = 30.0


class CpuPostprocessor:
    """Reference/fallback implementation matching the original compositor."""

    name = "cpu"

    def resize(self, width: int, height: int) -> None:
        pass

    def upload_tile(self, rgb: np.ndarray, depth: np.ndarray | None, *, x: int, y: int) -> None:
        pass

    @staticmethod
    def _debug_depth(depth: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth) & (depth > 0.0)
        value = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(valid):
            near = float(np.min(depth[valid])); far = float(np.max(depth[valid]))
            width = max(far - near, 1.0e-12)
            value[valid] = np.asarray((1.0 - np.clip((depth[valid] - near) / width, 0.0, 1.0)) * 255.0, dtype=np.uint8)
        return np.repeat(value[..., None], 3, axis=2)

    @staticmethod
    def _apply_ssao(settings: PostprocessSettings, depth: np.ndarray | None, rgb: np.ndarray) -> np.ndarray:
        if depth is None or settings.ssao_slices <= 0 or settings.ssao_radius <= 0.0:
            return rgb
        h, w = depth.shape
        valid_center = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid_center): return rgb
        direction_count = max(2, min(8, settings.ssao_slices * 2))
        step_count = max(1, min(4, settings.ssao_steps))
        radius = max(1.0, settings.ssao_radius); thickness = max(1.0e-6, settings.ssao_thickness)
        occlusion = np.zeros((h, w), np.float32); samples = np.zeros((h, w), np.float32)
        for di in range(direction_count):
            angle = math.pi * di / direction_count
            for si in range(1, step_count + 1):
                distance = radius * (si / step_count) ** 2
                dx = int(round(math.cos(angle) * distance)); dy = int(round(math.sin(angle) * distance))
                if dx == 0 and dy == 0: continue
                x0=max(0,-dx); x1=min(w,w-dx); y0=max(0,-dy); y1=min(h,h-dy)
                if x1 <= x0 or y1 <= y0: continue
                center=depth[y0:y1,x0:x1]; neighbor=depth[y0+dy:y1+dy,x0+dx:x1+dx]
                valid=valid_center[y0:y1,x0:x1] & np.isfinite(neighbor) & (neighbor > 0.0)
                occlusion[y0:y1,x0:x1] += (valid & (neighbor < center - thickness)).astype(np.float32)
                samples[y0:y1,x0:x1] += valid.astype(np.float32)
        ratio=np.divide(occlusion,samples,out=np.zeros_like(occlusion),where=samples>0.0)
        ao=np.power(np.clip(1.0-ratio,0.0,1.0),max(0.0,settings.ssao_strength))
        return np.clip(rgb.astype(np.float32)*ao[...,None],0.0,255.0).astype(np.uint8)

    @staticmethod
    def _box_blur(image: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0: return image.astype(np.float32, copy=False)
        source=image.astype(np.float32,copy=False); scalar=source.ndim==2
        if scalar: source=source[...,None]
        padded=np.pad(source,((radius,radius),(radius,radius),(0,0)),mode="edge")
        integral=np.pad(padded,((1,0),(1,0),(0,0)),mode="constant").cumsum(0).cumsum(1)
        size=2*radius+1
        summed=integral[size:,size:]-integral[:-size,size:]-integral[size:,:-size]+integral[:-size,:-size]
        blurred=summed/float(size*size)
        return blurred[...,0] if scalar else blurred

    @classmethod
    def _apply_dof(cls, settings: PostprocessSettings, depth: np.ndarray | None, rgb: np.ndarray) -> np.ndarray:
        if settings.dof_mode != "postprocess" or settings.aperture_size <= 0.0 or depth is None: return rgb
        focus=max(1e-6,settings.focus_distance); fov=math.radians(max(1e-6,settings.camera_fov))
        focal_pixels=0.5*rgb.shape[0]/max(math.tan(0.5*fov),1e-12)
        inverse=np.zeros_like(depth,np.float32); valid=np.isfinite(depth)&(depth>0.0); inverse[valid]=1.0/depth[valid]
        radius=np.clip(settings.aperture_size*focal_pixels*np.abs(inverse-1.0/focus),0.0,32.0)
        source=rgb.astype(np.float32,copy=False); result=source.copy(); previous_radius=0; previous=source
        for current in (1,2,4,8,16,32):
            blurred=cls._box_blur(source,current); mask=(radius>previous_radius)&(radius<=current)
            if np.any(mask):
                t=np.clip((radius[mask]-previous_radius)/float(current-previous_radius),0.0,1.0)
                result[mask]=previous[mask]*(1.0-t[:,None])+blurred[mask]*t[:,None]
            previous_radius=current; previous=blurred
        result[radius>32]=previous[radius>32]
        return np.clip(result,0.0,255.0).astype(np.uint8)

    def process(self, rgb: np.ndarray, depth: np.ndarray | None, settings: PostprocessSettings) -> np.ndarray:
        if settings.debug and depth is not None: return self._debug_depth(depth)
        result=self._apply_ssao(settings,depth,rgb.copy())
        return self._apply_dof(settings,depth,result)

    def close(self) -> None: pass


def create_postprocessor():
    """Prefer Vulkan, but never make Vulkan availability an application requirement."""
    try:
        from vtkweb.rendering.vulkan_postprocessor import VulkanPostprocessor
        return VulkanPostprocessor()
    except Exception as exc:
        print(f"[compositor] Vulkan unavailable, using CPU postprocessor: {exc}", flush=True)
        return CpuPostprocessor()
