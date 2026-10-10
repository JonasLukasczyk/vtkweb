from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4




@dataclass(frozen=True)
class RenderedFrame:
    """Raw top-to-bottom RGB24 tile."""

    rgb: bytes
    width: int
    height: int


REPRESENTATION_KINDS = (
    "surface",
    "wireframe",
    "outline",
    "dvr",
    "vpt",
)

DIRECTIONAL_LIGHTS = (
    {"name": "light_pos_x", "label": "+X", "direction": (1.0, 0.0, 0.0), "default": 0.0},
    {"name": "light_neg_x", "label": "-X", "direction": (-1.0, 0.0, 0.0), "default": 0.0},
    {"name": "light_pos_y", "label": "+Y", "direction": (0.0, 1.0, 0.0), "default": 0.0},
    {"name": "light_neg_y", "label": "-Y", "direction": (0.0, -1.0, 0.0), "default": 0.0},
    {"name": "light_pos_z", "label": "+Z", "direction": (0.0, 0.0, 1.0), "default": 1.0},
    {"name": "light_neg_z", "label": "-Z", "direction": (0.0, 0.0, -1.0), "default": 0.0},
)
DIRECTIONAL_LIGHT_PROPERTY_NAMES = tuple(light["name"] for light in DIRECTIONAL_LIGHTS)
DIRECTIONAL_LIGHT_DIRECTIONS = tuple(light["direction"] for light in DIRECTIONAL_LIGHTS)

VIEW_PROPERTY_GROUPS = {
    "environment": {"label": "Environment", "icon": "mdi-weather-sunny", "expanded": True},
    "camera": {"label": "Camera", "icon": "mdi-camera-outline", "expanded": True},
    "performance": {"label": "Performance", "icon": "mdi-speedometer", "expanded": False},
}

VIEW_PROPERTY_SPECS = {
    "hdri": {"name": "hdri", "label": "HDRI", "kind": "str", "default": "", "group": "environment"},
    "background_color": {
        "name": "background_color",
        "label": "Background",
        "kind": "color",
        "default": "#1a1a1a",
        "group": "environment",
    },
    "world_ambient_color": {
        "name": "world_ambient_color",
        "label": "World Ambient Color",
        "kind": "color",
        "default": "#ffffff",
        "group": "environment",
    },
    "world_ambient_intensity": {
        "name": "world_ambient_intensity",
        "label": "Ambient Intensity",
        "kind": "float",
        "default": 1.0,
        "min": 0.0,
        "step": 0.1,
        "group": "environment",
    },
    "camera": {
        "name": "camera",
        "label": "Camera",
        "kind": "camera",
        "default": None,
        "ui": False,
        "group": "camera",
    },
    "camera_focal_length_mm": {
        "name": "camera_focal_length_mm",
        "label": "Focal length",
        "kind": "float",
        "default": 44.78460969082653,
        "min": 1.0,
        "step": 1.0,
        "group": "camera",
    },
    "camera_focus_distance": {
        "name": "camera_focus_distance",
        "label": "Focus distance",
        "kind": "float",
        "default": 0.0,
        "min": 0.0,
        "step": 0.1,
        "group": "camera",
    },
    "camera_aperture_size": {
        "name": "camera_aperture_size",
        "label": "Aperture size",
        "kind": "float",
        "default": 0.0,
        "min": 0.0,
        "step": 0.001,
        "group": "camera",
    },
    "fps_limit": {
        "name": "fps_limit",
        "label": "FPS Limit",
        "kind": "int",
        "default": 30,
        "min": 1,
        "step": 1,
        "group": "performance",
    },
    "distributed": {
        "name": "distributed",
        "label": "Distributed Rendering",
        "kind": "bool",
        "default": False,
        "group": "performance",
    },
    **{
        light["name"]: {
            "name": light["name"],
            "label": light["label"],
            "kind": "float",
            "default": light["default"],
            "min": 0.0,
            "step": 0.1,
            "ui": False,
            "group": "lighting",
        }
        for light in DIRECTIONAL_LIGHTS
    },
}

VIEW_PROPERTY_NAMES = tuple(VIEW_PROPERTY_SPECS)
DEFAULT_VIEW_PROPERTIES = {
    name: spec["default"] for name, spec in VIEW_PROPERTY_SPECS.items()
}

REPRESENTATION_PROPERTY_GROUPS = {
    "appearance": {"label": "Appearance", "icon": "mdi-palette-outline", "expanded": True},
    "sampling": {"label": "Sampling", "icon": "mdi-grid", "expanded": True},
    "lighting": {"label": "Material", "icon": "mdi-texture-box", "expanded": True},
    "scattering": {"label": "Scattering", "icon": "mdi-blur", "expanded": False},
    "volume_resources": {"label": "Caching", "icon": "mdi-cube-outline", "expanded": False},
}

REPRESENTATION_PROPERTY_SPECS = {
    "color_by": {"label": "Color by", "kind": "array", "default": None, "kinds": {"surface", "wireframe", "dvr", "vpt"}, "group": "appearance"},
    "color": {"label": "Color", "kind": "color", "default": "#ffffff", "kinds": {"surface", "wireframe", "dvr", "vpt"}, "group": "appearance"},
    "line_width": {"label": "Line width", "kind": "float", "default": 0.01, "min": 0.0, "step": 0.001, "kinds": {"wireframe", "outline"}, "group": "appearance"},
    "tube_sides": {"label": "Tube sides", "kind": "int", "default": 3, "min": 3, "step": 1, "kinds": {"wireframe", "outline"}, "group": "appearance"},
    "interpolation": {"label": "Interpolation", "kind": "choice", "default": "linear", "options": (("Trilinear", "linear"), ("Nearest", "nearest")), "kinds": {"dvr", "vpt"}, "group": "sampling"},
    "blend_mode": {"label": "Blend", "kind": "choice", "default": "composite", "options": (("Composite", "composite"), ("Maximum intensity", "maximum"), ("Minimum intensity", "minimum")), "kinds": {"dvr"}, "group": "appearance"},
    "shade": {"label": "Shading", "kind": "bool", "default": True, "kinds": {"dvr"}, "group": "lighting"},
    "ambient": {"label": "Ambient", "kind": "float", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05, "kinds": {"dvr"}, "group": "lighting"},
    "diffuse": {"label": "Diffuse", "kind": "float", "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05, "kinds": {"dvr"}, "group": "lighting"},
    "global_illumination_reach": {"label": "Global illumination reach", "kind": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "kinds": {"dvr"}, "group": "sampling"},
    "volumetric_scattering_blending": {"label": "Scattering strength", "kind": "float", "default": 2.0, "min": 0.0, "max": 2.0, "step": 0.05, "kinds": {"dvr"}, "group": "scattering"},
    "scattering_anisotropy": {"label": "Scattering anisotropy", "kind": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05, "kinds": {"dvr"}, "group": "scattering"},
    "environment_scattering_strength": {"label": "Environment scatter strength", "kind": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "kinds": {"dvr"}, "group": "scattering"},
    "environment_scattering_samples": {"label": "Environment bake directions", "kind": "int", "default": 0, "min": 0, "max": 256, "step": 1, "kinds": {"dvr"}, "group": "scattering"},
    "environment_scattering_step_factor": {"label": "Environment lighting resolution factor", "kind": "float", "default": 4.0, "min": 1.0, "max": 32.0, "step": 0.5, "kinds": {"dvr"}, "group": "scattering"},
    "vpt_transport": {"label": "VPT transport", "kind": "choice", "default": "delta", "options": (("Stochastic absorption (delta tracking)", "delta"), ("Deterministic absorption diagnostic", "diagnostic")), "kinds": {"vpt"}, "group": "sampling"},
    "vpt_spp_per_frame": {"label": "VPT samples per frame", "kind": "int", "default": 1, "min": 1, "max": 64, "step": 1, "kinds": {"vpt"}, "group": "sampling"},
    "vpt_performance_log": {"label": "VPT performance logging", "kind": "bool", "default": False, "kinds": {"vpt"}, "group": "sampling"},
    "vpt_majorant_brick_size": {"label": "VPT majorant brick size", "kind": "choice", "default": "8", "options": (("4 voxels", "4"), ("8 voxels", "8"), ("16 voxels", "16")), "kinds": {"vpt"}, "group": "volume_resources"},
    "vpt_transmittance_cache": {"label": "VPT approximate directional transmittance cache (experimental)", "kind": "bool", "default": False, "kinds": {"vpt"}, "group": "volume_resources"},
    "vpt_transmittance_cache_directions": {"label": "VPT transmittance cache directions", "kind": "choice", "default": "26", "options": (("6 axes", "6"), ("26 neighbors", "26")), "kinds": {"vpt"}, "group": "volume_resources"},
    "vpt_max_depth": {"label": "Max scattering depth", "kind": "int", "default": 1, "min": 1, "max": 16, "step": 1, "kinds": {"vpt"}, "group": "scattering"},
    "vpt_anisotropy": {"label": "Scattering anisotropy (g)", "kind": "float", "default": 0.0, "min": -0.9, "max": 0.9, "step": 0.05, "kinds": {"vpt"}, "group": "scattering"},
    "scattering_albedo": {"label": "Scattering albedo", "kind": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "kinds": {"vpt"}, "group": "scattering"},
    "sample_distance": {"label": "Sample distance", "kind": "float", "default": 1.0, "min": 0.000001, "step": "any", "kinds": {"dvr"}, "group": "sampling"},
    "preintegration": {"label": "Pre-integration", "kind": "choice", "default": "512", "options": (("Off", "0"), ("256", "256"), ("512", "512"), ("1024", "1024"), ("2048", "2048"), ("4096", "4096")), "kinds": {"dvr"}, "group": "sampling"},
    "scalar_volume": {"label": "Scalar volume", "kind": "choice", "default": "f32", "options": (("FP32", "f32"), ("FP16", "f16")), "kinds": {"dvr"}, "group": "volume_resources"},
    "shadow_volume": {"label": "Shadow volume", "kind": "choice", "default": "f32", "options": (("Off", "off"), ("FP32", "f32"), ("FP16", "f16"), ("U8", "u8")), "kinds": {"dvr"}, "group": "volume_resources"},
    "environment_volume": {"label": "Environment volume", "kind": "choice", "default": "off", "options": (("Off", "off"), ("FP32", "f32"), ("FP16", "f16")), "kinds": {"dvr"}, "group": "volume_resources"},
}

DEFAULT_REPRESENTATION_PROPERTIES = {
    name: spec["default"] for name, spec in REPRESENTATION_PROPERTY_SPECS.items()
}

def normalize_representation_property(name: str, value: Any) -> Any:
    spec = REPRESENTATION_PROPERTY_SPECS.get(name)
    if spec is None:
        return value
    kind = spec["kind"]
    if kind == "bool":
        return bool(value)
    if kind == "int":
        value = int(round(float(value)))
    elif kind == "float":
        value = float(value)
    elif kind == "choice":
        value = str(value).lower()
        allowed = {item[1] for item in spec.get("options", ())}
        if value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}")
        return value
    if "min" in spec:
        value = max(spec["min"], value)
    if "max" in spec:
        value = min(spec["max"], value)
    return value



@dataclass
class RenderView:
    name: str
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class Representation:
    node_id: str
    output_port: int = 0
    kind: str = "outline"
    properties: dict[str, Any] = field(default_factory=dict)
    view_ids: set[str] = field(default_factory=set)
    id: str = field(default_factory=lambda: uuid4().hex)


class RenderingBackend(ABC):
    name: str

    @abstractmethod
    def add_view(self, view: RenderView) -> None: ...

    @abstractmethod
    def remove_view(self, view_id: str) -> None: ...

    @abstractmethod
    def add_representation(
        self, representation: Representation, view: RenderView, source: Any
    ) -> None: ...

    @abstractmethod
    def update_representation(
        self, representation: Representation, view: RenderView, source: Any
    ) -> None: ...

    @abstractmethod
    def remove_representation(self, representation_id: str, view_id: str) -> None: ...

    @abstractmethod
    def get_view_property(self, view_id: str, name: str) -> Any: ...

    @abstractmethod
    def set_view_property(self, view_id: str, name: str, value: Any) -> None: ...

    @abstractmethod
    def reset_camera(self, view_id: str) -> None: ...


@runtime_checkable
class FrameRenderingBackend(Protocol):
    """Capabilities required by the server-side frame scheduler.

    Each logical view owns one dedicated render worker. The worker repeatedly
    asks the backend to render its current state and publishes every completed
    frame. State invalidation and progressive accumulation are backend concerns,
    not scheduler concerns.
    """

    def set_render_size(self, view_id: str, width: int, height: int) -> bool: ...
    def has_renderable_scene(self, view_id: str) -> bool: ...
    def render_frame(
        self, view_id: str, *, region=None, full_size=None
    ) -> RenderedFrame | None: ...
