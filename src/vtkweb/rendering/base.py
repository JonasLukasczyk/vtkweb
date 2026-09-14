from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4


REPRESENTATION_KINDS = (
    "surface",
    "wireframe",
    "outline",
    "volume",
)

VIEW_PROPERTY_SPECS = {
    "background_color": {
        "name": "background_color",
        "label": "Background",
        "kind": "color",
        "default": "#1a1a1a",
    },
    "world_ambient_color": {
        "name": "world_ambient_color",
        "label": "World Ambient Color",
        "kind": "color",
        "default": "#ffffff",
    },
    "world_ambient_intensity": {
        "name": "world_ambient_intensity",
        "label": "Ambient Intensity",
        "kind": "float",
        "default": 1.0,
        "min": 0.0,
        "step": 0.1,
    },
    "camera": {
        "name": "camera",
        "label": "Camera",
        "kind": "camera",
        "default": None,
        "ui": False,
    },
    "debug": {
        "name": "debug",
        "label": "Debug",
        "kind": "bool",
        "default": False,
    },
    "fps_limit": {
        "name": "fps_limit",
        "label": "FPS Limit",
        "kind": "int",
        "default": 30,
        "min": 1,
        "step": 1,
    },
    "distributed": {
        "name": "distributed",
        "label": "Distributed Rendering",
        "kind": "bool",
        "default": False,
    },
}

VIEW_PROPERTY_NAMES = tuple(VIEW_PROPERTY_SPECS)
DEFAULT_VIEW_PROPERTIES = {
    name: spec["default"] for name, spec in VIEW_PROPERTY_SPECS.items()
}


def view_property_state(name: str, value: Any) -> dict[str, Any]:
    """Return serializable metadata and value for one view property."""
    spec = dict(VIEW_PROPERTY_SPECS[name])
    spec.pop("default", None)
    spec["value"] = value
    return spec


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
    ) -> bytes | None: ...
