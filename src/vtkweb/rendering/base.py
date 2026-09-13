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

VIEW_PROPERTY_NAMES = (
    "background_color",
    "world_ambient_color",
    "world_ambient_intensity",
    "camera",
)

DEFAULT_VIEW_PROPERTIES = {
    "background_color": "#1a1a1a",
    "world_ambient_color": "#ffffff",
    "world_ambient_intensity": 1.0,
}


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
    def render_frame(self, view_id: str) -> bytes | None: ...
