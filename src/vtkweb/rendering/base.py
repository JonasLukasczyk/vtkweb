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


@dataclass
class RenderView:
    name: str
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class Representation:
    node_id: str
    output_port: int = 0
    kind: str = "outline"

    # Serializable, renderer-agnostic representation properties. Backends
    # consume the keys they understand and ignore the rest.
    properties: dict[str, Any] = field(default_factory=dict)

    view_ids: set[str] = field(default_factory=set)

    id: str = field(default_factory=lambda: uuid4().hex)


class RenderingBackend(ABC):
    name: str

    @abstractmethod
    def add_view(
        self,
        view: RenderView,
    ) -> None:
        pass

    @abstractmethod
    def remove_view(
        self,
        view_id: str,
    ) -> None:
        pass

    @abstractmethod
    def rename_view(
        self,
        view_id: str,
        new_view_id: str,
    ) -> None:
        pass

    @abstractmethod
    def add_representation(
        self,
        representation: Representation,
        view: RenderView,
        source: Any,
    ) -> None:
        pass

    @abstractmethod
    def update_representation(
        self,
        representation: Representation,
        view: RenderView,
        source: Any,
    ) -> None:
        pass

    @abstractmethod
    def remove_representation(
        self,
        representation_id: str,
        view_id: str,
    ) -> None:
        pass

    @abstractmethod
    def set_view_property(self, view_id: str, name: str, value: Any) -> None:
        pass

    @abstractmethod
    def reset_camera(
        self,
        view_id: str,
    ) -> None:
        pass

    @abstractmethod
    def get_camera_state(
        self,
        view_id: str,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def set_camera_state(
        self,
        view_id: str,
        value: dict[str, Any],
    ) -> None:
        pass


@runtime_checkable
class ProgressiveRenderingBackend(Protocol):
    """Runtime capabilities required by the progressive render scheduler."""

    def set_render_size(self, view_id: str, width: int, height: int) -> bool: ...
    def interact_camera(
        self, view_id: str, mode: str, dx: float, dy: float, viewport_height: float
    ) -> int: ...
    def render_snapshot(self, view_id: str) -> tuple[int, dict[str, Any]]: ...
    def has_renderable_scene(self, view_id: str) -> bool: ...
    def clear_accumulation(self, view_id: str, generation: int | None = None) -> None: ...
    def accumulation_generation(self, view_id: str) -> int: ...
    def render_pass(self, view_id: str, camera: dict[str, Any], *, spp: int = 1) -> Any: ...
    def accumulate_pass(
        self,
        view_id: str,
        sample: Any,
        *,
        spp: int,
        generation: int,
    ) -> None: ...
    def encoded_frame(self, image: Any) -> bytes: ...
    def encoded_accumulated_frame(self, view_id: str) -> bytes | None: ...
