from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


REPRESENTATION_KINDS = (
    "surface",
    "wireframe",
    "outline",
)


@dataclass
class ViewSettings:
    background_color: tuple[
        float,
        float,
        float,
    ] = (0.1, 0.1, 0.1)


@dataclass
class RenderView:
    name: str
    settings: ViewSettings = field(default_factory=ViewSettings)
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass
class Representation:
    node_id: str
    output_port: int = 0
    kind: str = "outline"

    array_name: str | None = None
    association: str = "point"
    scalar_range: (
        tuple[
            float,
            float,
        ]
        | None
    ) = None
    color: str = "#ffffff"

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
    def set_view_settings(
        self,
        view: RenderView,
    ) -> None:
        pass

    @abstractmethod
    def reset_camera(
        self,
        view_id: str,
    ) -> None:
        pass
