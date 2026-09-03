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
class Representation:
    node_id: str
    kind: str = "surface"
    visible: bool = True

    array_name: str | None = None
    association: str = "point"
    scalar_range: tuple[float, float] | None = None

    id: str = field(
        default_factory=lambda: uuid4().hex
    )


@dataclass
class ViewSettings:
    background_color: tuple[
        float,
        float,
        float,
    ] = (0.1, 0.1, 0.1)


class RenderingBackend(ABC):
    name: str

    @abstractmethod
    def add_representation(
        self,
        representation: Representation,
        source: Any,
    ) -> None:
        pass

    @abstractmethod
    def update_representation(
        self,
        representation: Representation,
        source: Any,
    ) -> None:
        pass

    @abstractmethod
    def remove_representation(
        self,
        representation_id: str,
    ) -> None:
        pass

    @abstractmethod
    def set_view_settings(
        self,
        settings: ViewSettings,
    ) -> None:
        pass

    @abstractmethod
    def reset_camera(self) -> None:
        pass
