from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ViewType:
    create: Callable[..., str]
    remove: Callable[[str], None]


class ViewRegistry:
    def __init__(self) -> None:
        self._types: dict[str, ViewType] = {}

    def register(self, name: str, *, create: Callable[..., str], remove: Callable[[str], None]) -> None:
        if name in self._types:
            raise ValueError(f"View type already registered: {name}")
        self._types[name] = ViewType(create=create, remove=remove)

    def create(self, type_name: str, **kwargs) -> str:
        try:
            view_type = self._types[type_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._types))
            raise ValueError(f"Unknown view type '{type_name}'. Available: {available}") from exc
        return view_type.create(**kwargs)

    def remove(self, name: str, view_id: str) -> None:
        self._types[name].remove(view_id)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._types))
