from __future__ import annotations

from vtkweb.views.registry import ViewRegistry
from vtkweb.views import dummy_view, vtk_view


class ViewManager:
    """Registry-backed manager for heterogeneous workspace view content."""

    def __init__(self, state, rendering) -> None:
        self.state = state
        self.rendering = rendering
        self.registry = ViewRegistry()
        vtk_view.register(self.registry, rendering)
        dummy_view.register(self.registry, state)

    @property
    def views(self) -> tuple[dict, ...]:
        return tuple(dict(value) for value in self.state.views.values())

    def create_view(self, view_type: str, *, name=None, view_id=None, **kwargs) -> str:
        return self.registry.create(view_type, name=name, view_id=view_id, **kwargs)

    def remove_view(self, view_id: str) -> None:
        value = self.state.views[view_id]
        self.registry.remove(value["type"], view_id)

    def get(self, view_id: str) -> dict:
        return dict(self.state.views[view_id])
