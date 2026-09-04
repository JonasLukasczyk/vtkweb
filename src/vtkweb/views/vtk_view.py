from __future__ import annotations


def register(registry, rendering) -> None:
    def create(*, name=None, view_id=None, **_kwargs) -> str:
        return rendering.add_view(name=name, view_id=view_id).id

    registry.register("vtk", create=create, remove=rendering.remove_view)
