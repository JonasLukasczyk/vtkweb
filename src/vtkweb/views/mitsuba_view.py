from __future__ import annotations


def register(registry, rendering) -> None:
    def create(*, name=None, view_id=None, **_kwargs) -> str:
        return rendering.add_view(
            name=name,
            view_id=view_id,
            view_type="mitsuba",
        ).id

    registry.register("mitsuba", create=create, remove=rendering.remove_view)
