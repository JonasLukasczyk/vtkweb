from __future__ import annotations

from uuid import uuid4


def register(registry, state) -> None:
    def create(*, name=None, view_id=None, **kwargs) -> str:
        view_id = view_id or uuid4().hex
        if view_id in state.views:
            raise ValueError(f"View ID already exists: {view_id}")
        views = dict(state.views)
        views[view_id] = {
            "id": view_id,
            "type": "dummy",
            "name": name or f"Dummy {sum(v.get('type') == 'dummy' for v in views.values()) + 1}",
            "message": kwargs.get("message", "Dummy view"),
        }
        state.views = views
        return view_id

    def remove(view_id: str) -> None:
        views = dict(state.views)
        views.pop(view_id, None)
        state.views = views

    registry.register("dummy", create=create, remove=remove)
