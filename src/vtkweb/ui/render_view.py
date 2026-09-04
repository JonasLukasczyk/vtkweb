from __future__ import annotations

from trame.widgets import html
from trame.widgets import vtk as vtk_widgets

from vtkweb.rendering import (
    RenderManager,
    VTKRenderingBackend,
)


def build_render_view(
    state,
    ctrl,
    rendering: RenderManager,
) -> None:
    backend = rendering.backend

    if not isinstance(
        backend,
        VTKRenderingBackend,
    ):
        raise NotImplementedError(
            f"No render-view adapter for backend '{backend.name}'"
        )

    view = rendering.active_view
    render_window = backend.get_render_window(view.id)

    vtk_view = None

    def reset_active_view() -> None:
        vtk_view.reset_camera()
        vtk_view.update()

    ctrl.trigger("render_view_reset")(reset_active_view)

    with html.Div(
        tabindex=0,
        style=("height:100%;width:100%;outline:none;"),
        raw_attrs=[
            "@keydown.space.prevent=\"trigger('render_view_reset')\"",
        ],
    ):
        vtk_view = vtk_widgets.VtkLocalView(
            render_window,
            ref=f"render_view_{view.id}",
            tabindex=0,
            style=("height:100%;width:100%;outline:none;"),
            click=(
                ctrl.set_active_view,
                f"['{view.id}']",
            ),
        )

    @state.change(
        "pipeline",
        "representations",
        "views",
    )
    def update_render_view(**_):
        vtk_view.update()
