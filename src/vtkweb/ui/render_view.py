from __future__ import annotations

from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify3 as v3

from vtkweb.rendering import (
    RenderManager,
    VTKRenderingBackend,
)


def build_render_view(
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

    render_window = (
        backend.get_render_window(
            view.id
        )
    )

    with v3.VCol(
        cols=6,
        classes="pa-0",
        style="height:100vh;",
    ):
        vtk_view = vtk_widgets.VtkLocalView(
            render_window,
            ref=f"render_view_{view.id}",
            style=(
                "height:100%;"
                "width:100%;"
            ),
            click=(
                ctrl.set_active_render_view,
                f"['{view.id}']",
            ),
        )

    ctrl.view_update = (
        vtk_view.update
    )

    ctrl.view_reset_camera = (
        vtk_view.reset_camera
    )
