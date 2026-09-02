from __future__ import annotations

from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify3 as v3

from vtkweb.rendering import RenderManager


def build_render_view(
    ctrl,
    rendering: RenderManager,
) -> None:
    with v3.VCol(
        cols=6,
        classes="pa-0",
        style="height:100vh;",
    ):
        view = vtk_widgets.VtkLocalView(
            rendering.render_window,
            ref="view",
            style=(
                "height:100%;"
                "width:100%;"
            ),
        )

        ctrl.view_update = view.update

        ctrl.view_reset_camera = (
            view.reset_camera
        )
