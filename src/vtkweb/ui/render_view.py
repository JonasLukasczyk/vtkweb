from __future__ import annotations

from trame.widgets import vtk as vtk_widgets
from trame.widgets import vuetify3 as v3

from vtkweb.rendering import RenderManager, VTKRenderingBackend


def build_render_view(
    state,
    ctrl,
    rendering: RenderManager,
) -> None:
    backend = rendering.backend
    if not isinstance(backend, VTKRenderingBackend):
        raise NotImplementedError(
            f"No render-view adapter for backend '{backend.name}'"
        )

    view = rendering.active_view
    render_window = backend.get_render_window(view.id)
    vtk_view = None

    def on_keydown(code: str) -> None:
        if code == "Space":
            ctrl.reset_camera(state.active_view_id)
            vtk_view.update()

    ctrl.render_view_keydown = on_keydown

    with v3.VCol(
        cols=6,
        classes="pa-0",
        style="height:100vh;",
    ):
        vtk_view = vtk_widgets.VtkLocalView(
            render_window,
            ref=f"render_view_{view.id}",
            tabindex=0,
            style="height:100%;width:100%;outline:none;",
            click=(
                ctrl.set_active_view,
                f"['{view.id}']",
            ),
            keydown=(
                ctrl.render_view_keydown,
                "[$event.code]",
            ),
        )

    @state.change("pipeline", "representations", "views")
    def update_render_view(**_):
        vtk_view.update()
