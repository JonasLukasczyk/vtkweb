from __future__ import annotations

from trame.widgets import html

from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering import RenderManager


def initialize_view_tab(
    state,
    ctrl,
    pipeline: PipelineGraph,
    rendering: RenderManager,
) -> None:
    # View data and active_view_id are authoritative RenderManager state.

    def set_active_render_view(
        view_id: str,
    ) -> None:
        rendering.set_active_view(view_id)

        if pipeline.active_node_id is not None:
            ctrl.update_representation_state(pipeline.active_node_id)

    def set_view_background_color(
        value: str,
    ) -> None:
        rendering.set_background_color(
            rendering.active_view_id,
            _hex_to_rgb(value),
        )
        ctrl.view_update()

    ctrl.set_active_render_view = set_active_render_view
    ctrl.set_view_background_color = set_view_background_color


def build_view_tab(
    ctrl,
) -> None:
    html.Div(
        "View",
        classes="vtkweb-section-title",
    )

    with html.Label(
        classes="vtkweb-color-box",
    ):
        html.Span(
            "Background",
            classes="vtkweb-control-label",
        )

        html.Input(
            type="color",
            value=("views[active_view_id]?.background_color || '#1a1a1a'",),
            input=(
                ctrl.set_view_background_color,
                "[$event.target.value]",
            ),
        )


def _hex_to_rgb(
    value: str,
) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )
