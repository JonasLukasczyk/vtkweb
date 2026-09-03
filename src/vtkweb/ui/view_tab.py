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
    state.active_render_view_id = (
        rendering.active_view_id
    )

    state.view_background_color = (
        _rgb_to_hex(
            rendering
            .active_view
            .settings
            .background_color
        )
    )

    # -------------------------------------------------------------------------
    # Active view
    # -------------------------------------------------------------------------

    def set_active_render_view(
        view_id: str,
    ) -> None:
        rendering.set_active_view(
            view_id
        )

        view = rendering.active_view

        with state:
            state.active_render_view_id = (
                view.id
            )

            state.view_background_color = (
                _rgb_to_hex(
                    view.settings.background_color
                )
            )

        ctrl.update_node_visibility_state()

        if (
            pipeline.active_node_id
            is not None
        ):
            ctrl.update_representation_state(
                pipeline.active_node_id
            )

    # -------------------------------------------------------------------------
    # Background
    # -------------------------------------------------------------------------

    def set_view_background_color(
        value: str,
    ) -> None:
        rendering.set_background_color(
            rendering.active_view_id,
            _hex_to_rgb(value),
        )

        state.view_background_color = (
            value
        )

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    ctrl.set_active_render_view = (
        set_active_render_view
    )

    ctrl.set_view_background_color = (
        set_view_background_color
    )


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
            value=("view_background_color",),
            input=(
                ctrl.set_view_background_color,
                "[$event.target.value]",
            ),
        )


def _rgb_to_hex(
    color: tuple[
        float,
        float,
        float,
    ],
) -> str:
    values = [
        round(
            max(
                0.0,
                min(
                    1.0,
                    component,
                ),
            )
            * 255
        )
        for component in color
    ]

    return (
        f"#{values[0]:02x}"
        f"{values[1]:02x}"
        f"{values[2]:02x}"
    )


def _hex_to_rgb(
    value: str,
) -> tuple[
    float,
    float,
    float,
]:
    value = value.lstrip("#")

    return (
        int(
            value[0:2],
            16,
        )
        / 255.0,
        int(
            value[2:4],
            16,
        )
        / 255.0,
        int(
            value[4:6],
            16,
        )
        / 255.0,
    )
