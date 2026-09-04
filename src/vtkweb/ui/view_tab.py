from __future__ import annotations

from trame.widgets import html


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
                "[active_view_id,$event.target.value]",
            ),
        )
