from __future__ import annotations

from trame.widgets import html


def build_view_tab(ctrl) -> None:
    html.Div(
        "View",
        classes="vtkweb-section-title",
    )

    with html.Label(
        classes="vtkweb-color-box",
    ):
        html.Span(
            "Background"
        )

        html.Input(
            type="color",
            value=("view_background_color",),
            input=(
                ctrl.set_view_background_color,
                "[$event.target.value]",
            ),
        )
