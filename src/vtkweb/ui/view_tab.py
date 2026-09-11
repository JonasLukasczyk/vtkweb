from __future__ import annotations

from trame.widgets import html


def build_view_tab(
    ctrl,
) -> None:
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
                ctrl.set_view_property,
                "[active_view_id,'background_color',$event.target.value]",
            ),
        )

    with html.Label(
        classes="vtkweb-color-box mt-1",
    ):
        html.Span(
            "World Ambient Color",
            classes="vtkweb-control-label",
        )

        html.Input(
            type="color",
            value=("views[active_view_id]?.world_ambient_color || '#ffffff'",),
            input=(
                ctrl.set_view_property,
                "[active_view_id,'world_ambient_color',$event.target.value]",
            ),
        )

    with html.Label(
        classes="vtkweb-input-box mt-1",
    ):
        html.Span(
            "Ambient Intensity",
            classes="vtkweb-control-label",
        )

        html.Input(
            type="number",
            min="0",
            step="0.1",
            value=("views[active_view_id]?.world_ambient_intensity ?? 1.0",),
            change=(
                ctrl.set_view_property,
                "[active_view_id,'world_ambient_intensity',Number($event.target.value)]",
            ),
        )
