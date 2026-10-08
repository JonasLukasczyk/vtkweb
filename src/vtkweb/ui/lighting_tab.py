from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from vtkweb.rendering.base import DIRECTIONAL_LIGHTS


def build_lighting_tab(ctrl) -> None:
    with html.Div(classes="vtkweb-lighting-tab"):
        with html.Div(classes="vtkweb-property-group"):
            with html.Div(classes="vtkweb-property-group-header"):
                v3.VIcon(
                    "mdi-lightbulb-outline",
                    size="small",
                    classes="vtkweb-property-group-icon",
                )
                html.Span("Directional lights")

            with html.Div(classes="vtkweb-property-group-body"):
                for light in DIRECTIONAL_LIGHTS:
                    name = light["name"]
                    with html.Label(classes="vtkweb-input-box"):
                        html.Span(light["label"], classes="vtkweb-control-label")
                        html.Input(
                            type="number",
                            min="0",
                            step="0.1",
                            value=(f"views[active_view_id]?.properties?.{name}",),
                            change=(
                                ctrl.set_view_property,
                                f"[active_view_id,'{name}',Math.max(0,Number($event.target.value))]",
                            ),
                        )

        html.Div(
            "Intensity is unitless. A value of 0 disables the light.",
            classes="text-caption mt-2 opacity-70",
        )
