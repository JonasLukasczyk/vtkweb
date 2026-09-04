from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from vtkweb.ui.properties_tab import (
    build_properties_tab,
)
from vtkweb.ui.representations_tab import (
    build_representations_tab,
)
from vtkweb.ui.view_tab import (
    build_view_tab,
)


def initialize_inspector(
    state,
    ctrl,
) -> None:
    state.inspector_tab = "properties"

    def set_inspector_tab(
        value: str,
    ) -> None:
        state.inspector_tab = value

    ctrl.set_inspector_tab = set_inspector_tab


def build_inspector_view(
    ctrl,
) -> None:
    with html.Div(
        classes="pa-3",
        style=(
            "height:100%;"
            "width:100%;"
            "min-width:0;"
            "min-height:0;"
            "overflow:hidden;"
            "display:flex;"
            "flex-direction:column;"
        ),
    ):
        with v3.VCard(
            classes="mt-2 pa-3",
            style=(
                "width:100%;"
                "min-width:0;"
                "min-height:0;"
                "height:100%;"
                "display:flex;"
                "flex-direction:column;"
            ),
        ):
            with v3.VTabs(
                model_value=("inspector_tab",),
                density="compact",
                # grow=True,
                update_modelValue=(
                    ctrl.set_inspector_tab,
                    "[$event]",
                ),
            ):
                with v3.VTab(
                    value="properties",
                ):
                    v3.VIcon("mdi-tune")

                with v3.VTab(
                    value="representations",
                ):
                    v3.VIcon("mdi-cube-scan")

                with v3.VTab(
                    value="view",
                ):
                    v3.VIcon("mdi-monitor-edit")

            v3.VDivider(
                classes="mb-3",
            )

            with html.Div(
                style=("flex:1;min-height:0;overflow-y:auto;"),
            ):
                with html.Div(
                    v_if=("inspector_tab === 'properties'"),
                ):
                    build_properties_tab(ctrl)

                with html.Div(
                    v_if=("inspector_tab === 'representations'"),
                ):
                    build_representations_tab(ctrl)

                with html.Div(
                    v_if=("inspector_tab === 'view'"),
                ):
                    build_view_tab(ctrl)
