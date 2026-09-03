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


def build_inspector_view(ctrl) -> None:
    with v3.VCol(
        cols=3,
        classes="pa-3",
        style=(
            "height:100vh;"
            "overflow-y:auto;"
            "min-width:0;"
        ),
    ):
        v3.VLabel("Inspector")

        with v3.VCard(
            classes="mt-2 pa-3",
            style=(
                "width:100%;"
                "min-width:0;"
            ),
        ):
            v3.VCardTitle(
                "{{ active_node_name }}",
                classes="pa-0 mb-2",
            )

            with v3.VTabs(
                model_value=("inspector_tab",),
                density="compact",
                grow=True,
                update_modelValue=(ctrl.set_inspector_tab, "[$event]",),
            ):
                v3.VTab(
                    "Properties",
                    value="properties",
                )

                v3.VTab(
                    "Representations",
                    value="representations",
                )

                v3.VTab(
                    "View",
                    value="view",
                )

            v3.VDivider(
                classes="mb-3"
            )

            with html.Div(
                v_if=(
                    "inspector_tab === "
                    "'properties'"
                ),
            ):
                build_properties_tab(
                    ctrl
                )

            with html.Div(
                v_if=(
                    "inspector_tab === "
                    "'representations'"
                ),
            ):
                build_representations_tab(
                    ctrl
                )

            with html.Div(
                v_if=(
                    "inspector_tab === "
                    "'view'"
                ),
            ):
                build_view_tab(
                    ctrl
                )
