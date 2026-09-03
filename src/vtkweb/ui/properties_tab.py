from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def build_properties_tab(ctrl) -> None:
    html.Div(
        "Filter",
        classes="vtkweb-section-title",
    )

    # -------------------------------------------------------------------------
    # Input arrays
    # -------------------------------------------------------------------------

    with html.Div(
        classes="vtkweb-prop-list mb-2",
    ):
        with html.Div(
            v_for="array in input_arrays",
            key="array.index",
            classes="vtkweb-select-box",
        ):
            html.Span(
                "{{ array.label }}",
                classes="vtkweb-control-label",
            )

            v3.VSelect(
                items=("array.items",),
                model_value=("array.value",),
                clearable=True,
                density="compact",
                hide_details=True,
                variant="plain",
                classes="vtkweb-select-control",
                update_modelValue=(
                    ctrl.set_input_array,
                    "[array.index, $event]",
                ),
            )

    # -------------------------------------------------------------------------
    # Elevation presets
    # -------------------------------------------------------------------------

    with v3.VRow(
        v_if=(
            "active_node_type === "
            "'vtkElevationFilter'"
        ),
        dense=True,
        classes="mb-3",
    ):
        for axis in ("x", "y", "z"):
            with v3.VCol(cols=4):
                v3.VBtn(
                    axis.upper(),
                    block=True,
                    size="small",
                    click=(
                        ctrl.set_elevation_axis,
                        f"['{axis}']",
                    ),
                )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    with html.Div(
        classes="vtkweb-prop-list",
    ):
        with html.Div(
            v_for="property in filter_properties",
            key="property.name",
            classes="vtkweb-prop-item",
        ):

            # -----------------------------------------------------------------
            # Boolean
            # -----------------------------------------------------------------

            with html.Label(
                v_if="property.kind === 'bool'",
                classes="vtkweb-bool-row",
            ):
                html.Span(
                    "{{ property.label }}",
                    classes="vtkweb-control-label",
                )

                html.Input(
                    type="checkbox",
                    checked=(
                        "Boolean(property.value)",
                    ),
                    change=(
                        ctrl.set_filter_property,
                        (
                            "[property.name,"
                            "$event.target.checked]"
                        ),
                    ),
                )

            # -----------------------------------------------------------------
            # Scalar / string
            # -----------------------------------------------------------------

            with html.Label(
                v_if=(
                    "property.kind === 'int' || "
                    "property.kind === 'float' || "
                    "property.kind === 'str'"
                ),
                classes="vtkweb-input-box",
            ):
                html.Span(
                    "{{ property.label }}",
                    classes="vtkweb-control-label",
                )

                html.Input(
                    type=(
                        "property.kind === 'str' "
                        "? 'text' : 'number'"
                    ),
                    step=(
                        "property.kind === 'int' "
                        "? 1 : 'any'"
                    ),
                    value=("property.value",),
                    change=(
                        ctrl.set_filter_property,
                        (
                            "[property.name,"
                            "$event.target.value]"
                        ),
                    ),
                )

            # -----------------------------------------------------------------
            # Vector
            # -----------------------------------------------------------------

            with html.Div(
                v_if="property.kind === 'vector'",
                classes="vtkweb-vector-box",
            ):
                html.Span(
                    "{{ property.label }}",
                    classes="vtkweb-control-label",
                )

                with html.Div(
                    classes="vtkweb-vector-fields",
                ):
                    html.Input(
                        v_for=(
                            "(component,index) "
                            "in property.value"
                        ),
                        key="index",
                        type="number",
                        step="any",
                        value=("component",),
                        change=(
                            ctrl.set_filter_vector_component,
                            (
                                "[property.name,"
                                "index,"
                                "$event.target.value]"
                            ),
                        ),
                    )

            # -----------------------------------------------------------------
            # Scalar list
            # -----------------------------------------------------------------

            with html.Div(
                v_if=(
                    "property.kind === "
                    "'scalar_list'"
                ),
                classes="vtkweb-list-inline",
            ):
                html.Span(
                    "{{ property.label }}",
                    classes="vtkweb-control-label",
                )

                with html.Div(
                    classes="vtkweb-list-inline-values",
                ):
                    html.Input(
                        v_for=(
                            "(value,index) "
                            "in property.value"
                        ),
                        key="index",
                        type="number",
                        step="any",
                        value=("value",),
                        change=(
                            ctrl.set_filter_list_value,
                            (
                                "[property.name,"
                                "index,"
                                "$event.target.value]"
                            ),
                        ),
                    )

                html.Button(
                    "−",
                    type="button",
                    classes="vtkweb-list-inline-button",
                    disabled=(
                        "property.value.length === 0"
                    ),
                    click=(
                        ctrl.remove_filter_list_value,
                        (
                            "[property.name,"
                            "property.value.length - 1]"
                        ),
                    ),
                )

                html.Button(
                    "+",
                    type="button",
                    classes="vtkweb-list-inline-button",
                    click=(
                        ctrl.add_filter_list_value,
                        "[property.name]",
                    ),
                )
