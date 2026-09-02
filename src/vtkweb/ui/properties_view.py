from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3


PROPERTY_STYLE = """
.vtkweb-select-box {
    display: flex;
    align-items: center;

    width: 100%;
    min-width: 0;
    height: 28px;

    border: 1px solid rgba(128, 128, 128, 0.5);
    border-radius: 4px;

    background: rgba(128, 128, 128, 0.08);

    box-sizing: border-box;
    overflow: hidden;
}

.vtkweb-select-box:hover {
    border-color: rgba(128, 128, 128, 0.8);
}

.vtkweb-select-box:focus-within {
    border-color: #4f7df3;
    background: rgba(79, 125, 243, 0.06);
}

.vtkweb-select-prefix {
    flex: 0 0 auto;

    padding: 0 6px 0 8px;

    font-size: 12px;
    line-height: 26px;
    opacity: 0.82;

    white-space: nowrap;
    user-select: none;
}

.vtkweb-select-control {
    flex: 1 1 auto;
    min-width: 0;
    width: 0;
}

.vtkweb-select-control .v-input__control {
    min-height: 26px !important;
    height: 26px !important;
}

.vtkweb-select-control .v-field {
    min-height: 26px !important;
    height: 26px !important;

    padding: 0 !important;

    background: transparent !important;
    box-shadow: none !important;
}

.vtkweb-select-control .v-field__outline {
    display: none !important;
}

.vtkweb-select-control .v-field__input {
    min-height: 26px !important;
    height: 26px !important;

    padding: 0 4px !important;

    font-size: 12px;
}

.vtkweb-select-control .v-field__append-inner {
    min-height: 26px !important;
    padding-top: 0 !important;
    align-items: center;
}

.vtkweb-section-title {
    margin-bottom: 6px;
    font-size: 12px;
    font-weight: 600;
    opacity: 0.8;
}

.vtkweb-prop-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 100%;
    min-width: 0;
}

.vtkweb-prop-item {
    width: 100%;
    min-width: 0;
}

/* ------------------------------------------------------------------------- */
/* Scalar / string input                                                    */
/* ------------------------------------------------------------------------- */

.vtkweb-input-box {
    display: flex;
    align-items: center;

    width: 100%;
    min-width: 0;
    height: 28px;

    border: 1px solid rgba(128, 128, 128, 0.5);
    border-radius: 4px;

    background: rgba(128, 128, 128, 0.08);

    box-sizing: border-box;
    overflow: hidden;

    cursor: text;
}

.vtkweb-input-box:hover {
    border-color: rgba(128, 128, 128, 0.8);
}

.vtkweb-input-box:focus-within {
    border-color: #4f7df3;
    background: rgba(79, 125, 243, 0.06);
}

.vtkweb-input-prefix {
    flex: 0 0 auto;

    padding: 0 6px 0 8px;

    font-size: 12px;
    line-height: 26px;

    opacity: 0.82;

    white-space: nowrap;

    cursor: text;
    user-select: none;
}

.vtkweb-input-box input {
    flex: 1 1 auto;
    min-width: 0;
    width: auto;
    height: 26px;

    margin: 0;
    padding: 0 8px;

    border: 0 !important;
    border-radius: 0;
    outline: 0 !important;
    box-shadow: none !important;

    background: transparent !important;
    color: inherit;

    font: inherit;
    font-size: 12px;
    text-align: right;
    font-variant-numeric: tabular-nums;

    box-sizing: border-box;
}

.vtkweb-input-box input:focus,
.vtkweb-input-box input:focus-visible {
    border: 0 !important;
    outline: 0 !important;
    box-shadow: none !important;
}

.vtkweb-input-box input[type="number"] {
    appearance: textfield;
    -moz-appearance: textfield;
}

.vtkweb-input-box input[type="number"]::-webkit-inner-spin-button,
.vtkweb-input-box input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
}

/* ------------------------------------------------------------------------- */
/* Vector input                                                              */
/* ------------------------------------------------------------------------- */

.vtkweb-vector-box {
    display: flex;
    align-items: center;

    width: 100%;
    min-width: 0;
    height: 28px;

    border: 1px solid rgba(128, 128, 128, 0.5);
    border-radius: 4px;

    background: rgba(128, 128, 128, 0.08);

    box-sizing: border-box;
    overflow: hidden;
}

.vtkweb-vector-box:hover {
    border-color: rgba(128, 128, 128, 0.8);
}

.vtkweb-vector-box:focus-within {
    border-color: #4f7df3;
    background: rgba(79, 125, 243, 0.06);
}

.vtkweb-vector-fields {
    flex: 1 1 auto;

    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));

    min-width: 0;
    height: 100%;
}

.vtkweb-vector-fields input {
    min-width: 0;
    width: 100%;
    height: 100%;

    margin: 0;
    padding: 0 5px;

    border: 0 !important;
    border-left: 1px solid rgba(128, 128, 128, 0.25) !important;
    border-radius: 0;
    outline: 0 !important;
    box-shadow: none !important;

    background: transparent !important;
    color: inherit;

    font: inherit;
    font-size: 12px;
    text-align: right;
    font-variant-numeric: tabular-nums;

    box-sizing: border-box;

    appearance: textfield;
    -moz-appearance: textfield;
}

.vtkweb-vector-fields input:focus,
.vtkweb-vector-fields input:focus-visible {
    outline: 0 !important;
    box-shadow: none !important;
}

.vtkweb-vector-fields input::-webkit-inner-spin-button,
.vtkweb-vector-fields input::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
}

/* ------------------------------------------------------------------------- */
/* Boolean                                                                   */
/* ------------------------------------------------------------------------- */

.vtkweb-bool-row {
    display: flex;
    align-items: center;
    justify-content: space-between;

    width: 100%;
    min-width: 0;
    height: 28px;

    padding: 0 8px;

    border: 1px solid rgba(128, 128, 128, 0.5);
    border-radius: 4px;

    background: rgba(128, 128, 128, 0.08);

    box-sizing: border-box;
}

.vtkweb-bool-row:hover {
    border-color: rgba(128, 128, 128, 0.8);
}

.vtkweb-bool-row:focus-within {
    border-color: #4f7df3;
}

.vtkweb-bool-label {
    min-width: 0;

    font-size: 12px;
    opacity: 0.82;

    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;

    cursor: pointer;
}

/* ------------------------------------------------------------------------- */
/* Scalar color range                                                        */
/* ------------------------------------------------------------------------- */

.vtkweb-range-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto;
    gap: 6px;

    width: 100%;
    margin-top: 8px;
}

.vtkweb-range-input {
    width: 100%;
    min-width: 0;
    height: 28px;

    padding: 0 7px;

    border: 1px solid rgba(128, 128, 128, 0.5);
    border-radius: 4px;
    outline: none;

    background: rgba(128, 128, 128, 0.08);
    color: inherit;

    font-size: 12px;
    text-align: right;

    box-sizing: border-box;
}

.vtkweb-range-input:focus {
    border-color: #4f7df3;
}
"""


def build_properties_view(ctrl) -> None:
    with v3.VCol(
        cols=3,
        classes="pa-3",
        style="height:100vh;overflow-y:auto;min-width:0;",
    ):
        v3.VLabel("Properties")

        with v3.VCard(
            classes="mt-2 pa-3",
            style="width:100%;min-width:0;",
        ):
            v3.VCardTitle(
                "{{ active_node_name }}",
                classes="pa-0 mb-3",
            )

            # -----------------------------------------------------------------
            # Filter
            # -----------------------------------------------------------------

            html.Div(
                "Filter",
                classes="vtkweb-section-title",
            )

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

            with html.Div(
                classes="vtkweb-prop-list",
            ):
                with html.Div(
                    v_for="property in filter_properties",
                    key="property.name",
                    classes="vtkweb-prop-item",
                ):

                    # Boolean
                    with html.Label(
                        v_if="property.kind === 'bool'",
                        classes="vtkweb-bool-row",
                    ):
                        html.Span(
                            "{{ property.label }}",
                            classes="vtkweb-bool-label",
                        )

                        html.Input(
                            type="checkbox",
                            checked=(
                                "Boolean(property.value)",
                            ),
                            change=(
                                ctrl.set_filter_property,
                                (
                                    "[property.name, "
                                    "$event.target.checked]"
                                ),
                            ),
                        )

                    # Scalar / string
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
                            classes="vtkweb-input-prefix",
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
                                    "[property.name, "
                                    "$event.target.value]"
                                ),
                            ),
                        )

                    # Vector
                    with html.Div(
                        v_if="property.kind === 'vector'",
                        classes="vtkweb-vector-box",
                    ):
                        html.Span(
                            "{{ property.label }}",
                            classes="vtkweb-input-prefix",
                        )

                        with html.Div(
                            classes="vtkweb-vector-fields",
                        ):
                            html.Input(
                                v_for=(
                                    "(component, index) "
                                    "in property.value"
                                ),
                                key="index",
                                type="number",
                                step="any",
                                value=("component",),
                                change=(
                                    ctrl.set_filter_vector_component,
                                    (
                                        "[property.name, "
                                        "index, "
                                        "$event.target.value]"
                                    ),
                                ),
                            )

            # -----------------------------------------------------------------
            # Display
            # -----------------------------------------------------------------

            v3.VDivider(classes="my-4")

            html.Div(
                "Display",
                classes="vtkweb-section-title",
            )

            with html.Div(
                classes="vtkweb-select-box",
            ):
                html.Span(
                    "Representation",
                    classes="vtkweb-select-prefix",
                )

                v3.VSelect(
                    items=("representation_items",),
                    model_value=("representation_mode",),
                    density="compact",
                    hide_details=True,
                    variant="plain",
                    classes="vtkweb-select-control",
                    update_modelValue=(
                        ctrl.set_representation_mode,
                        "[$event]",
                    ),
                )
            # -----------------------------------------------------------------
            # Coloring
            # -----------------------------------------------------------------

            v3.VDivider(classes="my-4")

            html.Div(
                "Coloring",
                classes="vtkweb-section-title",
            )

            with html.Div(
                classes="vtkweb-select-box",
            ):
                html.Span(
                    "Color by",
                    classes="vtkweb-select-prefix",
                )

                v3.VSelect(
                    items=("color_array_items",),
                    model_value=("color_array",),
                    clearable=True,
                    density="compact",
                    hide_details=True,
                    variant="plain",
                    classes="vtkweb-select-control",
                    update_modelValue=(
                        ctrl.set_color_array,
                        "[$event]",
                    ),
                )

            with html.Div(
                v_if="color_array !== null",
                classes="vtkweb-range-row",
            ):
                html.Input(
                    type="number",
                    step="any",
                    value=("color_range_min",),
                    classes="vtkweb-range-input",
                    change=(
                        ctrl.set_color_range_min,
                        "[$event.target.value]",
                    ),
                )

                html.Input(
                    type="number",
                    step="any",
                    value=("color_range_max",),
                    classes="vtkweb-range-input",
                    change=(
                        ctrl.set_color_range_max,
                        "[$event.target.value]",
                    ),
                )

                v3.VBtn(
                    "Fit",
                    size="small",
                    click=ctrl.fit_color_range,
                )
