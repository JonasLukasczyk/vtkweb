from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3


PROPERTY_STYLE = """
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

/* Common compact input */

.vtkweb-input-box,
.vtkweb-vector-box,
.vtkweb-select-box,
.vtkweb-list-row,
.vtkweb-bool-row {
    width: 100%;
    min-width: 0;
    min-height: 28px;

    border: 1px solid rgba(128,128,128,0.5);
    border-radius: 4px;

    background: rgba(128,128,128,0.08);

    box-sizing: border-box;
}

.vtkweb-input-box:hover,
.vtkweb-vector-box:hover,
.vtkweb-select-box:hover {
    border-color: rgba(128,128,128,0.8);
}

.vtkweb-input-box:focus-within,
.vtkweb-vector-box:focus-within,
.vtkweb-select-box:focus-within,
.vtkweb-list-row:focus-within {
    border-color: #4f7df3;
    background: rgba(79,125,243,0.06);
}

/* Scalar */

.vtkweb-input-box {
    display: flex;
    align-items: center;

    height: 28px;
    overflow: hidden;

    cursor: text;
}

.vtkweb-input-prefix,
.vtkweb-select-prefix {
    flex: 0 0 auto;

    padding: 0 6px 0 8px;

    font-size: 12px;
    line-height: 26px;
    opacity: 0.82;

    white-space: nowrap;
    user-select: none;
}

.vtkweb-input-box input {
    flex: 1 1 auto;

    min-width: 0;
    height: 26px;

    padding: 0 8px;

    border: 0;
    outline: 0;
    box-shadow: none;

    background: transparent;
    color: inherit;

    font: inherit;
    font-size: 12px;

    text-align: right;

    appearance: textfield;
    -moz-appearance: textfield;
}

/* Vector */

.vtkweb-vector-box {
    display: flex;
    align-items: center;

    height: 28px;
    overflow: hidden;
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

    padding: 0 5px;

    border: 0;
    border-left:
        1px solid rgba(128,128,128,0.25);

    outline: 0;

    background: transparent;
    color: inherit;

    font: inherit;
    font-size: 12px;
    text-align: right;

    box-sizing: border-box;

    appearance: textfield;
    -moz-appearance: textfield;
}

/* Boolean */

.vtkweb-bool-row {
    display: flex;
    align-items: center;
    justify-content: space-between;

    height: 28px;

    padding: 0 8px;
}

/* List */

.vtkweb-list-property {
    display: flex;
    flex-direction: column;
    gap: 4px;

    width: 100%;
}

.vtkweb-list-header {
    display: flex;
    align-items: center;
    justify-content: space-between;

    height: 26px;

    padding-left: 8px;

    font-size: 12px;
    opacity: 0.82;
}

.vtkweb-list-add {
    height: 24px;

    padding: 0 8px;

    border: 1px solid rgba(128,128,128,0.5);
    border-radius: 4px;

    background: rgba(128,128,128,0.08);
    color: inherit;

    cursor: pointer;
}

.vtkweb-list-row {
    display: flex;
    align-items: center;

    height: 28px;

    overflow: hidden;
}

.vtkweb-list-index {
    padding: 0 7px;

    font-size: 11px;
    opacity: 0.45;

    user-select: none;
}

.vtkweb-list-row input {
    flex: 1 1 auto;

    min-width: 0;
    height: 26px;

    padding: 0 8px;

    border: 0;
    outline: 0;

    background: transparent;
    color: inherit;

    font: inherit;
    font-size: 12px;
    text-align: right;
}

.vtkweb-list-remove {
    width: 28px;
    align-self: stretch;

    border: 0;
    border-left:
        1px solid rgba(128,128,128,0.25);

    background: transparent;
    color: inherit;

    cursor: pointer;
}

/* Select */

.vtkweb-select-box {
    display: flex;
    align-items: center;

    height: 28px;

    overflow: hidden;
}

.vtkweb-select-control {
    flex: 1 1 auto;

    min-width: 0;
    width: 0;
}

.vtkweb-select-control .v-input__control,
.vtkweb-select-control .v-field {
    min-height: 26px !important;
    height: 26px !important;
}

.vtkweb-select-control .v-field {
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

/* Color range */

.vtkweb-range-row {
    display: grid;
    grid-template-columns:
        minmax(0,1fr)
        minmax(0,1fr)
        auto;

    gap: 6px;

    width: 100%;
    margin-top: 8px;
}

.vtkweb-range-input {
    width: 100%;
    min-width: 0;
    height: 28px;

    padding: 0 7px;

    border:
        1px solid rgba(128,128,128,0.5);
    border-radius: 4px;
    outline: none;

    background:
        rgba(128,128,128,0.08);
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
        style=(
            "height:100vh;"
            "overflow-y:auto;"
            "min-width:0;"
        ),
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

            # -------------------------------------------------------------
            # Filter
            # -------------------------------------------------------------

            html.Div(
                "Filter",
                classes="vtkweb-section-title",
            )

            # Input arrays
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
                        classes="vtkweb-select-prefix",
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

            # Elevation presets
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

            # Generic properties
            with html.Div(
                classes="vtkweb-prop-list",
            ):
                with html.Div(
                    v_for=(
                        "property in "
                        "filter_properties"
                    ),
                    key="property.name",
                    classes="vtkweb-prop-item",
                ):

                    with html.Label(
                        v_if="property.kind === 'bool'",
                        classes="vtkweb-bool-row",
                    ):
                        html.Span(
                            "{{ property.label }}"
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
                                    "[property.name,"
                                    "$event.target.value]"
                                ),
                            ),
                        )

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

                    with html.Div(
                        v_if=(
                            "property.kind === "
                            "'scalar_list'"
                        ),
                        classes="vtkweb-list-property",
                    ):
                        with html.Div(
                            classes="vtkweb-list-header",
                        ):
                            html.Span(
                                "{{ property.label }}"
                            )

                            html.Button(
                                "+",
                                type="button",
                                classes="vtkweb-list-add",
                                click=(
                                    ctrl.add_filter_list_value,
                                    "[property.name]",
                                ),
                            )

                        with html.Div(
                            v_for=(
                                "(value,index) "
                                "in property.value"
                            ),
                            key="index",
                            classes="vtkweb-list-row",
                        ):
                            html.Span(
                                "{{ index }}",
                                classes="vtkweb-list-index",
                            )

                            html.Input(
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
                                "×",
                                type="button",
                                classes="vtkweb-list-remove",
                                click=(
                                    ctrl.remove_filter_list_value,
                                    (
                                        "[property.name,"
                                        "index]"
                                    ),
                                ),
                            )

            # -------------------------------------------------------------
            # Display
            # -------------------------------------------------------------

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

            # -------------------------------------------------------------
            # Coloring
            # -------------------------------------------------------------

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
