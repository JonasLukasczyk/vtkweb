from __future__ import annotations

import vtk

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from vtkweb.pipeline import PipelineGraph


def initialize_properties_tab(
    state,
    ctrl,
    pipeline: PipelineGraph,
) -> None:
    def set_elevation_axis(
        node_id: str,
        axis: str,
    ) -> None:
        if node_id not in pipeline.nodes:
            return

        node = pipeline.nodes[node_id]
        if not isinstance(node.processor, vtk.vtkElevationFilter):
            return

        processor = node.processor
        input_data = processor.GetInputDataObject(0, 0)
        if input_data is None:
            return

        xmin, xmax, ymin, ymax, zmin, zmax = input_data.GetBounds()
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        cz = (zmin + zmax) / 2

        if axis == "x":
            low, high = (xmin, cy, cz), (xmax, cy, cz)
        elif axis == "y":
            low, high = (cx, ymin, cz), (cx, ymax, cz)
        else:
            low, high = (cx, cy, zmin), (cx, cy, zmax)

        processor.SetLowPoint(*low)
        processor.SetHighPoint(*high)
        processor.Update()
        pipeline.sync_node_from_runtime(node_id)

    ctrl.set_elevation_axis = set_elevation_axis


def build_properties_tab(
    ctrl,
) -> None:
    # -------------------------------------------------------------------------
    # Input arrays
    # -------------------------------------------------------------------------

    with html.Div(
        classes="vtkweb-prop-list mb-2",
    ):
        with html.Div(
            v_for=(
                "array in Object.values("
                "pipeline.nodes[active_node_id]?."
                "input_arrays || {})"
            ),
            key=("array.index",),
            classes="vtkweb-select-box",
        ):
            html.Span(
                "{{ array.label }}",
                classes="vtkweb-control-label",
            )

            v3.VSelect(
                classes="vtkweb-compact-select",
                model_value=("array.value ?? null",),
                items=("[{ title: 'None', value: null }, ...array.items]",),
                item_title="title",
                item_value="value",
                density="compact",
                variant="plain",
                hide_details=True,
                update_modelValue=(
                    ctrl.set_node_input_array,
                    "[active_node_id, array.index, $event]",
                ),
            )

    # -------------------------------------------------------------------------
    # Elevation presets
    # -------------------------------------------------------------------------

    with v3.VRow(
        v_if=("pipeline.nodes[active_node_id]?.class_name === 'vtkElevationFilter'"),
        dense=True,
        classes="mb-3",
    ):
        for axis in (
            "x",
            "y",
            "z",
        ):
            with v3.VCol(cols=4):
                v3.VBtn(
                    axis.upper(),
                    block=True,
                    size="small",
                    click=(
                        ctrl.set_elevation_axis,
                        f"[active_node_id, '{axis}']",
                    ),
                )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    with html.Div(
        classes="vtkweb-prop-list",
    ):
        with html.Div(
            v_for=(
                "property in Object.values("
                "pipeline.nodes[active_node_id]?."
                "properties || {})"
            ),
            key=("property.name",),
            classes="vtkweb-prop-item",
        ):
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
                    checked=("Boolean(property.value)",),
                    change=(
                        ctrl.set_node_property,
                        ("[active_node_id,property.name,$event.target.checked]"),
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
                    classes="vtkweb-control-label",
                )
                html.Input(
                    type=("property.kind === 'str' ? 'text' : 'number'",),
                    step=("property.kind === 'int' ? 1 : 'any'",),
                    value=("property.value",),
                    change=(
                        ctrl.set_node_property,
                        ("[active_node_id,property.name,$event.target.value]"),
                    ),
                )

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
                        v_for=("(component,index) in property.value"),
                        key=("index",),
                        type="number",
                        step="any",
                        value=("component",),
                        change=(
                            ctrl.set_node_vector_component,
                            (
                                "[active_node_id,property.name,index,$event.target.value]"
                            ),
                        ),
                    )

            with html.Div(
                v_if=("property.kind === 'scalar_list'"),
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
                        v_for=("(value,index) in property.value"),
                        key=("index",),
                        type="number",
                        step="any",
                        value=("value",),
                        change=(
                            ctrl.set_node_list_value,
                            (
                                "[active_node_id,property.name,index,$event.target.value]"
                            ),
                        ),
                    )

                html.Button(
                    "−",
                    type="button",
                    classes="vtkweb-list-inline-button",
                    disabled=("property.value.length === 0",),
                    click=(
                        ctrl.remove_node_list_value,
                        ("[active_node_id,property.name,property.value.length - 1]"),
                    ),
                )
                html.Button(
                    "+",
                    type="button",
                    classes="vtkweb-list-inline-button",
                    click=(
                        ctrl.add_node_list_value,
                        "[active_node_id,property.name]",
                    ),
                )
