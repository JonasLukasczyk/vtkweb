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
    # Properties and input-array descriptors are part of state.pipeline. This
    # module only owns the UI actions which mutate them through PipelineGraph.

    def active_node_id() -> str | None:
        return pipeline.active_node_id

    def update_properties_state(
        node_id: str | None,
    ) -> None:
        # Compatibility hook for existing callers. There is no second property
        # list to synchronize anymore. Explicit callers may use this to pull
        # direct/raw VTK mutations back into state.
        if node_id is not None:
            pipeline.sync_node_from_runtime(node_id)

    def set_active_input_array(
        index: int,
        value,
    ) -> None:
        node_id = active_node_id()
        if node_id is None:
            return

        pipeline.set_input_array(
            node_id,
            int(index),
            value,
        )
        ctrl.view_update()

    def set_filter_property(
        name: str,
        value,
    ) -> None:
        node_id = active_node_id()
        if node_id is None:
            return

        pipeline.set_property(
            node_id,
            name,
            value,
        )
        ctrl.update_representation_state(node_id)
        ctrl.view_update()

    def set_filter_vector_component(
        name: str,
        index: int,
        value,
    ) -> None:
        node_id = active_node_id()
        if node_id is None:
            return

        pipeline.set_vector_component(
            node_id,
            name,
            int(index),
            value,
        )
        ctrl.update_representation_state(node_id)
        ctrl.view_update()

    def set_filter_list_value(
        name: str,
        index: int,
        value,
    ) -> None:
        node_id = active_node_id()
        if node_id is None:
            return

        pipeline.set_list_value(
            node_id,
            name,
            int(index),
            value,
        )
        ctrl.update_representation_state(node_id)
        ctrl.view_update()

    def add_filter_list_value(
        name: str,
    ) -> None:
        node_id = active_node_id()
        if node_id is None:
            return

        pipeline.add_list_value(
            node_id,
            name,
        )
        ctrl.update_representation_state(node_id)
        ctrl.view_update()

    def remove_filter_list_value(
        name: str,
        index: int,
    ) -> None:
        node_id = active_node_id()
        if node_id is None:
            return

        pipeline.remove_list_value(
            node_id,
            name,
            int(index),
        )
        ctrl.update_representation_state(node_id)
        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Elevation helper
    # -------------------------------------------------------------------------

    def set_elevation_axis(
        axis: str,
    ) -> None:
        node = pipeline.active_node

        if node is None or not isinstance(
            node.algorithm,
            vtk.vtkElevationFilter,
        ):
            return

        algorithm = node.algorithm
        input_data = algorithm.GetInputDataObject(
            0,
            0,
        )

        if input_data is None:
            return

        (
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        ) = input_data.GetBounds()

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        cz = (zmin + zmax) / 2

        if axis == "x":
            low = (xmin, cy, cz)
            high = (xmax, cy, cz)
        elif axis == "y":
            low = (cx, ymin, cz)
            high = (cx, ymax, cz)
        else:
            low = (cx, cy, zmin)
            high = (cx, cy, zmax)

        algorithm.SetLowPoint(*low)
        algorithm.SetHighPoint(*high)
        algorithm.Update()

        pipeline.sync_node_from_runtime(node.id)
        ctrl.update_representation_state(node.id)
        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    ctrl.update_properties_state = update_properties_state
    ctrl.set_input_array = set_active_input_array
    ctrl.set_filter_property = set_filter_property
    ctrl.set_filter_vector_component = set_filter_vector_component
    ctrl.set_filter_list_value = set_filter_list_value
    ctrl.add_filter_list_value = add_filter_list_value
    ctrl.remove_filter_list_value = remove_filter_list_value
    ctrl.set_elevation_axis = set_elevation_axis


def build_properties_tab(
    ctrl,
) -> None:
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
            v_for=(
                "array in Object.values("
                "pipeline.nodes[active_node_id]?."
                "input_arrays || {})"
            ),
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
            v_for=(
                "property in Object.values("
                "pipeline.nodes[active_node_id]?."
                "properties || {})"
            ),
            key="property.name",
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
                        ctrl.set_filter_property,
                        ("[property.name,$event.target.checked]"),
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
                    type=("property.kind === 'str' ? 'text' : 'number'"),
                    step=("property.kind === 'int' ? 1 : 'any'"),
                    value=("property.value",),
                    change=(
                        ctrl.set_filter_property,
                        ("[property.name,$event.target.value]"),
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
                        key="index",
                        type="number",
                        step="any",
                        value=("component",),
                        change=(
                            ctrl.set_filter_vector_component,
                            ("[property.name,index,$event.target.value]"),
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
                        key="index",
                        type="number",
                        step="any",
                        value=("value",),
                        change=(
                            ctrl.set_filter_list_value,
                            ("[property.name,index,$event.target.value]"),
                        ),
                    )

                html.Button(
                    "−",
                    type="button",
                    classes="vtkweb-list-inline-button",
                    disabled=("property.value.length === 0"),
                    click=(
                        ctrl.remove_filter_list_value,
                        ("[property.name,property.value.length - 1]"),
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
