from __future__ import annotations

import vtk

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from vtkweb.input_arrays import (
    inspect_input_arrays,
    set_input_array,
)
from vtkweb.pipeline import PipelineGraph
from vtkweb.properties import (
    inspect_properties,
    set_property,
)


def initialize_properties_tab(
    state,
    ctrl,
    pipeline: PipelineGraph,
) -> None:
    state.filter_properties = []
    state.input_arrays = []

    # -------------------------------------------------------------------------
    # State synchronization
    # -------------------------------------------------------------------------

    def update_properties_state(
        node_id: str | None,
    ) -> None:
        if node_id is None:
            with state:
                state.filter_properties = []
                state.input_arrays = []
            return

        algorithm = (
            pipeline.nodes[node_id].algorithm
        )

        properties = inspect_properties(
            algorithm
        )

        input_arrays = inspect_input_arrays(
            algorithm
        )

        with state:
            state.filter_properties = [
                {
                    "name": prop.name,
                    "label": prop.label,
                    "kind": prop.kind,
                    "value": prop.value,
                    "size": prop.size,
                }
                for prop in properties
            ]

            state.input_arrays = [
                {
                    "index": item.index,
                    "label": item.label,
                    "value": item.value,
                    "items": item.items,
                }
                for item in input_arrays
            ]

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def get_descriptor(
        name: str,
    ):
        return next(
            prop
            for prop in inspect_properties(
                pipeline.active_node.algorithm
            )
            if prop.name == name
        )

    def apply_property(
        descriptor,
        value,
    ) -> None:
        algorithm = (
            pipeline.active_node.algorithm
        )

        set_property(
            algorithm,
            descriptor,
            value,
        )

        algorithm.Update()

        ctrl.refresh_node(
            pipeline.active_node_id
        )

    # -------------------------------------------------------------------------
    # Input arrays
    # -------------------------------------------------------------------------

    def set_active_input_array(
        index: int,
        value,
    ) -> None:
        if not value:
            return

        algorithm = (
            pipeline.active_node.algorithm
        )

        descriptor = next(
            item
            for item in inspect_input_arrays(
                algorithm
            )
            if item.index == int(index)
        )

        set_input_array(
            algorithm,
            descriptor,
            value,
        )

        algorithm.Update()

        ctrl.refresh_node(
            pipeline.active_node_id
        )

    # -------------------------------------------------------------------------
    # Generic properties
    # -------------------------------------------------------------------------

    def set_filter_property(
        name: str,
        value,
    ) -> None:
        apply_property(
            get_descriptor(name),
            value,
        )

    def set_filter_vector_component(
        name: str,
        index: int,
        value,
    ) -> None:
        descriptor = get_descriptor(
            name
        )

        values = list(
            descriptor.value
        )

        values[int(index)] = float(
            value
        )

        apply_property(
            descriptor,
            values,
        )

    def set_filter_list_value(
        name: str,
        index: int,
        value,
    ) -> None:
        if value in ("", None):
            return

        descriptor = get_descriptor(
            name
        )

        values = list(
            descriptor.value
        )

        values[int(index)] = float(
            value
        )

        apply_property(
            descriptor,
            values,
        )

    def add_filter_list_value(
        name: str,
    ) -> None:
        descriptor = get_descriptor(
            name
        )

        values = list(
            descriptor.value
        )

        values.append(
            values[-1]
            if values
            else 0.0
        )

        apply_property(
            descriptor,
            values,
        )

    def remove_filter_list_value(
        name: str,
        index: int,
    ) -> None:
        descriptor = get_descriptor(
            name
        )

        values = list(
            descriptor.value
        )

        index = int(index)

        if (
            index < 0
            or index >= len(values)
        ):
            return

        del values[index]

        apply_property(
            descriptor,
            values,
        )

    # -------------------------------------------------------------------------
    # Elevation helper
    # -------------------------------------------------------------------------

    def set_elevation_axis(
        axis: str,
    ) -> None:
        algorithm = (
            pipeline.active_node.algorithm
        )

        if not isinstance(
            algorithm,
            vtk.vtkElevationFilter,
        ):
            return

        input_data = (
            algorithm.GetInputDataObject(
                0,
                0,
            )
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
            low = (
                xmin,
                cy,
                cz,
            )
            high = (
                xmax,
                cy,
                cz,
            )

        elif axis == "y":
            low = (
                cx,
                ymin,
                cz,
            )
            high = (
                cx,
                ymax,
                cz,
            )

        else:
            low = (
                cx,
                cy,
                zmin,
            )
            high = (
                cx,
                cy,
                zmax,
            )

        algorithm.SetLowPoint(
            *low
        )

        algorithm.SetHighPoint(
            *high
        )

        algorithm.Update()

        ctrl.refresh_node(
            pipeline.active_node_id
        )

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    ctrl.update_properties_state = (
        update_properties_state
    )

    ctrl.set_input_array = (
        set_active_input_array
    )

    ctrl.set_filter_property = (
        set_filter_property
    )

    ctrl.set_filter_vector_component = (
        set_filter_vector_component
    )

    ctrl.set_filter_list_value = (
        set_filter_list_value
    )

    ctrl.add_filter_list_value = (
        add_filter_list_value
    )

    ctrl.remove_filter_list_value = (
        remove_filter_list_value
    )

    ctrl.set_elevation_axis = (
        set_elevation_axis
    )


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

            # Vector
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

            # Scalar list
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
