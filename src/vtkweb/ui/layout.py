from __future__ import annotations

import vtk

from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import client
from trame.widgets import vuetify3 as v3

from vtkweb.catalog import AlgorithmCatalog
from vtkweb.pipeline import PipelineGraph
from vtkweb.properties import inspect_properties, set_property
from vtkweb.rendering import RenderManager
from vtkweb.ui.filter_browser import build_filter_browser
from vtkweb.ui.pipeline_view import build_pipeline_view
from vtkweb.ui.properties_view import (
    PROPERTY_STYLE,
    build_properties_view,
)
from vtkweb.ui.render_view import build_render_view


def build_ui(
    server,
    pipeline: PipelineGraph,
    rendering: RenderManager,
    catalog: AlgorithmCatalog,
) -> None:
    state = server.state
    ctrl = server.controller

    # -------------------------------------------------------------------------
    # State
    # -------------------------------------------------------------------------

    state.active_node_id = pipeline.active_node_id
    state.active_node_name = pipeline.active_node.name
    state.active_node_type = (
        pipeline.active_node.algorithm.GetClassName()
    )

    state.filter_properties = []

    state.node_visibility = {
        node.id: node.visible
        for node in pipeline.nodes.values()
    }

    state.representation_items = [
        {
            "title": "Surface",
            "value": "surface",
        },
        {
            "title": "Wireframe",
            "value": "wireframe",
        },
    ]
    state.representation_mode = "surface"

    state.color_array_items = []
    state.color_array = None
    state.color_range_min = 0.0
    state.color_range_max = 1.0

    state.filtered_filter_catalog = []

    # -------------------------------------------------------------------------
    # State synchronization
    # -------------------------------------------------------------------------

    def update_property_state(node_id: str) -> None:
        algorithm = pipeline.nodes[node_id].algorithm
        properties = inspect_properties(algorithm)

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

    def update_representation_state(node_id: str) -> None:
        arrays = rendering.get_arrays(node_id)
        representation = rendering.representations[node_id]

        items = []

        for name in arrays["point"]:
            items.append(
                {
                    "title": f"{name} (Point)",
                    "value": f"point:{name}",
                }
            )

        for name in arrays["cell"]:
            items.append(
                {
                    "title": f"{name} (Cell)",
                    "value": f"cell:{name}",
                }
            )

        with state:
            state.representation_mode = (
                representation.representation_mode
            )
            state.color_array_items = items

            if representation.array_name is None:
                state.color_array = None
            else:
                state.color_array = (
                    f"{representation.association}:"
                    f"{representation.array_name}"
                )

            if representation.scalar_range is not None:
                state.color_range_min = (
                    representation.scalar_range[0]
                )
                state.color_range_max = (
                    representation.scalar_range[1]
                )

    # -------------------------------------------------------------------------
    # Active node
    # -------------------------------------------------------------------------

    def set_active_node(node_id: str) -> None:
        pipeline.set_active_node(node_id)

        with state:
            state.active_node_id = node_id
            state.active_node_name = pipeline.active_node.name
            state.active_node_type = (
                pipeline.active_node.algorithm.GetClassName()
            )

        update_property_state(node_id)
        update_representation_state(node_id)

    # -------------------------------------------------------------------------
    # Pipeline interaction
    # -------------------------------------------------------------------------

    def toggle_visibility(node_id: str) -> None:
        rendering.toggle_visibility(node_id)

        state.node_visibility = {
            **state.node_visibility,
            node_id: pipeline.nodes[node_id].visible,
        }

        ctrl.view_update()

    def node_click(
        node_id: str,
        shift_key: bool = False,
    ) -> None:
        if shift_key:
            toggle_visibility(node_id)
        else:
            set_active_node(node_id)

    # -------------------------------------------------------------------------
    # Filter browser
    # -------------------------------------------------------------------------

    def open_filter_browser() -> None:
        with state:
            state.filter_browser_query = ""
            state.filter_browser_open = True

    def create_filter(class_name: str) -> None:
        source_node = pipeline.active_node

        descriptor = next(
            item
            for item in catalog.algorithms
            if item.class_name == class_name
        )

        algorithm = catalog.create(class_name)

        new_node = pipeline.add_node(
            algorithm,
            name=descriptor.label,
        )

        edge = pipeline.connect(
            source_node.id,
            new_node.id,
            source_port=0,
            target_port=0,
        )

        rendering.add_representation(new_node.id)

        ctrl.pipeline_add_node(new_node)
        ctrl.pipeline_add_edge(edge)

        pipeline.set_active_node(new_node.id)

        with state:
            state.active_node_id = new_node.id
            state.active_node_name = new_node.name
            state.active_node_type = algorithm.GetClassName()

            state.node_visibility = {
                **state.node_visibility,
                new_node.id: new_node.visible,
            }

            state.filter_browser_query = ""
            state.filter_browser_open = False

        algorithm.Update()

        update_property_state(new_node.id)
        update_representation_state(new_node.id)

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Filter properties
    # -------------------------------------------------------------------------

    def get_descriptor(name: str):
        return next(
            prop
            for prop in inspect_properties(
                pipeline.active_node.algorithm
            )
            if prop.name == name
        )

    def set_filter_property(
        name: str,
        value,
    ) -> None:
        algorithm = pipeline.active_node.algorithm

        set_property(
            algorithm,
            get_descriptor(name),
            value,
        )

        update_property_state(
            pipeline.active_node_id
        )

        ctrl.view_update()

    def set_filter_vector_component(
        name: str,
        index: int,
        value,
    ) -> None:
        algorithm = pipeline.active_node.algorithm
        descriptor = get_descriptor(name)

        values = list(descriptor.value)
        values[int(index)] = float(value)

        set_property(
            algorithm,
            descriptor,
            values,
        )

        update_property_state(
            pipeline.active_node_id
        )

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Elevation helpers
    # -------------------------------------------------------------------------

    def set_elevation_axis(axis: str) -> None:
        algorithm = pipeline.active_node.algorithm

        if not isinstance(
            algorithm,
            vtk.vtkElevationFilter,
        ):
            return

        algorithm.Update()

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

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)

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

        update_property_state(
            pipeline.active_node_id
        )

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def set_representation_mode(value: str) -> None:
        rendering.set_representation_mode(
            pipeline.active_node_id,
            value,
        )

        state.representation_mode = value

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Coloring
    # -------------------------------------------------------------------------

    def set_color_array(value) -> None:
        node_id = pipeline.active_node_id

        if not value:
            rendering.set_array(
                node_id,
                None,
            )

            state.color_array = None
            ctrl.view_update()
            return

        association, array_name = value.split(
            ":",
            1,
        )

        rendering.set_array(
            node_id,
            array_name,
            association,
        )

        representation = rendering.representations[
            node_id
        ]

        with state:
            state.color_array = value

            if representation.scalar_range is not None:
                state.color_range_min = (
                    representation.scalar_range[0]
                )
                state.color_range_max = (
                    representation.scalar_range[1]
                )

        ctrl.view_update()

    def set_color_range_min(value) -> None:
        rendering.set_scalar_range(
            pipeline.active_node_id,
            float(value),
            float(state.color_range_max),
        )

        ctrl.view_update()

    def set_color_range_max(value) -> None:
        rendering.set_scalar_range(
            pipeline.active_node_id,
            float(state.color_range_min),
            float(value),
        )

        ctrl.view_update()

    def fit_color_range() -> None:
        node_id = pipeline.active_node_id
        representation = rendering.representations[
            node_id
        ]

        if representation.array_name is None:
            return

        scalar_range = rendering.get_array_range(
            node_id,
            representation.array_name,
            representation.association,
        )

        if scalar_range is None:
            return

        minimum, maximum = scalar_range

        rendering.set_scalar_range(
            node_id,
            minimum,
            maximum,
        )

        with state:
            state.color_range_min = minimum
            state.color_range_max = maximum

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    ctrl.node_click = node_click
    ctrl.set_active_node = set_active_node
    ctrl.toggle_visibility = toggle_visibility

    ctrl.open_filter_browser = open_filter_browser
    ctrl.create_filter = create_filter

    ctrl.set_filter_property = set_filter_property
    ctrl.set_filter_vector_component = (
        set_filter_vector_component
    )

    ctrl.set_elevation_axis = set_elevation_axis

    ctrl.set_representation_mode = (
        set_representation_mode
    )

    ctrl.set_color_array = set_color_array
    ctrl.set_color_range_min = (
        set_color_range_min
    )
    ctrl.set_color_range_max = (
        set_color_range_max
    )
    ctrl.fit_color_range = fit_color_range

    update_property_state(
        pipeline.active_node_id
    )
    update_representation_state(
        pipeline.active_node_id
    )

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------

    with SinglePageWithDrawerLayout(
        server,
        show_drawer=False,
        width=220,
    ) as layout:
        layout.title.set_text("vtkweb")

        with layout.drawer:
            with v3.VList(
                density="compact",
                nav=True,
            ):
                v3.VListItem(
                    title="Add Filter",
                    prepend_icon="mdi-plus",
                    click=ctrl.open_filter_browser,
                )

        with layout.content:
            client.Style(PROPERTY_STYLE)

            build_filter_browser(
                state,
                ctrl,
                catalog,
            )

            with v3.VContainer(
                fluid=True,
                classes="fill-height pa-0",
            ):
                with v3.VRow(
                    no_gutters=True,
                    classes="fill-height",
                ):
                    build_pipeline_view(
                        ctrl,
                        pipeline,
                    )

                    build_render_view(
                        ctrl,
                        rendering,
                    )

                    build_properties_view(
                        ctrl,
                    )
